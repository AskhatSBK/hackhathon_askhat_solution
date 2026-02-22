"""
Index builder: BM25 + FAISS dense indexes + hierarchical tree structures.
Indexes are built once and saved to disk; loaded at server startup.
"""

import json
import os
import pickle
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from src.data_loader import Protocol, RUSSIAN_STOPWORDS


# ---------------------------------------------------------------------------
# Device detection: NVIDIA CUDA / AMD ROCm / Apple MPS / CPU
# ---------------------------------------------------------------------------

def get_device() -> str:
    """Return the best available compute device.

    Priority: env override > CUDA (covers NVIDIA + AMD ROCm) > MPS > CPU.
    AMD ROCm exposes itself as 'cuda' in PyTorch when the ROCm-patched build
    is installed, so torch.cuda.is_available() returns True on ROCm too.
    """
    override = os.getenv("DEVICE", "").strip().lower()
    if override:
        return override
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass
    return "cpu"


def _is_rocm() -> bool:
    """Return True when running on AMD ROCm (not NVIDIA CUDA)."""
    try:
        import torch
        return torch.cuda.is_available() and (
            hasattr(torch.version, "hip") and torch.version.hip is not None
        )
    except ImportError:
        return False


class LlamaCppEmbedder:
    """GGUF embedding model via llama-cpp-python.

    Exposes the same encode() interface as SentenceTransformer so the rest
    of the codebase needs no changes.  Use it for quantized GGUF models such
    as Qwen/Qwen3-Embedding-0.6B-GGUF.

    Download the .gguf file first, e.g.:
        python -c "
        from huggingface_hub import hf_hub_download
        hf_hub_download('Qwen/Qwen3-Embedding-0.6B-GGUF',
                        'Qwen3-Embedding-0.6B-Q4_K_M.gguf',
                        local_dir='data/models')"

    Then set env vars:
        EMBED_BACKEND=llama-cpp
        EMBED_MODEL_PATH=data/models/Qwen3-Embedding-0.6B-Q4_K_M.gguf
    """

    prompts: dict = {}  # no built-in prompts; prefix via encode_texts() env vars

    def __init__(self, model_path: str, n_ctx: int = 2048):
        from llama_cpp import Llama
        import llama_cpp
        # Silence "init: embeddings required but..." noise from llama.cpp internals
        llama_cpp.llama_log_set(None, None)
        n_threads = int(os.getenv("EMBED_N_THREADS", str(os.cpu_count() or 4)))
        print(f"Loading GGUF embedder: {model_path} (n_threads={n_threads}, n_ctx={n_ctx})")
        self._llm = Llama(
            model_path=model_path,
            embedding=True,
            n_ctx=n_ctx,
            n_batch=n_ctx,          # process full context in one decode step
            n_threads=n_threads,
            n_threads_batch=n_threads,  # batch thread pool (used for embedding)
            verbose=False,
        )

    def encode(
        self,
        texts: list[str],
        normalize_embeddings: bool = True,
        batch_size: int = 1,  # n_seq_max=1: always process one text per call
        show_progress_bar: bool = False,
        **_,
    ) -> "np.ndarray":
        import numpy as np

        all_vecs = []
        items = list(range(len(texts)))
        if show_progress_bar:
            try:
                from tqdm import tqdm
                items = tqdm(items, desc="Encoding GGUF")
            except ImportError:
                pass

        for i in items:
            result = self._llm.create_embedding(texts[i])
            all_vecs.append(result["data"][0]["embedding"])

        embeddings = np.array(all_vecs, dtype=np.float32)
        if normalize_embeddings:
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            embeddings = embeddings / np.where(norms == 0, 1.0, norms)
        return embeddings


class OllamaEmbedder:
    """Embedding via Ollama's /api/embed endpoint.

    Set env vars:
        EMBED_PROVIDER=ollama
        EMBED_MODEL=nomic-embed-text          # model name as known to Ollama
        EMBED_BASE_URL=http://localhost:11434/api/embed   # full endpoint URL
    """

    prompts: dict = {}

    def __init__(self, url: str = "http://localhost:11434/v1/embeddings", model: str = "qwen3-embedding:0.6b"):
        self._url = url
        self._model = model
        print(f"Ollama embedder → {self._url}  model={model}")

    def encode(
        self,
        texts: list[str],
        normalize_embeddings: bool = True,
        batch_size: int = 32,
        show_progress_bar: bool = False,
        **_,
    ) -> "np.ndarray":
        import json
        import urllib.request
        import numpy as np

        all_vecs = []
        batches = [texts[i : i + batch_size] for i in range(0, len(texts), batch_size)]
        if show_progress_bar:
            try:
                from tqdm import tqdm
                batches = tqdm(batches, desc="Encoding via Ollama")
            except ImportError:
                pass

        for batch in batches:
            payload = json.dumps({"model": self._model, "input": batch}).encode()
            req = urllib.request.Request(
                self._url,
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req) as resp:
                data = json.loads(resp.read())
            # Ollama >=0.5: {"embeddings": [[...], [...]]}  (batch)
            # Ollama older: {"embedding": [...]}            (single)
            if "embeddings" in data:
                for vec in data["embeddings"]:
                    all_vecs.append(vec)
            else:
                all_vecs.append(data["embedding"])

        embeddings = np.array(all_vecs, dtype=np.float32)
        if normalize_embeddings:
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            embeddings = embeddings / np.where(norms == 0, 1.0, norms)
        return embeddings


class APIEmbedder:
    """Remote embedding model via OpenAI-compatible /v1/embeddings API.

    Use when the embedding model runs in a separate container (vLLM or SGLang
    on ROCm/CUDA). Both indexing and query encoding call this HTTP endpoint.

    Set env vars:
        EMBED_BACKEND=api
        EMBED_API_BASE=http://embed:8000/v1
        EMBED_API_MODEL=Qwen/Qwen3-Embedding-0.6B
        EMBED_API_KEY=placeholder  (optional)
    """

    prompts: dict = {}  # no built-in prompts; prefix handled by encode_texts()

    def __init__(self, base_url: str, model: str, api_key: str = "placeholder"):
        from openai import OpenAI
        # ngrok free tier requires this header to bypass the browser warning page
        extra_headers = {"ngrok-skip-browser-warning": "true"} if "ngrok" in base_url else {}
        self._client = OpenAI(base_url=base_url, api_key=api_key, default_headers=extra_headers)
        self._model = model
        print(f"API embedder → {base_url}  model={model}")

    def encode(
        self,
        texts: list[str],
        normalize_embeddings: bool = True,
        batch_size: int = 32,
        show_progress_bar: bool = False,
        **_,
    ) -> "np.ndarray":
        import numpy as np

        all_vecs = []
        batches = [texts[i : i + batch_size] for i in range(0, len(texts), batch_size)]
        if show_progress_bar:
            try:
                from tqdm import tqdm
                batches = tqdm(batches, desc="Encoding via API")
            except ImportError:
                pass

        for batch in batches:
            response = self._client.embeddings.create(model=self._model, input=batch)
            for item in sorted(response.data, key=lambda x: x.index):
                all_vecs.append(item.embedding)

        embeddings = np.array(all_vecs, dtype=np.float32)
        if normalize_embeddings:
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            embeddings = embeddings / np.where(norms == 0, 1.0, norms)
        return embeddings


def load_embed_model(stored_model_name: str = None):
    """Factory: return the right embedder based on EMBED_PROVIDER in .env.

    EMBED_PROVIDER=local (default)
        Loads the model locally via sentence-transformers (CPU).
    EMBED_PROVIDER=openai | together | groq | runpod | custom
        Calls a remote /v1/embeddings endpoint via APIEmbedder.
        Requires EMBED_BASE_URL for runpod/custom providers.
    """
    from src.config import EMBED_PROVIDER, EMBED_MODEL, EMBED_MODEL_PATH, EMBED_N_CTX, get_embed_base_url, get_embed_api_key

    if EMBED_PROVIDER == "llama-cpp":
        path = EMBED_MODEL_PATH
        if not path:
            raise ValueError(
                "EMBED_MODEL_PATH must point to a .gguf file when EMBED_PROVIDER=llama-cpp. "
                "Run: python scripts/download_model.py"
            )
        return LlamaCppEmbedder(model_path=path, n_ctx=EMBED_N_CTX)

    if EMBED_PROVIDER == "ollama":
        url = get_embed_base_url() or "http://localhost:11434/v1/embeddings"
        return OllamaEmbedder(url=url, model=EMBED_MODEL)

    if EMBED_PROVIDER != "local":
        base_url = get_embed_base_url()
        if not base_url:
            raise ValueError(
                f"No base URL for EMBED_PROVIDER={EMBED_PROVIDER!r}. "
                "Set EMBED_BASE_URL in .env."
            )
        return APIEmbedder(base_url=base_url, model=EMBED_MODEL, api_key=get_embed_api_key())

    # local — sentence-transformers on CPU
    from sentence_transformers import SentenceTransformer
    name = stored_model_name or EMBED_MODEL
    device = get_device()
    print(f"Loading embedding model {name!r} on device={device}")
    return SentenceTransformer(name, device=device)


def encode_texts(model, texts: list[str], is_query: bool, **kwargs) -> "np.ndarray":
    """Encode texts using the right prompt strategy for the model.

    Priority:
      1. Model's built-in prompts (SentenceTransformer + E5/Qwen3-HF/etc.)
         → uses prompt_name="query" or "passage" automatically
      2. Manual prefix via env vars or sensible defaults:
         - LlamaCppEmbedder (GGUF):
             EMBED_QUERY_PREFIX defaults to Qwen3 instruction format
             EMBED_PASSAGE_PREFIX defaults to "" (no prefix)
         - Legacy SentenceTransformer without prompts:
             EMBED_QUERY_PREFIX defaults to "query: "
             EMBED_PASSAGE_PREFIX defaults to "passage: "
    """
    prompt_key = "query" if is_query else "passage"

    # Use built-in prompt config when available (works for E5-instruct, Qwen3-HF, etc.)
    if hasattr(model, "prompts") and prompt_key in (model.prompts or {}):
        return model.encode(texts, prompt_name=prompt_key, **kwargs)

    # Determine defaults based on backend
    # LlamaCpp and API backends both serve Qwen3-style models that need instruction prefix
    if isinstance(model, (LlamaCppEmbedder, APIEmbedder, OllamaEmbedder)):
        q_default = (
            "Instruct: Given a medical query, retrieve the most relevant "
            "clinical protocol\nQuery: "
        )
        p_default = ""
    else:
        q_default = "query: "
        p_default = "passage: "

    prefix = os.getenv(
        "EMBED_QUERY_PREFIX" if is_query else "EMBED_PASSAGE_PREFIX",
        q_default if is_query else p_default,
    )
    return model.encode([prefix + t for t in texts], **kwargs)


# ---------------------------------------------------------------------------
# Chunk dataclass
# ---------------------------------------------------------------------------

@dataclass
class Chunk:
    chunk_id: str
    protocol_id: str
    protocol_title: str
    icd_codes: list[str]
    section_type: str
    text: str
    node_id: Optional[str] = None


# ---------------------------------------------------------------------------
# Tree dataclass
# ---------------------------------------------------------------------------

@dataclass
class TreeNode:
    node_id: str
    title: str
    summary: str
    section_type: str
    text_start: int
    text_end: int
    icd_codes: list[str]
    children: list["TreeNode"] = field(default_factory=list)


def treenode_to_dict(node: TreeNode) -> dict:
    d = {
        "node_id": node.node_id,
        "title": node.title,
        "summary": node.summary,
        "section_type": node.section_type,
        "text_start": node.text_start,
        "text_end": node.text_end,
        "icd_codes": node.icd_codes,
        "children": [treenode_to_dict(c) for c in node.children],
    }
    return d


def dict_to_treenode(d: dict) -> TreeNode:
    return TreeNode(
        node_id=d["node_id"],
        title=d["title"],
        summary=d["summary"],
        section_type=d["section_type"],
        text_start=d["text_start"],
        text_end=d["text_end"],
        icd_codes=d["icd_codes"],
        children=[dict_to_treenode(c) for c in d.get("children", [])],
    )


# ---------------------------------------------------------------------------
# Lemmatization (lazy-load pymystem3)
# ---------------------------------------------------------------------------

_mystem = None

def get_mystem():
    global _mystem
    if _mystem is None:
        try:
            from pymystem3 import Mystem
            _mystem = Mystem()
        except ImportError:
            _mystem = None
    return _mystem


def lemmatize(text: str) -> list[str]:
    """Lemmatize Russian text, fallback to simple whitespace split."""
    m = get_mystem()
    if m is not None:
        try:
            lemmas = m.lemmatize(text)
            return [
                l.strip().lower() for l in lemmas
                if l.strip() and l.strip().lower() not in RUSSIAN_STOPWORDS
                and len(l.strip()) > 1
            ]
        except Exception:
            pass
    # Fallback: simple tokenization
    tokens = re.findall(r'[а-яёa-z]{2,}', text.lower())
    return [t for t in tokens if t not in RUSSIAN_STOPWORDS]


# ---------------------------------------------------------------------------
# Tree building (rule-based)
# ---------------------------------------------------------------------------

SECTION_HEADERS = [
    ("introduction", re.compile(
        r'(?:ВВОДНАЯ\s+ЧАСТЬ|Код\(?ы?\)?\s+МКБ|ОБЩАЯ\s+ИНФОРМАЦИЯ|ОПРЕДЕЛЕНИЕ)',
        re.IGNORECASE
    )),
    ("diagnostic_criteria", re.compile(
        r'(?:ДИАГНОСТИЧЕСКИЕ\s+КРИТЕРИИ|ДИАГНОСТИКА\b|КРИТЕРИИ\s+ДИАГНОСТИКИ)',
        re.IGNORECASE
    )),
    ("clinical_signs", re.compile(
        r'(?:КЛИНИЧЕСКИЕ\s+ПРИЗНАКИ|КЛИНИЧЕСКИЕ\s+КРИТЕРИИ|СИМПТОМЫ)',
        re.IGNORECASE
    )),
    ("differential_diagnosis", re.compile(
        r'(?:ДИФФЕРЕНЦИАЛЬНЫЙ\s+ДИАГНОЗ|ДИФФЕРЕНЦИАЛЬНАЯ\s+ДИАГНОСТИКА)',
        re.IGNORECASE
    )),
    ("treatment", re.compile(
        r'(?:ТАКТИКА\s+ЛЕЧЕНИЯ|ЛЕЧЕНИЕ\b|МЕДИКАМЕНТОЗНОЕ\s+ЛЕЧЕНИЕ)',
        re.IGNORECASE
    )),
    ("hospitalization", re.compile(
        r'(?:ПОКАЗАНИЯ\s+ДЛЯ\s+ГОСПИТАЛИЗАЦИИ|ГОСПИТАЛИЗАЦИЯ)',
        re.IGNORECASE
    )),
]

ICD_CODE_RE = re.compile(r'\b([A-Z]\d{2}(?:\.\d{1,2})?)\b')


def _make_summary(text: str, max_chars: int = 300) -> str:
    """Return first sentence(s) up to max_chars as a summary."""
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    summary = ""
    for s in sentences:
        if len(summary) + len(s) > max_chars:
            break
        summary += s + " "
    return summary.strip() or text[:max_chars].strip()


def build_tree_from_protocol(protocol: Protocol) -> TreeNode:
    """Build a hierarchical tree structure from a protocol's text."""
    text = protocol.text

    # Find all section header positions (use match start — text has no newlines)
    section_positions = []
    for stype, pattern in SECTION_HEADERS:
        for m in pattern.finditer(text):
            section_positions.append((m.start(), m.end(), stype, m.group().strip()))

    section_positions.sort(key=lambda x: x[0])

    # Deduplicate: keep first occurrence of each type
    seen_types = set()
    deduped = []  # (content_start, header_end, stype, header)
    for pos_start, pos_end, stype, header in section_positions:
        if stype not in seen_types:
            seen_types.add(stype)
            deduped.append((pos_start, pos_end, stype, header))

    # Root node = entire protocol
    root = TreeNode(
        node_id=f"{protocol.protocol_id}_root",
        title=protocol.title or protocol.source_file,
        summary=_make_summary(text),
        section_type="root",
        text_start=0,
        text_end=len(text),
        icd_codes=list(set(protocol.icd_codes)),
    )

    if not deduped:
        return root

    # Create child nodes for each section (content starts after header end)
    for i, (pos_start, pos_end, stype, header) in enumerate(deduped):
        # Content ends where next section begins
        next_start = deduped[i + 1][0] if i + 1 < len(deduped) else len(text)
        section_text = text[pos_end:next_start].strip()
        node_icd = list(set(ICD_CODE_RE.findall(section_text)))

        child = TreeNode(
            node_id=f"{protocol.protocol_id}_{stype}",
            title=header,
            summary=_make_summary(section_text),
            section_type=stype,
            text_start=pos_end,
            text_end=next_start,
            icd_codes=node_icd,
        )
        root.children.append(child)

    return root


def format_tree_for_prompt(tree: TreeNode, protocol_id: str, icd_codes: list[str]) -> str:
    """Format a tree as a compact text representation for LLM prompts."""
    icd_str = ", ".join(icd_codes[:5]) if icd_codes else "N/A"
    lines = [f"Protocol: {tree.title} ({protocol_id}) | ICD: {icd_str}"]

    for i, child in enumerate(tree.children):
        prefix = "└──" if i == len(tree.children) - 1 else "├──"
        lines.append(
            f"{prefix} [{child.node_id}] {child.title}: {child.summary[:150]}"
        )

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

MAX_CHUNK_CHARS = 4000  # ~1000 tokens


def split_long_text(
    text: str,
    protocol_id: str,
    section_type: str,
    icd_codes: list[str],
    protocol_title: str,
    base_chunk_id: str,
) -> list[Chunk]:
    """Split a long section into overlapping sub-chunks."""
    overlap = 400  # characters
    chunks = []
    start = 0
    part = 0
    while start < len(text):
        end = min(start + MAX_CHUNK_CHARS, len(text))
        chunk_text = text[start:end]
        chunks.append(Chunk(
            chunk_id=f"{base_chunk_id}_part{part}",
            protocol_id=protocol_id,
            protocol_title=protocol_title,
            icd_codes=icd_codes,
            section_type=section_type,
            text=chunk_text,
            node_id=None,
        ))
        if end == len(text):
            break
        start = end - overlap
        part += 1
    return chunks


def build_chunks_from_protocol(protocol: Protocol) -> list[Chunk]:
    """Build section-aware chunks from a protocol."""
    chunks = []

    if protocol.sections:
        for stype, section_text in protocol.sections.items():
            if not section_text.strip():
                continue
            base_id = f"{protocol.protocol_id}_{stype}"
            if len(section_text) <= MAX_CHUNK_CHARS:
                chunks.append(Chunk(
                    chunk_id=base_id,
                    protocol_id=protocol.protocol_id,
                    protocol_title=protocol.title,
                    icd_codes=protocol.icd_codes,
                    section_type=stype,
                    text=section_text,
                    node_id=f"{protocol.protocol_id}_{stype}",
                ))
            else:
                chunks.extend(split_long_text(
                    section_text, protocol.protocol_id, stype,
                    protocol.icd_codes, protocol.title, base_id
                ))
    else:
        # No sections parsed — chunk entire text
        base_id = f"{protocol.protocol_id}_full"
        if len(protocol.text) <= MAX_CHUNK_CHARS:
            chunks.append(Chunk(
                chunk_id=base_id,
                protocol_id=protocol.protocol_id,
                protocol_title=protocol.title,
                icd_codes=protocol.icd_codes,
                section_type="full_text",
                text=protocol.text,
            ))
        else:
            chunks.extend(split_long_text(
                protocol.text, protocol.protocol_id, "full_text",
                protocol.icd_codes, protocol.title, base_id
            ))

    return chunks


# ---------------------------------------------------------------------------
# Index building and loading
# ---------------------------------------------------------------------------

INDEX_DIR = Path("data/indexes")
TREES_DIR = Path("data/trees")


def build_or_load_indexes(protocols: dict[str, Protocol]):
    """Build or load BM25 + FAISS indexes. Returns (bm25, faiss, embed_model, chunks)."""
    INDEX_DIR.mkdir(parents=True, exist_ok=True)

    bm25_path = INDEX_DIR / "bm25.pkl"
    faiss_path = INDEX_DIR / "faiss.index"
    chunks_path = INDEX_DIR / "chunks.pkl"

    if bm25_path.exists() and faiss_path.exists() and chunks_path.exists():
        print("Loading pre-built indexes from disk...")
        return _load_indexes()

    print("Building indexes from scratch...")
    return _build_indexes(protocols)


def _load_indexes():
    bm25_path = INDEX_DIR / "bm25.pkl"
    faiss_path = INDEX_DIR / "faiss.index"
    chunks_path = INDEX_DIR / "chunks.pkl"

    import faiss as faiss_lib

    with open(bm25_path, "rb") as f:
        bm25_data = pickle.load(f)

    bm25_index = bm25_data["bm25"]

    faiss_index = faiss_lib.read_index(str(faiss_path))
    faiss_index = _maybe_gpu_faiss(faiss_index, faiss_lib)

    with open(chunks_path, "rb") as f:
        chunks_list = pickle.load(f)

    # Pass stored model name so sentence-transformers backend uses the same model
    # that was used at index build time. Ignored by api/llama-cpp backends.
    stored_name = bm25_data.get("embed_model_name")
    embed_model = load_embed_model(stored_model_name=stored_name)

    return bm25_index, faiss_index, embed_model, chunks_list


def _maybe_gpu_faiss(faiss_index, faiss_lib):
    """Move FAISS index to GPU for NVIDIA CUDA only (not ROCm — no faiss-gpu for AMD)."""
    try:
        if not _is_rocm():
            import torch
            if torch.cuda.is_available():
                res = faiss_lib.StandardGpuResources()
                gpu_index = faiss_lib.index_cpu_to_gpu(res, 0, faiss_index)
                print("FAISS index moved to NVIDIA GPU")
                return gpu_index
    except Exception as e:
        print(f"FAISS GPU unavailable, using CPU: {e}")
    return faiss_index


def _build_indexes(protocols: dict[str, Protocol]):
    from rank_bm25 import BM25Okapi
    import faiss as faiss_lib

    embed_model = load_embed_model()
    from src.config import EMBED_MODEL
    embed_model_name = EMBED_MODEL  # stored in pkl so _load_indexes reloads same model

    # Build chunks
    print("Building chunks...")
    all_chunks: list[Chunk] = []
    for p in protocols.values():
        all_chunks.extend(build_chunks_from_protocol(p))
    print(f"Total chunks: {len(all_chunks)}")

    # BM25
    print("Building BM25 index...")
    tokenized = [lemmatize(c.text) for c in all_chunks]
    bm25 = BM25Okapi(tokenized)

    # Dense embeddings
    import numpy as np
    print("Building FAISS dense index...")
    embeddings = encode_texts(
        embed_model,
        [c.text for c in all_chunks],
        is_query=False,
        normalize_embeddings=True,
        show_progress_bar=True,
        batch_size=32,
    )
    cpu_faiss_index = faiss_lib.IndexFlatIP(embeddings.shape[1])
    cpu_faiss_index.add(embeddings.astype(np.float32))
    faiss_index = _maybe_gpu_faiss(cpu_faiss_index, faiss_lib)

    # Save
    print("Saving indexes to disk...")
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    with open(INDEX_DIR / "bm25.pkl", "wb") as f:
        pickle.dump({
            "bm25": bm25,
            "tokenized": tokenized,
            "embed_model_name": embed_model_name,
        }, f)
    # Always save the CPU index (GPU indexes cannot be serialized)
    save_index = faiss_lib.index_gpu_to_cpu(cpu_faiss_index) if faiss_index is not cpu_faiss_index else cpu_faiss_index
    faiss_lib.write_index(save_index, str(INDEX_DIR / "faiss.index"))
    with open(INDEX_DIR / "chunks.pkl", "wb") as f:
        pickle.dump(all_chunks, f)

    # Save metadata
    meta = {
        "num_protocols": len(protocols),
        "num_chunks": len(all_chunks),
        "embed_model": embed_model_name,
    }
    with open(INDEX_DIR / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    return bm25, faiss_index, embed_model, all_chunks


def build_or_load_trees(protocols: dict[str, Protocol]) -> dict[str, TreeNode]:
    """Build or load tree structures for all protocols."""
    TREES_DIR.mkdir(parents=True, exist_ok=True)
    trees = {}

    for pid, protocol in protocols.items():
        tree_path = TREES_DIR / f"{pid}.json"
        if tree_path.exists():
            with open(tree_path) as f:
                trees[pid] = dict_to_treenode(json.load(f))
        else:
            tree = build_tree_from_protocol(protocol)
            trees[pid] = tree
            with open(tree_path, "w", encoding="utf-8") as f:
                json.dump(treenode_to_dict(tree), f, ensure_ascii=False, indent=2)

    return trees
