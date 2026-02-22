"""
3-strategy retrieval with RRF fusion:
  A: PageIndex tree navigation (LLM over compact tree summaries)
  B: Hybrid BM25 + dense with RRF
  C: ICD-10 direct lookup
"""

import json
import re
from dataclasses import dataclass
from typing import Optional

from src.config import LLM_MODEL, prepare_system
from src.indexer import Chunk, TreeNode, lemmatize, format_tree_for_prompt, encode_texts


# ---------------------------------------------------------------------------
# Global state (set by server.py at startup)
# ---------------------------------------------------------------------------

bm25_index = None
faiss_index = None
embed_model = None
chunks_list: list[Chunk] = []
trees: dict[str, TreeNode] = {}
protocols_map = {}  # protocol_id -> Protocol
icd_to_protocols: dict[str, list] = {}


def init_retriever(
    bm25, faiss, embed, chunks, tree_map, proto_map, icd_map
):
    global bm25_index, faiss_index, embed_model, chunks_list
    global trees, protocols_map, icd_to_protocols
    bm25_index = bm25
    faiss_index = faiss
    embed_model = embed
    chunks_list = chunks
    trees = tree_map
    protocols_map = proto_map
    icd_to_protocols = icd_map


# ---------------------------------------------------------------------------
# Strategy B: Hybrid BM25 + Dense with RRF
# ---------------------------------------------------------------------------

def hybrid_search(queries: list[str], top_k: int = 15) -> list[Chunk]:
    """Run multiple queries through BM25 and dense search, fuse with RRF."""
    import numpy as np
    if not chunks_list:
        return []

    all_results: dict[int, dict] = {}

    for query in queries:
        # Dense search
        q_emb = encode_texts(
            embed_model,
            [query],
            is_query=True,
            normalize_embeddings=True,
        )
        scores, indices = faiss_index.search(q_emb.astype(np.float32), top_k * 2)
        for rank, idx in enumerate(indices[0]):
            idx = int(idx)
            if idx < 0 or idx >= len(chunks_list):
                continue
            entry = all_results.setdefault(idx, {"dense_ranks": [], "bm25_ranks": []})
            entry["dense_ranks"].append(rank)

        # BM25 search
        q_tokens = lemmatize(query)
        if q_tokens:
            bm25_scores = bm25_index.get_scores(q_tokens)
            bm25_top = np.argsort(bm25_scores)[::-1][:top_k * 2]
            for rank, idx in enumerate(bm25_top):
                idx = int(idx)
                entry = all_results.setdefault(idx, {"dense_ranks": [], "bm25_ranks": []})
                entry["bm25_ranks"].append(rank)

    # RRF fusion
    k = 60
    rrf_scores: dict[int, float] = {}
    for idx, ranks in all_results.items():
        score = sum(1.0 / (k + r) for r in ranks["dense_ranks"])
        score += sum(1.0 / (k + r) for r in ranks["bm25_ranks"])
        rrf_scores[idx] = score

    sorted_indices = sorted(rrf_scores, key=rrf_scores.get, reverse=True)[:top_k]
    return [chunks_list[i] for i in sorted_indices]


# ---------------------------------------------------------------------------
# Strategy C: ICD-10 Direct Lookup
# ---------------------------------------------------------------------------

def icd_lookup(candidate_icd_codes: list[str]) -> list[Chunk]:
    """Direct lookup by hypothesized ICD codes."""
    results = []
    seen_protocol_ids = set()

    for code in candidate_icd_codes:
        code = code.strip().upper()
        # Match exact code or by prefix (code[:3])
        for full_code, protocols in icd_to_protocols.items():
            if full_code == code or full_code.startswith(code[:3]):
                for p in protocols:
                    if p.protocol_id in seen_protocol_ids:
                        continue
                    seen_protocol_ids.add(p.protocol_id)

                    # Prefer diagnostic_criteria section
                    section_text = (
                        p.sections.get("diagnostic_criteria", "")
                        or p.sections.get("clinical_signs", "")
                        or p.text
                    )
                    if section_text:
                        results.append(Chunk(
                            chunk_id=f"{p.protocol_id}_icd_lookup",
                            protocol_id=p.protocol_id,
                            protocol_title=p.title,
                            icd_codes=p.icd_codes,
                            section_type="diagnostic_criteria",
                            text=section_text[:3000],
                        ))
        if len(results) >= 10:
            break

    return results[:10]


# ---------------------------------------------------------------------------
# Strategy A: PageIndex Tree Navigation
# ---------------------------------------------------------------------------

async def tree_search(
    query: str,
    candidate_chunks: list[Chunk],
    llm_client,
) -> list[Chunk]:
    """Use LLM to navigate tree structures and select relevant sections."""
    if not trees or not candidate_chunks:
        return []

    # Get unique protocols from candidate chunks
    seen = set()
    candidate_pids = []
    for c in candidate_chunks:
        if c.protocol_id not in seen:
            seen.add(c.protocol_id)
            candidate_pids.append(c.protocol_id)
        if len(candidate_pids) >= 10:
            break

    # Build compact tree summaries
    tree_summaries = []
    for pid in candidate_pids:
        if pid not in trees:
            continue
        tree = trees[pid]
        protocol = protocols_map.get(pid)
        icd_codes = protocol.icd_codes if protocol else []
        tree_summaries.append(format_tree_for_prompt(tree, pid, icd_codes))

    if not tree_summaries:
        return []

    prompt = f"""You are a medical search assistant. Given patient symptoms, identify which protocol sections contain the relevant diagnostic criteria.

Patient symptoms: {query}

Available protocol sections:
{chr(10).join(tree_summaries)}

For each relevant section, return a JSON array:
[{{"protocol_id": "...", "node_id": "...", "relevance": "high/medium"}}]

Return ONLY the JSON array, nothing else. Select 3-8 most relevant sections."""

    try:
        response = await llm_client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": prepare_system("Reasoning: low\nYou are a medical protocol search assistant. Return only valid JSON array."),
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.1,
            max_tokens=600,
        )

        raw = response.choices[0].message.content.strip()
        raw = re.sub(r'```json\s*', '', raw)
        raw = re.sub(r'```\s*', '', raw)

        # Extract JSON array
        match = re.search(r'\[.*\]', raw, re.DOTALL)
        if not match:
            return []
        selected_nodes = json.loads(match.group())

        results = []
        for node_sel in selected_nodes[:8]:
            pid = node_sel.get("protocol_id", "")
            node_id = node_sel.get("node_id", "")
            protocol = protocols_map.get(pid)
            if not protocol:
                continue

            # Get text for this node from trees
            tree = trees.get(pid)
            section_text = _get_node_text(tree, node_id, protocol.text)
            if not section_text:
                continue

            results.append(Chunk(
                chunk_id=f"{pid}_{node_id}_tree",
                protocol_id=pid,
                protocol_title=protocol.title,
                icd_codes=protocol.icd_codes,
                section_type=node_id.replace(f"{pid}_", ""),
                text=section_text[:3000],
                node_id=node_id,
            ))

        return results

    except Exception as e:
        print(f"Tree search failed: {e}")
        return []


def _get_node_text(tree: Optional[TreeNode], node_id: str, full_text: str) -> str:
    """Extract text for a specific tree node."""
    if tree is None:
        return ""

    if tree.node_id == node_id:
        return full_text[tree.text_start:tree.text_end]

    for child in tree.children:
        result = _get_node_text(child, node_id, full_text)
        if result:
            return result

    return ""


# ---------------------------------------------------------------------------
# Fusion and deduplication
# ---------------------------------------------------------------------------

def deduplicate_chunks(chunks: list[Chunk]) -> list[Chunk]:
    """Deduplicate chunks by protocol_id + section_type, keep first occurrence."""
    seen = set()
    result = []
    for c in chunks:
        key = (c.protocol_id, c.section_type)
        if key not in seen:
            seen.add(key)
            result.append(c)
    return result


def select_top_chunks(
    chunks: list[Chunk],
    max_chunks: int = 8,
    max_chars: int = 18000,
) -> list[Chunk]:
    """Select top chunks by source priority within token budget."""
    # Source priority: tree_search > icd_lookup > section-aware > rest
    priority = {"_tree": 0, "icd_lookup": 1, "diagnostic_criteria": 2}

    def chunk_priority(c: Chunk) -> int:
        if c.chunk_id.endswith("_tree"):
            return 0
        if "icd_lookup" in c.chunk_id:
            return 1
        if c.section_type == "diagnostic_criteria":
            return 2
        return 3

    sorted_chunks = sorted(chunks, key=chunk_priority)

    selected = []
    total_chars = 0
    for c in sorted_chunks:
        if len(selected) >= max_chunks:
            break
        if total_chars + len(c.text) > max_chars:
            continue
        selected.append(c)
        total_chars += len(c.text)

    return selected


# ---------------------------------------------------------------------------
# Main retrieval function
# ---------------------------------------------------------------------------

async def retrieve(
    query: str,
    analysis: dict,
    llm_client,
) -> tuple[list[Chunk], dict]:
    """Main retrieval: runs all 3 strategies and fuses results."""

    sub_queries = analysis.get("sub_queries", [query])
    candidate_icd_codes = analysis.get("candidate_icd_codes", [])
    hyde_passage = analysis.get("hyde_passage", "")

    # Include HyDE passage as an extra query: it's written in protocol language,
    # which matches indexed protocol text far better than raw patient descriptions.
    all_queries = [query] + sub_queries
    if hyde_passage:
        all_queries.append(hyde_passage)

    # Strategy B: Hybrid search (fast, no LLM)
    hybrid_results = hybrid_search(queries=all_queries, top_k=20)

    # Strategy C: ICD direct lookup (fast, no LLM)
    icd_results = icd_lookup(candidate_icd_codes) if candidate_icd_codes else []

    # Strategy A: Tree search over top candidates from hybrid
    tree_results = await tree_search(query, hybrid_results[:10], llm_client)

    # Merge: tree first (highest priority), then ICD lookup, then hybrid
    all_chunks = tree_results + icd_results + hybrid_results
    deduped = deduplicate_chunks(all_chunks)
    final_context = select_top_chunks(deduped, max_chunks=8, max_chars=18000)

    return final_context, analysis
