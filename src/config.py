"""
Inference provider configuration for LLM and Embedding.

Switch providers by changing .env — no code changes needed.

LLM providers  (LLM_PROVIDER):
  oss      – OpenAI-compatible custom endpoint (default, DataSaur competition)
  openai   – OpenAI API
  gemini   – Google AI Studio
  together – Together AI
  groq     – Groq
  runpod   – RunPod serverless (needs LLM_BASE_URL)
  custom   – Any OpenAI-compatible endpoint (needs LLM_BASE_URL)

Embedding providers  (EMBED_PROVIDER):
  local    – sentence-transformers in-process on CPU (default)
  openai   – OpenAI Embeddings API
  together – Together AI embeddings
  runpod   – RunPod serverless embedding (needs EMBED_BASE_URL)
  custom   – Any OpenAI-compatible /v1/embeddings endpoint (needs EMBED_BASE_URL)
"""

import os
import re

# ---------------------------------------------------------------------------
# LLM
# ---------------------------------------------------------------------------

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "oss").lower()

# (preset_base_url, default_model)
# preset_base_url=None means the URL must be supplied via LLM_BASE_URL env var
_LLM_PRESETS: dict[str, tuple[str | None, str | None]] = {
    "oss":      (None,                                                         "oss-120b"),
    "openai":   ("https://api.openai.com/v1",                                 "gpt-4o-mini"),
    "gemini":   ("https://generativelanguage.googleapis.com/v1beta/openai/",  "gemini-2.5-flash"),
    "together": ("https://api.together.xyz/v1",                               "meta-llama/Llama-3.3-70B-Instruct-Turbo-Free"),
    "groq":     ("https://api.groq.com/openai/v1",                            "llama-3.3-70b-versatile"),
    "runpod":   (None,                                                         None),
    "custom":   (None,                                                         None),
}

_llm_preset_url, _llm_preset_model = _LLM_PRESETS.get(LLM_PROVIDER, (None, None))

LLM_MODEL = os.getenv("LLM_MODEL") or _llm_preset_model or "oss-120b"

# Matches "Reasoning: high\n" / "Reasoning: low\n" — oss-120b specific prefix
_REASONING_RE = re.compile(r"^Reasoning:\s*\w+\n", re.MULTILINE)


def _llm_base_url() -> str:
    if _llm_preset_url:
        return _llm_preset_url
    # oss uses legacy env var for backwards compat
    if LLM_PROVIDER == "oss":
        return os.getenv("GPT_OSS_BASE_URL", "")
    return os.getenv("LLM_BASE_URL", "")


def _llm_api_key() -> str:
    # LLM_API_KEY is the universal key; provider-specific vars are fallbacks
    explicit = os.getenv("LLM_API_KEY", "")
    if explicit:
        return explicit
    if LLM_PROVIDER == "oss":
        return os.getenv("GPT_OSS_API_KEY", "placeholder")
    if LLM_PROVIDER == "gemini":
        return os.getenv("GOOGLE_AI_API_KEY", "")
    return "placeholder"


def make_llm_client():
    """Create an AsyncOpenAI client for the configured LLM provider."""
    from openai import AsyncOpenAI

    base_url = _llm_base_url()
    if not base_url:
        raise ValueError(
            f"No base URL for LLM_PROVIDER={LLM_PROVIDER!r}. "
            "Set LLM_BASE_URL in .env (or GPT_OSS_BASE_URL for oss)."
        )
    return AsyncOpenAI(
        base_url=base_url,
        api_key="placeholder",  # required by the client; actual auth via header below
        default_headers={"x-litellm-api-key": _llm_api_key()},
    )


def prepare_system(content: str) -> str:
    """Strip 'Reasoning: xxx\\n' prefix — only oss-120b understands it."""
    if LLM_PROVIDER == "oss":
        return content
    return _REASONING_RE.sub("", content, count=1)


# ---------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------

EMBED_PROVIDER = os.getenv("EMBED_PROVIDER", "local").lower()

# (preset_base_url, default_model)
# preset_base_url=None + provider="local" → sentence-transformers in-process
# preset_base_url=None + other providers  → URL must come from EMBED_BASE_URL
_EMBED_PRESETS: dict[str, tuple[str | None, str | None]] = {
    "local":      (None,                                         "Qwen/Qwen3-Embedding-0.6B"),
    "llama-cpp":  (None,                                         "Qwen3-Embedding-0.6B-Q4_K_M.gguf"),
    "ollama":     ("http://localhost:11434/api/embed",           "nomic-embed-text"),
    "openai":     ("https://api.openai.com/v1",                 "text-embedding-3-small"),
    "together":   ("https://api.together.xyz/v1",               "togethercomputer/m2-bert-80M-32k"),
    "runpod":     (None,                                         None),
    "custom":     (None,                                         None),
}

# llama-cpp specific: path to the .gguf file and context length
EMBED_MODEL_PATH: str = os.getenv("EMBED_MODEL_PATH", "")
EMBED_N_CTX: int = int(os.getenv("EMBED_N_CTX", "2048"))

_embed_preset_url, _embed_preset_model = _EMBED_PRESETS.get(EMBED_PROVIDER, (None, None))

EMBED_MODEL = os.getenv("EMBED_MODEL") or _embed_preset_model or "Qwen/Qwen3-Embedding-0.6B"


def get_embed_base_url() -> str:
    # Explicit env var always wins over preset (needed e.g. for Ollama inside Docker)
    explicit = os.getenv("EMBED_BASE_URL", os.getenv("EMBED_API_BASE", ""))
    if explicit:
        return explicit
    return _embed_preset_url or ""


def get_embed_api_key() -> str:
    explicit = os.getenv("EMBED_API_KEY", "")
    if explicit:
        return explicit
    # Reuse LLM key when both providers are the same service (e.g. both openai)
    if EMBED_PROVIDER == LLM_PROVIDER:
        return _llm_api_key()
    return "placeholder"
