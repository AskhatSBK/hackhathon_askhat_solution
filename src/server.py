"""
Main FastAPI server for the Qazcode Clinical Diagnosis Assistant.
Serves POST /diagnose endpoint and a web UI at GET /.
"""

import os
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()  # load .env before anything reads os.getenv()

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from src.config import make_llm_client
from src.data_loader import (
    load_all_protocols,
    build_icd_lookup,
    get_all_valid_icd_codes,
    load_few_shot_examples,
)
from src.indexer import build_or_load_indexes, build_or_load_trees
from src.retriever import init_retriever, hybrid_search, retrieve
from src.query_analyzer import analyze_query
from src.generator import generate_diagnosis
import src.postprocessor as postprocessor

# ---------------------------------------------------------------------------
# Global state
# ---------------------------------------------------------------------------

_protocols = {}
_few_shot_examples = []
_llm_client = None


# ---------------------------------------------------------------------------
# Lifespan (startup / shutdown)
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    global _protocols, _few_shot_examples, _llm_client

    print("Loading protocols...")
    _protocols = load_all_protocols("data/corpus/")
    icd_to_protocols = build_icd_lookup(_protocols)

    # Populate global valid ICD codes for postprocessor
    postprocessor.ALL_VALID_ICD_CODES.update(
        get_all_valid_icd_codes(_protocols)
    )

    print("Loading indexes...")
    bm25, faiss, embed, chunks = build_or_load_indexes(_protocols)

    print("Loading trees...")
    tree_map = build_or_load_trees(_protocols)

    # Initialise retriever module globals
    init_retriever(bm25, faiss, embed, chunks, tree_map, _protocols, icd_to_protocols)

    print("Loading few-shot examples...")
    _few_shot_examples = load_few_shot_examples("data/test_set/", n=3)

    print("Initialising LLM client...")
    _llm_client = make_llm_client()

    print("Server ready!")
    yield


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(title="Qazcode Clinical Diagnosis Assistant", lifespan=lifespan)


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

class DiagnoseRequest(BaseModel):
    symptoms: str = ""


class DiagnosisItem(BaseModel):
    rank: int
    diagnosis: str
    icd10_code: str
    explanation: str


class DiagnoseResponse(BaseModel):
    diagnoses: list[DiagnosisItem]


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.post("/diagnose", response_model=DiagnoseResponse)
async def diagnose(request: DiagnoseRequest):
    """Main diagnosis endpoint evaluated by evaluate.py."""
    query = request.symptoms or ""

    if not query.strip():
        return DiagnoseResponse(diagnoses=[])

    try:
        # Stage 1: Query analysis
        analysis = await analyze_query(query, _llm_client)

        # Stage 2: Multi-strategy retrieval
        context_chunks, analysis = await retrieve(query, analysis, _llm_client)

        # Stage 3: Diagnosis generation
        diagnoses = await generate_diagnosis(
            query, context_chunks, analysis, _llm_client, _few_shot_examples
        )

        return DiagnoseResponse(
            diagnoses=[DiagnosisItem(**d) for d in diagnoses]
        )

    except Exception as e:
        print(f"Pipeline error: {e}")
        return _fallback_response(query)


def _fallback_response(query: str) -> DiagnoseResponse:
    """BM25-only fallback when the full pipeline fails."""
    try:
        results = hybrid_search([query], top_k=3)
        diagnoses = []
        rank = 1
        seen_codes: set[str] = set()
        for chunk in results:
            for code in chunk.icd_codes:
                if code not in seen_codes and rank <= 3:
                    seen_codes.add(code)
                    diagnoses.append(DiagnosisItem(
                        rank=rank,
                        diagnosis=chunk.protocol_title or "Неизвестный диагноз",
                        icd10_code=code,
                        explanation=chunk.text[:200],
                    ))
                    rank += 1
        if diagnoses:
            return DiagnoseResponse(diagnoses=diagnoses)
    except Exception:
        pass
    return DiagnoseResponse(diagnoses=[])


@app.get("/", response_class=HTMLResponse)
async def web_ui():
    """Serve the web UI."""
    ui_path = Path("src/static/index.html")
    if ui_path.exists():
        return ui_path.read_text(encoding="utf-8")
    return "<h1>Qazcode Diagnosis API</h1><p>POST /diagnose with {\"symptoms\": \"...\"}</p>"


@app.get("/health")
async def health():
    return {"status": "ok", "protocols": len(_protocols)}
