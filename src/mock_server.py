"""
Mock diagnostic server that returns random ICD-10 codes.

Usage:
    uv run uvicorn src.mock_server:app --host 127.0.0.1 --port 8000

Docker:
    docker build -t mock-server .
    docker run -p 8000:8000 mock-server

Runs on http://127.0.0.1:8000/diagnose
"""

import random
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI
from pydantic import BaseModel


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("\n🏥 Mock Diagnostic Server (FastAPI)")
    print("=" * 40)
    print("Endpoint: /diagnose")
    print("Method:   POST")
    print('Body:     {"symptoms": "..."}')
    print("Docs:     /docs")
    print("=" * 40)
    print("\nPress Ctrl+C to stop\n")
    yield


app = FastAPI(title="Mock Diagnostic Server", lifespan=lifespan)

ICD_CODES = [
    "A00.13",
    "A01.14",
    "A02.15",
    "A03.16",
    "A04.17",
    "A05.18",
    "A06",
    "A07.19",
    "A08",
    "A09.20",
    "B00",
    "B01.21",
    "B02.22",
    "B03",
    "B04.23",
    "B05.24",
    "J00",
    "J01.25",
    "J02",
    "J03.26",
    "J04",
    "J05",
    "J06",
    "K00",
    "K01",
    "K02",
    "K03",
    "K04",
    "K05",
    "L00",
    "L01",
    "L02",
    "L03",
    "L04",
    "M00",
    "M01",
    "M02",
    "M03",
    "N00",
    "N01",
    "N02",
    "N03",
]


class DiagnoseRequest(BaseModel):
    symptoms: Optional[str] = ""


class Diagnosis(BaseModel):
    rank: int
    diagnosis: str
    icd10_code: str
    explanation: str


class DiagnoseResponse(BaseModel):
    diagnoses: list[Diagnosis]


@app.post("/diagnose", response_model=DiagnoseResponse)
async def handle_diagnose(request: DiagnoseRequest) -> DiagnoseResponse:
    """Handle POST /diagnose requests with random diagnoses."""
    symptoms = request.symptoms or ""

    codes = random.sample(ICD_CODES, min(5, len(ICD_CODES)))
    diagnoses = []

    for rank, code in enumerate(codes, start=1):
        diagnoses.append(
            Diagnosis(
                rank=rank,
                diagnosis=f"Simulated diagnosis for {code}",
                icd10_code=code,
                explanation=f"Based on symptoms: {symptoms[:100]}..."
                if symptoms
                else "No symptoms provided",
            )
        )

    return DiagnoseResponse(diagnoses=diagnoses)
