"""
Stage 3: Diagnosis generation using GPT-OSS with Reasoning: high.
Takes retrieved context chunks and generates ranked diagnoses with ICD-10 codes.
"""

import json

from src.config import LLM_MODEL, prepare_system
from src.indexer import Chunk
from src.postprocessor import (
    parse_diagnosis_json,
    validate_icd_codes,
    add_ranks,
    format_few_shot_examples,
    ALL_VALID_ICD_CODES,
)


async def generate_diagnosis(
    query: str,
    context_chunks: list[Chunk],
    analysis: dict,
    llm_client,
    few_shot_examples: list[dict],
) -> list[dict]:
    """
    Call GPT-OSS with high reasoning to produce final ranked diagnoses.
    Returns list of dicts with keys: rank, diagnosis, icd10_code, explanation.
    """

    # Build context string and collect available ICD codes
    context_parts = []
    available_icd_codes: set[str] = set()

    for chunk in context_chunks:
        context_parts.append(
            f"[Protocol: {chunk.protocol_id} | ICD: {', '.join(chunk.icd_codes)} | "
            f"Section: {chunk.section_type}]\n{chunk.text}"
        )
        available_icd_codes.update(chunk.icd_codes)

    context_str = "\n\n---\n\n".join(context_parts)
    icd_list = ", ".join(sorted(available_icd_codes))

    examples_str = format_few_shot_examples(few_shot_examples)

    system_prompt = """Reasoning: high

You are an expert physician specializing in Kazakhstan clinical protocols (Клинические протоколы Республики Казахстан).

YOUR TASK: Given patient symptoms and clinical protocol excerpts, identify the most likely diagnoses.

STEP-BY-STEP APPROACH:
1. Identify the PRIMARY complaint and mechanism: trauma? infection? chronic condition? which organ system?
2. For each retrieved protocol excerpt, score how well its diagnostic criteria match the symptoms
3. Rank diagnoses by that match score — Rank 1 must have the strongest symptom-criteria alignment
4. Use the ICD-10 code from the best-matching protocol excerpt

RULES:
- Do NOT assign a diagnosis that contradicts the clear clinical picture
  (e.g. do NOT diagnose diabetes for a patient with a clear trauma injury)
- If NONE of the retrieved excerpts match the symptoms, you may use any correct ICD-10 code
  from the Kazakhstan clinical protocol corpus based on your medical knowledge
- The ICD-10 code must belong to the Kazakhstan clinical protocol corpus
- Return valid JSON only — no markdown, no extra text

OUTPUT FORMAT:
{
  "diagnoses": [
    {
      "rank": 1,
      "diagnosis": "Disease name in Russian",
      "icd10_code": "X00.0",
      "explanation": "Which specific patient symptoms match which diagnostic criteria from the protocol"
    }
  ]
}

Return 3-5 diagnoses. Rank 1 = most likely based on symptom-criteria match."""

    user_prompt = f"""## Patient Symptoms
{query}

## Extracted Symptoms
{', '.join(analysis.get('normalized_symptoms', []))}

## Clinical Protocol Excerpts
{context_str}

## ICD-10 Codes in Retrieved Protocols
{icd_list}

## Reference Examples
{examples_str}

Which clinical condition best explains these symptoms? Match to the protocol excerpts above and return JSON."""

    try:
        response = await llm_client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": prepare_system(system_prompt)},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.2,
            max_tokens=2000,
        )

        raw_text = response.choices[0].message.content
        print(f"[DEBUG LLM] raw_text[:300]: {raw_text[:300]!r}")
        diagnoses_raw = parse_diagnosis_json(raw_text)
        print(f"[DEBUG LLM] parse_diagnosis_json -> {len(diagnoses_raw)} items")

        if not diagnoses_raw:
            print("[DEBUG LLM] FALLBACK: parse returned empty")
            return _fallback_from_context(context_chunks)

        # Validate ICD codes against corpus
        validated = validate_icd_codes(
            diagnoses_raw,
            available_icd_codes,
            {},  # icd_lookup dict not needed here
        )
        print(f"[DEBUG LLM] validate_icd_codes -> {len(validated)} items (context_codes={sorted(available_icd_codes)[:5]})")

        if not validated:
            print("[DEBUG LLM] FALLBACK: validate returned empty")
            return _fallback_from_context(context_chunks)

        # Ensure ranks are correct
        return add_ranks(validated)

    except Exception as e:
        print(f"Diagnosis generation failed: {e}")
        return _fallback_from_context(context_chunks)


def _fallback_from_context(context_chunks: list[Chunk]) -> list[dict]:
    """Emergency fallback: return top protocols from retrieved context."""
    seen_codes = set()
    diagnoses = []
    rank = 1

    for chunk in context_chunks[:5]:
        for code in chunk.icd_codes:
            if code not in seen_codes and rank <= 5:
                seen_codes.add(code)
                diagnoses.append({
                    "rank": rank,
                    "diagnosis": chunk.protocol_title or "Неизвестный диагноз",
                    "icd10_code": code,
                    "explanation": chunk.text[:200],
                })
                rank += 1

    return diagnoses
