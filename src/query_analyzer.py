"""
Stage 1: Query analysis and decomposition using GPT-OSS.
Decomposes patient symptom queries into sub-queries, extracts symptoms,
hypothesizes candidate ICD-10 codes, and generates a HyDE passage.
"""

import json
import re

from src.config import LLM_MODEL, prepare_system


async def analyze_query(query: str, llm_client) -> dict:
    """
    Lightweight LLM call to decompose and enrich the query.
    Also generates a HyDE (Hypothetical Document Embedding) passage —
    a fake protocol diagnostic-criteria excerpt that semantically matches
    what the correct protocol should look like.
    """
    prompt = f"""You are a medical search query optimizer for Kazakhstan clinical protocols (Клинические протоколы Казахстана).

Given patient symptoms in Russian or Kazakh, produce:
1. "sub_queries": 3 different search queries in Russian, each targeting a different angle:
   - Query 1: Focused on the main symptom complex and mechanism (e.g. trauma, infection)
   - Query 2: Using formal medical/clinical terminology for the suspected condition
   - Query 3: Focused on differential diagnosis keywords
2. "normalized_symptoms": Extract individual symptoms as a list (in Russian)
3. "candidate_icd_codes": 5-8 most likely ICD-10 code prefixes (e.g., "J03", "I10", "S22")
4. "hyde_passage": Write a 80-120 word FAKE clinical protocol excerpt in Russian that describes
   the DIAGNOSTIC CRITERIA for the most likely condition. Write it as if it were the
   "ДИАГНОСТИЧЕСКИЕ КРИТЕРИИ" section of the relevant Kazakhstan clinical protocol.
   Include: typical complaints, physical findings, and clinical signs that match the symptoms.

Patient symptoms: "{query}"

Return ONLY valid JSON, no markdown, no explanation:
{{"sub_queries": [...], "normalized_symptoms": [...], "candidate_icd_codes": [...], "hyde_passage": "..."}}"""

    try:
        response = await llm_client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": prepare_system("Reasoning: low\nReturn only valid JSON.")},
                {"role": "user", "content": prompt},
            ],
            temperature=0.1,
            max_tokens=900,
        )

        raw = response.choices[0].message.content.strip()
        # Strip markdown if present
        raw = re.sub(r'```json\s*', '', raw)
        raw = re.sub(r'```\s*', '', raw)

        result = json.loads(raw)

        return {
            "sub_queries": result.get("sub_queries", [query]),
            "normalized_symptoms": result.get("normalized_symptoms", [query]),
            "candidate_icd_codes": result.get("candidate_icd_codes", []),
            "hyde_passage": result.get("hyde_passage", ""),
        }

    except Exception as e:
        print(f"Query analysis failed: {e}")
        return {
            "sub_queries": [query],
            "normalized_symptoms": [query],
            "candidate_icd_codes": [],
            "hyde_passage": "",
        }
