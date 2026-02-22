"""
Post-processing: JSON parsing, ICD-10 code validation and correction.
"""

import json
import re

# Populated at startup by server.py
ALL_VALID_ICD_CODES: set[str] = set()


def parse_diagnosis_json(raw_text: str) -> list[dict]:
    """Parse LLM output, handling common failure modes."""
    # Strip markdown code blocks if present
    text = re.sub(r'```json\s*', '', raw_text)
    text = re.sub(r'```\s*', '', text)
    text = text.strip()

    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict) and "diagnoses" in parsed:
            return parsed["diagnoses"]
        elif isinstance(parsed, list):
            return parsed
    except json.JSONDecodeError:
        pass

    # Try to extract JSON object from the text
    match = re.search(r'\{.*\}', text, re.DOTALL)
    if match:
        try:
            parsed = json.loads(match.group())
            if isinstance(parsed, dict) and "diagnoses" in parsed:
                return parsed["diagnoses"]
        except Exception:
            pass

    # Try to extract JSON array
    match = re.search(r'\[.*\]', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except Exception:
            pass

    return []


def validate_icd_codes(
    diagnoses: list[dict],
    context_codes: set[str],
    icd_lookup: dict,
) -> list[dict]:
    """
    Validate ICD-10 codes against the Kazakhstan clinical protocol corpus.

    Strategy:
    1. Exact match in the full corpus — keep as-is (preferred)
    2. Code like "J03" that is a valid 3-char prefix: expand to the most specific
       matching code in context_codes (retrieved protocols), else in full corpus
    3. Completely unknown code — drop it (do not remap to an unrelated code)
    4. Remove duplicates
    """
    seen_codes = set()
    validated = []

    for d in diagnoses:
        code = d.get("icd_code", "") or d.get("icd10_code", "")
        code = code.strip().upper()

        if not code:
            continue

        # Normalize field to icd10_code (match evaluate.py expectation)
        d_normalized = {
            "diagnosis": d.get("diagnosis", d.get("name", "Unknown")),
            "icd10_code": code,
            "explanation": d.get("explanation", ""),
        }

        # 1. Exact match in corpus — accept directly
        if code in ALL_VALID_ICD_CODES:
            if code not in seen_codes:
                seen_codes.add(code)
                validated.append(d_normalized)
            continue

        # 2. Prefix expansion only when the code is a 3-char prefix (e.g. "J03", "S22")
        #    and matches at least one corpus code with that exact prefix.
        #    Never remap a specific code (e.g. "J03.5") to a different one.
        if len(code) == 3:
            # Try context codes first (most relevant to this query)
            for valid_code in sorted(context_codes):
                if valid_code.startswith(code) and valid_code not in seen_codes:
                    d_normalized["icd10_code"] = valid_code
                    seen_codes.add(valid_code)
                    validated.append(d_normalized)
                    break
            else:
                # Fall back to full corpus prefix match
                for valid_code in sorted(ALL_VALID_ICD_CODES):
                    if valid_code.startswith(code) and valid_code not in seen_codes:
                        d_normalized["icd10_code"] = valid_code
                        seen_codes.add(valid_code)
                        validated.append(d_normalized)
                        break
            continue

        # 3. Specific code not in corpus — drop rather than remap to wrong disease
        print(f"[validate] Dropping unknown ICD code: {code!r}")

    return validated


def add_ranks(diagnoses: list[dict]) -> list[dict]:
    """Add rank field to diagnoses list (1-indexed)."""
    for i, d in enumerate(diagnoses):
        d["rank"] = i + 1
    return diagnoses


def format_few_shot_examples(examples: list[dict]) -> str:
    """Format few-shot examples for the diagnosis prompt."""
    if not examples:
        return "No examples provided."

    parts = []
    for i, ex in enumerate(examples, 1):
        parts.append(
            f"Example {i}:\n"
            f"Symptoms: {ex['query'][:300]}\n"
            f"Correct ICD-10: {ex['gt']}\n"
            f"Valid codes in protocol: {', '.join(ex.get('icd_codes', [])[:5])}"
        )
    return "\n\n".join(parts)
