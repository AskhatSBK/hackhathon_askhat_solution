"""
Data loader for Kazakhstan clinical protocol JSONL corpus.
Loads protocols, parses sections, and builds ICD-10 lookup tables.
"""

import json
import re
from dataclasses import dataclass, field
from pathlib import Path


# Russian stopwords for BM25
RUSSIAN_STOPWORDS = {
    "и", "в", "во", "не", "что", "он", "на", "я", "с", "со", "как", "а", "то",
    "все", "она", "так", "его", "но", "да", "ты", "к", "у", "же", "вы", "за",
    "бы", "по", "только", "ее", "мне", "было", "вот", "от", "меня", "еще", "нет",
    "о", "из", "ему", "теперь", "когда", "даже", "ну", "вдруг", "ли", "если",
    "уже", "или", "ни", "быть", "был", "него", "до", "вас", "нибудь", "опять",
    "уж", "вам", "ведь", "там", "потом", "себя", "ничего", "ей", "может", "они",
    "тут", "где", "есть", "надо", "ней", "для", "мы", "тебя", "их", "чем",
    "была", "сам", "чтоб", "без", "будто", "чего", "раз", "тоже", "себе",
    "под", "будет", "ж", "тогда", "кто", "этот", "того", "потому", "этого",
    "какой", "совсем", "ним", "здесь", "этом", "один", "почти", "мой", "тем",
    "чтобы", "нее", "сейчас", "были", "куда", "зачем", "всех", "никогда",
    "можно", "при", "наконец", "два", "об", "другой", "хоть", "после", "над",
    "больше", "тот", "через", "эти", "нас", "про", "всего", "них", "какая",
    "много", "разве", "три", "эту", "моя", "впрочем", "хорошо", "свою",
    "этой", "перед", "иногда", "лучше", "чуть", "том", "нельзя", "такой",
    "им", "более", "всегда", "конечно", "всю", "между",
}

# Section header patterns (ordered by priority)
SECTION_PATTERNS = [
    ("diagnostic_criteria", re.compile(
        r'(?:ДИАГНОСТИЧЕСКИЕ\s+КРИТЕРИИ|диагностические\s+критерии|'
        r'ДИАГНОСТИКА|диагностика\b|КРИТЕРИИ\s+ДИАГНОСТИКИ)',
        re.IGNORECASE
    )),
    ("clinical_signs", re.compile(
        r'(?:КЛИНИЧЕСКИЕ\s+ПРИЗНАКИ|клинические\s+признаки|'
        r'КЛИНИЧЕСКИЕ\s+КРИТЕРИИ|клинические\s+критерии|'
        r'СИМПТОМЫ|симптомы)',
        re.IGNORECASE
    )),
    ("differential_diagnosis", re.compile(
        r'(?:ДИФФЕРЕНЦИАЛЬНЫЙ\s+ДИАГНОЗ|дифференциальный\s+диагноз|'
        r'ДИФФЕРЕНЦИАЛЬНАЯ\s+ДИАГНОСТИКА)',
        re.IGNORECASE
    )),
    ("treatment", re.compile(
        r'(?:ТАКТИКА\s+ЛЕЧЕНИЯ|тактика\s+лечения|ЛЕЧЕНИЕ\b|лечение\b|'
        r'МЕДИКАМЕНТОЗНОЕ\s+ЛЕЧЕНИЕ)',
        re.IGNORECASE
    )),
    ("hospitalization", re.compile(
        r'(?:ПОКАЗАНИЯ\s+ДЛЯ\s+ГОСПИТАЛИЗАЦИИ|показания\s+для\s+госпитализации|'
        r'ГОСПИТАЛИЗАЦИЯ)',
        re.IGNORECASE
    )),
    ("introduction", re.compile(
        r'(?:ВВОДНАЯ\s+ЧАСТЬ|вводная\s+часть|Код\(?ы?\)?\s+МКБ|'
        r'ОБЩАЯ\s+ИНФОРМАЦИЯ|ОПРЕДЕЛЕНИЕ)',
        re.IGNORECASE
    )),
]

ICD_CODE_PATTERN = re.compile(r'\b([A-Z]\d{2}(?:\.\d{1,2})?)\b')


@dataclass
class Protocol:
    protocol_id: str
    source_file: str
    title: str
    icd_codes: list[str]
    text: str
    sections: dict[str, str] = field(default_factory=dict)
    all_icd_codes: list[str] = field(default_factory=list)  # extracted from text


def parse_sections(text: str) -> dict[str, str]:
    """Parse protocol text into named sections using regex header detection."""
    sections = {}

    # Find all section header positions (use match start, not line start — text has no newlines)
    matches = []
    for section_type, pattern in SECTION_PATTERNS:
        for m in pattern.finditer(text):
            matches.append((m.start(), m.end(), section_type, m.group()))

    if not matches:
        # No sections found — treat entire text as diagnostic_criteria
        sections["full_text"] = text
        return sections

    # Sort by match start position
    matches.sort(key=lambda x: x[0])

    # Deduplicate — keep first occurrence of each section type
    seen = set()
    deduped = []
    for pos_start, pos_end, stype, header in matches:
        if stype not in seen:
            seen.add(stype)
            # Section content starts after the header text
            deduped.append((pos_end, stype))

    # Extract text between section boundaries
    for i, (pos, stype) in enumerate(deduped):
        if i + 1 < len(deduped):
            next_pos = deduped[i + 1][0]
            section_text = text[pos:next_pos].strip()
        else:
            section_text = text[pos:].strip()
        sections[stype] = section_text

    # Everything before first section = introduction (if not already found)
    if deduped and "introduction" not in sections:
        intro_text = text[:deduped[0][0]].strip()
        if intro_text:
            sections["introduction"] = intro_text

    return sections


def extract_icd_codes(text: str) -> list[str]:
    """Extract all ICD-10 codes mentioned in text."""
    codes = ICD_CODE_PATTERN.findall(text)
    return list(dict.fromkeys(codes))  # unique, order-preserving


def load_all_protocols(corpus_path: str) -> dict[str, "Protocol"]:
    """Load all protocols from JSONL file. Returns dict keyed by protocol_id."""
    path = Path(corpus_path)

    # Support both directory (with protocols_corpus.jsonl) and direct file path
    if path.is_dir():
        jsonl_file = path / "protocols_corpus.jsonl"
    else:
        jsonl_file = path

    protocols = {}
    with open(jsonl_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                raw_title = data.get("title", "").strip()
                # The corpus "title" field contains approval status ("Одобрен"), not
                # the disease name. Fall back to source_file stem or ICD codes.
                if not raw_title or raw_title.lower() in {"одобрен", "утверждён", "утвержден"}:
                    sf = data.get("source_file", "")
                    stem = sf.rsplit(".", 1)[0].replace("_", " ").replace("-", " ").strip() if sf else ""
                    icd_codes = data.get("icd_codes", [])
                    if stem and not stem.startswith("p_"):
                        raw_title = stem
                    elif icd_codes:
                        raw_title = f"Клинический протокол ({', '.join(icd_codes[:3])})"
                    else:
                        raw_title = "Клинический протокол"
                p = Protocol(
                    protocol_id=data["protocol_id"],
                    source_file=data.get("source_file", ""),
                    title=raw_title,
                    icd_codes=data.get("icd_codes", []),
                    text=data.get("text", ""),
                )
                p.sections = parse_sections(p.text)
                p.all_icd_codes = extract_icd_codes(p.text)
                protocols[p.protocol_id] = p
            except (json.JSONDecodeError, KeyError) as e:
                print(f"Warning: skipping malformed line: {e}")

    print(f"Loaded {len(protocols)} protocols")
    return protocols


def build_icd_lookup(protocols: dict[str, Protocol]) -> dict[str, list[Protocol]]:
    """Build mapping from ICD-10 code to list of protocols."""
    lookup: dict[str, list[Protocol]] = {}
    for p in protocols.values():
        # Include codes from the icd_codes field
        all_codes = set(p.icd_codes) | set(p.all_icd_codes)
        for code in all_codes:
            lookup.setdefault(code, []).append(p)
    return lookup


def get_all_valid_icd_codes(protocols: dict[str, Protocol]) -> set[str]:
    """Get all ICD-10 codes known from the entire corpus."""
    codes = set()
    for p in protocols.values():
        codes.update(p.icd_codes)
        codes.update(p.all_icd_codes)
    return codes


def load_few_shot_examples(test_set_dir: str, n: int = 3) -> list[dict]:
    """Load a few examples from the test set for few-shot prompting."""
    test_dir = Path(test_set_dir)
    examples = []
    for json_file in sorted(test_dir.glob("*.json"))[:n]:
        try:
            with open(json_file) as f:
                data = json.load(f)
            examples.append({
                "query": data.get("query", ""),
                "gt": data.get("gt", ""),
                "icd_codes": data.get("icd_codes", []),
                "title": data.get("title", ""),
            })
        except Exception:
            pass
    return examples
