"""
clean_ceb_data.py — Re-clean ceb_all_data.txt to produce a corrected ceb_all_clean.txt.

Input format (ceb_all_data.txt):
    book_title,grade_level,full_text (including title, author info, metadata, KATAPUSAN)

Output format (ceb_all_clean.txt):
    book_title,grade_level,cleaned_text (narrative only)

Cleaning steps per entry:
    1. Remove the repeated book title from the start of the text
    2. Remove author/illustrator credits
    3. Remove language/genre metadata
    4. Remove copyright/license boilerplate and URLs
    5. Remove "KATAPUSAN" / "WAKAS" end markers
    6. Remove instructional/metadata preambles
    7. Normalize whitespace
"""

import re
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
INPUT_PATH = PROJECT_DIR / "clean" / "ceb_all_data.txt"
OUTPUT_PATH = PROJECT_DIR / "clean" / "ceb_all_clean.txt"

# ── Language and genre vocabulary ────────────────────────────────────────────

LANG_NAMES = (
    r"(?:Cebuano|Bikol|Bikolano|Tagalog|Hiligaynon|Minasbate[nñ]?o?|"
    r"Karay-?a|Rinconada|Filipino)"
)

GENRE_TYPES = (
    r"(?:Personal\s+Development|Story\s*[Bb]ook|Community\s+Living|"
    r"Science|Math(?:ematics)?|Mother\s+Tongue|Reader|"
    r"Animal\s+Stories|Traditional\s+Stor(?:y|ies)|Health|Environment|"
    r"Non\s*[- ]?[Ff]iction|Primer|Dictionary|Agriculture|Culture|"
    r"Philippine\s+Nat'l\s+School\s+for\s+the\s+Blind)"
)

# ── Credit patterns ──────────────────────────────────────────────────────────

# Credit trigger verbs
_CREDIT_TRIGGERS = (
    r"(?:Gisulat|Sinulat|Gisuwat|Gidibuho|Gidibuhu|Isinulat|Hinikay|"
    r"Sinulatan|Gihulagway|Written|Illustrated)"
)

# Full credit pattern: trigger + optional "nila"/"ni"/"by"/":"  + name tokens
_CREDIT_RE = re.compile(
    _CREDIT_TRIGGERS +
    r"(?:\s+(?:nila|ug\s+Gihulagway))?"
    r"(?:\s+ni|\s+n|\s+by)?\s*:?\s*"
    r"(?:[A-Z(][\w.\-()]*(?:\s+[A-Z(][\w.\-()]*){0,7})"
    r"(?:\s*,\s*[A-Z][\w.\-]*(?:\s+[A-Z][\w.\-]*){0,5})*"
    r"\s*",
    re.UNICODE
)

# ── Boilerplate / copyright ──────────────────────────────────────────────────

_BOILERPLATE_RE = re.compile(
    r"(?:"
    r"Copyright\s*©.*?(?=Pila|Usa|Ang|Si|Sa|Aduna|$)"
    r"|You\s+are\s+free\s+to\s+make\s+commercial.*?(?=Pila|Usa|Ang|Si|Sa|Aduna|$)"
    r"|You\s+may\s+adapt.*?(?=Pila|Usa|Ang|Si|Sa|Aduna|$)"
    r"|You\s+must\s+keep.*?(?=Pila|Usa|Ang|Si|Sa|Aduna|$)"
    r"|Translations-If\s+you.*?(?=Pila|Usa|Ang|Si|Sa|Aduna|$)"
    r"|Adaptations-If\s+you.*?(?=Pila|Usa|Ang|Si|Sa|Aduna|$)"
    r"|This\s+translation\s+was\s+not.*?(?=Pila|Usa|Ang|Si|Sa|Aduna|$)"
    r"|USAID\s+shall\s+not.*?(?=Pila|Usa|Ang|Si|Sa|Aduna|$)"
    r"|This\s+is\s+an\s+adaptation.*?(?=Pila|Usa|Ang|Si|Sa|Aduna|$)"
    r"|Views\s+and\s+opinions.*?(?=Pila|Usa|Ang|Si|Sa|Aduna|$)"
    r")",
    re.IGNORECASE | re.DOTALL
)

# Simpler boilerplate — cut everything from Copyright/© to the last boilerplate sentence
_BOILERPLATE_BLOCK_RE = re.compile(
    r"(?:Copyright\s*)?©.*?endorsed\s+by\s+USAID\.?\s*",
    re.IGNORECASE | re.DOTALL
)

# ── URL pattern ──────────────────────────────────────────────────────────────
_URL_RE = re.compile(r"https?://\S+")

# ── Location metadata ────────────────────────────────────────────────────────
_LOCATION_RE = re.compile(
    r"\s*(?:Cebu\s+City|Davao|Manila),?\s*(?:Cebu,?\s*)?(?:Philippines)?\s*",
    re.IGNORECASE
)

# ── Level/Grade markers ──────────────────────────────────────────────────────
_LEVEL_RE = re.compile(r"\s*Level\s*,?\s*Grade\s*", re.IGNORECASE)

# ── "Sugilanon sa libro" and instructional metadata ──────────────────────────
_SUGILANON_RE = re.compile(
    r"Sugilanon\s+sa\s+libro:\s*[^.]*\.\s*", re.IGNORECASE
)
_GIGAMIT_RE = re.compile(
    r"Mga\s+(?:gigamit\s+nga\s+letra|sige\s+ug\s+gamit\s+nga\s+pulong|"
    r"bag-ong\s+pulong\s+nga\s+makat-unan)\s*:\s*[^A-Z]*(?=[A-Z]|\Z)",
    re.DOTALL
)


def clean_text(text: str, title: str) -> str:
    """Clean a single entry's text field."""
    # Remove BOM
    text = text.replace("\ufeff", "")

    # 1. Remove repeated title at the very start of text only
    escaped_title = re.escape(title)
    text = re.sub(r"^\s*" + escaped_title + r"\s+", "", text, count=1)

    # 2. Remove Level/Grade markers
    text = _LEVEL_RE.sub(" ", text)

    # 3. Remove language/genre metadata blocks
    # Pattern: language name optionally followed by genre type
    lang_genre_re = re.compile(
        r"\s*" + LANG_NAMES + r"(?:\s+" + GENRE_TYPES + r")?\s*",
        re.IGNORECASE
    )
    text = lang_genre_re.sub(" ", text, count=5)

    # Also genre alone
    genre_only_re = re.compile(r"\s*" + GENRE_TYPES + r"\s*", re.IGNORECASE)
    text = genre_only_re.sub(" ", text, count=3)

    # 4. Remove "Sugilanon sa libro" and instructional lines
    text = _SUGILANON_RE.sub("", text)
    text = _GIGAMIT_RE.sub("", text)

    # 5. Remove credit lines (multiple passes for multiple credits)
    for _ in range(6):
        new_text = _CREDIT_RE.sub("", text, count=1)
        if new_text == text:
            break
        text = new_text

    # 6. Remove second occurrence of title at the start (some entries repeat it after credits)
    text = re.sub(r"^\s*" + escaped_title + r"\s+", "", text, count=1)

    # 7. Remove copyright/boilerplate block
    text = _BOILERPLATE_BLOCK_RE.sub("", text)

    # 8. Remove URLs
    text = _URL_RE.sub("", text)

    # 9. Remove location metadata
    text = _LOCATION_RE.sub(" ", text, count=3)

    # 10. Remove KATAPUSAN / WAKAS end marker (only at end)
    text = re.sub(r"\s*\b(?:KATAPUSAN|WAKAS|Katapusan)\b\s*$", "", text)

    # 11. Remove HTML entities
    text = re.sub(r"&quot;", '"', text)
    text = re.sub(r"&amp;", "&", text)
    text = re.sub(r"&#\d+;", "", text)
    text = re.sub(r"&[a-zA-Z]+;", "", text)

    # 12. Remove orphaned leading fragments from credit parsing
    # e.g. "ni: Name Name" at the start
    text = re.sub(r"^\s*ni:\s*[A-Z][\w.\-]*(?:\s+[A-Z][\w.\-]*){0,4}\s*", "", text)
    # e.g. just "ni" + short names
    text = re.sub(r"^\s*ni\s+[A-Z][\w.\-]*(?:\s+[A-Z][\w.\-]*){0,3}\s*,?\s*", "", text)

    # 13. Clean up leading punctuation artifacts
    text = re.sub(r"^[\s,.:;]+", "", text)

    # 14. Normalize whitespace
    text = re.sub(r"[ \t]+", " ", text)
    text = text.strip()

    return text


def reassemble_entries(raw_lines: list[str]) -> list[str]:
    """Reassemble multi-line entries into single lines.
    
    A valid entry starts with: Title,grade,text where grade is 1, 2, or 3.
    Lines that don't match this pattern are continuations of the previous entry.
    """
    # Pattern: non-empty title, comma, single digit 1-3, comma, text
    entry_start_re = re.compile(r"^[^,]+,[123],")
    
    entries = []
    current = None
    
    for line in raw_lines:
        stripped = line.strip()
        if not stripped:
            continue
        
        if entry_start_re.match(stripped):
            # This is a new entry
            if current is not None:
                entries.append(current)
            current = stripped
        else:
            # This is a continuation of the previous entry
            if current is not None:
                current += " " + stripped
            else:
                # Orphan line at the start — shouldn't happen
                current = stripped
    
    # Don't forget the last entry
    if current is not None:
        entries.append(current)
    
    return entries


def process():
    """Read ceb_all_data.txt, clean each entry, write ceb_all_clean.txt."""
    raw_lines = INPUT_PATH.read_text(encoding="utf-8", errors="ignore").splitlines()
    
    # First, reassemble multi-line entries
    entries = reassemble_entries(raw_lines)
    print(f"Raw lines: {len(raw_lines)}, Reassembled entries: {len(entries)}")

    output_lines = []
    warnings = []

    for i, line in enumerate(entries):
        # Parse: title,grade_level,text (split on first 2 commas only)
        parts = line.split(",", 2)
        if len(parts) < 3:
            warnings.append(f"Entry {i}: too few commas, skipping: {line[:60]}")
            continue

        title = parts[0].strip()
        grade = parts[1].strip()
        text = parts[2].strip()

        if grade not in ("1", "2", "3"):
            warnings.append(f"Entry {i}: unexpected grade '{grade}': {title}")
            continue

        cleaned = clean_text(text, title)

        if not cleaned:
            warnings.append(f"Entry {i}: empty text after cleaning: {title}")

        output_lines.append(f"{title},{grade},{cleaned}")

    if warnings:
        print(f"Warnings ({len(warnings)}):")
        for w in warnings:
            print(f"  {w}")

    # Write output
    OUTPUT_PATH.write_text("\n".join(output_lines) + "\n", encoding="utf-8")
    print(f"\nWrote {len(output_lines)} entries to {OUTPUT_PATH.name}")

    return output_lines


if __name__ == "__main__":
    process()
