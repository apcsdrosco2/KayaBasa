"""Denoising and normalization (proposal Sections 3.1.2 and 3.1.3)."""

from __future__ import annotations

import re

# ara-close-lang (Section 3.1.2.1.1)
_NAME_TOKEN = r"(?:[A-ZÑÁÉÍÓÚ][\w'’.\-]*)"
_CREDIT_RE = re.compile(
    r"\b(?i:sinulat|gisulat|gidibuho|isinulat|written|illustrated"
    r"|hinikay|sinulatan|gidrowing)"
    r"(?:\s+(?i:ni|nila|by|ng))?\s*:?\s*"
    rf"(?:{_NAME_TOKEN}\s*(?=[A-ZÑÁÉÍÓÚ]|[.,;:!?]|$)){{0,5}}"
)

# Subject/genre tags are bounded to a short span for the same reason: an
_SUBJECT_TAG_RE = re.compile(
    r"(?i)(cebuano|bikol|bicol|tagalog|filipino)\s+"
    r"(personal development|story ?book|community living|science|mathematics"
    r"|mother tongue|araling panlipunan|edukasyon)[^.!?\n]{0,40}",
)


def clean_ara_text(text: str, title: str) -> str:
    """Strip the metadata concatenated into field 3 of an ara-close-lang line."""
    if not text:
        return ""

    # 1. Remove the repeated title at the start of the text field (~100% of lines).
    if title:
        text = re.sub(r"^\s*" + re.escape(title) + r"\s*", "", text)

    # 2. Remove author and illustrator credit lines (42% of Cebuano lines).
    text = _CREDIT_RE.sub("", text)

    # 3. Remove subject/genre tags embedded by DepEd or the publisher.
    text = _SUBJECT_TAG_RE.sub("", text)

    # 4. Remove the KATAPUSAN end-of-document marker (93% of Cebuano lines).
    text = re.sub(r"\bKATAPUSAN\b", "", text)

    # 5. Remove HTML character entities (8% of Cebuano lines).
    text = re.sub(r"&quot;", '"', text)
    text = re.sub(r"&amp;", "&", text)
    text = re.sub(r"&#\d+;", "", text)
    text = re.sub(r"&[a-zA-Z]+;", "", text)

    # 6. Remove URLs and web artifacts.
    text = re.sub(r"https?://\S+", "", text)
    text = re.sub(r"www\.\S+", "", text)

    # 7. Collapse excess whitespace.
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{2,}", "\n", text)

    return text.strip()


# BasahaCorpus (Section 3.1.2.1.2)
BASAHA_NOISE_SENTINELS = [
    r"Frog.s Exercise",
    r"About Cat and Dog",
    r"Let.s Read is an initiative",
    r"Table of Contents",
    r"Brought to you by",
    r"Original Story",
    r"For full terms of use",
    r"Contributing translators",
]
SENTINEL_RE = re.compile("|".join(BASAHA_NOISE_SENTINELS), flags=re.IGNORECASE)

# Function words shared across the Central Philippine languages. A short header
PHILIPPINE_FUNCTION_WORDS = re.compile(
    r"\b(ang|si|ng|sa|na|at|kag|ug|ni|kay|nag|mag|mga|an|su|iya|siya|nga)\b",
    re.IGNORECASE,
)

# How many leading lines the name-stripping heuristic is allowed to inspect.
HEADER_LINE_WINDOW = 10


def clean_basaha_text(text: str) -> str:
    """Truncate the Let's Read Asia export at the narrative boundary."""
    if not text:
        return ""

    # 1. Truncate at the first sentinel (start of boilerplate).
    match = SENTINEL_RE.search(text)
    if match:
        text = text[: match.start()]

    # 2. Drop header lines that carry no Philippine function word (author names).
    cleaned_lines = []
    for i, line in enumerate(text.split("\n")):
        stripped = line.strip()
        if (
            i < HEADER_LINE_WINDOW
            and stripped
            and not PHILIPPINE_FUNCTION_WORDS.search(stripped)
        ):
            continue
        cleaned_lines.append(line)
    text = "\n".join(cleaned_lines)

    # 3. Remove "Page N" navigation lines.
    text = re.sub(r"^\s*Page\s+\d+\s*$", "", text, flags=re.MULTILINE)

    # 4. Remove Guide / Cover / Start of Story navigation lines.
    text = re.sub(
        r"^\s*(Guide|Cover|Start of Story|Copyright)\s*$",
        "",
        text,
        flags=re.MULTILINE | re.IGNORECASE,
    )

    # 5. Collapse excess blank lines (sources carry up to five between pages).
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


# Normalization (Section 3.1.3), applied uniformly to all seven languages
def normalize_text(text: str) -> str:
    """Standardize formatting and encoding without altering morphology."""
    if not text:
        return ""

    # 1. Decode any remaining HTML entities.
    text = re.sub(r"&nbsp;", " ", text)
    text = re.sub(r"&amp;", "&", text)
    text = re.sub(r"&#\d+;", "", text)

    # 2. Insert a missing space after closing punctuation: 'anihon,"an' -> 'anihon," an'.
    text = re.sub(r'([,\.!\?;:"\)])\s*(?=[A-Za-z])', r"\1 ", text)

    # 3. Normalize hyphenation in reduplicated forms: "dali - dali" -> "dali-dali".
    text = re.sub(r"\s*-\s*", "-", text)
    text = re.sub(r"-{2,}", "-", text)

    # 4. Collapse internal whitespace.
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" +\n", "\n", text)

    # 5. Morphological preservation: no stemming, no lemmatization. Native
    return text.strip()


def prepare_corpus(df, apply_denoising: bool = True):
    """Return a copy of ``df`` with the ``text`` column prepared."""
    out = df.copy()
    if not apply_denoising:
        return out

    is_ara = out["split_role"] == "high_resource"
    if is_ara.any():
        out.loc[is_ara, "text"] = out.loc[is_ara].apply(
            lambda r: clean_ara_text(r["text"], r["title"]), axis=1
        )
    if (~is_ara).any():
        out.loc[~is_ara, "text"] = out.loc[~is_ara, "text"].apply(clean_basaha_text)
    out["text"] = out["text"].apply(normalize_text)
    return out
