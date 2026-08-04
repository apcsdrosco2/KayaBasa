"""
clean_cebuano.py — Cleaner for ara-close-lang Cebuano documents.

Preamble structure (inline CSV text field):
    [TITLE] [CREDIT*] [GENRE_TAG?] NARRATIVE [KATAPUSAN]

Credits use Cebuano/Filipino trigger verbs (Gisulat, Sinulat, Gidibuho, …)
followed by "ni" and 1-6 name tokens.  Genre tag is a language name +
optional genre-type phrase.  KATAPUSAN marks end of story.
"""

import regex
import ftfy

# ── Genre type vocabulary ────────────────────────────────────────────────────

_ANY_GENRE_TYPE = (
    r"(?i:personal\s+development|story\s*book|community\s+living|"
    r"science|mathematics|mother\s+tongue|reader|"
    r"animal\s+stories|traditional\s+story|health|environment|"
    r"non\s+fiction|primer|dictionary|agriculture)"
)

# ── Name-token guards ────────────────────────────────────────────────────────
# Stop NAME_TOKEN from consuming credit-trigger words or genre-related words.

_STOP_TRIGGERS = (
    r"(?!(?i:sinulat|gisulat|gidibu(?:h[io]|uh[io])|isinulat|hinikay|sinulatan|"
    r"written\s+|illustrated\s+))"
)
_STOP_GENRE = (
    r"(?!(?i:cebuano|bikol|bikolano|tagalog|hiligaynon|minasbate|karay|rinconada|"
    r"personal|development|story|mathematics|science|community|living|mother|tongue|"
    r"reader|animal|traditional|health|environment|non|fiction|primer|dictionary|"
    r"agriculture|pinulongan|petsa)\b)"
)
_NAME_TOKEN = r"(?:" + _STOP_TRIGGERS + _STOP_GENRE + r"\p{Lu}[\p{L}.\-]*\.?\s*)"

# ── One full credit line ─────────────────────────────────────────────────────

_ONE_CREDIT = (
    r"(?:(?i:sinulat|gisulat|gidibu(?:h[io]|uh[io])|isinulat|hinikay|sinulatan|"
    r"written\s+by|illustrated\s+by)"
    r"(?:\s*(?:ni|by))?\s*:?\s*"
    + _NAME_TOKEN + r"{0,6}\s*)"
)

# ── Non-standard metadata ────────────────────────────────────────────────────

_LANG_VALUE = (
    r"(?i:cebuano|bikol|bikolano|tagalog|hiligaynon|"
    r"minasbate[nñ]?o?|karay-?a|rinconada)"
)
_META_BLOCK = (
    r"(?:"
        r"(?i:Pinulongan)\s*:\s*" + _LANG_VALUE + r"\s*"
        r"|"
        r"(?i:Petsa\s+sa\s+Pagsulat)\s*:\s*\p{Lu}[\p{L}]*\s*,?\s*"
    r")*"
)

# ── Genre block (language name + optional genre type) ───────────────────────

_GENRE_BLOCK = (
    r"(?:(?i:cebuano|bikol|bikolano|tagalog|hiligaynon|"
    r"minasbate[nñ]?o?|karay-?a|rinconada)\s*"
    r"(?:" + _ANY_GENRE_TYPE + r"\s*)?)?"
)

# ── Unified preamble pattern ─────────────────────────────────────────────────
# Handles CREDIT*→META→LANG_GENRE and CREDIT*→GENRE_TYPE→LANG orderings.

_PREAMBLE_RE = regex.compile(
    r"^"
    + _ONE_CREDIT + r"*"
    + _META_BLOCK
    + _GENRE_BLOCK
    + _ONE_CREDIT + r"*"
    + r"(?:" + _ANY_GENRE_TYPE + r"\s*)?"
    + r"(?:" + _LANG_VALUE + r"\s*)?"
    + _META_BLOCK
)


def clean(text: str, title: str) -> str:
    """Remove title repetition, credits, genre tag, and KATAPUSAN from a
    Cebuano ara-close-lang document string."""
    text = ftfy.fix_text(text)
    # 1. Remove repeated title at start
    text = regex.sub(r"^\s*" + regex.escape(title) + r"\s*", "", text)
    # 2. Remove preamble (credits + genre tag) in one anchored pass
    text = _PREAMBLE_RE.sub("", text, count=1)
    # 2b. Orphaned leading colon + short value + comma (Petsa sa Pagsulat leak)
    text = regex.sub(r"^:\s*(?:\p{L}+\s*){0,3},?\s*(?=\p{Lu}|\Z)", "", text)
    # 3. Remove KATAPUSAN / Katapusan end-of-story marker.
    #    Anchor to end-of-string so "Sa katapusan," in narrative prose is NOT removed.
    text = regex.sub(r"\s*\bKATAPUSAN\b\s*$", "", text, flags=regex.IGNORECASE)
    # 4. HTML entities
    text = regex.sub(r"&quot;", '"', text)
    text = regex.sub(r"&amp;",  "&", text)
    text = regex.sub(r"&#\d+;", "", text)
    text = regex.sub(r"&[a-zA-Z]+;", "", text)
    # 5. URLs
    text = regex.sub(r"https?://\S+", "", text)
    # 6. Whitespace
    text = regex.sub(r"[ \t]+", " ", text)
    text = regex.sub(r"\n{2,}", "\n", text)
    return text.strip()
