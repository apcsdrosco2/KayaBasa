"""
clean_bikol.py — Cleaner for ara-close-lang Bikol (Central Bikol) documents.

CSV title format: "StoryName___Central_Bikol[-v1TIMESTAMP]"
Text field format: "[Title] [NarrativeText] [Q&A?] [Boilerplate]"

The noise is at the END: publisher boilerplate (ABC+, USAID, Asia Foundation,
Room to Read, Bilum Books, Pum Anh, Creative Commons) and sometimes a
comprehension-question / activity section just before the boilerplate.
The start is mostly clean apart from the repeated title.
"""

import regex
import ftfy

# ── Boilerplate sentinels (end of text) ──────────────────────────────────────
# Cut the text at the first match.  All of these signal the end of narrative
# and the start of publisher / licence attribution.

_BOILERPLATE = regex.compile(
    r"(?i)(?:"
    r"All\s+Children\s+Reading"
    r"|The\s+Asia\s+Foundation"
    r"|An?\s+Pum\s+Anh"
    r"|Bilum\s+Books"
    r"|Creative\s+Commons"
    r"|Room\s+to\s+Read"
    r"|\bUSAID\b"
    r"|This\s+book\s+was\s+(?:adapted|made|prepared)"
    r"|This\s+storybook\s+received"
    r"|This\s+product\s+is\s+made\s+possible"
    r"|An\s+produktong\s+ini\s+(?:ay|naging)"
    r"|An\s+librong\s+ini\s+na\s+binuo"
    r"|An\s+stroybook\s+naini"
    r"|Brought\s+to\s+(?:you|You)\s+by"
    r"|Let.s\s+Read\s+is\s+an\s+initiative"
    r"|For\s+full\s+terms\s+of\s+use"
    r"|Contributing\s+translators"
    r"|ABC\+.*?Advancing\s+Basic\s+Education"
    r"|ABC\+\s+is\s+a\s+partnership"
    r"|Pum\s+Anh\s+Lao\s+sarong"
    r"|nagpapasalamat\s+kami\s+sa"   # "Mga Pagmidbid" opening in Bikol
    r")",
)

# ── Activity / comprehension section sentinels ───────────────────────────────
# These mark the end of the narrative and the start of educational scaffolding.

_ACTIVITY = regex.compile(
    r"(?i)(?:"
    r"Mga\s+Hapot\s+sa\s+Pag.?intindi"
    r"|Kahapotan\s+[Kk]an\s+Mga\s+Naukdan"
    r"|Pagkasabot\s+sa\s+mga\s+Hapot"
    r"|Mga\s+Pagmidbid"          # Acknowledgements section
    r"|Dapit\s+sa\s+[Kk]agsurat"  # About the Author
    r"|Dapit\s+sa\s+[Kk]agkurit"  # About the Illustrator
    r"|(?:^|\n)\s*Aktibidad\s*(?:\n|$)"
    r"|(?:^|\n)\s*Gibohon\s*(?:\n|$)"
    r"|(?:^|\n)\s*Activity\s*(?:\n|$)"
    r")",
)


def get_display_title(raw_csv_title: str) -> str:
    """Convert a CSV filename-style title to the display title found in text.

    "StoryName___Central_Bikol-v1..." → "StoryName"
    Underscores become spaces; the ___Central_Bikol[...] suffix is dropped.
    """
    base = raw_csv_title.split("___")[0]
    return base.replace("_", " ")


# Leading attribution phrases that appear inline right after the title in some
# Bikol docs (e.g. "All Children Reading - Cambodia Sin Thuokna …").
# Strip these before the end-boilerplate search so they don't cut the text.
_LEADING_ATRIB = regex.compile(
    r"(?i)^(?:All\s+Children\s+Reading|Room\s+to\s+Read)(\s*[-–]\s*\w+)?\s+"
)


def clean(text: str, display_title: str) -> str:
    """Remove title repetition, end boilerplate, and Q&A sections from a
    Bikol ara-close-lang document string."""
    text = ftfy.fix_text(text)

    # 1. Strip repeated display title from start
    if display_title:
        text = regex.sub(
            r"^\s*" + regex.escape(display_title) + r"\s*", "", text
        )

    # 1b. Strip leading publisher attribution if it appears right at the start
    #     (some docs: "All Children Reading - Cambodia AuthorName Narrative…")
    text = _LEADING_ATRIB.sub("", text)

    # 2. Cut at activity / comprehension section (appears before boilerplate)
    m = _ACTIVITY.search(text)
    if m:
        text = text[: m.start()]

    # 3. Cut at end boilerplate
    m = _BOILERPLATE.search(text)
    if m:
        text = text[: m.start()]

    # 4. HTML entities
    text = regex.sub(r"&nbsp;", " ", text)
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
