"""
clean_basaha.py — Cleaner for BasahaCorpus documents (all four languages:
Hiligaynon, Minasbate, Karay-a, Rinconada).

File format:
    Line 1:    Story title
    Lines 2-N: [author name lines separated by blank lines]
    Body:      Narrative pages (many blank lines between pages)
    Tail:      [Activity / Q&A section]
               [Publisher boilerplate: Let's Read, Bilum Books, USAID, …]
               [Table of Contents / Guide / Cover / Copyright section]

All of those tail sections must be removed; only the narrative body is kept.
"""

import regex
import ftfy

# ── Boilerplate sentinels ────────────────────────────────────────────────────
# Cut the text at the first match.  Same sources as Bikol (Let's Read / Asia
# Foundation / Bilum Books / Room to Read / USAID / ABC+).

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
    r"|Brought\s+to\s+(?:you|You)\s+by"
    r"|Let.s\s+Read\s+is\s+an\s+initiative"
    r"|For\s+full\s+terms\s+of\s+use"
    r"|Contributing\s+translators"
    r"|Original\s+Story"
    r"|ABC\+.*?Advancing\s+Basic\s+Education"
    r"|ABC\+\s+is\s+a\s+partnership"
    r"|Pum\s+Anh\s+Lao\s+sarong"
    r")",
)

# ── Activity / comprehension section sentinels ───────────────────────────────

_ACTIVITY = regex.compile(
    r"(?im)(?:"
    r"^Mga\s+Hapot\s+sa\s+Pag.?intindi"
    r"|^Kahapotan\s+[Kk]an\s+Mga\s+Naukdan"
    r"|^Pagkasabot\s+sa\s+mga\s+Hapot"
    r"|^Mga\s+Pagmidbid"
    r"|^Dapit\s+sa\s+[Kk]agsurat"
    r"|^Dapit\s+sa\s+[Kk]agkurit"
    r"|^Ulubrahun\b"           # Hiligaynon: Activity
    r"|^Aktibidad\b"
    r"|^Gibohon\b"
    r"|^Activity\b"
    r"|^Ehersisyo\b"           # Exercise (Minasbate: Ehersisyo ni Paka)
    r"|^Mga\s+Eksampol\b"      # Examples section
    r")",
)

# ── Leading-preamble line detectors ─────────────────────────────────────────
# Some files (typically sourced from All Children Reading - Cambodia / Room to
# Read batches) place a publisher attribution line right after the title and
# before the narrative.  These must be stripped as preamble, not treated as
# end-boilerplate sentinels that would cut the text at position zero.

_LEADING_ATRIB_LINE = regex.compile(
    r"(?i)^(?:"
    r"All\s+Children\s+Reading"
    r"|Room\s+to\s+Read"
    r"|The\s+Asia\s+Foundation\s*[-–]\s*Let.?s\s+Read"
    r")(\s*[-–]\s*\w+)?$"
)

# Author names appear on their own lines in the first ~12 lines after the
# title.  A line qualifies as an author-name line when it is short, every
# word starts with an uppercase letter, and none of the words is a common
# Philippine / Bisayan function word.

_PHIL_FUNC = regex.compile(
    r"(?i)\b(?:ang|an|si|sa|mga|ng|ni|na|at|kag|ug|kay|sang|ini|ito|"
    r"siya|ako|kami|sinda|nag|mag|kan|sin|dili|wala|amo|man|lang|"
    r"ta|ba|gid|da|pa|ra|ya)\b"
)
_ALL_CAP_WORDS = regex.compile(r"^\s*(?:\p{Lu}[\p{L}.\-]+\.?\s+)*\p{Lu}[\p{L}.\-]+\.?\s*$")


def _is_author_line(line: str) -> bool:
    """Return True if the line looks like a foreign author/illustrator name."""
    s = line.strip()
    if not s or len(s) > 50:
        return False
    if not _ALL_CAP_WORDS.match(s):
        return False
    if _PHIL_FUNC.search(s):
        return False
    # Require at least 2 words (first + last name minimum)
    words = s.split()
    return len(words) >= 2


def clean(text: str) -> tuple[str, bool]:
    """Remove title, author lines, activity sections, and boilerplate from a
    BasahaCorpus document string.

    Returns:
        (cleaned_text, had_boilerplate_match)
    """
    text = ftfy.fix_text(text)
    lines = text.split("\n")

    # ── Step 1: strip the title (first non-empty line) ───────────────────────
    start = 0
    for i, line in enumerate(lines):
        if line.strip():
            start = i + 1   # skip this line (it is the title)
            break
    lines = lines[start:]

    # ── Step 2: strip preamble lines within the first 15 lines ──────────────
    # This covers:
    #  (a) leading attribution lines  ("All Children Reading - Cambodia")
    #  (b) foreign author/illustrator name lines ("Srun Rida", "Ramendra Kumar")
    filtered = []
    for i, line in enumerate(lines):
        if i < 15:
            s = line.strip()
            if _LEADING_ATRIB_LINE.match(s):
                continue
            if _is_author_line(line):
                continue
        filtered.append(line)
    text = "\n".join(filtered)

    # ── Step 3: cut at activity / comprehension section ──────────────────────
    m = _ACTIVITY.search(text)
    had_activity = m is not None
    if m:
        text = text[: m.start()]

    # ── Step 4: cut at end boilerplate ───────────────────────────────────────
    m = _BOILERPLATE.search(text)
    had_boilerplate = m is not None
    if m:
        text = text[: m.start()]

    # ── Step 5: remove structural page markers ───────────────────────────────
    text = regex.sub(r"(?im)^\s*Page\s+\d+\s*$", "", text)
    text = regex.sub(
        r"(?im)^\s*(?:Guide|Cover|Start\s+of\s+Story|Copyright)\s*$",
        "", text,
    )
    text = regex.sub(r"(?im)^\s*Table\s+of\s+Contents\s*$", "", text)

    # ── Step 6: HTML entities ────────────────────────────────────────────────
    text = regex.sub(r"&nbsp;", " ", text)
    text = regex.sub(r"&quot;", '"', text)
    text = regex.sub(r"&amp;",  "&", text)
    text = regex.sub(r"&#\d+;", "", text)
    text = regex.sub(r"&[a-zA-Z]+;", "", text)

    # ── Step 7: URLs ─────────────────────────────────────────────────────────
    text = regex.sub(r"https?://\S+", "", text)

    # ── Step 8: whitespace ───────────────────────────────────────────────────
    text = regex.sub(r"[ \t]+", " ", text)
    text = regex.sub(r"\n{3,}", "\n\n", text)

    matched = had_boilerplate or had_activity
    return text.strip(), matched
