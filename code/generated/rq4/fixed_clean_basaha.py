"""
fixed_clean_basaha.py — corrected re-implementation of the block-classification logic
in data-cleaning-repo/build_datasets.py's clean_basaha(), used to stage the RQ4
low-resource corpus. Self-contained (no dependency on data-cleaning-repo at runtime),
since that's a separate repo outside this worktree.

Two bugs are fixed relative to the original (see RQ4_PLAN.md §2A for the full
investigation, root-cause traces, and before/after regression results — zero
regressions found across all 769 raw documents):

Bug A: the original _ALLCAP_WORDS regex doesn't allow parentheses, so an author-name
block containing a parenthesized nickname (e.g. "Tassaya (Toffy) Charupatanapongse")
isn't recognized as a skippable name line. The next block (a publisher-attribution
line) then gets wrongly treated as the start of the real body, and the trailing-
boilerplate cutter truncates the entire real story to nothing.

Bug B: several _PUB/_BASAHA_TAIL sentinel phrases meant to catch TRAILING publisher
boilerplate (e.g. "Ini na libro", "(?:The )?Asia Foundation") also match inside the
FRONT acknowledgement sentence many documents open with ("Ini na libro ginhimo san
[org] sa bulig san [funder]..."), because that sentence isn't recognized as skippable
leading attribution (only short name-only lines and blocks literally starting with a
hardcoded English org name are skipped). This truncates the real narrative that
follows down to 0-3 words.

Fix A: allow a parenthesized token within a name-block run.
Fix B: also skip a leading block if the _PUB sentinel matches within roughly its
first 70 characters (i.e. the block *is* substantially a publisher/funder
acknowledgement, not narrative) rather than only matching blocks that start with
one of a few hardcoded English org names.
"""

import regex

# ── Verbatim from data-cleaning-repo/build_datasets.py (publisher/tail vocabulary) ──
# Kept identical to the original so Fix B's window-match reuses the exact same
# sentinel vocabulary, not a redefinition that could silently drift from it.

_PUB = (
    r"All\s+Children\s+Reading|Room\s+to\s+Read|(?:The\s+)?Asia\s+Foundation|"
    r"Bilum\s+Books|Creative\s+Commons|\bUSAID\b|Let.?s\s+Read|Brought\s+to\s+you\s+by|"
    r"This\s+(?:book|product|storybook|story)\s+(?:was|is|received)|ABC\+|Pum\s+Anh|"
    r"For\s+full\s+terms\s+of\s+use|Contributing\s+translators|Enabling\s+Writers|"
    r"\(Decodable\)|\bCopyright\b|©|Litara\s+Foundation|Third\s+Story|Original\s+Story|"
    r"An\s+(?:produktong|librong|stroybook)\s+ini|\bGrade\s*\d*\b|creative\s+non[\s-]?fiction|"
    r"Ini\s+nga\s+storybook|nakabaton\s+sang\s+bulig|Ini\s+na\s+libro|An\s+libro\s+na\s+ini|"
    r"Ang?\s+storybook\s+na\s+ini"
)

_BASAHA_TAIL = regex.compile(
    r"(?im)(?:"
    r"^\s*(?:Mga\s+Hapot|Kahapotan|Pagkasabot\s+sa\s+mga\s+Hapot|Mga\s+Pagmidbid|"
    r"Dapit\s+sa\s+[Kk]ag\w+|Ulubrahun|Aktibidad|Gibohon|Activity|Ehersisyo|"
    r"Mga\s+Eksampol)\b"
    r"|" + _PUB +
    r")"
)

_LEAD_ATTRIB = regex.compile(
    r"(?i)^(?:All\s+Children\s+Reading|Room\s+to\s+Read|The\s+Asia\s+Foundation|"
    r"Let.?s\s+Read|Pratham\s+Books|Book\s+Dash|African\s+Storybook|StoryWeaver|"
    r"Bloom\s+Library|Global\s+Digital\s+Library)\b.*$"
)

_PHIL_FUNC = regex.compile(
    r"(?i)\b(?:ang|an|si|sa|mga|ng|ni|na|at|kag|ug|kay|sang|ini|ito|siya|ako|"
    r"kami|sinda|nag|mag|kan|sin|dili|wala|amo|man|lang|ta|ba|gid|da|pa|ra|ya|"
    r"may|isa|ka)\b"
)

_HTML_ENT = [
    (regex.compile(r"&nbsp;"), " "),
    (regex.compile(r"&quot;"), '"'),
    (regex.compile(r"&amp;"), "&"),
    (regex.compile(r"&#\d+;"), ""),
    (regex.compile(r"&[a-zA-Z]+;"), ""),
]


def _basic_clean(text: str) -> str:
    """Fix mojibake/encoding artefacts and HTML entities — not normalization."""
    import ftfy
    text = ftfy.fix_text(text)
    text = text.replace("﻿", "")  # stray BOM
    for pat, repl in _HTML_ENT:
        text = pat.sub(repl, text)
    text = regex.sub(r"https?://\S+", "", text)
    return text


def _to_field(text: str) -> str:
    """Collapse to a single pipe-safe line (raw words preserved)."""
    text = regex.sub(r"\s+", " ", text).strip()
    return text.replace("|", "/")


# ── Fix A: allow a parenthesized token within a name run ──────────────────────
_ALLCAP_WORDS_FIXED = regex.compile(
    r"^(?:\(?\p{Lu}[\p{L}.'\-]*\)?\.?\s+)*\(?\p{Lu}[\p{L}.'\-]*\)?\.?$"
)


def _is_name_block(block: str) -> bool:
    s = block.strip()
    if not s or len(s) > 60 or "\n" in s.strip("\n").strip():
        s = regex.sub(r"\s+", " ", s)
    if len(s) > 60:
        return False
    if not _ALLCAP_WORDS_FIXED.match(s):
        return False
    if _PHIL_FUNC.search(s):
        return False
    return len(s.split()) >= 2


# ── Fix B: recognize a leading block that is substantially a publisher/funder
# acknowledgement (a native-language sentence mentioning the org inline), not just
# blocks that literally start with an English org name. ──────────────────────
# 200 chars (was 70): a Rinconada-dialect variant of this same acknowledgement
# template ("A librong adi ay ginibo kan Srijanalaya sa tabang kan programang Book
# for Asia kan The Asia Foundation...") runs the org mention past the 70-char
# window, so that phrasing fell through to the tail-cutter instead of being
# recognized as a skippable leading block — same failure family as Bug B, just a
# too-tight window. 200 chars comfortably covers every acknowledgement-sentence
# variant observed (Bikol/Minasbate/Karay-a/Rinconada spellings), while still being
# far shorter than any real narrative block would run before incidentally
# mentioning one of _PUB's terms — a leading block is, by construction, either this
# acknowledgement sentence in its entirety or the start of the real story; no
# observed real narrative opening mentions "Asia Foundation"/"Grade"/etc. at all.
_ACK_WINDOW_CHARS = 200


def _is_ack_block(block: str) -> bool:
    bs = regex.sub(r"\s+", " ", block).strip()
    if not bs:
        return False
    m = regex.search(r"(?i)" + _PUB, bs)
    return bool(m and m.start() < _ACK_WINDOW_CHARS)


# ── Fix C: structural page/section markers ("Table of Contents", "Guide", "Cover",
# "Page N", "Copyright", "Start of Story") are supposed to be stripped by the
# original code's final regex.sub(r"(?im)^...$", ...) — but that runs AFTER
# body.replace("\n", " ") has already collapsed every block onto one line, so the
# `^...$` per-line anchors can never match once several of these markers have been
# joined by spaces (e.g. "Table of Contents Guide Cover" as one string never matches
# a pattern requiring the WHOLE line to be just "Guide"). This is latent in the
# original build_datasets.py too, not something Fix A/B introduced — it only
# surfaces when a raw file's entire narrative is missing and ONLY these structural
# markers remain (3 confirmed Hiligaynon files: the raw .txt genuinely contains
# nothing but a title [+ author names] + "Table of Contents / Guide / Cover", no
# story body at all — same corrupt-source class as Minasbate's "Paglinis san
# Lawas"). Fixed by dropping any block that consists ENTIRELY of these markers
# before joining, so a file with nothing else left correctly resolves to "" / empty
# instead of a 3-4 word false-positive survivor. ─────────────────────────────
_STRUCTURAL_MARKER_BLOCK = regex.compile(
    r"(?i)^\s*(?:Page\s+\d+|Guide|Cover|Copyright|Start\s+of\s+Story|"
    r"Table\s+of\s+Contents)\s*$"
)


def _is_structural_marker_block(block: str) -> bool:
    bs = regex.sub(r"\s+", " ", block).strip()
    return bool(_STRUCTURAL_MARKER_BLOCK.match(bs))


# ── Fix D: an acknowledgement sentence can itself be split across 2+ blank-line-
# separated blocks in the raw source (a mid-sentence formatting line break), e.g.
# block N = "Ading librong adi ginibo ka Srijanalaya sa tabang ka programang" and
# block N+1 = "Books for Asia ka The Asia Foundation. An Srijanalaya usad na NGO...".
# Fix B's per-block _is_ack_block check correctly flags block N+1 (it contains the
# org mention), but by then block N — which contains NO _PUB match on its own —
# has already been committed as body-start, locking in seen_body=True before N+1 is
# even examined. Fixed by looking ahead up to 2 blocks: if the block under
# consideration doesn't itself qualify to skip, but combining it with the next
# block(s) (up to a small char budget) produces a combined span that DOES contain a
# _PUB match, treat the whole span as one skippable acknowledgement unit instead of
# committing to body at the first (org-mention-free) half of the sentence. ──────
_LOOKAHEAD_MAX_BLOCKS = 2
_LOOKAHEAD_MAX_CHARS = 260


def _leading_span_is_ack(blocks: list, start: int) -> int:
    """If blocks[start:start+k] together form a skippable acknowledgement span,
    return k (number of blocks to skip). Otherwise return 0.

    Checks for a _PUB match FIRST, then uses the char budget only to decide
    whether it's still worth extending the span further when no match has been
    found yet — not as a gate that can hide a match sitting just past the budget
    in an otherwise-short combined span (that inverted order was a bug: a 294-char
    combined span with the org mention at character 82 was being discarded before
    ever being checked, because 294 > a naive "stop early" length cutoff)."""
    acc = ""
    for k in range(1, _LOOKAHEAD_MAX_BLOCKS + 1):
        if start + k - 1 >= len(blocks):
            break
        bs = regex.sub(r"\s+", " ", blocks[start + k - 1]).strip()
        acc = (acc + " " + bs).strip() if acc else bs
        if regex.search(r"(?i)" + _PUB, acc):
            return k
        if len(acc) > _LOOKAHEAD_MAX_CHARS:
            break
    return 0


def clean_basaha(text: str):
    """Clean one BasahaCorpus document. Returns (narrative, flag)."""
    text = _basic_clean(text).replace("\r\n", "\n").replace("\r", "\n")

    blocks = [b for b in regex.split(r"\n\s*\n", text) if b.strip()]
    if not blocks:
        return "", "empty"
    body_blocks = []
    seen_body = False
    i = 0
    while i < len(blocks):
        b = blocks[i]
        bs = regex.sub(r"\s+", " ", b).strip()
        if not seen_body:
            if i == 0 or _is_name_block(b) or _LEAD_ATTRIB.match(bs) or _is_ack_block(b):
                i += 1
                continue
            span = _leading_span_is_ack(blocks, i)  # Fix D
            if span:
                i += span
                continue
        seen_body = True
        body_blocks.append(b)
        i += 1

    # Fix C: drop structural-marker-only blocks wherever they occur, not just via
    # the (broken, order-dependent) post-join regex below — kept as a belt-and-
    # braces second pass in case a marker ends up merged into a larger block.
    body_blocks = [b for b in body_blocks if not _is_structural_marker_block(b)]

    body = "\n".join(body_blocks)

    m = _BASAHA_TAIL.search(body)
    if m:
        body = body[: m.start()]
    body = body.replace("\n", " ")
    body = regex.sub(r"(?im)^\s*(?:Page\s+\d+|Guide|Cover|Copyright|"
                      r"Start\s+of\s+Story|Table\s+of\s+Contents)\s*$", " ", body)
    flag = "" if body.strip() else "empty"
    return _to_field(body), flag
