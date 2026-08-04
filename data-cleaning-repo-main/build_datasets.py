"""
build_datasets.py — Narrow raw corpora down to narrative text.

Goal (deliberately NOT normalization): strip metadata, keep the raw native
narrative.  Output schema is exactly three fields, pipe-delimited:

    language|level|text

One file per language per level (6 languages x 3 levels = 18 files) plus a
single combined master.  Rows whose narrative/metadata boundary could not be
resolved confidently are still emitted (cut at the best guess) AND logged to a
review file — per the "native-word cut + flag" strategy.

Sources:
  ara-close-lang  Cebuano / Bikol — one doc per CSV line, metadata is a
                  preamble at the START (repeated title, credits, English genre)
                  and, for some, an end marker (KATAPUSAN) or publisher tail.
  BasahaCorpus    Hiligaynon / Minasbate / Karay-a / Rinconada — one .txt per
                  doc, structured by blank lines: title block, author-name
                  blocks, narrative body, then activity/publisher tail.
"""

import os
import glob
import regex
import ftfy

# ── Paths ─────────────────────────────────────────────────────────────────────

ARA_CEBUANO = "ara-close-lang/data/cebuano/ceb_all_data.txt"
ARA_BIKOL   = "ara-close-lang/data/bikol/bik_all_data.txt"
BASAHA_BASE = "BasahaCorpus-HierarchicalCrosslingualARA/data/raw"
BASAHA_LANGS = ["hiligaynon", "minasbate", "karay-a", "rinconada"]
OUT_DIR = "output"

SEP = "|"

# ── Shared vocabulary ─────────────────────────────────────────────────────────

# Native sentence-openers — words that typically begin a narrative sentence and
# are frequently followed by a *capitalised* proper noun (e.g. "Si Nita ...").
# Defined first because the credit pattern uses it to stop consuming names.
_OPENERS = {
    "si", "sila", "siya", "sinda", "ikaw", "kamo", "ang", "an", "ako", "kami",
    "kita", "ini", "kini", "kana", "usa", "sarong", "saro", "aduna", "adunay",
    "dunay", "may", "mayo", "naa", "igwa", "sa", "kan", "sin", "nin", "mga",
    "kada", "karon", "ngunyan", "kaniadto", "niadto", "kaidto", "idtong",
    "hain", "sain", "bako", "diri", "dai", "wala", "kay", "kun", "kon", "iyo",
}
_OPENER_ALT = "|".join(sorted(_OPENERS))

# Native function words (openers + high-frequency particles across the six
# languages).  Metadata spans — credits, genre tags, publisher tails — contain
# none of these; a run rich in them is narrative.  Used to stop a publisher
# marker from truncating a story that continues after it.
_NATIVE_FUNC = set(_OPENERS) | {
    "ang", "an", "si", "sa", "mga", "ng", "ni", "na", "at", "kag", "ug", "kay",
    "sang", "ini", "ito", "siya", "ako", "kami", "sinda", "nag", "mag", "kan",
    "sin", "nin", "dili", "wala", "amo", "man", "lang", "ta", "ba", "gid", "da",
    "pa", "ra", "ya", "may", "isa", "ka", "kita", "kini", "kana", "usa", "asin",
    "saka", "nga", "og", "niya", "nila", "iyang", "akong",
}
_WORD_RE = regex.compile(r"\p{L}+")

# Unambiguously English function words (NOT native function words — "an", "sa",
# "ug", "na", "si" etc. are deliberately excluded).  Used so an English
# translated title/credit ("Fatima the Spinner and the Tent") is not mistaken
# for native prose just because it contains a lowercase "the"/"and".
_ENG_STOP = {
    "the", "and", "of", "to", "for", "with", "from", "his", "her", "their",
    "this", "that", "was", "were", "are", "is", "you", "your", "she", "he",
    "they", "them", "what", "who", "when", "where", "into", "about", "story",
    "book", "inc", "resources", "drivers",
}

# Credit trigger verbs (Cebuano/Bikol/Filipino): "written by" / "illustrated by".
_TRIG = (
    r"(?:gisulat|sinulat|gisuwat|sinuwat|gihuwat|isinulat|ginsurat|gindrowing|"
    r"gidrowing|gidibu(?:h[io]|uh[io])|gihulagway|ginhulagway|sininurat|gihikay|hinikay|"
    r"tinipon|gindrawing|gihubad|istorya|tigmantala|"
    r"written\s+by|illustrated\s+by|story\s+by|adapted\s+by|drawn\s+by|"
    r"translated\s+by)"
)
# Language names and English genre/level words — these separate a credit from the
# narrative, so name tokens must not consume them (otherwise the run eats through
# the genre tag into the first story word, e.g. "... Cebuano Nangadto sila ...").
_LANG_ALT = r"cebuano|bikol(?:ano)?|tagalog|hiligaynon|minasbate\w*|karay-?a|rinconada"
_GENRE_ALT = (
    r"personal|development|story|book|community|living|animal|stories|traditional|"
    r"science|math\w*|health|environment|primer|reader|dictionary|agriculture|"
    r"fiction|non|mother|tongue|folk|tale|decodable|grade|level|stage"
)

# A trigger word + optional ni/by + up to ~6 capitalised name tokens is a credit
# line and is deleted.  A name token is a capitalised word that is NOT a trigger,
# native opener, or language/genre word, AND is followed by another capital, a
# separator, or end-of-credit.  The trailing guard is the key: a capital word
# followed by a *lowercase* word starts a sentence (narrative) and is left alone,
# so "... CALUMPITA Gihimo ning ..." stops at CALUMPITA, never eating "Gihimo".
_NAME_STOP = "|".join([_OPENER_ALT, _LANG_ALT, _GENRE_ALT])
_SEP = r"(?:[&,]|\b(?i:and|ug|asin|saka|y)\b)"
_NAME_TOK = (
    r"(?:"
    r"(?:" + _SEP + r"\s*)?"
    r"(?i:(?!" + _TRIG + r"\b)(?!(?:" + _NAME_STOP + r")\b))"
    r"\p{Lu}[\p{L}.''\-]*\.?"
    r"(?=\s+(?:\p{Lu}|" + _SEP + r")|\s*$)"
    r"\s*"
    r")"
)
# NOTE: no global (?i) flag — under (?i), \p{Lu} becomes case-insensitive and the
# name-run would greedily eat lowercase narrative words.  Case-insensitivity is
# scoped to the trigger and connector words only; \p{Lu} stays strictly upper.
_CREDIT = regex.compile(
    r"\b(?i:" + _TRIG + r")\b(?:\s*(?i:ni(?:la)?|by))*\s*:?\s*" + _NAME_TOK + r"{0,12}"
)

# Recognisable metadata words: a skipped prefix containing any of these is
# confidently metadata (credit / language / genre), so the cut is NOT flagged.
# A skip with none of them is a bare proper-noun run (the "Green Star" case) and
# IS flagged for review.
_META_RE = regex.compile(r"(?i)\b(?:" + _TRIG + r"|" + _LANG_ALT + r"|" + _GENRE_ALT + r")\b")

# Multi-word English genre phrases.  Unlike single words (science/health), these
# never occur in the native narratives, so they are removed wherever they appear
# — even without a preceding language name (e.g. a title repeat "… Story Book …").
_GENRE_PHRASE = regex.compile(
    r"(?i)\b(?:story\s*book|animal\s+stories|personal\s+development|"
    r"community\s+living|traditional\s+story|mother\s+tongue|non[\s-]?fiction|"
    r"folk\s*tale|\(?decodable\)?)\b"
)

# Language name + optional English genre tag (closed vocabulary).
_GENRE = regex.compile(
    r"(?i)\b(?:cebuano|bikol(?:ano)?|tagalog|hiligaynon|minasbate\w*|"
    r"karay-?a|rinconada)\b\s*"
    r"(?:personal\s+development|story\s*book|community\s+living|animal\s+stories|"
    r"traditional\s+story|non[\s-]?fiction|mother\s+tongue|science|math\w*|"
    r"health|environment|primer|reader|dictionary|agriculture|"
    r"folk\s*tale|fiction)?"
)

# Native metadata labels written as "Label: value" — language ("Pinulongan:
# Cebuano") and write/draw dates ("Petsa sa Pagsulat/Paghulad: Mayo ,").  These
# occur only in front matter, never in the narrative, so they are removed
# wherever found.  Exactly one value word (+ trailing comma) is consumed after
# the colon, which covers the closed set of values (a language or a month).
_ARA_META = regex.compile(
    r"\b(?i:Pinulongan|Petsa(?:\s+sa\s+\p{L}+)?)\s*:\s*(?:\p{L}+\s*,?\s*)?"
)

# Editorial-board advisory + magazine-publisher preamble (the Liwayway/Bisaya
# Cebuano readers): "Ubos sa pagtambag nila ni: <board names> … Editor sa
# Bisaya, Liwayway Publications, Manila".  The board name-list is far longer than
# a single credit's name cap and is split by a parenthetical, so the credit
# remover can't fully consume it.  Both anchors occur only in this boilerplate,
# so the whole span — advisory phrase through publisher line — is removed at once.
_ARA_EDITORIAL = regex.compile(
    r"(?is)\bUbos\s+sa\s+pagtambag\b.*?\bLiwayway\s+Publications[,\s]*Manila\b"
)

# Publisher / copyright tail vocabulary (shared).  None of these words occur in
# the native narratives, so truncating at the first hit is safe.  Kept loose on
# purpose — the data writes "Ang Asia Foundation", "nga Let's Read", "Copyright
# ©  … Enabling Writers", "(Decodable)", not the textbook forms.
_PUB = (
    r"Story\s+Alt\s+Img|Sync\s+Translation|Translation\s*Publish|"     # Let's Read export UI footer
    r"Table\s+of\s+Contents|Guide\s+Cover|Comprehension\s+Questions|"  # navigation / activity headers
    r"All\s+Children\s+Reading|Room\s+to\s+Read|(?:The\s+)?Asia\s+Foundation|"
    r"Bilum\s+Books|Creative\s+Commons|\bUSAID\b|Let.?s\s+Read|Brought\s+to\s+you\s+by|"
    r"This\s+(?:book|product|storybook|story)\s+(?:was|is|received)|ABC\+|Pum\s+Anh|"
    r"For\s+full\s+terms\s+of\s+use|Contributing\s+translators|Enabling\s+Writers|"
    r"\(Decodable\)|\bCopyright\b|©|Litara\s+Foundation|Third\s+Story|Original\s+Story|"
    r"An\s+(?:produktong|librong|stroybook)\s+ini|\bGrade\s+\d+\b|creative\s+non[\s-]?fiction|"
    r"Ini\s+nga\s+storybook|nakabaton\s+sang\s+bulig|Ini\s+na\s+libro|An\s+libro\s+na\s+ini|"
    r"Ang?\s+storybook\s+na\s+ini"
)

# CANVAS license-attribution sentence (a publisher boilerplate block that, in
# BasahaCorpus and some ara docs, sits as a LEADING paragraph between the title
# and the real story: "An nag-surat, nag-guhit, kag an CANVAS gina-imbitaran …
# Salamat." / "Ineenganyo ku Kagsurat, Ilustrador saka CANVAS … Mabalos.").  It
# must be removed *in place* (not used as a tail sentinel — truncating at it
# decapitates the narrative that follows).  The whole sentence containing CANVAS
# is removed, plus an optional trailing "Salamat./Mabalos." closer; CANVAS is an
# org name that never appears in the narratives, so this is safe.
_CANVAS_LICENSE = regex.compile(
    r"[^.!?]*\bCANVAS\b[^.!?]*[.!?]\s*(?:(?i:Salamat|Mabalos)\s*[.!?]\s*)?",
    regex.IGNORECASE,
)

# "About the Author / About the Illustrator" bio tail, plus the closing
# "made together with the authors, illustrators, publishers" acknowledgment
# sentence.  Both sit at the END of a document with the real story before them,
# so the marker is used to truncate the tail (backing up to the start of the
# sentence that contains it).  The bio-header words (Kagsurat/Ilustrador/Tagsulat
# …) and the credit enumeration ("tagsulat, ilustrador") never occur in the
# narratives, so this never cuts story text.
_BIO_TAIL = regex.compile(
    r"(?i)(?:"
    r"\b(?:Dapit|Nahanungod|Mahanungod|Mahitungod|Bahin|Parte|Manungod|Mahin?ungod)"
    r"\s+(?:sa|kan|ni|kang|ki)\s+"
    r"(?:Kagsurat|Para?surat|Para?kurit|Ilustrador|Manunulat|Awtor|Kagdrowing|"
    r"Paradrowing|Paradibuho|Tagsulat|Tagguhit|Tagdrowing|Proyekto)\b"
    r"|\btagsulat,\s*ilustrador\b"
    r"|\bilustrador\s+(?:ug|kag|asin|saka)\s+(?:publisher|editor)\b"
    # closing credit enumeration ("… parasurat, parakurit, editor …"); the
    # agentive role noun parakurit is corpus-unique to these acknowledgments,
    # so it is a safe standalone tail sentinel.
    r"|\bparakurit\b"
    r")"
)


def _cut_bio_tail(body: str):
    """Truncate an end-of-doc author/illustrator bio or credit acknowledgment.
    Returns (body, flag)."""
    m = _BIO_TAIL.search(body)
    if not m:
        return body, ""
    cut = m.start()
    prev = max(body.rfind(".", 0, cut), body.rfind("!", 0, cut),
               body.rfind("?", 0, cut))
    if prev != -1:
        cut = prev + 1            # cut at the start of the marker's sentence
    return body[:cut].strip(), "bio_tail"


# End-of-narrative markers / publisher tails for ara-close-lang.
_ARA_END = regex.compile(r"(?:\bKATAPUSAN\b|(?i:" + _PUB + r"))")
# Publisher vocabulary alone (KATAPUSAN handled separately): these can appear as a
# leading tag, an inline note, or a trailing tail, so a match is not by itself an
# end-of-story signal — see clean_ara.
_PUB_ONLY = regex.compile(r"(?i:" + _PUB + r")")

# BasahaCorpus tail: activity/comprehension sections + publisher boilerplate.
_BASAHA_TAIL = regex.compile(
    r"(?im)(?:"
    r"^\s*(?:Mga\s+Hapot|Kahapotan|Pagkasabot\s+sa\s+mga\s+Hapot|Mga\s+Pagmidbid|"
    r"Dapit\s+sa\s+[Kk]ag\w+|Ulubrahun|Aktibidad|Gibohon|Activity|Ehersisyo|"
    r"Mga\s+Eksampol)\b"
    r"|" + _PUB +
    r")"
)

# Comprehension-question tail detector: a sequential numbered run "1. … 2. … 3.".
# Whether such a run is cut or kept depends on its question density, decided in
# clean_basaha — a run carrying >=2 question marks is an activity/Q&A section,
# while a run with none is an in-story list (family members, washing steps).
_NUM_QA = regex.compile(r"\b1\.\s.+?\b2\.\s.+?\b3\.\s", regex.S)
# A Q&A section header sitting immediately before the numbered run; pulled into
# the cut so "… bulak. Mga pamangkot:" does not dangle at the narrative tail.
_QA_HEAD = regex.compile(
    r"(?i)\s*(?:(?:Mga\s+)?Pamangkot\w*|(?:Mga\s+)?Hunga\w*|Palamangkutanon|"
    r"Pamangkutanon|Comprehension\s+Questions?|(?:Mga\s+)?Giya\s+sa\s+Pagtuon|"
    r"Pagsabot|Pagtuon)\s*:?\s*$"
)
# Page-number / word-echo interleaving: >=3 standalone short-number tokens left
# mid-body by paginated exports.  Too messy to strip without risking real words,
# so such a document is flagged for manual review and kept ("flag, don't mangle").
_DIGIT_INTERLEAVE = regex.compile(r"(?:\b\d{1,4}\b[ ]+){3,}")

_HTML_ENT = [
    (regex.compile(r"&nbsp;"), " "),
    (regex.compile(r"&quot;"), '"'),
    (regex.compile(r"&amp;"),  "&"),
    (regex.compile(r"&#\d+;"), ""),
    (regex.compile(r"&[a-zA-Z]+;"), ""),
]

# Foreign-script tokens.  All six target languages use a Latin orthography only,
# so any whitespace token carrying a non-Latin letter (Lao/Thai/Han/etc. runs
# that leak in from Let's Read Asia parallel-translation exports) is noise and
# the whole token is dropped.  The class [^\P{L}\p{Latin}] = "a letter that is
# not Latin script" (a negated union, which sidesteps the && set-intersection
# operator).  \p{Latin} covers ñ, accented vowels, and the Rinconada schwa
# (U+0259), so native orthography is preserved; combining marks are \p{M}, not
# \p{L}, so accents are never mistaken for foreign letters.
_FOREIGN_TOK = regex.compile(r"\S*[^\P{L}\p{Latin}]\S*")


def _basic_clean(text: str) -> str:
    """Fix mojibake/encoding artefacts and HTML entities — not normalization."""
    text = ftfy.fix_text(text)
    text = text.replace("﻿", "")          # stray BOM
    for pat, repl in _HTML_ENT:
        text = pat.sub(repl, text)
    text = regex.sub(r"https?://\S+", "", text)
    text = _FOREIGN_TOK.sub(" ", text)    # drop non-Latin-script tokens
    return text


def _to_field(text: str) -> str:
    """Collapse to a single pipe-safe line (raw words preserved)."""
    text = regex.sub(r"\s+", " ", text).strip()
    return text.replace(SEP, "/")              # protect the delimiter


# Unambiguously-English vocabulary — words that do NOT occur in the six target
# languages (native "an", "sa", "at", "si", "na" are deliberately excluded so
# native prose is never mistaken for English).  Used to flag/drop the occasional
# all-English junk document (e.g. a "Hello world … A lion, a snake" test export).
_ENG_ONLY = set((
    "the and of with for was were are his her their they them this that "
    "hello world lion cow duck otter elephant bird snake blue sky from have "
    "story about into would could should there what when where who"
).split())


def _english_ratio(text: str) -> float:
    toks = [w.strip(".,!?\"'()").lower() for w in text.split()]
    toks = [w for w in toks if w.isalpha()]
    if not toks:
        return 0.0
    return sum(1 for w in toks if w in _ENG_ONLY) / len(toks)


def _strip_leading_english(text: str):
    """Drop a leading English art-caption sentence glued to the front of native
    narrative (Let's Read parallel exports, e.g. "Horn Samon, A lion, cow, duck …
    in a blue sky. <native story…>").  Only the FIRST sentence is removed, and
    only when it is overwhelmingly English (>=60% English-only words over >=4
    tokens) AND native prose follows; ordinary native first sentences score ~0
    and are untouched.  Returns (text, flag)."""
    m = regex.match(r"^\s*(.*?[.!?])(\s+)(\p{Lu}.*)$", text, regex.S)
    if not m:
        return text, ""
    first, rest = m.group(1), m.group(3)
    toks = [w.strip(".,!?\"'()").lower() for w in first.split()]
    toks = [w for w in toks if w.isalpha()]
    if len(toks) < 4:
        return text, ""
    frac = sum(1 for w in toks if w in _ENG_ONLY) / len(toks)
    if frac >= 0.60 and len(rest.split()) >= 3:
        return rest, "english_caption"
    return text, ""


def _strip_leading_title(text: str, title: str) -> str:
    """Remove the title if the body opens by repeating it."""
    title = title.strip()
    if not title:
        return text
    parts = [regex.escape(p) for p in title.split()]
    pat = regex.compile(r"^\s*" + r"\s+".join(parts) + r"\b\s*", regex.IGNORECASE)
    return pat.sub("", text, count=1)


def _narrative_start(text: str):
    """Find where the narrative begins after leading metadata.

    Returns (start_index, skipped_text). start_index is the earliest of:
      (a) a known native opener token, or
      (b) a Capitalised token immediately followed by a lowercase-initial token
          (a real sentence start; author-name runs are all-capitalised).
    Returns (None, text) if neither signal is found.
    """
    toks = list(regex.finditer(r"\S+", text))
    cores = [regex.sub(r"^[^\p{L}]+|[^\p{L}]+$", "", t.group()) for t in toks]

    def is_lower(j):
        # A *native* lowercase word signals real prose; a lowercase English
        # stopword does not (it belongs to a translated title/credit).
        return (j < len(cores) and cores[j] and cores[j][0].islower()
                and cores[j].lower() not in _ENG_STOP)

    for i, t in enumerate(toks):
        core = cores[i]
        if not core:
            continue
        if core[0].islower():
            # Narrative sentences are capitalised, so a lowercase token here is a
            # leftover metadata fragment ("sa ni Evelyn ...", "for the Blind ...").
            # Skip it and keep scanning for the real (capitalised) start.
            continue
        # (a) native opener — but only if a lowercase function word follows
        #     within 1-2 tokens (real prose: "Si Nita ang ..."), which
        #     distinguishes it from an author run ("Sin Thuokna Nakakasurat").
        if core.lower() in _OPENERS and (is_lower(i + 1) or is_lower(i + 2)):
            return (t.start(), text[:t.start()])
        # (b) a Capitalised word immediately followed by a lowercase-initial word
        if is_lower(i + 1):
            return (t.start(), text[:t.start()])
    return (None, text)


# In-text title repeat that the CSV-title strip cannot catch (the in-text title
# differs from the CSV title field, e.g. body "Ang Dala ni Alma ug Mama Si Alma
# …" under CSV title "Ang Kinabuhi ni Edward").  The leading title is a Title-
# Case run — every word is Capitalised or a native function word, with no
# sentence punctuation — immediately followed by a native opener that begins the
# real narrative.  Ordinary prose breaks this pattern at its first lowercase verb
# ("Si Pipay adunay …"), so it is not matched.
_FUNC_LOWER = r"ni|ng|nga|ug|sa|si|an|at|kan|nin|kay|asin|saka|na|kag|sang|sin|ka|y"
_TITLE_WORD = r"(?:\p{Lu}[\p{Ll}'’\-]*|(?:" + _FUNC_LOWER + r"))"
_OPENER_WORD = (
    r"(?:Si|Ang|An|Adunay|Aduna|Dunay|Igwa|May|Usa|Sarong|Saro|Kang|Ining|"
    r"Naa|Kini|Kana|Ini|Niadto|Kaniadto)"
)
_TITLE_REPEAT = regex.compile(
    r"^(?P<title>" + _TITLE_WORD + r"(?:\s+" + _TITLE_WORD + r"){1,7})\s+"
    r"(?P<rest>" + _OPENER_WORD + r"\s+\p{L}.*)$", regex.S
)


def _strip_title_repeat(body: str):
    """Cut a leading repeated in-text title.  Returns (body, flag)."""
    m = _TITLE_REPEAT.match(body)
    if not m:
        return body, ""
    title, rest = m.group("title"), m.group("rest")
    # Require a genuine Title-Case run (>=2 capitalised words) and a substantive
    # remainder, so a short list ("Si Juan Si Pedro") is not over-trimmed.
    if len(regex.findall(r"\p{Lu}[\p{Ll}]", title)) < 2 or len(rest.split()) < 3:
        return body, ""
    return rest, "title_repeat"


def clean_ara(text: str, title: str):
    """Clean one ara-close-lang (Cebuano/Bikol) document.

    Returns (narrative, flag) where flag is "" (confident) or a reason string.
    """
    text = _basic_clean(text)
    text = _strip_leading_title(text, title)
    text = _CANVAS_LICENSE.sub(" ", text)   # drop CANVAS license sentence (keep story)
    text = _ARA_EDITORIAL.sub(" ", text)    # drop editorial-board + publisher preamble
    text = _CREDIT.sub(" ", text)           # delete credit lines anywhere
    text = _ARA_META.sub(" ", text)         # delete "Pinulongan:/Petsa sa …:" labels
    text = _GENRE.sub(" ", text)            # delete language + genre tag
    text = _GENRE_PHRASE.sub(" ", text)     # delete standalone genre phrases
    text = regex.sub(r"\s+", " ", text).strip()

    start, skipped = _narrative_start(text)
    if start is None:
        flag = "no_opener"                            # nothing looked like prose
        body = text
    else:
        body = text[start:]
        skipped = skipped.strip()
        if not skipped:
            flag = ""                                 # clean front, confident
        elif _META_RE.search(skipped):
            flag = ""                                 # skipped recognised metadata
        else:
            flag = "residual_prefix"                  # bare names (e.g. "Green Star")

    # End-of-story handling.  KATAPUSAN reliably marks the end, so truncate there.
    mk = regex.search(r"\bKATAPUSAN\b", body)
    if mk:
        body = body[: mk.start()]

    # A publisher / attribution phrase (_PUB) may sit at the tail, but it can also
    # appear as a LEADING source tag or an inline note with the whole story AFTER
    # it (e.g. "… Sokha The Asia Foundation - Let's Read … <1000-word story>"), or
    # be a plain narrative loanword.  Truncating at the first hit would then
    # amputate the narrative.  A hit is treated as a real tail — and truncated —
    # only when almost no native narrative follows it; if substantial native text
    # follows, the body is kept intact and the document is flagged for review
    # ("flag, don't mangle").
    # Walk each publisher/attribution marker (_PUB): if almost no native narrative
    # follows, it is a genuine trailing tail → truncate; otherwise the story
    # continues, so the marker is a leading source tag or inline note — delete the
    # boilerplate window (the anchor plus the following non-native attribution
    # tokens, up to where native prose resumes) in place and keep scanning.
    for _ in range(10):
        mp = _PUB_ONLY.search(body)
        if not mp:
            break
        after_native = sum(1 for w in _WORD_RE.findall(body[mp.end():].lower())
                           if w in _NATIVE_FUNC)
        if after_native <= 3:
            body = body[: mp.start()]                 # genuine trailing tail
            break
        seg = body[mp.start():]
        anchor_len = mp.end() - mp.start()
        cut_end = len(seg)
        for wm in _WORD_RE.finditer(seg):
            if wm.start() >= anchor_len and wm.group().lower() in _NATIVE_FUNC:
                cut_end = wm.start()                  # native prose resumes here
                break
        body = (body[: mp.start()] + " " + seg[cut_end:]).strip()
        if not flag:
            flag = "pub_marker_in_body"
    # Cut an end-of-doc author/illustrator bio or credit acknowledgment tail.
    body, bflag = _cut_bio_tail(body)
    if bflag:
        flag = bflag
    # Cut a leading in-text title repeat (logged via flag for audit).
    body, tflag = _strip_title_repeat(body)
    if tflag:
        flag = tflag
    # Cut a leading English art-caption sentence (keep the native story after it).
    body, eflag = _strip_leading_english(body)
    if eflag:
        flag = eflag
    return _to_field(body), flag


# ── BasahaCorpus ──────────────────────────────────────────────────────────────

_PHIL_FUNC = regex.compile(
    r"(?i)\b(?:ang|an|si|sa|mga|ng|ni|na|at|kag|ug|kay|sang|ini|ito|siya|ako|"
    r"kami|sinda|nag|mag|kan|sin|dili|wala|amo|man|lang|ta|ba|gid|da|pa|ra|ya|"
    r"may|isa|ka)\b"
)
_ALLCAP_WORDS = regex.compile(r"^(?:\p{Lu}[\p{L}.'\-]*\.?\s+)*\p{Lu}[\p{L}.'\-]*\.?$")

# Publisher attribution that appears *before* the narrative (a preamble line to
# drop, NOT a tail sentinel that should truncate the story).
_LEAD_ATTRIB = regex.compile(
    r"(?i)^(?:All\s+Children\s+Reading|Room\s+to\s+Read|The\s+Asia\s+Foundation|"
    r"Let.?s\s+Read|Pratham\s+Books|Book\s+Dash|African\s+Storybook|StoryWeaver|"
    r"Bloom\s+Library|Global\s+Digital\s+Library)\b.*$"
)


def _is_name_block(block: str) -> bool:
    """A short, fully title-cased block with no function words = author name."""
    s = block.strip()
    if not s or len(s) > 60 or "\n" in s.strip("\n").strip():
        # collapse internal newlines first; a name block is one logical line
        s = regex.sub(r"\s+", " ", s)
    if len(s) > 60:
        return False
    if not _ALLCAP_WORDS.match(s):
        return False
    if _PHIL_FUNC.search(s):
        return False
    return len(s.split()) >= 2


def clean_basaha(text: str):
    """Clean one BasahaCorpus document. Returns (narrative, flag)."""
    text = _basic_clean(text).replace("\r\n", "\n").replace("\r", "\n")
    text = _CANVAS_LICENSE.sub(" ", text)   # drop CANVAS license block (keep story)

    # Split on blank-line gaps; drop the title block (first) and any leading
    # author-name / publisher-attribution blocks, then keep the rest as body.
    blocks = [b for b in regex.split(r"\n\s*\n", text) if b.strip()]
    if not blocks:
        return "", "empty"
    body_blocks = []
    seen_body = False
    for i, b in enumerate(blocks):
        bs = regex.sub(r"\s+", " ", b).strip()
        if not seen_body and (i == 0 or _is_name_block(b) or _LEAD_ATTRIB.match(bs)):
            continue            # title / author name / leading attribution → skip
        seen_body = True
        body_blocks.append(b)

    body = "\n".join(body_blocks)

    # Cut the activity/publisher tail — only within the body, so a *leading*
    # attribution can never truncate the whole story.
    m = _BASAHA_TAIL.search(body)
    if m:
        body = body[: m.start()]
    # Cut a comprehension-question tail: a numbered run whose span carries >=2
    # question marks is a Q&A activity section.  A numbered run WITHOUT questions
    # is an in-story list and is left intact (and flagged below) — the
    # "flag, don't mangle" distinction.
    m = _NUM_QA.search(body)
    if m and m.group().count("?") >= 2:
        cut = m.start()
        h = _QA_HEAD.search(body[:cut])     # also drop an immediately-preceding header
        if h:
            cut = h.start()
        body = body[:cut]
    body = body.replace("\n", " ")
    body = regex.sub(r"(?im)^\s*(?:Page\s+\d+|Guide|Cover|Copyright|"
                     r"Start\s+of\s+Story|Table\s+of\s+Contents)\s*$", " ", body)
    body, bflag = _cut_bio_tail(_to_field(body))         # author/illustrator bio tail
    body, tflag = _strip_title_repeat(_to_field(body))   # leading in-text title repeat
    body, eflag = _strip_leading_english(_to_field(body))  # leading English caption
    field = _to_field(body)
    num = _NUM_QA.search(field)
    if not field.strip():
        flag = "empty"
    elif _DIGIT_INTERLEAVE.search(field):
        flag = "digit_interleave"          # page-number/echo noise: kept, flagged
    elif num and num.group().count("?") < 2:
        flag = "numbered_list"             # in-story numbered list: kept, flagged
    elif eflag:
        flag = eflag                       # English caption cut: kept, flagged
    elif tflag:
        flag = tflag                       # title repeat cut: kept, flagged
    elif bflag:
        flag = bflag                       # bio tail cut: kept, flagged
    else:
        flag = ""
    return field, flag


# ── Loading ───────────────────────────────────────────────────────────────────

_LINE = regex.compile(r"^(?P<title>.*?),(?P<level>\d+),(?P<text>.*)$")


def load_ara(path, language):
    rows = []
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line.strip():
                continue
            m = _LINE.match(line)
            if not m:
                continue
            raw_title = m.group("title").strip()
            title = raw_title.split("___")[0].replace("_", " ") if language == "bikol" else raw_title
            narrative, flag = clean_ara(m.group("text"), title)
            rows.append((language, int(m.group("level")), title, narrative, flag))
    return rows


def load_basaha(language):
    rows = []
    for grade, level in [("grade 1", 1), ("grade 2", 2), ("grade 3", 3)]:
        for fp in sorted(glob.glob(os.path.join(BASAHA_BASE, language, grade, "*.txt"))):
            with open(fp, encoding="utf-8", errors="replace") as f:
                narrative, flag = clean_basaha(f.read())
            title = os.path.basename(fp)[:-4].split("___")[0].replace("_", " ")
            rows.append((language, level, title, narrative, flag))
    return rows


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    rows = []
    rows += load_ara(ARA_CEBUANO, "cebuano")
    rows += load_ara(ARA_BIKOL, "bikol")
    for lang in BASAHA_LANGS:
        rows += load_basaha(lang)
    # Tagalog (high-resource; cleaned by the dedicated, messiest-source module).
    import clean_tagalog
    rows += clean_tagalog.load_tagalog()

    languages = ["tagalog", "cebuano", "bikol", "hiligaynon", "minasbate", "karay-a", "rinconada"]

    # Flags that drop a row from the datasets (still logged for review).  A body
    # of fewer than 4 words is never a real story here (a stray credit word, an
    # acknowledgement lead-in); an English-heavy body is a junk export; an exact
    # duplicate within a language is redundant.
    MIN_WORDS = 4
    DROP_FLAGS = {"too_short", "empty", "english_heavy", "duplicate",
                  "residual_boilerplate"}

    # Mark English-heavy and too-short rows (only override an otherwise-clean flag).
    rows = [[L, lvl, title, text, flag] for (L, lvl, title, text, flag) in rows]
    for r in rows:
        L, lvl, title, text, flag = r
        if text.strip() and not flag and _english_ratio(text) >= 0.40:
            r[4] = "english_heavy"
        elif text.strip() and len(text.split()) < MIN_WORDS and not flag:
            r[4] = "too_short"

    # Exact-duplicate removal within a language (keep the first occurrence).
    seen = set()
    for r in rows:
        L, lvl, title, text, flag = r
        emittable = (bool(text.strip()) and len(text.split()) >= MIN_WORDS
                     and flag not in DROP_FLAGS)
        if not emittable:
            continue
        key = (L, text.strip())
        if key in seen:
            r[4] = "duplicate"
        else:
            seen.add(key)

    def emit(text, flag):
        return (bool(text.strip()) and len(text.split()) >= MIN_WORDS
                and flag not in DROP_FLAGS)

    # Per language/level files + combined master.
    master_path = os.path.join(OUT_DIR, "all_languages.txt")
    written = {}
    with open(master_path, "w", encoding="utf-8", newline="\n") as master:
        master.write(f"language{SEP}level{SEP}text\n")
        for lang in languages:
            for level in (1, 2, 3):
                fp = os.path.join(OUT_DIR, f"{lang}_level{level}.txt")
                n = 0
                with open(fp, "w", encoding="utf-8", newline="\n") as out:
                    out.write(f"language{SEP}level{SEP}text\n")
                    for L, lvl, title, text, flag in rows:
                        if L == lang and lvl == level and emit(text, flag):
                            rec = f"{L}{SEP}{lvl}{SEP}{text}\n"
                            out.write(rec)
                            master.write(rec)
                            n += 1
                written[(lang, level)] = n

    # Review log for flagged rows.
    flagged = [r for r in rows if r[4]]
    flag_path = os.path.join(OUT_DIR, "flagged_for_review.txt")
    with open(flag_path, "w", encoding="utf-8", newline="\n") as fl:
        fl.write(f"language{SEP}level{SEP}flag{SEP}title{SEP}text\n")
        for L, lvl, title, text, flag in flagged:
            fl.write(f"{L}{SEP}{lvl}{SEP}{flag}{SEP}{title}{SEP}{text}\n")

    # Report.
    print("Per-language/level counts (rows with narrative text):")
    for lang in languages:
        line = "  %-11s " % lang + "  ".join(
            f"L{lvl}={written[(lang, lvl)]:>3}" for lvl in (1, 2, 3))
        print(line)
    total = sum(written.values())
    print(f"\nTotal narrative rows: {total}")
    print(f"Wrote 18 per-bucket files + {master_path}")

    from collections import Counter
    fc = Counter(r[4] for r in flagged)
    print(f"\nFlagged for review: {len(flagged)}  {dict(fc)}")
    print(f"  -> {flag_path}")


if __name__ == "__main__":
    main()
