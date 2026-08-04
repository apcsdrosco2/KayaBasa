"""
clean_tagalog.py — Narrow the Tagalog corpus down to narrative text.

Tagalog is the messiest source: there is NO comma schema and NO separate title
field.  The data ships as three per-level files

    data/tag_lvl{1,2,3}_data.txt

(level comes from the filename), one document per non-empty line.  Each document
mixes several noise layers that differ by level:

  * lvl1  ABC+/DepEd decodables: a long English DepEd/USAID/Creative-Commons
          license block, Tagalog (and occasionally Bikol) credits, page-number
          markers, and a Bikol/Tagalog comprehension-question tail.  A few are
          bilingual (a Bikol body AND a Tagalog body in one record).
  * lvl2  Let's Read translations: title + author + illustrator names, pervasive
          page numbers, an Asia-Foundation license tail.
  * lvl3  mixed: a leading CANVAS license sentence, DepEd / Asia-Foundation tails.

Strategy (reuses the build_datasets helpers; Tagalog-specific bits are here):
  1. encoding repair + foreign-script / HTML cleanup           (bd._basic_clean)
  2. drop whole English sentences  (license blocks, Author:/Illustrator:,
     "Brought to you by", DepEd/USAID/CC boilerplate)          → _drop_english
  3. remove Tagalog/Bikol credit lines (Isinulat/Iginuhit/Sinuri/Isinurat ni…)
  4. remove structural markers (GRADE 1, DONATED PROPERTY, NOT FOR SALE, etc.)
  5. cut the comprehension-question tail (Tabang sa Pagkatuto / Pag-aadal …)
  6. cut the publisher tail (Asia Foundation / Grade: N Level / Theme: …)
  7. strip page-number tokens interleaved in the body
  8. drop the leading title + author-name run        (bd._narrative_start)
  9. flag bilingual decodables for manual review (a Bikol body is present)

Output: output/tagalog_level{1,2,3}.txt  (schema language|level|text), and the
rows are wired into build_datasets' combined master + feature matrix as a
high_resource language.
"""

import os
import glob
import regex

import build_datasets as bd

SEP = bd.SEP
TAG_GLOB = "data/tag_lvl{lvl}_data.txt"

# ── English-sentence detection ────────────────────────────────────────────────
# English function / content words that are dense in the license-and-credit
# boilerplate but absent from Tagalog narrative.  Tagalog function words (ang,
# ng, sa, na, ay, mga, si, ni …) are disjoint from these, so a sentence whose
# tokens are mostly drawn from this set is boilerplate, not story.
_ENG = set((
    "the a an and or of to in on at is are was were be been being this that these "
    "those it its his her their our your my for with from by as not no but if then "
    "you we they he she him them us please under over into out about above below "
    "work works working licensed license commons attribution noncommercial "
    "international copy distribute transmit adapt adaptation translation translations "
    "commercial purposes purpose conditions content contents illustration illustrations "
    "reproduced reproduce original developed development project education department "
    "region learning resource resources management curriculum division edition rights "
    "reserved reserved. agency united states usaid abc advancing basic philippines "
    "partnership implemented together foundation asia released some terms use attribution "
    "author illustrator published publisher publishing books book story brought program "
    "supports early reading skills habits develop next generation critical thinkers "
    "creative innovators pacific read more like get further information visit donated "
    "property sale first grade level letters curricular theme reviewed central office "
    "permission written photocopying mechanical electronic means form any without part "
    "material may transmitted modified version full view copy visit page cover title "
    "contents guide all by-nc creativecommons org licenses http https www com html "
    "label following create translate change changes images image mismo halt foundation- "
    "contributing translators inc ltd co pvt pratham storyweaver bloom "
    "digitized digitised digitization digitisation bookmobile scanned archive gutenberg"
).split())


def _eng_frac(sentence: str) -> float:
    toks = [w.strip(".,!?\"'()[]:;·—–").lower() for w in sentence.split()]
    toks = [w for w in toks if w and any(c.isalpha() for c in w)]
    if not toks:
        return 0.0
    return sum(1 for w in toks if w in _ENG) / len(toks)


# Tagalog/Filipino function words — dense in real narrative, absent from English
# boilerplate.  A multi-word segment carrying NONE of these is English boilerplate;
# this catches varied license / legal / credit text without enumerating English.
_TL_FUNC = set((
    "ang ng sa na ay mga si ni sina kay kina ito iyon iyan ako ikaw siya kami tayo "
    "kayo sila niya nila nito ka mo ko at kung ngunit pero dahil para nang wala "
    "hindi huwag isa isang nasa kaya rin din lang naman po ba"
).split())


def _drop_english(text: str) -> str:
    """Remove whole sentences that are English boilerplate.  A segment is dropped
    when it is multi-word (>=5 tokens) but carries NO Tagalog function word (the
    license / legal / credit text), or when it is >=55% known-English — so a lone
    English loanword inside Tagalog prose still survives."""
    out = []
    for s in regex.split(r"(?<=[.!?])\s+", text):
        toks = [w.strip(".,!?\"'()[]:;·—–").lower() for w in s.split()]
        toks = [w for w in toks if w and any(c.isalpha() for c in w)]
        if not toks:
            out.append(s)
            continue
        native = sum(1 for w in toks if w in _TL_FUNC)
        if len(toks) >= 5 and native == 0:
            continue
        if len(toks) >= 3 and _eng_frac(s) >= 0.55:
            continue
        out.append(s)
    return " ".join(out)


# ── Tagalog/Bikol credit & structural markers ─────────────────────────────────
# Author / illustrator / editor credit clauses (Tagalog AND the Bikol/Hiligaynon
# forms that appear in the bilingual ABC+ / DepEd decodables).  Editor lists run
# long (5-7 names), so the name run must consume the WHOLE comma/connector list —
# but stop before the story.  This reuses the ara cleaner's guard: a name token is
# a capitalised word (NOT a trigger / native opener / language / genre word) that
# is followed by another capital, a separator, or end-of-credit; a capital word
# followed by a *lowercase* word is a sentence start (narrative) and halts the run.
_TL_TRIG = (
    r"(?:Isinulat|Isinurat|Iginuhit|Idrinowing|Idinrowing|Idinibuho|Sinuri|Inedit|"
    r"Isinalin|Itinongol|Inilarawan|Gini?nuhit|Sinulat|Ginsulat|Ginlaragway|"
    r"Ginhulagway|Gin-?edit|Ginguhit|Limbitco|Isinatitik|Isulat|Isinalarawan|"
    r"Isinaayos|Kinulayan|Tagawasto|Inilathala|Kinunan)"
)
_TL_SEP = r"(?:[&,]|\b(?i:at|saka|asin|kag|nina|nira|nanday|kanday|ni|and|y)\b)"
_TL_NAMESTOP = "|".join(sorted(bd._OPENERS)) + r"|ay|nang|" + bd._LANG_ALT + r"|" + bd._GENRE_ALT
_TL_CONN = r"(?i:at|saka|asin|kag|nina|nira|nanday|kanday|y)"
_TL_NAME_TOK = (
    r"(?:"
    r"(?i:(?!" + _TL_TRIG + r"\b)(?!(?:" + _TL_NAMESTOP + r")\b))"
    r"\p{Lu}[\p{L}.'’\-]*\.?"
    # trailing guard: a name may be followed by a comma/&, another capital, or a
    # connector word — but NOT by a lowercase story word, which halts the run.
    r"(?=\s*[,;&]|\s+(?:\p{Lu}|" + _TL_CONN + r"\b)|\s*$)"
    r"\s*[,;&]?\s*"
    r"(?:" + _TL_CONN + r"\s+)?"
    r")"
)
_TL_CREDIT = regex.compile(
    r"\b(?i:" + _TL_TRIG + r")\b(?:\s*(?i:nina|nira|nila|nanday|kanday|ni|by))*\s*:?\s*"
    + _TL_NAME_TOK + r"{0,40}"
)

# Standalone structural / front- and back-matter markers, removed wherever they
# occur (none appear inside narrative).
_TL_STRUCT = regex.compile(
    r"(?i)(?:"
    r"DONATED\s+PROPERTY|NOT\s+FOR\s+SALE|·|"
    r"\bGRADE\s*\d+\b|\bGrade\s*:?\s*\d+\b|\bLevel\s+Letters\b[^.]*|"
    r"\bCurricular\s+Theme\b[^.]*|\bTheme\s*:[^.]*|\bLevel\b\s*$"
    r")"
)

# English credit / production labels.  These are short "Label: Name" fragments
# (not full sentences, so _drop_english misses them) that recur in the DepEd and
# Let's Read front matter.  The label and the capitalised name run after the
# colon are removed.  All-caps role lines (PRINCIPAL I, DISTRICT SUPERVISOR …)
# are removed as whole tokens.  None of these phrases occur in the narratives.
_TL_LABEL = regex.compile(
    r"(?i)\b(?:Writer|Author|Illustrator|Illustrated\s+by|Written\s+by|Editor|"
    r"Contributor|Contributors|Story\s+by|Adapted\s+by|Translated\s+by|Layout|"
    r"Guhit\s+ni|Kuwento\s+ni|Kwento\s+ni|Iginuhit\s+ni|Isinatitik\s+ni|"
    r"Binigyang\s+[Kk]ulay(?:\s+ni)?|Disenyo\s+at\s+[Kk]ulay(?:\s+ni)?|Kulay\s+ni|Disenyo\s+ni|"
    r"Reviewer|Lay-?out\s+Artist|Language\s+(?:Editor|Reviewer)|Content\s+(?:Editor|Reviewer))\b"
    r"\s*:?\s*(?:\p{Lu}[\p{L}.'’\-]*\.?[\s,]*){0,6}"
)

# DepEd / LRMDS / publishing-house institutional block.  Once any of these
# appears the remainder of the document is production/credits metadata, so the
# body is truncated at the FIRST hit.  All 16 "Department of Education" residues
# are of this kind ("Published by the…", "distribution within the…", "approval
# of…"); none occur inside Tagalog narrative.
_TL_INSTITUTION = regex.compile(
    r"(?i)(?:"
    r"Department\s+of\s+Education|DepEd\b|"
    r"LEARNING\s+RESOURCES?\s+MANAGEMENT|LRMDS|LRMS\b|"
    r"Schools?\s+Division|District\s+Supervisor|Public\s+Schools|"
    r"Bureau\s+of|Bilum\s+Books|SIL\s+(?:International|PNG)|"
    r"Published\s+by|Republic\s+of\s+the\s+Philippines|"
    r"Principal\s+I+\b|CESO\b|Schools?\s+Division\s+Superintendent"
    r")"
)

# Short English / production boilerplate fragments that slip past _drop_english
# because they fall under the sentence-length floor or lack a trailing period
# (image credits, rights notices, "You are free/may/must …").  None occur in the
# Tagalog narrative, so they are removed in place wherever they appear.
_TL_BOILER = regex.compile(
    r"(?i)(?:"
    r"All\s+rights?\s+reserved\.?|"
    r"No\s+part\s+of\s+this\s+(?:material|book|work)\b[^.]*\.?|"
    r"You\s+(?:are\s+free|may|must)\b[^.]*|"
    r"Images?\s+by\b[^.]*|The\s+Art\s+Of\s+Reading\b[^.]*|"
    r"Resources?\s+for\s+the\s+Blind\b[^,.]*|"
    r"DONATED\s+PROPERTY|NOT\s+FOR\s+SALE|First\s+Edition\b[^.]*|"
    r"SIL\s+International\b[^.]*|Copyright\s+Page|Published\s+by\b[^.]*|"
    r"LEARNING\s+RESOURCES?\s+MANAGEMENT\b[^.]*|"
    r"Approved\s+by\b[^.]*|Noted\s+by\b[^.]*|Prepared\s+by\b[^.]*|"
    r"Quality\s+Assured\s+by\b[^.]*|Reviewed\s+by\b[^.]*|"
    r"Leveled\s+Reader\b[^.]*|IPINAG?BI?BILI\b|IPINAGBABAWAL\b[^.]*"
    r")"
)

# Comprehension-question tail: a Tagalog or Bikol activity header.  Everything
# from the header to end-of-doc is the activity section, not narrative.
_TL_QA_HEAD = regex.compile(
    r"(?i)\b(?:Tabang\s+sa\s+(?:Pagkatuto|Pag-?aadal)|Gabay\s+sa\s+Pag-?aaral|"
    r"Mga\s+Tanong|Sagutin|Pamprosesong\s+Tanong|Gawain|"
    r"Mga\s+Kasanayan|Mga\s+Kompetensi\w*|LEARNING\s+COMPETENC\w+|"
    r"Learning\s+Competenc\w+|Mga\s+Layunin)\b"
)

# Publisher / license / closing tail.  Everything from the FIRST anchor to the
# end of the document is boilerplate, not narrative, so the body is truncated
# there.  Combines the shared bd._PUB vocabulary with Tagalog-specific anchors
# (Inilimbag = "printed/published by", "Maraming salamat sa pakikinig" = the
# stock "thanks for listening" closer, "End of story").  NOTE: "Department of
# Education" is deliberately excluded — it occurs inside genuine Tagalog
# narrative (e.g. "… ng Department of Education noong nakaraang taon …").
_TL_PUB = regex.compile(
    r"(?i)(?:" + bd._PUB + r"|"
    r"Brought\s+to\s+you\s+by|Original\s+Story|Released\s+under|"
    r"Some\s+rights\s+reserved|First\s+Edition|For\s+full\s+terms|"
    r"Contributing\s+translators|CC[\s.\-]?BY|Inilimbag\s+ng|End\s+of\s+story|"
    r"This\s+(?:material|(?:e-?)?book|work)\s+was\s+digiti[sz]ed|"
    r"Maraming\s+salamat\s+sa\s+pakikinig)"
)

# Page-number markers interleaved in the body: a whitespace-isolated 1-3 digit
# integer NOT followed by a period (so a numbered list "1." survives).
_PAGE_NUM = regex.compile(r"(?<!\S)\d{1,3}(?!\S)(?!\.)")

# Bilingual decodable signal: the Bikol comprehension header or Bikol article/
# question words that only occur when a full Bikol body is concatenated in.
_BIKOL_SIGNAL = regex.compile(
    r"(?i)\bTabang\s+sa\s+Pag-?aadal\b|\b(?:Uno\s+a|Tauno|Siisay\s+a|Maati\s+a)\b"
)


def clean_tagalog(text: str):
    """Clean one Tagalog document. Returns (narrative, flag)."""
    bilingual = bool(_BIKOL_SIGNAL.search(text))

    text = bd._basic_clean(text)
    # Strip a stray "Title,Level," CSV prefix that leaked into some level files
    # (e.g. "Kubo ni Itay,2,Si itay ay …" -> "Si itay ay …").
    text = regex.sub(r"^\s*[^,\n]{1,60},\s*[1-3]\s*,\s*(?=\p{Lu})", "", text)
    text = bd._CANVAS_LICENSE.sub(" ", text)   # leading CANVAS license sentence
    text = _drop_english(text)                 # English license / credit sentences
    text = _TL_BOILER.sub(" ", text)           # short English boilerplate fragments
    text = _TL_LABEL.sub(" ", text)            # "Writer:/Illustrator:/Guhit ni …"
    text = _TL_CREDIT.sub(" ", text)           # Tagalog/Bikol author credits
    text = _TL_STRUCT.sub(" ", text)           # GRADE 1 / DONATED PROPERTY / Theme:
    text = regex.sub(r"\s+", " ", text).strip()

    # A few DepEd decodables open with an institutional header block
    # ("Republic of the Philippines Department of Education … Elementary School")
    # that sits BEFORE the title and story.  If an institutional marker appears in
    # the leading front matter, delete that header span in place (up to the first
    # credit/title boundary) rather than truncating the whole document at it.
    lead = _TL_INSTITUTION.search(text)
    if lead and lead.start() < 60:
        # cut from the header to just before the first native sentence start, so
        # the trailing story survives; _narrative_start (below) finds the prose.
        text = text[lead.start():]
        s2, _ = bd._narrative_start(text)
        text = text[s2:] if s2 is not None else text

    # Cut the comprehension-question / publisher / institutional tails (whichever
    # comes first past the opening).  Everything from the earliest anchor to
    # end-of-doc is the activity or production-credits section, not narrative.
    cut = len(text)
    for pat in (_TL_QA_HEAD, _TL_PUB, _TL_INSTITUTION):
        m = pat.search(text)
        if m and m.start() >= 40:          # ignore a still-leading remnant
            cut = min(cut, m.start())
    body = text[:cut]

    # Strip page-number markers, then collapse whitespace.
    body = _PAGE_NUM.sub(" ", body)
    body = regex.sub(r"\s+", " ", body).strip()

    # Drop the leading title + author-name run (capitalised tokens before the
    # first real Tagalog sentence).  bd._narrative_start's rule (b) — a capital
    # word followed by a lowercase native word — locates the prose start.
    start, skipped = bd._narrative_start(body)
    if start is not None:
        body = body[start:]
        skipped = skipped.strip()
        flag = "" if (not skipped or bd._META_RE.search(skipped)) else "residual_prefix"
    else:
        flag = "no_opener"

    body = bd._cut_bio_tail(body)[0]           # trailing About-the-Author bio
    # Final-pass tail cut: catch any publisher/license anchor exposed only after
    # the front-matter trim (insurance against ordering effects).
    m = _TL_PUB.search(body)
    if m:
        body = body[: m.start()]
    field = bd._to_field(body)

    # Final safety net by Tagalog function-word density.  Real narrative is dense
    # with ang/ng/sa/na/si/ni…; a multi-word result with almost none is residual
    # credit / admin / English boilerplate, not a story, so it is dropped and
    # logged.  Density (rather than a signature blocklist) keeps genuine stories —
    # even those with an all-caps title or a leading author block — that a keyword
    # blocklist would wrongly discard.
    ftoks = regex.findall(r"\p{L}+", field.lower())
    if len(ftoks) >= 5:
        nat_ratio = sum(1 for w in ftoks if w in _TL_FUNC) / len(ftoks)
        if nat_ratio < 0.08:
            return field, "residual_boilerplate"
    if bilingual:
        flag = "bilingual"                     # Bikol body present: flag, keep
    return field, flag


def load_tagalog():
    """Load all three Tagalog level files. Returns rows as
    (language, level, title, narrative, flag) — matching build_datasets."""
    rows = []
    for lvl in (1, 2, 3):
        path = TAG_GLOB.format(lvl=lvl)
        if not os.path.isfile(path):
            continue
        for line in open(path, encoding="utf-8", errors="replace").read().splitlines():
            if not line.strip():
                continue
            narrative, flag = clean_tagalog(line)
            title = " ".join(narrative.split()[:6])     # derived; no title field exists
            rows.append(("tagalog", lvl, title, narrative, flag))
    return rows


if __name__ == "__main__":
    # Standalone run: write tagalog_level{1,2,3}.txt + a flag log, without
    # touching the combined corpus (that wiring lives in build_datasets.main).
    os.makedirs(bd.OUT_DIR, exist_ok=True)
    rows = load_tagalog()
    from collections import Counter
    counts = Counter()
    for lvl in (1, 2, 3):
        fp = os.path.join(bd.OUT_DIR, f"tagalog_level{lvl}.txt")
        with open(fp, "w", encoding="utf-8", newline="\n") as out:
            out.write(f"language{SEP}level{SEP}text\n")
            for L, l, title, text, flag in rows:
                if l == lvl and text.strip() and len(text.split()) >= 4 \
                        and flag not in {"empty", "too_short"}:
                    out.write(f"{L}{SEP}{l}{SEP}{text}\n")
                    counts[lvl] += 1
    print("Tagalog rows written per level:", dict(counts))
    print("Flags:", dict(Counter(r[4] for r in rows if r[4])))
