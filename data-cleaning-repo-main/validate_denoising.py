"""
validate_denoising.py — Native-speaker-free validation of the denoised outputs.

The problem this solves: with no native speakers available, how do we check that
cleaning (build_datasets.py) + normalization (normalize_datasets.py) removed only
noise and left the native narrative intact?  The answer exploits the fact that our
"denoising" only DELETES structurally- or English-identifiable spans (titles,
credits, KATAPUSAN, English genre tags, publisher boilerplate) — it never rewrites
target-language text.  Validating deletion is a structural judgment, not a semantic
one, so it needs no fluency.  Three checks:

  Part A — Extraction faithfulness (Cebuano, Bikol).
      The original authors' published feature files (data/<lang>/trad.csv) were
      produced by their own scripts, whose cleaner() only strips non-alphabetic
      characters — it does NOT remove titles/credits/KATAPUSAN.  So running the
      UNMODIFIED original TRAD.py on our ingested RAW text must reproduce their
      published word_count / sentence_count exactly.  A high match rate proves our
      ingestion + tokenization + extraction is faithful, with zero involvement from
      our cleaning.  This is the ground-truth anchor.

  Part B — Denoising accountability (Cebuano, Bikol).
      Because Part A pins the extractor, every token our cleaning removes must be
      metadata.  For each document we verify (i) the cleaned narrative is a
      deletion-only subset of the raw (cleaning never invents tokens), and (ii) any
      NATIVE function word that was removed is explained by the title — never by
      narrative being eaten.  Documents that fail go to a small review queue; those
      are the ONLY documents a human would need to look at.

  Part C — Structural invariants (all shipped outputs).
      - normalization preserves every letter/accent (ñ, accents, Rinconada schwa)
      - no English publisher/boilerplate leaked into the final narratives
      - normalize_text is idempotent
      - no empty / degenerate rows survived to the outputs

Run:    python validate_denoising.py
Writes: output/validation_report.txt   (full summary)
        output/validation_queue.txt     (docs needing a human glance)
"""

import os
import csv
import sys
import difflib
import unicodedata
from collections import Counter

import regex

# Original, UNMODIFIED extractor (its cleaner strips only non-alpha — see module).
sys.path.append(os.path.join("ara-close-lang", "code"))
from TRAD import word_count_per_doc, sentence_count_per_doc  # noqa: E402

# Our own pipeline modules — importing runs no work (main() is __main__-guarded).
import build_datasets as bd            # noqa: E402
from normalize_datasets import normalize_text  # noqa: E402

SEP = "|"
OUT_DIR = "output"
NORM_DIR = "output_normalized"

# Languages that ship BOTH raw text and the authors' published features.
ARA = {
    "cebuano": ("ara-close-lang/data/cebuano/ceb_all_data.txt",
                "ara-close-lang/data/cebuano/trad.csv"),
    "bikol":   ("ara-close-lang/data/bikol/bik_all_data.txt",
                "ara-close-lang/data/bikol/trad.csv"),
}

_WORD = regex.compile(r"\p{L}+")

# Closed-class native function words across the six languages (from build_datasets'
# _OPENERS and _PHIL_FUNC).  A credit / genre / KATAPUSAN / publisher span contains
# NONE of these — only names and English.  So a removed native function word that
# the title cannot account for is the signature of narrative being deleted.
_NATIVE_FUNC = set(bd._OPENERS) | {
    "ang", "an", "si", "sa", "mga", "ng", "ni", "na", "at", "kag", "ug", "kay",
    "sang", "ini", "ito", "siya", "ako", "kami", "sinda", "nag", "mag", "kan",
    "sin", "nin", "dili", "wala", "amo", "man", "lang", "ta", "ba", "gid", "da",
    "pa", "ra", "ya", "may", "isa", "ka", "kita", "kini", "kana", "usa", "asin",
    "saka", "nga", "sang", "y",
}


def toks(s):
    return _WORD.findall(s.lower())


def iter_raw(path):
    """Yield (raw_title, level, text) from an ara-close-lang flat file."""
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line.strip():
                continue
            m = bd._LINE.match(line)
            if not m:
                continue
            yield m.group("title").strip(), int(m.group("level")), m.group("text")


# ── Part A ────────────────────────────────────────────────────────────────────

def part_a(lang):
    raw_path, csv_path = ARA[lang]
    ref = {}
    with open(csv_path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            ref[(row["book_title"].strip(), int(row["grade_level"]))] = (
                int(row["word_count"]), int(row["sentence_count"]))
    total = both = wc = sc = missing = 0
    mism = []
    for raw_title, level, text in iter_raw(raw_path):
        key = (raw_title, level)                  # (title, grade) — titles are not unique
        if key not in ref:
            missing += 1
            continue
        total += 1
        w, s = word_count_per_doc(text), sentence_count_per_doc(text)
        rw, rs = ref[key]
        wc += (w == rw)
        sc += (s == rs)
        if w == rw and s == rs:
            both += 1
        else:
            mism.append((raw_title, (w, s), (rw, rs)))
    return dict(total=total, both=both, wc=wc, sc=sc, missing=missing, mism=mism)


# ── Part B ────────────────────────────────────────────────────────────────────

def part_b(lang):
    raw_path, _ = ARA[lang]
    n = introduced = 0
    queue = []
    for raw_title, level, text in iter_raw(raw_path):
        n += 1
        clean_title = (raw_title.split("___")[0].replace("_", " ")
                       if lang == "bikol" else raw_title)
        narrative, flag = bd.clean_ara(text, clean_title)

        raw_seq = toks(bd._basic_clean(text))        # deletion baseline (post-ftfy)
        cln_seq = toks(narrative)

        if not cln_seq:
            queue.append((lang, level, raw_title, "over_deleted_empty",
                          f"raw={len(raw_seq)}w -> 0w (flag={flag or '-'})"))
            continue

        # Align cleaned against raw.  Cleaning only deletes, so the narrative should
        # be a contiguous run of raw with a deleted PREAMBLE (leading) and a deleted
        # TAIL (trailing).  A delete block that is neither leading nor trailing is an
        # INTERIOR cut; if it carries native function words it is prose being removed
        # from the middle of the story — the one pattern worth a human's eyes.  A
        # leading credit's "ni"/"by" or a genre tag never trips this, because those
        # deletions sit at position 0.
        sm = difflib.SequenceMatcher(a=raw_seq, b=cln_seq, autojunk=False)
        invented = []
        interior = Counter()
        example = ""
        for tag, i1, i2, j1, j2 in sm.get_opcodes():
            if tag == "equal":
                continue
            invented.extend(cln_seq[j1:j2])          # cleaned tokens with no raw match
            block = raw_seq[i1:i2]
            if i1 == 0 or i2 == len(raw_seq):
                continue                             # leading preamble / trailing tail
            fw = [w for w in block if w in _NATIVE_FUNC]
            if fw:
                interior.update(fw)
                if not example:
                    example = " ".join(block[:12])

        if invented:
            introduced += 1
            queue.append((lang, level, raw_title, "introduced_token",
                          "cleaning changed/added tokens: "
                          + " ".join(sorted(set(invented))[:8])))
        elif sum(interior.values()) >= 2:
            det = " ".join(f"{w}x{c}" for w, c in interior.most_common())
            queue.append((lang, level, raw_title, "interior_deletion",
                          f"removed mid-narrative native words [{det}] "
                          f"near: '{example}' (flag={flag or '-'})"))
    return dict(n=n, introduced=introduced, queue=queue)


# ── Part C ────────────────────────────────────────────────────────────────────

def read_rows(path):
    rows = []
    if not os.path.isfile(path):
        return rows
    with open(path, encoding="utf-8") as f:
        f.readline()  # header
        for line in f:
            line = line.rstrip("\n")
            if not line.strip():
                continue
            p = line.split(SEP, 2)
            if len(p) == 3:
                rows.append((p[0].strip(), int(p[1]), p[2]))
    return rows


def letters(s):
    """NFC-normalized multiset of letter codepoints (accents composed on both sides
    so decomposed-vs-composed is not counted as a difference)."""
    s = unicodedata.normalize("NFC", s)
    return Counter(c for c in s if c.isalpha())


def part_c():
    """Validate the normalize_text FUNCTION on the clean rows (source of truth),
    then separately check the on-disk output_normalized/ files are up to date.
    Correctness never depends on the possibly-stale normalized files."""
    _PUB_RE = regex.compile(bd._PUB, regex.IGNORECASE)
    langs = ["tagalog", "cebuano", "bikol", "hiligaynon", "minasbate",
             "karay-a", "rinconada"]

    grapheme_loss = []     # normalize_text dropped/changed a letter (accent safety)
    non_idem = 0           # normalize_text(normalize_text(t)) != normalize_text(t)
    leakage = []           # residual English/publisher boilerplate in a narrative
    degenerate = []        # <4-word rows that reached the outputs
    stale = []             # output_normalized bucket not reproduced from output/
    checked = 0

    for lang in langs:
        for level in (1, 2, 3):
            clean_rows = read_rows(os.path.join(OUT_DIR, f"{lang}_level{level}.txt"))
            if not clean_rows:
                continue

            # (1) function-level checks on the authoritative clean text
            expected_norm = []
            for _, _, ctext in clean_rows:
                checked += 1
                nt = normalize_text(ctext)
                expected_norm.append(nt)
                if letters(ctext) != letters(nt):            # accent / letter drop
                    lost = letters(ctext) - letters(nt)
                    grapheme_loss.append((lang, level,
                                          "".join(sorted(lost.elements()))[:20],
                                          ctext[:50]))
                if normalize_text(nt) != nt:                 # idempotence
                    non_idem += 1
                if not ctext.strip() or len(ctext.split()) < 4:
                    degenerate.append((lang, level, ctext[:60]))
                if _PUB_RE.search(ctext) or bd._english_ratio(ctext) >= 0.40:
                    m = _PUB_RE.search(ctext)
                    why = f"boilerplate:'{m.group()}'" if m else \
                          f"english_ratio={bd._english_ratio(ctext):.2f}"
                    leakage.append((lang, level, why, ctext[:70]))

            # (2) freshness of the shipped normalized file vs what we just computed
            norm_rows = read_rows(os.path.join(NORM_DIR, f"{lang}_level{level}.txt"))
            on_disk = [t for _, _, t in norm_rows]
            if on_disk != expected_norm:
                stale.append((lang, level, len(clean_rows), len(on_disk)))

    return dict(grapheme_loss=grapheme_loss, non_idem=non_idem, leakage=leakage,
                degenerate=degenerate, stale=stale, checked=checked)


# ── Report ────────────────────────────────────────────────────────────────────

def main():
    lines = []
    queue = []

    def out(s=""):
        print(s)
        lines.append(s)

    out("=" * 72)
    out("DENOISING VALIDATION  (no native speaker required)")
    out("=" * 72)

    out("\n[Part A] Extraction faithfulness — original scripts on RAW text must")
    out("         reproduce the authors' published features exactly.")
    for lang in ARA:
        a = part_a(lang)
        out(f"  {lang:9s}  exact(word+sentence): {a['both']}/{a['total']}"
            f"   word_count: {a['wc']}/{a['total']}"
            f"   sentence_count: {a['sc']}/{a['total']}"
            f"   (unmatched titles: {a['missing']})")
        for title, got, ref in a["mism"][:5]:
            out(f"      MISMATCH  {title!r}  got{got}  published{ref}")

    out("\n[Part B] Denoising accountability — removed tokens must be metadata.")
    for lang in ARA:
        b = part_b(lang)
        n_del = sum(1 for q in b["queue"] if q[3] == "interior_deletion")
        n_emp = sum(1 for q in b["queue"] if q[3] == "over_deleted_empty")
        out(f"  {lang:9s}  docs: {b['n']}   deletion-only invariant broken: "
            f"{b['introduced']}   over-deleted->empty: {n_emp}   "
            f"interior-deletion (needs review): {n_del}")
        queue += b["queue"]

    out("\n[Part C] Structural invariants (normalize_text on the clean rows).")
    c = part_c()
    out(f"  grapheme/accent preservation: "
        f"{c['checked'] - len(c['grapheme_loss'])}/{c['checked']} rows keep every letter")
    for lang, lvl, lost, ex in c["grapheme_loss"][:5]:
        out(f"      LETTERS LOST  {lang} L{lvl}  '{lost}'  :: {ex!r}")
    out(f"  normalize_text idempotent: "
        f"{c['checked'] - c['non_idem']}/{c['checked']} rows")
    out(f"  English/publisher leakage in narratives: {len(c['leakage'])}")
    for lang, lvl, why, ex in c["leakage"][:8]:
        out(f"      LEAK  {lang} L{lvl}  {why}  :: {ex!r}")
    if c["degenerate"]:
        out(f"  degenerate (<4 words) rows: {len(c['degenerate'])}")
    if c["stale"]:
        out(f"  STALE output_normalized/ (not reproduced from output/): "
            f"{[(l, lv) for l, lv, _, _ in c['stale']]}")
        out(f"      -> re-run normalize_datasets.py; note it also OMITS tagalog "
            f"from its LANGUAGES list, so Tagalog is never normalized.")

    # Fold Part C leakage into the human queue too.
    for lang, lvl, why, ex in c["leakage"]:
        queue.append((lang, lvl, "-", "boilerplate_leak", f"{why} :: {ex}"))

    out("\n" + "-" * 72)
    out(f"HUMAN REVIEW QUEUE: {len(queue)} document(s) need a glance "
        f"(everything else is verified structurally).")
    out("-" * 72)

    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, "validation_report.txt"), "w",
              encoding="utf-8", newline="\n") as f:
        f.write("\n".join(lines) + "\n")
    with open(os.path.join(OUT_DIR, "validation_queue.txt"), "w",
              encoding="utf-8", newline="\n") as f:
        f.write(f"language{SEP}level{SEP}title{SEP}reason{SEP}detail\n")
        for lang, lvl, title, reason, detail in queue:
            f.write(f"{lang}{SEP}{lvl}{SEP}{title}{SEP}{reason}{SEP}{detail}\n")

    out(f"\nWrote {OUT_DIR}/validation_report.txt and {OUT_DIR}/validation_queue.txt")


if __name__ == "__main__":
    main()
