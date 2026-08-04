"""
normalize_datasets.py — Structural normalization of the cleaned narratives.

This is the 3.1.3 step that follows cleaning (build_datasets.py).  It is
deliberately *structural only*: it makes the already-clean native narrative
consistent for tokenization and feature extraction.  It does NOT do linguistic
normalization — no stemming, no lemmatization — because in these agglutinative
languages the affixes (nag-, mag-, naka-, gi-, pa-, -an, -in, -on, -han) and
reduplication carry the readability signal the model is built to capture.

Input : output/all_languages.txt          (cleaned, RAW narrative — kept as-is)
Output: output_normalized/<lang>_level<N>.txt  (18 files) + all_languages.txt
        same schema:  language|level|text

The cleaned output/ files are never modified, so the raw narrative remains
available for the noisy-vs-denoised comparison later (3.1.5).
"""

import os
import unicodedata
import regex

IN_MASTER = os.path.join("output", "all_languages.txt")
OUT_DIR = "output_normalized"
SEP = "|"

LANGUAGES = ["tagalog", "cebuano", "bikol", "hiligaynon", "minasbate",
             "karay-a", "rinconada"]
LEVELS = (1, 2, 3)

# ── HTML-entity safety net ─────────────────────────────────────────────────────
# build_datasets already strips these; kept here so the normalizer is correct on
# any input, not only build_datasets output.  Cheap.
_HTML_ENT = [
    (regex.compile(r"&nbsp;"), " "),
    (regex.compile(r"&quot;"), '"'),
    (regex.compile(r"&amp;"),  "&"),
    (regex.compile(r"&#\d+;"), ""),
    (regex.compile(r"&[a-zA-Z]+;"), ""),
]

# Insert a missing space after closing punctuation before a letter.
# NOTE: the straight double quote " is intentionally EXCLUDED — it is ambiguous
# (open vs close), and including it would split opening-quote dialogue such as
# "Maayo -> " Maayo.  Unicode-aware: \p{L} treats ñ, accents, and the schwa as
# letters, so a space is not wrongly inserted at those boundaries.
_PUNCT_SPACE = regex.compile(r'([,.!?;:\)])\s*(?=\p{L})')

# Join spaced affix / reduplication hyphens: "mag -uban" -> "mag-uban",
# "dali - dali" -> "dali-dali".  Both sides must be letters, so a numeric or
# punctuation neighbour is left alone.  (See README caveat: in this corpus " - "
# is affix/reduplication only; clause dashes use em/en dashes, handled below.)
_HYPHEN_JOIN = regex.compile(r'(?<=\p{L})\s*-\s*(?=\p{L})')
_HYPHEN_MULTI = regex.compile(r'(?<=\p{L})-{2,}(?=\p{L})')

# Normalize spaced em/en dashes used as clause separators to a single form.
# Kept distinct from the ASCII hyphen so word-internal hyphens are not touched.
_DASH = regex.compile(r'\s*[—–]\s*')

_WS = regex.compile(r'[ \t]+')


def normalize_text(text: str) -> str:
    # 1. Canonical Unicode form so accented vowels / ñ / schwa are single
    #    codepoints (composed), not letter + combining mark.
    text = unicodedata.normalize("NFC", text)
    # 2. HTML-entity safety net
    for pat, repl in _HTML_ENT:
        text = pat.sub(repl, text)
    # 3. Space after closing punctuation before a letter (quote excluded)
    text = _PUNCT_SPACE.sub(r"\1 ", text)
    # 4. Join affix / reduplication hyphens; collapse multi-hyphens between letters
    text = _HYPHEN_JOIN.sub("-", text)
    text = _HYPHEN_MULTI.sub("-", text)
    # 5. Normalize spaced em/en dashes (clause separators)
    text = _DASH.sub(" — ", text)
    # 6. Collapse internal whitespace
    text = _WS.sub(" ", text)
    # 7. Morphological preservation: NO stemming, NO lemmatization.
    return text.strip()


def read_master(path):
    """Yield (language, level, text) data rows from a master file, skip header."""
    with open(path, encoding="utf-8") as f:
        first = f.readline()  # header: language|level|text
        for line in f:
            line = line.rstrip("\n")
            if not line.strip():
                continue
            parts = line.split(SEP, 2)
            if len(parts) < 3:
                continue
            lang, level, text = parts[0].strip(), parts[1].strip(), parts[2]
            yield lang, int(level), text


def main():
    if not os.path.isfile(IN_MASTER):
        raise SystemExit(f"missing {IN_MASTER} — run build_datasets.py first")
    os.makedirs(OUT_DIR, exist_ok=True)

    rows = [(L, lvl, normalize_text(t)) for L, lvl, t in read_master(IN_MASTER)]
    rows = [r for r in rows if r[2].strip()]   # guard: never emit an empty body

    written = {}
    master_path = os.path.join(OUT_DIR, "all_languages.txt")
    with open(master_path, "w", encoding="utf-8", newline="\n") as master:
        master.write(f"language{SEP}level{SEP}text\n")
        for lang in LANGUAGES:
            for level in LEVELS:
                fp = os.path.join(OUT_DIR, f"{lang}_level{level}.txt")
                n = 0
                with open(fp, "w", encoding="utf-8", newline="\n") as out:
                    out.write(f"language{SEP}level{SEP}text\n")
                    for L, lvl, text in rows:
                        if L == lang and lvl == level:
                            rec = f"{L}{SEP}{lvl}{SEP}{text}\n"
                            out.write(rec)
                            master.write(rec)
                            n += 1
                written[(lang, level)] = n

    print("Normalized rows per language/level:")
    for lang in LANGUAGES:
        print("  %-11s " % lang + "  ".join(
            f"L{lvl}={written[(lang, lvl)]:>3}" for lvl in LEVELS))
    print(f"\nTotal: {sum(written.values())}")
    print(f"Wrote 18 per-bucket files + {master_path}")


if __name__ == "__main__":
    main()
