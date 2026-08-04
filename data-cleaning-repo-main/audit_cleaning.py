"""
audit_cleaning.py — An indexed ledger of everything the cleaner removed.

For every source document it records each removed span with a character index,
so the cleaning can be reviewed and validated word-by-word instead of trusted.
The ledger is produced by diffing the encoding-normalized raw text against the
final cleaned narrative, so it captures exactly what cleaning deleted.

Each removed span is classified (title, credit, genre tag, metadata label, end
marker, author bio, publisher, foreign/entity, names, or — the one that matters —
REVIEW_narrative: a span that contains native function words but matches no known
metadata pattern, i.e. a possible false-positive deletion of real story text).

Outputs (pipe-delimited, one row per removed span):
    output/cleaning_ledger.txt   every removal, with index + category
    output/cleaning_review.txt    only the REVIEW_narrative spans, worst first

Run:  python audit_cleaning.py
"""

import os
import glob
import difflib
from collections import Counter

import regex

import build_datasets as bd

SEP = "|"
OUT_DIR = "output"

ARA = [("ara-close-lang/data/cebuano/ceb_all_data.txt", "cebuano"),
       ("ara-close-lang/data/bikol/bik_all_data.txt", "bikol")]

# Role nouns that only occur in author/illustrator/project bios & acknowledgments.
_BIO_WORDS = regex.compile(
    r"(?i)\b(?:kagsurat|para?surat|para?kurit|ilustrador|tagsulat|tagguhit|"
    r"tagdrowing|awtor|manunulat|proyekto|paradibuho|paradrowing|kagdrowing)\b")
_LANGNAME = regex.compile(r"(?i)\b(?:" + bd._LANG_ALT + r")\b")


def classify(span, title, position):
    """Best-effort category for a removed span. Order = most specific first."""
    s = span.strip()
    if not s:
        return "whitespace"
    low = s.lower()
    natives = [w for w in bd._WORD_RE.findall(low) if w in bd._NATIVE_FUNC]

    if regex.search(r"\bKATAPUSAN\b", s, regex.IGNORECASE):
        return "end_marker"
    if bd._BIO_TAIL.search(s) or _BIO_WORDS.search(s):
        return "author_bio"
    if bd._CREDIT.search(s) or regex.search(bd._TRIG, s, regex.IGNORECASE):
        return "credit"
    if bd._PUB_ONLY.search(s):
        return "publisher"
    if bd._GENRE.search(s) or bd._GENRE_PHRASE.search(s) or _LANGNAME.search(s):
        return "genre_tag"
    if bd._ARA_META.search(s):
        return "meta_label"
    if bd._FOREIGN_TOK.search(s) or regex.search(r"&[a-zA-Z#0-9]+;", s):
        return "foreign_entity"

    toks = bd._WORD_RE.findall(s)
    cores = [t.lower() for t in toks]
    title_cores = bd._WORD_RE.findall(title.lower())

    # Leading title block (+ a trailing author-name run).  The title is written in
    # the target language, so it legitimately carries function words; match it on
    # letter-cores so punctuation differences ("Si Pogi, Si Pogs") don't defeat it.
    # Only if native words remain BETWEEN the title and the author names is the span
    # a real suspect (dialogue/story text pulled into the front-matter cut).
    if position == "leading" and title_cores and cores[:len(title_cores)] == title_cores:
        rest = toks[len(title_cores):]
        k = len(rest)
        while k > 0 and rest[k - 1][:1].isupper():   # peel trailing capitalized names
            k -= 1
        middle = [t.lower() for t in rest[:k]]
        if not any(w in bd._NATIVE_FUNC for w in middle):
            return "title_author"

    # A run of only capitalized tokens with no native function word = a name /
    # title fragment (author-name run or a repeated Title Case heading).
    if toks and not natives and sum(1 for t in toks if t[:1].isupper()) >= len(toks) - 1:
        return "names_or_title"
    if title and low and title.lower() in low:
        return "title"
    if natives:
        return "REVIEW_narrative"      # native words, no metadata signature → suspect
    return "other"


def removed_spans(raw_text, cleaned):
    """Yield (char_start, char_end, position, span_text) for each removed run,
    indexed into the encoding-normalized raw text."""
    ref = bd._basic_clean(raw_text)
    ref_toks = [(m.group().lower(), m.start(), m.end())
                for m in bd._WORD_RE.finditer(ref)]
    ref_cores = [t[0] for t in ref_toks]
    cln_cores = [w.lower() for w in bd._WORD_RE.findall(cleaned)]

    sm = difflib.SequenceMatcher(a=ref_cores, b=cln_cores, autojunk=False)
    n = len(ref_cores)
    for tag, i1, i2, _j1, _j2 in sm.get_opcodes():
        if tag == "equal" or i1 == i2:
            continue
        start = ref_toks[i1][1]
        end = ref_toks[i2 - 1][2]
        position = ("leading" if i1 == 0 else
                    "trailing" if i2 == n else "interior")
        yield start, end, position, ref[start:end], (i2 - i1)


def iter_docs():
    for path, lang in ARA:
        if not os.path.isfile(path):
            continue
        with open(path, encoding="utf-8", errors="replace") as f:
            for line in f:
                m = bd._LINE.match(line.rstrip("\n"))
                if not m:
                    continue
                rt = m.group("title").strip()
                title = (rt.split("___")[0].replace("_", " ")
                         if lang == "bikol" else rt)
                cleaned, _flag = bd.clean_ara(m.group("text"), title)
                yield lang, int(m.group("level")), title, m.group("text"), cleaned

    for lang in bd.BASAHA_LANGS:
        for grade, level in [("grade 1", 1), ("grade 2", 2), ("grade 3", 3)]:
            pat = os.path.join(bd.BASAHA_BASE, lang, grade, "*.txt")
            for fp in sorted(glob.glob(pat)):
                with open(fp, encoding="utf-8", errors="replace") as f:
                    raw = f.read()
                cleaned, _flag = bd.clean_basaha(raw)
                title = os.path.basename(fp)[:-4].split("___")[0].replace("_", " ")
                yield lang, level, title, raw, cleaned


def field(s):
    return regex.sub(r"\s+", " ", s).strip().replace(SEP, "/")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    ledger, review = [], []
    cat_words = Counter()
    docs = 0

    for lang, level, title, raw, cleaned in iter_docs():
        docs += 1
        for start, end, position, span, nwords in removed_spans(raw, cleaned):
            cat = classify(span, title, position)
            row = (lang, level, title, start, end, nwords, position, cat, field(span))
            ledger.append(row)
            cat_words[cat] += nwords
            if cat == "REVIEW_narrative":
                review.append(row)

    header = SEP.join(["language", "level", "title", "char_start", "char_end",
                       "words", "position", "category", "removed_text"]) + "\n"
    with open(os.path.join(OUT_DIR, "cleaning_ledger.txt"), "w",
              encoding="utf-8", newline="\n") as f:
        f.write(header)
        for r in ledger:
            f.write(SEP.join(str(x) for x in r) + "\n")

    review.sort(key=lambda r: r[5], reverse=True)     # worst (most words) first
    with open(os.path.join(OUT_DIR, "cleaning_review.txt"), "w",
              encoding="utf-8", newline="\n") as f:
        f.write(header)
        for r in review:
            f.write(SEP.join(str(x) for x in r) + "\n")

    total_words = sum(cat_words.values())
    print(f"Documents audited: {docs}")
    print(f"Removed spans: {len(ledger)}   Removed words: {total_words}")
    print("\nWords removed by category (higher = more removed):")
    for cat, w in cat_words.most_common():
        bar = "#" * int(40 * w / max(total_words, 1))
        print(f"  {cat:16s} {w:6d}  {bar}")
    review_words = cat_words.get("REVIEW_narrative", 0)
    print(f"\nREVIEW_narrative: {len(review)} spans / {review_words} words "
          f"({100*review_words/max(total_words,1):.1f}% of all removals) "
          f"-> output/cleaning_review.txt")
    print("Full ledger -> output/cleaning_ledger.txt")


if __name__ == "__main__":
    main()
