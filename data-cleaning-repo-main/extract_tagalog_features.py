"""
extract_tagalog_features.py — Features + n-grams for the cleaned Tagalog corpus.

Reads the cleaned, narrative-only files produced by clean_tagalog_levels.py

    data/tag_lvl{1,2,3}_cleaned_data.txt   (records separated by "\\n\\n=====\\n\\n")

and, for every document, computes the same feature set as the ara-close-lang
reference (data/tagalog/tag_features.csv), reusing that repo's extractors:

  * TRAD.py    — surface readability (word/sentence/syllable counts, polysyllables)
  * SYLL.py    — phonotactic syllable-shape densities (V, CV, VC, CVC … CCVCCC)
  * CLGSNGO.py — cross-lingual character bi/trigram similarity to the Tagalog,
                 Bikol and Cebuano top-25 anchor profiles

Outputs
  output/tagalog_features.csv         one row per document (ara schema + grade_level)
  output/tagalog_top25_2gram.txt      corpus's own top-25 character bigrams  (ngram,count)
  output/tagalog_top25_3gram.txt      corpus's own top-25 character trigrams (ngram,count)

The grade_level column is the source level (1/2/3).  Surface/phonotactic features
run on an accent-folded copy (à→a, ñ→n …) for the ASCII-only TRAD/SYLL inventories;
CLGSNGO keeps the unfolded text (it strips non-ASCII itself), matching the
published CROSSNGO behaviour.
"""

import os
import sys
import shutil
import unicodedata
from collections import Counter

import pandas as pd
import nltk

SRC = "data/tag_lvl{lvl}_cleaned_data.txt"
OUT_CSV = "output/tagalog_features.csv"
OUT_2GRAM = "output/tagalog_top25_2gram.txt"
OUT_3GRAM = "output/tagalog_top25_3gram.txt"
LEVELS = (1, 2, 3)
SEP = "\n\n=====\n\n"

nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)

# CLGSNGO.py reads its anchor lists from the hardcoded relative dir "ngrams list/".
_NGRAM_SRC = "ara-close-lang/data/ngrams"
_NGRAM_DST = "ngrams list"
os.makedirs(_NGRAM_DST, exist_ok=True)
for _f in os.listdir(_NGRAM_SRC):
    if not os.path.exists(os.path.join(_NGRAM_DST, _f)):
        shutil.copy2(os.path.join(_NGRAM_SRC, _f), os.path.join(_NGRAM_DST, _f))

sys.path.insert(0, "ara-close-lang/code")
from TRAD import (                                    # noqa: E402
    word_count_per_doc, sentence_count_per_doc, ave_word_length,
    ave_syllable_count_of_word, polysyll_count_per_doc, ave_phrase_count_per_doc,
)
from SYLL import (                                    # noqa: E402
    get_consonant_cluster, get_v, get_cv, get_vc, get_cvc, get_vcc,
    get_cvcc, get_ccv, get_ccvc, get_ccvcc, get_ccvccc,
)
from CLGSNGO import (                                 # noqa: E402
    get_bigram_CLGSNGO, get_trigram_CLGSNGO, clean as ngram_clean, get_ngrams,
)


def _fold(text: str) -> str:
    """Accent-fold for the ASCII-only TRAD/SYLL inventories (counts accented
    native vowels instead of dropping them)."""
    return "".join(c for c in unicodedata.normalize("NFD", text)
                   if not unicodedata.combining(c))


def _safe(fn, *args, default=0.0):
    try:
        return fn(*args)
    except ZeroDivisionError:
        return default


def _doc_features(text: str, level: int, doc_id: str) -> dict:
    tf = _fold(text)
    wc = word_count_per_doc(tf)
    sc = sentence_count_per_doc(tf)
    row = {
        "book_title": doc_id,
        "word_count": wc,
        "sentence_count": sc,
        "phrase_count_per_sentence": _safe(ave_phrase_count_per_doc, tf),
        "average_word_len": _safe(ave_word_length, tf),
        "average_sentence_len": (wc / sc if sc else float(wc)),
        "average_syllable_count": _safe(ave_syllable_count_of_word, tf),
        "polysyll_count": polysyll_count_per_doc(tf),
        "consonant_cluster_density": get_consonant_cluster(tf),
        "v_density": get_v(tf),
        "cv_density": get_cv(tf),
        "vc_density": get_vc(tf),
        "cvc_density": get_cvc(tf),
        "vcc_density": get_vcc(tf),
        "cvcc_density": get_cvcc(tf),
        "ccvc_density": get_ccvc(tf),
        "ccv_density": get_ccv(tf),
        "ccvcc_density": get_ccvcc(tf),
        "ccvccc_density": get_ccvccc(tf),
    }
    row["tagalog_bigram_sim"], row["bikol_bigram_sim"], row["cebuano_bigram_sim"] = \
        get_bigram_CLGSNGO(text)
    row["tagalog_trigram_sim"], row["bikol_trigram_sim"], row["cebuano_trigam_sim"] = \
        get_trigram_CLGSNGO(text)
    row["grade_level"] = level
    return row


def _load_docs():
    """Yield (level, index, text) for every cleaned narrative record."""
    for lvl in LEVELS:
        path = SRC.format(lvl=lvl)
        if not os.path.isfile(path):
            print(f"  lvl{lvl}: {path} not found, skipped")
            continue
        docs = [d.strip() for d in open(path, encoding="utf-8").read().split(SEP)
                if d.strip()]
        for i, d in enumerate(docs):
            yield lvl, i, d


def _write_top_ngrams(corpus: str, n: int, path: str, top: int = 25):
    """Top-`top` CLGSNGO-style character n-grams (lowercased a-z, no spaces)."""
    counts = Counter(get_ngrams(ngram_clean(corpus), n))
    with open(path, "w", encoding="utf-8", newline="\n") as out:
        for gram, cnt in counts.most_common(top):
            out.write(f"{gram},{cnt}\n")
    return counts.most_common(top)


def main():
    docs = list(_load_docs())
    print(f"Loaded {len(docs)} cleaned Tagalog documents")

    rows = [_doc_features(text, lvl, f"tag_l{lvl}_{i:03d}")
            for lvl, i, text in docs]
    df = pd.DataFrame(rows)
    os.makedirs("output", exist_ok=True)
    df.to_csv(OUT_CSV, index=False)
    print(f"Saved {len(df)} rows x {len(df.columns)} cols -> {OUT_CSV}")
    print("Per-level counts:", df.groupby("grade_level").size().to_dict())

    corpus = " ".join(text for _, _, text in docs)
    top2 = _write_top_ngrams(corpus, 2, OUT_2GRAM)
    top3 = _write_top_ngrams(corpus, 3, OUT_3GRAM)
    print(f"Top-25 character n-grams -> {OUT_2GRAM}, {OUT_3GRAM}")
    print("  top bigrams:", [g for g, _ in top2[:8]])
    print("  top trigrams:", [g for g, _ in top3[:8]])


if __name__ == "__main__":
    main()
