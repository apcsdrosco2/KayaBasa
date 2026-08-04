"""
extract_features.py — Stage 3.1.4
Extracts TRAD, SYLL, and CLGSNGO features from the normalized corpus.
Input:  output_normalized/all_languages.txt   (language|level|text)
Output: output/all_features.csv               (keyed on doc_id)

The pipeline upstream of this stage emits per-language/level text files, not the
single serialized DataFrame the original draft assumed.  This script therefore
builds the keyed table itself: a stable doc_id, the integer label (= reading
level), the source language, and a split_role (ara-close-lang languages are the
high-resource training set; BasahaCorpus languages are the low-resource transfer
set).
"""

import os
import sys
import shutil
import unicodedata
import pandas as pd
import nltk
from nltk import word_tokenize

# ── Corpus loading / schema bridge ────────────────────────────────────────────

CORPUS_PATH = "output_normalized/all_languages.txt"
SEP = "|"
SPLIT_ROLE = {
    "tagalog": "high_resource",
    "cebuano": "high_resource", "bikol": "high_resource",
    "hiligaynon": "low_resource", "minasbate": "low_resource",
    "karay-a": "low_resource", "rinconada": "low_resource",
}


def load_corpus(path: str) -> pd.DataFrame:
    """Read language|level|text into a keyed DataFrame (doc_id, label, …)."""
    records = []
    for line in open(path, encoding="utf-8").read().splitlines()[1:]:  # skip header
        if not line.strip():
            continue
        language, level, text = line.split(SEP, 2)
        records.append({"language": language, "label": int(level), "text": text})
    df = pd.DataFrame(records)
    df.insert(0, "doc_id", range(len(df)))
    df["split_role"] = df["language"].map(SPLIT_ROLE)
    return df


def fold_accents(text: str) -> str:
    """Map accented Latin letters to their base form for the ASCII-only TRAD /
    SYLL inventories (à/á/â→a, ó/ô→o, é→e, ñ→n, š→s, ạ→a …) so accented native
    vowels are counted instead of silently stripped.  Applied only to the surface
    / phonotactic feature inputs; the stored corpus text is never modified, and
    CLGSNGO keeps the unfolded text (see extract_features)."""
    nfd = unicodedata.normalize("NFD", text)
    return "".join(c for c in nfd if not unicodedata.combining(c))

# ── NLTK data ─────────────────────────────────────────────────────────────────

nltk.download("punkt",     quiet=True)
nltk.download("punkt_tab", quiet=True)

# ── N-gram path fix ───────────────────────────────────────────────────────────
# CLGSNGO.py reads anchor lists from the hardcoded relative path "ngrams list/".
# We mirror the files from their actual location into that directory once.

_NGRAM_SRC = "ara-close-lang/data/ngrams"
_NGRAM_DST = "ngrams list"

os.makedirs(_NGRAM_DST, exist_ok=True)
for _fname in os.listdir(_NGRAM_SRC):
    _src = os.path.join(_NGRAM_SRC, _fname)
    _dst = os.path.join(_NGRAM_DST, _fname)
    if not os.path.exists(_dst):
        shutil.copy2(_src, _dst)

# ── Import feature scripts ────────────────────────────────────────────────────
# SYLL.py does `from TRAD import word_count_per_doc` at module level,
# so both must be importable from the same path entry.

sys.path.insert(0, "ara-close-lang/code")

from TRAD import (
    word_count_per_doc, sentence_count_per_doc,
    ave_word_length, word_count_per_sentence,
    ave_syllable_count_of_word, polysyll_count_per_doc,
    ave_phrase_count_per_doc,
)
from SYLL import (
    get_consonant_cluster, get_cv, get_cvc,
    get_ccvc, get_ccvccc,
)
from CLGSNGO import get_bigram_CLGSNGO, get_trigram_CLGSNGO

# ── Type-token ratio (project addition per §3.1.4.1) ─────────────────────────
# The repository scripts do not provide this; it is computed here on
# alphabetic tokens after normalization.

def type_token_ratio(text: str) -> float:
    toks = [t.lower() for t in word_tokenize(text) if t.isalpha()]
    return len(set(toks)) / len(toks) if toks else 0.0

# ── Safe wrappers for TRAD functions with latent divide-by-zero ──────────────
# word_count_per_sentence in TRAD.py guards sentence_count locally but still
# calls sentence_count_per_doc() again in the return, so the guard is a no-op.
# We replicate the correct arithmetic here.

def safe_mean_sentence_len(text: str) -> float:
    wc = word_count_per_doc(text)
    sc = sentence_count_per_doc(text)
    return wc / sc if sc > 0 else float(wc)

def safe_ave_syllable(text: str) -> float:
    try:
        return ave_syllable_count_of_word(text)
    except ZeroDivisionError:
        return 0.0

def safe_ave_word_len(text: str) -> float:
    try:
        return ave_word_length(text)
    except ZeroDivisionError:
        return 0.0

# ── Feature extraction per document ──────────────────────────────────────────

def extract_features(row) -> dict:
    t = row["text"]
    tf = fold_accents(t)        # accent-folded copy for the ASCII-only extractors
    feats = {"doc_id": row["doc_id"]}

    # Surface statistics (TRAD.py) — on folded text so accented words keep their
    # true length / syllable count instead of having the accent stripped.
    feats["word_count"]        = word_count_per_doc(tf)
    feats["sentence_count"]    = sentence_count_per_doc(tf)
    feats["mean_word_len"]     = safe_ave_word_len(tf)
    feats["mean_sentence_len"] = safe_mean_sentence_len(tf)
    feats["mean_syll_per_word"]= safe_ave_syllable(tf)
    feats["polysyll_count"]    = polysyll_count_per_doc(tf)
    feats["phrase_count"]      = ave_phrase_count_per_doc(tf)
    feats["type_token_ratio"]  = type_token_ratio(tf)

    # Phonotactic densities (SYLL.py — ASCII vowel/consonant inventory)
    feats["cv"]           = get_cv(tf)
    feats["cvc"]          = get_cvc(tf)
    feats["ccvc"]         = get_ccvc(tf)
    feats["ccvccc"]       = get_ccvccc(tf)
    feats["cons_cluster"] = get_consonant_cluster(tf)

    # Cross-lingual n-gram overlap (CLGSNGO.py — 3 anchors × 2 orders = 6 features)
    # Uses the UNFOLDED text: CLGSNGO strips non-ASCII itself, and that behaviour
    # is preserved for comparability with the published CROSSNGO feature.
    feats["tag_bi"],  feats["bik_bi"],  feats["ceb_bi"]  = get_bigram_CLGSNGO(t)
    feats["tag_tri"], feats["bik_tri"], feats["ceb_tri"] = get_trigram_CLGSNGO(t)

    return feats

# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("Loading normalized corpus...")
    df = load_corpus(CORPUS_PATH)
    print(f"  {len(df)} documents across {df['language'].nunique()} languages")

    print("Extracting features...")
    feature_rows = []
    for i, (_, row) in enumerate(df.iterrows()):
        feature_rows.append(extract_features(row))
        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{len(df)}")
    print(f"  {len(df)}/{len(df)} done")

    X = pd.DataFrame(feature_rows)

    # Merge label, language, split_role on the stable doc_id key
    X = X.merge(
        df[["doc_id", "label", "language", "split_role"]],
        on="doc_id", how="left",
    )

    # ── Sanity checks ─────────────────────────────────────────────────────────
    feat_cols = [c for c in X.columns if c not in ("doc_id", "label", "language", "split_role")]

    all_null = X[feat_cols].isnull().all(axis=0)
    if all_null.any():
        raise AssertionError(f"Entirely-null feature columns: {list(all_null[all_null].index)}")

    constant = X[feat_cols].nunique() == 1
    if constant.any():
        print(f"  Warning: constant feature columns: {list(constant[constant].index)}")

    ttr_bad = ~X["type_token_ratio"].between(0, 1, inclusive="right")
    if ttr_bad.any():
        raise AssertionError(f"TTR out of (0,1] for doc_ids: {X.loc[ttr_bad,'doc_id'].tolist()}")

    ngram_missing = (X[["tag_bi","bik_bi","ceb_bi","tag_tri","bik_tri","ceb_tri"]] == 0).all(axis=1)
    if ngram_missing.any():
        print(f"  Warning: {ngram_missing.sum()} docs have all-zero n-gram overlap (very short texts?)")

    # ── Save ──────────────────────────────────────────────────────────────────
    os.makedirs("output", exist_ok=True)
    out_path = "output/all_features.csv"
    X.to_csv(out_path, index=False)

    print(f"\nSaved {len(X)} rows x {len(X.columns)} columns to {out_path}")
    print(f"\nFeature columns ({len(feat_cols)}):", feat_cols)
    print("\nPer-language row counts:")
    print(X.groupby(["language", "split_role"])["doc_id"].count().to_string())
    print("\nFeature value ranges:")
    print(X[feat_cols].agg(["min", "max", "mean"]).round(4).to_string())

if __name__ == "__main__":
    main()
