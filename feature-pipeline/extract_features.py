"""
extract_features.py — KAYABASA §3.2.2  Feature Extraction Driver
=================================================================
Reads the cleaned, normalized corpus produced by the data-cleaning pipeline
and writes the 14-feature matrix used by the hybrid model.

Feature groups (14 total — see thesis Table VII):
  TRAD    (4)  mean_sentence_len, mean_word_len, polysyll_freq, type_token_ratio
  SYLL    (4)  syll_cv, syll_cvc, syll_ccvc, syll_ccvccc
  CLGSNGO (6)  tag_bi, bik_bi, ceb_bi, tag_tri, bik_tri, ceb_tri

  Input:   ../data-cleaning-repo-main/output_normalized/all_languages.txt
  Output:  output/all_features.csv

Run from this folder:
    cd feature-pipeline
    python extract_features.py

To import the pipeline in model training code:
    from feature_pipeline import FeaturePipeline, FEATURE_COLS, load_corpus
"""

import os
import sys

# ── Resolve paths relative to this script's location ──────────────────────

_HERE       = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT  = os.path.join(_HERE, "..", "data-cleaning-repo-main")

CORPUS_PATH = os.path.join(_REPO_ROOT, "output_normalized", "all_languages.txt")
OUT_PATH    = os.path.join(_HERE, "output", "all_features.csv")

# ── Imports ────────────────────────────────────────────────────────────────

from feature_pipeline import FeaturePipeline, FEATURE_COLS, load_corpus

# ── Main ──────────────────────────────────────────────────────────────────

def main():
    if not os.path.exists(CORPUS_PATH):
        print(f"ERROR: Corpus not found at:\n  {CORPUS_PATH}")
        print("Make sure the data-cleaning pipeline has been run first.")
        print("  cd ../data-cleaning-repo-main && python normalize_datasets.py")
        sys.exit(1)

    print(f"Loading corpus from:\n  {CORPUS_PATH}")
    df = load_corpus(CORPUS_PATH)
    print(f"  {len(df)} documents · {df['language'].nunique()} languages")
    print(f"  Languages: {sorted(df['language'].unique())}")

    print("\nInitialising FeaturePipeline...")
    pipe = FeaturePipeline()          # ngram_dir auto-resolved to ./ngrams list/

    print(f"\nExtracting {len(FEATURE_COLS)} features per document...")
    X = pipe.fit_transform(df)

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    X.to_csv(OUT_PATH, index=False)
    print(f"\n✓ Saved {len(X)} rows × {len(X.columns)} columns")
    print(f"  → {OUT_PATH}")

    feat_cols = [c for c in X.columns
                 if c not in ("doc_id", "label", "language", "split_role")]

    print(f"\nFeature columns ({len(feat_cols)}):")
    for i, c in enumerate(feat_cols, 1):
        print(f"  {i:>2}. {c}")

    print("\nPer-language document counts:")
    print(X.groupby(["language", "split_role"])["doc_id"].count().to_string())

    print("\nFeature value ranges (mean / min / max):")
    print(X[feat_cols].agg(["mean", "min", "max"]).round(4).to_string())


if __name__ == "__main__":
    main()
