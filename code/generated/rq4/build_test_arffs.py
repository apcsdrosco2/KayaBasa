"""
build_test_arffs.py — Stage D of the RQ4 pipeline.

Merges each low-resource language's Stage B features (24-dim trad_clgsngo,
code/generated/rq4/generated/{lang}_trad_clgsngo.csv) with its Stage C embeddings
(768-dim XLM-R, code/generated/rq4/embeddings/{lang}_xlmr_features.csv, computed on
Colab per COLAB_RQ4_EMBEDDINGS.md) by row order — both were derived from the same
`clean/rq4/{lang}_all_clean.txt` in the same deterministic row order (stage_raw_text.py
iterates raw files via a sorted glob; the embeddings CSV has no doc_id column, so row
order is the only alignment key, verified explicitly below by row count before
trusting positional alignment). Writes one {lang}_test_all_xlmr.arff per language,
matching the exact 793-attribute schema (24 trad_clgsngo + xlmr_000..xlmr_767 + class)
of the existing `{lang}_trad_clgsngo_xlmr.arff` files used for Tagalog/Cebuano/Bikolano.

Usage: python build_test_arffs.py
"""

import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PAIRWISE_ARFF_DIR = SCRIPT_DIR.parent / "pairwise" / "arff"
sys.path.insert(0, str(PAIRWISE_ARFF_DIR))

import pandas as pd
import train_test_split as tts  # reuses save_arff for consistent ARFF formatting

FEATURES_DIR = SCRIPT_DIR / "generated"
EMBEDDINGS_DIR = SCRIPT_DIR / "embeddings"
OUT_DIR = SCRIPT_DIR / "generated"

LANGUAGES = ["hiligaynon", "minasbate", "karay-a", "rinconada"]

EXPECTED_COUNTS = {"hiligaynon": 133, "minasbate": 268, "karay-a": 173, "rinconada": 195}


def main():
    grand_total = 0
    for lang in LANGUAGES:
        feat_path = FEATURES_DIR / f"{lang}_trad_clgsngo.csv"
        emb_path = EMBEDDINGS_DIR / f"{lang}_xlmr_features.csv"
        if not feat_path.exists():
            raise FileNotFoundError(f"Missing Stage B output: {feat_path}")
        if not emb_path.exists():
            raise FileNotFoundError(f"Missing Stage C output: {emb_path}")

        feat_df = pd.read_csv(feat_path)
        emb_df = pd.read_csv(emb_path, header=None)
        emb_df.columns = [f"xlmr_{i:03d}" for i in range(emb_df.shape[1])]

        # Row-count check BEFORE trusting positional alignment (no shared doc_id key
        # exists between these two files, so count parity is the only cheap guard —
        # both must equal the known-good staged total for this language).
        assert len(feat_df) == len(emb_df), (
            f"{lang}: row count mismatch — features={len(feat_df)}, embeddings={len(emb_df)}"
        )
        assert len(feat_df) == EXPECTED_COUNTS[lang], (
            f"{lang}: expected {EXPECTED_COUNTS[lang]} rows, got {len(feat_df)}"
        )
        assert emb_df.shape[1] == 768, f"{lang}: expected 768 embedding columns, got {emb_df.shape[1]}"

        trad_clgsngo_cols = [c for c in feat_df.columns if c not in ("doc_id", "flag", "class")]
        merged = pd.concat(
            [feat_df[trad_clgsngo_cols].reset_index(drop=True),
             emb_df.reset_index(drop=True),
             feat_df[["class"]].reset_index(drop=True)],
            axis=1,
        )
        assert len(merged.columns) == 793, f"{lang}: expected 793 columns, got {len(merged.columns)}"

        out_path = OUT_DIR / f"{lang}_test_all_xlmr.arff"
        tts.save_arff(merged, f"readability_{lang}_test", out_path)
        merged.to_csv(OUT_DIR / f"{lang}_test_all_xlmr.csv", index=False)
        print(f"{lang:<12} {len(merged)} rows, {len(merged.columns)} columns -> {out_path}")
        grand_total += len(merged)

    print(f"\nGrand total: {grand_total} (expected 769)")
    assert grand_total == 769, f"Expected 769, got {grand_total}"
    print("OK — 4 zero-shot test ARFFs built, 769 rows total, 793 columns each.")


if __name__ == "__main__":
    main()
