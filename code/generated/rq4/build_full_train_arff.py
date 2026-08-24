"""
build_full_train_arff.py — Stage E of the RQ4 pipeline.

RO4 requires training the hybrid model on **all** Tagalog+Cebuano+Bikolano documents,
with no held-out split (the low-resource languages are the test set instead). The
existing `splits/all_languages/all_languages_train_all_xlmr.arff` is NOT this — it's
the 80%-train portion of `train_test_split.py`'s stratified 80/20 split (611 rows,
~80% of 764), built for the RO1-3 monolingual/bilingual comparisons.

This script instead reuses `train_test_split.load_language_features()` directly, which
already loads each language's FULL, unsplit `all_xlmr` DataFrame (24 trad_clgsngo
features + 768 XLM-R dims + class) — no slicing, no train/test split logic needed.
Concatenates Tagalog + Cebuano + Bikolano's full DataFrames and writes the result as
one ARFF, matching the exact 793-attribute schema (792 features + class) via the same
`save_arff()` helper already used throughout this pipeline.

Usage: python build_full_train_arff.py
"""

import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PAIRWISE_ARFF_DIR = SCRIPT_DIR.parent / "pairwise" / "arff"
sys.path.insert(0, str(PAIRWISE_ARFF_DIR))

import pandas as pd
import train_test_split as tts  # reuses load_language_features, save_arff, LANG_CONFIG

OUT_DIR = SCRIPT_DIR / "generated"
OUT_DIR.mkdir(parents=True, exist_ok=True)

LANGUAGES = ["tagalog", "cebuano", "bikol"]


def main():
    frames = []
    for lang in LANGUAGES:
        features = tts.load_language_features(lang)
        df = features["all_xlmr"]
        print(f"{lang:<10} {len(df)} rows, {len(df.columns)} columns")
        frames.append(df)

    full = pd.concat(frames, axis=0, ignore_index=True)
    print(f"\nConcatenated: {len(full)} rows, {len(full.columns)} columns")

    out_path = OUT_DIR / "all_languages_full_train_all_xlmr.arff"
    tts.save_arff(full, "readability_all_languages_full", out_path)
    full.to_csv(OUT_DIR / "all_languages_full_train_all_xlmr.csv", index=False)
    print(f"-> {out_path}")

    # Sanity check against Chapter 3's reported per-language totals (no held-out
    # split, so this must be the FULL count, not ~80% of it).
    expected_total = 764  # 265 Tagalog + 349 Cebuano + 150 Bikol, per Chapter 3's table
    assert len(full) == expected_total, (
        f"Expected {expected_total} rows (full Tagalog+Cebuano+Bikol, no split), "
        f"got {len(full)} — check load_language_features() isn't returning a split subset."
    )
    assert len(full.columns) == 793, f"Expected 793 columns (792 features + class), got {len(full.columns)}"
    print(f"\nOK — {len(full)} rows verified (full, unsplit, matches Chapter 3), 793 columns.")


if __name__ == "__main__":
    main()
