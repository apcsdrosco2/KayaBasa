"""
train_test_split_cv.py — Stratified 5-fold CV splits, matching the ACL'23 paper.

Imperial & Kochmar (ACL Findings 2023), the paper this project reproduces, evaluate
with "a stratified k-fold approach with k==5 to have well-represented samples per
class for a small-dataset scenario" — not a single train/test split. train_test_split.py
implements a single stratified 80/20 split instead; with per-language totals of
150-349 documents split three ways by class, that single split can leave a test cell
with as few as ~5 documents (e.g. Bikol class 2), making individual result cells noisy.

This script instead builds 5 independent stratified folds per language. For each fold
index i, the monolingual train/test ARFFs are that language's fold-i split, and the
bilingual (pairwise) train ARFFs are the union of two languages' fold-i train portions
— exactly generalizing train_test_split.py's single-split design to 5 folds. The
paper's text doesn't specify how folds are built for its cross-lingual conditions;
this per-language-independent-folds-combined-by-index approach is our interpretation,
chosen because it reproduces Table 4's per-test-language column structure.

Reuses train_test_split.py's ARFF I/O and feature-loading helpers rather than
duplicating them — that module guards its own single-split pipeline behind
`if __name__ == "__main__"` specifically so it can be imported here safely.

Output structure:
  pairwise/arff/splits_cv/
    fold{0..4}/
      monolingual/
        {lang}_train_{featureset}.arff/.csv
        {lang}_test_{featureset}.arff/.csv
      bilingual/
        paired_{langA}_{langB}_train_{featureset}.arff/.csv
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import StratifiedKFold

from train_test_split import (
    LANG_CONFIG,
    PAIRS,
    FEATURE_SETS,
    SEED,
    load_language_features,
    save_both,
)

N_FOLDS = 5

SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = SCRIPT_DIR / "splits_cv"


def build_folds():
    """Load each language's features once and compute its 5 stratified folds.

    Returns (lang_features, lang_folds):
      lang_features[lang]        -> dict of feature-set DataFrames (from load_language_features)
      lang_folds[lang][fold_idx] -> (train_idx, test_idx) numpy arrays
    """
    lang_features = {lang: load_language_features(lang) for lang in LANG_CONFIG}
    lang_folds = {}

    for lang in LANG_CONFIG:
        labels = lang_features[lang]["trad"]["class"].astype(str)
        n = len(labels)
        skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
        lang_folds[lang] = list(skf.split(np.zeros(n), labels))

    return lang_features, lang_folds


def main():
    print("=" * 60)
    print(f"Building {N_FOLDS} stratified folds per language")
    print("=" * 60)

    lang_features, lang_folds = build_folds()

    for lang in LANG_CONFIG:
        labels = lang_features[lang]["trad"]["class"].astype(str)
        n = len(labels)
        print(f"\n  {lang.upper()}: {n} total documents")
        for fold_i, (train_idx, test_idx) in enumerate(lang_folds[lang]):
            train_dist = dict(labels.iloc[train_idx].value_counts().sort_index())
            test_dist = dict(labels.iloc[test_idx].value_counts().sort_index())
            print(
                f"    Fold {fold_i}: {len(train_idx)} train {train_dist}, "
                f"{len(test_idx)} test {test_dist}"
            )

    for fold_i in range(N_FOLDS):
        mono_dir = OUTPUT_DIR / f"fold{fold_i}" / "monolingual"
        bi_dir = OUTPUT_DIR / f"fold{fold_i}" / "bilingual"
        mono_dir.mkdir(parents=True, exist_ok=True)
        bi_dir.mkdir(parents=True, exist_ok=True)

        # Monolingual: this fold's per-language train/test split
        for lang in LANG_CONFIG:
            train_idx, test_idx = lang_folds[lang][fold_i]
            for feat_name in FEATURE_SETS:
                df = lang_features[lang][feat_name]
                train_df = df.iloc[train_idx].reset_index(drop=True)
                test_df = df.iloc[test_idx].reset_index(drop=True)

                save_both(train_df, f"{lang}_train_{feat_name}", mono_dir, f"{lang}_train_{feat_name}")
                save_both(test_df, f"{lang}_test_{feat_name}", mono_dir, f"{lang}_test_{feat_name}")

        # Bilingual: stack this fold's train portions for each language pair
        for lang1, lang2 in PAIRS:
            pair_name = f"{lang1}_{lang2}"
            train_idx1, _ = lang_folds[lang1][fold_i]
            train_idx2, _ = lang_folds[lang2][fold_i]

            for feat_name in FEATURE_SETS:
                train1 = lang_features[lang1][feat_name].iloc[train_idx1].reset_index(drop=True)
                train2 = lang_features[lang2][feat_name].iloc[train_idx2].reset_index(drop=True)
                stacked_train = pd.concat([train1, train2], axis=0, ignore_index=True)

                filename = f"paired_{pair_name}_train_{feat_name}"
                save_both(stacked_train, filename, bi_dir, filename)

        print(f"\n  Fold {fold_i} saved to: {OUTPUT_DIR / f'fold{fold_i}'}")

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"""
{N_FOLDS}-fold stratified CV splits (seed={SEED}), matching the paper's methodology.
Feature sets: {", ".join(FEATURE_SETS)}

For fold i:
  Monolingual: splits_cv/fold{{i}}/monolingual/{{lang}}_train_{{featureset}}.arff
               splits_cv/fold{{i}}/monolingual/{{lang}}_test_{{featureset}}.arff
  Bilingual:   splits_cv/fold{{i}}/bilingual/paired_{{langA}}_{{langB}}_train_{{featureset}}.arff
               (test sets are that fold's monolingual test sets)

train_and_evaluate_cv.py runs the full experimental matrix across all {N_FOLDS} folds
and reports mean +/- std per cell.
""")


if __name__ == "__main__":
    main()
