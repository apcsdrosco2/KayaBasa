"""
train_and_evaluate_cv.py — 5-fold CV version of train_and_evaluate.py.

Usage:
  python train_and_evaluate_cv.py rf     # RandomForest baseline (matches the paper)
  python train_and_evaluate_cv.py mlp    # XLM-R+MLP hybrid

Runs the same monolingual / bilingual (pairwise) / all-languages x feature-set x
test-language matrix as train_and_evaluate.py, but repeated once per fold against
train_test_split_cv.py's splits_cv/fold{0..4}/ directories, then aggregates each
cell's accuracy across the 5 folds as mean +/- std instead of reporting a single
number from one split. This is what lets us tell whether a "pattern" in the
single-split results is real signal or just noise from a small held-out set.

Outputs (results_cv_rf/ or results_cv_mlp_h256_n10/, depending on the classifier arg):
  results_summary_cv.csv       — one row per (fold, train_set, test_set, feature_set)
  results_summary_cv_agg.csv   — grouped mean/std of every metric across folds
  results_table_cv.txt         — formatted mean+/-std accuracy matrix
  results_table_cv_mean.csv    — mean-accuracy matrix (same shape as results_table.txt)
  results_table_cv_std.csv     — std-accuracy matrix
  results_detailed.txt         — full Weka output per fold/experiment (streamed)
"""

import sys
import pandas as pd
from datetime import datetime

import cv_common as cc

CLASSIFIER_CONFIGS = {
    "rf": {
        "classifier": "weka.classifiers.trees.RandomForest",
        # Matches the paper's Table 6 settings exactly (batchSize/bagSizePercent are
        # already Weka's defaults, so left implicit here as they were in
        # train_and_evaluate.py's original RandomForest run).
        "options": ["-I", "100", "-K", "0", "-depth", "0", "-S", "1"],
        "label": "RandomForest (baseline, matches paper)",
        "results_dir_name": "results_cv_rf",
        "header_lines": [
            "Weka 3.8.7 | numIterations=100, maxDepth=unlimited, bagSizePercent=100,",
            "             numFeatures=int(log(#predictors)+1), seed=1",
        ],
    },
    "mlp": {
        "classifier": "weka.classifiers.functions.MultilayerPerceptron",
        "options": [
            "-L", "0.3", "-M", "0.2", "-N", "10", "-V", "0", "-E", "20",
            "-H", "256", "-S", "1", "-o",
        ],
        "label": "MultilayerPerceptron (hybrid, XLM-R + MLP)",
        "results_dir_name": "results_cv_mlp_h256_n10",
        "header_lines": [
            "Weka 3.8.7 | learningRate=0.3, momentum=0.2, trainingTime=10, validationSetSize=0,",
            "             hiddenLayers=256, seed=1",
        ],
    },
}


def run_fold(fold: int, classifier: str, options: list, detailed_f) -> list:
    """Run the monolingual/bilingual/all-languages matrix for one fold. Returns result rows."""
    fold_dir = cc.SPLITS_CV_DIR / f"fold{fold}"
    mono_dir = fold_dir / "monolingual"
    bi_dir = fold_dir / "bilingual"
    all_train_dir = fold_dir / "all_languages"

    cc.build_all_languages_arff(mono_dir, all_train_dir)

    rows = []

    def record(train_set, test_lang, feat):
        test_arff = mono_dir / f"{test_lang}_test_{feat}.arff"
        metrics = cc.run_weka(train_arff, test_arff, classifier, options)
        rows.append({
            "fold": fold,
            "train_set": train_set,
            "test_set": f"{test_lang}_test",
            "feature_set": feat,
            "accuracy": metrics["accuracy"],
            "f1_weighted": metrics["f1_weighted"],
            "precision_weighted": metrics["precision_weighted"],
            "recall_weighted": metrics["recall_weighted"],
            "kappa": metrics["kappa"],
        })
        detailed_f.write(
            f"\n{'─' * 70}\n"
            f"Fold: {fold} | Train: {train_set} | Test: {test_lang} | Features: {feat}\n"
            f"{'─' * 70}\n"
            f"{metrics['raw_output']}\n"
        )

    # 1. Monolingual: train on each language, test on each language
    for train_lang in cc.LANGUAGES:
        for feat in cc.FEATURE_SETS:
            train_arff = mono_dir / f"{train_lang}_train_{feat}.arff"
            for test_lang in cc.LANGUAGES:
                record(f"mono_{train_lang}", test_lang, feat)

    # 2. Bilingual: train on each pair, test on each language
    for lang1, lang2 in cc.PAIRS:
        for feat in cc.FEATURE_SETS:
            train_arff, pair_name = cc.find_bilingual_train_arff(bi_dir, lang1, lang2, feat)
            for test_lang in cc.LANGUAGES:
                record(f"bi_{pair_name}", test_lang, feat)

    # 3. All languages: train on all 3, test on each language
    for feat in cc.FEATURE_SETS:
        train_arff = all_train_dir / f"all_languages_train_{feat}.arff"
        for test_lang in cc.LANGUAGES:
            record("all_languages", test_lang, feat)

    return rows


def main():
    if len(sys.argv) != 2 or sys.argv[1] not in CLASSIFIER_CONFIGS:
        print("Usage: python train_and_evaluate_cv.py {rf|mlp}")
        sys.exit(1)

    config = CLASSIFIER_CONFIGS[sys.argv[1]]
    classifier = config["classifier"]
    options = config["options"]

    results_dir = cc.SCRIPT_DIR / config["results_dir_name"]
    results_dir.mkdir(parents=True, exist_ok=True)

    detailed_f = open(results_dir / "results_detailed.txt", "w", encoding="utf-8")
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    detailed_f.write(f"Weka {config['label']} — {cc.N_FOLDS}-fold CV Results — {timestamp}\n{'=' * 70}\n")

    all_results = []
    for fold in range(cc.N_FOLDS):
        print("\n" + "=" * 70)
        print(f"FOLD {fold} ({config['label']})")
        print("=" * 70)
        all_results.extend(run_fold(fold, classifier, options, detailed_f))
        print(f"  Fold {fold} -> done ({len(all_results)} total rows so far)")

    detailed_f.close()

    results_df = pd.DataFrame(all_results)
    results_df["Model"] = results_df["train_set"].map(cc.TRAIN_MAP)
    results_df["Test_Lang"] = results_df["test_set"].map(cc.TEST_MAP)
    results_df["Features"] = results_df["feature_set"].map(cc.FEAT_MAP)
    results_df["col"] = results_df["Test_Lang"] + " | " + results_df["Features"]

    results_df.to_csv(results_dir / "results_summary_cv.csv", index=False)

    # Aggregate every metric (not just accuracy) across folds, per cell
    metric_cols = ["accuracy", "f1_weighted", "precision_weighted", "recall_weighted", "kappa"]
    agg_df = (
        results_df.groupby(["train_set", "test_set", "feature_set"])[metric_cols]
        .agg(["mean", "std"])
    )
    agg_df.columns = [f"{metric}_{stat}" for metric, stat in agg_df.columns]
    agg_df = agg_df.fillna(0.0).reset_index()
    agg_df.to_csv(results_dir / "results_summary_cv_agg.csv", index=False)

    # Formatted mean+/-std accuracy table
    table_path = results_dir / "results_table_cv.txt"
    title = f"Weka {config['label']} Accuracy (%) - {cc.N_FOLDS}-fold CV Train/Test Matrix"
    mean_pivot, std_pivot = cc.write_results_table_cv(results_df, table_path, title, config["header_lines"])
    mean_pivot.to_csv(results_dir / "results_table_cv_mean.csv")
    std_pivot.to_csv(results_dir / "results_table_cv_std.csv")

    print("\n" + "=" * 70)
    print("RESULTS TABLE (mean +/- std across folds)")
    print("=" * 70)
    print(open(table_path, "r", encoding="utf-8").read())
    print(f"\nTotal experiments: {len(all_results)} ({cc.N_FOLDS} folds x {len(all_results) // cc.N_FOLDS} cells/fold)")


if __name__ == "__main__":
    main()
