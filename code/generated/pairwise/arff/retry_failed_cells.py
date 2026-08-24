"""
retry_failed_cells.py — re-runs exactly the cells that failed with a JVM
out-of-memory crash (accuracy == kappa == f1_weighted == 0.0, the parser's
default when Weka's expected output sections are missing), patches them into
the existing results_summary_cv.csv, and regenerates results_summary_cv_agg.csv
+ the formatted table for that results directory.

Usage:
  python retry_failed_cells.py <results_dir_name> <classifier> <options...> -- <feature_sets_scope>

Simplify by hardcoding the two known-affected runs below instead of a generic CLI.
"""

import pandas as pd

import cv_common as cc
from train_and_evaluate_cv import mlp_options


def retry(results_dir_name: str, classifier: str, options: list, header_lines: list, title: str, feat_labels):
    results_dir = cc.SCRIPT_DIR / results_dir_name
    summary_path = results_dir / "results_summary_cv.csv"
    detailed_path = results_dir / "results_detailed.txt"

    df = pd.read_csv(summary_path)
    failed = df[(df.accuracy == 0.0) & (df.kappa == 0.0) & (df.f1_weighted == 0.0)]
    print(f"{results_dir_name}: {len(failed)} failed cells to retry")

    detailed_f = open(detailed_path, "a", encoding="utf-8")
    detailed_f.write(f"\n\n{'=' * 70}\nRETRY PASS (lower -Xmx after JVM OOM crashes)\n{'=' * 70}\n")

    for idx, row in failed.iterrows():
        fold = int(row.fold)
        train_set = row.train_set
        test_lang = row.test_set.replace("_test", "")
        feat = row.feature_set

        fold_dir = cc.SPLITS_CV_DIR / f"fold{fold}"
        mono_dir = fold_dir / "monolingual"
        bi_dir = fold_dir / "bilingual"
        all_train_dir = fold_dir / "all_languages"

        test_arff = mono_dir / f"{test_lang}_test_{feat}.arff"
        if train_set.startswith("mono_"):
            train_lang = train_set[len("mono_"):]
            train_arff = mono_dir / f"{train_lang}_train_{feat}.arff"
        elif train_set == "all_languages":
            train_arff = all_train_dir / f"all_languages_train_{feat}.arff"
        elif train_set.startswith("bi_"):
            rest = train_set[len("bi_"):]
            lang1, lang2 = rest.split("_")
            train_arff, _ = cc.find_bilingual_train_arff(bi_dir, lang1, lang2, feat)
        else:
            raise ValueError(f"Unrecognized train_set: {train_set}")

        metrics = cc.run_weka(train_arff, test_arff, classifier, options, xmx="1536m")
        print(f"  fold={fold} train={train_set} test={test_lang} feat={feat} -> "
              f"acc={metrics['accuracy']} f1m={metrics['f1_macro']:.3f}")

        for col, key in [
            ("accuracy", "accuracy"), ("f1_weighted", "f1_weighted"),
            ("precision_weighted", "precision_weighted"), ("recall_weighted", "recall_weighted"),
            ("kappa", "kappa"), ("f1_macro", "f1_macro"),
        ]:
            df.loc[idx, col] = metrics[key]

        detailed_f.write(
            f"\n{'─' * 70}\n"
            f"[RETRY] Fold: {fold} | Train: {train_set} | Test: {test_lang} | Features: {feat}\n"
            f"{'─' * 70}\n"
            f"{metrics['raw_output']}\n"
        )

    detailed_f.close()

    still_bad = ((df.accuracy == 0.0) & (df.kappa == 0.0) & (df.f1_weighted == 0.0)).sum()
    print(f"  Remaining zero-accuracy rows after retry: {still_bad}")

    df.to_csv(summary_path, index=False)

    metric_cols = ["accuracy", "f1_weighted", "precision_weighted", "recall_weighted", "kappa", "f1_macro"]
    agg_df = (
        df.groupby(["train_set", "test_set", "feature_set"])[metric_cols]
        .agg(["mean", "std"])
    )
    agg_df.columns = [f"{metric}_{stat}" for metric, stat in agg_df.columns]
    agg_df = agg_df.fillna(0.0).reset_index()
    agg_df.to_csv(results_dir / "results_summary_cv_agg.csv", index=False)

    df["Model"] = df["train_set"].map(cc.TRAIN_MAP)
    df["Test_Lang"] = df["test_set"].map(cc.TEST_MAP)
    df["Features"] = df["feature_set"].map(cc.FEAT_MAP)
    df["col"] = df["Test_Lang"] + " | " + df["Features"]

    mean_pivot, std_pivot = cc.write_results_table_cv(
        df, results_dir / "results_table_cv.txt", title, header_lines, feat_labels=feat_labels
    )
    mean_pivot.to_csv(results_dir / "results_table_cv_mean.csv")
    std_pivot.to_csv(results_dir / "results_table_cv_std.csv")
    print(f"  Regenerated agg CSV and table for {results_dir_name}")


if __name__ == "__main__":
    all_feat_labels = ["TRAD", "TRAD+CrossNGO", "mBERT Embdng", "ALL (mBERT)", "XLM-R Embdng", "ALL (XLM-R)"]
    emb_feat_labels = ["mBERT Embdng", "ALL (mBERT)", "XLM-R Embdng", "ALL (XLM-R)"]

    retry(
        "results_cv_mlp_h128_n10",
        "weka.classifiers.functions.MultilayerPerceptron",
        mlp_options(128, 10),
        [
            "Weka 3.8.7 | learningRate=0.3, momentum=0.2, trainingTime=10, validationSetSize=0,",
            "             hiddenLayers=128, seed=1",
        ],
        "Weka MultilayerPerceptron (hybrid, XLM-R + MLP) Accuracy (%) - 5-fold CV Train/Test Matrix",
        all_feat_labels,
    )

    print()

    retry(
        "results_cv_mlp_bagged_h128_n10_b10_emb",
        "weka.classifiers.meta.Bagging",
        [
            "-I", "10", "-S", "1", "-num-slots", "0", "-o",
            "-W", "weka.classifiers.functions.MultilayerPerceptron", "--",
        ] + mlp_options(128, 10, include_o=False),
        [
            "Weka 3.8.7 | Bagging(numIterations=10, numSlots=auto) of MultilayerPerceptron:",
            "             learningRate=0.3, momentum=0.2, trainingTime=10, hiddenLayers=128, seed=1",
        ],
        "Weka Bagged MultilayerPerceptron x10 (hybrid, XLM-R + MLP) Accuracy (%) - 5-fold CV Train/Test Matrix",
        emb_feat_labels,
    )
