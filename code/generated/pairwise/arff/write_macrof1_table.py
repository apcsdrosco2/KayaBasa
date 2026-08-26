"""
write_macrof1_table.py — Macro-F1 counterpart to a results_cv_*/results_table_cv.txt
(which only ever reports Accuracy). Writes a NEW file, results_table_cv_macrof1.txt,
alongside the existing accuracy table in the same folder — does not touch or
overwrite results_table_cv.txt. Matches the format already used by
code_noisy/results_noisy/results_noisy_mlp/results_table_cv_macrof1.txt.

Usage: python write_macrof1_table.py <results_dir_name> "<title>" "<header_line>" ["<header_line2>" ...]
Example:
  python write_macrof1_table.py results_cv_mlp_bagged_h128_n10_b10 \
      "Weka Bagged MultilayerPerceptron x10 (hybrid, XLM-R + MLP) Macro-F1 (%) - 5-fold CV Train/Test Matrix (all 6 feature sets)" \
      "Weka 3.8.7 | Bagging(numIterations=10, numSlots=auto) of MultilayerPerceptron:" \
      "             learningRate=0.3, momentum=0.2, trainingTime=10, hiddenLayers=128, seed=1"
"""

import sys
from pathlib import Path

import pandas as pd

import cv_common as cc


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)

    results_dir_name = sys.argv[1]
    title = sys.argv[2]
    header_lines = sys.argv[3:]

    results_dir = cc.SCRIPT_DIR / results_dir_name
    summary_path = results_dir / "results_summary_cv.csv"
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing {summary_path}")

    df = pd.read_csv(summary_path)
    if "f1_macro" not in df.columns:
        raise ValueError(f"{summary_path} has no f1_macro column — reparse it first (reparse_all_metrics.py)")

    # train_and_evaluate_cv.py adds these derived columns itself before writing
    # results_summary_cv.csv; results_cv_mlp_bagged_h128_n10_b10's copy has them
    # already (merged in by an earlier add-TRAD-features pass), but a plain,
    # never-modified results_cv_* folder (e.g. results_cv_rf) won't — derive them
    # the same way train_and_evaluate_cv.py does if missing.
    if "Model" not in df.columns:
        df["Model"] = df["train_set"].map(cc.TRAIN_MAP)
        df["Test_Lang"] = df["test_set"].map(cc.TEST_MAP)
        df["Features"] = df["feature_set"].map(cc.FEAT_MAP)
        df["col"] = df["Test_Lang"] + " | " + df["Features"]

    # Only render columns for feature sets actually present in this results dir (e.g.
    # the "_emb" folders only have 4 of the 6 feature sets) — same guard
    # write_results_table_cv's own feat_labels param is meant for.
    present_labels = set(df["Features"].unique())
    feat_labels = [cc.FEAT_MAP[f] for f in cc.FEATURE_SETS if cc.FEAT_MAP[f] in present_labels]

    out_path = results_dir / "results_table_cv_macrof1.txt"
    mean_pivot, std_pivot = cc.write_results_table_cv(
        df, out_path, title, header_lines, feat_labels=feat_labels, metric="f1_macro",
        scale=1.0, decimals=3, show_std=False,  # bare mean (0.547-style, no ±std),
        # matching Chapter4_Draft.md's convention: Macro F1 is always shown as a bare
        # mean there, never with a std dev (unlike Accuracy, which always has one).
    )
    mean_pivot.to_csv(results_dir / "results_table_cv_macrof1_mean.csv")
    std_pivot.to_csv(results_dir / "results_table_cv_macrof1_std.csv")
    print(f"Written: {out_path}")
    print(f"         {results_dir / 'results_table_cv_macrof1_mean.csv'}")
    print(f"         {results_dir / 'results_table_cv_macrof1_std.csv'}")
    print(open(out_path, "r", encoding="utf-8").read())


if __name__ == "__main__":
    main()
