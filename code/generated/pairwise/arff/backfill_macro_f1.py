"""
backfill_macro_f1.py — retroactively adds Macro-Averaged F1-Score to existing CV
result directories, by re-parsing their already-saved results_detailed.txt.

Macro-F1 was not part of the original metric set the CV pipeline captured (only
Weka's "Weighted Avg." F1 was parsed). Since every experiment's full Weka output —
including the per-class "=== Detailed Accuracy By Class ===" breakdown macro-F1 is
computed from — was already saved to results_detailed.txt, no experiments need to be
re-run: this script just re-parses that saved text with cv_common.parse_weka_output's
now-extended per-class F-Measure logic and backfills results_summary_cv.csv and
results_summary_cv_agg.csv with an f1_macro column.

Usage:
  python backfill_macro_f1.py results_cv_rf results_cv_mlp_h256_n10 ...
"""

import re
import sys
import pandas as pd

import cv_common as cc

BLOCK_HEADER_RE = re.compile(
    r"─{70}\nFold: (\d+) \| Train: (\S+) \| Test: (\S+) \| Features: (\S+)\n─{70}\n"
)


def parse_detailed_log(path) -> pd.DataFrame:
    """Split a results_detailed.txt into its per-experiment blocks (in file order)
    and re-run cv_common.parse_weka_output on each block's raw Weka output."""
    text = path.read_text(encoding="utf-8")
    matches = list(BLOCK_HEADER_RE.finditer(text))

    rows = []
    for i, m in enumerate(matches):
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        raw_output = text[start:end]
        metrics = cc.parse_weka_output(raw_output)
        rows.append({
            "fold": int(m.group(1)),
            "train_set": m.group(2),
            "test_set": f"{m.group(3)}_test",
            "feature_set": m.group(4),
            **metrics,
        })
    return pd.DataFrame(rows)


def backfill(results_dir_name: str):
    results_dir = cc.SCRIPT_DIR / results_dir_name
    detailed_path = results_dir / "results_detailed.txt"
    summary_path = results_dir / "results_summary_cv.csv"

    if not detailed_path.exists():
        print(f"  SKIP: {detailed_path} not found (not saved locally, or already cleaned up)")
        return

    parsed_df = parse_detailed_log(detailed_path)
    summary_df = pd.read_csv(summary_path)

    if len(parsed_df) != len(summary_df):
        print(f"  ERROR: {len(parsed_df)} parsed blocks vs {len(summary_df)} summary rows — skipping")
        return

    key_cols = ["fold", "train_set", "test_set", "feature_set"]
    mismatched = (parsed_df[key_cols].reset_index(drop=True) != summary_df[key_cols].reset_index(drop=True)).any(axis=1).sum()
    if mismatched:
        print(f"  ERROR: {mismatched} rows do not line up by (fold,train_set,test_set,feature_set) — skipping")
        return

    acc_diff = (parsed_df["accuracy"] - summary_df["accuracy"]).abs()
    bad_acc = (acc_diff > 0.01).sum()
    print(f"  Alignment check: {len(summary_df)} rows, {bad_acc} accuracy mismatches (>0.01)")

    summary_df["f1_macro"] = parsed_df["f1_macro"].values
    summary_df.to_csv(summary_path, index=False)
    print(f"  Wrote f1_macro into {summary_path}")

    metric_cols = ["accuracy", "f1_weighted", "precision_weighted", "recall_weighted", "kappa", "f1_macro"]
    agg_df = (
        summary_df.groupby(["train_set", "test_set", "feature_set"])[metric_cols]
        .agg(["mean", "std"])
    )
    agg_df.columns = [f"{metric}_{stat}" for metric, stat in agg_df.columns]
    agg_df = agg_df.fillna(0.0).reset_index()
    agg_path = results_dir / "results_summary_cv_agg.csv"
    agg_df.to_csv(agg_path, index=False)
    print(f"  Wrote {agg_path} (now includes f1_macro_mean/f1_macro_std)")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python backfill_macro_f1.py <results_dir_name> [<results_dir_name> ...]")
        sys.exit(1)
    for name in sys.argv[1:]:
        print(f"\n=== {name} ===")
        backfill(name)
