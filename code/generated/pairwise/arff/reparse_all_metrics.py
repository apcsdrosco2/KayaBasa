"""
reparse_all_metrics.py — rebuilds results_summary_cv.csv entirely from
results_detailed.txt, for directories where the row-recording code never
captured f1_macro (a bug: cv_common.parse_weka_output computes it, but
train_and_evaluate_cv.py's record() never copied it into the row dict).

Also correctly handles retry_failed_cells.py's appended "[RETRY] Fold: ..."
blocks: this parses BOTH the "Fold: ..." and "[RETRY] Fold: ..." header
forms, in file order, keyed by (fold, train_set, test_set, feature_set) —
since retries are appended after the originals, a later block for the same
key naturally overwrites the earlier (crashed) one, with no special-casing
needed.
"""

import re
import sys
import pandas as pd

import cv_common as cc

BLOCK_HEADER_RE = re.compile(
    r"(?:\[RETRY\] )?─{70}\nFold: (\d+) \| Train: (\S+) \| Test: (\S+) \| Features: (\S+)\n─{70}\n"
)
# The [RETRY] tag sits INSIDE the dashes in retry_failed_cells.py's actual output
# (dashes, then "[RETRY] Fold: ...", not "[RETRY] " then dashes) -- handle both
# by also trying the tag-after-dashes form.
BLOCK_HEADER_RE = re.compile(
    r"─{70}\n(?:\[\w+\] )?Fold: (\d+) \| Train: (\S+) \| Test: (\S+) \| Features: (\S+)\n─{70}\n"
)


def reparse(results_dir_name: str):
    results_dir = cc.SCRIPT_DIR / results_dir_name
    detailed_path = results_dir / "results_detailed.txt"
    summary_path = results_dir / "results_summary_cv.csv"

    text = detailed_path.read_text(encoding="utf-8")
    matches = list(BLOCK_HEADER_RE.finditer(text))

    by_key = {}
    order = []
    for i, m in enumerate(matches):
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        raw_output = text[start:end]
        key = (int(m.group(1)), m.group(2), f"{m.group(3)}_test", m.group(4))
        metrics = cc.parse_weka_output(raw_output)
        if key not in by_key:
            order.append(key)
        by_key[key] = metrics  # later occurrence (a retry) overwrites the earlier one

    rows = []
    for key in order:
        fold, train_set, test_set, feat = key
        metrics = by_key[key]
        rows.append({
            "fold": fold, "train_set": train_set, "test_set": test_set, "feature_set": feat,
            "accuracy": metrics["accuracy"], "f1_weighted": metrics["f1_weighted"],
            "precision_weighted": metrics["precision_weighted"], "recall_weighted": metrics["recall_weighted"],
            "kappa": metrics["kappa"], "f1_macro": metrics["f1_macro"],
        })
    new_df = pd.DataFrame(rows)

    old_df = pd.read_csv(summary_path)
    print(f"{results_dir_name}: {len(old_df)} old rows, {len(new_df)} reparsed rows")
    still_bad = ((new_df.accuracy == 0.0) & (new_df.kappa == 0.0) & (new_df.f1_weighted == 0.0)).sum()
    print(f"  Zero-accuracy rows after full reparse: {still_bad}")

    new_df.to_csv(summary_path, index=False)

    metric_cols = ["accuracy", "f1_weighted", "precision_weighted", "recall_weighted", "kappa", "f1_macro"]
    agg_df = new_df.groupby(["train_set", "test_set", "feature_set"])[metric_cols].agg(["mean", "std"])
    agg_df.columns = [f"{metric}_{stat}" for metric, stat in agg_df.columns]
    agg_df = agg_df.fillna(0.0).reset_index()
    agg_df.to_csv(results_dir / "results_summary_cv_agg.csv", index=False)
    print(f"  Rewrote {summary_path.name} and results_summary_cv_agg.csv")


if __name__ == "__main__":
    for name in sys.argv[1:]:
        reparse(name)
