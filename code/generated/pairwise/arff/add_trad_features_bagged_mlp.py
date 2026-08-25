"""
add_trad_features_bagged_mlp.py — adds the missing TRAD / TRAD+CrossNGO feature sets
to the bagged-MLP hybrid (H=128, N=10, 10 bags) CV results.

`run_final_hybrid_model.py` (and this script's underlying `train_and_evaluate_cv.py
mlp_bagged`) originally ran with `--features embedding`, which deliberately restricts
to the 4 neural-embedding feature sets (mbert, all, xlmr, all_xlmr) per cv_common.py's
own documented rationale — but the coordinator wants TRAD and TRAD+CrossNGO included
too for this specific model, for full comparability with Table 4.2's classifier x
feature-source matrix.

Rather than re-running the full 6-feature-set x 7-train-set x 3-test-lang x 5-fold
matrix from scratch (630 Weka calls, most of which would just reproduce the already-
computed mbert/all/xlmr/all_xlmr results), this only computes the 2 missing feature
sets (210 new calls) and merges them with the existing
results_cv_mlp_bagged_h128_n10_b10_emb/results_summary_cv.csv (420 rows, 4 feature
sets) into a new, complete results_cv_mlp_bagged_h128_n10_b10/ folder (no `_emb`
suffix, matching train_and_evaluate_cv.py's own naming convention for the
all-feature-sets case) — the original `_emb` folder is left untouched.

Usage: python add_trad_features_bagged_mlp.py
"""

import re
from datetime import datetime
from pathlib import Path

import pandas as pd

import cv_common as cc
from train_and_evaluate_cv import mlp_options, run_fold

# run_fold()'s own row dict never includes f1_macro (a known bug — see
# reparse_all_metrics.py's docstring: cv_common.parse_weka_output computes it, but
# train_and_evaluate_cv.py's record() never copies it into the row dict). Rather than
# trust run_fold's incomplete return value, this reparses the detailed log we write
# ourselves, the same way reparse_all_metrics.py backfills f1_macro for existing
# results directories — so the new rows and the existing (already-reparsed) rows use
# the identical parsing path, not two different ones that could silently disagree.
BLOCK_HEADER_RE = re.compile(
    r"─{70}\n(?:\[\w+\] )?Fold: (\d+) \| Train: (\S+) \| Test: (\S+) \| Features: (\S+)\n─{70}\n"
)


def reparse_detailed_log(detailed_path: Path) -> pd.DataFrame:
    text = detailed_path.read_text(encoding="utf-8")
    matches = list(BLOCK_HEADER_RE.finditer(text))
    rows = []
    for i, m in enumerate(matches):
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        metrics = cc.parse_weka_output(text[start:end])
        rows.append({
            "fold": int(m.group(1)), "train_set": m.group(2), "test_set": f"{m.group(3)}_test",
            "feature_set": m.group(4), "accuracy": metrics["accuracy"],
            "f1_weighted": metrics["f1_weighted"], "precision_weighted": metrics["precision_weighted"],
            "recall_weighted": metrics["recall_weighted"], "kappa": metrics["kappa"],
            "f1_macro": metrics["f1_macro"],
        })
    return pd.DataFrame(rows)

HIDDEN, EPOCHS, BAGS = 128, 10, 10
NEW_FEATURE_SETS = ["trad", "trad_clgsngo"]

EXISTING_RESULTS_DIR = cc.SCRIPT_DIR / "results_cv_mlp_bagged_h128_n10_b10_emb"
OUT_DIR = cc.SCRIPT_DIR / "results_cv_mlp_bagged_h128_n10_b10"


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    classifier = "weka.classifiers.meta.Bagging"
    options = [
        "-I", str(BAGS), "-S", "1", "-num-slots", "0", "-o",
        "-W", "weka.classifiers.functions.MultilayerPerceptron", "--",
    ] + mlp_options(HIDDEN, EPOCHS, include_o=False)

    detailed_path = OUT_DIR / "results_detailed_trad_features_only.txt"
    detailed_f = open(detailed_path, "w", encoding="utf-8")
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    detailed_f.write(
        f"Weka Bagged MultilayerPerceptron x{BAGS} (hybrid, XLM-R + MLP) — "
        f"TRAD/TRAD+CrossNGO only — {cc.N_FOLDS}-fold CV Results — {timestamp}\n"
        f"(mbert/all/xlmr/all_xlmr detailed logs are in "
        f"{EXISTING_RESULTS_DIR.name}/results_detailed.txt — not recomputed here)\n"
        f"{'=' * 70}\n"
    )

    for fold in range(cc.N_FOLDS):
        print(f"\n=== FOLD {fold} (TRAD, TRAD+CrossNGO only) ===")
        run_fold(fold, classifier, options, NEW_FEATURE_SETS, detailed_f)
        print(f"  Fold {fold} -> done")
    detailed_f.close()

    new_df = reparse_detailed_log(detailed_path)
    expected_new = cc.N_FOLDS * len(NEW_FEATURE_SETS) * 7 * 3  # folds x feats x train_sets x test_langs
    assert len(new_df) == expected_new, f"Expected {expected_new} new rows, got {len(new_df)}"

    existing_path = EXISTING_RESULTS_DIR / "results_summary_cv.csv"
    if not existing_path.exists():
        raise FileNotFoundError(f"Missing existing results: {existing_path}")
    existing_df = pd.read_csv(existing_path)
    raw_cols = ["fold", "train_set", "test_set", "feature_set", "accuracy",
                "f1_weighted", "precision_weighted", "recall_weighted", "kappa", "f1_macro"]
    existing_raw = existing_df[raw_cols]

    combined = pd.concat([existing_raw, new_df], axis=0, ignore_index=True)
    assert len(combined) == len(existing_raw) + len(new_df)
    assert combined["f1_macro"].notna().all(), "Some rows are missing f1_macro after merge"

    combined["Model"] = combined["train_set"].map(cc.TRAIN_MAP)
    combined["Test_Lang"] = combined["test_set"].map(cc.TEST_MAP)
    combined["Features"] = combined["feature_set"].map(cc.FEAT_MAP)
    combined["col"] = combined["Test_Lang"] + " | " + combined["Features"]

    combined.to_csv(OUT_DIR / "results_summary_cv.csv", index=False)

    metric_cols = ["accuracy", "f1_weighted", "precision_weighted", "recall_weighted", "kappa", "f1_macro"]
    agg_df = (
        combined.groupby(["train_set", "test_set", "feature_set"])[metric_cols]
        .agg(["mean", "std"])
    )
    agg_df.columns = [f"{metric}_{stat}" for metric, stat in agg_df.columns]
    agg_df = agg_df.fillna(0.0).reset_index()
    agg_df.to_csv(OUT_DIR / "results_summary_cv_agg.csv", index=False)

    table_path = OUT_DIR / "results_table_cv.txt"
    title = f"Weka Bagged MultilayerPerceptron x{BAGS} (hybrid, XLM-R + MLP) Accuracy (%) - {cc.N_FOLDS}-fold CV Train/Test Matrix (all 6 feature sets)"
    header_lines = [
        f"Weka 3.8.7 | Bagging(numIterations={BAGS}, numSlots=auto) of MultilayerPerceptron:",
        f"             learningRate=0.3, momentum=0.2, trainingTime={EPOCHS}, "
        f"hiddenLayers={HIDDEN}, seed=1",
    ]
    mean_pivot, std_pivot = cc.write_results_table_cv(combined, table_path, title, header_lines)
    mean_pivot.to_csv(OUT_DIR / "results_table_cv_mean.csv")
    std_pivot.to_csv(OUT_DIR / "results_table_cv_std.csv")

    print(f"\n=== Done: {len(new_df)} new rows (TRAD/TRAD+CrossNGO) + "
          f"{len(existing_raw)} existing rows (mbert/all/xlmr/all_xlmr) = {len(combined)} total ===")
    print(f"Written to: {OUT_DIR}")
    print(open(table_path, "r", encoding="utf-8").read())


if __name__ == "__main__":
    main()
