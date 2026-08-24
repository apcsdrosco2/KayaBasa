"""
tune_mlp_cv.py — cheap hyperparameter sweep for the MLP hybrid.

The MLP hyperparameters used so far (-H 256 -N 10) were picked purely to keep 5-fold
CV runtime tractable, never tuned for accuracy. This script sweeps a small grid over
hidden-layer size and epoch count, restricted to the monolingual diagonal (train lang
== test lang) on the embedding feature sets only, across all 5 CV folds — 240 Weka
calls instead of 630+, so results come back in a reasonable time and inform the much
larger full-matrix and bagged runs that follow.

Grid: H in {128, 256}, N in {10, 50}. L/M/V/E held at their existing values.

Output: tuning_results_mlp.csv (one row per H/N/lang/feature_set/fold) plus a printed
summary table of mean accuracy per (H, N) combo.
"""

import itertools
import pandas as pd

import cv_common as cc

OUT_CSV = cc.SCRIPT_DIR / "tuning_results_mlp.csv"

HIDDEN_GRID = [128, 256]
EPOCHS_GRID = [10, 50]

CLASSIFIER = "weka.classifiers.functions.MultilayerPerceptron"


def mlp_options(hidden: int, epochs: int) -> list:
    return [
        "-L", "0.3", "-M", "0.2", "-N", str(epochs), "-V", "0", "-E", "20",
        "-H", str(hidden), "-S", "1", "-o",
    ]


def main():
    combos = list(itertools.product(HIDDEN_GRID, EPOCHS_GRID))
    total_calls = len(combos) * len(cc.LANGUAGES) * len(cc.EMBEDDING_FEATURE_SETS) * cc.N_FOLDS
    print(
        f"Tuning sweep: {len(combos)} (H,N) combos x {len(cc.LANGUAGES)} langs x "
        f"{len(cc.EMBEDDING_FEATURE_SETS)} feature sets x {cc.N_FOLDS} folds "
        f"= {total_calls} Weka calls"
    )

    rows = []
    for hidden, epochs in combos:
        options = mlp_options(hidden, epochs)
        print(f"\n=== H={hidden} N={epochs} ===")
        for fold in range(cc.N_FOLDS):
            mono_dir = cc.SPLITS_CV_DIR / f"fold{fold}" / "monolingual"
            for lang in cc.LANGUAGES:
                for feat in cc.EMBEDDING_FEATURE_SETS:
                    train_arff = mono_dir / f"{lang}_train_{feat}.arff"
                    test_arff = mono_dir / f"{lang}_test_{feat}.arff"
                    metrics = cc.run_weka(train_arff, test_arff, CLASSIFIER, options)
                    rows.append({
                        "hidden": hidden,
                        "epochs": epochs,
                        "fold": fold,
                        "lang": lang,
                        "feature_set": feat,
                        "accuracy": metrics["accuracy"],
                        "kappa": metrics["kappa"],
                        "f1_weighted": metrics["f1_weighted"],
                    })
            print(f"  fold {fold} -> done")

    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False)

    summary = (
        df.groupby(["hidden", "epochs"])["accuracy"]
        .agg(["mean", "std"])
        .reset_index()
        .sort_values("mean", ascending=False)
    )

    print("\n" + "=" * 60)
    print("TUNING SUMMARY (mean accuracy, 3 langs x 4 featsets x 5 folds)")
    print("=" * 60)
    print(summary.to_string(index=False))

    best = summary.iloc[0]
    print(
        f"\nBest: H={int(best['hidden'])} N={int(best['epochs'])} "
        f"(mean accuracy {best['mean']:.2f} +/- {best['std']:.2f})"
    )
    print(f"\nFull results written to: {OUT_CSV}")


if __name__ == "__main__":
    main()
