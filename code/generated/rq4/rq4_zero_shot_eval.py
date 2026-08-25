"""
rq4_zero_shot_eval.py — Stage F of the RQ4 pipeline.

Zero-shot evaluation: train once on ALL 764 Tagalog+Cebuano+Bikolano documents (no
held-out split, Stage E), test on each of the 4 low-resource languages' full data
(Stage D), for the KayaBasa model (bagged MLP hybrid) x {trad_clgsngo (24-dim,
features-only), all_xlmr (24+768-dim, hybrid)} x 4 languages = 8 Weka calls.
RandomForest is NOT run here — per coordinator decision, this stage reports only the
KayaBasa model's own zero-shot transfer, not a baseline comparison. (An earlier
revision of this script also ran a RandomForest condition; that comparison was
dropped, not just hidden from the write-up.) Classifier config (H=128, N=10 epochs,
10 bags) is copied verbatim from train_and_evaluate_cv.py/run_final_hybrid_model.py —
same hyperparameters as RO1-RO3, per RQ4_PLAN.md Open Question 2 (no re-tuning for
this new condition). Reuses cv_common.run_weka/parse_weka_output for the actual Weka
invocation/parsing.

This is a genuine zero-shot condition, distinct from the pairwise/all-languages
conditions in the RO1-3 CV pipeline (Section 3.3.2 Phase 1): those measure whether
additional related-language training data improves performance on a target language
that is itself IN the training set. Here, none of the four low-resource languages
ever contribute training examples — training is exclusively Tagalog+Cebuano+Bikolano,
testing is exclusively the unseen language — so this is reported as its own,
separate result, not combined into RO1-3's transfer figures.

Builds two additional artifacts this stage needs that didn't exist yet:
  - all_languages_full_train_trad_clgsngo.arff (764 rows, 24 features + class) —
    the features-only counterpart to Stage E's all_xlmr training ARFF.
  - {lang}_test_trad_clgsngo.arff per low-resource language (24 features + class) —
    the features-only counterpart to Stage D's all_xlmr test ARFFs, built directly
    from the already-computed Stage B CSV (code/generated/rq4/generated/{lang}_trad_clgsngo.csv).

Usage: python rq4_zero_shot_eval.py
"""

import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PAIRWISE_ARFF_DIR = SCRIPT_DIR.parent / "pairwise" / "arff"
sys.path.insert(0, str(PAIRWISE_ARFF_DIR))

import pandas as pd
import train_test_split as tts
import cv_common as cc

GENERATED_DIR = SCRIPT_DIR / "generated"
RESULTS_DIR = PAIRWISE_ARFF_DIR / "results_rq4_zero_shot"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_LANGUAGES = ["tagalog", "cebuano", "bikol"]
TEST_LANGUAGES = ["hiligaynon", "minasbate", "karay-a", "rinconada"]
BISAYAN = ["hiligaynon", "minasbate", "karay-a"]  # vs. rinconada (Bikol-affiliated)

CLASSIFIERS = {
    "bagged_mlp": {
        "classifier": "weka.classifiers.meta.Bagging",
        "options": [
            "-I", "10", "-S", "1", "-num-slots", "0", "-o",
            "-W", "weka.classifiers.functions.MultilayerPerceptron", "--",
            "-L", "0.3", "-M", "0.2", "-N", "10", "-V", "0", "-E", "20", "-H", "128", "-S", "1",
        ],
        "label": "Bagged MLP x10 (KayaBasa model)",
    },
}

FEATURE_SETS = ["trad_clgsngo", "all_xlmr"]


def build_missing_trad_clgsngo_artifacts():
    """Build the trad_clgsngo-only train/test ARFFs this stage needs, if absent."""
    full_train_path = GENERATED_DIR / "all_languages_full_train_trad_clgsngo.arff"
    if not full_train_path.exists():
        frames = []
        for lang in TRAIN_LANGUAGES:
            df = tts.load_language_features(lang)["trad_clgsngo"]
            frames.append(df)
        full = pd.concat(frames, axis=0, ignore_index=True)
        assert len(full) == 764, f"Expected 764 rows, got {len(full)}"
        assert len(full.columns) == 25, f"Expected 25 columns (24 features + class), got {len(full.columns)}"
        tts.save_arff(full, "readability_all_languages_full_trad_clgsngo", full_train_path)
        print(f"Built {full_train_path} ({len(full)} rows)")

    for lang in TEST_LANGUAGES:
        out_path = GENERATED_DIR / f"{lang}_test_trad_clgsngo.arff"
        if out_path.exists():
            continue
        feat_path = GENERATED_DIR / f"{lang}_trad_clgsngo.csv"
        feat_df = pd.read_csv(feat_path)
        cols = [c for c in feat_df.columns if c not in ("doc_id", "flag")]
        df = feat_df[cols]
        assert len(df.columns) == 25, f"{lang}: expected 25 columns, got {len(df.columns)}"
        tts.save_arff(df, f"readability_{lang}_test_trad_clgsngo", out_path)
        print(f"Built {out_path} ({len(df)} rows)")


def write_results_table(results_df: pd.DataFrame, comparison_df: pd.DataFrame, path: Path):
    """Formatted Accuracy%/Macro-F1 table, same visual style as
    cv_common.write_results_table_cv's CV matrices (results_cv_rf/results_table_cv.txt,
    results_cv_mlp_bagged_h128_n10_b10/results_table_cv.txt) — but no fold mean/std,
    since this is a single zero-shot run, not a 5-fold average (see RQ4_PLAN.md /
    Chapter4_Draft.md §4.4's note on why no std dev is reported here)."""
    feat_cols = FEATURE_SETS  # ["trad_clgsngo", "all_xlmr"]
    feat_labels = [cc.FEAT_MAP[f] for f in feat_cols]
    lang_labels = {"hiligaynon": "Hiligaynon", "minasbate": "Minasbate",
                   "karay-a": "Karay-a", "rinconada": "Rinconada"}

    cw = 24  # column width per feature set (fits "Acc=100.00% F1=0.9999")
    row_label_w = 20  # fits "Bisayan (Hil+Kar+Min)" (the longest row label) in full
    group_w = cw
    total_w = row_label_w + 1 + (group_w + 1) * len(feat_cols)
    sep_line = "-" * row_label_w + "+" + ("-" * group_w + "+") * len(feat_cols)

    def cell(acc, f1):
        return f"Acc={acc:6.2f}% F1={f1:.4f}"

    pivot = {}
    for _, r in results_df.iterrows():
        pivot[(r["test_language"], r["feature_set"])] = (r["accuracy"], r["f1_macro"])

    with open(path, "w", encoding="utf-8") as f:
        f.write("Bagged MultilayerPerceptron x10 (KayaBasa model) - Zero-Shot Transfer "
                "to Low-Resource Languages\n")
        f.write("Weka 3.8.7 | Bagging(numIterations=10, numSlots=auto) of MultilayerPerceptron:\n")
        f.write("             learningRate=0.3, momentum=0.2, trainingTime=10, hiddenLayers=128, seed=1\n")
        f.write("Trained ONCE on all 764 Tagalog+Cebuano+Bikolano documents combined (no held-out\n")
        f.write("split); tested zero-shot on each language's full corpus. Single run, no fold\n")
        f.write("std dev (see Chapter4_Draft.md Sec 4.4). RandomForest is not run for this table.\n")
        f.write("=" * total_w + "\n\n")

        line = f"{'Language':<{row_label_w}}|"
        for feat_label in feat_labels:
            line += f"{feat_label:^{group_w}}|"
        f.write(line + "\n")
        f.write(sep_line + "\n")

        for lang in TEST_LANGUAGES:
            line = f"{lang_labels[lang]:<{row_label_w}}|"
            for feat in feat_cols:
                acc, f1 = pivot[(lang, feat)]
                line += f"{cell(acc, f1):^{group_w}}|"
            f.write(line + "\n")
        f.write(sep_line + "\n")

        # Bisayan (Hiligaynon+Karay-a+Minasbate) mean vs. Rinconada, per feature set
        comp_pivot = {}
        for _, r in comparison_df.iterrows():
            comp_pivot[(r["group"], r["feature_set"])] = (r["accuracy"], r["f1_macro"])
        display_labels = {"Bisayan (Hil+Kar+Min) mean": "Bisayan mean", "Rinconada": "Rinconada"}
        for group_label, display_label in display_labels.items():
            line = f"{display_label:<{row_label_w}}|"
            for feat in feat_cols:
                acc, f1 = comp_pivot[(group_label, feat)]
                line += f"{cell(acc, f1):^{group_w}}|"
            f.write(line + "\n")
        f.write("=" * total_w + "\n")


def main():
    build_missing_trad_clgsngo_artifacts()

    results = []
    detailed_lines = []

    for clf_key, clf_cfg in CLASSIFIERS.items():
        for feat in FEATURE_SETS:
            train_arff = GENERATED_DIR / f"all_languages_full_train_{feat}.arff"
            if not train_arff.exists():
                raise FileNotFoundError(f"Missing training ARFF: {train_arff}")

            for lang in TEST_LANGUAGES:
                test_arff = GENERATED_DIR / f"{lang}_test_{feat}.arff"
                if not test_arff.exists():
                    raise FileNotFoundError(f"Missing test ARFF: {test_arff}")

                print(f"Running: {clf_cfg['label']} | {feat} | test={lang} ...")
                metrics = cc.run_weka(train_arff, test_arff, clf_cfg["classifier"], clf_cfg["options"])

                row = {
                    "classifier": clf_key,
                    "classifier_label": clf_cfg["label"],
                    "feature_set": feat,
                    "test_language": lang,
                    "bisayan_affiliated": lang in BISAYAN,
                    "accuracy": metrics["accuracy"],
                    "kappa": metrics["kappa"],
                    "f1_weighted": metrics["f1_weighted"],
                    "f1_macro": metrics["f1_macro"],
                    "precision_weighted": metrics["precision_weighted"],
                    "recall_weighted": metrics["recall_weighted"],
                }
                results.append(row)
                detailed_lines.append(
                    f"\n{'='*80}\n{clf_cfg['label']} | {feat} | test={lang}\n{'='*80}\n"
                    + metrics["raw_output"]
                )

                # Known silent-failure signature (per this project's established
                # convention, cv_common.py/retry_failed_cells.py precedent): a JVM
                # crash under memory pressure parses as all-zero rather than raising.
                if metrics["accuracy"] == 0.0 and metrics["kappa"] == 0.0 and metrics["f1_weighted"] == 0.0:
                    print(f"  WARNING: possible silent Weka failure (all-zero metrics) for "
                          f"{clf_key}/{feat}/{lang} — check results_detailed.txt")
                else:
                    print(f"  accuracy={metrics['accuracy']:.2f}%  f1_macro={metrics['f1_macro']:.4f}")

    results_df = pd.DataFrame(results)
    results_df.to_csv(RESULTS_DIR / "results_summary_rq4.csv", index=False)

    with open(RESULTS_DIR / "results_detailed.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(detailed_lines))

    # Bisayan (Hiligaynon, Karay-a, Minasbate) vs. Rinconada (Bikol-affiliated) comparison
    comparison = (
        results_df.groupby(["classifier_label", "feature_set", "bisayan_affiliated"])
        [["accuracy", "f1_macro"]].mean().reset_index()
    )
    comparison["group"] = comparison["bisayan_affiliated"].map({True: "Bisayan (Hil+Kar+Min) mean", False: "Rinconada"})
    comparison = comparison.drop(columns=["bisayan_affiliated"])
    comparison.to_csv(RESULTS_DIR / "results_bisayan_vs_rinconada.csv", index=False)

    table_path = RESULTS_DIR / "results_table.txt"
    write_results_table(results_df, comparison, table_path)

    print(f"\n=== Done: {len(results)} Weka runs ===")
    print(f"Summary:   {RESULTS_DIR / 'results_summary_rq4.csv'}")
    print(f"Detailed:  {RESULTS_DIR / 'results_detailed.txt'}")
    print(f"Bisayan vs Rinconada: {RESULTS_DIR / 'results_bisayan_vs_rinconada.csv'}")
    print(f"Table:     {table_path}")
    print("\n" + open(table_path, "r", encoding="utf-8").read())

    n_silent_failures = ((results_df["accuracy"] == 0.0) & (results_df["kappa"] == 0.0) & (results_df["f1_weighted"] == 0.0)).sum()
    if n_silent_failures:
        print(f"\nWARNING: {n_silent_failures}/{len(results_df)} runs show the all-zero silent-failure "
              f"signature — check results_detailed.txt before trusting the summary.")


if __name__ == "__main__":
    main()
