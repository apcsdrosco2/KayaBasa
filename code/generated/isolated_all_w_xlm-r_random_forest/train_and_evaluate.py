"""
train_and_evaluate.py — Full experimental matrix using Weka's RandomForest.
Includes XLM-R embeddings. "all" has two variants: all_mbert and all_xlmr.

Setup:
  Train sets: TGL, BCL, CEB, TGL+BCL, BCL+CEB, CEB+TGL, ALL
  Test sets:  Each language
  Feature sets: trad, trad_clgsngo, mbert, xlmr, all_mbert, all_xlmr

Outputs:
  results/
    results_summary.csv
    results_detailed.txt
    results_table.txt
"""

import subprocess
import re
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

# ═══════════════════════════════════════════════════════════════════════════════
# Config
# ═══════════════════════════════════════════════════════════════════════════════

JAVA_PATH = r"C:\Program Files\Weka-3-8-7\jre\jre-25.0.2-full\bin\java.exe"
WEKA_JAR = r"C:\Program Files\Weka-3-8-7\weka.jar"

SCRIPT_DIR = Path(__file__).resolve().parent
SPLITS_DIR = SCRIPT_DIR / "splits"
MONO_DIR = SPLITS_DIR / "monolingual"
BI_DIR = SPLITS_DIR / "bilingual"
RESULTS_DIR = SCRIPT_DIR / "results"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

LANGUAGES = ["tagalog", "bikol", "cebuano"]
PAIRS = [("bikol", "tagalog"), ("cebuano", "bikol"), ("tagalog", "cebuano")]
FEATURE_SETS = ["trad", "trad_clgsngo", "mbert", "xlmr", "all_mbert", "all_xlmr"]

WEKA_CLASSIFIER = "weka.classifiers.trees.RandomForest"
WEKA_OPTIONS = ["-I", "100", "-K", "0", "-depth", "0", "-S", "1"]


# ═══════════════════════════════════════════════════════════════════════════════
# Weka CLI
# ═══════════════════════════════════════════════════════════════════════════════

def run_weka(train_arff: Path, test_arff: Path) -> dict:
    """Run Weka RandomForest via CLI and parse results."""
    cmd = [
        JAVA_PATH, "-Xmx4g",
        "-cp", WEKA_JAR,
        WEKA_CLASSIFIER,
        "-t", str(train_arff),
        "-T", str(test_arff),
    ] + WEKA_OPTIONS

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    output = result.stdout + result.stderr

    metrics = parse_weka_output(output)
    metrics["raw_output"] = output
    return metrics


def parse_weka_output(output: str) -> dict:
    """Extract metrics from Weka's test data output."""
    metrics = {
        "accuracy": 0.0,
        "kappa": 0.0,
        "f1_weighted": 0.0,
        "precision_weighted": 0.0,
        "recall_weighted": 0.0,
    }

    # Find test data section
    test_section = output
    test_marker = "=== Error on test data ==="
    if test_marker in output:
        test_section = output[output.index(test_marker):]

    m = re.search(r"Correctly Classified Instances\s+\d+\s+([\d.]+)\s*%", test_section)
    if m:
        metrics["accuracy"] = float(m.group(1))

    m = re.search(r"Kappa statistic\s+([\d.\-]+)", test_section)
    if m:
        metrics["kappa"] = float(m.group(1))

    m = re.search(r"Weighted Avg\.\s+([\d.?]+)\s+([\d.?]+)\s+([\d.?]+)", test_section)
    if m:
        try:
            metrics["precision_weighted"] = float(m.group(1))
            metrics["recall_weighted"] = float(m.group(2))
            metrics["f1_weighted"] = float(m.group(3))
        except ValueError:
            pass

    return metrics


# ═══════════════════════════════════════════════════════════════════════════════
# Build all-languages train ARFF files
# ═══════════════════════════════════════════════════════════════════════════════

ALL_TRAIN_DIR = SPLITS_DIR / "all_languages"
ALL_TRAIN_DIR.mkdir(parents=True, exist_ok=True)


def build_all_languages_arff():
    """Stack all 3 language train ARFFs into combined files."""
    for feat in FEATURE_SETS:
        all_lines_header = []
        all_lines_data = []
        header_done = False

        for lang in LANGUAGES:
            arff_path = MONO_DIR / f"{lang}_train_{feat}.arff"
            if not arff_path.exists():
                print(f"  WARNING: {arff_path.name} not found, skipping {feat}")
                continue

            with open(arff_path, "r", encoding="utf-8") as f:
                lines = f.readlines()

            in_data = False
            for line in lines:
                stripped = line.strip()
                if stripped.upper().startswith("@DATA"):
                    in_data = True
                    if not header_done:
                        all_lines_header.append(line)
                    continue
                if in_data:
                    if stripped:
                        all_lines_data.append(line)
                else:
                    if not header_done:
                        all_lines_header.append(line)
            header_done = True

        out_path = ALL_TRAIN_DIR / f"all_languages_train_{feat}.arff"
        with open(out_path, "w", encoding="utf-8") as f:
            f.writelines(all_lines_header)
            for data_line in all_lines_data:
                f.write(data_line)


print("Building all-languages train ARFF files...")
build_all_languages_arff()

# ═══════════════════════════════════════════════════════════════════════════════
# Run experiments
# ═══════════════════════════════════════════════════════════════════════════════

all_results = []
detailed_reports = []

timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
detailed_reports.append(f"Weka RandomForest Results (incl. XLM-R) — {timestamp}\n{'=' * 70}\n")

# ─── 1. Monolingual ──────────────────────────────────────────────────────────

print("\n" + "=" * 70)
print("MONOLINGUAL: Train on 1 language -> Test on each language")
print("=" * 70)

for train_lang in LANGUAGES:
    for feat in FEATURE_SETS:
        train_arff = MONO_DIR / f"{train_lang}_train_{feat}.arff"
        if not train_arff.exists():
            continue

        for test_lang in LANGUAGES:
            test_arff = MONO_DIR / f"{test_lang}_test_{feat}.arff"
            if not test_arff.exists():
                continue

            metrics = run_weka(train_arff, test_arff)
            all_results.append({
                "train_set": f"mono_{train_lang}",
                "test_set": f"{test_lang}_test",
                "feature_set": feat,
                "accuracy": metrics["accuracy"],
                "f1_weighted": metrics["f1_weighted"],
                "precision_weighted": metrics["precision_weighted"],
                "recall_weighted": metrics["recall_weighted"],
                "kappa": metrics["kappa"],
            })
            detailed_reports.append(
                f"\n{'─' * 70}\n"
                f"Train: {train_lang} | Test: {test_lang} | Features: {feat}\n"
                f"{'─' * 70}\n{metrics['raw_output']}\n"
            )

    print(f"  Train: {train_lang.upper()} -> done")

# ─── 2. Bilingual ────────────────────────────────────────────────────────────

print("\n" + "=" * 70)
print("BILINGUAL: Train on pair -> Test on each language")
print("=" * 70)

for lang1, lang2 in PAIRS:
    pair_name = f"{lang1}_{lang2}"

    for feat in FEATURE_SETS:
        train_arff = BI_DIR / f"paired_{pair_name}_train_{feat}.arff"
        if not train_arff.exists():
            # Try reverse
            pair_name_rev = f"{lang2}_{lang1}"
            train_arff = BI_DIR / f"paired_{pair_name_rev}_train_{feat}.arff"
            if train_arff.exists():
                pair_name = pair_name_rev
            else:
                continue

        for test_lang in LANGUAGES:
            test_arff = MONO_DIR / f"{test_lang}_test_{feat}.arff"
            if not test_arff.exists():
                continue

            metrics = run_weka(train_arff, test_arff)
            all_results.append({
                "train_set": f"bi_{pair_name}",
                "test_set": f"{test_lang}_test",
                "feature_set": feat,
                "accuracy": metrics["accuracy"],
                "f1_weighted": metrics["f1_weighted"],
                "precision_weighted": metrics["precision_weighted"],
                "recall_weighted": metrics["recall_weighted"],
                "kappa": metrics["kappa"],
            })
            detailed_reports.append(
                f"\n{'─' * 70}\n"
                f"Train: {pair_name} | Test: {test_lang} | Features: {feat}\n"
                f"{'─' * 70}\n{metrics['raw_output']}\n"
            )

    print(f"  Train: {pair_name.upper()} -> done")

# ─── 3. All languages ────────────────────────────────────────────────────────

print("\n" + "=" * 70)
print("ALL LANGUAGES: Train on all 3 -> Test on each language")
print("=" * 70)

for feat in FEATURE_SETS:
    train_arff = ALL_TRAIN_DIR / f"all_languages_train_{feat}.arff"
    if not train_arff.exists():
        continue

    for test_lang in LANGUAGES:
        test_arff = MONO_DIR / f"{test_lang}_test_{feat}.arff"
        if not test_arff.exists():
            continue

        metrics = run_weka(train_arff, test_arff)
        all_results.append({
            "train_set": "all_languages",
            "test_set": f"{test_lang}_test",
            "feature_set": feat,
            "accuracy": metrics["accuracy"],
            "f1_weighted": metrics["f1_weighted"],
            "precision_weighted": metrics["precision_weighted"],
            "recall_weighted": metrics["recall_weighted"],
            "kappa": metrics["kappa"],
        })
        detailed_reports.append(
            f"\n{'─' * 70}\n"
            f"Train: ALL | Test: {test_lang} | Features: {feat}\n"
            f"{'─' * 70}\n{metrics['raw_output']}\n"
        )

print("  Train: ALL -> done")

# ═══════════════════════════════════════════════════════════════════════════════
# Save results
# ═══════════════════════════════════════════════════════════════════════════════

results_df = pd.DataFrame(all_results)
results_df.to_csv(RESULTS_DIR / "results_summary.csv", index=False)

with open(RESULTS_DIR / "results_detailed.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(detailed_reports))

# ═══════════════════════════════════════════════════════════════════════════════
# Build formatted table
# ═══════════════════════════════════════════════════════════════════════════════

train_map = {
    "mono_tagalog": "TGL",
    "mono_bikol": "BCL",
    "mono_cebuano": "CEB",
    "bi_tagalog_bikol": "TGL+BCL",
    "bi_bikol_tagalog": "TGL+BCL",
    "bi_bikol_cebuano": "BCL+CEB",
    "bi_cebuano_bikol": "BCL+CEB",
    "bi_cebuano_tagalog": "CEB+TGL",
    "bi_tagalog_cebuano": "CEB+TGL",
    "all_languages": "ALL",
}

feat_labels = ["TRAD", "TRAD+CrossNGO", "mBERT", "XLM-R", "ALL(mBERT)", "ALL(XLM-R)"]
feat_map = {
    "trad": "TRAD",
    "trad_clgsngo": "TRAD+CrossNGO",
    "mbert": "mBERT",
    "xlmr": "XLM-R",
    "all_mbert": "ALL(mBERT)",
    "all_xlmr": "ALL(XLM-R)",
}
test_map = {"tagalog_test": "TGL", "bikol_test": "BCL", "cebuano_test": "CEB"}

results_df["Model"] = results_df["train_set"].map(train_map)
results_df["Test_Lang"] = results_df["test_set"].map(test_map)
results_df["Features"] = results_df["feature_set"].map(feat_map)
results_df["col"] = results_df["Test_Lang"] + " | " + results_df["Features"]

col_order = [f"{l} | {f}" for l in ["TGL", "BCL", "CEB"] for f in feat_labels]
row_order = ["TGL", "BCL", "CEB", "TGL+BCL", "BCL+CEB", "CEB+TGL", "ALL"]

pivot = results_df.pivot_table(values="accuracy", index="Model", columns="col")
pivot = pivot.reindex(index=row_order, columns=col_order)

# Save formatted table
cw = 11
out = RESULTS_DIR / "results_table.txt"
with open(out, "w", encoding="utf-8") as f:
    f.write("Weka RandomForest Accuracy (%) - incl. XLM-R\n")
    f.write("Weka 3.8.7 | numIterations=100, maxDepth=unlimited, bagSizePercent=100,\n")
    f.write("             numFeatures=int(log(#predictors)+1), seed=1\n")
    f.write("=" * 210 + "\n\n")

    line = f"{'Model':<10}|"
    for lang in ["TGL", "BCL", "CEB"]:
        group_width = cw * 6
        line += f"{lang:^{group_width}}|"
    f.write(line + "\n")
    f.write("-" * 10 + "+" + ("-" * (cw * 6) + "+") * 3 + "\n")

    line = f"{'':10}|"
    for _ in range(3):
        for feat in feat_labels:
            line += f"{feat:>{cw}}"
        line += "|"
    f.write(line + "\n")
    f.write("-" * 10 + "+" + ("-" * (cw * 6) + "+") * 3 + "\n")

    for model in row_order:
        line = f"{model:<10}|"
        for lang in ["TGL", "BCL", "CEB"]:
            for feat in feat_labels:
                col_key = f"{lang} | {feat}"
                if col_key in pivot.columns and model in pivot.index:
                    val = pivot.loc[model, col_key]
                    if pd.isna(val):
                        line += f"{'N/A':>{cw}}"
                    else:
                        line += f"{val:>{cw}.3f}"
                else:
                    line += f"{'N/A':>{cw}}"
            line += "|"
        f.write(line + "\n")
        if model == "CEB" or model == "CEB+TGL":
            f.write("-" * 10 + "+" + ("-" * (cw * 6) + "+") * 3 + "\n")

    f.write("=" * 210 + "\n")

pivot.to_csv(RESULTS_DIR / "results_table.csv")

print("\n" + "=" * 70)
print("RESULTS TABLE")
print("=" * 70)
print(open(out, "r").read())
print(f"\nTotal experiments: {len(all_results)}")
