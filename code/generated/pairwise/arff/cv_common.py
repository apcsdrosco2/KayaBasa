"""
cv_common.py — shared config/helpers for the 5-fold CV evaluation scripts.

train_and_evaluate_cv.py runs this same matrix twice — once for the RandomForest
baseline reproduction and once for the XLM-R+MLP hybrid — and the two runs should
differ ONLY in classifier/options. Everything else (Weka invocation, output parsing,
the all-languages ARFF stacking, and the results table layout) lives here once so the
two runs can't silently drift apart.
"""

import re
import subprocess
from pathlib import Path

from train_test_split import FEATURE_SETS

# The subset of FEATURE_SETS that actually uses a neural embedding. RandomForest is
# expected to win on the tiny 18/24-feature TRAD sets regardless of classifier choice,
# so "hybrid vs baseline" comparisons and MLP tuning/bagging runs restrict to these.
EMBEDDING_FEATURE_SETS = ["mbert", "all", "xlmr", "all_xlmr"]

JAVA_PATH = r"C:\Program Files\Weka-3-8-7\jre\jre-25.0.2-full\bin\java.exe"
WEKA_JAR = r"C:\Program Files\Weka-3-8-7\weka.jar"

SCRIPT_DIR = Path(__file__).resolve().parent
SPLITS_CV_DIR = SCRIPT_DIR / "splits_cv"
N_FOLDS = 5

LANGUAGES = ["tagalog", "bikol", "cebuano"]
PAIRS = [("tagalog", "bikol"), ("bikol", "cebuano"), ("cebuano", "tagalog")]

TRAIN_MAP = {
    "mono_tagalog": "TGL",
    "mono_bikol": "BCL",
    "mono_cebuano": "CEB",
    "bi_bikol_tagalog": "TGL+BCL",
    "bi_cebuano_bikol": "BCL+CEB",
    "bi_tagalog_cebuano": "CEB+TGL",
    "all_languages": "ALL",
}
ROW_ORDER = ["TGL", "BCL", "CEB", "TGL+BCL", "BCL+CEB", "CEB+TGL", "ALL"]

FEAT_LABELS = ["TRAD", "TRAD+CrossNGO", "mBERT Embdng", "ALL (mBERT)", "XLM-R Embdng", "ALL (XLM-R)"]
FEAT_MAP = {
    "trad": "TRAD",
    "trad_clgsngo": "TRAD+CrossNGO",
    "mbert": "mBERT Embdng",
    "all": "ALL (mBERT)",
    "xlmr": "XLM-R Embdng",
    "all_xlmr": "ALL (XLM-R)",
}
TEST_MAP = {"tagalog_test": "TGL", "bikol_test": "BCL", "cebuano_test": "CEB"}

COL_ORDER = [f"{l} | {f}" for l in ["TGL", "BCL", "CEB"] for f in FEAT_LABELS]


# ═══════════════════════════════════════════════════════════════════════════════
# Weka CLI runner
# ═══════════════════════════════════════════════════════════════════════════════

def run_weka(train_arff: Path, test_arff: Path, classifier: str, options: list) -> dict:
    """Run a Weka classifier via CLI and parse results."""
    cmd = [
        JAVA_PATH, "-Xmx4g",
        "-cp", WEKA_JAR,
        classifier,
        "-t", str(train_arff),
        "-T", str(test_arff),
    ] + options

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    output = result.stdout + result.stderr

    metrics = parse_weka_output(output)
    metrics["raw_output"] = output
    return metrics


def parse_weka_output(output: str) -> dict:
    """Extract metrics from Weka's test data output (not training data)."""
    metrics = {
        "accuracy": 0.0,
        "kappa": 0.0,
        "f1_weighted": 0.0,
        "precision_weighted": 0.0,
        "recall_weighted": 0.0,
        "f1_macro": 0.0,
    }

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

    # Macro-F1: the unweighted mean of each class's own F-Measure, from the
    # "=== Detailed Accuracy By Class ===" per-class rows (not the "Weighted
    # Avg." row above, which is weighted by class support). A class Weka
    # can't compute F-Measure for (e.g. never predicted, so precision is
    # undefined) prints "?" — treated as 0.0, matching scikit-learn's
    # zero_division=0 convention for macro-F1.
    class_block = re.search(
        r"=== Detailed Accuracy By Class ===\s*\n\s*TP Rate.*\n((?:.*\n)*?)Weighted Avg\.",
        test_section,
    )
    if class_block:
        f_measures = []
        for line in class_block.group(1).splitlines():
            parts = line.split()
            if len(parts) < 9:
                continue
            f_measure_str = parts[4]
            f_measures.append(0.0 if f_measure_str == "?" else float(f_measure_str))
        if f_measures:
            metrics["f1_macro"] = sum(f_measures) / len(f_measures)

    return metrics


# ═══════════════════════════════════════════════════════════════════════════════
# ARFF assembly helpers (per fold)
# ═══════════════════════════════════════════════════════════════════════════════

def build_all_languages_arff(mono_dir: Path, all_train_dir: Path):
    """Stack all 3 languages' train ARFFs (from mono_dir) into all_train_dir, per feature set."""
    all_train_dir.mkdir(parents=True, exist_ok=True)
    for feat in FEATURE_SETS:
        all_lines_header = []
        all_lines_data = []
        header_done = False

        for lang in LANGUAGES:
            arff_path = mono_dir / f"{lang}_train_{feat}.arff"
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

        out_path = all_train_dir / f"all_languages_train_{feat}.arff"
        with open(out_path, "w", encoding="utf-8") as f:
            f.writelines(all_lines_header)
            for data_line in all_lines_data:
                f.write(data_line)


def find_bilingual_train_arff(bi_dir: Path, lang1: str, lang2: str, feat: str):
    """Return (path, resolved_pair_name) for a pairwise train ARFF, trying both name orders."""
    pair_name = f"{lang1}_{lang2}"
    train_arff = bi_dir / f"paired_{pair_name}_train_{feat}.arff"
    if not train_arff.exists():
        pair_name_rev = f"{lang2}_{lang1}"
        rev_path = bi_dir / f"paired_{pair_name_rev}_train_{feat}.arff"
        if rev_path.exists():
            return rev_path, pair_name_rev
    return train_arff, pair_name


# ═══════════════════════════════════════════════════════════════════════════════
# Results table (mean +/- std across folds)
# ═══════════════════════════════════════════════════════════════════════════════

def _pivot(df, values_col, col_order):
    pivot = df.pivot_table(values=values_col, index="Model", columns="col")
    return pivot.reindex(index=ROW_ORDER, columns=col_order)


def write_results_table_cv(results_df, path: Path, title: str, header_lines: list, feat_labels=None):
    """Write a formatted mean+/-std accuracy matrix across folds.

    results_df must have one row per (fold, Model, col) with an 'accuracy' column
    (Model/col are the same label scheme train_and_evaluate.py's table uses).
    `feat_labels` restricts which feature-set columns are rendered (defaults to all
    of FEAT_LABELS) — pass a subset (e.g. the embedding-only labels) for runs that
    only evaluated those feature sets, so the table doesn't show blank TRAD columns.
    Returns (mean_pivot, std_pivot) so the caller can also save them as CSV.
    """
    feat_labels = feat_labels or FEAT_LABELS
    col_order = [f"{l} | {f}" for l in ["TGL", "BCL", "CEB"] for f in feat_labels]

    grouped = results_df.groupby(["Model", "col"])["accuracy"]
    mean_df = grouped.mean().reset_index()
    std_df = grouped.std().reset_index().fillna(0.0)  # std of 1 fold-count edge case -> 0

    mean_pivot = _pivot(mean_df, "accuracy", col_order)
    std_pivot = _pivot(std_df, "accuracy", col_order)

    cw = 16
    nfeat = len(feat_labels)
    group_w = cw * nfeat
    total_w = 10 + 1 + (group_w + 1) * 3
    sep_line = "-" * 10 + "+" + ("-" * group_w + "+") * 3

    with open(path, "w", encoding="utf-8") as f:
        f.write(f"{title}\n")
        for line in header_lines:
            f.write(f"{line}\n")
        f.write("=" * total_w + "\n\n")

        line = f"{'Model':<10}|"
        for lang in ["TGL", "BCL", "CEB"]:
            line += f"{lang:^{group_w}}|"
        f.write(line + "\n")
        f.write(sep_line + "\n")

        line = f"{'':10}|"
        for _ in range(3):
            for feat in feat_labels:
                line += f"{feat:>{cw}}"
            line += "|"
        f.write(line + "\n")
        f.write(sep_line + "\n")

        for model in ROW_ORDER:
            line = f"{model:<10}|"
            for lang in ["TGL", "BCL", "CEB"]:
                for feat in feat_labels:
                    col_key = f"{lang} | {feat}"
                    mean_val = mean_pivot.loc[model, col_key]
                    std_val = std_pivot.loc[model, col_key]
                    cell = f"{mean_val:.1f}\u00b1{std_val:.1f}"
                    line += f"{cell:>{cw}}"
                line += "|"
            f.write(line + "\n")
            if model == "CEB" or model == "CEB+TGL":
                f.write(sep_line + "\n")

        f.write("=" * total_w + "\n")

    return mean_pivot, std_pivot
