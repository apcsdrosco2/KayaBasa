"""
flesch_kincaid_baseline.py — Flesch-Kincaid Grade Level baseline (Chapter 3, §3.3.3).

Computes the classic Flesch-Kincaid Grade Level (FKGL) formula per document, using
the average_sentence_len (words/sentence) and average_syllable_count (syllables/word)
columns already produced by TRAD.py for the "trad" feature set — no retraining or
CV needed, since FKGL is a fixed arithmetic formula, not a learned model:

    FKGL = 0.39 * (words / sentence) + 11.8 * (syllables / word) - 15.59

The continuous FKGL score is mapped to a discrete L1/L2/L3 label using the same
grade-band cutoffs stated in the original proposal (Chapter 3, §3.3.3):
    FKGL <= 2.0        -> L1
    2.0 <  FKGL <= 3.0  -> L2
    FKGL >  3.0         -> L3

Run on the full 764-document corpus (all three languages) — not just a CV test
fold — because nothing is fit to the data; every document can be scored and
compared to its true label directly.

Outputs (code/generated/flesch_kincaid/):
    fkgl_scores.csv          — per-document FKGL score, predicted label, true label
    fkgl_results_summary.csv — per-language + overall accuracy/precision/recall/F1/kappa
    fkgl_confusion_matrices.csv — per-language + overall raw confusion matrices
"""

from pathlib import Path

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    precision_recall_fscore_support,
)

SCRIPT_DIR = Path(__file__).resolve().parent
GENERATED_DIR = SCRIPT_DIR / "generated"
OUT_DIR = GENERATED_DIR / "flesch_kincaid"

LANG_CONFIG = {
    "tagalog": "tag docus/tag_trad.csv",
    "cebuano": "ceb docus/ceb_trad.csv",
    "bikol": "bikol docus/bik_trad.csv",
}

LABELS = [1, 2, 3]


def fkgl_score(avg_sentence_len: float, avg_syllable_count: float) -> float:
    return 0.39 * avg_sentence_len + 11.8 * avg_syllable_count - 15.59


def fkgl_to_label(score: float) -> int:
    if score <= 2.0:
        return 1
    if score <= 3.0:
        return 2
    return 3


def evaluate(df: pd.DataFrame, name: str) -> dict:
    y_true = df["grade_level"].astype(int)
    y_pred = df["predicted_label"].astype(int)

    acc = accuracy_score(y_true, y_pred)
    kappa = cohen_kappa_score(y_true, y_pred)
    p_w, r_w, f1_w, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=LABELS, average="weighted", zero_division=0
    )
    p_m, r_m, f1_m, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=LABELS, average="macro", zero_division=0
    )

    return {
        "language": name,
        "n_docs": len(df),
        "accuracy_pct": round(acc * 100, 2),
        "kappa": round(kappa, 4),
        "precision_weighted": round(p_w, 4),
        "recall_weighted": round(r_w, 4),
        "f1_weighted": round(f1_w, 4),
        "precision_macro": round(p_m, 4),
        "recall_macro": round(r_m, 4),
        "f1_macro": round(f1_m, 4),
    }


def confusion_rows(df: pd.DataFrame, name: str) -> list:
    y_true = df["grade_level"].astype(int)
    y_pred = df["predicted_label"].astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=LABELS)
    rows = []
    for i, true_label in enumerate(LABELS):
        row = {"language": name, "true_label": f"L{true_label}"}
        for j, pred_label in enumerate(LABELS):
            row[f"pred_L{pred_label}"] = cm[i, j]
        rows.append(row)
    return rows


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    all_frames = []
    for lang, rel_path in LANG_CONFIG.items():
        path = GENERATED_DIR / rel_path
        df = pd.read_csv(path)
        df = df[["book_title", "average_sentence_len", "average_syllable_count", "grade_level"]].copy()
        df["language"] = lang
        df["fkgl_score"] = fkgl_score(df["average_sentence_len"], df["average_syllable_count"])
        df["predicted_label"] = df["fkgl_score"].apply(fkgl_to_label)
        all_frames.append(df)

    combined = pd.concat(all_frames, ignore_index=True)
    combined.to_csv(OUT_DIR / "fkgl_scores.csv", index=False)

    summary_rows = [evaluate(df, lang) for lang, df in zip(LANG_CONFIG, all_frames)]
    summary_rows.append(evaluate(combined, "ALL (pooled)"))
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(OUT_DIR / "fkgl_results_summary.csv", index=False)

    cm_rows = []
    for lang, df in zip(LANG_CONFIG, all_frames):
        cm_rows.extend(confusion_rows(df, lang))
    cm_rows.extend(confusion_rows(combined, "ALL (pooled)"))
    pd.DataFrame(cm_rows).to_csv(OUT_DIR / "fkgl_confusion_matrices.csv", index=False)

    print("=" * 78)
    print("FLESCH-KINCAID GRADE LEVEL BASELINE — RESULTS")
    print("=" * 78)
    print(summary_df.to_string(index=False))
    print()

    print("Predicted-label distribution per language (how FKGL actually voted):")
    for lang, df in zip(LANG_CONFIG, all_frames):
        dist = df["predicted_label"].value_counts().sort_index().to_dict()
        true_dist = df["grade_level"].value_counts().sort_index().to_dict()
        print(f"  {lang:10s} predicted={dist}  true={true_dist}")
    print()

    print(f"Wrote: {OUT_DIR / 'fkgl_scores.csv'}")
    print(f"Wrote: {OUT_DIR / 'fkgl_results_summary.csv'}")
    print(f"Wrote: {OUT_DIR / 'fkgl_confusion_matrices.csv'}")


if __name__ == "__main__":
    main()
