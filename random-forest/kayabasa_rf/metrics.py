"""Evaluation metrics (proposal Sections 3.3.1, 3.3.4, and 3.1.5)."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)

from . import config

LABELS = list(config.LABELS)
LABEL_NAMES = [config.LABEL_NAMES[label] for label in LABELS]


def non_adjacent_error_rate(y_true, y_pred) -> float:
    """Share of predictions that miss by two or more levels (L1 <-> L3)."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if y_true.size == 0:
        return float("nan")
    return float(np.mean(np.abs(y_true - y_pred) >= 2))


def adjacent_error_rate(y_true, y_pred) -> float:
    """Share of predictions that miss by exactly one level."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if y_true.size == 0:
        return float("nan")
    return float(np.mean(np.abs(y_true - y_pred) == 1))


def evaluate(y_true, y_pred) -> dict[str, float]:
    """Accuracy, macro-F1, per-class P/R/F1, and the two error-distance rates."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if y_true.size == 0:
        return {"n": 0}

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=LABELS, zero_division=0
    )
    out: dict[str, float] = {
        "n": int(y_true.size),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(
            f1_score(y_true, y_pred, labels=LABELS, average="macro", zero_division=0)
        ),
    }
    for i, name in enumerate(LABEL_NAMES):
        out[f"{name}_precision"] = float(precision[i])
        out[f"{name}_recall"] = float(recall[i])
        out[f"{name}_f1"] = float(f1[i])
        out[f"{name}_support"] = int(support[i])
    out["adjacent_error_rate"] = adjacent_error_rate(y_true, y_pred)
    out["non_adjacent_error_rate"] = non_adjacent_error_rate(y_true, y_pred)
    return out


def normalized_confusion_matrix(y_true, y_pred) -> pd.DataFrame:
    """Row-normalized confusion matrix (rows = true class), per Section 3.3.4."""
    cm = confusion_matrix(y_true, y_pred, labels=LABELS).astype(float)
    row_sums = cm.sum(axis=1, keepdims=True)
    # A class absent from this slice would otherwise divide by zero.
    normalized = np.divide(cm, row_sums, out=np.zeros_like(cm), where=row_sums > 0)
    return pd.DataFrame(
        normalized,
        index=[f"true_{name}" for name in LABEL_NAMES],
        columns=[f"pred_{name}" for name in LABEL_NAMES],
    )


def per_language_report(
    df: pd.DataFrame,
    true_col: str = "y_true",
    pred_col: str = "y_pred",
    group_col: str = "language",
) -> pd.DataFrame:
    """One metric row per language, plus a pooled ``ALL`` row."""
    rows = []
    for language, group in df.groupby(group_col):
        row = {group_col: language}
        row.update(evaluate(group[true_col], group[pred_col]))
        rows.append(row)
    rows.sort(key=lambda r: r[group_col])

    overall = {group_col: "ALL"}
    overall.update(evaluate(df[true_col], df[pred_col]))
    rows.append(overall)
    return pd.DataFrame(rows)


def fold_summary(
    fold_metrics: pd.DataFrame, by: list[str] | None = None
) -> pd.DataFrame:
    """Mean and standard deviation across folds."""
    by = by or []
    numeric = [
        c
        for c in fold_metrics.select_dtypes(include="number").columns
        if c != "fold"
    ]
    if by:
        grouped = fold_metrics.groupby(by)[numeric]
        out = pd.concat(
            [grouped.mean().add_suffix("_mean"), grouped.std(ddof=1).add_suffix("_std")],
            axis=1,
        )
        return out.reset_index()

    mean = fold_metrics[numeric].mean().add_suffix("_mean")
    std = fold_metrics[numeric].std(ddof=1).add_suffix("_std")
    return pd.concat([mean, std]).to_frame().T


# Paired comparison between two models on the same documents
def mcnemar_exact(correct_a, correct_b) -> dict[str, float]:
    """Exact McNemar test on paired correctness vectors."""
    from scipy.stats import binomtest

    correct_a = np.asarray(correct_a).astype(bool)
    correct_b = np.asarray(correct_b).astype(bool)

    only_a = int(np.sum(correct_a & ~correct_b))  # A right, B wrong
    only_b = int(np.sum(~correct_a & correct_b))  # B right, A wrong
    discordant = only_a + only_b

    if discordant == 0:
        return {
            "only_a_correct": 0,
            "only_b_correct": 0,
            "discordant": 0,
            "p_value": float("nan"),
        }

    result = binomtest(only_b, n=discordant, p=0.5, alternative="two-sided")
    return {
        "only_a_correct": only_a,
        "only_b_correct": only_b,
        "discordant": discordant,
        "p_value": float(result.pvalue),
    }


def bootstrap_macro_f1_delta(
    y_true,
    y_pred_a,
    y_pred_b,
    n_resamples: int = 2000,
    alpha: float = 0.05,
    seed: int = config.SEED,
) -> dict[str, float]:
    """Percentile CI for macro-F1(B) - macro-F1(A) by paired document resampling."""
    y_true = np.asarray(y_true)
    y_pred_a = np.asarray(y_pred_a)
    y_pred_b = np.asarray(y_pred_b)
    n = y_true.size

    def macro_f1(t, p):
        return f1_score(t, p, labels=LABELS, average="macro", zero_division=0)

    observed = macro_f1(y_true, y_pred_b) - macro_f1(y_true, y_pred_a)

    rng = np.random.default_rng(seed)
    deltas = np.empty(n_resamples)
    for i in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        deltas[i] = macro_f1(y_true[idx], y_pred_b[idx]) - macro_f1(
            y_true[idx], y_pred_a[idx]
        )

    lower, upper = np.percentile(deltas, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {
        "delta_macro_f1": float(observed),
        "ci_lower": float(lower),
        "ci_upper": float(upper),
        "ci_level": 1 - alpha,
        "n_resamples": n_resamples,
    }


def prediction_agreement(y_pred_a, y_pred_b) -> dict[str, float]:
    """How often two models agree, and Cohen's kappa on their predictions."""
    y_pred_a = np.asarray(y_pred_a)
    y_pred_b = np.asarray(y_pred_b)
    return {
        "raw_agreement": float(np.mean(y_pred_a == y_pred_b)),
        "cohens_kappa": float(cohen_kappa_score(y_pred_a, y_pred_b, labels=LABELS)),
    }
