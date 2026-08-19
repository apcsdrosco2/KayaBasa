"""Validation of the KAYABASA hybrid output against the Random Forest baseline."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from . import config, metrics

REQUIRED_COLUMNS = ("doc_id", "y_pred")
PROBA_COLUMNS = [f"proba_{name}" for name in metrics.LABEL_NAMES]

# Column aliases accepted on input, so a prediction file exported by the hybrid
COLUMN_ALIASES = {
    "id": "doc_id",
    "document_id": "doc_id",
    "doc": "doc_id",
    "label": "y_true",
    "true_label": "y_true",
    "gold": "y_true",
    "actual": "y_true",
    "prediction": "y_pred",
    "pred": "y_pred",
    "predicted": "y_pred",
    "predicted_label": "y_pred",
    "lang": "language",
}

PASS, WARN, FAIL = "PASS", "WARN", "FAIL"


@dataclass
class Check:
    name: str
    status: str
    detail: str

    @property
    def failed(self) -> bool:
        return self.status == FAIL


# Loading
def load_predictions(path: str | Path, name: str = "predictions") -> pd.DataFrame:
    """Read a prediction CSV and normalize its column names and label encoding."""
    df = pd.read_csv(path)
    df = df.rename(
        columns={
            c: COLUMN_ALIASES.get(c.strip().lower(), c.strip().lower())
            for c in df.columns
        }
    )

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"{name} file {path} is missing required column(s) {missing}. "
            f"Found: {list(df.columns)}"
        )

    for column in ("y_true", "y_pred"):
        if column in df.columns:
            df[column] = _coerce_labels(df[column], f"{name}.{column}")
    return df


def _coerce_labels(series: pd.Series, context: str) -> pd.Series:
    """Map L1/L2/L3 strings onto 1/2/3, leaving integers untouched."""
    if series.dtype == object:
        mapping = {"l1": 1, "l2": 2, "l3": 3, "1": 1, "2": 2, "3": 3}
        coerced = series.astype(str).str.strip().str.lower().map(mapping)
        if coerced.isna().any():
            bad = series[coerced.isna()].unique()[:5]
            raise ValueError(f"{context} contains unrecognized labels: {list(bad)}")
        return coerced.astype(int)
    return series.astype(int)


# Structural validation
def structural_checks(
    hybrid: pd.DataFrame,
    reference: pd.DataFrame,
    reference_name: str = "corpus",
) -> list[Check]:
    """Sanity-check the hybrid prediction file before any score is computed."""
    checks: list[Check] = []

    # 1. One row per document.
    duplicates = int(hybrid["doc_id"].duplicated().sum())
    checks.append(
        Check(
            "unique_doc_ids",
            PASS if duplicates == 0 else FAIL,
            "no duplicate doc_id"
            if duplicates == 0
            else f"{duplicates} duplicated doc_id values",
        )
    )

    # 2. Coverage against the corpus.
    ref_ids, hyb_ids = set(reference["doc_id"]), set(hybrid["doc_id"])
    missing, extra = ref_ids - hyb_ids, hyb_ids - ref_ids
    if not missing and not extra:
        coverage = Check(
            "coverage", PASS, f"all {len(ref_ids)} {reference_name} documents predicted"
        )
    elif missing:
        coverage = Check(
            "coverage",
            FAIL,
            f"{len(missing)} {reference_name} documents have no prediction "
            f"(e.g. {sorted(missing)[:3]}); reported metrics would silently "
            "exclude them",
        )
    else:
        coverage = Check(
            "coverage",
            WARN,
            f"{len(extra)} predicted documents are not in the {reference_name} "
            f"(e.g. {sorted(extra)[:3]}); they are ignored in the comparison",
        )
    checks.append(coverage)

    # 3. Label domain.
    bad_labels = sorted(set(hybrid["y_pred"]) - set(config.LABELS))
    checks.append(
        Check(
            "label_domain",
            PASS if not bad_labels else FAIL,
            "predictions are all L1/L2/L3"
            if not bad_labels
            else f"unexpected labels: {bad_labels}",
        )
    )

    # 4. Missing values.
    n_null = int(hybrid[["doc_id", "y_pred"]].isna().sum().sum())
    checks.append(
        Check(
            "no_missing_values",
            PASS if n_null == 0 else FAIL,
            "no nulls in doc_id/y_pred" if n_null == 0 else f"{n_null} null values",
        )
    )

    # 5. Gold labels must match the corpus, if the file supplies them.
    if "y_true" in hybrid.columns:
        merged = hybrid.merge(
            reference[["doc_id", "y_true"]], on="doc_id", suffixes=("_hyb", "_ref")
        )
        mismatches = int((merged["y_true_hyb"] != merged["y_true_ref"]).sum())
        checks.append(
            Check(
                "gold_labels_match_corpus",
                PASS if mismatches == 0 else FAIL,
                "hybrid gold labels agree with the corpus"
                if mismatches == 0
                else f"{mismatches} documents carry a different y_true than the "
                "corpus; the two models are not being scored against the same target",
            )
        )
    else:
        checks.append(
            Check(
                "gold_labels_match_corpus",
                WARN,
                "no y_true column supplied; corpus labels are used as ground truth",
            )
        )

    # 6. Degenerate output.
    distribution = hybrid["y_pred"].value_counts(normalize=True)
    top_share = float(distribution.iloc[0])
    if distribution.size < len(config.LABELS):
        diversity = Check(
            "prediction_diversity",
            FAIL,
            f"only {distribution.size} of {len(config.LABELS)} classes are ever "
            "predicted; macro-F1 is capped and the model cannot grade all levels",
        )
    elif top_share > 0.90:
        diversity = Check(
            "prediction_diversity",
            WARN,
            f"{top_share:.1%} of predictions are a single class; check for class "
            "collapse before trusting the accuracy figure",
        )
    else:
        shares = ", ".join(
            f"{config.LABEL_NAMES[int(k)]}={v:.1%}"
            for k, v in distribution.sort_index().items()
        )
        diversity = Check("prediction_diversity", PASS, f"all three levels predicted ({shares})")
    checks.append(diversity)

    # 7. Probability columns, when present, must be consistent with y_pred.
    if all(c in hybrid.columns for c in PROBA_COLUMNS):
        proba = hybrid[PROBA_COLUMNS].to_numpy(dtype=float)
        sums_ok = bool(np.allclose(proba.sum(axis=1), 1.0, atol=1e-3))
        argmax_labels = np.array(config.LABELS)[proba.argmax(axis=1)]
        n_argmax_mismatch = int((argmax_labels != hybrid["y_pred"].to_numpy()).sum())
        if sums_ok and n_argmax_mismatch == 0:
            status, detail = PASS, "probabilities sum to 1 and argmax matches y_pred"
        else:
            parts = []
            if not sums_ok:
                parts.append("rows do not sum to 1")
            if n_argmax_mismatch:
                parts.append(f"{n_argmax_mismatch} rows where argmax disagrees with y_pred")
            status, detail = WARN, "; ".join(parts)
        checks.append(Check("probability_consistency", status, detail))

    # 8. A near-perfect score on held-out data is a leakage smell, not a triumph.
    merged = hybrid[["doc_id", "y_pred"]].merge(
        reference[["doc_id", "y_true"]], on="doc_id", how="inner"
    )
    if len(merged):
        accuracy = float((merged["y_pred"] == merged["y_true"]).mean())
        if accuracy >= 0.995:
            checks.append(
                Check(
                    "plausible_accuracy",
                    WARN,
                    f"accuracy is {accuracy:.4f} on held-out data; verify that no "
                    "training document leaked into the evaluation set",
                )
            )
        else:
            checks.append(
                Check(
                    "plausible_accuracy",
                    PASS,
                    f"accuracy {accuracy:.4f} is in a plausible range",
                )
            )

    return checks


# Comparative validation
def compare_models(
    rf: pd.DataFrame,
    hybrid: pd.DataFrame,
    truth: pd.DataFrame,
    margin: float = config.HYBRID_SUPERIORITY_MARGIN,
    n_resamples: int = 2000,
) -> dict:
    """Score both models on the documents they share and apply the 5-point test."""
    paired = (
        truth[["doc_id", "language", "y_true"]]
        .merge(
            rf[["doc_id", "y_pred"]].rename(columns={"y_pred": "rf_pred"}), on="doc_id"
        )
        .merge(
            hybrid[["doc_id", "y_pred"]].rename(columns={"y_pred": "hybrid_pred"}),
            on="doc_id",
        )
    )
    if paired.empty:
        raise ValueError(
            "No documents are common to the corpus, the RF predictions, and the "
            "hybrid predictions. Check that both models use the same doc_id scheme."
        )

    rf_by_language = metrics.per_language_report(paired.rename(columns={"rf_pred": "y_pred"}))
    hybrid_by_language = metrics.per_language_report(
        paired.rename(columns={"hybrid_pred": "y_pred"})
    )

    comparison = rf_by_language[["language", "n", "accuracy", "macro_f1"]].merge(
        hybrid_by_language[["language", "accuracy", "macro_f1"]],
        on="language",
        suffixes=("_rf", "_hybrid"),
    )
    comparison["delta_macro_f1"] = (
        comparison["macro_f1_hybrid"] - comparison["macro_f1_rf"]
    )
    comparison["meets_margin"] = comparison["delta_macro_f1"] >= margin

    overall_rf = metrics.evaluate(paired["y_true"], paired["rf_pred"])
    overall_hybrid = metrics.evaluate(paired["y_true"], paired["hybrid_pred"])
    delta = overall_hybrid["macro_f1"] - overall_rf["macro_f1"]

    mcnemar = metrics.mcnemar_exact(
        paired["rf_pred"] == paired["y_true"],
        paired["hybrid_pred"] == paired["y_true"],
    )
    bootstrap = metrics.bootstrap_macro_f1_delta(
        paired["y_true"],
        paired["rf_pred"],
        paired["hybrid_pred"],
        n_resamples=n_resamples,
    )
    agreement = metrics.prediction_agreement(paired["rf_pred"], paired["hybrid_pred"])

    # Exclude the pooled ALL row when counting languages, or the pooled result
    language_rows = comparison[comparison["language"] != "ALL"]

    return {
        "n_documents": int(len(paired)),
        "n_languages": int(len(language_rows)),
        "overall_rf": overall_rf,
        "overall_hybrid": overall_hybrid,
        "delta_macro_f1": float(delta),
        "margin": margin,
        "meets_margin": bool(delta >= margin),
        "languages_meeting_margin": int(language_rows["meets_margin"].sum()),
        "mcnemar": mcnemar,
        "bootstrap": bootstrap,
        "agreement": agreement,
        "per_language": comparison,
        "rf_confusion": metrics.normalized_confusion_matrix(
            paired["y_true"], paired["rf_pred"]
        ),
        "hybrid_confusion": metrics.normalized_confusion_matrix(
            paired["y_true"], paired["hybrid_pred"]
        ),
        "paired": paired,
    }


def disagreement_examples(paired: pd.DataFrame, limit: int = 20) -> pd.DataFrame:
    """Documents where the hybrid is right and the RF is wrong, and vice versa."""
    rf_ok = paired["rf_pred"] == paired["y_true"]
    hybrid_ok = paired["hybrid_pred"] == paired["y_true"]
    interesting = paired[rf_ok != hybrid_ok].copy()
    if interesting.empty:
        return interesting

    hybrid_won = hybrid_ok[rf_ok != hybrid_ok].to_numpy()
    interesting["winner"] = np.where(hybrid_won, "hybrid", "random_forest")
    interesting["loser_error_distance"] = np.where(
        hybrid_won,
        (interesting["rf_pred"] - interesting["y_true"]).abs(),
        (interesting["hybrid_pred"] - interesting["y_true"]).abs(),
    )
    return interesting.sort_values(
        ["loser_error_distance", "language"], ascending=[False, True]
    ).head(limit)


# Reporting
def _format_checks(checks: list[Check]) -> str:
    lines = ["| Check | Result | Detail |", "| --- | --- | --- |"]
    lines += [f"| {c.name} | {c.status} | {c.detail} |" for c in checks]
    return "\n".join(lines)


def _to_markdown(df: pd.DataFrame, index: bool = False, floatfmt: str = ".4f") -> str:
    """Minimal DataFrame-to-Markdown renderer."""
    frame = df.reset_index() if index else df

    def cell(value) -> str:
        if isinstance(value, (float, np.floating)):
            return f"{value:{floatfmt}}"
        if isinstance(value, (bool, np.bool_)):
            return "yes" if value else "no"
        return str(value)

    header = "| " + " | ".join(str(c) for c in frame.columns) + " |"
    divider = "| " + " | ".join("---" for _ in frame.columns) + " |"
    rows = [
        "| " + " | ".join(cell(v) for v in row) + " |"
        for row in frame.itertuples(index=False, name=None)
    ]
    return "\n".join([header, divider, *rows])


def build_report(checks: list[Check], comparison: dict) -> str:
    """Render the validation result as Markdown, ready to paste into the thesis."""
    rf, hybrid = comparison["overall_rf"], comparison["overall_hybrid"]
    delta = comparison["delta_macro_f1"]
    boot, mcnemar = comparison["bootstrap"], comparison["mcnemar"]

    if comparison["meets_margin"]:
        verdict = (
            f"**MEETS the criterion.** KAYABASA exceeds the Random Forest baseline by "
            f"{delta:+.4f} Macro-F1, at or above the {comparison['margin']:.2f} margin "
            "required by Section 3.3.3."
        )
    else:
        verdict = (
            f"**DOES NOT meet the criterion.** KAYABASA differs from the Random Forest "
            f"baseline by {delta:+.4f} Macro-F1, short of the "
            f"{comparison['margin']:.2f} margin required by Section 3.3.3."
        )

    ci_note = (
        "The interval excludes zero, so the difference is unlikely to be resampling noise."
        if boot["ci_lower"] > 0 or boot["ci_upper"] < 0
        else "The interval spans zero, so the difference is within what document "
        "sampling alone could produce."
    )

    blocking = [c for c in checks if c.failed]
    integrity = (
        "No blocking issues found in the hybrid prediction file."
        if not blocking
        else f"**{len(blocking)} blocking issue(s) found. The comparison below is "
        "not trustworthy until they are resolved.**"
    )

    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    return f"""# KAYABASA hybrid output validation

Generated {generated_at} · seed {config.SEED} · {comparison['n_documents']} documents ·
{comparison['n_languages']} languages

## 1. Structural validation

{integrity}

{_format_checks(checks)}

## 2. Baseline comparison (Section 3.3.3)

{verdict}

| Metric | Random Forest | KAYABASA hybrid | Delta |
| --- | --- | --- | --- |
| Macro-F1 | {rf['macro_f1']:.4f} | {hybrid['macro_f1']:.4f} | {delta:+.4f} |
| Accuracy | {rf['accuracy']:.4f} | {hybrid['accuracy']:.4f} | {hybrid['accuracy'] - rf['accuracy']:+.4f} |
| L1 F1 | {rf['L1_f1']:.4f} | {hybrid['L1_f1']:.4f} | {hybrid['L1_f1'] - rf['L1_f1']:+.4f} |
| L2 F1 | {rf['L2_f1']:.4f} | {hybrid['L2_f1']:.4f} | {hybrid['L2_f1'] - rf['L2_f1']:+.4f} |
| L3 F1 | {rf['L3_f1']:.4f} | {hybrid['L3_f1']:.4f} | {hybrid['L3_f1'] - rf['L3_f1']:+.4f} |
| Adjacent error rate | {rf['adjacent_error_rate']:.4f} | {hybrid['adjacent_error_rate']:.4f} | {hybrid['adjacent_error_rate'] - rf['adjacent_error_rate']:+.4f} |
| Non-adjacent error rate | {rf['non_adjacent_error_rate']:.4f} | {hybrid['non_adjacent_error_rate']:.4f} | {hybrid['non_adjacent_error_rate'] - rf['non_adjacent_error_rate']:+.4f} |

Non-adjacent errors (L1 graded as L3 or the reverse) are the most harmful error
type for classroom use, so a hybrid that wins on Macro-F1 while making more of
them is not automatically the better model to deploy.

## 3. Is the difference real?

- Paired bootstrap ({boot['n_resamples']} resamples): Delta Macro-F1
  {boot['delta_macro_f1']:+.4f}, {boot['ci_level']:.0%} CI
  [{boot['ci_lower']:+.4f}, {boot['ci_upper']:+.4f}]. {ci_note}
- Exact McNemar test: {mcnemar['discordant']} discordant documents
  ({mcnemar['only_a_correct']} that only the Random Forest got right,
  {mcnemar['only_b_correct']} that only the hybrid got right),
  p = {mcnemar['p_value']:.4g}.
- Inter-model agreement: {comparison['agreement']['raw_agreement']:.4f} raw,
  Cohen's kappa {comparison['agreement']['cohens_kappa']:.4f}. Low agreement
  between two similarly-scoring models means they fail on different documents,
  which is the case where the transformer contributes something the handcrafted
  features cannot.

## 4. Per-language results (Section 3.1.6)

{_to_markdown(comparison['per_language'])}

Languages meeting the {comparison['margin']:.2f} margin individually:
{comparison['languages_meeting_margin']} of {comparison['n_languages']}.

## 5. Normalized confusion matrices (Section 3.3.4)

Random Forest baseline:

{_to_markdown(comparison['rf_confusion'], index=True)}

KAYABASA hybrid:

{_to_markdown(comparison['hybrid_confusion'], index=True)}
"""


def write_report(
    checks: list[Check],
    comparison: dict,
    out_dir: str | Path,
    stem: str = "validation_report",
) -> tuple[Path, Path]:
    """Write the Markdown report and a machine-readable JSON summary."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    md_path = out_dir / f"{stem}.md"
    md_path.write_text(build_report(checks, comparison), encoding="utf-8")

    payload = {
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "seed": config.SEED,
        "checks": [asdict(c) for c in checks],
        "blocking_failures": [c.name for c in checks if c.failed],
        "n_documents": comparison["n_documents"],
        "n_languages": comparison["n_languages"],
        "overall_rf": comparison["overall_rf"],
        "overall_hybrid": comparison["overall_hybrid"],
        "delta_macro_f1": comparison["delta_macro_f1"],
        "margin": comparison["margin"],
        "meets_margin": comparison["meets_margin"],
        "languages_meeting_margin": comparison["languages_meeting_margin"],
        "mcnemar": comparison["mcnemar"],
        "bootstrap": comparison["bootstrap"],
        "agreement": comparison["agreement"],
        "per_language": comparison["per_language"].to_dict(orient="records"),
    }
    json_path = out_dir / f"{stem}.json"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    return md_path, json_path
