"""Validate the KAYABASA hybrid output against the Random Forest baseline."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from kayabasa_rf import config, data_loading, validate_hybrid


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--hybrid-predictions",
        required=True,
        help="CSV exported by the KAYABASA hybrid model",
    )
    parser.add_argument(
        "--rf-predictions",
        default="results/rf_all_predictions.csv",
        help="Baseline predictions produced by run_baseline.py",
    )
    parser.add_argument(
        "--data-root",
        default="./corpora",
        help="Corpus root, used to recover the ground-truth labels",
    )
    parser.add_argument("--out", default="results", help="Where to write the report")
    parser.add_argument(
        "--margin",
        type=float,
        default=config.HYBRID_SUPERIORITY_MARGIN,
        help="Macro-F1 margin KAYABASA must clear (Section 3.3.3)",
    )
    parser.add_argument("--bootstrap-resamples", type=int, default=2000)
    parser.add_argument("--strict", action="store_true")
    parser.add_argument(
        "--already-clean",
        action="store_true",
        help="The corpus at --data-root is already denoised and normalized",
    )
    parser.add_argument(
        "--layout",
        choices=["auto", "repo", "tree"],
        default="auto",
        help="repo = the two GitHub repositories; tree = any folder structure",
    )
    return parser.parse_args()


def _require(path: str, what: str, remedy: str) -> None:
    """Fail with an instruction rather than a stack trace from inside pandas."""
    if not Path(path).exists():
        raise SystemExit(f"\n[error] {what} not found: {path}\n        {remedy}\n")


def main() -> int:
    args = parse_args()
    config.set_global_seed()

    # Check inputs before the corpus load, so a missing file is reported in a
    _require(
        args.rf_predictions,
        "Random Forest predictions",
        "Run the baseline first:\n"
        f"          python run_baseline.py --data-root {args.data_root}",
    )
    _require(
        args.hybrid_predictions,
        "Hybrid predictions",
        "Export the KAYABASA predictions to CSV (needs doc_id and y_pred columns),\n"
        "        or pass --hybrid-predictions with the correct path.",
    )

    # Ground truth comes from the corpus, never from the file under test: a
    df_all = data_loading.build_dataset(
        args.data_root,
        apply_denoising=True,
        strict=args.strict,
        already_clean=args.already_clean,
        layout=args.layout,
    )
    truth = df_all[["doc_id", "language", "label"]].rename(columns={"label": "y_true"})

    rf = validate_hybrid.load_predictions(args.rf_predictions, name="random forest")
    hybrid = validate_hybrid.load_predictions(
        args.hybrid_predictions, name="KAYABASA hybrid"
    )

    print(f"\n{'=' * 70}\nStructural validation\n{'=' * 70}")
    checks = validate_hybrid.structural_checks(hybrid, truth)
    for check in checks:
        print(f"  [{check.status}] {check.name}: {check.detail}")
    blocking = [c for c in checks if c.failed]

    print(f"\n{'=' * 70}\nBaseline comparison\n{'=' * 70}")
    comparison = validate_hybrid.compare_models(
        rf, hybrid, truth, margin=args.margin, n_resamples=args.bootstrap_resamples
    )

    print(
        f"  Random Forest macro-F1: {comparison['overall_rf']['macro_f1']:.4f}\n"
        f"  KAYABASA macro-F1:      {comparison['overall_hybrid']['macro_f1']:.4f}\n"
        f"  Delta:                  {comparison['delta_macro_f1']:+.4f} "
        f"(margin {args.margin:.2f})"
    )
    print(
        f"  Verdict: {'MEETS' if comparison['meets_margin'] else 'DOES NOT MEET'} "
        "the Section 3.3.3 criterion"
    )
    boot = comparison["bootstrap"]
    print(
        f"  95% CI on Delta: [{boot['ci_lower']:+.4f}, {boot['ci_upper']:+.4f}]   "
        f"McNemar p = {comparison['mcnemar']['p_value']:.4g}"
    )

    out_dir = Path(args.out)
    md_path, json_path = validate_hybrid.write_report(checks, comparison, out_dir)
    disagreements = validate_hybrid.disagreement_examples(comparison["paired"])
    if not disagreements.empty:
        disagreements.to_csv(out_dir / "model_disagreements.csv", index=False)
    print(f"\n[out] {md_path}\n[out] {json_path}")

    if blocking:
        print(
            f"\n[FAIL] {len(blocking)} structural check(s) failed: "
            f"{', '.join(c.name for c in blocking)}.\n"
            "       Fix these before reporting the comparison above."
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
