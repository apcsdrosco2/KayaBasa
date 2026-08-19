"""Train and evaluate the Random Forest baseline (proposal Table XIII)."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from kayabasa_rf import config, data_loading, features, metrics, rf_baseline, splits


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        default="./corpora",
        help="Directory holding ara-close-lang/ and BasahaCorpus-.../",
    )
    parser.add_argument("--out", default=".", help="Where to write artifacts")
    parser.add_argument(
        "--condition",
        choices=["denoised", "noisy"],
        default="denoised",
        help="denoised = Configuration B; noisy = Configuration A (Table XII)",
    )
    parser.add_argument(
        "--already-clean",
        action="store_true",
        help="The text in --data-root is already denoised and normalized, so "
        "skip the cleaning step instead of applying it a second time",
    )
    parser.add_argument(
        "--layout",
        choices=["auto", "repo", "tree"],
        default="auto",
        help="repo = the two GitHub repositories; tree = any folder structure, "
        "with language and level inferred from the path",
    )
    parser.add_argument(
        "--denoising-impact",
        action="store_true",
        help="Run both conditions and report Delta Macro-F1 (Section 3.1.5)",
    )
    parser.add_argument(
        "--raw-root",
        help="For --denoising-impact: the team's raw folder (noisy arm). "
        "Defaults to deriving the noisy arm from --data-root.",
    )
    parser.add_argument(
        "--clean-root",
        help="For --denoising-impact: the team's cleaned folder (denoised arm)",
    )
    parser.add_argument(
        "--tune",
        action="store_true",
        help="Grid-search RF hyperparameters inside each training fold",
    )
    parser.add_argument(
        "--anchor-from-train-only",
        action="store_true",
        help="Rebuild the CROSSNGO anchor profile per fold (no cross-split leakage)",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail instead of warning when a corpus path is missing",
    )
    return parser.parse_args()


def run_condition(
    data_root: str,
    paths: config.Paths,
    apply_denoising: bool,
    args: argparse.Namespace,
    save_artifacts: bool,
    already_clean: bool = False,
) -> dict:
    """One full pass: load, extract, cross-validate, transfer."""
    label = "denoised" if apply_denoising or already_clean else "noisy"
    print(f"\n{'=' * 70}\nCondition: {label}\n{'=' * 70}")

    df_all = data_loading.build_dataset(
        data_root,
        apply_denoising=apply_denoising,
        strict=args.strict,
        already_clean=already_clean,
        layout=args.layout,
        cache_path=paths.data / f"df_all_{label}.pkl" if save_artifacts else None,
    )

    feature_matrix = features.extract_features(
        df_all,
        out_csv=paths.features / f"all_features_{label}.csv"
        if save_artifacts
        else None,
    )

    df_high = df_all[df_all["split_role"] == "high_resource"].reset_index(drop=True)
    df_low = df_all[df_all["split_role"] == "low_resource"].reset_index(drop=True)
    if df_high.empty:
        raise ValueError(
            "No high-resource documents were loaded, so there is nothing to train on."
        )

    # The same saved folds are reused across conditions and across every model in
    folds = splits.get_or_create_folds(df_high, paths.splits / "fold_indices.pkl")
    if save_artifacts:
        splits.describe_folds(df_high, folds).to_csv(
            paths.results / "fold_composition.csv", index=False
        )

    fold_metrics, language_metrics, oof = rf_baseline.run_cross_validation(
        df_high,
        folds,
        features=feature_matrix,
        anchor_from_train_only=args.anchor_from_train_only,
        tune=args.tune,
    )

    model = rf_baseline.train_final_model(df_high, feature_matrix)
    cross_lingual, cross_lingual_report = rf_baseline.evaluate_cross_lingual(
        model, df_low, feature_matrix
    )

    if cross_lingual.empty:
        print("[phase2] no low-resource documents found; skipping transfer evaluation")
    else:
        pooled = cross_lingual_report[cross_lingual_report["language"] == "ALL"].iloc[0]
        print(
            f"[phase2] cross-lingual transfer macro-F1={pooled['macro_f1']:.4f} "
            f"acc={pooled['accuracy']:.4f} over {int(pooled['n'])} documents"
        )

    result = {
        "condition": label,
        "df_all": df_all,
        "features": feature_matrix,
        "fold_metrics": fold_metrics,
        "language_metrics": language_metrics,
        "oof": oof,
        "model": model,
        "cross_lingual": cross_lingual,
        "cross_lingual_report": cross_lingual_report,
    }

    if save_artifacts:
        _save(result, paths, model)
    return result


def _save(result: dict, paths: config.Paths, model) -> None:
    r = paths.results
    result["fold_metrics"].to_csv(r / "rf_fold_metrics.csv", index=False)
    metrics.fold_summary(result["fold_metrics"]).to_csv(
        r / "rf_fold_summary.csv", index=False
    )
    metrics.fold_summary(result["language_metrics"], by=["language"]).to_csv(
        r / "rf_language_summary.csv", index=False
    )
    result["language_metrics"].to_csv(r / "rf_language_metrics.csv", index=False)
    result["oof"].to_csv(r / "rf_oof_predictions.csv", index=False)

    # Out-of-fold predictions on the three training languages plus held-out
    all_predictions = [result["oof"].drop(columns=["fold"])]
    if not result["cross_lingual"].empty:
        all_predictions.append(result["cross_lingual"])
        result["cross_lingual"].to_csv(
            r / "rf_crosslingual_predictions.csv", index=False
        )
        result["cross_lingual_report"].to_csv(
            r / "rf_crosslingual_report.csv", index=False
        )
    combined = pd.concat(all_predictions, ignore_index=True)
    combined.to_csv(r / "rf_all_predictions.csv", index=False)

    metrics.per_language_report(combined).to_csv(
        r / "rf_seven_language_report.csv", index=False
    )
    metrics.normalized_confusion_matrix(combined["y_true"], combined["y_pred"]).to_csv(
        r / "rf_confusion_matrix.csv"
    )
    rf_baseline.feature_importances(model).to_csv(
        r / "rf_feature_importances.csv", index=False
    )
    rf_baseline.save_model(model, paths.models / "rf_baseline.joblib")
    print(f"[out] wrote baseline artifacts to {r}")


def main() -> None:
    args = parse_args()
    config.set_global_seed()
    paths = config.Paths(root=Path(args.out)).ensure()

    if args.denoising_impact:
        # Two sources for the two arms. If the team supplies its own raw/ and
        noisy_root = args.raw_root or args.data_root
        clean_root = args.clean_root or args.data_root
        team_cleaned = args.clean_root is not None

        # Only the denoised pass keeps its artifacts, since that is the
        noisy = run_condition(noisy_root, paths, False, args, save_artifacts=False)
        denoised = run_condition(
            clean_root,
            paths,
            apply_denoising=not team_cleaned,
            args=args,
            save_artifacts=True,
            already_clean=team_cleaned,
        )

        table, verdict = rf_baseline.denoising_impact(
            noisy["language_metrics"], denoised["language_metrics"]
        )
        table.to_csv(paths.results / "denoising_impact.csv", index=False)

        print(f"\n{'=' * 70}\nDenoising impact (Section 3.1.5)\n{'=' * 70}")
        print(table.to_string(index=False))
        status = "CONFIRMED" if verdict["confirmed"] else "NOT CONFIRMED"
        print(
            f"\nDelta Macro-F1 >= {verdict['threshold']:.2f} in "
            f"{verdict['languages_meeting_threshold']} of the high-resource "
            f"languages (need {verdict['languages_required']}): {status}"
        )
        return

    run_condition(
        args.data_root,
        paths,
        apply_denoising=args.condition == "denoised",
        args=args,
        save_artifacts=True,
        already_clean=args.already_clean,
    )
    print(
        "\nNext: run the hybrid, export its predictions, then\n"
        "  python run_validation.py --hybrid-predictions <path/to/predictions.csv>"
    )


if __name__ == "__main__":
    main()
