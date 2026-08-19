"""The Random Forest baseline itself (proposal Table XIII, Sections 3.1.5-3.3.3)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold

from . import config, metrics
from .features import FEATURE_NAMES, FeatureExtractor

PROBA_COLUMNS = [f"proba_{name}" for name in metrics.LABEL_NAMES]


# Feature plumbing
def _feature_matrix(features: pd.DataFrame, doc_ids: pd.Series) -> np.ndarray:
    """Align a feature frame to the given document order."""
    indexed = features.set_index("doc_id")
    missing = set(doc_ids) - set(indexed.index)
    if missing:
        raise KeyError(
            f"{len(missing)} documents have no features, e.g. {sorted(missing)[:3]}"
        )
    return indexed.loc[doc_ids, FEATURE_NAMES].to_numpy(dtype=float)


def _fold_features(
    df_train_fold: pd.DataFrame,
    df_apply: pd.DataFrame,
    anchor_language: str,
) -> pd.DataFrame:
    """Fit the CROSSNGO anchor profile on training-fold documents only."""
    anchor_texts = df_train_fold.loc[
        df_train_fold["language"] == anchor_language, "text"
    ].tolist()
    if not anchor_texts:
        anchor_texts = df_train_fold["text"].tolist()
    extractor = FeatureExtractor().fit(anchor_texts)
    return extractor.transform(df_apply)


# Model
def build_model(params: config.RFParams | None = None) -> RandomForestClassifier:
    params = params or config.RFParams()
    return RandomForestClassifier(**params.to_sklearn())


def _tune_within_fold(
    X_train: np.ndarray,
    y_train: np.ndarray,
    params: config.RFParams,
    seed: int = config.SEED,
) -> RandomForestClassifier:
    """Grid search using an inner CV over the training fold only."""
    inner = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
    search = GridSearchCV(
        build_model(params),
        param_grid=config.RF_GRID,
        scoring="f1_macro",
        cv=inner,
        n_jobs=params.n_jobs,
        refit=True,
    )
    search.fit(X_train, y_train)
    return search.best_estimator_


def _predict_frame(
    model: RandomForestClassifier, X: np.ndarray, df_slice: pd.DataFrame
) -> pd.DataFrame:
    proba = model.predict_proba(X)
    out = pd.DataFrame(
        {
            "doc_id": df_slice["doc_id"].to_numpy(),
            "language": df_slice["language"].to_numpy(),
            "split_role": df_slice["split_role"].to_numpy(),
            "y_true": df_slice["label"].to_numpy(),
            "y_pred": model.predict(X),
        }
    )
    # Reindex onto the canonical L1/L2/L3 order: a class missing from a training
    classes = list(model.classes_)
    for i, label in enumerate(config.LABELS):
        column = PROBA_COLUMNS[i]
        out[column] = proba[:, classes.index(label)] if label in classes else 0.0
    return out


# Phase 1: cross-validation
def run_cross_validation(
    df_train: pd.DataFrame,
    folds: list[dict[str, list[int]]],
    features: pd.DataFrame | None = None,
    params: config.RFParams | None = None,
    anchor_from_train_only: bool = False,
    tune: bool = False,
    anchor_language: str = config.ANCHOR_LANGUAGE,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run the 5-fold CV; returns (fold_metrics, per_language_metrics, oof_predictions)."""
    params = params or config.RFParams()
    if features is None and not anchor_from_train_only:
        raise ValueError(
            "Pass a precomputed feature frame, or set anchor_from_train_only=True "
            "to compute features inside each fold."
        )

    fold_rows, language_rows, oof_frames = [], [], []

    for fold_number, fold in enumerate(folds, start=1):
        train_slice = df_train.iloc[fold["train"]]
        val_slice = df_train.iloc[fold["val"]]

        fold_features = (
            _fold_features(train_slice, df_train, anchor_language)
            if anchor_from_train_only
            else features
        )

        X_train = _feature_matrix(fold_features, train_slice["doc_id"])
        X_val = _feature_matrix(fold_features, val_slice["doc_id"])
        y_train = train_slice["label"].to_numpy()
        y_val = val_slice["label"].to_numpy()

        model = (
            _tune_within_fold(X_train, y_train, params)
            if tune
            else build_model(params).fit(X_train, y_train)
        )

        predictions = _predict_frame(model, X_val, val_slice)
        predictions.insert(3, "fold", fold_number)
        oof_frames.append(predictions)

        fold_rows.append(
            {"fold": fold_number, **metrics.evaluate(y_val, predictions["y_pred"])}
        )
        for language, group in predictions.groupby("language"):
            language_rows.append(
                {
                    "fold": fold_number,
                    "language": language,
                    **metrics.evaluate(group["y_true"], group["y_pred"]),
                }
            )

        print(
            f"[cv] fold {fold_number}/{len(folds)}  "
            f"macro-F1={fold_rows[-1]['macro_f1']:.4f}  "
            f"acc={fold_rows[-1]['accuracy']:.4f}  "
            f"non-adjacent={fold_rows[-1]['non_adjacent_error_rate']:.4f}"
        )

    fold_metrics = pd.DataFrame(fold_rows)
    language_metrics = pd.DataFrame(language_rows)
    oof = pd.concat(oof_frames, ignore_index=True)

    print(
        f"[cv] macro-F1 across {len(folds)} folds: "
        f"{fold_metrics['macro_f1'].mean():.4f} "
        f"+/- {fold_metrics['macro_f1'].std(ddof=1):.4f}"
    )
    return fold_metrics, language_metrics, oof


# Phase 2: cross-lingual transfer
def train_final_model(
    df_train: pd.DataFrame,
    features: pd.DataFrame,
    params: config.RFParams | None = None,
) -> RandomForestClassifier:
    """Fit on the full high-resource set, for Phase 2 transfer evaluation."""
    X = _feature_matrix(features, df_train["doc_id"])
    y = df_train["label"].to_numpy()
    model = build_model(params).fit(X, y)
    print(f"[final] trained on {len(df_train)} high-resource documents")
    return model


def evaluate_cross_lingual(
    model: RandomForestClassifier,
    df_low: pd.DataFrame,
    features: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Predict the held-out low-resource languages (Section 3.3.2, Phase 2)."""
    if df_low.empty:
        return pd.DataFrame(), pd.DataFrame()
    X = _feature_matrix(features, df_low["doc_id"])
    predictions = _predict_frame(model, X, df_low)
    return predictions, metrics.per_language_report(predictions)


# Interpretability
def feature_importances(model: RandomForestClassifier) -> pd.DataFrame:
    """Impurity-based importances, sorted."""
    return (
        pd.DataFrame(
            {"feature": FEATURE_NAMES, "importance": model.feature_importances_}
        )
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )


# Section 3.1.5: denoising impact
def denoising_impact(
    noisy_language_metrics: pd.DataFrame,
    denoised_language_metrics: pd.DataFrame,
) -> tuple[pd.DataFrame, dict]:
    """Delta Macro-F1 = denoised - noisy, per language, with the proposal's verdict."""
    noisy = (
        noisy_language_metrics.groupby("language")["macro_f1"]
        .agg(["mean", "std"])
        .add_prefix("noisy_")
    )
    denoised = (
        denoised_language_metrics.groupby("language")["macro_f1"]
        .agg(["mean", "std"])
        .add_prefix("denoised_")
    )
    table = noisy.join(denoised, how="outer")
    table["delta_macro_f1"] = table["denoised_mean"] - table["noisy_mean"]
    table["meets_threshold"] = (
        table["delta_macro_f1"] >= config.DENOISING_DELTA_THRESHOLD
    )
    table = table.reset_index()

    n_passing = int(table["meets_threshold"].sum())
    verdict = {
        "threshold": config.DENOISING_DELTA_THRESHOLD,
        "languages_meeting_threshold": n_passing,
        "languages_required": config.DENOISING_MIN_LANGUAGES,
        "confirmed": bool(n_passing >= config.DENOISING_MIN_LANGUAGES),
        "mean_delta_macro_f1": float(table["delta_macro_f1"].mean()),
    }
    return table, verdict


# Persistence
def save_model(model: RandomForestClassifier, path: str | Path) -> None:
    import joblib

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "feature_names": FEATURE_NAMES}, path)
    print(f"[final] saved model -> {path}")


def load_model(path: str | Path) -> RandomForestClassifier:
    import joblib

    bundle = joblib.load(path)
    if bundle["feature_names"] != FEATURE_NAMES:
        raise ValueError(
            "The saved model expects different features than the current code:\n"
            f"  saved:   {bundle['feature_names']}\n"
            f"  current: {FEATURE_NAMES}"
        )
    return bundle["model"]
