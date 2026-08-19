"""Stratified 5-fold splits (proposal Section 3.1.6)."""

from __future__ import annotations

import pickle
from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedKFold

from . import config


def build_fold_indices(
    df_train: pd.DataFrame,
    n_splits: int = config.N_SPLITS,
    seed: int = config.SEED,
) -> list[dict[str, list[int]]]:
    """Stratify on the joint (language, label) key."""
    stratify_key = df_train["language"] + "_" + df_train["label"].astype(str)

    counts = stratify_key.value_counts()
    too_rare = counts[counts < n_splits]
    if not too_rare.empty:
        raise ValueError(
            f"These (language, label) strata have fewer than {n_splits} documents "
            f"and cannot be split {n_splits} ways: {too_rare.to_dict()}"
        )

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return [
        {"train": train_idx.tolist(), "val": val_idx.tolist()}
        for train_idx, val_idx in skf.split(df_train, stratify_key)
    ]


def get_or_create_folds(
    df_train: pd.DataFrame,
    splits_path: str | Path,
    n_splits: int = config.N_SPLITS,
    seed: int = config.SEED,
    force_regenerate: bool = False,
) -> list[dict[str, list[int]]]:
    """Load saved folds, or create and persist them on first use."""
    splits_path = Path(splits_path)

    if splits_path.exists() and not force_regenerate:
        with open(splits_path, "rb") as f:
            folds = pickle.load(f)
        n_docs = sum(len(fold["val"]) for fold in folds)
        if n_docs != len(df_train):
            raise ValueError(
                f"{splits_path} was built for {n_docs} documents but the current "
                f"training set has {len(df_train)}. Delete the file to regenerate, "
                "but note that every previously reported result used the old splits."
            )
        print(f"[splits] reusing saved folds from {splits_path}")
        return folds

    folds = build_fold_indices(df_train, n_splits=n_splits, seed=seed)
    splits_path.parent.mkdir(parents=True, exist_ok=True)
    with open(splits_path, "wb") as f:
        pickle.dump(folds, f)
    print(f"[splits] generated {n_splits} folds (seed={seed}) -> {splits_path}")
    return folds


def describe_folds(
    df_train: pd.DataFrame, folds: list[dict[str, list[int]]]
) -> pd.DataFrame:
    """Per-fold, per-language document counts, i.e. the proposal's Table VIII."""
    rows = []
    for i, fold in enumerate(folds, start=1):
        train_langs = df_train.iloc[fold["train"]]["language"].value_counts()
        val_langs = df_train.iloc[fold["val"]]["language"].value_counts()
        for language in sorted(df_train["language"].unique()):
            rows.append(
                {
                    "fold": i,
                    "language": language,
                    "train_docs": int(train_langs.get(language, 0)),
                    "val_docs": int(val_langs.get(language, 0)),
                }
            )
    return pd.DataFrame(rows)
