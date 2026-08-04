"""
split_datasets.py — Stage 3.1.6 Data Splitting (all languages).

Reads the unified normalized corpus (output_normalized/all_languages.txt — all
seven languages incl. Tagalog produced by clean_tagalog.py), assigns a split role
per language, and applies Stratified 5-Fold cross-validation to the HIGH-RESOURCE
training set (Tagalog, Cebuano, Bikol), stratified on the joint (language, label)
key under a fixed global seed.  The four BasahaCorpus LOW-RESOURCE languages
(Hiligaynon, Minasbate, Karay-a, Rinconada) are held entirely outside the CV loop
as the cross-lingual transfer test set (KayaBasa §3.1.6).

Row order matches extract_features.py's read of the same master, so doc_id is a
shared key across the feature matrix (output/all_features.csv) and these tables.

Outputs:
  splits/combined_corpus.csv      canonical doc table: doc_id, language, label, split_role
  splits/high_resource_index.csv  row-ordered HR docs (pos, doc_id, language, label)
                                  that the positional fold indices reference
  splits/low_resource_heldout.csv held-out transfer docs (doc_id, language, label)
  splits/fold_indices.pkl         list[ {'train':[...], 'val':[...]} ] x5
"""

import os
import random
import pickle

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

SEED = 42
HIGH_RESOURCE = {"tagalog", "cebuano", "bikol"}
NORM_DIR = "output_normalized"
NORM_MASTER = os.path.join(NORM_DIR, "all_languages.txt")
SPLIT_DIR = "splits"
SEP = "|"


def _load_combined() -> pd.DataFrame:
    """Read the unified normalized master (all 7 languages, incl. Tagalog produced
    by clean_tagalog.py) into doc_id / language / label / text / split_role.  Row
    order matches extract_features.py's read of the same file, so doc_id is shared
    across the feature matrix and the split tables."""
    rows = []
    for line in open(NORM_MASTER, encoding="utf-8").read().splitlines()[1:]:
        if not line.strip():
            continue
        lang, level, text = line.split(SEP, 2)
        rows.append({"language": lang, "label": int(level), "text": text})

    df = pd.DataFrame(rows)
    df.insert(0, "doc_id", range(len(df)))
    df["split_role"] = df["language"].apply(
        lambda l: "high_resource" if l in HIGH_RESOURCE else "low_resource")
    return df


def main():
    random.seed(SEED)
    np.random.seed(SEED)
    try:                                            # spec fixes torch seed too
        import torch
        torch.manual_seed(SEED)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(SEED)
    except Exception:
        pass

    df = _load_combined()
    os.makedirs(SPLIT_DIR, exist_ok=True)
    df[["doc_id", "language", "label", "split_role"]].to_csv(
        os.path.join(SPLIT_DIR, "combined_corpus.csv"), index=False)

    # ── Stratified 5-Fold on the high-resource set (stratify on language+label) ─
    df_train = df[df["split_role"] == "high_resource"].reset_index(drop=True)
    df_train["stratify_key"] = df_train["language"] + "_" + df_train["label"].astype(str)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    fold_indices = [{"train": tr.tolist(), "val": va.tolist()}
                    for tr, va in skf.split(df_train, df_train["stratify_key"])]
    with open(os.path.join(SPLIT_DIR, "fold_indices.pkl"), "wb") as f:
        pickle.dump(fold_indices, f)

    # tables the positional indices map back to
    (df_train.reset_index().rename(columns={"index": "pos"})
        [["pos", "doc_id", "language", "label"]]
        .to_csv(os.path.join(SPLIT_DIR, "high_resource_index.csv"), index=False))
    (df[df["split_role"] == "low_resource"][["doc_id", "language", "label"]]
        .to_csv(os.path.join(SPLIT_DIR, "low_resource_heldout.csv"), index=False))

    # ── Report + integrity checks ──────────────────────────────────────────────
    print(f"Combined corpus: {len(df)} docs across {df['language'].nunique()} languages")
    print(f"  HIGH-RESOURCE (5-fold CV): {len(df_train)} docs")
    print(df_train.groupby(["language", "label"]).size().to_string())
    print(f"  LOW-RESOURCE (held-out transfer): "
          f"{(df['split_role'] == 'low_resource').sum()} docs")
    print(df[df["split_role"] == "low_resource"].groupby("language").size().to_string())
    print("\nFold sizes (train/val):",
          [(len(f["train"]), len(f["val"])) for f in fold_indices])

    seen = sorted(i for f in fold_indices for i in f["val"])
    assert seen == list(range(len(df_train))), "validation coverage broken"
    assert all(set(f["train"]).isdisjoint(f["val"]) for f in fold_indices), "train/val overlap"
    print("Integrity OK: every HR doc validates exactly once; no train/val overlap")


if __name__ == "__main__":
    main()
