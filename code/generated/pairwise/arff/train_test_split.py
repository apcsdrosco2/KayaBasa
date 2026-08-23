"""
train_test_split.py — Split-then-stack for bilingual readability experiments.

Strategy:
  1. Split each language into train/test (80/20, stratified by grade level)
  2. For each language pair, stack the train splits into a bilingual train set
  3. Keep each language's test split as-is for individual evaluation

This enables comparing:
  - Monolingual baseline: train on lang_train → test on lang_test
  - Bilingual:           train on (langA_train + langB_train) → test on lang_test

If bilingual outperforms monolingual on the same test set, shared linguistic
knowledge is helping the model generalize.

Feature sets produced for each split:
  - trad           (18 features: traditional + syllable)
  - trad_clgsngo   (24 features: trad + cross-lingual ngrams)
  - mbert          (768 features: mBERT embeddings)
  - all            (792 features: trad + clgsngo + mbert)
  - xlmr           (768 features: XLM-R embeddings)
  - all_xlmr       (792 features: trad + clgsngo + xlmr)

Output structure:
  pairwise/arff/splits/
    monolingual/
      {lang}_train_{featureset}.arff/.csv
      {lang}_test_{featureset}.arff/.csv
    bilingual/
      paired_{langA}_{langB}_train_{featureset}.arff/.csv
      (test sets are the same monolingual test sets)

Uses a fixed random seed for reproducibility.
"""

import io
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split

SEED = 42
TEST_SIZE = 0.2

SCRIPT_DIR = Path(__file__).resolve().parent
GENERATED_DIR = SCRIPT_DIR.parent.parent  # code/generated/
OUTPUT_DIR = SCRIPT_DIR / "splits"
MONO_DIR = OUTPUT_DIR / "monolingual"
BI_DIR = OUTPUT_DIR / "bilingual"

# Language config
LANG_CONFIG = {
    "bikol": "bikol docus",
    "cebuano": "ceb docus",
    "tagalog": "tag docus",
}

# Language pairs
PAIRS = [
    ("bikol", "tagalog"),
    ("cebuano", "bikol"),
    ("tagalog", "cebuano"),
]

# Feature sets produced for every split (see module docstring)
FEATURE_SETS = ["trad", "trad_clgsngo", "mbert", "all", "xlmr", "all_xlmr"]


# ═══════════════════════════════════════════════════════════════════════════════
# ARFF I/O
# ═══════════════════════════════════════════════════════════════════════════════

def load_arff(arff_path: Path) -> pd.DataFrame:
    """Parse a Weka ARFF file into a DataFrame."""
    with open(arff_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    columns = []
    data_start = 0

    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.upper().startswith("@ATTRIBUTE"):
            col_name = stripped.split()[1]
            columns.append(col_name)
        elif stripped.upper().startswith("@DATA"):
            data_start = i + 1
            break

    data_lines = [l.strip() for l in lines[data_start:] if l.strip()]
    df = pd.read_csv(io.StringIO("\n".join(data_lines)), header=None, names=columns)
    return df


def save_arff(df: pd.DataFrame, relation_name: str, output_path: Path):
    """Write a DataFrame to ARFF format."""
    labels = sorted(df["class"].astype(str).unique().tolist())
    labels_str = "{" + ",".join(labels) + "}"

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(f"@RELATION {relation_name}\n\n")

        for col in df.columns[:-1]:
            f.write(f"@ATTRIBUTE {col} NUMERIC\n")

        f.write(f"@ATTRIBUTE class {labels_str}\n\n")

        f.write("@DATA\n")
        for row in df.itertuples(index=False, name=None):
            values = []
            for value in row[:-1]:
                if pd.isna(value) or value == "?":
                    values.append("?")
                else:
                    values.append(f"{float(value):.17g}")
            values.append(str(row[-1]))
            f.write(",".join(values) + "\n")


def save_both(df: pd.DataFrame, relation_name: str, output_dir: Path, filename: str):
    """Save as both CSV and ARFF."""
    df.to_csv(output_dir / f"{filename}.csv", index=False)
    save_arff(df, relation_name, output_dir / f"{filename}.arff")


# ═══════════════════════════════════════════════════════════════════════════════
# Feature loading
# ═══════════════════════════════════════════════════════════════════════════════

def load_language_features(lang: str) -> dict:
    """Load all feature sets for a language from arff folder.

    Returns dict with keys: trad, trad_clgsngo, mbert, all, xlmr, all_xlmr
    Each value is a DataFrame with features + 'class' column.
    """
    folder = LANG_CONFIG[lang]
    arff_dir = GENERATED_DIR / folder / "arff"

    # Load each feature set
    df_trad = load_arff(arff_dir / f"{lang}_trad.arff")
    df_trad_clg = load_arff(arff_dir / f"{lang}_trad_clgsngo.arff")
    df_mbert = load_arff(arff_dir / f"{lang}_mbert.arff")
    df_xlmr = load_arff(arff_dir / f"{lang}_xlmr.arff")

    # Build "all" = trad_clgsngo features + mbert features + class
    mbert_features = df_mbert.drop(columns=["class"])
    trad_clg_features = df_trad_clg.drop(columns=["class"])
    class_col = df_trad[["class"]]
    df_all = pd.concat([trad_clg_features, mbert_features, class_col], axis=1)

    # "all_xlmr" = trad_clgsngo features + xlmr features + class.
    # A pre-built version of this already ships as {lang}_trad_clgsngo_xlmr.arff,
    # so load it directly rather than re-concatenating (avoids any risk of the
    # two source files being in different row orders).
    df_all_xlmr = load_arff(arff_dir / f"{lang}_trad_clgsngo_xlmr.arff")

    return {
        "trad": df_trad,
        "trad_clgsngo": df_trad_clg,
        "mbert": df_mbert,
        "all": df_all,
        "xlmr": df_xlmr,
        "all_xlmr": df_all_xlmr,
    }


def main():
    """Run the single stratified 80/20 split-then-stack pipeline (writes to disk).

    Guarded behind __main__ so this module's helpers (load_arff, save_both,
    load_language_features, etc.) and constants (LANG_CONFIG, PAIRS,
    FEATURE_SETS, SEED, ...) can be imported elsewhere — e.g. by
    train_test_split_cv.py — without re-running this single-split pipeline as
    an import side effect.
    """
    MONO_DIR.mkdir(parents=True, exist_ok=True)
    BI_DIR.mkdir(parents=True, exist_ok=True)

    # ═══════════════════════════════════════════════════════════════════════════
    # Step 1: Split each language (stratified by class)
    # ═══════════════════════════════════════════════════════════════════════════

    print("=" * 60)
    print("STEP 1: Stratified train/test split per language")
    print("=" * 60)

    # Store split indices per language (same indices used across all feature sets)
    split_indices = {}

    for lang in LANG_CONFIG:
        features = load_language_features(lang)
        n = len(features["trad"])
        labels = features["trad"]["class"].astype(str)

        # Get stratified split indices
        indices = np.arange(n)
        train_idx, test_idx = train_test_split(
            indices, test_size=TEST_SIZE, random_state=SEED, stratify=labels
        )
        split_indices[lang] = (train_idx, test_idx)

        print(f"\n  {lang.upper()}: {n} total → {len(train_idx)} train, {len(test_idx)} test")

        # Save monolingual train/test for each feature set
        for feat_name, df in features.items():
            train_df = df.iloc[train_idx].reset_index(drop=True)
            test_df = df.iloc[test_idx].reset_index(drop=True)

            save_both(train_df, f"{lang}_train_{feat_name}", MONO_DIR, f"{lang}_train_{feat_name}")
            save_both(test_df, f"{lang}_test_{feat_name}", MONO_DIR, f"{lang}_test_{feat_name}")

        # Print class distribution
        train_labels = labels.iloc[train_idx]
        test_labels = labels.iloc[test_idx]
        print(f"    Train class dist: {dict(train_labels.value_counts().sort_index())}")
        print(f"    Test class dist:  {dict(test_labels.value_counts().sort_index())}")

    print(f"\n  Monolingual splits saved to: {MONO_DIR}")

    # ═══════════════════════════════════════════════════════════════════════════
    # Step 2: Stack train splits for each language pair
    # ═══════════════════════════════════════════════════════════════════════════

    print("\n" + "=" * 60)
    print("STEP 2: Stack bilingual train sets (pairwise)")
    print("=" * 60)

    for lang1, lang2 in PAIRS:
        pair_name = f"{lang1}_{lang2}"
        print(f"\n  Pair: {lang1.upper()} & {lang2.upper()}")

        features1 = load_language_features(lang1)
        features2 = load_language_features(lang2)

        train_idx1 = split_indices[lang1][0]
        train_idx2 = split_indices[lang2][0]

        for feat_name in FEATURE_SETS:
            # Stack only the TRAIN portions from each language
            train1 = features1[feat_name].iloc[train_idx1].reset_index(drop=True)
            train2 = features2[feat_name].iloc[train_idx2].reset_index(drop=True)
            stacked_train = pd.concat([train1, train2], axis=0, ignore_index=True)

            filename = f"paired_{pair_name}_train_{feat_name}"
            save_both(stacked_train, filename, BI_DIR, filename)

        # Report sizes
        n_train = len(features1[feat_name].iloc[train_idx1]) + len(features2[feat_name].iloc[train_idx2])
        print(f"    Bilingual train size: {n_train}")
        print(f"    Test: use monolingual/{lang1}_test_* and monolingual/{lang2}_test_*")

    print(f"\n  Bilingual train sets saved to: {BI_DIR}")

    # ═══════════════════════════════════════════════════════════════════════════
    # Summary
    # ═══════════════════════════════════════════════════════════════════════════

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"""
Experimental setup for each pair (e.g., bikol & tagalog):

  Monolingual baseline:
    Train: monolingual/bikol_train_{{featureset}}.arff
    Test:  monolingual/bikol_test_{{featureset}}.arff

  Bilingual experiment:
    Train: bilingual/paired_bikol_tagalog_train_{{featureset}}.arff
    Test:  monolingual/bikol_test_{{featureset}}.arff  (same test set!)

  Compare accuracy/F1 between the two conditions.
  If bilingual > monolingual, shared linguistic knowledge helps.

Feature sets: {", ".join(FEATURE_SETS)}
Split: {int((1-TEST_SIZE)*100)}/{int(TEST_SIZE*100)} train/test, stratified, seed={SEED}
""")


if __name__ == "__main__":
    main()
