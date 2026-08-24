"""
train_test_split.py — Split-then-stack for bilingual readability experiments.
Includes XLM-R embeddings alongside mBERT.

Strategy:
  1. Split each language into train/test (80/20, stratified by grade level)
  2. For each language pair, stack the train splits into a bilingual train set
  3. Keep each language's test split as-is for individual evaluation

Feature sets:
  - trad              (18 features: traditional + syllable)
  - trad_clgsngo      (24 features: trad + cross-lingual ngrams)
  - mbert             (768 features: mBERT embeddings)
  - xlmr              (768 features: XLM-R embeddings)
  - all_mbert         (792 features: trad + clgsngo + mbert)
  - all_xlmr          (792 features: trad + clgsngo + xlmr)

Output structure:
  splits/
    monolingual/
      {lang}_train_{featureset}.arff/.csv
      {lang}_test_{featureset}.arff/.csv
    bilingual/
      paired_{langA}_{langB}_train_{featureset}.arff/.csv
"""

import io
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split

SEED = 42

# 80/20 stratified split
TEST_SIZES = {
    "bikol": 30,       # 20% of 150
    "tagalog": 53,     # 20% of 265
    "cebuano": 70,     # 20% of 349
}

SCRIPT_DIR = Path(__file__).resolve().parent
GENERATED_DIR = SCRIPT_DIR.parent  # code/generated/
OUTPUT_DIR = SCRIPT_DIR / "splits"
MONO_DIR = OUTPUT_DIR / "monolingual"
BI_DIR = OUTPUT_DIR / "bilingual"

MONO_DIR.mkdir(parents=True, exist_ok=True)
BI_DIR.mkdir(parents=True, exist_ok=True)

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

# Feature sets: name -> arff filename pattern
# {lang} will be replaced with language name
FEATURE_SETS = {
    "trad": "{lang}_trad.arff",
    "trad_clgsngo": "{lang}_trad_clgsngo.arff",
    "mbert": "{lang}_mbert.arff",
    "xlmr": "{lang}_xlmr.arff",
    "all_mbert": "{lang}_trad_clgsngo_mbert.arff",
    "all_xlmr": "{lang}_trad_clgsngo_xlmr.arff",
}

# Special case: bikol has a typo in the mbert all file
BIKOL_ALL_MBERT = "bikol_trad_cglsngo_mbert.arff"


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

def get_arff_path(lang: str, feat_name: str) -> Path:
    """Get the ARFF file path for a language and feature set."""
    folder = LANG_CONFIG[lang]
    arff_dir = GENERATED_DIR / folder / "arff"

    # Handle bikol typo for all_mbert
    if feat_name == "all_mbert" and lang == "bikol":
        return arff_dir / BIKOL_ALL_MBERT

    filename = FEATURE_SETS[feat_name].format(lang=lang)
    return arff_dir / filename


def load_language_features(lang: str) -> dict:
    """Load all feature sets for a language.
    Returns dict: feat_name -> DataFrame (with 'class' column).
    """
    features = {}
    for feat_name in FEATURE_SETS:
        path = get_arff_path(lang, feat_name)
        if not path.exists():
            print(f"  WARNING: {path.name} not found!")
            continue
        features[feat_name] = load_arff(path)
    return features


# ═══════════════════════════════════════════════════════════════════════════════
# Step 1: Split each language (stratified by class)
# ═══════════════════════════════════════════════════════════════════════════════

print("=" * 60)
print("STEP 1: Stratified train/test split per language")
print("=" * 60)

split_indices = {}

for lang in LANG_CONFIG:
    features = load_language_features(lang)
    # Use trad to get row count and labels (all feature sets have same rows)
    n = len(features["trad"])
    labels = features["trad"]["class"].astype(str)

    # Stratified split with exact test size
    indices = np.arange(n)
    test_n = TEST_SIZES[lang]
    train_idx, test_idx = train_test_split(
        indices, test_size=test_n, random_state=SEED, stratify=labels
    )
    split_indices[lang] = (train_idx, test_idx)

    print(f"\n  {lang.upper()}: {n} total -> {len(train_idx)} train, {len(test_idx)} test")

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

# ═══════════════════════════════════════════════════════════════════════════════
# Step 2: Stack train splits for each language pair
# ═══════════════════════════════════════════════════════════════════════════════

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
        if feat_name not in features1 or feat_name not in features2:
            continue

        train1 = features1[feat_name].iloc[train_idx1].reset_index(drop=True)
        train2 = features2[feat_name].iloc[train_idx2].reset_index(drop=True)
        stacked_train = pd.concat([train1, train2], axis=0, ignore_index=True)

        filename = f"paired_{pair_name}_train_{feat_name}"
        save_both(stacked_train, filename, BI_DIR, filename)

    n_train = len(features1["trad"].iloc[train_idx1]) + len(features2["trad"].iloc[train_idx2])
    print(f"    Bilingual train size: {n_train}")

print(f"\n  Bilingual train sets saved to: {BI_DIR}")
print("\nDone.")
