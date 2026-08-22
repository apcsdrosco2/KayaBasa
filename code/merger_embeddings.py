import io
from pathlib import Path
import pandas as pd

# Path routing
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
GENERATED_DIR = SCRIPT_DIR / "generated"

LANGUAGES = {
    "bikol": "bik",
    "cebuano": "ceb",
    "tagalog": "tag"
}

SOURCE_FILES = {
    "bikol": PROJECT_DIR / "clean" / "bik_all_clean.txt",
    "cebuano": PROJECT_DIR / "clean" / "ceb_all_clean.txt",
    "tagalog": PROJECT_DIR / "clean" / "tag_all_clean.txt",
}


def load_arff_as_df(arff_path):
    """Parses a Weka ARFF file into a Pandas DataFrame."""
    with open(arff_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    data_start = 0
    columns = []

    for i, line in enumerate(lines):
        line = line.strip()
        if line.upper().startswith('@ATTRIBUTE'):
            col_name = line.split()[1]
            columns.append(col_name)
        elif line.upper().startswith('@DATA'):
            data_start = i + 1
            break

    data_lines = [line.strip() for line in lines[data_start:] if line.strip()]
    df = pd.read_csv(io.StringIO('\n'.join(data_lines)), header=None, names=columns)
    return df


def save_arff(df, lang_name, output_path):
    """Write a DataFrame to ARFF format compatible with Weka."""
    labels = sorted(df["class"].astype(str).unique().tolist())

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(f"@RELATION readability_{lang_name}_features_embeddings\n\n")

        for col in df.columns[:-1]:
            f.write(f"@ATTRIBUTE {col} NUMERIC\n")

        labels_str = "{" + ",".join(labels) + "}"
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


for lang_name, prefix in LANGUAGES.items():
    print(f"Processing {lang_name.upper()}...")

    # 1. Load the merged features CSV
    features_csv_path = GENERATED_DIR / f"{lang_name}_merged_features.csv"
    if not features_csv_path.exists():
        print(f"  Missing {features_csv_path.name}. Skipping.\n")
        continue

    df_features = pd.read_csv(features_csv_path)

    # Separate labels from features
    labels = df_features['class'].astype(str)
    df_features = df_features.drop(columns=['class'])

    # 2. Load mBERT embeddings
    mbert_path = GENERATED_DIR / f"{lang_name}_mbert_features.csv"
    if not mbert_path.exists():
        print(f"  Missing {mbert_path.name}. Skipping.\n")
        continue

    mbert = pd.read_csv(mbert_path, header=None)

    # 3. Filter mBERT rows to match valid source lines if needed
    source_lines = SOURCE_FILES[lang_name].read_text(
        encoding="utf-8", errors="ignore"
    ).splitlines()

    valid_source_indices = [
        index
        for index, line in enumerate(source_lines)
        if len(line.split(",", 2)) == 3 and line.split(",", 2)[2].strip()
    ]

    if len(mbert) == len(source_lines) and len(mbert) != len(valid_source_indices):
        mbert = mbert.iloc[valid_source_indices].reset_index(drop=True)

    mbert.columns = [f"mbert_{i:03d}" for i in range(mbert.shape[1])]

    # 4. Verify row counts match
    if len(df_features) != len(mbert):
        print(f"  Row count mismatch! Features: {len(df_features)}, mBERT: {len(mbert)}. Skipping.\n")
        continue

    # 5. Merge features and embeddings horizontally, then add class column
    class_col = pd.DataFrame({"class": labels.tolist()})
    merged = pd.concat([df_features, mbert, class_col], axis=1)

    # 6. Save as CSV
    csv_output = GENERATED_DIR / f"{lang_name}_merged_features_embeddings.csv"
    merged.to_csv(csv_output, index=False)
    print(f"  Saved: {csv_output.name}")

    # 7. Save as ARFF
    arff_output = GENERATED_DIR / f"{lang_name}_merged_features_embeddings.arff"
    save_arff(merged, lang_name, arff_output)
    print(f"  Saved: {arff_output.name}\n")

print("Done. All merged features + embeddings files are in the 'generated' folder.")
