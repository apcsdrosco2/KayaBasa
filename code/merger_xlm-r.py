import io
from pathlib import Path
import pandas as pd

# Path routing
SCRIPT_DIR = Path(__file__).resolve().parent
GENERATED_DIR = SCRIPT_DIR / "generated"

# Language config: (language_name, folder_name)
LANGUAGES = {
    "bikol": "bikol docus",
    "cebuano": "ceb docus",
    "tagalog": "tag docus"
}


def save_arff(df, lang_name, output_path):
    """Write a DataFrame to ARFF format compatible with Weka."""
    labels = sorted(df["class"].astype(str).unique().tolist())

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(f"@RELATION readability_{lang_name}_features_xlmr\n\n")

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


for lang_name, folder_name in LANGUAGES.items():
    print(f"Processing {lang_name.upper()}...")

    lang_dir = GENERATED_DIR / folder_name

    # 1. Load the merged features CSV
    features_csv_path = lang_dir / f"{lang_name}_merged_features.csv"
    if not features_csv_path.exists():
        print(f"  Missing {features_csv_path.name}. Skipping.\n")
        continue

    df_features = pd.read_csv(features_csv_path)

    # Separate labels from features
    labels = df_features["class"].astype(str)
    df_features = df_features.drop(columns=["class"])

    # 2. Load XLM-R embeddings
    xlmr_path = lang_dir / f"{lang_name}_xlmr_features.csv"
    if not xlmr_path.exists():
        print(f"  Missing {xlmr_path.name}. Skipping.\n")
        continue

    xlmr = pd.read_csv(xlmr_path, header=None)
    xlmr.columns = [f"xlmr_{i:03d}" for i in range(xlmr.shape[1])]
    print(f"  Features: {df_features.shape}, XLM-R: {xlmr.shape}")

    # 3. Verify row counts match
    if len(df_features) != len(xlmr):
        print(f"  Row count mismatch! Features: {len(df_features)}, XLM-R: {len(xlmr)}. Skipping.\n")
        continue

    # 4. Merge features and XLM-R embeddings horizontally, then add class column
    class_col = pd.DataFrame({"class": labels.tolist()})
    merged = pd.concat([df_features, xlmr, class_col], axis=1)

    # 5. Save as CSV
    csv_output = lang_dir / f"{lang_name}_merged_features_xlmr.csv"
    merged.to_csv(csv_output, index=False)
    print(f"  Saved: {csv_output.name}")

    # 6. Save as ARFF
    arff_output = lang_dir / f"{lang_name}_merged_features_xlmr.arff"
    save_arff(merged, lang_name, arff_output)
    print(f"  Saved: {arff_output.name}\n")

print("Done. All merged features + XLM-R files are in their respective language folders.")
