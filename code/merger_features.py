import pandas as pd
from pathlib import Path

# Set up paths relative to this script's location in the 'code' folder
project_dir = Path(__file__).resolve().parent
generated_dir = project_dir / 'generated'

# Map full language names to the prefixes used in the generated folder
languages = {
    "bikol": "bik",
    "cebuano": "ceb",
    "tagalog": "tag"
}

target_col = "grade_level"
columns_to_drop = ['text', 'book_title', target_col, 'Unnamed: 0']

def clean_df(df):
    """Drop non-feature columns from a dataframe."""
    drop_cols = [col for col in columns_to_drop if col in df.columns]
    return df.drop(columns=drop_cols)


def save_arff(df, lang_name, output_path):
    """Write a DataFrame to ARFF format compatible with Weka."""
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(f"@RELATION readability_{lang_name}\n\n")

        # Numeric attributes (all columns except the last 'class' column)
        for col in df.columns[:-1]:
            f.write(f"@ATTRIBUTE {col} NUMERIC\n")

        # Class attribute
        unique_labels = sorted(df['class'].unique().tolist())
        labels_str = "{" + ",".join(unique_labels) + "}"
        f.write(f"@ATTRIBUTE class {labels_str}\n\n")

        # Data section
        f.write("@DATA\n")
        for row in df.itertuples(index=False, name=None):
            f.write(",".join(map(str, row)) + "\n")


for lang_name, prefix in languages.items():
    print(f"Processing {lang_name.upper()}...")

    # 1. Define paths for the three feature CSV files
    trad_path = generated_dir / f"{prefix}_trad.csv"
    syll_path = generated_dir / f"{prefix}_syll.csv"
    clgsngo_path = generated_dir / f"{prefix}_clgsngo.csv"

    if not (trad_path.exists() and syll_path.exists() and clgsngo_path.exists()):
        print(f"  Missing one or more feature files for {lang_name}. Skipping.\n")
        continue

    # 2. Load feature files
    df_trad = pd.read_csv(trad_path)
    df_syll = pd.read_csv(syll_path)
    df_clgsngo = pd.read_csv(clgsngo_path)

    # 3. Extract target labels and clean dataframes
    labels = df_trad[target_col].astype(str)

    df_trad_clean = clean_df(df_trad)
    df_syll_clean = clean_df(df_syll)
    df_clgsngo_clean = clean_df(df_clgsngo)

    # 4. Merge all features horizontally
    merged_df = pd.concat([df_trad_clean, df_syll_clean, df_clgsngo_clean], axis=1)

    # Ensure all feature columns are numeric
    merged_df = merged_df.apply(pd.to_numeric, errors='coerce')

    # Add the target class column
    merged_df['class'] = labels

    # Replace NaNs with '?' (Weka's missing value marker)
    merged_df = merged_df.fillna("?")

    # 5. Save as CSV
    csv_output = generated_dir / f"{lang_name}_merged_features.csv"
    merged_df.to_csv(csv_output, index=False)
    print(f"  Saved: {csv_output.name}")

    # 6. Save as ARFF
    arff_output = generated_dir / f"{lang_name}_merged_features.arff"
    save_arff(merged_df, lang_name, arff_output)
    print(f"  Saved: {arff_output.name}\n")

print("Done. All merged feature files are in the 'generated' folder.")
