import pandas as pd
import arff
import os

# 1. Set the language you want to merge (options: "tagalog", "cebuano", "bikol")
language = "bikol"  # Change this to your desired language

# 2. Load the specific files for the chosen language
# Ensure your working directory is set to where these files are located
df_trad = pd.read_csv(f"trad_{language}.csv")
df_syll = pd.read_csv(f"syll_{language}.csv")
df_clgsngo = pd.read_csv(f"clgsngo_{language}.csv")

# NOTE: If you have generated the XLM-R embeddings, uncomment the line below:
# df_xlmr = pd.read_csv(f"xlmr_{language}.csv")

# 3. Set the name of your target column (e.g., 'label', 'level', or 'grade')
target_col = "grade_level"

# Extract the target labels from the first dataframe and ensure they are strings (Nominal)
labels = df_trad[target_col].astype(str)

# 4. Drop 'text', the target column, and any index columns from all dataframes
columns_to_drop = ['text', 'book_title', target_col, 'Unnamed: 0']

def clean_df(df):
    drop_cols = [col for col in columns_to_drop if col in df.columns]
    return df.drop(columns=drop_cols)

df_trad_clean = clean_df(df_trad)
df_syll_clean = clean_df(df_syll)
df_clgsngo_clean = clean_df(df_clgsngo)

# If using XLM-R, add df_xlmr_clean to this list:
dataframes_to_merge = [df_trad_clean, df_syll_clean, df_clgsngo_clean] 

# 5. Merge all features horizontally
merged_df = pd.concat(dataframes_to_merge, axis=1)

# Ensure all feature columns are numeric
merged_df = merged_df.apply(pd.to_numeric, errors='coerce')

# Add the target class column back to the very end of the dataframe
merged_df['class'] = labels

# 6. Build the ARFF structure
attributes = [(col, 'NUMERIC') for col in merged_df.columns[:-1]]
unique_labels = sorted(labels.unique().tolist())
attributes.append(('class', unique_labels))

data = merged_df.values.tolist()

arff_dic = {
    'description': f'Merged features for {language}',
    'relation': f'readability_{language}',
    'attributes': attributes,
    'data': data
}

# 7. Save the final ARFF file
output_filename = f"merged_features_{language}.arff"
with open(output_filename, "w", encoding="utf-8") as f:
    arff.dump(arff_dic, f)

print(f"Success! {output_filename} is ready to be loaded into Weka.")