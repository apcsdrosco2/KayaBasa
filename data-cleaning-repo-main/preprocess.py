"""
preprocess.py — Stages 3.1.1 to 3.1.3
Loads Cebuano + Bikol (high-resource) and Hiligaynon, Minasbate, Karay-a,
Rinconada (low-resource), cleans, normalizes, and saves the combined corpus.

Cleaning is delegated to corpus-specific modules:
  clean_cebuano.py  — preamble (credits + genre tag) at start, KATAPUSAN at end
  clean_bikol.py    — end boilerplate (ABC+/USAID/Asia Foundation/…) + Q&A sections
  clean_basaha.py   — title/author lines at start, same end boilerplate as Bikol

Outputs: output/corpus_cleaned.csv and output/corpus_cleaned.pkl
"""

import os
import glob
import pandas as pd
import regex

import clean_cebuano
import clean_bikol
import clean_basaha

# ── Paths ────────────────────────────────────────────────────────────────────

ARA_CEBUANO  = "ara-close-lang/data/cebuano/ceb_all_data.txt"
ARA_BIKOL    = "ara-close-lang/data/bikol/bik_all_data.txt"
BASAHA_BASE  = "BasahaCorpus-HierarchicalCrosslingualARA/data/raw"
BASAHA_LANGS = ["hiligaynon", "minasbate", "karay-a", "rinconada"]
OUT_DIR      = "output"

# ── 3.1.1.1  Cebuano loader ──────────────────────────────────────────────────

def load_cebuano(filepath: str) -> pd.DataFrame:
    """Load ara-close-lang Cebuano CSV (one doc per line: title,label,text)."""
    records = []
    with open(filepath, encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(",", 2)
            if len(parts) < 3:
                continue
            records.append({
                "language": "cebuano",
                "label":    int(parts[1].strip()),
                "title":    parts[0].strip(),
                "text":     parts[2].strip(),
            })
    return pd.DataFrame(records)

# ── 3.1.1.2  Bikol loader ────────────────────────────────────────────────────

def load_bikol(filepath: str) -> pd.DataFrame:
    """Load ara-close-lang Bikol CSV (one doc per line: title___Central_Bikol,label,text).

    The raw title field uses underscores and has a ___Central_Bikol[-vTIMESTAMP]
    suffix.  We store the clean display title (underscores → spaces, suffix dropped)
    because that is what appears at the start of the text field.
    """
    records = []
    with open(filepath, encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(",", 2)
            if len(parts) < 3:
                continue
            raw_title    = parts[0].strip()
            display_title = clean_bikol.get_display_title(raw_title)
            records.append({
                "language":      "bikol",
                "label":         int(parts[1].strip()),
                "title":         display_title,
                "text":          parts[2].strip(),
                "raw_csv_title": raw_title,   # kept for debugging; dropped before save
            })
    return pd.DataFrame(records)

# ── 3.1.1.3  BasahaCorpus loader ────────────────────────────────────────────

def load_basaha_language(base_dir: str, language: str) -> pd.DataFrame:
    """Load all .txt files for one BasahaCorpus language from grade 1/2/3 dirs."""
    records = []
    for grade_str, label_int in [("grade 1", 1), ("grade 2", 2), ("grade 3", 3)]:
        level_dir = os.path.join(base_dir, grade_str)
        if not os.path.isdir(level_dir):
            continue
        for fpath in sorted(glob.glob(os.path.join(level_dir, "*.txt"))):
            with open(fpath, encoding="utf-8", errors="replace") as f:
                text = f.read()
            records.append({
                "language":    language,
                "label":       label_int,
                "title":       os.path.basename(fpath)[:-4],
                "text":        text,
                "source_path": fpath,
            })
    return pd.DataFrame(records)

# ── 3.1.3  Normalization (applied after corpus-specific cleaning) ────────────

def normalize_text(text: str) -> str:
    text = regex.sub(r"&nbsp;", " ", text)
    text = regex.sub(r"&amp;",  "&", text)
    text = regex.sub(r"&#\d+;", "", text)
    # Ensure space after sentence-ending punctuation before next word
    text = regex.sub(r"([,.!?;:\"'»\)])\s*(?=\p{L})", r"\1 ", text)
    # Normalize hyphens inside words
    text = regex.sub(r"(?<=\p{L})\s*-\s*(?=\p{L})", "-", text)
    text = regex.sub(r"(?<=\p{L})-{2,}(?=\p{L})", "-", text)
    # Em-dash / en-dash
    text = regex.sub(r"\s*[–—]\s*", " — ", text)
    # Collapse internal whitespace
    text = regex.sub(r"[ \t]+", " ", text)
    text = regex.sub(r" +\n", "\n", text)
    return text.strip()

# ── Cleaning dispatch ────────────────────────────────────────────────────────

def clean_row(row, unmatched_log: list) -> str:
    language = row["language"]
    text     = str(row["text"])

    if language == "cebuano":
        text = clean_cebuano.clean(text, row["title"])

    elif language == "bikol":
        text = clean_bikol.clean(text, row["title"])

    else:
        # All four BasahaCorpus languages
        text, matched = clean_basaha.clean(text)
        if not matched:
            unmatched_log.append(row["doc_id"])

    return normalize_text(text)

# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # ── Load high-resource (ara-close-lang) ───────────────────────────────────
    print("Loading ara-close-lang (high-resource)...")
    df_ceb = load_cebuano(ARA_CEBUANO)
    df_bik = load_bikol(ARA_BIKOL)
    print(f"  Cebuano: {len(df_ceb)} | Bikol: {len(df_bik)}")

    df_high = pd.concat([df_ceb, df_bik], ignore_index=True)
    df_high["split_role"]  = "high_resource"
    df_high["source_path"] = ""

    # ── Load low-resource (BasahaCorpus) ──────────────────────────────────────
    print("Loading BasahaCorpus (low-resource)...")
    basaha_frames = []
    for lang in BASAHA_LANGS:
        lang_dir = os.path.join(BASAHA_BASE, lang)
        df_lang  = load_basaha_language(lang_dir, lang)
        print(f"  {lang}: {len(df_lang)} docs")
        basaha_frames.append(df_lang)

    df_low = pd.concat(basaha_frames, ignore_index=True)
    df_low["split_role"] = "low_resource"

    # ── Combine, dedup, assign doc_id ─────────────────────────────────────────
    df_all = pd.concat([df_high, df_low], ignore_index=True)

    # Drop the raw_csv_title debug column from Bikol before dedup
    if "raw_csv_title" in df_all.columns:
        df_all = df_all.drop(columns=["raw_csv_title"])

    before = len(df_all)
    df_all = df_all.drop_duplicates(subset=["language", "text"]).reset_index(drop=True)
    df_all.insert(0, "doc_id", range(len(df_all)))
    dropped = before - len(df_all)
    if dropped:
        print(f"\n  Dropped {dropped} duplicate (language, text) rows")

    df_all["text_raw"] = df_all["text"]  # frozen copy for §3.1.5 denoising experiment

    # ── Clean and normalize ───────────────────────────────────────────────────
    print("\nCleaning and normalizing...")
    unmatched_log: list = []
    df_all["text"] = df_all.apply(lambda r: clean_row(r, unmatched_log), axis=1)

    if unmatched_log:
        print(f"\n  Warning: {len(unmatched_log)} BasahaCorpus docs had no "
              f"boilerplate/activity sentinel match.")
        print(f"  doc_ids flagged: {unmatched_log[:20]}")
        flagged_path = os.path.join(OUT_DIR, "basaha_unmatched_docs.csv")
        df_all[df_all["doc_id"].isin(unmatched_log)][
            ["doc_id", "language", "label", "title", "source_path"]
        ].to_csv(flagged_path, index=False)
        print(f"  Saved to: {flagged_path}")

    # ── Drop stub documents (empty after cleaning) ────────────────────────────
    empty_mask = df_all["text"].str.strip().str.len() == 0
    if empty_mask.any():
        stubs = df_all[empty_mask][["doc_id", "language", "label", "title"]]
        print(f"\n  Warning: {len(stubs)} stub doc(s) with no narrative body — dropped:")
        print(stubs.to_string(index=False))
        df_all = df_all[~empty_mask].reset_index(drop=True)
        df_all["doc_id"] = range(len(df_all))

    # ── Integrity checks ──────────────────────────────────────────────────────
    assert df_all["doc_id"].is_unique,                "doc_id not unique"
    assert df_all["label"].isin([1, 2, 3]).all(),     "label outside {1,2,3}"
    assert not df_all.duplicated(subset=["language", "text"]).any(), "duplicates remain"

    # ── Label distribution ────────────────────────────────────────────────────
    print("\nLabel distribution:")
    dist = df_all.groupby(["language", "label"]).size().unstack(fill_value=0)
    dist["total"] = dist.sum(axis=1)
    print(dist.to_string())

    print("\nsplit_role counts:")
    print(df_all["split_role"].value_counts().to_string())

    # ── Save ──────────────────────────────────────────────────────────────────
    csv_path = os.path.join(OUT_DIR, "corpus_cleaned.csv")
    pkl_path = os.path.join(OUT_DIR, "corpus_cleaned.pkl")
    df_all.to_csv(csv_path, index=False, encoding="utf-8")
    df_all.to_pickle(pkl_path)

    print(f"\nSaved {len(df_all)} documents to:")
    print(f"  {csv_path}")
    print(f"  {pkl_path}")
    print("\nColumns:", list(df_all.columns))


if __name__ == "__main__":
    main()
