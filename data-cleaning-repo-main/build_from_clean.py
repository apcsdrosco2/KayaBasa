"""
build_from_clean.py — Build the 764-doc corpus directly from the clean/ folder.

The clean/ folder contains the pre-cleaned, thesis-aligned documents:
  clean/tag_all_clean.txt   — 265 Tagalog docs
  clean/ceb_all_clean.txt   — 349 Cebuano docs
  clean/bik_all_clean.txt   — 150 Bikolano docs
  Total: 764 high-resource documents (matches thesis Table IV)

Format in clean/ files: title,level,text  (comma-separated, one doc per line)

This script converts them to the standard pipeline format:
  language|level|text   (pipe-separated)

And produces:
  output/all_languages_raw.txt        — noisy (clean/ text = pre-normalization)
  output_normalized/all_languages.txt — denoised (after normalize_datasets.py)
  splits/fold_indices.pkl             — stratified 5-fold CV splits

Run:
  cd data-cleaning-repo-main
  python build_from_clean.py
  python normalize_datasets.py
  python split_datasets.py
"""

import os
import unicodedata

CLEAN_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "clean")
OUT_DIR   = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")

CLEAN_FILES = {
    "tagalog": "tag_all_clean.txt",
    "cebuano": "ceb_all_clean.txt",
    "bikol":   "bik_all_clean.txt",
}

# Low-resource languages are still sourced from BasahaCorpus via build_datasets.py
# This script only handles the high-resource languages from clean/

SEP = "|"


def read_clean_file(path: str, language: str):
    """
    Read a clean/ file (format: title,level,text) and yield (language, level, text).
    The title field is discarded — only level and text are used.
    """
    with open(path, encoding="utf-8") as f:
        lines = [l for l in f.read().splitlines() if l.strip()]

    skipped = 0
    for line in lines:
        # Split on comma, max 2 splits → [title, level, text]
        parts = line.split(",", 2)
        if len(parts) < 3:
            skipped += 1
            continue
        title, level_str, text = parts[0].strip(), parts[1].strip(), parts[2].strip()

        # Validate level
        if level_str not in ("1", "2", "3"):
            skipped += 1
            continue

        # Drop empty texts
        if not text:
            skipped += 1
            continue

        yield language, int(level_str), text

    if skipped:
        print(f"  [{language}] skipped {skipped} malformed rows")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # ── Read high-resource from clean/ ─────────────────────────────────────
    hr_rows = []
    for language, filename in CLEAN_FILES.items():
        path = os.path.join(CLEAN_DIR, filename)
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Clean file not found: {path}\n"
                f"Expected at: {CLEAN_DIR}/{filename}"
            )
        docs = list(read_clean_file(path, language))
        hr_rows.extend(docs)
        from collections import Counter
        lc = Counter(level for _, level, _ in docs)
        print(f"  {language}: {len(docs)} docs  "
              f"L1={lc[1]}  L2={lc[2]}  L3={lc[3]}")

    # ── Read low-resource from existing build_datasets.py output ───────────
    # build_datasets.py should have already been run to produce output/all_languages.txt
    # We take only the low-resource rows from it
    lr_rows = []
    old_master = os.path.join(OUT_DIR, "all_languages.txt")
    LR_LANGS = {"hiligaynon", "minasbate", "karay-a", "rinconada"}

    if os.path.exists(old_master):
        with open(old_master, encoding="utf-8") as f:
            for line in f.read().splitlines()[1:]:   # skip header
                if not line.strip():
                    continue
                parts = line.split(SEP, 2)
                if len(parts) < 3:
                    continue
                lang, level, text = parts[0].strip(), int(parts[1].strip()), parts[2]
                if lang in LR_LANGS:
                    lr_rows.append((lang, level, text))
        from collections import Counter
        lc = Counter(lang for lang, _, _ in lr_rows)
        print(f"\n  Low-resource (from existing output/all_languages.txt):")
        for lang, count in sorted(lc.items()):
            print(f"    {lang}: {count} docs")
    else:
        print("\n  WARNING: output/all_languages.txt not found.")
        print("  Run build_datasets.py first to get the low-resource languages.")
        print("  Proceeding with high-resource only.")

    all_rows = hr_rows + lr_rows

    # ── Write master file (raw/noisy — clean/ text is pre-normalization) ───
    master_path = os.path.join(OUT_DIR, "all_languages_raw.txt")
    with open(master_path, "w", encoding="utf-8", newline="\n") as f:
        f.write(f"language{SEP}level{SEP}text\n")
        for lang, level, text in all_rows:
            f.write(f"{lang}{SEP}{level}{SEP}{text}\n")

    # Also write to all_languages.txt (used by normalize_datasets.py as input)
    standard_path = os.path.join(OUT_DIR, "all_languages.txt")
    with open(standard_path, "w", encoding="utf-8", newline="\n") as f:
        f.write(f"language{SEP}level{SEP}text\n")
        for lang, level, text in all_rows:
            f.write(f"{lang}{SEP}{level}{SEP}{text}\n")

    # ── Report ──────────────────────────────────────────────────────────────
    from collections import Counter
    lang_counts = Counter(lang for lang, _, _ in all_rows)
    level_counts = {lang: Counter() for lang in lang_counts}
    for lang, level, _ in all_rows:
        level_counts[lang][level] += 1

    print(f"\nPer-language/level counts:")
    for lang in sorted(lang_counts):
        lc = level_counts[lang]
        role = "high_resource" if lang in {"tagalog", "cebuano", "bikol"} else "low_resource"
        print(f"  {lang:<14} L1={lc[1]:>3}  L2={lc[2]:>3}  L3={lc[3]:>3}  "
              f"total={lang_counts[lang]:>3}  [{role}]")

    hr_total = sum(lang_counts[l] for l in {"tagalog", "cebuano", "bikol"} if l in lang_counts)
    lr_total = sum(lang_counts[l] for l in LR_LANGS if l in lang_counts)
    print(f"\nHigh-resource total: {hr_total}  (thesis Table IV: 764)")
    print(f"Low-resource total:  {lr_total}  (held-out transfer set)")
    print(f"Grand total:         {len(all_rows)}")
    print(f"\nWrote: {master_path}")
    print(f"Wrote: {standard_path}")
    print(f"\nNext: python normalize_datasets.py")
    print(f"      python split_datasets.py")


if __name__ == "__main__":
    main()
