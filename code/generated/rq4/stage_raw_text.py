"""
stage_raw_text.py — Stage A of the RQ4 pipeline.

Reads the raw BasahaCorpus-HierarchicalCrosslingualARA `data/raw/{lang}/grade {N}/*.txt`
files, staged in this repo at `code/low_resource_data/raw/` (verified identical to a
fresh clone of https://github.com/imperialite/BasahaCorpus-HierarchicalCrosslingualARA
at commit bf3b40f8b557252af0dd9e4a2dae0782b0f29bce, its current upstream `main` tip —
see RQ4_PLAN.md §2A/§2C for the full document-count investigation), cleans each
document with the bug-fixed cleaner (fixed_clean_basaha.py — see RQ4_PLAN.md §2A for
the two root-caused bugs this fixes), and writes one pipe-delimited `level|flag|text`
file per language to `clean/rq4/`, mirroring the existing
`clean/{bik,ceb,tag}_all_clean.txt` convention used by the rest of the pipeline.

Corpus size: **769** documents — one row per raw `.txt` file, every single one,
Hiligaynon 133, Minasbate 268, Karay-a 173, Rinconada 195. **Per explicit user
directive (RQ4_PLAN.md §2C): cleaning must never drop a document.** Every raw file is
staged as a row, unconditionally, carrying whatever text `clean_basaha()` produced for
it (which may be empty or near-empty for a small number of degenerate source files —
see the `flag` column). 769 is 7 short of the paper's reported 776 total (Table VI:
Minasbate 271, Karay-a 177 vs. 268/173 actually on disk) — those 7 documents have no
raw file anywhere in the pinned upstream commit (confirmed via a full commit-history
walk showing the file count only ever increases, current HEAD is the historical max)
and are not recoverable with the tools available in this environment (Let's Read
Asia/Bloom Library are JS-only SPAs). **User-approved final count: 769**
(RQ4_PLAN.md status line).

Output format is `level|flag|text` (3 columns, header row) — the `flag` column records
whether cleaning found real narrative (`ok`) or a degenerate case
(`empty_after_cleaning`, `near_empty_after_cleaning`, `boilerplate_only`), so
downstream feature-extraction/embedding stages can see which rows are degenerate
without re-deriving that themselves, while still processing all 769 rows.

Usage: python stage_raw_text.py
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fixed_clean_basaha as fx

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent.parent  # .../KayaBasa

# RQ4_RAW_BASE lets a different environment (e.g. a fresh clone without this repo's
# local, untracked code/low_resource_data/ copy) point this at wherever it staged its
# own copy of BasahaCorpus-HierarchicalCrosslingualARA's data/raw — same structure
# expected either way: {lang}/grade {N}/*.txt. Defaults to this repo's local copy.
RAW_BASE = Path(os.environ["RQ4_RAW_BASE"]) if os.environ.get("RQ4_RAW_BASE") else \
    REPO_ROOT / "code" / "low_resource_data" / "raw"
OUT_DIR = REPO_ROOT / "clean" / "rq4"

LANGUAGES = ["hiligaynon", "minasbate", "karay-a", "rinconada"]
GRADES = [("grade 1", 1), ("grade 2", 2), ("grade 3", 3)]
MIN_WORDS = 4  # no longer used to drop rows — kept only as the threshold for the
                # `near_empty_after_cleaning` flag label, see classify_flag() below.

# NOT exclusion lists (per user directive, RQ4_PLAN.md §2C: cleaning must never drop a
# document). Kept only as documentation of which raw files are known to clean down to
# degenerate text, so classify_flag() below can label them accurately and an
# *unexpected* new degenerate case (a real regression) still surfaces distinctly rather
# than being silently absorbed into "ok".
KNOWN_BOILERPLATE_ONLY = {
    ("minasbate", 1, "Paglinis_san_Lawas___Minasbate"),
    ("rinconada", 1, "A_Palda_Kong_Pula___Rinconada-v12022.09.20T002148"),
}

KNOWN_STRUCTURAL_STUB = {
    ("hiligaynon", 1, "Buligan_ang_pispis___Hiligaynon"),
    ("hiligaynon", 2, "Makakita_Ako_Sang_Maathag.___Hiligaynon"),
    ("hiligaynon", 3, "Ang_Tilawit_nga_si_Ceraphim___Hiligaynon-v12022.10.13T025534"),
    ("minasbate", 1, "Gusto_magliwan_ni_Nin___Minasbate"),
    ("minasbate", 2, "Kaya_ko_mag_himo_san_mga_Bagay___Minasbate"),
    ("rinconada", 1, "Iskedyul_ni_Momo___Rinconada-v12022.09.06T074145"),
}


def classify_flag(key: tuple, text: str) -> str:
    """Label (never exclude) rows whose cleaned text is degenerate."""
    if key in KNOWN_BOILERPLATE_ONLY:
        return "boilerplate_only"
    if key in KNOWN_STRUCTURAL_STUB:
        return "empty_after_cleaning" if not text.strip() else "near_empty_after_cleaning"
    if not text.strip():
        return "empty_after_cleaning"  # a NEW, previously-unseen empty case
    if len(text.split()) < MIN_WORDS:
        return "near_empty_after_cleaning"  # a NEW, previously-unseen near-empty case
    return "ok"


def main():
    if not RAW_BASE.is_dir():
        raise FileNotFoundError(f"Missing raw data dir: {RAW_BASE}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    grand_total = 0
    flag_counts = {}

    for lang in LANGUAGES:
        rows = []  # (level, flag, text)
        for grade_dir, level in GRADES:
            src_dir = RAW_BASE / lang / grade_dir
            if not src_dir.is_dir():
                raise FileNotFoundError(f"Missing raw dir: {src_dir}")
            for fp in sorted(src_dir.glob("*.txt")):
                title = fp.stem
                raw = fp.read_text(encoding="utf-8", errors="replace")
                text, _ = fx.clean_basaha(raw)
                key = (lang, level, title)
                flag = classify_flag(key, text)
                rows.append((level, flag, text))
                flag_counts[flag] = flag_counts.get(flag, 0) + 1

        out_path = OUT_DIR / f"{lang}_all_clean.txt"
        with open(out_path, "w", encoding="utf-8", newline="\n") as f:
            f.write("level|flag|text\n")
            for level, flag, text in rows:
                f.write(f"{level}|{flag}|{text}\n")

        by_level = {1: 0, 2: 0, 3: 0}
        for level, _, _ in rows:
            by_level[level] += 1
        print(f"{lang:<12} L1={by_level[1]:>3} L2={by_level[2]:>3} L3={by_level[3]:>3} "
              f"total={len(rows):>3}  -> {out_path}")
        grand_total += len(rows)

    print(f"\nGrand total: {grand_total} (expected 769 — every raw file kept as a row, none dropped)")
    for flag, count in sorted(flag_counts.items()):
        print(f"  flag={flag}: {count}")
    assert grand_total == 769, f"Expected 769 (all raw files kept), got {grand_total}"
    print("OK — 769 rows staged, one per raw file, none dropped during cleaning.")


if __name__ == "__main__":
    main()
