"""
rq4_feature_extraction.py — Stage B of the RQ4 pipeline.

Reads the staged `clean/rq4/{lang}_all_clean.txt` files (level|flag|text, header row —
see stage_raw_text.py) and computes the exact 24-column `trad_clgsngo` feature schema
the trained hybrid model expects, by calling this project's own TRAD.py/SYLL.py/
CLGSNGO.py functions directly — the same modules used for Tagalog/Cebuano/Bikolano,
verified column-for-column against `code/generated/bikol docus/arff/bikol_trad_clgsngo.arff`'s
header (RQ4_PLAN.md §2). CLGSNGO's n-gram anchor files are staged at
`data-cleaning-repo-main/ngrams list/` (its hardcoded path) as part of this stage.

Per explicit user directive (RQ4_PLAN.md §2C): every one of the 769 staged rows gets a
feature row here too, none dropped — including the ~8 degenerate rows whose cleaned
text is empty or near-empty. TRAD/SYLL/CLGSNGO's own functions divide by word count or
n-gram count internally and would raise ZeroDivisionError on empty text, so every
feature call here is wrapped to fall back to 0.0 on that specific error rather than
crash or skip the row — this only ever fires for rows already flagged non-`ok` by
Stage A, and is logged so a NEW unexpected zero-division wouldn't be silently absorbed.

Usage: python rq4_feature_extraction.py
"""

import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
CODE_DIR = SCRIPT_DIR.parent.parent  # .../code
REPO_ROOT = CODE_DIR.parent
sys.path.insert(0, str(CODE_DIR))

import TRAD
import SYLL
import CLGSNGO
import pandas as pd

CLEAN_DIR = REPO_ROOT / "clean" / "rq4"
OUT_DIR = SCRIPT_DIR / "generated"
OUT_DIR.mkdir(parents=True, exist_ok=True)

LANGUAGES = ["hiligaynon", "minasbate", "karay-a", "rinconada"]

TRAD_FUNCS = [
    ("word_count", TRAD.word_count_per_doc),
    ("sentence_count", TRAD.sentence_count_per_doc),
    ("phrase_count_per_sentence", TRAD.ave_phrase_count_per_doc),
    ("average_word_len", TRAD.ave_word_length),
    ("average_sentence_len", TRAD.word_count_per_sentence),
    ("average_syllable_count", TRAD.ave_syllable_count_of_word),
    ("polysyll_count", TRAD.polysyll_count_per_doc),
]

SYLL_FUNCS = [
    ("consonant_cluster_density", SYLL.get_consonant_cluster),
    ("v_density", SYLL.get_v),
    ("cv_density", SYLL.get_cv),
    ("vc_density", SYLL.get_vc),
    ("cvc_density", SYLL.get_cvc),
    ("vcc_density", SYLL.get_vcc),
    ("cvcc_density", SYLL.get_cvcc),
    ("ccvc_density", SYLL.get_ccvc),
    ("ccv_density", SYLL.get_ccv),
    ("ccvcc_density", SYLL.get_ccvcc),
    ("ccvccc_density", SYLL.get_ccvccc),
]

# Exact 24-column order, matching bikol_trad_clgsngo.arff's header verbatim (including
# the existing repo's "cebuano_trigam_sim" typo — kept, not fixed, for schema parity).
COLUMN_ORDER = (
    [name for name, _ in TRAD_FUNCS]
    + [name for name, _ in SYLL_FUNCS]
    + [
        "tagalog_bigram_sim", "bikol_bigram_sim", "cebuano_bigram_sim",
        "tagalog_trigram_sim", "bikol_trigram_sim", "cebuano_trigam_sim",
    ]
)


def safe_call(func, text, zero_div_log, doc_id, feat_name):
    try:
        return func(text)
    except ZeroDivisionError:
        zero_div_log.append((doc_id, feat_name))
        return 0.0


def main():
    grand_total = 0
    zero_div_log = []  # (doc_id, feature_name) pairs that hit the empty-text fallback

    for lang in LANGUAGES:
        in_path = CLEAN_DIR / f"{lang}_all_clean.txt"
        if not in_path.exists():
            raise FileNotFoundError(f"Missing staged file: {in_path} — run stage_raw_text.py first")

        rows = []
        with open(in_path, "r", encoding="utf-8") as f:
            header = f.readline()  # "level|flag|text"
            for idx, line in enumerate(f):
                parts = line.rstrip("\n").split("|", 2)
                if len(parts) < 3:
                    raise AssertionError(f"{lang} line {idx}: malformed row {parts!r}")
                level, flag, text = parts
                doc_id = f"{lang}_{idx:04d}"

                row = {"doc_id": doc_id, "flag": flag}
                for name, func in TRAD_FUNCS:
                    row[name] = safe_call(func, text, zero_div_log, doc_id, name)
                for name, func in SYLL_FUNCS:
                    row[name] = safe_call(func, text, zero_div_log, doc_id, name)

                try:
                    t_b, b_b, c_b = CLGSNGO.get_bigram_CLGSNGO(text)
                except ZeroDivisionError:
                    zero_div_log.append((doc_id, "bigram_CLGSNGO"))
                    t_b, b_b, c_b = 0.0, 0.0, 0.0
                try:
                    t_t, b_t, c_t = CLGSNGO.get_trigram_CLGSNGO(text)
                except ZeroDivisionError:
                    zero_div_log.append((doc_id, "trigram_CLGSNGO"))
                    t_t, b_t, c_t = 0.0, 0.0, 0.0

                row["tagalog_bigram_sim"] = t_b
                row["bikol_bigram_sim"] = b_b
                row["cebuano_bigram_sim"] = c_b
                row["tagalog_trigram_sim"] = t_t
                row["bikol_trigram_sim"] = b_t
                row["cebuano_trigam_sim"] = c_t
                row["class"] = level

                rows.append(row)

        df = pd.DataFrame(rows, columns=["doc_id", "flag"] + COLUMN_ORDER + ["class"])
        out_path = OUT_DIR / f"{lang}_trad_clgsngo.csv"
        df.to_csv(out_path, index=False)
        print(f"{lang:<12} rows={len(df):>3}  -> {out_path}")
        grand_total += len(df)

    print(f"\nGrand total: {grand_total} (expected 769 — one feature row per staged row, none dropped)")
    assert grand_total == 769, f"Expected 769, got {grand_total}"
    if zero_div_log:
        print(f"\n{len(zero_div_log)} feature calls fell back to 0.0 on empty/near-empty text:")
        for doc_id, feat_name in zero_div_log:
            print(f"  {doc_id}: {feat_name}")
    print("\nOK — 769 feature rows written, none dropped.")


if __name__ == "__main__":
    main()
