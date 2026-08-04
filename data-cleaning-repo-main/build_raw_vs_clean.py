"""
build_raw_vs_clean.py — Aligned raw-vs-denoised corpus for the noisy-vs-denoised
comparison (KayaBasa §3.1.5).

For every document in the denoised corpus (output_normalized/all_languages.txt),
this emits BOTH the raw ingested text and the final cleaned+normalized text under
the SAME doc_id — the same key used by output/all_features.csv and splits/.  A
noisy baseline can therefore extract features from `text_raw` and evaluate under
the exact same folds as the denoised model, isolating the effect of denoising.

    text_raw    raw text after ingestion only — NO cleaning, NO normalization;
                whitespace is collapsed to a single line so it fits a CSV cell,
                but all noise (metadata, credits, entities, boilerplate) is kept.
    text_clean  cleaned + normalized narrative == the output_normalized master row.

Output: output/corpus_raw_vs_clean.csv
        columns: doc_id, language, label, split_role, text_raw, text_clean

The build reproduces build_datasets.py's emit order and then VERIFIES, row for row,
that text_clean equals output_normalized/all_languages.txt.  It aborts on any
mismatch, so the doc_id can be trusted to align with the feature matrix and splits.

Run:  python build_raw_vs_clean.py
"""

import os
import glob

import regex
import pandas as pd

import build_datasets as bd
import clean_tagalog
from normalize_datasets import normalize_text

SEP = "|"
HIGH_RESOURCE = {"tagalog", "cebuano", "bikol"}
LANGUAGES = ["tagalog", "cebuano", "bikol", "hiligaynon", "minasbate", "karay-a", "rinconada"]
MIN_WORDS = 4
DROP_FLAGS = {"too_short", "empty", "english_heavy", "duplicate", "residual_boilerplate"}
NORM_MASTER = os.path.join("output_normalized", "all_languages.txt")


def _ws(s: str) -> str:
    return regex.sub(r"\s+", " ", s).strip()


# ── Loaders that keep the RAW text alongside the cleaned narrative ─────────────
# Each mirrors build_datasets' loader but returns [lang, level, title, raw, text, flag].

def load_ara(path, language):
    out = []
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line.strip():
                continue
            m = bd._LINE.match(line)
            if not m:
                continue
            raw_title = m.group("title").strip()
            title = (raw_title.split("___")[0].replace("_", " ")
                     if language == "bikol" else raw_title)
            raw = m.group("text")
            narrative, flag = bd.clean_ara(raw, title)
            out.append([language, int(m.group("level")), title, raw, narrative, flag])
    return out


def load_basaha(language):
    out = []
    for grade, level in [("grade 1", 1), ("grade 2", 2), ("grade 3", 3)]:
        for fp in sorted(glob.glob(os.path.join(bd.BASAHA_BASE, language, grade, "*.txt"))):
            with open(fp, encoding="utf-8", errors="replace") as f:
                raw = f.read()
            narrative, flag = bd.clean_basaha(raw)
            title = os.path.basename(fp)[:-4].split("___")[0].replace("_", " ")
            out.append([language, level, title, raw, narrative, flag])
    return out


def load_tagalog():
    out = []
    for lvl in (1, 2, 3):
        path = clean_tagalog.TAG_GLOB.format(lvl=lvl)
        if not os.path.isfile(path):
            continue
        for line in open(path, encoding="utf-8", errors="replace").read().splitlines():
            if not line.strip():
                continue
            narrative, flag = clean_tagalog.clean_tagalog(line)
            title = " ".join(narrative.split()[:6])
            out.append(["tagalog", lvl, title, line, narrative, flag])
    return out


def emittable(text, flag):
    return (bool(text.strip()) and len(text.split()) >= MIN_WORDS
            and flag not in DROP_FLAGS)


def main():
    rows = []
    rows += load_ara(bd.ARA_CEBUANO, "cebuano")
    rows += load_ara(bd.ARA_BIKOL, "bikol")
    for lang in bd.BASAHA_LANGS:
        rows += load_basaha(lang)
    rows += load_tagalog()

    # Same english-heavy / too-short marking as build_datasets.
    for r in rows:
        _L, _lvl, _title, _raw, text, flag = r
        if text.strip() and not flag and bd._english_ratio(text) >= 0.40:
            r[5] = "english_heavy"
        elif text.strip() and len(text.split()) < MIN_WORDS and not flag:
            r[5] = "too_short"

    # Same within-language exact-duplicate marking (keep first).
    seen = set()
    for r in rows:
        L, _lvl, _title, _raw, text, flag = r
        if not emittable(text, flag):
            continue
        key = (L, text.strip())
        if key in seen:
            r[5] = "duplicate"
        else:
            seen.add(key)

    # Emit in build_datasets' order: languages x levels, original row order within.
    records = []
    for lang in LANGUAGES:
        for level in (1, 2, 3):
            for L, lvl, _title, raw, text, flag in rows:
                if L == lang and lvl == level and emittable(text, flag):
                    records.append({
                        "language": L,
                        "label": lvl,
                        "split_role": "high_resource" if L in HIGH_RESOURCE else "low_resource",
                        "text_raw": _ws(raw),
                        "text_clean": normalize_text(text),
                    })

    df = pd.DataFrame(records)
    df.insert(0, "doc_id", range(len(df)))

    # ── Verify alignment with the normalized master (doc_id must match) ─────────
    master = []
    with open(NORM_MASTER, encoding="utf-8") as f:
        f.readline()
        for line in f:
            line = line.rstrip("\n")
            if not line.strip():
                continue
            L, lvl, text = line.split(SEP, 2)
            master.append((L, int(lvl), text))

    if len(master) != len(df):
        raise SystemExit(f"ABORT: row-count mismatch — master {len(master)} vs built {len(df)}")

    mism = 0
    for i, (mL, mlvl, mtext) in enumerate(master):
        if not (df.at[i, "language"] == mL and df.at[i, "label"] == mlvl
                and df.at[i, "text_clean"] == mtext):
            mism += 1
            if mism <= 5:
                print(f"  MISMATCH at doc_id {i}: built ({df.at[i,'language']},"
                      f"{df.at[i,'label']}) vs master ({mL},{mlvl})")
    if mism:
        raise SystemExit(f"ABORT: {mism} rows do not match the normalized master; "
                         f"doc_id would be misaligned with features/splits.")

    out_path = os.path.join("output", "corpus_raw_vs_clean.csv")
    df[["doc_id", "language", "label", "split_role", "text_raw", "text_clean"]] \
        .to_csv(out_path, index=False)

    print(f"Wrote {len(df)} rows to {out_path}")
    print("doc_id verified to align with output_normalized/all_languages.txt "
          "(and thus output/all_features.csv + splits/).")
    print("\nPer-language counts:")
    print(df.groupby(["language", "split_role"])["doc_id"].count().to_string())
    rw = df["text_raw"].str.split().str.len()
    cw = df["text_clean"].str.split().str.len()
    print(f"\nMean words per doc — raw: {rw.mean():.1f}   denoised: {cw.mean():.1f}   "
          f"(removed on average {(rw - cw).mean():.1f} words/doc)")


if __name__ == "__main__":
    main()
