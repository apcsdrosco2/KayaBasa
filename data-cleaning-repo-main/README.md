# KayaBasa — Data Cleaning & Denoising

This repository prepares children's storybook corpora in **seven Philippine
languages** for automatic readability assessment. It narrows raw, cluttered
source files down to clean narrative text — removing titles, author/illustrator
credits, license/publisher boilerplate, page numbers, and activity sections —
then normalizes the text, extracts readability features, splits the data for
modeling, and **verifies the cleaning without native speakers**.

- **Languages:** Tagalog, Cebuano, Bikol (high-resource) · Hiligaynon, Minasbate, Karay-a, Rinconada (low-resource transfer set)
- **Corpus:** 1,480 documents across 3 reading levels (L1/L2/L3)
- **Design principle:** cleaning only *deletes* metadata/noise; it never rewrites the story. This is what lets the results be verified structurally, without fluency in the languages.

---

## Pipeline

Run in this order from the repository root:

```bash
python build_datasets.py       # raw sources        -> output/ (cleaned, per language/level)
python normalize_datasets.py   # output/            -> output_normalized/ (cleaned + normalized)
python extract_features.py     # output_normalized/ -> output/all_features.csv (features)
python split_datasets.py       # output_normalized/ -> splits/ (5-fold CV + held-out set)
```

Verification and the noisy-vs-denoised corpus (optional, run any time):

```bash
python validate_denoising.py   # gate scorecard   -> output/validation_report.txt, validation_queue.txt
python audit_cleaning.py       # per-span ledger  -> output/cleaning_ledger.txt, cleaning_review.txt
python build_raw_vs_clean.py   # aligned raw+clean -> output/corpus_raw_vs_clean.csv
```

*Requirements:* Python 3, `regex`, `ftfy`, `pandas`, `nltk`, `scikit-learn`. On
Windows, run with `PYTHONIOENCODING=utf-8` so native text prints correctly.

---

## Where the data lives

**Raw sources** (the "noisy" text, mixed formats, not `doc_id`-keyed):

| Language(s) | Location |
|---|---|
| Cebuano | `ara-close-lang/data/cebuano/ceb_all_data.txt` |
| Bikol | `ara-close-lang/data/bikol/bik_all_data.txt` |
| Hiligaynon / Minasbate / Karay-a / Rinconada | `BasahaCorpus-HierarchicalCrosslingualARA/data/raw/<lang>/grade {1,2,3}/*.txt` |
| Tagalog | `data/tag_lvl{1,2,3}_data.txt` |

**Cleaned / denoised text** (schema `language|level|text`):

| What | Location |
|---|---|
| Cleaned, **not** normalized (intermediate) | `output/all_languages.txt` + `output/<lang>_level<N>.txt` |
| Cleaned **+ normalized** — the modeling corpus | `output_normalized/all_languages.txt` + `output_normalized/<lang>_level<N>.txt` |

**Modeling artifacts** (all keyed by the same `doc_id` = row order of `output_normalized/all_languages.txt`):

| What | Location |
|---|---|
| Feature matrix (1480 × 23) | `output/all_features.csv` |
| Aligned raw-vs-denoised text (for the noisy-vs-denoised comparison) | `output/corpus_raw_vs_clean.csv` |
| Canonical doc table | `splits/combined_corpus.csv` |
| 5-fold CV indices | `splits/fold_indices.pkl` + `splits/high_resource_index.csv` |
| Held-out transfer set | `splits/low_resource_heldout.csv` |

> **Using both raw and cleaned data for models:** use `output/corpus_raw_vs_clean.csv`.
> It carries `text_raw` (raw, no cleaning) and `text_clean` (final denoised) for the
> **same `doc_id`**, so a noisy baseline and the denoised model can be evaluated under
> the identical `splits/fold_indices.pkl` — isolating the effect of denoising.

---

## Verification

The cleaning is checked with two instruments (no native speakers required):

- **`validate_denoising.py`** — reference reproduction (Cebuano 349/349, Bikol 148/150 against the original authors' published features), deletion accountability, and structural invariants (every letter/accent preserved, normalization idempotent, zero English/publisher leakage).
- **`audit_cleaning.py`** — an indexed, per-span ledger of *everything removed* from every document (`output/cleaning_ledger.txt`), plus a short review list of anything that looks like real story text rather than metadata (`output/cleaning_review.txt`).

Outputs of a passing run: grapheme preservation 1480/1480, 0 leakage, and only a handful of review items (all confirmed to be metadata such as bilingual title variants).

Full details:
- **Technical report:** [supplementary_context/Denoising_Evaluation_Report.md](supplementary_context/Denoising_Evaluation_Report.md)
- **Plain-language guide (non-technical):** [supplementary_context/Denoising_Evaluation_PlainLanguage.md](supplementary_context/Denoising_Evaluation_PlainLanguage.md)

---

## Key scripts

| Script | Role |
|---|---|
| `build_datasets.py` | Clean raw Cebuano/Bikol/BasahaCorpus down to narrative (calls `clean_tagalog.py` for Tagalog) |
| `clean_tagalog.py` | Tagalog cleaner (DepEd/Let's Read sources) |
| `normalize_datasets.py` | Structural normalization (no stemming/lemmatization — affixes carry the readability signal) |
| `extract_features.py` | TRAD / SYLL / CLGSNGO readability features → `output/all_features.csv` |
| `split_datasets.py` | Split roles + stratified 5-fold CV on the high-resource set |
| `validate_denoising.py` | Gate scorecard (reproduction, accountability, invariants) |
| `audit_cleaning.py` | Per-span removal ledger + narrative-loss review list |
| `build_raw_vs_clean.py` | Aligned raw-vs-denoised corpus for the §3.1.5 comparison |
