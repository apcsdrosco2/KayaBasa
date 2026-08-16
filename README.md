# KAYABASA

**KAYABASA: A Denoised Hybrid Transformer–Feature Model for Evaluating the Readability of Filipino Educational Texts**

> Bachelor of Science in Computer Science Thesis  
> Asia Pacific College — School of Computing and Information Technologies  
> Maria Sophea M. Balidio · Fredreich Martin H. Roxas · Jesmark David C. Presbitero · Suzanne Marie D. Rosco  
> 2026

---

## Overview

KAYABASA is a hybrid machine learning model designed to automatically classify the difficulty level of Key Stage 1 (Grades 1–3) early-grade reading materials across seven Central Philippine languages: **Tagalog, Bikolano, Cebuano, Hiligaynon, Minasbate, Karay-a, and Rinconada**.

The model combines two complementary components:

| Component | Role |
|---|---|
| **XLM-RoBERTa** (transformer) | Captures semantic context and discourse-level meaning of the text |
| **Regex Feature Pipeline** | Extracts 14 explicit linguistic features grounded in Philippine language morphology |

A **Multi-Layer Perceptron (MLP)** classifier fuses both representations and predicts one of three readability levels: **L1 (Grade 1)**, **L2 (Grade 2)**, or **L3 (Grade 3)**.

A **denoising pipeline** runs before both components to remove OCR artifacts, metadata boilerplate, and encoding noise from the raw educational texts.

---

## Repository Structure

```
KayaBasa/
├── README.md
├── data-cleaning-repo-main/          # Phase 1 — Data cleaning & normalization
│   ├── build_datasets.py             #   Load and merge ara-close-lang + BasahaCorpus
│   ├── clean_ara.py / clean_basaha.py #  Denoising for each corpus
│   ├── normalize_datasets.py         #   Text normalization (encoding, spacing, etc.)
│   ├── split_datasets.py             #   Stratified 5-fold CV splits
│   ├── validate_denoising.py         #   Denoising impact quantification
│   ├── output/                       #   Denoised corpus files (per language)
│   ├── output_normalized/            #   Final normalized corpus (all_languages.txt)
│   └── splits/                       #   Saved fold indices (fold_indices.pkl)
│
├── feature-pipeline/                 # Phase 2 — Feature extraction & training
│   ├── feature_pipeline.py           #   Core 14-feature regex pipeline (importable)
│   ├── extract_features.py           #   Driver: runs pipeline, saves feature CSV
│   ├── train_features.py             #   Features-only training (Configs A & B)
│   ├── ngrams list/                  #   Pre-computed top-25% anchor n-grams
│   │   ├── tag_top25_2gram.txt / tag_top25_3gram.txt
│   │   ├── bik_top25_2gram.txt / bik_top25_3gram.txt
│   │   └── ceb_top25_2gram.txt / ceb_top25_3gram.txt
│   └── output/
│       ├── all_features.csv          #   14-feature matrix (1480 docs)
│       └── results_config_B.json     #   Config B training results
│
└── raw/                              # Raw cloned datasets (not committed)
    ├── ara-close-lang/
    └── BasahaCorpus-HierarchicalCrosslingualARA/
```

---

## Model Architecture

```
  Raw Text
     │
     ▼
 ┌─────────────────────┐
 │  Denoising Pipeline │  ← removes metadata, boilerplate, encoding artifacts
 └─────────────────────┘
     │
     ▼
 ┌────────────────────────────────────────┐
 │            Cleaned Text                │
 └────────────────────────────────────────┘
     │                        │
     ▼                        ▼
 ┌───────────────┐     ┌───────────────────────┐
 │  XLM-RoBERTa  │     │  Regex Feature        │
 │  (fine-tuned) │     │  Pipeline             │
 │  768-dim [CLS]│     │  14-dim feature vector│
 └───────────────┘     └───────────────────────┘
     │                        │
     └──────────┬─────────────┘
                ▼
         Concatenate
         (782-dim input)
                │
                ▼
        ┌──────────────┐
        │  MLP + Softmax│
        │  (256 hidden) │
        └──────────────┘
                │
                ▼
        L1 / L2 / L3
```

---

## Feature Set (Thesis Table VII)

The regex pipeline extracts **14 features** across three groups:

### Group 1 — TRAD: Sentence-Level Surface Statistics (4 features)

| Feature | Description |
|---|---|
| `mean_sentence_len` | Mean number of words per sentence |
| `mean_word_len` | Mean number of characters per word |
| `polysyll_freq` | Proportion of words with ≥ 3 syllables |
| `type_token_ratio` | Vocabulary richness (unique words / total words) |

### Group 2 — SYLL: Phonotactic Decoding Load (4 features)

Frequencies of the four canonical Philippine consonant-vowel syllable patterns:

| Feature | Pattern | Example |
|---|---|---|
| `syll_cv` | CV | *ma*, *ba*, *la* |
| `syll_cvc` | CVC | *mag*, *kin*, *san* |
| `syll_ccvc` | CCVC | *krus*, *plas*, *trak* |
| `syll_ccvccc` | CCVCCC | *strikto*, *strengt* |

### Group 3 — CLGSNGO: Cross-Lingual N-Gram Overlap (6 features)

Rank-Biased Overlap (RBO) between the document's character n-gram distribution and the top-25% n-grams of each high-resource anchor language:

| Feature | Description |
|---|---|
| `tag_bi` / `tag_tri` | Bigram / trigram overlap with Tagalog anchor |
| `bik_bi` / `bik_tri` | Bigram / trigram overlap with Bikolano anchor |
| `ceb_bi` / `ceb_tri` | Bigram / trigram overlap with Cebuano anchor |

---

## Setup

### Prerequisites

- Python 3.10+
- `pip install pandas scikit-learn`

> The feature pipeline uses only Python built-ins, `pandas`, and `scikit-learn`. No NLP libraries (spaCy, NLTK, etc.) are required — by design, since morphological parsers do not exist for Minasbate, Karay-a, or Rinconada.

### 1. Clone the datasets

```bash
git clone https://github.com/imperialite/ara-close-lang.git raw/ara-close-lang
git clone https://github.com/imperialite/BasahaCorpus-HierarchicalCrosslingualARA.git raw/BasahaCorpus-HierarchicalCrosslingualARA
```

### 2. Run the data-cleaning pipeline

```bash
cd data-cleaning-repo-main
python build_datasets.py       # loads and merges both corpora
python normalize_datasets.py   # denoises + normalizes → output_normalized/all_languages.txt
python split_datasets.py       # generates stratified 5-fold splits → splits/fold_indices.pkl
```

### 3. Extract linguistic features

```bash
cd ../feature-pipeline
python extract_features.py
# Output: output/all_features.csv  (14 features × 1480 documents)
```

### 4. Train the features-only baseline (Config B)

```bash
python train_features.py
# Output: output/results_config_B.json
```

### 5. Train the full hybrid model (Colab recommended for GPU)

See the [Google Colab Training Guide](#google-colab-training-guide) below.

---

## Datasets

| Corpus | Languages | Role | Source |
|---|---|---|---|
| `ara-close-lang` | Tagalog, Cebuano, Bikolano | High-resource (training) | [imperialite/ara-close-lang](https://github.com/imperialite/ara-close-lang) |
| `BasahaCorpus` | Hiligaynon, Minasbate, Karay-a, Rinconada | Low-resource (cross-lingual test) | [imperialite/BasahaCorpus-HierarchicalCrosslingualARA](https://github.com/imperialite/BasahaCorpus-HierarchicalCrosslingualARA) |

All texts are Key Stage 1 narrative texts labeled at three difficulty levels (L1, L2, L3) aligned with the DepEd MTB-MLE curriculum.

---

## Ablation Study Configurations (Thesis Table XII)

| Config | Data | Components | Purpose |
|---|---|---|---|
| A | Noisy | Features only (RF baseline) | Noisy features baseline |
| **B** | **Denoised** | **Features only (RF baseline)** | **✅ Done — Macro-F1: 0.6487** |
| C | Noisy | Transformer only (XLM-R) | Noisy transformer baseline |
| D | Denoised | Transformer only (XLM-R) | Denoised transformer |
| E | Noisy | Hybrid (XLM-R + features + MLP) | Noisy hybrid baseline |
| **F** | **Denoised** | **Hybrid (XLM-R + features + MLP)** | **Full KAYABASA model** |

**Primary metric:** Macro-Averaged F1-Score (equal weight per class regardless of document count)

---

## Current Results (Config B — Features-Only Baseline)

| Language | Macro-F1 | Accuracy |
|---|:---:|:---:|
| Cebuano | 0.7431 | 0.7594 |
| Bikolano | 0.6591 | 0.6933 |
| Tagalog | 0.4796 | 0.4812 |
| **Overall (mean ± std)** | **0.6487 ± 0.0198** | **0.6554 ± 0.0202** |

---

## Google Colab Training Guide

For the full hybrid model (XLM-RoBERTa + features + MLP), GPU is required. Use Google Colab (free T4 GPU). See [`COLAB_GUIDE.md`](COLAB_GUIDE.md) for step-by-step instructions.

---

## References

- Imperial, J. & Ong, E. (2021). Diverse Linguistic Features for Assessing Reading Difficulty of Educational Filipino Texts. *ICCE 2021*.
- Reyes, L. et al. (2022). A Baseline Readability Model for Cebuano. *ACL Workshop 2022*.
- Imperial, J. & Kochmar, E. (2023). BasahaCorpus: An Expanded Linguistic Resource for Readability Assessment in Central Philippine Languages. *EMNLP 2023*.
- Imperial, J. (2023). Automatic Readability Assessment for Closely Related Languages. *EMNLP 2023*.
- Conneau, A. et al. (2020). Unsupervised Cross-lingual Representation Learning at Scale. *ACL 2020*.
