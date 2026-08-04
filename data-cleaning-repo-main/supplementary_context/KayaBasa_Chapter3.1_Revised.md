# 3.1 DATA PROCESSING (Revised)

This section specifies the data processing pipeline that produces the inputs for model
development and evaluation. The pipeline is organized into seven stages: data collection
(3.1.1), data cleaning (3.1.2), data normalization (3.1.3), feature engineering (3.1.4),
denoising impact quantification (3.1.5), data splitting (3.1.6), and the consolidated data
processing outputs and integrity verification that constitute the deliverables of this stage
(3.1.7). All paths, filenames, and directory structures below correspond to the actual state
of the two source repositories as cloned; where the repository structure differs from earlier
descriptions, the repository is treated as authoritative.

---

## 3.1.1 Data Collection

The text corpora are drawn from two publicly accessible GitHub repositories maintained by
Imperial and Kochmar: `ara-close-lang` [17] and `BasahaCorpus-HierarchicalCrosslingualARA`
[16]. Both distribute early-grade (Grades 1–3) educational narrative texts in Central
Philippine languages, organized by the three readability levels (L1, L2, L3) defined by the
MTB-MLE curriculum. The repositories are cloned locally:

```bash
git clone https://github.com/imperialite/ara-close-lang.git
git clone https://github.com/imperialite/BasahaCorpus-HierarchicalCrosslingualARA.git
```

All data access, preprocessing, and feature extraction reference these local clones. No API
authentication or web scraping is involved. Because the two repositories use different file
formats, directory conventions, and label encodings, each is ingested by a distinct loading
procedure.

### 3.1.1.0 Environment Prerequisites

The feature extraction scripts shipped in both repositories depend on the NLTK tokenizer.
The required tokenizer data must be downloaded once before any extraction is attempted,
otherwise the scripts raise a `LookupError` at runtime:

```python
import nltk
nltk.download('punkt')
nltk.download('punkt_tab')   # required for NLTK >= 3.8.2
```

The Unicode-aware normalization and encoding-repair steps require the third-party
`regex` and `ftfy` libraries in addition to `pandas`, `numpy`, and `scikit-learn`:

```bash
pip install pandas numpy scikit-learn nltk regex ftfy
```

### 3.1.1.1 ara-close-lang Dataset

**Repository structure.** The `ara-close-lang` repository contains a `/data/` directory with one
subdirectory per language (`/cebuano/`, `/bikol/`, `/tagalog/`) and a `/code/` directory holding
the feature extraction scripts (`TRAD.py`, `SYLL.py`, `CLGSNGO.py`, and their corresponding
`*_parser.py` driver scripts). Anchor n-gram lists are stored under `/data/ngrams/` and
`/data/mutual_intelligibility/ngrams/`.

**File format.** Each language is stored as a single flat `.txt` file (e.g., `ceb_all_data.txt`),
in which each line represents one complete document under a fixed comma-separated schema:

```
<Title>,<Label>,<raw text field including title repeat, credits, body, KATAPUSAN>
```

The integer readability label (1, 2, or 3) is embedded as the second field and does not require
inference from a path. The loader splits on the first two commas only:

```python
import pandas as pd

def load_ara_language(filepath: str, language: str) -> pd.DataFrame:
    records = []
    with open(filepath, encoding='utf-8', errors='replace') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(',', 2)
            if len(parts) < 3:
                continue
            records.append({
                'language': language,
                'label': int(parts[1].strip()),
                'title': parts[0].strip(),
                'text': parts[2].strip(),
            })
    return pd.DataFrame(records)

df_ceb = load_ara_language('ara-close-lang/data/cebuano/ceb_all_data.txt', 'cebuano')
df_bik = load_ara_language('ara-close-lang/data/bikol/bik_all_data.txt', 'bikol')
```

**Tagalog raw-text restriction.** Inspection of the repository confirms that
`/data/tagalog/` contains only pre-extracted numerical feature files
(`tag_features.csv`, `syll.csv`, `trad.csv`, `clgsngo.csv`, `tag_mbert_features.csv`);
no raw `tag_all_data.txt` is distributed, because the original Tagalog corpus was compiled
from Adarna House publications under a restricted license. Because the KayaBasa hybrid
architecture requires raw text to compute XLM-RoBERTa embeddings, a substitute Tagalog
corpus must be curated. This curation is the single largest manual workload in the data
processing stage and proceeds under the following constraints:

- **Sourcing.** Texts are drawn only from Bloom Library (bloomlibrary.org) and Let's Read
  Asia (letsreadasia.org), replicating the sourcing methodology of the original collection
  pipelines [16, 17]. Only Creative-Commons-licensed materials permitting non-commercial
  academic use are retained.
- **Format parity.** The substitute corpus is written to
  `ara-close-lang/data/tagalog/tag_all_data.txt` in the exact one-document-per-line,
  comma-separated schema above, with the integer label as the second field.
- **Distributional parity.** The target is approximately 265 documents with L1/L2/L3
  proportions matching the original feature file (72/96/97). Majority-class overage is
  corrected via `sklearn.utils.resample()`.
- **Label alignment.** Publisher reading-level metadata is cross-validated against the
  per-tier means and standard deviations of Average Sentence Length, Average Word Length,
  and Syllable Count per Word reported in the original `tag_features.csv`. Documents
  deviating by more than one standard deviation from their assigned tier are reclassified
  or excluded.

Until the substitute corpus passes the label-alignment check, Tagalog is excluded from the
high-resource training set and the pipeline runs on Cebuano and Bikol only; this allows the
remaining stages to be developed and validated in parallel with corpus curation.

### 3.1.1.2 BasahaCorpus Dataset

**Repository structure.** The BasahaCorpus repository stores raw texts under
`/data/raw/<language>/` rather than directly under `/data/<language>/`. Each language
directory contains three grade-level subdirectories named `grade 1`, `grade 2`, and
`grade 3` (lowercase, with a space), **not** `L1/L2/L3`. The four target language
directories are `hiligaynon`, `minasbate`, `karay-a` (hyphenated), and `rinconada`. Each
document is a separate `.txt` file whose readability label is determined by its grade-level
subdirectory. Feature extraction scripts are provided under `/code/feature extraction/`.

**Loader.** Documents are loaded by walking each language's grade-level subdirectories and
assigning the label from the directory name. The corrected loader uses the actual paths and
records the source path for traceability:

```python
import os, glob
import pandas as pd

def load_basaha_language(base_dir: str, language: str) -> pd.DataFrame:
    records = []
    for grade_str, label_int in [('grade 1', 1), ('grade 2', 2), ('grade 3', 3)]:
        level_dir = os.path.join(base_dir, grade_str)
        if not os.path.isdir(level_dir):
            continue
        for fpath in glob.glob(os.path.join(level_dir, '*.txt')):
            with open(fpath, encoding='utf-8', errors='replace') as f:
                text = f.read()
            records.append({
                'language': language,
                'label': label_int,
                'title': os.path.basename(fpath)[:-4],
                'text': text,
                'source_path': fpath,
            })
    return pd.DataFrame(records)

base = 'BasahaCorpus-HierarchicalCrosslingualARA/data/raw'
lang_dirs = {
    'hiligaynon': os.path.join(base, 'hiligaynon'),
    'minasbate':  os.path.join(base, 'minasbate'),
    'karay-a':    os.path.join(base, 'karay-a'),
    'rinconada':  os.path.join(base, 'rinconada'),
}
df_basaha = pd.concat(
    [load_basaha_language(d, lang) for lang, d in lang_dirs.items()],
    ignore_index=True
)
```

**Deduplication and count reconciliation.** The corpus contains literal duplicate files
(for example, a Hiligaynon Grade 1 story appears both as `…Hiligaynon.txt` and
`…Hiligaynon (1).txt`). Exact-text duplicates are removed before any further processing:

```python
df_basaha = df_basaha.drop_duplicates(subset=['language', 'text']).reset_index(drop=True)
```

The on-disk document counts after deduplication are recomputed empirically and reported in
the final corpus summary table rather than copied from earlier proposal drafts, because the
disk counts diverge from previously stated figures (for example, Minasbate Grade 1 yields 123
files on disk against a previously stated 124). The Results chapter reports the reconciled
counts produced by the loader, not the planning estimates.

### 3.1.1.3 Combined Dataset, Identifiers, and Role Assignment

The two DataFrames are concatenated after verifying identical column schemas. Two columns
are added before any text mutation:

- `doc_id`: a stable, unique integer assigned per document. This identifier is the join key
  used to merge feature tables in 3.1.4, replacing positional concatenation.
- `split_role`: `'high_resource'` for ara-close-lang languages (primary training set) and
  `'low_resource'` for BasahaCorpus languages (cross-lingual transfer test set only).

A verbatim copy of the raw text is preserved in a separate column **before** cleaning, because
the denoising impact experiment in 3.1.5 requires both a noisy and a denoised version of the
same documents:

```python
df_all = pd.concat([df_ceb, df_bik, df_basaha], ignore_index=True)
df_all.insert(0, 'doc_id', range(len(df_all)))
df_all['split_role'] = df_all['language'].map(
    lambda l: 'high_resource' if l in {'tagalog', 'cebuano', 'bikol'} else 'low_resource'
)
df_all['text_raw'] = df_all['text']   # frozen pre-cleaning copy for 3.1.5
```

---

## 3.1.2 Data Cleaning

Cleaning removes non-narrative structural elements. Because the two corpora have different
noise profiles, cleaning is corpus-specific and dispatched on `split_role`. All cleaning
operates on the `text` column; `text_raw` is never modified.

### 3.1.2.1 Encoding Repair

Both corpora contain mojibake and stray control characters from OCR and web export. Rather
than substituting invalid bytes with replacement characters, encoding is repaired with `ftfy`,
which reconstructs the intended characters and preserves the accented graphemes that carry
phonotactic signal in the target languages:

```python
import ftfy
df_all['text'] = df_all['text'].apply(lambda t: ftfy.fix_text(str(t)))
```

### 3.1.2.2 ara-close-lang Cleaning

The text field concatenates a repeated title, author and illustrator credits, the narrative body,
and a `KATAPUSAN` marker. The credit-removal pattern is bounded by name-token structure
rather than by the first punctuation mark, so that periods inside author initials (for example,
"Joan P. Sanchez") do not prematurely terminate the match. Genre tags are matched across
all source languages rather than Cebuano alone:

```python
import regex  # supports \p{L} Unicode property classes

CREDIT_TRIGGER = (
    r'(?i)(sinulat|gisulat|gidibuho|isinulat|hinikay|sinulatan|'
    r'written\s+by|illustrated\s+by)\s*(ni|by)?\s*:?\s*'
)
# consume 1-6 capitalized name tokens, tolerating single-letter initials with periods
NAME_RUN = r'(?:\p{Lu}[\p{L}.\-]*\.?\s*){1,6}'

GENRE_TAG = (
    r'(?i)(cebuano|bikol|bikolano|tagalog|hiligaynon|minasbate[nñ]?o?|karay-?a|rinconada)\s+'
    r'(personal\s+development|story\s*book|community\s+living|science|'
    r'mathematics|mother\s+tongue|reader)[^.!?\n]*'
)

def clean_ara_text(text: str, title: str) -> str:
    # 1. Remove the repeated title at the start of the field
    text = regex.sub(r'^\s*' + regex.escape(title) + r'\s*', '', text)
    # 2. Remove author / illustrator credit runs
    text = regex.sub(CREDIT_TRIGGER + NAME_RUN, '', text)
    # 3. Remove publisher / DepEd genre tags for any source language
    text = regex.sub(GENRE_TAG, '', text)
    # 4. Remove the KATAPUSAN end-of-document marker
    text = regex.sub(r'\bKATAPUSAN\b', '', text)
    # 5. Remove residual HTML character entities
    text = regex.sub(r'&quot;', '"', text)
    text = regex.sub(r'&amp;', '&', text)
    text = regex.sub(r'&#\d+;', '', text)
    text = regex.sub(r'&[a-zA-Z]+;', '', text)
    # 6. Remove URLs
    text = regex.sub(r'https?://\S+', '', text)
    # 7. Collapse excess whitespace
    text = regex.sub(r'[ \t]+', ' ', text)
    text = regex.sub(r'\n{2,}', '\n', text)
    return text.strip()
```

The credit-removal step is acknowledged as heuristic: it is bounded to a maximum of six name
tokens to limit the risk of consuming narrative text, and its output is verified by the manual
sampling procedure in 3.1.2.4.

### 3.1.2.3 BasahaCorpus Cleaning

Each BasahaCorpus document is a full Let's Read Asia export in which the narrative body is
followed by English-language boilerplate. Cleaning truncates the document at the first
boilerplate sentinel, then removes navigation artefacts. The sentinel list is expanded, and any
document in which **no** sentinel matches is flagged for manual inspection rather than silently
retained, because an unmatched document risks carrying English boilerplate into the training
signal:

```python
BASAHA_NOISE_SENTINELS = [
    r"Frog.?s\s+Exercise", r"About\s+Cat\s+and\s+Dog",
    r"Let.?s\s+Read\s+is\s+an\s+initiative", r"Table\s+of\s+Contents",
    r"Brought\s+to\s+you\s+by", r"Original\s+Story",
    r"For\s+full\s+terms\s+of\s+use", r"Contributing\s+translators",
    r"This\s+book\s+was\s+(made|brought)", r"Asia\s+Foundation",
    r"All\s+rights\s+reserved", r"Creative\s+Commons",
]
SENTINEL_RE = regex.compile('|'.join(BASAHA_NOISE_SENTINELS), flags=regex.IGNORECASE)

# Strict name-line pattern: 1-4 capitalized tokens, no terminal punctuation,
# no Philippine function words. Used only for leading author lines.
FUNCTION_WORDS = regex.compile(
    r'(?i)\b(ang|si|ng|sa|na|at|kag|ug|ni|kay|nag|mag|mga|an|in|kan|su|si)\b'
)
NAME_LINE = regex.compile(r'^\s*(?:\p{Lu}[\p{L}.\-]+\s*){1,4}$')

def clean_basaha_text(text: str) -> tuple[str, bool]:
    matched = bool(SENTINEL_RE.search(text))
    if matched:
        text = text[:SENTINEL_RE.search(text).start()]
    # Remove leading author/title-only lines (first 3 lines, strict pattern only)
    lines = text.split('\n')
    out = []
    for i, line in enumerate(lines):
        s = line.strip()
        if i < 3 and s and NAME_LINE.match(s) and not FUNCTION_WORDS.search(s):
            continue
        out.append(line)
    text = '\n'.join(out)
    # Remove page / navigation artefacts
    text = regex.sub(r'^\s*Page\s+\d+\s*$', '', text, flags=regex.MULTILINE)
    text = regex.sub(r'^\s*(Guide|Cover|Start of Story|Copyright)\s*$', '',
                     text, flags=regex.MULTILINE | regex.IGNORECASE)
    text = regex.sub(r'\n{3,}', '\n\n', text)
    return text.strip(), matched
```

The leading-line removal is restricted to the first three lines and to a strict capitalized
name pattern that excludes any line containing a Philippine function word or terminal
punctuation, narrowing the earlier heuristic so that short opening narrative lines and one-word
titles are not erroneously deleted.

### 3.1.2.4 Cleaning Validation

Two automated integrity checks run over the full corpus, and one manual check is performed:

1. **Sentinel coverage (BasahaCorpus).** The proportion of documents for which a boilerplate
   sentinel matched is logged per language and per grade level. Any document with no match is
   added to a manual-review queue.
2. **Length sanity.** Documents whose cleaned length falls below a minimum token threshold
   (indicating over-deletion) or retains characteristic boilerplate tokens (indicating
   under-deletion) are flagged automatically.
3. **Stratified manual sample.** Ten documents per language (70 total) are inspected before and
   after cleaning to confirm that (a) the full narrative body is retained and (b) no metadata
   remains. Any false positive (narrative content removed) triggers a targeted pattern revision
   and a full re-run of the cleaning pass. Final UTF-8 standardization is applied after cleaning.

---

## 3.1.3 Data Normalization

Following cleaning, each text undergoes structural normalization to produce a consistent
machine-readable format. Because Philippine languages are highly agglutinative, encoding
tense, aspect, voice, and grammatical focus through affixation and reduplication, stemming and
lemmatization are deliberately not applied; stripping affixes would destroy the morphological
markers the model is designed to capture.

The normalization regexes are Unicode-aware. The earlier ASCII-only character classes
(`[A-Za-z]`) are replaced with the Unicode letter property `\p{L}`, so that ñ, the digraph
*ng*, accented vowels, and the Rinconada schwa are treated as letters rather than as
boundaries. Hyphen normalization is constrained to inter-word reduplication hyphens and does
not fuse dash punctuation used as a clause separator:

```python
import regex

def normalize_text(text: str) -> str:
    # 1. Decode any residual HTML entities
    text = regex.sub(r'&nbsp;', ' ', text)
    text = regex.sub(r'&amp;', '&', text)
    text = regex.sub(r'&#\d+;', '', text)
    # 2. Insert a missing space after closing punctuation before a letter (Unicode-aware)
    text = regex.sub(r'([,.!?;:"\)])\s*(?=\p{L})', r'\1 ', text)
    # 3. Join reduplication hyphens only when both sides are word characters
    #    e.g. "dali - dali" -> "dali-dali"; leaves " — " clause dashes intact
    text = regex.sub(r'(?<=\p{L})\s*-\s*(?=\p{L})', '-', text)
    text = regex.sub(r'(?<=\p{L})-{2,}(?=\p{L})', '-', text)
    # 4. Normalize spaced em/en dashes used as separators (do not fuse to words)
    text = regex.sub(r'\s*[—–]\s*', ' — ', text)
    # 5. Collapse internal whitespace
    text = regex.sub(r'[ \t]+', ' ', text)
    text = regex.sub(r' +\n', '\n', text)
    # 6. Morphological preservation: no stemming, no lemmatization.
    #    Native prefixes (nag-, mag-, naka-, maka-, gi-, pa-) and
    #    suffixes (-an, -in, -on, -han, -um-) are left fully intact.
    return text.strip()

df_all['text'] = df_all['text'].apply(normalize_text)
```

---

## 3.1.4 Feature Engineering

Three groups of linguistic features are extracted. The feature extractors are the rule-based
scripts shipped in each repository's code directory. These scripts as distributed have no
command-line interface, read a hardcoded input filename, and write hardcoded output filenames;
they cannot be driven by `--input`/`--output`/`--anchor` arguments. The pipeline therefore
imports the underlying feature functions directly and applies them per document under the
project's own driver, attaching the `doc_id` join key to every output row.

### 3.1.4.1 Feature Groups

**Phonotactic decoding load (`SYLL.py`).** The syllable module segments words into
consonant–vowel patterns and computes per-word densities for the canonical Philippine
structures. The shipped module exposes densities for the consonant cluster (`kk`), V, CV, VC,
CVC, VCC, CVCC, CCV, CCVC, CCVCC, and CCVCCC patterns; the full set is retained and the
selected subset is documented in the feature summary. The vowel and consonant inventories in
`SYLL.py` are ASCII-only (`[aeiou]`, `[bcdfghjklmnpqrstvwxyz]`); for the languages whose
orthographies use accented vowels or the Rinconada schwa, the inventories are extended to the
graphemes actually attested per language before counting, so that phonotactic load is not
systematically undercounted. Any residual orthography not covered is recorded as a limitation
in 3.5.2.

**Sentence-level surface statistics (`TRAD.py`).** The shipped module computes word count,
sentence count, phrase count per sentence, mean word length, mean sentence length, mean
syllable count per word, and polysyllabic word count. **Type-token ratio is not provided by the
repository script and is added by the project**, computed on alphabetic tokens after
normalization:

```python
from nltk import word_tokenize

def type_token_ratio(text: str) -> float:
    toks = [t.lower() for t in word_tokenize(text) if t.isalpha()]
    return len(set(toks)) / len(toks) if toks else 0.0
```

**Cross-lingual n-gram overlap (`CLGSNGO.py`).** The module computes character bigram and
trigram Rank-Biased Overlap against the three anchor languages (Tagalog, Bikol, Cebuano),
producing six similarity features per document. There is no single-anchor parameter; all three
anchor similarities are retained as features, consistent with the original BasahaCorpus design.
The module reads its anchor n-gram lists from a relative path `ngrams list/…` that does not
exist in the repository. Before extraction, the six path constants are repointed to the actual
repository locations (`ara-close-lang/data/ngrams/…`), or the anchor lists are copied into a
directory named exactly `ngrams list/` in the working directory. The `CLGSNGO` cleaner
removes non-ASCII characters before n-gram extraction; this behavior is preserved for
comparability with the published CROSSNGO feature, and its effect on accented orthographies
is noted as a limitation.

### 3.1.4.2 Extraction Driver and Feature Merge

The cleaned, normalized corpus is passed to each extractor per document, and the three
resulting tables are merged on `doc_id` rather than by row position, eliminating the silent
misalignment risk of positional concatenation:

```python
import sys, pandas as pd
sys.path.append('ara-close-lang/code')          # expose TRAD, SYLL, CLGSNGO
from TRAD import (word_count_per_doc, sentence_count_per_doc, ave_word_length,
                  word_count_per_sentence, ave_syllable_count_of_word,
                  polysyll_count_per_doc, ave_phrase_count_per_doc)
from SYLL import (get_consonant_cluster, get_cv, get_cvc, get_ccvc, get_ccvccc)
from CLGSNGO import get_bigram_CLGSNGO, get_trigram_CLGSNGO

def extract_features(row) -> dict:
    t = row['text']
    feats = {'doc_id': row['doc_id']}
    # surface statistics
    feats['word_count']        = word_count_per_doc(t)
    feats['sentence_count']    = sentence_count_per_doc(t)
    feats['mean_word_len']     = ave_word_length(t)
    feats['mean_sentence_len'] = word_count_per_sentence(t)
    feats['mean_syll_per_word']= ave_syllable_count_of_word(t)
    feats['polysyll_count']    = polysyll_count_per_doc(t)
    feats['type_token_ratio']  = type_token_ratio(t)
    # phonotactic densities (subset retained for the study)
    feats['cv']      = get_cv(t)
    feats['cvc']     = get_cvc(t)
    feats['ccvc']    = get_ccvc(t)
    feats['ccvccc']  = get_ccvccc(t)
    feats['cons_cluster'] = get_consonant_cluster(t)
    # cross-lingual n-gram overlap (3 anchors x 2 orders = 6 features)
    feats['tag_bi'], feats['bik_bi'], feats['ceb_bi'] = get_bigram_CLGSNGO(t)
    feats['tag_tri'], feats['bik_tri'], feats['ceb_tri'] = get_trigram_CLGSNGO(t)
    return feats

feature_rows = [extract_features(r) for _, r in df_all.iterrows()]
X_features = pd.DataFrame(feature_rows)
# merge on the stable key, carrying label and language for downstream use
X_features = X_features.merge(
    df_all[['doc_id', 'label', 'language', 'split_role']], on='doc_id', how='left'
)
X_features.to_csv('features/all_features.csv', index=False)
```

### 3.1.4.3 Summary of Extracted Features

| Feature group | Features extracted | Source |
|---|---|---|
| Phonotactic decoding load | CV, CVC, CCVC, CCVCCC densities; consonant-cluster density | `SYLL.py` (inventory extended per language) |
| Surface statistics | word count, sentence count, mean word length, mean sentence length, mean syllables/word, polysyllabic count, **type-token ratio** | `TRAD.py` + project addition |
| Cross-lingual n-gram overlap | bigram and trigram RBO vs. Tagalog, Bikol, Cebuano anchors (6 features) | `CLGSNGO.py` (anchor paths repointed) |

---

## 3.1.5 Denoising Impact Quantification

To justify the cleaning and normalization pipeline as a substantive pipeline component rather
than a cosmetic step, a controlled experiment compares classification performance on noisy
versus denoised text. The term *denoising* in this study refers specifically to this corpus
cleaning and normalization pipeline; it is distinct from the masked-language-modeling
denoising objective inherited from XLM-RoBERTa pre-training, which is not trained by this
study. This distinction is stated explicitly to avoid conflating the two.

**Experimental design.** Two versions of the high-resource corpus are prepared from the
frozen `text_raw` column and the processed `text` column: a *noisy* version (raw text after
ingestion only, bypassing cleaning and normalization) and a *denoised* version (the fully
cleaned and normalized corpus from 3.1.2 and 3.1.3). Both undergo identical feature extraction
(3.1.4) and are evaluated under identical Stratified 5-Fold conditions (3.1.6) using a
feature-only Random Forest classifier, isolating the effect of preprocessing from any
transformer contribution. The Random Forest used here is the same configuration as the
feature-only baseline defined for the ablation study, instantiated at this stage:

```python
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators=300, random_state=42)
```

**Metrics reported.** For each fold and each high-resource language: Macro-Averaged F1-Score
on the validation fold, per-class F1 for L1/L2/L3, and the non-adjacent misclassification rate
(for example, L1 predicted as L3), which is the most harmful error type in a readability grading
context. Means and standard deviations are reported across the five folds for both conditions.

**Acceptance criterion.** The primary evidence metric is ΔMacro-F1 = denoised − noisy.
Denoising is confirmed to constitute a meaningful pipeline contribution if ΔMacro-F1 ≥ 0.03
in **at least two of the three** high-resource languages (Tagalog, Cebuano, Bikol). This
corrects the earlier, logically impossible criterion of "four of the three languages."

---

## 3.1.6 Data Splitting

Stratified 5-Fold Cross-Validation is applied exclusively to the high-resource training set
(Tagalog, Cebuano, Bikol). Under a 5-fold scheme, each iteration uses 80% of the training
corpus for training and the remaining 20% for validation; the rotation is performed five times
so that every document serves as a validation instance exactly once, producing five independent
scores whose mean and standard deviation are reported. The 80/20 ratio maximizes the training
signal available to XLM-RoBERTa fine-tuning and the MLP under the constrained document
counts while reserving a held-out fold large enough for stable per-class F1 estimates.

A global random seed is fixed across all libraries before any splitting or model
initialization, and folds are stratified on the joint (language, label) key to preserve both
distributions:

```python
import random, numpy as np, torch, pickle
from sklearn.model_selection import StratifiedKFold

SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

df_train = df_all[df_all['split_role'] == 'high_resource'].copy()
df_train['stratify_key'] = df_train['language'] + '_' + df_train['label'].astype(str)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
fold_indices = [
    {'train': tr.tolist(), 'val': va.tolist()}
    for tr, va in skf.split(df_train, df_train['stratify_key'])
]
with open('splits/fold_indices.pkl', 'wb') as f:
    pickle.dump(fold_indices, f)
```

The saved fold indices are loaded at the start of every ablation run (feature-only,
transformer-only, full hybrid), ensuring results are directly comparable across
configurations. The BasahaCorpus low-resource languages (Hiligaynon, Minasbate, Karay-a,
Rinconada) are held entirely outside the cross-validation loop and are used only in the Phase 2
cross-lingual transfer evaluation, with no additional fine-tuning.

**Per-language evaluation commitment.** Evaluation results are reported separately for each of
the four low-resource languages before any cross-language averaging. For each language, the
Results chapter presents Precision, Recall, and F1-Score per class, Macro-Averaged F1, overall
Accuracy, and a normalized confusion matrix. Consolidated seven-language averages are computed
and reported only after the individual language results have been discussed.

---

## 3.1.7 Data Processing Outputs and Integrity Verification

This subsection defines the concrete artefacts that constitute the deliverables of the data
processing stage and the integrity checks that gate their acceptance. These are the outputs
required this term.

**Deliverables.**

1. **Combined corpus (serialized).** A single serialized DataFrame (`df_all`) containing, per
   document: `doc_id`, `language`, `label`, `title`, cleaned-and-normalized `text`, frozen
   `text_raw`, `source_path`, and `split_role`. This is the canonical corpus referenced by all
   downstream code.
2. **Feature matrix.** `features/all_features.csv`, keyed on `doc_id`, containing the surface,
   phonotactic, and cross-lingual n-gram features defined in 3.1.4.3, with `label`, `language`,
   and `split_role` merged in.
3. **Denoising impact table.** A per-language table of ΔMacro-F1 (denoised − noisy) with means
   and standard deviations across five folds, plus the non-adjacent misclassification rates, as
   specified in 3.1.5.
4. **Saved cross-validation splits.** `splits/fold_indices.pkl`, generated once under seed 42 and
   reused by every ablation configuration.
5. **Substitute Tagalog corpus.** `ara-close-lang/data/tagalog/tag_all_data.txt`, formatted to
   the one-document-per-line schema and passing the label-alignment check of 3.1.1.1, together
   with its sourcing-and-licensing log.
6. **Reconciled corpus summary tables.** Document, sentence, and vocabulary distributions
   recomputed from the loaded corpus after deduplication, superseding the planning-stage counts.

**Integrity verification.** Each deliverable is accepted only after the following checks pass and
are logged:

- **Schema and key integrity.** Every row carries a unique `doc_id`; the feature matrix and the
  corpus contain identical `doc_id` sets; no duplicate `(language, text)` pairs remain.
- **Label integrity.** Per-language L1/L2/L3 counts in the loaded corpus match the reconciled
  summary tables; no document carries a label outside {1, 2, 3}.
- **Cleaning integrity.** The 70-document stratified manual sample passes the
  retain-narrative / remove-metadata check; the BasahaCorpus sentinel-coverage log contains no
  unreviewed unmatched documents.
- **Feature sanity.** No feature column is entirely null or constant; densities lie in plausible
  ranges; the type-token ratio lies in (0, 1]; cross-lingual overlap features are populated for
  all documents (confirming the anchor n-gram paths resolved correctly).
- **Reproducibility.** Re-running the full pipeline from the cloned repositories under seed 42
  reproduces identical `all_features.csv` and identical fold indices, traceable to a tagged
  commit of the project repository.

---

## Appendix 3.1.A — Summary of Revisions and the Defects They Address

| Section | Revision | Defect corrected |
|---|---|---|
| 3.1.1.0 | Added NLTK and library prerequisites | `word_tokenize` raised `LookupError`; `regex`/`ftfy` undeclared |
| 3.1.1.1 | Tagalog stated as raw-text-absent; substitute corpus framed as the primary manual task; pipeline runs on Cebuano+Bikol until ready | `tag_all_data.txt` does not exist in the repo |
| 3.1.1.2 | Corrected BasahaCorpus paths (`data/raw/`, `grade 1/2/3`, `karay-a`); added dedup; empirical count reconciliation | Loader returned zero documents; duplicates and count mismatches |
| 3.1.1.3 | Added `doc_id`, `split_role`, frozen `text_raw` | No join key; in-place mutation destroyed the raw text needed by 3.1.5 |
| 3.1.2.1 | Encoding repair via `ftfy` instead of `errors='replace'` | Replacement characters injected into features |
| 3.1.2.2 | Credit regex bounded by name-token structure; genre tags generalized to all languages | Periods in initials truncated the match; Cebuano-only genre pattern |
| 3.1.2.3 | Expanded sentinel list; unmatched-document flagging; stricter leading-line removal | Boilerplate leakage; over-deletion of opening narrative |
| 3.1.3 | Unicode-aware classes (`\p{L}`); constrained hyphen joining | ASCII-only regex mishandled ñ, accents, schwa; dash fusion |
| 3.1.4 | Direct function import instead of fictional CLI; added type-token ratio; repointed CLGSNGO anchor paths; per-language phonotactic inventory; merge on `doc_id` | Parser scripts have no CLI; TTR absent; broken anchor paths; positional-merge misalignment |
| 3.1.5 | Random Forest instantiated here; corrected acceptance criterion; uses frozen raw copy | RF undefined; "four of three" was impossible; raw copy unavailable |
| 3.1.6 | Retained; clarified high-resource set | (No material defect) |
| 3.1.7 | New section: enumerated deliverables and integrity gates | Outputs and acceptance criteria for the term were unspecified |
