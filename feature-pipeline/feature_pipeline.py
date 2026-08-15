"""
feature_pipeline.py — KAYABASA §2.3.4 / §3.2.2 Regex Feature Extraction Pipeline
==================================================================================
Implements the *exact* three feature groups defined in the thesis (Table VII).
All extra features that are not in the paper have been removed.

Feature groups (14 features total — matches Table VII exactly):
  ┌──────────────────────────────┬─────────────────────────────────────────────┐
  │ Group                        │ Features                                    │
  ├──────────────────────────────┼─────────────────────────────────────────────┤
  │ TRAD — Sentence-level        │ mean_sentence_len, mean_word_len,           │
  │ surface statistics (4)       │ polysyll_freq, type_token_ratio             │
  ├──────────────────────────────┼─────────────────────────────────────────────┤
  │ SYLL — Phonotactic decoding  │ syll_cv, syll_cvc, syll_ccvc, syll_ccvccc  │
  │ load (4)                     │ (canonical Philippine CV-pattern inventory) │
  ├──────────────────────────────┼─────────────────────────────────────────────┤
  │ CLGSNGO — Cross-lingual      │ tag_bi, bik_bi, ceb_bi,                    │
  │ n-gram overlap (6)           │ tag_tri, bik_tri, ceb_tri                  │
  └──────────────────────────────┴─────────────────────────────────────────────┘

Design principles
-----------------
  • Pure Python + regex only.  No spaCy, no Stanza, no Filipino NLP library.
    This is the correct choice for a low-resource cross-lingual setting:
    morphological parsers do not exist for Minasbate, Karay-a, or Rinconada.
    Regex feature extractors follow Imperial & Ong (2021) and Imperial &
    Kochmar (2023) exactly.

  • No stemming or lemmatization.  Affixes are the readability signal for
    agglutinative Philippine languages (§3.1.3 normalization principle).

  • Per-language phonotactic inventories.  The SYLL extractor uses per-language
    vowel sets so that accented vowels (á, é, í, ó, ú) and the Rinconada
    schwa (ə) are counted rather than silently dropped (§3.1.4.1).

  • Reproducible.  No randomness; given the same text, always returns the
    same feature vector. Safe to call from any fold in 5-fold CV.

Usage
-----
  from feature_pipeline import FeaturePipeline, FEATURE_COLS

  pipe = FeaturePipeline()
  X = pipe.fit_transform(df)   # df must have columns: doc_id, text, label,
                               #   language, split_role

  X[FEATURE_COLS]   # 14-column numeric input to the MLP

References
----------
  [10] Imperial & Ong (2021) — Random Forest + handcrafted features for
       Filipino ARA; established locally-adapted feature engineering.
  [25] Reyes et al. (2022) — SVM/RF baseline for Cebuano ARA; validated
       surface and phonotactic features for Central Philippine languages.
  [35] Imperial & Kochmar (2023) — BasahaCorpus; cross-lingual ARA with
       CROSSNGO feature set for closely related Philippine languages.
"""

from __future__ import annotations

import os
import re
from collections import Counter
from typing import Dict, List

import pandas as pd

# ---------------------------------------------------------------------------
# Default CLGSNGO anchor directory
# (relative to this file so the module is path-agnostic)
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_NGRAM_DIR = os.path.join(_HERE, "ngrams list")


# ===========================================================================
# 1.  Per-language orthographic inventories
# ===========================================================================
# The SYLL extractor uses per-language vowel sets so that accented vowels and
# the Rinconada schwa (ə / ǝ) are counted as vowels, not ignored.
# ---------------------------------------------------------------------------

_ASCII_VOWELS  = set("aeiou")
_ACCENT_VOWELS = set("áéíóúàèìòùâêîôûäëïöü")

EXTENDED_VOWELS: Dict[str, frozenset] = {
    "tagalog":    frozenset(_ASCII_VOWELS | _ACCENT_VOWELS),
    "cebuano":    frozenset(_ASCII_VOWELS | {"á", "é", "í", "ó", "ú"}),
    "bikol":      frozenset(_ASCII_VOWELS | {"á", "é", "í", "ó", "ú"}),
    "hiligaynon": frozenset(_ASCII_VOWELS | {"á", "é", "í", "ó", "ú"}),
    "minasbate":  frozenset(_ASCII_VOWELS | {"á", "é", "í", "ó", "ú"}),
    "karay-a":    frozenset(_ASCII_VOWELS | {"á", "é", "í", "ó", "ú"}),
    "rinconada":  frozenset(_ASCII_VOWELS | {"á", "é", "í", "ó", "ú", "ə", "ǝ"}),
}
_DEFAULT_VOWELS = frozenset(_ASCII_VOWELS | _ACCENT_VOWELS)


# ---------------------------------------------------------------------------
# Feature column names — stable identifiers used by model training code
# These 14 names match Table VII of the thesis exactly.
# ---------------------------------------------------------------------------

FEATURE_COLS: List[str] = [
    # TRAD — sentence-level surface statistics (4)
    "mean_sentence_len",
    "mean_word_len",
    "polysyll_freq",
    "type_token_ratio",
    # SYLL — phonotactic decoding load, canonical Philippine patterns (4)
    "syll_cv",
    "syll_cvc",
    "syll_ccvc",
    "syll_ccvccc",
    # CLGSNGO — cross-lingual character n-gram overlap (6)
    "tag_bi",
    "bik_bi",
    "ceb_bi",
    "tag_tri",
    "bik_tri",
    "ceb_tri",
]

assert len(FEATURE_COLS) == 14, f"Expected 14 features, got {len(FEATURE_COLS)}"


# ===========================================================================
# 2.  Low-level utility functions
# ===========================================================================

def _alpha_tokens(text: str) -> List[str]:
    """Alphabetic word tokens (lowercased), preserving non-ASCII letters."""
    # Regex split on non-alpha characters; keep tokens that contain only letters
    # (including accented chars and the Rinconada schwa)
    tokens = re.findall(r"[a-záéíóúàèìòùâêîôûäëïöüəǝ]+", text.lower())
    return tokens


def _split_sentences(text: str) -> List[str]:
    """Split text into sentences using terminal punctuation as delimiters."""
    # Split on . ? ! followed by whitespace or end of string
    sents = re.split(r"(?<=[.?!])\s+", text.strip())
    return [s for s in sents if s.strip()]


def _count_syllables(word: str, vowels: frozenset) -> int:
    """Count syllables by counting vowel-cluster nuclei."""
    count = 0
    in_vowel = False
    for ch in word.lower():
        is_v = ch in vowels
        if is_v and not in_vowel:
            count += 1
        in_vowel = is_v
    return max(count, 1)


# ===========================================================================
# 3.  TRAD — Sentence-level surface statistics  (§3.2.2.2 / trad_parser.py)
# ===========================================================================
# Paper-defined features (4):
#   mean_sentence_len   — mean number of words per sentence
#   mean_word_len       — mean number of characters per word
#   polysyll_freq       — proportion of words with ≥ 3 syllables
#   type_token_ratio    — vocabulary richness (unique words / total words)
# ---------------------------------------------------------------------------

class _TRADExtractor:
    def extract(self, text: str, vowels: frozenset = _DEFAULT_VOWELS) -> dict:
        sents  = _split_sentences(text)
        words  = _alpha_tokens(text)
        n_sents = max(len(sents), 1)
        n_words = len(words)

        mean_sentence_len = n_words / n_sents if n_sents else float(n_words)
        mean_word_len     = (sum(len(w) for w in words) / n_words
                             if n_words else 0.0)

        syll_counts = [_count_syllables(w, vowels) for w in words]
        polysyll_freq = (sum(1 for s in syll_counts if s >= 3) / n_words
                         if n_words else 0.0)

        type_token_ratio = (len(set(words)) / n_words if words else 0.0)

        return {
            "mean_sentence_len": mean_sentence_len,
            "mean_word_len":     mean_word_len,
            "polysyll_freq":     polysyll_freq,
            "type_token_ratio":  type_token_ratio,
        }


# ===========================================================================
# 4.  SYLL — Phonotactic decoding load  (§3.2.2.1 / syll_parse.py)
# ===========================================================================
# Paper-defined features (4) — the 4 canonical Philippine syllable patterns:
#   syll_cv      — proportion of words matching CV pattern  (e.g. "ma")
#   syll_cvc     — proportion matching CVC                  (e.g. "mag")
#   syll_ccvc    — proportion matching CCVC                 (e.g. "krus")
#   syll_ccvccc  — proportion matching CCVCCC               (e.g. "strikto")
#
# Note: the paper (§2.3.4) lists exactly CV, CVC, CCVC, CCVCCC. Other
# patterns present in the original code (V, VC, VCC, CVCC, CCV, CCVCC,
# cons_cluster) are NOT in the thesis and have been removed.
# ---------------------------------------------------------------------------

def _cv_pattern(word: str, vowels: frozenset) -> str:
    """Convert a word to its C/V skeleton, e.g. 'mag' → 'CVC'."""
    result = []
    for ch in word.lower():
        if ch in vowels:
            result.append("V")
        elif ch.isalpha() or ch in ("ə", "ǝ"):
            result.append("C")
    return "".join(result)


class _SYLLExtractor:
    # Only the 4 patterns named in the thesis (Table VII)
    _PATTERNS = {
        "syll_cv":      re.compile(r"^CV$"),
        "syll_cvc":     re.compile(r"^CVC$"),
        "syll_ccvc":    re.compile(r"^CCVC$"),
        "syll_ccvccc":  re.compile(r"^CCVCCC$"),
    }

    def extract(self, text: str, vowels: frozenset = _DEFAULT_VOWELS) -> dict:
        words = _alpha_tokens(text)
        n = len(words)
        if n == 0:
            return {k: 0.0 for k in self._PATTERNS}

        counts = {k: 0 for k in self._PATTERNS}
        for w in words:
            skeleton = _cv_pattern(w, vowels)
            for feat, pat in self._PATTERNS.items():
                if pat.fullmatch(skeleton):
                    counts[feat] += 1

        return {k: v / n for k, v in counts.items()}


# ===========================================================================
# 5.  CLGSNGO — Cross-lingual character n-gram overlap  (§3.2.2.3)
# ===========================================================================
# Paper-defined features (6):
#   tag_bi, bik_bi, ceb_bi   — character bigram  RBO overlap
#   tag_tri, bik_tri, ceb_tri — character trigram RBO overlap
#
# Anchor lists are the top-25% most frequent n-grams from each high-resource
# language corpus, pre-computed and stored in ngrams list/.
# ---------------------------------------------------------------------------

def _clgsngo_clean(text: str) -> str:
    """Lowercase and strip non-alpha chars — matches upstream CLGSNGO.clean()."""
    return re.sub(r"[^a-z]", "", text.lower())


def _get_ngrams(text: str, n: int) -> Counter:
    return Counter(text[i:i+n] for i in range(len(text) - n + 1))


def _rbo_overlap(doc_counts: Counter, anchor_list: List[str],
                 p: float = 0.98) -> float:
    """Rank-Biased Overlap between document n-gram ranks and anchor ranks."""
    if not doc_counts or not anchor_list:
        return 0.0
    doc_ranked = [gram for gram, _ in doc_counts.most_common()]
    score = 0.0
    depth = min(len(anchor_list), len(doc_ranked))
    anchor_set: set = set()
    doc_set:    set = set()
    for d in range(1, depth + 1):
        anchor_set.add(anchor_list[d - 1])
        doc_set.add(doc_ranked[d - 1])
        overlap = len(anchor_set & doc_set) / d
        score  += (p ** (d - 1)) * overlap
    return (1 - p) * score


class _CLGSNGOExtractor:
    _ANCHOR_FILES = {
        "tagalog": ("tag_top25_2gram.txt", "tag_top25_3gram.txt"),
        "bikol":   ("bik_top25_2gram.txt", "bik_top25_3gram.txt"),
        "cebuano": ("ceb_top25_2gram.txt", "ceb_top25_3gram.txt"),
    }

    def __init__(self, ngram_dir: str):
        self._ngram_dir = ngram_dir
        self._anchors: Dict[str, Dict[int, List[str]]] = {}
        self._load_anchors()

    def _load_anchors(self):
        for lang, (bi_file, tri_file) in self._ANCHOR_FILES.items():
            self._anchors[lang] = {}
            for order, fname in ((2, bi_file), (3, tri_file)):
                path = os.path.join(self._ngram_dir, fname)
                if not os.path.exists(path):
                    raise FileNotFoundError(
                        f"CLGSNGO anchor file not found: {path}\n"
                        f"Expected in: {self._ngram_dir}"
                    )
                grams = []
                with open(path, encoding="utf-8") as f:
                    for line in f:
                        parts = line.strip().split(",")
                        if parts and parts[0]:
                            grams.append(parts[0])
                self._anchors[lang][order] = grams

    def extract(self, text: str) -> dict:
        cleaned  = _clgsngo_clean(text)
        bigrams  = _get_ngrams(cleaned, 2)
        trigrams = _get_ngrams(cleaned, 3)
        result   = {}
        for lang_key, prefix in [("tagalog", "tag"), ("bikol", "bik"),
                                  ("cebuano", "ceb")]:
            result[f"{prefix}_bi"]  = _rbo_overlap(bigrams,
                                                    self._anchors[lang_key][2])
            result[f"{prefix}_tri"] = _rbo_overlap(trigrams,
                                                    self._anchors[lang_key][3])
        return result


# ===========================================================================
# 6.  FeaturePipeline — public API
# ===========================================================================

class FeaturePipeline:
    """Extracts the 14-feature KAYABASA feature set from a corpus DataFrame.

    Parameters
    ----------
    ngram_dir : str, optional
        Path to the CLGSNGO anchor n-gram files directory.
        Defaults to the ``ngrams list/`` folder beside this script.

    Example
    -------
    >>> pipe = FeaturePipeline()
    >>> X = pipe.fit_transform(df)
    >>> X[FEATURE_COLS]   # 14-column numeric DataFrame for the MLP
    """

    def __init__(self, ngram_dir: str = _DEFAULT_NGRAM_DIR):
        self._trad = _TRADExtractor()
        self._syll = _SYLLExtractor()
        self._clgs = _CLGSNGOExtractor(ngram_dir=ngram_dir)

    def fit(self, df: pd.DataFrame) -> "FeaturePipeline":
        required = {"doc_id", "text", "label", "language", "split_role"}
        missing  = required - set(df.columns)
        if missing:
            raise ValueError(f"DataFrame missing columns: {missing}")
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        rows  = []
        total = len(df)
        for i, (_, row) in enumerate(df.iterrows()):
            if (i + 1) % 100 == 0 or (i + 1) == total:
                print(f"  [{i+1}/{total}] extracting features...", flush=True)
            rows.append(self._extract_one(row))

        X = pd.DataFrame(rows)
        X = X.merge(
            df[["doc_id", "label", "language", "split_role"]],
            on="doc_id", how="left",
        )
        self._sanity_checks(X)
        return X

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.fit(df).transform(df)

    # ── Internal ────────────────────────────────────────────────────────────

    def _extract_one(self, row: pd.Series) -> dict:
        text     = str(row["text"])
        language = str(row.get("language", "")).lower()
        vowels   = EXTENDED_VOWELS.get(language, _DEFAULT_VOWELS)

        feats: dict = {"doc_id": row["doc_id"]}
        feats.update(self._trad.extract(text, vowels))
        feats.update(self._syll.extract(text, vowels))
        feats.update(self._clgs.extract(text))
        return feats

    @staticmethod
    def _sanity_checks(X: pd.DataFrame) -> None:
        feat_cols = [c for c in X.columns
                     if c not in ("doc_id", "label", "language", "split_role")]

        # No entirely-null columns
        all_null = X[feat_cols].isnull().all(axis=0)
        if all_null.any():
            raise AssertionError(
                f"Entirely-null columns: {list(all_null[all_null].index)}")

        # Warn on constant columns (no discriminative power)
        constant = X[feat_cols].nunique() == 1
        if constant.any():
            print(f"  [WARNING] Constant columns: "
                  f"{list(constant[constant].index)}")

        # type_token_ratio must be in (0, 1]
        ttr_bad = ~X["type_token_ratio"].between(0, 1, inclusive="right")
        if ttr_bad.any():
            raise AssertionError(
                f"TTR out of (0,1] for doc_ids: "
                f"{X.loc[ttr_bad, 'doc_id'].tolist()}")

        # CLGSNGO scores should be non-negative
        clgs_cols = ["tag_bi", "bik_bi", "ceb_bi", "tag_tri", "bik_tri",
                     "ceb_tri"]
        if (X[clgs_cols] < 0).any(axis=None):
            raise AssertionError("Negative CLGSNGO overlap values found.")
        all_zero = (X[clgs_cols] == 0).all(axis=1)
        if all_zero.any():
            print(f"  [WARNING] {all_zero.sum()} docs have all-zero "
                  f"CLGSNGO overlap")

        # Ratio / density columns must be non-negative
        ratio_cols = [c for c in feat_cols
                      if c.endswith("_freq") or c.endswith("_ratio")
                      or c.startswith("syll_")]
        if (X[ratio_cols] < 0).any(axis=None):
            raise AssertionError("Negative ratio values found.")

        print(f"\n  [SANITY OK] {len(X)} docs × {len(feat_cols)} features. "
              f"No critical integrity violations.")


# ===========================================================================
# 7.  Corpus loader (shared with extract_features.py)
# ===========================================================================

SPLIT_ROLE = {
    "tagalog":    "high_resource",
    "cebuano":    "high_resource",
    "bikol":      "high_resource",
    "hiligaynon": "low_resource",
    "minasbate":  "low_resource",
    "karay-a":    "low_resource",
    "rinconada":  "low_resource",
}


def load_corpus(path: str, sep: str = "|") -> pd.DataFrame:
    """Read a ``language|level|text`` corpus file into a keyed DataFrame.

    Parameters
    ----------
    path : str
        Path to the normalized corpus file
        (e.g. output_normalized/all_languages.txt).
    sep : str
        Field separator (default ``|``).

    Returns
    -------
    DataFrame with columns: doc_id, language, label, text, split_role
    """
    records = []
    with open(path, encoding="utf-8") as f:
        lines = f.read().splitlines()
    for line in lines[1:]:          # skip header
        if not line.strip():
            continue
        parts = line.split(sep, 2)
        if len(parts) < 3:
            continue
        language, level, text = parts
        records.append({
            "language": language.strip(),
            "label":    int(level.strip()),
            "text":     text.strip(),
        })
    df = pd.DataFrame(records)
    df.insert(0, "doc_id", range(len(df)))
    df["split_role"] = df["language"].map(SPLIT_ROLE).fillna("unknown")
    return df
