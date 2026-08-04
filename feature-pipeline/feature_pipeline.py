"""
feature_pipeline.py — KAYABASA §3.1.4 Feature Engineering Pipeline
====================================================================
A self-contained, importable feature extraction module for the KAYABASA
hybrid readability model. Implements all three feature groups described in
§3.1.4.1 and §3.1.4.3, plus Philippine-specific morphological features
validated by Imperial & Ong (2021) and Reyes et al. (2022).

Feature groups (35 features total):
  ┌─────────────────────────────┬────────────────────────────────────────────┐
  │ Group                       │ Features                                   │
  ├─────────────────────────────┼────────────────────────────────────────────┤
  │ TRAD — Surface statistics   │ word_count, sentence_count, mean_word_len, │
  │ (8 features)                │ mean_sentence_len, mean_syll_per_word,     │
  │                             │ polysyll_count, phrase_count,              │
  │                             │ type_token_ratio                           │
  ├─────────────────────────────┼────────────────────────────────────────────┤
  │ SYLL — Phonotactic load     │ syll_v, syll_cv, syll_vc, syll_cvc,       │
  │ (11 features)               │ syll_vcc, syll_cvcc, syll_ccv, syll_ccvc, │
  │                             │ syll_ccvcc, syll_ccvccc, cons_cluster      │
  ├─────────────────────────────┼────────────────────────────────────────────┤
  │ CLGSNGO — Cross-lingual     │ tag_bi, bik_bi, ceb_bi,                   │
  │ n-gram overlap (6 features) │ tag_tri, bik_tri, ceb_tri                 │
  ├─────────────────────────────┼────────────────────────────────────────────┤
  │ MORPH — Philippine          │ affix_density, reduplication_rate,         │
  │ morphological features      │ prefix_density, suffix_density,            │
  │ (7 features)                │ ng_digraph_density, function_word_ratio,   │
  │                             │ hapax_ratio                                │
  ├─────────────────────────────┼────────────────────────────────────────────┤
  │ SYNT — Syntactic / narrative│ question_ratio, dialogue_ratio,            │
  │ structure (3 features)      │ punct_density                              │
  ├─────────────────────────────┴────────────────────────────────────────────┤
  │ Metadata (not model inputs) │ doc_id, label, language, split_role        │
  └─────────────────────────────┴────────────────────────────────────────────┘

Usage
-----
  from feature_pipeline import FeaturePipeline, FEATURE_COLS

  pipe = FeaturePipeline()
  X = pipe.fit_transform(df)           # df has columns: doc_id, text, label,
                                       #   language, split_role

  # X has columns: doc_id + FEATURE_COLS + label, language, split_role
  X[FEATURE_COLS]  # the 35-column numeric input to the MLP

Design principles
-----------------
  • Pure Python + regex only.  No spaCy, no Stanza, no Filipino NLP library.
    This is the correct choice for a low-resource cross-lingual setting:
    morphological parsers do not exist for Minasbate, Karay-a, or Rinconada.
    Regex feature extractors follow Imperial & Ong (2021) exactly.

  • No stemming or lemmatization.  Affixes are the readability signal for
    agglutinative Philippine languages (§3.1.3 normalization principle).

  • Per-language phonotactic inventories.  SYLL.py uses ASCII-only vowel/
    consonant sets; this module extends them per language so that accented
    vowels (á, é, í, ó, ú) and the Rinconada schwa (ə) are counted, not
    silently dropped (§3.1.4.1 inventory extension).

  • Reproducible.  No randomness; given the same text, always returns the
    same feature vector. Safe to call from any fold in 5-fold CV.

References
----------
  [10] Imperial & Ong (2021) — Random Forest + handcrafted features for
       Filipino ARA; established that Philippine languages require locally
       adapted feature engineering (affix density, syllable patterns).
  [25] Reyes et al. (2022) — SVM/RF baseline for Cebuano ARA; validated
       surface and phonotactic features for Central Philippine languages.
  [14] Imperial & Kochmar (2023) — cross-lingual transfer for closely
       related Philippine language pairs; motivates CLGSNGO feature group.
"""

from __future__ import annotations

import os
import re
import unicodedata
from collections import Counter
from typing import Dict, List

import nltk
import pandas as pd

# ---------------------------------------------------------------------------
# NLTK bootstrap (punkt tokenizer required for sentence splitting)
# ---------------------------------------------------------------------------
nltk.download("punkt",     quiet=True)
nltk.download("punkt_tab", quiet=True)

from nltk.tokenize import sent_tokenize, word_tokenize  # noqa: E402

# ---------------------------------------------------------------------------
# Default CLGSNGO anchor directory
# (relative to this file's location so the module is path-agnostic)
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_NGRAM_DIR = os.path.join(_HERE, "ngrams list")


# ===========================================================================
# 1.  Per-language orthographic inventories
# ===========================================================================
# SYLL.py uses ASCII-only sets: vowels = [aeiou], consonants = [bcdfg...xyz].
# This module extends them so that accented vowels and language-specific
# graphemes are counted rather than ignored.  The Rinconada schwa (ə) is a
# lexically distinctive grapheme in that language and must be treated as a
# vowel for syllabification to be correct.
# ---------------------------------------------------------------------------

_ASCII_VOWELS     = set("aeiou")
_ASCII_CONSONANTS = set("bcdfghjklmnpqrstvwxyz")
_ACCENT_VOWELS    = set("áéíóúàèìòùâêîôûäëïöü")

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
# ---------------------------------------------------------------------------

FEATURE_COLS: List[str] = [
    # TRAD — surface statistics (8)
    "word_count",
    "sentence_count",
    "mean_word_len",
    "mean_sentence_len",
    "mean_syll_per_word",
    "polysyll_count",
    "phrase_count",
    "type_token_ratio",
    # SYLL — phonotactic densities, full 11-pattern inventory (11)
    "syll_v",
    "syll_cv",
    "syll_vc",
    "syll_cvc",
    "syll_vcc",
    "syll_cvcc",
    "syll_ccv",
    "syll_ccvc",
    "syll_ccvcc",
    "syll_ccvccc",
    "cons_cluster",
    # CLGSNGO — cross-lingual character n-gram overlap (6)
    "tag_bi",
    "bik_bi",
    "ceb_bi",
    "tag_tri",
    "bik_tri",
    "ceb_tri",
    # MORPH — Philippine morphological features (7)
    "affix_density",
    "prefix_density",
    "suffix_density",
    "reduplication_rate",
    "ng_digraph_density",
    "function_word_ratio",
    "hapax_ratio",
    # SYNT — syntactic / narrative structure (3)
    "question_ratio",
    "dialogue_ratio",
    "punct_density",
]

assert len(FEATURE_COLS) == 35, f"Expected 35 features, got {len(FEATURE_COLS)}"


# ===========================================================================
# 2.  Low-level utility functions
# ===========================================================================

def _alpha_tokens(text: str) -> List[str]:
    """Alphabetic word tokens (lowercased), preserving non-ASCII letters."""
    return [t.lower() for t in word_tokenize(text)
            if all(c.isalpha() or c in ("ə", "ǝ") for c in t) and t]


def _count_syllables_in_word(word: str, vowels: frozenset) -> int:
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
# 3.  TRAD — Surface statistics  (§3.1.4.1 / TRAD.py parity)
# ===========================================================================

class _TRADExtractor:
    _PHRASE_BREAK = re.compile(r"[,;:]")

    def extract(self, text: str) -> dict:
        sents    = sent_tokenize(text)
        words    = _alpha_tokens(text)
        n_sents  = max(len(sents), 1)
        n_words  = len(words)

        word_count     = n_words
        sentence_count = len(sents)
        mean_word_len  = sum(len(w) for w in words) / n_words if n_words else 0.0
        mean_sentence_len = n_words / n_sents if n_sents else float(n_words)

        syll_counts      = [_count_syllables_in_word(w, _ASCII_VOWELS) for w in words]
        mean_syll_per_word = sum(syll_counts) / n_words if n_words else 0.0
        polysyll_count   = sum(1 for s in syll_counts if s >= 3)

        phrase_breaks  = sum(len(self._PHRASE_BREAK.findall(s)) for s in sents)
        phrase_count   = phrase_breaks / n_sents if n_sents else 0.0
        type_token_ratio = len(set(words)) / len(words) if words else 0.0

        return {
            "word_count":         word_count,
            "sentence_count":     sentence_count,
            "mean_word_len":      mean_word_len,
            "mean_sentence_len":  mean_sentence_len,
            "mean_syll_per_word": mean_syll_per_word,
            "polysyll_count":     polysyll_count,
            "phrase_count":       phrase_count,
            "type_token_ratio":   type_token_ratio,
        }


# ===========================================================================
# 4.  SYLL — Full phonotactic CV-pattern inventory  (§3.1.4.1 / SYLL.py)
# ===========================================================================

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
    _PATTERNS = {
        "syll_v":      re.compile(r"^V$"),
        "syll_cv":     re.compile(r"^CV$"),
        "syll_vc":     re.compile(r"^VC$"),
        "syll_cvc":    re.compile(r"^CVC$"),
        "syll_vcc":    re.compile(r"^VCC$"),
        "syll_cvcc":   re.compile(r"^CVCC$"),
        "syll_ccv":    re.compile(r"^CCV$"),
        "syll_ccvc":   re.compile(r"^CCVC$"),
        "syll_ccvcc":  re.compile(r"^CCVCC$"),
        "syll_ccvccc": re.compile(r"^CCVCCC$"),
    }
    _CONS_CLUSTER = re.compile(r"CC")

    def extract(self, text: str, vowels: frozenset = _DEFAULT_VOWELS) -> dict:
        words = _alpha_tokens(text)
        n = len(words)
        if n == 0:
            return {k: 0.0 for k in list(self._PATTERNS) + ["cons_cluster"]}

        counts = {k: 0 for k in list(self._PATTERNS) + ["cons_cluster"]}
        for w in words:
            skeleton = _cv_pattern(w, vowels)
            for feat, pat in self._PATTERNS.items():
                if pat.fullmatch(skeleton):
                    counts[feat] += 1
            if self._CONS_CLUSTER.search(skeleton):
                counts["cons_cluster"] += 1

        return {k: v / n for k, v in counts.items()}


# ===========================================================================
# 5.  CLGSNGO — Cross-lingual character n-gram overlap  (§3.1.4.1)
# ===========================================================================

def _clgsngo_clean(text: str) -> str:
    """Remove non-ASCII/non-alpha chars; lowercase — matches upstream CLGSNGO.clean()."""
    return re.sub(r"[^a-z]", "", text.lower())


def _get_ngrams(text: str, n: int) -> Counter:
    return Counter(text[i:i+n] for i in range(len(text) - n + 1))


def _rbo_overlap(doc_counts: Counter, anchor_list: List[str], p: float = 0.98) -> float:
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
        for lang_key, feat_prefix in [("tagalog","tag"), ("bikol","bik"), ("cebuano","ceb")]:
            result[f"{feat_prefix}_bi"]  = _rbo_overlap(bigrams,  self._anchors[lang_key][2])
            result[f"{feat_prefix}_tri"] = _rbo_overlap(trigrams, self._anchors[lang_key][3])
        return result


# ===========================================================================
# 6.  MORPH — Philippine morphological features  (Imperial & Ong 2021)
# ===========================================================================

_COMMON_PREFIXES = (
    r"naka(?=\w)", r"maka(?=\w)", r"pag(?=\w)", r"nag(?=\w)", r"mag(?=\w)",
    r"nang(?=\w)", r"mang(?=\w)", r"ma(?=\w)", r"pa(?=\w)", r"ka(?=\w)",
    r"na(?=\w)", r"gi(?=\w)", r"i(?=\w)", r"in(?=\w)", r"um(?=\w)",
)
_COMMON_SUFFIXES = (
    r"(?<=\w)han\b", r"(?<=\w)in\b", r"(?<=\w)an\b", r"(?<=\w)on\b",
    r"(?<=\w)ng\b",  r"(?<=\w)nila\b", r"(?<=\w)ko\b",
)
_PREFIX_RE  = re.compile("|".join(_COMMON_PREFIXES),  re.IGNORECASE)
_SUFFIX_RE  = re.compile("|".join(_COMMON_SUFFIXES),  re.IGNORECASE)
_REDUP_RE   = re.compile(r"\b([a-záéíóúàèìòùəǝ]{2,})-\1\b", re.IGNORECASE)
_NG_RE      = re.compile(r"\bng\b|\bnga\b|\bmga\b", re.IGNORECASE)

_FUNCTION_WORDS = frozenset([
    "ang", "ng", "sa", "na", "at", "ay", "si", "mga", "ko", "mo",
    "niya", "kami", "tayo", "kayo", "sila", "ito", "iyon", "dito",
    "og", "ug", "ni", "nako", "siya", "ila", "kini", "kana", "didto", "dinhi",
    "an", "ta", "sinda", "nia", "ninda",
    "o", "pero", "kaya", "dahil", "kung", "kapag", "habang",
])


class _MORPHExtractor:
    def extract(self, text: str) -> dict:
        words = _alpha_tokens(text)
        n     = len(words)
        if n == 0:
            return {k: 0.0 for k in [
                "affix_density","prefix_density","suffix_density",
                "reduplication_rate","ng_digraph_density",
                "function_word_ratio","hapax_ratio",
            ]}

        prefix_density    = len(_PREFIX_RE.findall(text)) / n
        suffix_density    = len(_SUFFIX_RE.findall(text)) / n
        affix_density     = prefix_density + suffix_density
        reduplication_rate = len(_REDUP_RE.findall(text.lower())) / n
        ng_digraph_density = len(_NG_RE.findall(text)) / n
        function_word_ratio = sum(1 for w in words if w in _FUNCTION_WORDS) / n

        freq = Counter(words)
        hapax_ratio = sum(1 for c in freq.values() if c == 1) / len(freq) if freq else 0.0

        return {
            "affix_density":       affix_density,
            "prefix_density":      prefix_density,
            "suffix_density":      suffix_density,
            "reduplication_rate":  reduplication_rate,
            "ng_digraph_density":  ng_digraph_density,
            "function_word_ratio": function_word_ratio,
            "hapax_ratio":         hapax_ratio,
        }


# ===========================================================================
# 7.  SYNT — Syntactic / narrative structure features
# ===========================================================================

_QUESTION_RE  = re.compile(r"\?")
_DIALOGUE_RE  = re.compile(r'[""«»\'"]')
_PUNCT_DENSE_RE = re.compile(r"[,;:]")


class _SYNTExtractor:
    def extract(self, text: str) -> dict:
        sents   = sent_tokenize(text)
        n_sents = max(len(sents), 1)
        return {
            "question_ratio": sum(1 for s in sents if _QUESTION_RE.search(s)) / n_sents,
            "dialogue_ratio": sum(1 for s in sents if _DIALOGUE_RE.search(s)) / n_sents,
            "punct_density":  sum(len(_PUNCT_DENSE_RE.findall(s)) for s in sents) / n_sents,
        }


# ===========================================================================
# 8.  FeaturePipeline — public API
# ===========================================================================

class FeaturePipeline:
    """Extracts the full KAYABASA feature set from a corpus DataFrame.

    Parameters
    ----------
    ngram_dir : str, optional
        Path to the CLGSNGO anchor n-gram files directory.
        Defaults to the ``ngrams list/`` folder beside this script.

    Example
    -------
    >>> pipe = FeaturePipeline()
    >>> X = pipe.fit_transform(df)
    >>> X[FEATURE_COLS]   # 35-column numeric DataFrame for the MLP
    """

    def __init__(self, ngram_dir: str = _DEFAULT_NGRAM_DIR):
        self._trad  = _TRADExtractor()
        self._syll  = _SYLLExtractor()
        self._clgs  = _CLGSNGOExtractor(ngram_dir=ngram_dir)
        self._morph = _MORPHExtractor()
        self._synt  = _SYNTExtractor()

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

    # ── Internal ───────────────────────────────────────────────────────────

    def _extract_one(self, row: pd.Series) -> dict:
        text     = str(row["text"])
        language = str(row.get("language", "")).lower()
        vowels   = EXTENDED_VOWELS.get(language, _DEFAULT_VOWELS)

        feats: dict = {"doc_id": row["doc_id"]}
        feats.update(self._trad.extract(text))
        feats.update(self._syll.extract(text, vowels))
        feats.update(self._clgs.extract(text))
        feats.update(self._morph.extract(text))
        feats.update(self._synt.extract(text))
        return feats

    @staticmethod
    def _sanity_checks(X: pd.DataFrame) -> None:
        feat_cols = [c for c in X.columns
                     if c not in ("doc_id","label","language","split_role")]

        all_null = X[feat_cols].isnull().all(axis=0)
        if all_null.any():
            raise AssertionError(f"Entirely-null columns: {list(all_null[all_null].index)}")

        constant = X[feat_cols].nunique() == 1
        if constant.any():
            print(f"  [WARNING] Constant columns: {list(constant[constant].index)}")

        ttr_bad = ~X["type_token_ratio"].between(0, 1, inclusive="right")
        if ttr_bad.any():
            raise AssertionError(f"TTR out of (0,1] for doc_ids: {X.loc[ttr_bad,'doc_id'].tolist()}")

        clgs_cols = ["tag_bi","bik_bi","ceb_bi","tag_tri","bik_tri","ceb_tri"]
        all_zero = (X[clgs_cols] == 0).all(axis=1)
        if all_zero.any():
            print(f"  [WARNING] {all_zero.sum()} docs have all-zero CLGSNGO overlap")

        density_cols = [c for c in feat_cols
                        if c.endswith("_density") or c.endswith("_ratio")
                        or c.startswith("syll_")]
        if (X[density_cols] < 0).any(axis=None):
            raise AssertionError("Negative density values found.")

        print(f"\n  [SANITY OK] {len(X)} docs × {len(feat_cols)} features. "
              f"No critical integrity violations.")


# ===========================================================================
# 9.  Corpus loader (shared with extract_features.py)
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
        Path to the normalized corpus file (e.g. output_normalized/all_languages.txt).
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
