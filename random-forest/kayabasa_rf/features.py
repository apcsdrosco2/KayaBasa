"""Handcrafted linguistic features (proposal Section 3.2.2, Table VII)."""

from __future__ import annotations

import re
from collections import Counter
from pathlib import Path

import pandas as pd

from . import config

# Orthography
VOWELS = set("aeiouáéíóúàèìòùäëïöüāēīōū")

# <ng> is a single consonant (velar nasal) in Philippine orthography. Treating
_NG_RE = re.compile(r"ng", re.IGNORECASE)
_NG_PLACEHOLDER = ""  # a character that cannot occur in the corpora

_WORD_RE = re.compile(r"[^\W\d_]+(?:[-'][^\W\d_]+)*", re.UNICODE)
_SENTENCE_SPLIT_RE = re.compile(r"[.!?]+(?:\s|$)|\n+")
_PUNCT_RE = re.compile(r"[^\w\s]", re.UNICODE)

# The four canonical Philippine syllable structures reported in Table VII.
CANONICAL_PATTERNS = ("CV", "CVC", "CCVC", "CCVCCC")

FEATURE_NAMES = [
    "syll_cv",
    "syll_cvc",
    "syll_ccvc",
    "syll_ccvccc",
    "trad_mean_sentence_len",
    "trad_mean_word_len",
    "trad_polysyllabic_ratio",
    "trad_type_token_ratio",
    "clgsngo_bigram_rbo",
    "clgsngo_trigram_rbo",
]

POLYSYLLABIC_MIN = 3


# Phonotactic decoding load (Section 3.2.2.1, syll_parse.py)
LEGAL_ONSET_CLUSTERS = frozenset(
    {
        "bl", "br", "by", "bw",
        "dr", "dy", "dw",
        "gl", "gr", "gw",
        "hw", "hy",
        "kl", "kr", "kw", "ky",
        "ly",
        "my", "mw",
        "ny", "nw",
        "pl", "pr", "pw", "py",
        "ry",
        "sk", "sp", "st", "sw", "sy",
        "tr", "ts", "tw", "ty",
    }
)


def _phonemes(word: str) -> list[tuple[str, str]]:
    """Split a word into (symbol, class) pairs, treating <ng> as one consonant."""
    folded = _NG_RE.sub(_NG_PLACEHOLDER, word.lower())
    out = []
    for ch in folded:
        if ch in VOWELS:
            out.append((ch, "V"))
        elif ch == _NG_PLACEHOLDER:
            out.append(("ng", "C"))
        elif ch.isalpha():
            out.append((ch, "C"))
    return out


def syllable_patterns(word: str) -> list[str]:
    """Segment one word into syllables and return each syllable's CV pattern."""
    # Reduplication hyphens ("dali-dali") are genuine syllable boundaries.
    if "-" in word or "'" in word:
        parts = re.split(r"[-']", word)
        return [p for part in parts if part for p in syllable_patterns(part)]

    units = _phonemes(word)
    nuclei = [i for i, (_, kind) in enumerate(units) if kind == "V"]
    if not nuclei:
        return []

    patterns = []
    for k, nucleus in enumerate(nuclei):
        previous = nuclei[k - 1] if k > 0 else -1
        following = nuclei[k + 1] if k + 1 < len(nuclei) else len(units)

        run = nucleus - previous - 1  # consonants sitting before this nucleus
        if k == 0:
            # A word-initial cluster has nowhere else to go.
            onset = run
        elif run >= 2:
            pair = "".join(symbol for symbol, _ in units[nucleus - 2 : nucleus])
            onset = 2 if pair in LEGAL_ONSET_CLUSTERS else 1
        else:
            onset = run

        trailing = following - nucleus - 1  # consonants sitting after this nucleus
        if k == len(nuclei) - 1:
            coda = trailing  # a word-final cluster closes the last syllable
        elif trailing >= 2:
            pair = "".join(symbol for symbol, _ in units[following - 2 : following])
            coda = trailing - (2 if pair in LEGAL_ONSET_CLUSTERS else 1)
        else:
            coda = 0  # the single consonant becomes the next syllable's onset

        patterns.append("C" * onset + "V" + "C" * coda)
    return patterns


def count_syllables(word: str) -> int:
    return len(syllable_patterns(word))


def syll_features(words: list[str]) -> dict[str, float]:
    """Frequencies of the four canonical patterns, as a share of all syllables."""
    counts: Counter = Counter()
    total = 0
    for word in words:
        for pattern in syllable_patterns(word):
            total += 1
            if pattern in CANONICAL_PATTERNS:
                counts[pattern] += 1
    if total == 0:
        return dict.fromkeys(FEATURE_NAMES[:4], 0.0)
    return {
        "syll_cv": counts["CV"] / total,
        "syll_cvc": counts["CVC"] / total,
        "syll_ccvc": counts["CCVC"] / total,
        "syll_ccvccc": counts["CCVCCC"] / total,
    }


# Sentence-level surface statistics (Section 3.2.2.2, trad_parser.py)
def trad_features(text: str, words: list[str]) -> dict[str, float]:
    n_words = len(words)
    if n_words == 0:
        return dict.fromkeys(FEATURE_NAMES[4:8], 0.0)

    sentences = [s for s in _SENTENCE_SPLIT_RE.split(text) if s.strip()]
    n_sentences = len(sentences) or 1
    polysyllabic = sum(1 for w in words if count_syllables(w) >= POLYSYLLABIC_MIN)
    return {
        "trad_mean_sentence_len": n_words / n_sentences,
        "trad_mean_word_len": sum(len(w) for w in words) / n_words,
        "trad_polysyllabic_ratio": polysyllabic / n_words,
        "trad_type_token_ratio": len({w.lower() for w in words}) / n_words,
    }


# CROSSNGO cross-lingual n-gram overlap (Section 3.2.2.3, CLGSNGO_parser.py)
def char_ngrams(text: str, n: int) -> Counter:
    """Character n-grams over letters and single spaces, lowercased."""
    cleaned = _PUNCT_RE.sub(" ", (text or "").lower())
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return Counter(cleaned[i : i + n] for i in range(len(cleaned) - n + 1))


def ranked_ngrams(counter: Counter, top_fraction: float | None = None) -> list[str]:
    """N-grams ordered by descending frequency, ties broken alphabetically."""
    ordered = sorted(counter.items(), key=lambda kv: (-kv[1], kv[0]))
    ranked = [ngram for ngram, _ in ordered]
    if top_fraction is not None and ranked:
        keep = max(1, int(round(len(ranked) * top_fraction)))
        ranked = ranked[:keep]
    return ranked


def rbo(list1: list[str], list2: list[str], p: float = config.RBO_P) -> float:
    """Rank-Biased Overlap, extrapolated form (Webber et al. 2010, eq. 32)."""
    if not list1 or not list2:
        return 0.0

    short, long = (list1, list2) if len(list1) <= len(list2) else (list2, list1)
    s, l = len(short), len(long)

    seen_short: set[str] = set()
    seen_long: set[str] = set()
    overlap = 0
    overlap_at_s = 0
    weighted_sum = 0.0

    for d in range(1, l + 1):
        e_long = long[d - 1]
        if d <= s:
            e_short = short[d - 1]
            seen_short.add(e_short)
            seen_long.add(e_long)
            overlap += (e_long in seen_short) + (e_short in seen_long)
            if e_short == e_long:
                overlap -= 1  # counted once from each side above
        else:
            seen_long.add(e_long)
            overlap += e_long in seen_short
        if d == s:
            overlap_at_s = overlap
        weighted_sum += (overlap / d) * (p**d)

    overlap_at_l = overlap

    # Ranks beyond the shorter list, where its content can no longer change.
    tail_sum = sum(
        (overlap_at_s * (d - s) / (d * s)) * (p**d) for d in range(s + 1, l + 1)
    )
    extrapolation = ((overlap_at_l - overlap_at_s) / l + overlap_at_s / s) * (p**l)

    return (1 - p) / p * (weighted_sum + tail_sum) + extrapolation


# Extractor
class FeatureExtractor:
    """Turns prepared text into the Table VII feature matrix."""

    def __init__(
        self,
        anchor_top_fraction: float = config.ANCHOR_TOP_FRACTION,
        rbo_p: float = config.RBO_P,
    ) -> None:
        self.anchor_top_fraction = anchor_top_fraction
        self.rbo_p = rbo_p
        self.anchor_bigrams: list[str] = []
        self.anchor_trigrams: list[str] = []
        self._fitted = False

    def fit(self, anchor_texts: list[str]) -> "FeatureExtractor":
        bigrams, trigrams = Counter(), Counter()
        for text in anchor_texts:
            bigrams.update(char_ngrams(text, 2))
            trigrams.update(char_ngrams(text, 3))
        self.anchor_bigrams = ranked_ngrams(bigrams, self.anchor_top_fraction)
        self.anchor_trigrams = ranked_ngrams(trigrams, self.anchor_top_fraction)
        self._fitted = True
        return self

    def transform_one(self, text: str) -> dict[str, float]:
        if not self._fitted:
            raise RuntimeError("FeatureExtractor.fit() must run before transform().")
        words = _WORD_RE.findall(text or "")
        row: dict[str, float] = {}
        row.update(syll_features(words))
        row.update(trad_features(text or "", words))
        row["clgsngo_bigram_rbo"] = rbo(
            ranked_ngrams(char_ngrams(text, 2)), self.anchor_bigrams, self.rbo_p
        )
        row["clgsngo_trigram_rbo"] = rbo(
            ranked_ngrams(char_ngrams(text, 3)), self.anchor_trigrams, self.rbo_p
        )
        return row

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        rows = [self.transform_one(text) for text in df["text"]]
        out = pd.DataFrame(rows, columns=FEATURE_NAMES, index=df.index)
        out.insert(0, "doc_id", df["doc_id"].to_numpy())
        return out

    def fit_transform(
        self, df: pd.DataFrame, anchor_language: str = config.ANCHOR_LANGUAGE
    ) -> pd.DataFrame:
        anchor_texts = df.loc[df["language"] == anchor_language, "text"].tolist()
        if not anchor_texts:
            print(
                f"[warn] anchor language {anchor_language!r} is absent; "
                "building the CROSSNGO profile from the full corpus instead"
            )
            anchor_texts = df["text"].tolist()
        self.fit(anchor_texts)
        return self.transform(df)


def extract_features(
    df: pd.DataFrame,
    anchor_language: str = config.ANCHOR_LANGUAGE,
    out_csv: str | Path | None = None,
) -> pd.DataFrame:
    """Build the full feature matrix for a prepared corpus."""
    print(
        f"[features] extracting {len(FEATURE_NAMES)} features "
        f"for {len(df)} documents (anchor: {anchor_language})"
    )
    features = FeatureExtractor().fit_transform(df, anchor_language=anchor_language)
    if out_csv is not None:
        out_csv = Path(out_csv)
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        features.to_csv(out_csv, index=False)
        print(f"[features] wrote {out_csv}")
    return features


def merge_parser_outputs(
    syll_csv: str | Path,
    trad_csv: str | Path,
    clgsngo_csv: str | Path,
    doc_ids: list[str] | None = None,
    out_csv: str | Path | None = None,
) -> pd.DataFrame:
    """Column-wise merge of the three imperialite parser outputs (Section 3.2.2.4)."""
    frames = [pd.read_csv(p) for p in (syll_csv, trad_csv, clgsngo_csv)]
    if len({len(f) for f in frames}) != 1:
        raise ValueError(
            f"Parser outputs disagree on row count: {[len(f) for f in frames]}. "
            "They must all be generated from the same combined.csv."
        )
    merged = pd.concat(frames, axis=1)
    if doc_ids is not None:
        if len(doc_ids) != len(merged):
            raise ValueError(
                f"doc_ids has {len(doc_ids)} entries but the parser outputs have "
                f"{len(merged)} rows; the row alignment would be wrong."
            )
        merged.insert(0, "doc_id", doc_ids)
    if out_csv is not None:
        merged.to_csv(out_csv, index=False)
    return merged
