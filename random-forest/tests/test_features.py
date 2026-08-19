"""Regression tests for the pieces where a silent error would be invisible."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from kayabasa_rf import metrics  # noqa: E402
from kayabasa_rf.denoise import (  # noqa: E402
    clean_ara_text,
    clean_basaha_text,
    normalize_text,
)
from kayabasa_rf.features import (  # noqa: E402
    count_syllables,
    ranked_ngrams,
    rbo,
    syllable_patterns,
    trad_features,
)

# Hand-checked against standard Philippine syllabification. The medial-cluster
SYLLABLE_CASES = {
    "aso": ["V", "CV"],
    "bata": ["CV", "CV"],
    "babae": ["CV", "CV", "V"],
    "nagbabasa": ["CVC", "CV", "CV", "CV"],
    "maganda": ["CV", "CVC", "CV"],
    "langoy": ["CV", "CVC"],
    "anak": ["V", "CVC"],
    "ngipin": ["CV", "CVC"],  # <ng> is one consonant, not two
    "sungka": ["CVC", "CV"],
    "prutas": ["CCV", "CVC"],  # a legal onset cluster stays together
    "trabaho": ["CCV", "CV", "CV"],
    "eksena": ["VC", "CV", "CV"],  # /ks/ is not a legal onset, so it splits
    "dali-dali": ["CV", "CV", "CV", "CV"],  # reduplication hyphen is a boundary
}


def test_syllable_patterns():
    for word, expected in SYLLABLE_CASES.items():
        assert syllable_patterns(word) == expected, word


def test_count_syllables_matches_patterns():
    assert count_syllables("nagbabasa") == 4
    assert count_syllables("aso") == 2
    assert count_syllables("") == 0
    assert count_syllables("123") == 0


def test_rbo_bounds():
    ranked = list("abcdefghij")
    assert rbo(ranked, ranked) == 1.0
    assert rbo(list("abcde"), list("vwxyz")) == 0.0
    assert rbo([], ranked) == 0.0
    # Reversing the ranking must cost a lot, since RBO is top-weighted.
    assert 0.0 < rbo(ranked, list(reversed(ranked))) < 0.7


def test_ranked_ngrams_is_deterministic_under_ties():
    from collections import Counter

    tied = Counter({"ba": 3, "ka": 3, "an": 3, "sa": 1})
    assert ranked_ngrams(tied)[:3] == ["an", "ba", "ka"]  # alphabetical tie-break
    assert ranked_ngrams(tied, top_fraction=0.25) == ["an"]


def test_trad_features_on_empty_input():
    empty = trad_features("", [])
    assert all(value == 0.0 for value in empty.values())


def test_trad_features_counts():
    text = "Ang bata ay masaya. Ang aso ay tumakbo."
    words = text.replace(".", "").split()
    computed = trad_features(text, words)
    assert computed["trad_mean_sentence_len"] == 8 / 2
    assert 0 < computed["trad_type_token_ratio"] <= 1


def test_denoising_removes_artifacts_but_keeps_narrative():
    raw = (
        "Akong Higala Gisulat ni: Joan P. Sanchez Gidibuho ni: Justin Pono "
        "Usa ka adlaw, milakaw si Ana. &quot; https://letsreadasia.org KATAPUSAN"
    )
    cleaned = normalize_text(clean_ara_text(raw, "Akong Higala"))
    assert "Usa ka adlaw" in cleaned
    assert "Gisulat ni" not in cleaned
    assert "KATAPUSAN" not in cleaned
    assert "letsreadasia" not in cleaned
    assert "&quot;" not in cleaned


def test_basaha_truncates_at_boundary():
    raw = "Si Ana kag ang iya ido.\nPage 3\nFrog's Exercise\nAnswer the questions."
    cleaned = clean_basaha_text(raw)
    assert "Si Ana" in cleaned
    assert "Exercise" not in cleaned
    assert "Page 3" not in cleaned


def test_normalize_preserves_affixes():
    # Stripping nag-/-an would destroy the morphological readability signal.
    assert "nagbabasa" in normalize_text("nagbabasa")
    assert normalize_text("dali - dali") == "dali-dali"
    assert normalize_text('anihon,"an') == 'anihon," an'


def test_non_adjacent_error_rate():
    assert metrics.non_adjacent_error_rate([1, 1, 2], [3, 1, 2]) == 1 / 3
    assert metrics.non_adjacent_error_rate([1, 2, 3], [1, 2, 3]) == 0.0
    assert metrics.adjacent_error_rate([1, 2, 3], [2, 2, 3]) == 1 / 3


def test_evaluate_reports_all_classes():
    result = metrics.evaluate([1, 2, 3, 1], [1, 2, 3, 2])
    assert result["n"] == 4
    assert 0 < result["macro_f1"] <= 1
    for name in ("L1", "L2", "L3"):
        assert f"{name}_f1" in result


def test_mcnemar_identifies_the_better_model():
    # B is right on ten documents where A is wrong, and never the reverse.
    correct_a = [False] * 10 + [True] * 10
    correct_b = [True] * 20
    result = metrics.mcnemar_exact(correct_a, correct_b)
    assert result["only_b_correct"] == 10
    assert result["only_a_correct"] == 0
    assert result["p_value"] < 0.01


if __name__ == "__main__":
    failures = 0
    for name, function in sorted(globals().items()):
        if name.startswith("test_") and callable(function):
            try:
                function()
                print(f"  PASS  {name}")
            except AssertionError as error:
                failures += 1
                print(f"  FAIL  {name}: {error}")
    print("\nall tests passed" if not failures else f"\n{failures} test(s) failed")
    sys.exit(1 if failures else 0)
