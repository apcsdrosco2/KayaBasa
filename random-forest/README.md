# Random Forest Baseline (KAYABASA)

Baseline model for the comparative analysis in Table XIII of the proposal. It
uses the handcrafted linguistic features only, with no transformer embeddings,
so it can be compared against the full hybrid model.

## Requirements

Python 3.10 or newer.

    pip install -r requirements.txt

Everything here is already installed in Google Colab.

## Running it

Put the corpus somewhere and point the script at it:

    python run_baseline.py --data-root ./data/clean --already-clean

Use --already-clean when the text has already been denoised. Leave it out if
you are giving it raw text and want the script to do the cleaning.

Results are written to results/, the trained model to models/, and the
cross-validation folds to splits/fold_indices.pkl.

To compare against the hybrid once it has predictions:

    python run_validation.py --hybrid-predictions predictions.csv --data-root ./data/clean

## Checking the corpus first

Before training it helps to check the document counts:

    python inspect_corpus.py --root ./data/clean

This prints how many documents were found per language and per level and
compares them to the counts in the proposal.

If there is a raw and a cleaned version of the corpus, this compares them and
lists any document that is in one but not the other:

    python inspect_corpus.py --compare ./data/raw ./data/clean

## Features

Ten features in total.

Syllable patterns (from syll_parse.py in the original repo):
CV, CVC, CCVC, CCVCCC, counted as a share of all syllables in the document.

Surface statistics (from trad_parser.py):
mean sentence length, mean word length, ratio of words with three or more
syllables, and type-token ratio.

CROSSNGO (from CLGSNGO_parser.py):
Rank-Biased Overlap of the document's character bigrams and trigrams against
the most frequent n-grams of an anchor language, Cebuano by default.

If you already have the three parser scripts from the imperialite repos, you
can run those instead and merge the output with merge_parser_outputs() in
features.py. Otherwise the code computes the same values itself.

Two notes on the syllable counting. "ng" is treated as one consonant, not two,
otherwise the CCVC and CCVCCC counts come out too high. And consonants between
two vowels are only kept together if they form a cluster that can actually start
a syllable in Filipino, so nagbabasa splits as nag-ba-ba-sa and not
na-gba-ba-sa. There are test cases for this in tests/test_features.py.

## Model settings

Random Forest with 100 trees, max_features = log2, no depth limit. These match
the WEKA defaults used in the BasahaCorpus README so the baseline is comparable
to the earlier work instead of being something I tuned myself.

Cross-validation is stratified 5-fold on the combined (language, label) key.
Seed is 42. The folds are generated once, saved, and reused on every later run.

## About the folds

splits/fold_indices.pkl needs to be shared with whoever trains the hybrid. Both
models have to be scored on the same splits, otherwise the comparison in Section
3.3.3 is not measuring the same thing.

## Validation output

run_validation.py does two things. First it checks the prediction file itself:
whether every document has a prediction, whether the labels are inside L1/L2/L3,
whether the gold labels match the corpus, and whether the model is predicting
all three levels instead of collapsing onto one. If any of these fail it says so
and exits with an error code.

Then it compares the two models. Besides the macro-F1 difference it reports a
paired bootstrap confidence interval, a McNemar test, how often the two models
agree, and how many errors are off by one level versus off by two. That last one
matters because grading an L1 text as L3 is worse than grading it as L2.

The report is written to results/validation_report.md.

## Testing without the corpus

    python make_synthetic_corpus.py --out ./corpora --emit-mock-hybrid
    python run_baseline.py --data-root ./corpora

This generates fake documents in the same folder layout so the pipeline can be
run end to end. The text is nonsense, so the scores mean nothing. It is only for
checking that the code works.

    python tests/test_features.py

## A few things worth knowing

The regex for removing author credits in the proposal ends with [^\n.!?]*, which
does not work on the ara-close-lang files because each document is on a single
line. That pattern keeps matching past the credit and deletes the start of the
story. In one case it reduced a whole document to '. Sanchez . "'. This version
only removes name-like words after the credit so it stops at the sentence.

The proposal also computes the features once over the whole corpus, which means
the validation documents affect the CROSSNGO anchor ranking. The effect is small
but it is still information crossing the split. Passing --anchor-from-train-only
rebuilds the anchor inside each fold if you want a number without that.

Feature importances come from impurity, which tends to favour continuous
features, so treat the ranking as a rough guide rather than a measurement.
