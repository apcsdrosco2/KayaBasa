"""
run_final_hybrid_model.py — reproduces this study's reported hybrid model with no
flags to remember: XLM-R embeddings + linguistic features, MLP classifier, bagged
(10 networks), hidden layer = 128, epochs = 10.

train_and_evaluate_cv.py is a general-purpose runner that takes the classifier and
every hyperparameter as command-line arguments (see its own docstring) — useful for
experimenting, but easy to get wrong if you don't already know which flags produced
the results in the thesis. This script does NOT duplicate that runner's logic; it
just calls its existing main() with the exact arguments that were actually used,
so anyone can reproduce the same run with a single, argument-free command:

    python run_final_hybrid_model.py

This is exactly equivalent to running:

    python train_and_evaluate_cv.py mlp_bagged --hidden 128 --epochs 10 --bags 10 --features embedding

...and writes to the same output folder, results_cv_mlp_bagged_h128_n10_b10_emb/
(monolingual / bilingual / all-languages-combined x the four embedding feature sets
[mbert, all, xlmr, all_xlmr] x 5 CV folds — see train_and_evaluate_cv.py and
cv_common.py for what each of those covers).

Requires train_test_split_cv.py to have already been run once (to build splits_cv/).
"""

import sys

from train_and_evaluate_cv import main as run_train_and_evaluate_cv

FIXED_ARGS = [
    "mlp_bagged",
    "--hidden", "128",
    "--epochs", "10",
    "--bags", "10",
    "--features", "embedding",
]

if __name__ == "__main__":
    sys.argv = ["train_and_evaluate_cv.py"] + FIXED_ARGS
    run_train_and_evaluate_cv()
