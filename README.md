# KayaBasa

A repository for the thesis: **KAYABASA: A Denoised Hybrid Transformer–Feature Model For Evaluating The Readability Of Filipino Educational Texts**.

For full documentation, setup instructions, and model architecture, see the [`main` branch README](https://github.com/YOUR_USERNAME/KayaBasa/blob/main/README.md).

---

## This Branch: `feat/feature-pipeline-refactor`

Refactors the feature extraction pipeline to match exactly the 14 features defined in the thesis (§2.3.4, Table VII). Removes extra features that were not specified in the paper.

**Changes in this branch:**
- `feature-pipeline/feature_pipeline.py` — trimmed from 35 to 14 features (TRAD × 4, SYLL × 4, CLGSNGO × 6)
- `feature-pipeline/extract_features.py` — updated driver script
- `feature-pipeline/train_features.py` — features-only training (Config B: Random Forest, 5-fold CV)
- `feature-pipeline/output/all_features.csv` — 1480 docs × 14 features
- `feature-pipeline/output/results_config_B.json` — Config B results (Macro-F1: 0.6487)
