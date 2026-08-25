# CHAPTER 4

# RESULTS AND DISCUSSION

This chapter reports the results of the pipeline described in the revised
Chapter 3. Findings are organized by Research Objective (RO), because each RO
in Section 1.4 stands for one numbered problem in the Statement of the Problem
(Section 1.2) and one Research Question (RQ) in Section 1.3 — RO1 answers
Problem 1 / RQ1, RO2 answers Problem 2 / RQ2, and so on.

**Two things to know before reading the tables.** First, "the KayaBasa model"
in this chapter always means the same specific model: XLM-R language
embeddings combined with 24 linguistic features, classified by a bagged
ensemble of 10 Weka MultilayerPerceptrons (H=128, N=10). It is this study's
hybrid model, and it is the model this study proposes.

Second, every result is reported as **Accuracy** and **Macro-Averaged F1-Score
(Macro F1)**. Accuracy is simply the percentage of documents the model
labeled correctly. Macro F1 scores the model separately on each of the three
reading levels (L1, L2, L3) and then averages those three scores equally. The
two metrics can disagree, because the three languages do not have equal
numbers of documents at each reading level (Chapter 3, Table in §3.1.1) — a
model can score well on Accuracy just by favoring the most common level,
without necessarily reading the less common levels correctly. **This chapter
uses Macro F1 to decide which model is better**, per Chapter 3 §3.3.1;
Accuracy is reported alongside for reference only. All numbers are the
average of 5 train/test splits (5-fold cross-validation), written as
mean ± standard deviation — the standard deviation shows how much the number
moved across those 5 splits; a small one (e.g. ±3) means a stable result, a
large one (e.g. ±10) means the number should be read as noisy, not exact.

**Data-integrity note:** an earlier pass of these experiments was affected by
a technical fault (Chapter 3, §3.2.5) that silently recorded some failed
training runs as 0% instead of raising an error. That fault has been fixed,
and every number in this chapter has been checked against the raw Weka
output. One correction from that check: Table 4.1's Bikolano and Cebuano
single-MLP numbers in an earlier draft of this chapter did not match any
verified run — they have been replaced below with the correct, checked
numbers.

## 4.1 Problem 1 / RQ1 / RO1 — Building the KayaBasa Model

The first objective was to build the KayaBasa model and confirm it can
classify text into L1/L2/L3 for all three available languages. It was built
and runs successfully for Tagalog, Bikolano, and Cebuano. The table below
also checks whether **bagging** (training 10 networks instead of 1, and
averaging their votes) actually helps, since it was added specifically to
reduce the instability of a single network (Chapter 3, §3.2.4).

**Table 4.1. Single MLP vs. bagged MLP (the KayaBasa model), trained on all
three languages combined and tested per language. Accuracy% / Macro F1.**

| Test Language | Single MLP | Bagged MLP (KayaBasa model) |
|---|---:|---:|
| Tagalog | 51.3±3.1% / 0.477 | 54.7±6.0% / 0.536 |
| Bikolano | 66.0±14.4% / 0.604 | 75.3±11.9% / 0.685 |
| Cebuano | 73.9±4.6% / 0.728 | 73.6±6.4% / 0.707 |

Bagging clearly helps Tagalog and Bikolano — Macro F1 rises by 0.059 and
0.081 — but makes almost no difference for Cebuano, where it is 0.021 lower
than the single network. Bagging is kept as the KayaBasa model's final design
because it helps two of the three languages and does not meaningfully hurt
the third.

## 4.2 Problem 2 / RQ2 / RO2 — Which Parts of the Model Matter

The proposal's original test design (Table XII) compared raw text against
cleaned text across several feature combinations. The test actually run here
instead compares **classifiers** (RandomForest vs. the tuned MLP) **and
feature sources** (linguistic features alone, linguistic features with
CrossNGO, embeddings alone, and everything combined), all on the same cleaned
text (Chapter 3, §3.1.2). This is the scope-change explained in Chapter 3,
§3.1.2.

**Table 4.2. Same-language train/test, by feature source and classifier.
Accuracy% / Macro F1.**

| Language | Feature Set | RandomForest | MLP (KayaBasa's classifier) |
|---|---|---|---|
| Tagalog | Linguistic features only | 53.2±9.5% / 0.510 | 45.7±7.1% / 0.349 |
| Tagalog | + CrossNGO | 54.7±5.5% / 0.527 | 43.0±4.5% / 0.320 |
| Tagalog | XLM-R embedding only | 56.6±5.7% / 0.537 | 54.7±3.0% / 0.529 |
| Tagalog | Everything combined | 59.2±6.1% / 0.560 | 54.7±3.5% / 0.527 |
| Bikolano | Linguistic features only | 74.0±7.2% / 0.638 | 73.3±4.7% / 0.538 |
| Bikolano | + CrossNGO | 74.0±7.2% / 0.622 | 70.7±6.4% / 0.518 |
| Bikolano | XLM-R embedding only | 75.3±5.6% / 0.553 | 71.3±9.6% / 0.638 |
| Bikolano | Everything combined | 76.7±4.1% / 0.601 | 75.3±7.7% / 0.681 |
| Cebuano | Linguistic features only | 76.8±7.9% / 0.744 | 67.6±6.2% / 0.607 |
| Cebuano | + CrossNGO | 76.2±7.2% / 0.728 | 66.2±4.7% / 0.588 |
| Cebuano | XLM-R embedding only | 72.8±5.4% / 0.691 | 72.5±6.5% / 0.689 |
| Cebuano | Everything combined | 74.8±6.0% / 0.709 | 72.2±8.3% / 0.689 |

Reading by Macro F1: under RandomForest, linguistic features alone already
score competitively for Bikolano and Cebuano, and adding embeddings mainly
helps Tagalog. Under the MLP — the KayaBasa model's own classifier — the
pattern is clearer: embeddings alone consistently beat linguistic features
alone (e.g. Bikolano Macro F1 goes from 0.518 to 0.638 with XLM-R alone), and
combining both lifts Macro F1 further for Bikolano and Cebuano. This
partly agrees with, but does not fully match, Imperial (2021)'s finding that
combining features and embeddings is not always better than embeddings
alone.

## 4.3 Problem 3 / RQ3 / RO3 — Comparing the KayaBasa Model Against Other Models

**Table 4.3. The KayaBasa model vs. this study's own RandomForest baseline vs.
the best published model vs. the Flesch-Kincaid formula. All models trained
on all three languages combined; feature set is the best-performing hybrid
combination (linguistic features + CrossNGO + XLM-R). Accuracy% / Macro F1.**

| Language | KayaBasa Model | RandomForest (this study) | Best Published Model (Imperial & Kochmar, 2023a) | Flesch-Kincaid |
|---|---:|---:|---:|---:|
| Tagalog | **54.7±6.0% / 0.536** | 48.7±8.6% / 0.477 | 32.7% / — | 36.6% / 0.179 |
| Bikolano | **75.3±11.9% / 0.685** | 74.0±9.8% / 0.620 | 79.3% / — | 36.7% / 0.179 |
| Cebuano | 73.6±6.4% / **0.707** | 73.9±5.6% / 0.691 | 75.6% / — | 23.5% / 0.127 |

By Macro F1 — the metric this study uses to decide which model is better,
since the three reading levels are not evenly represented in the data — the
KayaBasa model beats this study's own RandomForest baseline in **all three
languages**: Tagalog +5.9 points, Bikolano +6.5 points, Cebuano +1.6 points,
for an average of +4.6 points. (On raw Accuracy alone, Cebuano's numbers are
close enough — 73.6% vs. 73.9% — that neither model is clearly ahead; this is
exactly the kind of case Macro F1 is meant to settle, since Accuracy can be
misleading when reading levels are unevenly represented.)

The best published model (Imperial & Kochmar, 2023a) only reports Accuracy, so
it cannot be compared on Macro F1. On Accuracy, the KayaBasa model beats the
published Tagalog result (54.7% vs. 32.7%) but falls short for Bikolano and
Cebuano. This comparison is limited anyway, since the original study's exact
train/test split is not fully documented (Chapter 3, §3.1.3).

The Flesch-Kincaid formula — the traditional, English-calibrated readability
formula named in the Statement of the Problem — scores far below both
machine-learning models on every language and every metric. This directly
supports this study's starting claim that formulas built for English do not
work for these languages: on inspection, Flesch-Kincaid assigns nearly every
document to L3 regardless of its true difficulty level (Cohen's Kappa ≈ 0 in
every language, meaning its predictions are no better than chance).

Two other comparison points named in RQ3/RO3 — a majority-class baseline and
a document-length-only baseline — have not yet been built. This is an
outstanding gap, not a negative result.

## 4.4 Problem 4 / RQ4 / RO4 — Testing on Low-Resource Languages

The fourth objective tests whether the KayaBasa model generalizes **zero-shot**
to four related but unseen languages — Hiligaynon, Minasbate, and Karay-a
(Bisayan-affiliated) and Rinconada (Bikol-affiliated) — with no fine-tuning
and no target-language training data. The model was trained once on all 764
Tagalog+Cebuano+Bikolano documents combined, then evaluated as-is against
each low-resource language's full corpus. This condition is distinct from
the pairwise and all-languages conditions in RO1's training matrix (§3.3.2,
Phase 1): those measure whether additional related-language data improves
performance on a target language that is itself in the training set, so they
are not a test of transfer to a genuinely unseen language. Only these four
BasahaCorpus languages — held entirely outside the cross-validation
procedure, contributing no training examples at all — support a zero-shot
claim, so this result is reported on its own and not combined with RO1's
figures. This study's own RandomForest baseline is not run for this
objective; only the KayaBasa model's zero-shot transfer is reported.

**Corpus note.** The BasahaCorpus source (Imperial & Kochmar, 2023b — a
separate paper by the same authors as the RO3 comparison model, distinguished
here as 2023b) reports 776 documents across these four languages; this
study's low-resource corpus
is 769. The 7-document gap was investigated exhaustively — verified via a
full commit-history walk of the source repository (file counts only ever
increase; the current version is the historical maximum) — and traced to
7 specific documents never present in the publicly released data at all, not
to a cleaning or discovery fault on this study's end. All 769 documents that
do exist are included; none were filtered out during cleaning, including
documents whose extractable text is minimal after removing publisher/cover
boilerplate.

Unlike Tables 4.1–4.3, the numbers below are a **single train/test run**, not
a 5-fold average — there is only one way to construct "train on everything,
test on an unseen language" — so no standard deviation is reported.

**Table 4.4. Zero-shot transfer to four low-resource languages (KayaBasa
model only), by feature source. Trained once on all 764
Tagalog+Cebuano+Bikolano documents; tested on each language's full corpus.
Accuracy% / Macro F1.**

| Test Language | Feature Set | KayaBasa Model |
|---|---|---:|
| Hiligaynon | Linguistic features + CrossNGO | 60.2% / 0.468 |
| Hiligaynon | + XLM-R embedding | **70.7% / 0.682** |
| Minasbate | Linguistic features + CrossNGO | 59.3% / 0.457 |
| Minasbate | + XLM-R embedding | **54.5% / 0.524** |
| Karay-a | Linguistic features + CrossNGO | 66.5% / 0.537 |
| Karay-a | + XLM-R embedding | **67.6% / 0.629** |
| Rinconada | Linguistic features + CrossNGO | 60.5% / 0.426 |
| Rinconada | + XLM-R embedding | **56.4% / 0.500** |

(Bold marks the higher Macro F1 between the two feature sources for the same
language — the metric this chapter uses throughout, consistent with Tables
4.1–4.3.)

Reading by Macro F1: adding the XLM-R embedding to linguistic features
improves Macro F1 for every one of the four languages — Hiligaynon (0.468 →
0.682), Minasbate (0.457 → 0.524), Karay-a (0.537 → 0.629), and Rinconada
(0.426 → 0.500) — even though Accuracy alone moves in the opposite direction
for two of them (Minasbate 59.3% → 54.5%, Rinconada 60.5% → 56.4%). This is
exactly the kind of case this chapter's Macro F1 convention exists to catch:
the embedding-only model is scoring more evenly across reading levels even
where it accepts a lower Accuracy, rather than favoring whichever level is
most common in the unseen language's own class distribution.

Averaging the three Bisayan-affiliated languages (Hiligaynon, Karay-a,
Minasbate) against Rinconada (Bikol-affiliated) on the full-feature
condition: the KayaBasa model scores 64.3% / 0.612 mean Bisayan Macro F1 vs.
56.4% / 0.500 for Rinconada. By Macro F1, Rinconada scores lowest of all four
languages under both feature conditions in Table 4.4 (by raw Accuracy,
Minasbate is marginally lower than Rinconada in one cell — features-only,
59.3% vs. 60.5% — but Macro F1 is this chapter's deciding metric throughout). This
is a partial, not a clean, confirmation of the hypothesis that transfer is
easier within the same language subgroup the model was trained on: Bikolano
— the same subgroup as Rinconada — is one of the three training languages,
yet Rinconada still underperforms the three Bisayan-affiliated languages,
none of which the model was trained on directly. Explaining this gap is
outside this study's scope but is worth flagging for future work.

## 4.5 Summary of Findings

- **RO1 — done.** The KayaBasa model (XLM-R embeddings + linguistic features,
  bagged MLP) was built and classifies documents into L1/L2/L3 for all three
  available languages. Bagging improves Macro F1 for Tagalog and Bikolano and
  leaves Cebuano roughly unchanged.
- **RO2 — done**, with a scope change from the original raw-vs-cleaned-text
  test to a classifier × feature-source test, both on cleaned text (Chapter 3,
  §3.1.2). Embeddings alone consistently outperform linguistic features alone
  under the MLP; combining both helps most for Bikolano and Cebuano.
- **RO3 — mostly done.** By Macro F1, the KayaBasa model beats this study's own
  RandomForest baseline in all three languages (+4.6 points on average) and
  clears the Flesch-Kincaid floor by a wide margin. The comparison against the
  best published model is mixed. Two of the four comparison points named in
  RQ3/RO3 (majority-class baseline, document-length baseline) are not yet
  built.
- **RO4 — done.** The KayaBasa model was tested zero-shot on four low-resource
  languages (Hiligaynon, Minasbate, Karay-a, Rinconada; 769 documents, see
  §4.4's corpus note) — this study's own RandomForest baseline was not run
  for this objective. By Macro F1, adding the XLM-R embedding to linguistic
  features improves every language. Rinconada (Bikol-affiliated) scores
  lowest of the four languages under both feature conditions by Macro F1, a
  partial but not clean confirmation that transfer is easier within the
  training languages' subgroup, since Bikolano — Rinconada's own subgroup —
  was one of the three training languages.
