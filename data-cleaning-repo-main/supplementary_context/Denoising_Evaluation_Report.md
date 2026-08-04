# Denoising Evaluation & Verification Report

**Project:** KayaBasa — Hierarchical Cross-lingual Automatic Readability Assessment for Philippine languages
**Scope of this document:** how the denoising stage (cleaning → normalization) is *evaluated* and *verified*, i.e. how we establish that the cleaned/normalized text files are "truly cleaned" without access to native speakers.
**Status:** current as of the `denoising-validation` branch (regenerated corpus = 1,480 narrative rows across 7 languages).

---

## 1. What "truly cleaned" means here

Denoising in this project is a **deletion-only** transformation. It removes text that is metadata or noise — repeated titles, author/illustrator credits, English genre tags, publisher/license boilerplate, page numbers, activity/comprehension tails, `KATAPUSAN` end markers — and keeps the native-language narrative. It never rewrites, paraphrases, translates, or reorders the story.

Because the operation only deletes, "truly cleaned" decomposes into three claims that can each be checked mechanically:

1. **Nothing but junk was removed** (no false-positive deletion of narrative).
2. **All the junk was removed** (no false-negative retention of boilerplate).
3. **The text is machine-consistent** (every letter preserved through normalization; output is stable and reproducible).

Crucially, none of these is a *semantic* judgment ("does the cleaned text read correctly in Cebuano?"), which would require fluency. They are *structural* judgments about what was deleted and what remains — which is exactly why the absence of native-speaker reviewers does not block validation.

### Why native-speaker review is not the gate
The corpora are low-resource Central Philippine languages (Cebuano, Bikol, Hiligaynon, Minasbate, Karay-a, Rinconada) plus Tagalog. Fluent annotators are not available to the study at corpus scale. The manual 70-document sample described in the proposal (§3.1.2) is retained, but **repositioned as a spot-check of an automatically generated review queue**, not as the primary correctness gate.

---

## 2. The verification principle

> Validating a *deletion* is a judgment about **form**, not **meaning**. Metadata is identifiable by structure (position, capitalization, punctuation) or by language (English boilerplate inside a non-English narrative). Detecting whether such spans were removed — and whether anything else was — requires no fluency.

This principle is what every check below operationalizes. It also bounds what we can and cannot claim: we can certify that only structurally/linguistically identifiable noise was removed and that the narrative survives intact; we cannot, from these checks alone, certify subtle semantic fidelity — but since we never alter meaning-bearing text, semantic fidelity reduces to "did we delete only noise?", which the checks do cover.

---

## 3. Evaluation instruments

Two scripts, run against the raw source repositories under a fixed seed.

### 3.1 `validate_denoising.py` — the gate scorecard

Runs three families of checks and writes `output/validation_report.txt` (scorecard) and `output/validation_queue.txt` (documents needing a human glance).

**Part A — Extraction faithfulness (reference reproduction).**
The `ara-close-lang` repository ships the *original authors' published feature files* (`data/<lang>/trad.csv`) for the Cebuano and Bikol documents. Those authors' extraction scripts' internal `cleaner()` strips only non-alphabetic characters — it does **not** remove titles, credits, or `KATAPUSAN`. Therefore the published word- and sentence-counts are a **deterministic function of the raw text**, and running the *unmodified* original `TRAD.py` on our ingested raw text must reproduce them exactly.
- *What it proves:* our ingestion, tokenization, and feature extraction are faithful to the reference pipeline — independent of any cleaning. Any later difference between raw and cleaned features is therefore attributable solely to denoising.
- *Result:* **Cebuano 349/349**, **Bikol 148/150** exact on (word_count, sentence_count). The two Bikol exceptions differ by exactly two tokens with matching sentence counts (residual `&nbsp;` entity artifacts).
- *Coverage:* Cebuano and Bikol only — the sole languages with a published answer key (see §5).

**Part B — Denoising accountability.**
For each document, the raw and cleaned texts are aligned token-by-token (`difflib`). It asserts (i) *deletion-only* — cleaning introduces no token absent from the raw source — and (ii) no native function word is removed from the *interior* of a document (credits/genre/tails contain no native function words, so an interior native word removed is the signature of narrative loss).
- *Result:* 0 over-deletion-to-empty. Interior-deletion flags feed the review queue.
- **Known limitation (important):** Part B over-reports on **native-language boilerplate**. Author bios, acknowledgments, and publisher attributions are written *in the target language* (e.g. Bikol "an produktong ini ay naging posible…" = "this product was made possible…"), so removing them legitimately deletes native function words and trips the interior-deletion flag. Part B cannot tell these apart from real loss. **This is why `audit_cleaning.py` (§3.2), which classifies each removed span, is the authoritative narrative-loss check.**

**Part C — Structural invariants** (applied to the `normalize_text` function over all clean rows, so correctness never depends on possibly-stale on-disk files):
- **Grapheme/accent preservation** — normalization drops/alters no letter, so ñ, accented vowels, and the Rinconada schwa survive. *Result: 1480/1480.*
- **Idempotence** — re-normalizing an already-normalized document is a no-op. *Result: 1480/1480.*
- **Boilerplate non-leakage** — no residual English publisher/license phrase survives into a narrative (detected by an English-density scan; language-agnostic because the boilerplate is English while narratives are not). *Result: 0 leaks.*
- **Degenerate rows** — no emitted narrative under 4 words.
- **Freshness** — the on-disk `output_normalized/` files are exactly reproducible from `output/`.

### 3.2 `audit_cleaning.py` — the authoritative per-span removal ledger

Diffs `basic_clean(raw)` against the final cleaned narrative for **every** document and records **each removed span** with a character index and a category. Outputs:
- `output/cleaning_ledger.txt` — the complete indexed record: `language|level|title|char_start|char_end|words|position|category|removed_text`.
- `output/cleaning_review.txt` — only the suspects (`REVIEW_narrative`): spans that contain native function words but match **no** known metadata pattern, i.e. possible false-positive deletions.

**Categories** (most-specific first): `end_marker`, `author_bio`, `credit`, `publisher`, `genre_tag`, `meta_label`, `foreign_entity`, `title_author`, `names_or_title`, `title`, and — the only one that matters for narrative-loss — `REVIEW_narrative`.

- *What it proves:* every deleted span is accounted for as metadata, and the residual "narrative-like" removals are enumerated and small enough to inspect by hand.
- *Result (ceb + bik + BasahaCorpus, 1,268 documents, ~189,000 removed words):*

  | Category | Words removed | Share |
  |---|---:|---:|
  | publisher | 139,074 | 73.5% |
  | author_bio | 28,280 | 14.9% |
  | credit | 14,282 | 7.5% |
  | title_author | 6,176 | 3.3% |
  | end_marker | 1,102 | 0.6% |
  | genre_tag / names / title | ~293 | 0.2% |
  | **REVIEW_narrative** | **25 (6 spans)** | **0.0%** |

- *The 6 REVIEW spans, on inspection, are all correct removals* (not narrative loss): bilingual title variants (a story printed with both its Cebuano and Tagalog title, e.g. "Ang Elepante Sa Aking Tahanan"), an author byline ("Si Balaraw"), and title fragments ("Ang Una Na"). See §7.
- *Coverage:* Cebuano, Bikol, and the four BasahaCorpus languages. Tagalog is validated through `validate_denoising` Part C (structural, all 1,480 rows) plus the density-based safety net in `clean_tagalog.py` (see §4); extending the per-span ledger to Tagalog is a planned enhancement.

---

## 4. What changed recently, and why the evaluation was updated

The evaluation method was revised alongside three sets of cleaning fixes:

**(a) Publisher/end-marker redesign (Cebuano/Bikol).**
Previously the cleaner truncated each story at the *first* publisher or end-marker token. This catastrophically amputated stories when the token appeared early or inline — e.g. "Ang Una Na Operasyon ni Dr Sokha" (a source tag "The Asia Foundation - Let's Read" after the title) was cut from **1,057 words to 4**, and the ordinary loanword "Grade" ("mga tinun-an sa Grade") truncated others. The fix: a marker truncates **only when almost no native narrative follows it** (a genuine tail); otherwise the boilerplate window is deleted **in place** and the story is kept. `\bGrade\s+\d+\b` now requires a digit.
- *Effect on evaluation:* recovered stories now retain (correctly-removed) native-language publisher attributions in the deletion set, which inflated Part B's interior-deletion count. This is what motivated promoting `audit_cleaning.py`'s category-based check to authoritative status.

**(b) Tagalog integration.**
The Tagalog sources were never missing — they are at `data/tag_lvl{1,2,3}_data.txt`; the loader simply pointed at the wrong folder. After fixing the path (265 documents load), the DepEd-LRMDS Tagalog proved far messier than the other sources. The cleaner was hardened: the credit remover now consumes full comma-separated editor name-lists; **English/boilerplate detection is density-based** (a multi-word segment with *no Tagalog function word* is boilerplate — robust to arbitrary legal/admin English without enumerating it); and a final density test drops non-narrative admin pages (`residual_boilerplate`) rather than emitting mangled rows.
- *Evaluation lesson recorded:* a signature blocklist (English "Writer:", position codes, ALL-CAPS runs) was first tried and **wrongly dropped ~19 genuine stories** with all-caps titles. Dropping a document that contains a real story *is* a form of mangling; the density criterion replaced it and drops 0 real stories.

**(c) Normalizer coverage.** `normalize_datasets.py` now includes `tagalog` in its language list, so the normalized outputs are complete and reproducible.

**Net update to the methodology:** evaluation is now a **two-instrument framework** — `validate_denoising.py` for gate checks and reproduction, `audit_cleaning.py` for authoritative per-span narrative-loss accounting — with the explicit acceptance rubric in §6.

---

## 5. Coverage by language

Not all checks apply everywhere; only Part A needs an external answer key.

| Language | Source | Raw text | Answer key | A: reproduction | B/audit: accountability | C: invariants |
|---|---|:--:|:--:|:--:|:--:|:--:|
| Cebuano | ara-close-lang | ✓ | ✓ | ✓ | ✓ | ✓ |
| Bikol | ara-close-lang | ✓ | ✓ | ✓ | ✓ | ✓ |
| Hiligaynon | BasahaCorpus | ✓ | ✗ | ✗ | ✓ | ✓ |
| Minasbate | BasahaCorpus | ✓ | ✗ | ✗ | ✓ | ✓ |
| Karay-a | BasahaCorpus | ✓ | ✗ | ✗ | ✓ | ✓ |
| Rinconada | BasahaCorpus | ✓ | ✗ | ✗ | ✓ | ✓ |
| Tagalog | substitute corpus | ✓ | ✗ | ✗ | via C + density net | ✓ |

**On the missing answer key (5 languages):** BasahaCorpus ships no per-document feature files, and the published Tagalog features describe a *different* corpus, so reference reproduction is permanently unavailable for these. Its impact is bounded because (i) their dominant noise is *English* publisher boilerplate, which the language-agnostic leakage scan catches, and (ii) accountability (only-junk-removed) does not need a reference. A reference-free corroboration — that readability features increase monotonically across grade levels L1→L2→L3 — is recommended as a supplementary check.

---

## 6. Acceptance rubric — the operational definition of "truly cleaned"

A document (and the corpus) is accepted as clean when **all** of the following hold:

**Hard gates (must pass 100% — any violation is a defect):**
- Grapheme/accent preservation through normalization.
- `normalize_text` idempotence.
- No over-deletion to empty; no degenerate (<4-word) emitted rows.
- No English/publisher boilerplate leakage into a narrative.

**Reference gate (where an answer key exists):**
- Reproduction ≥ 99% on (word_count, sentence_count), with every remaining mismatch individually explained.

**Narrative-loss gate (authoritative — `audit_cleaning.py`):**
- The `REVIEW_narrative` queue is fully adjudicated and confirmed to contain no real narrative loss.

**Hygiene:**
- Non-narrative documents that cannot be recovered are *dropped and logged* (`residual_boilerplate`, `too_short`, `english_heavy`, `duplicate`, `empty`), never emitted mangled.
- Re-running the pipeline reproduces identical outputs.

Any confirmed narrative loss triggers a targeted pattern revision and a full re-run — the same loop as the proposal's §3.1.2.

---

## 7. Current results (evidence)

**Corpus:** 1,480 narrative rows.

| Language | L1 | L2 | L3 |
|---|--:|--:|--:|
| Tagalog | 68 | 96 | 75 |
| Cebuano | 164 | 100 | 81 |
| Bikol | 68 | 27 | 55 |
| Hiligaynon | 60 | 21 | 45 |
| Minasbate | 119 | 72 | 68 |
| Karay-a | 60 | 32 | 79 |
| Rinconada | 114 | 35 | 41 |

**Scorecard:**

| Check | Result |
|---|---|
| Reference reproduction | Cebuano 349/349; Bikol 148/150 (2 explained) |
| Grapheme/accent preservation | 1480/1480 |
| `normalize_text` idempotent | 1480/1480 |
| English/publisher leakage | 0 |
| Authoritative narrative-loss (`cleaning_review.txt`) | 6 spans / 25 words / 0.0% — all benign |

**The complete narrative-loss review queue (all 6, verified as correct removals):**

| Language·Level | Doc title | Removed span | Verdict |
|---|---|---|---|
| cebuano·3 | Ang Atup nga Hardin | `Ang Bubungang Hardin Kim Ann Arun Chea Sereyroth` | variant title + author names |
| cebuano·1 | Si Navy ug si Bora | `Si Navy at si Bora` | Tagalog-variant title |
| cebuano·3 | Ang Elepante sa Akong Balay | `Ang Elepante Sa Aking Tahanan` | Tagalog-variant title |
| cebuano·2 | Ang Una Na Operasyon ni Dr Sokha | `Ang Una Na` | title fragment |
| cebuano·1 | Ang Hardin Ni Balaraw | `Si Balaraw` | author byline |
| bikol·2 | Maniwala Ka Saro akong Kuneho | `Maniwala Ka` | title fragment |

**Review-flag tallies (kept-and-flagged or dropped-and-logged), 387 total:** residual_prefix 252, title_repeat 26, residual_boilerplate 22 (Tagalog, dropped), pub_marker_in_body 21, bio_tail 14, duplicate 10, empty 10, numbered_list 10, too_short 8, no_opener 6, english_caption 5, digit_interleave 2, bilingual 1.

---

## 8. Defects the evaluation caught and fixed (process efficacy)

The evaluation is not theoretical — it localized concrete, high-impact defects that a manual skim would likely have missed:

1. **Catastrophic amputation** — "Ang Una Na Operasyon ni Dr Sokha" (1,057→4 words) and other stories truncated at inline publisher tokens / the loanword "Grade". Now recovered.
2. **Silent whole-story loss** — the Bikol "Ngiyaw! Ngiyaw!" (587 words) collapsed to 0 by a leading publisher attribution. Now retained (254 words, flagged).
3. **Tagalog pipeline non-functional** — the loader pointed at a non-existent path; 0 Tagalog documents were being produced, and the shipped Tagalog outputs were stale. Now 239 documents cleaned and included.
4. **English boilerplate leakage** — residual English notices inside otherwise-native narratives. Now 0.

---

## 9. Known limitations & residual risk (declared)

- **No external answer key for 5 of 7 languages** (Part A is Cebuano/Bikol only). Mitigated by accountability + invariants + (recommended) grade-level distribution checks.
- **Part B over-flags native-language boilerplate** (bios, acknowledgments, publisher attributions written in the target language). This is a property of the coarse token-alignment heuristic, not a data defect; the per-span audit (§3.2) is the correct reference for narrative loss.
- **Two Bikol reproduction misses** (off by two tokens, `&nbsp;` artifacts).
- **Residual leading author-blocks** on a small number of otherwise-good stories are kept and flagged (`residual_prefix`) rather than removed — a deliberate "flag, don't mangle" choice.
- **Semantic corruption** that is structurally invisible cannot be fully excluded without a fluent reader; it is bounded by the deletion-only design and the small, adjudicated review queue, and is declared as a limitation.
- **Tagalog per-span audit** coverage is pending (currently validated via Part C + density net).
- **Downstream staleness:** the feature matrix (`output/all_features.csv`) and data splits are computed from the normalized text; after this regeneration they must be rebuilt (`extract_features.py`, `split_datasets.py`) before modeling.

---

## 10. Reproducibility

The full evaluation is reproducible from the raw source repositories:

```bash
python build_datasets.py        # raw -> output/ (cleaning)
python normalize_datasets.py    # output/ -> output_normalized/ (normalization)
python validate_denoising.py    # gate scorecard -> output/validation_report.txt, validation_queue.txt
python audit_cleaning.py        # per-span ledger -> output/cleaning_ledger.txt, cleaning_review.txt
```

All stochastic steps use a fixed seed; the checks are deterministic and the reports are versioned with the corpus.

---

## 11. Deliverables / artifact index

| Artifact | Produced by | Contents |
|---|---|---|
| `output/validation_report.txt` | `validate_denoising.py` | pass/fail scorecard (Parts A/B/C) |
| `output/validation_queue.txt` | `validate_denoising.py` | documents flagged for a human glance, with reasons |
| `output/cleaning_ledger.txt` | `audit_cleaning.py` | **every** removed span, indexed + categorized |
| `output/cleaning_review.txt` | `audit_cleaning.py` | narrative-loss suspects only (currently 6) |
| `output/flagged_for_review.txt` | `build_datasets.py` | per-document flags from the cleaning pass |
| `output_normalized/all_languages.txt` | `normalize_datasets.py` | final denoised corpus (schema `language|level|text`) |

---

*This report documents the evaluation methodology and its current results. It should be read alongside Chapter 3.1 (§3.1.2 Cleaning, §3.1.3 Normalization) of the thesis; the acceptance rubric in §6 is the concrete answer to "how do we know the files are truly cleaned?"*
