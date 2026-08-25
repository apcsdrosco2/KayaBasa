# RQ4 / RO4 Implementation Plan — Cross-Lingual Transfer to Low-Resource Languages

Branch: `ara-close-lang` (working tree, uncommitted). **Note: this pipeline was
originally built in a separate worktree on branch `feat/rq4-cross-lingual-transfer`;
it has since been consolidated into this primary repo/worktree instead, because the
raw source data (`code/low_resource_data/`) lives here, untracked, not in that
worktree — see §2D. The `feat/rq4-cross-lingual-transfer` worktree/branch is now
stale; this file and `code/generated/rq4/`, `clean/rq4/` here are the current, live
copies.**
Status: **Document count question CLOSED, final: 769 (user-approved). Stage A
(staging), Stage B (feature extraction), and Stage E (full training ARFF) all done.**
769/769 low-resource rows staged and featurized, nothing dropped. Stage E built
`code/generated/rq4/generated/all_languages_full_train_all_xlmr.arff` — 764 rows
(265 Tagalog + 349 Cebuano + 150 Bikol, full/unsplit, matches Chapter 3 exactly),
793 columns (24 trad_clgsngo + 768 xlmr + class), via `train_test_split.py`'s
`load_language_features()` (no train/test slicing). Remaining: Stage C (embeddings,
blocked on memory), Stage D (zero-shot test ARFFs per low-resource language, needs
Stage C's embeddings), Stage F (zero-shot Weka evaluation).

**Stage C smoke test (2026-08-25)**: tried a minimal tokenizer/model-load-only test
(`code/generated/rq4/smoke_test_model_load.py`) at 283MB free, weaker than the full
pilot. Result: failed **safely** this time (a Python `MemoryError`, not a host crash
like the previous attempt in §8) — but revealed the problem is worse than assumed:
just `import torch`/`transformers`, before touching the model at all, drops free
memory from 283MB to 82MB. The tokenizer's `json.load` over the 9MB `tokenizer.json`
then fails immediately at 82MB. Conclusion: need meaningfully more free memory than
previously estimated — at least ~1.5-2GB free for reasonable margin (import overhead
~200MB + model ~1.1GB + working memory), not just "more than 358MB." Not yet
retried since.

**Stage C moved to a remote/cloud agent (2026-08-25)**: after the local machine kept
failing, pushed the RQ4 code (scripts only, no data) to branch
`feat/rq4-remote-embeddings` (commit `308c9d7`) and dispatched a remote agent to run
Stage C there, sidestepping this machine's memory entirely. First remote attempt
landed in an unrelated stale worktree with none of this repo's state (confirmed
remote agents only see pushed git state, not local uncommitted files — this is why
the push was needed). Second attempt (in progress/pending as of this note) was given
the pushed branch name directly and instructed to independently clone
`imperialite/BasahaCorpus-HierarchicalCrosslingualARA` at `bf3b40f8` itself for the
raw text (via the new `RQ4_RAW_BASE` env var override added to `stage_raw_text.py`),
run staging + pilot + full embedding extraction, then commit+push just the 4 output
CSVs back to `feat/rq4-remote-embeddings`. Check that branch for results before
re-running anything.

**Result: staging succeeded (769/133/268/173/195, exact match), but the pilot
embedding extraction failed the same way as locally.** The remote agent checked
actual free memory before running anything and found **7.78GB total, 100-700MB
free (91-99% used)** — i.e. this "remote" agent was running on the *same physical
machine* as every local attempt, not a separate, higher-memory environment. Same
processes observed as the culprits (multiple VS Code windows, Word, multiple
claude.exe instances). The pilot crashed with a Rust-level allocator abort
(`memory allocation of 67067339 bytes failed`, in the tokenizers/sentencepiece
native bindings) before even reaching the model load — contained to the Python
subprocess this time, no host crash. **Conclusion: the remote-agent approach does
not sidestep this machine's memory constraint in this environment** — it needed to
be re-verified rather than assumed, and turned out not to hold. Back to square one
on Stage C: either free real memory on this one physical machine, or use a smaller-
footprint approach (quantized/ONNX XLM-R) instead of the full fp32 model.

**Root cause of "why was it fast the first time?" identified**: the original
Tagalog/Cebuano/Bikolano XLM-R embeddings were never computed on this local machine
at all — this project's own (now on `main`, not `ara-close-lang`) `COLAB_GUIDE.md`
confirms the established practice is Google Colab (free T4 GPU, ~12GB+ RAM), for
exactly this kind of GPU/memory-heavy step. Every attempt so far (local, and the
"remote" agent that turned out to be the same physical machine) was on the one
memory-starved 7.4GB machine — Colab is a genuinely different, adequately-resourced
machine, unlike the remote-agent attempt. `extract_embeddings_xlmr.py` was patched to
use CUDA when available (previously CPU-only in code, though always intended to run
frozen/no-fine-tuning either way). A copy-paste Colab walkthrough is at
`code/generated/rq4/COLAB_RQ4_EMBEDDINGS.md` — clone this branch + a fresh
BasahaCorpus clone, stage, pilot, full run, download the 4 output CSVs, drop them into
`code/generated/rq4/embeddings/` locally. This needs the user to actually run it in a
browser (no browser/Colab tool available here) — not yet done as of this note.

Note: `main`'s `COLAB_GUIDE.md`/`COLAB_AB_FEATURES.py` describe a **different, older,
fine-tuning-based** architecture (Config D: XLM-R + linear head, fine-tuned end-to-end,
5-fold CV) — this is the abandoned proposal-era direction (see the
`kayabasa-proposal-vs-implementation-gap` project note), NOT the current frozen-
embeddings-into-Weka pipeline this RQ4 work follows. Only the Colab-as-infrastructure
pattern was reused, not that script.

## Stage C/D/E/F — COMPLETE (2026-08-25)

User ran the Colab walkthrough successfully and uploaded the 4 embedding CSVs
(`{lang}_xlmr_features.csv`, 768-dim, one row per document). Verified before use:
133/173/268/195 rows (=769 total, matches exactly), zero NaN, plausible norms
(18.4-18.9, tightly clustered — the shared minimum norm across 3 languages is exactly
what's expected from the 6 `empty_after_cleaning` documents producing identical
trivial embeddings, a good sign these are real, not fabricated).

- **Stage D** (`build_test_arffs.py`): merged Stage B's 24-dim features with these
  embeddings by row order (verified via row-count parity, no shared doc_id column
  exists between the two source files) into `{lang}_test_all_xlmr.arff` per
  low-resource language — 793 columns, 769 rows total, matches the training ARFF's
  exact schema.
- **Stage F** (`rq4_zero_shot_eval.py`): also built the `trad_clgsngo`-only train/test
  ARFFs this stage needed (features-only condition), then ran the full 16-cell matrix
  ({RandomForest, Bagged MLP x10 h128} x {trad_clgsngo, all_xlmr} x 4 languages) via
  `cv_common.run_weka`. **All 16 runs succeeded — no silent-failure (all-zero) rows.**
  Outputs: `code/generated/pairwise/arff/results_rq4_zero_shot/{results_summary_rq4.csv,
  results_detailed.txt, results_bisayan_vs_rinconada.csv}`.

**Headline result**: RandomForest baseline slightly outperforms the bagged-MLP hybrid
on this zero-shot task for BOTH feature sets (e.g. all_xlmr: RF 69.7% mean
Bisayan accuracy / 0.624 macro-F1 vs. MLP 64.3% / 0.612) — opposite of the
in-distribution pattern RO1-3 found favoring the MLP hybrid. All_xlmr beats
trad_clgsngo-only for both classifiers. Rinconada (Bikol-affiliated) consistently
scores lower than the Bisayan-affiliated mean (Hiligaynon+Karay-a+Minasbate) across
every classifier/feature-set combination, consistent with the hypothesis that
cross-lingual transfer works better within the same language subgroup the model
was trained on (Tagalog/Cebuano/Bikolano — note Bikolano IS in the training set,
yet Rinconada, also Bikol-affiliated, still scores lowest; worth flagging as a
finding to discuss, not just confirming the hypothesis cleanly).

Pipeline is functionally complete end-to-end. Remaining: Stage G (write up results in
`docs/Chapter4_Draft.md`'s RO4 section — not yet done), and a decision on whether to
commit the RQ4 code + condensed result CSVs (not raw detailed logs, per project git
hygiene convention) — not yet committed/pushed to `ara-close-lang`.

Note on local working state: this repo's primary working branch is `ara-close-lang`;
the RQ4 code/plan doc are committed only on `feat/rq4-remote-embeddings`, which is
checked out in the remote agent's own worktree and so can't also be checked out here
— this file and the `code/generated/rq4/*.py` scripts are restored as untracked
working copies on `ara-close-lang` (via `git show feat/rq4-remote-embeddings:<path>`)
so local work can continue without conflicting with the agent's worktree. Paper's
Table VI reports 776 (Hiligaynon 133, Minasbate 271, Karay-a 177, Rinconada 195);
actual on disk (both the GitHub repo and this project's local
`code/low_resource_data/raw`) is 133/268/173/195 = 769. The 7-document gap (Minasbate
G1×1/G3×2, Karay-a G1×1/G3×3) was checked exhaustively and confirmed unrecoverable
from every source available to this environment (fresh clone, git object database,
GitHub API tree, the repo's own feature CSVs, its only fork, all PRs/refs, and the
local low_resource_data folder — all agree at 769; Let's Read Asia/Bloom Library are
JS-only SPAs this environment can't render). **769 is the final corpus, nothing
further to chase on the count.** `stage_raw_text.py` keeps every one of the 769 raw
files as a row, none dropped (§2C). Proceeding to Stage B (feature extraction).
Embedding-extraction pilot (Stage C) is still separately on hold pending free memory
(§8).

## Coordinator decisions incorporated (this revision)

1. **RandomForest baseline + features-only condition are IN SCOPE** for the RQ4
   zero-shot matrix, not optional.
2. **Document count: user approved 768** (external-source chasing closed — see §2A).
   **Implementation then surfaced 2 more real cleaning defects (§2B), correcting the
   verified final total to 761** — Hiligaynon 130, Minasbate 265, Karay-a 173,
   Rinconada 193, sourced entirely from
   `BasahaCorpus-HierarchicalCrosslingualARA`'s own `data/raw/`. No further chasing of
   external sources; this is a within-repo cleaning-logic correction, not a new
   attempt to reach 776.

## 0. Setup note (worktree was stale)

The worktree this task started in was not actually based on `ara-close-lang` — it was
still on old `main`/`feat/feature-pipeline-refactor` history and was missing the entire
`code/generated/pairwise/arff/` pipeline. Before any research, I branched
`feat/rq4-cross-lingual-transfer` directly off the local `ara-close-lang` ref (commit
`bc2b4e4`, "Fix missing f1_macro in newer CV runs; add reparse_all_metrics.py") instead
of the worktree's own stale branch. This worktree now has the correct pipeline files.
`docs/Chapter3_Revised.md` and `docs/Chapter4_Draft.md` are untracked-only and exist in
the **primary** worktree (`C:\Users\sophe\Desktop\KayaBasa\docs\`), not this one — I read
them there for grounding; they don't exist here yet.

## 1. What RO4 actually requires (restated from the task)

Train the combined model (XLM-R embeddings + 24 linguistic features → bagged MLP) on
**all 764** Tagalog+Cebuano+Bikolano documents, no held-out split. Test **zero-shot**
(no target-language training data, no fine-tuning) on each of Hiligaynon, Minasbate,
Karay-a, Rinconada's full data. Compare the three Bisayan-affiliated languages
(Hiligaynon, Karay-a, Minasbate) against Rinconada (Bikol-affiliated) specifically.

## 2. What I found during research (this changes the plan vs. the task's framing)

- **Document counts, as staged in `data-cleaning-repo/output/`**: 759 total across the
  four languages (matches Chapter 4's cited figure exactly once each file's header row
  is excluded from the raw `wc -l`). **This turned out to undercount by 9 genuinely
  recoverable documents due to a real pipeline bug — see §2A below for the full
  investigation and the corrected 768-document figure this plan now uses.**

  | Language | L1 | L2 | L3 | Total |
  |---|---|---|---|---|
  | Hiligaynon | 65 | 22 | 46 | 133 |
  | Minasbate | 121 | 73 | 68 | 262 |
  | Karay-a | 60 | 32 | 79 | 171 |
  | Rinconada | 117 | 35 | 41 | 193 |
  | **Total** | | | | **759 (superseded, see §2A → 768)** |

  Files: `data-cleaning-repo/output/{lang}_level{1,2,3}.txt`, pipe-delimited
  `language|level|text` with a header row. No empty rows. Text length ranges from 29 to
  ~10,600 characters. `data-cleaning-repo/output/flagged_for_review.txt` already tracks
  10 known quality issues across these four languages (6 minasbate, 2 karay-a, 2
  rinconada, 0 hiligaynon — mostly `residual_prefix`), which I did **not** attempt to
  fix — 10/759 is small, and I found at least one additional un-flagged junk row myself
  ("Table of Contents Guide Cover", hiligaynon L1) that its own flagging pass missed.
  This is a genuine residual data-quality gap; see Open Question 5.

- **`data-cleaning-repo/extract_features.py` has a real, previously-unconfirmed bug**:
  it only computes 5 of the main pipeline's 11 SYLL-group features (`cons_cluster`,
  `cv`, `cvc`, `ccvc`, `ccvccc` — missing `v`, `vc`, `vcc`, `cvcc`, `ccv`, `ccvcc`), so
  it produces a 19-column feature set, not the 24-column `trad_clgsngo` set the trained
  model actually expects. Its dependency chain (`preprocess.py` → `corpus_cleaned.pkl`
  → `extract_features.py`) is also unverified to run end-to-end, per the task brief.
  **Decision: don't use it at all.** Instead I'll call the KayaBasa repo's own
  `code/TRAD.py`, `code/SYLL.py`, `code/CLGSNGO.py` directly against the low-resource
  text — these are the exact modules that produced the 24-column feature set the model
  was trained on for Tagalog/Cebuano/Bikolano (verified column-for-column against
  `code/generated/tag docus/arff/tagalog_trad_clgsngo_xlmr.arff`'s header), so this
  guarantees schema parity by construction instead of by hoping two independent
  implementations agree. This also removes `preprocess.py`/`corpus_cleaned.pkl` from
  the critical path entirely — I'll read straight from the already-cleaned
  `output/{lang}_level{N}.txt` files.

- **`code/CLGSNGO.py` has a hardcoded, currently-broken path**: it reads anchor n-gram
  files from `<repo root>/data-cleaning-repo-main/ngrams list/`, a directory that does
  not exist in either worktree right now. The three existing languages' CLGSNGO
  features were evidently generated at some past point when that directory *did*
  exist locally, then the directory was removed — the resulting CSVs
  (`{lang}_clgsngo.csv`) are already committed, so this has been silently unused since.
  The actual anchor files exist at `data-cleaning-repo/ngrams list/*.txt` (6 files,
  tiny). **Decision: don't touch `CLGSNGO.py` (shared file, used by the RO1–3
  pipeline)** — instead stage a copy of those 6 files at the exact path it already
  expects (`data-cleaning-repo-main/ngrams list/` at repo root). This is adding
  previously-missing data, not changing behavior, and incidentally un-breaks that path
  for anyone re-running the existing pipeline too.

- **No XLM-R embedding extraction script exists anywhere in the repo.**
  `code/extract_embeddings.py` only does mBERT (`bert-base-multilingual-uncased`). The
  XLM-R embeddings used for Tagalog/Cebuano/Bikolano must have been produced on Colab
  per Chapter 3 §3.0 and the resulting CSVs copied in — there's no script to reuse, so
  I'll write one, mirroring `extract_embeddings.py`'s exact recipe (frozen
  `xlm-roberta-base`, 512-token truncation, mean-pool over attention mask, 768-dim).

- **`torch`/`transformers`/`sentencepiece` confirmed absent** from this Python 3.12.10
  environment. Weka 3.8.7 + bundled JRE confirmed present and working at the paths
  `cv_common.py` hardcodes. Machine: 8 cores, ~7.4GB RAM (matches `cv_common.py`'s own
  documented constraint), 140GB free disk — installing torch+transformers+sentencepiece
  (roughly 2–3GB with dependencies) plus downloading `xlm-roberta-base` (~1.1GB) is not
  a disk/space concern, only a time-and-memory one.

- **`train_test_split.py`'s `load_language_features(lang)` already loads each
  language's full, unsplit feature set** (not fold-split) — this is exactly what RO4's
  "train on all 764 documents, no held-out split" needs. I can build the RO4 training
  ARFF by concatenating `load_language_features("tagalog")["all_xlmr"]`,
  `["bikol"]["all_xlmr"]`, `["cebuano"]["all_xlmr"]` with no train/test slicing at
  all — no new splitting logic needed.

- **`cv_common.run_weka` / `parse_weka_output` are directly reusable** for the
  zero-shot scoring step (they don't know or care whether the test set's language was
  in the training data). I will reuse them rather than re-implementing Weka
  invocation or metric parsing.

## 2A. Document-count investigation (759 vs. target 776) — 9 recovered by a verified bug fix; the other 8 are a confirmed-genuine repo shortfall with a real, still-open recovery lead (not yet retrievable with this environment's tools)

**Status: 759 → 768 recovered via a fixed pipeline bug (verified, zero regressions on
the other 750 documents), independently re-confirmed as a real GitHub-repo shortfall
(not a discovery bug on my end — see the historical-max-count check below). The last 8
are NOT confirmed unrecoverable — I found a specific, credible lead (a Bloom Library
bookshelf series with matching per-language/grade collections) and specific candidate
titles, but could not retrieve actual content due to a tooling wall (both source sites
are JS-only SPAs my fetch tool can't render, and the Wayback Machine is blocked
outright). Full trail below, per the request not to silently settle on a lower number.**

### Method

`data-cleaning-repo/BasahaCorpus-HierarchicalCrosslingualARA/` is empty on disk (a
gitlink was committed without a `.gitmodules`, so it never resolves on a fresh clone).
I cloned the actual upstream repo (`imperialite/BasahaCorpus-HierarchicalCrosslingualARA`)
into scratchpad and checked out the exact commit the gitlink records (`bf3b40f`), which I
confirmed **is the current tip of upstream `main`** — there is no newer upstream data to
pull. All investigation below is against that exact pinned snapshot, entirely in
scratchpad; nothing in `data-cleaning-repo` or outside this worktree was modified.

### Two real, fixable bugs in `data-cleaning-repo/build_datasets.py`'s `clean_basaha()`

Both are in the block-classification logic that decides which leading blocks (title,
author names, publisher attribution) to skip before the real narrative body starts.

- **Bug A — parenthesized nicknames break name-block detection.** `_ALLCAP_WORDS`'s
  character class doesn't include `(` `)`, so an author line like
  `"Tassaya (Toffy) Charupatanapongse"` fails to match and isn't recognized as a
  skippable name block. The next block (a publisher-attribution line) then gets
  wrongly treated as the start of the real body, and the tail-boilerplate cutter
  (`_BASAHA_TAIL`) truncates the entire real story down to nothing, because it matches
  that publisher name near the very start of what it thinks is the body.
  Confirmed on: the same underlying "Lights! Camera! Action!" story, independently
  translated into 3 of the 4 languages (Minasbate G2, Karay-a G2, Rinconada G3) — same
  bug, same source book, three language editions.

- **Bug B — native-language acknowledgement sentences aren't recognized as skippable
  front matter.** Many documents open with a full sentence like *"Ini na libro ginhimo
  san Srijanalaya sa bulig san The Asia Foundation's Book for Asia..."* ("This book was
  made by ... with the help of ..."). `_LEAD_ATTRIB` only skips a block if it **starts
  with** a hardcoded English org name; a native sentence that merely **mentions** the
  org mid-sentence isn't skipped, so it's wrongly treated as body-start — and then the
  same `_PUB`/`_BASAHA_TAIL` sentinel (meant to catch *trailing* boilerplate, containing
  phrases like `"Ini na libro"`, `"(?:The )?Asia Foundation"`) matches within that
  sentence's own first few words, truncating the real story to 0–3 words before it ever
  begins. Confirmed on 6 documents across Minasbate/Karay-a/Rinconada, all sharing this
  same acknowledgement-sentence template in different dialect spellings ("sa bulig
  san/kang", "sa tabang ka").

I wrote a corrected `clean_basaha()` (scratchpad only:
`rq4_investigation/fixed_clean_basaha.py`) that (A) allows a parenthesized token in a
name-block run, and (B) also treats a leading block as skippable front matter if the
`_PUB` sentinel matches within roughly its first 70 characters (i.e. the block *is*
substantially a publisher/funder acknowledgement, not narrative). I ran this over
**every** raw file for all 4 languages/3 grades (not just the 10 already-flagged ones)
and diffed against the current `build_datasets.clean_basaha()` output:

| Language | Old kept | New kept | Recovered |
|---|---:|---:|---:|
| Hiligaynon | 65/22/46 = 133 | unchanged | 0 |
| Minasbate | 121/73/68 = 262 | 123/77/68 = 268 | +6 (net; see below) |
| Karay-a | 60/32/79 = 171 | 60/34/79 = 173 | +2 |
| Rinconada | 117/35/41 = 193 | 117/36/42 = 195 | +2 |
| **Total** | **759** | **769 raw** | **+10 raw** |

**Zero regressions**: no document that was previously kept got newly dropped by the fix.

**One of the 10 "recovered" documents is not a genuine recovery**: Minasbate G1
`Paglinis_san_Lawas` (title: "Cleaning the Body") crosses the 4-word minimum only
because the fixed cleaner picks up more of the same English legal boilerplate
("*Translation of work: If individuals or legal entities translate this work...*") —
the raw `.txt` file's own Table of Contents points to "Page 12" but **no page content
was ever included in the raw export**. This file genuinely contains zero narrative
text in the source; I'm not counting it as recovered. **Net genuine recovery: 9
documents** (759 → 768).

**Bonus finding, not part of the count question but worth flagging**: the same two bugs
also *silently corrupted* (truncated, not dropped) at least 4 more documents that
stayed above the 4-word floor and were therefore invisible in both the document count
and `flagged_for_review.txt` (which only logs sub-4-word drops) — e.g. Hiligaynon
`Ngiyaw_Ngiyaw` went from a 9-word truncated stub ("*Ini nga libro ginhimo sang
Srijanalaya sa bulig sang*") to its real ~380-word story once the fix is applied. These
were found by diffing word counts on documents kept under *both* versions, not by the
count investigation itself — I did not exhaustively hunt for more of these beyond the 4
found, since it's outside the specific question asked, but it suggests the true
data-quality benefit of this fix is larger than just the 9 recovered documents.

### The remaining 8-document gap (768 → target 776): confirmed unrecoverable from the current upstream source

Before any cleaning, the **raw file counts on disk** already fall short of the
proposal's Table II/III target for two languages — meaning no cleaning-logic fix could
ever close this part of the gap, the files simply aren't in the upstream repo:

| Language/Grade | Raw files on disk | Table II/III target | Gap | Cause |
|---|---:|---:|---:|---|
| Minasbate G1 | 123 | 124 | 1 | Raw source has 1 fewer file than the proposal stated |
| Minasbate G1 | (123, incl. above) | — | +1 more | `Paglinis_san_Lawas` exists but has zero narrative text (see above) — corrupt source file |
| Minasbate G3 | 68 | 70 | 2 | Raw source has 2 fewer files than the proposal stated |
| Karay-a G1 | 60 | 61 | 1 | Raw source has 1 fewer file than the proposal stated |
| Karay-a G3 | 79 | 82 | 3 | Raw source has 3 fewer files than the proposal stated |
| **Total gap** | | | **8** | |

I confirmed there's no newer upstream commit, no alternate folder, and no
non-`.txt`/hidden files being missed (checked directory listings directly). I also found
**independent corroboration already sitting in this project's own files**:
`data-cleaning-repo/supplementary_context/KayaBasa_Chapter3.1_Revised.md` (§3.1.1.2),
an internal design note, states verbatim: *"the disk counts diverge from previously
stated figures (for example, Minasbate Grade 1 yields 123 files on disk against a
previously stated 124). The Results chapter reports the reconciled counts produced by
the loader, not the planning estimates."* — i.e., a teammate already independently
discovered and documented this exact same discrepancy, and concluded the proposal's
Table II/III was a **pre-data-collection planning estimate**, not an empirically
verified count. My fresh investigation reaches the same conclusion via a different
path (a live re-clone of the pinned upstream commit).

**Update — user independently confirmed the target and asked me to keep investigating
before accepting 768.** The user checked the paper's own Table 3 directly (776 total,
matching exactly) and checked the GitHub repo's full commit history via the GitHub API,
confirming the only deletions ever made were placeholder scaffolding, never real
documents. I re-verified this independently and more rigorously than a deletion-filter
diff: I walked **every commit** in the repo's history that touched
`data/raw/minasbate` or `data/raw/karay-a` (`data/raw/kinaray-a` included) and measured
the file count at each one, not just filtered for delete-type diffs. Result: file count
increases monotonically at every commit for both languages, and **the current HEAD is
the historical maximum** — 268 for Minasbate, 173 for Karay-a, never higher at any
prior point. This closes the file-discovery-bug question definitively: it isn't a
missed rename, a squashed add+delete, or a scaffolding artifact — these files were
simply never committed to this repository, ever. I also re-verified case-sensitivity,
extension variants, and nesting depth via `git ls-tree -r` (case-sensitive, authoritative,
independent of my earlier OS-level directory walk): exactly 12 directories
(`hiligaynon|karay-a|minasbate|rinconada` × `grade 1|2|3`), all lowercase, no
alternate-spelling siblings, 769 files total, all `.txt` (one had a smart-quote
character in its filename, which is why a naive extension-regex miscounted it by 1 in
my first pass — confirmed to still be a normal `.txt` file, already correctly included
in every count above). **The "kinaray-a" spelling was real but was pure scaffolding**
(a placeholder file created at 22:51, deleted at 22:55, replaced by the correctly-spelled
`karay-a` folder — all within 4 minutes, before any real files were ever uploaded, per
commit timestamps) — no data was ever lost in that rename.

### Let's Read Asia / Bloom Library search — genuine attempt, blocked by tooling, not by absence of leads

I confirmed the paper's own data-availability statement (fetched from the arXiv HTML,
§ data availability): *"Let's Read Asia is an online library... The researchers obtained
explicit permission from Let's Read Asia to use this data... licensed under Creative
Commons BY 4.0."* I also pulled the paper's full Table 2 (word/sentence-count means per
language/grade) for future cross-validation of any candidate document — see table below.

I attempted direct access to both named source platforms and could not retrieve usable
content from either, for a structural reason, not a search-effort reason: **both
`letsreadasia.org` and `bloomlibrary.org` are pure client-side-rendered single-page
apps.** Every URL I fetched on either domain — site root, `robots.txt`, `sitemap.xml`,
specific collection pages, specific book pages, a specific book-reader/player page —
returned only an empty shell or a literal `"Loading..."` placeholder to `WebFetch`
(which does not execute JavaScript). I confirmed this isn't a fluke by trying ~10
different URLs across both domains. My fallback (the Wayback Machine, the standard
workaround for JS-SPA archaeology) is blocked outright at the tool-infrastructure level
(`"Claude Code is unable to fetch from web.archive.org"`). Bloom Library does have a
real read API (OPDS, documented at `docs.bloomlibrary.org`) but it requires an account +
API key I don't have (`https://api.bloomlibrary.org/v1/opds?...&key=ACCOUNT:KEY`) — I
have no credentials to use it. The Global Digital Library (a different platform I
initially suspected might mirror this content) turned out not to host these languages at
all, and its own API returned 403. I checked for another web/browser tool in this
environment (`ToolSearch` for playwright/puppeteer/browser-rendering tools) — none is
available; `WebFetch` is the only web-access tool I have, and it cannot render JS.

**What I did establish despite the access wall** — real, structural evidence the
documents likely exist somewhere, even though I can't retrieve them:
- Bloom Library organizes Philippine-language content into a per-language,
  per-grade bookshelf series named `ABCPhilippines-{Language}-Grade{N}` — I confirmed
  `ABCPhilippines-Minasbate-Grade1` exists and contains at least one real, findable book
  (`bloomlibrary.org/#!/ABCPhilippines/ABCPhilippines-Minasbate-Grade1/book/F9Kr6qbQCS`),
  and search results reference sibling collections (`...-Rinconada-Grade1`,
  `...-Cebuano-Grade1`, etc.) in the same series — consistent with the "ABC+: Advancing
  Basic Education in the Philippines" project boilerplate that already appears at the
  tail of several of our own raw `.txt` files. I could not enumerate this bookshelf's
  contents or get a count (JS wall), but its existence is a genuine, specific, and
  well-targeted lead for anyone with browser or API access to Bloom Library.
- I cross-referenced our own already-downloaded raw filenames across all 4 languages
  (normalizing away version timestamps and language suffixes) to find story titles that
  exist in ≥2 of the other 3 languages but are absent from Minasbate G1/G3 or Karay-a
  G1/G3 specifically — since this corpus is built from parallel translations of a shared
  story pool (e.g. "Lights! Camera! Action!" already confirmed translated into 3 of the
  4 languages). This produced concrete, specific candidate titles to check against the
  Bloom Library bookshelves above:
  - **Minasbate G1/G3 candidates**: "Aaloo-Maalo-Kaaloo", "Ang Buyog kag ang Elepante",
    "Ano Ayhan", "Emma", "Magisip Kita", "May Torotot sa Saranggutan", "Ngiyaw Ngiyaw",
    "Rudi", "Si Putu kag si Gutu", "Si Tata kag si Toto" (10–12 candidates; note these
    were found at *varying* grade levels in the other languages, so I can't assume a
    Minasbate version — if it exists — would land in G1 vs. G3).
  - **Karay-a G1/G3 candidates**: "An Balaybalay", "An Garden sa Atop", "An Maisog na si
    Tori", "Aninipot", "Ano Ayhan", "Dalagan Dalagan Dalagan", "Emma", "Magbilang Kita",
    "Nahadlok Ako", "Ngiyaw Ngiyaw", "Rudi", "Su Sirom saka su Tinapay", "Tamtam" (13–14
    candidates, same caveat).

**Honest conclusion on this lead**: I located what is very likely the right source
(Bloom Library's `ABCPhilippines` bookshelf series) and a specific, evidence-based
candidate title list, but **I cannot confirm, download, or word/sentence-count-validate
any specific missing document with the tools available in this environment** — the
blocker is JS rendering + missing API credentials, not an absence of leads.

**CLOSED — user decision.** The user independently hit the identical JS-rendering wall,
and separately checked a HuggingFace mirror with pre-computed BasahaCorpus-like feature
CSVs, which turned out to be a different/larger dataset vintage with no matching raw
text available — so that lead is closed too. **Final: 768 documents, sourced entirely
from `BasahaCorpus-HierarchicalCrosslingualARA`'s own `data/raw/` via the bug-fixed
cleaner, is the corpus this project uses for RQ4.** No further external-source chasing.

**Paper's Table 2 (arXiv HTML, for future validation once/if candidate texts are found)**:

| Language | Grade | Docs | Mean words | Mean sentences | Vocabulary |
|---|---|---:|---:|---:|---:|
| Hiligaynon | L1/L2/L3 | 65/22/46 | 198.8/296.5/610.0 | 20.7/39.1/57.3 | 2043/1539/4137 |
| Minasbate | L1/L2/L3 | 124/77/70 | 240.4/360.5/578.6 | 28.7/43.0/62.5 | 3836/4097/5520 |
| Karay-a | L1/L2/L3 | 61/34/82 | 191.0/410.9/569.1 | 21.4/47.7/60.3 | 1937/2309/6264 |
| Rinconada | L1/L2/L3 | 117/36/42 | 261.0/521.7/505.2 | 30.0/59.5/55.1 | 4222/3313/3958 |

### Related-but-separate finding (not fixed, flagged only)

4 raw files across Hiligaynon/Minasbate/Rinconada Grade 1 are exact byte-for-byte
duplicates re-downloaded under a `" (1).txt"` suffix (e.g.
`Aaloo-Maaloo-Kaaloo___Hiligaynon (1).txt`, identical to the base file). The current
pipeline does **not** deduplicate these, so they're currently double-counted as 2
separate documents. I have not touched this — deduplicating would *reduce* Hiligaynon
below its target (65 → 61), which is a separate judgment call from the recovery work
above, not something I'm folding into the count fix silently.

### What changes in the RQ4 pipeline plan because of this

Stage A (§3) no longer stages `data-cleaning-repo/output/{lang}_level{N}.txt` directly
— those files carry the 2 confirmed bugs. Instead, the RQ4 staging step will re-derive
cleaned text directly from the raw BasahaCorpus source (already cloned to scratchpad at
the exact pinned commit) using the corrected cleaner, entirely inside this worktree —
`data-cleaning-repo` itself is not modified, consistent with the "don't touch anything
outside the worktree" constraint. This made the RQ4 corpus 768 documents at the time
of this section — since corrected to **761**, see §2B.

## 2B. Post-implementation reconciliation — 768 → 761 (final, verified)

After the 768 figure was approved and I began Stage A implementation
(`code/generated/rq4/stage_raw_text.py`, `fixed_clean_basaha.py`), the coordinator
flagged a discrepancy: raw `wc -l` on the staged `clean/rq4/*.txt` files showed
772 (134/174/268/196), not 768. **That specific gap was trivial**: each file has a
`level|text` header row, so `wc -l` over-counts by exactly 4 (one per language file);
772 − 4 = 768 real rows, confirmed with a proper pandas read (excluding the header).

The coordinator also asked me to check whether "Table of Contents Guide Cover" (a junk
row spotted during the very first, pre-implementation exploration of
`data-cleaning-repo/output/hiligaynon_level1.txt`, a different/older file) had made it
into the new staged corpus. It had — as a literal survivor in 3 Hiligaynon documents,
one per grade. Chasing that down surfaced a real, previously-missed defect, which then
led to finding two more:

- **Fix C**: the structural-marker cleanup (`Table of Contents`/`Guide`/`Cover`/etc.)
  in the original `build_datasets.py` — and in my own Fix A/B version, since I'd
  carried this part over unchanged — runs its `^...$`-anchored regex *after*
  `body.replace("\n", " ")` has already collapsed every block onto one line, so it can
  never match once several markers have been joined by spaces (e.g. `"Table of
  Contents Guide Cover"` as one string never matches a pattern requiring the *whole*
  line to be just `"Guide"`). This let a 4-word residue survive the MIN_WORDS filter
  as a false-positive document whenever a raw file's entire narrative was already
  missing (title + optional author names + bare `Table of Contents / Guide / Cover`,
  no story body at all in the raw export). Fixed by dropping any block that is
  *entirely* a structural marker before joining, at the block level, regardless of
  position. This correctly resolved **6 documents** (3 Hiligaynon + 2 Minasbate + 1
  Rinconada) to genuinely empty — all 6 manually inspected and confirmed to contain
  zero real narrative in the raw file, same corrupt-source class as "Paglinis san
  Lawas" (§2A).
- **Fix B's window was too narrow**: while re-auditing every remaining document under
  ~15-20 words (not just the ones already flagged), I found two more truncated-to-a-
  fragment survivors, both a Rinconada-dialect phrasing of the same acknowledgement
  template ("*A librong adi ay ginibo kan Srijanalaya sa tabang kan programang Book
  for Asia kan The Asia Foundation...*") where the org mention fell past Fix B's
  70-character detection window. Widened to 200 characters (comfortably covers every
  phrasing variant observed, still far shorter than any real narrative would run
  before incidentally mentioning one of these terms).
- **Fix D (new)**: the acknowledgement sentence can itself be split across 2+ blank-
  line-separated blocks in the raw source (a mid-sentence line break) — Fix B's
  per-block check correctly flags the block *containing* the org mention, but by then
  the earlier half-sentence block (which has no org mention on its own) has already
  been committed as body-start, locking in `seen_body=True` before the second half is
  even examined. Fixed with a 2-block/260-character lookahead: if the current block
  doesn't itself qualify to skip, but combining it with the next block produces a
  combined span that *does* contain the org mention, the whole span is skipped as one
  unit. (First attempt at this fix had an ordering bug of its own — checking the
  length-budget cutoff before checking for the match, which silently discarded a
  match sitting just past the budget in an otherwise short combined span — caught by
  testing directly against the specific failing document rather than trusting the
  fix on inspection alone.)

**Two additional documents needed manual exclusion** (real words present, but
confirmed by direct inspection to be zero real narrative, so not caught by the
automatic empty-flag path): Minasbate "Paglinis san Lawas" (§2A, unchanged) and
Rinconada "A Palda Kong Pula" — title + a Lao-script author name (`ສຸກສະດາ ສຸດທິໄຊ`) +
an English co-author name + bare structural markers, no story. `_is_name_block`
requires a Latin uppercase letter (`\p{Lu}`) to recognize a name block, and Lao script
has no case distinction at all, so this specific name block was never classified as
skippable — a real, narrow gap I chose to handle as a manual exclusion (one document)
rather than generalize name-block detection to arbitrary scripts, which would carry
more regression risk for a single-document payoff.

**Final, re-verified total: 761 documents** — Hiligaynon 130, Minasbate 265, Karay-a
173, Rinconada 193. Verified via: (a) a full raw-vs-cleaned diff across all 769 raw
files at every fix stage, confirming exactly which documents changed and why; (b) a
systematic audit of every kept document under 20 words, manually inspecting each one's
full text and raw source rather than trusting the word count alone (two — "Dalagan!
Dalagan! Dalagan!" at 16 words and "Gusto ko an mga Kulay" at 18 words — were
confirmed genuinely complete, simple Grade-1-level stories, not truncation artifacts);
(c) `stage_raw_text.py` hard-asserts the exact expected count, exclusion counts, and
that they sum back to the unchanged raw total (769) every time it runs, so a future
change that silently breaks this accounting fails loudly instead of drifting.

This is very likely still not the theoretical ceiling — I stopped actively hunting
once the under-20-word audit came back clean twice in a row, not because I'm certain
no further defects exist, but per the "genuinely exhaust obvious leads, then report"
pattern established in §2A rather than open-ended perfectionism on a question that
wasn't the one asked. **761 supersedes every earlier figure (759, 768, 772) in this
document and in every script.**

## 2C. User directive (post-§2B): never drop documents during cleaning; target stays 776

The user rejected §2B's 761-document total and gave an explicit standing rule (also
saved to cross-session memory, so future sessions on this project should already know
it): **cleaning must never remove a document — not even one whose narrative body
cleans to empty or near-empty after boilerplate/front-matter stripping.** Concretely,
this means:

- The 6 "Fix C" documents (structural-marker-only survivors, §2B) and the 2 manual
  exclusions ("Paglinis san Lawas" §2A, "A Palda Kong Pula" §2B) must be **kept as rows
  in the staged corpus**, not dropped, regardless of how empty their cleaned text is.
  Reverting all §2B/§2A drops gets the corpus back to **769** (the full raw-file count,
  one row per raw `.txt` file, none filtered out) — this is the correct ceiling from
  currently-downloaded source, NOT 776.
- **769 is still 7 short of 776.** Those 7 (Minasbate G1×1/G3×2, Karay-a G1×1/G3×3, per
  §2A's table) have no corresponding raw file anywhere in the pinned upstream commit at
  all — verified via a full commit-history walk (file count increases monotonically,
  current HEAD is the historical max). No cleaning-logic change can produce text for a
  document that was never downloaded into this repo. Reaching 776 requires actually
  recovering those 7 documents' source text from Let's Read Asia / Bloom Library (both
  JS-only SPAs, blocked for the tools available in this environment so far — see §2A's
  "genuine attempt, blocked by tooling" section for exactly what was tried) or some
  other source, not a pipeline fix.
- **Practical follow-up needed, not yet done**: decide what "keep as a document" means
  for the ~8 effectively-empty rows in scoring — e.g. do they get a placeholder/whatever
  minimal text survived cleaning (even if 0-3 words), and is that acceptable input to
  TRAD/SYLL/CLGSNGO feature extraction and to the MLP, or does keeping them require a
  different accommodation (flag-and-keep vs. impute vs. accept degenerate features)?
  This wasn't resolved before the session ended — flag for whoever continues.

**Net status for whoever picks this up**: re-stage to 769 (un-drop everything §2A/§2B
dropped) as the immediate correct step, then keep pursuing the remaining 7 via new
avenues (a non-JS Let's Read Asia endpoint, direct book-title/candidate-title search
hits, `archive.org` proper as opposed to the blocked `web.archive.org`, or asking the
user directly for Bloom Library API credentials / manual retrieval) rather than
stopping at 769.

## 2D. Consolidation into the primary repo (this file's location)

The RQ4 pipeline (§2A–§2C) was originally implemented in an isolated worktree
(`.claude/worktrees/agent-a1dfa8516627ece45/`, branch `feat/rq4-cross-lingual-transfer`)
because that's where the task started. The user's own copy of the raw BasahaCorpus data
(`code/low_resource_data/{raw,features,ngrams}/`, untracked, 791 files, verified
identical to the pinned upstream commit — same 769 raw `.txt` files, same per-
language/grade breakdown) lives directly in this primary repo, not that worktree.
Rather than maintain two parallel copies, the pipeline has been copied over and
re-pointed at `code/low_resource_data/raw` directly (no separate `_basahacorpus_raw`
staging step needed here, since the data's already local):

- `code/generated/rq4/fixed_clean_basaha.py` — copied over unchanged.
- `code/generated/rq4/stage_raw_text.py` — re-pointed `RAW_BASE` at
  `code/low_resource_data/raw`; logic otherwise identical (769 rows, none dropped, per
  §2C). Re-run here, produces `clean/rq4/{lang}_all_clean.txt` (level|flag|text).
- `code/generated/rq4/rq4_feature_extraction.py` — new (Stage B, see §3). Calls
  `code/TRAD.py`/`SYLL.py`/`CLGSNGO.py` directly, wraps every feature call to fall back
  to `0.0` on `ZeroDivisionError` (only ever triggered by the 6 `empty_after_cleaning`
  rows) rather than crash or skip — 769/769 feature rows written, verified.
- CLGSNGO's hardcoded `data-cleaning-repo-main/ngrams list/` path staged with the
  tag/bik/ceb n-gram files from `code/low_resource_data/ngrams/` (6 files) — this path
  did not exist in the primary repo before now.

The `feat/rq4-cross-lingual-transfer` worktree/branch still exists but is now stale;
future work should happen here instead.

## 3. Planned pipeline (in order)

**Stage A — Stage the raw data (no code, just files + one loader convention)**
1. Copy `data-cleaning-repo/output/{hiligaynon,minasbate,karay-a,rinconada}_level{1,2,3}.txt`
   into the KayaBasa repo, e.g. `clean/rq4/{lang}_level{N}.txt` (sibling convention to
   the existing `clean/{bik,ceb,tag}_all_clean.txt`).
2. Copy `data-cleaning-repo/ngrams list/*.txt` (6 files) to
   `data-cleaning-repo-main/ngrams list/` at repo root, matching `CLGSNGO.py`'s
   hardcoded expectation (see §2).

**Stage B — Feature extraction (24-dim, exact schema match)**
3. New script `code/rq4_feature_extraction.py`: for each of the 4 languages, read the
   staged level files in a fixed order, assign a stable `doc_id` (language + row
   index), call `TRAD.py`/`SYLL.py`/`CLGSNGO.py`'s functions directly (same 7+11+6=24
   functions `trad_parser.py`/`syll_parser.py`/`CLGSNGO_parser.py` call), and write one
   CSV per language with **exactly** the 24 column names/order already used in
   `{lang}_trad_clgsngo.arff` (`word_count`, ..., `cebuano_trigam_sim` — yes, the
   existing repo's `cebuano_trigam_sim` typo is part of the schema I must match
   exactly, not "fix").

**Stage C — Embedding extraction (768-dim XLM-R, frozen)**
4. `pip install torch transformers sentencepiece` (only after this plan is approved).
5. New script `code/rq4_extract_embeddings_xlmr.py`, mirroring
   `extract_embeddings.py`'s recipe but with `xlm-roberta-base` instead of
   `bert-base-multilingual-uncased`, run once over the same staged per-language text
   in the same fixed row order as Stage B. Batch (not all-at-once) with periodic
   progress logging and incremental save-to-disk (e.g. flush every N documents) so a
   crash partway through doesn't lose completed work — this machine has a documented
   history of JVM OOM crashes under memory pressure (`cv_common.py`'s own comments),
   and a long CPU-bound torch job is a new, untested memory-pressure profile on the
   same 7.4GB machine.
6. **Before committing to the full 759-document run**, do a small pilot (e.g. 10–20
   documents) to measure real per-document wall-clock time on this machine and
   extrapolate total time, rather than guessing.

**Stage D — Assemble zero-shot test ARFFs**
7. New script (or extend the Stage C script) to merge each language's 24-dim features
   (Stage B) with its 768-dim embeddings (Stage C) by `doc_id`, matching the exact
   attribute order of `{lang}_trad_clgsngo_xlmr.arff` (24 linguistic features, then
   `xlmr_000`..`xlmr_767`, then `class` ∈ `{1,2,3}` — labels already match, no
   remapping needed since `level` is already 1/2/3 in the source data). Output:
   `{lang}_test_all_xlmr.arff` per low-resource language, in a new results-isolated
   folder (see §5).

**Stage E — Build the full-data (no split) training ARFF**
8. Small script reusing `train_test_split.load_language_features` to load
   Tagalog/Cebuano/Bikolano's full `all_xlmr` DataFrames (no slicing), concatenate all
   764 rows, save as `all_languages_full_train_all_xlmr.arff`.

**Stage F — Zero-shot evaluation**
9. New script `code/generated/pairwise/arff/rq4_zero_shot_eval.py` reusing
   `cv_common.run_weka`/`parse_weka_output`: train once on Stage E's ARFF with the
   established bagged-MLP config (`weka.classifiers.meta.Bagging` wrapping
   `MultilayerPerceptron`, H=128, N=10, 10 bags — the exact config
   `run_final_hybrid_model.py` already fixes for RO1–RO3, for consistency), score
   against each of the 4 Stage D test ARFFs. Persist full raw Weka output to a
   `results_detailed.txt` log and write a `results_summary_rq4.csv` +
   Bisayan-vs-Rinconada comparison table, following this project's established
   never-trust-only-the-incremental-CSV convention.

**Stage G — Validate and write up**
10. Sanity-check every stage per §4 below.
11. Report results back (accuracy, macro-F1 per language, Bisayan-mean vs. Rinconada
    comparison) — updating `docs/Chapter4_Draft.md`'s RO4 section is a natural
    follow-up but that file doesn't exist in this worktree (untracked, primary-worktree
    only) — flagged as Open Question 6.

## 4. Validation checkpoints (per this project's established practice of never trusting a stage blind)

| Stage | Check |
|---|---|
| Data staging | Row counts match today's measured 759 (133/262/171/193); no empty text; cross-reference against `flagged_for_review.txt`'s 10 known-issue rows |
| Feature extraction | Output shape 759×24 per the combined total; no all-null or all-constant columns; 2–3 documents' TRAD counts hand-verified; CLGSNGO bigram/trigram overlaps not all-zero (should show real overlap, especially Bisayan↔Cebuano) |
| Embedding extraction | Output shape matches text row count exactly per language; no NaN; embedding norms in a plausible range; explicit `doc_id`-order check against the feature CSV (not just positional trust) before merging — this is the exact class of silent-misalignment bug this project's conventions are built to catch |
| ARFF assembly | Row count = language's total docs; attribute list diffed line-by-line against the training ARFF's header (order and names must match exactly, since Weka scores by declared attribute structure) |
| Full-data training ARFF | Exactly 764 rows (307+223+234 per Chapter 3's table), all_xlmr schema |
| Weka zero-shot run | Check for the known silent-failure signature (accuracy==kappa==f1_weighted==0.0 simultaneously) on every cell; retry at lower `-Xmx` if seen, matching `retry_failed_cells.py`'s precedent; keep the raw log so results are always re-derivable from scratch |

## 5. Naming / isolation (matching existing conventions)

- New results folder: `code/generated/pairwise/arff/results_rq4_zero_shot/` (parallel
  to `results_cv_*`).
- `.gitignore` additions, extending the existing patterns: staged raw text under
  `clean/rq4/` — likely committed (small, same category as the already-committed
  `clean/*.txt`), embedding CSVs likely **not** committed (regenerable, large-ish:
  ~15MB total estimated), `results_rq4_zero_shot/results_detailed.txt` not committed
  (same rule as existing `results*/results_detailed.txt`).
- Only commit code + condensed summary/table outputs, per project git hygiene
  convention.

## 6. Open questions / judgment calls for the user

1. ~~Zero-shot evaluation scope~~ — **RESOLVED by coordinator**: RandomForest baseline
   + features-only (`trad_clgsngo`) condition are now in scope alongside the bagged-MLP
   hybrid on `all_xlmr`, matching RO2/RO3's comparison pattern. Stage F's matrix is now
   {RandomForest, Bagged MLP} × {`trad_clgsngo`, `all_xlmr`} × 4 test languages = 16
   Weka calls (still cheap, seconds each).
1a. ~~The 768-vs-776 decision~~ — **RESOLVED by user, 768 approved**, then **corrected
   to 761 during implementation (§2B)**, a within-repo cleaning-logic fix (2 more real
   defects found while re-auditing short documents), not a reopening of the
   776-vs-repo-shortfall question, which stays closed (§2A: raw file-count history
   fully reconstructed commit-by-commit — real repo shortfall, not a discovery bug;
   Let's Read Asia / Bloom Library / a HuggingFace mirror all checked and closed).
   **761 documents, sourced entirely from `BasahaCorpus-HierarchicalCrosslingualARA`'s
   own `data/raw/`, is the final, verified RQ4 low-resource corpus.**
2. **Hyperparameters for the final zero-shot model.** I'm defaulting to the
   already-tuned H=128, N=10, 10-bags config for consistency/reproducibility with
   RO1–RO3 rather than re-tuning specifically for the 764-doc full-data condition.
   Re-tuning is possible but likely low-value (§3.2.4 found the MLP largely
   insensitive to H/N in this range) and adds a nontrivial extra step.
3. **CLGSNGO anchor-file staging approach.** I'm placing the missing `ngrams list`
   files at the path `CLGSNGO.py` already hardcodes, rather than editing that shared
   file. If you'd rather fix the hardcoded path itself (e.g. to point somewhere more
   sensible), say so — that's a one-line change but touches code shared with RO1–3.
4. **Embedding extraction wall-clock time is genuinely unknown** until the Stage C
   pilot runs — plan is to measure on a small sample first and report back before
   committing to the full ~761-document run, rather than guess a number now.
5. **Residual known-junk documents.** After the §2A fix, `flagged_for_review.txt`'s
   remaining `residual_prefix` flags (185, mostly Cebuano/Bikol, not low-resource) and
   the one unflagged junk row I found myself ("Table of Contents Guide Cover",
   Hiligaynon L1) are still present. Recommend keeping them (small fraction, matches
   RO4's "full data" wording), but flagging in case you'd rather exclude known-bad rows
   before scoring. Separately, §2A also surfaced that the same 2 cleaning bugs likely
   silently truncated some currently-"fine" documents' text (4 confirmed by spot-check,
   not exhaustively searched) — worth a decision on whether to also re-clean the
   already-correctly-counted documents for quality, not just the 9 previously-dropped
   ones, since the fixed cleaner is a strict improvement with zero regressions found.
6. **Chapter 4 write-up.** `docs/Chapter4_Draft.md` exists only in the primary
   worktree, not this one. I can produce results and a Bisayan-vs-Rinconada comparison
   as data/tables regardless; actually editing that doc would need to happen in (or be
   copied into) the primary worktree, since I'm restricted to this one.

## 7. Explicitly not doing (until told otherwise)

- ~~Not installing torch/transformers/sentencepiece yet~~ — **now installed**
  (`torch 2.13.0+cpu`, `transformers 5.15.1`, `sentencepiece 0.2.2`), see §8.
- Not running `data-cleaning-repo/preprocess.py` or `extract_features.py` (bypassed
  per §2/§3).
- Not modifying `code/CLGSNGO.py`, `code/TRAD.py`, `code/SYLL.py`, or any other file
  shared with the RO1–3 pipeline.
- Not committing or pushing anything.
- **Not re-attempting the embedding pilot or anything that loads the XLM-R model**
  until the coordinator confirms it's safe (§8) — this machine is currently critically
  low on free RAM (~241MB free of 7.4GB, 96% used, confirmed via
  `GlobalMemoryStatusEx`).

## 8. Embedding-pilot status: crashed once, not yet successfully run

**Current state, for anyone picking this up**: `torch`/`transformers`/`sentencepiece`
are installed. `code/generated/rq4/extract_embeddings_xlmr.py` is written and already
has one real fix baked in (see its `main()` comments): the default `AutoTokenizer`
loading path in `transformers 5.15.1` parses the full ~9MB `tokenizer.json` via a
plain `json.load()`, which raised a genuine `MemoryError` 3 times in a row on this
machine even as free memory fluctuated (235MB → 695MB → 296MB, all failed); switching
to the legacy `XLMRobertaTokenizer` class directly (bypassing that code path
entirely, same vocab/tokenization, just a lower-footprint loader) fixed that specific
failure and successfully loaded once in isolation.

**However, the embedding pilot itself (`python extract_embeddings_xlmr.py --pilot 5`)
has NOT yet completed successfully.** The one dispatch attempt coincided with the
Claude Code host process itself crashing (Windows OOM, exit code 3221226505) — the
~1.1GB XLM-R model load very likely tipped this machine over given it was already at
~241–272MB free with multiple other heavy apps and a second Claude session running
concurrently. The worktree/branch/files were unaffected by that crash. **I have not
retried it** — waiting on confirmation that more memory is free before attempting
again, per explicit instruction. When it's safe to retry: the corpus staged at
`clean/rq4/*.txt` is the final, verified 761-document corpus (§2B) — no restaging
needed, just run the pilot.
