# RQ4 Stage C — XLM-R Embedding Extraction on Google Colab

Why Colab: the original Tagalog/Cebuano/Bikolano XLM-R embeddings were fast because
they were produced on Google Colab (free T4 GPU, ~12GB+ RAM) per this project's own
`COLAB_GUIDE.md` — never on the local machine. This machine has 7.4GB RAM and is
frequently down to a few hundred MB free, which is why every local/"remote" attempt at
this exact step (loading `xlm-roberta-base` + tokenizing) has failed or hung (see
`RQ4_PLAN.md` §8). Colab sidesteps this the same way it did the first time — it's
actually a different machine, unlike the earlier "remote agent" attempt which turned
out to be this same one.

This uses this project's **current** architecture — frozen XLM-R embeddings, no
fine-tuning, fed into Weka afterward — not the older fine-tuning approach in the
existing `COLAB_GUIDE.md` (that file is from an earlier, since-abandoned direction;
see the `kayabasa-proposal-vs-implementation-gap` project note). Same infrastructure
pattern (Colab + clone + run), different, current script.

## Step 1 — New notebook, GPU runtime

1. [colab.research.google.com](https://colab.research.google.com) → **New notebook**
2. **Runtime → Change runtime type → T4 GPU → Save**

## Step 2 — Clone the code and the raw corpus

```python
import os

# This repo's RQ4 branch (code only — stage_raw_text.py, extract_embeddings_xlmr.py, etc.)
!git clone --branch feat/rq4-remote-embeddings https://github.com/apcsdrosco2/KayaBasa.git /content/KayaBasa
# If the repo is private and this fails with an auth error, use a personal access token instead:
#   https://YOUR_TOKEN@github.com/apcsdrosco2/KayaBasa.git

# The raw low-resource text (public repo, pinned to the exact commit already verified
# in RQ4_PLAN.md §2A/§2D — 769 raw .txt files, 133/268/173/195 per language)
!git clone https://github.com/imperialite/BasahaCorpus-HierarchicalCrosslingualARA.git /content/basaha_src
os.chdir('/content/basaha_src')
!git checkout bf3b40f8b557252af0dd9e4a2dae0782b0f29bce
os.chdir('/content/KayaBasa')
```

## Step 3 — Install dependencies and confirm GPU

```python
!pip install transformers sentencepiece --quiet
import torch
print("GPU available:", torch.cuda.is_available())
print("Device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
```

## Step 4 — Stage the corpus (must print 769, none dropped)

```python
import os
os.environ["RQ4_RAW_BASE"] = "/content/basaha_src/data/raw"
!python code/generated/rq4/stage_raw_text.py
```

Expect: `Grand total: 769 ... OK — 769 rows staged, one per raw file, none dropped
during cleaning.` If the counts differ at all, stop and report back rather than
continuing — don't proceed on a mismatch.

## Step 5 — Pilot, then full extraction

```python
!python code/generated/rq4/extract_embeddings_xlmr.py --pilot 15
```

Check the printed device line says `cuda` (not `cpu`) and the pilot's shape/NaN/norm
checks pass (the script asserts this itself). Then:

```python
!python code/generated/rq4/extract_embeddings_xlmr.py
```

This writes `code/generated/rq4/embeddings/{lang}_xlmr_features.csv` (4 files, one row
per document, 768 float columns each, no header, row order matching
`clean/rq4/{lang}_all_clean.txt`).

## Step 6 — Bring the results back

```python
from google.colab import files
import shutil
shutil.make_archive('/content/rq4_embeddings', 'zip', 'code/generated/rq4/embeddings')
files.download('/content/rq4_embeddings.zip')
```

This downloads a zip through the browser. Unzip it into
`C:\Users\sophe\Desktop\KayaBasa\code\generated\rq4\embeddings\` on this machine
(4 CSVs), then tell Claude they're there — Stage D (zero-shot test ARFFs) picks up
from exactly that folder.
