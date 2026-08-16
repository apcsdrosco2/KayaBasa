# KAYABASA — Google Colab Training Guide

Step-by-step instructions to train the full KAYABASA hybrid model on Google Colab
(free T4 GPU). This covers the full ablation study (Configs C–F) plus cross-lingual
evaluation, picking up after the features-only baseline (Config B) which was already
run locally.

---

## Before You Start

Make sure you have completed locally:
- `data-cleaning-repo-main/` pipeline run (all_languages.txt exists)
- `feature-pipeline/output/all_features.csv` exists (14-feature matrix)
- `data-cleaning-repo-main/splits/fold_indices.pkl` exists

You will upload these files to Google Drive or re-generate them in Colab.

---

## Step 1 — Open Google Colab

1. Go to [https://colab.research.google.com](https://colab.research.google.com)
2. Click **New notebook**
3. In the top menu: **Runtime → Change runtime type → T4 GPU → Save**

---

## Step 2 — Mount Google Drive

```python
from google.colab import drive
drive.mount('/content/drive')
```

Create a folder in your Drive: `My Drive/KayaBasa/`  
Upload your repo files there, or continue to Step 3 to clone from GitHub.

---

## Step 3 — Clone the Repo and Datasets

```python
import os

# Clone your KayaBasa thesis repo
!git clone https://github.com/YOUR_USERNAME/KayaBasa.git /content/KayaBasa

# Clone the raw datasets
!git clone https://github.com/imperialite/ara-close-lang.git \
    /content/KayaBasa/raw/ara-close-lang

!git clone https://github.com/imperialite/BasahaCorpus-HierarchicalCrosslingualARA.git \
    /content/KayaBasa/raw/BasahaCorpus-HierarchicalCrosslingualARA

os.chdir('/content/KayaBasa')
print("Working directory:", os.getcwd())
```

> Replace `YOUR_USERNAME` with your GitHub username. If the repo is private,
> use a personal access token:
> `https://YOUR_TOKEN@github.com/YOUR_USERNAME/KayaBasa.git`

---

## Step 4 — Install Dependencies

```python
!pip install transformers datasets scikit-learn pandas numpy torch --quiet
```

Verify GPU is available:

```python
import torch
print("GPU available:", torch.cuda.is_available())
print("Device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
```

---

## Step 5 — Run the Data Cleaning Pipeline (if not already done)

Skip this step if you already have `output_normalized/all_languages.txt`.

```python
os.chdir('/content/KayaBasa/data-cleaning-repo-main')

!python build_datasets.py
!python normalize_datasets.py
!python split_datasets.py

print("Done! Corpus ready at output_normalized/all_languages.txt")
```

---

## Step 6 — Extract the 14 Linguistic Features

```python
os.chdir('/content/KayaBasa/feature-pipeline')

!python extract_features.py
# Output: output/all_features.csv  (1480 docs × 14 features)
```

---

## Step 7 — Config B: Features-Only Baseline (Random Forest)

This was already run locally (Macro-F1: 0.6487), but run it here too for
Colab reproducibility:

```python
!python train_features.py
# Output: output/results_config_B.json
```

---

## Step 8 — Config D: Transformer-Only (XLM-RoBERTa, Denoised)

This is the first GPU-intensive step. Create a new script in Colab:

```python
# ============================================================
# Config D: XLM-RoBERTa only, denoised text, 5-fold CV
# ============================================================

import pickle, os, json
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics import f1_score, accuracy_score

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_NAME = "xlm-roberta-base"
MAX_LEN = 512
BATCH_SIZE = 16
EPOCHS = 10
LR = 2e-5
PATIENCE = 3   # early stopping

CORPUS_PATH = "/content/KayaBasa/data-cleaning-repo-main/output_normalized/all_languages.txt"
FOLD_PKL    = "/content/KayaBasa/data-cleaning-repo-main/splits/fold_indices.pkl"
HR_IDX_CSV  = "/content/KayaBasa/data-cleaning-repo-main/splits/high_resource_index.csv"
OUT_DIR     = "/content/KayaBasa/feature-pipeline/output"
os.makedirs(OUT_DIR, exist_ok=True)

# --- Load corpus ---
import sys
sys.path.insert(0, '/content/KayaBasa/feature-pipeline')
from feature_pipeline import load_corpus

df = load_corpus(CORPUS_PATH)
df_hr = df[df["split_role"] == "high_resource"].reset_index(drop=True)
print(f"High-resource docs: {len(df_hr)}")

# --- Dataset class ---
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

class ReadabilityDataset(Dataset):
    def __init__(self, texts, labels, max_len=MAX_LEN):
        self.texts  = texts
        self.labels = labels
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = tokenizer(
            self.texts[idx],
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids":      enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "label":          torch.tensor(self.labels[idx], dtype=torch.long),
        }

# --- Model: XLM-R + linear head ---
class XLMRClassifier(nn.Module):
    def __init__(self, dropout=0.1, num_classes=3):
        super().__init__()
        self.xlmr    = AutoModel.from_pretrained(MODEL_NAME)
        self.dropout = nn.Dropout(dropout)
        self.head    = nn.Linear(768, num_classes)

    def forward(self, input_ids, attention_mask):
        out = self.xlmr(input_ids=input_ids, attention_mask=attention_mask)
        cls = out.last_hidden_state[:, 0, :]   # [CLS] token
        return self.head(self.dropout(cls))

# --- Training helpers ---
def train_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0
    for batch in loader:
        ids  = batch["input_ids"].to(DEVICE)
        mask = batch["attention_mask"].to(DEVICE)
        y    = batch["label"].to(DEVICE)
        optimizer.zero_grad()
        logits = model(ids, mask)
        loss   = criterion(logits, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

def eval_epoch(model, loader):
    model.eval()
    all_true, all_pred = [], []
    with torch.no_grad():
        for batch in loader:
            ids  = batch["input_ids"].to(DEVICE)
            mask = batch["attention_mask"].to(DEVICE)
            y    = batch["label"].to(DEVICE)
            logits = model(ids, mask)
            preds  = logits.argmax(dim=1)
            all_true.extend(y.cpu().tolist())
            all_pred.extend(preds.cpu().tolist())
    macro = f1_score(all_true, all_pred, average="macro", zero_division=0)
    acc   = accuracy_score(all_true, all_pred)
    return macro, acc, all_true, all_pred

# --- 5-Fold CV ---
with open(FOLD_PKL, "rb") as f:
    fold_indices = pickle.load(f)
hr_index = pd.read_csv(HR_IDX_CSV)

# Labels: 1→0, 2→1, 3→2
df_hr["label_enc"] = df_hr["label"] - 1

fold_results = []
for fold_num, splits in enumerate(fold_indices, 1):
    print(f"\n=== Fold {fold_num}/5 ===")

    train_doc_ids = hr_index.loc[splits["train"], "doc_id"].values
    val_doc_ids   = hr_index.loc[splits["val"],   "doc_id"].values

    # Build index: doc_id → row in df_hr
    id2row = {row["doc_id"]: i for i, row in df_hr.iterrows()}

    train_texts  = [df_hr.iloc[id2row[d]]["text"]      for d in train_doc_ids]
    train_labels = [df_hr.iloc[id2row[d]]["label_enc"] for d in train_doc_ids]
    val_texts    = [df_hr.iloc[id2row[d]]["text"]      for d in val_doc_ids]
    val_labels   = [df_hr.iloc[id2row[d]]["label_enc"] for d in val_doc_ids]

    train_ds = ReadabilityDataset(train_texts, train_labels)
    val_ds   = ReadabilityDataset(val_texts,   val_labels)
    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_dl   = DataLoader(val_ds,   batch_size=BATCH_SIZE)

    model     = XLMRClassifier().to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    best_macro, best_epoch, patience_count = 0, 0, 0
    for epoch in range(1, EPOCHS + 1):
        train_loss = train_epoch(model, train_dl, optimizer, criterion)
        val_macro, val_acc, _, _ = eval_epoch(model, val_dl)
        print(f"  Epoch {epoch}: loss={train_loss:.4f}  val_macro={val_macro:.4f}  acc={val_acc:.4f}")

        if val_macro > best_macro:
            best_macro, best_epoch = val_macro, epoch
            patience_count = 0
            torch.save(model.state_dict(), f"{OUT_DIR}/best_fold{fold_num}_D.pt")
        else:
            patience_count += 1
            if patience_count >= PATIENCE:
                print(f"  Early stopping at epoch {epoch} (best={best_macro:.4f} @ epoch {best_epoch})")
                break

    # Load best weights and get final predictions
    model.load_state_dict(torch.load(f"{OUT_DIR}/best_fold{fold_num}_D.pt"))
    val_macro, val_acc, y_true, y_pred = eval_epoch(model, val_dl)
    f1_per = f1_score(y_true, y_pred, average=None, labels=[0,1,2], zero_division=0).tolist()

    fold_results.append({
        "fold":      fold_num,
        "macro_f1":  round(val_macro, 4),
        "accuracy":  round(val_acc,   4),
        "f1_L1":     round(f1_per[0], 4),
        "f1_L2":     round(f1_per[1], 4),
        "f1_L3":     round(f1_per[2], 4),
    })
    print(f"  Fold {fold_num} Final: Macro-F1={val_macro:.4f}  Acc={val_acc:.4f}"
          f"  [L1={f1_per[0]:.3f} L2={f1_per[1]:.3f} L3={f1_per[2]:.3f}]")

# Summary
macro_vals = [r["macro_f1"] for r in fold_results]
print(f"\nConfig D Summary:")
print(f"  Mean Macro-F1 : {np.mean(macro_vals):.4f} +/- {np.std(macro_vals):.4f}")

results_D = {
    "config": "D (Denoised Transformer)",
    "folds": fold_results,
    "mean_macro_f1": round(float(np.mean(macro_vals)), 4),
    "std_macro_f1":  round(float(np.std(macro_vals)),  4),
}
with open(f"{OUT_DIR}/results_config_D.json", "w") as f:
    json.dump(results_D, f, indent=2)
print(f"Saved: {OUT_DIR}/results_config_D.json")
```

---

## Step 9 — Config F: Full KAYABASA Hybrid Model (XLM-R + Features + MLP)

After Config D completes, run the hybrid. The key change: concatenate the 768-dim
`[CLS]` vector with the 14-dim feature vector before the MLP head.

```python
# ============================================================
# Config F: Full KAYABASA — XLM-R + 14 features + MLP
# ============================================================

from sklearn.preprocessing import StandardScaler

FEATURE_COLS = [
    "mean_sentence_len", "mean_word_len", "polysyll_freq", "type_token_ratio",
    "syll_cv", "syll_cvc", "syll_ccvc", "syll_ccvccc",
    "tag_bi", "bik_bi", "ceb_bi", "tag_tri", "bik_tri", "ceb_tri",
]

# Load feature matrix (already extracted in Step 6)
feat_df = pd.read_csv(f"{OUT_DIR}/all_features.csv")
feat_df = feat_df[feat_df["split_role"] == "high_resource"].set_index("doc_id")

# --- Hybrid model: XLM-R + features concatenated ---
class KAYABASAHybrid(nn.Module):
    def __init__(self, n_features=14, hidden=256, dropout=0.1, num_classes=3):
        super().__init__()
        self.xlmr    = AutoModel.from_pretrained(MODEL_NAME)
        self.dropout = nn.Dropout(dropout)
        # MLP head receives 768 (transformer) + 14 (features) = 782
        self.mlp = nn.Sequential(
            nn.Linear(768 + n_features, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, input_ids, attention_mask, features):
        out = self.xlmr(input_ids=input_ids, attention_mask=attention_mask)
        cls = out.last_hidden_state[:, 0, :]          # 768-dim
        x   = torch.cat([cls, features], dim=1)       # 782-dim
        return self.mlp(self.dropout(x))

# --- Dataset with features ---
class HybridDataset(Dataset):
    def __init__(self, texts, features, labels, max_len=MAX_LEN):
        self.texts    = texts
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels   = labels
        self.max_len  = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = tokenizer(
            self.texts[idx],
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids":      enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "features":       self.features[idx],
            "label":          torch.tensor(self.labels[idx], dtype=torch.long),
        }

def train_epoch_hybrid(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0
    for batch in loader:
        ids  = batch["input_ids"].to(DEVICE)
        mask = batch["attention_mask"].to(DEVICE)
        feat = batch["features"].to(DEVICE)
        y    = batch["label"].to(DEVICE)
        optimizer.zero_grad()
        logits = model(ids, mask, feat)
        loss   = criterion(logits, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

def eval_epoch_hybrid(model, loader):
    model.eval()
    all_true, all_pred = [], []
    with torch.no_grad():
        for batch in loader:
            ids  = batch["input_ids"].to(DEVICE)
            mask = batch["attention_mask"].to(DEVICE)
            feat = batch["features"].to(DEVICE)
            y    = batch["label"].to(DEVICE)
            logits = model(ids, mask, feat)
            preds  = logits.argmax(dim=1)
            all_true.extend(y.cpu().tolist())
            all_pred.extend(preds.cpu().tolist())
    macro = f1_score(all_true, all_pred, average="macro", zero_division=0)
    acc   = accuracy_score(all_true, all_pred)
    return macro, acc, all_true, all_pred

# --- 5-Fold CV for Config F ---
fold_results_F = []
for fold_num, splits in enumerate(fold_indices, 1):
    print(f"\n=== Fold {fold_num}/5 (Config F - Hybrid) ===")

    train_doc_ids = hr_index.loc[splits["train"], "doc_id"].values
    val_doc_ids   = hr_index.loc[splits["val"],   "doc_id"].values

    id2row = {row["doc_id"]: i for i, row in df_hr.iterrows()}

    train_texts  = [df_hr.iloc[id2row[d]]["text"]      for d in train_doc_ids]
    train_labels = [df_hr.iloc[id2row[d]]["label_enc"] for d in train_doc_ids]
    val_texts    = [df_hr.iloc[id2row[d]]["text"]      for d in val_doc_ids]
    val_labels   = [df_hr.iloc[id2row[d]]["label_enc"] for d in val_doc_ids]

    # Scale features (fit on train, apply to val)
    scaler = StandardScaler()
    X_train_feat = scaler.fit_transform(feat_df.loc[train_doc_ids, FEATURE_COLS].values)
    X_val_feat   = scaler.transform(feat_df.loc[val_doc_ids,   FEATURE_COLS].values)

    train_ds = HybridDataset(train_texts, X_train_feat, train_labels)
    val_ds   = HybridDataset(val_texts,   X_val_feat,   val_labels)
    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_dl   = DataLoader(val_ds,   batch_size=BATCH_SIZE)

    model     = KAYABASAHybrid().to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    best_macro, best_epoch, patience_count = 0, 0, 0
    for epoch in range(1, EPOCHS + 1):
        loss = train_epoch_hybrid(model, train_dl, optimizer, criterion)
        val_macro, val_acc, _, _ = eval_epoch_hybrid(model, val_dl)
        print(f"  Epoch {epoch}: loss={loss:.4f}  val_macro={val_macro:.4f}")

        if val_macro > best_macro:
            best_macro, best_epoch = val_macro, epoch
            patience_count = 0
            torch.save(model.state_dict(), f"{OUT_DIR}/best_fold{fold_num}_F.pt")
        else:
            patience_count += 1
            if patience_count >= PATIENCE:
                print(f"  Early stop (best={best_macro:.4f} @ ep {best_epoch})")
                break

    model.load_state_dict(torch.load(f"{OUT_DIR}/best_fold{fold_num}_F.pt"))
    val_macro, val_acc, y_true, y_pred = eval_epoch_hybrid(model, val_dl)
    f1_per = f1_score(y_true, y_pred, average=None, labels=[0,1,2], zero_division=0).tolist()
    fold_results_F.append({
        "fold": fold_num, "macro_f1": round(val_macro,4),
        "accuracy": round(val_acc,4),
        "f1_L1": round(f1_per[0],4), "f1_L2": round(f1_per[1],4), "f1_L3": round(f1_per[2],4),
    })
    print(f"  Fold {fold_num}: Macro-F1={val_macro:.4f}  Acc={val_acc:.4f}"
          f"  [L1={f1_per[0]:.3f} L2={f1_per[1]:.3f} L3={f1_per[2]:.3f}]")

macro_vals_F = [r["macro_f1"] for r in fold_results_F]
print(f"\nConfig F (Full KAYABASA):")
print(f"  Mean Macro-F1 : {np.mean(macro_vals_F):.4f} +/- {np.std(macro_vals_F):.4f}")

results_F = {
    "config": "F (Full KAYABASA Hybrid)",
    "folds":  fold_results_F,
    "mean_macro_f1": round(float(np.mean(macro_vals_F)), 4),
    "std_macro_f1":  round(float(np.std(macro_vals_F)),  4),
}
with open(f"{OUT_DIR}/results_config_F.json", "w") as f:
    json.dump(results_F, f, indent=2)
print("Saved results_config_F.json")
```

---

## Step 10 — Cross-Lingual Evaluation (BasahaCorpus)

After Config F training, test on the held-out low-resource languages with no retraining:

```python
# Load best model from a chosen fold (or train on full high-resource set)
# Then predict on BasahaCorpus:

lr_df = df[df["split_role"] == "low_resource"].reset_index(drop=True)
lr_df["label_enc"] = lr_df["label"] - 1
lr_feat = pd.read_csv(f"{OUT_DIR}/all_features.csv")
lr_feat = lr_feat[lr_feat["split_role"] == "low_resource"].set_index("doc_id")

# Evaluate per language
for lang in ["hiligaynon", "minasbate", "karay-a", "rinconada"]:
    subset   = lr_df[lr_df["language"] == lang]
    doc_ids  = subset["doc_id"].values
    texts    = subset["text"].tolist()
    labels   = subset["label_enc"].tolist()
    features = scaler.transform(lr_feat.loc[doc_ids, FEATURE_COLS].values)

    ds = HybridDataset(texts, features, labels)
    dl = DataLoader(ds, batch_size=BATCH_SIZE)

    macro, acc, _, _ = eval_epoch_hybrid(model, dl)
    print(f"  {lang:<14}  Macro-F1={macro:.4f}  Acc={acc:.4f}")
```

---

## Step 11 — Save Results to Drive

```python
# Copy all output files to Google Drive for persistence
!cp -r /content/KayaBasa/feature-pipeline/output/ \
        /content/drive/MyDrive/KayaBasa/results/

print("Results saved to Google Drive.")
```

---

## Tips

- **Session timeout**: Colab free tier disconnects after ~1.5 hours of idle. Run Configs D and F in separate sessions; use Drive to persist model checkpoints (`.pt` files).
- **GPU memory**: If you get CUDA OOM errors, reduce `BATCH_SIZE` from 16 to 8.
- **Long documents**: Some texts exceed 512 tokens. The current implementation truncates at 512. A sliding-window approach (chunk + soft-vote) is described in §3.2.3 and can be added as an extension.
- **Save frequently**: Call `torch.save(model.state_dict(), ...)` after each fold.
