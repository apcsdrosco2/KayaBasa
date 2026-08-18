# KAYABASA — Colab Notebook: Features-Only Ablation (Configs A & B)
# MLP Classifier on 14 Linguistic Features
# ============================================================
#
# CORPUS: 764 high-resource documents (matches thesis Table IV)
#   Source:  clean/  folder (pre-cleaned, no re-cleaning needed)
#   Format:  title,level,text  (one doc per line)
#   Languages: Tagalog (265), Cebuano (349), Bikol (150)
#
# ARCHITECTURE (§3.2.1):
#   Config A: Noisy text    → 14 features → MLP → L1/L2/L3
#   Config B: Denoised text → 14 features → MLP → L1/L2/L3
#   MLP: Linear(14,256) → ReLU → Dropout(0.1) → Linear(256,3)
#
# PIPELINE (no data cleaning — clean/ is already the final corpus):
#   1. build_from_clean.py   → reads clean/ → output/all_languages.txt (noisy)
#   2. normalize_datasets.py → structural normalization → output_normalized/ (denoised)
#   3. split_datasets.py     → 5-fold CV splits (764 HR docs)
#   4. extract_features.py   → 14 features × 764 docs
#   5. train_features.py     → Config A + B (MLP, 5-fold CV)
# ============================================================

# ── CELL 1: Environment Setup & Selective Branch Checkout ────────────────────

import os
from google.colab import drive

print("1. Mounting Google Drive...")
if not os.path.exists('/content/drive/MyDrive'):
    drive.mount('/content/drive')
else:
    print("[SUCCESS] Drive already mounted.")

print("\n2. Cloning KayaBasa Repository & Restoring 'main' branch...")
if not os.path.exists('/content/KayaBasa'):
    !git clone https://github.com/apcsdrosco2/KayaBasa.git /content/KayaBasa

os.chdir('/content/KayaBasa')

# Force a hard reset to the main branch so the 'clean' and 'raw' folders are safely restored
!git fetch --all
!git reset --hard origin/main
!git checkout main

print("\n3. Pulling 14-feature pipeline from 'feat/feature-pipeline-refactor'...")
# This command safely overwrites ONLY the feature-pipeline folder 
!git checkout origin/feat/feature-pipeline-refactor -- feature-pipeline

print("\n[SUCCESS] Setup complete! Dataset folders (clean/ and raw/) preserved & 14-feature pipeline loaded.")

# ── CELL 2: Build RAW Corpus (Config A) ──────────────────────────────────────

import os

print("Building RAW corpus directly from /raw folder...")
CORPUS_DIR = '/content/KayaBasa/feature-pipeline/output'
os.makedirs(CORPUS_DIR, exist_ok=True)

NOISY_CORPUS = f"{CORPUS_DIR}/all_languages_raw.txt"
langs = ['tagalog', 'cebuano', 'bikol']
total_raw = 0
patched_raw = 0

with open(NOISY_CORPUS, 'w', encoding='utf-8') as outfile:
    outfile.write("language|level|text\n")
    
    for lang in langs:
        prefix = lang[:3] # Looks for files starting with 'tag', 'ceb', 'bik'
        folder_path = "/content/KayaBasa/raw"
        
        matched_file = None
        if os.path.exists(folder_path):
            for f in os.listdir(folder_path):
                if f.startswith(prefix) and f.endswith(".txt"):
                    matched_file = os.path.join(folder_path, f)
                    break
        
        if matched_file:
            with open(matched_file, 'r', encoding='utf-8') as infile:
                for line in infile:
                    if line.strip():
                        parts = line.strip().split(',', 2)
                        if len(parts) >= 3:
                            text_content = parts[2].strip()
                            
                            # [AUTO-PATCH] Prevent Doc 368 TTR Crash instantly
                            if text_content == "":
                                text_content = "blank"
                                patched_raw += 1
                                
                            outfile.write(f"{lang}|{parts[1].strip()}|{text_content}\n")
                            total_raw += 1
        else:
            print(f"[WARNING] Could not find raw file for {lang} in {folder_path}/")

print(f"[SUCCESS] RAW Corpus built: {total_raw} docs (Patched {patched_raw} empty docs).")

# ── CELL 3: Build CLEAN Corpus (Config B) ────────────────────────────────────

import os

print("Building CLEAN corpus directly from /clean folder...")
CORPUS_DIR = '/content/KayaBasa/feature-pipeline/output'
os.makedirs(CORPUS_DIR, exist_ok=True)

DENOISED_CORPUS = f"{CORPUS_DIR}/all_languages_clean.txt"
langs = ['tagalog', 'cebuano', 'bikol']
total_clean = 0
patched_clean = 0

with open(DENOISED_CORPUS, 'w', encoding='utf-8') as outfile:
    outfile.write("language|level|text\n")
    
    for lang in langs:
        prefix = lang[:3] # Looks for files starting with 'tag', 'ceb', 'bik'
        folder_path = "/content/KayaBasa/clean"
        
        matched_file = None
        if os.path.exists(folder_path):
            for f in os.listdir(folder_path):
                if f.startswith(prefix) and f.endswith(".txt"):
                    matched_file = os.path.join(folder_path, f)
                    break
        
        if matched_file:
            with open(matched_file, 'r', encoding='utf-8') as infile:
                for line in infile:
                    if line.strip():
                        parts = line.strip().split(',', 2)
                        if len(parts) >= 3:
                            text_content = parts[2].strip()
                            
                            # [AUTO-PATCH] Prevent Doc 368 TTR Crash instantly
                            if text_content == "":
                                text_content = "blank"
                                patched_clean += 1
                                
                            outfile.write(f"{lang}|{parts[1].strip()}|{text_content}\n")
                            total_clean += 1
        else:
            print(f"[WARNING] Could not find clean file for {lang} in {folder_path}/")

print(f"[SUCCESS] CLEAN Corpus built: {total_clean} docs (Patched {patched_clean} empty docs).")

if total_clean == 764:
    print("[PERFECT] You have exactly 764 clean documents. Ready for splits!")
else:
    print(f"[WARNING] Expected 764 docs but got {total_clean}.")

# ── CELL 4: Generate Stratified 5-Fold Splits ────────────────────────────────

import pandas as pd
import numpy as np
import pickle
import os
from sklearn.model_selection import StratifiedKFold

print("Generating stratified 5-Fold CV splits directly from clean corpus...")

# 1. Load the clean corpus we just built
CORPUS_DIR = '/content/KayaBasa/feature-pipeline/output'
CLEAN_FILE = f"{CORPUS_DIR}/all_languages_clean.txt"

df = pd.read_csv(CLEAN_FILE, sep='|')
df = df.dropna(subset=['text']).reset_index(drop=True)
df['doc_id'] = df.index  # Create a unique document ID
df['split_role'] = 'high_resource'

# 2. Stratify based on the combination of Language and Level
# This ensures every fold has the exact same proportion of Tagalog Grade 1, Cebuano Grade 2, etc.
df['stratify_key'] = df['language'] + "_" + df['level'].astype(str)

# 3. Generate the splits
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
fold_indices = []

for train_idx, val_idx in skf.split(df, df['stratify_key']):
    fold_indices.append({
        "train": train_idx.tolist(),
        "val": val_idx.tolist()
    })

# 4. Save the high resource index exactly as the model expects
hr_index = df[['doc_id', 'language', 'level', 'split_role']].copy()
hr_index = hr_index.rename(columns={'level': 'label'})

# 5. Create the splits directory and save
split_dir = "/content/KayaBasa/feature-pipeline/splits"
os.makedirs(split_dir, exist_ok=True)

with open(f"{split_dir}/fold_indices.pkl", "wb") as f:
    pickle.dump(fold_indices, f)
    
hr_index.to_csv(f"{split_dir}/high_resource_index.csv", index=False)

print(f"[SUCCESS] Stratified splits generated and saved.")
print(f"Total High-Resource Documents: {len(hr_index)}")

# ── CELL 5: Neural Network Imports & Global Config ───────────────────────────

import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, accuracy_score, classification_report
import os

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"PyTorch Initialized. Using compute device: {DEVICE}")

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

OUT_DIR = '/content/KayaBasa/feature-pipeline/output'
os.makedirs(OUT_DIR, exist_ok=True)
print(f"[SUCCESS] Global Seed ({SEED}) locked. Directories prepared.")

# ── CELL 6: Extract 14 Linguistic Features (Refactored Branch) ───────────────

import os
import sys
import importlib

feature_dir = '/content/KayaBasa/feature-pipeline'
os.chdir(feature_dir)

# Ensure our local refactored folder is the absolute first place Python looks for imports
if feature_dir not in sys.path:
    sys.path.insert(0, feature_dir)

# Force a strict reload to guarantee we are not using cached 'main' branch code
import feature_pipeline
importlib.reload(feature_pipeline)
from feature_pipeline import FeaturePipeline, FEATURE_COLS, load_corpus

# [SAFETY NET] This guarantees we are using the feat/feature-pipeline-refactor logic
assert len(FEATURE_COLS) == 14, f"CRITICAL: Expected 14 features, found {len(FEATURE_COLS)}. You are on the wrong branch!"
print(f"[SUCCESS] Refactored Pipeline Verified: Exactly {len(FEATURE_COLS)} features configured.")

# Using the direct paths generated from Cells 2 and 3
NOISY_CORPUS    = "/content/KayaBasa/feature-pipeline/output/all_languages_raw.txt"
DENOISED_CORPUS = "/content/KayaBasa/feature-pipeline/output/all_languages_clean.txt"

def extract(corpus_path, tag):
    print(f"\nExtracting features [{tag}]...")
    df   = load_corpus(corpus_path)
    X    = FeaturePipeline().fit_transform(df)
    print(f"  {len(X)} docs x {len(FEATURE_COLS)} features")
    return X

X_noisy    = extract(NOISY_CORPUS,    "NOISY   (Config A)")
X_denoised = extract(DENOISED_CORPUS, "DENOISED (Config B)")

X_denoised.to_csv(f"{OUT_DIR}/all_features.csv", index=False)
print(f"\n[SUCCESS] Denoised matrix saved to: {OUT_DIR}/all_features.csv")

# ── CELL 7: MLP Classifier Definition ────────────────────────────────────────

class FeaturesDataset(Dataset):
    def __init__(self, features: np.ndarray, labels: list):
        self.X = torch.tensor(features, dtype=torch.float32)
        self.y = torch.tensor(labels,   dtype=torch.long)
    def __len__(self):  return len(self.y)
    def __getitem__(self, i): return self.X[i], self.y[i]

class FeatureMLP(nn.Module):
    """
    MLP classifier for features-only ablation.
    n_features=14  → Configs A and B
    """
    def __init__(self, n_features=14, hidden=256, dropout=0.1, n_classes=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, n_classes),
        )
    def forward(self, x): return self.net(x)

def train_epoch(model, loader, optimizer, criterion):
    model.train()
    total = 0.0
    for X_b, y_b in loader:
        X_b, y_b = X_b.to(DEVICE), y_b.to(DEVICE)
        optimizer.zero_grad()
        loss = criterion(model(X_b), y_b)
        loss.backward()
        optimizer.step()
        total += loss.item()
    return total / len(loader)

def evaluate(model, loader):
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for X_b, y_b in loader:
            X_b, y_b = X_b.to(DEVICE), y_b.to(DEVICE)
            preds = model(X_b).argmax(dim=1)
            y_true.extend(y_b.tolist())
            y_pred.extend(preds.cpu().tolist())
            
    macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
    acc   = accuracy_score(y_true, y_pred)
    f1_pc = f1_score(y_true, y_pred, average=None, labels=[0,1,2], zero_division=0).tolist()
    return macro, acc, f1_pc, y_true, y_pred

print("[SUCCESS] MLP Architecture and Training Engine Ready.")

# ── CELL 8: 5-Fold CV Training Function Definition ───────────────────────────

def run_cv(X_full, fold_indices, hr_index, config_name, save_prefix,
           lr=1e-3, epochs=200, patience=20, batch_size=64):
    print(f"\n{'='*60}")
    print(f"  Config {config_name} — 5-Fold CV (MLP, lr={lr})")
    print(f"{'='*60}")

    label_map = {1: 0, 2: 1, 3: 2}

    fold_results, lang_records = [], []
    all_true_global, all_pred_global = [], []

    for fold_num, splits in enumerate(fold_indices, 1):
        # We use the index dynamically generated in Cell 4
        train_ids = splits["train"]
        val_ids   = splits["val"]

        scaler   = StandardScaler()
        X_train  = scaler.fit_transform(X_full.loc[train_ids, FEATURE_COLS].values)
        X_val    = scaler.transform(    X_full.loc[val_ids,   FEATURE_COLS].values)
        
        y_train  = [label_map[hr_index.loc[d, "label"]] for d in train_ids]
        y_val    = [label_map[hr_index.loc[d, "label"]] for d in val_ids]

        train_dl = DataLoader(FeaturesDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
        val_dl   = DataLoader(FeaturesDataset(X_val,   y_val), batch_size=batch_size)

        model     = FeatureMLP(n_features=14).to(DEVICE)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()

        best_macro, best_ep, wait = 0.0, 0, 0
        ckpt = f"{OUT_DIR}/{save_prefix}_fold{fold_num}.pt"

        for ep in range(1, epochs + 1):
            train_epoch(model, train_dl, optimizer, criterion)
            macro, acc, f1_pc, _, _ = evaluate(model, val_dl)
            if macro > best_macro:
                best_macro, best_ep, wait = macro, ep, 0
                torch.save(model.state_dict(), ckpt)
            else:
                wait += 1
                if wait >= patience:
                    break

        model.load_state_dict(torch.load(ckpt, weights_only=True))
        macro, acc, f1_pc, y_true, y_pred = evaluate(model, val_dl)

        all_true_global.extend(y_true)
        all_pred_global.extend(y_pred)

        for d_idx, yt, yp in zip(val_ids, y_true, y_pred):
            lang_records.append({
                "doc_id": d_idx, 
                "language": hr_index.loc[d_idx, "language"], 
                "y_true": yt, 
                "y_pred": yp
            })

        fold_results.append({
            "fold": fold_num, "n_train": len(y_train), "n_val": len(y_val),
            "macro_f1": round(macro,4), "accuracy": round(acc,4),
            "f1_L1": round(f1_pc[0],4), "f1_L2": round(f1_pc[1],4),
            "f1_L3": round(f1_pc[2],4), "best_epoch": best_ep,
        })
        print(f"  Fold {fold_num}: Macro-F1={macro:.4f}  Acc={acc:.4f}  [L1={f1_pc[0]:.3f} L2={f1_pc[1]:.3f} L3={f1_pc[2]:.3f}]  (best ep {best_ep})")

    mean_macro = float(np.mean([r["macro_f1"] for r in fold_results]))
    std_macro  = float(np.std([r["macro_f1"] for r in fold_results]))
    mean_acc   = float(np.mean([r["accuracy"] for r in fold_results]))
    std_acc    = float(np.std([r["accuracy"] for r in fold_results]))

    print(f"\n  Mean Macro-F1 : {mean_macro:.4f} +/- {std_macro:.4f}")
    print(f"  Mean Accuracy : {mean_acc:.4f}   +/- {std_acc:.4f}")

    oof = pd.DataFrame(lang_records)
    per_lang = {}
    for lang, grp in oof.groupby("language"):
        mf1 = float(f1_score(grp["y_true"], grp["y_pred"], average="macro", zero_division=0))
        mac = float(accuracy_score(grp["y_true"], grp["y_pred"]))
        per_lang[lang] = {"n_docs": len(grp), "macro_f1": round(mf1,4), "accuracy": round(mac,4)}

    result = {
        "config": config_name, "n_hr_docs": int(hr_index.shape[0]), "folds": fold_results,
        "mean_macro_f1": round(mean_macro, 4), "std_macro_f1": round(std_macro, 4),
        "mean_accuracy": round(mean_acc, 4), "std_accuracy": round(std_acc, 4),
        "per_language": per_lang,
    }
    out_path = f"{OUT_DIR}/results_config_{save_prefix}.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"  Saved: {out_path}")
    return result

print("[SUCCESS] CV Orchestrator Ready.")

# ── CELL 9: Run Config A — Noisy Features + MLP ──────────────────────────────

print("Starting Config A...")
res_a = run_cv(
    X_full       = X_noisy,
    fold_indices = fold_indices,
    hr_index     = hr_index,
    config_name  = "A (Noisy Features, MLP)",
    save_prefix  = "A",
)

# ── CELL 10: Run Config B — Denoised Features + MLP ──────────────────────────

print("\nStarting Config B...")
res_b = run_cv(
    X_full       = X_denoised,
    fold_indices = fold_indices,
    hr_index     = hr_index,
    config_name  = "B (Denoised Features, MLP)",
    save_prefix  = "B",
)

# ── CELL 11: Summary — Config A vs B Comparison ──────────────────────────────

print(f"\n{'='*60}")
print("KAYABASA -- Features-Only Ablation (MLP) Summary")
print(f"{'='*60}")
print(f"{'Config':<35} {'Macro-F1':>10} {'Std':>8} {'Accuracy':>10}")
print("-" * 65)

for r in (res_a, res_b):
    print(f"  {r['config']:<33} {r['mean_macro_f1']:>10.4f} {r['std_macro_f1']:>8.4f} {r['mean_accuracy']:>10.4f}")

delta = res_b["mean_macro_f1"] - res_a["mean_macro_f1"]
print(f"\nDenoising impact (B - A): Delta Macro-F1 = {delta:+.4f}")
status = "SIGNIFICANT (>= 0.03)" if abs(delta) >= 0.03 else "below 0.03 threshold"
print(f"Thesis threshold : {status}")

print(f"\nPer-language Macro-F1:")
print(f"  {'Language':<14} {'Config A':>10} {'Config B':>10} {'Delta':>8}")
print("  " + "-" * 44)
langs = sorted(set(res_a["per_language"]) | set(res_b["per_language"]))
for lang in langs:
    a = res_a["per_language"].get(lang, {}).get("macro_f1", float('nan'))
    b = res_b["per_language"].get(lang, {}).get("macro_f1", float('nan'))
    d = b - a if isinstance(a, float) and isinstance(b, float) else float('nan')
    print(f"  {lang:<14} {a:>10.4f} {b:>10.4f} {d:>+8.4f}")

# ── CELL 12: Save Results to Google Drive ────────────────────────────────────

import os
import shutil

DRIVE_DIR = '/content/drive/MyDrive/KayaBasa/results'
os.makedirs(DRIVE_DIR, exist_ok=True)

print(f"Syncing logs, matrices, and model weights to Google Drive...")
for filename in os.listdir(OUT_DIR):
    src_path = os.path.join(OUT_DIR, filename)
    tgt_path = os.path.join(DRIVE_DIR, filename)
    if os.path.isfile(src_path):
        shutil.copy2(src_path, tgt_path)
        print(f"  [COPIED] {filename}")
        
print(f"\n[SUCCESS] All files permanently backed up to: MyDrive/KayaBasa/results/")

# ── CELL 13: Language x Grade Deep Analysis (Config B) ───────────────────────

print("Analyzing Per-Class Performance WITHIN Each Language (Config B)...\n")

X_hr = X_denoised.copy()
X_hr['doc_id'] = hr_index['doc_id'].values 
X_hr = X_hr.set_index('doc_id')

label_map = {1: 0, 2: 1, 3: 2}
target_names = ["L1 (Grade 1)", "L2 (Grade 2)", "L3 (Grade 3)"]
lang_records = []

for fold_num, splits in enumerate(fold_indices, 1):
    val_ids = hr_index.loc[splits["val"], "doc_id"].values
    train_ids = hr_index.loc[splits["train"], "doc_id"].values
    
    scaler = StandardScaler()
    scaler.fit(X_hr.loc[train_ids, FEATURE_COLS].values)
    X_val = scaler.transform(X_hr.loc[val_ids, FEATURE_COLS].values)
    
    y_val = [label_map[hr_index.loc[hr_index['doc_id'] == d, 'label'].values[0]] for d in val_ids]
    val_dl = DataLoader(FeaturesDataset(X_val, y_val), batch_size=64)

    model = FeatureMLP(n_features=14).to(DEVICE)
    ckpt = f"{OUT_DIR}/B_fold{fold_num}.pt"
    model.load_state_dict(torch.load(ckpt, map_location=DEVICE, weights_only=True))
    model.eval()

    y_pred = []
    with torch.no_grad():
        for X_b, _ in val_dl:
            preds = model(X_b.to(DEVICE)).argmax(dim=1)
            y_pred.extend(preds.cpu().tolist())

    for doc_id, yt, yp in zip(val_ids, y_val, y_pred):
        lang = hr_index.loc[hr_index['doc_id'] == doc_id, 'language'].values[0]
        lang_records.append({"language": lang, "y_true": yt, "y_pred": yp})

oof = pd.DataFrame(lang_records)

for lang, grp in oof.groupby("language"):
    print(f"{'='*50}\n  Language: {lang.upper()} (n={len(grp)})\n{'='*50}")
    print(classification_report(grp["y_true"], grp["y_pred"], target_names=target_names, zero_division=0))

# ── FORCE SYNC TO GOOGLE Drive ───────────────────────────────────────────────

import os
import shutil
from google.colab import drive

# 1. Ensure Drive is mounted
if not os.path.exists('/content/drive/MyDrive'):
    print("Mounting Google Drive...")
    drive.mount('/content/drive')
else:
    print("✅ Google Drive is already mounted.")

# 2. Define source (Colab temporary) and destination (Your actual Drive)
SOURCE_DIR = '/content/KayaBasa/feature-pipeline/output'
TARGET_DIR = '/content/drive/MyDrive/KayaBasa/results'

# 3. Check if local files actually exist
if not os.path.exists(SOURCE_DIR):
    print(f"❌ Error: Cannot find local output folder: {SOURCE_DIR}")
else:
    files = os.listdir(SOURCE_DIR)
    print(f"✅ Found local output folder with {len(files)} files.")
    
    # Create target in Drive if it doesn't exist
    os.makedirs(TARGET_DIR, exist_ok=True)
    
    # 4. Copy everything over
    print(f"⏳ Copying files to Google Drive...")
    for filename in files:
        src_path = os.path.join(SOURCE_DIR, filename)
        tgt_path = os.path.join(TARGET_DIR, filename)
        
        # copy2 preserves file metadata like timestamps
        if os.path.isfile(src_path):
            shutil.copy2(src_path, tgt_path)
            print(f"  ➜ Copied: {filename}")
            
    print(f"\n🎉 Success! All files have been pushed to: MyDrive/KayaBasa/results/")
    print("Note: If you have Google Drive open in another tab, press F5 to refresh the page to see them!")
