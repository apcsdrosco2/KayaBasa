# KAYABASA — Colab Notebook: Features-Only Ablation (Configs A & B)
# MLP Classifier on 14 Linguistic Features
# ============================================================
#
# IMPORTANT NOTE ON DOCUMENT COUNT
# ---------------------------------
# The thesis paper (Table IV) cites 764 high-resource documents.
# After the denoising and filtering pipeline (§3.1.3), 30 documents
# were removed (10 duplicates, 10 empty, 8 too short, 2 digit-interleave).
# This leaves 734 high-resource docs used in training — consistent with
# the paper's §3.5.2 discussion of filtering trade-offs.
#
# ARCHITECTURE NOTE
# -----------------
# Configs A and B use an MLP classifier (not Random Forest).
# The MLP is the same classifier head used in the full KAYABASA model (Config F),
# but receives only the 14-dim feature vector instead of 782-dim hybrid input.
# This ensures the ablation is a fair head-to-head comparison of
# what each INPUT TYPE contributes, not what each CLASSIFIER contributes.
#
# Config A: Noisy text   → 14 features → MLP → L1/L2/L3
# Config B: Denoised text → 14 features → MLP → L1/L2/L3
# ============================================================


# ── CELL 1: Setup ────────────────────────────────────────────────────────────

import os
import sys
import json
import pickle
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score, accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from google.colab import drive

# Mount Google Drive for checkpoint persistence
drive.mount('/content/drive')
os.makedirs('/content/drive/MyDrive/KayaBasa/results', exist_ok=True)

print("PyTorch version:", torch.__version__)
print("GPU available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("Device:", torch.cuda.get_device_name(0))

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── CELL 2: Clone repositories ───────────────────────────────────────────────

print("Cloning KayaBasa repository...")
# Switch to the feature-pipeline branch which has the 14-feature pipeline
os.system("git clone https://github.com/YOUR_USERNAME/KayaBasa.git /content/KayaBasa")
os.chdir('/content/KayaBasa')
os.system("git checkout feat/feature-pipeline-refactor")

print("Cloning raw datasets...")
os.system("git clone https://github.com/imperialite/ara-close-lang.git /content/KayaBasa/raw/ara-close-lang")
os.system("git clone https://github.com/imperialite/BasahaCorpus-HierarchicalCrosslingualARA.git /content/KayaBasa/raw/BasahaCorpus-HierarchicalCrosslingualARA")

# ── CELL 3: Install dependencies ─────────────────────────────────────────────

os.system("pip install ftfy regex scikit-learn pandas numpy --quiet")
print("Dependencies installed.")

# ── CELL 4: Generate both corpora ────────────────────────────────────────────
# Produces:
#   output/all_languages_raw.txt        ← NOISY  (Config A)
#   output_normalized/all_languages.txt ← DENOISED (Config B)
#   splits/fold_indices.pkl             ← 5-fold CV splits
#   splits/high_resource_index.csv      ← doc_id → language, label mapping

os.chdir('/content/KayaBasa/data-cleaning-repo-main')

os.system("python build_datasets.py")       # raw, noisy corpus
os.system("python normalize_datasets.py")   # denoised + normalized corpus
os.system("python split_datasets.py")       # stratified 5-fold CV splits

# Verify outputs
assert os.path.exists("output/all_languages_raw.txt"),         "Raw corpus missing!"
assert os.path.exists("output_normalized/all_languages.txt"),  "Normalized corpus missing!"
assert os.path.exists("splits/fold_indices.pkl"),              "Fold splits missing!"
assert os.path.exists("splits/high_resource_index.csv"),       "HR index missing!"
print("All corpus files generated successfully.")

# ── CELL 5: Extract features from both corpora ───────────────────────────────

os.chdir('/content/KayaBasa/feature-pipeline')
sys.path.insert(0, '/content/KayaBasa/feature-pipeline')
from feature_pipeline import FeaturePipeline, FEATURE_COLS, load_corpus

NOISY_CORPUS    = "/content/KayaBasa/data-cleaning-repo-main/output/all_languages_raw.txt"
DENOISED_CORPUS = "/content/KayaBasa/data-cleaning-repo-main/output_normalized/all_languages.txt"
OUT_DIR         = "/content/KayaBasa/feature-pipeline/output"
os.makedirs(OUT_DIR, exist_ok=True)

def extract_features(corpus_path, label):
    print(f"\nExtracting features [{label}] from:\n  {corpus_path}")
    df   = load_corpus(corpus_path)
    pipe = FeaturePipeline()
    X    = pipe.fit_transform(df)
    print(f"  Done: {len(X)} docs x {len(FEATURE_COLS)} features")
    return X

X_noisy    = extract_features(NOISY_CORPUS,    "NOISY")
X_denoised = extract_features(DENOISED_CORPUS, "DENOISED")

# Save denoised feature matrix (used by Config F later)
X_denoised.to_csv(f"{OUT_DIR}/all_features.csv", index=False)
print(f"\nSaved: {OUT_DIR}/all_features.csv")

# ── CELL 6: Verify document count ────────────────────────────────────────────

hr_index   = pd.read_csv("/content/KayaBasa/data-cleaning-repo-main/splits/high_resource_index.csv")
n_hr_docs  = len(hr_index)

print(f"\nHigh-resource training documents: {n_hr_docs}")
print("(Note: thesis paper cites 764; 30 docs removed during denoising/filtering)")
print("  10 duplicates + 10 empty + 8 too-short + 2 digit-interleave = 30 dropped")
print(f"\nPer-language breakdown:")
print(hr_index.groupby(['language','label']).size().to_string())

# ── CELL 7: Load fold splits ──────────────────────────────────────────────────

FOLD_PKL = "/content/KayaBasa/data-cleaning-repo-main/splits/fold_indices.pkl"
with open(FOLD_PKL, "rb") as f:
    fold_indices = pickle.load(f)

print(f"\n5-Fold CV splits loaded:")
for i, s in enumerate(fold_indices, 1):
    print(f"  Fold {i}: train={len(s['train'])} docs, val={len(s['val'])} docs")

# ── CELL 8: MLP Classifier Definition ────────────────────────────────────────
#
# The MLP classifier is the same head used in the full KAYABASA hybrid model (Config F).
# For Configs A & B (features-only), the input is the 14-dim feature vector directly.
# For Config F (hybrid), the input will be [768-dim XLM-R CLS] + [14-dim features] = 782-dim.
#
# Architecture (§3.2.1):
#   Input → Linear(14, 256) → ReLU → Dropout(0.1) → Linear(256, 3) → Softmax

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

class FeaturesDataset(Dataset):
    """Dataset for features-only (Configs A & B)."""
    def __init__(self, features: np.ndarray, labels: list):
        self.X = torch.tensor(features, dtype=torch.float32)
        self.y = torch.tensor(labels,   dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class FeatureMLP(nn.Module):
    """
    MLP classifier for features-only ablation.
    Matches the MLP head of the full KAYABASA hybrid model (§3.2.1).
    n_features=14  for Configs A and B
    n_features=782 for Config F (XLM-R [CLS] + 14 features)
    """
    def __init__(self, n_features: int = 14, hidden: int = 256,
                 dropout: float = 0.1, n_classes: int = 3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, n_classes),
        )

    def forward(self, x):
        return self.net(x)


def train_one_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0.0
    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
        optimizer.zero_grad()
        loss = criterion(model(X_batch), y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


def evaluate(model, loader):
    model.eval()
    all_true, all_pred = [], []
    with torch.no_grad():
        for X_batch, y_batch in loader:
            preds = model(X_batch.to(DEVICE)).argmax(dim=1)
            all_true.extend(y_batch.tolist())
            all_pred.extend(preds.cpu().tolist())
    macro = f1_score(all_true, all_pred, average="macro", zero_division=0)
    acc   = accuracy_score(all_true, all_pred)
    f1_pc = f1_score(all_true, all_pred, average=None, labels=[0,1,2],
                     zero_division=0).tolist()
    return macro, acc, f1_pc, all_true, all_pred


# ── CELL 9: 5-Fold CV function ────────────────────────────────────────────────

def run_cv(X_full: pd.DataFrame, fold_indices: list, hr_index: pd.DataFrame,
           config_name: str, save_prefix: str,
           lr: float = 1e-3, epochs: int = 200, patience: int = 20,
           batch_size: int = 64) -> dict:
    """
    Stratified 5-Fold CV with MLP on the 14 linguistic features.
    Scaler is fit on train set each fold to prevent data leakage.
    """
    print(f"\n{'='*60}")
    print(f"  Config {config_name} — 5-Fold CV (MLP)")
    print(f"{'='*60}")

    # Filter to high-resource only and index by doc_id
    X_hr = X_full[X_full["split_role"] == "high_resource"].set_index("doc_id")

    # Labels: L1/L2/L3 → 0/1/2
    label_map = {1: 0, 2: 1, 3: 2}

    fold_results = []
    all_y_true_global, all_y_pred_global = [], []
    lang_records = []

    for fold_num, splits in enumerate(fold_indices, 1):
        train_ids = hr_index.loc[splits["train"], "doc_id"].values
        val_ids   = hr_index.loc[splits["val"],   "doc_id"].values

        # Feature matrices
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_hr.loc[train_ids, FEATURE_COLS].values)
        X_val   = scaler.transform(X_hr.loc[val_ids,   FEATURE_COLS].values)

        y_train = [label_map[X_hr.loc[d, "label"]] for d in train_ids]
        y_val   = [label_map[X_hr.loc[d, "label"]] for d in val_ids]

        train_dl = DataLoader(FeaturesDataset(X_train, y_train),
                              batch_size=batch_size, shuffle=True)
        val_dl   = DataLoader(FeaturesDataset(X_val, y_val),
                              batch_size=batch_size)

        model     = FeatureMLP(n_features=14).to(DEVICE)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()

        best_macro, best_ep, wait = 0.0, 0, 0
        ckpt_path = f"{OUT_DIR}/{save_prefix}_fold{fold_num}.pt"

        for ep in range(1, epochs + 1):
            train_one_epoch(model, train_dl, optimizer, criterion)
            macro, acc, f1_pc, _, _ = evaluate(model, val_dl)
            if macro > best_macro:
                best_macro, best_ep = macro, ep
                wait = 0
                torch.save(model.state_dict(), ckpt_path)
            else:
                wait += 1
                if wait >= patience:
                    break

        # Load best checkpoint
        model.load_state_dict(torch.load(ckpt_path))
        macro, acc, f1_pc, y_true, y_pred = evaluate(model, val_dl)

        all_y_true_global.extend(y_true)
        all_y_pred_global.extend(y_pred)

        # Per-language records for this fold
        for doc_id, yt, yp in zip(val_ids, y_true, y_pred):
            lang_records.append({
                "doc_id": doc_id,
                "language": X_hr.loc[doc_id, "language"],
                "y_true": yt,
                "y_pred": yp,
            })

        fold_results.append({
            "fold":     fold_num,
            "n_train":  len(y_train),
            "n_val":    len(y_val),
            "macro_f1": round(macro, 4),
            "accuracy": round(acc,   4),
            "f1_L1":    round(f1_pc[0], 4),
            "f1_L2":    round(f1_pc[1], 4),
            "f1_L3":    round(f1_pc[2], 4),
            "best_epoch": best_ep,
        })
        print(f"  Fold {fold_num}: Macro-F1={macro:.4f}  Acc={acc:.4f}"
              f"  [L1={f1_pc[0]:.3f} L2={f1_pc[1]:.3f} L3={f1_pc[2]:.3f}]"
              f"  (best ep={best_ep})")

    # Aggregate
    macro_vals = [r["macro_f1"] for r in fold_results]
    acc_vals   = [r["accuracy"] for r in fold_results]
    mean_macro = float(np.mean(macro_vals))
    std_macro  = float(np.std(macro_vals))
    mean_acc   = float(np.mean(acc_vals))
    std_acc    = float(np.std(acc_vals))

    print(f"\n  Mean Macro-F1 : {mean_macro:.4f} +/- {std_macro:.4f}")
    print(f"  Mean Accuracy : {mean_acc:.4f}   +/- {std_acc:.4f}")

    # Per-language breakdown
    oof_df = pd.DataFrame(lang_records)
    per_lang = {}
    for lang, grp in oof_df.groupby("language"):
        per_lang[lang] = {
            "n_docs":   len(grp),
            "macro_f1": round(float(f1_score(grp["y_true"], grp["y_pred"],
                                             average="macro", zero_division=0)), 4),
            "accuracy": round(float(accuracy_score(grp["y_true"], grp["y_pred"])), 4),
        }

    print("\n  Per-language Macro-F1:")
    for lang, m in per_lang.items():
        print(f"    {lang:<14}  F1={m['macro_f1']:.4f}  Acc={m['accuracy']:.4f}")

    # Full classification report
    print("\n  Classification Report (all folds combined):")
    print(classification_report(
        all_y_true_global, all_y_pred_global,
        target_names=["L1 (Grade 1)", "L2 (Grade 2)", "L3 (Grade 3)"],
        zero_division=0
    ))

    result = {
        "config":        config_name,
        "n_hr_docs":     len(hr_index),
        "folds":         fold_results,
        "mean_macro_f1": round(mean_macro, 4),
        "std_macro_f1":  round(std_macro,  4),
        "mean_accuracy": round(mean_acc,   4),
        "std_accuracy":  round(std_acc,    4),
        "per_language":  per_lang,
    }

    out_path = f"{OUT_DIR}/results_config_{save_prefix}.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\n  Saved: {out_path}")
    return result


# ── CELL 10: Run Config B — Denoised Features + MLP ─────────────────────────

res_b = run_cv(
    X_full       = X_denoised,
    fold_indices = fold_indices,
    hr_index     = hr_index,
    config_name  = "B (Denoised Features, MLP)",
    save_prefix  = "B",
)

# ── CELL 11: Run Config A — Noisy Features + MLP ────────────────────────────

res_a = run_cv(
    X_full       = X_noisy,
    fold_indices = fold_indices,
    hr_index     = hr_index,
    config_name  = "A (Noisy Features, MLP)",
    save_prefix  = "A",
)

# ── CELL 12: Summary — A vs B Comparison ─────────────────────────────────────

print("\n" + "="*60)
print("KAYABASA -- Features-Only Ablation Summary (MLP)")
print("="*60)
print(f"{'Config':<30} {'Macro-F1':>10} {'Std':>8} {'Accuracy':>10}")
print("-"*60)
for r in (res_a, res_b):
    print(f"{r['config']:<30} {r['mean_macro_f1']:>10.4f}"
          f" {r['std_macro_f1']:>8.4f} {r['mean_accuracy']:>10.4f}")

delta = res_b["mean_macro_f1"] - res_a["mean_macro_f1"]
print(f"\nDenoising impact (B - A): Delta Macro-F1 = {delta:+.4f}")
threshold = "CONFIRMED (>= 0.03)" if delta >= 0.03 else "below 0.03 threshold"
print(f"Thesis denoising threshold (Section 3.1.5): {threshold}")

print("\nPer-language Macro-F1:")
print(f"{'Language':<14} {'Config A':>10} {'Config B':>10} {'Delta':>8}")
print("-"*44)
langs = sorted(set(res_a["per_language"]) | set(res_b["per_language"]))
for lang in langs:
    a = res_a["per_language"].get(lang, {}).get("macro_f1", "-")
    b = res_b["per_language"].get(lang, {}).get("macro_f1", "-")
    d = f"{b-a:+.4f}" if isinstance(a, float) and isinstance(b, float) else "-"
    print(f"{lang:<14} {a:>10.4f} {b:>10.4f} {d:>8}")

# ── CELL 13: Save to Google Drive ─────────────────────────────────────────────

import shutil
shutil.copytree(OUT_DIR,
                '/content/drive/MyDrive/KayaBasa/results',
                dirs_exist_ok=True)
print("\nResults saved to Google Drive: My Drive/KayaBasa/results/")
print("  - results_config_A.json")
print("  - results_config_B.json")
print("  - all_features.csv       (14-feature matrix, denoised)")
