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


# ── CELL 1: Mount Drive & Install Dependencies ───────────────────────────────

from google.colab import drive
drive.mount('/content/drive')

import os, sys, json, pickle
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score, accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler

print("PyTorch:", torch.__version__)
print("GPU available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("Device:", torch.cuda.get_device_name(0))

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.system("pip install ftfy regex scikit-learn pandas numpy --quiet")
os.makedirs('/content/drive/MyDrive/KayaBasa/results', exist_ok=True)
print("Dependencies ready.")


# ── CELL 2: Clone Repositories ───────────────────────────────────────────────

print("Cloning KayaBasa...")
os.system("git clone https://github.com/apcsdrosco2/KayaBasa.git /content/KayaBasa")
os.chdir('/content/KayaBasa')
# Use feature-pipeline branch (has the 14-feature pipeline)
os.system("git checkout feat/feature-pipeline-refactor")

print("Cloning BasahaCorpus (low-resource languages)...")
os.system(
    "git clone https://github.com/imperialite/BasahaCorpus-HierarchicalCrosslingualARA.git "
    "/content/KayaBasa/raw/BasahaCorpus-HierarchicalCrosslingualARA"
)

# Verify the clean/ folder is present (764 pre-cleaned HR docs)
clean_files = [
    '/content/KayaBasa/clean/tag_all_clean.txt',
    '/content/KayaBasa/clean/ceb_all_clean.txt',
    '/content/KayaBasa/clean/bik_all_clean.txt',
]
for f in clean_files:
    assert os.path.exists(f), f"Missing: {f}"

from collections import Counter
total = 0
for lang, path in zip(['tagalog','cebuano','bikol'], clean_files):
    with open(path, encoding='utf-8') as fh:
        lines = [l for l in fh.read().splitlines() if l.strip()]
    lc = Counter(l.split(',',2)[1].strip() for l in lines if len(l.split(',',2))>=2)
    total += len(lines)
    print(f"  {lang:<10}: {len(lines)} docs  L1={lc.get('1',0)}  L2={lc.get('2',0)}  L3={lc.get('3',0)}")

print(f"\n  TOTAL: {total} docs  (thesis 764 = {total == 764})")


# ── CELL 3: Build Corpus from clean/ (no re-cleaning) ────────────────────────
# build_from_clean.py reads clean/ → output/all_languages.txt (noisy/raw copy)
# This is Config A's source (text before normalize_datasets.py normalization).

os.chdir('/content/KayaBasa/data-cleaning-repo-main')
os.system("python build_from_clean.py")

# Verify
assert os.path.exists("output/all_languages.txt"),        "output/all_languages.txt missing"
assert os.path.exists("output/all_languages_raw.txt"),    "output/all_languages_raw.txt missing"
print("Noisy corpus ready: output/all_languages.txt")


# ── CELL 4: Structural Normalization → Denoised Corpus (Config B) ────────────
# normalize_datasets.py applies structural fixes only (no stemming/lemmatization):
#   - NFC Unicode normalisation
#   - space after closing punctuation
#   - affix/reduplication hyphen joining
#   - whitespace collapse
# Input:  output/all_languages.txt
# Output: output_normalized/all_languages.txt  ← Config B source

os.system("python normalize_datasets.py")
assert os.path.exists("output_normalized/all_languages.txt"), "Normalized corpus missing"
print("Denoised corpus ready: output_normalized/all_languages.txt")


# ── CELL 5: Generate Stratified 5-Fold CV Splits (764 HR docs) ───────────────

os.system("python split_datasets.py")
assert os.path.exists("splits/fold_indices.pkl"),         "fold_indices.pkl missing"
assert os.path.exists("splits/high_resource_index.csv"), "high_resource_index.csv missing"

hr_index = pd.read_csv("splits/high_resource_index.csv")
print(f"\n5-Fold CV splits ready. HR docs: {len(hr_index)}")
print(hr_index.groupby(['language','label']).size().to_string())

with open("splits/fold_indices.pkl","rb") as f:
    fold_indices = pickle.load(f)
print(f"\nFold sizes (train / val):")
for i, s in enumerate(fold_indices, 1):
    print(f"  Fold {i}: train={len(s['train'])}  val={len(s['val'])}")


# ── CELL 6: Extract 14 Linguistic Features from Both Corpora ─────────────────

os.chdir('/content/KayaBasa/feature-pipeline')
sys.path.insert(0, '/content/KayaBasa/feature-pipeline')
from feature_pipeline import FeaturePipeline, FEATURE_COLS, load_corpus

OUT_DIR = '/content/KayaBasa/feature-pipeline/output'
os.makedirs(OUT_DIR, exist_ok=True)

NOISY_CORPUS    = "/content/KayaBasa/data-cleaning-repo-main/output/all_languages_raw.txt"
DENOISED_CORPUS = "/content/KayaBasa/data-cleaning-repo-main/output_normalized/all_languages.txt"

def extract(corpus_path, tag):
    print(f"\nExtracting features [{tag}]...")
    df   = load_corpus(corpus_path)
    pipe = FeaturePipeline()
    X    = pipe.fit_transform(df)
    print(f"  {len(X)} docs × {len(FEATURE_COLS)} features")
    return X

X_noisy    = extract(NOISY_CORPUS,    "NOISY   (Config A)")
X_denoised = extract(DENOISED_CORPUS, "DENOISED (Config B)")

# Save denoised matrix (reused in Config F later)
X_denoised.to_csv(f"{OUT_DIR}/all_features.csv", index=False)
print(f"\nSaved: {OUT_DIR}/all_features.csv")

# Sanity check
hr_n = len(hr_index)
hr_noisy    = X_noisy[X_noisy["split_role"]=="high_resource"]
hr_denoised = X_denoised[X_denoised["split_role"]=="high_resource"]
print(f"\nHR docs in feature matrix: noisy={len(hr_noisy)}  denoised={len(hr_denoised)}")
assert len(hr_noisy)    == hr_n, f"Expected {hr_n}, got {len(hr_noisy)}"
assert len(hr_denoised) == hr_n, f"Expected {hr_n}, got {len(hr_denoised)}"
print(f"Sanity OK: both matrices have {hr_n} HR docs.")


# ── CELL 7: MLP Classifier Definition ────────────────────────────────────────
# Same MLP head used in the full KAYABASA hybrid (Config F).
# For A/B: input is 14-dim feature vector.
# For F:   input is 768 (XLM-R CLS) + 14 (features) = 782-dim.

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)


class FeaturesDataset(Dataset):
    def __init__(self, features: np.ndarray, labels: list):
        self.X = torch.tensor(features, dtype=torch.float32)
        self.y = torch.tensor(labels,   dtype=torch.long)
    def __len__(self):  return len(self.y)
    def __getitem__(self, i): return self.X[i], self.y[i]


class FeatureMLP(nn.Module):
    """
    MLP classifier for features-only ablation (§3.2.1).
    n_features=14  → Configs A and B
    n_features=782 → Config F (XLM-R CLS + 14 features)
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
            preds = model(X_b.to(DEVICE)).argmax(dim=1)
            y_true.extend(y_b.tolist())
            y_pred.extend(preds.cpu().tolist())
    macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
    acc   = accuracy_score(y_true, y_pred)
    f1_pc = f1_score(y_true, y_pred, average=None, labels=[0,1,2], zero_division=0).tolist()
    return macro, acc, f1_pc, y_true, y_pred


# ── CELL 8: 5-Fold CV Training Function ──────────────────────────────────────

def run_cv(X_full, fold_indices, hr_index, config_name, save_prefix,
           lr=1e-3, epochs=200, patience=20, batch_size=64):
    """
    Stratified 5-Fold CV with MLP on the 14 linguistic features.
    StandardScaler is fit only on the train split each fold (no data leakage).
    """
    print(f"\n{'='*60}")
    print(f"  Config {config_name} — 5-Fold CV (MLP, lr={lr})")
    print(f"{'='*60}")

    X_hr = X_full[X_full["split_role"]=="high_resource"].set_index("doc_id")
    label_map = {1: 0, 2: 1, 3: 2}

    fold_results, lang_records = [], []
    all_true_global, all_pred_global = [], []

    for fold_num, splits in enumerate(fold_indices, 1):
        train_ids = hr_index.loc[splits["train"], "doc_id"].values
        val_ids   = hr_index.loc[splits["val"],   "doc_id"].values

        scaler   = StandardScaler()
        X_train  = scaler.fit_transform(X_hr.loc[train_ids, FEATURE_COLS].values)
        X_val    = scaler.transform(    X_hr.loc[val_ids,   FEATURE_COLS].values)
        y_train  = [label_map[X_hr.loc[d, "label"]] for d in train_ids]
        y_val    = [label_map[X_hr.loc[d, "label"]] for d in val_ids]

        train_dl = DataLoader(FeaturesDataset(X_train, y_train),
                              batch_size=batch_size, shuffle=True)
        val_dl   = DataLoader(FeaturesDataset(X_val,   y_val),
                              batch_size=batch_size)

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

        model.load_state_dict(torch.load(ckpt))
        macro, acc, f1_pc, y_true, y_pred = evaluate(model, val_dl)

        all_true_global.extend(y_true)
        all_pred_global.extend(y_pred)

        for doc_id, yt, yp in zip(val_ids, y_true, y_pred):
            lang_records.append({
                "doc_id":   doc_id,
                "language": X_hr.loc[doc_id, "language"],
                "y_true":   yt,
                "y_pred":   yp,
            })

        fold_results.append({
            "fold": fold_num, "n_train": len(y_train), "n_val": len(y_val),
            "macro_f1": round(macro,4), "accuracy": round(acc,4),
            "f1_L1": round(f1_pc[0],4), "f1_L2": round(f1_pc[1],4),
            "f1_L3": round(f1_pc[2],4), "best_epoch": best_ep,
        })
        print(f"  Fold {fold_num}: Macro-F1={macro:.4f}  Acc={acc:.4f}"
              f"  [L1={f1_pc[0]:.3f} L2={f1_pc[1]:.3f} L3={f1_pc[2]:.3f}]"
              f"  (best ep {best_ep})")

    macro_vals = [r["macro_f1"] for r in fold_results]
    acc_vals   = [r["accuracy"] for r in fold_results]
    mean_macro = float(np.mean(macro_vals))
    std_macro  = float(np.std(macro_vals))
    mean_acc   = float(np.mean(acc_vals))
    std_acc    = float(np.std(acc_vals))

    print(f"\n  Mean Macro-F1 : {mean_macro:.4f} +/- {std_macro:.4f}")
    print(f"  Mean Accuracy : {mean_acc:.4f}   +/- {std_acc:.4f}")

    # Per-language out-of-fold breakdown
    oof = pd.DataFrame(lang_records)
    per_lang = {}
    print("\n  Per-language Macro-F1 (out-of-fold):")
    for lang, grp in oof.groupby("language"):
        mf1 = float(f1_score(grp["y_true"], grp["y_pred"], average="macro", zero_division=0))
        mac = float(accuracy_score(grp["y_true"], grp["y_pred"]))
        per_lang[lang] = {"n_docs": len(grp), "macro_f1": round(mf1,4), "accuracy": round(mac,4)}
        print(f"    {lang:<14}  F1={mf1:.4f}  Acc={mac:.4f}  (n={len(grp)})")

    # Full classification report
    print("\n  Classification Report (all folds combined):")
    print(classification_report(
        all_true_global, all_pred_global,
        target_names=["L1 (Grade 1)", "L2 (Grade 2)", "L3 (Grade 3)"],
        zero_division=0
    ))

    result = {
        "config": config_name,
        "n_hr_docs":     int(hr_index.shape[0]),
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
    print(f"  Saved: {out_path}")
    return result


# ── CELL 9: Run Config B — Denoised Features + MLP ───────────────────────────

res_b = run_cv(
    X_full       = X_denoised,
    fold_indices = fold_indices,
    hr_index     = hr_index,
    config_name  = "B (Denoised Features, MLP)",
    save_prefix  = "B",
)


# ── CELL 10: Run Config A — Noisy Features + MLP ─────────────────────────────

res_a = run_cv(
    X_full       = X_noisy,
    fold_indices = fold_indices,
    hr_index     = hr_index,
    config_name  = "A (Noisy Features, MLP)",
    save_prefix  = "A",
)


# ── CELL 11: Summary — Config A vs B Comparison ──────────────────────────────

print(f"\n{'='*60}")
print("KAYABASA -- Features-Only Ablation (MLP) Summary")
print(f"{'='*60}")
print(f"{'Config':<35} {'Macro-F1':>10} {'Std':>8} {'Accuracy':>10}")
print("-"*65)
for r in (res_a, res_b):
    print(f"  {r['config']:<33} {r['mean_macro_f1']:>10.4f}"
          f" {r['std_macro_f1']:>8.4f} {r['mean_accuracy']:>10.4f}")

delta = res_b["mean_macro_f1"] - res_a["mean_macro_f1"]
print(f"\nDenoising impact (B - A): Delta Macro-F1 = {delta:+.4f}")
status = "SIGNIFICANT (>= 0.03)" if delta >= 0.03 else "below 0.03 threshold (expected for features)"
print(f"Thesis §3.1.5 threshold : {status}")

print(f"\n{'Per-language Macro-F1':}")
print(f"  {'Language':<14} {'Config A':>10} {'Config B':>10} {'Delta':>8}")
print("  " + "-"*44)
langs = sorted(set(res_a["per_language"]) | set(res_b["per_language"]))
for lang in langs:
    a = res_a["per_language"].get(lang, {}).get("macro_f1", float('nan'))
    b = res_b["per_language"].get(lang, {}).get("macro_f1", float('nan'))
    d = b - a if isinstance(a, float) and isinstance(b, float) else float('nan')
    print(f"  {lang:<14} {a:>10.4f} {b:>10.4f} {d:>+8.4f}")


# ── CELL 12: Save Results to Google Drive ────────────────────────────────────

import shutil
shutil.copytree(OUT_DIR, '/content/drive/MyDrive/KayaBasa/results', dirs_exist_ok=True)
print("\nResults saved to Google Drive: My Drive/KayaBasa/results/")
print("  results_config_A.json  — Config A (Noisy + MLP)")
print("  results_config_B.json  — Config B (Denoised + MLP)")
print("  all_features.csv       — 14-feature matrix (denoised, all 1480 docs)")
print("  *.pt                   — Best MLP checkpoints per fold")
