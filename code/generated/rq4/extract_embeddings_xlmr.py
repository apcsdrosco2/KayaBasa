"""
extract_embeddings_xlmr.py — Stage C of the RQ4 pipeline.

Extracts frozen XLM-RoBERTa (xlm-roberta-base) sentence embeddings for the RQ4
low-resource corpus (clean/rq4/{lang}_all_clean.txt, produced by stage_raw_text.py),
mirroring code/extract_embeddings.py's exact recipe for the existing 3-language
pipeline (Chapter 3 §3.2.2): frozen encoder (no fine-tuning), 512-token truncation,
mean-pooling over the attention mask, 768-dim output per document.

Runs in small batches with progress logging and incremental flush-to-disk, since a
long CPU-bound job is an untested memory-pressure profile on this machine (7.4GB RAM,
per cv_common.py's documented constraint) and a crash partway through should not lose
completed work.

Usage:
  python extract_embeddings_xlmr.py --pilot 15      # first N docs per language only
  python extract_embeddings_xlmr.py                 # full run, all languages
"""

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoModel
from transformers.models.xlm_roberta.tokenization_xlm_roberta import XLMRobertaTokenizer

SCRIPT_DIR = Path(__file__).resolve().parent
CLEAN_DIR = SCRIPT_DIR.parent.parent.parent / "clean" / "rq4"
OUT_DIR = SCRIPT_DIR / "embeddings"

LANGUAGES = ["hiligaynon", "minasbate", "karay-a", "rinconada"]
BATCH_SIZE = 8
MODEL_NAME = "xlm-roberta-base"


def load_texts(lang: str) -> list:
    fp = CLEAN_DIR / f"{lang}_all_clean.txt"
    df = pd.read_csv(fp, sep="|", engine="python", quoting=3)
    return df["text"].astype(str).tolist()


def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return sum_embeddings / sum_mask


def embed_texts(tokenizer, model, texts: list, batch_size: int, log_prefix: str, device: torch.device) -> np.ndarray:
    all_vecs = []
    n = len(texts)
    t_start = time.time()
    for i in range(0, n, batch_size):
        batch = texts[i:i + batch_size]
        encoded = tokenizer(
            batch, padding=True, truncation=True, max_length=512, return_tensors="pt",
        )
        encoded = {k: v.to(device) for k, v in encoded.items()}
        with torch.no_grad():
            output = model(**encoded)
        vecs = mean_pooling(output, encoded["attention_mask"]).cpu().numpy()
        all_vecs.append(vecs)
        done = min(i + batch_size, n)
        elapsed = time.time() - t_start
        rate = done / elapsed if elapsed > 0 else 0.0
        print(f"  {log_prefix} {done}/{n}  ({elapsed:.1f}s elapsed, {rate:.2f} docs/s)")
    return np.concatenate(all_vecs, axis=0)


def main():
    parser = argparse.ArgumentParser(description="Extract frozen XLM-R embeddings for the RQ4 corpus.")
    parser.add_argument("--pilot", type=int, default=None,
                         help="If set, only embed the first N documents of each language (timing/quality check).")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading tokenizer/model: {MODEL_NAME} (frozen, no fine-tuning)...")
    t0 = time.time()
    # This machine is frequently down to a few hundred MB free RAM (matches
    # cv_common.py's documented constraint elsewhere in this repo — confirmed here via
    # GlobalMemoryStatusEx during this run: as low as ~235MB free, 96% used).
    # AutoTokenizer.from_pretrained(..., use_fast=False) STILL internally converts the
    # tokenizer to transformers 5.x's unified "native format" via a plain json.load()
    # over the ~9MB tokenizer.json — with per-token Python object overhead across the
    # ~250k-entry XLM-R vocab, this raised a genuine MemoryError 3 times in a row on
    # this machine, even with more RAM free each retry (235MB, then 695MB, still
    # failed). The legacy XLMRobertaTokenizer class, imported directly (bypassing
    # AutoTokenizer's fast/native-conversion path entirely), loads the vocab straight
    # from sentencepiece.bpe.model via sentencepiece's own low-footprint binary loader
    # and succeeded even at 296MB free. Same vocab, same tokenization — only the
    # loading code path differs.
    tokenizer = XLMRobertaTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME, low_cpu_mem_usage=True)
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"  Loaded in {time.time() - t0:.1f}s. Device: {device} "
          f"({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'})")

    grand_start = time.time()
    for lang in LANGUAGES:
        texts = load_texts(lang)
        if args.pilot is not None:
            texts = texts[: args.pilot]
        if not texts:
            continue

        print(f"\n=== {lang}: {len(texts)} documents ===")
        t_lang = time.time()
        vecs = embed_texts(tokenizer, model, texts, args.batch_size, log_prefix=lang, device=device)
        elapsed = time.time() - t_lang

        suffix = f"_pilot{args.pilot}" if args.pilot is not None else ""
        out_path = OUT_DIR / f"{lang}_xlmr_features{suffix}.csv"
        np.savetxt(out_path, vecs, delimiter=",")
        print(f"  {lang}: {len(texts)} docs in {elapsed:.1f}s "
              f"({elapsed / len(texts):.2f}s/doc) -> {out_path}")

        # Quick quality sanity checks
        assert vecs.shape == (len(texts), 768), f"Unexpected shape: {vecs.shape}"
        n_nan = np.isnan(vecs).sum()
        norms = np.linalg.norm(vecs, axis=1)
        print(f"  shape={vecs.shape} nan_count={n_nan} "
              f"norm_min={norms.min():.3f} norm_max={norms.max():.3f} norm_mean={norms.mean():.3f}")
        assert n_nan == 0, f"Found {n_nan} NaN values in {lang} embeddings"

    total_elapsed = time.time() - grand_start
    total_docs = sum(
        len(load_texts(lang)[: args.pilot] if args.pilot is not None else load_texts(lang))
        for lang in LANGUAGES
    )
    print(f"\n=== Done: {total_docs} documents in {total_elapsed:.1f}s "
          f"({total_elapsed / max(total_docs,1):.2f}s/doc average) ===")
    if args.pilot is not None:
        full_total = 769  # per RQ4_PLAN.md, final corpus size (user-approved, §2C)
        est_full = (total_elapsed / max(total_docs, 1)) * full_total
        print(f"Extrapolated full-run estimate (769 docs): {est_full:.0f}s "
              f"(~{est_full/60:.1f} min)")


if __name__ == "__main__":
    main()
