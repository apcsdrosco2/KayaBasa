"""
smoke_test_model_load.py — lower-risk precursor to the Stage C pilot.

Loads the XLM-R tokenizer + model ONLY (no batch embedding, no text processing) and
exits immediately. This isolates "does the model load survive at current memory
levels" from the full pilot, since the model load itself (~1.1GB) is the highest
single memory spike in the whole Stage C pipeline, and this machine has a documented
history of crashing at this exact step (RQ4_PLAN.md §8).

Prints free memory before and after the load so the result is unambiguous either way.
"""

import ctypes
import os
import time

# Set BEFORE importing torch/transformers: the tokenizers crate (Rust, via rayon) and
# OpenMP/torch's own thread pool each pre-allocate per-thread memory arenas at import/
# init time. Forcing single-threaded avoids that multiplied-by-core-count overhead,
# which matters when a single-digit number of MB can be the difference between an
# allocation succeeding or aborting on this machine (RQ4_PLAN.md §8 documents a Rust-
# level allocator abort inside the tokenizers crate at the tokenizer-load step, before
# the model itself is even touched).
os.environ.setdefault("RAYON_NUM_THREADS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")


def free_mb() -> float:
    class MEMORYSTATUSEX(ctypes.Structure):
        _fields_ = [
            ("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
            ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
            ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
            ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
            ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
        ]
    m = MEMORYSTATUSEX()
    m.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
    return m.ullAvailPhys / 1e6


print(f"Free memory before import: {free_mb():.0f} MB")

import torch  # noqa: E402

torch.set_num_threads(1)

from transformers import AutoModel  # noqa: E402
from transformers.models.xlm_roberta.tokenization_xlm_roberta import XLMRobertaTokenizer  # noqa: E402

print(f"Free memory after transformers/torch import: {free_mb():.0f} MB")

t0 = time.time()
tokenizer = XLMRobertaTokenizer.from_pretrained("xlm-roberta-base")
print(f"Tokenizer loaded in {time.time() - t0:.1f}s. Free memory: {free_mb():.0f} MB")

t1 = time.time()
# float16 halves the model's resident weight memory (~1.1GB -> ~550MB) vs. the
# default float32. This is a quality/precision trade-off relative to the recipe used
# for Tagalog/Cebuano/Bikolano's embeddings (full float32) -- flagged clearly in this
# script's own output and in RQ4_PLAN.md, not silently substituted.
model = AutoModel.from_pretrained("xlm-roberta-base", torch_dtype=torch.float16, low_cpu_mem_usage=True)
model.eval()
print(f"Model loaded in {time.time() - t1:.1f}s (dtype=float16). Free memory: {free_mb():.0f} MB")

# One trivial forward pass on a single short string, to confirm it's actually usable,
# not just resident in memory.
encoded = tokenizer(["Ini na libro."], padding=True, truncation=True, max_length=512, return_tensors="pt")
with torch.no_grad():
    output = model(**encoded)
print(f"Forward pass output shape: {tuple(output[0].shape)}. Free memory: {free_mb():.0f} MB")

print("\nSMOKE TEST PASSED — tokenizer + model load + one forward pass all survived.")
