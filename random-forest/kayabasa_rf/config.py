"""Central configuration for the KAYABASA Random Forest baseline."""

from __future__ import annotations

import os
import random
from dataclasses import dataclass, field
from pathlib import Path

# Reproducibility (proposal Section 3.4.4)
SEED = 42


def set_global_seed(seed: int = SEED) -> None:
    """Fix every RNG this pipeline touches, before any split or model init."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:  # numpy is required at runtime, tolerated at import time
        pass


# Corpus layout (proposal Sections 3.1.1.1 - 3.1.1.3)
LABELS = (1, 2, 3)
LABEL_NAMES = {1: "L1", 2: "L2", 3: "L3"}

HIGH_RESOURCE = ("tagalog", "cebuano", "bikolano")
LOW_RESOURCE = ("hiligaynon", "minasbate", "karaya", "rinconada")

# One consolidated ``.txt`` per language; label is field 2 of each line.
ARA_FILES = {
    "tagalog": "ara-close-lang/data/tagalog/tag_all_data.txt",
    "cebuano": "ara-close-lang/data/cebuano/ceb_all_data.txt",
    "bikolano": "ara-close-lang/data/bikol/bik_all_data.txt",
}

# One ``.txt`` per document; label comes from the /L1/, /L2/, /L3/ subdirectory.
BASAHA_DIRS = {
    "hiligaynon": "BasahaCorpus-HierarchicalCrosslingualARA/data/hiligaynon",
    "minasbate": "BasahaCorpus-HierarchicalCrosslingualARA/data/minasbate",
    "karaya": "BasahaCorpus-HierarchicalCrosslingualARA/data/karaya",
    "rinconada": "BasahaCorpus-HierarchicalCrosslingualARA/data/rinconada",
}

# Anchor language for CROSSNGO cross-lingual n-gram overlap (Section 3.2.2.3).
ANCHOR_LANGUAGE = "cebuano"

# Only the top quartile of the anchor profile is compared (Section 3.2.2.3).
ANCHOR_TOP_FRACTION = 0.25

# Persistence parameter for Rank-Biased Overlap. 0.9 puts roughly 86% of the
RBO_P = 0.9


# Cross-validation (proposal Section 3.1.6)
N_SPLITS = 5


# Random Forest baseline (proposal Table XIII)
@dataclass(frozen=True)
class RFParams:
    """Defaults mirror the WEKA RandomForest settings documented in the BasahaCorpus README, so the baseline is comparable to prior Philippine ARA work (Imperial & Kochmar 2023) rather than an arbitrarily weakened rival."""

    n_estimators: int = 100
    max_features: str = "log2"
    max_depth: int | None = None
    min_samples_leaf: int = 1
    class_weight: str | None = None
    n_jobs: int = -1
    random_state: int = SEED

    def to_sklearn(self) -> dict:
        return {
            "n_estimators": self.n_estimators,
            "max_features": self.max_features,
            "max_depth": self.max_depth,
            "min_samples_leaf": self.min_samples_leaf,
            "class_weight": self.class_weight,
            "n_jobs": self.n_jobs,
            "random_state": self.random_state,
        }


# Optional grid, searched *inside* each training fold only, never on validation.
RF_GRID = {
    "n_estimators": [100, 300, 500],
    "max_features": ["log2", "sqrt"],
    "min_samples_leaf": [1, 2, 4],
}


# Decision thresholds stated in the proposal
HYBRID_SUPERIORITY_MARGIN = 0.05

# Section 3.1.5 - denoising counts as a real pipeline contribution when
DENOISING_DELTA_THRESHOLD = 0.03
DENOISING_MIN_LANGUAGES = 2


# Output layout
@dataclass
class Paths:
    root: Path = field(default_factory=lambda: Path("."))

    @property
    def data(self) -> Path:
        return self.root / "data"

    @property
    def features(self) -> Path:
        return self.root / "features"

    @property
    def splits(self) -> Path:
        return self.root / "splits"

    @property
    def models(self) -> Path:
        return self.root / "models"

    @property
    def results(self) -> Path:
        return self.root / "results"

    def ensure(self) -> "Paths":
        for p in (self.data, self.features, self.splits, self.models, self.results):
            p.mkdir(parents=True, exist_ok=True)
        return self
