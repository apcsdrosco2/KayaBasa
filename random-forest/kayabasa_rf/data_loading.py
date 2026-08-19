"""Corpus ingestion (proposal Section 3.1.1)."""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

from . import config
from .denoise import prepare_corpus

SCHEMA = ["doc_id", "language", "label", "title", "text", "split_role"]

# Document counts documented in Table VI and the ara-close-lang summary.
EXPECTED_COUNTS = {
    "tagalog": 265,
    "cebuano": 349,
    "bikolano": 150,
    "hiligaynon": 133,
    "minasbate": 271,
    "karaya": 177,
    "rinconada": 195,
}


def load_ara_language(filepath: str | Path, language: str) -> pd.DataFrame:
    """Read one consolidated ara-close-lang ``.txt`` file."""
    records = []
    with open(filepath, encoding="utf-8", errors="replace") as f:
        for lineno, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            parts = line.split(",", 2)
            if len(parts) < 3:
                continue
            try:
                label = int(parts[1].strip())
            except ValueError:
                # A malformed label means the line schema is broken. Skipping it
                raise ValueError(
                    f"{filepath}:{lineno + 1} has a non-integer label: {parts[1]!r}"
                ) from None
            if label not in config.LABELS:
                raise ValueError(
                    f"{filepath}:{lineno + 1} has label {label}, "
                    f"expected one of {config.LABELS}"
                )
            records.append(
                {
                    "language": language,
                    "label": label,
                    "title": parts[0].strip(),
                    "text": parts[2].strip(),
                }
            )
    return pd.DataFrame(records)


def load_basaha_language(base_dir: str | Path, language: str) -> pd.DataFrame:
    """Read one BasahaCorpus language directory."""
    records = []
    for level_str, label_int in (("L1", 1), ("L2", 2), ("L3", 3)):
        level_dir = os.path.join(base_dir, level_str)
        if not os.path.isdir(level_dir):
            continue
        for fname in sorted(os.listdir(level_dir)):
            if not fname.endswith(".txt"):
                continue
            fpath = os.path.join(level_dir, fname)
            with open(fpath, encoding="utf-8", errors="replace") as f:
                text = f.read()
            records.append(
                {
                    "language": language,
                    "label": label_int,
                    "title": fname.replace(".txt", ""),
                    "text": text,
                }
            )
    return pd.DataFrame(records)


# Loading an arbitrary folder tree (a team's own raw/ and clean/ directories)
LANGUAGE_ALIASES = {
    "tagalog": ("tagalog", "filipino", "tgl", "tag"),
    "cebuano": ("cebuano", "binisaya", "bisaya", "ceb"),
    "bikolano": ("bikolano", "bicolano", "bikol", "bicol", "bik"),
    "hiligaynon": ("hiligaynon", "ilonggo", "hil"),
    "minasbate": ("minasbate", "masbatenyo", "masbate", "min"),
    "karaya": ("kinaray-a", "kinaraya", "karay-a", "karaya", "kar"),
    "rinconada": ("rinconada", "rin"),
}

# Directory names that encode a readability level.
LEVEL_ALIASES = {
    1: ("l1", "level1", "1", "grade1", "g1"),
    2: ("l2", "level2", "2", "grade2", "g2"),
    3: ("l3", "level3", "3", "grade3", "g3"),
}

_ALIAS_TO_LANGUAGE = {
    alias: language
    for language, aliases in LANGUAGE_ALIASES.items()
    for alias in aliases
}
_ALIAS_TO_LEVEL = {
    alias: label for label, aliases in LEVEL_ALIASES.items() for alias in aliases
}


def _normalize_component(component: str) -> str:
    return component.strip().lower().replace(" ", "").replace("_", "-")


def _infer_language(parts: tuple[str, ...]) -> str | None:
    """Find a language name anywhere in a file's path components."""
    ordered_aliases = sorted(_ALIAS_TO_LANGUAGE.items(), key=lambda kv: -len(kv[0]))
    for raw_part in reversed(parts):
        part = _normalize_component(raw_part)
        if part in _ALIAS_TO_LANGUAGE:
            return _ALIAS_TO_LANGUAGE[part]
        # Names glued to other text: "ceb_all_data", "01-cebuano", "cebuano.txt".
        for alias, language in ordered_aliases:
            if alias in part:
                return language
    return None


def _infer_label(parts: tuple[str, ...]) -> int | None:
    """Find an L1/L2/L3 marker among a file's parent directories."""
    for raw_part in reversed(parts[:-1]):  # directories only, not the filename
        part = _normalize_component(raw_part).replace("-", "")
        if part in _ALIAS_TO_LEVEL:
            return _ALIAS_TO_LEVEL[part]
    return None


def _looks_consolidated(path: Path) -> bool:
    """True when a .txt holds many documents, one per line, as ``Title,2,text``."""
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split(",", 2)
                return (
                    len(parts) >= 3
                    and parts[1].strip().isdigit()
                    and int(parts[1].strip()) in config.LABELS
                )
    except OSError:
        return False
    return False


def load_from_tree(root: str | Path, strict: bool = False) -> pd.DataFrame:
    """Load every ``.txt`` under ``root``, inferring language and label from the path."""
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f"Corpus folder not found: {root}")

    records: list[dict] = []
    skipped: list[str] = []

    for path in sorted(root.rglob("*.txt")):
        parts = path.relative_to(root).parts
        language = _infer_language(parts)
        if language is None:
            skipped.append(f"{path}  (no language found in path)")
            continue

        if _looks_consolidated(path):
            records.extend(
                load_ara_language(path, language).to_dict(orient="records")
            )
            continue

        label = _infer_label(parts)
        if label is None:
            skipped.append(f"{path}  (no L1/L2/L3 folder in path)")
            continue

        with open(path, encoding="utf-8", errors="replace") as f:
            records.append(
                {
                    "language": language,
                    "label": label,
                    "title": path.stem,
                    "text": f.read(),
                }
            )

    if skipped:
        message = f"{len(skipped)} file(s) could not be placed:\n  " + "\n  ".join(
            skipped[:15]
        )
        if len(skipped) > 15:
            message += f"\n  ... and {len(skipped) - 15} more"
        if strict:
            raise ValueError(message)
        print(f"[warn] {message}")

    if not records:
        raise FileNotFoundError(
            f"No usable .txt documents found under {root}. Expected either "
            "<language>/L1|L2|L3/*.txt or a consolidated <language>/*_all_data.txt."
        )

    df = pd.DataFrame(records)
    df["split_role"] = [
        "high_resource" if lang in config.HIGH_RESOURCE else "low_resource"
        for lang in df["language"]
    ]
    # Sort before assigning ids so doc_id is stable across the raw and clean
    df = df.sort_values(["language", "label", "title"], kind="stable").reset_index(
        drop=True
    )
    df["doc_id"] = [
        f"{lang}_{i:04d}"
        for lang, i in zip(
            df["language"], df.groupby("language").cumcount(), strict=True
        )
    ]
    df = df[SCHEMA]
    _report_counts(df)
    return df


def load_corpus(data_root: str | Path, strict: bool = False) -> pd.DataFrame:
    """Load both corpora and assign ``split_role`` (Section 3.1.1.3)."""
    data_root = Path(data_root)
    frames, missing = [], []

    for language, rel in config.ARA_FILES.items():
        path = data_root / rel
        if not path.exists():
            missing.append(str(path))
            continue
        df = load_ara_language(path, language)
        df["split_role"] = "high_resource"
        frames.append(df)

    for language, rel in config.BASAHA_DIRS.items():
        path = data_root / rel
        if not path.is_dir():
            missing.append(str(path))
            continue
        df = load_basaha_language(path, language)
        df["split_role"] = "low_resource"
        frames.append(df)

    if missing:
        message = "Corpus paths not found:\n  " + "\n  ".join(missing)
        if strict:
            raise FileNotFoundError(message)
        print(f"[warn] {message}")

    if not frames:
        raise FileNotFoundError(
            f"No corpus data found under {data_root}. Clone the imperialite "
            "repositories there, or run make_synthetic_corpus.py for a smoke test."
        )

    df_all = pd.concat(frames, ignore_index=True)

    # Stable, human-readable document id used to join every downstream artifact
    df_all["doc_id"] = [
        f"{lang}_{i:04d}"
        for lang, i in zip(
            df_all["language"], df_all.groupby("language").cumcount(), strict=True
        )
    ]
    df_all = df_all[SCHEMA]

    _report_counts(df_all)
    return df_all


def _report_counts(df: pd.DataFrame) -> None:
    for language, group in df.groupby("language"):
        expected = EXPECTED_COUNTS.get(language)
        got = len(group)
        flag = ""
        if expected is not None and got != expected:
            flag = f"  [warn] expected {expected} per the proposal"
        by_label = ", ".join(
            f"{config.LABEL_NAMES[lab]}={int((group['label'] == lab).sum())}"
            for lab in config.LABELS
        )
        print(f"  {language:<12} {got:>4} docs ({by_label}){flag}")


def _has_repo_layout(data_root: Path) -> bool:
    """True when the two imperialite repositories sit under ``data_root`` verbatim."""
    return any((data_root / rel).exists() for rel in config.ARA_FILES.values()) or any(
        (data_root / rel).is_dir() for rel in config.BASAHA_DIRS.values()
    )


def build_dataset(
    data_root: str | Path,
    apply_denoising: bool = True,
    strict: bool = False,
    cache_path: str | Path | None = None,
    already_clean: bool = False,
    layout: str = "auto",
) -> pd.DataFrame:
    """Load, prepare, and optionally cache the corpus."""
    data_root = Path(data_root)

    if layout == "auto":
        layout = "repo" if _has_repo_layout(data_root) else "tree"

    if already_clean:
        condition = "pre-cleaned by the team"
    else:
        condition = "denoised" if apply_denoising else "noisy"
    print(f"[data] loading corpus from {data_root} (layout: {layout}, {condition})")

    df_all = (
        load_corpus(data_root, strict=strict)
        if layout == "repo"
        else load_from_tree(data_root, strict=strict)
    )
    df_all = prepare_corpus(
        df_all, apply_denoising=apply_denoising and not already_clean
    )

    empty = int(df_all["text"].str.strip().eq("").sum())
    if empty:
        print(f"[warn] {empty} documents are empty after preparation")

    if cache_path is not None:
        cache_path = Path(cache_path)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        df_all.to_pickle(cache_path)
        print(f"[data] cached prepared corpus -> {cache_path}")

    return df_all
