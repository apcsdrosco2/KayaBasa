"""Audit a corpus folder: what was found, what is missing, what was skipped."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from kayabasa_rf import config, data_loading


def counts_table(df: pd.DataFrame) -> pd.DataFrame:
    """Per-language counts by level, against the proposal's documented totals."""
    rows = []
    for language in sorted(df["language"].unique()):
        group = df[df["language"] == language]
        row = {"language": language}
        for label in config.LABELS:
            row[config.LABEL_NAMES[label]] = int((group["label"] == label).sum())
        row["total"] = len(group)
        expected = data_loading.EXPECTED_COUNTS.get(language)
        row["expected"] = expected if expected is not None else ""
        row["difference"] = (len(group) - expected) if expected is not None else ""
        rows.append(row)

    table = pd.DataFrame(rows)
    total_row = {
        "language": "TOTAL",
        **{
            config.LABEL_NAMES[label]: int((df["label"] == label).sum())
            for label in config.LABELS
        },
        "total": len(df),
        "expected": sum(data_loading.EXPECTED_COUNTS.values()),
        "difference": len(df) - sum(data_loading.EXPECTED_COUNTS.values()),
    }
    return pd.concat([table, pd.DataFrame([total_row])], ignore_index=True)


def compare_trees(raw: pd.DataFrame, clean: pd.DataFrame) -> pd.DataFrame:
    """Documents present in one tree but not the other."""
    key_columns = ["language", "label", "title"]
    raw_keys = set(map(tuple, raw[key_columns].to_numpy().tolist()))
    clean_keys = set(map(tuple, clean[key_columns].to_numpy().tolist()))

    rows = []
    for language, label, title in sorted(raw_keys | clean_keys):
        in_raw = (language, label, title) in raw_keys
        in_clean = (language, label, title) in clean_keys
        if not (in_raw and in_clean):
            rows.append(
                {
                    "language": language,
                    "level": config.LABEL_NAMES[label],
                    "title": title,
                    "in_raw": in_raw,
                    "in_clean": in_clean,
                }
            )
    return pd.DataFrame(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", help="A single corpus folder to audit")
    parser.add_argument(
        "--compare",
        nargs=2,
        metavar=("RAW", "CLEAN"),
        help="Compare a raw tree against its cleaned counterpart",
    )
    parser.add_argument("--out", default="results")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail instead of warning when a file cannot be placed",
    )
    args = parser.parse_args()

    if not args.root and not args.compare:
        parser.error("give --root FOLDER or --compare RAW CLEAN")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.root:
        print(f"\n{'=' * 70}\nAuditing {args.root}\n{'=' * 70}")
        df = data_loading.build_dataset(
            args.root, apply_denoising=False, strict=args.strict
        )
        table = counts_table(df)
        print(f"\n{table.to_string(index=False)}")
        table.to_csv(out_dir / "corpus_audit.csv", index=False)
        print(f"\n[out] {out_dir / 'corpus_audit.csv'}")

        empty = df[df["text"].str.strip() == ""]
        if not empty.empty:
            print(f"\n[warn] {len(empty)} document(s) are empty after loading:")
            for doc_id in empty["doc_id"].head(10):
                print(f"         {doc_id}")

    if args.compare:
        raw_root, clean_root = args.compare
        print(f"\n{'=' * 70}\nRaw tree: {raw_root}\n{'=' * 70}")
        raw = data_loading.build_dataset(
            raw_root, apply_denoising=False, strict=args.strict
        )
        print(f"\n{'=' * 70}\nClean tree: {clean_root}\n{'=' * 70}")
        clean = data_loading.build_dataset(
            clean_root, apply_denoising=False, already_clean=True, strict=args.strict
        )

        print(f"\n{'=' * 70}\nComparison\n{'=' * 70}")
        print(f"  raw:   {len(raw):>5} documents")
        print(f"  clean: {len(clean):>5} documents")
        print(f"  difference: {len(clean) - len(raw):+d}")

        diff = compare_trees(raw, clean)
        if diff.empty:
            print("\n  Every document appears in both trees.")
            return 0

        missing_from_clean = diff[~diff["in_clean"]]
        missing_from_raw = diff[~diff["in_raw"]]
        if not missing_from_clean.empty:
            print(
                f"\n  {len(missing_from_clean)} document(s) are in raw but NOT in clean:"
            )
            for _, row in missing_from_clean.head(20).iterrows():
                print(f"    {row['language']:<12} {row['level']:<4} {row['title']}")
            if len(missing_from_clean) > 20:
                print(f"    ... and {len(missing_from_clean) - 20} more")
        if not missing_from_raw.empty:
            print(
                f"\n  {len(missing_from_raw)} document(s) are in clean but NOT in raw:"
            )
            for _, row in missing_from_raw.head(20).iterrows():
                print(f"    {row['language']:<12} {row['level']:<4} {row['title']}")

        diff.to_csv(out_dir / "corpus_diff.csv", index=False)
        print(f"\n[out] {out_dir / 'corpus_diff.csv'}")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
