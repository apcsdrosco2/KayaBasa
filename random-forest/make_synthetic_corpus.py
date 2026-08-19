"""Generate a synthetic stand-in corpus so the pipeline can be smoke-tested."""

from __future__ import annotations

import argparse
import random
from pathlib import Path

from kayabasa_rf import config

ONSETS = ["b", "k", "d", "g", "h", "l", "m", "n", "ng", "p", "r", "s", "t", "w", "y"]
VOWELS = ["a", "e", "i", "o", "u"]
CODAS = ["", "", "", "n", "s", "t", "k", "ng"]

FUNCTION_WORDS = ["ang", "ng", "sa", "si", "kag", "ug", "ni", "na", "at", "mga"]

# label -> (syllables per content word, words per sentence, sentences per doc)
DIFFICULTY_PROFILE = {
    1: (2, 6, 8),
    2: (3, 10, 12),
    3: (4, 15, 16),
}

ARA_LANG_FILES = {
    "tagalog": ("tagalog", "tag_all_data.txt"),
    "cebuano": ("cebuano", "ceb_all_data.txt"),
    "bikolano": ("bikol", "bik_all_data.txt"),
}


def blurred_label(rng: random.Random, label: int, noise: float) -> int:
    """Occasionally write a document in a neighbouring level's style."""
    if noise <= 0 or rng.random() >= noise:
        return label
    return rng.choice([c for c in config.LABELS if abs(c - label) == 1])


def make_word(rng: random.Random, syllables: int) -> str:
    return "".join(
        rng.choice(ONSETS) + rng.choice(VOWELS) + rng.choice(CODAS)
        for _ in range(max(1, syllables))
    )


def make_sentence(rng: random.Random, label: int) -> str:
    syllables, words_per_sentence, _ = DIFFICULTY_PROFILE[label]
    words = []
    for i in range(max(2, words_per_sentence + rng.randint(-2, 2))):
        if i % 3 == 1:
            words.append(rng.choice(FUNCTION_WORDS))
        else:
            words.append(make_word(rng, syllables + rng.randint(-1, 1)))
    return " ".join(words).capitalize() + rng.choice([".", ".", ".", "!", "?"])


def make_narrative(rng: random.Random, label: int) -> str:
    _, _, n_sentences = DIFFICULTY_PROFILE[label]
    return " ".join(make_sentence(rng, label) for _ in range(n_sentences))


def make_ara_line(rng: random.Random, title: str, label: int, style: int) -> str:
    """One consolidated-file line, complete with the noise Table IV documents."""
    parts = [title]
    if rng.random() < 0.42:  # author/illustrator credits: 42% of lines
        parts.append(
            f"Gisulat ni: {make_word(rng, 2).capitalize()} "
            f"{make_word(rng, 2).capitalize()}"
        )
        parts.append(f"Gidibuho ni: {make_word(rng, 2).capitalize()}")
    parts.append(make_narrative(rng, style))
    if rng.random() < 0.08:  # HTML entities: 8% of lines
        parts.append("&quot;&#160;")
    if rng.random() < 0.03:  # publisher URLs: 3% of lines
        parts.append("https://letsreadasia.org")
    if rng.random() < 0.93:  # KATAPUSAN marker: 93% of lines
        parts.append("KATAPUSAN")
    # The body contains commas, which is exactly why the loader splits on the
    return f"{title},{label}," + " ".join(parts)


def make_basaha_document(rng: random.Random, title: str, style: int) -> str:
    """One Let's Read export, with the boilerplate block Table V documents."""
    return "\n".join(
        [
            f"{make_word(rng, 2).capitalize()} {make_word(rng, 2).capitalize()}",
            "",
            "Cover",
            title,
            "",
            make_narrative(rng, style),
            "",
            "Page 12",
            "",
            "Frog's Exercise",
            "Answer the following questions about the story you just read.",
            "Let's Read is an initiative of The Asia Foundation.",
            "For full terms of use, visit the website.",
            "Contributing translators: Anonymous",
        ]
    )


def write_corpus(
    out_root: Path, docs_per_level: int, seed: int, label_noise: float
) -> list[dict]:
    rng = random.Random(seed)
    manifest = []

    for language, (folder, filename) in ARA_LANG_FILES.items():
        path = out_root / "ara-close-lang" / "data" / folder / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        lines, index = [], 0
        for label in config.LABELS:
            for _ in range(docs_per_level):
                title = f"Titulo {make_word(rng, 2).capitalize()}"
                style = blurred_label(rng, label, label_noise)
                lines.append(make_ara_line(rng, title, label, style))
                manifest.append(
                    {
                        "doc_id": f"{language}_{index:04d}",
                        "language": language,
                        "y_true": label,
                    }
                )
                index += 1
        path.write_text("\n".join(lines), encoding="utf-8")
        print(f"  wrote {len(lines):>4} documents -> {path}")

    for language in config.LOW_RESOURCE:
        base = out_root / "BasahaCorpus-HierarchicalCrosslingualARA" / "data" / language
        for label in config.LABELS:
            level_dir = base / config.LABEL_NAMES[label]
            level_dir.mkdir(parents=True, exist_ok=True)
            for n in range(docs_per_level):
                title = f"Titulo_{make_word(rng, 2).capitalize()}_{n}"
                style = blurred_label(rng, label, label_noise)
                (level_dir / f"{title}.txt").write_text(
                    make_basaha_document(rng, title, style), encoding="utf-8"
                )
        # doc_id must follow the loader's ordering: L1 first, then L2, then L3,
        index = 0
        for label in config.LABELS:
            level_dir = base / config.LABEL_NAMES[label]
            for _ in sorted(p.name for p in level_dir.iterdir() if p.suffix == ".txt"):
                manifest.append(
                    {
                        "doc_id": f"{language}_{index:04d}",
                        "language": language,
                        "y_true": label,
                    }
                )
                index += 1
        print(f"  wrote {docs_per_level * 3:>4} documents -> {base}")

    return manifest


def write_mock_hybrid(
    manifest: list[dict], out_csv: Path, accuracy: float, seed: int
) -> None:
    """A fake hybrid prediction file, for exercising run_validation.py."""
    rng = random.Random(seed + 1)
    rows = ["doc_id,language,y_true,y_pred"]
    for record in manifest:
        truth = record["y_true"]
        if rng.random() < accuracy:
            prediction = truth
        elif rng.random() < 0.85:
            prediction = rng.choice([c for c in config.LABELS if abs(c - truth) == 1])
        else:
            distant = [c for c in config.LABELS if abs(c - truth) >= 2]
            prediction = rng.choice(distant) if distant else truth
        rows.append(f"{record['doc_id']},{record['language']},{truth},{prediction}")
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_csv.write_text("\n".join(rows), encoding="utf-8")
    print(f"  wrote mock hybrid predictions -> {out_csv}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="./corpora")
    parser.add_argument("--docs-per-level", type=int, default=40)
    parser.add_argument(
        "--label-noise",
        type=float,
        default=0.35,
        help="Share of documents written in a neighbouring level's style, so the "
        "classes overlap the way a real corpus does",
    )
    parser.add_argument("--emit-mock-hybrid", action="store_true")
    parser.add_argument("--hybrid-accuracy", type=float, default=0.78)
    parser.add_argument("--seed", type=int, default=config.SEED)
    args = parser.parse_args()

    out_root = Path(args.out)
    print(f"[synthetic] generating corpus under {out_root}")
    manifest = write_corpus(
        out_root, args.docs_per_level, args.seed, args.label_noise
    )

    if args.emit_mock_hybrid:
        write_mock_hybrid(
            manifest,
            Path("results/mock_hybrid_predictions.csv"),
            args.hybrid_accuracy,
            args.seed,
        )

    print(
        f"\n[synthetic] {len(manifest)} documents total. This is nonsense text for "
        "plumbing tests only;\n            any score it produces says nothing about "
        "Philippine readability."
    )


if __name__ == "__main__":
    main()
