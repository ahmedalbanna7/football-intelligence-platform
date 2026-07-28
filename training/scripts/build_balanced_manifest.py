from __future__ import annotations

import argparse
import json
from collections import Counter
from hashlib import sha256
from pathlib import Path

import yaml


IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a repeat-weighted YOLO train manifest for rare classes."
    )
    parser.add_argument("dataset", type=Path)
    parser.add_argument(
        "--repeat",
        action="append",
        default=[],
        metavar="CLASS=COUNT",
        help="Total repetitions for images containing CLASS.",
    )
    parser.add_argument("--config-name", default="data-balanced.yaml")
    parser.add_argument(
        "--max-empty-ratio",
        type=float,
        help="Keep at most this many empty images per positive image.",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def class_names(config: dict) -> list[str]:
    names = config.get("names", [])
    if isinstance(names, dict):
        return [str(names[index]) for index in sorted(names, key=lambda value: int(value))]
    return [str(name) for name in names]


def parse_repeats(values: list[str], names: list[str]) -> dict[int, int]:
    repeats: dict[int, int] = {}
    for value in values:
        name, separator, count_value = value.partition("=")
        if not separator or name not in names:
            raise ValueError(f"Invalid --repeat {value!r}; classes={names}")
        count = int(count_value)
        if count < 1 or count > 8:
            raise ValueError("Repeat counts must be between 1 and 8")
        repeats[names.index(name)] = count
    return repeats


def resolve_train(dataset: Path, configured: str) -> Path:
    path = Path(configured)
    candidates = [dataset / path]
    normalized = tuple(part for part in path.parts if part not in (".", ".."))
    if normalized:
        candidates.append(dataset.joinpath(*normalized))
    for candidate in candidates:
        if candidate.resolve().exists():
            return candidate.resolve()
    raise FileNotFoundError(f"Cannot resolve train split {configured!r}")


def label_path(image: Path) -> Path:
    parts = list(image.parts)
    parts[-2] = "labels"
    return Path(*parts).with_suffix(".txt")


def main() -> None:
    args = parse_args()
    dataset = args.dataset.resolve()
    config_path = dataset / "data.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    names = class_names(config)
    repeats = parse_repeats(args.repeat, names)
    train_root = resolve_train(dataset, str(config["train"]))
    images = sorted(
        path
        for path in train_root.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )
    image_classes: dict[Path, list[int]] = {}
    for image in images:
        image_classes[image] = [
            int(line.split()[0])
            for line in label_path(image).read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    positive_images = [
        image
        for image, classes in image_classes.items()
        if classes
    ]
    empty_images = [
        image
        for image, classes in image_classes.items()
        if not classes
    ]
    selected_empty_images = empty_images
    if args.max_empty_ratio is not None:
        if args.max_empty_ratio < 0:
            raise ValueError("--max-empty-ratio cannot be negative")
        empty_limit = int(round(len(positive_images) * args.max_empty_ratio))
        selected_empty_images = sorted(
            empty_images,
            key=lambda image: sha256(
                f"{args.seed}:{image.relative_to(dataset).as_posix()}".encode("utf-8")
            ).hexdigest(),
        )[:empty_limit]
    selected_images = sorted(positive_images + selected_empty_images)

    rows: list[str] = []
    source_instances: Counter[int] = Counter()
    weighted_instances: Counter[int] = Counter()
    repetition_counts: Counter[int] = Counter()
    for image in selected_images:
        classes = image_classes[image]
        source_instances.update(classes)
        repeat = max((repeats.get(class_id, 1) for class_id in set(classes)), default=1)
        repetition_counts[repeat] += 1
        relative = f"./{image.relative_to(dataset).as_posix()}"
        rows.extend(relative for _ in range(repeat))
        for _ in range(repeat):
            weighted_instances.update(classes)

    manifest_path = dataset / "train-balanced.txt"
    manifest_path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    balanced_config = dict(config)
    balanced_config["train"] = manifest_path.name
    output_config = dataset / args.config_name
    output_config.write_text(
        yaml.safe_dump(balanced_config, sort_keys=False),
        encoding="utf-8",
    )
    report = {
        "dataset": str(dataset),
        "source_images": len(images),
        "source_positive_images": len(positive_images),
        "source_empty_images": len(empty_images),
        "selected_images": len(selected_images),
        "selected_empty_images": len(selected_empty_images),
        "max_empty_ratio": args.max_empty_ratio,
        "weighted_entries": len(rows),
        "repeat_policy": {
            names[class_id]: count for class_id, count in sorted(repeats.items())
        },
        "images_by_repeat": dict(sorted(repetition_counts.items())),
        "source_instances": {
            name: source_instances[class_id] for class_id, name in enumerate(names)
        },
        "weighted_instances": {
            name: weighted_instances[class_id] for class_id, name in enumerate(names)
        },
        "manifest": str(manifest_path),
        "config": str(output_config),
    }
    (dataset / "balance-report.json").write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
