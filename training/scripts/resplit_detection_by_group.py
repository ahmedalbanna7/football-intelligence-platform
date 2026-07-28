from __future__ import annotations

import argparse
import itertools
import json
import random
import re
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import yaml


IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".webp"}
DEFAULT_SOURCE_PATTERN = r"^(?P<group>.+)_mp4-\d+_jpg$"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create leakage-resistant YOLO splits grouped by source video."
    )
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--val-groups", type=int, default=2)
    parser.add_argument("--test-groups", type=int, default=2)
    parser.add_argument("--group-pattern", default=DEFAULT_SOURCE_PATTERN)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def resolve_split(dataset: Path, configured: str) -> Path:
    path = Path(configured)
    candidates = [dataset / path]
    normalized = tuple(part for part in path.parts if part not in (".", ".."))
    if normalized:
        candidates.append(dataset.joinpath(*normalized))
    for candidate in candidates:
        if candidate.resolve().exists():
            return candidate.resolve()
    raise FileNotFoundError(f"Cannot resolve split {configured!r}")


def label_path(image_path: Path) -> Path:
    parts = list(image_path.parts)
    parts[-2] = "labels"
    return Path(*parts).with_suffix(".txt")


def source_group(path: Path, pattern: re.Pattern[str]) -> str:
    source_name = path.stem.split(".rf.", maxsplit=1)[0]
    match = pattern.match(source_name)
    if match is not None:
        return match.group("group")
    return source_name


def label_count(path: Path) -> int:
    return sum(
        1
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )


def choose_eval_groups(
    group_stats: dict[str, tuple[int, int]],
    val_count: int,
    test_count: int,
    seed: int,
) -> tuple[set[str], set[str]]:
    groups = sorted(group_stats)
    if val_count + test_count >= len(groups):
        raise ValueError("Evaluation groups must leave at least one training group")

    total_images = sum(group_stats[group][0] for group in groups)
    total_labels = sum(group_stats[group][1] for group in groups)
    val_target = val_count / len(groups)
    test_target = test_count / len(groups)
    def assignment_score(
        val_groups_tuple: tuple[str, ...],
        test_groups_tuple: tuple[str, ...],
    ) -> float:
        val_images = sum(group_stats[group][0] for group in val_groups_tuple)
        val_labels = sum(group_stats[group][1] for group in val_groups_tuple)
        test_images = sum(group_stats[group][0] for group in test_groups_tuple)
        test_labels = sum(group_stats[group][1] for group in test_groups_tuple)
        score = (
            (val_images / total_images - val_target) ** 2
            + (test_images / total_images - test_target) ** 2
            + (val_labels / max(total_labels, 1) - val_target) ** 2
            + (test_labels / max(total_labels, 1) - test_target) ** 2
        )
        return score + (1.0 if val_labels == 0 or test_labels == 0 else 0.0)

    candidates: list[tuple[tuple[str, ...], tuple[str, ...]]] = []
    if len(groups) <= 20:
        for val_groups_tuple in itertools.combinations(groups, val_count):
            remaining = [group for group in groups if group not in val_groups_tuple]
            candidates.extend(
                (val_groups_tuple, test_groups_tuple)
                for test_groups_tuple in itertools.combinations(remaining, test_count)
            )
    else:
        generator = random.Random(seed)
        seen: set[tuple[tuple[str, ...], tuple[str, ...]]] = set()
        for _ in range(50_000):
            shuffled = generator.sample(groups, len(groups))
            candidate = (
                tuple(sorted(shuffled[:val_count])),
                tuple(sorted(shuffled[val_count : val_count + test_count])),
            )
            if candidate not in seen:
                seen.add(candidate)
                candidates.append(candidate)

    best_score = float("inf")
    best: tuple[set[str], set[str]] | None = None
    best_key: tuple[tuple[str, ...], tuple[str, ...]] | None = None
    for val_groups_tuple, test_groups_tuple in candidates:
        score = assignment_score(val_groups_tuple, test_groups_tuple)
        key = (val_groups_tuple, test_groups_tuple)
        if score < best_score or (
            abs(score - best_score) < 1e-12
            and (best_key is None or key < best_key)
        ):
            best_score = score
            best_key = key
            best = (set(val_groups_tuple), set(test_groups_tuple))

    if best is None:
        raise RuntimeError("Unable to choose grouped validation/test splits")
    return best


def main() -> None:
    args = parse_args()
    source = args.source.resolve()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"Output already exists: {output}")

    config = yaml.safe_load((source / "data.yaml").read_text(encoding="utf-8"))
    group_pattern = re.compile(args.group_pattern)
    if "group" not in group_pattern.groupindex:
        raise ValueError("--group-pattern must define a named 'group' capture")
    records: list[tuple[Path, Path, str, int]] = []
    group_stats: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for split in ("train", "val", "test"):
        configured = config.get(split)
        if configured is None:
            continue
        split_root = resolve_split(source, str(configured))
        for image in sorted(
            path
            for path in split_root.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        ):
            label = label_path(image)
            labels = label_count(label)
            group = source_group(image, group_pattern)
            records.append((image, label, group, labels))
            group_stats[group][0] += 1
            group_stats[group][1] += labels

    frozen_stats = {
        group: (values[0], values[1]) for group, values in group_stats.items()
    }
    val_groups, test_groups = choose_eval_groups(
        frozen_stats,
        args.val_groups,
        args.test_groups,
        args.seed,
    )
    assignments = {
        group: (
            "valid"
            if group in val_groups
            else "test"
            if group in test_groups
            else "train"
        )
        for group in frozen_stats
    }

    output_counts: dict[str, Counter[str]] = defaultdict(Counter)
    for image, label, group, labels in records:
        split = assignments[group]
        images_root = output / split / "images"
        labels_root = output / split / "labels"
        images_root.mkdir(parents=True, exist_ok=True)
        labels_root.mkdir(parents=True, exist_ok=True)
        output_image = images_root / image.name
        output_label = labels_root / label.name
        if output_image.exists() or output_label.exists():
            raise FileExistsError(f"Duplicate output name: {image.name}")
        shutil.copy2(image, output_image)
        shutil.copy2(label, output_label)
        output_counts[split]["images"] += 1
        output_counts[split]["instances"] += labels
        output_counts[split]["groups"] = len(
            {candidate for candidate, assigned in assignments.items() if assigned == split}
        )

    output_config = dict(config)
    output_config.pop("path", None)
    output_config.update(
        {
            "train": "train/images",
            "val": "valid/images",
            "test": "test/images",
        }
    )
    (output / "data.yaml").write_text(
        yaml.safe_dump(output_config, sort_keys=False),
        encoding="utf-8",
    )
    report = {
        "source": str(source),
        "output": str(output),
        "splits": {
            split: dict(counts) for split, counts in sorted(output_counts.items())
        },
        "group_assignments": dict(sorted(assignments.items())),
    }
    (output / "split-report.json").write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
