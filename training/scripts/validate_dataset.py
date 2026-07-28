from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
import sys

import yaml


IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate a Roboflow YOLO dataset.")
    parser.add_argument("dataset", type=Path, help="Extracted dataset directory.")
    parser.add_argument("--task", choices=("detect", "pose"), default="detect")
    return parser.parse_args()


def class_names(config: dict) -> list[str]:
    names = config.get("names", [])
    if isinstance(names, dict):
        return [str(names[index]) for index in sorted(names, key=lambda value: int(value))]
    return [str(name) for name in names]


def resolve_split(dataset: Path, config: dict, split: str) -> Path | None:
    configured = config.get(split)
    if configured is None:
        return None
    if isinstance(configured, list):
        raise ValueError(f"{split}: list-based split paths are not supported by this validator")
    path = Path(str(configured))
    if path.is_absolute():
        return path.resolve()

    candidates = [dataset / path]
    normalized_parts = tuple(part for part in path.parts if part not in (".", ".."))
    if normalized_parts:
        candidates.append(dataset.joinpath(*normalized_parts))
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved.exists():
            return resolved
    return candidates[0].resolve()


def label_path(image_path: Path) -> Path:
    parts = list(image_path.parts)
    for index in range(len(parts) - 1, -1, -1):
        if parts[index] == "images":
            parts[index] = "labels"
            return Path(*parts).with_suffix(".txt")
    return image_path.with_suffix(".txt")


def validate_line(
    line: str,
    file_path: Path,
    line_number: int,
    class_count: int,
    task: str,
    pose_columns: int | None,
) -> tuple[int, int | None]:
    values = line.split()
    if len(values) < 5:
        raise ValueError(f"{file_path}:{line_number}: expected at least 5 columns")
    class_id = int(values[0])
    if class_id < 0 or class_id >= class_count:
        raise ValueError(f"{file_path}:{line_number}: class {class_id} is out of range")
    coordinates = [float(value) for value in values[1:]]
    if any(value < 0.0 or value > 1.0 for value in coordinates[:4]):
        raise ValueError(f"{file_path}:{line_number}: box coordinates are not normalized")
    if coordinates[2] <= 0.0 or coordinates[3] <= 0.0:
        raise ValueError(f"{file_path}:{line_number}: box width/height must be positive")
    if task == "pose":
        current_columns = len(values)
        if current_columns <= 5 or (current_columns - 5) % 3 != 0:
            raise ValueError(
                f"{file_path}:{line_number}: pose labels must contain x/y/visibility triplets"
            )
        if pose_columns is not None and current_columns != pose_columns:
            raise ValueError(
                f"{file_path}:{line_number}: inconsistent pose column count "
                f"{current_columns}, expected {pose_columns}"
            )
        pose_columns = current_columns
        keypoints = coordinates[4:]
        for offset in range(0, len(keypoints), 3):
            x, y, visibility = keypoints[offset : offset + 3]
            if x < 0.0 or x > 1.0 or y < 0.0 or y > 1.0:
                raise ValueError(f"{file_path}:{line_number}: keypoint is not normalized")
            if visibility not in (0.0, 1.0, 2.0):
                raise ValueError(f"{file_path}:{line_number}: invalid visibility value")
    return class_id, pose_columns


def main() -> int:
    args = parse_args()
    dataset = args.dataset.resolve()
    data_yaml = dataset / "data.yaml"
    if not data_yaml.exists():
        print(f"Missing {data_yaml}", file=sys.stderr)
        return 2

    config = yaml.safe_load(data_yaml.read_text(encoding="utf-8")) or {}
    names = class_names(config)
    if not names:
        print("Dataset has no class names", file=sys.stderr)
        return 2

    total_images = 0
    empty_labels = 0
    missing_labels = 0
    instances: Counter[int] = Counter()
    pose_columns: int | None = None
    errors: list[str] = []
    split_report: dict[str, dict[str, int]] = {}

    for split in ("train", "val", "test"):
        try:
            split_path = resolve_split(dataset, config, split)
        except ValueError as exc:
            errors.append(str(exc))
            continue
        if split_path is None:
            continue
        if not split_path.exists():
            errors.append(f"{split}: missing directory {split_path}")
            continue
        images = sorted(
            path for path in split_path.rglob("*") if path.suffix.lower() in IMAGE_SUFFIXES
        )
        split_instances = 0
        for image_path in images:
            total_images += 1
            current_label = label_path(image_path)
            if not current_label.exists():
                missing_labels += 1
                continue
            lines = [
                line.strip()
                for line in current_label.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            if not lines:
                empty_labels += 1
                continue
            for line_number, line in enumerate(lines, start=1):
                try:
                    class_id, pose_columns = validate_line(
                        line,
                        current_label,
                        line_number,
                        len(names),
                        args.task,
                        pose_columns,
                    )
                    instances[class_id] += 1
                    split_instances += 1
                except (TypeError, ValueError) as exc:
                    errors.append(str(exc))
        split_report[split] = {"images": len(images), "instances": split_instances}

    print(f"Dataset: {dataset}")
    print(f"Task: {args.task}")
    print(f"Classes: {names}")
    print(f"Splits: {split_report}")
    print(f"Images: {total_images}")
    print(f"Missing labels: {missing_labels}")
    print(f"Empty labels: {empty_labels}")
    print("Instances:")
    for class_id, name in enumerate(names):
        print(f"  {class_id} {name}: {instances[class_id]}")
    if args.task == "pose" and pose_columns is not None:
        print(f"Pose keypoints: {(pose_columns - 5) // 3}")
    if errors:
        print(f"Validation errors: {len(errors)}", file=sys.stderr)
        for error in errors[:50]:
            print(f"  {error}", file=sys.stderr)
        return 1
    print("Validation: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
