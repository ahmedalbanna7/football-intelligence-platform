from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter
from pathlib import Path

import cv2
import yaml


IMAGE_SUFFIXES = {".jpeg", ".jpg", ".png", ".webp"}
OUTPUT_NAMES = ["ball", "goalkeeper", "player", "referee", "other"]
PRIMARY_CLASS_MAP = {0: 0, 1: 1, 2: 2, 3: 3}
AUXILIARY_CLASS_MAP = {1: 4}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge complementary labels without duplicating near-identical images."
    )
    parser.add_argument("--primary", type=Path, required=True)
    parser.add_argument("--auxiliary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def source_key(path: Path) -> str:
    return path.name.split(".rf.", maxsplit=1)[0]


def image_files(split_root: Path) -> list[Path]:
    images_root = split_root / "images"
    return sorted(
        path
        for path in images_root.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )


def label_for(image_path: Path) -> Path:
    return image_path.parent.parent / "labels" / f"{image_path.stem}.txt"


def read_labels(
    path: Path,
    class_map: dict[int, int],
    *,
    ignore_unmapped: bool = False,
) -> list[list[float]]:
    labels: list[list[float]] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        if not line.strip():
            continue
        values = line.split()
        if len(values) != 5:
            raise ValueError(f"{path}:{line_number}: expected detection label")
        source_class = int(values[0])
        if source_class not in class_map:
            if ignore_unmapped:
                continue
            raise ValueError(f"{path}:{line_number}: unmapped class {source_class}")
        labels.append([float(class_map[source_class]), *map(float, values[1:])])
    return labels


def box_iou(first: list[float], second: list[float]) -> float:
    _, first_x, first_y, first_w, first_h = first
    _, second_x, second_y, second_w, second_h = second
    first_xyxy = (
        first_x - first_w / 2,
        first_y - first_h / 2,
        first_x + first_w / 2,
        first_y + first_h / 2,
    )
    second_xyxy = (
        second_x - second_w / 2,
        second_y - second_h / 2,
        second_x + second_w / 2,
        second_y + second_h / 2,
    )
    intersection_width = max(
        0.0,
        min(first_xyxy[2], second_xyxy[2])
        - max(first_xyxy[0], second_xyxy[0]),
    )
    intersection_height = max(
        0.0,
        min(first_xyxy[3], second_xyxy[3])
        - max(first_xyxy[1], second_xyxy[1]),
    )
    intersection = intersection_width * intersection_height
    union = first_w * first_h + second_w * second_h - intersection
    return intersection / union if union > 1e-9 else 0.0


def append_unique(
    merged: list[list[float]],
    candidates: list[list[float]],
) -> tuple[int, int]:
    added = 0
    duplicates = 0
    for candidate in candidates:
        if any(
            int(existing[0]) == int(candidate[0])
            and box_iou(existing, candidate) >= 0.88
            for existing in merged
        ):
            duplicates += 1
            continue
        merged.append(candidate)
        added += 1
    return added, duplicates


def write_labels(path: Path, labels: list[list[float]]) -> None:
    lines = [
        f"{int(label[0])} " + " ".join(f"{value:.6f}" for value in label[1:])
        for label in labels
    ]
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def main() -> None:
    args = parse_args()
    primary = args.primary.resolve()
    auxiliary = args.auxiliary.resolve()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"Output already exists: {output}")

    report: dict[str, object] = {
        "classes": OUTPUT_NAMES,
        "splits": {},
    }
    class_counts: Counter[int] = Counter()
    unmatched_auxiliary: set[str] = set()

    for split in ("train", "valid", "test"):
        primary_images = image_files(primary / split)
        auxiliary_images = {
            source_key(path): path
            for path in image_files(auxiliary / split)
        }
        unmatched_auxiliary.update(auxiliary_images)
        output_images = output / split / "images"
        output_labels = output / split / "labels"
        output_images.mkdir(parents=True, exist_ok=False)
        output_labels.mkdir(parents=True, exist_ok=False)

        matched = 0
        auxiliary_added = 0
        duplicate_labels = 0
        normalized_geometry_matches = 0
        for primary_image in primary_images:
            key = source_key(primary_image)
            auxiliary_image = auxiliary_images.get(key)
            merged = read_labels(label_for(primary_image), PRIMARY_CLASS_MAP)
            if auxiliary_image is not None:
                primary_shape = cv2.imread(str(primary_image)).shape[:2]
                auxiliary_shape = cv2.imread(str(auxiliary_image)).shape[:2]
                if primary_shape != auxiliary_shape:
                    primary_ratio = primary_shape[1] / primary_shape[0]
                    auxiliary_ratio = auxiliary_shape[1] / auxiliary_shape[0]
                    if abs(primary_ratio - auxiliary_ratio) > 1e-4:
                        raise ValueError(
                            f"Image aspect-ratio mismatch for {key}: "
                            f"{primary_shape} != {auxiliary_shape}"
                        )
                    normalized_geometry_matches += 1
                added, duplicates = append_unique(
                    merged,
                    read_labels(
                        label_for(auxiliary_image),
                        AUXILIARY_CLASS_MAP,
                        ignore_unmapped=True,
                    ),
                )
                auxiliary_added += added
                duplicate_labels += duplicates
                matched += 1
                unmatched_auxiliary.discard(key)

            output_image = output_images / primary_image.name
            shutil.copy2(primary_image, output_image)
            write_labels(output_labels / f"{primary_image.stem}.txt", merged)
            class_counts.update(int(label[0]) for label in merged)

        report["splits"][split] = {
            "images": len(primary_images),
            "matched_auxiliary_images": matched,
            "auxiliary_labels_added": auxiliary_added,
            "duplicate_labels_skipped": duplicate_labels,
            "normalized_geometry_matches": normalized_geometry_matches,
        }

    data_config = {
        "train": "train/images",
        "val": "valid/images",
        "test": "test/images",
        "nc": len(OUTPUT_NAMES),
        "names": OUTPUT_NAMES,
    }
    (output / "data.yaml").write_text(
        yaml.safe_dump(data_config, sort_keys=False),
        encoding="utf-8",
    )
    report["instances"] = {
        OUTPUT_NAMES[class_id]: class_counts[class_id]
        for class_id in range(len(OUTPUT_NAMES))
    }
    report["unmatched_auxiliary_images"] = sorted(unmatched_auxiliary)
    (output / "merge-report.json").write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
