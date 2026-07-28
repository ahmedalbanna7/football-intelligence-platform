from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
import yaml


IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit YOLO detection labels, geometry, and split leakage."
    )
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--duplicate-iou", type=float, default=0.85)
    parser.add_argument("--near-threshold", type=int, default=4)
    return parser.parse_args()


def resolve_split(dataset: Path, configured: str) -> Path:
    path = Path(configured)
    if path.is_absolute():
        return path.resolve()
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


def read_boxes(path: Path) -> list[tuple[int, float, float, float, float]]:
    boxes: list[tuple[int, float, float, float, float]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        class_id, x, y, width, height = line.split()[:5]
        boxes.append((int(class_id), float(x), float(y), float(width), float(height)))
    return boxes


def box_iou(
    first: tuple[int, float, float, float, float],
    second: tuple[int, float, float, float, float],
) -> float:
    _, x1, y1, w1, h1 = first
    _, x2, y2, w2, h2 = second
    first_xyxy = (x1 - w1 / 2, y1 - h1 / 2, x1 + w1 / 2, y1 + h1 / 2)
    second_xyxy = (x2 - w2 / 2, y2 - h2 / 2, x2 + w2 / 2, y2 + h2 / 2)
    intersection_width = max(
        0.0, min(first_xyxy[2], second_xyxy[2]) - max(first_xyxy[0], second_xyxy[0])
    )
    intersection_height = max(
        0.0, min(first_xyxy[3], second_xyxy[3]) - max(first_xyxy[1], second_xyxy[1])
    )
    intersection = intersection_width * intersection_height
    union = w1 * h1 + w2 * h2 - intersection
    return intersection / union if union > 1e-9 else 0.0


def difference_hash(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Cannot decode image: {path}")
    resized = cv2.resize(image, (17, 16), interpolation=cv2.INTER_AREA)
    return np.packbits((resized[:, 1:] > resized[:, :-1]).reshape(-1))


def hamming(first: np.ndarray, second: np.ndarray) -> int:
    return int(np.unpackbits(np.bitwise_xor(first, second)).sum())


def main() -> None:
    args = parse_args()
    dataset = args.dataset.resolve()
    config = yaml.safe_load((dataset / "data.yaml").read_text(encoding="utf-8"))
    names = config["names"]
    if isinstance(names, dict):
        names = [names[index] for index in sorted(names, key=lambda value: int(value))]

    class_counts: Counter[int] = Counter()
    areas: dict[int, list[float]] = defaultdict(list)
    duplicate_pairs: list[dict[str, object]] = []
    outside_boxes: list[dict[str, object]] = []
    split_images: dict[str, list[Path]] = {}

    for split_name in ("train", "val", "test"):
        configured = config.get(split_name)
        if configured is None:
            continue
        split_root = resolve_split(dataset, str(configured))
        images = sorted(
            path
            for path in split_root.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        )
        split_images[split_name] = images
        for image in images:
            boxes = read_boxes(label_path(image))
            for box_index, box in enumerate(boxes):
                class_id, x, y, width, height = box
                class_counts[class_id] += 1
                areas[class_id].append(width * height)
                if (
                    x - width / 2 < -1e-4
                    or y - height / 2 < -1e-4
                    or x + width / 2 > 1.0001
                    or y + height / 2 > 1.0001
                ):
                    outside_boxes.append(
                        {
                            "image": str(image),
                            "box_index": box_index,
                            "box": box,
                        }
                    )
            for first_index, first in enumerate(boxes):
                for second_index in range(first_index + 1, len(boxes)):
                    second = boxes[second_index]
                    if first[0] != second[0]:
                        continue
                    overlap = box_iou(first, second)
                    if overlap >= args.duplicate_iou:
                        duplicate_pairs.append(
                            {
                                "image": str(image),
                                "class": names[first[0]],
                                "first": first_index,
                                "second": second_index,
                                "iou": round(overlap, 4),
                            }
                        )

    hashes = {
        split: [(path, difference_hash(path)) for path in images]
        for split, images in split_images.items()
    }
    leakage: list[dict[str, object]] = []
    split_names = list(hashes)
    for first_index, first_split in enumerate(split_names):
        for second_split in split_names[first_index + 1 :]:
            for first_path, first_hash in hashes[first_split]:
                distances = [
                    (hamming(first_hash, second_hash), second_path)
                    for second_path, second_hash in hashes[second_split]
                ]
                if not distances:
                    continue
                distance, second_path = min(distances, key=lambda item: item[0])
                if distance <= args.near_threshold:
                    leakage.append(
                        {
                            "first_split": first_split,
                            "first": str(first_path),
                            "second_split": second_split,
                            "second": str(second_path),
                            "distance": distance,
                        }
                    )

    area_report = {}
    for class_id, name in enumerate(names):
        values = np.asarray(areas[class_id], dtype=np.float64)
        area_report[name] = {
            "instances": class_counts[class_id],
            "median_area": round(float(np.median(values)), 8) if values.size else None,
            "p05_area": round(float(np.percentile(values, 5)), 8) if values.size else None,
            "tiny_under_0_001": int((values < 0.001).sum()) if values.size else 0,
        }

    report = {
        "dataset": str(dataset),
        "images": {split: len(images) for split, images in split_images.items()},
        "classes": area_report,
        "outside_boxes": {
            "count": len(outside_boxes),
            "samples": outside_boxes[:20],
        },
        "possible_duplicate_boxes": {
            "threshold": args.duplicate_iou,
            "count": len(duplicate_pairs),
            "samples": duplicate_pairs[:20],
        },
        "possible_split_leakage": {
            "dhash_threshold": args.near_threshold,
            "count": len(leakage),
            "samples": leakage[:20],
        },
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
