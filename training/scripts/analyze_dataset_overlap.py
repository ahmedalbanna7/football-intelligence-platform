from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np


IMAGE_SUFFIXES = {".jpeg", ".jpg", ".png", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure exact and perceptual overlap between two image datasets."
    )
    parser.add_argument("first", type=Path)
    parser.add_argument("second", type=Path)
    parser.add_argument("--near-threshold", type=int, default=6)
    return parser.parse_args()


def images(root: Path) -> list[Path]:
    return sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )


def exact_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def difference_hash(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Cannot decode image: {path}")
    resized = cv2.resize(image, (17, 16), interpolation=cv2.INTER_AREA)
    bits = resized[:, 1:] > resized[:, :-1]
    return np.packbits(bits.reshape(-1)).astype(np.uint8)


def hamming_distances(query: np.ndarray, candidates: np.ndarray) -> np.ndarray:
    xor = np.bitwise_xor(candidates, query)
    return np.unpackbits(xor, axis=1).sum(axis=1)


def main() -> None:
    args = parse_args()
    first_images = images(args.first.resolve())
    second_images = images(args.second.resolve())
    if not first_images or not second_images:
        raise ValueError("Both datasets must contain images")

    first_exact = {exact_hash(path) for path in first_images}
    second_exact = {exact_hash(path) for path in second_images}
    second_perceptual = np.stack([difference_hash(path) for path in second_images])

    minimum_distances: list[int] = []
    nearest_pairs: list[dict[str, int | str]] = []
    for path in first_images:
        distances = hamming_distances(difference_hash(path), second_perceptual)
        nearest_index = int(np.argmin(distances))
        minimum = int(distances[nearest_index])
        minimum_distances.append(minimum)
        if minimum <= args.near_threshold and len(nearest_pairs) < 25:
            nearest_pairs.append(
                {
                    "first": str(path),
                    "second": str(second_images[nearest_index]),
                    "hamming_distance": minimum,
                }
            )

    report = {
        "first_images": len(first_images),
        "second_images": len(second_images),
        "exact_duplicates": len(first_exact & second_exact),
        "near_threshold": args.near_threshold,
        "near_duplicates": sum(
            distance <= args.near_threshold for distance in minimum_distances
        ),
        "minimum_distance": min(minimum_distances),
        "median_nearest_distance": float(np.median(minimum_distances)),
        "p95_nearest_distance": float(np.percentile(minimum_distances, 95)),
        "sample_pairs": nearest_pairs,
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
