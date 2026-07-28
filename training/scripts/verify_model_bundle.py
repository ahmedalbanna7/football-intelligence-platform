from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify all versioned weights used by a model-registry bundle."
    )
    parser.add_argument(
        "--registry",
        type=Path,
        default=ROOT / "training" / "config" / "model-registry.yaml",
    )
    parser.add_argument("--bundle")
    return parser.parse_args()


def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def weight_entries(value: Any, location: str = ""):
    if isinstance(value, dict):
        if value.get("path") and value.get("sha256"):
            yield location, value
        for key, nested in value.items():
            yield from weight_entries(nested, f"{location}.{key}".strip("."))
    elif isinstance(value, list):
        for index, nested in enumerate(value):
            yield from weight_entries(nested, f"{location}[{index}]")


def main() -> int:
    args = parse_args()
    registry = yaml.safe_load(args.registry.read_text(encoding="utf-8"))
    bundle_name = args.bundle or registry["active_bundle"]
    bundles = registry["bundles"]
    pending = [bundle_name]
    visited: set[str] = set()
    failures: list[str] = []

    while pending:
        current = pending.pop()
        if current in visited:
            continue
        visited.add(current)
        bundle = bundles[current]
        fallback = bundle.get("fallback_bundle")
        if fallback:
            pending.append(str(fallback))
        for location, entry in weight_entries(bundle, current):
            path = ROOT / str(entry["path"])
            if not path.is_file():
                failures.append(f"{location}: missing {path}")
                continue
            actual = digest(path)
            expected = str(entry["sha256"]).lower()
            if actual != expected:
                failures.append(
                    f"{location}: SHA256 mismatch ({actual} != {expected})"
                )
                continue
            print(f"OK {location}: {path.name}")

    if failures:
        for failure in failures:
            print(f"ERROR {failure}")
        return 1
    print(f"Verified bundle: {bundle_name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
