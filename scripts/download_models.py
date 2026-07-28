#!/usr/bin/env python3
"""Download and verify the versioned production model bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = REPOSITORY_ROOT / "config" / "model-bundle-v2.json"
CHUNK_SIZE = 1024 * 1024
DOWNLOAD_ATTEMPTS = 3


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(CHUNK_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_is_valid(path: Path, expected_bytes: int, expected_sha256: str) -> bool:
    return (
        path.is_file()
        and path.stat().st_size == expected_bytes
        and sha256_file(path) == expected_sha256
    )


def download_once(url: str, temporary_path: Path, expected_bytes: int) -> None:
    request = Request(
        url,
        headers={"User-Agent": "football-intelligence-platform-model-setup/2.0"},
    )
    downloaded = 0
    last_percent = -1

    with urlopen(request, timeout=60) as response:
        with temporary_path.open("wb") as output:
            while True:
                chunk = response.read(CHUNK_SIZE)
                if not chunk:
                    break
                output.write(chunk)
                downloaded += len(chunk)
                percent = int(downloaded * 100 / expected_bytes)
                if percent >= last_percent + 10 or downloaded == expected_bytes:
                    print(f"  {min(percent, 100):3d}% ({downloaded}/{expected_bytes} bytes)")
                    last_percent = percent


def install_model(
    model: dict[str, Any],
    destination: Path,
    force: bool,
) -> None:
    name = str(model["name"])
    target = destination / name
    temporary = target.with_suffix(target.suffix + ".download")
    expected_bytes = int(model["bytes"])
    expected_sha256 = str(model["sha256"]).lower()

    if not force and file_is_valid(target, expected_bytes, expected_sha256):
        print(f"[ok] {name}")
        return

    temporary.unlink(missing_ok=True)
    print(f"[download] {name}")
    for attempt in range(1, DOWNLOAD_ATTEMPTS + 1):
        try:
            download_once(str(model["url"]), temporary, expected_bytes)
            if not file_is_valid(temporary, expected_bytes, expected_sha256):
                raise ValueError("downloaded file failed size or SHA-256 validation")
            temporary.replace(target)
            print(f"[installed] {name}")
            return
        except (HTTPError, URLError, TimeoutError, ValueError) as exc:
            temporary.unlink(missing_ok=True)
            if attempt == DOWNLOAD_ATTEMPTS:
                raise RuntimeError(
                    f"Unable to install {name} after {DOWNLOAD_ATTEMPTS} attempts: {exc}"
                ) from exc
            print(f"  attempt {attempt} failed: {exc}; retrying...")
            time.sleep(attempt * 2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download the verified football model bundle.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST,
        help="Path to the model bundle JSON manifest.",
    )
    parser.add_argument(
        "--destination",
        type=Path,
        help="Override the manifest destination directory.",
    )
    parser.add_argument(
        "--model",
        action="append",
        default=[],
        help="Download only the named model. May be repeated.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Redownload models even when the checksum is already valid.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest_path = args.manifest.resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    destination = (
        args.destination.resolve()
        if args.destination
        else REPOSITORY_ROOT / str(manifest["destination"])
    )
    destination.mkdir(parents=True, exist_ok=True)

    selected_names = set(args.model)
    models = [
        model
        for model in manifest["models"]
        if not selected_names or model["name"] in selected_names
    ]
    missing_names = selected_names - {str(model["name"]) for model in models}
    if missing_names:
        print(
            f"Unknown model name(s): {', '.join(sorted(missing_names))}",
            file=sys.stderr,
        )
        return 2

    print(
        f"Installing {manifest['bundle']} v{manifest['version']} "
        f"to {destination}"
    )
    try:
        for model in models:
            install_model(model, destination, args.force)
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    print(f"Verified {len(models)} model file(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
