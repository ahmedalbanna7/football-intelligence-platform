from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil

import torch
import yaml


ROOT = Path(__file__).resolve().parents[2]
TRAINING_CACHE = ROOT / "training" / "cache"
TRAINING_CACHE.mkdir(parents=True, exist_ok=True)
ULTRALYTICS_CONFIG = TRAINING_CACHE / "ultralytics"
ULTRALYTICS_CONFIG.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("YOLO_CONFIG_DIR", str(ULTRALYTICS_CONFIG))
os.environ.setdefault("TORCH_HOME", str(TRAINING_CACHE / "torch"))

from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a versioned YOLO candidate.")
    parser.add_argument("--task", choices=("detect", "pose"), required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fraction", type=float, default=1.0)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def resolve_model(value: str) -> str:
    candidate = Path(value)
    if candidate.is_absolute():
        return str(candidate)
    project_candidate = ROOT / candidate
    return str(project_candidate) if project_candidate.exists() else value


def resolve_split_path(data: Path, configured: str) -> Path:
    path = Path(configured)
    if path.is_absolute():
        return path.resolve()
    candidates = [data.parent / path]
    normalized_parts = tuple(part for part in path.parts if part not in (".", ".."))
    if normalized_parts:
        candidates.append(data.parent.joinpath(*normalized_parts))
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved.exists():
            return resolved
    raise FileNotFoundError(
        f"Dataset split does not exist. Configured={configured!r}, "
        f"checked={[str(candidate.resolve()) for candidate in candidates]}"
    )


def normalized_data_config(data: Path, name: str) -> Path:
    config = yaml.safe_load(data.read_text(encoding="utf-8")) or {}
    config.pop("path", None)
    for split in ("train", "val", "test"):
        configured = config.get(split)
        if configured is None:
            continue
        if isinstance(configured, list):
            config[split] = [
                str(resolve_split_path(data, str(item))) for item in configured
            ]
        else:
            config[split] = str(resolve_split_path(data, str(configured)))

    cache_dir = ROOT / "training" / "cache" / "dataset-configs"
    cache_dir.mkdir(parents=True, exist_ok=True)
    normalized = cache_dir / f"{name}.yaml"
    normalized.write_text(
        yaml.safe_dump(config, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    return normalized


def main() -> int:
    args = parse_args()
    data = args.data.resolve()
    if not data.exists():
        raise FileNotFoundError(f"Dataset config does not exist: {data}")
    if args.device != "cpu" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Use --device cpu only for smoke tests.")
    if not 0.0 < args.fraction <= 1.0:
        raise ValueError("--fraction must be greater than 0 and at most 1")

    project = ROOT / "training" / "runs"
    run_dir = project / args.name
    if run_dir.exists() and not args.resume:
        raise FileExistsError(
            f"{run_dir} already exists. Use a new versioned name or --resume."
        )

    normalized_data = normalized_data_config(data, args.name)
    model = YOLO(resolve_model(args.model), task=args.task)
    results = model.train(
        data=str(normalized_data),
        project=str(project),
        name=args.name,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        patience=args.patience,
        seed=args.seed,
        deterministic=True,
        amp=True,
        cache="disk",
        cos_lr=True,
        close_mosaic=10,
        plots=True,
        save=True,
        save_period=10,
        fraction=args.fraction,
        resume=args.resume,
    )

    best = run_dir / "weights" / "best.pt"
    if not best.exists():
        raise RuntimeError(f"Training completed without a best checkpoint: {results}")
    candidate = run_dir / f"{args.name}.pt"
    shutil.copy2(best, candidate)
    print(f"Candidate: {candidate}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
