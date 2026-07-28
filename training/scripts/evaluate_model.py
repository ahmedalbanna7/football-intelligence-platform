from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
TRAINING_CACHE = ROOT / "training" / "cache"
ULTRALYTICS_CONFIG = TRAINING_CACHE / "ultralytics"
ULTRALYTICS_CONFIG.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("YOLO_CONFIG_DIR", str(ULTRALYTICS_CONFIG))
os.environ.setdefault("TORCH_HOME", str(TRAINING_CACHE / "torch"))

from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate one versioned YOLO model on an isolated dataset split."
    )
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--task", choices=("detect", "pose"), required=True)
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--device", default="0")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def metric_summary(metric: Any | None, names: dict[int, str]) -> dict[str, Any] | None:
    if metric is None:
        return None
    raw_class_maps = getattr(metric, "maps", None)
    class_maps = list(raw_class_maps) if raw_class_maps is not None else []
    return {
        "precision": round(float(metric.mp), 6),
        "recall": round(float(metric.mr), 6),
        "map50": round(float(metric.map50), 6),
        "map50_95": round(float(metric.map), 6),
        "map50_95_by_class": {
            str(names.get(index, index)): round(float(value), 6)
            for index, value in enumerate(class_maps)
        },
    }


def main() -> int:
    args = parse_args()
    model_path = args.model.resolve()
    data_path = args.data.resolve()
    if not model_path.is_file():
        raise FileNotFoundError(model_path)
    if not data_path.is_file():
        raise FileNotFoundError(data_path)

    model = YOLO(str(model_path), task=args.task)
    results = model.val(
        data=str(data_path),
        split=args.split,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=2,
        plots=False,
        project=str(ROOT / "training" / "runs" / "evaluations"),
        name=args.name,
        verbose=True,
    )
    names = {
        int(index): str(name)
        for index, name in results.names.items()
    }
    report = {
        "name": args.name,
        "model": str(model_path),
        "data": str(data_path),
        "task": args.task,
        "split": args.split,
        "image_size": args.imgsz,
        "batch": args.batch,
        "names": names,
        "box": metric_summary(getattr(results, "box", None), names),
        "pose": metric_summary(getattr(results, "pose", None), names),
        "speed_ms_per_image": {
            str(key): round(float(value), 4)
            for key, value in results.speed.items()
        },
        "save_dir": str(results.save_dir),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
