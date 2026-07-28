from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
TRAINING_CACHE = ROOT / "training" / "cache"
TRAINING_CACHE.mkdir(parents=True, exist_ok=True)
ULTRALYTICS_CONFIG = TRAINING_CACHE / "ultralytics"
ULTRALYTICS_CONFIG.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("YOLO_CONFIG_DIR", str(ULTRALYTICS_CONFIG))
os.environ.setdefault("TORCH_HOME", str(TRAINING_CACHE / "torch"))

from ultralytics import YOLO


PERSON_CLASSES = {"person", "player", "goalkeeper", "referee"}
BALL_CLASSES = {"ball", "sports ball"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare YOLO weights on sampled video frames.")
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--frames", type=int, default=1000)
    parser.add_argument("--stride", type=int, default=10)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--device", default="0")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def green_foot_ratio(frame: np.ndarray, bbox: np.ndarray) -> float:
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    green = cv2.inRange(hsv, (24, 22, 20), (108, 255, 255))
    height, width = green.shape[:2]
    x1, _, x2, y2 = [float(value) for value in bbox]
    foot_x = int(round((x1 + x2) / 2))
    foot_y = int(round(y2))
    radius = max(4, int(round((x2 - x1) * 0.28)))
    left, right = max(0, foot_x - radius), min(width, foot_x + radius + 1)
    top, bottom = max(0, foot_y - radius), min(height, foot_y + radius + 1)
    if left >= right or top >= bottom:
        return 0.0
    return float(np.mean(green[top:bottom, left:right] > 0))


def sample_frames(video: Path, start_frame: int, count: int, stride: int):
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise ValueError(f"Could not open {video}")
    capture.set(cv2.CAP_PROP_POS_FRAMES, max(0, start_frame))
    for offset in range(count):
        ok, frame = capture.read()
        if not ok:
            break
        if offset % stride == 0:
            yield start_frame + offset, frame
    capture.release()


def evaluate_model(
    model_path: str,
    frames: list[tuple[int, np.ndarray]],
    imgsz: int,
    device: str,
) -> dict:
    model = YOLO(model_path)
    model.predict(
        frames[0][1],
        imgsz=imgsz,
        device=device,
        conf=0.15,
        max_det=100,
        verbose=False,
    )
    counts: Counter[str] = Counter()
    on_pitch_people = 0
    outside_people = 0
    confidences: list[float] = []
    latencies: list[float] = []
    per_frame: list[dict] = []
    for frame_index, frame in frames:
        started = perf_counter()
        result = model.predict(
            frame,
            imgsz=imgsz,
            device=device,
            conf=0.15,
            max_det=100,
            verbose=False,
        )[0]
        latencies.append((perf_counter() - started) * 1000)
        frame_counts: Counter[str] = Counter()
        boxes = result.boxes
        if boxes is not None and boxes.xyxy is not None:
            xyxy = boxes.xyxy.cpu().numpy()
            class_ids = boxes.cls.cpu().numpy()
            scores = boxes.conf.cpu().numpy()
            for bbox, class_id, score in zip(xyxy, class_ids, scores, strict=True):
                name = str(result.names.get(int(class_id), int(class_id))).lower()
                frame_counts[name] += 1
                counts[name] += 1
                confidences.append(float(score))
                if name in PERSON_CLASSES:
                    if green_foot_ratio(frame, bbox) >= 0.12:
                        on_pitch_people += 1
                    else:
                        outside_people += 1
        per_frame.append({"frame": frame_index, "counts": dict(frame_counts)})
    return {
        "model": model_path,
        "sampled_frames": len(frames),
        "counts": dict(counts),
        "on_pitch_person_observations": on_pitch_people,
        "outside_pitch_person_observations": outside_people,
        "average_confidence": round(float(np.mean(confidences)), 5) if confidences else None,
        "average_inference_ms": round(float(np.mean(latencies)), 2) if latencies else None,
        "median_inference_ms": round(float(np.median(latencies)), 2) if latencies else None,
        "p95_inference_ms": round(float(np.percentile(latencies, 95)), 2) if latencies else None,
        "per_frame": per_frame,
    }


def main() -> int:
    args = parse_args()
    frames = list(sample_frames(args.video, args.start_frame, args.frames, args.stride))
    if not frames:
        raise ValueError("No frames were sampled")
    report = {
        "video": str(args.video.resolve()),
        "start_frame": args.start_frame,
        "source_frames_scanned": args.frames,
        "stride": args.stride,
        "models": [
            evaluate_model(model, frames, args.imgsz, args.device)
            for model in args.models
        ],
    }
    rendered = json.dumps(report, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
        print(args.output)
    else:
        print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
