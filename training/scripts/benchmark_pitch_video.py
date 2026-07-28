from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from time import perf_counter
from typing import Any

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
TRAINING_CACHE = ROOT / "training" / "cache"
ULTRALYTICS_CONFIG = TRAINING_CACHE / "ultralytics"
ULTRALYTICS_CONFIG.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("YOLO_CONFIG_DIR", str(ULTRALYTICS_CONFIG))
os.environ.setdefault("TORCH_HOME", str(TRAINING_CACHE / "torch"))

from ultralytics import YOLO


PITCH_LENGTH_CM = 10500.0
PITCH_WIDTH_CM = 6800.0
PENALTY_AREA_LENGTH_CM = 1650.0
PENALTY_AREA_WIDTH_CM = 4032.0
GOAL_AREA_LENGTH_CM = 550.0
GOAL_AREA_WIDTH_CM = 1832.0
PENALTY_SPOT_DISTANCE_CM = 1100.0
CENTER_CIRCLE_RADIUS_CM = 915.0


def pitch_vertices() -> np.ndarray:
    penalty_width = PENALTY_AREA_WIDTH_CM
    goal_width = GOAL_AREA_WIDTH_CM
    penalty_length = PENALTY_AREA_LENGTH_CM
    goal_length = GOAL_AREA_LENGTH_CM
    penalty_spot = PENALTY_SPOT_DISTANCE_CM
    center_circle = CENTER_CIRCLE_RADIUS_CM
    return np.array(
        [
            (0, 0),
            (0, (PITCH_WIDTH_CM - penalty_width) / 2),
            (0, (PITCH_WIDTH_CM - goal_width) / 2),
            (0, (PITCH_WIDTH_CM + goal_width) / 2),
            (0, (PITCH_WIDTH_CM + penalty_width) / 2),
            (0, PITCH_WIDTH_CM),
            (goal_length, (PITCH_WIDTH_CM - goal_width) / 2),
            (goal_length, (PITCH_WIDTH_CM + goal_width) / 2),
            (penalty_spot, PITCH_WIDTH_CM / 2),
            (penalty_length, (PITCH_WIDTH_CM - penalty_width) / 2),
            (penalty_length, (PITCH_WIDTH_CM - goal_width) / 2),
            (penalty_length, (PITCH_WIDTH_CM + goal_width) / 2),
            (penalty_length, (PITCH_WIDTH_CM + penalty_width) / 2),
            (PITCH_LENGTH_CM / 2, 0),
            (PITCH_LENGTH_CM / 2, PITCH_WIDTH_CM / 2 - center_circle),
            (PITCH_LENGTH_CM / 2, PITCH_WIDTH_CM / 2 + center_circle),
            (PITCH_LENGTH_CM / 2, PITCH_WIDTH_CM),
            (
                PITCH_LENGTH_CM - penalty_length,
                (PITCH_WIDTH_CM - penalty_width) / 2,
            ),
            (
                PITCH_LENGTH_CM - penalty_length,
                (PITCH_WIDTH_CM - goal_width) / 2,
            ),
            (
                PITCH_LENGTH_CM - penalty_length,
                (PITCH_WIDTH_CM + goal_width) / 2,
            ),
            (
                PITCH_LENGTH_CM - penalty_length,
                (PITCH_WIDTH_CM + penalty_width) / 2,
            ),
            (PITCH_LENGTH_CM - penalty_spot, PITCH_WIDTH_CM / 2),
            (
                PITCH_LENGTH_CM - goal_length,
                (PITCH_WIDTH_CM - goal_width) / 2,
            ),
            (
                PITCH_LENGTH_CM - goal_length,
                (PITCH_WIDTH_CM + goal_width) / 2,
            ),
            (PITCH_LENGTH_CM, 0),
            (PITCH_LENGTH_CM, (PITCH_WIDTH_CM - penalty_width) / 2),
            (PITCH_LENGTH_CM, (PITCH_WIDTH_CM - goal_width) / 2),
            (PITCH_LENGTH_CM, (PITCH_WIDTH_CM + goal_width) / 2),
            (PITCH_LENGTH_CM, (PITCH_WIDTH_CM + penalty_width) / 2),
            (PITCH_LENGTH_CM, PITCH_WIDTH_CM),
            (PITCH_LENGTH_CM / 2 - center_circle, PITCH_WIDTH_CM / 2),
            (PITCH_LENGTH_CM / 2 + center_circle, PITCH_WIDTH_CM / 2),
        ],
        dtype=np.float32,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare pitch-pose models on sampled video geometry."
    )
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--frames", type=int, default=1000)
    parser.add_argument("--stride", type=int, default=12)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--device", default="0")
    parser.add_argument("--keypoint-confidence", type=float, default=0.34)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def sample_frames(
    video: Path,
    start_frame: int,
    count: int,
    stride: int,
) -> list[tuple[int, np.ndarray]]:
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise ValueError(f"Could not open {video}")
    capture.set(cv2.CAP_PROP_POS_FRAMES, max(0, start_frame))
    sampled: list[tuple[int, np.ndarray]] = []
    for offset in range(count):
        ok, frame = capture.read()
        if not ok:
            break
        if offset % stride == 0:
            sampled.append((start_frame + offset, frame))
    capture.release()
    return sampled


def reprojection_errors(
    source: np.ndarray,
    target: np.ndarray,
    homography: np.ndarray,
) -> np.ndarray:
    projected = cv2.perspectiveTransform(
        source.reshape(-1, 1, 2),
        homography,
    ).reshape(-1, 2)
    return np.linalg.norm(projected - target, axis=1)


def pitch_projection(
    homography: np.ndarray,
    targets: np.ndarray,
) -> np.ndarray | None:
    try:
        inverse = np.linalg.inv(homography)
    except np.linalg.LinAlgError:
        return None
    if not np.all(np.isfinite(inverse)):
        return None
    anchors = np.array(
        [
            [0.0, 0.0],
            [PITCH_LENGTH_CM, 0.0],
            [PITCH_LENGTH_CM, PITCH_WIDTH_CM],
            [0.0, PITCH_WIDTH_CM],
            [PITCH_LENGTH_CM / 2, PITCH_WIDTH_CM / 2],
            targets[8],
            targets[21],
        ],
        dtype=np.float32,
    )
    return cv2.perspectiveTransform(
        anchors.reshape(-1, 1, 2),
        inverse,
    ).reshape(-1, 2)


def summarize(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"median": None, "mean": None, "p95": None}
    array = np.asarray(values, dtype=np.float64)
    return {
        "median": round(float(np.median(array)), 4),
        "mean": round(float(np.mean(array)), 4),
        "p95": round(float(np.percentile(array, 95)), 4),
    }


def evaluate_model(
    model_path: str,
    frames: list[tuple[int, np.ndarray]],
    imgsz: int,
    device: str,
    confidence_threshold: float,
) -> dict[str, Any]:
    model = YOLO(model_path, task="pose")
    model.predict(
        frames[0][1],
        imgsz=imgsz,
        device=device,
        verbose=False,
    )
    targets = pitch_vertices()
    visible_counts: list[int] = []
    inference_ms: list[float] = []
    reprojection_error_cm: list[float] = []
    inlier_ratios: list[float] = []
    temporal_anchor_jumps: list[float] = []
    successful_homographies = 0
    wide_view_homographies = 0
    frames_with_pose = 0
    previous_projection: np.ndarray | None = None
    previous_frame_index: int | None = None
    per_frame: list[dict[str, Any]] = []

    for frame_index, frame in frames:
        started = perf_counter()
        result = model.predict(
            frame,
            imgsz=imgsz,
            device=device,
            verbose=False,
        )[0]
        inference_ms.append((perf_counter() - started) * 1000)
        keypoints = result.keypoints
        row: dict[str, Any] = {
            "frame": frame_index,
            "visible_keypoints": 0,
            "homography": False,
        }
        if keypoints is None or keypoints.xy is None:
            per_frame.append(row)
            visible_counts.append(0)
            continue

        source = keypoints.xy.cpu().numpy()
        if source.ndim == 3:
            source = source[0]
        confidence = keypoints.conf
        confidence_values = (
            np.ones(len(source), dtype=np.float32)
            if confidence is None
            else confidence.cpu().numpy()
        )
        if confidence_values.ndim == 2:
            confidence_values = confidence_values[0]
        count = min(len(source), len(targets), len(confidence_values))
        source = source[:count].astype(np.float32)
        frame_targets = targets[:count]
        confidence_values = confidence_values[:count]
        visible = (
            (confidence_values >= confidence_threshold)
            & (source[:, 0] > 1)
            & (source[:, 1] > 1)
        )
        visible_count = int(np.count_nonzero(visible))
        visible_counts.append(visible_count)
        row["visible_keypoints"] = visible_count
        row["mean_visible_confidence"] = (
            round(float(np.mean(confidence_values[visible])), 4)
            if visible_count
            else None
        )
        if visible_count:
            frames_with_pose += 1
        if visible_count < 4:
            per_frame.append(row)
            continue

        visible_source = source[visible]
        visible_target = frame_targets[visible]
        homography, inlier_mask = cv2.findHomography(
            visible_source,
            visible_target,
            cv2.RANSAC,
            320.0,
        )
        if homography is None or not np.all(np.isfinite(homography)):
            per_frame.append(row)
            continue
        inliers = (
            inlier_mask.reshape(-1).astype(bool)
            if inlier_mask is not None
            else np.ones(visible_count, dtype=bool)
        )
        inlier_count = int(np.count_nonzero(inliers))
        if inlier_count < 4:
            per_frame.append(row)
            continue

        errors = reprojection_errors(
            visible_source[inliers],
            visible_target[inliers],
            homography,
        )
        error = float(np.median(errors))
        inlier_ratio = inlier_count / max(1, visible_count)
        successful_homographies += 1
        reprojection_error_cm.append(error)
        inlier_ratios.append(inlier_ratio)

        target_span_x = float(np.ptp(visible_target[:, 0]))
        target_span_y = float(np.ptp(visible_target[:, 1]))
        hull_area = float(cv2.contourArea(cv2.convexHull(visible_source)))
        frame_area = float(max(1, frame.shape[0] * frame.shape[1]))
        is_wide_view = (
            visible_count >= 8
            and target_span_x >= PITCH_LENGTH_CM * 0.50
            and target_span_y >= PITCH_WIDTH_CM * 0.40
            and hull_area >= frame_area * 0.045
        )
        if is_wide_view:
            wide_view_homographies += 1

        projection = pitch_projection(homography, targets)
        if (
            projection is not None
            and previous_projection is not None
            and previous_frame_index is not None
        ):
            frame_gap = max(1, frame_index - previous_frame_index)
            diagonal = float(np.hypot(frame.shape[1], frame.shape[0]))
            jump = float(
                np.median(np.linalg.norm(projection - previous_projection, axis=1))
                / max(1.0, diagonal)
                / frame_gap
            )
            temporal_anchor_jumps.append(jump)
        if projection is not None:
            previous_projection = projection
            previous_frame_index = frame_index

        row.update(
            {
                "homography": True,
                "inliers": inlier_count,
                "inlier_ratio": round(inlier_ratio, 4),
                "median_reprojection_error_cm": round(error, 2),
                "target_span_cm": [
                    round(target_span_x, 1),
                    round(target_span_y, 1),
                ],
                "wide_view": is_wide_view,
            }
        )
        per_frame.append(row)

    sampled_count = len(frames)
    return {
        "model": model_path,
        "sampled_frames": sampled_count,
        "frames_with_pose": frames_with_pose,
        "pose_frame_ratio": round(frames_with_pose / max(1, sampled_count), 4),
        "successful_homographies": successful_homographies,
        "homography_success_ratio": round(
            successful_homographies / max(1, sampled_count),
            4,
        ),
        "wide_view_homographies": wide_view_homographies,
        "visible_keypoints": summarize([float(value) for value in visible_counts]),
        "inlier_ratio": summarize(inlier_ratios),
        "reprojection_error_cm": summarize(reprojection_error_cm),
        "temporal_anchor_jump_per_frame_ratio": summarize(temporal_anchor_jumps),
        "inference_ms": summarize(inference_ms),
        "per_frame": per_frame,
    }


def main() -> int:
    args = parse_args()
    frames = sample_frames(
        args.video,
        args.start_frame,
        args.frames,
        args.stride,
    )
    if not frames:
        raise ValueError("No frames were sampled")
    report = {
        "video": str(args.video.resolve()),
        "start_frame": args.start_frame,
        "source_frames_scanned": args.frames,
        "stride": args.stride,
        "image_size": args.imgsz,
        "keypoint_confidence": args.keypoint_confidence,
        "models": [
            evaluate_model(
                model,
                frames,
                args.imgsz,
                args.device,
                args.keypoint_confidence,
            )
            for model in args.models
        ],
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
