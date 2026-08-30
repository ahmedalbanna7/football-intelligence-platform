from __future__ import annotations

import math
from typing import Any


BALL_FRAME_STATES = {"visible", "occluded", "out_of_frame", "uncertain"}


def _finite_pair(value: Any, label: str) -> tuple[float, float]:
    if not isinstance(value, list) or len(value) != 2:
        raise ValueError(f"{label} must contain [x, y]")
    pair = (float(value[0]), float(value[1]))
    if not all(math.isfinite(item) for item in pair):
        raise ValueError(f"{label} must contain finite coordinates")
    return pair


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, math.ceil(percentile * len(ordered)) - 1))
    return float(ordered[index])


def validate_ball_ground_truth(
    payload: dict[str, Any],
    *,
    require_verified: bool = False,
) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError("Ball ground truth must be a JSON object")
    if str(payload.get("schema_version") or "") != "ball_ground_truth.v1":
        raise ValueError("Ball ground truth must use schema_version ball_ground_truth.v1")

    verification = payload.get("verification")
    if not isinstance(verification, dict):
        raise ValueError("Ball ground truth requires verification metadata")
    status = str(verification.get("status") or "draft").lower()
    if status not in {"draft", "verified"}:
        raise ValueError("Ball ground-truth verification status must be draft or verified")
    if require_verified and status != "verified":
        raise ValueError("Ball ground truth must be manually reviewed before evaluation")
    if status == "verified":
        if not str(verification.get("annotator") or "").strip():
            raise ValueError("Verified ball ground truth requires an annotator")
        if not str(verification.get("reviewed_at") or "").strip():
            raise ValueError("Verified ball ground truth requires reviewed_at")

    frames = payload.get("frames")
    if not isinstance(frames, list) or not frames:
        raise ValueError("Ball ground truth must contain at least one annotated frame")
    if len(frames) > 3001:
        raise ValueError("Ball ground truth cannot exceed 3001 annotated frames")

    resolution = payload.get("resolution") or []
    width = float(resolution[0]) if isinstance(resolution, list) and len(resolution) == 2 else None
    height = float(resolution[1]) if isinstance(resolution, list) and len(resolution) == 2 else None

    seen_frames: set[int] = set()
    state_counts = {state: 0 for state in sorted(BALL_FRAME_STATES)}
    verified_frames = 0
    for item in frames:
        if not isinstance(item, dict):
            raise ValueError("Every ball frame annotation must be an object")
        frame = int(item.get("frame", -1))
        if frame < 0:
            raise ValueError("Ball frame indexes must be non-negative")
        if frame in seen_frames:
            raise ValueError(f"Ball frame {frame} occurs more than once")
        seen_frames.add(frame)
        source_frame = item.get("source_frame")
        if source_frame is not None and int(source_frame) < 0:
            raise ValueError(f"Ball source frame at frame {frame} must be non-negative")

        state = str(item.get("state") or "uncertain").lower()
        if state not in BALL_FRAME_STATES:
            raise ValueError(f"Unsupported ball state {state!r} at frame {frame}")
        state_counts[state] += 1
        review_state = str(item.get("review_state") or "unverified").lower()
        if review_state not in {"unverified", "verified"}:
            raise ValueError(f"Ball frame {frame} has an invalid review_state")
        if review_state == "verified":
            verified_frames += 1
        elif status == "verified":
            raise ValueError(f"Ball frame {frame} is not verified")

        ball = item.get("ball")
        if state == "visible" and not isinstance(ball, dict):
            raise ValueError(f"Visible ball frame {frame} requires a ball position")
        if isinstance(ball, dict):
            center = _finite_pair(ball.get("center"), f"Ball center at frame {frame}")
            if width is not None and height is not None and not (
                0 <= center[0] <= width and 0 <= center[1] <= height
            ):
                raise ValueError(f"Ball center at frame {frame} is outside the source resolution")
            bbox = ball.get("bbox")
            if bbox is not None:
                if not isinstance(bbox, list) or len(bbox) != 4:
                    raise ValueError(f"Ball bbox at frame {frame} must contain four values")
                values = [float(value) for value in bbox]
                if not all(math.isfinite(value) for value in values):
                    raise ValueError(f"Ball bbox at frame {frame} must be finite")
                if values[2] <= values[0] or values[3] <= values[1]:
                    raise ValueError(f"Ball bbox at frame {frame} has invalid dimensions")
                if width is not None and height is not None and (
                    values[0] < 0
                    or values[1] < 0
                    or values[2] > width
                    or values[3] > height
                ):
                    raise ValueError(f"Ball bbox at frame {frame} is outside the source resolution")
            ball_height = ball.get("height_cm")
            if ball_height is not None and (
                not math.isfinite(float(ball_height)) or float(ball_height) < 0
            ):
                raise ValueError(f"Ball height at frame {frame} must be non-negative")

    return {
        "status": status,
        "frame_count": len(frames),
        "verified_frames": verified_frames,
        "state_counts": state_counts,
        "ready_for_evaluation": status == "verified" and verified_frames == len(frames),
    }


def evaluate_ball_tracking(
    visual_layers: dict[str, Any],
    ground_truth: dict[str, Any],
    *,
    source_frame_offset: int = 0,
    tolerance_pixels: float | None = None,
) -> dict[str, Any]:
    validation = validate_ball_ground_truth(ground_truth, require_verified=True)
    resolution = visual_layers.get("resolution") or ground_truth.get("resolution") or [1920, 1080]
    width = max(1.0, float(resolution[0]))
    height = max(1.0, float(resolution[1]))
    diagonal = math.hypot(width, height)
    tolerance = float(tolerance_pixels or max(8.0, diagonal * 0.012))

    ball_layer = visual_layers.get("ball") or {}
    image_path = ball_layer.get("image_path") or []
    predictions: dict[int, dict[str, Any]] = {}
    for sample in image_path:
        if not isinstance(sample, dict) or sample.get("frame") is None:
            continue
        frame = int(sample["frame"]) + int(source_frame_offset)
        if sample.get("x") is None or sample.get("y") is None:
            continue
        predictions[frame] = sample

    true_positives = 0
    false_positives = 0
    false_negatives = 0
    visible_frames = 0
    negative_frames = 0
    ignored_frames = 0
    observed_matches = 0
    interpolated_matches = 0
    center_errors: list[float] = []
    center_error_ratios: list[float] = []
    airborne_correct = 0
    airborne_measured = 0
    height_errors: list[float] = []
    frame_results: list[dict[str, Any]] = []
    current_visible_miss_gap = 0
    maximum_visible_miss_gap = 0

    frames = sorted(ground_truth["frames"], key=lambda item: int(item["frame"]))
    for item in frames:
        coordinate = int(item.get("source_frame", int(item["frame"]) + source_frame_offset))
        state = str(item.get("state") or "uncertain").lower()
        prediction = predictions.get(coordinate)
        result: dict[str, Any] = {
            "frame": int(item["frame"]),
            "source_frame": coordinate,
            "state": state,
            "prediction_present": prediction is not None,
        }
        if state in {"occluded", "uncertain"}:
            ignored_frames += 1
            result["outcome"] = "ignored"
            frame_results.append(result)
            continue
        if state == "out_of_frame":
            negative_frames += 1
            if prediction is None:
                result["outcome"] = "true_negative"
            else:
                false_positives += 1
                result["outcome"] = "false_positive"
            frame_results.append(result)
            continue

        visible_frames += 1
        ball = item["ball"]
        ground_x, ground_y = _finite_pair(ball.get("center"), "Ball center")
        if prediction is None:
            false_negatives += 1
            current_visible_miss_gap += 1
            maximum_visible_miss_gap = max(maximum_visible_miss_gap, current_visible_miss_gap)
            result["outcome"] = "false_negative"
            frame_results.append(result)
            continue

        error = math.hypot(float(prediction["x"]) - ground_x, float(prediction["y"]) - ground_y)
        result["center_error_pixels"] = round(error, 3)
        result["tolerance_pixels"] = round(tolerance, 3)
        if error <= tolerance:
            true_positives += 1
            current_visible_miss_gap = 0
            center_errors.append(error)
            center_error_ratios.append(error / diagonal)
            if bool(prediction.get("predicted", False)):
                interpolated_matches += 1
            else:
                observed_matches += 1
            result["outcome"] = "matched"
        else:
            false_positives += 1
            false_negatives += 1
            current_visible_miss_gap += 1
            maximum_visible_miss_gap = max(maximum_visible_miss_gap, current_visible_miss_gap)
            result["outcome"] = "localization_miss"

        if ball.get("airborne") is not None and prediction.get("airborne") is not None:
            airborne_measured += 1
            if bool(ball["airborne"]) == bool(prediction["airborne"]):
                airborne_correct += 1
        if ball.get("height_cm") is not None and prediction.get("height_cm") is not None:
            height_errors.append(abs(float(ball["height_cm"]) - float(prediction["height_cm"])))
        frame_results.append(result)

    precision = true_positives / max(1, true_positives + false_positives)
    recall = true_positives / max(1, true_positives + false_negatives)
    f1 = 2.0 * precision * recall / max(1e-12, precision + recall)
    median_error = _percentile(center_errors, 0.5)
    p95_error = _percentile(center_errors, 0.95)
    median_ratio = _percentile(center_error_ratios, 0.5)
    p95_ratio = _percentile(center_error_ratios, 0.95)
    airborne_accuracy = (
        airborne_correct / airborne_measured if airborne_measured else None
    )

    configured = ground_truth.get("release_thresholds") or {}
    thresholds = {
        "minimum_evaluated_frames": int(configured.get("minimum_evaluated_frames", 20)),
        "minimum_visible_frames": int(configured.get("minimum_visible_frames", 10)),
        "minimum_precision": float(configured.get("minimum_precision", 0.90)),
        "minimum_recall": float(configured.get("minimum_recall", 0.85)),
        "maximum_median_center_error_ratio": float(
            configured.get("maximum_median_center_error_ratio", 0.012)
        ),
        "maximum_p95_center_error_ratio": float(
            configured.get("maximum_p95_center_error_ratio", 0.03)
        ),
        "minimum_airborne_accuracy": float(configured.get("minimum_airborne_accuracy", 0.80)),
    }
    conditions = [
        {
            "code": "evaluated_sample_size",
            "passed": visible_frames + negative_frames
            >= thresholds["minimum_evaluated_frames"],
            "actual": visible_frames + negative_frames,
            "required": f">= {thresholds['minimum_evaluated_frames']} frames",
        },
        {
            "code": "visible_sample_size",
            "passed": visible_frames >= thresholds["minimum_visible_frames"],
            "actual": visible_frames,
            "required": f">= {thresholds['minimum_visible_frames']} visible frames",
        },
        {
            "code": "precision",
            "passed": precision >= thresholds["minimum_precision"],
            "actual": round(precision * 100.0, 3),
            "required": f">= {thresholds['minimum_precision'] * 100.0:.1f}%",
        },
        {
            "code": "recall",
            "passed": recall >= thresholds["minimum_recall"],
            "actual": round(recall * 100.0, 3),
            "required": f">= {thresholds['minimum_recall'] * 100.0:.1f}%",
        },
        {
            "code": "median_center_error",
            "passed": median_ratio is not None
            and median_ratio <= thresholds["maximum_median_center_error_ratio"],
            "actual": round((median_ratio or 0.0) * 100.0, 3) if median_ratio is not None else None,
            "required": f"<= {thresholds['maximum_median_center_error_ratio'] * 100.0:.1f}% diagonal",
        },
        {
            "code": "p95_center_error",
            "passed": p95_ratio is not None
            and p95_ratio <= thresholds["maximum_p95_center_error_ratio"],
            "actual": round((p95_ratio or 0.0) * 100.0, 3) if p95_ratio is not None else None,
            "required": f"<= {thresholds['maximum_p95_center_error_ratio'] * 100.0:.1f}% diagonal",
        },
    ]
    if airborne_measured:
        conditions.append(
            {
                "code": "airborne_state",
                "passed": airborne_accuracy is not None
                and airborne_accuracy >= thresholds["minimum_airborne_accuracy"],
                "actual": round((airborne_accuracy or 0.0) * 100.0, 3),
                "required": f">= {thresholds['minimum_airborne_accuracy'] * 100.0:.1f}%",
            }
        )

    return {
        "engine": "ball_ground_truth_quality_v1",
        "status": "measured",
        "coordinate_space": "source_video_pixels",
        "evaluated_frames": visible_frames + negative_frames,
        "visible_frames": visible_frames,
        "negative_frames": negative_frames,
        "ignored_frames": ignored_frames,
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "precision": round(precision * 100.0, 3),
        "recall": round(recall * 100.0, 3),
        "f1": round(f1 * 100.0, 3),
        "mean_center_error_pixels": round(sum(center_errors) / len(center_errors), 3)
        if center_errors
        else None,
        "median_center_error_pixels": round(median_error, 3) if median_error is not None else None,
        "p95_center_error_pixels": round(p95_error, 3) if p95_error is not None else None,
        "median_center_error_ratio": round(median_ratio, 6) if median_ratio is not None else None,
        "p95_center_error_ratio": round(p95_ratio, 6) if p95_ratio is not None else None,
        "observed_matches": observed_matches,
        "interpolated_matches": interpolated_matches,
        "maximum_visible_miss_gap": maximum_visible_miss_gap,
        "airborne_accuracy": round(airborne_accuracy * 100.0, 3)
        if airborne_accuracy is not None
        else None,
        "airborne_evaluated_frames": airborne_measured,
        "height_mae_cm": round(sum(height_errors) / len(height_errors), 3)
        if height_errors
        else None,
        "tolerance_pixels": round(tolerance, 3),
        "validation": validation,
        "release_gate": {
            "status": "passed" if conditions and all(item["passed"] for item in conditions) else "blocked",
            "conditions": conditions,
            "thresholds": thresholds,
        },
        "frames": frame_results,
    }
