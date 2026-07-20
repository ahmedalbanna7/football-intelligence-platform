from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np


def _iou(a: list[float], b: list[float]) -> float:
    x1 = max(float(a[0]), float(b[0]))
    y1 = max(float(a[1]), float(b[1]))
    x2 = min(float(a[2]), float(b[2]))
    y2 = min(float(a[3]), float(b[3]))
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, float(a[2]) - float(a[0])) * max(
        0.0,
        float(a[3]) - float(a[1]),
    )
    area_b = max(0.0, float(b[2]) - float(b[0])) * max(
        0.0,
        float(b[3]) - float(b[1]),
    )
    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0


def _normalise_frames(payload: dict[str, Any], prediction: bool) -> dict[int, list[dict[str, Any]]]:
    frames: dict[int, list[dict[str, Any]]] = defaultdict(list)
    identity_keys = (
        ("track_id", "canonical_track_id", "identity_id", "id")
        if prediction
        else ("identity_id", "player_id", "track_id", "id")
    )

    if isinstance(payload.get("frames"), list):
        source = []
        for frame_payload in payload["frames"]:
            if not isinstance(frame_payload, dict):
                continue
            frame_index = frame_payload.get("frame", frame_payload.get("frame_index"))
            for item in frame_payload.get("objects", frame_payload.get("annotations", [])):
                if isinstance(item, dict):
                    source.append({**item, "frame": frame_index})
    else:
        source = payload.get("observations", payload.get("annotations", []))

    if not isinstance(source, list):
        raise ValueError("Tracking data must contain a frames or observations list")

    for item in source:
        if not isinstance(item, dict):
            continue
        frame_value = item.get("frame", item.get("frame_index"))
        bbox = item.get("bbox", item.get("bbox_xyxy"))
        identity = next((item.get(key) for key in identity_keys if item.get(key) is not None), None)
        if frame_value is None or identity is None or not isinstance(bbox, list) or len(bbox) != 4:
            continue
        frames[int(frame_value)].append(
            {
                "identity": str(identity),
                "bbox": [float(value) for value in bbox],
            }
        )
    return dict(frames)


def _linear_assignment(scores: np.ndarray) -> list[tuple[int, int]]:
    if scores.size == 0:
        return []
    try:
        import lap

        costs = 1.0 - scores
        _, row_assignment, _ = lap.lapjv(costs, extend_cost=True)
        return [
            (row, int(column))
            for row, column in enumerate(row_assignment)
            if 0 <= int(column) < scores.shape[1]
        ]
    except (ImportError, TypeError, ValueError, RuntimeError):
        pairs: list[tuple[int, int]] = []
        used_rows: set[int] = set()
        used_columns: set[int] = set()
        ranked = [
            (float(scores[row, column]), row, column)
            for row in range(scores.shape[0])
            for column in range(scores.shape[1])
        ]
        for _, row, column in sorted(ranked, reverse=True):
            if row in used_rows or column in used_columns:
                continue
            pairs.append((row, column))
            used_rows.add(row)
            used_columns.add(column)
        return pairs


def _frame_matches(
    ground_truth: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
    threshold: float,
) -> list[tuple[str, str, float]]:
    if not ground_truth or not predictions:
        return []
    scores = np.array(
        [
            [_iou(gt_item["bbox"], prediction["bbox"]) for prediction in predictions]
            for gt_item in ground_truth
        ],
        dtype=np.float64,
    )
    return [
        (
            ground_truth[row]["identity"],
            predictions[column]["identity"],
            float(scores[row, column]),
        )
        for row, column in _linear_assignment(scores)
        if scores[row, column] >= threshold
    ]


def _identity_metrics(
    gt_frames: dict[int, list[dict[str, Any]]],
    prediction_frames: dict[int, list[dict[str, Any]]],
    threshold: float,
) -> dict[str, float | int]:
    pair_counts: dict[tuple[str, str], int] = defaultdict(int)
    matches_by_frame: dict[int, list[tuple[str, str, float]]] = {}
    all_frames = sorted(set(gt_frames) | set(prediction_frames))
    for frame_index in all_frames:
        matches = _frame_matches(
            gt_frames.get(frame_index, []),
            prediction_frames.get(frame_index, []),
            threshold,
        )
        matches_by_frame[frame_index] = matches
        for ground_truth_id, prediction_id, _ in matches:
            pair_counts[(ground_truth_id, prediction_id)] += 1

    gt_ids = sorted({item["identity"] for items in gt_frames.values() for item in items})
    prediction_ids = sorted(
        {item["identity"] for items in prediction_frames.values() for item in items}
    )
    identity_scores = np.zeros((len(gt_ids), len(prediction_ids)), dtype=np.float64)
    for row, ground_truth_id in enumerate(gt_ids):
        for column, prediction_id in enumerate(prediction_ids):
            identity_scores[row, column] = pair_counts.get(
                (ground_truth_id, prediction_id),
                0,
            )
    idtp = int(
        sum(identity_scores[row, column] for row, column in _linear_assignment(identity_scores))
    )
    gt_detections = sum(len(items) for items in gt_frames.values())
    prediction_detections = sum(len(items) for items in prediction_frames.values())
    idfn = max(0, gt_detections - idtp)
    idfp = max(0, prediction_detections - idtp)
    denominator = 2 * idtp + idfp + idfn
    idf1 = 2 * idtp / denominator if denominator else 0.0

    last_prediction_by_gt: dict[str, str] = {}
    id_switches = 0
    matched_frames_by_gt: dict[str, list[int]] = defaultdict(list)
    for frame_index in all_frames:
        for ground_truth_id, prediction_id, _ in matches_by_frame[frame_index]:
            previous = last_prediction_by_gt.get(ground_truth_id)
            if previous is not None and previous != prediction_id:
                id_switches += 1
            last_prediction_by_gt[ground_truth_id] = prediction_id
            matched_frames_by_gt[ground_truth_id].append(frame_index)

    fragmentation = 0
    for ground_truth_id in gt_ids:
        gt_presence = sorted(
            frame_index
            for frame_index, items in gt_frames.items()
            if any(item["identity"] == ground_truth_id for item in items)
        )
        matched = set(matched_frames_by_gt.get(ground_truth_id, []))
        segments = 0
        inside_segment = False
        for frame_index in gt_presence:
            if frame_index in matched and not inside_segment:
                segments += 1
                inside_segment = True
            elif frame_index not in matched:
                inside_segment = False
        fragmentation += max(segments - 1, 0)

    return {
        "idtp": idtp,
        "idfp": idfp,
        "idfn": idfn,
        "idf1": idf1,
        "id_switches": id_switches,
        "fragmentation": fragmentation,
    }


def _hota_at_threshold(
    gt_frames: dict[int, list[dict[str, Any]]],
    prediction_frames: dict[int, list[dict[str, Any]]],
    threshold: float,
) -> dict[str, float | int]:
    pair_counts: dict[tuple[str, str], int] = defaultdict(int)
    gt_counts: dict[str, int] = defaultdict(int)
    prediction_counts: dict[str, int] = defaultdict(int)
    matches: list[tuple[str, str]] = []
    true_positives = 0
    false_negatives = 0
    false_positives = 0

    for items in gt_frames.values():
        for item in items:
            gt_counts[item["identity"]] += 1
    for items in prediction_frames.values():
        for item in items:
            prediction_counts[item["identity"]] += 1

    for frame_index in sorted(set(gt_frames) | set(prediction_frames)):
        frame_matches = _frame_matches(
            gt_frames.get(frame_index, []),
            prediction_frames.get(frame_index, []),
            threshold,
        )
        true_positives += len(frame_matches)
        false_negatives += len(gt_frames.get(frame_index, [])) - len(frame_matches)
        false_positives += len(prediction_frames.get(frame_index, [])) - len(frame_matches)
        for ground_truth_id, prediction_id, _ in frame_matches:
            pair_counts[(ground_truth_id, prediction_id)] += 1
            matches.append((ground_truth_id, prediction_id))

    det_denominator = true_positives + false_negatives + false_positives
    detection_accuracy = true_positives / det_denominator if det_denominator else 0.0
    association_sum = 0.0
    for ground_truth_id, prediction_id in matches:
        true_positive_association = pair_counts[(ground_truth_id, prediction_id)]
        association_denominator = (
            gt_counts[ground_truth_id]
            + prediction_counts[prediction_id]
            - true_positive_association
        )
        if association_denominator:
            association_sum += true_positive_association / association_denominator
    association_accuracy = association_sum / true_positives if true_positives else 0.0
    return {
        "threshold": round(threshold, 2),
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "detection_accuracy": detection_accuracy,
        "association_accuracy": association_accuracy,
        "hota": float(np.sqrt(detection_accuracy * association_accuracy)),
    }


def evaluate_tracking(
    predictions: dict[str, Any],
    ground_truth: dict[str, Any],
    iou_threshold: float = 0.5,
) -> dict[str, Any]:
    if not 0.05 <= iou_threshold <= 0.95:
        raise ValueError("iou_threshold must be between 0.05 and 0.95")
    prediction_frames = _normalise_frames(predictions, prediction=True)
    ground_truth_frames = _normalise_frames(ground_truth, prediction=False)
    if not ground_truth_frames:
        raise ValueError("Ground truth does not contain valid frame annotations")
    if not prediction_frames:
        raise ValueError("Tracking run does not contain valid predictions")

    identity = _identity_metrics(
        ground_truth_frames,
        prediction_frames,
        iou_threshold,
    )
    thresholds = [round(value, 2) for value in np.arange(0.05, 1.0, 0.05)]
    hota_curve = [
        _hota_at_threshold(ground_truth_frames, prediction_frames, threshold)
        for threshold in thresholds
    ]
    hota = float(np.mean([float(item["hota"]) for item in hota_curve]))
    evaluated_frames = len(set(ground_truth_frames) | set(prediction_frames))
    return {
        "status": "measured",
        "protocol": "mot_identity_and_hota",
        "iou_threshold": iou_threshold,
        "evaluated_frames": evaluated_frames,
        "ground_truth_identities": len(
            {item["identity"] for items in ground_truth_frames.values() for item in items}
        ),
        "predicted_identities": len(
            {item["identity"] for items in prediction_frames.values() for item in items}
        ),
        "id_switches": int(identity["id_switches"]),
        "fragmentation": int(identity["fragmentation"]),
        "idf1": round(float(identity["idf1"]) * 100.0, 3),
        "hota": round(hota * 100.0, 3),
        "idtp": int(identity["idtp"]),
        "idfp": int(identity["idfp"]),
        "idfn": int(identity["idfn"]),
        "hota_curve": [
            {
                **item,
                "detection_accuracy": round(float(item["detection_accuracy"]) * 100.0, 3),
                "association_accuracy": round(float(item["association_accuracy"]) * 100.0, 3),
                "hota": round(float(item["hota"]) * 100.0, 3),
            }
            for item in hota_curve
        ],
        "warning": None,
    }
