from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np


def _validate_ground_truth(payload: dict[str, Any]) -> dict[str, Any]:
    verification = payload.get("verification")
    if not isinstance(verification, dict):
        verification = {}
    status = str(
        verification.get("status")
        or payload.get("verification_status")
        or payload.get("status")
        or "draft"
    ).lower()
    if status != "verified":
        raise ValueError(
            "Ground truth must be manually reviewed and marked verification.status='verified'"
        )
    annotator = verification.get("annotator") or payload.get("annotator")
    if not str(annotator or "").strip():
        raise ValueError("Verified ground truth must include verification.annotator")
    for frame_payload in payload.get("frames", []):
        if not isinstance(frame_payload, dict):
            continue
        for item in frame_payload.get("objects", frame_payload.get("annotations", [])):
            if isinstance(item, dict) and item.get("review_state") != "verified":
                raise ValueError(
                    "Every ground-truth object must have review_state='verified'"
                )
    return {
        "status": status,
        "annotator": str(annotator),
        "reviewed_at": verification.get("reviewed_at"),
    }


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


def _has_source_frames(payload: dict[str, Any]) -> bool:
    if isinstance(payload.get("frames"), list):
        return any(
            isinstance(frame_payload, dict)
            and (
                frame_payload.get("source_frame") is not None
                or any(
                    isinstance(item, dict) and item.get("source_frame") is not None
                    for item in frame_payload.get("objects", frame_payload.get("annotations", []))
                )
            )
            for frame_payload in payload["frames"]
        )
    source = payload.get("observations", payload.get("annotations", []))
    return isinstance(source, list) and any(
        isinstance(item, dict) and item.get("source_frame") is not None
        for item in source
    )


def _normalise_frames(
    payload: dict[str, Any],
    prediction: bool,
    use_source_frames: bool = False,
) -> dict[int, list[dict[str, Any]]]:
    frames: dict[int, list[dict[str, Any]]] = defaultdict(list)
    identity_metadata = payload.get("identities", {})
    if isinstance(identity_metadata, list):
        identity_metadata = {
            str(item.get("identity_id", item.get("id"))): item
            for item in identity_metadata
            if isinstance(item, dict) and item.get("identity_id", item.get("id")) is not None
        }
    if not isinstance(identity_metadata, dict):
        identity_metadata = {}
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
            frame_index = (
                frame_payload.get("source_frame")
                if use_source_frames and frame_payload.get("source_frame") is not None
                else frame_payload.get("frame", frame_payload.get("frame_index"))
            )
            for item in frame_payload.get("objects", frame_payload.get("annotations", [])):
                if isinstance(item, dict):
                    item_frame = (
                        item.get("source_frame")
                        if use_source_frames and item.get("source_frame") is not None
                        else frame_index
                    )
                    source.append({**item, "frame": item_frame})
    else:
        source = payload.get("observations", payload.get("annotations", []))

    if not isinstance(source, list):
        raise ValueError("Tracking data must contain a frames or observations list")

    for item in source:
        if not isinstance(item, dict):
            continue
        frame_value = (
            item.get("source_frame")
            if use_source_frames and item.get("source_frame") is not None
            else item.get("frame", item.get("frame_index"))
        )
        bbox = item.get("bbox", item.get("bbox_xyxy"))
        identity = next((item.get(key) for key in identity_keys if item.get(key) is not None), None)
        if frame_value is None or identity is None or not isinstance(bbox, list) or len(bbox) != 4:
            continue
        metadata = identity_metadata.get(str(identity), {})
        if not isinstance(metadata, dict):
            metadata = {}
        team = item.get("team")
        if team is None:
            team = item.get("team_number")
        if team is None:
            team = metadata.get("team", metadata.get("team_number"))
        role_name = item.get("role_name")
        if role_name is None:
            role_name = item.get("role")
        if role_name is None:
            role_name = metadata.get("role_name", metadata.get("role"))
        frames[int(frame_value)].append(
            {
                "identity": str(identity),
                "bbox": [float(value) for value in bbox],
                "team": team,
                "role_name": role_name,
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


DEFAULT_RELEASE_THRESHOLDS = {
    "minimum_idf1": 95.0,
    "minimum_hota": 90.0,
    "maximum_critical_id_switches": 0,
    "maximum_cross_team_switches": 0,
    "required_scenarios": ["crossing", "crowding", "reentry"],
    "required_camera_styles": ["tactical", "close_or_moving"],
}


def _prepare_prediction_frames(
    ground_truth_frames: dict[int, list[dict[str, Any]]],
    prediction_frames: dict[int, list[dict[str, Any]]],
    coverage: str,
) -> dict[int, list[dict[str, Any]]]:
    annotated_frames = set(ground_truth_frames)
    prepared = {
        frame_index: items
        for frame_index, items in prediction_frames.items()
        if frame_index in annotated_frames
    }
    if coverage == "selected_identities":
        prepared = {
            frame_index: [
                prediction
                for prediction in items
                if any(
                    _iou(prediction["bbox"], truth["bbox"]) >= 0.10
                    for truth in ground_truth_frames.get(frame_index, [])
                )
            ]
            for frame_index, items in prepared.items()
        }
    return prepared


def _cross_team_identity_switches(
    ground_truth_frames: dict[int, list[dict[str, Any]]],
    prediction_frames: dict[int, list[dict[str, Any]]],
    threshold: float,
) -> dict[str, Any]:
    teams_by_prediction: dict[str, set[str]] = defaultdict(set)
    labeled_objects = 0
    for frame_index in sorted(ground_truth_frames):
        truth_items = ground_truth_frames.get(frame_index, [])
        prediction_items = prediction_frames.get(frame_index, [])
        truth_by_identity = {item["identity"]: item for item in truth_items}
        for truth_id, prediction_id, _ in _frame_matches(
            truth_items,
            prediction_items,
            threshold,
        ):
            team = truth_by_identity[truth_id].get("team")
            if team in {None, "", 0, "0", "unknown"}:
                continue
            labeled_objects += 1
            teams_by_prediction[prediction_id].add(str(team))
    conflicts = {
        prediction_id: sorted(teams)
        for prediction_id, teams in teams_by_prediction.items()
        if len(teams) > 1
    }
    return {
        "status": "measured" if labeled_objects else "team_labels_required",
        "labeled_matches": labeled_objects,
        "cross_team_switches": len(conflicts) if labeled_objects else None,
        "conflicting_prediction_ids": conflicts,
    }


def _evaluate_frame_set(
    ground_truth_frames: dict[int, list[dict[str, Any]]],
    prediction_frames: dict[int, list[dict[str, Any]]],
    iou_threshold: float,
) -> dict[str, Any]:
    identity = _identity_metrics(ground_truth_frames, prediction_frames, iou_threshold)
    thresholds = [round(value, 2) for value in np.arange(0.05, 1.0, 0.05)]
    hota_curve = [
        _hota_at_threshold(ground_truth_frames, prediction_frames, threshold)
        for threshold in thresholds
    ]
    hota = float(np.mean([float(item["hota"]) for item in hota_curve]))
    cross_team = _cross_team_identity_switches(
        ground_truth_frames,
        prediction_frames,
        iou_threshold,
    )
    return {
        "evaluated_frames": len(ground_truth_frames),
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
        "cross_team": cross_team,
        "hota_curve": [
            {
                **item,
                "detection_accuracy": round(float(item["detection_accuracy"]) * 100.0, 3),
                "association_accuracy": round(float(item["association_accuracy"]) * 100.0, 3),
                "hota": round(float(item["hota"]) * 100.0, 3),
            }
            for item in hota_curve
        ],
    }


def _release_gate(
    metrics: dict[str, Any],
    clip_results: list[dict[str, Any]],
    ground_truth: dict[str, Any],
) -> dict[str, Any]:
    configured = ground_truth.get("release_thresholds") or {}
    thresholds = {**DEFAULT_RELEASE_THRESHOLDS, **configured}
    critical_clips = [clip for clip in clip_results if clip.get("critical")]
    critical_switches = sum(
        int(clip.get("id_switches", 0))
        for clip in (critical_clips or clip_results)
    )
    cross_team = metrics["cross_team"]
    fragment_reviews = ground_truth.get("fragment_reviews") or []
    reviewed_fragments = sum(
        1
        for item in fragment_reviews
        if isinstance(item, dict)
        and str(item.get("status", "")).lower() in {"verified", "resolved", "accepted"}
    )
    unresolved_fragments = max(0, int(metrics["fragmentation"]) - reviewed_fragments)
    scenarios = {
        str(clip.get("scenario") or "unknown").lower()
        for clip in clip_results
    }
    camera_styles = {
        str(clip.get("camera_style") or "unknown").lower()
        for clip in clip_results
    }
    required_scenarios = {
        str(value).lower() for value in thresholds["required_scenarios"]
    }
    required_camera_styles = {
        str(value).lower() for value in thresholds["required_camera_styles"]
    }
    conditions = [
        {
            "code": "critical_id_switches",
            "label": "Zero ID switches in critical clips",
            "passed": critical_switches <= int(thresholds["maximum_critical_id_switches"]),
            "actual": critical_switches,
            "required": f"<= {int(thresholds['maximum_critical_id_switches'])}",
        },
        {
            "code": "idf1",
            "label": "IDF1 threshold",
            "passed": float(metrics["idf1"]) >= float(thresholds["minimum_idf1"]),
            "actual": metrics["idf1"],
            "required": f">= {float(thresholds['minimum_idf1'])}",
        },
        {
            "code": "hota",
            "label": "HOTA threshold",
            "passed": float(metrics["hota"]) >= float(thresholds["minimum_hota"]),
            "actual": metrics["hota"],
            "required": f">= {float(thresholds['minimum_hota'])}",
        },
        {
            "code": "cross_team_switches",
            "label": "Zero identities crossing between teams",
            "passed": (
                cross_team["status"] == "measured"
                and int(cross_team["cross_team_switches"] or 0)
                <= int(thresholds["maximum_cross_team_switches"])
            ),
            "actual": cross_team["cross_team_switches"],
            "required": f"<= {int(thresholds['maximum_cross_team_switches'])}",
            "status": cross_team["status"],
        },
        {
            "code": "fragment_review",
            "label": "Every fragment is explained or corrected",
            "passed": unresolved_fragments == 0,
            "actual": unresolved_fragments,
            "required": "0 unresolved",
        },
        {
            "code": "scenario_coverage",
            "label": "Critical scenario coverage",
            "passed": required_scenarios.issubset(scenarios),
            "actual": sorted(scenarios),
            "required": sorted(required_scenarios),
            "missing": sorted(required_scenarios - scenarios),
        },
        {
            "code": "camera_coverage",
            "label": "Tactical and close/moving camera coverage",
            "passed": required_camera_styles.issubset(camera_styles),
            "actual": sorted(camera_styles),
            "required": sorted(required_camera_styles),
            "missing": sorted(required_camera_styles - camera_styles),
        },
    ]
    return {
        "status": "passed" if all(item["passed"] for item in conditions) else "blocked",
        "thresholds": thresholds,
        "conditions": conditions,
        "critical_clips": len(critical_clips),
        "reviewed_fragments": reviewed_fragments,
        "unresolved_fragments": unresolved_fragments,
    }


def evaluate_tracking(
    predictions: dict[str, Any],
    ground_truth: dict[str, Any],
    iou_threshold: float = 0.5,
) -> dict[str, Any]:
    if not 0.05 <= iou_threshold <= 0.95:
        raise ValueError("iou_threshold must be between 0.05 and 0.95")
    verification = _validate_ground_truth(ground_truth)
    use_source_frames = _has_source_frames(predictions) and _has_source_frames(ground_truth)
    all_prediction_frames = _normalise_frames(
        predictions,
        prediction=True,
        use_source_frames=use_source_frames,
    )
    ground_truth_frames = _normalise_frames(
        ground_truth,
        prediction=False,
        use_source_frames=use_source_frames,
    )
    if not ground_truth_frames:
        raise ValueError("Ground truth does not contain valid frame annotations")
    if not all_prediction_frames:
        raise ValueError("Tracking run does not contain valid predictions")

    for frame_index, items in ground_truth_frames.items():
        identities = [item["identity"] for item in items]
        duplicates = [identity for identity in set(identities) if identities.count(identity) > 1]
        if duplicates:
            raise ValueError(
                f"Ground truth identity {duplicates[0]!r} occurs more than once in frame {frame_index}"
            )

    coverage = str(ground_truth.get("coverage") or "all_visible_identities")
    prediction_frames = _prepare_prediction_frames(
        ground_truth_frames,
        all_prediction_frames,
        coverage,
    )
    if not prediction_frames:
        raise ValueError("Tracking run has no predictions in the annotated ground-truth frames")

    aggregate = _evaluate_frame_set(
        ground_truth_frames,
        prediction_frames,
        iou_threshold,
    )
    clip_results: list[dict[str, Any]] = []
    clips = ground_truth.get("clips") or []
    if not clips:
        clips = [
            {
                "start_frame": min(ground_truth_frames),
                "end_frame": max(ground_truth_frames),
                "scenario": "unknown",
                "camera_style": "unknown",
                "critical": False,
            }
        ]
    for index, clip in enumerate(clips):
        start_key = "source_start_frame" if use_source_frames else "start_frame"
        end_key = "source_end_frame" if use_source_frames else "end_frame"
        start_frame = int(clip.get(start_key, min(ground_truth_frames)))
        end_frame = int(clip.get(end_key, max(ground_truth_frames)))
        clip_truth = {
            frame: items
            for frame, items in ground_truth_frames.items()
            if start_frame <= frame <= end_frame
        }
        if not clip_truth:
            continue
        clip_predictions = _prepare_prediction_frames(
            clip_truth,
            all_prediction_frames,
            str(clip.get("coverage") or coverage),
        )
        clip_metrics = _evaluate_frame_set(clip_truth, clip_predictions, iou_threshold)
        clip_results.append(
            {
                "clip_index": index,
                "start_frame": start_frame,
                "end_frame": end_frame,
                "scenario": str(clip.get("scenario") or "unknown").lower(),
                "camera_style": str(clip.get("camera_style") or "unknown").lower(),
                "critical": bool(clip.get("critical", False)),
                **{key: value for key, value in clip_metrics.items() if key != "hota_curve"},
            }
        )

    result = {
        "status": "measured",
        "protocol": (
            "mot_identity_and_hota_partial_identity"
            if coverage == "selected_identities"
            else "mot_identity_and_hota"
        ),
        "evaluation_scope": "annotated_frames_only",
        "frame_coordinate_space": "source" if use_source_frames else "run_local",
        "coverage": coverage,
        "verification": verification,
        "iou_threshold": iou_threshold,
        **aggregate,
        "clips": clip_results,
        "warning": None,
    }
    result["release_gate"] = _release_gate(result, clip_results, ground_truth)
    return result


def evaluate_release_suite(
    cases: list[dict[str, Any]],
    thresholds: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if not cases:
        raise ValueError("At least one measured release case is required")
    configured = {**DEFAULT_RELEASE_THRESHOLDS, **(thresholds or {})}
    scenarios = {
        str(clip.get("scenario") or "unknown").lower()
        for case in cases
        for clip in case.get("clips", [])
    }
    camera_styles = {
        str(clip.get("camera_style") or "unknown").lower()
        for case in cases
        for clip in case.get("clips", [])
    }
    critical_switches = sum(
        int(clip.get("id_switches", 0))
        for case in cases
        for clip in case.get("clips", [])
        if clip.get("critical")
    )
    minimum_idf1 = min(float(case.get("idf1", 0.0)) for case in cases)
    minimum_hota = min(float(case.get("hota", 0.0)) for case in cases)
    cross_team_measured = all(
        (case.get("cross_team") or {}).get("status") == "measured"
        for case in cases
    )
    cross_team_switches = sum(
        int((case.get("cross_team") or {}).get("cross_team_switches") or 0)
        for case in cases
    )
    unresolved_fragments = sum(
        int((case.get("release_gate") or {}).get("unresolved_fragments") or 0)
        for case in cases
    )
    required_scenarios = {str(value).lower() for value in configured["required_scenarios"]}
    required_cameras = {str(value).lower() for value in configured["required_camera_styles"]}
    conditions = [
        {
            "code": "critical_id_switches",
            "passed": critical_switches <= int(configured["maximum_critical_id_switches"]),
            "actual": critical_switches,
        },
        {
            "code": "idf1",
            "passed": minimum_idf1 >= float(configured["minimum_idf1"]),
            "actual": minimum_idf1,
        },
        {
            "code": "hota",
            "passed": minimum_hota >= float(configured["minimum_hota"]),
            "actual": minimum_hota,
        },
        {
            "code": "cross_team_switches",
            "passed": cross_team_measured
            and cross_team_switches <= int(configured["maximum_cross_team_switches"]),
            "actual": cross_team_switches if cross_team_measured else None,
            "status": "measured" if cross_team_measured else "team_labels_required",
        },
        {
            "code": "fragment_review",
            "passed": unresolved_fragments == 0,
            "actual": unresolved_fragments,
        },
        {
            "code": "scenario_coverage",
            "passed": required_scenarios.issubset(scenarios),
            "missing": sorted(required_scenarios - scenarios),
        },
        {
            "code": "camera_coverage",
            "passed": required_cameras.issubset(camera_styles),
            "missing": sorted(required_cameras - camera_styles),
        },
    ]
    return {
        "status": "passed" if all(condition["passed"] for condition in conditions) else "blocked",
        "cases_count": len(cases),
        "thresholds": configured,
        "conditions": conditions,
        "coverage": {
            "scenarios": sorted(scenarios),
            "camera_styles": sorted(camera_styles),
        },
    }
