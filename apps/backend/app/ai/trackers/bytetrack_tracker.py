from types import SimpleNamespace
from typing import Any

import numpy as np

from app.core.config import settings


class TrackerResults:
    def __init__(
        self,
        detections: list[dict[str, Any]],
    ) -> None:
        self.detections = detections
        self.xywh = np.asarray(
            [
                self._xywh(detection)
                for detection in detections
            ],
            dtype=np.float32,
        ).reshape((-1, 4))
        self.conf = np.asarray(
            [
                float(detection.get("confidence") or 0)
                for detection in detections
            ],
            dtype=np.float32,
        )
        self.cls = np.asarray(
            [
                float(detection.get("class_id") or 0)
                for detection in detections
            ],
            dtype=np.float32,
        )

    def __len__(self) -> int:
        return len(self.detections)

    def __getitem__(self, index):
        if isinstance(index, np.ndarray):
            if index.dtype == bool:
                detections = [
                    detection
                    for detection, keep in zip(self.detections, index, strict=False)
                    if keep
                ]
            else:
                detections = [
                    self.detections[int(item)]
                    for item in index
                ]
            return TrackerResults(detections)
        if isinstance(index, slice):
            return TrackerResults(self.detections[index])
        return TrackerResults([self.detections[int(index)]])

    def _xywh(self, detection: dict[str, Any]) -> list[float]:
        bbox_xywh = detection.get("bbox_xywh")
        if bbox_xywh and len(bbox_xywh) == 4:
            return [
                float(value)
                for value in bbox_xywh
            ]

        bbox_xyxy = detection.get("bbox_xyxy") or [0, 0, 0, 0]
        x1, y1, x2, y2 = [
            float(value)
            for value in bbox_xyxy
        ]
        return [
            x1,
            y1,
            max(x2 - x1, 0.0),
            max(y2 - y1, 0.0),
        ]


class ByteTrackTracker:
    name = "ultralytics_bytetrack"
    fallback_name = "iou_center_linker"

    def track(self, detections: list[dict[str, Any]]) -> dict[str, Any]:
        if not detections:
            return self._result([], 0, mode="empty")

        if any(detection.get("stub_object_id") for detection in detections):
            return self._track_stub_objects(detections)

        real_result = self._track_with_ultralytics_bytetrack(detections)
        if real_result is not None:
            return real_result

        return self._track_by_iou_and_center(detections)

    def _track_with_ultralytics_bytetrack(
        self,
        detections: list[dict[str, Any]],
    ) -> dict[str, Any] | None:
        try:
            from ultralytics.trackers.byte_tracker import BYTETracker
        except Exception:
            return None

        args = SimpleNamespace(
            track_high_thresh=settings.TRACKER_HIGH_THRESH,
            track_low_thresh=settings.TRACKER_LOW_THRESH,
            new_track_thresh=settings.TRACKER_NEW_TRACK_THRESH,
            track_buffer=settings.TRACKER_BUFFER,
            match_thresh=settings.TRACKER_MATCH_THRESH,
            fuse_score=True,
        )
        tracker = BYTETracker(args)
        tracks_by_id: dict[int, dict[str, Any]] = {}
        detections_by_frame = self._detections_by_frame(detections)

        try:
            for frame_index in sorted(detections_by_frame):
                frame_detections = detections_by_frame[frame_index]
                tracker_results = TrackerResults(frame_detections)
                output = tracker.update(tracker_results)
                for row in output:
                    self._append_bytetrack_row(
                        tracks_by_id,
                        frame_detections,
                        frame_index,
                        row,
                    )
        except Exception:
            return None

        tracks = self._postprocess_tracks(list(tracks_by_id.values()))
        return self._result(
            tracks,
            detections_received=len(detections),
            mode="ultralytics_bytetrack",
            track_key="byte_track_id",
        )

    def _detections_by_frame(
        self,
        detections: list[dict[str, Any]],
    ) -> dict[int, list[dict[str, Any]]]:
        detections_by_frame: dict[int, list[dict[str, Any]]] = {}
        for detection in detections:
            frame_index = int(detection.get("frame_index") or 0)
            detections_by_frame.setdefault(frame_index, []).append(detection)
        return detections_by_frame

    def _append_bytetrack_row(
        self,
        tracks_by_id: dict[int, dict[str, Any]],
        frame_detections: list[dict[str, Any]],
        frame_index: int,
        row,
    ) -> None:
        x1, y1, x2, y2, track_id, score, class_id, _ = [
            float(value)
            for value in row[:8]
        ]
        track_id_int = int(track_id)
        matched_detection = self._nearest_detection(
            [x1, y1, x2, y2],
            frame_detections,
            int(class_id),
        )
        class_name = (
            matched_detection.get("class_name")
            if matched_detection
            else self._class_name_from_id(int(class_id))
        )
        track = tracks_by_id.setdefault(
            track_id_int,
            {
                "track_id": track_id_int,
                "object_key": f"{class_name}-{track_id_int}",
                "class_name": class_name,
                "frames": [],
                "first_frame": frame_index,
                "source_tracker": self.name,
            },
        )
        bbox_xyxy = [
            round(x1, 2),
            round(y1, 2),
            round(x2, 2),
            round(y2, 2),
        ]
        center = [
            round((x1 + x2) / 2, 2),
            round((y1 + y2) / 2, 2),
        ]
        area = round(max(x2 - x1, 0) * max(y2 - y1, 0), 2)
        timestamp_ms = (
            matched_detection.get("timestamp_ms")
            if matched_detection
            else None
        )
        timestamp_seconds = (
            matched_detection.get("timestamp_seconds")
            if matched_detection
            else None
        )
        frame = {
            "frame_index": frame_index,
            "timestamp_seconds": timestamp_seconds,
            "timestamp_ms": timestamp_ms,
            "bbox_xyxy": bbox_xyxy,
            "bbox_center": center,
            "center": center,
            "bbox_area": area,
            "confidence": round(float(score), 6),
            "class_name": class_name,
            "raw_class_name": (
                matched_detection.get("raw_class_name")
                if matched_detection
                else class_name
            ),
            "track_state": "active",
            "source_tracker": self.name,
        }
        track["frames"].append(frame)
        track["last_frame"] = frame_index
        track["average_confidence"] = self._average_confidence(track["frames"])

    def _nearest_detection(
        self,
        bbox_xyxy: list[float],
        frame_detections: list[dict[str, Any]],
        class_id: int,
    ) -> dict[str, Any] | None:
        same_class = [
            detection
            for detection in frame_detections
            if int(detection.get("class_id") or 0) == class_id
        ]
        candidates = same_class or frame_detections
        if not candidates:
            return None
        return max(
            candidates,
            key=lambda detection: self._iou(
                bbox_xyxy,
                detection.get("bbox_xyxy") or [],
            ),
        )

    def _class_name_from_id(self, class_id: int) -> str:
        if class_id == 1:
            return "ball"
        return "player"

    def _track_stub_objects(self, detections: list[dict[str, Any]]) -> dict[str, Any]:
        tracks_by_object: dict[str, dict[str, Any]] = {}

        for detection in detections:
            object_key = detection.get("stub_object_id") or detection["id"]
            track = tracks_by_object.setdefault(
                object_key,
                {
                    "track_id": len(tracks_by_object) + 1,
                    "object_key": object_key,
                    "class_name": detection.get("class_name"),
                    "frames": [],
                },
            )
            track["frames"].append(
                {
                    "frame_index": detection.get("frame_index"),
                    "timestamp_seconds": detection.get("timestamp_seconds"),
                    "timestamp_ms": detection.get("timestamp_ms"),
                    "bbox_xyxy": detection.get("bbox_xyxy"),
                    "bbox_center": detection.get("bbox_center") or detection.get("center"),
                    "center": detection.get("bbox_center") or detection.get("center"),
                    "bbox_area": detection.get("bbox_area") or detection.get("area"),
                    "confidence": detection.get("confidence"),
                    "class_name": detection.get("class_name"),
                    "track_state": "active",
                    "source_tracker": self.name,
                }
            )

        tracks = list(tracks_by_object.values())
        return self._result(
            tracks,
            detections_received=len(detections),
            mode="deterministic_stub",
            track_key="stub_object_id",
        )

    def _track_by_iou_and_center(
        self,
        detections: list[dict[str, Any]],
    ) -> dict[str, Any]:
        tracks: list[dict[str, Any]] = []
        active_tracks: list[dict[str, Any]] = []
        next_track_id = 1

        detections_by_frame: dict[int, list[dict[str, Any]]] = {}
        for detection in detections:
            frame_index = int(detection.get("frame_index") or 0)
            detections_by_frame.setdefault(frame_index, []).append(detection)

        for frame_index in sorted(detections_by_frame):
            frame_detections = sorted(
                detections_by_frame[frame_index],
                key=lambda item: float(item.get("confidence") or 0),
                reverse=True,
            )
            used_track_ids: set[int] = set()
            active_tracks = [
                track
                for track in active_tracks
                if frame_index - int(track.get("last_frame") or frame_index) <= 120
            ]

            for detection in frame_detections:
                track = self._best_track_match(
                    detection,
                    active_tracks,
                    used_track_ids,
                    frame_index,
                )
                if track is None:
                    track = {
                        "track_id": next_track_id,
                        "object_key": f"{detection.get('class_name', 'object')}-{next_track_id}",
                        "class_name": detection.get("class_name"),
                        "frames": [],
                        "first_frame": frame_index,
                    }
                    next_track_id += 1
                    tracks.append(track)
                    active_tracks.append(track)

                self._append_detection_to_track(track, detection, frame_index)
                used_track_ids.add(track["track_id"])

        return self._result(
            tracks,
            detections_received=len(detections),
            mode="iou_center_linker",
            track_key="class_name+iou+center_distance",
        )

    def _best_track_match(
        self,
        detection: dict[str, Any],
        tracks: list[dict[str, Any]],
        used_track_ids: set[int],
        frame_index: int,
    ) -> dict[str, Any] | None:
        best_track = None
        best_score = -1.0
        detection_class = detection.get("class_name")
        detection_bbox = detection.get("bbox_xyxy") or []
        detection_center = detection.get("center") or [0, 0]

        for track in tracks:
            if track["track_id"] in used_track_ids:
                continue
            if track.get("class_name") != detection_class:
                continue

            frame_gap = frame_index - int(track.get("last_frame") or frame_index)
            if frame_gap < 0 or frame_gap > 120:
                continue

            last_bbox = track.get("last_bbox_xyxy") or []
            last_center = track.get("last_center") or [0, 0]
            iou = self._iou(detection_bbox, last_bbox)
            distance = self._center_distance(detection_center, last_center)
            max_distance = self._max_center_distance(detection_class, detection_bbox)

            if iou < 0.08 and distance > max_distance:
                continue

            distance_score = max(0.0, 1 - (distance / max_distance))
            score = (iou * 0.7) + (distance_score * 0.3)
            if score > best_score:
                best_score = score
                best_track = track

        return best_track

    def _append_detection_to_track(
        self,
        track: dict[str, Any],
        detection: dict[str, Any],
        frame_index: int,
    ) -> None:
        frame = {
            "frame_index": detection.get("frame_index"),
            "timestamp_seconds": detection.get("timestamp_seconds"),
            "timestamp_ms": detection.get("timestamp_ms"),
            "bbox_xyxy": detection.get("bbox_xyxy"),
            "bbox_center": detection.get("bbox_center") or detection.get("center"),
            "center": detection.get("bbox_center") or detection.get("center"),
            "bbox_area": detection.get("bbox_area") or detection.get("area"),
            "confidence": detection.get("confidence"),
            "class_name": detection.get("class_name"),
            "raw_class_name": detection.get("raw_class_name"),
            "track_state": "active",
            "source_tracker": self.name,
        }
        track["frames"].append(frame)
        track["last_frame"] = frame_index
        track["last_bbox_xyxy"] = detection.get("bbox_xyxy")
        track["last_center"] = detection.get("center")
        track["average_confidence"] = self._average_confidence(track["frames"])

    def _postprocess_tracks(
        self,
        tracks: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        merged = self._merge_tracklets(tracks)
        return [
            track
            for track in merged
            if self._keep_track(track)
        ]

    def _merge_tracklets(
        self,
        tracks: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        ordered = sorted(
            tracks,
            key=lambda track: (
                track.get("class_name") or "",
                int(track.get("first_frame") or 0),
                int(track.get("track_id") or 0),
            ),
        )
        merged: list[dict[str, Any]] = []

        for track in ordered:
            target = self._merge_target(track, merged)
            if target is None:
                merged.append(track)
                continue

            target["frames"].extend(track.get("frames", []))
            target["frames"] = sorted(
                target["frames"],
                key=lambda frame: int(frame.get("frame_index") or 0),
            )
            target["last_frame"] = target["frames"][-1].get("frame_index")
            target["average_confidence"] = self._average_confidence(target["frames"])
            target["merged_track_ids"] = sorted(
                set(target.get("merged_track_ids", [target["track_id"]]))
                | set(track.get("merged_track_ids", [track["track_id"]])),
            )

        for new_id, track in enumerate(merged, start=1):
            track["original_track_id"] = track.get("track_id")
            track["track_id"] = new_id
            track["object_key"] = f"{track.get('class_name', 'object')}-{new_id}"
        return merged

    def _merge_target(
        self,
        track: dict[str, Any],
        candidates: list[dict[str, Any]],
    ) -> dict[str, Any] | None:
        if track.get("class_name") == "ball":
            return None
        frames = track.get("frames", [])
        if not frames:
            return None
        first_frame = int(frames[0].get("frame_index") or 0)
        first_center = frames[0].get("bbox_center") or frames[0].get("center")

        best_candidate = None
        best_distance = settings.TRACKER_MERGE_DISTANCE
        for candidate in candidates:
            if candidate.get("class_name") != track.get("class_name"):
                continue
            candidate_frames = candidate.get("frames", [])
            if not candidate_frames:
                continue
            last_frame = int(candidate_frames[-1].get("frame_index") or 0)
            gap = first_frame - last_frame
            if gap < 0 or gap > settings.TRACKER_MERGE_GAP_FRAMES:
                continue
            last_center = (
                candidate_frames[-1].get("bbox_center")
                or candidate_frames[-1].get("center")
            )
            distance = self._center_distance(first_center or [], last_center or [])
            if distance < best_distance:
                best_distance = distance
                best_candidate = candidate
        return best_candidate

    def _keep_track(self, track: dict[str, Any]) -> bool:
        if track.get("class_name") == "ball":
            return True
        return len(track.get("frames", [])) >= settings.TRACKER_MIN_PLAYER_OBSERVATIONS

    def _result(
        self,
        tracks: list[dict[str, Any]],
        detections_received: int,
        mode: str,
        track_key: str | None = None,
    ) -> dict[str, Any]:
        for track in tracks:
            track.pop("last_bbox_xyxy", None)
            track.pop("last_center", None)
            if "first_frame" not in track and track.get("frames"):
                track["first_frame"] = track["frames"][0].get("frame_index")
            if "last_frame" not in track and track.get("frames"):
                track["last_frame"] = track["frames"][-1].get("frame_index")

        return {
            "status": "ok",
            "data": {
                "tracks": tracks,
                "tracks_count": len(tracks),
                "track_observations_count": sum(
                    len(track.get("frames", []))
                    for track in tracks
                ),
                "detections_received": detections_received,
            },
            "meta": {
                "engine": self.name,
                "mode": mode,
                "track_key": track_key,
                "note": "Temporary lightweight tracker until full ByteTrack/DeepSORT integration.",
            },
        }

    def _iou(self, first: list[float], second: list[float]) -> float:
        if len(first) != 4 or len(second) != 4:
            return 0.0

        x1 = max(float(first[0]), float(second[0]))
        y1 = max(float(first[1]), float(second[1]))
        x2 = min(float(first[2]), float(second[2]))
        y2 = min(float(first[3]), float(second[3]))
        intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        if intersection == 0:
            return 0.0

        first_area = max(0.0, float(first[2]) - float(first[0])) * max(
            0.0,
            float(first[3]) - float(first[1]),
        )
        second_area = max(0.0, float(second[2]) - float(second[0])) * max(
            0.0,
            float(second[3]) - float(second[1]),
        )
        union = first_area + second_area - intersection
        return intersection / union if union else 0.0

    def _center_distance(
        self,
        first: list[float],
        second: list[float],
    ) -> float:
        if len(first) != 2 or len(second) != 2:
            return 10_000.0
        dx = float(first[0]) - float(second[0])
        dy = float(first[1]) - float(second[1])
        return (dx**2 + dy**2) ** 0.5

    def _max_center_distance(
        self,
        class_name: str | None,
        bbox_xyxy: list[float],
    ) -> float:
        if class_name == "ball":
            return 180.0
        if len(bbox_xyxy) != 4:
            return 140.0

        width = max(1.0, float(bbox_xyxy[2]) - float(bbox_xyxy[0]))
        height = max(1.0, float(bbox_xyxy[3]) - float(bbox_xyxy[1]))
        return max(90.0, min(260.0, ((width**2 + height**2) ** 0.5) * 1.6))

    def _average_confidence(self, frames: list[dict[str, Any]]) -> float | None:
        values = [
            frame.get("confidence")
            for frame in frames
            if frame.get("confidence") is not None
        ]
        if not values:
            return None
        return round(sum(values) / len(values), 6)
