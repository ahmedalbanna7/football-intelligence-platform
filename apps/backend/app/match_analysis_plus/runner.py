from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter
from typing import Any
import io
import json
import shutil
import subprocess

import cv2
import numpy as np

from app.core.config import settings
from app.services.minio_client import client


PERSON_ALIASES = {"person", "player", "goalkeeper", "referee"}
BALL_ALIASES = {"sports ball", "ball"}


@dataclass
class AnalysisObject:
    track_id: int
    class_name: str
    bbox: list[float]
    confidence: float | None = None
    raw_track_id: int | None = None


@dataclass
class StableTrackState:
    stable_id: int
    bbox: list[float]
    center: tuple[float, float]
    velocity: tuple[float, float]
    last_frame: int
    raw_ids_seen: set[int]
    appearance_hist: np.ndarray | None = None
    jersey_color: tuple[int, int, int] | None = None
    hits: int = 1


class TrackIdStabilizer:
    def __init__(self, max_gap_frames: int = 240) -> None:
        self.max_gap_frames = max_gap_frames
        self.next_stable_id = 1
        self.tracks: dict[int, StableTrackState] = {}
        self.raw_to_stable: dict[int, int] = {}
        self.raw_ids_seen: set[int] = set()
        self.raw_id_reassignments = 0
        self.appearance_matches = 0
        self.rejected_far_matches = 0

    def update(self, frame_index: int, players: list[AnalysisObject], frame: np.ndarray | None = None) -> list[AnalysisObject]:
        used_stable_ids: set[int] = set()
        stabilized: list[AnalysisObject] = []
        ordered_players = sorted(
            players,
            key=lambda item: (-(item.confidence or 0.0), item.bbox[0], item.bbox[1]),
        )

        for player in ordered_players:
            raw_id = player.raw_track_id if player.raw_track_id is not None else player.track_id
            self.raw_ids_seen.add(raw_id)
            appearance_hist, jersey_color = self._extract_appearance(frame, player.bbox)
            stable_id = self._stable_from_raw(
                raw_id,
                player,
                frame_index,
                used_stable_ids,
                appearance_hist,
                jersey_color,
            )
            if stable_id is None:
                stable_id = self._match_existing(
                    player,
                    frame_index,
                    used_stable_ids,
                    appearance_hist,
                    jersey_color,
                )
            if stable_id is None:
                stable_id = self._create_track(player, frame_index, raw_id, appearance_hist, jersey_color)
            self._update_track(stable_id, player, frame_index, raw_id, appearance_hist, jersey_color)
            used_stable_ids.add(stable_id)
            stabilized.append(
                AnalysisObject(
                    track_id=stable_id,
                    class_name=player.class_name,
                    bbox=player.bbox,
                    confidence=player.confidence,
                    raw_track_id=raw_id,
                )
            )
        return stabilized

    def summary(self) -> dict[str, Any]:
        raw_ids_per_stable = {
            stable_id: len(state.raw_ids_seen)
            for stable_id, state in self.tracks.items()
        }
        stable_count = max(len(self.tracks), 1)
        raw_count = len(self.raw_ids_seen)
        return {
            "engine": "appearance_motion_stabilizer_v2",
            "raw_track_ids_seen": raw_count,
            "stable_tracks_count": len(self.tracks),
            "max_gap_frames": self.max_gap_frames,
            "raw_id_reassignments": self.raw_id_reassignments,
            "appearance_matches": self.appearance_matches,
            "rejected_far_matches": self.rejected_far_matches,
            "tracks_with_multiple_raw_ids": sum(1 for count in raw_ids_per_stable.values() if count > 1),
            "max_raw_ids_per_stable_track": max(raw_ids_per_stable.values(), default=0),
            "avg_raw_ids_per_stable_track": round(raw_count / stable_count, 3),
            "fragmentation_reduction_percent": round(max(0, raw_count - len(self.tracks)) * 100 / max(raw_count, 1), 2),
            "raw_ids_per_stable_track": raw_ids_per_stable,
        }

    def _stable_from_raw(
        self,
        raw_id: int,
        player: AnalysisObject,
        frame_index: int,
        used_stable_ids: set[int],
        appearance_hist: np.ndarray | None,
        jersey_color: tuple[int, int, int] | None,
    ) -> int | None:
        stable_id = self.raw_to_stable.get(raw_id)
        if stable_id is None or stable_id in used_stable_ids:
            return None
        state = self.tracks.get(stable_id)
        if state is None or frame_index - state.last_frame > self.max_gap_frames:
            return None
        if self._is_compatible(state, player, frame_index, appearance_hist, jersey_color):
            return stable_id
        self.raw_id_reassignments += 1
        return None

    def _match_existing(
        self,
        player: AnalysisObject,
        frame_index: int,
        used_stable_ids: set[int],
        appearance_hist: np.ndarray | None,
        jersey_color: tuple[int, int, int] | None,
    ) -> int | None:
        best: tuple[float, int] | None = None
        for stable_id, state in self.tracks.items():
            if stable_id in used_stable_ids:
                continue
            gap = frame_index - state.last_frame
            if gap < 0 or gap > self.max_gap_frames:
                continue
            iou = self._iou(player.bbox, state.bbox)
            predicted = self._predicted_center(state, gap)
            distance = self._center_distance(self._center(player.bbox), predicted)
            max_distance = self._max_center_distance(player.bbox, state.bbox, gap)
            appearance = self._appearance_similarity(appearance_hist, state.appearance_hist)
            color_similarity = self._color_similarity(jersey_color, state.jersey_color)
            far_distance_gate = max_distance * (2.6 if gap > 8 else 1.8)
            if iou <= 0.02 and distance > far_distance_gate:
                self.rejected_far_matches += 1
                continue
            if distance > max_distance and not (appearance >= 0.80 and color_similarity >= 0.72):
                continue
            position_score = max(0.0, 1.0 - (distance / max_distance))
            score = (
                (iou * 2.2)
                + (position_score * 3.0)
                + (appearance * 2.4)
                + (color_similarity * 1.2)
                - (gap * 0.0025)
            )
            if best is None or score > best[0]:
                best = (score, stable_id)
        if best is not None:
            state = self.tracks[best[1]]
            if self._appearance_similarity(appearance_hist, state.appearance_hist) >= 0.70:
                self.appearance_matches += 1
        return best[1] if best is not None else None

    def _create_track(
        self,
        player: AnalysisObject,
        frame_index: int,
        raw_id: int,
        appearance_hist: np.ndarray | None,
        jersey_color: tuple[int, int, int] | None,
    ) -> int:
        stable_id = self.next_stable_id
        self.next_stable_id += 1
        self.tracks[stable_id] = StableTrackState(
            stable_id=stable_id,
            bbox=player.bbox,
            center=self._center(player.bbox),
            velocity=(0.0, 0.0),
            last_frame=frame_index,
            raw_ids_seen={raw_id},
            appearance_hist=appearance_hist,
            jersey_color=jersey_color,
        )
        self.raw_to_stable[raw_id] = stable_id
        return stable_id

    def _update_track(
        self,
        stable_id: int,
        player: AnalysisObject,
        frame_index: int,
        raw_id: int,
        appearance_hist: np.ndarray | None,
        jersey_color: tuple[int, int, int] | None,
    ) -> None:
        state = self.tracks[stable_id]
        new_center = self._center(player.bbox)
        frame_delta = max(frame_index - state.last_frame, 1)
        instant_velocity = (
            (new_center[0] - state.center[0]) / frame_delta,
            (new_center[1] - state.center[1]) / frame_delta,
        )
        state.velocity = (
            state.velocity[0] * 0.55 + instant_velocity[0] * 0.45,
            state.velocity[1] * 0.55 + instant_velocity[1] * 0.45,
        )
        state.bbox = player.bbox
        state.center = new_center
        state.last_frame = frame_index
        state.raw_ids_seen.add(raw_id)
        state.hits += 1
        if appearance_hist is not None:
            if state.appearance_hist is None:
                state.appearance_hist = appearance_hist
            else:
                state.appearance_hist = self._normalize_hist(state.appearance_hist * 0.80 + appearance_hist * 0.20)
        if jersey_color is not None:
            if state.jersey_color is None:
                state.jersey_color = jersey_color
            else:
                state.jersey_color = tuple(
                    int(round(state.jersey_color[index] * 0.85 + jersey_color[index] * 0.15))
                    for index in range(3)
                )
        self.raw_to_stable[raw_id] = stable_id

    def _is_compatible(
        self,
        state: StableTrackState,
        player: AnalysisObject,
        frame_index: int,
        appearance_hist: np.ndarray | None,
        jersey_color: tuple[int, int, int] | None,
    ) -> bool:
        gap = max(frame_index - state.last_frame, 1)
        distance = self._center_distance(self._center(player.bbox), self._predicted_center(state, gap))
        iou = self._iou(player.bbox, state.bbox)
        appearance = self._appearance_similarity(appearance_hist, state.appearance_hist)
        color_similarity = self._color_similarity(jersey_color, state.jersey_color)
        max_distance = self._max_center_distance(player.bbox, state.bbox, gap)
        if iou <= 0.02 and distance > max_distance * 2.6:
            return False
        return (
            distance <= max_distance
            or iou > 0.05
            or (distance <= max_distance * 2.0 and appearance >= 0.80 and color_similarity >= 0.72)
        )

    def _max_center_distance(self, bbox_a: list[float], bbox_b: list[float], gap: int) -> float:
        size = max(self._bbox_size(bbox_a), self._bbox_size(bbox_b), 1.0)
        return max(65.0, min(320.0, size * 1.55 + min(gap, 45) * 5.0))

    def _predicted_center(self, state: StableTrackState, gap: int) -> tuple[float, float]:
        capped_gap = min(gap, 45)
        return (
            state.center[0] + state.velocity[0] * capped_gap,
            state.center[1] + state.velocity[1] * capped_gap,
        )

    def _extract_appearance(
        self,
        frame: np.ndarray | None,
        bbox: list[float],
    ) -> tuple[np.ndarray | None, tuple[int, int, int] | None]:
        if frame is None:
            return None, None
        height, width = frame.shape[:2]
        x1, y1, x2, y2 = [int(round(value)) for value in bbox]
        x1, x2 = max(0, x1), min(width, x2)
        y1, y2 = max(0, y1), min(height, y2)
        if x2 <= x1 or y2 <= y1:
            return None, None
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return None, None
        torso_y2 = max(1, int(crop.shape[0] * 0.65))
        torso = crop[:torso_y2, :]
        if torso.size == 0:
            return None, None
        torso = cv2.resize(torso, (32, 48), interpolation=cv2.INTER_AREA)
        hsv = cv2.cvtColor(torso, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, [16, 8], [0, 180, 0, 256]).astype(np.float32).flatten()
        hist = self._normalize_hist(hist)
        pixels = torso.reshape(-1, 3).astype(np.float32)
        jersey_color = None
        if len(pixels) >= 6:
            jersey_color = tuple(int(value) for value in np.median(pixels, axis=0))
        return hist, jersey_color

    def _normalize_hist(self, hist: np.ndarray) -> np.ndarray:
        total = float(np.linalg.norm(hist))
        if total <= 1e-6:
            return hist
        return hist / total

    def _appearance_similarity(self, a: np.ndarray | None, b: np.ndarray | None) -> float:
        if a is None or b is None:
            return 0.0
        return float(max(0.0, min(1.0, np.dot(a, b))))

    def _color_similarity(
        self,
        a: tuple[int, int, int] | None,
        b: tuple[int, int, int] | None,
    ) -> float:
        if a is None or b is None:
            return 0.0
        distance = float(np.linalg.norm(np.array(a, dtype=np.float32) - np.array(b, dtype=np.float32)))
        return max(0.0, 1.0 - distance / 255.0)

    def _bbox_size(self, bbox: list[float]) -> float:
        return max(bbox[2] - bbox[0], bbox[3] - bbox[1])

    def _center(self, bbox: list[float]) -> tuple[float, float]:
        return ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)

    def _center_distance(self, a: tuple[float, float], b: tuple[float, float]) -> float:
        return float(np.hypot(a[0] - b[0], a[1] - b[1]))

    def _iou(self, a: list[float], b: list[float]) -> float:
        x1 = max(a[0], b[0])
        y1 = max(a[1], b[1])
        x2 = min(a[2], b[2])
        y2 = min(a[3], b[3])
        intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
        area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
        union = area_a + area_b - intersection
        return intersection / union if union > 0 else 0.0


@dataclass
class BallStaticCandidate:
    center: tuple[float, float]
    first_frame: int
    last_frame: int
    hits: int = 1
    rejected_hits: int = 0


class BallStaticFilter:
    def __init__(self, cell_size: int = 24, static_hits: int = 5) -> None:
        self.cell_size = cell_size
        self.static_hits = static_hits
        self.candidates: dict[tuple[int, int], BallStaticCandidate] = {}
        self.raw_seen = 0
        self.kept = 0
        self.filtered_static = 0

    def filter(
        self,
        frame_index: int,
        balls: list[AnalysisObject],
        players: list[AnalysisObject],
        frame_width: int,
    ) -> list[AnalysisObject]:
        kept: list[AnalysisObject] = []
        for ball in balls:
            self.raw_seen += 1
            center = self._center(ball.bbox)
            if self._near_player_foot(center, players, frame_width):
                kept.append(ball)
                self.kept += 1
                self._remember(frame_index, center)
                continue

            candidate = self._remember(frame_index, center)
            if candidate.hits >= self.static_hits:
                candidate.rejected_hits += 1
                self.filtered_static += 1
                continue

            kept.append(ball)
            self.kept += 1
        return kept

    def summary(self) -> dict[str, Any]:
        return {
            "engine": "static_field_marker_filter_v1",
            "raw_ball_observations": self.raw_seen,
            "kept_ball_observations": self.kept,
            "filtered_static_candidates": self.filtered_static,
            "static_hits_threshold": self.static_hits,
        }

    def _remember(self, frame_index: int, center: tuple[float, float]) -> BallStaticCandidate:
        key = (round(center[0] / self.cell_size), round(center[1] / self.cell_size))
        candidate = self.candidates.get(key)
        if candidate is None:
            candidate = BallStaticCandidate(center=center, first_frame=frame_index, last_frame=frame_index)
            self.candidates[key] = candidate
            return candidate
        candidate.hits += 1
        candidate.last_frame = frame_index
        alpha = 1.0 / candidate.hits
        candidate.center = (
            candidate.center[0] * (1.0 - alpha) + center[0] * alpha,
            candidate.center[1] * (1.0 - alpha) + center[1] * alpha,
        )
        return candidate

    def _near_player_foot(
        self,
        center: tuple[float, float],
        players: list[AnalysisObject],
        frame_width: int,
    ) -> bool:
        threshold = max(42.0, frame_width * 0.032)
        for player in players:
            foot = ((player.bbox[0] + player.bbox[2]) / 2, player.bbox[3])
            if float(np.hypot(foot[0] - center[0], foot[1] - center[1])) <= threshold:
                return True
        return False

    def _center(self, bbox: list[float]) -> tuple[float, float]:
        return ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)


class MatchAnalysisPlusRunner:
    supported_modes = {
        "PLAYER_DETECTION",
        "BALL_DETECTION",
        "PLAYER_TRACKING",
        "TEAM_CLASSIFICATION",
        "RADAR",
    }

    def __init__(self) -> None:
        self.model = None
        self.model_path = self._resolve_model_path()

    def run(
        self,
        run_id: int,
        match_id: int,
        bucket: str,
        object_name: str,
        artifact_prefix: str,
        mode: str = "PLAYER_TRACKING",
        max_frames: int = 450,
    ) -> dict[str, Any]:
        normalized_mode = (mode or "PLAYER_TRACKING").upper()
        if normalized_mode not in self.supported_modes:
            raise ValueError(f"Unsupported match analysis mode: {mode}")

        with TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            input_path = temp_path / Path(object_name).name
            raw_output_path = temp_path / "match_analysis_plus.avi"
            output_path = temp_path / "match_analysis_plus.mp4"
            thumbnail_path = temp_path / "thumbnail.jpg"

            self._download_video(bucket, object_name, input_path)
            summary = self._process_video(
                input_path=input_path,
                raw_output_path=raw_output_path,
                output_path=output_path,
                thumbnail_path=thumbnail_path,
                mode=normalized_mode,
                max_frames=max_frames,
            )

            output_object = f"{artifact_prefix}/output.mp4"
            summary_object = f"{artifact_prefix}/summary.json"
            thumbnail_object = f"{artifact_prefix}/thumbnail.jpg"
            self._put_file(bucket, output_object, output_path, "video/mp4")
            if thumbnail_path.exists():
                self._put_file(bucket, thumbnail_object, thumbnail_path, "image/jpeg")

            payload = {
                **summary,
                "run_id": run_id,
                "match_id": match_id,
                "mode": normalized_mode,
                "input_object": object_name,
                "output_object": output_object,
                "summary_object": summary_object,
                "thumbnail_object": thumbnail_object if thumbnail_path.exists() else None,
                "source_project": "apps/match-analysis-worker/sports-main",
                "worker": "match-analysis-worker",
            }
            self._put_json(bucket, summary_object, payload)
            return payload

    def _process_video(
        self,
        input_path: Path,
        raw_output_path: Path,
        output_path: Path,
        thumbnail_path: Path,
        mode: str,
        max_frames: int,
    ) -> dict[str, Any]:
        start = perf_counter()
        model = self._load_model()
        capture = cv2.VideoCapture(str(input_path))
        if not capture.isOpened():
            raise ValueError("Could not open uploaded video for match analysis")

        fps = capture.get(cv2.CAP_PROP_FPS) or 24.0
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 1280)
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 720)
        writer = cv2.VideoWriter(
            str(raw_output_path),
            cv2.VideoWriter_fourcc(*"MJPG"),
            fps,
            (width, height),
        )
        if not writer.isOpened():
            capture.release()
            raise ValueError("Could not open match analysis video writer")

        last_positions: dict[int, tuple[float, float, int]] = {}
        track_distance: dict[int, float] = {}
        track_speed: dict[int, float] = {}
        track_frames: dict[int, int] = {}
        team_by_track: dict[int, int] = {}
        team_colors: dict[int, tuple[int, int, int]] = {}
        ball_control: list[int] = []
        class_counts: dict[str, int] = {}
        confidence_values: list[float] = []
        track_stabilizer = TrackIdStabilizer()
        ball_filter = BallStaticFilter()
        frames_processed = 0
        detections_count = 0
        raw_detections_count = 0

        while max_frames <= 0 or frames_processed < max_frames:
            ok, frame = capture.read()
            if not ok:
                break

            raw_objects = self._detect_and_track(model, frame, mode)
            raw_detections_count += len(raw_objects)
            raw_players = [item for item in raw_objects if item.class_name == "player"]
            raw_balls = [item for item in raw_objects if item.class_name == "ball"]
            players = track_stabilizer.update(frames_processed, raw_players, frame)
            balls = ball_filter.filter(frames_processed, raw_balls, players, width)
            objects = players + balls
            detections_count += len(objects)
            for item in objects:
                class_counts[item.class_name] = class_counts.get(item.class_name, 0) + 1
                if item.confidence is not None:
                    confidence_values.append(item.confidence)

            self._assign_teams(frame, players, team_by_track, team_colors)
            self._update_movement(
                players=players,
                frame_index=frames_processed,
                fps=fps,
                frame_width=width,
                last_positions=last_positions,
                track_distance=track_distance,
                track_speed=track_speed,
                track_frames=track_frames,
            )
            current_control = self._ball_control(players, balls, team_by_track, ball_control)
            if current_control is not None:
                ball_control.append(current_control)

            annotated = frame.copy()
            self._draw_overlay(
                annotated,
                players=players,
                balls=balls,
                team_by_track=team_by_track,
                team_colors=team_colors,
                track_distance=track_distance,
                track_speed=track_speed,
                ball_control=ball_control,
                mode=mode,
            )
            if frames_processed == 0:
                cv2.imwrite(str(thumbnail_path), annotated)
            writer.write(annotated)
            frames_processed += 1

        capture.release()
        writer.release()
        if frames_processed == 0:
            raise ValueError("No frames were processed by match analysis")
        output_codec = self._transcode_for_browser(raw_output_path, output_path)

        tracks = [
            {
                "track_id": track_id,
                "team": team_by_track.get(track_id),
                "frames": track_frames.get(track_id, 0),
                "distance_m": round(track_distance.get(track_id, 0.0), 2),
                "last_speed_kmh": round(track_speed.get(track_id, 0.0), 2),
                "raw_ids_count": len(track_stabilizer.tracks[track_id].raw_ids_seen)
                if track_id in track_stabilizer.tracks
                else 0,
                "raw_ids_seen": sorted(track_stabilizer.tracks[track_id].raw_ids_seen)
                if track_id in track_stabilizer.tracks
                else [],
            }
            for track_id in sorted(track_frames)
        ]
        team_1 = sum(1 for item in ball_control if item == 1)
        team_2 = sum(1 for item in ball_control if item == 2)
        total_control = max(team_1 + team_2, 1)
        elapsed_ms = round((perf_counter() - start) * 1000, 2)
        processing_fps = round(frames_processed / max(elapsed_ms / 1000, 0.001), 3)

        return {
            "status": "ok",
            "engine": "match_analysis_plus",
            "model": str(self.model_path),
            "model_mode": "sports-main-light-yolo",
            "output_codec": output_codec,
            "output_content_type": "video/mp4",
            "frames_processed": frames_processed,
            "max_frames": max_frames,
            "fps": round(float(fps), 3),
            "processing_fps": processing_fps,
            "resolution": [width, height],
            "detections_count": detections_count,
            "raw_detections_count": raw_detections_count,
            "class_counts": class_counts,
            "confidence": {
                "avg": round(float(np.mean(confidence_values)), 4) if confidence_values else None,
                "min": round(float(np.min(confidence_values)), 4) if confidence_values else None,
                "max": round(float(np.max(confidence_values)), 4) if confidence_values else None,
            },
            "tracks_count": len(tracks),
            "raw_tracks_count": len(track_stabilizer.raw_ids_seen),
            "id_stabilizer": track_stabilizer.summary(),
            "ball_filter": ball_filter.summary(),
            "tracks": tracks[:250],
            "team_ball_control": {
                "team_1_percent": round(team_1 * 100 / total_control, 2),
                "team_2_percent": round(team_2 * 100 / total_control, 2),
            },
            "notes": [
                "sports-main source is vendored in apps/match-analysis-worker/sports-main",
                "specialized sports-main models are optional; this run uses the local YOLO model by default",
            ],
            "elapsed_ms": elapsed_ms,
        }

    def _detect_and_track(self, model: Any, frame: np.ndarray, mode: str) -> list[AnalysisObject]:
        if mode == "BALL_DETECTION":
            classes = [32]
        elif mode == "PLAYER_DETECTION":
            classes = [0]
        else:
            classes = None

        results = model.track(
            frame,
            persist=True,
            conf=max(settings.YOLO_CONFIDENCE, 0.15),
            imgsz=settings.YOLO_IMAGE_SIZE,
            device=settings.YOLO_DEVICE,
            verbose=False,
            tracker="bytetrack.yaml",
            classes=classes,
        )
        if not results:
            return []

        result = results[0]
        names = result.names or {}
        boxes = result.boxes
        if boxes is None or boxes.xyxy is None:
            return []

        xyxy = boxes.xyxy.cpu().numpy()
        cls = boxes.cls.cpu().numpy() if boxes.cls is not None else np.zeros(len(xyxy))
        conf = boxes.conf.cpu().numpy() if boxes.conf is not None else [None] * len(xyxy)
        ids = boxes.id.cpu().numpy() if boxes.id is not None else np.arange(1, len(xyxy) + 1)

        objects: list[AnalysisObject] = []
        for index, bbox in enumerate(xyxy):
            raw_name = str(names.get(int(cls[index]), int(cls[index]))).lower()
            class_name = self._map_class_name(raw_name)
            if class_name is None:
                continue
            objects.append(
                AnalysisObject(
                    track_id=int(ids[index]),
                    class_name=class_name,
                    bbox=[float(value) for value in bbox.tolist()],
                    confidence=float(conf[index]) if conf[index] is not None else None,
                    raw_track_id=int(ids[index]),
                )
            )
        return objects

    def _assign_teams(
        self,
        frame: np.ndarray,
        players: list[AnalysisObject],
        team_by_track: dict[int, int],
        team_colors: dict[int, tuple[int, int, int]],
    ) -> None:
        for player in players:
            if player.track_id in team_by_track:
                continue
            color = self._dominant_jersey_color(frame, player.bbox)
            if color is None:
                team_by_track[player.track_id] = 1
                continue
            if not team_colors:
                team_colors[1] = color
                team_by_track[player.track_id] = 1
                continue
            if len(team_colors) == 1:
                distance = np.linalg.norm(np.array(team_colors[1]) - np.array(color))
                if distance > 80:
                    team_colors[2] = color
                    team_by_track[player.track_id] = 2
                else:
                    team_by_track[player.track_id] = 1
                continue
            distances = {
                team: np.linalg.norm(np.array(team_color) - np.array(color))
                for team, team_color in team_colors.items()
            }
            team_by_track[player.track_id] = min(distances, key=distances.get)

    def _dominant_jersey_color(
        self,
        frame: np.ndarray,
        bbox: list[float],
    ) -> tuple[int, int, int] | None:
        height, width = frame.shape[:2]
        x1, y1, x2, y2 = [int(round(value)) for value in bbox]
        x1, x2 = max(0, x1), min(width, x2)
        y1, y2 = max(0, y1), min(height, y2)
        if x2 <= x1 or y2 <= y1:
            return None
        crop = frame[y1:y2, x1:x2]
        torso = crop[: max(1, crop.shape[0] // 2), :]
        if torso.size == 0:
            return None
        pixels = torso.reshape(-1, 3).astype(np.float32)
        if len(pixels) < 6:
            return None
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
        _, labels, centers = cv2.kmeans(pixels, 2, None, criteria, 3, cv2.KMEANS_PP_CENTERS)
        counts = np.bincount(labels.flatten())
        center = centers[int(np.argmax(counts))]
        return tuple(int(value) for value in center)

    def _update_movement(
        self,
        players: list[AnalysisObject],
        frame_index: int,
        fps: float,
        frame_width: int,
        last_positions: dict[int, tuple[float, float, int]],
        track_distance: dict[int, float],
        track_speed: dict[int, float],
        track_frames: dict[int, int],
    ) -> None:
        meters_per_pixel = 68.0 / max(frame_width, 1)
        for player in players:
            foot = ((player.bbox[0] + player.bbox[2]) / 2, player.bbox[3])
            previous = last_positions.get(player.track_id)
            track_frames[player.track_id] = track_frames.get(player.track_id, 0) + 1
            if previous is not None:
                distance = float(np.hypot(foot[0] - previous[0], foot[1] - previous[1]) * meters_per_pixel)
                frame_delta = max(frame_index - previous[2], 1)
                if distance < 15:
                    track_distance[player.track_id] = track_distance.get(player.track_id, 0.0) + distance
                    track_speed[player.track_id] = distance / (frame_delta / fps) * 3.6
            last_positions[player.track_id] = (foot[0], foot[1], frame_index)

    def _ball_control(
        self,
        players: list[AnalysisObject],
        balls: list[AnalysisObject],
        team_by_track: dict[int, int],
        ball_control: list[int],
    ) -> int | None:
        if not players or not balls:
            return ball_control[-1] if ball_control else None
        ball = balls[0]
        ball_center = ((ball.bbox[0] + ball.bbox[2]) / 2, (ball.bbox[1] + ball.bbox[3]) / 2)
        nearest: tuple[float, AnalysisObject] | None = None
        for player in players:
            foot = ((player.bbox[0] + player.bbox[2]) / 2, player.bbox[3])
            distance = float(np.hypot(foot[0] - ball_center[0], foot[1] - ball_center[1]))
            if nearest is None or distance < nearest[0]:
                nearest = (distance, player)
        if nearest is None or nearest[0] > 90:
            return ball_control[-1] if ball_control else None
        return team_by_track.get(nearest[1].track_id, 1)

    def _draw_overlay(
        self,
        frame: np.ndarray,
        players: list[AnalysisObject],
        balls: list[AnalysisObject],
        team_by_track: dict[int, int],
        team_colors: dict[int, tuple[int, int, int]],
        track_distance: dict[int, float],
        track_speed: dict[int, float],
        ball_control: list[int],
        mode: str,
    ) -> None:
        for player in players:
            team = team_by_track.get(player.track_id, 1)
            color = team_colors.get(team, (80, 220, 80) if team == 2 else (240, 240, 240))
            self._draw_player(frame, player, color, track_distance, track_speed)
        for ball in balls:
            self._draw_triangle(frame, ball.bbox, (0, 255, 255))
        self._draw_header(frame, mode)
        self._draw_ball_control(frame, ball_control)

    def _draw_player(
        self,
        frame: np.ndarray,
        player: AnalysisObject,
        color: tuple[int, int, int],
        track_distance: dict[int, float],
        track_speed: dict[int, float],
    ) -> None:
        x1, y1, x2, y2 = [int(round(value)) for value in player.bbox]
        center_x = int((x1 + x2) / 2)
        cv2.ellipse(frame, (center_x, y2), (max(10, (x2 - x1) // 2), 8), 0, -45, 235, color, 2)
        scale = self._font_scale(frame, 0.5)
        small = self._font_scale(frame, 0.38)
        thickness = self._thickness(frame)
        cv2.rectangle(frame, (center_x - 14, y2 + 4), (center_x + 14, y2 + 21), (245, 250, 245), cv2.FILLED)
        cv2.putText(frame, str(player.track_id), (center_x - 9, y2 + 18), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thickness)
        distance = track_distance.get(player.track_id)
        speed = track_speed.get(player.track_id)
        if distance is not None:
            cv2.putText(frame, f"{distance:.1f}m", (center_x - 17, y2 + 36), cv2.FONT_HERSHEY_SIMPLEX, small, (0, 0, 0), thickness)
        if speed is not None:
            cv2.putText(frame, f"{speed:.1f}km/h", (center_x - 24, y2 + 50), cv2.FONT_HERSHEY_SIMPLEX, small, (0, 0, 0), thickness)

    def _draw_triangle(self, frame: np.ndarray, bbox: list[float], color: tuple[int, int, int]) -> None:
        x = int((bbox[0] + bbox[2]) / 2)
        y = int(bbox[1])
        points = np.array([[x, y], [x - 9, y - 18], [x + 9, y - 18]])
        cv2.drawContours(frame, [points], 0, color, cv2.FILLED)
        cv2.drawContours(frame, [points], 0, (0, 0, 0), 1)

    def _draw_header(self, frame: np.ndarray, mode: str) -> None:
        scale = self._font_scale(frame, 0.48)
        thickness = self._thickness(frame)
        cv2.putText(frame, f"Match Analysis +  {mode}", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), thickness + 2)
        cv2.putText(frame, f"Match Analysis +  {mode}", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thickness)

    def _draw_ball_control(self, frame: np.ndarray, ball_control: list[int]) -> None:
        height, width = frame.shape[:2]
        scale = self._font_scale(frame, 0.44)
        thickness = self._thickness(frame)
        team_1 = sum(1 for item in ball_control if item == 1)
        team_2 = sum(1 for item in ball_control if item == 2)
        total = max(team_1 + team_2, 1)
        text = f"T1 {team_1 * 100 / total:.1f}%  T2 {team_2 * 100 / total:.1f}%"
        cv2.putText(frame, text, (max(10, width - 250), height - 18), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), thickness + 2)
        cv2.putText(frame, text, (max(10, width - 250), height - 18), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thickness)

    def _ui_scale(self, frame: np.ndarray) -> float:
        height, width = frame.shape[:2]
        return max(0.45, min(width / 1920, height / 1080))

    def _font_scale(self, frame: np.ndarray, base: float) -> float:
        return max(0.28, base * self._ui_scale(frame))

    def _thickness(self, frame: np.ndarray) -> int:
        return max(1, int(round(2 * self._ui_scale(frame))))

    def _map_class_name(self, raw_name: str) -> str | None:
        if raw_name in PERSON_ALIASES:
            return "player"
        if raw_name in BALL_ALIASES:
            return "ball"
        return None

    def _load_model(self) -> Any:
        if self.model is not None:
            return self.model
        from ultralytics import YOLO

        self.model = YOLO(str(self.model_path))
        return self.model

    def _resolve_model_path(self) -> Path:
        configured = Path(settings.YOLO_MODEL_PATH)
        if configured.exists():
            return configured
        return Path("yolo11n.pt")

    def _transcode_for_browser(self, source_path: Path, output_path: Path) -> str:
        ffmpeg = shutil.which("ffmpeg")
        if ffmpeg is None:
            source_path.replace(output_path)
            return "mjpg"
        command = [
            ffmpeg,
            "-y",
            "-i",
            str(source_path),
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "24",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(output_path),
        ]
        completed = subprocess.run(command, capture_output=True, text=True, check=False)
        if completed.returncode != 0 or not output_path.exists() or output_path.stat().st_size <= 1024:
            source_path.replace(output_path)
            return "mjpg"
        return "h264"

    def _download_video(self, bucket: str, object_name: str, local_path: Path) -> None:
        response = client.get_object(bucket, object_name)
        try:
            with local_path.open("wb") as file:
                for chunk in response.stream(1024 * 1024):
                    file.write(chunk)
        finally:
            response.close()
            response.release_conn()

    def _put_file(self, bucket: str, object_name: str, path: Path, content_type: str) -> None:
        with path.open("rb") as file:
            data = file.read()
        client.put_object(bucket, object_name, io.BytesIO(data), length=len(data), content_type=content_type)

    def _put_json(self, bucket: str, object_name: str, payload: dict[str, Any]) -> None:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        client.put_object(bucket, object_name, io.BytesIO(data), length=len(data), content_type="application/json")
