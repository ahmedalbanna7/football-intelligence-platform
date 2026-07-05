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
class FrameObject:
    track_id: int
    class_name: str
    bbox: list[float]
    confidence: float | None = None


class SimpleTeamAssigner:
    def __init__(self) -> None:
        self.team_centers: np.ndarray | None = None
        self.player_team: dict[int, int] = {}
        self.seed_colors: list[np.ndarray] = []
        self.team_colors_bgr: dict[int, tuple[int, int, int]] = {
            1: (220, 220, 220),
            2: (80, 220, 80),
        }

    def get_player_color(self, frame: np.ndarray, bbox: list[float]) -> np.ndarray | None:
        height, width = frame.shape[:2]
        x1, y1, x2, y2 = [int(round(value)) for value in bbox]
        x1, x2 = max(0, x1), min(width, x2)
        y1, y2 = max(0, y1), min(height, y2)
        if x2 <= x1 or y2 <= y1:
            return None

        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return None
        top_half = crop[: max(1, crop.shape[0] // 2), :]
        if top_half.size == 0:
            return None

        pixels = top_half.reshape((-1, 3)).astype(np.float32)
        if len(pixels) < 4:
            return None

        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
        _, labels, centers = cv2.kmeans(
            pixels,
            2,
            None,
            criteria,
            3,
            cv2.KMEANS_PP_CENTERS,
        )
        clustered = labels.reshape(top_half.shape[:2])
        corners = [
            int(clustered[0, 0]),
            int(clustered[0, -1]),
            int(clustered[-1, 0]),
            int(clustered[-1, -1]),
        ]
        background_cluster = max(set(corners), key=corners.count)
        player_cluster = 1 - background_cluster
        return centers[player_cluster]

    def warmup(self, frame: np.ndarray, players: list[FrameObject]) -> None:
        if self.team_centers is not None:
            return
        for player in players:
            color = self.get_player_color(frame, player.bbox)
            if color is not None:
                self.seed_colors.append(color)
            if len(self.seed_colors) >= 8:
                break
        if len(self.seed_colors) < 2:
            return

        samples = np.array(self.seed_colors, dtype=np.float32)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 1.0)
        _, _, centers = cv2.kmeans(
            samples,
            2,
            None,
            criteria,
            10,
            cv2.KMEANS_PP_CENTERS,
        )
        self.team_centers = centers
        self.team_colors_bgr = {
            1: tuple(int(value) for value in centers[0]),
            2: tuple(int(value) for value in centers[1]),
        }

    def assign(self, frame: np.ndarray, player: FrameObject) -> int:
        if player.track_id in self.player_team:
            return self.player_team[player.track_id]
        if self.team_centers is None:
            return 1

        color = self.get_player_color(frame, player.bbox)
        if color is None:
            return 1

        distances = np.linalg.norm(self.team_centers - color.reshape(1, 3), axis=1)
        team = int(np.argmin(distances)) + 1
        self.player_team[player.track_id] = team
        return team

    def color_for(self, team: int) -> tuple[int, int, int]:
        color = self.team_colors_bgr.get(team, (220, 220, 220))
        return tuple(int(value) for value in color)


class CameraMovementEstimator:
    def __init__(self) -> None:
        self.previous_gray: np.ndarray | None = None
        self.previous_features: np.ndarray | None = None
        self.cumulative = np.array([0.0, 0.0], dtype=np.float32)
        self.lk_params = {
            "winSize": (15, 15),
            "maxLevel": 2,
            "criteria": (
                cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                10,
                0.03,
            ),
        }

    def update(self, frame: np.ndarray) -> tuple[float, float]:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if self.previous_gray is None:
            self.previous_gray = gray
            self.previous_features = self._features(gray)
            return 0.0, 0.0

        if self.previous_features is None or len(self.previous_features) == 0:
            self.previous_features = self._features(self.previous_gray)
            self.previous_gray = gray
            return 0.0, 0.0

        next_features, status, _ = cv2.calcOpticalFlowPyrLK(
            self.previous_gray,
            gray,
            self.previous_features,
            None,
            **self.lk_params,
        )
        movement = np.array([0.0, 0.0], dtype=np.float32)
        if next_features is not None and status is not None:
            good_new = next_features[status.flatten() == 1]
            good_old = self.previous_features[status.flatten() == 1]
            if len(good_new) >= 4:
                deltas = (good_new - good_old).reshape(-1, 2)
                median = np.median(deltas, axis=0)
                if float(np.linalg.norm(median)) > 1.5:
                    movement = median.astype(np.float32)
                    self.cumulative += movement

        self.previous_gray = gray
        self.previous_features = self._features(gray)
        return float(movement[0]), float(movement[1])

    def adjust(self, point: tuple[float, float]) -> tuple[float, float]:
        return float(point[0] - self.cumulative[0]), float(point[1] - self.cumulative[1])

    def _features(self, gray: np.ndarray) -> np.ndarray | None:
        mask = np.zeros_like(gray)
        width = gray.shape[1]
        mask[:, : max(20, width // 12)] = 255
        mask[:, max(0, width - max(20, width // 12)) :] = 255
        return cv2.goodFeaturesToTrack(
            gray,
            maxCorners=120,
            qualityLevel=0.3,
            minDistance=5,
            blockSize=7,
            mask=mask,
        )


class FirstAnalysisRunner:
    def __init__(self) -> None:
        self.model = None
        self.model_path = self._resolve_model_path()

    def run(
        self,
        bucket: str,
        object_name: str,
        match_id: int,
        max_frames: int | None = None,
    ) -> dict[str, Any]:
        with TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            input_path = temp_path / Path(object_name).name
            output_path = temp_path / "first_analysis_output.mp4"
            self._download_video(bucket, object_name, input_path)

            result = self._process_video(
                input_path=input_path,
                output_path=output_path,
                max_frames=max_frames or settings.FIRST_ANALYSIS_MAX_FRAMES,
            )

            artifact_prefix = f"matches/{match_id}/first-analysis"
            output_object = f"{artifact_prefix}/output.mp4"
            summary_object = f"{artifact_prefix}/summary.json"
            self._put_file(bucket, output_object, output_path, "video/mp4")
            summary = {
                **result,
                "match_id": match_id,
                "input_object": object_name,
                "output_object": output_object,
                "summary_object": summary_object,
                "original_project_path": "apps/backend/football_analysis-main",
            }
            self._put_json(bucket, summary_object, summary)
            return summary

    def _process_video(
        self,
        input_path: Path,
        output_path: Path,
        max_frames: int,
    ) -> dict[str, Any]:
        start = perf_counter()
        model = self._load_model()
        capture = cv2.VideoCapture(str(input_path))
        if not capture.isOpened():
            raise ValueError("Could not open uploaded video for first analysis")

        fps = capture.get(cv2.CAP_PROP_FPS) or 24.0
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 1280)
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 720)
        raw_output_path = output_path.with_suffix(".avi")
        writer, output_codec = self._open_writer(raw_output_path, fps, width, height)

        team_assigner = SimpleTeamAssigner()
        camera = CameraMovementEstimator()
        last_positions: dict[int, tuple[float, float, int]] = {}
        total_distance: dict[int, float] = {}
        speed_by_track: dict[int, float] = {}
        ball_control: list[int] = []
        detections_count = 0
        player_track_ids: set[int] = set()
        ball_seen = 0
        frame_index = 0

        while max_frames <= 0 or frame_index < max_frames:
            ok, frame = capture.read()
            if not ok:
                break

            camera_dx, camera_dy = camera.update(frame)
            objects = self._detect_and_track(model, frame)
            players = [item for item in objects if item.class_name == "player"]
            balls = [item for item in objects if item.class_name == "ball"]
            detections_count += len(objects)
            ball_seen += len(balls)
            player_track_ids.update(item.track_id for item in players)
            team_assigner.warmup(frame, players)
            player_teams = {
                player.track_id: team_assigner.assign(frame, player)
                for player in players
            }

            current_team_control = self._assign_ball_control(
                players,
                balls,
                ball_control,
                player_teams,
            )
            if current_team_control is not None:
                ball_control.append(current_team_control)
            elif ball_control:
                ball_control.append(ball_control[-1])

            self._update_speed(
                players=players,
                camera=camera,
                frame_index=frame_index,
                fps=fps,
                frame_width=width,
                last_positions=last_positions,
                total_distance=total_distance,
                speed_by_track=speed_by_track,
            )

            annotated = frame.copy()
            for player in players:
                team = player_teams.get(player.track_id, 1)
                color = team_assigner.color_for(team)
                self._draw_player(
                    annotated,
                    player,
                    color=color,
                    speed=speed_by_track.get(player.track_id),
                    distance=total_distance.get(player.track_id),
                    has_ball=bool(
                        current_team_control is not None
                        and team == current_team_control
                        and self._is_nearest_to_ball(player, players, balls)
                    ),
                )
            for ball in balls:
                self._draw_triangle(annotated, ball.bbox, (0, 255, 0))

            self._draw_camera_movement(annotated, camera_dx, camera_dy)
            self._draw_ball_control(annotated, ball_control)
            writer.write(annotated)
            frame_index += 1

        capture.release()
        writer.release()
        if not raw_output_path.exists() or raw_output_path.stat().st_size <= 1024:
            raise ValueError("First analysis video writer produced an empty output")
        output_codec = self._transcode_for_browser(
            raw_output_path,
            output_path,
            fallback_codec=output_codec,
        )

        team_1_frames = sum(1 for item in ball_control if item == 1)
        team_2_frames = sum(1 for item in ball_control if item == 2)
        total_control = max(team_1_frames + team_2_frames, 1)
        return {
            "status": "ok",
            "engine": "first_analysis",
            "model": str(self.model_path),
            "output_codec": output_codec,
            "output_content_type": "video/mp4",
            "frames_processed": frame_index,
            "max_frames": max_frames,
            "fps": fps,
            "resolution": [width, height],
            "detections_count": detections_count,
            "player_tracks_count": len(player_track_ids),
            "ball_observations": ball_seen,
            "team_ball_control": {
                "team_1_percent": round(team_1_frames * 100 / total_control, 2),
                "team_2_percent": round(team_2_frames * 100 / total_control, 2),
            },
            "elapsed_ms": round((perf_counter() - start) * 1000, 2),
        }

    def _open_writer(
        self,
        output_path: Path,
        fps: float,
        width: int,
        height: int,
    ):
        candidates = [("MJPG", output_path)]
        for codec, path in candidates:
            writer = cv2.VideoWriter(
                str(path),
                cv2.VideoWriter_fourcc(*codec),
                fps,
                (width, height),
            )
            if writer.isOpened():
                return writer, codec
            writer.release()
        raise ValueError("Could not open the intermediate video writer")

    def _transcode_for_browser(
        self,
        source_path: Path,
        output_path: Path,
        fallback_codec: str,
    ) -> str:
        ffmpeg = shutil.which("ffmpeg")
        if ffmpeg is None:
            source_path.replace(output_path)
            return fallback_codec

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
            "23",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(output_path),
        ]
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=False,
        )
        if completed.returncode != 0:
            source_path.replace(output_path)
            return fallback_codec
        if not output_path.exists() or output_path.stat().st_size <= 1024:
            source_path.replace(output_path)
            return fallback_codec
        return "h264"

    def _detect_and_track(self, model: Any, frame: np.ndarray) -> list[FrameObject]:
        results = model.track(
            frame,
            persist=True,
            conf=max(settings.YOLO_CONFIDENCE, 0.15),
            imgsz=settings.YOLO_IMAGE_SIZE,
            device=settings.YOLO_DEVICE,
            verbose=False,
            tracker="bytetrack.yaml",
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

        objects: list[FrameObject] = []
        for index, bbox in enumerate(xyxy):
            raw_name = str(names.get(int(cls[index]), int(cls[index]))).lower()
            mapped = self._map_class_name(raw_name)
            if mapped is None:
                continue
            objects.append(
                FrameObject(
                    track_id=int(ids[index]),
                    class_name=mapped,
                    bbox=[float(value) for value in bbox.tolist()],
                    confidence=(
                        float(conf[index])
                        if conf[index] is not None
                        else None
                    ),
                )
            )
        return objects

    def _map_class_name(self, raw_name: str) -> str | None:
        if raw_name in PERSON_ALIASES:
            return "player"
        if raw_name in BALL_ALIASES:
            return "ball"
        return None

    def _update_speed(
        self,
        players: list[FrameObject],
        camera: CameraMovementEstimator,
        frame_index: int,
        fps: float,
        frame_width: int,
        last_positions: dict[int, tuple[float, float, int]],
        total_distance: dict[int, float],
        speed_by_track: dict[int, float],
    ) -> None:
        meters_per_pixel = 68.0 / max(frame_width, 1)
        for player in players:
            foot = self._foot_position(player.bbox)
            adjusted = camera.adjust(foot)
            previous = last_positions.get(player.track_id)
            if previous is not None:
                dx = adjusted[0] - previous[0]
                dy = adjusted[1] - previous[1]
                frame_delta = max(frame_index - previous[2], 1)
                distance_m = float(np.hypot(dx, dy) * meters_per_pixel)
                if distance_m < 15:
                    total_distance[player.track_id] = total_distance.get(player.track_id, 0.0) + distance_m
                    speed_by_track[player.track_id] = distance_m / (frame_delta / fps) * 3.6
            last_positions[player.track_id] = (adjusted[0], adjusted[1], frame_index)

    def _assign_ball_control(
        self,
        players: list[FrameObject],
        balls: list[FrameObject],
        ball_control: list[int],
        player_teams: dict[int, int],
    ) -> int | None:
        if not players or not balls:
            return ball_control[-1] if ball_control else None
        ball_center = self._center(balls[0].bbox)
        nearest: tuple[float, FrameObject] | None = None
        for player in players:
            left_foot = (player.bbox[0], player.bbox[3])
            right_foot = (player.bbox[2], player.bbox[3])
            distance = min(
                float(np.hypot(left_foot[0] - ball_center[0], left_foot[1] - ball_center[1])),
                float(np.hypot(right_foot[0] - ball_center[0], right_foot[1] - ball_center[1])),
            )
            if nearest is None or distance < nearest[0]:
                nearest = (distance, player)
        if nearest is None or nearest[0] > 80:
            return ball_control[-1] if ball_control else None
        return player_teams.get(nearest[1].track_id, 1)

    def _is_nearest_to_ball(
        self,
        player: FrameObject,
        players: list[FrameObject],
        balls: list[FrameObject],
    ) -> bool:
        if not balls:
            return False
        ball_center = self._center(balls[0].bbox)
        distances = [
            (
                float(np.hypot(self._foot_position(item.bbox)[0] - ball_center[0], self._foot_position(item.bbox)[1] - ball_center[1])),
                item.track_id,
            )
            for item in players
        ]
        return bool(distances and min(distances)[1] == player.track_id)

    def _draw_player(
        self,
        frame: np.ndarray,
        player: FrameObject,
        color: tuple[int, int, int],
        speed: float | None,
        distance: float | None,
        has_ball: bool,
    ) -> None:
        x1, y1, x2, y2 = [int(round(value)) for value in player.bbox]
        center_x = int((x1 + x2) / 2)
        width = max(12, int(x2 - x1))
        cv2.ellipse(
            frame,
            (center_x, y2),
            (max(10, width // 2), max(5, width // 6)),
            0,
            -45,
            235,
            color,
            3,
            cv2.LINE_4,
        )
        font_scale = self._font_scale(frame, 0.62)
        small_scale = self._font_scale(frame, 0.46)
        thickness = self._thickness(frame)
        cv2.rectangle(
            frame,
            (center_x - 15, y2 + 5),
            (center_x + 15, y2 + 22),
            (240, 250, 235),
            cv2.FILLED,
        )
        cv2.putText(
            frame,
            str(player.track_id),
            (center_x - 9, y2 + 19),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (0, 0, 0),
            thickness,
        )
        if speed is not None and distance is not None:
            cv2.putText(
                frame,
                f"{speed:.2f} km/h",
                (center_x - 24, y2 + 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                small_scale,
                (0, 0, 0),
                thickness,
            )
            cv2.putText(
                frame,
                f"{distance:.2f} m",
                (center_x - 20, y2 + 56),
                cv2.FONT_HERSHEY_SIMPLEX,
                small_scale,
                (0, 0, 0),
                thickness,
            )
        if has_ball:
            self._draw_triangle(frame, player.bbox, (0, 0, 255), above=True)

    def _draw_triangle(
        self,
        frame: np.ndarray,
        bbox: list[float],
        color: tuple[int, int, int],
        above: bool = False,
    ) -> None:
        x, y = self._center(bbox)
        if above:
            y = bbox[1]
        points = np.array(
            [
                [int(x), int(y)],
                [int(x - 11), int(y - 22)],
                [int(x + 11), int(y - 22)],
            ]
        )
        cv2.drawContours(frame, [points], 0, color, cv2.FILLED)
        cv2.drawContours(frame, [points], 0, (0, 0, 0), 2)

    def _draw_camera_movement(self, frame: np.ndarray, dx: float, dy: float) -> None:
        scale = self._font_scale(frame, 0.62)
        thickness = self._thickness(frame)
        line_height = int(26 * self._ui_scale(frame))
        overlay = frame.copy()
        panel_w = int(260 * self._ui_scale(frame))
        panel_h = int(58 * self._ui_scale(frame))
        cv2.rectangle(overlay, (0, 0), (panel_w, panel_h), (255, 255, 255), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
        cv2.putText(frame, f"Camera X: {dx:.2f}", (8, line_height), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thickness)
        cv2.putText(frame, f"Camera Y: {dy:.2f}", (8, line_height * 2), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thickness)

    def _draw_ball_control(self, frame: np.ndarray, ball_control: list[int]) -> None:
        scale = self._font_scale(frame, 0.62)
        thickness = self._thickness(frame)
        ui_scale = self._ui_scale(frame)
        height, width = frame.shape[:2]
        panel_w = int(360 * ui_scale)
        panel_h = int(66 * ui_scale)
        x1, y1 = max(0, width - panel_w - 12), max(0, height - panel_h - 12)
        overlay = frame.copy()
        cv2.rectangle(overlay, (x1, y1), (width - 12, height - 12), (255, 255, 255), -1)
        cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)
        team_1 = sum(1 for item in ball_control if item == 1)
        team_2 = sum(1 for item in ball_control if item == 2)
        total = max(team_1 + team_2, 1)
        cv2.putText(frame, f"T1 Ball: {team_1 * 100 / total:.2f}%", (x1 + 10, y1 + int(25 * ui_scale)), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thickness)
        cv2.putText(frame, f"T2 Ball: {team_2 * 100 / total:.2f}%", (x1 + 10, y1 + int(52 * ui_scale)), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thickness)

    def _ui_scale(self, frame: np.ndarray) -> float:
        height, width = frame.shape[:2]
        return max(0.45, min(width / 1920, height / 1080))

    def _font_scale(self, frame: np.ndarray, base: float) -> float:
        return max(0.28, base * self._ui_scale(frame))

    def _thickness(self, frame: np.ndarray) -> int:
        return max(1, int(round(2 * self._ui_scale(frame))))

    def _center(self, bbox: list[float]) -> tuple[float, float]:
        return (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2

    def _foot_position(self, bbox: list[float]) -> tuple[float, float]:
        return (bbox[0] + bbox[2]) / 2, bbox[3]

    def _load_model(self) -> Any:
        if self.model is not None:
            return self.model
        from ultralytics import YOLO

        self.model = YOLO(str(self.model_path))
        return self.model

    def _resolve_model_path(self) -> Path:
        project_model = Path("football_analysis-main/models/best.pt")
        if project_model.exists():
            return project_model
        configured = Path(settings.YOLO_MODEL_PATH)
        if configured.exists():
            return configured
        return Path("yolo11n.pt")

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
        client.put_object(
            bucket,
            object_name,
            io.BytesIO(data),
            length=len(data),
            content_type=content_type,
        )

    def _put_json(self, bucket: str, object_name: str, payload: dict[str, Any]) -> None:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        client.put_object(
            bucket,
            object_name,
            io.BytesIO(data),
            length=len(data),
            content_type="application/json",
        )
