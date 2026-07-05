from pathlib import Path
import io
import json
from tempfile import TemporaryDirectory
from typing import Any

from app.core.config import settings
from app.ai.detectors.detector_factory import DetectorFactory
from app.ai.event_detection.detector import EventDetectionEngine
from app.ai.jersey_number.recognizer import JerseyNumberRecognizer
from app.ai.llm.coach import LLMCoach
from app.ai.player_development.engine import DevelopmentEngine
from app.ai.team_classification.classifier import TeamClassifier
from app.ai.tactical_identity_layer import TacticalIdentityService
from app.ai.trackers.tracker_factory import TrackerFactory
from app.modules.analytics.service import AnalyticsEngine
from app.modules.reports.service import ReportEngine
from app.pipeline.frame_extractor import FrameExtractor
from app.services.minio_client import client


class VideoPipeline:
    def __init__(self) -> None:
        self.frame_extractor = FrameExtractor()
        self.detector = DetectorFactory.create("yolo")
        self.tracker = TrackerFactory.create("bytetrack")
        self.team_classifier = TeamClassifier()
        self.jersey_number_recognizer = JerseyNumberRecognizer()
        self.tactical_identity = TacticalIdentityService()
        self.event_detector = EventDetectionEngine()
        self.analytics = AnalyticsEngine()
        self.development = DevelopmentEngine()
        self.llm_coach = LLMCoach()
        self.reports = ReportEngine()

    def run(
        self,
        bucket: str,
        object_name: str,
        match_context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        context = match_context or {}

        with TemporaryDirectory() as temp_dir:
            local_path = Path(temp_dir) / Path(object_name).name
            frames_dir = Path(temp_dir) / "frames"
            self._download_video(bucket, object_name, local_path)

            frames_result = self.frame_extractor.extract(
                str(local_path),
                output_dir=str(frames_dir),
            )
            frames = frames_result.get("data", {}).get("frames", [])

            detections_result = self.detector.detect(frames)
            detections = detections_result.get("data", {}).get("detections", [])

            tracks_result = self.tracker.track(detections)
            tracks = tracks_result.get("data", {}).get("tracks", [])

            team_assignment_result = self.team_classifier.assign(
                tracks,
                match_context=context,
                frames=frames,
            )
            assigned_tracks = team_assignment_result.get("data", {}).get("tracks", [])

            crops_result = self._save_player_crops(
                bucket=bucket,
                object_name=object_name,
                tracks=assigned_tracks,
                frames=frames,
            )
            team_assignment_result.setdefault("data", {})["tracks"] = assigned_tracks

            jersey_number_result = self.jersey_number_recognizer.recognize(
                assigned_tracks,
                match_context=context,
            )
            identified_tracks = jersey_number_result.get("data", {}).get("tracks", [])

            tactical_identity_result = self.tactical_identity.resolve(
                identified_tracks,
                match_context=context,
            )

            events_result = self.event_detector.detect(
                identified_tracks,
                match_context=context,
            )
            analytics_result = self.analytics.analyze(
                identified_tracks,
                match_context=context,
            )
            artifacts_result = self._write_pipeline_artifacts(
                bucket=bucket,
                object_name=object_name,
                detections=detections,
                tracks=identified_tracks,
            )
            development_result = self.development.analyze(analytics_result)

            llm_result = self.llm_coach.summarize(
                {
                    "match_context": context,
                    "events": events_result,
                    "analytics": analytics_result,
                    "development": development_result,
                }
            )
            report_result = self.reports.build_json_report(
                {
                    "frames": self._sanitize_frames_result(frames_result),
                    "match_context": context,
                    "detections": detections_result,
                    "tracks": tracks_result,
                    "team_assignment": team_assignment_result,
                    "crops": crops_result,
                    "jersey_number_recognition": jersey_number_result,
                    "tactical_identity": tactical_identity_result,
                    "events": events_result,
                    "analytics": analytics_result,
                    "artifacts": artifacts_result,
                    "development": development_result,
                    "llm": llm_result,
                }
            )
            sanitized_frames_result = self._sanitize_frames_result(frames_result)

            return {
                "status": "ok",
                "data": {
                    "frames": sanitized_frames_result,
                    "detections": detections_result,
                    "tracks": tracks_result,
                    "team_assignment": team_assignment_result,
                    "crops": crops_result,
                    "jersey_number_recognition": jersey_number_result,
                    "tactical_identity": tactical_identity_result,
                    "events": events_result,
                    "analytics": analytics_result,
                    "artifacts": artifacts_result,
                    "development": development_result,
                    "llm": llm_result,
                    "report": report_result,
                },
                "meta": {
                    "bucket": bucket,
                    "object_name": object_name,
                    "match_context": context,
                    "video": frames_result.get("meta", {}),
                    "detections": detections_result.get("data", {}),
                    "tracks": tracks_result.get("data", {}),
                    "team_assignment": team_assignment_result.get("data", {}),
                    "crops": crops_result.get("data", {}),
                    "jersey_number_recognition": jersey_number_result.get("data", {}),
                    "tactical_identity": tactical_identity_result.get("data", {}),
                    "analytics": analytics_result.get("data", {}),
                    "artifacts": artifacts_result.get("data", {}),
                },
            }

    def _sanitize_frames_result(self, frames_result: dict[str, Any]) -> dict[str, Any]:
        sanitized = {
            **frames_result,
            "data": {
                **frames_result.get("data", {}),
                "frames": [
                    {
                        key: value
                        for key, value in frame.items()
                        if key != "image_path"
                    }
                    for frame in frames_result.get("data", {}).get("frames", [])
                ],
            },
        }
        return sanitized

    def _download_video(self, bucket: str, object_name: str, local_path: Path) -> None:
        response = client.get_object(bucket, object_name)
        try:
            with local_path.open("wb") as file:
                for chunk in response.stream(1024 * 1024):
                    file.write(chunk)
        finally:
            response.close()
            response.release_conn()

    def _write_pipeline_artifacts(
        self,
        bucket: str,
        object_name: str,
        detections: list[dict[str, Any]],
        tracks: list[dict[str, Any]],
    ) -> dict[str, Any]:
        artifact_prefix = self._artifact_prefix(object_name)
        track_observations = self._flatten_track_observations(tracks)
        track_summaries = self._build_track_summaries(tracks)

        artifacts = {
            "detections_jsonl": f"{artifact_prefix}/detections.jsonl",
            "track_observations_jsonl": f"{artifact_prefix}/track_observations.jsonl",
            "tracks_summary_json": f"{artifact_prefix}/tracks_summary.json",
            "crops_prefix": f"{artifact_prefix}/crops",
        }

        try:
            self._put_text_object(
                bucket,
                artifacts["detections_jsonl"],
                self._to_jsonl(detections),
                "application/x-ndjson",
            )
            self._put_text_object(
                bucket,
                artifacts["track_observations_jsonl"],
                self._to_jsonl(track_observations),
                "application/x-ndjson",
            )
            self._put_text_object(
                bucket,
                artifacts["tracks_summary_json"],
                json.dumps(track_summaries, ensure_ascii=False),
                "application/json",
            )
            status = "ok"
            error = None
        except Exception as exc:
            status = "failed"
            error = str(exc)

        return {
            "status": status,
            "data": {
                "artifacts": artifacts,
                "detections_count": len(detections),
                "track_observations_count": len(track_observations),
                "tracks_count": len(track_summaries),
                "sample_detections": detections[:settings.YOLO_ARTIFACT_SAMPLE_LIMIT],
                "sample_track_observations": track_observations[
                    :settings.YOLO_ARTIFACT_SAMPLE_LIMIT
                ],
            },
            "meta": {
                "storage": "minio_jsonl",
                "bucket": bucket,
                "error": error,
            },
        }

    def _artifact_prefix(self, object_name: str) -> str:
        parts = object_name.split("/")
        if len(parts) >= 2 and parts[0] == "matches":
            return f"matches/{parts[1]}/artifacts"
        return f"artifacts/{Path(object_name).stem}"

    def _flatten_track_observations(
        self,
        tracks: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        observations: list[dict[str, Any]] = []
        for track in tracks:
            track_id = track.get("track_id")
            class_name = track.get("class_name")
            for frame in track.get("frames", []):
                observations.append(
                    {
                        "track_id": track_id,
                        "frame_index": frame.get("frame_index"),
                        "timestamp_ms": frame.get("timestamp_ms"),
                        "class_name": class_name,
                        "confidence": frame.get("confidence"),
                        "bbox_xyxy": frame.get("bbox_xyxy"),
                        "bbox_center": frame.get("bbox_center") or frame.get("center"),
                        "bbox_area": frame.get("bbox_area"),
                        "team_label": track.get("team_context"),
                        "team_confidence": track.get("team_assignment_confidence"),
                        "player_id": track.get("resolved_player_id"),
                        "pitch_x": None,
                        "pitch_y": None,
                        "zone": track.get("zone"),
                        "jersey_number": track.get("recognized_shirt_number"),
                        "track_state": frame.get("track_state", "active"),
                        "source_tracker": frame.get("source_tracker"),
                    }
                )
        return observations

    def _build_track_summaries(
        self,
        tracks: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        summaries: list[dict[str, Any]] = []
        for track in tracks:
            frames = track.get("frames", [])
            timestamps = [
                frame.get("timestamp_ms")
                for frame in frames
                if frame.get("timestamp_ms") is not None
            ]
            confidences = [
                float(frame.get("confidence"))
                for frame in frames
                if frame.get("confidence") is not None
            ]
            first_timestamp = timestamps[0] if timestamps else None
            last_timestamp = timestamps[-1] if timestamps else None
            summaries.append(
                {
                    "track_id": track.get("track_id"),
                    "class_name": track.get("class_name"),
                    "first_frame_index": (
                        frames[0].get("frame_index") if frames else None
                    ),
                    "last_frame_index": (
                        frames[-1].get("frame_index") if frames else None
                    ),
                    "first_timestamp_ms": first_timestamp,
                    "last_timestamp_ms": last_timestamp,
                    "duration_ms": (
                        round(last_timestamp - first_timestamp, 2)
                        if first_timestamp is not None and last_timestamp is not None
                        else None
                    ),
                    "observation_count": len(frames),
                    "avg_confidence": (
                        round(sum(confidences) / len(confidences), 6)
                        if confidences
                        else None
                    ),
                    "team_label": track.get("team_context"),
                    "team_confidence": track.get("team_assignment_confidence"),
                    "resolved_player_id": track.get("resolved_player_id"),
                    "identity_confidence": track.get("identity_confidence"),
                }
            )
        return summaries

    def _to_jsonl(self, rows: list[dict[str, Any]]) -> str:
        return "\n".join(
            json.dumps(row, ensure_ascii=False)
            for row in rows
        )

    def _put_text_object(
        self,
        bucket: str,
        object_name: str,
        text: str,
        content_type: str,
    ) -> None:
        payload = text.encode("utf-8")
        client.put_object(
            bucket,
            object_name,
            io.BytesIO(payload),
            length=len(payload),
            content_type=content_type,
        )

    def _save_player_crops(
        self,
        bucket: str,
        object_name: str,
        tracks: list[dict[str, Any]],
        frames: list[dict[str, Any]],
    ) -> dict[str, Any]:
        if not settings.SAVE_PLAYER_CROPS:
            return {
                "status": "skipped",
                "data": {
                    "crops_count": 0,
                    "jersey_crops_count": 0,
                    "crops_prefix": None,
                },
                "meta": {"reason": "SAVE_PLAYER_CROPS=false"},
            }

        try:
            import cv2
        except Exception as exc:
            return {
                "status": "skipped",
                "data": {
                    "crops_count": 0,
                    "jersey_crops_count": 0,
                    "crops_prefix": None,
                },
                "meta": {"reason": str(exc)},
            }

        frame_lookup = {
            frame.get("frame_index"): frame
            for frame in frames
            if frame.get("image_path")
        }
        crops_prefix = f"{self._artifact_prefix(object_name)}/crops"
        crops_count = 0
        jersey_crops_count = 0

        for track in tracks:
            if track.get("class_name") != "player":
                continue

            samples = []
            saved_for_track = 0
            for track_frame in track.get("frames", []):
                frame_index = int(track_frame.get("frame_index") or 0)
                if frame_index % max(settings.CROP_EVERY_N_FRAMES, 1) != 0:
                    continue
                if saved_for_track >= settings.MAX_CROPS_PER_TRACK:
                    break

                frame = frame_lookup.get(frame_index)
                if frame is None:
                    continue

                image = cv2.imread(frame["image_path"])
                if image is None:
                    continue

                crop = self._crop_bbox(image, track_frame.get("bbox_xyxy"))
                if crop is None:
                    continue

                crop_object = (
                    f"{crops_prefix}/track_{track.get('track_id')}/"
                    f"frame_{frame_index}.jpg"
                )
                self._put_image_object(bucket, crop_object, crop)
                crops_count += 1

                jersey_crop = self._crop_jersey(crop)
                jersey_object = None
                if jersey_crop is not None:
                    jersey_object = (
                        f"{crops_prefix}/track_{track.get('track_id')}/"
                        f"frame_{frame_index}_jersey.jpg"
                    )
                    self._put_image_object(bucket, jersey_object, jersey_crop)
                    jersey_crops_count += 1

                track_frame["crop_path"] = crop_object
                track_frame["jersey_crop_path"] = jersey_object
                samples.append(
                    {
                        "frame_index": frame_index,
                        "crop_path": crop_object,
                        "jersey_crop_path": jersey_object,
                    }
                )
                saved_for_track += 1

            track["crop_samples"] = samples

        return {
            "status": "ok",
            "data": {
                "crops_count": crops_count,
                "jersey_crops_count": jersey_crops_count,
                "crops_prefix": crops_prefix,
            },
            "meta": {
                "crop_every_n_frames": settings.CROP_EVERY_N_FRAMES,
                "max_crops_per_track": settings.MAX_CROPS_PER_TRACK,
            },
        }

    def _crop_bbox(self, image, bbox_xyxy: list[float] | None):
        if not bbox_xyxy or len(bbox_xyxy) != 4:
            return None

        height, width = image.shape[:2]
        x1, y1, x2, y2 = [
            int(round(value))
            for value in bbox_xyxy
        ]
        pad_x = max(2, int((x2 - x1) * 0.08))
        pad_y = max(2, int((y2 - y1) * 0.08))
        x1 = max(0, min(x1 - pad_x, width - 1))
        x2 = max(0, min(x2 + pad_x, width))
        y1 = max(0, min(y1 - pad_y, height - 1))
        y2 = max(0, min(y2 + pad_y, height))
        if x2 <= x1 or y2 <= y1:
            return None
        crop = image[y1:y2, x1:x2]
        if crop.size == 0:
            return None
        return crop

    def _crop_jersey(self, crop):
        height, width = crop.shape[:2]
        if height < 8 or width < 4:
            return None
        y1 = int(height * 0.18)
        y2 = int(height * 0.62)
        jersey = crop[y1:y2, :]
        if jersey.size == 0:
            return None
        return jersey

    def _put_image_object(
        self,
        bucket: str,
        object_name: str,
        image,
    ) -> None:
        try:
            import cv2

            ok, encoded = cv2.imencode(".jpg", image)
            if not ok:
                return
            payload = encoded.tobytes()
            client.put_object(
                bucket,
                object_name,
                io.BytesIO(payload),
                length=len(payload),
                content_type="image/jpeg",
            )
        except Exception:
            return
