from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from app.ai.detectors.detector_factory import DetectorFactory
from app.ai.event_detection.detector import EventDetectionEngine
from app.ai.llm.coach import LLMCoach
from app.ai.player_development.engine import DevelopmentEngine
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
        self.event_detector = EventDetectionEngine()
        self.analytics = AnalyticsEngine()
        self.development = DevelopmentEngine()
        self.llm_coach = LLMCoach()
        self.reports = ReportEngine()

    def run(self, bucket: str, object_name: str) -> dict[str, Any]:
        with TemporaryDirectory() as temp_dir:
            local_path = Path(temp_dir) / Path(object_name).name
            self._download_video(bucket, object_name, local_path)

            frames_result = self.frame_extractor.extract(str(local_path))
            frames = frames_result.get("data", {}).get("frames", [])

            detections_result = self.detector.detect(frames)
            detections = detections_result.get("data", {}).get("detections", [])

            tracks_result = self.tracker.track(detections)
            tracks = tracks_result.get("data", {}).get("tracks", [])

            events_result = self.event_detector.detect(tracks)
            analytics_result = self.analytics.analyze(tracks)
            development_result = self.development.analyze(analytics_result)

            llm_result = self.llm_coach.summarize(
                {
                    "events": events_result,
                    "analytics": analytics_result,
                    "development": development_result,
                }
            )
            report_result = self.reports.build_json_report(
                {
                    "frames": frames_result,
                    "detections": detections_result,
                    "tracks": tracks_result,
                    "events": events_result,
                    "analytics": analytics_result,
                    "development": development_result,
                    "llm": llm_result,
                }
            )

            return {
                "status": "ok",
                "data": {
                    "frames": frames_result,
                    "detections": detections_result,
                    "tracks": tracks_result,
                    "events": events_result,
                    "analytics": analytics_result,
                    "development": development_result,
                    "llm": llm_result,
                    "report": report_result,
                },
                "meta": {
                    "bucket": bucket,
                    "object_name": object_name,
                },
            }

    def _download_video(self, bucket: str, object_name: str, local_path: Path) -> None:
        response = client.get_object(bucket, object_name)
        try:
            with local_path.open("wb") as file:
                for chunk in response.stream(1024 * 1024):
                    file.write(chunk)
        finally:
            response.close()
            response.release_conn()
