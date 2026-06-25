from typing import Any


class YOLODetector:
    name = "yolo_stub"

    def detect(self, frames: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "status": "ok",
            "data": {
                "detections": [],
                "frames_processed": len(frames),
            },
            "meta": {
                "engine": self.name,
                "model": "future_yolov11_or_yolov12",
            },
        }
