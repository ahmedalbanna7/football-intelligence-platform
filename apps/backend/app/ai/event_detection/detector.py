from typing import Any


class EventDetectionEngine:
    def detect(self, tracks: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "status": "ok",
            "data": {
                "events": [],
                "tracks_received": len(tracks),
            },
            "meta": {
                "engine": "event_detection_stub",
            },
        }
