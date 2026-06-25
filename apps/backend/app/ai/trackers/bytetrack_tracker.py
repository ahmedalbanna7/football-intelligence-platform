from typing import Any


class ByteTrackTracker:
    name = "bytetrack_stub"

    def track(self, detections: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "status": "ok",
            "data": {
                "tracks": [],
                "detections_received": len(detections),
            },
            "meta": {
                "engine": self.name,
            },
        }
