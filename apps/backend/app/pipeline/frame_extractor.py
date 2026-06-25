from pathlib import Path
from typing import Any


class FrameExtractor:
    def __init__(self, sample_rate: int = 30) -> None:
        self.sample_rate = sample_rate

    def extract(self, video_path: str) -> dict[str, Any]:
        try:
            import cv2
        except ImportError:
            return {
                "status": "skipped",
                "data": {"frames": []},
                "meta": {
                    "reason": "opencv_not_installed",
                    "video_path": video_path,
                },
            }

        capture = cv2.VideoCapture(video_path)
        frames: list[dict[str, Any]] = []
        frame_index = 0

        while capture.isOpened():
            ok, frame = capture.read()
            if not ok:
                break

            if frame_index % self.sample_rate == 0:
                frames.append(
                    {
                        "frame_index": frame_index,
                        "shape": list(frame.shape),
                    }
                )

            frame_index += 1

        capture.release()

        return {
            "status": "ok",
            "data": {
                "frames": frames,
                "source": Path(video_path).name,
            },
            "meta": {
                "sample_rate": self.sample_rate,
                "total_frames_seen": frame_index,
            },
        }
