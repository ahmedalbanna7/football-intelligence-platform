from pathlib import Path
from typing import Any

from app.core.config import settings


class FrameExtractor:
    def __init__(
        self,
        sample_rate: int | None = None,
        max_frames: int | None = None,
    ) -> None:
        self.sample_rate = sample_rate or settings.FRAME_SAMPLE_RATE
        self.max_frames = max_frames or settings.FRAME_MAX_FRAMES

    def extract(
        self,
        video_path: str,
        output_dir: str | None = None,
    ) -> dict[str, Any]:
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
        if not capture.isOpened():
            return {
                "status": "failed",
                "data": {"frames": []},
                "meta": {
                    "reason": "video_open_failed",
                    "video_path": video_path,
                },
            }

        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0)
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        duration_seconds = frame_count / fps if fps > 0 else None

        frames: list[dict[str, Any]] = []
        frame_index = 0
        frame_output_dir: Path | None = None
        if output_dir is not None:
            frame_output_dir = Path(output_dir)
            frame_output_dir.mkdir(parents=True, exist_ok=True)

        while capture.isOpened():
            ok, frame = capture.read()
            if not ok:
                break

            if frame_index % self.sample_rate == 0:
                timestamp_seconds = frame_index / fps if fps > 0 else None
                frame_image_path = None
                if frame_output_dir is not None:
                    frame_image_path = str(frame_output_dir / f"frame_{frame_index}.jpg")
                    cv2.imwrite(frame_image_path, frame)

                frames.append(
                    {
                        "frame_index": frame_index,
                        "timestamp_seconds": timestamp_seconds,
                        "shape": list(frame.shape),
                        "height": int(frame.shape[0]),
                        "width": int(frame.shape[1]),
                        "channels": int(frame.shape[2]) if len(frame.shape) > 2 else 1,
                        "image_path": frame_image_path,
                    }
                )

                if len(frames) >= self.max_frames:
                    break

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
                "max_frames": self.max_frames,
                "fps": fps,
                "duration_seconds": duration_seconds,
                "frame_count": frame_count,
                "resolution": {
                    "width": width,
                    "height": height,
                },
                "sampled_frames_count": len(frames),
                "total_frames_seen": frame_index,
            },
        }
