from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract evenly distributed frames for annotation and regression tests."
    )
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int, default=120)
    parser.add_argument("--start-second", type=float, default=0.0)
    parser.add_argument("--end-second", type=float)
    parser.add_argument("--jpeg-quality", type=int, default=92)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.video.is_file():
        raise FileNotFoundError(args.video)
    if args.count < 1:
        raise ValueError("--count must be at least 1")

    capture = cv2.VideoCapture(str(args.video))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open video: {args.video}")

    fps = float(capture.get(cv2.CAP_PROP_FPS) or 25.0)
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    start_frame = max(0, int(args.start_second * fps))
    requested_end = (
        int(args.end_second * fps) if args.end_second is not None else total_frames - 1
    )
    end_frame = min(max(start_frame, requested_end), max(0, total_frames - 1))
    span = max(1, end_frame - start_frame)
    frame_indexes = sorted(
        {
            min(end_frame, start_frame + round(span * index / max(1, args.count - 1)))
            for index in range(args.count)
        }
    )

    args.output.mkdir(parents=True, exist_ok=True)
    written: list[dict[str, int | float | str]] = []
    for frame_index in frame_indexes:
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = capture.read()
        if not ok:
            continue
        output_name = f"{args.video.stem}_frame_{frame_index:08d}.jpg"
        output_path = args.output / output_name
        if not cv2.imwrite(
            str(output_path),
            frame,
            [cv2.IMWRITE_JPEG_QUALITY, max(1, min(100, args.jpeg_quality))],
        ):
            raise RuntimeError(f"Failed to write {output_path}")
        written.append(
            {
                "image": output_name,
                "frame": frame_index,
                "second": round(frame_index / fps, 3),
            }
        )
    capture.release()

    manifest = {
        "source": str(args.video),
        "fps": fps,
        "total_frames": total_frames,
        "start_frame": start_frame,
        "end_frame": end_frame,
        "requested_samples": args.count,
        "written_samples": len(written),
        "frames": written,
    }
    manifest_path = args.output / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
