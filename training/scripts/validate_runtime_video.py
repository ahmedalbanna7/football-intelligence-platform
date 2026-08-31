from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from types import ModuleType, SimpleNamespace
import sys


ROOT = Path(__file__).resolve().parents[2]
BACKEND = ROOT / "apps" / "backend"
TRAINING_CACHE = ROOT / "training" / "cache"
ULTRALYTICS_CONFIG = TRAINING_CACHE / "ultralytics"
ULTRALYTICS_CONFIG.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("YOLO_CONFIG_DIR", str(ULTRALYTICS_CONFIG))
os.environ.setdefault("TORCH_HOME", str(TRAINING_CACHE / "torch"))
sys.path.insert(0, str(BACKEND))


def install_runtime_stubs(device: str) -> None:
    config_module = ModuleType("app.core.config")
    config_module.settings = SimpleNamespace(
        YOLO_MODEL_PATH="yolo11n.pt",
        YOLO_IMAGE_SIZE=640,
        YOLO_DEVICE=device,
        YOLO_MAX_DETECTIONS=100,
        MATCH_ANALYSIS_PLAYER_MODEL_PATH="models/football-objects-v2.pt",
        MATCH_ANALYSIS_PLAYER_MODEL_FALLBACK_PATH=(
            "models/football-player-detection.pt"
        ),
        MATCH_ANALYSIS_BALL_MODEL_PATH="models/football-ball-v2.pt",
        MATCH_ANALYSIS_BALL_MODEL_FALLBACK_PATH=(
            "models/football-player-detection.pt"
        ),
        MATCH_ANALYSIS_PITCH_MODEL_PATH="models/football-pitch-v2.pt",
        MATCH_ANALYSIS_PITCH_MODEL_FALLBACK_PATH=(
            "models/football-pitch-detection.pt"
        ),
        MATCH_ANALYSIS_IMAGE_SIZE=960,
        MATCH_ANALYSIS_CONFIDENCE=0.20,
        MATCH_ANALYSIS_RADAR_STRIDE=12,
        MATCH_ANALYSIS_TRACKER=(
            "app/match_analysis_plus/trackers/botsort_reid.yaml"
        ),
    )
    sys.modules["app.core.config"] = config_module

    minio_module = ModuleType("app.services.minio_client")
    minio_module.BUCKET_NAME = "local-validation"
    minio_module.client = SimpleNamespace()
    sys.modules["app.services.minio_client"] = minio_module


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the real Match Analysis pipeline against a local video."
    )
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--frames", type=int, default=1000)
    parser.add_argument("--device", default="0")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--selection-only", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    install_runtime_stubs(args.device)

    from app.match_analysis_plus.runner import MatchAnalysisPlusRunner

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    crops_dir = output_dir / "tracking-quality-crops"
    crops_dir.mkdir(parents=True, exist_ok=True)
    runner = MatchAnalysisPlusRunner()
    if args.selection_only:
        import cv2

        capture = cv2.VideoCapture(str(args.video.resolve()))
        if not capture.isOpened():
            raise ValueError(f"Could not open {args.video}")
        source_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        capture.set(cv2.CAP_PROP_POS_FRAMES, max(0, args.start_frame))
        runner._select_model_for_video(capture, args.start_frame)
        runner._select_pitch_model_for_video(
            capture,
            args.start_frame,
            source_frames,
        )
        capture.release()
        selection = {
            "objects_model": str(runner.model_path),
            "objects": runner.model_selection,
            "ball_model": str(runner.ball_model_path),
            "pitch_model": str(runner.pitch_model_path),
            "pitch": runner.pitch_model_selection,
        }
        (output_dir / "model-selection.json").write_text(
            json.dumps(selection, indent=2),
            encoding="utf-8",
        )
        print(json.dumps(selection, indent=2))
        return 0

    summary = runner._process_video(
        input_path=args.video.resolve(),
        raw_output_path=output_dir / "output.avi",
        output_path=output_dir / "output.mp4",
        thumbnail_path=output_dir / "thumbnail.jpg",
        quality_crops_dir=crops_dir,
        quality_predictions_path=output_dir / "tracking-quality.jsonl",
        mode="FULL_ANALYSIS",
        max_frames=args.frames,
        start_frame=args.start_frame,
        calibration_points=[],
        team_context={},
    )
    visual_layers = summary.pop("_visual_layers_payload", None)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    if visual_layers is not None:
        (output_dir / "visual-layers.json").write_text(
            json.dumps(visual_layers, indent=2),
            encoding="utf-8",
        )

    required = {
        "status": summary.get("status"),
        "frames_processed": summary.get("frames_processed"),
        "objects_model": summary.get("model"),
        "ball_model": summary.get("ball_model"),
        "pitch_model": summary.get("pitch_model"),
        "pitch_selection": (
            summary.get("pitch_model_selection", {}).get("selected")
        ),
        "tracks_count": summary.get("tracks_count"),
        "track_roles": summary.get("track_role_counts"),
        "team_classifier": (
            summary.get("team_classifier", {}).get("engine")
        ),
        "pitch_reliable_frames": (
            summary.get("metric_tracking", {}).get("reliable_frames")
        ),
        "ball_observed_frames": (
            summary.get("ball_filter", {})
            .get("tracker", {})
            .get("observed_frames")
        ),
        "processing_fps": summary.get("processing_fps"),
    }
    print(json.dumps(required, indent=2))
    if summary.get("status") != "ok" or not summary.get("frames_processed"):
        raise RuntimeError("Runtime regression did not complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
