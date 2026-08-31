from datetime import datetime, timedelta, timezone
import json
import io
import subprocess
from typing import Any

from fastapi import APIRouter, Query, Response
from fastapi import Depends
from fastapi import HTTPException
from pydantic import BaseModel
from pydantic import Field
from sqlalchemy import desc
from sqlalchemy.orm import Session

from app.db.dependencies import get_db
from app.models.match import Match
from app.models.match_analysis_run import MatchAnalysisRun
from app.models.match_video import MatchVideo
from app.queues.events import MatchAnalysisRequestedEvent
from app.queues.publisher import publish_match_analysis_requested
from app.services.minio_client import BUCKET_NAME, client
from app.tracking_quality.metrics import evaluate_release_suite
from app.tracking_quality.service import TrackingQualityService
from app.match_analysis_plus.reports_v2 import ReportsV2Builder

router = APIRouter()


class MatchAnalysisRunRequest(BaseModel):
    mode: str = "FULL_ANALYSIS"
    max_frames: int = 450
    start_frame: int = Field(default=0, ge=0)
    calibration_points: list[dict[str, float]] = Field(default_factory=list)
    reuse_run_id: int | None = Field(default=None, ge=1)


class TrackCorrectionRequest(BaseModel):
    action: str
    source_track_id: int
    target_track_id: int | None = None
    split_frame: int | None = None
    assigned_player_id: int | None = None
    assigned_team_number: int | None = None
    assigned_role_name: str | None = None
    note: str | None = Field(default=None, max_length=1000)


class TrackingBenchmarkRequest(BaseModel):
    ground_truth: dict[str, Any]
    iou_threshold: float = Field(default=0.5, ge=0.05, le=0.95)


class TrackingGroundTruthDraftRequest(BaseModel):
    start_frame: int = Field(default=0, ge=0)
    end_frame: int = Field(ge=0)
    sample_every_frames: int = Field(default=5, ge=1, le=120)
    track_ids: list[int] = Field(default_factory=list)
    scenario: str = Field(default="general", max_length=60)
    camera_style: str = Field(default="tactical", max_length=60)
    critical: bool = False


class TrackingGroundTruthSaveRequest(BaseModel):
    ground_truth: dict[str, Any]


class BallGroundTruthDraftRequest(BaseModel):
    start_frame: int = Field(default=0, ge=0)
    end_frame: int = Field(ge=0)
    sample_every_frames: int = Field(default=3, ge=1, le=120)
    scenario: str = Field(default="general", max_length=60)
    camera_style: str = Field(default="tactical", max_length=60)
    critical: bool = False


class BallGroundTruthBenchmarkRequest(BaseModel):
    ground_truth: dict[str, Any]
    tolerance_pixels: float | None = Field(default=None, ge=1.0, le=500.0)


class TrackingCriticalRangeSuggestionRequest(BaseModel):
    padding_frames: int = Field(default=20, ge=5, le=150)
    max_ranges: int = Field(default=12, ge=1, le=50)


class TrackingCriticalRange(BaseModel):
    start_frame: int = Field(ge=0)
    end_frame: int = Field(ge=0)
    scenario: str = Field(default="critical", max_length=60)


class TrackingReleasePlanRequest(BaseModel):
    clip_size: int = Field(default=750, ge=500, le=1000)
    overlap_frames: int = Field(default=0, ge=0, le=250)
    camera_style: str = Field(default="tactical", max_length=60)
    critical_ranges: list[TrackingCriticalRange] = Field(default_factory=list)


class TrackingReleaseSuiteCase(BaseModel):
    match_id: int
    run_id: int
    ground_truth: dict[str, Any]
    iou_threshold: float = Field(default=0.5, ge=0.05, le=0.95)


class TrackingReleaseSuiteRequest(BaseModel):
    cases: list[TrackingReleaseSuiteCase] = Field(min_length=1)
    thresholds: dict[str, Any] = Field(default_factory=dict)


class ReportComparisonCase(BaseModel):
    match_id: int
    run_id: int


class ReportComparisonRequest(BaseModel):
    cases: list[ReportComparisonCase] = Field(min_length=2, max_length=8)


quality_service = TrackingQualityService()
report_builder = ReportsV2Builder()


def load_json_object(object_name: str) -> dict[str, Any]:
    response = client.get_object(BUCKET_NAME, object_name)
    try:
        return json.loads(response.read().decode("utf-8"))
    finally:
        response.close()
        response.release_conn()


def load_binary_object(object_name: str) -> bytes:
    response = client.get_object(BUCKET_NAME, object_name)
    try:
        return response.read()
    finally:
        response.close()
        response.release_conn()


def get_latest_video(db: Session, match_id: int) -> MatchVideo | None:
    return (
        db.query(MatchVideo)
        .filter(MatchVideo.match_id == match_id)
        .order_by(MatchVideo.id.desc())
        .first()
    )


def serialize_run(run: MatchAnalysisRun) -> dict[str, Any]:
    quality = run.quality_assessment
    return {
        "id": run.id,
        "match_id": run.match_id,
        "video_id": run.video_id,
        "mode": run.mode,
        "status": run.status,
        "source": run.source,
        "max_frames": run.max_frames,
        "analysis_config": run.analysis_config_json or {},
        "output_object": run.output_object,
        "summary_object": run.summary_object,
        "thumbnail_object": run.thumbnail_object,
        "summary": run.summary_json,
        "error_message": run.error_message,
        "created_at": run.created_at,
        "started_at": run.started_at,
        "finished_at": run.finished_at,
        "quality": {
            "status": quality.status,
            "average_identity_confidence": quality.average_identity_confidence,
            "tracks_needing_review": quality.tracks_needing_review,
            "benchmark_status": quality.benchmark_status,
            "idf1": quality.idf1,
            "hota": quality.hota,
        }
        if quality is not None
        else None,
    }


def get_run_or_404(db: Session, match_id: int, run_id: int) -> MatchAnalysisRun:
    run = (
        db.query(MatchAnalysisRun)
        .filter(MatchAnalysisRun.match_id == match_id)
        .filter(MatchAnalysisRun.id == run_id)
        .first()
    )
    if run is None:
        raise HTTPException(status_code=404, detail="Match analysis run not found")
    return run


def resolve_player_detection_mode(
    db: Session,
    source_run: MatchAnalysisRun,
) -> str | None:
    """Recover detector provenance through nested detection-cache reuse."""

    cached_placeholders = {"cached-football-detections"}
    current: MatchAnalysisRun | None = source_run
    seen_run_ids: set[int] = set()
    fallback: str | None = None

    while current is not None and current.id not in seen_run_ids:
        seen_run_ids.add(current.id)
        summary = current.summary_json or {}
        config = current.analysis_config_json or {}
        candidates = (
            summary.get("player_detection_mode"),
            config.get("reuse_model_mode"),
            summary.get("model_mode"),
        )
        for candidate in candidates:
            if not candidate:
                continue
            mode = str(candidate)
            fallback = fallback or mode
            if mode not in cached_placeholders:
                return mode

        parent_run_id = config.get("reuse_run_id")
        if parent_run_id is None:
            break
        try:
            current = db.get(MatchAnalysisRun, int(parent_run_id))
        except (TypeError, ValueError):
            break

    return fallback


@router.get("/options/modes")
def get_match_analysis_modes():
    return {
        "items": [
            {
                "value": "FULL_ANALYSIS",
                "label": "Full match analysis",
                "description": "Players, ball, stable tracking, teams, movement, possession, and pitch radar.",
            },
        ]
    }


@router.post("/quality/release-gate/suite")
def evaluate_tracking_release_suite(
    payload: TrackingReleaseSuiteRequest,
    db: Session = Depends(get_db),
):
    measured_cases: list[dict[str, Any]] = []
    case_results: list[dict[str, Any]] = []
    try:
        for release_case in payload.cases:
            run = get_run_or_404(db, release_case.match_id, release_case.run_id)
            metrics = quality_service.benchmark(
                db,
                run,
                release_case.ground_truth,
                release_case.iou_threshold,
            )
            measured_cases.append(metrics)
            case_results.append(
                {
                    "match_id": release_case.match_id,
                    "run_id": release_case.run_id,
                    "metrics": metrics,
                }
            )
        return {
            "suite": evaluate_release_suite(measured_cases, payload.thresholds),
            "cases": case_results,
        }
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/{match_id}")
def list_match_analysis_runs(
    match_id: int,
    db: Session = Depends(get_db),
):
    match = db.get(Match, match_id)
    if match is None:
        raise HTTPException(status_code=404, detail="Match not found")

    runs = (
        db.query(MatchAnalysisRun)
        .filter(MatchAnalysisRun.match_id == match_id)
        .order_by(desc(MatchAnalysisRun.created_at))
        .all()
    )
    return {
        "match_id": match.id,
        "match_title": match.title,
        "runs": [serialize_run(run) for run in runs],
        "latest": serialize_run(runs[0]) if runs else None,
    }


@router.post("/{match_id}/run")
async def run_match_analysis_plus(
    match_id: int,
    payload: MatchAnalysisRunRequest,
    db: Session = Depends(get_db),
):
    match = db.get(Match, match_id)
    if match is None:
        raise HTTPException(status_code=404, detail="Match not found")

    video = get_latest_video(db, match_id)
    if video is None:
        raise HTTPException(status_code=404, detail="No uploaded video found for this match")

    reuse_detections_object: str | None = None
    reuse_model_mode: str | None = None
    reuse_ball_detection_mode: str | None = None
    if payload.reuse_run_id is not None:
        source_run = get_run_or_404(db, match_id, payload.reuse_run_id)
        if source_run.video_id != video.id:
            raise HTTPException(status_code=400, detail="Detection cache belongs to another video")
        reuse_detections_object = (
            ((source_run.summary_json or {}).get("performance") or {}).get("detection_cache") or {}
        ).get("object_name")
        if not reuse_detections_object:
            raise HTTPException(status_code=400, detail="Selected run has no reusable detection cache")
        reuse_ball_detection_mode = (source_run.summary_json or {}).get(
            "ball_detection_mode"
        )
        reuse_model_mode = resolve_player_detection_mode(db, source_run)

    run = MatchAnalysisRun(
        match_id=match_id,
        video_id=video.id,
        mode="FULL_ANALYSIS",
        status="queued",
        source="native-runner",
        max_frames=max(payload.max_frames, 0),
        analysis_config_json={
            "start_frame": max(payload.start_frame, 0),
            "calibration_points": payload.calibration_points,
            "reuse_run_id": payload.reuse_run_id,
            "reuse_detections_object": reuse_detections_object,
            "reuse_model_mode": reuse_model_mode,
            "reuse_ball_detection_mode": reuse_ball_detection_mode,
            "runtime_progress": {
                "stage": "queued",
                "processed_frames": 0,
                "percent": 0.0,
                "eta_seconds": None,
            },
        },
    )
    db.add(run)
    db.commit()
    db.refresh(run)

    artifact_prefix = f"matches/{match_id}/match-analysis-plus/runs/{run.id}"
    try:
        await publish_match_analysis_requested(
            MatchAnalysisRequestedEvent(
                run_id=run.id,
                match_id=match_id,
                video_id=video.id,
                bucket=BUCKET_NAME,
                object_name=video.object_name,
                artifact_prefix=artifact_prefix,
                mode=run.mode,
                max_frames=run.max_frames,
                start_frame=max(payload.start_frame, 0),
                calibration_points=payload.calibration_points,
                reuse_detections_object=reuse_detections_object,
                reuse_model_mode=reuse_model_mode,
                reuse_ball_detection_mode=reuse_ball_detection_mode,
            )
        )
    except Exception as exc:
        run.status = "failed"
        run.error_message = str(exc)
        run.finished_at = datetime.now(timezone.utc).replace(tzinfo=None)
        db.commit()
        db.refresh(run)
        raise HTTPException(
            status_code=502,
            detail=f"Could not queue Match Analysis job: {exc}",
        ) from exc

    match.status = "queued"
    db.commit()
    db.refresh(run)
    return serialize_run(run)


@router.get("/{match_id}/calibration-frame")
def get_calibration_frame(
    match_id: int,
    frame_index: int = Query(default=0, ge=0),
    db: Session = Depends(get_db),
):
    video = get_latest_video(db, match_id)
    if video is None:
        raise HTTPException(status_code=404, detail="No uploaded video found for this match")

    source_url = client.presigned_get_object(
        BUCKET_NAME,
        video.object_name,
        expires=timedelta(minutes=10),
    )
    probe = subprocess.run(
        [
            "ffprobe", "-v", "error", "-select_streams", "v:0",
            "-show_entries", "stream=avg_frame_rate,nb_frames,width,height",
            "-of", "json", source_url,
        ],
        capture_output=True,
        check=False,
        timeout=45,
    )
    if probe.returncode != 0:
        raise HTTPException(status_code=422, detail="Could not inspect match video")
    stream = (json.loads(probe.stdout.decode("utf-8")) or {}).get("streams", [{}])[0]
    rate_parts = str(stream.get("avg_frame_rate") or "25/1").split("/", 1)
    fps = float(rate_parts[0]) / max(float(rate_parts[1]) if len(rate_parts) > 1 else 1.0, 1e-6)
    source_frames = int(stream.get("nb_frames") or 0)
    selected_frame = min(frame_index, max(0, source_frames - 1)) if source_frames else frame_index
    timestamp = selected_frame / max(fps, 1e-6)
    extraction = subprocess.run(
        [
            "ffmpeg", "-v", "error", "-ss", f"{timestamp:.6f}", "-i", source_url,
            "-frames:v", "1", "-q:v", "2", "-f", "image2pipe", "-vcodec", "mjpeg", "pipe:1",
        ],
        capture_output=True,
        check=False,
        timeout=60,
    )
    if extraction.returncode != 0 or not extraction.stdout:
        raise HTTPException(status_code=422, detail="Could not read calibration frame")
    return Response(
        content=extraction.stdout,
        media_type="image/jpeg",
        headers={
            "X-Frame-Index": str(selected_frame),
            "X-Video-Width": str(stream.get("width") or 0),
            "X-Video-Height": str(stream.get("height") or 0),
            "Cache-Control": "no-store",
        },
    )


@router.get("/{match_id}/runs/{run_id}")
def get_match_analysis_run(
    match_id: int,
    run_id: int,
    db: Session = Depends(get_db),
):
    run = get_run_or_404(db, match_id, run_id)
    return serialize_run(run)


@router.get("/{match_id}/runs/{run_id}/report")
def get_match_analysis_report(
    match_id: int,
    run_id: int,
    db: Session = Depends(get_db),
):
    run = get_run_or_404(db, match_id, run_id)
    reports = (run.summary_json or {}).get("reports_v2") or {}
    object_name = (reports.get("artifacts") or {}).get("json")
    if not object_name:
        raise HTTPException(status_code=404, detail="Reports v2 is not available for this run")
    return load_json_object(object_name)


@router.get("/{match_id}/runs/{run_id}/report.pdf")
def get_match_analysis_report_pdf(
    match_id: int,
    run_id: int,
    db: Session = Depends(get_db),
):
    run = get_run_or_404(db, match_id, run_id)
    reports = (run.summary_json or {}).get("reports_v2") or {}
    object_name = (reports.get("artifacts") or {}).get("pdf")
    if not object_name:
        raise HTTPException(status_code=404, detail="Reports v2 PDF is not available for this run")
    return Response(
        content=load_binary_object(object_name),
        media_type="application/pdf",
        headers={"Content-Disposition": f'inline; filename="match-{match_id}-run-{run_id}.pdf"'},
    )


@router.post("/reports/compare")
def compare_match_analysis_reports(
    payload: ReportComparisonRequest,
    db: Session = Depends(get_db),
):
    reports: list[tuple[int, int, dict[str, Any]]] = []
    for case in payload.cases:
        run = get_run_or_404(db, case.match_id, case.run_id)
        report_summary = (run.summary_json or {}).get("reports_v2") or {}
        object_name = (report_summary.get("artifacts") or {}).get("json")
        if not object_name:
            raise HTTPException(
                status_code=400,
                detail=f"Reports v2 is not available for run {case.run_id}",
            )
        reports.append((case.match_id, case.run_id, load_json_object(object_name)))
    return report_builder.compare(reports)


@router.get("/{match_id}/runs/{run_id}/quality")
def get_tracking_quality(
    match_id: int,
    run_id: int,
    db: Session = Depends(get_db),
):
    run = get_run_or_404(db, match_id, run_id)
    return quality_service.get_quality(db, run)


@router.post("/{match_id}/quality/release-gate/plan")
def build_tracking_release_plan(
    match_id: int,
    payload: TrackingReleasePlanRequest,
    db: Session = Depends(get_db),
):
    video = get_latest_video(db, match_id)
    if video is None:
        raise HTTPException(status_code=404, detail="No uploaded video found for this match")
    if payload.overlap_frames >= payload.clip_size:
        raise HTTPException(status_code=400, detail="overlap_frames must be smaller than clip_size")
    for critical_range in payload.critical_ranges:
        if critical_range.end_frame < critical_range.start_frame:
            raise HTTPException(status_code=400, detail="Critical range end must follow its start")

    source_url = client.presigned_get_object(
        BUCKET_NAME,
        video.object_name,
        expires=timedelta(minutes=10),
    )
    probe = subprocess.run(
        [
            "ffprobe", "-v", "error", "-select_streams", "v:0",
            "-show_entries", "stream=avg_frame_rate,nb_frames,duration",
            "-of", "json", source_url,
        ],
        capture_output=True,
        check=False,
        timeout=45,
    )
    if probe.returncode != 0:
        raise HTTPException(status_code=422, detail="Could not inspect match video")
    stream = (json.loads(probe.stdout.decode("utf-8")) or {}).get("streams", [{}])[0]
    rate_parts = str(stream.get("avg_frame_rate") or "25/1").split("/", 1)
    fps = float(rate_parts[0]) / max(float(rate_parts[1]) if len(rate_parts) > 1 else 1.0, 1e-6)
    source_frames = int(stream.get("nb_frames") or 0)
    if source_frames <= 0:
        source_frames = int(round(float(stream.get("duration") or 0.0) * fps))
    if source_frames <= 0:
        raise HTTPException(status_code=422, detail="Video frame count is unavailable")

    stride = payload.clip_size - payload.overlap_frames
    clips: list[dict[str, Any]] = []
    start_frame = 0
    while start_frame < source_frames:
        end_frame = min(source_frames - 1, start_frame + payload.clip_size - 1)
        matched_ranges = [
            item
            for item in payload.critical_ranges
            if item.start_frame <= end_frame and item.end_frame >= start_frame
        ]
        scenarios = sorted({item.scenario.lower() for item in matched_ranges})
        clips.append(
            {
                "index": len(clips),
                "start_frame": start_frame,
                "end_frame": end_frame,
                "frame_count": end_frame - start_frame + 1,
                "camera_style": payload.camera_style.lower(),
                "critical": bool(matched_ranges),
                "scenarios": scenarios or ["baseline"],
                "run_request": {
                    "mode": "FULL_ANALYSIS",
                    "start_frame": start_frame,
                    "max_frames": end_frame - start_frame + 1,
                },
            }
        )
        if end_frame >= source_frames - 1:
            break
        start_frame += stride
    return {
        "match_id": match_id,
        "video_id": video.id,
        "source_frames": source_frames,
        "fps": round(fps, 4),
        "clip_size": payload.clip_size,
        "overlap_frames": payload.overlap_frames,
        "clips_count": len(clips),
        "critical_clips_count": sum(1 for clip in clips if clip["critical"]),
        "clips": clips,
    }


@router.post("/{match_id}/runs/{run_id}/quality/corrections")
def create_tracking_correction(
    match_id: int,
    run_id: int,
    payload: TrackCorrectionRequest,
    db: Session = Depends(get_db),
):
    run = get_run_or_404(db, match_id, run_id)
    try:
        correction = quality_service.apply_correction(db, run, payload.model_dump())
        recalculation = None
        if payload.action.lower() in {
            "reject",
            "merge",
            "split",
            "assign_player",
            "change_team",
            "change_role",
        }:
            recalculation = quality_service.recalculate(db, run)
        response = quality_service.get_quality(db, run)
        response["correction_id"] = correction.id
        response["recalculation"] = recalculation
        return response
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/{match_id}/runs/{run_id}/quality/corrections/{correction_id}/undo")
def undo_tracking_correction(
    match_id: int,
    run_id: int,
    correction_id: int,
    db: Session = Depends(get_db),
):
    run = get_run_or_404(db, match_id, run_id)
    try:
        quality_service.undo_correction(db, run, correction_id)
        quality_service.recalculate(db, run)
        return quality_service.get_quality(db, run)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/{match_id}/runs/{run_id}/quality/recalculate")
def recalculate_tracking_quality(
    match_id: int,
    run_id: int,
    db: Session = Depends(get_db),
):
    run = get_run_or_404(db, match_id, run_id)
    try:
        result = quality_service.recalculate(db, run)
        return {
            **result,
            "quality": quality_service.get_quality(db, run),
        }
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/{match_id}/runs/{run_id}/quality/benchmark")
def benchmark_tracking_quality(
    match_id: int,
    run_id: int,
    payload: TrackingBenchmarkRequest,
    db: Session = Depends(get_db),
):
    run = get_run_or_404(db, match_id, run_id)
    try:
        metrics = quality_service.benchmark(
            db,
            run,
            payload.ground_truth,
            payload.iou_threshold,
        )
        return {
            "metrics": metrics,
            "quality": quality_service.get_quality(db, run),
        }
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/{match_id}/runs/{run_id}/quality/ground-truth/draft")
def build_tracking_ground_truth_draft(
    match_id: int,
    run_id: int,
    payload: TrackingGroundTruthDraftRequest,
    db: Session = Depends(get_db),
):
    run = get_run_or_404(db, match_id, run_id)
    try:
        return quality_service.build_ground_truth_draft(
            db=db,
            run=run,
            start_frame=payload.start_frame,
            end_frame=payload.end_frame,
            sample_every_frames=payload.sample_every_frames,
            track_ids=payload.track_ids,
            scenario=payload.scenario,
            camera_style=payload.camera_style,
            critical=payload.critical,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/{match_id}/runs/{run_id}/quality/ground-truth")
def get_tracking_ground_truth(
    match_id: int,
    run_id: int,
    db: Session = Depends(get_db),
):
    run = get_run_or_404(db, match_id, run_id)
    try:
        return quality_service.get_tracking_ground_truth(db, run)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.put("/{match_id}/runs/{run_id}/quality/ground-truth")
def save_tracking_ground_truth(
    match_id: int,
    run_id: int,
    payload: TrackingGroundTruthSaveRequest,
    db: Session = Depends(get_db),
):
    run = get_run_or_404(db, match_id, run_id)
    try:
        return quality_service.save_tracking_ground_truth(db, run, payload.ground_truth)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/{match_id}/runs/{run_id}/quality/ball-ground-truth/draft")
def build_ball_ground_truth_draft(
    match_id: int,
    run_id: int,
    payload: BallGroundTruthDraftRequest,
    db: Session = Depends(get_db),
):
    run = get_run_or_404(db, match_id, run_id)
    try:
        return quality_service.build_ball_ground_truth_draft(
            run=run,
            start_frame=payload.start_frame,
            end_frame=payload.end_frame,
            sample_every_frames=payload.sample_every_frames,
            scenario=payload.scenario,
            camera_style=payload.camera_style,
            critical=payload.critical,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/{match_id}/runs/{run_id}/quality/ball-ground-truth")
def get_ball_ground_truth(
    match_id: int,
    run_id: int,
    db: Session = Depends(get_db),
):
    run = get_run_or_404(db, match_id, run_id)
    try:
        return quality_service.get_ball_ground_truth(run)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.put("/{match_id}/runs/{run_id}/quality/ball-ground-truth")
def save_ball_ground_truth(
    match_id: int,
    run_id: int,
    payload: TrackingGroundTruthSaveRequest,
    db: Session = Depends(get_db),
):
    run = get_run_or_404(db, match_id, run_id)
    try:
        return quality_service.save_ball_ground_truth(db, run, payload.ground_truth)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/{match_id}/runs/{run_id}/quality/ball-ground-truth/benchmark")
def benchmark_ball_ground_truth(
    match_id: int,
    run_id: int,
    payload: BallGroundTruthBenchmarkRequest,
    db: Session = Depends(get_db),
):
    run = get_run_or_404(db, match_id, run_id)
    try:
        return quality_service.benchmark_ball_ground_truth(
            db,
            run,
            payload.ground_truth,
            payload.tolerance_pixels,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/{match_id}/runs/{run_id}/quality/critical-ranges/suggest")
def suggest_tracking_critical_ranges(
    match_id: int,
    run_id: int,
    payload: TrackingCriticalRangeSuggestionRequest,
    db: Session = Depends(get_db),
):
    run = get_run_or_404(db, match_id, run_id)
    try:
        return quality_service.suggest_critical_ranges(
            db=db,
            run=run,
            padding_frames=payload.padding_frames,
            max_ranges=payload.max_ranges,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
