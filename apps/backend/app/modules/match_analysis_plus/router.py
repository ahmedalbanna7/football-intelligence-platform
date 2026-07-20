from datetime import datetime
from typing import Any

from fastapi import APIRouter
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
from app.services.minio_client import BUCKET_NAME
from app.tracking_quality.service import TrackingQualityService

router = APIRouter()


class MatchAnalysisRunRequest(BaseModel):
    mode: str = "FULL_ANALYSIS"
    max_frames: int = 450


class TrackCorrectionRequest(BaseModel):
    action: str
    source_track_id: int
    target_track_id: int | None = None
    split_frame: int | None = None
    assigned_player_id: int | None = None
    assigned_team_number: int | None = None
    note: str | None = Field(default=None, max_length=1000)


class TrackingBenchmarkRequest(BaseModel):
    ground_truth: dict[str, Any]
    iou_threshold: float = Field(default=0.5, ge=0.05, le=0.95)


quality_service = TrackingQualityService()


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

    run = MatchAnalysisRun(
        match_id=match_id,
        video_id=video.id,
        mode="FULL_ANALYSIS",
        status="queued",
        source="sports-main",
        max_frames=max(payload.max_frames, 0),
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
            )
        )
    except Exception as exc:
        run.status = "failed"
        run.error_message = str(exc)
        run.finished_at = datetime.utcnow()
        db.commit()
        db.refresh(run)
        raise HTTPException(
            status_code=502,
            detail=f"Could not queue Match Analysis + job: {exc}",
        ) from exc

    match.status = "queued"
    db.commit()
    db.refresh(run)
    return serialize_run(run)


@router.get("/{match_id}/runs/{run_id}")
def get_match_analysis_run(
    match_id: int,
    run_id: int,
    db: Session = Depends(get_db),
):
    run = get_run_or_404(db, match_id, run_id)
    return serialize_run(run)


@router.get("/{match_id}/runs/{run_id}/quality")
def get_tracking_quality(
    match_id: int,
    run_id: int,
    db: Session = Depends(get_db),
):
    run = get_run_or_404(db, match_id, run_id)
    return quality_service.get_quality(db, run)


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
        if payload.action.lower() in {"reject", "merge", "split", "assign_player", "change_team"}:
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
