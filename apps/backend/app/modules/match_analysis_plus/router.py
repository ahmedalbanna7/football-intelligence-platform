from datetime import datetime
from typing import Any

from fastapi import APIRouter
from fastapi import Depends
from fastapi import HTTPException
from pydantic import BaseModel
from sqlalchemy import desc
from sqlalchemy.orm import Session

from app.db.dependencies import get_db
from app.models.match import Match
from app.models.match_analysis_run import MatchAnalysisRun
from app.models.match_video import MatchVideo
from app.queues.events import MatchAnalysisRequestedEvent
from app.queues.publisher import publish_match_analysis_requested
from app.services.minio_client import BUCKET_NAME

router = APIRouter()


class MatchAnalysisRunRequest(BaseModel):
    mode: str = "FULL_ANALYSIS"
    max_frames: int = 450


def get_latest_video(db: Session, match_id: int) -> MatchVideo | None:
    return (
        db.query(MatchVideo)
        .filter(MatchVideo.match_id == match_id)
        .order_by(MatchVideo.id.desc())
        .first()
    )


def serialize_run(run: MatchAnalysisRun) -> dict[str, Any]:
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
    }


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
    run = (
        db.query(MatchAnalysisRun)
        .filter(MatchAnalysisRun.match_id == match_id)
        .filter(MatchAnalysisRun.id == run_id)
        .first()
    )
    if run is None:
        raise HTTPException(status_code=404, detail="Match analysis run not found")
    return serialize_run(run)
