from fastapi import APIRouter
from fastapi import Depends
from fastapi import HTTPException
from fastapi import Query
from sqlalchemy.orm import Session

from app.db.dependencies import get_db
from app.first_analysis import FirstAnalysisRunner
from app.models.match import Match
from app.models.match_video import MatchVideo
from app.services.minio_client import BUCKET_NAME
from app.services.minio_client import client

router = APIRouter()


def get_latest_video(db: Session, match_id: int) -> MatchVideo | None:
    return (
        db.query(MatchVideo)
        .filter(MatchVideo.match_id == match_id)
        .order_by(MatchVideo.id.desc())
        .first()
    )


@router.post("/{match_id}/run")
def run_first_analysis(
    match_id: int,
    max_frames: int = Query(450, ge=0, le=200000),
    db: Session = Depends(get_db),
):
    match = db.get(Match, match_id)
    if match is None:
        raise HTTPException(status_code=404, detail="Match not found")

    video = get_latest_video(db, match_id)
    if video is None:
        raise HTTPException(status_code=404, detail="No uploaded video found for this match")

    try:
        result = FirstAnalysisRunner().run(
            bucket=BUCKET_NAME,
            object_name=video.object_name,
            match_id=match_id,
            max_frames=max_frames,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return result


@router.get("/{match_id}")
def get_first_analysis(
    match_id: int,
    db: Session = Depends(get_db),
):
    match = db.get(Match, match_id)
    if match is None:
        raise HTTPException(status_code=404, detail="Match not found")

    object_name = f"matches/{match_id}/first-analysis/summary.json"
    try:
        response = client.get_object(BUCKET_NAME, object_name)
        try:
            payload = b"".join(response.stream(32 * 1024))
        finally:
            response.close()
            response.release_conn()
        return {
            "exists": True,
            "summary": __import__("json").loads(payload.decode("utf-8")),
        }
    except Exception:
        return {
            "exists": False,
            "summary": None,
        }
