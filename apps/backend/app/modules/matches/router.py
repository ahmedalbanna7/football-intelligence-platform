import io
from uuid import uuid4

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from sqlalchemy.orm import Session

from app.db.dependencies import get_db
from app.models.match import Match
from app.models.match_video import MatchVideo
from app.queues.events import VideoUploadedEvent
from app.queues.publisher import publish_video_uploaded
from app.services.minio_client import (
    BUCKET_NAME,
    client,
)

router = APIRouter()


@router.post("/upload")
async def upload_match(
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")

    safe_filename = file.filename or "match-video"
    match = Match(
        title=safe_filename,
        status="uploaded",
    )
    db.add(match)
    db.flush()

    object_name = f"matches/{match.id}/{uuid4().hex}_{safe_filename}"

    client.put_object(
        BUCKET_NAME,
        object_name,
        io.BytesIO(content),
        len(content),
        content_type=file.content_type,
    )

    video = MatchVideo(
        match_id=match.id,
        object_name=object_name,
    )
    db.add(video)
    db.flush()

    db.commit()
    db.refresh(match)
    db.refresh(video)

    event = VideoUploadedEvent(
        match_id=match.id,
        video_id=video.id,
        bucket=BUCKET_NAME,
        object_name=object_name,
        filename=file.filename or object_name,
        content_type=file.content_type,
    )
    try:
        await publish_video_uploaded(event)
    except Exception as exc:
        match.status = "queue_failed"
        db.commit()
        raise HTTPException(
            status_code=503,
            detail="Video uploaded, but queue publishing failed",
        ) from exc

    match.status = "queued"
    db.commit()
    db.refresh(match)

    return {
        "match_id": match.id,
        "video_id": video.id,
        "filename": file.filename,
        "object_name": object_name,
        "status": match.status,
    }
