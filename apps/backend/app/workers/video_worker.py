import asyncio
from datetime import datetime
import traceback

from sqlalchemy.orm import Session

from app.db.database import SessionLocal
from app.models.match import Match
from app.models.player_roster_entry import PlayerRosterEntry
from app.models.track_player_assignment import TrackPlayerAssignment
from app.models.video_processing_job import VideoProcessingJob
from app.pipeline.video_pipeline import VideoPipeline
from app.queues.events import VideoUploadedEvent
from app.queues.consumer import consume_video_uploaded_events


def update_match_status(db: Session, match_id: int, status: str) -> None:
    match = db.get(Match, match_id)
    if match is None:
        return

    match.status = status
    db.commit()


def create_processing_job(
    db: Session,
    event: VideoUploadedEvent,
) -> VideoProcessingJob:
    job = VideoProcessingJob(
        match_id=event.match_id,
        video_id=event.video_id,
        status="processing",
    )
    db.add(job)
    db.commit()
    db.refresh(job)
    return job


def finish_processing_job(
    db: Session,
    job: VideoProcessingJob,
    status: str,
    result_json: dict | None = None,
    error_message: str | None = None,
) -> None:
    job.status = status
    job.finished_at = datetime.utcnow()
    job.result_json = result_json
    job.error_message = error_message
    db.commit()


def auto_assign_tracks_from_roster(
    db: Session,
    job: VideoProcessingJob,
    result: dict,
) -> list[dict]:
    tracks = (
        result
        .get("data", {})
        .get("jersey_number_recognition", {})
        .get("data", {})
        .get("tracks", [])
    )
    assignments: list[dict] = []

    for track in tracks:
        shirt_number = track.get("recognized_shirt_number")
        team_context = track.get("team_context")
        track_id = track.get("track_id")

        if shirt_number is None or track_id is None:
            continue
        if team_context not in {"primary_team", "opponent_team", "club_internal"}:
            continue

        roster_entry = find_roster_entry(
            db,
            match_id=job.match_id,
            team_context=team_context,
            shirt_number=shirt_number,
        )
        assignment = upsert_track_assignment_from_roster(
            db,
            job=job,
            track_id=track_id,
            team_context=team_context,
            shirt_number=shirt_number,
            player_name=(
                roster_entry.player_name
                if roster_entry is not None
                else f"Player #{shirt_number}"
            ),
            position=roster_entry.position if roster_entry is not None else None,
        )
        assignments.append(
            {
                "track_id": assignment.track_id,
                "team_context": assignment.team_context,
                "player_name": assignment.player_name,
                "shirt_number": assignment.shirt_number,
                "position": assignment.position,
                "source": "recognized_shirt_number",
            }
        )

    return assignments


def find_roster_entry(
    db: Session,
    match_id: int,
    team_context: str,
    shirt_number: int,
) -> PlayerRosterEntry | None:
    if team_context in {"primary_team", "club_internal"}:
        lookup_team_context = "primary_team"
        lookup_match_id = None
    else:
        lookup_team_context = "opponent_team"
        lookup_match_id = match_id

    query = (
        db.query(PlayerRosterEntry)
        .filter(PlayerRosterEntry.team_context == lookup_team_context)
        .filter(PlayerRosterEntry.shirt_number == shirt_number)
    )
    if lookup_match_id is None:
        query = query.filter(PlayerRosterEntry.match_id.is_(None))
    else:
        query = query.filter(PlayerRosterEntry.match_id == lookup_match_id)

    return query.first()


def upsert_track_assignment_from_roster(
    db: Session,
    job: VideoProcessingJob,
    track_id: int,
    team_context: str,
    shirt_number: int,
    player_name: str,
    position: str | None,
) -> TrackPlayerAssignment:
    assignment = (
        db.query(TrackPlayerAssignment)
        .filter(TrackPlayerAssignment.match_id == job.match_id)
        .filter(TrackPlayerAssignment.processing_job_id == job.id)
        .filter(TrackPlayerAssignment.track_id == track_id)
        .first()
    )

    if assignment is None:
        assignment = TrackPlayerAssignment(
            match_id=job.match_id,
            processing_job_id=job.id,
            track_id=track_id,
            team_context=team_context,
            player_name=player_name,
            shirt_number=shirt_number,
            position=position,
        )
        db.add(assignment)
    else:
        assignment.team_context = team_context
        assignment.player_name = player_name
        assignment.shirt_number = shirt_number
        assignment.position = position

    db.commit()
    db.refresh(assignment)
    return assignment


async def run_worker() -> None:
    pipeline = VideoPipeline()

    async for message, event in consume_video_uploaded_events():
        db = SessionLocal()
        job: VideoProcessingJob | None = None
        try:
            job = create_processing_job(db, event)
            update_match_status(db, event.match_id, "processing")
            result = pipeline.run(
                event.bucket,
                event.object_name,
                match_context=event.match_context,
            )
            auto_assignments = auto_assign_tracks_from_roster(db, job, result)
            result.setdefault("meta", {})["auto_player_assignments"] = auto_assignments
            finish_processing_job(db, job, "processed", result_json=result)
            update_match_status(db, event.match_id, "processed")
            await message.ack()
        except Exception as exc:
            if job is not None:
                finish_processing_job(
                    db,
                    job,
                    "failed",
                    error_message=str(exc),
                )
            update_match_status(db, event.match_id, "failed")
            traceback.print_exc()
            await message.nack(requeue=False)
        finally:
            db.close()


if __name__ == "__main__":
    asyncio.run(run_worker())
