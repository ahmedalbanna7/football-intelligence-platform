import asyncio
from datetime import UTC, datetime
import traceback

from sqlalchemy.orm import Session

from app.db.database import SessionLocal
from app.match_analysis_plus import MatchAnalysisPlusRunner
from app.models.match import Match
from app.models.match_analysis_run import MatchAnalysisRun
from app.queues.consumer import consume_match_analysis_requested_events
from app.queues.events import MatchAnalysisRequestedEvent
from app.tracking_quality.service import TrackingQualityService


quality_service = TrackingQualityService()


def update_match_status(db: Session, match_id: int, status: str) -> None:
    match = db.get(Match, match_id)
    if match is None:
        return
    match.status = status
    db.commit()


def mark_run_processing(db: Session, event: MatchAnalysisRequestedEvent) -> MatchAnalysisRun:
    run = db.get(MatchAnalysisRun, event.run_id)
    if run is None:
        raise ValueError(f"Match analysis run {event.run_id} not found")
    run.status = "processing"
    run.started_at = datetime.now(UTC).replace(tzinfo=None)
    run.finished_at = None
    run.error_message = None
    db.commit()
    db.refresh(run)
    return run


def finish_run(
    db: Session,
    run: MatchAnalysisRun,
    status: str,
    summary: dict | None = None,
    error_message: str | None = None,
) -> None:
    run.status = status
    run.finished_at = datetime.now(UTC).replace(tzinfo=None)
    run.error_message = error_message
    if summary is not None:
        run.output_object = summary.get("output_object")
        run.summary_object = summary.get("summary_object")
        run.thumbnail_object = summary.get("thumbnail_object")
        run.summary_json = summary
    db.commit()
    if summary is not None and status == "processed":
        quality_service.sync_from_summary(db, run, summary)


async def run_worker() -> None:
    runner = MatchAnalysisPlusRunner()

    async for message, event in consume_match_analysis_requested_events():
        db = SessionLocal()
        run: MatchAnalysisRun | None = None
        try:
            run = mark_run_processing(db, event)
            update_match_status(db, event.match_id, "processing")
            summary = await asyncio.to_thread(
                runner.run,
                run_id=event.run_id,
                match_id=event.match_id,
                bucket=event.bucket,
                object_name=event.object_name,
                artifact_prefix=event.artifact_prefix,
                mode=event.mode,
                max_frames=event.max_frames,
            )
            finish_run(db, run, "processed", summary=summary)
            update_match_status(db, event.match_id, "processed")
            await message.ack()
        except Exception as exc:
            if run is not None:
                finish_run(db, run, "failed", error_message=str(exc))
            update_match_status(db, event.match_id, "failed")
            traceback.print_exc()
            await message.nack(requeue=False)
        finally:
            db.close()


if __name__ == "__main__":
    asyncio.run(run_worker())
