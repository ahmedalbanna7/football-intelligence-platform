import asyncio
from datetime import UTC, datetime
import traceback
from typing import Any

from sqlalchemy.orm import Session

from app.db.database import SessionLocal
from app.match_analysis_plus import MatchAnalysisPlusRunner
from app.models.match import Match
from app.models.match_analysis_run import MatchAnalysisRun
from app.queues.consumer import consume_match_analysis_requested_events
from app.queues.events import MatchAnalysisRequestedEvent
from app.tracking_quality.service import TrackingQualityService


quality_service = TrackingQualityService()


def _selected_kit_object(
    team: Any | None,
    source: str | None,
) -> str | None:
    if team is None:
        return None
    primary = getattr(team, "primary_kit_image_object_name", None)
    alternate = getattr(team, "alternate_kit_image_object_name", None)
    if source == "alternate":
        return alternate or primary
    return primary or alternate


def build_analysis_team_context(match: Match) -> dict[str, Any]:
    primary_reference = match.primary_team_profile or match.primary_team
    secondary_team = match.another_team or match.opponent_team
    primary_name = (
        match.primary_team_name
        or getattr(primary_reference, "team_name", None)
        or getattr(primary_reference, "name", None)
        or "Team 1"
    )
    secondary_name = (
        match.another_team_name
        or match.opponent_team_name
        or getattr(secondary_team, "name", None)
        or "Team 2"
    )

    return {
        "match_category": match.match_category,
        "match_type": match.match_type,
        "matchup_type": match.matchup_type,
        "team_labels": {
            "1": primary_name,
            "2": secondary_name,
        },
        "kit_references": {
            "team_1_selected": _selected_kit_object(
                primary_reference,
                match.primary_team_kit_source,
            ),
            "team_1_primary": getattr(
                primary_reference,
                "primary_kit_image_object_name",
                None,
            ),
            "team_1_alternate": getattr(
                primary_reference,
                "alternate_kit_image_object_name",
                None,
            ),
            "team_2_selected": _selected_kit_object(
                secondary_team,
                match.another_team_kit_source,
            ),
            "team_2_primary": getattr(
                secondary_team,
                "primary_kit_image_object_name",
                None,
            ),
            "team_2_alternate": getattr(
                secondary_team,
                "alternate_kit_image_object_name",
                None,
            ),
        },
    }


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


def run_is_already_complete(run: MatchAnalysisRun | None) -> bool:
    return bool(
        run is not None
        and run.status == "processed"
        and run.summary_json
        and run.output_object
    )


async def settle_message(message: Any, action: str) -> bool:
    try:
        if action == "ack":
            await message.ack()
        else:
            await message.nack(requeue=False)
        return True
    except Exception:
        print(
            f"RabbitMQ {action} failed; the robust consumer will reconnect "
            "and idempotency will handle any redelivery.",
            flush=True,
        )
        traceback.print_exc()
        return False


async def process_event(
    runner: MatchAnalysisPlusRunner,
    message: Any,
    event: MatchAnalysisRequestedEvent,
) -> None:
    db = SessionLocal()
    run: MatchAnalysisRun | None = None
    try:
        run = db.get(MatchAnalysisRun, event.run_id)
        if run_is_already_complete(run):
            update_match_status(db, event.match_id, "processed")
            await settle_message(message, "ack")
            return

        run = mark_run_processing(db, event)
        update_match_status(db, event.match_id, "processing")
        match = db.get(Match, event.match_id)
        team_context = build_analysis_team_context(match) if match is not None else {}
        summary = await asyncio.to_thread(
            runner.run,
            run_id=event.run_id,
            match_id=event.match_id,
            bucket=event.bucket,
            object_name=event.object_name,
            artifact_prefix=event.artifact_prefix,
            mode=event.mode,
            max_frames=event.max_frames,
            start_frame=event.start_frame,
            calibration_points=event.calibration_points,
            team_context=team_context,
        )
        finish_run(db, run, "processed", summary=summary)
        update_match_status(db, event.match_id, "processed")
    except Exception as exc:
        if run is not None and not run_is_already_complete(run):
            finish_run(db, run, "failed", error_message=str(exc))
            update_match_status(db, event.match_id, "failed")
        traceback.print_exc()
        await settle_message(message, "nack")
        return
    finally:
        db.close()

    await settle_message(message, "ack")


async def run_worker() -> None:
    runner = MatchAnalysisPlusRunner()

    while True:
        try:
            async for message, event in consume_match_analysis_requested_events():
                await process_event(runner, message, event)
        except asyncio.CancelledError:
            raise
        except Exception:
            print(
                "Match Analysis + consumer connection failed; reconnecting.",
                flush=True,
            )
            traceback.print_exc()
            await asyncio.sleep(2)


if __name__ == "__main__":
    asyncio.run(run_worker())
