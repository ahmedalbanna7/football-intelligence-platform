import asyncio
from datetime import UTC, datetime
import traceback
from typing import Any

from sqlalchemy.orm import Session

from app.db.database import SessionLocal
from app.match_analysis_plus import MatchAnalysisPlusRunner
from app.models.match import Match
from app.models.match_analysis_run import MatchAnalysisRun
from app.models.player import Player
from app.models.player_roster_entry import PlayerRosterEntry
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


def _serialize_roster_player(player: Any, player_id: int | None = None) -> dict[str, Any]:
    return {
        "player_id": player_id,
        "name": getattr(player, "player_name", None) or getattr(player, "name", None),
        "shirt_number": getattr(player, "shirt_number", None)
        if hasattr(player, "shirt_number")
        else getattr(player, "jersey_number", None),
        "primary_zone": getattr(player, "primary_zone", None),
        "position_label": getattr(player, "position_label", None),
    }


def build_analysis_team_context(
    match: Match,
    db: Session | None = None,
) -> dict[str, Any]:
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
    primary_roster: list[dict[str, Any]] = []
    secondary_roster: list[dict[str, Any]] = []
    if db is not None:
        primary_entries = (
            db.query(PlayerRosterEntry)
            .filter(PlayerRosterEntry.match_id.is_(None))
            .filter(PlayerRosterEntry.team_context == "primary_team")
            .order_by(PlayerRosterEntry.shirt_number)
            .all()
        )
        primary_roster = [_serialize_roster_player(item) for item in primary_entries]
        if getattr(secondary_team, "id", None) is not None:
            secondary_players = (
                db.query(Player)
                .filter(Player.team_id == secondary_team.id)
                .order_by(Player.jersey_number, Player.name)
                .all()
            )
            secondary_roster = [
                _serialize_roster_player(item, item.id)
                for item in secondary_players
            ]

    return {
        "match_category": match.match_category,
        "match_type": match.match_type,
        "matchup_type": match.matchup_type,
        "analysis_scope": match.analysis_scope,
        "analyze_primary_players": bool(match.analyze_primary_players),
        "analyze_opponent_players": bool(match.analyze_opponent_players),
        "formations": {
            "1": match.formation,
            "2": match.another_formation,
        },
        "rosters": {
            "1": primary_roster,
            "2": secondary_roster,
        },
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
            "team_1_goalkeeper": getattr(
                primary_reference,
                "goalkeeper_kit_image_object_name",
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
            "team_2_goalkeeper": getattr(
                secondary_team,
                "goalkeeper_kit_image_object_name",
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


def update_run_progress(run_id: int, progress: dict[str, Any]) -> None:
    progress_db = SessionLocal()
    try:
        progress_run = progress_db.get(MatchAnalysisRun, run_id)
        if progress_run is None:
            return
        config = dict(progress_run.analysis_config_json or {})
        config["runtime_progress"] = progress
        progress_run.analysis_config_json = config
        progress_db.commit()
    finally:
        progress_db.close()


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
        update_run_progress(
            event.run_id,
            {
                "stage": "starting_worker",
                "processed_frames": 0,
                "total_frames": event.max_frames if event.max_frames > 0 else None,
                "percent": 0.0,
                "processing_fps": 0.0,
                "eta_seconds": None,
                "cache_hit_frames": 0,
            },
        )
        match = db.get(Match, event.match_id)
        team_context = build_analysis_team_context(match, db) if match is not None else {}
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
            reuse_detections_object=event.reuse_detections_object,
            progress_callback=lambda progress: update_run_progress(event.run_id, progress),
        )
        finish_run(db, run, "processed", summary=summary)
        update_match_status(db, event.match_id, "processed")
    except Exception as exc:
        if run is not None and not run_is_already_complete(run):
            update_run_progress(
                event.run_id,
                {
                    "stage": "failed",
                    "processed_frames": 0,
                    "total_frames": event.max_frames if event.max_frames > 0 else None,
                    "percent": None,
                    "processing_fps": 0.0,
                    "eta_seconds": None,
                    "cache_hit_frames": 0,
                    "error": str(exc),
                },
            )
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
