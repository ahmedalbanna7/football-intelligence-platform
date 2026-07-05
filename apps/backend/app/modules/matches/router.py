import io
import json
from pathlib import Path
from uuid import uuid4

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, Request, UploadFile
from pydantic import BaseModel
from sqlalchemy import desc
from sqlalchemy.orm import Session
from starlette.responses import Response, StreamingResponse

from app.core.config import settings
from app.db.dependencies import get_db
from app.models.match import Match
from app.models.match_analysis_run import MatchAnalysisRun
from app.models.match_lineup import MatchLineup
from app.models.match_substitution import MatchSubstitution
from app.models.match_video import MatchVideo
from app.models.player import Player
from app.models.player_match_stats import PlayerMatchStat
from app.models.player_roster_entry import PlayerRosterEntry
from app.models.primary_team_profile import PrimaryTeamProfile
from app.models.team import Team
from app.models.track_identity_assignment import TrackIdentityAssignment
from app.models.track_player_assignment import TrackPlayerAssignment
from app.models.video_processing_job import VideoProcessingJob
from app.queues.events import MatchAnalysisRequestedEvent
from app.queues.events import VideoUploadedEvent
from app.queues.publisher import publish_match_analysis_requested
from app.queues.publisher import publish_video_uploaded
from app.services.minio_client import (
    BUCKET_NAME,
    client,
)
from app.ai.tactical_identity_layer.zone_model import validate_zone
from app.ai.tactical_identity_layer.zone_model import validate_zones

router = APIRouter()

MATCH_CATEGORIES = {
    "competitive",
    "friendly",
    "internal_scrimmage",
    "academy_match",
}

MATCH_TYPES = {
    "my_team_vs_opponent",
    "opponent_vs_opponent",
    "internal_scrimmage",
    "academy_match",
    # Legacy values kept for existing data/API clients.
    "official_vs_opponent",
    "friendly_vs_opponent",
}

ANALYSIS_SCOPES = {
    "both_teams_full",
    "my_team_full",
    "opponent_full",
    "opponent_team_only",
    "my_team",
    "another_team",
    "both",
    "none",
}

TEAM_DIRECTIONS = {
    "left_to_right",
    "right_to_left",
    "unknown",
}

PRIMARY_KIT_SOURCES = {
    "primary",
    "alternate",
    "auto",
    "unknown",
}

ASSIGNABLE_TEAM_CONTEXTS = {
    "primary_team",
    "opponent_team",
    "club_internal",
}


class TrackAssignmentRequest(BaseModel):
    track_id: int
    player_name: str
    team_context: str | None = None
    shirt_number: int | None = None
    position: str | None = None


class RosterPlayerRequest(BaseModel):
    player_name: str
    shirt_number: int
    position: str | None = None


class LineupPlayerRequest(BaseModel):
    player_id: int
    jersey_number: int | None = None
    starting_zone: str | None = None
    expected_zones: list[str] = []
    is_starter: bool = True
    start_minute: int = 0


class MatchLineupRequest(BaseModel):
    team_id: int | None = None
    players: list[LineupPlayerRequest]


class SubstitutionRequest(BaseModel):
    team_id: int | None = None
    minute: int
    second: int | None = None
    player_out_id: int | None = None
    player_in_id: int
    player_in_zone: str | None = None
    expected_zones: list[str] = []
    notes: str | None = None


class MatchSubstitutionsRequest(BaseModel):
    substitutions: list[SubstitutionRequest]


def get_active_primary_team_profile(db: Session) -> PrimaryTeamProfile | None:
    return (
        db.query(PrimaryTeamProfile)
        .order_by(desc(PrimaryTeamProfile.updated_at))
        .first()
    )


async def upload_optional_asset(
    file: UploadFile | None,
    object_prefix: str,
) -> str | None:
    if file is None:
        return None

    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail=f"{file.filename} is empty")

    safe_filename = file.filename or "asset"
    object_name = f"{object_prefix}/{uuid4().hex}_{safe_filename}"
    client.put_object(
        BUCKET_NAME,
        object_name,
        io.BytesIO(content),
        len(content),
        content_type=file.content_type,
    )
    return object_name


def default_player_analysis_flags(analysis_scope: str) -> tuple[bool, bool]:
    if analysis_scope in {"my_team_full", "my_team"}:
        return True, False
    if analysis_scope in {"opponent_full", "another_team"}:
        return False, True
    if analysis_scope in {"opponent_team_only", "none"}:
        return False, False
    return True, True


def validate_match_context(
    match_category: str,
    match_type: str,
    analysis_scope: str,
    opponent_team_name: str | None,
    primary_team_direction: str,
    opponent_team_direction: str,
    primary_team_kit_source: str,
) -> None:
    if match_category not in MATCH_CATEGORIES:
        raise HTTPException(
            status_code=400,
            detail=f"match_category must be one of: {sorted(MATCH_CATEGORIES)}",
        )

    if match_type not in MATCH_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"match_type must be one of: {sorted(MATCH_TYPES)}",
        )

    if analysis_scope not in ANALYSIS_SCOPES:
        raise HTTPException(
            status_code=400,
            detail=f"analysis_scope must be one of: {sorted(ANALYSIS_SCOPES)}",
        )

    if primary_team_direction not in TEAM_DIRECTIONS:
        raise HTTPException(
            status_code=400,
            detail=f"primary_team_direction must be one of: {sorted(TEAM_DIRECTIONS)}",
        )

    if opponent_team_direction not in TEAM_DIRECTIONS:
        raise HTTPException(
            status_code=400,
            detail=f"opponent_team_direction must be one of: {sorted(TEAM_DIRECTIONS)}",
        )

    if primary_team_kit_source not in PRIMARY_KIT_SOURCES:
        raise HTTPException(
            status_code=400,
            detail=f"primary_team_kit_source must be one of: {sorted(PRIMARY_KIT_SOURCES)}",
        )

def get_or_create_opponent_team(
    db: Session,
    opponent_team_id: int | None,
    opponent_team_name: str | None,
) -> Team | None:
    if opponent_team_id is not None:
        team = db.get(Team, opponent_team_id)
        if team is None:
            raise HTTPException(status_code=404, detail="Opponent team not found")
        if team.team_type == "primary":
            raise HTTPException(
                status_code=400,
                detail="Primary team cannot be selected as opponent",
            )
        return team

    name = (opponent_team_name or "").strip()
    if not name:
        return None

    primary_profile = get_active_primary_team_profile(db)
    if primary_profile is not None and name == primary_profile.team_name:
        raise HTTPException(
            status_code=400,
            detail="Opponent name cannot match the primary team name",
        )

    team = db.query(Team).filter(Team.name == name).first()
    if team is not None:
        if team.team_type == "primary":
            raise HTTPException(
                status_code=400,
                detail="Primary team cannot be used as opponent",
            )
        if team.team_type != "opponent":
            team.team_type = "opponent"
        return team

    team = Team(name=name, team_type="opponent")
    db.add(team)
    db.flush()
    return team


def get_or_create_named_team(
    db: Session,
    team_id: int | None,
    team_name: str | None,
    *,
    allow_primary: bool = False,
) -> Team | None:
    if team_id is not None:
        team = db.get(Team, team_id)
        if team is None:
            raise HTTPException(status_code=404, detail="Team not found")
        if not allow_primary and team.team_type == "primary":
            raise HTTPException(
                status_code=400,
                detail="Primary team cannot be selected for this team slot",
            )
        return team

    name = (team_name or "").strip()
    if not name:
        return None

    primary_profile = get_active_primary_team_profile(db)
    if (
        not allow_primary
        and primary_profile is not None
        and name.lower() == primary_profile.team_name.lower()
    ):
        raise HTTPException(
            status_code=400,
            detail="Another team name cannot match the primary team name",
        )

    team = db.query(Team).filter(Team.name == name).first()
    if team is not None:
        if not allow_primary and team.team_type == "primary":
            raise HTTPException(
                status_code=400,
                detail="Primary team cannot be used for this team slot",
            )
        if team.team_type != "primary":
            team.team_type = "opponent"
        return team

    team = Team(name=name, team_type="opponent")
    db.add(team)
    db.flush()
    return team


async def apply_opponent_upload_details(
    db: Session,
    team: Team | None,
    opponent_kit_image: UploadFile | None,
    opponent_alternate_kit_image: UploadFile | None,
    opponent_players_json: str | None,
) -> None:
    if team is None:
        return

    object_prefix = f"team-assets/opponents/{team.id}"
    primary_kit_object_name = await upload_optional_asset(
        opponent_kit_image,
        object_prefix,
    )
    alternate_kit_object_name = await upload_optional_asset(
        opponent_alternate_kit_image,
        object_prefix,
    )
    if primary_kit_object_name is not None:
        team.primary_kit_image_object_name = primary_kit_object_name
    if alternate_kit_object_name is not None:
        team.alternate_kit_image_object_name = alternate_kit_object_name

    if not opponent_players_json:
        return

    try:
        payload = json.loads(opponent_players_json)
    except json.JSONDecodeError as exc:
        raise HTTPException(
            status_code=400,
            detail="opponent_players_json must be a JSON array",
        ) from exc

    if not isinstance(payload, list):
        raise HTTPException(
            status_code=400,
            detail="opponent_players_json must be a JSON array",
        )

    for item in payload:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name") or item.get("player_name") or "").strip()
        if not name:
            continue
        jersey_number = item.get("jersey_number")
        if jersey_number in ("", None):
            jersey_number = item.get("shirt_number")
        jersey_number = int(jersey_number) if jersey_number not in ("", None) else None

        existing_query = db.query(Player).filter(Player.team_id == team.id)
        if jersey_number is not None:
            existing_query = existing_query.filter(Player.jersey_number == jersey_number)
        else:
            existing_query = existing_query.filter(Player.name == name)
        existing = existing_query.first()
        if existing is not None:
            existing.name = name
            existing.position_label = item.get("position_label") or existing.position_label
            continue

        player = Player(
            team_id=team.id,
            name=name,
            jersey_number=jersey_number,
            age=int(item.get("age") or 0),
            position=item.get("primary_zone") or item.get("position_label") or "unknown",
            primary_zone=validate_zone(item.get("primary_zone")),
            secondary_zones=validate_zones(item.get("secondary_zones") or []),
            position_label=item.get("position_label"),
            preferred_side=item.get("preferred_side") or "unknown",
            notes=item.get("notes"),
        )
        db.add(player)


def build_match_context(match: Match) -> dict:
    primary_kit_image = None
    if match.primary_team_profile is not None:
        if match.primary_team_kit_source == "alternate":
            primary_kit_image = (
                match.primary_team_profile.alternate_kit_image_object_name
                or match.primary_team_profile.primary_kit_image_object_name
            )
        else:
            primary_kit_image = match.primary_team_profile.primary_kit_image_object_name

    return {
        "match_category": match.match_category,
        "match_type": match.match_type,
        "matchup_type": match.matchup_type,
        "analysis_scope": match.analysis_scope,
        "primary_team_name": match.primary_team_name,
        "primary_team_id": match.primary_team_id,
        "opponent_team_id": match.opponent_team_id,
        "opponent_team_name": match.opponent_team_name,
        "another_team_id": match.another_team_id,
        "another_team_name": match.another_team_name,
        "formation": match.formation,
        "primary_formation": match.formation,
        "another_formation": match.another_formation,
        "primary_team_direction": match.primary_team_direction,
        "opponent_team_direction": match.opponent_team_direction,
        "primary_team_kit_source": match.primary_team_kit_source,
        "another_team_kit_source": match.another_team_kit_source,
        "player_analysis": {
            "analyze_primary_players": match.analyze_primary_players,
            "analyze_opponent_players": match.analyze_opponent_players,
        },
        "kit_references": {
            "primary_team_selected_kit_image": primary_kit_image,
            "primary_team_primary_kit_image": (
                match.primary_team_profile.primary_kit_image_object_name
                if match.primary_team_profile is not None else None
            ),
            "primary_team_alternate_kit_image": (
                match.primary_team_profile.alternate_kit_image_object_name
                if match.primary_team_profile is not None else None
            ),
            "opponent_team_kit_image": (
                (match.another_team or match.opponent_team).primary_kit_image_object_name
                if (match.another_team or match.opponent_team) is not None else None
            ),
            "opponent_team_alternate_kit_image": (
                (match.another_team or match.opponent_team).alternate_kit_image_object_name
                if (match.another_team or match.opponent_team) is not None else None
            ),
        },
        "analysis_targets": get_analysis_targets(match),
        "tactical_identity": build_tactical_identity_context(match),
    }


def get_analysis_targets(match: Match) -> dict:
    scope = match.analysis_scope
    is_internal = match.match_category in {"internal_scrimmage", "academy_match"} or match.match_type in {"internal_scrimmage", "academy_match"}

    if is_internal:
        return {
            "teams": [
                team
                for team in [match.primary_team_name, match.another_team_name or "club_internal"]
                if team
            ],
            "include_primary_players": scope in {"my_team", "both", "my_team_full", "both_teams_full"},
            "include_opponent_players": scope in {"another_team", "both", "opponent_full", "both_teams_full"},
            "include_opponent_team_summary": scope in {"another_team", "both", "opponent_full", "both_teams_full"},
        }

    return {
        "teams": [
            team
            for team in [
                match.primary_team_name if match.match_type == "my_team_vs_opponent" else match.opponent_team_name,
                match.another_team_name or match.opponent_team_name,
            ]
            if team
        ],
        "include_primary_players": scope in {"my_team", "both", "my_team_full", "both_teams_full"},
        "include_opponent_players": scope in {"another_team", "both", "opponent_full", "both_teams_full"},
        "include_opponent_team_summary": scope in {"another_team", "both", "opponent_full", "opponent_team_only", "both_teams_full"},
    }


def serialize_player_identity(player: Player | None) -> dict | None:
    if player is None:
        return None
    return {
        "player_id": player.id,
        "name": player.name,
        "jersey_number": player.jersey_number,
        "primary_zone": player.primary_zone,
        "secondary_zones": player.secondary_zones or [],
        "position_label": player.position_label,
        "preferred_side": player.preferred_side,
    }


def serialize_lineup_entry(entry: MatchLineup) -> dict:
    player = serialize_player_identity(entry.player)
    return {
        "id": entry.id,
        "match_id": entry.match_id,
        "team_id": entry.team_id,
        "player_id": entry.player_id,
        "player": player,
        "player_name": player.get("name") if player else None,
        "jersey_number": entry.jersey_number,
        "starting_zone": entry.starting_zone,
        "expected_zones": entry.expected_zones or [],
        "is_starter": entry.is_starter,
        "start_minute": entry.start_minute,
    }


def serialize_substitution(entry: MatchSubstitution) -> dict:
    player_in = serialize_player_identity(entry.player_in)
    player_out = serialize_player_identity(entry.player_out)
    return {
        "id": entry.id,
        "match_id": entry.match_id,
        "team_id": entry.team_id,
        "minute": entry.minute,
        "second": entry.second,
        "player_out_id": entry.player_out_id,
        "player_out": player_out,
        "player_in_id": entry.player_in_id,
        "player_in": player_in,
        "player_in_jersey_number": (
            player_in.get("jersey_number")
            if player_in is not None
            else None
        ),
        "player_in_zone": entry.player_in_zone,
        "expected_zones": entry.expected_zones or [],
        "notes": entry.notes,
    }


def serialize_track_identity_assignment(
    assignment: TrackIdentityAssignment,
) -> dict:
    return {
        "id": assignment.id,
        "match_id": assignment.match_id,
        "frame_index": assignment.frame_index,
        "timestamp_ms": assignment.timestamp_ms,
        "track_id": assignment.track_id,
        "team_id": assignment.team_id,
        "resolved_player_id": assignment.resolved_player_id,
        "resolved_player": serialize_player_identity(assignment.resolved_player),
        "confidence": assignment.confidence,
        "candidate_scores": assignment.candidate_scores,
        "zone": assignment.zone,
        "pitch_x": assignment.pitch_x,
        "pitch_y": assignment.pitch_y,
        "source": assignment.source,
        "created_at": assignment.created_at,
    }


def build_tactical_identity_context(match: Match) -> dict:
    lineup = [
        serialize_lineup_entry(entry)
        for entry in sorted(
            match.lineup_entries,
            key=lambda item: (item.start_minute, item.jersey_number or 999),
        )
    ]
    substitutions = [
        serialize_substitution(entry)
        for entry in sorted(
            match.substitutions,
            key=lambda item: (item.minute, item.second or 0),
        )
    ]
    return {
        "primary_team_id": match.primary_team_id,
        "formation": match.formation,
        "lineup": lineup,
        "substitutions": substitutions,
    }


def summarize_sequence(value: object) -> dict:
    if isinstance(value, list):
        return {"count": len(value)}
    return {"count": 0}


def summarize_processing_result(result: dict | None) -> dict | None:
    if not isinstance(result, dict):
        return None

    data = result.get("data")
    if not isinstance(data, dict):
        return {
            "status": result.get("status"),
            "meta": result.get("meta"),
        }

    frames = data.get("frames") if isinstance(data.get("frames"), dict) else {}
    frames_data = frames.get("data") if isinstance(frames.get("data"), dict) else {}
    detections = (
        data.get("detections")
        if isinstance(data.get("detections"), dict)
        else {}
    )
    detections_data = (
        detections.get("data")
        if isinstance(detections.get("data"), dict)
        else {}
    )
    tracking = data.get("tracks")
    if not isinstance(tracking, dict):
        tracking = data.get("tracking")
    if not isinstance(tracking, dict):
        tracking = {}
    tracking_data = (
        tracking.get("data")
        if isinstance(tracking.get("data"), dict)
        else {}
    )
    analytics = (
        data.get("analytics")
        if isinstance(data.get("analytics"), dict)
        else {}
    )
    tactical_identity = (
        data.get("tactical_identity")
        if isinstance(data.get("tactical_identity"), dict)
        else {}
    )
    tactical_identity_data = (
        tactical_identity.get("data")
        if isinstance(tactical_identity.get("data"), dict)
        else {}
    )
    events = data.get("events") if isinstance(data.get("events"), dict) else {}
    events_data = events.get("data") if isinstance(events.get("data"), dict) else {}
    artifacts = (
        data.get("artifacts")
        if isinstance(data.get("artifacts"), dict)
        else {}
    )
    artifacts_data = (
        artifacts.get("data")
        if isinstance(artifacts.get("data"), dict)
        else {}
    )
    crops = data.get("crops") if isinstance(data.get("crops"), dict) else {}
    crops_data = crops.get("data") if isinstance(crops.get("data"), dict) else {}

    return {
        "status": result.get("status"),
        "frames": {
            "status": frames.get("status"),
            "frames_processed": frames_data.get("frames_processed"),
            "frames_sampled": frames_data.get("frames_sampled"),
            "fps": frames_data.get("fps"),
            "duration_seconds": frames_data.get("duration_seconds"),
        },
        "detections": {
            "status": detections.get("status"),
            "engine": (detections.get("meta") or {}).get("engine"),
            "model": (detections.get("meta") or {}).get("model"),
            "mode": (detections.get("meta") or {}).get("mode"),
            "device": (detections.get("meta") or {}).get("device"),
            "fallback_reason": (detections.get("meta") or {}).get(
                "fallback_reason"
            ),
            "frames_processed": detections_data.get("frames_processed"),
            "frames_requested": detections_data.get("frames_requested"),
            "frames_skipped": detections_data.get("frames_skipped"),
            "detections_count": detections_data.get("detections_count"),
            "class_counts": detections_data.get("class_counts"),
            "raw_class_counts": detections_data.get("raw_class_counts"),
            "confidence": detections_data.get("confidence"),
            "elapsed_ms": (detections.get("meta") or {}).get("elapsed_ms"),
        },
        "tracking": {
            "status": tracking.get("status"),
            "engine": (tracking.get("meta") or {}).get("engine"),
            "mode": (tracking.get("meta") or {}).get("mode"),
            "detections_received": tracking_data.get("detections_received"),
            "tracks_count": tracking_data.get("tracks_count"),
        },
        "events": {
            "status": events.get("status"),
            "events_count": events_data.get("events_count"),
            "event_types": (events.get("meta") or {}).get("event_types"),
            "tracks_received": events_data.get("tracks_received"),
        },
        "analytics": {
            "status": analytics.get("status"),
            "engine": (analytics.get("meta") or {}).get("engine"),
        },
        "tactical_identity": {
            "status": tactical_identity.get("status"),
            "engine": (tactical_identity.get("meta") or {}).get("engine"),
            "assignments_count": tactical_identity_data.get("assignments_count"),
            "resolved_count": tactical_identity_data.get("resolved_count"),
        },
        "artifacts": {
            "status": artifacts.get("status"),
            "storage": (artifacts.get("meta") or {}).get("storage"),
            "paths": artifacts_data.get("artifacts"),
            "detections_count": artifacts_data.get("detections_count"),
            "tracks_count": artifacts_data.get("tracks_count"),
            "track_observations_count": artifacts_data.get(
                "track_observations_count"
            ),
        },
        "crops": {
            "status": crops.get("status"),
            "crops_count": crops_data.get("crops_count"),
            "jersey_crops_count": crops_data.get("jersey_crops_count"),
            "crops_prefix": crops_data.get("crops_prefix"),
        },
    }


def serialize_processing_job(
    job: VideoProcessingJob | None,
    include_result: bool = False,
) -> dict | None:
    if job is None:
        return None

    payload = {
        "id": job.id,
        "video_id": job.video_id,
        "status": job.status,
        "started_at": job.started_at,
        "finished_at": job.finished_at,
        "error_message": job.error_message,
        "result_summary": summarize_processing_result(job.result_json),
    }

    if include_result:
        payload["result"] = job.result_json

    return payload


def serialize_track_assignment(
    assignment: TrackPlayerAssignment | None,
) -> dict | None:
    if assignment is None:
        return None

    return {
        "id": assignment.id,
        "track_id": assignment.track_id,
        "team_context": assignment.team_context,
        "player_name": assignment.player_name,
        "shirt_number": assignment.shirt_number,
        "position": assignment.position,
        "created_at": assignment.created_at,
        "updated_at": assignment.updated_at,
    }


def get_latest_processing_job(
    db: Session,
    match_id: int,
) -> VideoProcessingJob | None:
    active_or_processed = (
        db.query(VideoProcessingJob)
        .filter(VideoProcessingJob.match_id == match_id)
        .filter(VideoProcessingJob.status != "failed")
        .order_by(desc(VideoProcessingJob.started_at))
        .first()
    )
    if active_or_processed is not None:
        return active_or_processed

    return (
        db.query(VideoProcessingJob)
        .filter(VideoProcessingJob.match_id == match_id)
        .order_by(desc(VideoProcessingJob.started_at))
        .first()
    )


def ensure_team_exists(db: Session, team_id: int | None) -> None:
    if team_id is None:
        return
    if db.get(Team, team_id) is None:
        raise HTTPException(status_code=404, detail=f"Team {team_id} not found")


def ensure_player_exists(db: Session, player_id: int) -> Player:
    player = db.get(Player, player_id)
    if player is None:
        raise HTTPException(status_code=404, detail=f"Player {player_id} not found")
    return player


def replace_match_lineup(
    db: Session,
    match: Match,
    payload: MatchLineupRequest,
) -> list[MatchLineup]:
    ensure_team_exists(db, payload.team_id)
    (
        db.query(MatchLineup)
        .filter(MatchLineup.match_id == match.id)
        .filter(MatchLineup.team_id == payload.team_id)
        .delete(synchronize_session=False)
    )

    entries = []
    for item in payload.players:
        player = ensure_player_exists(db, item.player_id)
        entry = MatchLineup(
            match_id=match.id,
            team_id=payload.team_id,
            player_id=player.id,
            jersey_number=item.jersey_number or player.jersey_number,
            starting_zone=validate_zone(item.starting_zone or player.primary_zone),
            expected_zones=validate_zones(
                item.expected_zones
                or player.secondary_zones
                or [player.primary_zone]
            ),
            is_starter=item.is_starter,
            start_minute=item.start_minute,
        )
        db.add(entry)
        entries.append(entry)

    db.commit()
    for entry in entries:
        db.refresh(entry)
    return entries


def create_match_substitutions(
    db: Session,
    match: Match,
    payload: MatchSubstitutionsRequest,
) -> list[MatchSubstitution]:
    entries = []
    for item in payload.substitutions:
        ensure_team_exists(db, item.team_id)
        player_in = ensure_player_exists(db, item.player_in_id)
        if item.player_out_id is not None:
            ensure_player_exists(db, item.player_out_id)

        entry = MatchSubstitution(
            match_id=match.id,
            team_id=item.team_id,
            minute=item.minute,
            second=item.second,
            player_out_id=item.player_out_id,
            player_in_id=player_in.id,
            player_in_zone=validate_zone(
                item.player_in_zone or player_in.primary_zone
            ),
            expected_zones=validate_zones(
                item.expected_zones
                or player_in.secondary_zones
                or [player_in.primary_zone]
            ),
            notes=item.notes,
        )
        db.add(entry)
        entries.append(entry)

    db.commit()
    for entry in entries:
        db.refresh(entry)
    return entries


def parse_tactical_context_payload(raw_payload: str | None) -> dict:
    if not raw_payload:
        return {}
    try:
        payload = json.loads(raw_payload)
    except json.JSONDecodeError as exc:
        raise HTTPException(
            status_code=400,
            detail="tactical_context must be valid JSON",
        ) from exc
    if not isinstance(payload, dict):
        raise HTTPException(
            status_code=400,
            detail="tactical_context must be a JSON object",
        )
    return payload


def apply_upload_tactical_context(
    db: Session,
    match: Match,
    payload: dict,
) -> None:
    team_id = payload.get("primary_team_id")
    if team_id is not None:
        ensure_team_exists(db, int(team_id))
        match.primary_team_id = int(team_id)

    if payload.get("formation") is not None:
        match.formation = str(payload.get("formation"))

    lineup = payload.get("lineup")
    if lineup:
        replace_match_lineup(
            db,
            match,
            MatchLineupRequest(
                team_id=match.primary_team_id,
                players=lineup,
            ),
        )

    substitutions = payload.get("substitutions")
    if substitutions:
        create_match_substitutions(
            db,
            match,
            MatchSubstitutionsRequest(substitutions=substitutions),
        )


def get_team_assignment_tracks(job: VideoProcessingJob) -> list[dict]:
    result = job.result_json or {}
    jersey_tracks = (
        result
        .get("data", {})
        .get("jersey_number_recognition", {})
        .get("data", {})
        .get("tracks", [])
    )
    if jersey_tracks:
        return jersey_tracks

    return (
        result
        .get("data", {})
        .get("team_assignment", {})
        .get("data", {})
        .get("tracks", [])
    )


def serialize_roster_entry(entry: PlayerRosterEntry) -> dict:
    return {
        "id": entry.id,
        "match_id": entry.match_id,
        "team_context": entry.team_context,
        "player_name": entry.player_name,
        "shirt_number": entry.shirt_number,
        "position": entry.position,
        "created_at": entry.created_at,
        "updated_at": entry.updated_at,
    }


def get_track_assignment_map(
    db: Session,
    match_id: int,
    processing_job_id: int,
) -> dict[int, TrackPlayerAssignment]:
    assignments = (
        db.query(TrackPlayerAssignment)
        .filter(TrackPlayerAssignment.match_id == match_id)
        .filter(TrackPlayerAssignment.processing_job_id == processing_job_id)
        .all()
    )
    return {
        assignment.track_id: assignment
        for assignment in assignments
    }


def serialize_track_with_assignment(
    track: dict,
    assignment: TrackPlayerAssignment | None,
) -> dict:
    frames = track.get("frames", [])
    crop_samples = track.get("crop_samples") or []
    shirt_number_source = track.get("shirt_number_source")
    recognized_shirt_number = track.get("recognized_shirt_number")
    shirt_number_confidence = track.get("shirt_number_confidence")
    if shirt_number_source == "stub_object_key":
        recognized_shirt_number = None
        shirt_number_confidence = None
        shirt_number_source = "legacy_stub_hidden"
    return {
        "track_id": track.get("track_id"),
        "object_key": track.get("object_key"),
        "class_name": track.get("class_name"),
        "team_context": track.get("team_context"),
        "team_assignment_source": track.get("team_assignment_source"),
        "team_assignment_confidence": track.get("team_assignment_confidence"),
        "recognized_shirt_number": recognized_shirt_number,
        "shirt_number_confidence": shirt_number_confidence,
        "shirt_number_source": shirt_number_source,
        "frames_count": len(frames),
        "first_frame": frames[0].get("frame_index") if frames else None,
        "last_frame": frames[-1].get("frame_index") if frames else None,
        "crop_samples": crop_samples[:5],
        "dominant_colors": track.get("dominant_colors", []),
        "kit_match_score": track.get("kit_match_score"),
        "assignment": serialize_track_assignment(assignment),
    }


def get_pipeline_identity_assignments(job: VideoProcessingJob) -> list[dict]:
    result = job.result_json or {}
    assignments = (
        result
        .get("data", {})
        .get("tactical_identity", {})
        .get("data", {})
        .get("assignments", [])
    )
    for assignment in assignments:
        snapshot = assignment.get("track_snapshot")
        if not isinstance(snapshot, dict):
            continue
        if snapshot.get("shirt_number_confidence") == 0.61:
            snapshot["recognized_shirt_number"] = None
            snapshot["shirt_number_confidence"] = None
            snapshot["shirt_number_source"] = "legacy_stub_hidden"
    return assignments


def compact_report_payload(report: dict) -> dict:
    data = report.get("data")
    if not isinstance(data, dict):
        return report

    data.pop("sections", None)
    data["debug_sections_available"] = data.get("debug_sections_available", [])

    yolo_tracking = data.get("yolo_tracking")
    if isinstance(yolo_tracking, dict):
        if "sample_detections" in yolo_tracking:
            yolo_tracking["sample_detections_count"] = len(
                yolo_tracking.pop("sample_detections") or []
            )
        if "sample_track_observations" in yolo_tracking:
            yolo_tracking["sample_track_observations_count"] = len(
                yolo_tracking.pop("sample_track_observations") or []
            )

    identity = data.get("identity")
    if isinstance(identity, dict) and isinstance(identity.get("assignments"), list):
        identity["assignments_count"] = len(identity["assignments"])
        identity["assignments"] = identity["assignments"][:80]

    return report


def manual_assignments_as_identity(
    assignments: list[TrackPlayerAssignment],
) -> list[dict]:
    return [
        {
            "track_id": assignment.track_id,
            "team_context": assignment.team_context,
            "team_id": None,
            "resolved_player_id": None,
            "resolved_player": {
                "player_id": None,
                "name": assignment.player_name,
                "jersey_number": assignment.shirt_number,
                "zone": assignment.position,
            },
            "confidence": 1.0,
            "zone": assignment.position,
            "candidates": [
                {
                    "player_id": 0,
                    "name": assignment.player_name,
                    "jersey_number": assignment.shirt_number,
                    "zone": assignment.position,
                    "score": 1.0,
                    "reasons": {
                        "manual_assignment": 1.0,
                    },
                }
            ],
            "track_snapshot": {
                "manual_assignment_id": assignment.id,
                "shirt_number": assignment.shirt_number,
                "source": "manual_track_assignment",
            },
            "source": "manual_track_assignment_v1",
        }
        for assignment in assignments
    ]


def make_simple_pdf(title: str, lines: list[str]) -> bytes:
    escaped_lines = [
        line.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
        for line in lines
    ]
    text_commands = ["BT", "/F1 12 Tf", "50 790 Td"]
    for index, line in enumerate(escaped_lines[:42]):
        if index:
            text_commands.append("0 -18 Td")
        text_commands.append(f"({line[:95]}) Tj")
    text_commands.append("ET")
    stream = "\n".join(text_commands).encode("latin-1", errors="replace")

    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 595 842] /Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
        b"<< /Length " + str(len(stream)).encode("ascii") + b" >>\nstream\n" + stream + b"\nendstream",
    ]
    pdf = bytearray(b"%PDF-1.4\n")
    offsets = [0]
    for number, obj in enumerate(objects, start=1):
        offsets.append(len(pdf))
        pdf.extend(f"{number} 0 obj\n".encode("ascii"))
        pdf.extend(obj)
        pdf.extend(b"\nendobj\n")
    xref_offset = len(pdf)
    pdf.extend(f"xref\n0 {len(objects) + 1}\n".encode("ascii"))
    pdf.extend(b"0000000000 65535 f \n")
    for offset in offsets[1:]:
        pdf.extend(f"{offset:010d} 00000 n \n".encode("ascii"))
    pdf.extend(
        (
            f"trailer << /Size {len(objects) + 1} /Root 1 0 R >>\n"
            f"startxref\n{xref_offset}\n%%EOF\n"
        ).encode("ascii")
    )
    return bytes(pdf)


def find_roster_entry_for_track(
    db: Session,
    match_id: int,
    team_context: str,
    shirt_number: int,
) -> dict | None:
    if team_context in {"primary_team", "club_internal"}:
        entry = (
            db.query(PlayerRosterEntry)
            .filter(PlayerRosterEntry.match_id.is_(None))
            .filter(PlayerRosterEntry.team_context == "primary_team")
            .filter(PlayerRosterEntry.shirt_number == shirt_number)
            .first()
        )
        if entry is None:
            return None
        return {
            "player_name": entry.player_name,
            "position": entry.position,
        }

    match = db.get(Match, match_id)
    if match is not None and match.opponent_team_id is not None:
        player = (
            db.query(Player)
            .filter(Player.team_id == match.opponent_team_id)
            .filter(Player.jersey_number == shirt_number)
            .first()
        )
        if player is not None:
            return {
                "player_name": player.name,
                "position": player.primary_zone or player.position_label or player.position,
            }

    entry = (
        db.query(PlayerRosterEntry)
        .filter(PlayerRosterEntry.match_id == match_id)
        .filter(PlayerRosterEntry.team_context == "opponent_team")
        .filter(PlayerRosterEntry.shirt_number == shirt_number)
        .first()
    )
    if entry is None:
        return None
    return {
        "player_name": entry.player_name,
        "position": entry.position,
    }


def upsert_auto_track_assignment(
    db: Session,
    match_id: int,
    job_id: int,
    track: dict,
) -> TrackPlayerAssignment | None:
    track_id = track.get("track_id")
    team_context = track.get("team_context")
    shirt_number = track.get("recognized_shirt_number")

    if track_id is None or shirt_number is None:
        return None
    if team_context not in ASSIGNABLE_TEAM_CONTEXTS:
        return None

    roster_entry = find_roster_entry_for_track(
        db,
        match_id=match_id,
        team_context=team_context,
        shirt_number=shirt_number,
    )

    assignment = (
        db.query(TrackPlayerAssignment)
        .filter(TrackPlayerAssignment.match_id == match_id)
        .filter(TrackPlayerAssignment.processing_job_id == job_id)
        .filter(TrackPlayerAssignment.track_id == track_id)
        .first()
    )

    if roster_entry is None:
        return None

    player_name = roster_entry["player_name"]
    position = roster_entry["position"] if roster_entry is not None else None

    if assignment is None:
        assignment = TrackPlayerAssignment(
            match_id=match_id,
            processing_job_id=job_id,
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


def serialize_match(
    db: Session,
    match: Match,
    include_result: bool = False,
) -> dict:
    latest_job = get_latest_processing_job(db, match.id)
    job_payload = serialize_processing_job(latest_job)
    latest_match_analysis_run = (
        db.query(MatchAnalysisRun)
        .filter(MatchAnalysisRun.match_id == match.id)
        .order_by(desc(MatchAnalysisRun.created_at))
        .first()
    )

    if job_payload is not None and not include_result:
        job_payload.pop("result", None)

    return {
        "id": match.id,
        "title": match.title,
        "status": match.status,
        "match_context": build_match_context(match),
        "created_at": match.created_at,
        "videos": [
            {
                "id": video.id,
                "object_name": video.object_name,
            }
            for video in match.videos
        ],
        "latest_processing_job": job_payload,
        "latest_match_analysis_run": (
            {
                "id": latest_match_analysis_run.id,
                "status": latest_match_analysis_run.status,
                "mode": latest_match_analysis_run.mode,
                "max_frames": latest_match_analysis_run.max_frames,
                "output_object": latest_match_analysis_run.output_object,
                "summary_object": latest_match_analysis_run.summary_object,
                "summary": latest_match_analysis_run.summary_json,
                "error_message": latest_match_analysis_run.error_message,
                "created_at": latest_match_analysis_run.created_at,
                "started_at": latest_match_analysis_run.started_at,
                "finished_at": latest_match_analysis_run.finished_at,
            }
            if latest_match_analysis_run is not None
            else None
        ),
    }


@router.post("/upload")
async def upload_match(
    file: UploadFile = File(...),
    match_category: str = Form("competitive"),
    match_type: str = Form("my_team_vs_opponent"),
    analysis_scope: str = Form("both"),
    opponent_team_name: str | None = Form(None),
    opponent_team_id: int | None = Form(None),
    opponent_kit_image: UploadFile | None = File(None),
    opponent_alternate_kit_image: UploadFile | None = File(None),
    opponent_players_json: str | None = Form(None),
    another_team_name: str | None = Form(None),
    another_team_id: int | None = Form(None),
    another_kit_image: UploadFile | None = File(None),
    another_alternate_kit_image: UploadFile | None = File(None),
    another_players_json: str | None = Form(None),
    primary_team_direction: str = Form("unknown"),
    opponent_team_direction: str = Form("unknown"),
    primary_team_kit_source: str = Form("auto"),
    another_team_kit_source: str = Form("auto"),
    primary_team_id: int | None = Form(None),
    formation: str | None = Form(None),
    another_formation: str | None = Form(None),
    tactical_context: str | None = Form(None),
    analyze_primary_players: bool | None = Form(None),
    analyze_opponent_players: bool | None = Form(None),
    db: Session = Depends(get_db),
):
    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")

    primary_team_profile = get_active_primary_team_profile(db)
    if match_category in {"internal_scrimmage", "academy_match"}:
        match_type = match_category
    if match_type in {"official_vs_opponent", "friendly_vs_opponent"}:
        match_category = "friendly" if match_type == "friendly_vs_opponent" else "competitive"
        match_type = "my_team_vs_opponent"

    ensure_team_exists(db, primary_team_id)

    opponent_team = get_or_create_named_team(
        db,
        team_id=opponent_team_id,
        team_name=opponent_team_name,
    )
    another_team = get_or_create_named_team(
        db,
        team_id=another_team_id,
        team_name=another_team_name,
    )

    if match_type == "my_team_vs_opponent" and another_team is None:
        another_team = opponent_team

    resolved_opponent_team_name = (
        opponent_team.name
        if opponent_team is not None
        else opponent_team_name
    )
    resolved_another_team_name = (
        another_team.name
        if another_team is not None
        else another_team_name
    )
    validate_match_context(
        match_category=match_category,
        match_type=match_type,
        analysis_scope=analysis_scope,
        opponent_team_name=resolved_another_team_name or resolved_opponent_team_name,
        primary_team_direction=primary_team_direction,
        opponent_team_direction=opponent_team_direction,
        primary_team_kit_source=primary_team_kit_source,
    )
    await apply_opponent_upload_details(
        db,
        team=opponent_team,
        opponent_kit_image=opponent_kit_image,
        opponent_alternate_kit_image=opponent_alternate_kit_image,
        opponent_players_json=opponent_players_json,
    )
    await apply_opponent_upload_details(
        db,
        team=another_team,
        opponent_kit_image=another_kit_image,
        opponent_alternate_kit_image=another_alternate_kit_image,
        opponent_players_json=another_players_json,
    )
    tactical_payload = parse_tactical_context_payload(tactical_context)

    default_analyze_primary, default_analyze_opponent = default_player_analysis_flags(
        analysis_scope
    )
    analyze_primary = (
        default_analyze_primary
        if analyze_primary_players is None
        else analyze_primary_players
    )
    analyze_opponent = (
        default_analyze_opponent
        if analyze_opponent_players is None
        else analyze_opponent_players
    )

    safe_filename = file.filename or "match-video"
    match = Match(
        title=safe_filename,
        status="uploaded",
        match_type=match_type,
        match_category=match_category,
        matchup_type=match_type,
        analysis_scope=analysis_scope,
        primary_team_name=(
            primary_team_profile.team_name
            if primary_team_profile is not None
            else None
        ),
        primary_team_id=primary_team_id,
        formation=formation,
        opponent_team_id=opponent_team.id if opponent_team is not None else None,
        opponent_team_name=resolved_opponent_team_name,
        another_team_id=another_team.id if another_team is not None else None,
        another_team_name=resolved_another_team_name,
        another_formation=another_formation,
        another_team_kit_source=another_team_kit_source,
        primary_team_direction=primary_team_direction,
        opponent_team_direction=opponent_team_direction,
        primary_team_kit_source=primary_team_kit_source,
        primary_team_profile_id=(
            primary_team_profile.id
            if primary_team_profile is not None
            else None
        ),
        analyze_primary_players=analyze_primary,
        analyze_opponent_players=analyze_opponent,
    )
    db.add(match)
    db.flush()

    apply_upload_tactical_context(db, match, tactical_payload)

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

    match_analysis_run: MatchAnalysisRun | None = None
    if settings.MATCH_ANALYSIS_AUTO_QUEUE_ON_UPLOAD:
        match_analysis_run = MatchAnalysisRun(
            match_id=match.id,
            video_id=video.id,
            mode=settings.MATCH_ANALYSIS_DEFAULT_MODE.upper(),
            status="queued",
            source="sports-main",
            max_frames=max(settings.MATCH_ANALYSIS_DEFAULT_MAX_FRAMES, 0),
        )
        db.add(match_analysis_run)
        db.flush()

    db.commit()
    db.refresh(match)
    db.refresh(video)
    if match_analysis_run is not None:
        db.refresh(match_analysis_run)

    match_context = build_match_context(match)
    if match_analysis_run is not None:
        try:
            await publish_match_analysis_requested(
                MatchAnalysisRequestedEvent(
                    run_id=match_analysis_run.id,
                    match_id=match.id,
                    video_id=video.id,
                    bucket=BUCKET_NAME,
                    object_name=object_name,
                    artifact_prefix=(
                        f"matches/{match.id}/match-analysis-plus/runs/"
                        f"{match_analysis_run.id}"
                    ),
                    mode=match_analysis_run.mode,
                    max_frames=match_analysis_run.max_frames,
                    match_context=match_context,
                )
            )
        except Exception as exc:
            match.status = "queue_failed"
            match_analysis_run.status = "failed"
            match_analysis_run.error_message = str(exc)
            db.commit()
            return {
                "match_id": match.id,
                "video_id": video.id,
                "match_analysis_run_id": match_analysis_run.id,
                "filename": file.filename,
                "object_name": object_name,
                "status": match.status,
                "queue_error": str(exc),
                "match_context": match_context,
            }
    else:
        event = VideoUploadedEvent(
            match_id=match.id,
            video_id=video.id,
            bucket=BUCKET_NAME,
            object_name=object_name,
            filename=file.filename or object_name,
            content_type=file.content_type,
            match_context=match_context,
        )
        try:
            await publish_video_uploaded(event)
        except Exception as exc:
            match.status = "queue_failed"
            db.commit()
            return {
                "match_id": match.id,
                "video_id": video.id,
                "filename": file.filename,
                "object_name": object_name,
                "status": match.status,
                "queue_error": str(exc),
                "match_context": match_context,
            }

    match.status = "queued"
    db.commit()
    db.refresh(match)

    return {
        "match_id": match.id,
        "video_id": video.id,
        "filename": file.filename,
        "object_name": object_name,
        "status": match.status,
        "match_analysis_run_id": (
            match_analysis_run.id
            if match_analysis_run is not None
            else None
        ),
        "match_context": build_match_context(match),
    }


@router.post("/{match_id}/reprocess")
async def reprocess_match(
    match_id: int,
    db: Session = Depends(get_db),
):
    match = db.get(Match, match_id)
    if match is None:
        raise HTTPException(status_code=404, detail="Match not found")

    video = (
        db.query(MatchVideo)
        .filter(MatchVideo.match_id == match_id)
        .order_by(desc(MatchVideo.id))
        .first()
    )
    if video is None:
        raise HTTPException(status_code=404, detail="No video found for this match")

    event = VideoUploadedEvent(
        match_id=match.id,
        video_id=video.id,
        bucket=BUCKET_NAME,
        object_name=video.object_name,
        filename=Path(video.object_name).name,
        content_type="video/mp4",
        match_context=build_match_context(match),
    )
    try:
        await publish_video_uploaded(event)
    except Exception as exc:
        match.status = "queue_failed"
        db.commit()
        raise HTTPException(
            status_code=503,
            detail="Reprocess queue publishing failed",
        ) from exc

    match.status = "queued"
    db.commit()
    return {
        "match_id": match.id,
        "video_id": video.id,
        "status": match.status,
        "queued": True,
    }


@router.get("/")
def list_matches(
    limit: int = 20,
    offset: int = 0,
    db: Session = Depends(get_db),
):
    limit = min(max(limit, 1), 100)
    offset = max(offset, 0)

    matches = (
        db.query(Match)
        .order_by(desc(Match.created_at))
        .offset(offset)
        .limit(limit)
        .all()
    )

    return {
        "items": [
            serialize_match(db, match)
            for match in matches
        ],
        "limit": limit,
        "offset": offset,
    }


def delete_match_objects(match_id: int) -> int:
    deleted = 0
    prefix = f"matches/{match_id}/"
    objects = client.list_objects(BUCKET_NAME, prefix=prefix, recursive=True)
    for item in objects:
        client.remove_object(BUCKET_NAME, item.object_name)
        deleted += 1
    return deleted


@router.delete("/{match_id}")
def delete_match(
    match_id: int,
    db: Session = Depends(get_db),
):
    match = db.get(Match, match_id)
    if match is None:
        raise HTTPException(status_code=404, detail="Match not found")

    deleted_objects = delete_match_objects(match.id)
    for model in [
        TrackPlayerAssignment,
        TrackIdentityAssignment,
        MatchLineup,
        MatchSubstitution,
        PlayerMatchStat,
        VideoProcessingJob,
        MatchVideo,
    ]:
        db.query(model).filter(model.match_id == match_id).delete(
            synchronize_session=False
        )
    db.query(Match).filter(Match.id == match_id).delete(synchronize_session=False)
    db.commit()
    return {
        "match_id": match_id,
        "deleted": True,
        "deleted_objects": deleted_objects,
    }


@router.get("/options/analysis-context")
def get_analysis_context_options():
    return {
        "match_categories": {
            "competitive": "Competitive match.",
            "friendly": "Friendly match.",
            "internal_scrimmage": "Internal club split-squad match.",
            "academy_match": "Academy or development match.",
        },
        "match_types": {
            "my_team_vs_opponent": "My team against another team.",
            "opponent_vs_opponent": "Two non-primary teams.",
            "internal_scrimmage": "Internal split-squad match.",
            "academy_match": "Academy/development match.",
        },
        "analysis_scopes": {
            "my_team": "Analyze my team only.",
            "another_team": "Analyze the other/second team only.",
            "both": "Analyze both sides.",
            "none": "Do not run player/team analysis.",
        },
        "recommended_defaults": {
            "opponent_match": {
                "match_category": "competitive",
                "match_type": "my_team_vs_opponent",
                "analysis_scope": "both",
                "primary_team_source": "/primary-team",
                "required_fields": ["another_team_name", "formation", "another_formation"],
            },
            "internal_or_academy": {
                "match_category": "internal_scrimmage",
                "match_type": "internal_scrimmage",
                "analysis_scope": "my_team",
                "primary_team_source": "/primary-team",
                "required_fields": ["formation", "another_formation"],
            },
        },
        "team_directions": {
            "left_to_right": "Team attacks from left to right in the first half.",
            "right_to_left": "Team attacks from right to left in the first half.",
            "unknown": "Direction is not known yet.",
        },
        "primary_kit_sources": {
            "auto": "Use primary kit references and infer the other kit by difference.",
            "primary": "Use primary team home/main kit reference image.",
            "alternate": "Use primary team alternate kit reference image.",
            "unknown": "Do not force a primary team kit reference.",
        },
        "player_analysis_flags": {
            "analyze_primary_players": "Include player-level report for my team.",
            "analyze_opponent_players": "Include player-level report for opponent team.",
        },
    }


@router.get("/artifacts/object")
def get_artifact_object(
    request: Request,
    object_name: str = Query(...),
):
    allowed_prefixes = (
        "matches/",
        "team-assets/",
    )
    if not object_name.startswith(allowed_prefixes):
        raise HTTPException(status_code=400, detail="Unsupported object path")

    content_type = "application/octet-stream"
    if object_name.lower().endswith((".jpg", ".jpeg")):
        content_type = "image/jpeg"
    elif object_name.lower().endswith(".png"):
        content_type = "image/png"
    elif object_name.lower().endswith(".json"):
        content_type = "application/json"
    elif object_name.lower().endswith(".jsonl"):
        content_type = "application/x-ndjson"
    elif object_name.lower().endswith(".mp4"):
        content_type = "video/mp4"
    elif object_name.lower().endswith(".avi"):
        content_type = "video/x-msvideo"
    elif object_name.lower().endswith(".webm"):
        content_type = "video/webm"

    try:
        stat = client.stat_object(BUCKET_NAME, object_name)
    except Exception as exc:
        raise HTTPException(status_code=404, detail="Object not found") from exc

    range_header = request.headers.get("range")
    if range_header and content_type.startswith("video/"):
        try:
            units, byte_range = range_header.split("=", 1)
            if units.strip().lower() != "bytes":
                raise ValueError("Unsupported range unit")
            start_text, end_text = byte_range.split("-", 1)
            start = int(start_text) if start_text else 0
            end = int(end_text) if end_text else stat.size - 1
            end = min(end, stat.size - 1)
            if start < 0 or end < start:
                raise ValueError("Invalid byte range")
        except ValueError as exc:
            raise HTTPException(status_code=416, detail="Invalid range") from exc

        length = end - start + 1
        response = client.get_object(
            BUCKET_NAME,
            object_name,
            offset=start,
            length=length,
        )
        return StreamingResponse(
            response.stream(32 * 1024),
            status_code=206,
            media_type=content_type,
            headers={
                "Accept-Ranges": "bytes",
                "Content-Range": f"bytes {start}-{end}/{stat.size}",
                "Content-Length": str(length),
            },
        )

    try:
        response = client.get_object(BUCKET_NAME, object_name)
    except Exception as exc:
        raise HTTPException(status_code=404, detail="Object not found") from exc

    return StreamingResponse(
        response.stream(32 * 1024),
        media_type=content_type,
        headers={
            "Accept-Ranges": "bytes" if content_type.startswith("video/") else "none",
            "Content-Length": str(stat.size),
        },
    )


@router.get("/{match_id}")
def get_match(
    match_id: int,
    db: Session = Depends(get_db),
):
    match = db.get(Match, match_id)
    if match is None:
        raise HTTPException(status_code=404, detail="Match not found")

    return serialize_match(db, match, include_result=True)


@router.get("/{match_id}/tracks")
def get_match_tracks(
    match_id: int,
    db: Session = Depends(get_db),
):
    match = db.get(Match, match_id)
    if match is None:
        raise HTTPException(status_code=404, detail="Match not found")

    job = get_latest_processing_job(db, match_id)
    if job is None:
        raise HTTPException(
            status_code=404,
            detail="No processing job found for this match",
        )

    tracks = get_team_assignment_tracks(job)
    assignments = get_track_assignment_map(db, match_id, job.id)

    return {
        "match_id": match.id,
        "match_status": match.status,
        "job_id": job.id,
        "job_status": job.status,
        "tracks": [
            serialize_track_with_assignment(
                track,
                assignments.get(track.get("track_id")),
            )
            for track in tracks
        ],
    }


@router.post("/{match_id}/opponent-players")
def upsert_opponent_player(
    match_id: int,
    payload: RosterPlayerRequest,
    db: Session = Depends(get_db),
):
    match = db.get(Match, match_id)
    if match is None:
        raise HTTPException(status_code=404, detail="Match not found")

    entry = (
        db.query(PlayerRosterEntry)
        .filter(PlayerRosterEntry.match_id == match_id)
        .filter(PlayerRosterEntry.team_context == "opponent_team")
        .filter(PlayerRosterEntry.shirt_number == payload.shirt_number)
        .first()
    )
    if entry is None:
        entry = PlayerRosterEntry(
            match_id=match_id,
            team_context="opponent_team",
            player_name=payload.player_name,
            shirt_number=payload.shirt_number,
            position=payload.position,
        )
        db.add(entry)
    else:
        entry.player_name = payload.player_name
        entry.position = payload.position

    db.commit()
    db.refresh(entry)
    return serialize_roster_entry(entry)


@router.get("/{match_id}/roster")
def get_match_roster(
    match_id: int,
    db: Session = Depends(get_db),
):
    match = db.get(Match, match_id)
    if match is None:
        raise HTTPException(status_code=404, detail="Match not found")

    primary_entries = (
        db.query(PlayerRosterEntry)
        .filter(PlayerRosterEntry.match_id.is_(None))
        .filter(PlayerRosterEntry.team_context == "primary_team")
        .order_by(PlayerRosterEntry.shirt_number)
        .all()
    )
    opponent_entries = (
        db.query(PlayerRosterEntry)
        .filter(PlayerRosterEntry.match_id == match_id)
        .filter(PlayerRosterEntry.team_context == "opponent_team")
        .order_by(PlayerRosterEntry.shirt_number)
        .all()
    )

    return {
        "match_id": match.id,
        "primary_team": [
            serialize_roster_entry(entry)
            for entry in primary_entries
        ],
        "opponent_team": [
            serialize_roster_entry(entry)
            for entry in opponent_entries
        ],
    }


@router.post("/{match_id}/lineup")
def replace_lineup(
    match_id: int,
    payload: MatchLineupRequest,
    db: Session = Depends(get_db),
):
    match = db.get(Match, match_id)
    if match is None:
        raise HTTPException(status_code=404, detail="Match not found")

    if payload.team_id is not None:
        match.primary_team_id = payload.team_id
    entries = replace_match_lineup(db, match, payload)
    db.refresh(match)
    return {
        "match_id": match.id,
        "team_id": payload.team_id,
        "lineup": [
            serialize_lineup_entry(entry)
            for entry in entries
        ],
    }


@router.get("/{match_id}/lineup")
def get_lineup(
    match_id: int,
    db: Session = Depends(get_db),
):
    match = db.get(Match, match_id)
    if match is None:
        raise HTTPException(status_code=404, detail="Match not found")

    return {
        "match_id": match.id,
        "formation": match.formation,
        "primary_team_id": match.primary_team_id,
        "lineup": [
            serialize_lineup_entry(entry)
            for entry in match.lineup_entries
        ],
    }


@router.post("/{match_id}/substitutions")
def add_substitutions(
    match_id: int,
    payload: MatchSubstitutionsRequest,
    db: Session = Depends(get_db),
):
    match = db.get(Match, match_id)
    if match is None:
        raise HTTPException(status_code=404, detail="Match not found")

    entries = create_match_substitutions(db, match, payload)
    return {
        "match_id": match.id,
        "substitutions": [
            serialize_substitution(entry)
            for entry in entries
        ],
    }


@router.get("/{match_id}/substitutions")
def get_substitutions(
    match_id: int,
    db: Session = Depends(get_db),
):
    match = db.get(Match, match_id)
    if match is None:
        raise HTTPException(status_code=404, detail="Match not found")

    return {
        "match_id": match.id,
        "substitutions": [
            serialize_substitution(entry)
            for entry in match.substitutions
        ],
    }


@router.post("/{match_id}/track-assignments")
def upsert_track_assignment(
    match_id: int,
    payload: TrackAssignmentRequest,
    db: Session = Depends(get_db),
):
    match = db.get(Match, match_id)
    if match is None:
        raise HTTPException(status_code=404, detail="Match not found")

    job = get_latest_processing_job(db, match_id)
    if job is None:
        raise HTTPException(
            status_code=404,
            detail="No processing job found for this match",
        )

    tracks = get_team_assignment_tracks(job)
    track = next(
        (
            item
            for item in tracks
            if item.get("track_id") == payload.track_id
        ),
        None,
    )
    if track is None:
        raise HTTPException(status_code=404, detail="Track not found")

    team_context = payload.team_context or track.get("team_context")
    if team_context not in ASSIGNABLE_TEAM_CONTEXTS:
        raise HTTPException(
            status_code=400,
            detail=f"team_context must be one of: {sorted(ASSIGNABLE_TEAM_CONTEXTS)}",
        )

    assignment = (
        db.query(TrackPlayerAssignment)
        .filter(TrackPlayerAssignment.match_id == match_id)
        .filter(TrackPlayerAssignment.processing_job_id == job.id)
        .filter(TrackPlayerAssignment.track_id == payload.track_id)
        .first()
    )

    if assignment is None:
        assignment = TrackPlayerAssignment(
            match_id=match_id,
            processing_job_id=job.id,
            track_id=payload.track_id,
            team_context=team_context,
            player_name=payload.player_name,
            shirt_number=payload.shirt_number,
            position=payload.position,
        )
        db.add(assignment)
    else:
        assignment.team_context = team_context
        assignment.player_name = payload.player_name
        assignment.shirt_number = payload.shirt_number
        assignment.position = payload.position

    db.commit()
    db.refresh(assignment)

    return {
        "match_id": match.id,
        "job_id": job.id,
        "track": serialize_track_with_assignment(track, assignment),
    }


@router.post("/{match_id}/auto-assign-tracks")
def auto_assign_tracks(
    match_id: int,
    db: Session = Depends(get_db),
):
    match = db.get(Match, match_id)
    if match is None:
        raise HTTPException(status_code=404, detail="Match not found")

    job = get_latest_processing_job(db, match_id)
    if job is None:
        raise HTTPException(
            status_code=404,
            detail="No processing job found for this match",
        )

    tracks = get_team_assignment_tracks(job)
    assignments = [
        assignment
        for assignment in [
            upsert_auto_track_assignment(db, match_id, job.id, track)
            for track in tracks
        ]
        if assignment is not None
    ]

    return {
        "match_id": match.id,
        "job_id": job.id,
        "assigned_count": len(assignments),
        "assignments": [
            serialize_track_assignment(assignment)
            for assignment in assignments
        ],
    }


@router.get("/{match_id}/identity-assignments")
def get_identity_assignments(
    match_id: int,
    db: Session = Depends(get_db),
):
    match = db.get(Match, match_id)
    if match is None:
        raise HTTPException(status_code=404, detail="Match not found")

    job = get_latest_processing_job(db, match_id)
    manual_query = (
        db.query(TrackPlayerAssignment)
        .filter(TrackPlayerAssignment.match_id == match_id)
    )
    if job is not None:
        manual_query = manual_query.filter(
            TrackPlayerAssignment.processing_job_id == job.id
        )
    manual_assignments = (
        manual_query
        .order_by(TrackPlayerAssignment.track_id)
        .all()
    )
    if manual_assignments:
        return {
            "match_id": match.id,
            "source": "manual_track_assignments",
            "job_id": job.id if job is not None else None,
            "assignments": manual_assignments_as_identity(manual_assignments),
        }

    db_assignments = (
        db.query(TrackIdentityAssignment)
        .filter(TrackIdentityAssignment.match_id == match_id)
        .order_by(TrackIdentityAssignment.track_id)
        .all()
    )
    if db_assignments:
        return {
            "match_id": match.id,
            "source": "database",
            "assignments": [
                serialize_track_identity_assignment(assignment)
                for assignment in db_assignments
            ],
        }

    pipeline_assignments = (
        get_pipeline_identity_assignments(job)
        if job is not None
        else []
    )
    return {
        "match_id": match.id,
        "source": "latest_processing_job",
        "job_id": job.id if job is not None else None,
        "assignments": pipeline_assignments,
    }


@router.get("/{match_id}/tracks/{track_id}/identity")
def get_track_identity(
    match_id: int,
    track_id: int,
    db: Session = Depends(get_db),
):
    match = db.get(Match, match_id)
    if match is None:
        raise HTTPException(status_code=404, detail="Match not found")

    db_assignment = (
        db.query(TrackIdentityAssignment)
        .filter(TrackIdentityAssignment.match_id == match_id)
        .filter(TrackIdentityAssignment.track_id == track_id)
        .order_by(desc(TrackIdentityAssignment.created_at))
        .first()
    )
    if db_assignment is not None:
        return {
            "match_id": match.id,
            "track_id": track_id,
            "source": "database",
            "identity": serialize_track_identity_assignment(db_assignment),
        }

    job = get_latest_processing_job(db, match_id)
    assignments = get_pipeline_identity_assignments(job) if job is not None else []
    identity = next(
        (
            item
            for item in assignments
            if item.get("track_id") == track_id
        ),
        None,
    )
    if identity is None:
        raise HTTPException(status_code=404, detail="Track identity not found")

    return {
        "match_id": match.id,
        "track_id": track_id,
        "source": "latest_processing_job",
        "job_id": job.id if job is not None else None,
        "identity": identity,
    }


@router.get("/{match_id}/processing")
def get_match_processing(
    match_id: int,
    include_result: bool = Query(
        default=False,
        description="Return the full raw pipeline result. Keep false for Swagger.",
    ),
    db: Session = Depends(get_db),
):
    match = db.get(Match, match_id)
    if match is None:
        raise HTTPException(status_code=404, detail="Match not found")

    job = get_latest_processing_job(db, match_id)
    if job is None:
        return {
            "match_id": match.id,
            "match_status": match.status,
            "match_context": build_match_context(match),
            "job": None,
        }

    return {
        "match_id": match.id,
        "match_status": match.status,
        "match_context": build_match_context(match),
        "job": serialize_processing_job(job, include_result=include_result),
    }


@router.get("/{match_id}/report")
def get_match_report(
    match_id: int,
    db: Session = Depends(get_db),
):
    match = db.get(Match, match_id)
    if match is None:
        raise HTTPException(status_code=404, detail="Match not found")

    job = get_latest_processing_job(db, match_id)
    if job is None:
        raise HTTPException(
            status_code=404,
            detail="No processing job found for this match",
        )

    if job.status != "processed":
        return {
            "match_id": match.id,
            "match_status": match.status,
            "job_status": job.status,
            "report": None,
        }

    result = job.result_json or {}
    report = result.get("data", {}).get("report")
    if report is None:
        raise HTTPException(
            status_code=404,
            detail="Processing job does not contain a report",
        )
    report = compact_report_payload(report)

    assignments = get_track_assignment_map(db, match_id, job.id)
    manual_identity = manual_assignments_as_identity(list(assignments.values()))
    if isinstance(report, dict):
        report_data = report.setdefault("data", {})
        identity_data = report_data.setdefault("identity", {})
        identity_data["manual_assignments"] = manual_identity
        identity_data["manual_resolved_count"] = len(manual_identity)

    return {
        "match_id": match.id,
        "match_status": match.status,
        "match_context": build_match_context(match),
        "job_id": job.id,
        "player_assignments": [
            serialize_track_assignment(assignment)
            for assignment in assignments.values()
        ],
        "report": report,
    }


@router.get("/{match_id}/report.pdf")
def get_match_report_pdf(
    match_id: int,
    db: Session = Depends(get_db),
):
    payload = get_match_report(match_id, db)
    report_data = payload.get("report", {}).get("data", {})
    summary = report_data.get("summary", {})
    counts = report_data.get("counts", {})
    lines = [
        "Sports Intelligence Match Report",
        f"Match ID: {match_id}",
        f"Primary Team: {summary.get('primary_team_name') or '-'}",
        f"Opponent: {summary.get('opponent_team_name') or '-'}",
        f"Match Type: {summary.get('match_type') or '-'}",
        f"Analysis Scope: {summary.get('analysis_scope') or '-'}",
        "",
        "Counts",
        f"Detections: {counts.get('detections', 0)}",
        f"Tracks: {counts.get('tracks', 0)}",
        f"Events: {counts.get('events', 0)}",
        f"Identity assignments: {counts.get('identity_assignments', 0)}",
        f"Identity resolved: {counts.get('identity_resolved', 0)}",
        f"Player crops: {counts.get('player_crops', 0)}",
        f"Jersey crops: {counts.get('jersey_crops', 0)}",
    ]
    manual_assignments = (
        report_data
        .get("identity", {})
        .get("manual_assignments", [])
    )
    if manual_assignments:
        lines.extend(["", "Manual Track Assignments"])
        for item in manual_assignments[:20]:
            player = item.get("resolved_player") or {}
            lines.append(
                f"Track {item.get('track_id')}: {player.get('name')} #{player.get('jersey_number') or '-'}"
            )

    pdf = make_simple_pdf(f"match-{match_id}-report", lines)
    return Response(
        pdf,
        media_type="application/pdf",
        headers={
            "Content-Disposition": f'attachment; filename="match-{match_id}-report.pdf"'
        },
    )
