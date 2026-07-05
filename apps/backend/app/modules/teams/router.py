import io
from uuid import uuid4

from fastapi import APIRouter
from fastapi import Depends
from fastapi import File
from fastapi import Form
from fastapi import HTTPException
from fastapi import UploadFile
from pydantic import BaseModel, Field
from sqlalchemy import or_
from sqlalchemy.orm import Session

from app.ai.tactical_identity_layer.zone_model import PREFERRED_SIDES
from app.ai.tactical_identity_layer.zone_model import validate_zone
from app.ai.tactical_identity_layer.zone_model import validate_zones
from app.db.dependencies import get_db
from app.models.match import Match
from app.models.player import Player
from app.models.primary_team_profile import PrimaryTeamProfile
from app.models.team import Team
from app.modules.players.router import serialize_player
from app.services.minio_client import BUCKET_NAME
from app.services.minio_client import client

router = APIRouter()


class TeamCreateRequest(BaseModel):
    name: str
    team_type: str = "opponent"
    notes: str | None = None


class TeamPlayerCreateRequest(BaseModel):
    name: str
    jersey_number: int | None = None
    age: int = 0
    primary_zone: str | None = None
    secondary_zones: list[str] = Field(default_factory=list)
    position_label: str | None = None
    preferred_side: str | None = "unknown"
    notes: str | None = None


def serialize_team(team: Team) -> dict:
    return {
        "id": team.id,
        "name": team.name,
        "team_type": team.team_type,
        "primary_kit_image_object_name": team.primary_kit_image_object_name,
        "alternate_kit_image_object_name": team.alternate_kit_image_object_name,
        "notes": team.notes,
    }


def normalize_team_type(team_type: str | None) -> str:
    value = (team_type or "opponent").strip().lower()
    if value not in {"opponent", "primary", "academy", "other"}:
        raise HTTPException(
            status_code=400,
            detail="team_type must be one of: opponent, primary, academy, other",
        )
    return value


async def upload_optional_asset(
    file: UploadFile | None,
    object_prefix: str,
) -> str | None:
    if file is None:
        return None

    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail=f"{file.filename} is empty")

    safe_filename = file.filename or "team-kit"
    object_name = f"{object_prefix}/{uuid4().hex}_{safe_filename}"
    client.put_object(
        BUCKET_NAME,
        object_name,
        io.BytesIO(content),
        len(content),
        content_type=file.content_type,
    )
    return object_name


@router.post("/")
def create_team(
    payload: TeamCreateRequest,
    db: Session = Depends(get_db),
):
    name = payload.name.strip()
    if not name:
        raise HTTPException(status_code=400, detail="Team name is required")

    existing = db.query(Team).filter(Team.name == name).first()
    if existing is not None:
        existing.team_type = normalize_team_type(payload.team_type)
        if payload.notes is not None:
            existing.notes = payload.notes
        db.commit()
        db.refresh(existing)
        return serialize_team(existing)

    team = Team(
        name=name,
        team_type=normalize_team_type(payload.team_type),
        notes=payload.notes,
    )
    db.add(team)
    db.commit()
    db.refresh(team)
    return serialize_team(team)


@router.get("/")
def list_teams(
    team_type: str | None = None,
    db: Session = Depends(get_db),
):
    query = db.query(Team)
    if team_type:
        query = query.filter(Team.team_type == normalize_team_type(team_type))
    if team_type == "opponent":
        primary_profile = (
            db.query(PrimaryTeamProfile)
            .order_by(PrimaryTeamProfile.updated_at.desc())
            .first()
        )
        if primary_profile is not None:
            query = query.filter(Team.name != primary_profile.team_name)
    teams = query.order_by(Team.name).all()
    return {
        "items": [
            serialize_team(team)
            for team in teams
        ]
    }


@router.delete("/{team_id}")
def delete_team(
    team_id: int,
    db: Session = Depends(get_db),
):
    team = db.get(Team, team_id)
    if team is None:
        raise HTTPException(status_code=404, detail="Team not found")
    if team.team_type == "primary":
        raise HTTPException(status_code=400, detail="Primary team cannot be deleted from Teams")

    linked_match = (
        db.query(Match)
        .filter(
            or_(
                Match.opponent_team_id == team.id,
                Match.primary_team_id == team.id,
                Match.another_team_id == team.id,
            )
        )
        .first()
    )
    if linked_match is not None:
        raise HTTPException(
            status_code=409,
            detail="Team is linked to matches. Delete those matches first.",
        )

    deleted_players = db.query(Player).filter(Player.team_id == team.id).delete()
    db.delete(team)
    db.commit()
    return {"team_id": team_id, "deleted": True, "deleted_players": deleted_players}


@router.post("/{team_id}/profile")
async def update_team_profile(
    team_id: int,
    name: str | None = Form(None),
    team_type: str | None = Form(None),
    notes: str | None = Form(None),
    primary_kit_image: UploadFile | None = File(None),
    alternate_kit_image: UploadFile | None = File(None),
    db: Session = Depends(get_db),
):
    team = db.get(Team, team_id)
    if team is None:
        raise HTTPException(status_code=404, detail="Team not found")

    if name is not None and name.strip():
        existing = (
            db.query(Team)
            .filter(Team.name == name.strip())
            .filter(Team.id != team.id)
            .first()
        )
        if existing is not None:
            raise HTTPException(status_code=409, detail="Team name already exists")
        team.name = name.strip()

    if team_type is not None:
        team.team_type = normalize_team_type(team_type)
    if notes is not None:
        team.notes = notes

    object_prefix = f"team-assets/opponents/{team.id}"
    primary_kit_object_name = await upload_optional_asset(
        primary_kit_image,
        object_prefix,
    )
    alternate_kit_object_name = await upload_optional_asset(
        alternate_kit_image,
        object_prefix,
    )
    if primary_kit_object_name is not None:
        team.primary_kit_image_object_name = primary_kit_object_name
    if alternate_kit_object_name is not None:
        team.alternate_kit_image_object_name = alternate_kit_object_name

    db.commit()
    db.refresh(team)
    return serialize_team(team)


@router.post("/{team_id}/players")
def create_team_player(
    team_id: int,
    payload: TeamPlayerCreateRequest,
    db: Session = Depends(get_db),
):
    team = db.get(Team, team_id)
    if team is None:
        raise HTTPException(status_code=404, detail="Team not found")

    primary_zone = validate_zone(payload.primary_zone)
    secondary_zones = validate_zones(payload.secondary_zones)
    preferred_side = (payload.preferred_side or "unknown").strip().lower()
    if preferred_side not in PREFERRED_SIDES:
        raise HTTPException(
            status_code=400,
            detail=f"preferred_side must be one of: {sorted(PREFERRED_SIDES)}",
        )

    player = Player(
        team_id=team.id,
        name=payload.name,
        jersey_number=payload.jersey_number,
        age=payload.age,
        position=primary_zone or payload.position_label or "unknown",
        primary_zone=primary_zone,
        secondary_zones=secondary_zones,
        position_label=payload.position_label,
        preferred_side=preferred_side,
        notes=payload.notes,
    )
    db.add(player)
    db.commit()
    db.refresh(player)
    return serialize_player(player)


@router.get("/{team_id}/players")
def list_team_players(
    team_id: int,
    db: Session = Depends(get_db),
):
    team = db.get(Team, team_id)
    if team is None:
        raise HTTPException(status_code=404, detail="Team not found")

    players = (
        db.query(Player)
        .filter(Player.team_id == team.id)
        .order_by(Player.jersey_number, Player.name)
        .all()
    )
    return {
        "team": serialize_team(team),
        "players": [
            serialize_player(player)
            for player in players
        ],
    }


@router.delete("/{team_id}/players/{player_id}")
def delete_team_player(
    team_id: int,
    player_id: int,
    db: Session = Depends(get_db),
):
    team = db.get(Team, team_id)
    if team is None:
        raise HTTPException(status_code=404, detail="Team not found")

    player = db.get(Player, player_id)
    if player is None or player.team_id != team.id:
        raise HTTPException(status_code=404, detail="Player not found for this team")

    db.delete(player)
    db.commit()
    return {"team_id": team_id, "player_id": player_id, "deleted": True}


@router.get("/{team_id}/matches")
def list_team_matches(
    team_id: int,
    db: Session = Depends(get_db),
):
    team = db.get(Team, team_id)
    if team is None:
        raise HTTPException(status_code=404, detail="Team not found")

    matches = (
        db.query(Match)
        .filter(
            or_(
                Match.opponent_team_id == team.id,
                Match.primary_team_id == team.id,
            )
        )
        .order_by(Match.created_at.desc())
        .all()
    )
    return {
        "team": serialize_team(team),
        "matches": [
            {
                "id": match.id,
                "title": match.title,
                "status": match.status,
                "match_type": match.match_type,
                "analysis_scope": match.analysis_scope,
                "opponent_team_name": match.opponent_team_name,
                "created_at": match.created_at,
            }
            for match in matches
        ],
    }
