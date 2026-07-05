import io
from uuid import uuid4

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from pydantic import BaseModel, Field
from sqlalchemy import desc
from sqlalchemy.orm import Session

from app.ai.tactical_identity_layer.zone_model import PREFERRED_SIDES
from app.ai.tactical_identity_layer.zone_model import validate_zone
from app.ai.tactical_identity_layer.zone_model import validate_zones
from app.db.dependencies import get_db
from app.models.player_roster_entry import PlayerRosterEntry
from app.models.primary_team_profile import PrimaryTeamProfile
from app.models.team import Team
from app.services.minio_client import BUCKET_NAME, client

router = APIRouter()


class RosterPlayerRequest(BaseModel):
    player_name: str
    shirt_number: int
    position: str | None = None
    primary_zone: str | None = None
    secondary_zones: list[str] = Field(default_factory=list)
    position_label: str | None = None
    preferred_side: str | None = "unknown"
    notes: str | None = None


def serialize_profile(profile: PrimaryTeamProfile | None) -> dict | None:
    if profile is None:
        return None

    return {
        "id": profile.id,
        "team_name": profile.team_name,
        "primary_kit_image_object_name": profile.primary_kit_image_object_name,
        "alternate_kit_image_object_name": profile.alternate_kit_image_object_name,
        "created_at": profile.created_at,
        "updated_at": profile.updated_at,
    }


def get_active_profile(db: Session) -> PrimaryTeamProfile | None:
    return (
        db.query(PrimaryTeamProfile)
        .order_by(desc(PrimaryTeamProfile.updated_at))
        .first()
    )


def serialize_roster_entry(entry: PlayerRosterEntry) -> dict:
    return {
        "id": entry.id,
        "team_context": entry.team_context,
        "player_name": entry.player_name,
        "shirt_number": entry.shirt_number,
        "position": entry.position,
        "primary_zone": entry.primary_zone,
        "secondary_zones": entry.secondary_zones or [],
        "position_label": entry.position_label,
        "preferred_side": entry.preferred_side,
        "notes": entry.notes,
        "created_at": entry.created_at,
        "updated_at": entry.updated_at,
    }


def normalize_roster_payload(payload: RosterPlayerRequest) -> dict:
    player_name = payload.player_name.strip()
    if not player_name:
        raise HTTPException(status_code=400, detail="Player name is required")
    if payload.shirt_number <= 0:
        raise HTTPException(status_code=400, detail="shirt_number must be greater than zero")

    try:
        primary_zone = validate_zone(payload.primary_zone)
        secondary_zones = validate_zones(payload.secondary_zones)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    preferred_side = (payload.preferred_side or "unknown").strip().lower()
    if preferred_side not in PREFERRED_SIDES:
        raise HTTPException(
            status_code=400,
            detail=f"preferred_side must be one of: {sorted(PREFERRED_SIDES)}",
        )

    position_label = (payload.position_label or "").strip() or None
    legacy_position = (payload.position or "").strip() or None
    return {
        "player_name": player_name,
        "shirt_number": payload.shirt_number,
        "position": primary_zone or position_label or legacy_position,
        "primary_zone": primary_zone,
        "secondary_zones": secondary_zones,
        "position_label": position_label,
        "preferred_side": preferred_side,
        "notes": (payload.notes or "").strip() or None,
    }


async def upload_optional_asset(
    file: UploadFile | None,
    object_prefix: str,
) -> str | None:
    if file is None:
        return None

    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail=f"{file.filename} is empty")

    safe_filename = file.filename or "kit-image"
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
async def upsert_primary_team_profile(
    team_name: str = Form(...),
    primary_kit_image: UploadFile | None = File(None),
    alternate_kit_image: UploadFile | None = File(None),
    db: Session = Depends(get_db),
):
    profile = get_active_profile(db)
    if profile is None:
        profile = PrimaryTeamProfile(team_name=team_name)
        db.add(profile)
        db.flush()
    else:
        profile.team_name = team_name

    object_prefix = f"team-assets/primary-team/{profile.id}"

    primary_kit_object_name = await upload_optional_asset(
        primary_kit_image,
        object_prefix,
    )
    alternate_kit_object_name = await upload_optional_asset(
        alternate_kit_image,
        object_prefix,
    )

    if primary_kit_object_name is not None:
        profile.primary_kit_image_object_name = primary_kit_object_name
    if alternate_kit_object_name is not None:
        profile.alternate_kit_image_object_name = alternate_kit_object_name

    team = db.query(Team).filter(Team.name == team_name).first()
    if team is not None:
        team.team_type = "primary"
        if profile.primary_kit_image_object_name:
            team.primary_kit_image_object_name = profile.primary_kit_image_object_name
        if profile.alternate_kit_image_object_name:
            team.alternate_kit_image_object_name = profile.alternate_kit_image_object_name

    db.commit()
    db.refresh(profile)

    return serialize_profile(profile)


@router.get("/")
def get_primary_team_profile(db: Session = Depends(get_db)):
    profile = get_active_profile(db)
    if profile is None:
        raise HTTPException(status_code=404, detail="Primary team profile not configured")

    return serialize_profile(profile)


@router.post("/players")
def upsert_primary_team_player(
    payload: RosterPlayerRequest,
    db: Session = Depends(get_db),
):
    profile = get_active_profile(db)
    if profile is None:
        raise HTTPException(status_code=400, detail="Configure /primary-team first")

    entry = (
        db.query(PlayerRosterEntry)
        .filter(PlayerRosterEntry.match_id.is_(None))
        .filter(PlayerRosterEntry.team_context == "primary_team")
        .filter(PlayerRosterEntry.shirt_number == payload.shirt_number)
        .first()
    )

    normalized = normalize_roster_payload(payload)

    if entry is None:
        entry = PlayerRosterEntry(
            match_id=None,
            team_context="primary_team",
            **normalized,
        )
        db.add(entry)
    else:
        entry.player_name = normalized["player_name"]
        entry.position = normalized["position"]
        entry.primary_zone = normalized["primary_zone"]
        entry.secondary_zones = normalized["secondary_zones"]
        entry.position_label = normalized["position_label"]
        entry.preferred_side = normalized["preferred_side"]
        entry.notes = normalized["notes"]

    db.commit()
    db.refresh(entry)
    return serialize_roster_entry(entry)


@router.get("/players")
def list_primary_team_players(db: Session = Depends(get_db)):
    entries = (
        db.query(PlayerRosterEntry)
        .filter(PlayerRosterEntry.match_id.is_(None))
        .filter(PlayerRosterEntry.team_context == "primary_team")
        .order_by(PlayerRosterEntry.shirt_number)
        .all()
    )
    return {
        "items": [
            serialize_roster_entry(entry)
            for entry in entries
        ]
    }


@router.delete("/players/{entry_id}")
def delete_primary_team_player(
    entry_id: int,
    db: Session = Depends(get_db),
):
    entry = db.get(PlayerRosterEntry, entry_id)
    if (
        entry is None
        or entry.match_id is not None
        or entry.team_context != "primary_team"
    ):
        raise HTTPException(status_code=404, detail="Primary team player not found")

    db.delete(entry)
    db.commit()
    return {"entry_id": entry_id, "deleted": True}
