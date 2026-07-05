from fastapi import APIRouter
from fastapi import Depends
from fastapi import HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.ai.tactical_identity_layer.zone_model import PREFERRED_SIDES
from app.ai.tactical_identity_layer.zone_model import validate_zone
from app.ai.tactical_identity_layer.zone_model import validate_zones
from app.db.dependencies import get_db
from app.models.player import Player

router = APIRouter()


class PlayerUpdateRequest(BaseModel):
    name: str | None = None
    jersey_number: int | None = None
    primary_zone: str | None = None
    secondary_zones: list[str] | None = None
    position_label: str | None = None
    preferred_side: str | None = None
    notes: str | None = None


def serialize_player(player: Player) -> dict:
    return {
        "id": player.id,
        "team_id": player.team_id,
        "name": player.name,
        "jersey_number": player.jersey_number,
        "age": player.age,
        "position": player.position,
        "primary_zone": player.primary_zone,
        "secondary_zones": player.secondary_zones or [],
        "position_label": player.position_label,
        "preferred_side": player.preferred_side,
        "notes": player.notes,
    }


def apply_player_update(player: Player, payload: PlayerUpdateRequest) -> None:
    if payload.name is not None:
        player.name = payload.name
    if payload.jersey_number is not None:
        player.jersey_number = payload.jersey_number
    if payload.primary_zone is not None:
        player.primary_zone = validate_zone(payload.primary_zone)
        player.position = player.primary_zone or player.position
    if payload.secondary_zones is not None:
        player.secondary_zones = validate_zones(payload.secondary_zones)
    if payload.position_label is not None:
        player.position_label = payload.position_label
    if payload.preferred_side is not None:
        preferred_side = payload.preferred_side.strip().lower()
        if preferred_side not in PREFERRED_SIDES:
            raise HTTPException(
                status_code=400,
                detail=f"preferred_side must be one of: {sorted(PREFERRED_SIDES)}",
            )
        player.preferred_side = preferred_side
    if payload.notes is not None:
        player.notes = payload.notes


@router.post("/")
def create_player():
    return {"message": "player created"}


@router.get("/")
def get_players():
    return {"message": "players"}


@router.patch("/{player_id}")
def update_player(
    player_id: int,
    payload: PlayerUpdateRequest,
    db: Session = Depends(get_db),
):
    player = db.get(Player, player_id)
    if player is None:
        raise HTTPException(status_code=404, detail="Player not found")

    apply_player_update(player, payload)
    db.commit()
    db.refresh(player)
    return serialize_player(player)
