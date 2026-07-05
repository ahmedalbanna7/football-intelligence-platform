from typing import Any

from pydantic import BaseModel, Field


class IdentityCandidate(BaseModel):
    player_id: int
    team_id: int | None = None
    name: str | None = None
    jersey_number: int | None = None
    zone: str | None = None
    score: float
    reasons: dict[str, float | None] = Field(default_factory=dict)


class TrackIdentityResolution(BaseModel):
    track_id: int
    team_context: str | None = None
    team_id: int | None = None
    resolved_player_id: int | None = None
    resolved_player: dict | None = None
    confidence: float | None = None
    zone: str | None = None
    source: str = "tactical_identity_stub_v1"
    candidates: list[IdentityCandidate] = Field(default_factory=list)
    track_snapshot: dict[str, Any] = Field(default_factory=dict)
