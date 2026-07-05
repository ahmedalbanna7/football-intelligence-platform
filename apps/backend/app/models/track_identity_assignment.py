from datetime import datetime

from sqlalchemy import DateTime
from sqlalchemy import Float
from sqlalchemy import ForeignKey
from sqlalchemy import Integer
from sqlalchemy import JSON
from sqlalchemy import String
from sqlalchemy.orm import Mapped
from sqlalchemy.orm import mapped_column
from sqlalchemy.orm import relationship

from app.models.base import Base


class TrackIdentityAssignment(Base):
    __tablename__ = "track_identity_assignments"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    match_id: Mapped[int] = mapped_column(ForeignKey("matches.id"), nullable=False)
    frame_index: Mapped[int | None] = mapped_column(Integer, nullable=True)
    timestamp_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)
    track_id: Mapped[int] = mapped_column(Integer, nullable=False)
    team_id: Mapped[int | None] = mapped_column(ForeignKey("teams.id"), nullable=True)
    resolved_player_id: Mapped[int | None] = mapped_column(
        ForeignKey("players.id"),
        nullable=True,
    )
    confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    candidate_scores: Mapped[dict | list | None] = mapped_column(JSON, nullable=True)
    zone: Mapped[str | None] = mapped_column(String(30), nullable=True)
    pitch_x: Mapped[float | None] = mapped_column(Float, nullable=True)
    pitch_y: Mapped[float | None] = mapped_column(Float, nullable=True)
    source: Mapped[str] = mapped_column(
        String(80),
        default="tactical_identity_stub_v1",
        nullable=False,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=datetime.utcnow,
        nullable=False,
    )

    match = relationship("Match")
    team = relationship("Team")
    resolved_player = relationship("Player")
