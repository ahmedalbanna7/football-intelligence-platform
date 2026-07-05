from datetime import datetime

from sqlalchemy import Boolean
from sqlalchemy import DateTime
from sqlalchemy import ForeignKey
from sqlalchemy import Integer
from sqlalchemy import JSON
from sqlalchemy import String
from sqlalchemy import UniqueConstraint
from sqlalchemy.orm import Mapped
from sqlalchemy.orm import mapped_column
from sqlalchemy.orm import relationship

from app.models.base import Base


class MatchLineup(Base):
    __tablename__ = "match_lineups"
    __table_args__ = (
        UniqueConstraint(
            "match_id",
            "team_id",
            "player_id",
            name="uq_match_lineup_match_team_player",
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    match_id: Mapped[int] = mapped_column(ForeignKey("matches.id"), nullable=False)
    team_id: Mapped[int | None] = mapped_column(ForeignKey("teams.id"), nullable=True)
    player_id: Mapped[int] = mapped_column(ForeignKey("players.id"), nullable=False)
    jersey_number: Mapped[int | None] = mapped_column(Integer, nullable=True)
    starting_zone: Mapped[str | None] = mapped_column(String(30), nullable=True)
    expected_zones: Mapped[list[str] | None] = mapped_column(JSON, nullable=True)
    is_starter: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    start_minute: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=datetime.utcnow,
        nullable=False,
    )

    match = relationship("Match", back_populates="lineup_entries")
    team = relationship("Team")
    player = relationship("Player")
