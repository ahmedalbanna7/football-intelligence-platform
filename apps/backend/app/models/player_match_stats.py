# app/models/player_match_stats.py

from sqlalchemy import Float
from sqlalchemy import ForeignKey
from sqlalchemy import Integer

from sqlalchemy.orm import Mapped
from sqlalchemy.orm import mapped_column
from sqlalchemy.orm import relationship

from app.models.base import Base


class PlayerMatchStat(Base):

    __tablename__ = "player_match_stats"

    id: Mapped[int] = mapped_column(
        primary_key=True
    )

    player_id: Mapped[int] = mapped_column(
        ForeignKey("players.id")
    )

    match_id: Mapped[int] = mapped_column(
        ForeignKey("matches.id")
    )

    distance: Mapped[float] = mapped_column(
        Float,
        default=0
    )

    max_speed: Mapped[float] = mapped_column(
        Float,
        default=0
    )

    passes_completed: Mapped[int] = mapped_column(
        Integer,
        default=0
    )

    passes_failed: Mapped[int] = mapped_column(
        Integer,
        default=0
    )

    shots: Mapped[int] = mapped_column(
        Integer,
        default=0
    )

    goals: Mapped[int] = mapped_column(
        Integer,
        default=0
    )

    player = relationship(
        "Player",
        back_populates="match_stats"
    )

    match = relationship(
        "Match",
        back_populates="player_stats"
    )