# app/models/player.py

from sqlalchemy import ForeignKey
from sqlalchemy import Integer
from sqlalchemy import JSON
from sqlalchemy import String
from sqlalchemy import Text

from sqlalchemy.orm import Mapped
from sqlalchemy.orm import mapped_column
from sqlalchemy.orm import relationship

from app.models.base import Base


class Player(Base):

    __tablename__ = "players"

    id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True
    )

    name: Mapped[str] = mapped_column(
        String(255)
    )

    jersey_number: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True
    )

    age: Mapped[int] = mapped_column(
        Integer,
        default=0
    )

    position: Mapped[str] = mapped_column(
        String(50),
        default="unknown"
    )

    primary_zone: Mapped[str | None] = mapped_column(
        String(30),
        nullable=True
    )

    secondary_zones: Mapped[list[str] | None] = mapped_column(
        JSON,
        nullable=True
    )

    position_label: Mapped[str | None] = mapped_column(
        String(120),
        nullable=True
    )

    preferred_side: Mapped[str | None] = mapped_column(
        String(30),
        nullable=True
    )

    notes: Mapped[str | None] = mapped_column(
        Text,
        nullable=True
    )

    team_id: Mapped[int] = mapped_column(
        ForeignKey("teams.id")
    )

    team = relationship(
        "Team",
        back_populates="players"
    )

    match_stats = relationship(
        "PlayerMatchStat",
        back_populates="player"
    )

    development_reports = relationship(
    "PlayerDevelopmentReport",
    back_populates="player",
    cascade="all, delete-orphan"
    )
