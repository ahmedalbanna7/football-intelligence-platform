# app/models/player.py

from sqlalchemy import ForeignKey
from sqlalchemy import Integer
from sqlalchemy import String

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

    age: Mapped[int] = mapped_column(
        Integer
    )

    position: Mapped[str] = mapped_column(
        String(50)
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