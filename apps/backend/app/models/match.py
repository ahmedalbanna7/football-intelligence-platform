# app/models/match.py

from datetime import datetime

from sqlalchemy import DateTime
from sqlalchemy import String

from sqlalchemy.orm import Mapped
from sqlalchemy.orm import mapped_column
from sqlalchemy.orm import relationship

from app.models.base import Base


class Match(Base):

    __tablename__ = "matches"

    id: Mapped[int] = mapped_column(
        primary_key=True
    )

    title: Mapped[str] = mapped_column(
        String(255)
    )

    status: Mapped[str] = mapped_column(
        String(50),
        default="uploaded"
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=datetime.utcnow
    )

    videos = relationship(
        "MatchVideo",
        back_populates="match"
    )

    player_stats = relationship(
        "PlayerMatchStat",
        back_populates="match"
    )