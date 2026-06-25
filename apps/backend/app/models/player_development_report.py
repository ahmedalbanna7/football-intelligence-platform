from datetime import datetime

from sqlalchemy import DateTime
from sqlalchemy import Float
from sqlalchemy import ForeignKey
from sqlalchemy import Integer
from sqlalchemy import Text

from sqlalchemy.orm import Mapped
from sqlalchemy.orm import mapped_column
from sqlalchemy.orm import relationship

from app.models.base import Base


class PlayerDevelopmentReport(Base):

    __tablename__ = "player_development_reports"

    id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True
    )

    player_id: Mapped[int] = mapped_column(
        ForeignKey("players.id"),
        nullable=False
    )

    passing_score: Mapped[float] = mapped_column(
        Float,
        default=0
    )

    shooting_score: Mapped[float] = mapped_column(
        Float,
        default=0
    )

    dribbling_score: Mapped[float] = mapped_column(
        Float,
        default=0
    )

    speed_score: Mapped[float] = mapped_column(
        Float,
        default=0
    )

    stamina_score: Mapped[float] = mapped_column(
        Float,
        default=0
    )

    strengths: Mapped[str] = mapped_column(
        Text,
        nullable=True
    )

    weaknesses: Mapped[str] = mapped_column(
        Text,
        nullable=True
    )

    recommended_drills: Mapped[str] = mapped_column(
        Text,
        nullable=True
    )

    ai_summary: Mapped[str] = mapped_column(
        Text,
        nullable=True
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=datetime.utcnow
    )

    player = relationship(
        "Player",
        back_populates="development_reports"
    )