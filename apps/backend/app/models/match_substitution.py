from datetime import datetime

from sqlalchemy import DateTime
from sqlalchemy import ForeignKey
from sqlalchemy import Integer
from sqlalchemy import JSON
from sqlalchemy import String
from sqlalchemy import Text
from sqlalchemy.orm import Mapped
from sqlalchemy.orm import mapped_column
from sqlalchemy.orm import relationship

from app.models.base import Base


class MatchSubstitution(Base):
    __tablename__ = "match_substitutions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    match_id: Mapped[int] = mapped_column(ForeignKey("matches.id"), nullable=False)
    team_id: Mapped[int | None] = mapped_column(ForeignKey("teams.id"), nullable=True)
    minute: Mapped[int] = mapped_column(Integer, nullable=False)
    second: Mapped[int | None] = mapped_column(Integer, nullable=True)
    player_out_id: Mapped[int | None] = mapped_column(
        ForeignKey("players.id"),
        nullable=True,
    )
    player_in_id: Mapped[int] = mapped_column(ForeignKey("players.id"), nullable=False)
    player_in_zone: Mapped[str | None] = mapped_column(String(30), nullable=True)
    expected_zones: Mapped[list[str] | None] = mapped_column(JSON, nullable=True)
    notes: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=datetime.utcnow,
        nullable=False,
    )

    match = relationship("Match", back_populates="substitutions")
    team = relationship("Team")
    player_out = relationship("Player", foreign_keys=[player_out_id])
    player_in = relationship("Player", foreign_keys=[player_in_id])
