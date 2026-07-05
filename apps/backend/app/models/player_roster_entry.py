from datetime import datetime

from sqlalchemy import DateTime
from sqlalchemy import ForeignKey
from sqlalchemy import Integer
from sqlalchemy import JSON
from sqlalchemy import String
from sqlalchemy import Text
from sqlalchemy import UniqueConstraint
from sqlalchemy.orm import Mapped
from sqlalchemy.orm import mapped_column
from sqlalchemy.orm import relationship

from app.models.base import Base


class PlayerRosterEntry(Base):
    __tablename__ = "player_roster_entries"
    __table_args__ = (
        UniqueConstraint(
            "match_id",
            "team_context",
            "shirt_number",
            name="uq_player_roster_match_team_number",
        ),
    )

    id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
    )

    match_id: Mapped[int | None] = mapped_column(
        ForeignKey("matches.id"),
        nullable=True,
    )

    team_context: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
    )

    player_name: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
    )

    shirt_number: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
    )

    position: Mapped[str | None] = mapped_column(
        String(80),
        nullable=True,
    )

    primary_zone: Mapped[str | None] = mapped_column(
        String(30),
        nullable=True,
    )

    secondary_zones: Mapped[list[str] | None] = mapped_column(
        JSON,
        nullable=True,
    )

    position_label: Mapped[str | None] = mapped_column(
        String(120),
        nullable=True,
    )

    preferred_side: Mapped[str | None] = mapped_column(
        String(30),
        nullable=True,
    )

    notes: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=datetime.utcnow,
        nullable=False,
    )

    updated_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=datetime.utcnow,
        onupdate=datetime.utcnow,
        nullable=False,
    )

    match = relationship("Match")
