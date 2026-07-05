# app/models/team.py

from sqlalchemy import String
from sqlalchemy import Text
from sqlalchemy.orm import Mapped
from sqlalchemy.orm import mapped_column
from sqlalchemy.orm import relationship

from app.models.base import Base


class Team(Base):

    __tablename__ = "teams"

    id: Mapped[int] = mapped_column(
        primary_key=True
    )

    name: Mapped[str] = mapped_column(
        String(255),
        unique=True,
        nullable=False
    )

    team_type: Mapped[str] = mapped_column(
        String(50),
        default="opponent",
        nullable=False,
    )

    primary_kit_image_object_name: Mapped[str | None] = mapped_column(
        String(500),
        nullable=True,
    )

    alternate_kit_image_object_name: Mapped[str | None] = mapped_column(
        String(500),
        nullable=True,
    )

    notes: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
    )

    players = relationship(
        "Player",
        back_populates="team"
    )
