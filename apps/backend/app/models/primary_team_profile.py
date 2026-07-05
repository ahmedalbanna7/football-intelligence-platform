from datetime import datetime

from sqlalchemy import DateTime
from sqlalchemy import Integer
from sqlalchemy import String
from sqlalchemy.orm import Mapped
from sqlalchemy.orm import mapped_column

from app.models.base import Base


class PrimaryTeamProfile(Base):
    __tablename__ = "primary_team_profiles"

    id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
    )

    team_name: Mapped[str] = mapped_column(
        String(255),
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
