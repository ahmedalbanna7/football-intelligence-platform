from datetime import datetime

from sqlalchemy import DateTime
from sqlalchemy import ForeignKey
from sqlalchemy import Integer
from sqlalchemy import String
from sqlalchemy import UniqueConstraint
from sqlalchemy.orm import Mapped
from sqlalchemy.orm import mapped_column
from sqlalchemy.orm import relationship

from app.models.base import Base


class TrackPlayerAssignment(Base):
    __tablename__ = "track_player_assignments"
    __table_args__ = (
        UniqueConstraint(
            "match_id",
            "processing_job_id",
            "track_id",
            name="uq_track_player_assignment_match_job_track",
        ),
    )

    id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
    )

    match_id: Mapped[int] = mapped_column(
        ForeignKey("matches.id"),
        nullable=False,
    )

    processing_job_id: Mapped[int] = mapped_column(
        ForeignKey("video_processing_jobs.id"),
        nullable=False,
    )

    track_id: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
    )

    team_context: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
    )

    player_name: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
    )

    shirt_number: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True,
    )

    position: Mapped[str | None] = mapped_column(
        String(80),
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
    processing_job = relationship("VideoProcessingJob")
