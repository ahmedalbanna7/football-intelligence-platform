from datetime import datetime
from typing import Any

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


class VideoProcessingJob(Base):
    __tablename__ = "video_processing_jobs"

    id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
    )

    match_id: Mapped[int] = mapped_column(
        ForeignKey("matches.id"),
        nullable=False,
    )

    video_id: Mapped[int] = mapped_column(
        ForeignKey("match_videos.id"),
        nullable=False,
    )

    status: Mapped[str] = mapped_column(
        String(50),
        default="processing",
        nullable=False,
    )

    started_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=datetime.utcnow,
        nullable=False,
    )

    finished_at: Mapped[datetime | None] = mapped_column(
        DateTime,
        nullable=True,
    )

    error_message: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
    )

    result_json: Mapped[dict[str, Any] | None] = mapped_column(
        JSON,
        nullable=True,
    )

    match = relationship(
        "Match",
        back_populates="processing_jobs",
    )

    video = relationship(
        "MatchVideo",
        back_populates="processing_jobs",
    )
