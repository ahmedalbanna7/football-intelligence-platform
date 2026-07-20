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


class MatchAnalysisRun(Base):
    __tablename__ = "match_analysis_runs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    match_id: Mapped[int] = mapped_column(ForeignKey("matches.id"), nullable=False)
    video_id: Mapped[int] = mapped_column(ForeignKey("match_videos.id"), nullable=False)
    mode: Mapped[str] = mapped_column(String(80), default="PLAYER_TRACKING", nullable=False)
    status: Mapped[str] = mapped_column(String(50), default="processing", nullable=False)
    source: Mapped[str] = mapped_column(String(120), default="sports-main", nullable=False)
    max_frames: Mapped[int] = mapped_column(Integer, default=450, nullable=False)
    output_object: Mapped[str | None] = mapped_column(String(500), nullable=True)
    summary_object: Mapped[str | None] = mapped_column(String(500), nullable=True)
    thumbnail_object: Mapped[str | None] = mapped_column(String(500), nullable=True)
    summary_json: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)
    started_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)

    match = relationship("Match")
    video = relationship("MatchVideo")
    quality_assessment = relationship(
        "TrackingQualityAssessment",
        back_populates="run",
        cascade="all, delete-orphan",
        uselist=False,
    )
    track_review_items = relationship(
        "TrackReviewItem",
        back_populates="run",
        cascade="all, delete-orphan",
    )
    track_review_corrections = relationship(
        "TrackReviewCorrection",
        back_populates="run",
        cascade="all, delete-orphan",
    )
