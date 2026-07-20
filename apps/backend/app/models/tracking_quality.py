from datetime import datetime
from typing import Any

from sqlalchemy import Boolean
from sqlalchemy import DateTime
from sqlalchemy import Float
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


class TrackingQualityAssessment(Base):
    __tablename__ = "tracking_quality_assessments"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    run_id: Mapped[int] = mapped_column(
        ForeignKey("match_analysis_runs.id", ondelete="CASCADE"),
        unique=True,
        nullable=False,
    )
    status: Mapped[str] = mapped_column(String(40), default="pending", nullable=False)
    tracker_engine: Mapped[str | None] = mapped_column(String(120), nullable=True)
    reid_enabled: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    reid_model: Mapped[str | None] = mapped_column(String(255), nullable=True)
    average_identity_confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    suspected_id_switches: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    fragmented_tracks: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    tracks_needing_review: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    benchmark_status: Mapped[str] = mapped_column(
        String(50),
        default="ground_truth_required",
        nullable=False,
    )
    id_switches: Mapped[int | None] = mapped_column(Integer, nullable=True)
    idf1: Mapped[float | None] = mapped_column(Float, nullable=True)
    hota: Mapped[float | None] = mapped_column(Float, nullable=True)
    fragmentation: Mapped[int | None] = mapped_column(Integer, nullable=True)
    predictions_object: Mapped[str | None] = mapped_column(String(500), nullable=True)
    ground_truth_object: Mapped[str | None] = mapped_column(String(500), nullable=True)
    metrics_json: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    thresholds_json: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=datetime.utcnow,
        onupdate=datetime.utcnow,
        nullable=False,
    )
    reviewed_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)

    run = relationship("MatchAnalysisRun", back_populates="quality_assessment")


class TrackReviewItem(Base):
    __tablename__ = "track_review_items"
    __table_args__ = (
        UniqueConstraint("run_id", "track_id", name="uq_track_review_run_track"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    run_id: Mapped[int] = mapped_column(
        ForeignKey("match_analysis_runs.id", ondelete="CASCADE"),
        nullable=False,
    )
    track_id: Mapped[int] = mapped_column(Integer, nullable=False)
    canonical_track_id: Mapped[int] = mapped_column(Integer, nullable=False)
    team_number: Mapped[int | None] = mapped_column(Integer, nullable=True)
    assigned_player_id: Mapped[int | None] = mapped_column(
        ForeignKey("players.id", ondelete="SET NULL"),
        nullable=True,
    )
    status: Mapped[str] = mapped_column(String(40), default="pending", nullable=False)
    identity_confidence: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    reid_confidence: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    motion_consistency: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    team_consistency: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    switch_risk: Mapped[str] = mapped_column(String(20), default="medium", nullable=False)
    fragment_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    raw_id_transitions: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    first_frame: Mapped[int | None] = mapped_column(Integer, nullable=True)
    last_frame: Mapped[int | None] = mapped_column(Integer, nullable=True)
    observation_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    raw_track_ids: Mapped[list[int] | None] = mapped_column(JSON, nullable=True)
    issue_codes: Mapped[list[str] | None] = mapped_column(JSON, nullable=True)
    crop_objects: Mapped[list[dict[str, Any]] | None] = mapped_column(JSON, nullable=True)
    observations_json: Mapped[list[dict[str, Any]] | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=datetime.utcnow,
        onupdate=datetime.utcnow,
        nullable=False,
    )

    run = relationship("MatchAnalysisRun", back_populates="track_review_items")
    assigned_player = relationship("Player")


class TrackReviewCorrection(Base):
    __tablename__ = "track_review_corrections"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    run_id: Mapped[int] = mapped_column(
        ForeignKey("match_analysis_runs.id", ondelete="CASCADE"),
        nullable=False,
    )
    action: Mapped[str] = mapped_column(String(40), nullable=False)
    source_track_id: Mapped[int | None] = mapped_column(Integer, nullable=True)
    target_track_id: Mapped[int | None] = mapped_column(Integer, nullable=True)
    split_frame: Mapped[int | None] = mapped_column(Integer, nullable=True)
    assigned_player_id: Mapped[int | None] = mapped_column(
        ForeignKey("players.id", ondelete="SET NULL"),
        nullable=True,
    )
    assigned_team_number: Mapped[int | None] = mapped_column(Integer, nullable=True)
    before_json: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    after_json: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    note: Mapped[str | None] = mapped_column(Text, nullable=True)
    undone: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)

    run = relationship("MatchAnalysisRun", back_populates="track_review_corrections")
    assigned_player = relationship("Player")
