# app/models/match.py

from datetime import datetime

from sqlalchemy import DateTime
from sqlalchemy import ForeignKey
from sqlalchemy import Integer
from sqlalchemy import String
from sqlalchemy import Boolean

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

    match_type: Mapped[str] = mapped_column(
        String(50),
        default="official_vs_opponent"
    )

    match_category: Mapped[str] = mapped_column(
        String(50),
        default="competitive"
    )

    matchup_type: Mapped[str] = mapped_column(
        String(80),
        default="my_team_vs_opponent"
    )

    analysis_scope: Mapped[str] = mapped_column(
        String(80),
        default="both_teams_full"
    )

    primary_team_name: Mapped[str | None] = mapped_column(
        String(255),
        nullable=True
    )

    primary_team_id: Mapped[int | None] = mapped_column(
        Integer,
        ForeignKey("teams.id"),
        nullable=True
    )

    formation: Mapped[str | None] = mapped_column(
        String(30),
        nullable=True
    )

    opponent_team_name: Mapped[str | None] = mapped_column(
        String(255),
        nullable=True
    )

    opponent_team_id: Mapped[int | None] = mapped_column(
        Integer,
        ForeignKey("teams.id"),
        nullable=True,
    )

    another_team_name: Mapped[str | None] = mapped_column(
        String(255),
        nullable=True,
    )

    another_team_id: Mapped[int | None] = mapped_column(
        Integer,
        ForeignKey("teams.id"),
        nullable=True,
    )

    another_formation: Mapped[str | None] = mapped_column(
        String(30),
        nullable=True,
    )

    another_team_kit_source: Mapped[str] = mapped_column(
        String(50),
        default="auto"
    )

    primary_team_profile_id: Mapped[int | None] = mapped_column(
        Integer,
        ForeignKey("primary_team_profiles.id"),
        nullable=True
    )

    primary_team_direction: Mapped[str] = mapped_column(
        String(50),
        default="unknown"
    )

    opponent_team_direction: Mapped[str] = mapped_column(
        String(50),
        default="unknown"
    )

    primary_team_kit_source: Mapped[str] = mapped_column(
        String(50),
        default="primary"
    )

    analyze_primary_players: Mapped[bool] = mapped_column(
        Boolean,
        default=True
    )

    analyze_opponent_players: Mapped[bool] = mapped_column(
        Boolean,
        default=True
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=datetime.utcnow
    )

    videos = relationship(
        "MatchVideo",
        back_populates="match",
        cascade="all, delete-orphan"
    )

    player_stats = relationship(
        "PlayerMatchStat",
        back_populates="match"
    )

    processing_jobs = relationship(
        "VideoProcessingJob",
        back_populates="match",
        cascade="all, delete-orphan"
    )

    primary_team_profile = relationship(
        "PrimaryTeamProfile"
    )

    primary_team = relationship(
        "Team",
        foreign_keys=[primary_team_id],
    )

    opponent_team = relationship(
        "Team",
        foreign_keys=[opponent_team_id],
    )

    another_team = relationship(
        "Team",
        foreign_keys=[another_team_id],
    )

    lineup_entries = relationship(
        "MatchLineup",
        back_populates="match",
        cascade="all, delete-orphan"
    )

    substitutions = relationship(
        "MatchSubstitution",
        back_populates="match",
        cascade="all, delete-orphan"
    )
