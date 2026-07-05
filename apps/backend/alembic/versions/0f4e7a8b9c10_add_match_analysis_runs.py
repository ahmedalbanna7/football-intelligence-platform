"""add_match_analysis_runs

Revision ID: 0f4e7a8b9c10
Revises: f1a2b3c4d5e6
Create Date: 2026-07-03 04:55:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "0f4e7a8b9c10"
down_revision: Union[str, Sequence[str], None] = "f1a2b3c4d5e6"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "match_analysis_runs",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("match_id", sa.Integer(), nullable=False),
        sa.Column("video_id", sa.Integer(), nullable=False),
        sa.Column("mode", sa.String(length=80), nullable=False),
        sa.Column("status", sa.String(length=50), nullable=False),
        sa.Column("source", sa.String(length=120), nullable=False),
        sa.Column("max_frames", sa.Integer(), nullable=False),
        sa.Column("output_object", sa.String(length=500), nullable=True),
        sa.Column("summary_object", sa.String(length=500), nullable=True),
        sa.Column("thumbnail_object", sa.String(length=500), nullable=True),
        sa.Column("summary_json", sa.JSON(), nullable=True),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("started_at", sa.DateTime(), nullable=True),
        sa.Column("finished_at", sa.DateTime(), nullable=True),
        sa.ForeignKeyConstraint(["match_id"], ["matches.id"]),
        sa.ForeignKeyConstraint(["video_id"], ["match_videos.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_match_analysis_runs_match_id",
        "match_analysis_runs",
        ["match_id"],
    )
    op.create_index(
        "ix_match_analysis_runs_status",
        "match_analysis_runs",
        ["status"],
    )


def downgrade() -> None:
    op.drop_index("ix_match_analysis_runs_status", table_name="match_analysis_runs")
    op.drop_index("ix_match_analysis_runs_match_id", table_name="match_analysis_runs")
    op.drop_table("match_analysis_runs")
