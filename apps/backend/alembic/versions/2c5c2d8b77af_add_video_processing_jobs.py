"""add_video_processing_jobs

Revision ID: 2c5c2d8b77af
Revises: 8b0fdcea7c2c
Create Date: 2026-06-25 23:55:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "2c5c2d8b77af"
down_revision: Union[str, Sequence[str], None] = "8b0fdcea7c2c"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "video_processing_jobs",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("match_id", sa.Integer(), nullable=False),
        sa.Column("video_id", sa.Integer(), nullable=False),
        sa.Column("status", sa.String(length=50), nullable=False),
        sa.Column("started_at", sa.DateTime(), nullable=False),
        sa.Column("finished_at", sa.DateTime(), nullable=True),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("result_json", sa.JSON(), nullable=True),
        sa.ForeignKeyConstraint(["match_id"], ["matches.id"]),
        sa.ForeignKeyConstraint(["video_id"], ["match_videos.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_video_processing_jobs_match_id",
        "video_processing_jobs",
        ["match_id"],
    )
    op.create_index(
        "ix_video_processing_jobs_video_id",
        "video_processing_jobs",
        ["video_id"],
    )


def downgrade() -> None:
    op.drop_index(
        "ix_video_processing_jobs_video_id",
        table_name="video_processing_jobs",
    )
    op.drop_index(
        "ix_video_processing_jobs_match_id",
        table_name="video_processing_jobs",
    )
    op.drop_table("video_processing_jobs")
