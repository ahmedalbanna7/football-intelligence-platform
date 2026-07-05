"""add_track_player_assignments

Revision ID: 91c4ad8f3f8f
Revises: 84c0cbcd0e32
Create Date: 2026-06-26 01:25:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "91c4ad8f3f8f"
down_revision: Union[str, Sequence[str], None] = "84c0cbcd0e32"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "track_player_assignments",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("match_id", sa.Integer(), nullable=False),
        sa.Column("processing_job_id", sa.Integer(), nullable=False),
        sa.Column("track_id", sa.Integer(), nullable=False),
        sa.Column("team_context", sa.String(length=50), nullable=False),
        sa.Column("player_name", sa.String(length=255), nullable=False),
        sa.Column("shirt_number", sa.Integer(), nullable=True),
        sa.Column("position", sa.String(length=80), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(["match_id"], ["matches.id"]),
        sa.ForeignKeyConstraint(["processing_job_id"], ["video_processing_jobs.id"]),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "match_id",
            "processing_job_id",
            "track_id",
            name="uq_track_player_assignment_match_job_track",
        ),
    )
    op.create_index(
        "ix_track_player_assignments_match_id",
        "track_player_assignments",
        ["match_id"],
    )
    op.create_index(
        "ix_track_player_assignments_processing_job_id",
        "track_player_assignments",
        ["processing_job_id"],
    )


def downgrade() -> None:
    op.drop_index(
        "ix_track_player_assignments_processing_job_id",
        table_name="track_player_assignments",
    )
    op.drop_index(
        "ix_track_player_assignments_match_id",
        table_name="track_player_assignments",
    )
    op.drop_table("track_player_assignments")
