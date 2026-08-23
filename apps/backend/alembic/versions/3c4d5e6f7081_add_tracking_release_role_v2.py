"""add tracking release gate and participant roles v2

Revision ID: 3c4d5e6f7081
Revises: 2b3c4d5e6f70
Create Date: 2026-08-23 12:00:00.000000
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "3c4d5e6f7081"
down_revision: Union[str, Sequence[str], None] = "2b3c4d5e6f70"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "tracking_quality_assessments",
        sa.Column("release_gate_status", sa.String(length=40), server_default="not_ready", nullable=False),
    )
    op.add_column(
        "tracking_quality_assessments",
        sa.Column("release_gate_json", sa.JSON(), nullable=True),
    )
    op.add_column(
        "track_review_items",
        sa.Column("role_name", sa.String(length=40), server_default="player", nullable=False),
    )
    op.add_column(
        "track_review_items",
        sa.Column("role_confidence", sa.Float(), server_default="0", nullable=False),
    )
    op.add_column(
        "track_review_items",
        sa.Column("role_locked", sa.Boolean(), server_default=sa.false(), nullable=False),
    )
    op.add_column("track_review_items", sa.Column("role_evidence_json", sa.JSON(), nullable=True))
    op.add_column(
        "track_review_corrections",
        sa.Column("assigned_role_name", sa.String(length=40), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("track_review_corrections", "assigned_role_name")
    op.drop_column("track_review_items", "role_evidence_json")
    op.drop_column("track_review_items", "role_locked")
    op.drop_column("track_review_items", "role_confidence")
    op.drop_column("track_review_items", "role_name")
    op.drop_column("tracking_quality_assessments", "release_gate_json")
    op.drop_column("tracking_quality_assessments", "release_gate_status")
