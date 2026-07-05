"""add_match_analysis_context

Revision ID: 5f81c4470c9a
Revises: 2c5c2d8b77af
Create Date: 2026-06-26 00:20:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "5f81c4470c9a"
down_revision: Union[str, Sequence[str], None] = "2c5c2d8b77af"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "matches",
        sa.Column(
            "match_type",
            sa.String(length=50),
            nullable=False,
            server_default="official_vs_opponent",
        ),
    )
    op.add_column(
        "matches",
        sa.Column(
            "analysis_scope",
            sa.String(length=80),
            nullable=False,
            server_default="both_teams_full",
        ),
    )
    op.add_column(
        "matches",
        sa.Column("primary_team_name", sa.String(length=255), nullable=True),
    )
    op.add_column(
        "matches",
        sa.Column("opponent_team_name", sa.String(length=255), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("matches", "opponent_team_name")
    op.drop_column("matches", "primary_team_name")
    op.drop_column("matches", "analysis_scope")
    op.drop_column("matches", "match_type")
