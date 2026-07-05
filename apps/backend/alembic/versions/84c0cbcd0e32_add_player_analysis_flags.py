"""add_player_analysis_flags

Revision ID: 84c0cbcd0e32
Revises: 72dd6213073e
Create Date: 2026-06-26 01:05:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "84c0cbcd0e32"
down_revision: Union[str, Sequence[str], None] = "72dd6213073e"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "matches",
        sa.Column(
            "analyze_primary_players",
            sa.Boolean(),
            nullable=False,
            server_default=sa.true(),
        ),
    )
    op.add_column(
        "matches",
        sa.Column(
            "analyze_opponent_players",
            sa.Boolean(),
            nullable=False,
            server_default=sa.true(),
        ),
    )


def downgrade() -> None:
    op.drop_column("matches", "analyze_opponent_players")
    op.drop_column("matches", "analyze_primary_players")
