"""add_player_roster_entries

Revision ID: a2a5c2f02e8a
Revises: 91c4ad8f3f8f
Create Date: 2026-06-26 01:45:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "a2a5c2f02e8a"
down_revision: Union[str, Sequence[str], None] = "91c4ad8f3f8f"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "player_roster_entries",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("match_id", sa.Integer(), nullable=True),
        sa.Column("team_context", sa.String(length=50), nullable=False),
        sa.Column("player_name", sa.String(length=255), nullable=False),
        sa.Column("shirt_number", sa.Integer(), nullable=False),
        sa.Column("position", sa.String(length=80), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(["match_id"], ["matches.id"]),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "match_id",
            "team_context",
            "shirt_number",
            name="uq_player_roster_match_team_number",
        ),
    )
    op.create_index(
        "ix_player_roster_entries_match_id",
        "player_roster_entries",
        ["match_id"],
    )
    op.create_index(
        "ix_player_roster_entries_team_context",
        "player_roster_entries",
        ["team_context"],
    )


def downgrade() -> None:
    op.drop_index(
        "ix_player_roster_entries_team_context",
        table_name="player_roster_entries",
    )
    op.drop_index(
        "ix_player_roster_entries_match_id",
        table_name="player_roster_entries",
    )
    op.drop_table("player_roster_entries")
