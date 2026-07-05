"""add_roster_entry_tactical_fields

Revision ID: f1a2b3c4d5e6
Revises: e9f0a1b2c3d4
Create Date: 2026-06-28 03:10:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "f1a2b3c4d5e6"
down_revision: Union[str, Sequence[str], None] = "e9f0a1b2c3d4"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column("player_roster_entries", sa.Column("primary_zone", sa.String(length=30), nullable=True))
    op.add_column("player_roster_entries", sa.Column("secondary_zones", sa.JSON(), nullable=True))
    op.add_column("player_roster_entries", sa.Column("position_label", sa.String(length=120), nullable=True))
    op.add_column("player_roster_entries", sa.Column("preferred_side", sa.String(length=30), nullable=True))
    op.add_column("player_roster_entries", sa.Column("notes", sa.Text(), nullable=True))


def downgrade() -> None:
    op.drop_column("player_roster_entries", "notes")
    op.drop_column("player_roster_entries", "preferred_side")
    op.drop_column("player_roster_entries", "position_label")
    op.drop_column("player_roster_entries", "secondary_zones")
    op.drop_column("player_roster_entries", "primary_zone")
