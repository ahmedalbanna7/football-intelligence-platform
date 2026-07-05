"""add_opponent_team_profiles

Revision ID: d8a7b6c5e4f3
Revises: c4b6d0f1a9e2
Create Date: 2026-06-27 02:30:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "d8a7b6c5e4f3"
down_revision: Union[str, Sequence[str], None] = "c4b6d0f1a9e2"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "teams",
        sa.Column(
            "team_type",
            sa.String(length=50),
            nullable=False,
            server_default="opponent",
        ),
    )
    op.add_column(
        "teams",
        sa.Column("primary_kit_image_object_name", sa.String(length=500), nullable=True),
    )
    op.add_column(
        "teams",
        sa.Column("alternate_kit_image_object_name", sa.String(length=500), nullable=True),
    )
    op.add_column("teams", sa.Column("notes", sa.Text(), nullable=True))

    op.add_column("matches", sa.Column("opponent_team_id", sa.Integer(), nullable=True))
    op.create_foreign_key(
        "fk_matches_opponent_team_id_teams",
        "matches",
        "teams",
        ["opponent_team_id"],
        ["id"],
    )
    op.execute(
        """
        UPDATE teams
        SET team_type = 'primary'
        WHERE name IN (
            SELECT team_name FROM primary_team_profiles
        )
        OR lower(name) IN (
            SELECT lower(team_name) FROM primary_team_profiles
        )
        """
    )


def downgrade() -> None:
    op.drop_constraint("fk_matches_opponent_team_id_teams", "matches", type_="foreignkey")
    op.drop_column("matches", "opponent_team_id")
    op.drop_column("teams", "notes")
    op.drop_column("teams", "alternate_kit_image_object_name")
    op.drop_column("teams", "primary_kit_image_object_name")
    op.drop_column("teams", "team_type")
