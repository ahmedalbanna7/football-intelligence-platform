"""add_primary_team_profile

Revision ID: 72dd6213073e
Revises: 5f81c4470c9a
Create Date: 2026-06-26 00:45:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "72dd6213073e"
down_revision: Union[str, Sequence[str], None] = "5f81c4470c9a"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "primary_team_profiles",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("team_name", sa.String(length=255), nullable=False),
        sa.Column("primary_kit_image_object_name", sa.String(length=500), nullable=True),
        sa.Column("alternate_kit_image_object_name", sa.String(length=500), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )

    op.add_column(
        "matches",
        sa.Column("primary_team_profile_id", sa.Integer(), nullable=True),
    )
    op.create_foreign_key(
        "fk_matches_primary_team_profile_id",
        "matches",
        "primary_team_profiles",
        ["primary_team_profile_id"],
        ["id"],
    )

    op.add_column(
        "matches",
        sa.Column(
            "primary_team_direction",
            sa.String(length=50),
            nullable=False,
            server_default="unknown",
        ),
    )
    op.add_column(
        "matches",
        sa.Column(
            "opponent_team_direction",
            sa.String(length=50),
            nullable=False,
            server_default="unknown",
        ),
    )
    op.add_column(
        "matches",
        sa.Column(
            "primary_team_kit_source",
            sa.String(length=50),
            nullable=False,
            server_default="primary",
        ),
    )


def downgrade() -> None:
    op.drop_column("matches", "primary_team_kit_source")
    op.drop_column("matches", "opponent_team_direction")
    op.drop_column("matches", "primary_team_direction")
    op.drop_constraint(
        "fk_matches_primary_team_profile_id",
        "matches",
        type_="foreignkey",
    )
    op.drop_column("matches", "primary_team_profile_id")
    op.drop_table("primary_team_profiles")
