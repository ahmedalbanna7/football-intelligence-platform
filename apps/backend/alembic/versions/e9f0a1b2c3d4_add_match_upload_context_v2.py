"""add_match_upload_context_v2

Revision ID: e9f0a1b2c3d4
Revises: d8a7b6c5e4f3
Create Date: 2026-06-28 01:30:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "e9f0a1b2c3d4"
down_revision: Union[str, Sequence[str], None] = "d8a7b6c5e4f3"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "matches",
        sa.Column(
            "match_category",
            sa.String(length=50),
            nullable=False,
            server_default="competitive",
        ),
    )
    op.add_column(
        "matches",
        sa.Column(
            "matchup_type",
            sa.String(length=80),
            nullable=False,
            server_default="my_team_vs_opponent",
        ),
    )
    op.add_column("matches", sa.Column("another_team_name", sa.String(length=255), nullable=True))
    op.add_column("matches", sa.Column("another_team_id", sa.Integer(), nullable=True))
    op.add_column("matches", sa.Column("another_formation", sa.String(length=30), nullable=True))
    op.add_column(
        "matches",
        sa.Column(
            "another_team_kit_source",
            sa.String(length=50),
            nullable=False,
            server_default="auto",
        ),
    )
    op.create_foreign_key(
        "fk_matches_another_team_id_teams",
        "matches",
        "teams",
        ["another_team_id"],
        ["id"],
    )

    op.execute(
        """
        UPDATE matches
        SET match_category = CASE
            WHEN match_type = 'friendly_vs_opponent' THEN 'friendly'
            WHEN match_type = 'internal_scrimmage' THEN 'internal_scrimmage'
            WHEN match_type = 'academy_match' THEN 'academy_match'
            ELSE 'competitive'
        END,
        matchup_type = CASE
            WHEN match_type IN ('internal_scrimmage', 'academy_match') THEN match_type
            ELSE 'my_team_vs_opponent'
        END,
        another_team_name = opponent_team_name,
        another_team_id = opponent_team_id
        """
    )


def downgrade() -> None:
    op.drop_constraint("fk_matches_another_team_id_teams", "matches", type_="foreignkey")
    op.drop_column("matches", "another_team_kit_source")
    op.drop_column("matches", "another_formation")
    op.drop_column("matches", "another_team_id")
    op.drop_column("matches", "another_team_name")
    op.drop_column("matches", "matchup_type")
    op.drop_column("matches", "match_category")
