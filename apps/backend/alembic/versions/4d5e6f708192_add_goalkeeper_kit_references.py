"""add goalkeeper kit references

Revision ID: 4d5e6f708192
Revises: 3c4d5e6f7081
Create Date: 2026-08-24 12:00:00.000000
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "4d5e6f708192"
down_revision: Union[str, Sequence[str], None] = "3c4d5e6f7081"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "teams",
        sa.Column("goalkeeper_kit_image_object_name", sa.String(length=500), nullable=True),
    )
    op.add_column(
        "primary_team_profiles",
        sa.Column("goalkeeper_kit_image_object_name", sa.String(length=500), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("primary_team_profiles", "goalkeeper_kit_image_object_name")
    op.drop_column("teams", "goalkeeper_kit_image_object_name")
