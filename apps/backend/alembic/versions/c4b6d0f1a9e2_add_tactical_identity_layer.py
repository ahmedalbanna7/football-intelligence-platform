"""add_tactical_identity_layer

Revision ID: c4b6d0f1a9e2
Revises: a2a5c2f02e8a
Create Date: 2026-06-26 18:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "c4b6d0f1a9e2"
down_revision: Union[str, Sequence[str], None] = "a2a5c2f02e8a"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column("players", sa.Column("jersey_number", sa.Integer(), nullable=True))
    op.add_column("players", sa.Column("primary_zone", sa.String(length=30), nullable=True))
    op.add_column("players", sa.Column("secondary_zones", sa.JSON(), nullable=True))
    op.add_column("players", sa.Column("position_label", sa.String(length=120), nullable=True))
    op.add_column("players", sa.Column("preferred_side", sa.String(length=30), nullable=True))
    op.add_column("players", sa.Column("notes", sa.Text(), nullable=True))

    op.add_column("matches", sa.Column("primary_team_id", sa.Integer(), nullable=True))
    op.add_column("matches", sa.Column("formation", sa.String(length=30), nullable=True))
    op.create_foreign_key(
        "fk_matches_primary_team_id_teams",
        "matches",
        "teams",
        ["primary_team_id"],
        ["id"],
    )

    op.create_table(
        "match_lineups",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("match_id", sa.Integer(), nullable=False),
        sa.Column("team_id", sa.Integer(), nullable=True),
        sa.Column("player_id", sa.Integer(), nullable=False),
        sa.Column("jersey_number", sa.Integer(), nullable=True),
        sa.Column("starting_zone", sa.String(length=30), nullable=True),
        sa.Column("expected_zones", sa.JSON(), nullable=True),
        sa.Column("is_starter", sa.Boolean(), nullable=False),
        sa.Column("start_minute", sa.Integer(), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(["match_id"], ["matches.id"]),
        sa.ForeignKeyConstraint(["player_id"], ["players.id"]),
        sa.ForeignKeyConstraint(["team_id"], ["teams.id"]),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "match_id",
            "team_id",
            "player_id",
            name="uq_match_lineup_match_team_player",
        ),
    )
    op.create_index("ix_match_lineups_match_id", "match_lineups", ["match_id"])
    op.create_index("ix_match_lineups_team_id", "match_lineups", ["team_id"])

    op.create_table(
        "match_substitutions",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("match_id", sa.Integer(), nullable=False),
        sa.Column("team_id", sa.Integer(), nullable=True),
        sa.Column("minute", sa.Integer(), nullable=False),
        sa.Column("second", sa.Integer(), nullable=True),
        sa.Column("player_out_id", sa.Integer(), nullable=True),
        sa.Column("player_in_id", sa.Integer(), nullable=False),
        sa.Column("player_in_zone", sa.String(length=30), nullable=True),
        sa.Column("expected_zones", sa.JSON(), nullable=True),
        sa.Column("notes", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(["match_id"], ["matches.id"]),
        sa.ForeignKeyConstraint(["player_in_id"], ["players.id"]),
        sa.ForeignKeyConstraint(["player_out_id"], ["players.id"]),
        sa.ForeignKeyConstraint(["team_id"], ["teams.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_match_substitutions_match_id", "match_substitutions", ["match_id"])
    op.create_index("ix_match_substitutions_team_id", "match_substitutions", ["team_id"])

    op.create_table(
        "track_identity_assignments",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("match_id", sa.Integer(), nullable=False),
        sa.Column("frame_index", sa.Integer(), nullable=True),
        sa.Column("timestamp_ms", sa.Integer(), nullable=True),
        sa.Column("track_id", sa.Integer(), nullable=False),
        sa.Column("team_id", sa.Integer(), nullable=True),
        sa.Column("resolved_player_id", sa.Integer(), nullable=True),
        sa.Column("confidence", sa.Float(), nullable=True),
        sa.Column("candidate_scores", sa.JSON(), nullable=True),
        sa.Column("zone", sa.String(length=30), nullable=True),
        sa.Column("pitch_x", sa.Float(), nullable=True),
        sa.Column("pitch_y", sa.Float(), nullable=True),
        sa.Column("source", sa.String(length=80), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(["match_id"], ["matches.id"]),
        sa.ForeignKeyConstraint(["resolved_player_id"], ["players.id"]),
        sa.ForeignKeyConstraint(["team_id"], ["teams.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_track_identity_assignments_match_track",
        "track_identity_assignments",
        ["match_id", "track_id"],
    )
    op.create_index(
        "ix_track_identity_assignments_resolved_player_id",
        "track_identity_assignments",
        ["resolved_player_id"],
    )


def downgrade() -> None:
    op.drop_index(
        "ix_track_identity_assignments_resolved_player_id",
        table_name="track_identity_assignments",
    )
    op.drop_index(
        "ix_track_identity_assignments_match_track",
        table_name="track_identity_assignments",
    )
    op.drop_table("track_identity_assignments")

    op.drop_index("ix_match_substitutions_team_id", table_name="match_substitutions")
    op.drop_index("ix_match_substitutions_match_id", table_name="match_substitutions")
    op.drop_table("match_substitutions")

    op.drop_index("ix_match_lineups_team_id", table_name="match_lineups")
    op.drop_index("ix_match_lineups_match_id", table_name="match_lineups")
    op.drop_table("match_lineups")

    op.drop_constraint("fk_matches_primary_team_id_teams", "matches", type_="foreignkey")
    op.drop_column("matches", "formation")
    op.drop_column("matches", "primary_team_id")

    op.drop_column("players", "notes")
    op.drop_column("players", "preferred_side")
    op.drop_column("players", "position_label")
    op.drop_column("players", "secondary_zones")
    op.drop_column("players", "primary_zone")
    op.drop_column("players", "jersey_number")
