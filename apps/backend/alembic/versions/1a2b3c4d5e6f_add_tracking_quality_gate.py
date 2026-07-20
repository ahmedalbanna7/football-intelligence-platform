"""add_tracking_quality_gate

Revision ID: 1a2b3c4d5e6f
Revises: 0f4e7a8b9c10
Create Date: 2026-07-20 13:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "1a2b3c4d5e6f"
down_revision: Union[str, Sequence[str], None] = "0f4e7a8b9c10"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "tracking_quality_assessments",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("run_id", sa.Integer(), nullable=False),
        sa.Column("status", sa.String(length=40), nullable=False),
        sa.Column("tracker_engine", sa.String(length=120), nullable=True),
        sa.Column("reid_enabled", sa.Boolean(), nullable=False),
        sa.Column("reid_model", sa.String(length=255), nullable=True),
        sa.Column("average_identity_confidence", sa.Float(), nullable=True),
        sa.Column("suspected_id_switches", sa.Integer(), nullable=False),
        sa.Column("fragmented_tracks", sa.Integer(), nullable=False),
        sa.Column("tracks_needing_review", sa.Integer(), nullable=False),
        sa.Column("benchmark_status", sa.String(length=50), nullable=False),
        sa.Column("id_switches", sa.Integer(), nullable=True),
        sa.Column("idf1", sa.Float(), nullable=True),
        sa.Column("hota", sa.Float(), nullable=True),
        sa.Column("fragmentation", sa.Integer(), nullable=True),
        sa.Column("predictions_object", sa.String(length=500), nullable=True),
        sa.Column("ground_truth_object", sa.String(length=500), nullable=True),
        sa.Column("metrics_json", sa.JSON(), nullable=True),
        sa.Column("thresholds_json", sa.JSON(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
        sa.Column("reviewed_at", sa.DateTime(), nullable=True),
        sa.ForeignKeyConstraint(["run_id"], ["match_analysis_runs.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("run_id"),
    )
    op.create_table(
        "track_review_items",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("run_id", sa.Integer(), nullable=False),
        sa.Column("track_id", sa.Integer(), nullable=False),
        sa.Column("canonical_track_id", sa.Integer(), nullable=False),
        sa.Column("team_number", sa.Integer(), nullable=True),
        sa.Column("assigned_player_id", sa.Integer(), nullable=True),
        sa.Column("status", sa.String(length=40), nullable=False),
        sa.Column("identity_confidence", sa.Float(), nullable=False),
        sa.Column("reid_confidence", sa.Float(), nullable=False),
        sa.Column("motion_consistency", sa.Float(), nullable=False),
        sa.Column("team_consistency", sa.Float(), nullable=False),
        sa.Column("switch_risk", sa.String(length=20), nullable=False),
        sa.Column("fragment_count", sa.Integer(), nullable=False),
        sa.Column("raw_id_transitions", sa.Integer(), nullable=False),
        sa.Column("first_frame", sa.Integer(), nullable=True),
        sa.Column("last_frame", sa.Integer(), nullable=True),
        sa.Column("observation_count", sa.Integer(), nullable=False),
        sa.Column("raw_track_ids", sa.JSON(), nullable=True),
        sa.Column("issue_codes", sa.JSON(), nullable=True),
        sa.Column("crop_objects", sa.JSON(), nullable=True),
        sa.Column("observations_json", sa.JSON(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(["assigned_player_id"], ["players.id"], ondelete="SET NULL"),
        sa.ForeignKeyConstraint(["run_id"], ["match_analysis_runs.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("run_id", "track_id", name="uq_track_review_run_track"),
    )
    op.create_index("ix_track_review_items_run_id", "track_review_items", ["run_id"])
    op.create_index("ix_track_review_items_switch_risk", "track_review_items", ["switch_risk"])
    op.create_table(
        "track_review_corrections",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("run_id", sa.Integer(), nullable=False),
        sa.Column("action", sa.String(length=40), nullable=False),
        sa.Column("source_track_id", sa.Integer(), nullable=True),
        sa.Column("target_track_id", sa.Integer(), nullable=True),
        sa.Column("split_frame", sa.Integer(), nullable=True),
        sa.Column("assigned_player_id", sa.Integer(), nullable=True),
        sa.Column("assigned_team_number", sa.Integer(), nullable=True),
        sa.Column("before_json", sa.JSON(), nullable=True),
        sa.Column("after_json", sa.JSON(), nullable=True),
        sa.Column("note", sa.Text(), nullable=True),
        sa.Column("undone", sa.Boolean(), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(["assigned_player_id"], ["players.id"], ondelete="SET NULL"),
        sa.ForeignKeyConstraint(["run_id"], ["match_analysis_runs.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_track_review_corrections_run_id",
        "track_review_corrections",
        ["run_id"],
    )


def downgrade() -> None:
    op.drop_index(
        "ix_track_review_corrections_run_id",
        table_name="track_review_corrections",
    )
    op.drop_table("track_review_corrections")
    op.drop_index("ix_track_review_items_switch_risk", table_name="track_review_items")
    op.drop_index("ix_track_review_items_run_id", table_name="track_review_items")
    op.drop_table("track_review_items")
    op.drop_table("tracking_quality_assessments")
