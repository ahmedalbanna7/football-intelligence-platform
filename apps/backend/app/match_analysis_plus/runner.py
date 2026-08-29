from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter
from typing import Any
from collections.abc import Callable
import colorsys
import gc
import io
import json
import shutil
import subprocess

import cv2
import numpy as np

from app.core.config import settings
from app.match_analysis_plus.analytics_v1 import AnalyticsRealV1
from app.match_analysis_plus.reports_v2 import ReportsV2Builder
from app.services.minio_client import BUCKET_NAME, client


PLAYER_ALIASES = {"person", "player"}
GOALKEEPER_ALIASES = {"goalkeeper", "goal keeper"}
REFEREE_ALIASES = {"referee", "official"}
ASSISTANT_REFEREE_ALIASES = {"assistant referee", "assistant_referee", "linesman"}
STAFF_ALIASES = {"staff", "coach", "team staff"}
PERSON_ALIASES = (
    PLAYER_ALIASES
    | GOALKEEPER_ALIASES
    | REFEREE_ALIASES
    | ASSISTANT_REFEREE_ALIASES
    | STAFF_ALIASES
)
BALL_ALIASES = {"sports ball", "ball"}
PARTICIPANT_ROLES = {
    "player",
    "goalkeeper",
    "referee",
    "assistant_referee",
    "staff_outside_pitch",
}
ANALYTICS_ROLES = {"player", "goalkeeper"}
TEAM_DISPLAY_COLORS = {
    0: (0, 215, 255),
    1: (245, 245, 245),
    2: (48, 72, 224),
}
PITCH_LENGTH_CM = 10500.0
PITCH_WIDTH_CM = 6800.0
PENALTY_AREA_LENGTH_CM = 1650.0
PENALTY_AREA_WIDTH_CM = 4032.0
GOAL_AREA_LENGTH_CM = 550.0
GOAL_AREA_WIDTH_CM = 1832.0
GOAL_WIDTH_CM = 732.0
PENALTY_SPOT_DISTANCE_CM = 1100.0
CENTER_CIRCLE_RADIUS_CM = 915.0
VISUAL_LAYER_SCHEMA_VERSION = 2
VISUAL_LAYER_SAMPLE_RATE_HZ = 6.0
QUALITY_REVIEW_SAMPLE_RATE_HZ = 2.0
QUALITY_MAX_CROPS_PER_TRACK = 6
QUALITY_MAX_REVIEW_OBSERVATIONS_PER_TRACK = 600
QUALITY_THRESHOLDS = {
    "approve_identity_confidence": 0.82,
    "review_identity_confidence": 0.68,
    "high_risk_identity_confidence": 0.52,
    "high_risk_fragments": 2,
    "high_risk_raw_id_transitions": 4,
}
TRACK_VISUAL_PALETTE = (
    "#ef4444",
    "#0ea5e9",
    "#22c55e",
    "#f59e0b",
    "#8b5cf6",
    "#ec4899",
    "#14b8a6",
    "#f97316",
    "#6366f1",
    "#84cc16",
    "#06b6d4",
    "#e11d48",
    "#a855f7",
    "#10b981",
    "#eab308",
    "#3b82f6",
    "#d946ef",
    "#65a30d",
    "#0891b2",
    "#dc2626",
    "#7c3aed",
    "#059669",
    "#ca8a04",
    "#2563eb",
)


@dataclass
class AnalysisObject:
    track_id: int
    class_name: str
    bbox: list[float]
    confidence: float | None = None
    raw_track_id: int | None = None
    is_predicted: bool = False
    role_name: str = "player"
    pitch_position: tuple[float, float] | None = None
    pitch_velocity: tuple[float, float] | None = None
    height_cm: float = 0.0
    trajectory_3d_confidence: float = 0.0


@dataclass
class StableTrackState:
    stable_id: int
    bbox: list[float]
    center: tuple[float, float]
    foot: tuple[float, float]
    velocity: tuple[float, float]
    foot_velocity: tuple[float, float]
    last_frame: int
    raw_ids_seen: set[int]
    appearance_hist: np.ndarray | None = None
    appearance_gallery: list[np.ndarray] = field(default_factory=list)
    jersey_color: tuple[int, int, int] | None = None
    jersey_family: str | None = None
    jersey_family_votes: dict[str, int] = field(default_factory=dict)
    role_name: str = "player"
    role_votes: dict[str, float] = field(default_factory=dict)
    role_locked: bool = False
    last_observed_role: str = "player"
    role_streak: int = 1
    bbox_height: float = 0.0
    depth_proxy: float = 0.0
    depth_velocity: float = 0.0
    last_reliable_frame: int = 0
    identity_locked: bool = False
    occlusion_hits: int = 0
    reliable_hits: int = 1
    hits: int = 1
    consecutive_hits: int = 1
    confirmed: bool = False
    first_frame: int = 0
    last_raw_id: int | None = None
    raw_id_transitions: int = 0
    fragments: int = 1
    max_observation_gap: int = 0
    assignment_scores: list[float] = field(default_factory=list)
    detection_confidences: list[float] = field(default_factory=list)
    max_motion_gate_ratio: float = 0.0
    native_reid_reentries: int = 0


@dataclass
class ParticipantRoleState:
    track_id: int
    role_name: str = "player"
    confidence: float = 0.0
    locked: bool = False
    observations: int = 0
    consecutive_role_frames: int = 0
    last_candidate_role: str = "player"
    scores: dict[str, float] = field(
        default_factory=lambda: {role: 0.0 for role in PARTICIPANT_ROLES}
    )
    inside_pitch_observations: int = 0
    outside_pitch_observations: int = 0
    near_goal_observations: int = 0
    goalkeeper_zone_observations: int = 0
    penalty_area_observations: int = 0
    kit_outlier_observations: int = 0
    kit_outlier_score_sum: float = 0.0
    touchline_observations: int = 0
    team_affinity_observations: int = 0
    detector_role_observations: dict[str, int] = field(default_factory=dict)
    evidence: list[str] = field(default_factory=list)


class PlayerValidityFilter:
    """Reject obvious field fixtures before they can receive a stable identity."""

    def __init__(self) -> None:
        self.raw_seen = 0
        self.kept = 0
        self.rejected_implausible_shape = 0
        self.rejected_field_fixture = 0
        self.rejected_sparse_foreground = 0
        self.specialized_observations = 0

    def filter(
        self,
        players: list[AnalysisObject],
        frame: np.ndarray,
        specialized_detector: bool = False,
    ) -> list[AnalysisObject]:
        kept: list[AnalysisObject] = []
        minimum_height = (
            3.0
            if specialized_detector
            else max(4.0, float(frame.shape[0]) * 0.012)
        )
        for player in players:
            self.raw_seen += 1
            width = max(1.0, player.bbox[2] - player.bbox[0])
            height = max(1.0, player.bbox[3] - player.bbox[1])
            aspect_ratio = width / height
            if not self._has_plausible_person_geometry(
                player,
                minimum_height=minimum_height,
            ):
                self.rejected_implausible_shape += 1
                continue
            if specialized_detector:
                self.specialized_observations += 1
            elif self._looks_like_thin_field_fixture(frame, player.bbox, aspect_ratio):
                self.rejected_field_fixture += 1
                continue
            kept.append(player)
            self.kept += 1
        return kept

    @staticmethod
    def geometry_candidates(
        players: list[AnalysisObject],
        frame_height: int | None = None,
    ) -> list[AnalysisObject]:
        """Broad physical-body guard that never creates canonical tracks."""

        minimum_height = max(4.0, float(frame_height or 1080) * 0.012)
        return [
            player
            for player in players
            if PlayerValidityFilter._has_plausible_person_geometry(
                player,
                minimum_height=minimum_height,
            )
        ]

    @staticmethod
    def _has_plausible_person_geometry(
        player: AnalysisObject,
        minimum_height: float,
    ) -> bool:
        width = max(1.0, player.bbox[2] - player.bbox[0])
        height = max(1.0, player.bbox[3] - player.bbox[1])
        aspect_ratio = width / height
        return height >= minimum_height and 0.105 <= aspect_ratio <= 1.18

    def summary(self) -> dict[str, Any]:
        return {
            "engine": "player_validity_filter_v1",
            "raw_player_detections": self.raw_seen,
            "kept_player_detections": self.kept,
            "rejected_implausible_shape": self.rejected_implausible_shape,
            "rejected_field_fixtures": self.rejected_field_fixture,
            "rejected_sparse_foreground": self.rejected_sparse_foreground,
            "specialized_detector_observations": self.specialized_observations,
        }

    def _looks_like_thin_field_fixture(
        self,
        frame: np.ndarray,
        bbox: list[float],
        aspect_ratio: float,
    ) -> bool:
        frame_height, frame_width = frame.shape[:2]
        x1, y1, x2, y2 = [int(round(value)) for value in bbox]
        x1, x2 = max(0, x1), min(frame_width, x2)
        y1, y2 = max(0, y1), min(frame_height, y2)
        if x2 - x1 < 3 or y2 - y1 < 18:
            return True
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return True

        (
            background_fill,
            background_column_support,
            middle_fill,
            middle_row_median,
        ) = self._background_relative_shape(crop)
        if (
            aspect_ratio < 0.36
            and middle_fill < 0.36
            and middle_row_median < 0.30
        ):
            self.rejected_sparse_foreground += 1
            return True
        if (
            aspect_ratio < 1.05
            and background_fill < 0.22
            and background_column_support < 0.16
        ):
            self.rejected_sparse_foreground += 1
            return True

        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        green_field = (
            (hsv[:, :, 0] >= 28)
            & (hsv[:, :, 0] <= 96)
            & (hsv[:, :, 1] >= 32)
            & (hsv[:, :, 2] >= 28)
        )
        non_field = ~green_field
        lower = non_field[int(non_field.shape[0] * 0.38) :, :]
        if lower.size == 0:
            return False
        lower_fill = float(np.mean(lower))
        lower_column_support = float(np.mean(np.mean(lower, axis=0) >= 0.22))

        if aspect_ratio < 0.16 and lower_fill < 0.30:
            return True
        if aspect_ratio < 1.0 and lower_fill < 0.18 and lower_column_support < 0.22:
            return True
        return (
            aspect_ratio < 0.30
            and lower_fill < 0.17
            and lower_column_support < 0.34
        )

    def _background_relative_shape(
        self,
        crop: np.ndarray,
    ) -> tuple[float, float, float, float]:
        height, width = crop.shape[:2]
        border_size = max(1, min(height, width) // 10)
        border = np.concatenate(
            [
                crop[:border_size, :, :].reshape(-1, 3),
                crop[-border_size:, :, :].reshape(-1, 3),
                crop[border_size:-border_size or None, :border_size, :].reshape(-1, 3),
                crop[border_size:-border_size or None, -border_size:, :].reshape(-1, 3),
            ],
            axis=0,
        )
        background_bgr = np.median(border, axis=0).astype(np.uint8)
        crop_lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB).astype(np.float32)
        background_lab = cv2.cvtColor(
            background_bgr.reshape(1, 1, 3),
            cv2.COLOR_BGR2LAB,
        ).astype(np.float32)[0, 0]
        foreground = np.linalg.norm(crop_lab - background_lab, axis=2) >= 22.0
        lower = foreground[int(height * 0.38) :, :]
        middle = foreground[int(height * 0.34) : max(int(height * 0.72), 1), :]
        if lower.size == 0:
            return 1.0, 1.0, 1.0, 1.0
        return (
            float(np.mean(lower)),
            float(np.mean(np.mean(lower, axis=0) >= 0.22)),
            float(np.mean(middle)) if middle.size else 1.0,
            float(np.median(np.mean(middle, axis=1))) if middle.size else 1.0,
        )


class PitchOccupancyFilter:
    """Keep only people whose ground contact point belongs to the playing surface."""

    def __init__(self, grace_frames: int = 3) -> None:
        self.grace_frames = grace_frames
        self.last_inside_by_raw_id: dict[int, int] = {}
        self.raw_seen = 0
        self.kept = 0
        self.rejected_outside_pitch = 0
        self.rejected_non_field_foot = 0
        self.kept_official_margin = 0
        self.metric_decisions = 0
        self.visual_fallback_decisions = 0
        self.last_visual_mask: np.ndarray | None = None

    def filter(
        self,
        frame_index: int,
        players: list[AnalysisObject],
        frame: np.ndarray,
        radar: "PitchRadar",
    ) -> list[AnalysisObject]:
        kept: list[AnalysisObject] = []
        visual_mask = radar.playing_surface_mask(frame)
        self.last_visual_mask = visual_mask
        for player in players:
            self.raw_seen += 1
            raw_id = player.raw_track_id if player.raw_track_id is not None else player.track_id
            foot = ((player.bbox[0] + player.bbox[2]) / 2, player.bbox[3])
            metric_available = radar.is_reliable(0.52)
            if metric_available:
                self.metric_decisions += 1
                inside = radar.contains_image_point(foot, margin_cm=90.0)
                if (
                    not inside
                    and player.role_name in {"referee", "assistant_referee"}
                    and radar.contains_image_point(foot, margin_cm=600.0)
                ):
                    inside = True
                    self.kept_official_margin += 1
            else:
                self.visual_fallback_decisions += 1
                inside = self._mask_contains(visual_mask, foot)

            if inside:
                self.last_inside_by_raw_id[raw_id] = frame_index
                kept.append(player)
                self.kept += 1
                continue

            last_inside = self.last_inside_by_raw_id.get(raw_id)
            if last_inside is not None and frame_index - last_inside <= self.grace_frames:
                kept.append(player)
                self.kept += 1
                continue

            if metric_available:
                self.rejected_outside_pitch += 1
            else:
                self.rejected_non_field_foot += 1
        return kept

    def summary(self) -> dict[str, Any]:
        return {
            "engine": "metric_pitch_occupancy_filter_v2",
            "raw_player_candidates": self.raw_seen,
            "kept_player_candidates": self.kept,
            "rejected_outside_pitch": self.rejected_outside_pitch,
            "rejected_non_field_foot": self.rejected_non_field_foot,
            "metric_decisions": self.metric_decisions,
            "visual_fallback_decisions": self.visual_fallback_decisions,
            "kept_official_margin": self.kept_official_margin,
            "grace_frames": self.grace_frames,
        }

    def _mask_contains(
        self,
        mask: np.ndarray | None,
        point: tuple[float, float],
    ) -> bool:
        if mask is None:
            return True
        x = int(round(point[0]))
        y = int(round(point[1]))
        if x < 0 or y < 0 or x >= mask.shape[1] or y >= mask.shape[0]:
            return False
        radius = max(5, int(round(mask.shape[1] * 0.004)))
        y1, y2 = max(0, y - radius), min(mask.shape[0], y + radius + 1)
        x1, x2 = max(0, x - radius), min(mask.shape[1], x + radius + 1)
        return float(np.mean(mask[y1:y2, x1:x2] > 0)) >= 0.20


class TrackIdStabilizer:
    def __init__(
        self,
        max_gap_frames: int = 120,
        confirmation_hits: int = 4,
        hidden_hold_frames: int = 18,
        long_gap_frames: int = 30,
    ) -> None:
        self.max_gap_frames = max_gap_frames
        self.confirmation_hits = confirmation_hits
        self.hidden_hold_frames = hidden_hold_frames
        self.long_gap_frames = long_gap_frames
        self.next_stable_id = 1
        self.tracks: dict[int, StableTrackState] = {}
        self.raw_to_stable: dict[int, int] = {}
        self.raw_ids_seen: set[int] = set()
        self.raw_id_reassignments = 0
        self.appearance_matches = 0
        self.rejected_far_matches = 0
        self.rejected_appearance_mismatches = 0
        self.rejected_jersey_mismatches = 0
        self.rejected_color_family_mismatches = 0
        self.rejected_depth_mismatches = 0
        self.rejected_direction_mismatches = 0
        self.locked_identity_rejections = 0
        self.crowded_visual_freezes = 0
        self.prediction_ambiguity_freezes = 0
        self.hidden_occlusion_holds = 0
        self.suppressed_ambiguous_detections = 0
        self.suppressed_uncertain_associations = 0
        self.suppressed_duplicate_candidates = 0
        self.suppressed_tentative_outputs = 0
        self.discarded_tentative_tracks = 0
        self.global_assignment_frames = 0
        self.global_assignment_fallbacks = 0
        self.motion_matches = 0
        self.raw_id_identity_mismatch_ignores = 0
        self.raw_id_motion_conflict_ignores = 0
        self.rejected_hard_motion_jumps = 0
        self.rejected_long_gap_reentries = 0
        self.rejected_role_conflicts = 0
        self.frozen_identity_visual_updates = 0

    def update(self, frame_index: int, players: list[AnalysisObject], frame: np.ndarray | None = None) -> list[AnalysisObject]:
        self._expire_tentative_tracks(frame_index)
        candidates: list[dict[str, Any]] = []
        for player in players:
            raw_id = player.raw_track_id if player.raw_track_id is not None else player.track_id
            self.raw_ids_seen.add(raw_id)
            appearance_hist, jersey_color = self._extract_appearance(frame, player.bbox)
            current_crowding = self._is_crowded_detection(player, players)
            severe_overlap = self._is_severe_overlap_detection(player, players)
            prediction_ambiguity = self._is_prediction_ambiguous(player, players, frame_index)
            matching_visual_reliable = appearance_hist is not None
            visual_quality = 0.45 if severe_overlap else (0.65 if current_crowding else 1.0)
            update_visual_reliable = (
                matching_visual_reliable
                and not current_crowding
                and not prediction_ambiguity
            )
            if not update_visual_reliable:
                self.crowded_visual_freezes += 1
                if prediction_ambiguity:
                    self.prediction_ambiguity_freezes += 1
            candidates.append(
                {
                    "player": player,
                    "raw_id": raw_id,
                    "appearance_hist": appearance_hist,
                    "jersey_color": jersey_color,
                    "jersey_family": self._jersey_family(jersey_color),
                    "matching_visual_reliable": matching_visual_reliable,
                    "visual_quality": visual_quality,
                    "update_visual_reliable": update_visual_reliable,
                    "identity_ambiguous": prediction_ambiguity,
                    "crowded": current_crowding,
                    "severe_overlap": severe_overlap,
                }
            )

        pair_scores: list[tuple[float, int, int]] = []
        for candidate_index, candidate in enumerate(candidates):
            if candidate["identity_ambiguous"]:
                continue
            for stable_id, state in self.tracks.items():
                gap = frame_index - state.last_frame
                if gap < 0 or gap > self.max_gap_frames:
                    continue
                score = self._candidate_score(
                    state=state,
                    player=candidate["player"],
                    frame_index=frame_index,
                    raw_id=candidate["raw_id"],
                    appearance_hist=candidate["appearance_hist"],
                    jersey_color=candidate["jersey_color"],
                    jersey_family=candidate["jersey_family"],
                    visual_reliable=candidate["matching_visual_reliable"],
                    visual_quality=candidate["visual_quality"],
                )
                if score is not None:
                    pair_scores.append((score, stable_id, candidate_index))

        assigned_candidates, assigned_scores = self._solve_global_assignment(pair_scores)
        if candidates and self.tracks:
            self.global_assignment_frames += 1
        for candidate_index, score in assigned_scores.items():
            if score >= 3.0:
                self.motion_matches += 1

        uncertain_candidates: set[int] = set()
        for candidate_index, stable_id in assigned_candidates.items():
            if self._assignment_is_uncertain(
                candidate_index=candidate_index,
                stable_id=stable_id,
                assigned_score=assigned_scores[candidate_index],
                pair_scores=pair_scores,
                crowded=bool(candidates[candidate_index]["crowded"]),
                severe_overlap=bool(candidates[candidate_index]["severe_overlap"]),
                frame_index=frame_index,
            ):
                uncertain_candidates.add(candidate_index)
                self.suppressed_uncertain_associations += 1

        stabilized: list[AnalysisObject] = []
        updated_stable_ids: set[int] = set()
        for candidate_index, candidate in enumerate(candidates):
            if candidate["identity_ambiguous"]:
                self.suppressed_ambiguous_detections += 1
                continue
            if candidate_index in uncertain_candidates:
                continue
            player = candidate["player"]
            raw_id = candidate["raw_id"]
            appearance_hist = candidate["appearance_hist"]
            jersey_color = candidate["jersey_color"]
            visual_reliable = candidate["update_visual_reliable"]
            stable_id = assigned_candidates.get(candidate_index)
            created = stable_id is None
            if stable_id is None:
                if candidate["severe_overlap"] or self._near_confirmed_prediction(
                    player,
                    frame_index,
                ):
                    self.suppressed_duplicate_candidates += 1
                    continue
                stable_id = self._create_track(player, frame_index, raw_id, appearance_hist, jersey_color)
            if not created:
                self._update_track(
                    stable_id,
                    player,
                    frame_index,
                    raw_id,
                    appearance_hist,
                    jersey_color,
                    visual_reliable,
                    assigned_scores.get(candidate_index),
                )
            updated_stable_ids.add(stable_id)
            state = self.tracks[stable_id]
            if not state.confirmed:
                self.suppressed_tentative_outputs += 1
                continue
            stabilized.append(
                AnalysisObject(
                    track_id=stable_id,
                    class_name=player.class_name,
                    bbox=player.bbox,
                    confidence=player.confidence,
                    raw_track_id=raw_id,
                    role_name=state.role_name,
                )
            )

        for stable_id, state in self.tracks.items():
            if stable_id in updated_stable_ids:
                continue
            gap = frame_index - state.last_frame
            state.consecutive_hits = 0
            if state.confirmed and 0 < gap <= self.hidden_hold_frames:
                self.hidden_occlusion_holds += 1
        return stabilized

    def summary(self) -> dict[str, Any]:
        confirmed_tracks = {
            stable_id: state
            for stable_id, state in self.tracks.items()
            if state.confirmed
        }
        raw_ids_per_stable = {
            stable_id: len(state.raw_ids_seen)
            for stable_id, state in confirmed_tracks.items()
        }
        stable_count = max(len(confirmed_tracks), 1)
        raw_count = len(self.raw_ids_seen)
        motion_gate_ratios = [
            state.max_motion_gate_ratio
            for state in confirmed_tracks.values()
        ]
        return {
            "engine": "identity_isolation_stabilizer_v5_conservative_reid",
            "raw_track_ids_seen": raw_count,
            "stable_tracks_count": len(confirmed_tracks),
            "internal_track_candidates": len(self.tracks),
            "tentative_tracks_count": len(self.tracks) - len(confirmed_tracks),
            "max_gap_frames": self.max_gap_frames,
            "confirmation_hits": self.confirmation_hits,
            "hidden_hold_frames": self.hidden_hold_frames,
            "long_gap_frames": self.long_gap_frames,
            "raw_id_reassignments": self.raw_id_reassignments,
            "appearance_matches": self.appearance_matches,
            "rejected_far_matches": self.rejected_far_matches,
            "rejected_appearance_mismatches": self.rejected_appearance_mismatches,
            "rejected_jersey_mismatches": self.rejected_jersey_mismatches,
            "rejected_color_family_mismatches": self.rejected_color_family_mismatches,
            "rejected_depth_mismatches": self.rejected_depth_mismatches,
            "rejected_direction_mismatches": self.rejected_direction_mismatches,
            "locked_identity_rejections": self.locked_identity_rejections,
            "crowded_visual_freezes": self.crowded_visual_freezes,
            "prediction_ambiguity_freezes": self.prediction_ambiguity_freezes,
            "hidden_occlusion_holds": self.hidden_occlusion_holds,
            "predicted_boxes_rendered": 0,
            "suppressed_ambiguous_detections": self.suppressed_ambiguous_detections,
            "suppressed_uncertain_associations": self.suppressed_uncertain_associations,
            "suppressed_duplicate_candidates": self.suppressed_duplicate_candidates,
            "suppressed_tentative_outputs": self.suppressed_tentative_outputs,
            "discarded_tentative_tracks": self.discarded_tentative_tracks,
            "global_assignment_frames": self.global_assignment_frames,
            "global_assignment_fallbacks": self.global_assignment_fallbacks,
            "motion_matches": self.motion_matches,
            "raw_id_identity_mismatch_ignores": self.raw_id_identity_mismatch_ignores,
            "raw_id_motion_conflict_ignores": self.raw_id_motion_conflict_ignores,
            "rejected_hard_motion_jumps": self.rejected_hard_motion_jumps,
            "rejected_long_gap_reentries": self.rejected_long_gap_reentries,
            "rejected_role_conflicts": self.rejected_role_conflicts,
            "frozen_identity_visual_updates": self.frozen_identity_visual_updates,
            "tracks_with_multiple_raw_ids": sum(1 for count in raw_ids_per_stable.values() if count > 1),
            "max_raw_ids_per_stable_track": max(raw_ids_per_stable.values(), default=0),
            "avg_raw_ids_per_stable_track": round(raw_count / stable_count, 3),
            "fragmentation_reduction_percent": round(max(0, raw_count - len(confirmed_tracks)) * 100 / max(raw_count, 1), 2),
            "raw_ids_per_stable_track": raw_ids_per_stable,
            "identity_locked_tracks": sum(1 for state in confirmed_tracks.values() if state.identity_locked),
            "max_accepted_motion_gate_ratio": round(max(motion_gate_ratios, default=0.0), 4),
            "tracks_near_motion_gate": sum(1 for ratio in motion_gate_ratios if ratio >= 0.82),
            "tracks_over_motion_gate": sum(1 for ratio in motion_gate_ratios if ratio > 1.0),
        }

    def _solve_global_assignment(
        self,
        pair_scores: list[tuple[float, int, int]],
    ) -> tuple[dict[int, int], dict[int, float]]:
        if not pair_scores:
            return {}, {}

        stable_ids = sorted({stable_id for _, stable_id, _ in pair_scores})
        candidate_ids = sorted({candidate_id for _, _, candidate_id in pair_scores})
        stable_index = {stable_id: index for index, stable_id in enumerate(stable_ids)}
        candidate_index = {candidate_id: index for index, candidate_id in enumerate(candidate_ids)}
        scores = np.full((len(stable_ids), len(candidate_ids)), -1e6, dtype=np.float64)
        score_by_pair: dict[tuple[int, int], float] = {}
        for score, stable_id, candidate_id in pair_scores:
            row = stable_index[stable_id]
            column = candidate_index[candidate_id]
            scores[row, column] = max(scores[row, column], score)
            score_by_pair[(stable_id, candidate_id)] = max(
                score_by_pair.get((stable_id, candidate_id), -1e6),
                score,
            )

        try:
            import lap

            finite_scores = scores[scores > -1e5]
            ceiling = float(np.max(finite_scores)) + 1.0
            costs = np.where(scores > -1e5, ceiling - scores, 1e6)
            _, row_assignment, _ = lap.lapjv(costs, extend_cost=True, cost_limit=100.0)
            assigned: dict[int, int] = {}
            assigned_scores: dict[int, float] = {}
            for row, column in enumerate(row_assignment):
                if column < 0 or column >= len(candidate_ids) or costs[row, column] >= 1e5:
                    continue
                stable_id = stable_ids[row]
                candidate_id = candidate_ids[column]
                assigned[candidate_id] = stable_id
                assigned_scores[candidate_id] = score_by_pair[(stable_id, candidate_id)]
            return assigned, assigned_scores
        except (ImportError, TypeError, ValueError, RuntimeError):
            self.global_assignment_fallbacks += 1
            assigned = {}
            assigned_scores = {}
            used_stable_ids: set[int] = set()
            for score, stable_id, candidate_id in sorted(
                pair_scores,
                reverse=True,
                key=lambda item: item[0],
            ):
                if stable_id in used_stable_ids or candidate_id in assigned:
                    continue
                assigned[candidate_id] = stable_id
                assigned_scores[candidate_id] = score
                used_stable_ids.add(stable_id)
            return assigned, assigned_scores

    def _assignment_is_uncertain(
        self,
        candidate_index: int,
        stable_id: int,
        assigned_score: float,
        pair_scores: list[tuple[float, int, int]],
        crowded: bool,
        severe_overlap: bool,
        frame_index: int,
    ) -> bool:
        competing_scores = [
            score
            for score, other_stable_id, other_candidate_index in pair_scores
            if (
                other_candidate_index == candidate_index
                and other_stable_id != stable_id
            )
            or (
                other_stable_id == stable_id
                and other_candidate_index != candidate_index
            )
        ]
        if not competing_scores:
            return False
        gap = max(frame_index - self.tracks[stable_id].last_frame, 1)
        required_margin = (
            0.90
            if severe_overlap
            else 0.52
            if crowded
            else 0.42
            if gap > 8
            else 0.20
        )
        return assigned_score - max(competing_scores) < required_margin

    def _near_confirmed_prediction(
        self,
        player: AnalysisObject,
        frame_index: int,
    ) -> bool:
        candidate_foot = self._foot(player.bbox)
        for state in self.tracks.values():
            if not state.confirmed:
                continue
            gap = frame_index - state.last_frame
            if gap < 0 or gap > self.hidden_hold_frames:
                continue
            predicted_foot = self._predicted_foot(state, max(gap, 1))
            gate = max(
                38.0,
                min(
                    160.0,
                    max(state.bbox_height, self._bbox_height(player.bbox)) * 0.42
                    + gap * 3.0,
                ),
            )
            if self._center_distance(candidate_foot, predicted_foot) <= gate:
                return True
        return False

    def _expire_tentative_tracks(self, frame_index: int) -> None:
        expired_ids = [
            stable_id
            for stable_id, state in self.tracks.items()
            if not state.confirmed and frame_index - state.last_frame > 6
        ]
        if not expired_ids:
            return
        expired = set(expired_ids)
        for stable_id in expired_ids:
            del self.tracks[stable_id]
        self.raw_to_stable = {
            raw_id: stable_id
            for raw_id, stable_id in self.raw_to_stable.items()
            if stable_id not in expired
        }
        self.discarded_tentative_tracks += len(expired_ids)

    def _stable_from_raw(
        self,
        raw_id: int,
        player: AnalysisObject,
        frame_index: int,
        used_stable_ids: set[int],
        appearance_hist: np.ndarray | None,
        jersey_color: tuple[int, int, int] | None,
    ) -> int | None:
        stable_id = self.raw_to_stable.get(raw_id)
        if stable_id is None or stable_id in used_stable_ids:
            return None
        state = self.tracks.get(stable_id)
        if state is None or frame_index - state.last_frame > self.max_gap_frames:
            return None
        if self._raw_id_identity_mismatch(state, player, frame_index, appearance_hist, jersey_color):
            self.raw_id_identity_mismatch_ignores += 1
            return None
        if self._is_compatible(state, player, frame_index, appearance_hist, jersey_color):
            return stable_id
        self.raw_id_reassignments += 1
        return None

    def _match_existing(
        self,
        player: AnalysisObject,
        frame_index: int,
        used_stable_ids: set[int],
        appearance_hist: np.ndarray | None,
        jersey_color: tuple[int, int, int] | None,
    ) -> int | None:
        best: tuple[float, int] | None = None
        for stable_id, state in self.tracks.items():
            if stable_id in used_stable_ids:
                continue
            gap = frame_index - state.last_frame
            if gap < 0 or gap > self.max_gap_frames:
                continue
            iou = self._iou(player.bbox, state.bbox)
            predicted = self._predicted_center(state, gap)
            distance = self._center_distance(self._center(player.bbox), predicted)
            max_distance = self._max_center_distance(player.bbox, state.bbox, gap)
            appearance = self._appearance_similarity(appearance_hist, state.appearance_hist)
            color_similarity = self._color_similarity(jersey_color, state.jersey_color)
            if self._is_locked_jersey_mismatch(state, jersey_color, appearance, color_similarity):
                self.rejected_jersey_mismatches += 1
                continue
            far_distance_gate = max_distance * (2.6 if gap > 8 else 1.8)
            if iou <= 0.02 and distance > far_distance_gate:
                self.rejected_far_matches += 1
                continue
            if distance > max_distance and not (appearance >= 0.80 and color_similarity >= 0.72):
                continue
            position_score = max(0.0, 1.0 - (distance / max_distance))
            score = (
                (iou * 2.2)
                + (position_score * 3.0)
                + (appearance * 2.4)
                + (color_similarity * 1.2)
                - (gap * 0.0025)
            )
            if best is None or score > best[0]:
                best = (score, stable_id)
        if best is not None:
            state = self.tracks[best[1]]
            if self._appearance_similarity(appearance_hist, state.appearance_hist) >= 0.70:
                self.appearance_matches += 1
        return best[1] if best is not None else None

    def _candidate_score(
        self,
        state: StableTrackState,
        player: AnalysisObject,
        frame_index: int,
        raw_id: int,
        appearance_hist: np.ndarray | None,
        jersey_color: tuple[int, int, int] | None,
        jersey_family: str | None,
        visual_reliable: bool,
        visual_quality: float,
    ) -> float | None:
        gap = max(frame_index - state.last_frame, 1)
        center = self._center(player.bbox)
        foot = self._foot(player.bbox)
        predicted_center = self._predicted_center(state, gap)
        predicted_foot = self._predicted_foot(state, gap)
        center_distance = self._center_distance(center, predicted_center)
        foot_distance = self._center_distance(foot, predicted_foot)
        lateral_gap = abs(foot[0] - predicted_foot[0])
        ground_depth_gap = abs(foot[1] - predicted_foot[1])
        max_distance = self._max_center_distance(player.bbox, state.bbox, gap)
        iou = self._iou(player.bbox, state.bbox)
        appearance = self._state_appearance_similarity(appearance_hist, state)
        color_similarity = self._color_similarity(jersey_color, state.jersey_color)
        family_confidence = self._jersey_family_confidence(state)
        height = self._bbox_height(player.bbox)
        height_ratio = min(height, state.bbox_height) / max(height, state.bbox_height, 1.0)
        depth_proxy = self._depth_proxy(player.bbox)
        predicted_depth = state.depth_proxy + state.depth_velocity * min(gap, 45)
        perspective_depth_gap = abs(depth_proxy - predicted_depth)
        direction_similarity = self._direction_similarity(state, foot, gap)
        same_raw_owner = self.raw_to_stable.get(raw_id) == state.stable_id
        trusted_visual = visual_reliable and visual_quality >= 0.80
        raw_motion_conflict = same_raw_owner and self._raw_id_motion_conflict(
            state=state,
            player=player,
            frame_index=frame_index,
        )
        raw_identity_conflict = same_raw_owner and (
            raw_motion_conflict
            or self._raw_id_identity_mismatch(
                state,
                player,
                frame_index,
                appearance_hist,
                jersey_color,
            )
        )
        if raw_identity_conflict:
            self.raw_id_identity_mismatch_ignores += 1
            if raw_motion_conflict:
                self.raw_id_motion_conflict_ignores += 1
        same_raw = same_raw_owner and not raw_identity_conflict
        bootstrap_raw = same_raw and not trusted_visual and state.reliable_hits < 4
        observed_role = self._normalized_role(player.role_name)

        direct_foot_distance = self._center_distance(foot, state.foot)
        hard_motion_gate = self._hard_motion_gate(state, player.bbox, gap)
        if direct_foot_distance > hard_motion_gate and iou < 0.04:
            self.rejected_hard_motion_jumps += 1
            if state.identity_locked:
                self.locked_identity_rejections += 1
            return None

        if gap > self.long_gap_frames:
            native_reid_evidence = (
                same_raw
                and trusted_visual
                and appearance >= 0.62
                and color_similarity >= 0.42
            )
            if not native_reid_evidence:
                self.rejected_long_gap_reentries += 1
                return None

        # Role labels are semantic metadata, not identity evidence. A temporary
        # referee/goalkeeper class error must never force an otherwise valid
        # physical identity to jump to a different stable track.

        family_mismatch = self._color_family_mismatch(state.jersey_family, jersey_family)
        if family_mismatch and family_confidence >= 0.70:
            self.rejected_color_family_mismatches += 1
            if (
                (state.identity_locked or state.reliable_hits >= 8)
                and trusted_visual
                and color_similarity < 0.62
                and appearance < 0.84
            ):
                self.locked_identity_rejections += 1
                return None

        if trusted_visual and self._is_locked_jersey_mismatch(
            state,
            jersey_color,
            appearance,
            color_similarity,
        ):
            self.rejected_jersey_mismatches += 1
            if state.identity_locked:
                self.locked_identity_rejections += 1
            return None

        if self._is_strong_color_conflict(
            state,
            jersey_color,
            appearance,
            color_similarity,
            trusted_visual,
        ):
            self.rejected_jersey_mismatches += 1
            return None

        if state.hits >= 6 and not (same_raw and trusted_visual) and not bootstrap_raw:
            allowed_depth_gap = max(42.0, min(180.0, state.bbox_height * 1.25 + gap * 3.5))
            if ground_depth_gap > allowed_depth_gap and iou < 0.04 and appearance < 0.78:
                self.rejected_depth_mismatches += 1
                if state.identity_locked:
                    self.locked_identity_rejections += 1
                return None
            if perspective_depth_gap > 0.68 and height_ratio < 0.52 and appearance < 0.80:
                self.rejected_depth_mismatches += 1
                if state.identity_locked:
                    self.locked_identity_rejections += 1
                return None

        far_gate = max_distance * (1.25 if gap <= 4 else 1.55)
        if (
            center_distance > far_gate
            and foot_distance > far_gate
            and not (same_raw and trusted_visual)
            and not bootstrap_raw
        ):
            self.rejected_far_matches += 1
            if state.identity_locked:
                self.locked_identity_rejections += 1
            return None

        if state.identity_locked and gap <= 8 and direction_similarity < -0.55 and not bootstrap_raw:
            if not trusted_visual or appearance < 0.84:
                self.rejected_direction_mismatches += 1
                self.locked_identity_rejections += 1
                return None

        if state.hits >= 5 and appearance_hist is not None and state.appearance_hist is not None:
            if appearance < 0.32 and color_similarity < 0.66:
                self.rejected_appearance_mismatches += 1
                if state.identity_locked:
                    self.locked_identity_rejections += 1
                return None

        center_score = max(0.0, 1.0 - center_distance / max(max_distance, 1.0))
        foot_score = max(0.0, 1.0 - foot_distance / max(max_distance, 1.0))
        lateral_scale = max(35.0, min(240.0, state.bbox_height * 1.2 + gap * 4.0))
        depth_scale = max(28.0, min(150.0, state.bbox_height * 0.8 + gap * 2.5))
        lateral_score = max(0.0, 1.0 - lateral_gap / lateral_scale)
        ground_depth_score = max(0.0, 1.0 - ground_depth_gap / depth_scale)
        perspective_depth_score = max(0.0, 1.0 - perspective_depth_gap / 0.55)
        direction_score = (direction_similarity + 1.0) / 2.0
        raw_bonus = 0.0
        if bootstrap_raw:
            raw_bonus = 2.4
        elif same_raw and trusted_visual:
            raw_bonus = 2.8 if gap <= 3 else 1.8 if gap <= 12 else 0.95

        family_penalty = 0.0
        if family_mismatch and family_confidence >= 0.55:
            family_penalty = 0.65

        score = (
            foot_score * 2.4
            + center_score * 1.5
            + lateral_score * 1.25
            + ground_depth_score * 1.45
            + perspective_depth_score * 1.0
            + direction_score * (1.55 if not trusted_visual else 1.05)
            + iou * 0.9
            + appearance * (1.9 * visual_quality if visual_reliable else 0.0)
            + color_similarity * (1.35 * visual_quality if visual_reliable else 0.0)
            + height_ratio * 0.85
            + raw_bonus
            - family_penalty
            - gap * 0.004
        )
        min_score = 4.05 if gap <= 4 else 4.45
        if same_raw and trusted_visual:
            min_score -= 0.35
        if bootstrap_raw:
            min_score -= 0.9
        if state.identity_locked and not trusted_visual:
            min_score += 0.18
        if score < min_score:
            return None
        return score

    def _create_track(
        self,
        player: AnalysisObject,
        frame_index: int,
        raw_id: int,
        appearance_hist: np.ndarray | None,
        jersey_color: tuple[int, int, int] | None,
    ) -> int:
        stable_id = self.next_stable_id
        self.next_stable_id += 1
        jersey_family = self._jersey_family(jersey_color)
        observed_role = self._normalized_role(player.role_name)
        self.tracks[stable_id] = StableTrackState(
            stable_id=stable_id,
            bbox=player.bbox,
            center=self._center(player.bbox),
            foot=self._foot(player.bbox),
            velocity=(0.0, 0.0),
            foot_velocity=(0.0, 0.0),
            last_frame=frame_index,
            raw_ids_seen={raw_id},
            appearance_hist=appearance_hist,
            appearance_gallery=[appearance_hist.copy()] if appearance_hist is not None else [],
            jersey_color=jersey_color,
            jersey_family=jersey_family,
            jersey_family_votes={jersey_family: 1} if jersey_family is not None else {},
            role_name="player",
            role_votes={observed_role: self._role_vote_weight(observed_role)},
            last_observed_role=observed_role,
            bbox_height=self._bbox_height(player.bbox),
            depth_proxy=self._depth_proxy(player.bbox),
            last_reliable_frame=frame_index if appearance_hist is not None else 0,
            reliable_hits=1 if appearance_hist is not None else 0,
            first_frame=frame_index,
            last_raw_id=raw_id,
            detection_confidences=[player.confidence]
            if player.confidence is not None
            else [],
        )
        self.raw_to_stable[raw_id] = stable_id
        return stable_id

    def _update_track(
        self,
        stable_id: int,
        player: AnalysisObject,
        frame_index: int,
        raw_id: int,
        appearance_hist: np.ndarray | None,
        jersey_color: tuple[int, int, int] | None,
        visual_reliable: bool,
        assignment_score: float | None,
    ) -> None:
        state = self.tracks[stable_id]
        new_center = self._center(player.bbox)
        new_foot = self._foot(player.bbox)
        new_depth = self._depth_proxy(player.bbox)
        frame_delta = max(frame_index - state.last_frame, 1)
        motion_gate = self._hard_motion_gate(state, player.bbox, frame_delta)
        state.max_motion_gate_ratio = max(
            state.max_motion_gate_ratio,
            self._center_distance(new_foot, state.foot) / max(motion_gate, 1.0),
        )
        if frame_delta > self.long_gap_frames and self.raw_to_stable.get(raw_id) == stable_id:
            state.native_reid_reentries += 1
        state.max_observation_gap = max(state.max_observation_gap, frame_delta)
        if frame_delta > 2:
            state.fragments += 1
        if state.last_raw_id is not None and raw_id != state.last_raw_id:
            state.raw_id_transitions += 1
        state.last_raw_id = raw_id
        if assignment_score is not None:
            state.assignment_scores.append(float(assignment_score))
            if len(state.assignment_scores) > 240:
                state.assignment_scores.pop(0)
        if player.confidence is not None:
            state.detection_confidences.append(float(player.confidence))
            if len(state.detection_confidences) > 240:
                state.detection_confidences.pop(0)
        instant_velocity = (
            (new_center[0] - state.center[0]) / frame_delta,
            (new_center[1] - state.center[1]) / frame_delta,
        )
        instant_foot_velocity = (
            (new_foot[0] - state.foot[0]) / frame_delta,
            (new_foot[1] - state.foot[1]) / frame_delta,
        )
        instant_depth_velocity = (new_depth - state.depth_proxy) / frame_delta
        position_weight = 0.16 if not visual_reliable else 0.30
        foot_weight = 0.14 if not visual_reliable else 0.28
        depth_weight = 0.12 if not visual_reliable else 0.24
        state.velocity = (
            state.velocity[0] * (1.0 - position_weight) + instant_velocity[0] * position_weight,
            state.velocity[1] * (1.0 - position_weight) + instant_velocity[1] * position_weight,
        )
        state.foot_velocity = (
            state.foot_velocity[0] * (1.0 - foot_weight) + instant_foot_velocity[0] * foot_weight,
            state.foot_velocity[1] * (1.0 - foot_weight) + instant_foot_velocity[1] * foot_weight,
        )
        state.depth_velocity = state.depth_velocity * (1.0 - depth_weight) + instant_depth_velocity * depth_weight
        state.bbox = player.bbox
        state.center = new_center
        state.foot = new_foot
        state.depth_proxy = state.depth_proxy * 0.76 + new_depth * 0.24
        state.last_frame = frame_index
        state.raw_ids_seen.add(raw_id)
        state.bbox_height = state.bbox_height * 0.82 + self._bbox_height(player.bbox) * 0.18
        state.hits += 1
        state.consecutive_hits = state.consecutive_hits + 1 if frame_delta <= 2 else 1
        self._update_role_state(state, self._normalized_role(player.role_name))
        if not visual_reliable:
            state.occlusion_hits += 1
        identity_visual_update = visual_reliable
        if state.identity_locked and appearance_hist is not None:
            appearance_to_identity = self._state_appearance_similarity(appearance_hist, state)
            color_to_identity = self._color_similarity(jersey_color, state.jersey_color)
            observed_family = self._jersey_family(jersey_color)
            if (
                appearance_to_identity < 0.74
                or color_to_identity < 0.52
                or self._color_family_mismatch(state.jersey_family, observed_family)
            ):
                identity_visual_update = False
                self.frozen_identity_visual_updates += 1
        if identity_visual_update and appearance_hist is not None:
            state.last_reliable_frame = frame_index
            state.reliable_hits += 1
            if state.appearance_hist is None:
                state.appearance_hist = appearance_hist
            else:
                state.appearance_hist = self._normalize_hist(state.appearance_hist * 0.88 + appearance_hist * 0.12)
            if not state.appearance_gallery or self._appearance_similarity(
                appearance_hist,
                state.appearance_gallery[-1],
            ) < 0.985:
                state.appearance_gallery.append(appearance_hist.copy())
                if len(state.appearance_gallery) > 12:
                    state.appearance_gallery.pop(0)
        if identity_visual_update and jersey_color is not None:
            observed_family = self._jersey_family(jersey_color)
            family_confidence = self._jersey_family_confidence(state)
            accept_color = (
                state.jersey_color is None
                or state.jersey_family is None
                or observed_family == state.jersey_family
                or family_confidence < 0.55
            )
            if state.jersey_color is None:
                state.jersey_color = jersey_color
            elif accept_color:
                state.jersey_color = tuple(
                    int(round(state.jersey_color[index] * 0.85 + jersey_color[index] * 0.15))
                    for index in range(3)
                )
            if observed_family is not None:
                state.jersey_family_votes[observed_family] = state.jersey_family_votes.get(observed_family, 0) + 1
                state.jersey_family = max(
                    state.jersey_family_votes,
                    key=state.jersey_family_votes.get,
                )
        state.identity_locked = state.identity_locked or (
            state.hits >= 10
            and state.reliable_hits >= 8
            and state.jersey_family is not None
            and self._jersey_family_confidence(state) >= 0.55
        )
        state.confirmed = state.confirmed or (
            state.hits >= self.confirmation_hits
            and state.consecutive_hits >= min(3, self.confirmation_hits)
            and state.reliable_hits >= 2
        )
        self.raw_to_stable[raw_id] = stable_id

    def _normalized_role(self, role_name: str | None) -> str:
        normalized = str(role_name or "player").lower()
        if normalized in GOALKEEPER_ALIASES:
            return "goalkeeper"
        if normalized in ASSISTANT_REFEREE_ALIASES:
            return "assistant_referee"
        if normalized in STAFF_ALIASES or normalized == "staff_outside_pitch":
            return "staff_outside_pitch"
        if normalized in REFEREE_ALIASES:
            return "referee"
        return "player"

    def _role_vote_weight(self, role_name: str) -> float:
        if role_name in {"referee", "assistant_referee"}:
            return 0.70
        if role_name == "staff_outside_pitch":
            return 0.55
        if role_name == "goalkeeper":
            return 0.85
        return 1.0

    def _update_role_state(self, state: StableTrackState, observed_role: str) -> None:
        if observed_role == state.last_observed_role:
            state.role_streak += 1
        else:
            state.last_observed_role = observed_role
            state.role_streak = 1
        state.role_votes[observed_role] = (
            state.role_votes.get(observed_role, 0.0)
            + self._role_vote_weight(observed_role)
        )
        if state.role_locked:
            return

        total = sum(state.role_votes.values())
        best_role = max(state.role_votes, key=state.role_votes.get)
        best_confidence = state.role_votes[best_role] / max(total, 1e-6)
        required_streak = 8 if best_role != "player" else 4
        if state.last_observed_role == best_role and state.role_streak >= required_streak:
            if best_confidence >= (0.74 if best_role != "player" else 0.58):
                state.role_name = best_role
        if state.hits >= 16 and best_confidence >= 0.84:
            state.role_name = best_role
            state.role_locked = True

    def _is_compatible(
        self,
        state: StableTrackState,
        player: AnalysisObject,
        frame_index: int,
        appearance_hist: np.ndarray | None,
        jersey_color: tuple[int, int, int] | None,
    ) -> bool:
        gap = max(frame_index - state.last_frame, 1)
        distance = self._center_distance(self._center(player.bbox), self._predicted_center(state, gap))
        iou = self._iou(player.bbox, state.bbox)
        appearance = self._appearance_similarity(appearance_hist, state.appearance_hist)
        color_similarity = self._color_similarity(jersey_color, state.jersey_color)
        max_distance = self._max_center_distance(player.bbox, state.bbox, gap)
        if self._is_locked_jersey_mismatch(state, jersey_color, appearance, color_similarity):
            return False
        if state.hits >= 5 and appearance_hist is not None and state.appearance_hist is not None:
            if appearance < 0.38 and color_similarity < 0.62:
                self.rejected_appearance_mismatches += 1
                return False
        if iou <= 0.02 and distance > max_distance * 2.6:
            return False
        return (
            distance <= max_distance
            or iou > 0.05
            or (distance <= max_distance * 2.0 and appearance >= 0.80 and color_similarity >= 0.72)
        )

    def _raw_id_identity_mismatch(
        self,
        state: StableTrackState,
        player: AnalysisObject,
        frame_index: int,
        appearance_hist: np.ndarray | None,
        jersey_color: tuple[int, int, int] | None,
    ) -> bool:
        if state.hits < 4:
            return False
        gap = max(frame_index - state.last_frame, 1)
        distance = self._center_distance(self._center(player.bbox), self._predicted_center(state, gap))
        max_distance = self._max_center_distance(player.bbox, state.bbox, gap)
        iou = self._iou(player.bbox, state.bbox)
        appearance = self._appearance_similarity(appearance_hist, state.appearance_hist)
        color_similarity = self._color_similarity(jersey_color, state.jersey_color)
        if self._is_locked_jersey_mismatch(state, jersey_color, appearance, color_similarity):
            return True
        if state.appearance_hist is not None and appearance_hist is not None and appearance < 0.30 and color_similarity < 0.68:
            return True
        if distance > max_distance * 1.4 and iou < 0.08 and appearance < 0.55 and color_similarity < 0.72:
            return True
        return False

    def _raw_id_motion_conflict(
        self,
        state: StableTrackState,
        player: AnalysisObject,
        frame_index: int,
    ) -> bool:
        """Ignore a native ID when another established trajectory owns the detection.

        Native trackers can preserve the numeric ID while assigning it to the other
        participant after an overlap.  The stable layer therefore treats raw IDs as
        evidence, not truth, whenever the candidate is substantially closer to a
        competing predicted foot position.
        """
        if state.hits < 3:
            return False
        gap = frame_index - state.last_frame
        if gap < 0 or gap > self.hidden_hold_frames:
            return False

        candidate_foot = self._foot(player.bbox)
        own_prediction = self._predicted_foot(state, max(gap, 1))
        own_distance = self._center_distance(candidate_foot, own_prediction)
        separation_margin = max(12.0, self._bbox_height(player.bbox) * 0.20)
        if own_distance <= separation_margin:
            return False

        for other_state in self.tracks.values():
            if other_state.stable_id == state.stable_id or not other_state.confirmed:
                continue
            other_gap = frame_index - other_state.last_frame
            if other_gap < 0 or other_gap > self.hidden_hold_frames:
                continue
            other_prediction = self._predicted_foot(other_state, max(other_gap, 1))
            other_distance = self._center_distance(candidate_foot, other_prediction)
            if other_distance + separation_margin < own_distance:
                return True
        return False

    def _max_center_distance(self, bbox_a: list[float], bbox_b: list[float], gap: int) -> float:
        size = max(self._bbox_size(bbox_a), self._bbox_size(bbox_b), 1.0)
        return max(42.0, min(240.0, size * 1.10 + min(gap, 30) * 4.0))

    def _hard_motion_gate(
        self,
        state: StableTrackState,
        candidate_bbox: list[float],
        gap: int,
    ) -> float:
        """Maximum plausible image-plane foot displacement before identity is rejected.

        Appearance can distinguish teams, but it must never override impossible motion.
        A camera cut may therefore start new identities; that is safer than assigning an
        existing identity to a different person and corrupting every downstream metric.
        """
        height = max(state.bbox_height, self._bbox_height(candidate_bbox), 1.0)
        if gap <= 2:
            gate = height * 0.62 + gap * 12.0
        elif gap <= 12:
            gate = height * 0.78 + gap * 8.0
        elif gap <= self.long_gap_frames:
            gate = height * 1.05 + gap * 5.0
        else:
            gate = height * 1.35 + min(gap, self.max_gap_frames) * 3.0
        return max(28.0, min(260.0, gate))

    def _is_locked_jersey_mismatch(
        self,
        state: StableTrackState,
        jersey_color: tuple[int, int, int] | None,
        appearance: float,
        color_similarity: float,
    ) -> bool:
        if (
            not state.identity_locked
            or self._jersey_family_confidence(state) < 0.65
            or state.jersey_color is None
            or jersey_color is None
        ):
            return False
        state_hsv = self._bgr_to_hsv(state.jersey_color)
        candidate_hsv = self._bgr_to_hsv(jersey_color)
        hue_gap = self._hue_gap(state_hsv[0], candidate_hsv[0])
        both_colored = state_hsv[1] >= 45 and candidate_hsv[1] >= 45
        if both_colored and hue_gap >= 25 and color_similarity < 0.54 and appearance < 0.62:
            return True
        if both_colored and hue_gap >= 18 and color_similarity < 0.65 and appearance < 0.66:
            return True
        return color_similarity < 0.36 and appearance < 0.50

    def _is_strong_color_conflict(
        self,
        state: StableTrackState,
        jersey_color: tuple[int, int, int] | None,
        appearance: float,
        color_similarity: float,
        visual_reliable: bool,
    ) -> bool:
        if (
            not visual_reliable
            or state.reliable_hits < 1
            or state.jersey_color is None
            or jersey_color is None
        ):
            return False
        state_hsv = self._bgr_to_hsv(state.jersey_color)
        candidate_hsv = self._bgr_to_hsv(jersey_color)
        both_colored = state_hsv[1] >= 72 and candidate_hsv[1] >= 72
        hue_gap = self._hue_gap(state_hsv[0], candidate_hsv[0])
        return (
            both_colored
            and color_similarity < 0.62
            and (
                hue_gap >= 32
                or (hue_gap >= 24 and appearance < 0.55)
            )
        )

    def _predicted_center(self, state: StableTrackState, gap: int) -> tuple[float, float]:
        capped_gap = min(gap, 45)
        return (
            state.center[0] + state.velocity[0] * capped_gap,
            state.center[1] + state.velocity[1] * capped_gap,
        )

    def _predicted_foot(self, state: StableTrackState, gap: int) -> tuple[float, float]:
        capped_gap = min(gap, 45)
        return (
            state.foot[0] + state.foot_velocity[0] * capped_gap,
            state.foot[1] + state.foot_velocity[1] * capped_gap,
        )

    def _extract_appearance(
        self,
        frame: np.ndarray | None,
        bbox: list[float],
    ) -> tuple[np.ndarray | None, tuple[int, int, int] | None]:
        if frame is None:
            return None, None
        height, width = frame.shape[:2]
        x1, y1, x2, y2 = [int(round(value)) for value in bbox]
        x1, x2 = max(0, x1), min(width, x2)
        y1, y2 = max(0, y1), min(height, y2)
        if x2 <= x1 or y2 <= y1:
            return None, None
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return None, None
        person = cv2.resize(crop, (32, 64), interpolation=cv2.INTER_AREA)
        torso = person[8:34, 9:23]
        if torso.size == 0:
            return None, None

        region_descriptors = [
            self._region_hist(person[1:19, 8:24]),
            self._region_hist(torso),
            self._region_hist(person[34:61, 6:26]),
        ]
        lab_torso = self._region_lab_hist(torso)
        gray = cv2.cvtColor(person, cv2.COLOR_BGR2GRAY)
        gradients = self._gradient_descriptor(gray)
        appearance = self._normalize_hist(
            np.concatenate(
                [
                    region_descriptors[0] * 0.85,
                    region_descriptors[1] * 1.35,
                    region_descriptors[2] * 1.0,
                    lab_torso * 1.15,
                    gradients * 0.55,
                ]
            ).astype(np.float32)
        )
        inner_torso = torso[2:-2, 2:-2] if torso.shape[0] > 6 and torso.shape[1] > 6 else torso
        hsv = cv2.cvtColor(inner_torso, cv2.COLOR_BGR2HSV)
        valid_mask = hsv[:, :, 2] >= 32
        pixels = inner_torso[valid_mask].reshape(-1, 3).astype(np.float32)
        jersey_color = None
        if len(pixels) >= 6:
            jersey_color = tuple(int(value) for value in np.median(pixels, axis=0))
        return appearance, jersey_color

    def _region_hist(self, region: np.ndarray) -> np.ndarray:
        if region.size == 0:
            return np.zeros(128, dtype=np.float32)
        hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, [16, 8], [0, 180, 0, 256]).astype(np.float32).flatten()
        return self._normalize_hist(hist)

    def _region_lab_hist(self, region: np.ndarray) -> np.ndarray:
        if region.size == 0:
            return np.zeros(64, dtype=np.float32)
        lab = cv2.cvtColor(region, cv2.COLOR_BGR2LAB)
        hist = cv2.calcHist([lab], [1, 2], None, [8, 8], [0, 256, 0, 256]).astype(np.float32).flatten()
        return self._normalize_hist(hist)

    def _gradient_descriptor(self, gray: np.ndarray) -> np.ndarray:
        normalized = gray.astype(np.float32) / 255.0
        gradient_y, gradient_x = np.gradient(normalized)
        magnitude = np.hypot(gradient_x, gradient_y)
        orientation = (np.arctan2(gradient_y, gradient_x) + np.pi) * (8.0 / (2.0 * np.pi))
        descriptor: list[np.ndarray] = []
        for row in range(4):
            for column in range(2):
                y1, y2 = row * 16, (row + 1) * 16
                x1, x2 = column * 16, (column + 1) * 16
                bins = np.floor(orientation[y1:y2, x1:x2]).astype(np.int32) % 8
                weights = magnitude[y1:y2, x1:x2]
                descriptor.append(np.bincount(bins.ravel(), weights=weights.ravel(), minlength=8).astype(np.float32))
        return self._normalize_hist(np.concatenate(descriptor))

    def _normalize_hist(self, hist: np.ndarray) -> np.ndarray:
        total = float(np.linalg.norm(hist))
        if total <= 1e-6:
            return hist
        return hist / total

    def _appearance_similarity(self, a: np.ndarray | None, b: np.ndarray | None) -> float:
        if a is None or b is None:
            return 0.0
        return float(max(0.0, min(1.0, np.dot(a, b))))

    def _state_appearance_similarity(
        self,
        candidate: np.ndarray | None,
        state: StableTrackState,
    ) -> float:
        if candidate is None:
            return 0.0
        aggregate = self._appearance_similarity(candidate, state.appearance_hist)
        if not state.appearance_gallery:
            return aggregate
        gallery_scores = sorted(
            (
            self._appearance_similarity(candidate, reference)
            for reference in state.appearance_gallery
            ),
            reverse=True,
        )
        robust_count = min(3, len(gallery_scores))
        robust_gallery = float(np.mean(gallery_scores[:robust_count]))
        return float(np.clip(aggregate * 0.62 + robust_gallery * 0.38, 0.0, 1.0))

    def _color_similarity(
        self,
        a: tuple[int, int, int] | None,
        b: tuple[int, int, int] | None,
    ) -> float:
        if a is None or b is None:
            return 0.0
        distance = float(np.linalg.norm(np.array(a, dtype=np.float32) - np.array(b, dtype=np.float32)))
        return max(0.0, 1.0 - distance / 441.672)

    def _bgr_to_hsv(self, color: tuple[int, int, int]) -> tuple[int, int, int]:
        pixel = np.array([[list(color)]], dtype=np.uint8)
        hsv = cv2.cvtColor(pixel, cv2.COLOR_BGR2HSV)[0][0]
        return int(hsv[0]), int(hsv[1]), int(hsv[2])

    def _hue_gap(self, hue_a: int, hue_b: int) -> int:
        diff = abs(hue_a - hue_b)
        return min(diff, 180 - diff)

    def _jersey_family(self, color: tuple[int, int, int] | None) -> str | None:
        if color is None:
            return None
        hue, saturation, value = self._bgr_to_hsv(color)
        if value < 48:
            return "black"
        if saturation < 36:
            if value >= 188:
                return "white"
            return "gray"
        if hue < 8 or hue >= 172:
            return "red"
        if hue < 18:
            return "orange"
        if hue < 36:
            return "yellow"
        if hue < 82:
            return "green"
        if hue < 100:
            return "cyan"
        if hue < 132:
            return "blue"
        if hue < 158:
            return "purple"
        return "magenta"

    def _color_family_mismatch(self, state_family: str | None, candidate_family: str | None) -> bool:
        if state_family is None or candidate_family is None or state_family == candidate_family:
            return False
        compatible_neighbors = {
            frozenset(("black", "gray")),
            frozenset(("gray", "white")),
            frozenset(("red", "orange")),
            frozenset(("orange", "yellow")),
            frozenset(("yellow", "green")),
            frozenset(("green", "cyan")),
            frozenset(("cyan", "blue")),
            frozenset(("blue", "purple")),
            frozenset(("purple", "magenta")),
            frozenset(("magenta", "red")),
        }
        return frozenset((state_family, candidate_family)) not in compatible_neighbors

    def _jersey_family_confidence(self, state: StableTrackState) -> float:
        total = sum(state.jersey_family_votes.values())
        if total <= 0 or state.jersey_family is None:
            return 0.0
        return state.jersey_family_votes.get(state.jersey_family, 0) / total

    def _direction_similarity(
        self,
        state: StableTrackState,
        candidate_foot: tuple[float, float],
        gap: int,
    ) -> float:
        previous_velocity = np.array(state.foot_velocity, dtype=np.float32)
        candidate_velocity = np.array(
            (
                (candidate_foot[0] - state.foot[0]) / max(gap, 1),
                (candidate_foot[1] - state.foot[1]) / max(gap, 1),
            ),
            dtype=np.float32,
        )
        previous_speed = float(np.linalg.norm(previous_velocity))
        candidate_speed = float(np.linalg.norm(candidate_velocity))
        if previous_speed < 0.35 or candidate_speed < 0.35:
            return 0.0
        return float(
            np.clip(
                np.dot(previous_velocity, candidate_velocity) / (previous_speed * candidate_speed),
                -1.0,
                1.0,
            )
        )

    def _bbox_size(self, bbox: list[float]) -> float:
        return max(bbox[2] - bbox[0], bbox[3] - bbox[1])

    def _bbox_height(self, bbox: list[float]) -> float:
        return max(1.0, bbox[3] - bbox[1])

    def _depth_proxy(self, bbox: list[float]) -> float:
        # A larger on-screen player is usually closer to a monocular broadcast camera.
        return float(np.log(self._bbox_height(bbox)))

    def _identity_confidence(self, state: StableTrackState) -> float:
        history = min(1.0, state.reliable_hits / 12.0)
        appearance = min(1.0, len(state.appearance_gallery) / 5.0)
        occlusion_ratio = state.occlusion_hits / max(state.hits, 1)
        continuity = max(0.0, 1.0 - min(occlusion_ratio, 0.65))
        lock_bonus = 0.12 if state.identity_locked else 0.0
        fragment_penalty = min(0.22, max(0, state.fragments - 1) * 0.055)
        transition_penalty = min(0.16, state.raw_id_transitions * 0.018)
        motion_penalty = max(0.0, state.max_motion_gate_ratio - 0.68) * 0.34
        return float(
            np.clip(
                history * 0.42
                + appearance * 0.28
                + continuity * 0.30
                + lock_bonus
                - fragment_penalty
                - transition_penalty
                - motion_penalty,
                0.0,
                1.0,
            )
        )

    def quality_report(
        self,
        track_frames: dict[int, int],
        team_by_track: dict[int, int],
        review_observations: dict[int, list[dict[str, Any]]],
        crop_files: dict[int, list[dict[str, Any]]],
        tracker_runtime: dict[str, Any],
    ) -> dict[str, Any]:
        tracks: list[dict[str, Any]] = []
        for track_id in sorted(track_frames):
            state = self.tracks.get(track_id)
            if state is None or not state.confirmed:
                continue
            identity_confidence = self._identity_confidence(state)
            appearance_consistency = self._appearance_consistency(state)
            gallery_coverage = min(1.0, len(state.appearance_gallery) / 6.0)
            motion_consistency = self._motion_consistency(state)
            transition_consistency = max(
                0.0,
                1.0 - state.raw_id_transitions / max(state.hits * 0.08, 1.0),
            )
            reid_confidence = (
                appearance_consistency * 0.40
                + gallery_coverage * 0.12
                + motion_consistency * 0.30
                + transition_consistency * 0.18
            )
            team_consistency = self._jersey_family_confidence(state)
            fragment_count = max(0, state.fragments - 1)
            issues: list[str] = []
            if identity_confidence < QUALITY_THRESHOLDS["review_identity_confidence"]:
                issues.append("low_identity_confidence")
            if fragment_count > 0:
                issues.append("fragmented_track")
            if state.raw_id_transitions >= QUALITY_THRESHOLDS["high_risk_raw_id_transitions"]:
                issues.append("frequent_raw_id_transitions")
            if team_consistency < 0.62:
                issues.append("team_color_unstable")
            if reid_confidence < 0.58:
                issues.append("appearance_inconsistent")
            if state.max_motion_gate_ratio >= 0.82:
                issues.append("near_motion_gate")
            if track_frames.get(track_id, 0) < self.confirmation_hits * 3:
                issues.append("short_track")

            high_risk = (
                identity_confidence < QUALITY_THRESHOLDS["high_risk_identity_confidence"]
                or fragment_count >= QUALITY_THRESHOLDS["high_risk_fragments"]
                or state.raw_id_transitions >= QUALITY_THRESHOLDS["high_risk_raw_id_transitions"]
            )
            medium_risk = (
                identity_confidence < QUALITY_THRESHOLDS["approve_identity_confidence"]
                or fragment_count > 0
                or reid_confidence < 0.72
                or team_consistency < 0.72
            )
            switch_risk = "high" if high_risk else "medium" if medium_risk else "low"
            observations = review_observations.get(track_id, [])
            tracks.append(
                {
                    "track_id": track_id,
                    "team": team_by_track.get(track_id),
                    "identity_confidence": round(identity_confidence, 4),
                    "reid_confidence": round(float(np.clip(reid_confidence, 0.0, 1.0)), 4),
                    "motion_consistency": round(motion_consistency, 4),
                    "team_consistency": round(team_consistency, 4),
                    "switch_risk": switch_risk,
                    "fragment_count": fragment_count,
                    "raw_id_transitions": state.raw_id_transitions,
                    "first_frame": state.first_frame,
                    "last_frame": state.last_frame,
                    "observation_count": track_frames.get(track_id, 0),
                    "raw_track_ids": sorted(state.raw_ids_seen),
                    "max_motion_gate_ratio": round(state.max_motion_gate_ratio, 4),
                    "native_reid_reentries": state.native_reid_reentries,
                    "role_locked": state.role_locked,
                    "issue_codes": issues,
                    "crop_files": crop_files.get(track_id, []),
                    "review_observations": observations,
                }
            )

        confidences = [float(track["identity_confidence"]) for track in tracks]
        review_tracks = [track for track in tracks if track["switch_risk"] != "low"]
        high_risk_tracks = [track for track in tracks if track["switch_risk"] == "high"]
        fragmented_tracks = [track for track in tracks if int(track["fragment_count"]) > 0]
        motion_gate_ratios = [float(track["max_motion_gate_ratio"]) for track in tracks]
        return {
            "engine": "tracking_quality_gate_v2_ground_truth",
            "tracker_runtime": tracker_runtime,
            "overview": {
                "status": "needs_review" if review_tracks else "quality_check_passed",
                "tracks_evaluated": len(tracks),
                "average_identity_confidence": round(float(np.mean(confidences)), 4)
                if confidences
                else None,
                "suspected_id_switches": len(high_risk_tracks),
                "fragmented_tracks": len(fragmented_tracks),
                "tracks_needing_review": len(review_tracks),
                "low_risk_tracks": len(tracks) - len(review_tracks),
                "max_accepted_motion_gate_ratio": round(max(motion_gate_ratios, default=0.0), 4),
                "tracks_near_motion_gate": sum(1 for ratio in motion_gate_ratios if ratio >= 0.82),
                "tracks_over_motion_gate": sum(1 for ratio in motion_gate_ratios if ratio > 1.0),
            },
            "benchmark": {
                "status": "ground_truth_required",
                "id_switches": None,
                "idf1": None,
                "hota": None,
                "fragmentation": None,
                "message": "Upload frame-level ground truth to measure IDF1, HOTA, exact ID switches, and fragmentation.",
            },
            "thresholds": QUALITY_THRESHOLDS,
            "tracks": tracks,
        }

    def _appearance_consistency(self, state: StableTrackState) -> float:
        if not state.appearance_gallery or state.appearance_hist is None:
            return 0.0
        similarities = [
            self._appearance_similarity(reference, state.appearance_hist)
            for reference in state.appearance_gallery
        ]
        return float(np.clip(np.mean(similarities), 0.0, 1.0))

    def _motion_consistency(self, state: StableTrackState) -> float:
        if state.assignment_scores:
            score_quality = float(
                np.clip((np.mean(state.assignment_scores) - 3.5) / 3.5, 0.0, 1.0)
            )
        else:
            score_quality = 0.45
        gap_quality = max(0.0, 1.0 - min(state.max_observation_gap, 30) / 30.0)
        fragment_quality = max(0.0, 1.0 - max(0, state.fragments - 1) * 0.16)
        gate_quality = max(0.0, 1.0 - max(0.0, state.max_motion_gate_ratio - 0.45) / 0.55)
        return float(
            np.clip(
                score_quality * 0.36
                + gap_quality * 0.20
                + fragment_quality * 0.20
                + gate_quality * 0.24,
                0.0,
                1.0,
            )
        )

    def _center(self, bbox: list[float]) -> tuple[float, float]:
        return ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)

    def _foot(self, bbox: list[float]) -> tuple[float, float]:
        return ((bbox[0] + bbox[2]) / 2, bbox[3])

    def _center_distance(self, a: tuple[float, float], b: tuple[float, float]) -> float:
        return float(np.hypot(a[0] - b[0], a[1] - b[1]))

    def _iou(self, a: list[float], b: list[float]) -> float:
        x1 = max(a[0], b[0])
        y1 = max(a[1], b[1])
        x2 = min(a[2], b[2])
        y2 = min(a[3], b[3])
        intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
        area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
        union = area_a + area_b - intersection
        return intersection / union if union > 0 else 0.0

    def _is_crowded_detection(self, player: AnalysisObject, players: list[AnalysisObject]) -> bool:
        for other in players:
            if other is player:
                continue
            if self._iou(player.bbox, other.bbox) >= 0.18:
                return True
            if self._center_distance(self._foot(player.bbox), self._foot(other.bbox)) <= max(
                18.0,
                min(self._bbox_height(player.bbox), self._bbox_height(other.bbox)) * 0.28,
            ):
                return True
        return False

    def _is_severe_overlap_detection(
        self,
        player: AnalysisObject,
        players: list[AnalysisObject],
    ) -> bool:
        for other in players:
            if other is player:
                continue
            overlap = self._iou(player.bbox, other.bbox)
            foot_distance = self._center_distance(
                self._foot(player.bbox),
                self._foot(other.bbox),
            )
            shared_height = min(
                self._bbox_height(player.bbox),
                self._bbox_height(other.bbox),
            )
            if overlap >= 0.34:
                return True
            if overlap >= 0.08 and foot_distance <= max(12.0, shared_height * 0.17):
                return True
        return False

    def _is_prediction_ambiguous(
        self,
        player: AnalysisObject,
        players: list[AnalysisObject],
        frame_index: int,
    ) -> bool:
        candidate_foot = self._foot(player.bbox)
        nearby_predictions = 0
        largest_gate = 55.0
        for state in self.tracks.values():
            if state.hits < 2 and state.reliable_hits < 1:
                continue
            gap = frame_index - state.last_frame
            if gap < 0 or gap > min(self.max_gap_frames, 18):
                continue
            predicted_foot = self._predicted_foot(state, max(gap, 1))
            gate = max(
                55.0,
                min(220.0, max(state.bbox_height, self._bbox_height(player.bbox)) * 0.55 + gap * 4.0),
            )
            largest_gate = max(largest_gate, gate)
            if self._center_distance(candidate_foot, predicted_foot) <= gate:
                nearby_predictions += 1
        if nearby_predictions < 2:
            return False
        nearby_detections = sum(
            1
            for other in players
            if self._center_distance(candidate_foot, self._foot(other.bbox)) <= largest_gate
        )
        return nearby_predictions > nearby_detections

class TeamColorClassifier:
    """Resolve stable team identity from kit references and temporal appearance."""

    def __init__(
        self,
        reference_palettes_bgr: dict[int, list[tuple[int, int, int]]] | None = None,
        goalkeeper_reference_palettes_bgr: dict[int, list[tuple[int, int, int]]] | None = None,
        team_labels: dict[int, str] | None = None,
    ) -> None:
        self.reference_palettes_bgr = {
            int(team): list(colors)
            for team, colors in (reference_palettes_bgr or {}).items()
            if colors
        }
        self.reference_seeded_teams = set(self.reference_palettes_bgr)
        self.goalkeeper_reference_palettes_bgr = {
            int(team): list(colors)
            for team, colors in (goalkeeper_reference_palettes_bgr or {}).items()
            if colors
        }
        self.team_labels = {
            int(team): str(label)
            for team, label in (team_labels or {}).items()
            if label
        }
        self.anchors: dict[int, tuple[int, int, int]] = {
            team: colors[0]
            for team, colors in self.reference_palettes_bgr.items()
        }
        self.appearance_anchors: dict[int, np.ndarray] = {}
        self.track_votes: dict[int, dict[int, float]] = {}
        self.track_observation_counts: dict[int, int] = {}
        self.locked_team_by_track: dict[int, int] = {}
        self.team_assignments: dict[int, int] = {}
        self.track_confidence: dict[int, float] = {}
        self.observations = 0
        self.anchor_initializations = 0
        self.ambiguous_observations = 0
        self.official_tracks: set[int] = set()
        self.goalkeeper_tracks: set[int] = set()
        self.goalkeeper_reference_matches = 0
        self.assignment_sources: dict[int, str] = {}
        self.prevented_team_switches = 0

    def update(
        self,
        players: list[AnalysisObject],
        track_states: dict[int, StableTrackState],
        team_by_track: dict[int, int],
    ) -> None:
        samples: dict[int, tuple[tuple[int, int, int], np.ndarray | None]] = {}
        outfield_track_ids: set[int] = set()
        for player in players:
            state = track_states.get(player.track_id)
            if state is None or state.jersey_color is None:
                continue
            role_name = str(getattr(state, "role_name", player.role_name or "player"))
            if role_name in {"referee", "assistant_referee", "staff_outside_pitch"}:
                team_by_track[player.track_id] = 0
                self.team_assignments[player.track_id] = 0
                self.track_confidence[player.track_id] = 1.0
                self.official_tracks.add(player.track_id)
                self.goalkeeper_tracks.discard(player.track_id)
                continue
            self.official_tracks.discard(player.track_id)
            if role_name == "goalkeeper":
                self.goalkeeper_tracks.add(player.track_id)
            else:
                self.goalkeeper_tracks.discard(player.track_id)
                outfield_track_ids.add(player.track_id)
            samples[player.track_id] = (
                state.jersey_color,
                getattr(state, "appearance_hist", None),
            )
            self.observations += 1

        if not samples:
            return
        anchor_samples = [
            samples[track_id]
            for track_id in outfield_track_ids
            if track_id in samples
        ]
        if not anchor_samples:
            anchor_samples = list(samples.values())
        created_second_anchor = self._ensure_anchors(anchor_samples)
        if created_second_anchor:
            self.track_votes.clear()
            self.track_observation_counts.clear()
            if not self.reference_seeded_teams:
                self.locked_team_by_track.clear()

        assignments: dict[int, int] = {}
        for track_id, (color, appearance) in samples.items():
            role_name = str(getattr(track_states.get(track_id), "role_name", "player"))
            team, confidence, source = self._nearest_team(
                color,
                appearance,
                role_name=role_name,
            )
            assignments[track_id] = team
            self.assignment_sources[track_id] = source
            if source == "goalkeeper_kit_reference":
                self.goalkeeper_reference_matches += 1
            votes = self.track_votes.setdefault(track_id, {1: 0.0, 2: 0.0})
            self.track_observation_counts[track_id] = (
                self.track_observation_counts.get(track_id, 0) + 1
            )
            votes[1] *= 0.90
            votes[2] *= 0.90
            votes[team] += 0.50 + confidence
            proposed_team = max(votes, key=votes.get)
            total_votes = max(votes[1] + votes[2], 1e-6)
            vote_confidence = abs(votes[1] - votes[2]) / total_votes
            locked_team = self.locked_team_by_track.get(track_id)
            if locked_team is not None:
                if proposed_team != locked_team:
                    self.prevented_team_switches += 1
                team_by_track[track_id] = locked_team
            else:
                previous_team = team_by_track.get(track_id)
                if previous_team in {1, 2} and proposed_team != previous_team:
                    challenger = votes[proposed_team]
                    incumbent = votes[previous_team]
                    if challenger < incumbent * 1.35 + 0.75:
                        proposed_team = previous_team
                        self.prevented_team_switches += 1
                team_by_track[track_id] = proposed_team
                required_observations = (
                    6
                    if proposed_team in self.reference_seeded_teams
                    else 12
                )
                if (
                    self.track_observation_counts[track_id] >= required_observations
                    and vote_confidence >= 0.68
                ):
                    self.locked_team_by_track[track_id] = proposed_team
            self.team_assignments[track_id] = team_by_track[track_id]
            self.track_confidence[track_id] = round(
                min(1.0, confidence * 0.55 + vote_confidence * 0.45),
                4,
            )
            if confidence < 0.58:
                self.ambiguous_observations += 1

        if len(self.anchors) == 2:
            self._update_anchors(
                samples,
                assignments,
                outfield_track_ids,
            )

    def summary(self) -> dict[str, Any]:
        quality_gate = self.quality_gate()
        return {
            "engine": "team_identity_v3_shadow_robust",
            "kit_anchors_bgr": {
                str(team): list(color)
                for team, color in sorted(self.anchors.items())
            },
            "reference_palettes_bgr": {
                str(team): [list(color) for color in colors]
                for team, colors in sorted(self.reference_palettes_bgr.items())
            },
            "reference_seeded_teams": sorted(self.reference_seeded_teams),
            "goalkeeper_reference_palettes_bgr": {
                str(team): [list(color) for color in colors]
                for team, colors in sorted(self.goalkeeper_reference_palettes_bgr.items())
            },
            "reference_source": (
                "stored_kit_images"
                if self.reference_seeded_teams
                else "online_appearance_clustering"
            ),
            "team_labels": {
                str(team): label
                for team, label in sorted(self.team_labels.items())
            },
            "appearance_anchor_dimensions": {
                str(team): int(anchor.size)
                for team, anchor in sorted(self.appearance_anchors.items())
            },
            "classified_tracks": len(self.track_votes),
            "track_confidence": {
                str(track_id): confidence
                for track_id, confidence in sorted(self.track_confidence.items())
            },
            "assignment_sources": {
                str(track_id): source
                for track_id, source in sorted(self.assignment_sources.items())
            },
            "locked_team_by_track": {
                str(track_id): team
                for track_id, team in sorted(self.locked_team_by_track.items())
            },
            "locked_team_tracks": len(self.locked_team_by_track),
            "prevented_team_switches": self.prevented_team_switches,
            "team_track_counts": {
                str(team): sum(
                    1 for assigned_team in self.team_assignments.values()
                    if assigned_team == team
                )
                for team in (1, 2)
            },
            "color_observations": self.observations,
            "ambiguous_observations": self.ambiguous_observations,
            "official_tracks": sorted(self.official_tracks),
            "goalkeeper_tracks": sorted(self.goalkeeper_tracks),
            "goalkeeper_reference_matches": self.goalkeeper_reference_matches,
            "anchor_initializations": self.anchor_initializations,
            "quality_gate": quality_gate,
            "manual_correction_available": True,
            "roster_linkage": "prepared_for_manual_assignment_or_jersey_ocr",
        }

    def quality_gate(self) -> dict[str, Any]:
        confidences = [
            value
            for track_id, value in self.track_confidence.items()
            if track_id not in self.official_tracks
        ]
        average_confidence = float(np.mean(confidences)) if confidences else 0.0
        ambiguous_ratio = self.ambiguous_observations / max(self.observations, 1)
        anchor_separation = None
        if 1 in self.anchors and 2 in self.anchors:
            anchor_separation = self._color_distance(self.anchors[1], self.anchors[2])
        team_track_counts = {
            team: sum(
                1
                for track_id, assigned_team in self.team_assignments.items()
                if track_id not in self.official_tracks and assigned_team == team
            )
            for team in (1, 2)
        }
        classified_team_tracks = sum(team_track_counts.values())
        minimum_team_support = max(
            1,
            min(3, int(np.ceil(classified_team_tracks * 0.10))),
        )
        conditions = [
            {
                "code": "two_team_anchors",
                "passed": len(self.anchors) == 2,
                "value": len(self.anchors),
                "required": 2,
            },
            {
                "code": "average_track_confidence",
                "passed": average_confidence >= 0.72,
                "value": round(average_confidence, 4),
                "required": 0.72,
            },
            {
                "code": "bounded_ambiguity",
                "passed": ambiguous_ratio <= 0.30,
                "value": round(ambiguous_ratio, 4),
                "required": "<= 0.30",
            },
            {
                "code": "kit_separation",
                "passed": anchor_separation is not None and anchor_separation >= 0.12,
                "value": round(anchor_separation, 4) if anchor_separation is not None else None,
                "required": ">= 0.12",
            },
            {
                "code": "both_teams_observed",
                "passed": all(
                    team_track_counts[team] >= minimum_team_support
                    for team in (1, 2)
                ),
                "value": {
                    str(team): team_track_counts[team]
                    for team in (1, 2)
                },
                "required": f">= {minimum_team_support} tracks per team",
            },
        ]
        failed = [condition["code"] for condition in conditions if not condition["passed"]]
        return {
            "status": "passed" if not failed else "needs_review",
            "conditions": conditions,
            "failed_conditions": failed,
            "average_track_confidence": round(average_confidence, 4),
            "ambiguous_observation_ratio": round(ambiguous_ratio, 4),
            "team_track_counts": {
                str(team): team_track_counts[team]
                for team in (1, 2)
            },
            "similar_kits_detected": anchor_separation is not None and anchor_separation < 0.18,
            "shadow_invariant_color_distance": True,
        }

    def _ensure_anchors(
        self,
        samples: list[tuple[tuple[int, int, int], np.ndarray | None]],
    ) -> bool:
        if not self.anchors:
            if len(samples) == 1:
                color, appearance = samples[0]
                self.anchors[1] = color
                if appearance is not None:
                    self.appearance_anchors[1] = appearance.copy()
                self.anchor_initializations += 1
                return False
            best_pair: tuple[
                tuple[tuple[int, int, int], np.ndarray | None],
                tuple[tuple[int, int, int], np.ndarray | None],
            ] | None = None
            best_distance = -1.0
            for first_index, first in enumerate(samples):
                for second in samples[first_index + 1 :]:
                    distance = self._sample_distance(first, second)
                    if distance > best_distance:
                        best_distance = distance
                        best_pair = (first, second)
            if best_pair is not None and best_distance >= 0.30:
                self.anchors = {
                    1: best_pair[0][0],
                    2: best_pair[1][0],
                }
                for team, sample in ((1, best_pair[0]), (2, best_pair[1])):
                    if sample[1] is not None:
                        self.appearance_anchors[team] = sample[1].copy()
                self.anchor_initializations += 2
                return True
            color, appearance = samples[0]
            self.anchors[1] = color
            if appearance is not None:
                self.appearance_anchors[1] = appearance.copy()
            self.anchor_initializations += 1
            return False

        if len(self.anchors) == 1:
            anchor = self.anchors[1]
            anchor_appearance = self.appearance_anchors.get(1)
            candidate = max(
                samples,
                key=lambda sample: self._sample_distance(
                    (anchor, anchor_appearance),
                    sample,
                ),
            )
            if self._sample_distance((anchor, anchor_appearance), candidate) >= 0.30:
                self.anchors[2] = candidate[0]
                if candidate[1] is not None:
                    self.appearance_anchors[2] = candidate[1].copy()
                self.anchor_initializations += 1
                return True
        return False

    def _nearest_team(
        self,
        color: tuple[int, int, int],
        appearance: np.ndarray | None,
        role_name: str = "player",
    ) -> tuple[int, float, str]:
        distances: dict[int, float] = {}
        for team, anchor in self.anchors.items():
            goalkeeper_colors = self.goalkeeper_reference_palettes_bgr.get(team, [])
            if role_name == "goalkeeper" and goalkeeper_colors:
                reference_colors = goalkeeper_colors
            else:
                reference_colors = self.reference_palettes_bgr.get(team) or [anchor]
            distances[team] = min(
                self._sample_distance(
                    (color, appearance),
                    (
                        reference_color,
                        self.appearance_anchors.get(team),
                    ),
                )
                for reference_color in reference_colors
            )
        team = min(distances, key=distances.get)
        source = (
            "goalkeeper_kit_reference"
            if role_name == "goalkeeper" and team in self.goalkeeper_reference_palettes_bgr
            else "stored_kit_reference"
            if team in self.reference_seeded_teams
            else "online_appearance_cluster"
        )
        if len(distances) < 2:
            return team, 0.50, source
        ordered = sorted(distances.values())
        margin = max(0.0, ordered[1] - ordered[0])
        confidence = max(
            0.50,
            min(
                0.99,
                0.50
                + margin * 0.95
                + max(0.0, 0.22 - ordered[0]) * 0.45,
            ),
        )
        return team, confidence, source

    def _update_anchors(
        self,
        samples: dict[int, tuple[tuple[int, int, int], np.ndarray | None]],
        assignments: dict[int, int],
        outfield_track_ids: set[int],
    ) -> None:
        for team in (1, 2):
            team_samples = [
                samples[track_id]
                for track_id, assigned_team in assignments.items()
                if assigned_team == team and track_id in outfield_track_ids
            ]
            if not team_samples:
                continue
            sample_color = tuple(
                int(value)
                for value in np.median(
                    np.array([sample[0] for sample in team_samples], dtype=np.float32),
                    axis=0,
                )
            )
            anchor = self.anchors[team]
            observation_weight = 0.01 if team in self.reference_seeded_teams else 0.03
            self.anchors[team] = tuple(
                int(
                    round(
                        anchor[index] * (1.0 - observation_weight)
                        + sample_color[index] * observation_weight
                    )
                )
                for index in range(3)
            )
            appearance_samples = [
                sample[1]
                for sample in team_samples
                if sample[1] is not None
            ]
            if appearance_samples:
                sample_appearance = np.mean(
                    np.stack(appearance_samples).astype(np.float32),
                    axis=0,
                )
                norm = float(np.linalg.norm(sample_appearance))
                if norm > 1e-6:
                    sample_appearance /= norm
                    existing = self.appearance_anchors.get(team)
                    if existing is None:
                        self.appearance_anchors[team] = sample_appearance
                    else:
                        blended = existing * 0.97 + sample_appearance * 0.03
                        blended_norm = float(np.linalg.norm(blended))
                        self.appearance_anchors[team] = (
                            blended / blended_norm
                            if blended_norm > 1e-6
                            else blended
                        )

    def _sample_distance(
        self,
        first: tuple[tuple[int, int, int], np.ndarray | None],
        second: tuple[tuple[int, int, int], np.ndarray | None],
    ) -> float:
        color_distance = self._color_distance(first[0], second[0])
        if first[1] is None or second[1] is None:
            return color_distance
        appearance_similarity = float(
            max(0.0, min(1.0, np.dot(first[1], second[1])))
        )
        appearance_distance = 1.0 - appearance_similarity
        return color_distance * 0.72 + appearance_distance * 0.28

    def kit_outlier_score(
        self,
        team: int | None,
        state: StableTrackState,
    ) -> float:
        """Return how unlike an outfield kit this participant appears."""
        if team not in {1, 2} or state.jersey_color is None:
            return 0.0
        reference_colors = self.reference_palettes_bgr.get(team)
        if not reference_colors:
            anchor = self.anchors.get(team)
            reference_colors = [anchor] if anchor is not None else []
        if not reference_colors:
            return 0.0
        distance = min(
            self._sample_distance(
                (reference_color, self.appearance_anchors.get(team)),
                (state.jersey_color, state.appearance_hist),
            )
            for reference_color in reference_colors
        )
        return float(np.clip((distance - 0.14) / 0.52, 0.0, 1.0))

    def _color_distance(
        self,
        first: tuple[int, int, int],
        second: tuple[int, int, int],
    ) -> float:
        pixels = np.array([[list(first), list(second)]], dtype=np.uint8)
        hsv = cv2.cvtColor(pixels, cv2.COLOR_BGR2HSV)[0]
        lab = cv2.cvtColor(pixels, cv2.COLOR_BGR2LAB)[0].astype(np.float32)
        hue_gap = abs(float(hsv[0, 0]) - float(hsv[1, 0]))
        hue_gap = min(hue_gap, 180.0 - hue_gap) / 90.0
        saturation_gap = abs(float(hsv[0, 1]) - float(hsv[1, 1])) / 255.0
        value_gap = abs(float(hsv[0, 2]) - float(hsv[1, 2])) / 255.0
        both_colored = hsv[0, 1] >= 42 and hsv[1, 1] >= 42
        chroma_gap = float(np.linalg.norm(lab[0, 1:3] - lab[1, 1:3])) / 181.0
        if both_colored:
            return hue_gap * 0.58 + saturation_gap * 0.16 + chroma_gap * 0.22 + value_gap * 0.04
        return hue_gap * 0.08 + saturation_gap * 0.34 + chroma_gap * 0.38 + value_gap * 0.20


class ParticipantRoleClassifierV2:
    """Resolve a stable semantic role without participating in identity matching."""

    def __init__(self) -> None:
        self.states: dict[int, ParticipantRoleState] = {}
        self.metric_observations = 0
        self.visual_fallback_observations = 0
        self.locked_role_count = 0
        self.prevented_role_changes = 0

    def update(
        self,
        players: list[AnalysisObject],
        track_states: dict[int, StableTrackState],
        team_by_track: dict[int, int],
        team_classifier: TeamColorClassifier,
        radar: "PitchRadar",
        frame: np.ndarray,
        surface_mask: np.ndarray | None = None,
    ) -> None:
        if surface_mask is None:
            surface_mask = radar.playing_surface_mask(frame)
        for player in players:
            stable_state = track_states.get(player.track_id)
            if stable_state is None:
                continue
            role_state = self.states.setdefault(
                player.track_id,
                ParticipantRoleState(track_id=player.track_id),
            )
            role_state.observations += 1
            for role in PARTICIPANT_ROLES:
                role_state.scores[role] *= 0.975

            foot = (
                (float(player.bbox[0]) + float(player.bbox[2])) / 2.0,
                float(player.bbox[3]),
            )
            metric_reliable = radar.is_reliable(0.62)
            pitch_point = radar.transform_point(foot) if metric_reliable else None
            if metric_reliable:
                self.metric_observations += 1
                inside_pitch = radar.contains_image_point(foot, margin_cm=40.0)
                surface_support = 1.0 if inside_pitch else 0.0
            else:
                self.visual_fallback_observations += 1
                surface_support = self._surface_support(surface_mask, foot)
                inside_pitch = surface_support >= 0.20

            if inside_pitch:
                role_state.inside_pitch_observations += 1
            else:
                role_state.outside_pitch_observations += 1

            near_goal = False
            near_touchline = False
            goalkeeper_zone = False
            penalty_area = False
            if pitch_point is not None:
                x_cm, y_cm = pitch_point
                goal_line_distance = min(x_cm, PITCH_LENGTH_CM - x_cm)
                goal_center_distance = abs(y_cm - PITCH_WIDTH_CM / 2.0)
                near_goal = goal_line_distance <= 1850.0
                penalty_area = (
                    goal_line_distance <= PENALTY_AREA_LENGTH_CM + 180.0
                    and goal_center_distance <= PENALTY_AREA_WIDTH_CM / 2.0 + 180.0
                )
                goalkeeper_zone = (
                    goal_line_distance <= PENALTY_AREA_LENGTH_CM + 500.0
                    and goal_center_distance <= PENALTY_AREA_WIDTH_CM / 2.0 + 550.0
                )
                near_touchline = min(y_cm, PITCH_WIDTH_CM - y_cm) <= 260.0
                if near_goal:
                    role_state.near_goal_observations += 1
                if goalkeeper_zone:
                    role_state.goalkeeper_zone_observations += 1
                if penalty_area:
                    role_state.penalty_area_observations += 1
                if near_touchline:
                    role_state.touchline_observations += 1

            team_number = team_by_track.get(player.track_id)
            team_confidence = float(
                team_classifier.track_confidence.get(player.track_id, 0.0)
            )
            team_affinity = team_number in {1, 2} and team_confidence >= 0.62
            if team_affinity:
                role_state.team_affinity_observations += 1
            kit_outlier_resolver = getattr(
                team_classifier,
                "kit_outlier_score",
                None,
            )
            kit_outlier_score = (
                float(kit_outlier_resolver(team_number, stable_state))
                if callable(kit_outlier_resolver)
                else 0.0
            )
            role_state.kit_outlier_score_sum += kit_outlier_score
            if kit_outlier_score >= 0.30:
                role_state.kit_outlier_observations += 1

            detector_ratios = self._detector_role_ratios(stable_state)
            for role, ratio in detector_ratios.items():
                if ratio >= 0.45:
                    role_state.detector_role_observations[role] = (
                        role_state.detector_role_observations.get(role, 0) + 1
                    )

            scores = role_state.scores
            scores["player"] += 0.50 + detector_ratios["player"] * 1.10
            scores["goalkeeper"] += detector_ratios["goalkeeper"] * 2.40
            scores["referee"] += detector_ratios["referee"] * 2.20
            scores["assistant_referee"] += detector_ratios["assistant_referee"] * 2.30
            scores["staff_outside_pitch"] += detector_ratios["staff_outside_pitch"] * 2.20

            if team_affinity:
                scores["player"] += 1.35
                if detector_ratios["goalkeeper"] >= 0.35 and near_goal:
                    scores["goalkeeper"] += 0.85
            if near_goal and detector_ratios["goalkeeper"] >= 0.22:
                scores["goalkeeper"] += 0.70
            goalkeeper_zone_ratio = (
                role_state.goalkeeper_zone_observations
                / max(role_state.observations, 1)
            )
            penalty_area_ratio = (
                role_state.penalty_area_observations
                / max(role_state.observations, 1)
            )
            average_kit_outlier = (
                role_state.kit_outlier_score_sum
                / max(role_state.observations, 1)
            )
            if goalkeeper_zone and team_affinity:
                scores["goalkeeper"] += 0.14 + kit_outlier_score * 1.75
            if (
                role_state.observations >= 10
                and goalkeeper_zone_ratio >= 0.68
                and penalty_area_ratio >= 0.48
                and average_kit_outlier >= 0.24
            ):
                scores["goalkeeper"] += 2.80
            if near_touchline and (
                detector_ratios["referee"] + detector_ratios["assistant_referee"]
            ) >= 0.45:
                # Generic football detectors commonly label both officials as
                # referee. Persistent metric touchline evidence disambiguates
                # the assistant without changing the physical track identity.
                scores["assistant_referee"] += 3.10
            if not inside_pitch:
                scores["staff_outside_pitch"] += 6.5 if metric_reliable else 3.5
            elif surface_support >= 0.45:
                scores["player"] += 0.35

            candidate = max(scores, key=scores.get)
            candidate = self._eligible_candidate(
                role_state,
                candidate,
                detector_ratios,
            )
            confidence = self._confidence(scores, candidate)
            if candidate in {"goalkeeper", "referee", "assistant_referee"}:
                detector_support = detector_ratios[candidate]
                if candidate == "assistant_referee":
                    detector_support += detector_ratios["referee"] * 0.5
                touchline_assistant_support = (
                    candidate == "assistant_referee"
                    and (
                        detector_ratios["referee"]
                        + detector_ratios["assistant_referee"]
                    ) >= 0.78
                    and role_state.touchline_observations
                    / max(role_state.observations, 1) >= 0.72
                )
                if detector_support >= 0.78 or touchline_assistant_support:
                    confidence = max(confidence, 0.86)
                geometric_goalkeeper_support = (
                    candidate == "goalkeeper"
                    and goalkeeper_zone_ratio >= 0.68
                    and penalty_area_ratio >= 0.48
                    and average_kit_outlier >= 0.24
                    and team_affinity
                )
                if geometric_goalkeeper_support:
                    confidence = max(confidence, 0.86)
            if candidate == "staff_outside_pitch":
                outside_ratio = role_state.outside_pitch_observations / max(
                    role_state.observations,
                    1,
                )
                if outside_ratio >= 0.80:
                    confidence = max(confidence, 0.90)
            if candidate == role_state.last_candidate_role:
                role_state.consecutive_role_frames += 1
            else:
                role_state.last_candidate_role = candidate
                role_state.consecutive_role_frames = 1

            if role_state.locked:
                if candidate != role_state.role_name:
                    self.prevented_role_changes += 1
            else:
                role_state.role_name = candidate
                role_state.confidence = confidence
                protected_goalkeeper_candidate = (
                    candidate == "player"
                    and goalkeeper_zone_ratio >= 0.58
                    and average_kit_outlier >= 0.22
                )
                required_observations = (
                    42
                    if protected_goalkeeper_candidate
                    else 20
                    if candidate == "player"
                    else 14
                )
                required_streak = 10 if candidate == "player" else 8
                required_confidence = 0.84 if candidate == "player" else 0.78
                if (
                    role_state.observations >= required_observations
                    and role_state.consecutive_role_frames >= required_streak
                    and confidence >= required_confidence
                ):
                    role_state.locked = True
                    self.locked_role_count += 1

            role_state.evidence = self._evidence_codes(
                role_state,
                detector_ratios,
                team_affinity,
                metric_reliable,
            )
            stable_state.role_name = role_state.role_name
            stable_state.role_locked = role_state.locked
            player.role_name = role_state.role_name

    def get(self, track_id: int) -> ParticipantRoleState | None:
        return self.states.get(track_id)

    def summary(self) -> dict[str, Any]:
        role_counts: dict[str, int] = {}
        review_required = 0
        for state in self.states.values():
            role_counts[state.role_name] = role_counts.get(state.role_name, 0) + 1
            if state.confidence < 0.72 or not state.locked:
                review_required += 1
        return {
            "engine": "participant_role_classifier_v2_temporal_geometry",
            "role_counts": role_counts,
            "tracks_evaluated": len(self.states),
            "locked_roles": sum(1 for state in self.states.values() if state.locked),
            "tracks_needing_role_review": review_required,
            "metric_observations": self.metric_observations,
            "visual_fallback_observations": self.visual_fallback_observations,
            "prevented_role_changes": self.prevented_role_changes,
            "roles": {
                str(track_id): {
                    "role_name": state.role_name,
                    "confidence": round(state.confidence, 4),
                    "locked": state.locked,
                    "observations": state.observations,
                    "evidence": list(state.evidence),
                    "inside_pitch_ratio": round(
                        state.inside_pitch_observations / max(state.observations, 1),
                        4,
                    ),
                    "outside_pitch_ratio": round(
                        state.outside_pitch_observations / max(state.observations, 1),
                        4,
                    ),
                    "goalkeeper_zone_ratio": round(
                        state.goalkeeper_zone_observations / max(state.observations, 1),
                        4,
                    ),
                    "penalty_area_ratio": round(
                        state.penalty_area_observations / max(state.observations, 1),
                        4,
                    ),
                    "kit_outlier_score": round(
                        state.kit_outlier_score_sum / max(state.observations, 1),
                        4,
                    ),
                }
                for track_id, state in sorted(self.states.items())
            },
        }

    def _eligible_candidate(
        self,
        state: ParticipantRoleState,
        candidate: str,
        detector_ratios: dict[str, float],
    ) -> str:
        observations = max(state.observations, 1)
        outside_ratio = state.outside_pitch_observations / observations
        near_goal_ratio = state.near_goal_observations / observations
        goalkeeper_zone_ratio = state.goalkeeper_zone_observations / observations
        penalty_area_ratio = state.penalty_area_observations / observations
        average_kit_outlier = state.kit_outlier_score_sum / observations
        touchline_ratio = state.touchline_observations / observations
        team_affinity_ratio = state.team_affinity_observations / observations
        if candidate == "staff_outside_pitch":
            if state.observations < 6 or outside_ratio < 0.62:
                return "player"
        elif candidate == "assistant_referee":
            official_ratio = detector_ratios["referee"] + detector_ratios["assistant_referee"]
            if state.observations < 10 or official_ratio < 0.48 or touchline_ratio < 0.45:
                return "referee" if official_ratio >= 0.58 else "player"
        elif candidate == "referee":
            if state.observations < 8 or detector_ratios["referee"] < 0.52:
                return "player"
        elif candidate == "goalkeeper":
            detector_support = detector_ratios["goalkeeper"] >= 0.46
            geometric_support = (
                goalkeeper_zone_ratio >= 0.68
                and penalty_area_ratio >= 0.48
                and average_kit_outlier >= 0.24
                and team_affinity_ratio >= 0.35
            )
            if state.observations < 10 or not (detector_support or geometric_support):
                return "player"
        return candidate

    def _detector_role_ratios(
        self,
        state: StableTrackState,
    ) -> dict[str, float]:
        normalized = {role: 0.0 for role in PARTICIPANT_ROLES}
        for role, score in state.role_votes.items():
            target = role if role in PARTICIPANT_ROLES else "player"
            normalized[target] += max(0.0, float(score))
        total = max(sum(normalized.values()), 1e-6)
        return {role: value / total for role, value in normalized.items()}

    def _confidence(self, scores: dict[str, float], candidate: str) -> float:
        ordered = sorted((max(0.0, value) for value in scores.values()), reverse=True)
        total = max(sum(ordered), 1e-6)
        margin = (ordered[0] - ordered[1]) / max(ordered[0], 1e-6) if len(ordered) > 1 else 1.0
        share = max(0.0, scores[candidate]) / total
        return float(np.clip(share * 0.62 + margin * 0.38, 0.0, 1.0))

    def _surface_support(
        self,
        mask: np.ndarray | None,
        point: tuple[float, float],
    ) -> float:
        if mask is None:
            return 0.5
        x, y = int(round(point[0])), int(round(point[1]))
        if x < 0 or y < 0 or x >= mask.shape[1] or y >= mask.shape[0]:
            return 0.0
        radius = max(5, int(round(mask.shape[1] * 0.004)))
        crop = mask[
            max(0, y - radius) : min(mask.shape[0], y + radius + 1),
            max(0, x - radius) : min(mask.shape[1], x + radius + 1),
        ]
        return float(np.mean(crop > 0)) if crop.size else 0.0

    def _evidence_codes(
        self,
        state: ParticipantRoleState,
        detector_ratios: dict[str, float],
        team_affinity: bool,
        metric_reliable: bool,
    ) -> list[str]:
        observations = max(state.observations, 1)
        evidence = ["metric_pitch_geometry" if metric_reliable else "visual_pitch_fallback"]
        if team_affinity:
            evidence.append("team_kit_affinity")
        if state.outside_pitch_observations / observations >= 0.40:
            evidence.append("outside_pitch_history")
        if state.near_goal_observations / observations >= 0.50:
            evidence.append("goal_area_history")
        if state.goalkeeper_zone_observations / observations >= 0.68:
            evidence.append("persistent_goalkeeper_zone")
        if state.kit_outlier_score_sum / observations >= 0.24:
            evidence.append("outfield_kit_outlier")
        if state.touchline_observations / observations >= 0.45:
            evidence.append("touchline_history")
        strongest_detector = max(detector_ratios, key=detector_ratios.get)
        if detector_ratios[strongest_detector] >= 0.45:
            evidence.append(f"detector_{strongest_detector}")
        return evidence


@dataclass
class BallStaticCandidate:
    candidate_id: int
    image_origin: tuple[float, float]
    image_center: tuple[float, float]
    first_frame: int
    last_frame: int
    pitch_origin: tuple[float, float] | None = None
    pitch_center: tuple[float, float] | None = None
    raw_ids: set[int] = field(default_factory=set)
    hits: int = 1
    aspect_sum: float = 1.0
    max_image_displacement: float = 0.0
    max_pitch_displacement: float = 0.0
    confirmed_static: bool = False
    confirmed_moving: bool = False

    @property
    def average_aspect(self) -> float:
        return self.aspect_sum / max(1, self.hits)


@dataclass
class BallReacquisitionHypothesis:
    """One temporally consistent ball candidate during a lost-track search."""

    hypothesis_id: int
    position: np.ndarray
    origin: np.ndarray
    first_frame: int
    last_frame: int
    previous_position: np.ndarray
    previous_frame: int
    velocity: np.ndarray = field(
        default_factory=lambda: np.zeros(2, dtype=np.float64)
    )
    acceleration: np.ndarray = field(
        default_factory=lambda: np.zeros(2, dtype=np.float64)
    )
    hits: int = 1
    strong_hits: int = 0
    ground_hits: int = 0
    max_displacement: float = 0.0
    confidence_sum: float = 0.0
    size_sum: float = 0.0
    last_detection: AnalysisObject | None = None
    body_owner_id: int | None = None


class BallStaticFilter:
    def __init__(
        self,
        static_hits: int = 3,
    ) -> None:
        self.static_hits = static_hits
        self.candidates: dict[int, BallStaticCandidate] = {}
        self.next_candidate_id = 1
        self.raw_seen = 0
        self.kept = 0
        self.filtered_static = 0
        self.suppressed_tentative = 0
        self.forwarded_tentative = 0
        self.pitch_stabilized_observations = 0
        self.penalty_spot_rejections = 0
        self.outside_pitch_rejections = 0

    def filter(
        self,
        frame_index: int,
        balls: list[AnalysisObject],
        players: list[AnalysisObject],
        frame_width: int,
        pitch_transform: Any | None = None,
    ) -> list[AnalysisObject]:
        kept: list[AnalysisObject] = []
        for ball in balls:
            self.raw_seen += 1
            center = self._center(ball.bbox)
            pitch_center = pitch_transform(center) if pitch_transform is not None else None
            if pitch_center is not None:
                self.pitch_stabilized_observations += 1

            width = max(1.0, ball.bbox[2] - ball.bbox[0])
            height = max(1.0, ball.bbox[3] - ball.bbox[1])
            near_player = self._near_player_foot(center, players, frame_width)
            near_penalty_spot = self._near_known_penalty_spot(pitch_center)
            candidate = self._observe(
                frame_index=frame_index,
                image_center=center,
                pitch_center=pitch_center,
                raw_track_id=ball.raw_track_id,
                aspect=width / height,
                frame_width=frame_width,
                near_player=near_player and not near_penalty_spot,
            )

            if pitch_center is None and pitch_transform is not None:
                self.outside_pitch_rejections += 1
                continue

            if near_penalty_spot and not near_player and not candidate.confirmed_moving:
                candidate.confirmed_static = True
                candidate.confirmed_moving = False
                self.penalty_spot_rejections += 1
                self.filtered_static += 1
                continue

            if near_player:
                kept.append(ball)
                self.kept += 1
                continue

            if width / height >= 1.55:
                candidate.confirmed_static = True
                candidate.confirmed_moving = False
                self.filtered_static += 1
                continue

            if candidate.confirmed_moving:
                kept.append(ball)
                self.kept += 1
                continue

            if candidate.confirmed_static:
                self.filtered_static += 1
                continue
            # The dedicated model often sees a fast, tiny ball only once or
            # twice. Forward round tentative candidates to the continuity
            # tracker; it owns motion, body-overlap, and reacquisition gates.
            if 0.48 <= width / height <= 1.70:
                kept.append(ball)
                self.kept += 1
                self.forwarded_tentative += 1
                continue
            self.suppressed_tentative += 1
        return kept

    def static_marker_centers(self, frame_index: int) -> list[tuple[float, float]]:
        markers = [
            candidate
            for candidate in self.candidates.values()
            if candidate.confirmed_static
            and candidate.average_aspect >= 1.45
            and frame_index - candidate.last_frame <= 12
        ]
        markers.sort(key=lambda candidate: (-candidate.hits, candidate.candidate_id))
        return [candidate.image_center for candidate in markers]

    def summary(self) -> dict[str, Any]:
        return {
            "engine": "metric_static_ball_filter_v4",
            "raw_ball_observations": self.raw_seen,
            "kept_ball_observations": self.kept,
            "filtered_static_candidates": self.filtered_static,
            "suppressed_tentative_observations": self.suppressed_tentative,
            "forwarded_tentative_observations": self.forwarded_tentative,
            "static_hits_threshold": self.static_hits,
            "pitch_stabilized_observations": self.pitch_stabilized_observations,
            "penalty_spot_rejections": self.penalty_spot_rejections,
            "outside_pitch_rejections": self.outside_pitch_rejections,
            "confirmed_static_markers": sum(
                1 for candidate in self.candidates.values() if candidate.confirmed_static
            ),
            "motion_confirmed_candidates": sum(
                1 for candidate in self.candidates.values() if candidate.confirmed_moving
            ),
        }

    def _observe(
        self,
        frame_index: int,
        image_center: tuple[float, float],
        pitch_center: tuple[float, float] | None,
        raw_track_id: int | None,
        aspect: float,
        frame_width: int,
        near_player: bool,
    ) -> BallStaticCandidate:
        candidate = self._find_candidate(
            frame_index,
            image_center,
            pitch_center,
            raw_track_id,
            frame_width,
            aspect,
        )
        if candidate is None:
            candidate = BallStaticCandidate(
                candidate_id=self.next_candidate_id,
                image_origin=image_center,
                image_center=image_center,
                first_frame=frame_index,
                last_frame=frame_index,
                pitch_origin=pitch_center,
                pitch_center=pitch_center,
                raw_ids={raw_track_id} if raw_track_id is not None else set(),
                aspect_sum=aspect,
            )
            self.candidates[candidate.candidate_id] = candidate
            self.next_candidate_id += 1
            return candidate

        candidate.hits += 1
        candidate.last_frame = frame_index
        candidate.aspect_sum += aspect
        if raw_track_id is not None:
            candidate.raw_ids.add(raw_track_id)
        candidate.max_image_displacement = max(
            candidate.max_image_displacement,
            float(np.hypot(
                image_center[0] - candidate.image_origin[0],
                image_center[1] - candidate.image_origin[1],
            )),
        )
        if pitch_center is not None:
            if candidate.pitch_origin is None:
                candidate.pitch_origin = pitch_center
            candidate.max_pitch_displacement = max(
                candidate.max_pitch_displacement,
                float(np.hypot(
                    pitch_center[0] - candidate.pitch_origin[0],
                    pitch_center[1] - candidate.pitch_origin[1],
                )),
            )
            candidate.pitch_center = pitch_center
        candidate.image_center = image_center

        flat_marker = candidate.average_aspect >= 1.45
        image_motion_threshold = max(24.0, frame_width * 0.006)
        if (
            candidate.confirmed_static
            and flat_marker
            and not near_player
            and (
                aspect >= 1.45
                or candidate.max_image_displacement <= image_motion_threshold * 2.0
            )
        ):
            candidate.confirmed_moving = False
            return candidate
        pitch_is_moving = candidate.max_pitch_displacement >= 110.0
        image_is_moving = candidate.max_image_displacement >= image_motion_threshold
        if near_player or pitch_is_moving or (image_is_moving and not flat_marker):
            candidate.confirmed_moving = True
            candidate.confirmed_static = False
        elif candidate.hits >= self.static_hits and flat_marker:
            image_is_static = candidate.max_image_displacement <= image_motion_threshold
            ground_is_static = (
                candidate.max_pitch_displacement <= 65.0
                if candidate.pitch_origin is not None and candidate.pitch_center is not None
                else image_is_static
            )
            if ground_is_static:
                candidate.confirmed_static = True
        return candidate

    def _find_candidate(
        self,
        frame_index: int,
        image_center: tuple[float, float],
        pitch_center: tuple[float, float] | None,
        raw_track_id: int | None,
        frame_width: int,
        aspect: float,
    ) -> BallStaticCandidate | None:
        active = [
            candidate
            for candidate in self.candidates.values()
            if frame_index - candidate.last_frame <= 12
        ]
        if raw_track_id is not None:
            raw_matches = [
                candidate
                for candidate in active
                if raw_track_id in candidate.raw_ids
                and not (candidate.confirmed_static and aspect < 1.35)
                and not (candidate.confirmed_moving and aspect >= 1.55)
            ]
            if raw_matches:
                return max(raw_matches, key=lambda candidate: candidate.last_frame)

        image_gate = max(42.0, frame_width * 0.015)
        best: BallStaticCandidate | None = None
        best_score = float("inf")
        for candidate in active:
            if candidate.confirmed_static and aspect < 1.35:
                continue
            if candidate.confirmed_moving and aspect >= 1.55:
                continue
            image_distance = float(np.hypot(
                image_center[0] - candidate.image_center[0],
                image_center[1] - candidate.image_center[1],
            ))
            pitch_distance: float | None = None
            if pitch_center is not None and candidate.pitch_center is not None:
                pitch_distance = float(np.hypot(
                    pitch_center[0] - candidate.pitch_center[0],
                    pitch_center[1] - candidate.pitch_center[1],
                ))
            if image_distance > image_gate and (
                pitch_distance is None or pitch_distance > 160.0
            ):
                continue
            score = min(
                image_distance / image_gate,
                pitch_distance / 160.0 if pitch_distance is not None else float("inf"),
            )
            if score < best_score:
                best = candidate
                best_score = score
        return best

    def _near_player_foot(
        self,
        center: tuple[float, float],
        players: list[AnalysisObject],
        frame_width: int,
    ) -> bool:
        threshold = max(36.0, frame_width * 0.022)
        for player in players:
            foot = ((player.bbox[0] + player.bbox[2]) / 2, player.bbox[3])
            if float(np.hypot(foot[0] - center[0], foot[1] - center[1])) <= threshold:
                return True
        return False

    def _center(self, bbox: list[float]) -> tuple[float, float]:
        return ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)

    def _near_known_penalty_spot(
        self,
        pitch_center: tuple[float, float] | None,
    ) -> bool:
        if pitch_center is None:
            return False
        spots = (
            (PENALTY_SPOT_DISTANCE_CM, PITCH_WIDTH_CM / 2),
            (PITCH_LENGTH_CM - PENALTY_SPOT_DISTANCE_CM, PITCH_WIDTH_CM / 2),
        )
        return any(
            float(np.hypot(pitch_center[0] - spot[0], pitch_center[1] - spot[1])) <= 58.0
            for spot in spots
        )


class BallTrackerV4:
    """Single-ball tracker with guarded ground and airborne motion states."""

    def __init__(
        self,
        max_interpolation_frames: int = 6,
        max_airborne_interpolation_frames: int = 36,
        max_reacquisition_frames: int = 90,
        fps: float = 25.0,
    ) -> None:
        self.fps = max(1.0, float(fps))
        self.gravity_cm_per_frame2 = 981.0 / (self.fps * self.fps)
        self.max_interpolation_frames = max_interpolation_frames
        self.max_airborne_interpolation_frames = max(
            max_interpolation_frames,
            max_airborne_interpolation_frames,
        )
        self.max_reacquisition_frames = max(
            max_interpolation_frames + 1,
            max_reacquisition_frames,
        )
        self.position: np.ndarray | None = None
        self.flow_position: np.ndarray | None = None
        self.observed_position: np.ndarray | None = None
        self.observed_velocity = np.zeros(2, dtype=np.float64)
        self.observed_position_frame = -1
        self.velocity = np.zeros(2, dtype=np.float64)
        self.kalman_state: np.ndarray | None = None
        self.kalman_covariance = np.diag([120.0, 120.0, 36.0, 36.0]).astype(np.float64)
        self.predicted_state: np.ndarray | None = None
        self.predicted_covariance: np.ndarray | None = None
        self.size = np.array([12.0, 12.0], dtype=np.float64)
        self.last_frame = -1
        self.last_observed_frame = -1
        self.pending_position: np.ndarray | None = None
        self.pending_pitch_position: np.ndarray | None = None
        self.pending_frame = -1
        self.pitch_position: np.ndarray | None = None
        self.pitch_velocity = np.zeros(2, dtype=np.float64)
        self.last_pitch_frame = -1
        self.position_3d: np.ndarray | None = None
        self.velocity_3d = np.zeros(3, dtype=np.float64)
        self.last_3d_frame = -1
        self.trajectory_3d_confidence = 0.0
        self.flight_start_frame = -1
        self.maximum_height_cm = 0.0
        self.three_d_observed_frames = 0
        self.three_d_predicted_frames = 0
        self.three_d_projection_failures = 0
        self.three_d_low_confidence_fallbacks = 0
        self.three_d_observation_projection_rejections = 0
        self.confidence = 0.0
        self.observed_frames = 0
        self.interpolated_frames = 0
        self.rejected_gate = 0
        self.reinitializations = 0
        self.expired_track_resets = 0
        self.current_interpolation_streak = 0
        self.maximum_interpolation_streak = 0
        self.mahalanobis_rejections = 0
        self.metric_jump_rejections = 0
        self.player_body_rejections = 0
        self.implausible_size_rejections = 0
        self.ambiguous_reacquisitions = 0
        self.optical_flow_attempts = 0
        self.optical_flow_successes = 0
        self.optical_flow_rejections = 0
        self.predicted_body_suppressions = 0
        self.dormant_reacquisitions = 0
        self.airborne_entries = 0
        self.airborne_observed_frames = 0
        self.metric_conflict_acceptances = 0
        self.ground_reacquisitions = 0
        self.airborne_body_pass_throughs = 0
        self.airborne_trajectory_rejections = 0
        self.airborne_flow_drift_rejections = 0
        self.ground_contact_confirmations = 0
        self.challenger_position: np.ndarray | None = None
        self.challenger_origin: np.ndarray | None = None
        self.challenger_first_frame = -1
        self.challenger_last_frame = -1
        self.challenger_hits = 0
        self.challenger_strong_hits = 0
        self.challenger_max_displacement = 0.0
        self.challenger_previous_position: np.ndarray | None = None
        self.challenger_previous_frame = -1
        self.challenger_velocity = np.zeros(2, dtype=np.float64)
        self.challenger_ground_hits = 0
        self.reinitialization_velocity = np.zeros(2, dtype=np.float64)
        self.challenger_prediction_suppressions = 0
        self.airborne_challenger_promotions = 0
        self.challengers_started = 0
        self.challenger_promotions = 0
        self.challenger_static_rejections = 0
        self.challenger_reinitialization_ready = False
        self.reacquisition_hypotheses: list[BallReacquisitionHypothesis] = []
        self.next_reacquisition_hypothesis_id = 1
        self.peak_reacquisition_hypotheses = 0
        self.reacquisition_competing_rejections = 0
        self.flight_mode = False
        self.flight_clear_hits = 0
        self.selected_measurement_details: dict[str, Any] = {}
        self.previous_gray: np.ndarray | None = None
        self.track_id = 1
        self.pitch_path: list[dict[str, float | int | bool]] = []
        self.image_path: list[dict[str, float | int | bool]] = []

    def update(
        self,
        frame_index: int,
        detections: list[AnalysisObject],
        players: list[AnalysisObject],
        frame_width: int,
        pitch_transform: Any | None = None,
        frame: np.ndarray | None = None,
        pitch_camera: "PitchRadar | None" = None,
    ) -> list[AnalysisObject]:
        self._expire_lost_track(frame_index)
        predicted = self._predict(frame_index)
        predicted_3d = self._predict_3d(frame_index)
        frame_shape = (
            frame.shape
            if frame is not None
            else (int(round(frame_width * 9 / 16)), frame_width)
        )
        camera_projection = (
            pitch_camera.camera_projection_matrix(frame_shape)
            if pitch_camera is not None and pitch_camera.is_reliable(0.52)
            else None
        )
        projected_3d = self._project_3d(predicted_3d, camera_projection)
        trusted_3d_projection = (
            projected_3d is not None
            and self.trajectory_3d_confidence >= 0.48
        )
        if self.flight_mode and trusted_3d_projection:
            predicted = projected_3d
        trajectory_prediction = self._trajectory_prediction(frame_index)
        current_gray = (
            cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if frame is not None
            else None
        )
        flow_prediction = self._optical_flow_prediction(current_gray, frame_width)
        if (
            flow_prediction is not None
            and self.flight_mode
            and (trusted_3d_projection or trajectory_prediction is not None)
            and float(
                np.linalg.norm(
                    flow_prediction
                    - (
                        projected_3d
                        if trusted_3d_projection
                        else trajectory_prediction
                    )
                )
            )
            > self._trajectory_gate(frame_index, frame_width)
        ):
            flow_prediction = None
            self.optical_flow_rejections += 1
            self.airborne_flow_drift_rejections += 1
        flow_prediction_valid = flow_prediction is not None
        if flow_prediction is not None:
            self.optical_flow_successes += 1
            self.flow_position = flow_prediction.copy()
            if self.flight_mode and trusted_3d_projection:
                predicted = projected_3d * 0.68 + flow_prediction * 0.32
            elif self.flight_mode and trajectory_prediction is not None:
                predicted = flow_prediction * 0.72 + trajectory_prediction * 0.28
            else:
                predicted = (
                    flow_prediction
                    if predicted is None
                    else flow_prediction * 0.72 + predicted * 0.28
                )
            if self.predicted_state is not None:
                self.predicted_state[:2] = predicted
        challenger = self._consider_global_challenger(
            frame_index,
            detections,
            players,
            frame_width,
            predicted,
        )
        if challenger is not None:
            self._prepare_challenger_reinitialization()
            predicted = None
            measurement = challenger
            center = np.array(self._center(challenger.bbox), dtype=np.float64)
            near_foot, _ = self._player_relation(center, players, frame_width)
            self.selected_measurement_details = {
                "near_foot": near_foot,
                "ground_contact": self._ground_contact(
                    center,
                    players,
                    frame_width,
                ),
                "dormant": True,
                "metric_conflict": False,
                "image_consistent": True,
                "body_pass_through": False,
            }
        else:
            measurement = self._select_measurement(
                frame_index,
                detections,
                predicted,
                players,
                frame_width,
                pitch_transform,
            )
        if measurement is not None:
            center = self._center(measurement.bbox)
            measured = np.array(center, dtype=np.float64)
            measured_pitch = self._pitch_measurement(measured, pitch_transform)
            previous_gap = (
                frame_index - self.last_observed_frame
                if self.last_observed_frame >= 0
                else 0
            )
            if self.position is None:
                challenger_confirmed = self.challenger_reinitialization_ready
                self.challenger_reinitialization_ready = False
                if not challenger_confirmed and not self._confirm_initial_measurement(
                    frame_index,
                    measured,
                    measured_pitch,
                    frame_width,
                ):
                    self._remember_gray(current_gray)
                    return []
                self.position = measured
                self.velocity = self.reinitialization_velocity.copy()
                self.reinitialization_velocity[:] = 0.0
                self.kalman_state = np.array(
                    [
                        measured[0],
                        measured[1],
                        self.velocity[0],
                        self.velocity[1],
                    ],
                    dtype=np.float64,
                )
                self.kalman_covariance = np.diag(
                    [64.0, 64.0, 25.0, 25.0]
                ).astype(np.float64)
                self.reinitializations += 1
            else:
                self._correct_kalman(
                    measured,
                    float(measurement.confidence or 0.5),
                )
            if previous_gap > self.max_interpolation_frames:
                self.dormant_reacquisitions += 1
            self._update_flight_state(frame_index, measured_pitch)
            self._update_observed_trajectory(frame_index, measured, frame_width)
            self._update_3d_state(
                frame_index=frame_index,
                measurement=measured,
                measured_pitch=measured_pitch,
                predicted_3d=predicted_3d,
                camera_projection=camera_projection,
            )
            self.flow_position = measured.copy()
            self.size = self.size * 0.80 + np.array(
                [
                    max(2.0, measurement.bbox[2] - measurement.bbox[0]),
                    max(2.0, measurement.bbox[3] - measurement.bbox[1]),
                ],
                dtype=np.float64,
            ) * 0.20
            self.last_frame = frame_index
            self.last_observed_frame = frame_index
            self.confidence = min(0.99, self.confidence * 0.72 + float(measurement.confidence or 0.5) * 0.28 + 0.18)
            if self.flight_mode:
                self.confidence = max(self.confidence, 0.42)
            self.observed_frames += 1
            self.current_interpolation_streak = 0
            output = self._object(measurement.raw_track_id, predicted=False)
            self._record_image(frame_index, output, predicted=False)
            self._record_pitch(frame_index, output, pitch_transform, predicted=False)
            self._remember_gray(current_gray)
            return [output]

        if self.position is None or self.last_observed_frame < 0:
            self._remember_gray(current_gray)
            return []
        if self._challenger_suppresses_prediction(frame_index):
            self.challenger_prediction_suppressions += 1
            self._remember_gray(current_gray)
            return []
        gap = frame_index - self.last_observed_frame
        interpolation_limit = (
            self.max_airborne_interpolation_frames
            if self.flight_mode and flow_prediction_valid
            else self.max_interpolation_frames
        )
        if gap > interpolation_limit:
            self.confidence *= 0.75
            self._remember_gray(current_gray)
            return []
        if self.predicted_state is None or self.predicted_covariance is None:
            self._remember_gray(current_gray)
            return []
        self.kalman_state = self.predicted_state.copy()
        self.kalman_state[2:] *= 0.96
        self.kalman_covariance = self.predicted_covariance.copy()
        self.position = self.kalman_state[:2].copy()
        self.velocity = self.kalman_state[2:].copy()
        if self.flight_mode and predicted_3d is not None:
            self._commit_3d_prediction(frame_index, predicted_3d)
            projected = self._project_3d(self.position_3d, camera_projection)
            if projected is not None and self.trajectory_3d_confidence >= 0.48:
                self.position = projected
                self.kalman_state[:2] = projected
            elif predicted is not None:
                # Monocular height is under-constrained. A low-confidence 3D
                # estimate may enrich analytics, but it must never drag the
                # visible marker away from the image/flow trajectory.
                self.position = predicted.copy()
                self.kalman_state[:2] = self.position
                self.three_d_low_confidence_fallbacks += 1
            else:
                self.three_d_projection_failures += 1
        elif not self.flight_mode:
            self._predict_pitch_state(frame_index)
        self.last_frame = frame_index
        self.confidence *= 0.97 if self.flight_mode and flow_prediction_valid else 0.82
        confidence_floor = 0.14 if self.flight_mode and flow_prediction_valid else 0.22
        if self.confidence < confidence_floor:
            self._remember_gray(current_gray)
            return []
        _, overlaps_body = self._player_relation(self.position, players, frame_width)
        if overlaps_body and not (self.flight_mode and flow_prediction_valid):
            self.predicted_body_suppressions += 1
            self.confidence *= 0.72
            self._remember_gray(current_gray)
            return []
        self.interpolated_frames += 1
        self.current_interpolation_streak += 1
        self.maximum_interpolation_streak = max(
            self.maximum_interpolation_streak,
            self.current_interpolation_streak,
        )
        output = self._object(None, predicted=True)
        self._record_image(frame_index, output, predicted=True)
        self._record_pitch(frame_index, output, pitch_transform, predicted=True)
        self._remember_gray(current_gray)
        return [output]

    def summary(self) -> dict[str, Any]:
        return {
            "engine": "metric_image_fusion_ball_tracker_v6_multihypothesis_3d",
            "track_id": self.track_id,
            "observed_frames": self.observed_frames,
            "interpolated_frames": self.interpolated_frames,
            "rejected_motion_gate": self.rejected_gate,
            "reinitializations": self.reinitializations,
            "expired_track_resets": self.expired_track_resets,
            "current_confidence": round(self.confidence, 4),
            "max_interpolation_frames": self.max_interpolation_frames,
            "max_airborne_interpolation_frames": self.max_airborne_interpolation_frames,
            "max_reacquisition_frames": self.max_reacquisition_frames,
            "maximum_interpolation_streak": self.maximum_interpolation_streak,
            "mahalanobis_rejections": self.mahalanobis_rejections,
            "metric_jump_rejections": self.metric_jump_rejections,
            "player_body_rejections": self.player_body_rejections,
            "implausible_size_rejections": self.implausible_size_rejections,
            "ambiguous_reacquisitions": self.ambiguous_reacquisitions,
            "optical_flow_attempts": self.optical_flow_attempts,
            "optical_flow_successes": self.optical_flow_successes,
            "optical_flow_rejections": self.optical_flow_rejections,
            "predicted_body_suppressions": self.predicted_body_suppressions,
            "dormant_reacquisitions": self.dormant_reacquisitions,
            "airborne_entries": self.airborne_entries,
            "airborne_observed_frames": self.airborne_observed_frames,
            "metric_conflict_acceptances": self.metric_conflict_acceptances,
            "ground_reacquisitions": self.ground_reacquisitions,
            "airborne_body_pass_throughs": self.airborne_body_pass_throughs,
            "airborne_trajectory_rejections": self.airborne_trajectory_rejections,
            "airborne_flow_drift_rejections": self.airborne_flow_drift_rejections,
            "ground_contact_confirmations": self.ground_contact_confirmations,
            "challengers_started": self.challengers_started,
            "challenger_promotions": self.challenger_promotions,
            "challenger_static_rejections": self.challenger_static_rejections,
            "challenger_prediction_suppressions": self.challenger_prediction_suppressions,
            "airborne_challenger_promotions": self.airborne_challenger_promotions,
            "active_reacquisition_hypotheses": len(
                self.reacquisition_hypotheses
            ),
            "peak_reacquisition_hypotheses": self.peak_reacquisition_hypotheses,
            "reacquisition_competing_rejections": (
                self.reacquisition_competing_rejections
            ),
            "flight_mode_active": self.flight_mode,
            "trajectory_3d": {
                "active": self.position_3d is not None,
                "confidence": round(self.trajectory_3d_confidence, 4),
                "observed_frames": self.three_d_observed_frames,
                "predicted_frames": self.three_d_predicted_frames,
                "maximum_height_cm": round(self.maximum_height_cm, 2),
                "projection_failures": self.three_d_projection_failures,
                "low_confidence_image_fallbacks": (
                    self.three_d_low_confidence_fallbacks
                ),
                "observation_projection_rejections": (
                    self.three_d_observation_projection_rejections
                ),
                "gravity_cm_per_frame2": round(self.gravity_cm_per_frame2, 5),
            },
            "state_covariance_trace": round(float(np.trace(self.kalman_covariance)), 3),
            "metric_state_active": self.pitch_position is not None,
            "pitch_samples": len(self.pitch_path),
            "image_samples": len(self.image_path),
        }

    def _predict(self, frame_index: int) -> np.ndarray | None:
        if self.kalman_state is None:
            return None
        delta_frames = max(1, frame_index - self.last_frame)
        transition = np.array(
            [
                [1.0, 0.0, delta_frames, 0.0],
                [0.0, 1.0, 0.0, delta_frames],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        process_scale = max(1.0, float(delta_frames))
        process_noise = np.diag(
            [5.0, 5.0, 1.8, 1.8]
        ).astype(np.float64) * process_scale
        self.predicted_state = transition @ self.kalman_state
        self.predicted_covariance = (
            transition @ self.kalman_covariance @ transition.T + process_noise
        )
        return self.predicted_state[:2].copy()

    def _optical_flow_prediction(
        self,
        current_gray: np.ndarray | None,
        frame_width: int,
    ) -> np.ndarray | None:
        if (
            current_gray is None
            or self.previous_gray is None
            or self.flow_position is None
            or self.previous_gray.shape != current_gray.shape
        ):
            return None
        self.optical_flow_attempts += 1
        previous_point = np.array(
            [[[float(self.flow_position[0]), float(self.flow_position[1])]]],
            dtype=np.float32,
        )
        criteria = (
            cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
            20,
            0.01,
        )
        next_points, status, error = cv2.calcOpticalFlowPyrLK(
            self.previous_gray,
            current_gray,
            previous_point,
            None,
            winSize=(21, 21),
            maxLevel=3,
            criteria=criteria,
        )
        if (
            next_points is None
            or status is None
            or int(status.reshape(-1)[0]) != 1
            or error is None
            or float(error.reshape(-1)[0]) > 28.0
        ):
            self.optical_flow_rejections += 1
            return None
        backward_points, backward_status, _ = cv2.calcOpticalFlowPyrLK(
            current_gray,
            self.previous_gray,
            next_points,
            None,
            winSize=(21, 21),
            maxLevel=3,
            criteria=criteria,
        )
        if (
            backward_points is None
            or backward_status is None
            or int(backward_status.reshape(-1)[0]) != 1
        ):
            self.optical_flow_rejections += 1
            return None
        forward = next_points.reshape(-1, 2)[0].astype(np.float64)
        backward = backward_points.reshape(-1, 2)[0].astype(np.float64)
        forward_backward_error = float(
            np.linalg.norm(backward - previous_point.reshape(-1, 2)[0])
        )
        displacement = float(np.linalg.norm(forward - self.flow_position))
        if (
            forward_backward_error > 2.5
            or displacement > max(45.0, frame_width * 0.07)
            or not np.isfinite(forward).all()
        ):
            self.optical_flow_rejections += 1
            return None
        return forward

    def _remember_gray(self, current_gray: np.ndarray | None) -> None:
        if current_gray is not None:
            self.previous_gray = current_gray

    def _consider_global_challenger(
        self,
        frame_index: int,
        detections: list[AnalysisObject],
        players: list[AnalysisObject],
        frame_width: int,
        predicted: np.ndarray | None,
    ) -> AnalysisObject | None:
        observation_gap = (
            frame_index - self.last_observed_frame
            if self.last_observed_frame >= 0
            else 0
        )
        dormant = (
            self.position is not None
            and self.last_observed_frame >= 0
            and observation_gap > self.max_interpolation_frames
        )
        stale_flight = dormant and self.flight_mode
        if not dormant:
            self._clear_challenger()
            return None

        hypothesis_ttl = 14 if stale_flight else 24
        retained_hypotheses: list[BallReacquisitionHypothesis] = []
        for hypothesis in self.reacquisition_hypotheses:
            hypothesis_age = frame_index - hypothesis.last_frame
            if hypothesis_age <= hypothesis_ttl:
                retained_hypotheses.append(hypothesis)
            elif hypothesis.hits >= 2 and hypothesis.max_displacement < 30.0:
                self.challenger_static_rejections += 1
        self.reacquisition_hypotheses = retained_hypotheses

        eligible: list[
            tuple[float, AnalysisObject, bool, float, int | None]
        ] = []
        for detection in detections:
            confidence = float(detection.confidence or 0.0)
            width = max(1.0, detection.bbox[2] - detection.bbox[0])
            height = max(1.0, detection.bbox[3] - detection.bbox[1])
            aspect = width / height
            if (
                confidence < (0.14 if stale_flight else 0.32)
                or not 0.42 <= aspect <= 1.80
                or max(width, height) > max(32.0, frame_width * 0.024)
            ):
                continue
            center = np.array(self._center(detection.bbox), dtype=np.float64)
            near_foot, overlaps_body = self._player_relation(
                center,
                players,
                frame_width,
            )
            body_owner_id = self._player_body_owner_id(center, players)
            if overlaps_body and not stale_flight:
                continue
            ground_contact = self._ground_contact(center, players, frame_width)
            distance_from_stale = (
                float(np.linalg.norm(center - predicted))
                if predicted is not None
                else frame_width
            )
            score = (
                confidence
                + (0.28 if ground_contact else 0.14 if near_foot else 0.0)
                + min(0.16, distance_from_stale / max(frame_width, 1) * 0.22)
                - abs(np.log(max(width / height, 1e-6))) * 0.04
                - (0.16 if overlaps_body else 0.0)
            )
            size = (width + height) / 2.0
            eligible.append(
                (score, detection, ground_contact, size, body_owner_id)
            )

        # Duplicate model boxes at the same location must not create several
        # independent votes for the same physical object.
        deduplicated: list[
            tuple[float, AnalysisObject, bool, float, int | None]
        ] = []
        for item in sorted(eligible, key=lambda value: value[0], reverse=True):
            center = np.array(self._center(item[1].bbox), dtype=np.float64)
            if any(
                float(
                    np.linalg.norm(
                        center
                        - np.array(self._center(existing[1].bbox), dtype=np.float64)
                    )
                )
                <= max(7.0, frame_width * 0.004)
                for existing in deduplicated
            ):
                continue
            deduplicated.append(item)
        eligible = deduplicated

        matched_hypotheses: set[int] = set()
        matched_detections: set[int] = set()
        association_candidates: list[tuple[float, int, int]] = []
        for hypothesis_index, hypothesis in enumerate(
            self.reacquisition_hypotheses
        ):
            elapsed = max(1, frame_index - hypothesis.last_frame)
            predicted_position = (
                hypothesis.position + hypothesis.velocity * elapsed
            )
            link_gate = max(42.0, frame_width * 0.022) + max(
                0,
                elapsed - 1,
            ) * max(38.0, frame_width * 0.020)
            average_size = hypothesis.size_sum / max(hypothesis.hits, 1)
            for detection_index, (
                score,
                detection,
                _,
                size,
                body_owner_id,
            ) in enumerate(eligible):
                center = np.array(
                    self._center(detection.bbox),
                    dtype=np.float64,
                )
                distance = float(np.linalg.norm(center - predicted_position))
                if distance > link_gate:
                    continue
                size_delta = abs(
                    np.log(max(size, 1e-6) / max(average_size, 1e-6))
                )
                if size_delta > 1.15:
                    continue
                if (
                    hypothesis.body_owner_id is not None
                    and body_owner_id is not None
                    and hypothesis.body_owner_id != body_owner_id
                ):
                    continue
                association_cost = (
                    distance / max(link_gate, 1.0)
                    + size_delta * 0.18
                    - score * 0.08
                )
                association_candidates.append(
                    (association_cost, hypothesis_index, detection_index)
                )

        for _, hypothesis_index, detection_index in sorted(
            association_candidates,
            key=lambda item: item[0],
        ):
            if (
                hypothesis_index in matched_hypotheses
                or detection_index in matched_detections
            ):
                continue
            hypothesis = self.reacquisition_hypotheses[hypothesis_index]
            (
                _,
                detection,
                ground_contact,
                size,
                body_owner_id,
            ) = eligible[detection_index]
            center = np.array(self._center(detection.bbox), dtype=np.float64)
            elapsed = max(1, frame_index - hypothesis.previous_frame)
            instant_velocity = (center - hypothesis.previous_position) / elapsed
            instant_acceleration = instant_velocity - hypothesis.velocity
            hypothesis.acceleration = (
                hypothesis.acceleration * 0.55 + instant_acceleration * 0.45
            )
            hypothesis.velocity = (
                hypothesis.velocity * 0.38 + instant_velocity * 0.62
            )
            hypothesis.previous_position = center.copy()
            hypothesis.previous_frame = frame_index
            hypothesis.position = center
            hypothesis.last_frame = frame_index
            hypothesis.hits += 1
            hypothesis.strong_hits += int(
                float(detection.confidence or 0.0) >= 0.28
            )
            hypothesis.ground_hits += int(ground_contact)
            hypothesis.confidence_sum += float(detection.confidence or 0.0)
            hypothesis.size_sum += size
            hypothesis.last_detection = detection
            if hypothesis.body_owner_id is None and body_owner_id is not None:
                hypothesis.body_owner_id = body_owner_id
            hypothesis.max_displacement = max(
                hypothesis.max_displacement,
                float(np.linalg.norm(center - hypothesis.origin)),
            )
            matched_hypotheses.add(hypothesis_index)
            matched_detections.add(detection_index)

        for detection_index, (
            _,
            detection,
            ground_contact,
            size,
            body_owner_id,
        ) in enumerate(eligible):
            if detection_index in matched_detections:
                continue
            center = np.array(self._center(detection.bbox), dtype=np.float64)
            confidence = float(detection.confidence or 0.0)
            self.reacquisition_hypotheses.append(
                BallReacquisitionHypothesis(
                    hypothesis_id=self.next_reacquisition_hypothesis_id,
                    position=center.copy(),
                    origin=center.copy(),
                    first_frame=frame_index,
                    last_frame=frame_index,
                    previous_position=center.copy(),
                    previous_frame=frame_index,
                    strong_hits=int(confidence >= 0.28),
                    ground_hits=int(ground_contact),
                    confidence_sum=confidence,
                    size_sum=size,
                    last_detection=detection,
                    body_owner_id=body_owner_id,
                )
            )
            self.next_reacquisition_hypothesis_id += 1
            self.challengers_started += 1

        def hypothesis_rank(hypothesis: BallReacquisitionHypothesis) -> float:
            average_confidence = hypothesis.confidence_sum / max(hypothesis.hits, 1)
            span = max(1, hypothesis.last_frame - hypothesis.first_frame)
            movement_rate = hypothesis.max_displacement / span
            acceleration = float(np.linalg.norm(hypothesis.acceleration))
            return (
                hypothesis.hits * 0.52
                + average_confidence * 2.2
                + min(2.0, movement_rate / max(frame_width * 0.006, 1.0))
                - min(1.5, acceleration / max(frame_width * 0.055, 1.0))
            )

        if len(self.reacquisition_hypotheses) > 8:
            self.reacquisition_hypotheses = sorted(
                self.reacquisition_hypotheses,
                key=hypothesis_rank,
                reverse=True,
            )[:8]
        self.peak_reacquisition_hypotheses = max(
            self.peak_reacquisition_hypotheses,
            len(self.reacquisition_hypotheses),
        )
        promotable: list[BallReacquisitionHypothesis] = []
        for hypothesis in self.reacquisition_hypotheses:
            span = hypothesis.last_frame - hypothesis.first_frame
            average_confidence = (
                hypothesis.confidence_sum / max(hypothesis.hits, 1)
            )
            acceleration = float(np.linalg.norm(hypothesis.acceleration))
            motion_confirmed = (
                hypothesis.max_displacement >= max(30.0, frame_width * 0.014)
                and acceleration <= max(150.0, frame_width * 0.080)
            )
            airborne_confirmed = (
                stale_flight
                and hypothesis.hits >= 3
                and span >= 4
                and average_confidence >= 0.15
                and motion_confirmed
                and (
                    hypothesis.ground_hits >= 1
                    or hypothesis.strong_hits >= 1
                    or hypothesis.max_displacement
                    >= max(90.0, frame_width * 0.055)
                )
            )
            dormant_confirmed = (
                not stale_flight
                and hypothesis.hits >= 3
                and hypothesis.strong_hits >= 2
                and span >= 4
                and motion_confirmed
            )
            if (
                (airborne_confirmed or dormant_confirmed)
                and hypothesis.last_detection is not None
                and hypothesis.last_frame == frame_index
            ):
                promotable.append(hypothesis)

        if not promotable:
            return None
        winner = max(promotable, key=hypothesis_rank)
        self.reacquisition_competing_rejections += max(0, len(promotable) - 1)
        self.challenger_position = winner.position.copy()
        self.challenger_origin = winner.origin.copy()
        self.challenger_first_frame = winner.first_frame
        self.challenger_last_frame = winner.last_frame
        self.challenger_hits = winner.hits
        self.challenger_strong_hits = winner.strong_hits
        self.challenger_max_displacement = winner.max_displacement
        self.challenger_previous_position = winner.previous_position.copy()
        self.challenger_previous_frame = winner.previous_frame
        self.challenger_velocity = winner.velocity.copy()
        self.challenger_ground_hits = winner.ground_hits
        self.challenger_promotions += 1
        if stale_flight:
            self.airborne_challenger_promotions += 1
        return winner.last_detection

    def _prepare_challenger_reinitialization(self) -> None:
        origin = (
            self.challenger_origin.copy()
            if self.challenger_origin is not None
            else None
        )
        origin_frame = self.challenger_first_frame
        reinitialization_velocity = self.challenger_velocity.copy()
        self.position = None
        self.flow_position = None
        self.observed_position = None
        self.observed_velocity[:] = 0.0
        self.observed_position_frame = -1
        self.kalman_state = None
        self.predicted_state = None
        self.predicted_covariance = None
        self.velocity[:] = 0.0
        self.last_frame = -1
        self.last_observed_frame = -1
        self.pitch_position = None
        self.pitch_velocity[:] = 0.0
        self.last_pitch_frame = -1
        self.position_3d = None
        self.velocity_3d[:] = 0.0
        self.last_3d_frame = -1
        self.trajectory_3d_confidence = 0.0
        self.flight_start_frame = -1
        self.confidence = 0.0
        self.current_interpolation_streak = 0
        self.flight_mode = False
        self.flight_clear_hits = 0
        self.selected_measurement_details = {}
        self.pending_position = origin
        self.pending_pitch_position = None
        self.pending_frame = origin_frame
        self.reinitialization_velocity = reinitialization_velocity
        self.challenger_reinitialization_ready = True
        self._clear_challenger()

    def _clear_challenger(self) -> None:
        self.reacquisition_hypotheses.clear()
        self.challenger_position = None
        self.challenger_origin = None
        self.challenger_first_frame = -1
        self.challenger_last_frame = -1
        self.challenger_hits = 0
        self.challenger_strong_hits = 0
        self.challenger_max_displacement = 0.0
        self.challenger_previous_position = None
        self.challenger_previous_frame = -1
        self.challenger_velocity[:] = 0.0
        self.challenger_ground_hits = 0

    def _challenger_suppresses_prediction(self, frame_index: int) -> bool:
        return (
            bool(self.reacquisition_hypotheses)
            and self.last_observed_frame >= 0
            and frame_index - self.last_observed_frame
            > self.max_interpolation_frames
        )

    def _trajectory_prediction(self, frame_index: int) -> np.ndarray | None:
        if self.observed_position is None or self.observed_position_frame < 0:
            return None
        elapsed = max(0, frame_index - self.observed_position_frame)
        return self.observed_position + self.observed_velocity * elapsed

    def _trajectory_gate(self, frame_index: int, frame_width: int) -> float:
        elapsed = max(1, frame_index - max(self.observed_position_frame, 0))
        return max(72.0, frame_width * 0.045) + elapsed * max(
            1.5,
            frame_width * 0.0018,
        )

    def _update_observed_trajectory(
        self,
        frame_index: int,
        measurement: np.ndarray,
        frame_width: int,
    ) -> None:
        if self.observed_position is not None and self.observed_position_frame >= 0:
            elapsed = max(1, frame_index - self.observed_position_frame)
            observed_velocity = (measurement - self.observed_position) / elapsed
            speed = float(np.linalg.norm(observed_velocity))
            maximum_speed = max(32.0, frame_width * 0.08)
            if speed > maximum_speed:
                observed_velocity *= maximum_speed / speed
            blend = 0.52 if elapsed <= self.max_interpolation_frames else 0.36
            self.observed_velocity = (
                self.observed_velocity * (1.0 - blend)
                + observed_velocity * blend
            )
        else:
            self.observed_velocity[:] = 0.0
        self.observed_position = measurement.copy()
        self.observed_position_frame = frame_index

    def _predict_3d(self, frame_index: int) -> np.ndarray | None:
        if self.position_3d is None or self.last_3d_frame < 0:
            return None
        elapsed = max(0, frame_index - self.last_3d_frame)
        predicted = self.position_3d + self.velocity_3d * elapsed
        if self.flight_mode and elapsed > 0:
            predicted[2] -= (
                0.5 * self.gravity_cm_per_frame2 * elapsed * elapsed
            )
        predicted[0] = float(np.clip(predicted[0], 0.0, PITCH_LENGTH_CM))
        predicted[1] = float(np.clip(predicted[1], 0.0, PITCH_WIDTH_CM))
        predicted[2] = max(0.0, float(predicted[2]))
        return predicted

    def _project_3d(
        self,
        point_3d: np.ndarray | None,
        projection: np.ndarray | None,
    ) -> np.ndarray | None:
        if point_3d is None or projection is None:
            return None
        homogeneous = projection @ np.array(
            [point_3d[0], point_3d[1], point_3d[2], 1.0],
            dtype=np.float64,
        )
        if (
            not np.all(np.isfinite(homogeneous))
            or abs(float(homogeneous[2])) <= 1e-9
        ):
            return None
        return (homogeneous[:2] / homogeneous[2]).astype(np.float64)

    def _backproject_3d(
        self,
        image_point: np.ndarray,
        height_cm: float,
        projection: np.ndarray,
    ) -> np.ndarray | None:
        u, v = float(image_point[0]), float(image_point[1])
        first = projection[0] - u * projection[2]
        second = projection[1] - v * projection[2]
        system = np.array(
            [[first[0], first[1]], [second[0], second[1]]],
            dtype=np.float64,
        )
        target = -np.array(
            [
                first[2] * height_cm + first[3],
                second[2] * height_cm + second[3],
            ],
            dtype=np.float64,
        )
        if abs(float(np.linalg.det(system))) <= 1e-9:
            return None
        xy = np.linalg.solve(system, target)
        if not np.all(np.isfinite(xy)):
            return None
        if not (
            -1200.0 <= xy[0] <= PITCH_LENGTH_CM + 1200.0
            and -1200.0 <= xy[1] <= PITCH_WIDTH_CM + 1200.0
        ):
            return None
        return np.array([xy[0], xy[1], height_cm], dtype=np.float64)

    def _update_3d_state(
        self,
        frame_index: int,
        measurement: np.ndarray,
        measured_pitch: np.ndarray | None,
        predicted_3d: np.ndarray | None,
        camera_projection: np.ndarray | None,
    ) -> None:
        if not self.flight_mode:
            self._update_pitch_state(frame_index, measured_pitch)
            if self.pitch_position is not None:
                self.position_3d = np.array(
                    [self.pitch_position[0], self.pitch_position[1], 0.0],
                    dtype=np.float64,
                )
                self.velocity_3d = np.array(
                    [self.pitch_velocity[0], self.pitch_velocity[1], 0.0],
                    dtype=np.float64,
                )
                self.last_3d_frame = frame_index
                self.trajectory_3d_confidence = max(
                    self.trajectory_3d_confidence * 0.75,
                    0.78,
                )
                self.three_d_observed_frames += 1
            return

        expected = predicted_3d
        if expected is None and self.pitch_position is not None:
            expected = np.array(
                [
                    self.pitch_position[0],
                    self.pitch_position[1],
                    max(35.0, float(self.velocity_3d[2])),
                ],
                dtype=np.float64,
            )
        measurement_3d = None
        residual_cm = float("inf")
        if camera_projection is not None and expected is not None:
            expected_height = max(0.0, float(expected[2]))
            best: tuple[float, np.ndarray] | None = None
            for height_cm in np.linspace(0.0, 3200.0, 65):
                candidate = self._backproject_3d(
                    measurement,
                    float(height_cm),
                    camera_projection,
                )
                if candidate is None:
                    continue
                horizontal_residual = float(
                    np.linalg.norm(candidate[:2] - expected[:2])
                )
                vertical_weight = 0.04 if expected_height < 80.0 else 0.18
                score = horizontal_residual + abs(
                    float(candidate[2]) - expected_height
                ) * vertical_weight
                if best is None or score < best[0]:
                    best = (score, candidate)
            if best is not None:
                residual_cm = best[0]
                measurement_3d = best[1]
                if not bool(self.selected_measurement_details.get("ground_contact")):
                    measurement_3d[2] = max(35.0, measurement_3d[2])

        if measurement_3d is None:
            measurement_3d = (
                expected.copy()
                if expected is not None
                else np.array(
                    [
                        measured_pitch[0] if measured_pitch is not None else 0.0,
                        measured_pitch[1] if measured_pitch is not None else 0.0,
                        35.0,
                    ],
                    dtype=np.float64,
                )
            )
            residual_cm = 1800.0

        projected_measurement = self._project_3d(
            measurement_3d,
            camera_projection,
        )
        projection_residual_px = (
            float(np.linalg.norm(projected_measurement - measurement))
            if projected_measurement is not None
            else float("inf")
        )
        if projection_residual_px > 8.0:
            # Never publish a 3D state that cannot reproduce the observation
            # which created it. Keep it as low-confidence analytics metadata;
            # the image tracker remains authoritative for rendering.
            self.three_d_observation_projection_rejections += 1
            residual_cm = max(residual_cm, 2200.0)

        previous = self.position_3d.copy() if self.position_3d is not None else None
        previous_frame = self.last_3d_frame
        if previous is not None and previous_frame >= 0:
            elapsed = max(1, frame_index - previous_frame)
            observed_velocity = (measurement_3d - previous) / elapsed
            horizontal_speed = float(np.linalg.norm(observed_velocity[:2]))
            if horizontal_speed > 320.0:
                observed_velocity[:2] *= 320.0 / horizontal_speed
            observed_velocity[2] = float(
                np.clip(observed_velocity[2], -120.0, 120.0)
            )
            blend = 0.58 if residual_cm <= 900.0 else 0.34
            self.velocity_3d = (
                self.velocity_3d * (1.0 - blend)
                + observed_velocity * blend
            )
        elif self.velocity_3d[2] <= 0.0:
            self.velocity_3d[2] = 48.0

        # The selected point lies on the current image ray. Blending XYZ with
        # a previous ray creates a point that projects to neither observation,
        # which was the source of the marker jump near distant players.
        self.position_3d = measurement_3d.copy()
        self.position_3d[0] = float(
            np.clip(self.position_3d[0], 0.0, PITCH_LENGTH_CM)
        )
        self.position_3d[1] = float(
            np.clip(self.position_3d[1], 0.0, PITCH_WIDTH_CM)
        )
        self.position_3d[2] = max(0.0, float(self.position_3d[2]))
        self.pitch_position = self.position_3d[:2].copy()
        self.pitch_velocity = self.velocity_3d[:2].copy()
        self.last_pitch_frame = frame_index
        self.last_3d_frame = frame_index
        self.maximum_height_cm = max(
            self.maximum_height_cm,
            float(self.position_3d[2]),
        )
        self.trajectory_3d_confidence = float(
            np.clip(
                np.exp(-residual_cm / 1400.0)
                * np.exp(-projection_residual_px / 10.0),
                0.08,
                0.96,
            )
        )
        self.three_d_observed_frames += 1

    def _commit_3d_prediction(
        self,
        frame_index: int,
        predicted_3d: np.ndarray,
    ) -> None:
        elapsed = max(1, frame_index - max(self.last_3d_frame, 0))
        self.position_3d = predicted_3d.copy()
        self.velocity_3d[2] -= self.gravity_cm_per_frame2 * elapsed
        self.pitch_position = self.position_3d[:2].copy()
        self.pitch_velocity = self.velocity_3d[:2].copy()
        self.last_pitch_frame = frame_index
        self.last_3d_frame = frame_index
        self.maximum_height_cm = max(
            self.maximum_height_cm,
            float(self.position_3d[2]),
        )
        self.trajectory_3d_confidence *= 0.985
        self.three_d_predicted_frames += 1

    def _correct_kalman(
        self,
        measurement: np.ndarray,
        detection_confidence: float,
    ) -> None:
        if self.predicted_state is None or self.predicted_covariance is None:
            return
        observation = np.array(
            [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]],
            dtype=np.float64,
        )
        noise_value = 24.0 - min(1.0, max(0.0, detection_confidence)) * 16.0
        measurement_noise = np.eye(2, dtype=np.float64) * noise_value
        innovation = measurement - observation @ self.predicted_state
        innovation_covariance = (
            observation @ self.predicted_covariance @ observation.T
            + measurement_noise
        )
        gain = (
            self.predicted_covariance
            @ observation.T
            @ np.linalg.inv(innovation_covariance)
        )
        self.kalman_state = self.predicted_state + gain @ innovation
        identity = np.eye(4, dtype=np.float64)
        self.kalman_covariance = (
            identity - gain @ observation
        ) @ self.predicted_covariance
        self.position = self.kalman_state[:2].copy()
        self.velocity = self.kalman_state[2:].copy()

    def _expire_lost_track(self, frame_index: int) -> None:
        if self.position is None or self.last_observed_frame < 0:
            return
        if frame_index - self.last_observed_frame <= self.max_reacquisition_frames:
            return
        self.position = None
        self.flow_position = None
        self.observed_position = None
        self.observed_velocity[:] = 0.0
        self.observed_position_frame = -1
        self.kalman_state = None
        self.predicted_state = None
        self.predicted_covariance = None
        self.velocity[:] = 0.0
        self.last_frame = -1
        self.last_observed_frame = -1
        self.pending_position = None
        self.pending_pitch_position = None
        self.pending_frame = -1
        self.pitch_position = None
        self.pitch_velocity[:] = 0.0
        self.last_pitch_frame = -1
        self.position_3d = None
        self.velocity_3d[:] = 0.0
        self.last_3d_frame = -1
        self.trajectory_3d_confidence = 0.0
        self.flight_start_frame = -1
        self.reinitialization_velocity[:] = 0.0
        self.confidence = 0.0
        self.current_interpolation_streak = 0
        self.flight_mode = False
        self.flight_clear_hits = 0
        self.selected_measurement_details = {}
        self.challenger_reinitialization_ready = False
        self._clear_challenger()
        self.expired_track_resets += 1

    def _select_measurement(
        self,
        frame_index: int,
        detections: list[AnalysisObject],
        predicted: np.ndarray | None,
        players: list[AnalysisObject],
        frame_width: int,
        pitch_transform: Any | None,
    ) -> AnalysisObject | None:
        plausible: list[tuple[float, AnalysisObject, dict[str, Any]]] = []
        frame_gap = max(1, frame_index - max(self.last_frame, 0))
        observation_gap = max(
            1,
            frame_index - max(self.last_observed_frame, 0),
        )
        dormant = (
            self.last_observed_frame >= 0
            and observation_gap > self.max_interpolation_frames
        )
        self.selected_measurement_details = {}
        predicted_pitch = None
        trajectory_prediction = self._trajectory_prediction(frame_index)
        trajectory_gate = self._trajectory_gate(frame_index, frame_width)
        if self.pitch_position is not None:
            pitch_gap = max(1, frame_index - max(self.last_pitch_frame, 0))
            predicted_pitch = self.pitch_position + self.pitch_velocity * pitch_gap
        for detection in detections:
            width = max(1.0, detection.bbox[2] - detection.bbox[0])
            height = max(1.0, detection.bbox[3] - detection.bbox[1])
            aspect = width / height
            if not 0.45 <= aspect <= 1.75:
                continue
            center = np.array(self._center(detection.bbox), dtype=np.float64)
            if max(width, height) > max(28.0, frame_width * 0.045):
                self.implausible_size_rejections += 1
                continue
            near_foot, overlaps_body = self._player_relation(center, players, frame_width)
            ground_contact = self._ground_contact(center, players, frame_width)
            body_pass_through = False
            measured_pitch = self._pitch_measurement(center, pitch_transform)
            proximity_bonus = 0.12 if near_foot else 0.0
            trajectory_distance = None
            if predicted is None:
                if overlaps_body:
                    self.player_body_rejections += 1
                    continue
                score = float(detection.confidence or 0.0) + proximity_bonus
                distance = None
                gate = None
                metric_distance = None
                metric_gate = None
                metric_rejected = False
                if measured_pitch is not None and self.pitch_position is not None:
                    elapsed = max(1, frame_index - self.last_pitch_frame)
                    reachable = 550.0 + elapsed * 260.0
                    metric_distance = float(np.linalg.norm(measured_pitch - self.pitch_position))
                    if metric_distance > reachable:
                        self.metric_jump_rejections += 1
                        continue
                    score += max(0.0, 1.0 - metric_distance / reachable) * 0.25
            else:
                gate = max(
                    48.0,
                    frame_width * 0.04,
                    float(np.linalg.norm(self.velocity)) * frame_gap * 2.8 + 24.0,
                )
                if dormant:
                    gate = max(
                        gate,
                        min(
                            frame_width * 0.28,
                            48.0
                            + observation_gap
                            * max(6.0, frame_width * 0.009),
                        ),
                    )
                distance = float(np.linalg.norm(center - predicted))
                metric_distance = None
                metric_gate = None
                if measured_pitch is not None and predicted_pitch is not None:
                    metric_distance = float(np.linalg.norm(measured_pitch - predicted_pitch))
                    metric_gate = max(
                        380.0,
                        float(np.linalg.norm(self.pitch_velocity)) * frame_gap * 2.4 + 220.0,
                    )
                image_rejected = distance > gate
                metric_rejected = (
                    metric_distance is not None
                    and metric_gate is not None
                    and metric_distance > metric_gate
                )
                # A ground-plane projection cannot validate a teleport in the
                # camera image. This guard prevents a false foot candidate from
                # stealing the track while the real ball is airborne.
                if image_rejected:
                    self.rejected_gate += 1
                    if metric_rejected:
                        self.metric_jump_rejections += 1
                    continue
                trajectory_distance = (
                    float(np.linalg.norm(center - trajectory_prediction))
                    if trajectory_prediction is not None
                    else None
                )
                if (
                    self.flight_mode
                    and trajectory_distance is not None
                    and trajectory_distance > trajectory_gate
                ):
                    self.airborne_trajectory_rejections += 1
                    continue
                image_consistent = distance <= gate * 0.68
                if overlaps_body:
                    body_pass_through = (
                        self.flight_mode
                        and image_consistent
                        and distance <= max(24.0, gate * 0.45)
                    )
                    if not body_pass_through:
                        self.player_body_rejections += 1
                        continue
                mahalanobis = 0.0
                if self.predicted_covariance is not None and not image_rejected:
                    covariance = self.predicted_covariance[:2, :2] + np.eye(2) * 12.0
                    residual = center - predicted
                    mahalanobis = float(residual.T @ np.linalg.inv(covariance) @ residual)
                    if mahalanobis > 16.0 and (
                        metric_distance is None or metric_rejected
                    ):
                        self.mahalanobis_rejections += 1
                        continue
                score = (
                    max(-0.25, 1.0 - distance / gate) * 0.35
                    - min(1.0, mahalanobis / 16.0) * 0.15
                    + float(detection.confidence or 0.0) * 0.35
                    + proximity_bonus
                )
                if metric_distance is not None and metric_gate is not None:
                    if (
                        (self.flight_mode or metric_rejected)
                        and image_consistent
                        and not near_foot
                    ):
                        score += 0.18
                    else:
                        score += max(
                            -0.35,
                            1.0 - metric_distance / metric_gate,
                        ) * 0.65
                if trajectory_distance is not None and self.flight_mode:
                    score += max(
                        -0.25,
                        1.0 - trajectory_distance / trajectory_gate,
                    ) * 0.55
            details = {
                "near_foot": near_foot,
                "ground_contact": ground_contact,
                "dormant": dormant,
                "image_distance": distance,
                "image_gate": gate,
                "metric_distance": metric_distance,
                "metric_gate": metric_gate,
                "metric_conflict": bool(metric_rejected),
                "image_consistent": bool(
                    distance is None
                    or gate is None
                    or distance <= gate * 0.68
                ),
                "body_pass_through": body_pass_through,
                "trajectory_distance": trajectory_distance,
                "trajectory_gate": trajectory_gate,
            }
            plausible.append((score, detection, details))
        if not plausible:
            return None
        plausible.sort(key=lambda item: item[0], reverse=True)
        if len(plausible) > 1 and plausible[0][0] - plausible[1][0] < 0.07:
            self.ambiguous_reacquisitions += 1
            if predicted is None:
                return None
        self.selected_measurement_details = plausible[0][2]
        if bool(self.selected_measurement_details.get("body_pass_through", False)):
            self.airborne_body_pass_throughs += 1
        return plausible[0][1]

    def _update_flight_state(
        self,
        frame_index: int,
        measured_pitch: np.ndarray | None,
    ) -> None:
        details = self.selected_measurement_details
        near_foot = bool(details.get("near_foot", False))
        ground_contact = bool(details.get("ground_contact", False))
        metric_conflict = bool(details.get("metric_conflict", False))
        image_consistent = bool(details.get("image_consistent", False))
        dormant = bool(details.get("dormant", False))
        should_enter_flight = (
            image_consistent
            and not near_foot
            and (metric_conflict or dormant)
        )
        if should_enter_flight and not self.flight_mode:
            self.flight_mode = True
            self.flight_clear_hits = 0
            self.flight_start_frame = frame_index
            if self.position_3d is not None:
                self.velocity_3d[2] = max(self.velocity_3d[2], 48.0)
            self.airborne_entries += 1
        if self.flight_mode:
            self.airborne_observed_frames += 1
            if metric_conflict and image_consistent:
                self.metric_conflict_acceptances += 1
            if ground_contact and image_consistent:
                self.flight_clear_hits += 1
                if self.flight_clear_hits >= 3:
                    self.flight_mode = False
                    self.flight_clear_hits = 0
                    self.pitch_position = (
                        measured_pitch.copy() if measured_pitch is not None else None
                    )
                    self.pitch_velocity[:] = 0.0
                    self.last_pitch_frame = (
                        frame_index if measured_pitch is not None else -1
                    )
                    self.ground_reacquisitions += 1
                    self.ground_contact_confirmations += 1
                    self.flight_start_frame = -1
                    if self.position_3d is not None:
                        self.position_3d[2] = 0.0
                    self.velocity_3d[2] = 0.0
            else:
                self.flight_clear_hits = 0

    def _confirm_initial_measurement(
        self,
        frame_index: int,
        measurement: np.ndarray,
        pitch_measurement: np.ndarray | None,
        frame_width: int,
    ) -> bool:
        if self.pending_position is None or frame_index - self.pending_frame > 3:
            self.pending_position = measurement
            self.pending_pitch_position = pitch_measurement
            self.pending_frame = frame_index
            return False
        elapsed = max(1, frame_index - self.pending_frame)
        image_gate = max(52.0, frame_width * 0.04) * elapsed
        image_confirmed = float(np.linalg.norm(measurement - self.pending_position)) <= image_gate
        pitch_confirmed = False
        if pitch_measurement is not None and self.pending_pitch_position is not None:
            pitch_confirmed = (
                float(np.linalg.norm(pitch_measurement - self.pending_pitch_position))
                <= 520.0 * elapsed
            )
        confirmed = image_confirmed or pitch_confirmed
        self.pending_position = measurement
        self.pending_pitch_position = pitch_measurement
        self.pending_frame = frame_index
        return confirmed

    def _object(self, raw_track_id: int | None, predicted: bool) -> AnalysisObject:
        assert self.position is not None
        half = self.size / 2.0
        return AnalysisObject(
            track_id=self.track_id,
            raw_track_id=raw_track_id,
            class_name="ball",
            bbox=[
                float(self.position[0] - half[0]),
                float(self.position[1] - half[1]),
                float(self.position[0] + half[0]),
                float(self.position[1] + half[1]),
            ],
            confidence=round(self.confidence, 4),
            is_predicted=predicted,
            role_name="ball",
            pitch_position=(
                (
                    float(self.position_3d[0]),
                    float(self.position_3d[1]),
                )
                if self.position_3d is not None
                else None
            ),
            pitch_velocity=(
                (float(self.pitch_velocity[0]), float(self.pitch_velocity[1]))
                if self.pitch_position is not None
                else None
            ),
            height_cm=(
                float(self.position_3d[2])
                if self.position_3d is not None
                else 0.0
            ),
            trajectory_3d_confidence=self.trajectory_3d_confidence,
        )

    def _record_pitch(
        self,
        frame_index: int,
        ball: AnalysisObject,
        pitch_transform: Any | None,
        predicted: bool,
    ) -> None:
        point = ball.pitch_position
        if point is None and pitch_transform is not None and not self.flight_mode:
            point = pitch_transform(self._center(ball.bbox))
        if point is None:
            return
        self.pitch_path.append(
            {
                "frame": frame_index,
                "x": round(float(point[0]), 2),
                "y": round(float(point[1]), 2),
                "predicted": predicted,
                "confidence": round(self.confidence, 4),
                "height_cm": round(float(ball.height_cm), 2),
                "trajectory_3d_confidence": round(
                    float(ball.trajectory_3d_confidence),
                    4,
                ),
            }
        )

    def _record_image(
        self,
        frame_index: int,
        ball: AnalysisObject,
        predicted: bool,
    ) -> None:
        center = self._center(ball.bbox)
        self.image_path.append(
            {
                "frame": frame_index,
                "x": round(float(center[0]), 2),
                "y": round(float(center[1]), 2),
                "predicted": predicted,
                "confidence": round(self.confidence, 4),
                "airborne": self.flight_mode,
                "height_cm": round(float(ball.height_cm), 2),
                "trajectory_3d_confidence": round(
                    float(ball.trajectory_3d_confidence),
                    4,
                ),
            }
        )

    def _player_relation(
        self,
        center: np.ndarray,
        players: list[AnalysisObject],
        frame_width: int,
    ) -> tuple[bool, bool]:
        foot_threshold = max(28.0, frame_width * 0.022)
        near_foot = False
        for player in players:
            x1, y1, x2, y2 = player.bbox
            player_height = max(1.0, y2 - y1)
            player_width = max(1.0, x2 - x1)
            foot = np.array([(x1 + x2) / 2, y2], dtype=np.float64)
            if float(np.linalg.norm(center - foot)) <= max(
                foot_threshold,
                player_height * 0.34,
            ):
                near_foot = True
            torso_left = x1 + player_width * 0.12
            torso_right = x2 - player_width * 0.12
            torso_bottom = y1 + player_height * 0.72
            if torso_left <= center[0] <= torso_right and y1 <= center[1] <= torso_bottom:
                return near_foot, True
        return near_foot, False

    def _player_body_owner_id(
        self,
        center: np.ndarray,
        players: list[AnalysisObject],
    ) -> int | None:
        for player in players:
            x1, y1, x2, y2 = player.bbox
            player_height = max(1.0, y2 - y1)
            player_width = max(1.0, x2 - x1)
            if (
                x1 + player_width * 0.12
                <= center[0]
                <= x2 - player_width * 0.12
                and y1
                <= center[1]
                <= y1 + player_height * 0.72
            ):
                owner_id = (
                    player.raw_track_id
                    if player.raw_track_id is not None
                    else player.track_id
                )
                return int(owner_id) if owner_id is not None else None
        return None

    def _ground_contact(
        self,
        center: np.ndarray,
        players: list[AnalysisObject],
        frame_width: int,
    ) -> bool:
        horizontal_floor = max(12.0, frame_width * 0.008)
        vertical_floor = max(10.0, frame_width * 0.006)
        for player in players:
            x1, y1, x2, y2 = player.bbox
            player_height = max(1.0, y2 - y1)
            foot_x = (x1 + x2) / 2.0
            if (
                abs(float(center[0]) - foot_x)
                <= max(horizontal_floor, player_height * 0.14)
                and abs(float(center[1]) - y2)
                <= max(vertical_floor, player_height * 0.16)
            ):
                return True
        return False

    def _pitch_measurement(
        self,
        center: np.ndarray,
        pitch_transform: Any | None,
    ) -> np.ndarray | None:
        if pitch_transform is None:
            return None
        point = pitch_transform((float(center[0]), float(center[1])))
        if point is None:
            return None
        return np.array(point, dtype=np.float64)

    def _update_pitch_state(
        self,
        frame_index: int,
        measurement: np.ndarray | None,
    ) -> None:
        if measurement is None:
            return
        if self.pitch_position is not None and self.last_pitch_frame >= 0:
            elapsed = max(1, frame_index - self.last_pitch_frame)
            observed_velocity = (measurement - self.pitch_position) / elapsed
            speed = float(np.linalg.norm(observed_velocity))
            if speed > 350.0:
                observed_velocity *= 350.0 / speed
            self.pitch_velocity = self.pitch_velocity * 0.58 + observed_velocity * 0.42
        else:
            self.pitch_velocity[:] = 0.0
        self.pitch_position = measurement.copy()
        self.last_pitch_frame = frame_index

    def _predict_pitch_state(self, frame_index: int) -> None:
        if self.pitch_position is None or self.last_pitch_frame < 0:
            return
        elapsed = max(1, frame_index - self.last_pitch_frame)
        self.pitch_position = self.pitch_position + self.pitch_velocity * elapsed
        self.pitch_velocity *= 0.94
        self.last_pitch_frame = frame_index

    def _center(self, bbox: list[float]) -> tuple[float, float]:
        return ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)


# Backward-compatible imports for existing integrations and saved test suites.
BallTrackerV3 = BallTrackerV4
BallTrackerV2 = BallTrackerV4


class PossessionTracker:
    def __init__(
        self,
        confirmation_frames: int = 3,
        hold_frames: int = 4,
        fps: float = 30.0,
        maximum_link_gap_seconds: float = 3.0,
    ) -> None:
        self.confirmation_frames = confirmation_frames
        self.hold_frames = hold_frames
        self.fps = max(1.0, fps)
        self.maximum_link_gap_frames = max(
            1,
            int(round(maximum_link_gap_seconds * max(1.0, fps))),
        )
        self.current_player: int | None = None
        self.current_team: int | None = None
        self.candidate_player: int | None = None
        self.candidate_hits = 0
        self.last_observed_frame = -1
        self.team_frames: dict[int, int] = {1: 0, 2: 0}
        self.player_frames: dict[int, int] = {}
        self.transitions = 0
        self.unassigned_frames = 0
        self.completed_passes = 0
        self.turnovers = 0
        self.far_ball_releases = 0
        self.predicted_holds = 0
        self.stale_possession_resets = 0
        self.events: list[dict[str, Any]] = []
        self.release_frame: int | None = None
        self.release_pitch: tuple[float, float] | None = None
        self.last_control_pitch: tuple[float, float] | None = None

    def update(
        self,
        frame_index: int,
        players: list[AnalysisObject],
        balls: list[AnalysisObject],
        team_by_track: dict[int, int],
        pitch_transform: Any,
    ) -> tuple[int | None, int | None]:
        if not balls:
            if (
                self.current_team is not None
                and self.release_frame is None
                and frame_index - self.last_observed_frame <= self.hold_frames
            ):
                self._count_current()
                return self.current_player, self.current_team
            self.unassigned_frames += 1
            return None, None
        ball = balls[0]
        ball_center = (
            (ball.bbox[0] + ball.bbox[2]) / 2,
            (ball.bbox[1] + ball.bbox[3]) / 2,
        )
        ball_pitch = (
            ball.pitch_position
            if ball.pitch_position is not None
            else pitch_transform(ball_center)
        )
        if (
            ball.height_cm >= 100.0
            or (
                ball.height_cm >= 65.0
                and ball.trajectory_3d_confidence >= 0.10
            )
        ):
            if self.current_player is not None and self.release_frame is None:
                self.release_frame = frame_index
                self.release_pitch = ball_pitch or self.last_control_pitch
                self.far_ball_releases += 1
            self.candidate_player = None
            self.candidate_hits = 0
            if not ball.is_predicted:
                self.last_observed_frame = frame_index
            self.unassigned_frames += 1
            return None, None
        if ball.is_predicted:
            if (
                self.current_team is not None
                and self.release_frame is None
                and frame_index - self.last_observed_frame <= min(3, self.hold_frames)
            ):
                self._count_current()
                self.predicted_holds += 1
                return self.current_player, self.current_team
            self.unassigned_frames += 1
            return None, None
        nearest = self._nearest_player(ball, players, pitch_transform)
        control_radius_cm = self._control_radius_cm(ball)
        if nearest is None or nearest[0] > control_radius_cm:
            if self.current_player is not None and self.release_frame is None:
                self.release_frame = frame_index
                self.release_pitch = ball_pitch or self.last_control_pitch
                self.far_ball_releases += 1
            self.candidate_player = None
            self.candidate_hits = 0
            self.last_observed_frame = frame_index
            self.unassigned_frames += 1
            return None, None
        player = nearest[1]
        if self.candidate_player == player.track_id:
            self.candidate_hits += 1
        else:
            self.candidate_player = player.track_id
            self.candidate_hits = 1
        if self.candidate_hits < self.confirmation_frames:
            self.last_observed_frame = frame_index
            if self.current_player == player.track_id and self.release_frame is None:
                self._count_current()
                return self.current_player, self.current_team
            self.unassigned_frames += 1
            return None, None
        if self.candidate_hits >= self.confirmation_frames:
            if (
                self.release_frame is not None
                and frame_index - self.release_frame > self.maximum_link_gap_frames
            ):
                self.current_player = None
                self.current_team = None
                self.release_pitch = None
                self.last_control_pitch = None
                self.stale_possession_resets += 1
            if self.current_player is not None and self.current_player != player.track_id:
                verified_transition = self._record_transition(
                    frame_index=frame_index,
                    next_player=player.track_id,
                    next_team=team_by_track.get(player.track_id),
                    ball_pitch=ball_pitch,
                )
                if verified_transition:
                    self.transitions += 1
            self.current_player = player.track_id
            self.current_team = team_by_track.get(player.track_id)
            self.last_control_pitch = ball_pitch
            self.release_frame = None
            self.release_pitch = None
        self.last_observed_frame = frame_index
        self._count_current()
        return self.current_player, self.current_team

    def summary(self) -> dict[str, Any]:
        assigned_frames = sum(self.team_frames.values())
        total_frames = assigned_frames + self.unassigned_frames
        percentage_denominator = max(1, assigned_frames)
        assigned_coverage = assigned_frames / max(1, total_frames)
        return {
            "engine": "metric_ball_possession_and_pass_detection_v4",
            "team_1_percent": round(
                self.team_frames[1] * 100 / percentage_denominator,
                2,
            ),
            "team_2_percent": round(
                self.team_frames[2] * 100 / percentage_denominator,
                2,
            ),
            "assigned_frames": assigned_frames,
            "total_frames": total_frames,
            "assigned_coverage": round(assigned_coverage, 4),
            "quality_status": (
                "passed" if assigned_coverage >= 0.15 else "needs_review"
            ),
            "player_frames": {str(key): value for key, value in sorted(self.player_frames.items())},
            "transitions": self.transitions,
            "completed_passes": self.completed_passes,
            "turnovers": self.turnovers,
            "events": self.events,
            "unassigned_frames": self.unassigned_frames,
            "far_ball_releases": self.far_ball_releases,
            "predicted_holds": self.predicted_holds,
            "stale_possession_resets": self.stale_possession_resets,
            "confirmation_frames": self.confirmation_frames,
            "maximum_link_gap_frames": self.maximum_link_gap_frames,
            "maximum_control_distance_cm": 180.0,
            "fast_ball_control_distance_cm": 90.0,
            "airborne_release_height_cm": 65.0,
            "pass_detection": {
                "minimum_travel_m": 1.2,
                "same_team_required": True,
                "uses_canonical_track_ids": True,
                "predicted_ball_frames_can_hold_but_not_change_possession": True,
                "requires_ball_approach_direction_when_fast": True,
            },
        }

    def _record_transition(
        self,
        frame_index: int,
        next_player: int,
        next_team: int | None,
        ball_pitch: tuple[float, float] | None,
    ) -> bool:
        previous_player = self.current_player
        previous_team = self.current_team
        start_frame = self.release_frame if self.release_frame is not None else self.last_observed_frame
        start_pitch = self.release_pitch or self.last_control_pitch
        travel_m = None
        if start_pitch is not None and ball_pitch is not None:
            travel_m = float(np.hypot(
                ball_pitch[0] - start_pitch[0],
                ball_pitch[1] - start_pitch[1],
            ) / 100.0)
        duration_frames = max(1, frame_index - max(0, start_frame))
        ball_speed_mps = (
            travel_m / (duration_frames / self.fps)
            if travel_m is not None
            else None
        )
        same_team = previous_team is not None and previous_team == next_team
        event_type = "possession_change"
        confidence = 0.68
        verified = ball_speed_mps is None or ball_speed_mps <= 50.0
        reason = None
        if not verified:
            event_type = "unverified_reacquisition"
            confidence = 0.18
            reason = "implausible_ball_speed"
        elif (
            same_team
            and previous_player is not None
            and previous_player != next_player
            and travel_m is not None
            and travel_m >= 1.2
        ):
            event_type = "completed_pass"
            self.completed_passes += 1
            confidence = min(0.98, 0.72 + min(0.20, travel_m / 50.0))
        elif previous_team is not None and next_team is not None and previous_team != next_team:
            event_type = "turnover"
            self.turnovers += 1
            confidence = 0.86
        self.events.append(
            {
                "type": event_type,
                "frame": frame_index,
                "start_frame": start_frame,
                "duration_frames": duration_frames,
                "from_track_id": previous_player,
                "to_track_id": next_player,
                "from_team": previous_team,
                "to_team": next_team,
                "travel_m": round(travel_m, 3) if travel_m is not None else None,
                "ball_speed_mps": (
                    round(ball_speed_mps, 3)
                    if ball_speed_mps is not None
                    else None
                ),
                "confidence": round(confidence, 4),
                "verified": verified,
                "reason": reason,
            }
        )
        return verified

    def _count_current(self) -> None:
        if self.current_team in self.team_frames:
            self.team_frames[self.current_team] += 1
        if self.current_player is not None:
            self.player_frames[self.current_player] = self.player_frames.get(self.current_player, 0) + 1

    def _nearest_player(
        self,
        ball: AnalysisObject,
        players: list[AnalysisObject],
        pitch_transform: Any,
    ) -> tuple[float, AnalysisObject] | None:
        ball_center = ((ball.bbox[0] + ball.bbox[2]) / 2, (ball.bbox[1] + ball.bbox[3]) / 2)
        ball_pitch = (
            ball.pitch_position
            if ball.pitch_position is not None
            else pitch_transform(ball_center)
        )
        nearest: tuple[float, AnalysisObject] | None = None
        for player in players:
            if player.is_predicted or player.role_name not in ANALYTICS_ROLES:
                continue
            foot = ((player.bbox[0] + player.bbox[2]) / 2, player.bbox[3])
            player_pitch = pitch_transform(foot)
            if ball_pitch is not None and player_pitch is not None:
                distance = float(np.hypot(player_pitch[0] - ball_pitch[0], player_pitch[1] - ball_pitch[1]))
                if ball.pitch_velocity is not None:
                    velocity = np.array(ball.pitch_velocity, dtype=np.float64)
                    speed_cm_per_frame = float(np.linalg.norm(velocity))
                    to_player = np.array(
                        [
                            player_pitch[0] - ball_pitch[0],
                            player_pitch[1] - ball_pitch[1],
                        ],
                        dtype=np.float64,
                    )
                    to_player_norm = float(np.linalg.norm(to_player))
                    if speed_cm_per_frame > 1e-6 and to_player_norm > 1e-6:
                        approach_alignment = float(
                            np.dot(velocity, to_player)
                            / (speed_cm_per_frame * to_player_norm)
                        )
                        speed_mps = speed_cm_per_frame * self.fps / 100.0
                        if (
                            speed_mps >= 5.0
                            and approach_alignment < -0.20
                            and distance > 70.0
                        ):
                            continue
            else:
                player_height = max(1.0, player.bbox[3] - player.bbox[1])
                distance = float(np.hypot(foot[0] - ball_center[0], foot[1] - ball_center[1]) * 180.0 / player_height)
            if nearest is None or distance < nearest[0]:
                nearest = (distance, player)
        return nearest

    def _control_radius_cm(self, ball: AnalysisObject) -> float:
        if ball.pitch_velocity is None:
            return 180.0
        speed_mps = float(np.linalg.norm(ball.pitch_velocity)) * self.fps / 100.0
        if speed_mps >= 12.0:
            return 90.0
        if speed_mps >= 7.0:
            return 125.0
        return 180.0


class PitchRadar:
    def __init__(
        self,
        model: Any | None,
        stride: int = 12,
        manual_points: list[dict[str, float]] | None = None,
    ) -> None:
        self.model = model
        self.stride = max(1, stride)
        self.manual_points = manual_points or []
        self.homography: np.ndarray | None = None
        self.calibration_mode: str | None = None
        self.calibration_confidence = 0.0
        self.calibration_source = "unavailable"
        self.frame_confidence: list[dict[str, Any]] = []
        self.manual_calibrations = 0
        self.manual_calibration_rejections = 0
        self.model_refreshes = 0
        self.temporal_blends = 0
        self.last_calibrated_frame = -1
        self.attempts = 0
        self.successes = 0
        self.assisted_calibrations = 0
        self.bootstrap_keypoint_calibrations = 0
        self.wide_view_calibrations = 0
        self.rejected_local_calibrations = 0
        self.rejected_geometry = 0
        self.line_refinements = 0
        self.goal_geometry_attempts = 0
        self.goal_geometry_calibrations = 0
        self.goal_geometry_rejections = 0
        self.last_line_alignment_score: float | None = None
        self.previous_tracking_gray: np.ndarray | None = None
        self.previous_tracking_mask: np.ndarray | None = None
        self.previous_tracking_scale: float | None = None
        self.camera_tracking_attempts = 0
        self.camera_tracking_successes = 0
        self.camera_tracking_failures = 0
        self.last_camera_inliers = 0
        self.last_camera_inlier_ratio: float | None = None
        self.last_camera_reprojection_error_px: float | None = None
        self.camera_cuts: list[dict[str, Any]] = []
        self.camera_cut_recoveries = 0
        self.awaiting_camera_cut_recovery = False
        self.last_camera_cut_frame: int | None = None
        self.visual_marker_observations = 0
        self.visual_marker_tracks: list[dict[str, Any]] = []
        self.next_visual_marker_id = 1
        self.rendered_frames = 0
        self.last_visible_keypoints = 0
        self.last_inliers = 0
        self.last_reprojection_error_cm: float | None = None
        self.last_target_span_cm: tuple[float, float] = (0.0, 0.0)
        self.last_player_valid_ratio: float | None = None
        self.ball_3d_projection_attempts = 0
        self.ball_3d_projection_successes = 0
        self.ball_3d_backprojection_attempts = 0
        self.ball_3d_backprojection_successes = 0
        self.errors = 0

    def update(
        self,
        frame: np.ndarray,
        frame_index: int,
        players: list[AnalysisObject] | None = None,
        static_markers: list[tuple[float, float]] | None = None,
    ) -> None:
        players = players or []
        try:
            if frame_index == 0 and self.manual_points:
                self._apply_manual_calibration(frame, frame_index, players)
            self._track_camera_motion(frame, frame_index, players)
            players = self._visual_pitch_players(frame, players)
            should_calibrate = (
                frame_index == 0
                or frame_index % self.stride == 0
                or self.homography is None
            )
            if should_calibrate:
                self.attempts += 1
                self.goal_geometry_attempts += 1
                visual_detections = self._detect_flat_white_markers(frame, players)
                self.visual_marker_observations += len(visual_detections)
                confirmed_visual_markers = self._track_visual_markers(
                    frame_index,
                    visual_detections,
                    frame.shape[1],
                )
                marker_candidates = self._merge_marker_candidates(
                    [*(static_markers or []), *visual_detections],
                    confirmed_visual_markers,
                )
                metric_geometry = self._goal_area_metric_homography(
                    frame,
                    players,
                    marker_candidates,
                )
                if metric_geometry is not None:
                    homography, error, inliers, player_ratio, line_score = metric_geometry
                    if self._geometry_agrees_with_current(homography, players):
                        self._accept_homography(
                            homography,
                            frame_index,
                            error,
                            inliers,
                            "metric_goal_area_geometry",
                            player_ratio,
                            line_score=line_score,
                        )
                        self.last_line_alignment_score = round(line_score, 4)
                        self.goal_geometry_calibrations += 1
                        return
                    self.goal_geometry_rejections += 1

            if self.model is None or frame_index % self.stride != 0:
                return

            results = self.model.predict(
                frame,
                imgsz=max(960, settings.YOLO_IMAGE_SIZE),
                device=settings.YOLO_DEVICE,
                verbose=False,
            )
            if not results or results[0].keypoints is None:
                return
            keypoints = results[0].keypoints
            source = keypoints.xy.cpu().numpy()
            if source.ndim == 3:
                if source.shape[0] == 0:
                    return
                source = source[0]
            if source.ndim != 2 or source.shape[0] == 0 or source.shape[1] < 2:
                return
            confidence = keypoints.conf
            if confidence is None:
                confidence_values = np.ones(len(source), dtype=np.float32)
            else:
                confidence_values = confidence.cpu().numpy()
                if confidence_values.ndim == 2:
                    confidence_values = confidence_values[0]

            target = self._pitch_vertices()
            count = min(len(source), len(target), len(confidence_values))
            source = source[:count].astype(np.float32)
            target = target[:count].astype(np.float32)
            confidence_values = confidence_values[:count]
            visible = (
                (confidence_values >= 0.34)
                & (source[:, 0] > 1)
                & (source[:, 1] > 1)
            )
            self.last_visible_keypoints = int(np.count_nonzero(visible))
            if self.last_visible_keypoints < 4:
                return

            visible_target = target[visible]
            span_x = float(np.ptp(visible_target[:, 0]))
            span_y = float(np.ptp(visible_target[:, 1]))
            self.last_target_span_cm = (round(span_x, 1), round(span_y, 1))

            source_hull_area = float(cv2.contourArea(cv2.convexHull(source[visible])))
            frame_area = float(max(1, frame.shape[0] * frame.shape[1]))
            is_wide_view = (
                self.last_visible_keypoints >= 8
                and span_x >= PITCH_LENGTH_CM * 0.50
                and span_y >= PITCH_WIDTH_CM * 0.40
                and source_hull_area >= frame_area * 0.045
            )
            is_bootstrap_view = self._is_constrained_bootstrap_candidate(
                visible_keypoints=self.last_visible_keypoints,
                span_x=span_x,
                span_y=span_y,
                source_hull_ratio=source_hull_area / frame_area,
            )
            if not is_wide_view and not is_bootstrap_view:
                self.rejected_local_calibrations += 1
                return

            homography, inlier_mask = cv2.findHomography(
                source[visible],
                target[visible],
                cv2.RANSAC,
                320.0,
            )
            if homography is None or not np.all(np.isfinite(homography)):
                self.rejected_geometry += 1
                return
            inliers = (
                inlier_mask.reshape(-1).astype(bool)
                if inlier_mask is not None
                else np.ones(self.last_visible_keypoints, dtype=bool)
            )
            if int(np.count_nonzero(inliers)) < 4:
                self.rejected_geometry += 1
                return
            errors = self._reprojection_errors(
                source[visible][inliers],
                target[visible][inliers],
                homography,
            )
            error = float(np.median(errors))
            p90_error = float(np.percentile(errors, 90))
            player_ok, player_ratio, _ = self._validate_player_projection(
                homography,
                players,
                end=None,
            )
            line_score = self._metric_line_alignment_score(frame, homography)
            normal_geometry_rejected = (
                not np.isfinite(error)
                or error > 260.0
                or p90_error > 480.0
                or not player_ok
                or line_score < 0.30
            )
            bootstrap_geometry_rejected = is_bootstrap_view and (
                not np.isfinite(error)
                or error > 140.0
                or p90_error > 280.0
                or not player_ok
                or player_ratio < 0.72
                or line_score < 0.45
            )
            if normal_geometry_rejected or bootstrap_geometry_rejected:
                self.rejected_geometry += 1
                return
            calibration_mode = (
                "constrained_bootstrap_keypoints"
                if is_bootstrap_view and not is_wide_view
                else "wide_view_keypoints"
            )
            self._accept_homography(
                homography,
                frame_index,
                error,
                int(np.count_nonzero(inliers)),
                calibration_mode,
                player_ratio,
                line_score=line_score,
            )
            self.last_line_alignment_score = round(line_score, 4)
            if calibration_mode == "wide_view_keypoints":
                self.wide_view_calibrations += 1
            else:
                self.assisted_calibrations += 1
                self.bootstrap_keypoint_calibrations += 1
            self.model_refreshes += 1
        except (AttributeError, IndexError, RuntimeError, TypeError, ValueError, cv2.error):
            self.errors += 1

    def _is_constrained_bootstrap_candidate(
        self,
        *,
        visible_keypoints: int,
        span_x: float,
        span_y: float,
        source_hull_ratio: float,
    ) -> bool:
        return (
            self.homography is None
            and visible_keypoints >= 6
            and span_x >= PITCH_LENGTH_CM * 0.40
            and span_y >= PITCH_WIDTH_CM * 0.55
            and source_hull_ratio >= 0.08
        )

    def _visual_pitch_players(
        self,
        frame: np.ndarray,
        players: list[AnalysisObject],
    ) -> list[AnalysisObject]:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        green = cv2.inRange(hsv, (24, 22, 20), (108, 255, 255))
        height, width = green.shape[:2]
        accepted: list[AnalysisObject] = []
        for player in players:
            foot_x = int(round((player.bbox[0] + player.bbox[2]) / 2))
            foot_y = int(round(player.bbox[3]))
            radius = max(5, int(round((player.bbox[2] - player.bbox[0]) * 0.24)))
            x1, x2 = max(0, foot_x - radius), min(width, foot_x + radius + 1)
            y1, y2 = max(0, foot_y - radius), min(height, foot_y + radius + 1)
            if x1 >= x2 or y1 >= y2:
                continue
            if float(np.mean(green[y1:y2, x1:x2] > 0)) >= 0.16:
                accepted.append(player)
        return accepted

    def _accept_homography(
        self,
        homography: np.ndarray,
        frame_index: int,
        error: float,
        inliers: int,
        mode: str,
        player_ratio: float,
        line_score: float | None = None,
    ) -> None:
        candidate = homography / homography[2, 2]
        if self.homography is not None and mode != "manual_correspondences":
            candidate = self._temporally_stabilize(candidate, line_score or 0.0)
        self.homography = candidate
        self.calibration_mode = mode
        self.calibration_source = "manual" if mode == "manual_correspondences" else "automatic"
        self.last_calibrated_frame = frame_index
        self.last_reprojection_error_cm = round(error, 2)
        self.last_inliers = inliers
        self.last_player_valid_ratio = round(player_ratio, 4)
        if line_score is not None:
            self.last_line_alignment_score = round(float(line_score), 4)
        error_score = max(0.0, 1.0 - error / 420.0)
        inlier_score = min(1.0, inliers / 10.0)
        line_quality = min(1.0, max(0.0, line_score if line_score is not None else 0.72))
        self.calibration_confidence = round(
            min(0.99, 0.34 * error_score + 0.20 * inlier_score + 0.26 * player_ratio + 0.20 * line_quality),
            4,
        )
        self.successes += 1
        if self.awaiting_camera_cut_recovery:
            self.camera_cut_recoveries += 1
            self.awaiting_camera_cut_recovery = False
            if self.camera_cuts:
                self.camera_cuts[-1]["recovered_frame"] = frame_index
                self.camera_cuts[-1]["recovery_frames"] = (
                    frame_index - int(self.camera_cuts[-1]["frame"])
                )

    def _temporally_stabilize(
        self,
        candidate: np.ndarray,
        candidate_line_score: float,
    ) -> np.ndarray:
        if self.homography is None:
            return candidate
        anchors = np.array(
            [[0.0, 0.0], [PITCH_LENGTH_CM, 0.0], [PITCH_LENGTH_CM, PITCH_WIDTH_CM], [0.0, PITCH_WIDTH_CM], [PITCH_LENGTH_CM / 2, PITCH_WIDTH_CM / 2]],
            dtype=np.float32,
        )
        try:
            current_inverse = np.linalg.inv(self.homography)
            candidate_inverse = np.linalg.inv(candidate)
            current_image = cv2.perspectiveTransform(anchors.reshape(-1, 1, 2), current_inverse).reshape(-1, 2)
            candidate_image = cv2.perspectiveTransform(anchors.reshape(-1, 1, 2), candidate_inverse).reshape(-1, 2)
        except (np.linalg.LinAlgError, cv2.error):
            return candidate
        finite = np.all(np.isfinite(current_image), axis=1) & np.all(np.isfinite(candidate_image), axis=1)
        if int(np.count_nonzero(finite)) < 4:
            return candidate
        median_jump = float(np.median(np.linalg.norm(current_image[finite] - candidate_image[finite], axis=1)))
        if median_jump > 180.0 and candidate_line_score < 0.62:
            return self.homography
        alpha = 0.42 if median_jump < 90.0 else 0.72
        blended_image = current_image[finite] * (1.0 - alpha) + candidate_image[finite] * alpha
        blended, _ = cv2.findHomography(blended_image.astype(np.float32), anchors[finite], 0)
        if blended is None or not np.all(np.isfinite(blended)):
            return candidate
        self.temporal_blends += 1
        return blended / blended[2, 2]

    def _apply_manual_calibration(
        self,
        frame: np.ndarray,
        frame_index: int,
        players: list[AnalysisObject],
    ) -> None:
        valid = [
            point
            for point in self.manual_points
            if {"image_x", "image_y", "pitch_x", "pitch_y"}.issubset(point)
        ]
        if len(valid) < 4:
            self.manual_calibration_rejections += 1
            return
        source = np.array([[point["image_x"], point["image_y"]] for point in valid], dtype=np.float32)
        target = np.array([[point["pitch_x"], point["pitch_y"]] for point in valid], dtype=np.float32)
        if float(cv2.contourArea(cv2.convexHull(source))) < frame.shape[0] * frame.shape[1] * 0.002:
            self.manual_calibration_rejections += 1
            return
        homography, _ = cv2.findHomography(source, target, cv2.RANSAC, 4.0)
        if homography is None or not np.all(np.isfinite(homography)):
            self.manual_calibration_rejections += 1
            return
        player_ok, player_ratio, _ = self._validate_player_projection(homography, players, end=None)
        line_score = self._metric_line_alignment_score(frame, homography)
        if not player_ok or line_score < 0.24:
            self.manual_calibration_rejections += 1
            return
        error = float(np.median(self._reprojection_errors(source, target, homography)))
        self._accept_homography(
            homography,
            frame_index,
            error,
            len(valid),
            "manual_correspondences",
            player_ratio,
            line_score=line_score,
        )
        self.calibration_confidence = max(self.calibration_confidence, min(0.96, 0.68 + line_score * 0.28))
        self.manual_calibrations += 1

    def _merge_marker_candidates(
        self,
        first: list[tuple[float, float]],
        second: list[tuple[float, float]],
    ) -> list[tuple[float, float]]:
        merged: list[tuple[float, float]] = []
        for point in [*first, *second]:
            if any(float(np.hypot(point[0] - item[0], point[1] - item[1])) < 55.0 for item in merged):
                continue
            merged.append(point)
        return merged

    def _track_camera_motion(
        self,
        frame: np.ndarray,
        frame_index: int,
        players: list[AnalysisObject],
    ) -> None:
        current_gray, current_mask, current_scale = self._camera_tracking_sample(
            frame,
            players,
        )
        previous_gray = self.previous_tracking_gray
        previous_mask = self.previous_tracking_mask
        previous_scale = self.previous_tracking_scale
        self.previous_tracking_gray = current_gray
        self.previous_tracking_mask = current_mask
        self.previous_tracking_scale = current_scale
        if (
            previous_gray is not None
            and previous_gray.shape == current_gray.shape
            and self._is_camera_cut(previous_gray, current_gray)
        ):
            self.camera_cuts.append(
                {
                    "frame": frame_index,
                    "recovered_frame": None,
                    "recovery_frames": None,
                }
            )
            self.last_camera_cut_frame = frame_index
            self.awaiting_camera_cut_recovery = True
            self.homography = None
            self.calibration_mode = None
            self.calibration_confidence = 0.0
            self.calibration_source = "camera_cut"
            self.last_calibrated_frame = -1
            return
        if (
            self.homography is None
            or previous_gray is None
            or previous_mask is None
            or previous_scale is None
            or abs(previous_scale - current_scale) > 1e-6
        ):
            return

        self.camera_tracking_attempts += 1
        self.calibration_confidence = max(0.0, self.calibration_confidence * 0.996)
        points = cv2.goodFeaturesToTrack(
            previous_gray,
            mask=previous_mask,
            maxCorners=500,
            qualityLevel=0.008,
            minDistance=10,
            blockSize=7,
        )
        if points is None or len(points) < 24:
            self.camera_tracking_failures += 1
            return

        current_points, forward_status, _ = cv2.calcOpticalFlowPyrLK(
            previous_gray,
            current_gray,
            points,
            None,
            winSize=(25, 25),
            maxLevel=3,
            criteria=(
                cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                30,
                0.01,
            ),
        )
        if current_points is None or forward_status is None:
            self.camera_tracking_failures += 1
            return
        backward_points, backward_status, _ = cv2.calcOpticalFlowPyrLK(
            current_gray,
            previous_gray,
            current_points,
            None,
            winSize=(25, 25),
            maxLevel=3,
            criteria=(
                cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                30,
                0.01,
            ),
        )
        if backward_points is None or backward_status is None:
            self.camera_tracking_failures += 1
            return

        previous_xy = points.reshape(-1, 2)
        current_xy = current_points.reshape(-1, 2)
        backward_xy = backward_points.reshape(-1, 2)
        valid = (
            (forward_status.reshape(-1) == 1)
            & (backward_status.reshape(-1) == 1)
            & (np.linalg.norm(previous_xy - backward_xy, axis=1) <= 1.6)
        )
        current_height, current_width = current_gray.shape[:2]
        rounded = np.rint(current_xy).astype(np.int32)
        inside = (
            (rounded[:, 0] >= 0)
            & (rounded[:, 0] < current_width)
            & (rounded[:, 1] >= 0)
            & (rounded[:, 1] < current_height)
        )
        green = np.zeros(len(valid), dtype=bool)
        inside_indexes = np.flatnonzero(inside)
        if len(inside_indexes):
            green[inside_indexes] = (
                current_mask[
                    rounded[inside_indexes, 1],
                    rounded[inside_indexes, 0],
                ]
                > 0
            )
        valid &= inside & green
        previous_xy = previous_xy[valid]
        current_xy = current_xy[valid]
        if len(previous_xy) < 24:
            self.camera_tracking_failures += 1
            return

        current_to_previous, inlier_mask = cv2.findHomography(
            current_xy,
            previous_xy,
            cv2.RANSAC,
            2.2,
        )
        if current_to_previous is None or inlier_mask is None:
            self.camera_tracking_failures += 1
            return
        inliers = inlier_mask.reshape(-1).astype(bool)
        inlier_count = int(np.count_nonzero(inliers))
        inlier_ratio = float(inlier_count / max(1, len(inliers)))
        projected = cv2.perspectiveTransform(
            current_xy.reshape(-1, 1, 2).astype(np.float32),
            current_to_previous,
        ).reshape(-1, 2)
        reprojection_error = float(
            np.median(np.linalg.norm(projected[inliers] - previous_xy[inliers], axis=1))
        ) if inlier_count else float("inf")
        if (
            inlier_count < 22
            or inlier_ratio < 0.68
            or not np.isfinite(reprojection_error)
            or reprojection_error > 1.8
            or not self._reasonable_camera_delta(
                current_to_previous,
                current_width,
                current_height,
            )
        ):
            self.camera_tracking_failures += 1
            return

        scale_matrix = np.array(
            [
                [current_scale, 0.0, 0.0],
                [0.0, current_scale, 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        current_to_previous_original = (
            np.linalg.inv(scale_matrix)
            @ current_to_previous
            @ scale_matrix
        )
        candidate = self.homography @ current_to_previous_original
        if abs(float(candidate[2, 2])) <= 1e-9:
            self.camera_tracking_failures += 1
            return
        candidate /= candidate[2, 2]
        player_ok, _, _ = self._validate_player_projection(candidate, players, end=None)
        if not player_ok:
            self.camera_tracking_failures += 1
            return

        self.homography = candidate
        self.last_calibrated_frame = frame_index
        if self.calibration_mode and "+camera_motion" not in self.calibration_mode:
            self.calibration_mode = f"{self.calibration_mode}+camera_motion"
        self.camera_tracking_successes += 1
        flow_quality = min(1.0, max(0.0, inlier_ratio * (1.0 - reprojection_error / 8.0)))
        self.calibration_confidence = round(
            min(0.99, self.calibration_confidence * 0.94 + flow_quality * 0.06),
            4,
        )
        self.calibration_source = "camera_motion"
        self.last_camera_inliers = inlier_count
        self.last_camera_inlier_ratio = round(inlier_ratio, 4)
        self.last_camera_reprojection_error_px = round(reprojection_error, 3)

    def _is_camera_cut(
        self,
        previous_gray: np.ndarray,
        current_gray: np.ndarray,
    ) -> bool:
        previous_hist = cv2.calcHist([previous_gray], [0], None, [32], [0, 256])
        current_hist = cv2.calcHist([current_gray], [0], None, [32], [0, 256])
        cv2.normalize(previous_hist, previous_hist)
        cv2.normalize(current_hist, current_hist)
        histogram_correlation = float(
            cv2.compareHist(previous_hist, current_hist, cv2.HISTCMP_CORREL)
        )
        mean_difference = float(
            np.mean(cv2.absdiff(previous_gray, current_gray))
        )
        return (
            histogram_correlation < 0.42 and mean_difference > 34.0
        ) or mean_difference > 72.0

    def _camera_tracking_sample(
        self,
        frame: np.ndarray,
        players: list[AnalysisObject],
    ) -> tuple[np.ndarray, np.ndarray, float]:
        frame_height, frame_width = frame.shape[:2]
        scale = min(1.0, 960.0 / max(1, frame_width))
        if scale < 1.0:
            sample = cv2.resize(
                frame,
                (
                    int(round(frame_width * scale)),
                    int(round(frame_height * scale)),
                ),
                interpolation=cv2.INTER_AREA,
            )
        else:
            sample = frame
        gray = cv2.cvtColor(sample, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(sample, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, (25, 28, 25), (105, 255, 255))
        mask = cv2.erode(mask, np.ones((5, 5), dtype=np.uint8))
        for player in players:
            x1, y1, x2, y2 = player.bbox
            width = x2 - x1
            height = y2 - y1
            cv2.rectangle(
                mask,
                (
                    max(0, int(round((x1 - width * 0.15) * scale))),
                    max(0, int(round((y1 - height * 0.10) * scale))),
                ),
                (
                    min(mask.shape[1] - 1, int(round((x2 + width * 0.15) * scale))),
                    min(mask.shape[0] - 1, int(round((y2 + height * 0.10) * scale))),
                ),
                0,
                cv2.FILLED,
            )
        return gray, mask, scale

    def _reasonable_camera_delta(
        self,
        transform: np.ndarray,
        width: int,
        height: int,
    ) -> bool:
        corners = np.float32(
            [[0, 0], [width, 0], [width, height], [0, height]]
        )
        projected = cv2.perspectiveTransform(
            corners.reshape(-1, 1, 2),
            transform,
        ).reshape(-1, 2)
        if not np.all(np.isfinite(projected)):
            return False
        displacement = np.linalg.norm(projected - corners, axis=1)
        if float(np.max(displacement)) > float(np.hypot(width, height)) * 0.12:
            return False
        source_area = float(max(1, width * height))
        projected_area = abs(float(cv2.contourArea(projected.astype(np.float32))))
        ratio = projected_area / source_area
        return 0.84 <= ratio <= 1.18

    def _goal_area_metric_homography(
        self,
        frame: np.ndarray,
        players: list[AnalysisObject],
        marker_candidates: list[tuple[float, float]],
    ) -> tuple[np.ndarray, float, int, float, float] | None:
        if not marker_candidates:
            return None
        frame_height, frame_width = frame.shape[:2]
        scale = min(1.0, 1920.0 / max(1, frame_width))
        if scale < 1.0:
            sample = cv2.resize(
                frame,
                (
                    int(round(frame_width * scale)),
                    int(round(frame_height * scale)),
                ),
                interpolation=cv2.INTER_AREA,
            )
        else:
            sample = frame
        sample_height, sample_width = sample.shape[:2]
        hsv = cv2.cvtColor(sample, cv2.COLOR_BGR2HSV)
        white = cv2.inRange(hsv, (0, 0, 165), (180, 80, 255))
        goal_posts = self._detect_goal_post_pair(white)
        if goal_posts is None:
            return None
        left_post, right_post = goal_posts

        green = cv2.inRange(hsv, (25, 35, 30), (100, 255, 255))
        green_near = cv2.dilate(green, np.ones((17, 17), dtype=np.uint8))
        field_white = cv2.inRange(hsv, (0, 0, 150), (180, 92, 255))
        field_mask = cv2.bitwise_and(field_white, green_near)
        field_mask = cv2.morphologyEx(
            field_mask,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_RECT, (11, 3)),
        )
        segments = self._hough_segments(
            field_mask,
            threshold=max(45, sample_width // 42),
            min_length=max(60, sample_width // 20),
            max_gap=max(24, sample_width // 64),
        )
        if not segments:
            return None

        post_bottom_left = left_post["bottom"]
        post_bottom_right = right_post["bottom"]
        goal_angle = self._segment_angle(
            (
                float(post_bottom_left[0]),
                float(post_bottom_left[1]),
                float(post_bottom_right[0]),
                float(post_bottom_right[1]),
            )
        )
        lateral: list[tuple[np.ndarray, tuple[float, float, float, float], float]] = []
        depth: list[tuple[np.ndarray, tuple[float, float, float, float], float]] = []
        for segment in segments:
            line = self._line_from_segment(segment)
            if line is None:
                continue
            length = float(np.hypot(
                segment[2] - segment[0],
                segment[3] - segment[1],
            ))
            gap = self._angle_gap(goal_angle, self._segment_angle(segment))
            item = (line, segment, length)
            if gap <= 11.0:
                lateral.append(item)
            elif gap >= 25.0 and length >= sample_height * 0.08:
                depth.append(item)
        if not lateral or not depth:
            return None

        goal_item = min(
            lateral,
            key=lambda item: (
                self._point_line_distance(tuple(post_bottom_left), item[0])
                + self._point_line_distance(tuple(post_bottom_right), item[0])
                - item[2] * 0.01
            ),
        )
        goal_line = goal_item[0]
        post_left = self._line_intersection(left_post["line"], goal_line)
        post_right = self._line_intersection(right_post["line"], goal_line)
        if post_left is None or post_right is None:
            return None
        post_left_xy = np.array(post_left, dtype=np.float64)
        post_right_xy = np.array(post_right, dtype=np.float64)

        marker: np.ndarray | None = None
        marker_distance = 0.0
        for original_marker in marker_candidates:
            candidate = np.array(original_marker, dtype=np.float64) * scale
            absolute = self._point_line_distance(tuple(candidate), goal_line)
            if not (
                sample_height * 0.14
                <= absolute
                <= sample_height * 0.68
            ):
                continue
            x_min = min(post_left_xy[0], post_right_xy[0]) - sample_width * 0.20
            x_max = max(post_left_xy[0], post_right_xy[0]) + sample_width * 0.20
            if not x_min <= candidate[0] <= x_max:
                continue
            if absolute > marker_distance:
                marker = candidate
                marker_distance = absolute
        if marker is None:
            return None
        marker_signed = float(
            goal_line[0] * marker[0]
            + goal_line[1] * marker[1]
            + goal_line[2]
        )
        marker_sign = 1.0 if marker_signed >= 0 else -1.0

        front_candidates = []
        for item in lateral:
            midpoint = (
                (item[1][0] + item[1][2]) / 2,
                (item[1][1] + item[1][3]) / 2,
            )
            signed = float(
                goal_line[0] * midpoint[0]
                + goal_line[1] * midpoint[1]
                + goal_line[2]
            )
            ratio = signed * marker_sign / marker_distance
            if (
                0.18 <= ratio <= 0.75
                and item[2] >= sample_width * 0.28
            ):
                front_candidates.append(
                    (
                        abs(ratio - 0.42) - item[2] / sample_width * 0.03,
                        item,
                    )
                )
        if not front_candidates:
            return None
        _, front_item = min(front_candidates, key=lambda item: item[0])
        front_line = front_item[0]

        goal_direction = post_right_xy - post_left_xy
        direction_norm = float(np.linalg.norm(goal_direction))
        if direction_norm <= 1e-6:
            return None
        goal_direction /= direction_norm
        post_scalars = sorted(
            (
                float(np.dot(post_left_xy, goal_direction)),
                float(np.dot(post_right_xy, goal_direction)),
            )
        )
        post_span = post_scalars[1] - post_scalars[0]
        side_candidates = []
        for item in depth:
            back_corner = self._line_intersection(item[0], goal_line)
            front_corner = self._line_intersection(item[0], front_line)
            if back_corner is None or front_corner is None:
                continue
            back_xy = np.array(back_corner, dtype=np.float64)
            front_xy = np.array(front_corner, dtype=np.float64)
            if not (
                -sample_width * 0.08 <= back_xy[0] <= sample_width * 1.08
                and -sample_height * 0.08 <= back_xy[1] <= sample_height * 1.08
                and -sample_width * 0.08 <= front_xy[0] <= sample_width * 1.08
                and -sample_height * 0.08 <= front_xy[1] <= sample_height * 1.08
            ):
                continue
            scalar = float(np.dot(back_xy, goal_direction))
            outside = min(
                abs(scalar - post_scalars[0]),
                abs(scalar - post_scalars[1]),
            )
            if (
                post_scalars[0] <= scalar <= post_scalars[1]
                or not post_span * 0.20 <= outside <= post_span * 2.2
            ):
                continue
            first_endpoint = np.array(item[1][:2], dtype=np.float64)
            second_endpoint = np.array(item[1][2:], dtype=np.float64)
            endpoint_distance = min(
                float(np.linalg.norm(front_xy - first_endpoint)),
                float(np.linalg.norm(front_xy - second_endpoint)),
                float(np.linalg.norm(back_xy - first_endpoint)),
                float(np.linalg.norm(back_xy - second_endpoint)),
            )
            side_candidates.append(
                (
                    endpoint_distance - item[2] * 0.12,
                    back_xy,
                    front_xy,
                    scalar,
                )
            )
        if not side_candidates:
            return None
        _, back_corner, front_corner, side_scalar = min(
            side_candidates,
            key=lambda item: item[0],
        )

        center_y = PITCH_WIDTH_CM / 2
        side_y = (
            center_y - GOAL_AREA_WIDTH_CM / 2
            if side_scalar < post_scalars[0]
            else center_y + GOAL_AREA_WIDTH_CM / 2
        )
        ordered_posts = sorted(
            (post_left_xy, post_right_xy),
            key=lambda point: float(np.dot(point, goal_direction)),
        )
        source_sample = np.array(
            [
                back_corner,
                front_corner,
                ordered_posts[0],
                ordered_posts[1],
                marker,
            ],
            dtype=np.float32,
        )
        target = np.array(
            [
                [PITCH_LENGTH_CM, side_y],
                [PITCH_LENGTH_CM - GOAL_AREA_LENGTH_CM, side_y],
                [PITCH_LENGTH_CM, center_y - GOAL_WIDTH_CM / 2],
                [PITCH_LENGTH_CM, center_y + GOAL_WIDTH_CM / 2],
                [PITCH_LENGTH_CM - PENALTY_SPOT_DISTANCE_CM, center_y],
            ],
            dtype=np.float32,
        )
        homography_sample, _ = cv2.findHomography(source_sample, target, 0)
        if homography_sample is None or not np.all(np.isfinite(homography_sample)):
            return None
        scale_matrix = np.array(
            [[scale, 0.0, 0.0], [0.0, scale, 0.0], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
        homography = homography_sample @ scale_matrix
        errors = self._reprojection_errors(
            source_sample / scale,
            target,
            homography,
        )
        error = float(np.median(errors))
        if not np.isfinite(error) or error > 140.0 or float(np.max(errors)) > 280.0:
            return None
        player_ok, player_ratio, _ = self._validate_player_projection(
            homography,
            players,
            end="right",
        )
        if not player_ok:
            return None
        line_score = self._metric_line_alignment_score(frame, homography)
        if line_score < 0.48:
            return None
        return homography, error, len(source_sample), player_ratio, line_score

    def _detect_goal_post_pair(
        self,
        white_mask: np.ndarray,
    ) -> tuple[dict[str, Any], dict[str, Any]] | None:
        height, width = white_mask.shape[:2]
        segments = self._hough_segments(
            white_mask,
            threshold=max(35, width // 55),
            min_length=max(35, height // 30),
            max_gap=max(24, width // 64),
        )
        vertical = []
        for segment in segments:
            length = float(np.hypot(
                segment[2] - segment[0],
                segment[3] - segment[1],
            ))
            if length < height * 0.08 or abs(self._segment_angle(segment)) < 65:
                continue
            top = min(segment[1], segment[3])
            bottom = max(segment[1], segment[3])
            if top > height * 0.60 or bottom > height * 0.75:
                continue
            vertical.append((segment, length))

        clusters: list[list[tuple[tuple[float, float, float, float], float]]] = []
        for item in sorted(
            vertical,
            key=lambda value: (value[0][0] + value[0][2]) / 2,
        ):
            center_x = (item[0][0] + item[0][2]) / 2
            cluster = next(
                (
                    group
                    for group in clusters
                    if abs(
                        center_x
                        - float(np.median([
                            (entry[0][0] + entry[0][2]) / 2
                            for entry in group
                        ]))
                    )
                    <= width * 0.018
                ),
                None,
            )
            if cluster is None:
                clusters.append([item])
            else:
                cluster.append(item)

        post_candidates: list[dict[str, Any]] = []
        for cluster in clusters:
            points = np.array(
                [
                    point
                    for segment, _ in cluster
                    for point in (
                        (segment[0], segment[1]),
                        (segment[2], segment[3]),
                    )
                ],
                dtype=np.float32,
            )
            top_y = float(np.min(points[:, 1]))
            bottom_y = float(np.max(points[:, 1]))
            if bottom_y - top_y < height * 0.11:
                continue
            vx, vy, x0, y0 = [
                float(value)
                for value in cv2.fitLine(
                    points,
                    cv2.DIST_L2,
                    0,
                    0.01,
                    0.01,
                ).reshape(-1)
            ]
            line = np.array([vy, -vx, vx * y0 - vy * x0], dtype=np.float64)
            norm = float(np.hypot(line[0], line[1]))
            if norm <= 1e-6 or abs(float(line[0])) <= 1e-6:
                continue
            line /= norm
            top_x = -(line[1] * top_y + line[2]) / line[0]
            bottom_x = -(line[1] * bottom_y + line[2]) / line[0]
            post_candidates.append(
                {
                    "line": line,
                    "top": np.array([top_x, top_y], dtype=np.float64),
                    "bottom": np.array([bottom_x, bottom_y], dtype=np.float64),
                    "span": bottom_y - top_y,
                }
            )

        dilated = cv2.dilate(white_mask, np.ones((5, 5), dtype=np.uint8))
        best_pair: tuple[dict[str, Any], dict[str, Any]] | None = None
        best_score = -1.0
        for first_index, first in enumerate(post_candidates):
            for second in post_candidates[first_index + 1 :]:
                left, right = sorted(
                    (first, second),
                    key=lambda item: float(item["top"][0]),
                )
                separation = float(right["top"][0] - left["top"][0])
                if not width * 0.12 <= separation <= width * 0.62:
                    continue
                if abs(float(left["top"][1] - right["top"][1])) > height * 0.12:
                    continue
                crossbar_support = -1.0
                for segment in segments:
                    segment_length = float(np.hypot(
                        segment[2] - segment[0],
                        segment[3] - segment[1],
                    ))
                    if (
                        segment_length < separation * 0.65
                        or abs(self._segment_angle(segment)) > 25
                    ):
                        continue
                    crossbar_line = self._line_from_segment(segment)
                    if crossbar_line is None:
                        continue
                    left_touch = self._line_intersection(left["line"], crossbar_line)
                    right_touch = self._line_intersection(right["line"], crossbar_line)
                    if left_touch is None or right_touch is None:
                        continue
                    if not (
                        left["top"][1] - height * 0.04
                        <= left_touch[1]
                        <= left["top"][1] + height * 0.15
                        and right["top"][1] - height * 0.04
                        <= right_touch[1]
                        <= right["top"][1] + height * 0.15
                    ):
                        continue
                    support = self._mask_line_support(
                        dilated,
                        np.array(left_touch, dtype=np.float64),
                        np.array(right_touch, dtype=np.float64),
                    )
                    crossbar_support = max(crossbar_support, support)
                if crossbar_support < 0.35:
                    continue
                score = (
                    min(float(left["span"]), float(right["span"]))
                    + crossbar_support * height * 0.2
                )
                if score > best_score:
                    best_pair = (left, right)
                    best_score = score
        return best_pair

    def _hough_segments(
        self,
        mask: np.ndarray,
        threshold: int,
        min_length: int,
        max_gap: int,
    ) -> list[tuple[float, float, float, float]]:
        detected = cv2.HoughLinesP(
            mask,
            1,
            np.pi / 720,
            threshold=threshold,
            minLineLength=min_length,
            maxLineGap=max_gap,
        )
        if detected is None:
            return []
        return [
            tuple(float(value) for value in segment)
            for segment in detected.reshape(-1, 4)
        ]

    def _mask_line_support(
        self,
        mask: np.ndarray,
        first: np.ndarray,
        second: np.ndarray,
        samples: int = 100,
    ) -> float:
        points = np.rint(np.linspace(first, second, samples)).astype(np.int32)
        inside = (
            (points[:, 0] >= 0)
            & (points[:, 0] < mask.shape[1])
            & (points[:, 1] >= 0)
            & (points[:, 1] < mask.shape[0])
        )
        if int(np.count_nonzero(inside)) < 8:
            return 0.0
        points = points[inside]
        return float(np.mean(mask[points[:, 1], points[:, 0]] > 0))

    def _metric_line_alignment_score(
        self,
        frame: np.ndarray,
        homography: np.ndarray,
    ) -> float:
        try:
            inverse = np.linalg.inv(homography)
        except np.linalg.LinAlgError:
            return 0.0
        frame_height, frame_width = frame.shape[:2]
        scale = min(1.0, 960.0 / max(1, frame_width))
        if scale < 1.0:
            sample = cv2.resize(
                frame,
                (
                    int(round(frame_width * scale)),
                    int(round(frame_height * scale)),
                ),
                interpolation=cv2.INTER_AREA,
            )
        else:
            sample = frame
        hsv = cv2.cvtColor(sample, cv2.COLOR_BGR2HSV)
        green = cv2.inRange(hsv, (25, 30, 28), (105, 255, 255))
        green_near = cv2.dilate(green, np.ones((11, 11), dtype=np.uint8))
        white = cv2.inRange(hsv, (0, 0, 145), (180, 96, 255))
        line_mask = cv2.bitwise_and(white, green_near)
        line_mask = cv2.dilate(line_mask, np.ones((7, 7), dtype=np.uint8))

        center_x = PITCH_LENGTH_CM / 2
        center_y = PITCH_WIDTH_CM / 2
        penalty_half = PENALTY_AREA_WIDTH_CM / 2
        goal_half = GOAL_AREA_WIDTH_CM / 2
        segments = [
            ((0.0, 0.0), (PITCH_LENGTH_CM, 0.0)),
            ((0.0, PITCH_WIDTH_CM), (PITCH_LENGTH_CM, PITCH_WIDTH_CM)),
            ((0.0, 0.0), (0.0, PITCH_WIDTH_CM)),
            ((PITCH_LENGTH_CM, 0.0), (PITCH_LENGTH_CM, PITCH_WIDTH_CM)),
            ((center_x, 0.0), (center_x, PITCH_WIDTH_CM)),
        ]
        for end_x, direction in ((0.0, 1.0), (PITCH_LENGTH_CM, -1.0)):
            penalty_x = end_x + direction * PENALTY_AREA_LENGTH_CM
            goal_x = end_x + direction * GOAL_AREA_LENGTH_CM
            segments.extend(
                [
                    ((penalty_x, center_y - penalty_half), (penalty_x, center_y + penalty_half)),
                    ((end_x, center_y - penalty_half), (penalty_x, center_y - penalty_half)),
                    ((end_x, center_y + penalty_half), (penalty_x, center_y + penalty_half)),
                    ((goal_x, center_y - goal_half), (goal_x, center_y + goal_half)),
                    ((end_x, center_y - goal_half), (goal_x, center_y - goal_half)),
                    ((end_x, center_y + goal_half), (goal_x, center_y + goal_half)),
                ]
            )
        circle = [
            (
                center_x + CENTER_CIRCLE_RADIUS_CM * np.cos(angle),
                center_y + CENTER_CIRCLE_RADIUS_CM * np.sin(angle),
            )
            for angle in np.linspace(0.0, 2.0 * np.pi, 17)
        ]
        segments.extend(zip(circle[:-1], circle[1:]))
        penalty_arc_angle = float(np.arccos(
            (PENALTY_AREA_LENGTH_CM - PENALTY_SPOT_DISTANCE_CM)
            / CENTER_CIRCLE_RADIUS_CM
        ))
        left_arc = [
            (
                PENALTY_SPOT_DISTANCE_CM + CENTER_CIRCLE_RADIUS_CM * np.cos(angle),
                center_y + CENTER_CIRCLE_RADIUS_CM * np.sin(angle),
            )
            for angle in np.linspace(-penalty_arc_angle, penalty_arc_angle, 9)
        ]
        right_arc = [
            (
                PITCH_LENGTH_CM - PENALTY_SPOT_DISTANCE_CM
                + CENTER_CIRCLE_RADIUS_CM * np.cos(angle),
                center_y + CENTER_CIRCLE_RADIUS_CM * np.sin(angle),
            )
            for angle in np.linspace(
                np.pi - penalty_arc_angle,
                np.pi + penalty_arc_angle,
                9,
            )
        ]
        segments.extend(zip(left_arc[:-1], left_arc[1:]))
        segments.extend(zip(right_arc[:-1], right_arc[1:]))
        scores: list[float] = []
        for first, second in segments:
            pitch_points = np.linspace(first, second, 100).astype(np.float32)
            image_points = cv2.perspectiveTransform(
                pitch_points.reshape(-1, 1, 2),
                inverse,
            ).reshape(-1, 2)
            image_points *= scale
            finite = np.all(np.isfinite(image_points), axis=1)
            rounded = np.rint(image_points).astype(np.int32)
            inside = (
                finite
                & (rounded[:, 0] >= 0)
                & (rounded[:, 0] < line_mask.shape[1])
                & (rounded[:, 1] >= 0)
                & (rounded[:, 1] < line_mask.shape[0])
            )
            if int(np.count_nonzero(inside)) < 12:
                continue
            valid_points = rounded[inside]
            scores.append(float(np.mean(
                line_mask[valid_points[:, 1], valid_points[:, 0]] > 0
            )))
        if len(scores) < 3:
            return 0.0
        scores.sort(reverse=True)
        return float(np.mean(scores[: min(7, len(scores))]))

    def _geometry_agrees_with_current(
        self,
        candidate: np.ndarray,
        players: list[AnalysisObject],
    ) -> bool:
        if self.homography is None or not self.calibration_mode:
            return True
        if not self.calibration_mode.startswith("metric_goal_area_geometry"):
            return True
        points = [
            ((player.bbox[0] + player.bbox[2]) / 2, player.bbox[3])
            for player in players
            if not player.is_predicted
        ]
        if len(points) < 3:
            return True
        source = np.array(points, dtype=np.float32).reshape(-1, 1, 2)
        current_xy = cv2.perspectiveTransform(source, self.homography).reshape(-1, 2)
        candidate_xy = cv2.perspectiveTransform(source, candidate).reshape(-1, 2)
        finite = np.all(np.isfinite(current_xy), axis=1) & np.all(np.isfinite(candidate_xy), axis=1)
        if int(np.count_nonzero(finite)) < 3:
            return False
        discrepancy = np.linalg.norm(current_xy[finite] - candidate_xy[finite], axis=1)
        return float(np.median(discrepancy)) <= 180.0

    def _track_visual_markers(
        self,
        frame_index: int,
        detections: list[tuple[float, float]],
        frame_width: int,
    ) -> list[tuple[float, float]]:
        gate = max(70.0, frame_width * 0.025)
        used_track_ids: set[int] = set()
        for point in detections:
            best_track: dict[str, Any] | None = None
            best_distance = float("inf")
            for track in self.visual_marker_tracks:
                if int(track["id"]) in used_track_ids:
                    continue
                if frame_index - int(track["last_frame"]) > self.stride * 3:
                    continue
                center = track["center"]
                distance = float(np.hypot(point[0] - center[0], point[1] - center[1]))
                if distance <= gate and distance < best_distance:
                    best_track = track
                    best_distance = distance
            if best_track is None:
                best_track = {
                    "id": self.next_visual_marker_id,
                    "center": point,
                    "last_frame": frame_index,
                    "hits": 1,
                }
                self.next_visual_marker_id += 1
                self.visual_marker_tracks.append(best_track)
            else:
                best_track["center"] = point
                best_track["last_frame"] = frame_index
                best_track["hits"] = int(best_track["hits"]) + 1
            used_track_ids.add(int(best_track["id"]))

        confirmed = [
            track
            for track in self.visual_marker_tracks
            if int(track["hits"]) >= 2
            and frame_index - int(track["last_frame"]) <= self.stride * 2
        ]
        confirmed.sort(key=lambda track: (-int(track["hits"]), int(track["id"])))
        return [track["center"] for track in confirmed[:4]]

    def _detect_flat_white_markers(
        self,
        frame: np.ndarray,
        players: list[AnalysisObject],
    ) -> list[tuple[float, float]]:
        frame_height, frame_width = frame.shape[:2]
        scale = min(1.0, 1920.0 / max(1, frame_width))
        if scale < 1.0:
            sample = cv2.resize(
                frame,
                (int(round(frame_width * scale)), int(round(frame_height * scale))),
                interpolation=cv2.INTER_AREA,
            )
        else:
            sample = frame
        sample_height, sample_width = sample.shape[:2]
        hsv = cv2.cvtColor(sample, cv2.COLOR_BGR2HSV)
        green = cv2.inRange(hsv, (25, 35, 28), (100, 255, 255))
        green_near = cv2.dilate(
            green,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13)),
        )
        white = cv2.inRange(hsv, (0, 0, 155), (180, 78, 255))
        mask = cv2.bitwise_and(white, green_near)
        mask = cv2.morphologyEx(
            mask,
            cv2.MORPH_OPEN,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
        )
        component_count, _, stats, centroids = cv2.connectedComponentsWithStats(
            mask,
            connectivity=8,
        )
        min_area = max(18, int(sample_width * sample_height * 0.000008))
        max_area = max(min_area + 1, int(sample_width * sample_height * 0.00055))
        candidates: list[tuple[float, float, float]] = []
        for index in range(1, component_count):
            x, y, width, height, area = [int(value) for value in stats[index]]
            if area < min_area or area > max_area or height <= 0:
                continue
            aspect = width / height
            fill_ratio = area / max(1.0, float(width * height))
            if not (1.35 <= aspect <= 3.8 and fill_ratio >= 0.34):
                continue
            if not (
                sample_width * 0.004 <= width <= sample_width * 0.055
                and sample_height * 0.004 <= height <= sample_height * 0.05
            ):
                continue
            center_x = float(centroids[index][0] / scale)
            center_y = float(centroids[index][1] / scale)
            if center_y < frame_height * 0.16:
                continue
            if self._inside_player_region((center_x, center_y), players):
                continue
            candidates.append((center_x, center_y, float(area)))
        candidates.sort(key=lambda item: item[2], reverse=True)
        return [(item[0], item[1]) for item in candidates[:4]]

    def _inside_player_region(
        self,
        point: tuple[float, float],
        players: list[AnalysisObject],
    ) -> bool:
        for player in players:
            x1, y1, x2, y2 = player.bbox
            width = x2 - x1
            height = y2 - y1
            if (
                x1 - width * 0.18 <= point[0] <= x2 + width * 0.18
                and y1 - height * 0.12 <= point[1] <= y2 + height * 0.12
            ):
                return True
        return False

    def _penalty_area_homography(
        self,
        frame: np.ndarray,
        source: np.ndarray,
        target: np.ndarray,
        confidence: np.ndarray,
        players: list[AnalysisObject],
        static_markers: list[tuple[float, float]],
    ) -> tuple[np.ndarray, float, int, float] | None:
        if not static_markers:
            return None

        groups = (
            {
                "end": "right",
                "front_corner": 17,
                "front_reference": 19,
                "spot": 21,
                "goal_corner": 25,
                "goal_reference": 26,
            },
            {
                "end": "right",
                "front_corner": 20,
                "front_reference": 18,
                "spot": 21,
                "goal_corner": 28,
                "goal_reference": 27,
            },
            {
                "end": "left",
                "front_corner": 9,
                "front_reference": 11,
                "spot": 8,
                "goal_corner": 1,
                "goal_reference": 2,
            },
            {
                "end": "left",
                "front_corner": 12,
                "front_reference": 10,
                "spot": 8,
                "goal_corner": 4,
                "goal_reference": 3,
            },
        )
        required_keys = ("front_reference", "goal_corner", "goal_reference")
        available_groups = []
        for group in groups:
            indexes = [int(group[key]) for key in required_keys]
            if max(indexes) >= len(source):
                continue
            if any(confidence[index] < 0.42 for index in indexes):
                continue
            score = sum(float(confidence[index]) for index in indexes)
            available_groups.append((score, group))
        available_groups.sort(key=lambda item: item[0], reverse=True)

        best: tuple[np.ndarray, float, int, float] | None = None
        best_score = float("inf")
        for _, group in available_groups:
            front_reference_index = int(group["front_reference"])
            refined = self._refine_penalty_corner(
                frame,
                tuple(float(value) for value in source[front_reference_index]),
            )
            if refined is None:
                continue
            corner, front_line = refined
            front_reference = self._project_to_line(
                tuple(float(value) for value in source[front_reference_index]),
                front_line,
            )
            for marker in static_markers[:3]:
                source_points = np.array(
                    [
                        corner,
                        front_reference,
                        marker,
                        source[int(group["goal_corner"])],
                        source[int(group["goal_reference"])],
                    ],
                    dtype=np.float32,
                )
                target_points = np.array(
                    [
                        target[int(group["front_corner"])],
                        target[front_reference_index],
                        target[int(group["spot"])],
                        target[int(group["goal_corner"])],
                        target[int(group["goal_reference"])],
                    ],
                    dtype=np.float32,
                )
                homography, _ = cv2.findHomography(source_points, target_points, 0)
                if homography is None or not np.all(np.isfinite(homography)):
                    continue
                errors = self._reprojection_errors(
                    source_points,
                    target_points,
                    homography,
                )
                error = float(np.median(errors))
                max_error = float(np.max(errors))
                player_ok, player_ratio, player_median_x = self._validate_player_projection(
                    homography,
                    players,
                    end=str(group["end"]),
                )
                if (
                    not np.isfinite(error)
                    or error > 850.0
                    or max_error > 1250.0
                    or not player_ok
                ):
                    continue
                end_penalty = 0.0
                if group["end"] == "right" and player_median_x < 9000.0:
                    end_penalty = 5000.0
                if group["end"] == "left" and player_median_x > 3000.0:
                    end_penalty = 5000.0
                score = error + end_penalty + (1.0 - player_ratio) * 2000.0
                if score < best_score:
                    best = (homography, error, len(source_points), player_ratio)
                    best_score = score
            if best is not None:
                self.line_refinements += 1
                break
        return best

    def _refine_penalty_corner(
        self,
        frame: np.ndarray,
        front_reference: tuple[float, float],
    ) -> tuple[tuple[float, float], np.ndarray] | None:
        frame_height, frame_width = frame.shape[:2]
        scale = min(1.0, 1920.0 / max(1, frame_width))
        if scale < 1.0:
            sample = cv2.resize(
                frame,
                (int(round(frame_width * scale)), int(round(frame_height * scale))),
                interpolation=cv2.INTER_AREA,
            )
        else:
            sample = frame
        sample_height, sample_width = sample.shape[:2]
        hsv = cv2.cvtColor(sample, cv2.COLOR_BGR2HSV)
        green = cv2.inRange(hsv, (25, 35, 30), (100, 255, 255))
        green_near = cv2.dilate(
            green,
            np.ones((max(9, sample_width // 95), max(9, sample_width // 95)), np.uint8),
        )
        white = cv2.inRange(hsv, (0, 0, 145), (180, 92, 255))
        line_mask = cv2.bitwise_and(white, green_near)
        line_mask = cv2.morphologyEx(
            line_mask,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_RECT, (9, 3)),
        )
        detected = cv2.HoughLinesP(
            line_mask,
            1,
            np.pi / 360,
            threshold=max(55, sample_width // 24),
            minLineLength=max(70, int(sample_width * 0.08)),
            maxLineGap=max(24, int(sample_width * 0.025)),
        )
        if detected is None:
            return None

        segments: list[tuple[np.ndarray, tuple[float, float, float, float], float]] = []
        for values in detected.reshape(-1, 4):
            x1, y1, x2, y2 = (float(value) / scale for value in values)
            segment = (x1, y1, x2, y2)
            length = float(np.hypot(x2 - x1, y2 - y1))
            line = self._line_from_segment(segment)
            if line is None:
                continue
            segments.append((line, segment, length))

        front_candidates = [
            item
            for item in segments
            if item[2] >= frame_width * 0.25
            and self._point_line_distance(front_reference, item[0]) <= frame_height * 0.07
        ]
        if not front_candidates:
            return None
        front_line, front_segment, _ = min(
            front_candidates,
            key=lambda item: (
                self._point_line_distance(front_reference, item[0]),
                -item[2],
            ),
        )
        front_angle = self._segment_angle(front_segment)

        best_corner: tuple[float, float] | None = None
        best_score = float("inf")
        for side_line, side_segment, side_length in segments:
            if side_length < frame_width * 0.08:
                continue
            angle_gap = self._angle_gap(front_angle, self._segment_angle(side_segment))
            if angle_gap < 20.0:
                continue
            corner = self._line_intersection(front_line, side_line)
            if corner is None:
                continue
            if not (
                -frame_width * 0.05 <= corner[0] <= frame_width * 1.05
                and -frame_height * 0.05 <= corner[1] <= frame_height * 1.05
            ):
                continue
            endpoint_distance = min(
                float(np.hypot(corner[0] - side_segment[0], corner[1] - side_segment[1])),
                float(np.hypot(corner[0] - side_segment[2], corner[1] - side_segment[3])),
            )
            if endpoint_distance > frame_width * 0.045:
                continue
            far_endpoint_distance = max(
                float(np.hypot(corner[0] - side_segment[0], corner[1] - side_segment[1])),
                float(np.hypot(corner[0] - side_segment[2], corner[1] - side_segment[3])),
            )
            if far_endpoint_distance < frame_height * 0.08:
                continue
            score = endpoint_distance - side_length * 0.02
            if score < best_score:
                best_corner = corner
                best_score = score
        if best_corner is None:
            return None
        return best_corner, front_line

    def _validate_player_projection(
        self,
        homography: np.ndarray,
        players: list[AnalysisObject],
        end: str | None,
    ) -> tuple[bool, float, float]:
        if not players:
            return True, 1.0, PITCH_LENGTH_CM / 2
        feet = np.array(
            [
                ((player.bbox[0] + player.bbox[2]) / 2, player.bbox[3])
                for player in players
            ],
            dtype=np.float32,
        )
        transformed = cv2.perspectiveTransform(
            feet.reshape(-1, 1, 2),
            homography,
        ).reshape(-1, 2)
        finite = np.all(np.isfinite(transformed), axis=1)
        inside = (
            finite
            & (transformed[:, 0] >= -300.0)
            & (transformed[:, 0] <= PITCH_LENGTH_CM + 300.0)
            & (transformed[:, 1] >= -300.0)
            & (transformed[:, 1] <= PITCH_WIDTH_CM + 300.0)
        )
        ratio = float(np.count_nonzero(inside) / max(1, len(players)))
        median_x = float(np.median(transformed[inside, 0])) if np.any(inside) else -1.0
        end_ok = True
        if end == "right":
            end_ok = median_x >= PITCH_LENGTH_CM - 2500.0
        elif end == "left":
            end_ok = 0.0 <= median_x <= 2500.0
        return ratio >= 0.75 and end_ok, ratio, median_x

    def _reprojection_errors(
        self,
        source: np.ndarray,
        target: np.ndarray,
        homography: np.ndarray,
    ) -> np.ndarray:
        projected = cv2.perspectiveTransform(
            source.astype(np.float32).reshape(-1, 1, 2),
            homography,
        ).reshape(-1, 2)
        return np.linalg.norm(projected - target, axis=1)

    def _line_from_segment(
        self,
        segment: tuple[float, float, float, float],
    ) -> np.ndarray | None:
        x1, y1, x2, y2 = segment
        line = np.cross(
            np.array([x1, y1, 1.0], dtype=np.float64),
            np.array([x2, y2, 1.0], dtype=np.float64),
        )
        norm = float(np.hypot(line[0], line[1]))
        if norm <= 1e-6:
            return None
        return line / norm

    def _point_line_distance(
        self,
        point: tuple[float, float],
        line: np.ndarray,
    ) -> float:
        return abs(float(line[0] * point[0] + line[1] * point[1] + line[2]))

    def _project_to_line(
        self,
        point: tuple[float, float],
        line: np.ndarray,
    ) -> tuple[float, float]:
        signed_distance = float(line[0] * point[0] + line[1] * point[1] + line[2])
        return (
            point[0] - line[0] * signed_distance,
            point[1] - line[1] * signed_distance,
        )

    def _line_intersection(
        self,
        first: np.ndarray,
        second: np.ndarray,
    ) -> tuple[float, float] | None:
        point = np.cross(first, second)
        if abs(float(point[2])) <= 1e-6:
            return None
        return (float(point[0] / point[2]), float(point[1] / point[2]))

    def _segment_angle(self, segment: tuple[float, float, float, float]) -> float:
        return float(np.degrees(np.arctan2(
            segment[3] - segment[1],
            segment[2] - segment[0],
        )))

    def _angle_gap(self, first: float, second: float) -> float:
        gap = abs(first - second) % 180.0
        return min(gap, 180.0 - gap)

    def draw(
        self,
        frame: np.ndarray,
        frame_index: int,
        players: list[AnalysisObject],
        balls: list[AnalysisObject],
        team_by_track: dict[int, int],
    ) -> None:
        if (
            not self.is_reliable()
            or self.last_calibrated_frame < 0
            or frame_index - self.last_calibrated_frame > self.stride * 3
        ):
            return
        frame_height, frame_width = frame.shape[:2]
        radar_width = max(360, min(720, int(frame_width * 0.30)))
        radar_height = int(round(radar_width * PITCH_WIDTH_CM / PITCH_LENGTH_CM))
        radar = np.full((radar_height, radar_width, 3), (43, 108, 45), dtype=np.uint8)
        margin = max(10, int(round(radar_width * 0.026)))
        line_color = (225, 238, 225)
        thickness = max(1, int(round(radar_width / 360)))

        def pitch_point(x_cm: float, y_cm: float) -> tuple[int, int]:
            usable_width = radar_width - margin * 2
            usable_height = radar_height - margin * 2
            return (
                int(round(margin + x_cm / PITCH_LENGTH_CM * usable_width)),
                int(round(margin + y_cm / PITCH_WIDTH_CM * usable_height)),
            )

        cv2.rectangle(
            radar,
            pitch_point(0, 0),
            pitch_point(PITCH_LENGTH_CM, PITCH_WIDTH_CM),
            line_color,
            thickness,
        )
        cv2.line(
            radar,
            pitch_point(PITCH_LENGTH_CM / 2, 0),
            pitch_point(PITCH_LENGTH_CM / 2, PITCH_WIDTH_CM),
            line_color,
            thickness,
        )
        center = pitch_point(PITCH_LENGTH_CM / 2, PITCH_WIDTH_CM / 2)
        circle_radius = max(
            3,
            int(round(CENTER_CIRCLE_RADIUS_CM / PITCH_LENGTH_CM * (radar_width - margin * 2))),
        )
        cv2.circle(radar, center, circle_radius, line_color, thickness)
        cv2.circle(radar, center, max(2, thickness + 1), line_color, cv2.FILLED)
        penalty_y1 = (PITCH_WIDTH_CM - PENALTY_AREA_WIDTH_CM) / 2
        penalty_y2 = (PITCH_WIDTH_CM + PENALTY_AREA_WIDTH_CM) / 2
        goal_y1 = (PITCH_WIDTH_CM - GOAL_AREA_WIDTH_CM) / 2
        goal_y2 = (PITCH_WIDTH_CM + GOAL_AREA_WIDTH_CM) / 2
        cv2.rectangle(radar, pitch_point(0, penalty_y1), pitch_point(PENALTY_AREA_LENGTH_CM, penalty_y2), line_color, thickness)
        cv2.rectangle(radar, pitch_point(PITCH_LENGTH_CM - PENALTY_AREA_LENGTH_CM, penalty_y1), pitch_point(PITCH_LENGTH_CM, penalty_y2), line_color, thickness)
        cv2.rectangle(radar, pitch_point(0, goal_y1), pitch_point(GOAL_AREA_LENGTH_CM, goal_y2), line_color, thickness)
        cv2.rectangle(radar, pitch_point(PITCH_LENGTH_CM - GOAL_AREA_LENGTH_CM, goal_y1), pitch_point(PITCH_LENGTH_CM, goal_y2), line_color, thickness)
        cv2.circle(radar, pitch_point(PENALTY_SPOT_DISTANCE_CM, PITCH_WIDTH_CM / 2), max(2, thickness + 1), line_color, cv2.FILLED)
        cv2.circle(radar, pitch_point(PITCH_LENGTH_CM - PENALTY_SPOT_DISTANCE_CM, PITCH_WIDTH_CM / 2), max(2, thickness + 1), line_color, cv2.FILLED)
        penalty_arc_angle = float(np.arccos(
            (PENALTY_AREA_LENGTH_CM - PENALTY_SPOT_DISTANCE_CM)
            / CENTER_CIRCLE_RADIUS_CM
        ))
        for penalty_x, angles in (
            (
                PENALTY_SPOT_DISTANCE_CM,
                np.linspace(-penalty_arc_angle, penalty_arc_angle, 28),
            ),
            (
                PITCH_LENGTH_CM - PENALTY_SPOT_DISTANCE_CM,
                np.linspace(
                    np.pi - penalty_arc_angle,
                    np.pi + penalty_arc_angle,
                    28,
                ),
            ),
        ):
            arc = np.array(
                [
                    pitch_point(
                        penalty_x + CENTER_CIRCLE_RADIUS_CM * np.cos(angle),
                        PITCH_WIDTH_CM / 2 + CENTER_CIRCLE_RADIUS_CM * np.sin(angle),
                    )
                    for angle in angles
                ],
                dtype=np.int32,
            )
            cv2.polylines(radar, [arc], False, line_color, thickness, cv2.LINE_AA)

        for player in players:
            pitch_xy = self.transform_point(
                ((player.bbox[0] + player.bbox[2]) / 2, player.bbox[3])
            )
            if pitch_xy is None:
                continue
            point = pitch_point(*pitch_xy)
            team = team_by_track.get(player.track_id, 1)
            color = TEAM_DISPLAY_COLORS.get(team, TEAM_DISPLAY_COLORS[1])
            radius = max(8, int(round(radar_width / 70)))
            cv2.circle(radar, point, radius, color, cv2.FILLED)
            cv2.circle(radar, point, radius, (20, 24, 20), thickness)
            text_color = (15, 15, 15) if team == 1 else (255, 255, 255)
            label = str(player.track_id)
            font_scale = max(0.28, radar_width / 1500)
            text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)[0]
            cv2.putText(
                radar,
                label,
                (point[0] - text_size[0] // 2, point[1] + text_size[1] // 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                text_color,
                1,
                cv2.LINE_AA,
            )

        for ball in balls[:1]:
            pitch_xy = ball.pitch_position or self.transform_point(
                (
                    (ball.bbox[0] + ball.bbox[2]) / 2,
                    (ball.bbox[1] + ball.bbox[3]) / 2,
                )
            )
            if pitch_xy is not None:
                cv2.circle(radar, pitch_point(*pitch_xy), max(5, radar_width // 120), (0, 230, 255), cv2.FILLED)
                cv2.circle(radar, pitch_point(*pitch_xy), max(5, radar_width // 120), (20, 20, 20), thickness)

        x1 = (frame_width - radar_width) // 2
        y1 = frame_height - radar_height - max(14, frame_height // 80)
        region = frame[y1 : y1 + radar_height, x1 : x1 + radar_width]
        cv2.addWeighted(radar, 0.82, region, 0.18, 0.0, dst=region)
        cv2.rectangle(
            frame,
            (x1, y1),
            (x1 + radar_width - 1, y1 + radar_height - 1),
            (24, 30, 24),
            thickness,
        )
        self.rendered_frames += 1

    def quality_gate(self) -> dict[str, Any]:
        confidence_values = [float(item["confidence"]) for item in self.frame_confidence]
        total_frames = len(confidence_values)
        reliable_frames = sum(1 for value in confidence_values if value >= 0.58)
        reliable_ratio = reliable_frames / max(1, total_frames)
        average_confidence = float(np.mean(confidence_values)) if confidence_values else 0.0
        longest_unreliable_streak = 0
        current_unreliable_streak = 0
        for item in self.frame_confidence:
            if bool(item.get("reliable")):
                current_unreliable_streak = 0
            else:
                current_unreliable_streak += 1
                longest_unreliable_streak = max(
                    longest_unreliable_streak,
                    current_unreliable_streak,
                )
        unrecovered_cuts = sum(
            1 for item in self.camera_cuts if item.get("recovered_frame") is None
        )
        maximum_allowed_gap = max(self.stride * 4, int(round(total_frames * 0.08)))
        reprojection_ok = (
            self.last_reprojection_error_cm is not None
            and self.last_reprojection_error_cm <= 260.0
        )
        line_alignment_ok = (
            self.last_line_alignment_score is not None
            and self.last_line_alignment_score >= 0.30
        )
        conditions = [
            {
                "code": "homography_available",
                "passed": self.homography is not None,
                "value": self.calibration_mode,
                "required": "valid metric homography",
            },
            {
                "code": "reliable_frame_coverage",
                "passed": reliable_ratio >= 0.90,
                "value": round(reliable_ratio, 4),
                "required": 0.90,
            },
            {
                "code": "average_confidence",
                "passed": average_confidence >= 0.62,
                "value": round(average_confidence, 4),
                "required": 0.62,
            },
            {
                "code": "maximum_unreliable_streak",
                "passed": longest_unreliable_streak <= maximum_allowed_gap,
                "value": longest_unreliable_streak,
                "required": maximum_allowed_gap,
            },
            {
                "code": "camera_cut_recovery",
                "passed": unrecovered_cuts == 0,
                "value": unrecovered_cuts,
                "required": 0,
            },
            {
                "code": "metric_reprojection_error",
                "passed": reprojection_ok,
                "value": self.last_reprojection_error_cm,
                "required": "<= 260 cm",
            },
            {
                "code": "pitch_line_alignment",
                "passed": line_alignment_ok,
                "value": self.last_line_alignment_score,
                "required": ">= 0.30",
            },
        ]
        failed = [item["code"] for item in conditions if not item["passed"]]
        status = "passed" if not failed else "needs_manual_calibration"
        return {
            "status": status,
            "metric_outputs_verified": status == "passed",
            "conditions": conditions,
            "failed_conditions": failed,
            "reliable_ratio": round(reliable_ratio, 4),
            "longest_unreliable_streak_frames": longest_unreliable_streak,
            "camera_cuts_detected": len(self.camera_cuts),
            "camera_cuts_recovered": self.camera_cut_recoveries,
            "unrecovered_camera_cuts": unrecovered_cuts,
            "manual_fallback_available": True,
            "manual_fallback_used": self.manual_calibrations > 0,
        }

    def summary(self) -> dict[str, Any]:
        confidence_values = [float(item["confidence"]) for item in self.frame_confidence]
        return {
            "engine": "pitch_calibration_v3_quality_gate",
            "model_available": self.model is not None,
            "model_image_size": max(960, settings.YOLO_IMAGE_SIZE),
            "calibration_mode": self.calibration_mode,
            "calibration_source": self.calibration_source,
            "confidence": {
                "current": round(self.calibration_confidence, 4),
                "average": round(float(np.mean(confidence_values)), 4) if confidence_values else 0.0,
                "minimum": round(float(np.min(confidence_values)), 4) if confidence_values else 0.0,
                "reliable_frames": sum(1 for value in confidence_values if value >= 0.58),
                "total_frames": len(confidence_values),
                "threshold": 0.58,
                "per_frame": self.frame_confidence,
            },
            "calibration_attempts": self.attempts,
            "successful_calibrations": self.successes,
            "manual_calibrations": self.manual_calibrations,
            "manual_calibration_rejections": self.manual_calibration_rejections,
            "model_refreshes": self.model_refreshes,
            "temporal_blends": self.temporal_blends,
            "assisted_calibrations": self.assisted_calibrations,
            "bootstrap_keypoint_calibrations": self.bootstrap_keypoint_calibrations,
            "wide_view_calibrations": self.wide_view_calibrations,
            "rejected_local_calibrations": self.rejected_local_calibrations,
            "rejected_geometry": self.rejected_geometry,
            "line_refinements": self.line_refinements,
            "goal_geometry_attempts": self.goal_geometry_attempts,
            "goal_geometry_calibrations": self.goal_geometry_calibrations,
            "goal_geometry_rejections": self.goal_geometry_rejections,
            "last_line_alignment_score": self.last_line_alignment_score,
            "camera_tracking": {
                "engine": "field_masked_bidirectional_lk_homography",
                "attempts": self.camera_tracking_attempts,
                "successes": self.camera_tracking_successes,
                "failures": self.camera_tracking_failures,
                "last_inliers": self.last_camera_inliers,
                "last_inlier_ratio": self.last_camera_inlier_ratio,
                "last_reprojection_error_px": self.last_camera_reprojection_error_px,
            },
            "camera_cuts": self.camera_cuts,
            "quality_gate": self.quality_gate(),
            "visual_marker_observations": self.visual_marker_observations,
            "rendered_frames": self.rendered_frames,
            "last_visible_keypoints": self.last_visible_keypoints,
            "last_inliers": self.last_inliers,
            "last_reprojection_error_cm": self.last_reprojection_error_cm,
            "last_target_span_cm": {
                "x": self.last_target_span_cm[0],
                "y": self.last_target_span_cm[1],
            },
            "last_player_valid_ratio": self.last_player_valid_ratio,
            "coordinate_system": "metric_pitch_ground_plane_centimeters",
            "pitch_template": {
                "name": "standard_105x68",
                "length_cm": PITCH_LENGTH_CM,
                "width_cm": PITCH_WIDTH_CM,
                "penalty_area_length_cm": PENALTY_AREA_LENGTH_CM,
                "penalty_area_width_cm": PENALTY_AREA_WIDTH_CM,
                "goal_area_length_cm": GOAL_AREA_LENGTH_CM,
                "goal_area_width_cm": GOAL_AREA_WIDTH_CM,
                "goal_width_cm": GOAL_WIDTH_CM,
                "penalty_spot_distance_cm": PENALTY_SPOT_DISTANCE_CM,
                "penalty_arc_radius_cm": CENTER_CIRCLE_RADIUS_CM,
                "center_circle_radius_cm": CENTER_CIRCLE_RADIUS_CM,
            },
            "projection_model": "metric_pinhole_3d_from_dynamic_pitch_homography",
            "ground_plane_3d": {"z_cm": 0.0},
            "ball_3d_projection": {
                "intrinsics": "frame_scaled_pinhole_estimate",
                "projection_attempts": self.ball_3d_projection_attempts,
                "projection_successes": self.ball_3d_projection_successes,
                "backprojection_attempts": self.ball_3d_backprojection_attempts,
                "backprojection_successes": self.ball_3d_backprojection_successes,
            },
            "errors": self.errors,
        }

    def record_frame_confidence(self, frame_index: int) -> None:
        if self.homography is None:
            self.calibration_confidence = 0.0
            self.calibration_source = "unavailable"
        elif self.last_calibrated_frame >= 0:
            stale = max(0, frame_index - self.last_calibrated_frame)
            if stale > self.stride * 2:
                self.calibration_confidence = max(
                    0.0,
                    self.calibration_confidence * (0.985 ** (stale - self.stride * 2)),
                )
        self.frame_confidence.append(
            {
                "frame": frame_index,
                "confidence": round(self.calibration_confidence, 4),
                "source": self.calibration_source,
                "reliable": self.is_reliable(),
                "reprojection_error_cm": self.last_reprojection_error_cm,
                "line_alignment_score": self.last_line_alignment_score,
                "camera_cut": frame_index == self.last_camera_cut_frame,
            }
        )

    def is_reliable(self, threshold: float = 0.58) -> bool:
        return self.homography is not None and self.calibration_confidence >= threshold

    def contains_image_point(
        self,
        point: tuple[float, float],
        margin_cm: float = 0.0,
    ) -> bool:
        transformed = self._raw_transform(point)
        if transformed is None:
            return False
        return (
            -margin_cm <= transformed[0] <= PITCH_LENGTH_CM + margin_cm
            and -margin_cm <= transformed[1] <= PITCH_WIDTH_CM + margin_cm
        )

    def playing_surface_mask(self, frame: np.ndarray) -> np.ndarray | None:
        frame_height, frame_width = frame.shape[:2]
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        green = cv2.inRange(hsv, (24, 24, 22), (108, 255, 255))
        green = cv2.morphologyEx(
            green,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21)),
        )
        green = cv2.dilate(green, np.ones((9, 9), dtype=np.uint8))
        if self.homography is None or not self.is_reliable(0.52):
            component_count, labels, stats, _ = cv2.connectedComponentsWithStats(green, 8)
            if component_count <= 1:
                return green
            min_area = frame_height * frame_width * 0.025
            valid_labels = [index for index in range(1, component_count) if stats[index, cv2.CC_STAT_AREA] >= min_area]
            if not valid_labels:
                return green
            return np.where(np.isin(labels, valid_labels), 255, 0).astype(np.uint8)

        inverse = self.pitch_to_video_matrix(require_reliable=False)
        if inverse is None:
            return green
        pitch_boundary = np.array(
            [[0.0, 0.0], [PITCH_LENGTH_CM, 0.0], [PITCH_LENGTH_CM, PITCH_WIDTH_CM], [0.0, PITCH_WIDTH_CM]],
            dtype=np.float32,
        )
        projected = cv2.perspectiveTransform(pitch_boundary.reshape(-1, 1, 2), inverse).reshape(-1, 2)
        if not np.all(np.isfinite(projected)):
            return green
        polygon = np.zeros((frame_height, frame_width), dtype=np.uint8)
        cv2.fillConvexPoly(polygon, np.rint(projected).astype(np.int32), 255)
        line_tolerance = cv2.dilate(green, np.ones((15, 15), dtype=np.uint8))
        return cv2.bitwise_and(polygon, line_tolerance)

    def _raw_transform(self, point: tuple[float, float]) -> tuple[float, float] | None:
        if self.homography is None:
            return None
        transformed = cv2.perspectiveTransform(
            np.array(point, dtype=np.float32).reshape(1, 1, 2),
            self.homography,
        ).reshape(2)
        x_cm, y_cm = float(transformed[0]), float(transformed[1])
        if not np.isfinite(x_cm) or not np.isfinite(y_cm):
            return None
        return x_cm, y_cm

    def transform_point(
        self,
        point: tuple[float, float],
        require_reliable: bool = True,
    ) -> tuple[float, float] | None:
        if require_reliable and not self.is_reliable():
            return None
        transformed = self._raw_transform(point)
        if transformed is None:
            return None
        x_cm, y_cm = transformed
        if x_cm < -250 or x_cm > PITCH_LENGTH_CM + 250:
            return None
        if y_cm < -250 or y_cm > PITCH_WIDTH_CM + 250:
            return None
        return (
            min(PITCH_LENGTH_CM, max(0.0, x_cm)),
            min(PITCH_WIDTH_CM, max(0.0, y_cm)),
        )

    def pitch_to_video_matrix(self, require_reliable: bool = True) -> np.ndarray | None:
        """Return the current metric-pitch to source-video projection."""
        if self.homography is None or (require_reliable and not self.is_reliable()):
            return None
        try:
            inverse = np.linalg.inv(self.homography)
        except np.linalg.LinAlgError:
            return None
        if not np.all(np.isfinite(inverse)):
            return None
        scale = float(inverse[2, 2])
        if abs(scale) < 1e-9:
            return None
        return inverse / scale

    def camera_projection_matrix(
        self,
        frame_shape: tuple[int, int] | tuple[int, int, int],
        require_reliable: bool = True,
    ) -> np.ndarray | None:
        """Lift the pitch homography into a metric pinhole camera model.

        The pitch supplies the known Z=0 plane. Intrinsics are estimated from
        the tactical frame size, then the two planar rotation axes and camera
        translation are recovered from the homography. The returned matrix
        preserves the accepted ground-plane projection exactly while adding a
        physically oriented vertical axis for airborne objects.
        """
        ground_projection = self.pitch_to_video_matrix(require_reliable)
        if ground_projection is None:
            return None
        frame_height, frame_width = int(frame_shape[0]), int(frame_shape[1])
        focal_base = float(max(frame_width, frame_height))
        best: tuple[float, np.ndarray, bool] | None = None
        for focal_multiplier in (1.20, 1.55, 1.90, 2.30, 2.80, 0.95):
            focal = focal_base * focal_multiplier
            intrinsics = np.array(
                [
                    [focal, 0.0, frame_width / 2.0],
                    [0.0, focal, frame_height / 2.0],
                    [0.0, 0.0, 1.0],
                ],
                dtype=np.float64,
            )
            inverse_intrinsics = np.linalg.inv(intrinsics)
            for sign in (1.0, -1.0):
                signed_ground = ground_projection * sign
                first_axis = inverse_intrinsics @ signed_ground[:, 0]
                second_axis = inverse_intrinsics @ signed_ground[:, 1]
                first_norm = float(np.linalg.norm(first_axis))
                second_norm = float(np.linalg.norm(second_axis))
                metric_scale = (first_norm + second_norm) / 2.0
                if metric_scale <= 1e-9:
                    continue
                axis_orthogonality = abs(
                    float(np.dot(first_axis, second_axis))
                    / max(first_norm * second_norm, 1e-9)
                )
                scale_mismatch = abs(first_norm - second_norm) / metric_scale
                first_rotation = first_axis / metric_scale
                second_rotation = second_axis / metric_scale
                vertical_rotation = np.cross(first_rotation, second_rotation)
                vertical_norm = float(np.linalg.norm(vertical_rotation))
                if vertical_norm <= 1e-9:
                    continue
                vertical_rotation /= vertical_norm
                approximate_rotation = np.column_stack(
                    (first_rotation, second_rotation, vertical_rotation)
                )
                left, _, right = np.linalg.svd(approximate_rotation)
                rotation = left @ right
                if np.linalg.det(rotation) < 0:
                    rotation[:, 2] *= -1.0
                translation = (
                    inverse_intrinsics @ signed_ground[:, 2]
                ) / metric_scale
                camera_center = -rotation.T @ translation
                camera_height = float(camera_center[2])
                physically_plausible = 700.0 <= camera_height <= 7500.0
                if camera_height <= 0.0:
                    height_penalty = 20.0
                else:
                    height_penalty = abs(np.log(camera_height / 2800.0))
                physical_penalty = 0.0 if physically_plausible else 8.0
                score = -(
                    axis_orthogonality * 5.0
                    + scale_mismatch * 5.0
                    + height_penalty * 0.18
                    + physical_penalty
                )
                vertical_column = intrinsics @ (
                    rotation[:, 2] * metric_scale
                )
                projection = np.column_stack(
                    (
                        signed_ground[:, 0],
                        signed_ground[:, 1],
                        vertical_column,
                        signed_ground[:, 2],
                    )
                )
                if best is None or score > best[0]:
                    best = (score, projection, physically_plausible)
        if best is None or not best[2]:
            return None
        return best[1]

    def project_pitch_3d(
        self,
        point_xyz: tuple[float, float, float] | np.ndarray,
        frame_shape: tuple[int, int] | tuple[int, int, int],
        require_reliable: bool = True,
    ) -> tuple[float, float] | None:
        self.ball_3d_projection_attempts += 1
        projection = self.camera_projection_matrix(frame_shape, require_reliable)
        if projection is None:
            return None
        point = np.array(
            [float(point_xyz[0]), float(point_xyz[1]), float(point_xyz[2]), 1.0],
            dtype=np.float64,
        )
        image = projection @ point
        if not np.all(np.isfinite(image)) or abs(float(image[2])) <= 1e-9:
            return None
        pixel = image[:2] / image[2]
        frame_height, frame_width = int(frame_shape[0]), int(frame_shape[1])
        if not (
            -frame_width * 0.25 <= pixel[0] <= frame_width * 1.25
            and -frame_height * 0.25 <= pixel[1] <= frame_height * 1.25
        ):
            return None
        self.ball_3d_projection_successes += 1
        return float(pixel[0]), float(pixel[1])

    def backproject_image_at_height(
        self,
        point: tuple[float, float] | np.ndarray,
        height_cm: float,
        frame_shape: tuple[int, int] | tuple[int, int, int],
        require_reliable: bool = True,
    ) -> tuple[float, float] | None:
        self.ball_3d_backprojection_attempts += 1
        projection = self.camera_projection_matrix(frame_shape, require_reliable)
        if projection is None:
            return None
        u, v = float(point[0]), float(point[1])
        z = max(0.0, float(height_cm))
        first = projection[0] - u * projection[2]
        second = projection[1] - v * projection[2]
        system = np.array(
            [[first[0], first[1]], [second[0], second[1]]],
            dtype=np.float64,
        )
        target = -np.array(
            [
                first[2] * z + first[3],
                second[2] * z + second[3],
            ],
            dtype=np.float64,
        )
        if abs(float(np.linalg.det(system))) <= 1e-9:
            return None
        xy = np.linalg.solve(system, target)
        if not np.all(np.isfinite(xy)):
            return None
        if not (
            -1500.0 <= xy[0] <= PITCH_LENGTH_CM + 1500.0
            and -1500.0 <= xy[1] <= PITCH_WIDTH_CM + 1500.0
        ):
            return None
        self.ball_3d_backprojection_successes += 1
        return float(xy[0]), float(xy[1])

    def _pitch_vertices(self) -> np.ndarray:
        penalty_width = PENALTY_AREA_WIDTH_CM
        goal_width = GOAL_AREA_WIDTH_CM
        penalty_length = PENALTY_AREA_LENGTH_CM
        goal_length = GOAL_AREA_LENGTH_CM
        penalty_spot = PENALTY_SPOT_DISTANCE_CM
        center_circle = CENTER_CIRCLE_RADIUS_CM
        return np.array(
            [
                (0, 0),
                (0, (PITCH_WIDTH_CM - penalty_width) / 2),
                (0, (PITCH_WIDTH_CM - goal_width) / 2),
                (0, (PITCH_WIDTH_CM + goal_width) / 2),
                (0, (PITCH_WIDTH_CM + penalty_width) / 2),
                (0, PITCH_WIDTH_CM),
                (goal_length, (PITCH_WIDTH_CM - goal_width) / 2),
                (goal_length, (PITCH_WIDTH_CM + goal_width) / 2),
                (penalty_spot, PITCH_WIDTH_CM / 2),
                (penalty_length, (PITCH_WIDTH_CM - penalty_width) / 2),
                (penalty_length, (PITCH_WIDTH_CM - goal_width) / 2),
                (penalty_length, (PITCH_WIDTH_CM + goal_width) / 2),
                (penalty_length, (PITCH_WIDTH_CM + penalty_width) / 2),
                (PITCH_LENGTH_CM / 2, 0),
                (PITCH_LENGTH_CM / 2, PITCH_WIDTH_CM / 2 - center_circle),
                (PITCH_LENGTH_CM / 2, PITCH_WIDTH_CM / 2 + center_circle),
                (PITCH_LENGTH_CM / 2, PITCH_WIDTH_CM),
                (PITCH_LENGTH_CM - penalty_length, (PITCH_WIDTH_CM - penalty_width) / 2),
                (PITCH_LENGTH_CM - penalty_length, (PITCH_WIDTH_CM - goal_width) / 2),
                (PITCH_LENGTH_CM - penalty_length, (PITCH_WIDTH_CM + goal_width) / 2),
                (PITCH_LENGTH_CM - penalty_length, (PITCH_WIDTH_CM + penalty_width) / 2),
                (PITCH_LENGTH_CM - penalty_spot, PITCH_WIDTH_CM / 2),
                (PITCH_LENGTH_CM - goal_length, (PITCH_WIDTH_CM - goal_width) / 2),
                (PITCH_LENGTH_CM - goal_length, (PITCH_WIDTH_CM + goal_width) / 2),
                (PITCH_LENGTH_CM, 0),
                (PITCH_LENGTH_CM, (PITCH_WIDTH_CM - penalty_width) / 2),
                (PITCH_LENGTH_CM, (PITCH_WIDTH_CM - goal_width) / 2),
                (PITCH_LENGTH_CM, (PITCH_WIDTH_CM + goal_width) / 2),
                (PITCH_LENGTH_CM, (PITCH_WIDTH_CM + penalty_width) / 2),
                (PITCH_LENGTH_CM, PITCH_WIDTH_CM),
                (PITCH_LENGTH_CM / 2 - center_circle, PITCH_WIDTH_CM / 2),
                (PITCH_LENGTH_CM / 2 + center_circle, PITCH_WIDTH_CM / 2),
            ],
            dtype=np.float32,
        )


class MatchAnalysisPlusRunner:
    legacy_modes = {
        "PLAYER_DETECTION",
        "BALL_DETECTION",
        "PLAYER_TRACKING",
        "TEAM_CLASSIFICATION",
        "RADAR",
    }
    supported_modes = {
        "FULL_ANALYSIS",
        *legacy_modes,
    }

    def __init__(self) -> None:
        self.models: dict[str, Any] = {}
        self.pitch_model = None
        self.general_model_path = self._resolve_asset_path(settings.YOLO_MODEL_PATH) or Path(
            settings.YOLO_MODEL_PATH
        )
        self.specialized_model_path = self._resolve_asset_path(
            settings.MATCH_ANALYSIS_PLAYER_MODEL_PATH
        )
        self.legacy_specialized_model_path = self._resolve_asset_path(
            settings.MATCH_ANALYSIS_PLAYER_MODEL_FALLBACK_PATH
        )
        self.model_path = self.specialized_model_path or self.general_model_path
        self.ball_model_path = (
            self._resolve_asset_path(settings.MATCH_ANALYSIS_BALL_MODEL_PATH)
            or self._resolve_asset_path(
                settings.MATCH_ANALYSIS_BALL_MODEL_FALLBACK_PATH
            )
        )
        self.pitch_model_path = None
        self.pitch_model_candidates = self._unique_model_paths(
            [
                (
                    "pitch_v2",
                    self._resolve_asset_path(settings.MATCH_ANALYSIS_PITCH_MODEL_PATH),
                ),
                (
                    "pitch_v1_fallback",
                    self._resolve_asset_path(
                        settings.MATCH_ANALYSIS_PITCH_MODEL_FALLBACK_PATH
                    ),
                ),
            ]
        )
        self.model_mode = "unloaded"
        self.active_image_size = max(640, settings.MATCH_ANALYSIS_IMAGE_SIZE)
        self.model_selection: dict[str, Any] = {}
        self.pitch_model_selection: dict[str, Any] = {
            "strategy": "unavailable",
            "selected": None,
            "candidates": {},
        }
        self.ball_detection_mode = "shared_detector"
        self.analytics_engine = AnalyticsRealV1()
        self.report_builder = ReportsV2Builder()

    def run(
        self,
        run_id: int,
        match_id: int,
        bucket: str,
        object_name: str,
        artifact_prefix: str,
        mode: str = "FULL_ANALYSIS",
        max_frames: int = 450,
        start_frame: int = 0,
        calibration_points: list[dict[str, float]] | None = None,
        team_context: dict[str, Any] | None = None,
        reuse_detections_object: str | None = None,
        reuse_model_mode: str | None = None,
        reuse_ball_detection_mode: str | None = None,
        progress_callback: Callable[[dict[str, Any]], None] | None = None,
    ) -> dict[str, Any]:
        requested_mode = (mode or "FULL_ANALYSIS").upper()
        if requested_mode not in self.supported_modes:
            raise ValueError(f"Unsupported match analysis mode: {mode}")
        normalized_mode = "FULL_ANALYSIS"

        with TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            input_path = temp_path / Path(object_name).name
            raw_output_path = temp_path / "match_analysis_plus.avi"
            output_path = temp_path / "match_analysis_plus.mp4"
            thumbnail_path = temp_path / "thumbnail.jpg"
            quality_crops_dir = temp_path / "tracking-quality-crops"
            quality_predictions_path = temp_path / "tracking_quality_predictions.jsonl"
            detection_cache_path = temp_path / "detections.jsonl"
            reuse_detection_cache_path = temp_path / "reused_detections.jsonl"
            quality_crops_dir.mkdir(parents=True, exist_ok=True)

            if progress_callback is not None:
                progress_callback(
                    {
                        "stage": "downloading_source",
                        "processed_frames": 0,
                        "total_frames": max_frames if max_frames > 0 else None,
                        "percent": 0.0,
                        "processing_fps": 0.0,
                        "eta_seconds": None,
                        "cache_hit_frames": 0,
                    }
                )
            self._download_video(bucket, object_name, input_path)
            if reuse_detections_object:
                self._download_video(
                    bucket,
                    reuse_detections_object,
                    reuse_detection_cache_path,
                )
            if progress_callback is not None:
                progress_callback(
                    {
                        "stage": "preparing_models",
                        "processed_frames": 0,
                        "total_frames": max_frames if max_frames > 0 else None,
                        "percent": 0.0,
                        "processing_fps": 0.0,
                        "eta_seconds": None,
                        "cache_hit_frames": 0,
                    }
                )
            summary = self._process_video(
                input_path=input_path,
                raw_output_path=raw_output_path,
                output_path=output_path,
                thumbnail_path=thumbnail_path,
                quality_crops_dir=quality_crops_dir,
                quality_predictions_path=quality_predictions_path,
                mode=normalized_mode,
                max_frames=max_frames,
                start_frame=max(0, start_frame),
                calibration_points=calibration_points or [],
                team_context=team_context or {},
                detection_cache_path=detection_cache_path,
                reuse_detection_cache_path=(
                    reuse_detection_cache_path if reuse_detections_object else None
                ),
                reuse_model_mode=reuse_model_mode,
                reuse_ball_detection_mode=reuse_ball_detection_mode,
                progress_callback=progress_callback,
            )

            visual_layers_payload = summary.pop("_visual_layers_payload", None)
            analytics_payload = summary.pop("_analytics_payload", None)

            if progress_callback is not None:
                progress_callback(
                    {
                        "stage": "building_artifacts",
                        "processed_frames": summary.get("frames_processed", 0),
                        "total_frames": summary.get("frames_processed", 0),
                        "percent": 99.0,
                        "processing_fps": summary.get("processing_fps", 0.0),
                        "eta_seconds": None,
                        "cache_hit_frames": (summary.get("performance") or {}).get(
                            "cache_hit_frames",
                            0,
                        ),
                    }
                )

            output_object = f"{artifact_prefix}/output.mp4"
            summary_object = f"{artifact_prefix}/summary.json"
            thumbnail_object = f"{artifact_prefix}/thumbnail.jpg"
            visual_layers_object = f"{artifact_prefix}/visual_layers.json"
            analytics_object = f"{artifact_prefix}/analytics_v1.json"
            detection_cache_object = f"{artifact_prefix}/performance/detections.jsonl"
            report_object = f"{artifact_prefix}/reports-v2/report.json"
            report_pdf_object = f"{artifact_prefix}/reports-v2/report.pdf"
            team_chart_object = f"{artifact_prefix}/reports-v2/team-overview.png"
            heatmap_atlas_object = f"{artifact_prefix}/reports-v2/player-heatmaps.png"
            quality_predictions_object = (
                f"{artifact_prefix}/tracking-quality/predictions.jsonl"
            )
            self._put_file(bucket, output_object, output_path, "video/mp4")
            if thumbnail_path.exists():
                self._put_file(bucket, thumbnail_object, thumbnail_path, "image/jpeg")
            if visual_layers_payload is not None:
                self._put_json(bucket, visual_layers_object, visual_layers_payload)
                summary["visual_layers"] = {
                    "status": "ready",
                    "object_name": visual_layers_object,
                    "schema_version": visual_layers_payload["schema_version"],
                    "tracks_count": len(visual_layers_payload["tracks"]),
                    "movement_sample_rate_hz": visual_layers_payload[
                        "movement_sample_rate_hz"
                    ],
                    "heatmap_sample_rate_hz": visual_layers_payload[
                        "heatmap_sample_rate_hz"
                    ],
                }
            if analytics_payload is not None:
                self._put_json(bucket, analytics_object, analytics_payload)
                summary["analytics_real_v1"] = {
                    **analytics_payload,
                    "object_name": analytics_object,
                }
                report_payload = self.report_builder.build(
                    analytics_payload,
                    summary,
                    team_context or {},
                )
                report_payload["artifacts"] = {
                    "json": report_object,
                    "pdf": report_pdf_object,
                    "team_chart": team_chart_object,
                    "player_heatmaps": heatmap_atlas_object,
                }
                self._put_bytes(
                    bucket,
                    team_chart_object,
                    self.report_builder.team_chart_png(report_payload),
                    "image/png",
                )
                self._put_bytes(
                    bucket,
                    heatmap_atlas_object,
                    self.report_builder.heatmap_atlas_png(report_payload),
                    "image/png",
                )
                self._put_bytes(
                    bucket,
                    report_pdf_object,
                    self.report_builder.pdf(report_payload),
                    "application/pdf",
                )
                self._put_json(bucket, report_object, report_payload)
                summary["reports_v2"] = {
                    "status": "ready",
                    "schema_version": report_payload["schema_version"],
                    "teams_count": len(report_payload["teams"]),
                    "players_count": len(report_payload["players"]),
                    "artifacts": report_payload["artifacts"],
                }
            if detection_cache_path.exists():
                self._put_file(
                    bucket,
                    detection_cache_object,
                    detection_cache_path,
                    "application/x-ndjson",
                )
                performance = dict(summary.get("performance") or {})
                performance["detection_cache"] = {
                    "status": "ready",
                    "object_name": detection_cache_object,
                    "source": "reused" if reuse_detections_object else "generated",
                    "reusable_without_yolo": True,
                }
                summary["performance"] = performance
            tracking_quality = summary.get("tracking_quality")
            if tracking_quality is not None:
                for track in tracking_quality.get("tracks", []):
                    crop_objects: list[dict[str, Any]] = []
                    for crop in track.pop("crop_files", []):
                        crop_path = quality_crops_dir / str(crop["file_name"])
                        if not crop_path.exists():
                            continue
                        crop_object = (
                            f"{artifact_prefix}/tracking-quality/crops/{crop_path.name}"
                        )
                        self._put_file(bucket, crop_object, crop_path, "image/jpeg")
                        crop_objects.append(
                            {
                                "frame": int(crop["frame"]),
                                "object_name": crop_object,
                                "confidence": crop.get("confidence"),
                            }
                        )
                    track["crop_objects"] = crop_objects
                if quality_predictions_path.exists():
                    self._put_file(
                        bucket,
                        quality_predictions_object,
                        quality_predictions_path,
                        "application/x-ndjson",
                    )
                    tracking_quality["predictions_object"] = quality_predictions_object

            payload = {
                **summary,
                "run_id": run_id,
                "match_id": match_id,
                "mode": normalized_mode,
                "input_object": object_name,
                "output_object": output_object,
                "summary_object": summary_object,
                "thumbnail_object": thumbnail_object if thumbnail_path.exists() else None,
                "source_project": "apps/match-analysis-worker/sports-main",
                "worker": "match-analysis-worker",
            }
            self._put_json(bucket, summary_object, payload)
            if progress_callback is not None:
                progress_callback(
                    {
                        "stage": "completed",
                        "processed_frames": summary.get("frames_processed", 0),
                        "total_frames": summary.get("frames_processed", 0),
                        "percent": 100.0,
                        "processing_fps": summary.get("processing_fps", 0.0),
                        "eta_seconds": 0.0,
                        "cache_hit_frames": (summary.get("performance") or {}).get(
                            "cache_hit_frames",
                            0,
                        ),
                    }
                )
            return payload

    def _process_video(
        self,
        input_path: Path,
        raw_output_path: Path,
        output_path: Path,
        thumbnail_path: Path,
        quality_crops_dir: Path,
        quality_predictions_path: Path,
        mode: str,
        max_frames: int,
        start_frame: int,
        calibration_points: list[dict[str, float]],
        team_context: dict[str, Any],
        detection_cache_path: Path,
        reuse_detection_cache_path: Path | None,
        reuse_model_mode: str | None,
        reuse_ball_detection_mode: str | None,
        progress_callback: Callable[[dict[str, Any]], None] | None,
    ) -> dict[str, Any]:
        start = perf_counter()
        capture = cv2.VideoCapture(str(input_path))
        if not capture.isOpened():
            raise ValueError("Could not open uploaded video for match analysis")

        fps = capture.get(cv2.CAP_PROP_FPS) or 24.0
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 1280)
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 720)
        source_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        requested_start_frame = max(0, start_frame)
        if source_frames > 0:
            requested_start_frame = min(requested_start_frame, max(0, source_frames - 1))
        if requested_start_frame:
            capture.set(cv2.CAP_PROP_POS_FRAMES, requested_start_frame)
        actual_start_frame = int(capture.get(cv2.CAP_PROP_POS_FRAMES) or requested_start_frame)
        cached_objects_by_frame = self._load_detection_cache(
            reuse_detection_cache_path,
            expected_start_frame=actual_start_frame,
        )
        cached_specialized_ball_active = bool(
            cached_objects_by_frame
            and reuse_ball_detection_mode
            in {
                "dedicated_football_model_every_2_frames",
                "dedicated_football_model_every_frame_high_resolution",
                "cached_dedicated_ball_detections",
            }
        )
        cached_specialized_player_active = bool(
            cached_objects_by_frame
            and reuse_model_mode
            in {
                "football-specialized-yolo",
                "cached-football-specialized-detections",
            }
        )
        model = None
        if cached_objects_by_frame:
            self.model_mode = "cached-football-detections"
            self.ball_detection_mode = (
                "cached_dedicated_ball_detections"
                if cached_specialized_ball_active
                else "cached_shared_ball_detections"
            )
            self.model_selection = {
                "strategy": "reusable_detection_cache_v1",
                "selected": "cached_detections",
                "reason": "YOLO inference skipped for cached frames",
                "cached_frames": len(cached_objects_by_frame),
                "analysis_image_size": self.active_image_size,
                "candidates": {},
            }
        else:
            model = self._select_model_for_video(capture, actual_start_frame)
        player_detection_mode = (
            "football-specialized-yolo"
            if cached_specialized_player_active
            else self.model_mode
        )
        pitch_model = self._select_pitch_model_for_video(
            capture,
            actual_start_frame,
            source_frames,
        )
        if model is not None:
            self._reset_tracker_state(model)
        dedicated_ball_model = None
        if (
            model is not None
            and
            self.ball_model_path is not None
            and str(self.ball_model_path) != str(self.model_path)
        ):
            dedicated_ball_model = self._load_model(self.ball_model_path)
            self.ball_detection_mode = "dedicated_football_model_every_frame_high_resolution"
        elif not cached_objects_by_frame:
            self.ball_detection_mode = "shared_football_detector"
        writer = cv2.VideoWriter(
            str(raw_output_path),
            cv2.VideoWriter_fourcc(*"MJPG"),
            fps,
            (width, height),
        )
        if not writer.isOpened():
            capture.release()
            raise ValueError("Could not open match analysis video writer")

        last_positions: dict[int, tuple[float, float, int]] = {}
        track_distance: dict[int, float] = {}
        track_speed: dict[int, float] = {}
        track_frames: dict[int, int] = {}
        track_video_samples: dict[int, list[list[int]]] = {}
        track_pitch_samples: dict[int, list[dict[str, float | int]]] = {}
        pitch_to_video_samples: list[list[float | int]] = []
        team_by_track: dict[int, int] = {}
        quality_review_observations: dict[int, list[dict[str, Any]]] = {}
        quality_crop_files: dict[int, list[dict[str, Any]]] = {}
        tracker_runtime: dict[str, Any] = {}
        ball_control: list[int] = []
        class_counts: dict[str, int] = {}
        participant_role_counts: dict[str, int] = {}
        confidence_values: list[float] = []
        player_filter = PlayerValidityFilter()
        pitch_occupancy_filter = PitchOccupancyFilter()
        track_stabilizer = TrackIdStabilizer()
        kit_reference_config = self._load_kit_reference_config(
            team_context,
        )
        team_classifier = TeamColorClassifier(
            reference_palettes_bgr=kit_reference_config["palettes"],
            goalkeeper_reference_palettes_bgr=kit_reference_config[
                "goalkeeper_palettes"
            ],
            team_labels=kit_reference_config["team_labels"],
        )
        participant_role_classifier = ParticipantRoleClassifierV2()
        ball_filter = BallStaticFilter()
        ball_tracker = BallTrackerV4(fps=fps)
        possession_tracker = PossessionTracker(fps=fps)
        radar = PitchRadar(
            pitch_model,
            settings.MATCH_ANALYSIS_RADAR_STRIDE,
            manual_points=calibration_points,
        )
        frames_processed = 0
        detections_count = 0
        raw_detections_count = 0
        available_source_frames = max(0, source_frames - actual_start_frame) if source_frames else 0
        expected_frames = (
            min(available_source_frames, max_frames)
            if available_source_frames > 0 and max_frames > 0
            else max(available_source_frames, max_frames)
        )
        quality_review_interval = max(
            1,
            int(round(fps / QUALITY_REVIEW_SAMPLE_RATE_HZ)),
            int(np.ceil(expected_frames / QUALITY_MAX_REVIEW_OBSERVATIONS_PER_TRACK))
            if expected_frames > 0
            else 1,
        )
        quality_predictions_file = quality_predictions_path.open("w", encoding="utf-8")
        detection_cache_file = detection_cache_path.open("w", encoding="utf-8")
        cache_hit_frames = 0
        inference_frames = 0
        detection_elapsed_seconds = 0.0
        rendering_elapsed_seconds = 0.0

        while max_frames <= 0 or frames_processed < max_frames:
            ok, frame = capture.read()
            if not ok:
                break

            detection_started = perf_counter()
            cached_objects = cached_objects_by_frame.get(frames_processed)
            if cached_objects is not None:
                raw_objects = [self._deserialize_analysis_object(item) for item in cached_objects]
                cache_hit_frames += 1
            else:
                if model is None:
                    raise ValueError(
                        "Detection cache does not cover the requested frame range; run a fresh analysis first"
                    )
                raw_objects = self._detect_and_track(
                    model,
                    frame,
                    mode,
                    include_ball=dedicated_ball_model is None,
                )
                ball_stride = max(1, settings.MATCH_ANALYSIS_BALL_DETECTION_STRIDE)
                if dedicated_ball_model is not None and frames_processed % ball_stride == 0:
                    raw_objects.extend(self._detect_dedicated_balls(dedicated_ball_model, frame))
                inference_frames += 1
            detection_cache_file.write(
                json.dumps(
                    {
                        "frame": frames_processed,
                        "source_frame": actual_start_frame + frames_processed,
                        "objects": [self._serialize_analysis_object(item) for item in raw_objects],
                    },
                    separators=(",", ":"),
                )
                + "\n"
            )
            detection_elapsed_seconds += perf_counter() - detection_started
            if not tracker_runtime and model is not None:
                tracker_runtime = self._tracker_runtime_diagnostics(model)
            elif not tracker_runtime and cached_objects_by_frame:
                tracker_runtime = {
                    "engine": "cached_raw_tracks",
                    "reid": {"requested": True, "active": True, "model": "source_run"},
                    "stable_identity_layer": "identity_isolation_stabilizer_v5_conservative_reid",
                }
            raw_detections_count += len(raw_objects)
            detected_players = [
                item for item in raw_objects if item.class_name == "player"
            ]
            ball_guard_players = PlayerValidityFilter.geometry_candidates(
                detected_players,
                frame_height=frame.shape[0],
            )
            shape_valid_players = player_filter.filter(
                detected_players,
                frame,
                specialized_detector=(
                    player_detection_mode == "football-specialized-yolo"
                ),
            )
            raw_balls = [item for item in raw_objects if item.class_name == "ball"]
            radar.update(
                frame,
                frames_processed,
                players=shape_valid_players,
                static_markers=ball_filter.static_marker_centers(frames_processed),
            )
            radar.record_frame_confidence(frames_processed)
            raw_players = pitch_occupancy_filter.filter(
                frames_processed,
                shape_valid_players,
                frame,
                radar,
            )
            players = track_stabilizer.update(frames_processed, raw_players, frame)
            participant_role_classifier.update(
                players=players,
                track_states=track_stabilizer.tracks,
                team_by_track=team_by_track,
                team_classifier=team_classifier,
                radar=radar,
                frame=frame,
                surface_mask=pitch_occupancy_filter.last_visual_mask,
            )
            # Resolve team once per frame after roles so officials and staff
            # cannot pollute kit anchors or inflate classifier observations.
            team_classifier.update(players, track_stabilizer.tracks, team_by_track)
            analytics_players = [
                player for player in players if player.role_name in ANALYTICS_ROLES
            ]
            reliable_pitch_transform = radar.transform_point if radar.is_reliable() else None
            # Ball geometry sees every plausible physical participant,
            # including identity candidates withheld by appearance and pitch
            # filters. These boxes can reject shoes/body patches but can never
            # create labels, possession owners, analytics rows, or reports.
            filtered_balls = ball_filter.filter(
                frames_processed,
                raw_balls,
                ball_guard_players,
                width,
                pitch_transform=reliable_pitch_transform,
            )
            balls = ball_tracker.update(
                frames_processed,
                filtered_balls,
                ball_guard_players,
                width,
                pitch_transform=reliable_pitch_transform,
                frame=frame,
                pitch_camera=radar,
            )
            self._record_pitch_projection(
                frame_index=frames_processed,
                fps=fps,
                radar=radar,
                samples=pitch_to_video_samples,
            )
            objects = players + balls
            detected_objects = [item for item in objects if not item.is_predicted]
            detections_count += len(detected_objects)
            for item in detected_objects:
                class_counts[item.class_name] = class_counts.get(item.class_name, 0) + 1
                if item.class_name == "player":
                    participant_role_counts[item.role_name] = (
                        participant_role_counts.get(item.role_name, 0) + 1
                    )
                if item.confidence is not None:
                    confidence_values.append(item.confidence)

            self._record_tracking_quality(
                frame=frame,
                frame_index=frames_processed,
                source_frame_index=actual_start_frame + frames_processed,
                fps=fps,
                players=players,
                team_by_track=team_by_track,
                predictions_file=quality_predictions_file,
                review_interval=quality_review_interval,
                review_observations=quality_review_observations,
                crop_files=quality_crop_files,
                crops_dir=quality_crops_dir,
            )
            self._update_movement(
                players=players,
                frame_index=frames_processed,
                fps=fps,
                pitch_transform=radar.transform_point,
                last_positions=last_positions,
                track_distance=track_distance,
                track_speed=track_speed,
                track_frames=track_frames,
                track_video_samples=track_video_samples,
                track_pitch_samples=track_pitch_samples,
            )
            _, current_control = possession_tracker.update(
                frames_processed,
                analytics_players,
                balls,
                team_by_track,
                pitch_transform=radar.transform_point,
            )
            if current_control is not None:
                ball_control.append(current_control)

            rendering_started = perf_counter()
            annotated = frame.copy()
            self._draw_overlay(
                annotated,
                players=players,
                balls=balls,
                team_by_track=team_by_track,
                track_distance=track_distance,
                track_speed=track_speed,
                ball_control=ball_control,
                mode=mode,
            )
            radar.draw(
                annotated,
                frame_index=frames_processed,
                players=analytics_players,
                balls=balls,
                team_by_track=team_by_track,
            )
            if frames_processed == 0:
                cv2.imwrite(str(thumbnail_path), annotated)
            writer.write(annotated)
            rendering_elapsed_seconds += perf_counter() - rendering_started
            frames_processed += 1
            if progress_callback is not None and (
                frames_processed == 1
                or frames_processed % 20 == 0
                or (expected_frames and frames_processed >= expected_frames)
            ):
                elapsed_seconds = max(perf_counter() - start, 0.001)
                rate = frames_processed / elapsed_seconds
                remaining = max(0, expected_frames - frames_processed) if expected_frames else None
                progress_callback(
                    {
                        "stage": "processing",
                        "processed_frames": frames_processed,
                        "total_frames": expected_frames or None,
                        "percent": round(frames_processed * 100.0 / expected_frames, 2)
                        if expected_frames
                        else None,
                        "processing_fps": round(rate, 3),
                        "eta_seconds": round(remaining / max(rate, 1e-6), 1)
                        if remaining is not None
                        else None,
                        "cache_hit_frames": cache_hit_frames,
                    }
                )
            if frames_processed == 1 or frames_processed % 50 == 0:
                elapsed_seconds = max(perf_counter() - start, 0.001)
                print(
                    "Match Analysis + progress "
                    f"{frames_processed}/{expected_frames or '?'} frames "
                    f"({frames_processed / elapsed_seconds:.3f} FPS)",
                    flush=True,
                )

        quality_predictions_file.close()
        detection_cache_file.close()
        capture.release()
        writer.release()
        if frames_processed == 0:
            raise ValueError("No frames were processed by match analysis")
        output_codec = self._transcode_for_browser(raw_output_path, output_path)

        tracks: list[dict[str, Any]] = []
        track_role_counts: dict[str, int] = {}
        for track_id in sorted(track_frames):
            state = track_stabilizer.tracks.get(track_id)
            role_state = participant_role_classifier.get(track_id)
            video_samples = track_video_samples.get(track_id, [])
            pitch_samples = track_pitch_samples.get(track_id, [])
            stable_role = (
                role_state.role_name
                if role_state is not None
                else state.role_name if state is not None else "player"
            )
            track_role_counts[stable_role] = track_role_counts.get(stable_role, 0) + 1
            tracks.append(
                {
                    "track_id": track_id,
                    "team": team_by_track.get(track_id),
                    "frames": track_frames.get(track_id, 0),
                    "distance_m": round(track_distance.get(track_id, 0.0), 2),
                    "last_speed_kmh": round(track_speed.get(track_id, 0.0), 2),
                    "pitch_position_cm": {
                        "x": round(last_positions[track_id][0], 2),
                        "y": round(last_positions[track_id][1], 2),
                    }
                    if track_id in last_positions
                    else None,
                    "ground_position_3d_cm": {
                        "x": round(last_positions[track_id][0], 2),
                        "y": round(last_positions[track_id][1], 2),
                        "z": 0.0,
                    }
                    if track_id in last_positions
                    else None,
                    "first_frame": video_samples[0][0] if video_samples else None,
                    "last_frame": video_samples[-1][0] if video_samples else None,
                    "movement_samples": len(video_samples),
                    "heatmap_samples": len(pitch_samples),
                    "identity_locked": state.identity_locked if state is not None else False,
                    "identity_confidence": round(track_stabilizer._identity_confidence(state), 4)
                    if state is not None
                    else 0.0,
                    "position_2d_px": {
                        "x": round(state.foot[0], 2),
                        "y": round(state.foot[1], 2),
                    }
                    if state is not None
                    else None,
                    "position_3d_proxy": {
                        "x_px": round(state.foot[0], 2),
                        "ground_y_px": round(state.foot[1], 2),
                        "camera_depth_log_height": round(state.depth_proxy, 5),
                    }
                    if state is not None
                    else None,
                    "velocity_2d_px_per_frame": [
                        round(state.foot_velocity[0], 4),
                        round(state.foot_velocity[1], 4),
                    ]
                    if state is not None
                    else None,
                    "depth_velocity": round(state.depth_velocity, 6) if state is not None else None,
                    "occlusion_frames": state.occlusion_hits if state is not None else 0,
                    "appearance_references": len(state.appearance_gallery) if state is not None else 0,
                    "raw_ids_count": len(state.raw_ids_seen) if state is not None else 0,
                    "raw_ids_seen": sorted(state.raw_ids_seen) if state is not None else [],
                    "jersey_family": state.jersey_family if state is not None else None,
                    "jersey_family_confidence": round(track_stabilizer._jersey_family_confidence(state), 4)
                    if state is not None
                    else 0.0,
                    "jersey_family_votes": dict(state.jersey_family_votes) if state is not None else {},
                    "jersey_color_bgr": list(state.jersey_color)
                    if state is not None and state.jersey_color is not None
                    else None,
                    "jersey_color_hsv": list(track_stabilizer._bgr_to_hsv(state.jersey_color))
                    if state is not None and state.jersey_color is not None
                    else None,
                    "role_name": stable_role,
                    "role_confidence": round(
                        role_state.confidence,
                        4,
                    )
                    if role_state is not None
                    else 0.0,
                    "role_locked": role_state.locked if role_state is not None else False,
                    "role_evidence": list(role_state.evidence) if role_state is not None else [],
                    "role_votes": dict(state.role_votes) if state is not None else {},
                    "team_confidence": team_classifier.track_confidence.get(track_id, 0.0),
                }
            )
        team_1 = possession_tracker.team_frames[1]
        team_2 = possession_tracker.team_frames[2]
        total_control = max(team_1 + team_2, 1)
        elapsed_ms = round((perf_counter() - start) * 1000, 2)
        processing_fps = round(frames_processed / max(elapsed_ms / 1000, 0.001), 3)
        tracking_quality = track_stabilizer.quality_report(
            track_frames=track_frames,
            team_by_track=team_by_track,
            review_observations=quality_review_observations,
            crop_files=quality_crop_files,
            tracker_runtime=tracker_runtime or self._tracker_runtime_diagnostics(model),
        )
        role_summary = participant_role_classifier.summary()
        for quality_track in tracking_quality.get("tracks", []):
            role_state = participant_role_classifier.get(int(quality_track["track_id"]))
            if role_state is None:
                continue
            quality_track["role_name"] = role_state.role_name
            quality_track["role_confidence"] = round(role_state.confidence, 4)
            quality_track["role_locked"] = role_state.locked
            quality_track["role_evidence"] = list(role_state.evidence)
            if role_state.confidence < 0.72:
                quality_track.setdefault("issue_codes", []).append("role_needs_review")
        quality_by_track = {
            int(track["track_id"]): track
            for track in tracking_quality.get("tracks", [])
        }
        visual_layers_payload = self._build_visual_layers_payload(
            fps=fps,
            frames_processed=frames_processed,
            width=width,
            height=height,
            track_frames=track_frames,
            track_video_samples=track_video_samples,
            track_pitch_samples=track_pitch_samples,
            pitch_to_video_samples=pitch_to_video_samples,
            team_by_track=team_by_track,
            role_by_track={
                int(track["track_id"]): str(track["role_name"])
                for track in tracks
            },
            team_confidence_by_track=team_classifier.track_confidence,
            quality_by_track=quality_by_track,
            pitch_confidence_samples=radar.frame_confidence,
            ball_pitch_path=ball_tracker.pitch_path,
            ball_image_path=ball_tracker.image_path,
        )
        radar_summary = radar.summary()
        pitch_quality_gate = radar_summary["quality_gate"]
        ball_tracker_summary = ball_tracker.summary()
        ball_filter_summary = ball_filter.summary()
        possession_summary = possession_tracker.summary()
        ball_quality_gate = self._ball_quality_gate(
            frames_processed=frames_processed,
            tracker_summary=ball_tracker_summary,
            filter_summary=ball_filter_summary,
            dedicated_model_active=(
                dedicated_ball_model is not None or cached_specialized_ball_active
            ),
            possession_summary=possession_summary,
        )
        team_identity_summary = team_classifier.summary()
        analytics_real_v1 = self.analytics_engine.build(
            layers=visual_layers_payload,
            possession=possession_summary,
            pitch_gate=pitch_quality_gate,
            ball_gate=ball_quality_gate,
            team_identity=team_identity_summary,
            team_context=team_context,
        )

        return {
            "status": "ok",
            "engine": "match_analysis_plus",
            "model": str(self.model_path),
            "model_mode": self.model_mode,
            "player_detection_mode": player_detection_mode,
            "model_selection": self.model_selection,
            "ball_model": str(self.ball_model_path or self.model_path),
            "ball_detection_mode": self.ball_detection_mode,
            "pitch_model": str(self.pitch_model_path) if self.pitch_model_path is not None else None,
            "pitch_model_selection": self.pitch_model_selection,
            "tracker": settings.MATCH_ANALYSIS_TRACKER,
            "output_codec": output_codec,
            "output_content_type": "video/mp4",
            "frames_processed": frames_processed,
            "max_frames": max_frames,
            "source_total_frames": source_frames,
            "source_start_frame": actual_start_frame,
            "source_end_frame": actual_start_frame + frames_processed - 1,
            "fps": round(float(fps), 3),
            "processing_fps": processing_fps,
            "performance": self._performance_summary(
                frames_processed=frames_processed,
                processing_fps=processing_fps,
                cache_hit_frames=cache_hit_frames,
                inference_frames=inference_frames,
                detection_elapsed_seconds=detection_elapsed_seconds,
                rendering_elapsed_seconds=rendering_elapsed_seconds,
            ),
            "resolution": [width, height],
            "detections_count": detections_count,
            "raw_detections_count": raw_detections_count,
            "class_counts": class_counts,
            "participant_role_counts": participant_role_counts,
            "track_role_counts": track_role_counts,
            "confidence": {
                "avg": round(float(np.mean(confidence_values)), 4) if confidence_values else None,
                "min": round(float(np.min(confidence_values)), 4) if confidence_values else None,
                "max": round(float(np.max(confidence_values)), 4) if confidence_values else None,
            },
            "tracks_count": len(tracks),
            "raw_tracks_count": len(track_stabilizer.raw_ids_seen),
            "player_filter": player_filter.summary(),
            "pitch_occupancy_filter": pitch_occupancy_filter.summary(),
            "team_classifier": team_identity_summary,
            "participant_role_classifier": role_summary,
            "kit_references": kit_reference_config["summary"],
            "id_stabilizer": track_stabilizer.summary(),
            "tracking_quality": tracking_quality,
            "ball_filter": {
                **ball_filter_summary,
                "tracker": ball_tracker_summary,
                "quality_gate": ball_quality_gate,
            },
            "radar": radar_summary,
            "metric_tracking": {
                "coordinate_system": "pitch_centimeters",
                "ground_plane_z_cm": 0.0,
                "trajectory_sample_rate_hz": VISUAL_LAYER_SAMPLE_RATE_HZ,
                "heatmap_ready": any(item["reliable"] for item in radar.frame_confidence),
                "reliable_frames": sum(1 for item in radar.frame_confidence if item["reliable"]),
                "quality_verified": pitch_quality_gate["metric_outputs_verified"],
                "quality_gate_status": pitch_quality_gate["status"],
                "distance_speed_units": "metric_only_when_quality_gate_passes",
            },
            "tracks": tracks[:250],
            "_visual_layers_payload": visual_layers_payload,
            "team_ball_control": {
                "team_1_percent": round(team_1 * 100 / total_control, 2),
                "team_2_percent": round(team_2 * 100 / total_control, 2),
            },
            "possession": possession_summary,
            "analysis_scope": team_context.get("analysis_scope") or "both_teams_full",
            "analytics_real_v1": analytics_real_v1,
            "_analytics_payload": analytics_real_v1,
            "notes": [
                "sports-main source is vendored in apps/match-analysis-worker/sports-main",
                "every run executes player, ball, tracking, team classification, and pitch radar analysis",
                "field-fixture filtering runs before stable player identity assignment",
                "distance and speed use metric ground-plane coordinates only after validated pitch calibration",
                "unverified metric outputs remain visible for review but are not release-grade until the pitch quality gate passes",
            ],
            "elapsed_ms": elapsed_ms,
        }

    def _record_tracking_quality(
        self,
        frame: np.ndarray,
        frame_index: int,
        source_frame_index: int,
        fps: float,
        players: list[AnalysisObject],
        team_by_track: dict[int, int],
        predictions_file: Any,
        review_interval: int,
        review_observations: dict[int, list[dict[str, Any]]],
        crop_files: dict[int, list[dict[str, Any]]],
        crops_dir: Path,
    ) -> None:
        crop_interval = max(1, int(round(fps * 1.5)))
        frame_height, frame_width = frame.shape[:2]
        for player in players:
            if player.is_predicted:
                continue
            bbox = [round(float(value), 2) for value in player.bbox]
            observation = {
                "frame": frame_index,
                "source_frame": source_frame_index,
                "track_id": player.track_id,
                "raw_track_id": player.raw_track_id,
                "bbox": bbox,
                "team": team_by_track.get(player.track_id),
                "role_name": player.role_name,
                "confidence": round(float(player.confidence), 4)
                if player.confidence is not None
                else None,
            }
            predictions_file.write(json.dumps(observation, separators=(",", ":")) + "\n")
            sampled = review_observations.setdefault(player.track_id, [])
            if not sampled or frame_index - int(sampled[-1]["frame"]) >= review_interval:
                sampled.append(observation)

            track_crops = crop_files.setdefault(player.track_id, [])
            if len(track_crops) >= QUALITY_MAX_CROPS_PER_TRACK:
                continue
            if track_crops and frame_index - int(track_crops[-1]["frame"]) < crop_interval:
                continue
            x1, y1, x2, y2 = player.bbox
            width = max(1.0, x2 - x1)
            height = max(1.0, y2 - y1)
            if width < 18 or height < 42:
                continue
            pad_x = width * 0.08
            pad_y = height * 0.05
            left = max(0, int(round(x1 - pad_x)))
            top = max(0, int(round(y1 - pad_y)))
            right = min(frame_width, int(round(x2 + pad_x)))
            bottom = min(frame_height, int(round(y2 + pad_y)))
            crop = frame[top:bottom, left:right]
            if crop.size == 0:
                continue
            file_name = f"track_{player.track_id:04d}_frame_{frame_index:07d}.jpg"
            if cv2.imwrite(
                str(crops_dir / file_name),
                crop,
                [cv2.IMWRITE_JPEG_QUALITY, 90],
            ):
                track_crops.append(
                    {
                        "frame": frame_index,
                        "file_name": file_name,
                        "confidence": observation["confidence"],
                    }
                )

    def _tracker_runtime_diagnostics(self, model: Any) -> dict[str, Any]:
        predictor = getattr(model, "predictor", None)
        trackers = getattr(predictor, "trackers", None) or []
        tracker = trackers[0] if trackers else None
        args = getattr(tracker, "args", None)
        requested_reid = bool(getattr(args, "with_reid", False))
        reid_model = getattr(args, "model", None)
        encoder = getattr(tracker, "encoder", None)
        native_features = requested_reid and str(reid_model).lower() == "auto"
        try:
            import ultralytics

            ultralytics_version = getattr(ultralytics, "__version__", None)
        except ImportError:
            ultralytics_version = None
        return {
            "engine": type(tracker).__name__ if tracker is not None else "unavailable",
            "ultralytics_version": ultralytics_version,
            "config": self._resolve_tracker_config(),
            "global_motion_compensation": getattr(args, "gmc_method", None),
            "track_buffer": getattr(args, "track_buffer", None),
            "reid": {
                "requested": requested_reid,
                "active": bool(requested_reid and (encoder is not None or native_features)),
                "model": str(reid_model) if reid_model is not None else None,
                "encoder": type(encoder).__name__ if encoder is not None else None,
                "native_detector_features": native_features,
            },
            "stable_identity_layer": "identity_isolation_stabilizer_v5_conservative_reid",
        }

    def _reset_tracker_state(self, model: Any) -> None:
        predictor = getattr(model, "predictor", None)
        trackers = getattr(predictor, "trackers", None) or []
        for tracker in trackers:
            reset = getattr(tracker, "reset", None)
            if callable(reset):
                reset()
        if predictor is not None and hasattr(predictor, "vid_path"):
            predictor.vid_path = [None for _ in getattr(predictor, "vid_path", [None])]

    def _load_detection_cache(
        self,
        path: Path | None,
        expected_start_frame: int,
    ) -> dict[int, list[dict[str, Any]]]:
        if path is None or not path.exists():
            return {}
        cached: dict[int, list[dict[str, Any]]] = {}
        with path.open("r", encoding="utf-8") as file:
            for line in file:
                if not line.strip():
                    continue
                payload = json.loads(line)
                source_frame = int(payload.get("source_frame", payload.get("frame", 0)))
                local_frame = source_frame - expected_start_frame
                if local_frame < 0:
                    continue
                cached[local_frame] = list(payload.get("objects") or [])
        return cached

    def _serialize_analysis_object(self, item: AnalysisObject) -> dict[str, Any]:
        return {
            "track_id": int(item.track_id),
            "class_name": item.class_name,
            "bbox": [round(float(value), 3) for value in item.bbox],
            "confidence": round(float(item.confidence), 5) if item.confidence is not None else None,
            "raw_track_id": item.raw_track_id,
            "is_predicted": bool(item.is_predicted),
            "role_name": item.role_name,
        }

    def _deserialize_analysis_object(self, payload: dict[str, Any]) -> AnalysisObject:
        return AnalysisObject(
            track_id=int(payload["track_id"]),
            class_name=str(payload["class_name"]),
            bbox=[float(value) for value in payload["bbox"]],
            confidence=float(payload["confidence"]) if payload.get("confidence") is not None else None,
            raw_track_id=int(payload["raw_track_id"]) if payload.get("raw_track_id") is not None else None,
            is_predicted=bool(payload.get("is_predicted", False)),
            role_name=str(payload.get("role_name") or "player"),
        )

    def _performance_summary(
        self,
        frames_processed: int,
        processing_fps: float,
        cache_hit_frames: int,
        inference_frames: int,
        detection_elapsed_seconds: float,
        rendering_elapsed_seconds: float,
    ) -> dict[str, Any]:
        cuda_available = False
        cuda_devices = 0
        try:
            import torch

            cuda_available = bool(torch.cuda.is_available())
            cuda_devices = int(torch.cuda.device_count()) if cuda_available else 0
        except Exception:
            pass
        configured_device = str(settings.YOLO_DEVICE)
        gpu_requested = configured_device.lower() not in {"cpu", "none", ""}
        return {
            "engine": "separated_detection_rendering_pipeline_v1",
            "processing_fps": processing_fps,
            "frames_processed": frames_processed,
            "configured_device": configured_device,
            "cuda_available": cuda_available,
            "cuda_devices": cuda_devices,
            "gpu_active": gpu_requested and cuda_available,
            "cache_hit_frames": cache_hit_frames,
            "yolo_inference_frames": inference_frames,
            "yolo_skipped_frames": cache_hit_frames,
            "detection_seconds": round(detection_elapsed_seconds, 3),
            "rendering_seconds": round(rendering_elapsed_seconds, 3),
            "detection_cache_reusable": True,
            "stateful_tracking_batch_size": 1,
            "auxiliary_inference_batch_size": int(settings.YOLO_BATCH_SIZE),
            "batch_policy": "stateful tracking remains ordered; auxiliary and future cache precompute may batch safely",
        }

    def _detect_and_track(
        self,
        model: Any,
        frame: np.ndarray,
        mode: str,
        include_ball: bool = True,
    ) -> list[AnalysisObject]:
        classes = self._target_class_ids(model, include_ball=include_ball)
        results = model.track(
            frame,
            persist=True,
            conf=max(0.05, settings.MATCH_ANALYSIS_CONFIDENCE),
            imgsz=self.active_image_size,
            device=settings.YOLO_DEVICE,
            max_det=settings.YOLO_MAX_DETECTIONS,
            verbose=False,
            tracker=self._resolve_tracker_config(),
            classes=classes,
        )
        if not results:
            return []

        result = results[0]
        names = result.names or {}
        boxes = result.boxes
        if boxes is None or boxes.xyxy is None:
            return []

        xyxy = boxes.xyxy.cpu().numpy()
        cls = boxes.cls.cpu().numpy() if boxes.cls is not None else np.zeros(len(xyxy))
        conf = boxes.conf.cpu().numpy() if boxes.conf is not None else [None] * len(xyxy)
        ids = boxes.id.cpu().numpy() if boxes.id is not None else np.arange(1, len(xyxy) + 1)

        objects: list[AnalysisObject] = []
        for index, bbox in enumerate(xyxy):
            raw_name = str(names.get(int(cls[index]), int(cls[index]))).lower()
            class_name = self._map_class_name(raw_name)
            if class_name is None:
                continue
            objects.append(
                AnalysisObject(
                    track_id=int(ids[index]),
                    class_name=class_name,
                    bbox=[float(value) for value in bbox.tolist()],
                    confidence=float(conf[index]) if conf[index] is not None else None,
                    raw_track_id=int(ids[index]),
                    role_name=self._map_role_name(raw_name),
                )
            )
        return objects

    def _detect_dedicated_balls(self, model: Any, frame: np.ndarray) -> list[AnalysisObject]:
        classes = self._target_class_ids(
            model,
            include_players=False,
            include_ball=True,
        )
        if not classes:
            return []
        results = model.predict(
            frame,
            conf=max(0.03, settings.MATCH_ANALYSIS_BALL_CONFIDENCE),
            imgsz=max(640, settings.MATCH_ANALYSIS_BALL_IMAGE_SIZE),
            device=settings.YOLO_DEVICE,
            max_det=20,
            verbose=False,
            classes=classes,
        )
        if not results or results[0].boxes is None:
            return []
        result = results[0]
        names = result.names or {}
        boxes = result.boxes
        xyxy = boxes.xyxy.cpu().numpy() if boxes.xyxy is not None else np.empty((0, 4))
        class_ids = boxes.cls.cpu().numpy() if boxes.cls is not None else np.empty(0)
        confidences = boxes.conf.cpu().numpy() if boxes.conf is not None else np.zeros(len(xyxy))
        balls: list[AnalysisObject] = []
        for index, bbox in enumerate(xyxy):
            raw_name = str(names.get(int(class_ids[index]), int(class_ids[index]))).lower()
            if raw_name not in BALL_ALIASES:
                continue
            balls.append(
                AnalysisObject(
                    track_id=index + 1,
                    class_name="ball",
                    bbox=[float(value) for value in bbox.tolist()],
                    confidence=float(confidences[index]),
                    raw_track_id=None,
                    role_name="ball",
                )
            )
        return balls

    def _update_movement(
        self,
        players: list[AnalysisObject],
        frame_index: int,
        fps: float,
        pitch_transform: Any,
        last_positions: dict[int, tuple[float, float, int]],
        track_distance: dict[int, float],
        track_speed: dict[int, float],
        track_frames: dict[int, int],
        track_video_samples: dict[int, list[list[int]]],
        track_pitch_samples: dict[int, list[dict[str, float | int]]],
    ) -> None:
        sample_interval = max(1, int(round(fps / VISUAL_LAYER_SAMPLE_RATE_HZ)))
        for player in players:
            if player.is_predicted:
                continue
            foot = ((player.bbox[0] + player.bbox[2]) / 2, player.bbox[3])
            track_frames[player.track_id] = track_frames.get(player.track_id, 0) + 1
            video_samples = track_video_samples.setdefault(player.track_id, [])
            if (
                not video_samples
                or frame_index - int(video_samples[-1][0]) >= sample_interval
            ):
                video_samples.append(
                    [frame_index, int(round(foot[0])), int(round(foot[1]))]
                )

            if player.role_name not in ANALYTICS_ROLES:
                continue

            pitch_xy = pitch_transform(foot)
            if pitch_xy is None:
                continue
            previous = last_positions.get(player.track_id)
            if previous is not None:
                frame_delta = max(frame_index - previous[2], 1)
                distance_m = float(np.hypot(
                    pitch_xy[0] - previous[0],
                    pitch_xy[1] - previous[1],
                ) / 100.0)
                elapsed = frame_delta / max(fps, 1e-6)
                instant_speed = distance_m / elapsed * 3.6
                if distance_m >= 0.025 and instant_speed <= 42.0:
                    track_distance[player.track_id] = (
                        track_distance.get(player.track_id, 0.0) + distance_m
                    )
                    prior_speed = track_speed.get(player.track_id, instant_speed)
                    track_speed[player.track_id] = (
                        prior_speed * 0.72 + instant_speed * 0.28
                    )
            last_positions[player.track_id] = (
                pitch_xy[0],
                pitch_xy[1],
                frame_index,
            )
            samples = track_pitch_samples.setdefault(player.track_id, [])
            if not samples or frame_index - int(samples[-1]["frame"]) >= sample_interval:
                samples.append(
                    {
                        "frame": frame_index,
                        "x": round(pitch_xy[0], 2),
                        "y": round(pitch_xy[1], 2),
                        "z": 0.0,
                    }
                )

    def _record_pitch_projection(
        self,
        frame_index: int,
        fps: float,
        radar: PitchRadar,
        samples: list[list[float | int]],
    ) -> None:
        sample_interval = max(1, int(round(fps / VISUAL_LAYER_SAMPLE_RATE_HZ)))
        if samples and frame_index - int(samples[-1][0]) < sample_interval:
            return
        matrix = radar.pitch_to_video_matrix()
        if matrix is None:
            return
        samples.append(
            [frame_index, *[round(float(value), 9) for value in matrix.reshape(-1)]]
        )

    def _build_visual_layers_payload(
        self,
        fps: float,
        frames_processed: int,
        width: int,
        height: int,
        track_frames: dict[int, int],
        track_video_samples: dict[int, list[list[int]]],
        track_pitch_samples: dict[int, list[dict[str, float | int]]],
        pitch_to_video_samples: list[list[float | int]],
        team_by_track: dict[int, int],
        role_by_track: dict[int, str] | None = None,
        team_confidence_by_track: dict[int, float] | None = None,
        quality_by_track: dict[int, dict[str, Any]] | None = None,
        pitch_confidence_samples: list[dict[str, Any]] | None = None,
        ball_pitch_path: list[dict[str, float | int | bool]] | None = None,
        ball_image_path: list[dict[str, float | int | bool]] | None = None,
    ) -> dict[str, Any]:
        quality_by_track = quality_by_track or {}
        role_by_track = role_by_track or {}
        team_confidence_by_track = team_confidence_by_track or {}
        visual_tracks: list[dict[str, Any]] = []
        for track_id in sorted(track_frames):
            video_path = track_video_samples.get(track_id, [])
            if not video_path:
                continue
            pitch_path = [
                [
                    int(sample["frame"]),
                    int(round(float(sample["x"]))),
                    int(round(float(sample["y"]))),
                ]
                for sample in track_pitch_samples.get(track_id, [])
            ]
            visual_tracks.append(
                {
                    "track_id": track_id,
                    "team": team_by_track.get(track_id),
                    "role_name": role_by_track.get(track_id, "player"),
                    "team_confidence": team_confidence_by_track.get(track_id),
                    "color": self._track_visual_color(track_id),
                    "frames": track_frames.get(track_id, 0),
                    "first_frame": int(video_path[0][0]),
                    "last_frame": int(video_path[-1][0]),
                    "video_path": video_path,
                    "pitch_path": pitch_path,
                    "identity_confidence": quality_by_track.get(track_id, {}).get(
                        "identity_confidence"
                    ),
                    "switch_risk": quality_by_track.get(track_id, {}).get("switch_risk"),
                }
            )

        return {
            "schema_version": VISUAL_LAYER_SCHEMA_VERSION,
            "coordinate_systems": {
                "video": "source_pixels",
                "pitch": "pitch_centimeters",
                "ground_plane_z_cm": 0.0,
            },
            "fps": round(float(fps), 4),
            "frames_processed": frames_processed,
            "duration_seconds": round(frames_processed / max(float(fps), 1e-6), 3),
            "resolution": [width, height],
            "movement_sample_rate_hz": VISUAL_LAYER_SAMPLE_RATE_HZ,
            "heatmap_sample_rate_hz": VISUAL_LAYER_SAMPLE_RATE_HZ,
            "pitch": {
                "length_cm": int(PITCH_LENGTH_CM),
                "width_cm": int(PITCH_WIDTH_CM),
            },
            "pitch_to_video": pitch_to_video_samples,
            "pitch_calibration": pitch_confidence_samples or [],
            "ball": {
                "track_id": 1,
                "pitch_path": ball_pitch_path or [],
                "image_path": ball_image_path or [],
            },
            "tracks": visual_tracks,
        }

    def _track_visual_color(self, track_id: int) -> str:
        if 1 <= track_id <= len(TRACK_VISUAL_PALETTE):
            return TRACK_VISUAL_PALETTE[track_id - 1]
        hue = (track_id * 0.61803398875) % 1.0
        red, green, blue = colorsys.hsv_to_rgb(hue, 0.78, 0.94)
        return f"#{int(red * 255):02x}{int(green * 255):02x}{int(blue * 255):02x}"

    def _ball_control(
        self,
        players: list[AnalysisObject],
        balls: list[AnalysisObject],
        team_by_track: dict[int, int],
        ball_control: list[int],
        pitch_transform: Any,
    ) -> int | None:
        players = [player for player in players if not player.is_predicted]
        if not players or not balls:
            return ball_control[-1] if ball_control else None
        ball = balls[0]
        ball_center = ((ball.bbox[0] + ball.bbox[2]) / 2, (ball.bbox[1] + ball.bbox[3]) / 2)
        ball_pitch = pitch_transform(ball_center)
        nearest: tuple[float, AnalysisObject] | None = None
        for player in players:
            foot = ((player.bbox[0] + player.bbox[2]) / 2, player.bbox[3])
            player_pitch = pitch_transform(foot)
            if ball_pitch is not None and player_pitch is not None:
                distance = float(np.hypot(
                    player_pitch[0] - ball_pitch[0],
                    player_pitch[1] - ball_pitch[1],
                ))
            else:
                distance = float(
                    np.hypot(foot[0] - ball_center[0], foot[1] - ball_center[1])
                    * 3.0
                )
            if nearest is None or distance < nearest[0]:
                nearest = (distance, player)
        if nearest is None or nearest[0] > 250.0:
            return ball_control[-1] if ball_control else None
        return team_by_track.get(nearest[1].track_id, 1)

    def _ball_quality_gate(
        self,
        frames_processed: int,
        tracker_summary: dict[str, Any],
        filter_summary: dict[str, Any],
        dedicated_model_active: bool,
        possession_summary: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        minimum_observed_frames = max(2, int(np.ceil(frames_processed * 0.15)))
        observed_frames = int(tracker_summary.get("observed_frames", 0))
        interpolated_frames = int(tracker_summary.get("interpolated_frames", 0))
        tracked_frames = observed_frames + interpolated_frames
        minimum_tracked_frames = max(2, int(np.ceil(frames_processed * 0.25)))
        pitch_samples = int(tracker_summary.get("pitch_samples", 0))
        maximum_interpolation = int(
            tracker_summary.get("maximum_interpolation_streak", 0)
        )
        interpolation_limit = int(
            tracker_summary.get(
                "max_airborne_interpolation_frames"
                if tracker_summary.get("airborne_entries", 0)
                else "max_interpolation_frames",
                10,
            )
        )
        conditions = [
            {
                "code": "specialized_ball_model",
                "passed": dedicated_model_active,
                "value": dedicated_model_active,
                "required": True,
            },
            {
                "code": "observed_ball_coverage",
                "passed": observed_frames >= minimum_observed_frames,
                "value": observed_frames,
                "required": minimum_observed_frames,
            },
            {
                "code": "bounded_interpolation",
                "passed": maximum_interpolation <= interpolation_limit,
                "value": maximum_interpolation,
                "required": interpolation_limit,
            },
            {
                "code": "tracked_ball_coverage",
                "passed": tracked_frames >= minimum_tracked_frames,
                "value": tracked_frames,
                "required": minimum_tracked_frames,
            },
            {
                "code": "metric_ball_path",
                "passed": pitch_samples > 0,
                "value": pitch_samples,
                "required": "> 0",
            },
            {
                "code": "static_false_positive_filter",
                "passed": "penalty_spot_rejections" in filter_summary,
                "value": {
                    "penalty_spot_rejections": filter_summary.get(
                        "penalty_spot_rejections", 0
                    ),
                    "static_candidates": filter_summary.get(
                        "filtered_static_candidates", 0
                    ),
                },
                "required": "active",
            },
            {
                "code": "player_body_false_positive_guard",
                "passed": "player_body_rejections" in tracker_summary,
                "value": tracker_summary.get("player_body_rejections", 0),
                "required": "active",
            },
            {
                "code": "metric_jump_guard",
                "passed": "metric_jump_rejections" in tracker_summary,
                "value": tracker_summary.get("metric_jump_rejections", 0),
                "required": "active",
            },
            {
                "code": "optical_flow_guard",
                "passed": "optical_flow_rejections" in tracker_summary,
                "value": {
                    "attempts": tracker_summary.get("optical_flow_attempts", 0),
                    "successes": tracker_summary.get("optical_flow_successes", 0),
                    "rejections": tracker_summary.get("optical_flow_rejections", 0),
                },
                "required": "active",
            },
        ]
        failed = [condition["code"] for condition in conditions if not condition["passed"]]
        possession_coverage = float(
            (possession_summary or {}).get("assigned_coverage", 0.0)
        )
        possession_ready = not failed and possession_coverage >= 0.15
        return {
            "status": "passed" if not failed else "needs_review",
            "conditions": conditions,
            "failed_conditions": failed,
            "possession_ready": possession_ready,
            "pass_detection_ready": possession_ready,
            "possession_assigned_coverage": round(possession_coverage, 4),
            "possession_minimum_coverage": 0.15,
        }

    def _draw_overlay(
        self,
        frame: np.ndarray,
        players: list[AnalysisObject],
        balls: list[AnalysisObject],
        team_by_track: dict[int, int],
        track_distance: dict[int, float],
        track_speed: dict[int, float],
        ball_control: list[int],
        mode: str,
    ) -> None:
        for player in players:
            team = team_by_track.get(player.track_id, 1)
            self._draw_player(frame, player, team, track_distance, track_speed)
        for ball in balls:
            self._draw_triangle(
                frame,
                ball.bbox,
                (0, 255, 255),
                filled=not ball.is_predicted,
            )
        self._draw_header(frame, mode)
        self._draw_ball_control(frame, ball_control)

    def _draw_player(
        self,
        frame: np.ndarray,
        player: AnalysisObject,
        team: int,
        track_distance: dict[int, float],
        track_speed: dict[int, float],
    ) -> None:
        x1, y1, x2, y2 = [int(round(value)) for value in player.bbox]
        center_x = int((x1 + x2) / 2)
        color = TEAM_DISPLAY_COLORS.get(team, TEAM_DISPLAY_COLORS[1])
        text_color = (15, 15, 15) if team in (0, 1) else (255, 255, 255)
        cv2.ellipse(frame, (center_x, y2), (max(10, (x2 - x1) // 2), 8), 0, -45, 235, color, 2)
        scale = self._font_scale(frame, 0.5)
        small = self._font_scale(frame, 0.38)
        thickness = self._thickness(frame)
        role_prefix = {
            "goalkeeper": "GK ",
            "referee": "REF ",
            "assistant_referee": "AR ",
            "staff_outside_pitch": "STAFF ",
        }.get(player.role_name, "")
        label = f"{role_prefix}{player.track_id}"
        text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)[0]
        label_width = max(28, text_size[0] + 10)
        label_height = max(18, text_size[1] + 8)
        label_left = center_x - label_width // 2
        label_top = y2 + 4
        cv2.rectangle(
            frame,
            (label_left, label_top),
            (label_left + label_width, label_top + label_height),
            color,
            cv2.FILLED,
        )
        cv2.rectangle(
            frame,
            (label_left, label_top),
            (label_left + label_width, label_top + label_height),
            (25, 25, 25),
            1,
        )
        cv2.putText(
            frame,
            label,
            (center_x - text_size[0] // 2, label_top + label_height - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            scale,
            text_color,
            thickness,
            cv2.LINE_AA,
        )
        distance = track_distance.get(player.track_id)
        speed = track_speed.get(player.track_id)
        if distance is not None:
            cv2.putText(frame, f"{distance:.1f}m", (center_x - 17, label_top + label_height + 15), cv2.FONT_HERSHEY_SIMPLEX, small, (0, 0, 0), thickness)
        if speed is not None:
            cv2.putText(frame, f"{speed:.1f}km/h", (center_x - 24, label_top + label_height + 29), cv2.FONT_HERSHEY_SIMPLEX, small, (0, 0, 0), thickness)

    def _draw_triangle(
        self,
        frame: np.ndarray,
        bbox: list[float],
        color: tuple[int, int, int],
        filled: bool = True,
    ) -> None:
        x = int((bbox[0] + bbox[2]) / 2)
        y = int(bbox[1])
        points = np.array([[x, y], [x - 9, y - 18], [x + 9, y - 18]])
        cv2.drawContours(frame, [points], 0, color, cv2.FILLED if filled else 2)
        cv2.drawContours(frame, [points], 0, (0, 0, 0), 1)

    def _draw_header(self, frame: np.ndarray, mode: str) -> None:
        scale = self._font_scale(frame, 0.48)
        thickness = self._thickness(frame)
        title = "Match Analysis +  FULL"
        cv2.putText(frame, title, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), thickness + 2)
        cv2.putText(frame, title, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thickness)

    def _draw_ball_control(self, frame: np.ndarray, ball_control: list[int]) -> None:
        height, width = frame.shape[:2]
        scale = self._font_scale(frame, 0.44)
        thickness = self._thickness(frame)
        team_1 = sum(1 for item in ball_control if item == 1)
        team_2 = sum(1 for item in ball_control if item == 2)
        total = max(team_1 + team_2, 1)
        text = f"T1 {team_1 * 100 / total:.1f}%  T2 {team_2 * 100 / total:.1f}%"
        cv2.putText(frame, text, (max(10, width - 250), height - 18), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), thickness + 2)
        cv2.putText(frame, text, (max(10, width - 250), height - 18), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thickness)

    def _ui_scale(self, frame: np.ndarray) -> float:
        height, width = frame.shape[:2]
        return max(0.45, min(width / 1920, height / 1080))

    def _font_scale(self, frame: np.ndarray, base: float) -> float:
        return max(0.28, base * self._ui_scale(frame))

    def _thickness(self, frame: np.ndarray) -> int:
        return max(1, int(round(2 * self._ui_scale(frame))))

    def _map_class_name(self, raw_name: str) -> str | None:
        if raw_name in PERSON_ALIASES:
            return "player"
        if raw_name in BALL_ALIASES:
            return "ball"
        return None

    def _map_role_name(self, raw_name: str) -> str:
        if raw_name in GOALKEEPER_ALIASES:
            return "goalkeeper"
        if raw_name in ASSISTANT_REFEREE_ALIASES:
            return "assistant_referee"
        if raw_name in STAFF_ALIASES:
            return "staff_outside_pitch"
        if raw_name in REFEREE_ALIASES:
            return "referee"
        if raw_name in BALL_ALIASES:
            return "ball"
        return "player"

    def _target_class_ids(
        self,
        model: Any,
        include_players: bool = True,
        include_ball: bool = True,
    ) -> list[int]:
        names = getattr(model, "names", {}) or {}
        if isinstance(names, list):
            names = dict(enumerate(names))
        class_ids = [
            int(class_id)
            for class_id, class_name in names.items()
            if (
                include_players
                and str(class_name).lower() in PERSON_ALIASES
            )
            or (
                include_ball
                and str(class_name).lower() in BALL_ALIASES
            )
        ]
        return sorted(class_ids)

    def _select_model_for_video(self, capture: cv2.VideoCapture, start_frame: int) -> Any:
        candidates = self._unique_model_paths(
            [
                ("general", self.general_model_path),
                ("football_objects_v2", self.specialized_model_path),
                (
                    "football_specialized_v1_fallback",
                    self.legacy_specialized_model_path,
                ),
            ]
        )

        original_position = int(capture.get(cv2.CAP_PROP_POS_FRAMES) or start_frame)
        ok, preview = capture.read()
        capture.set(cv2.CAP_PROP_POS_FRAMES, original_position)
        if not ok or preview is None:
            label, path = next(
                (
                    item
                    for item in candidates
                    if item[0] == "football_objects_v2"
                ),
                candidates[-1],
            )
            model = self._load_model(path)
            self._activate_model(label, path, model, {}, "preview_unavailable")
            return model

        scores: dict[str, dict[str, float | int | str]] = {}
        available: list[tuple[str, Path]] = []
        from ultralytics import YOLO

        for label, path in candidates:
            candidate_model = None
            try:
                candidate_model = YOLO(str(path))
                scores[label] = self._preview_model_score(
                    candidate_model,
                    preview,
                )
                available.append((label, path))
            except (ImportError, OSError, RuntimeError, ValueError) as exc:
                scores[label] = {
                    "valid_players": 0,
                    "raw_players": 0,
                    "confidence_sum": 0.0,
                    "error": type(exc).__name__,
                }
            finally:
                if candidate_model is not None:
                    del candidate_model
                gc.collect()
        if not available:
            raise RuntimeError("No YOLO model could be loaded for match analysis")

        selected_label, selected_path = max(
            available,
            key=lambda item: (
                int(scores[item[0]].get("valid_players", 0)),
                float(scores[item[0]].get("confidence_sum", 0.0)),
            ),
        )
        selected_model = self._load_model(selected_path)
        reason = "highest_on_pitch_player_coverage"
        self._activate_model(
            selected_label,
            selected_path,
            selected_model,
            scores,
            reason,
        )
        capture.set(cv2.CAP_PROP_POS_FRAMES, original_position)
        return selected_model

    def _preview_model_score(self, model: Any, frame: np.ndarray) -> dict[str, float | int]:
        results = model.predict(
            frame,
            conf=max(0.12, settings.MATCH_ANALYSIS_CONFIDENCE),
            imgsz=640,
            device=settings.YOLO_DEVICE,
            max_det=settings.YOLO_MAX_DETECTIONS,
            verbose=False,
            classes=self._target_class_ids(model),
        )
        if not results or results[0].boxes is None:
            return {"valid_players": 0, "raw_players": 0, "confidence_sum": 0.0}

        result = results[0]
        names = result.names or {}
        boxes = result.boxes
        xyxy = boxes.xyxy.cpu().numpy() if boxes.xyxy is not None else np.empty((0, 4))
        classes = boxes.cls.cpu().numpy() if boxes.cls is not None else np.empty(0)
        confidences = boxes.conf.cpu().numpy() if boxes.conf is not None else np.zeros(len(xyxy))
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        green = cv2.inRange(hsv, (24, 22, 20), (108, 255, 255))
        frame_height, frame_width = green.shape[:2]
        raw_players = 0
        valid_players = 0
        confidence_sum = 0.0
        for index, bbox in enumerate(xyxy):
            raw_name = str(names.get(int(classes[index]), int(classes[index]))).lower()
            if raw_name not in PERSON_ALIASES:
                continue
            raw_players += 1
            x1, y1, x2, y2 = [float(value) for value in bbox]
            box_width = max(1.0, x2 - x1)
            box_height = max(1.0, y2 - y1)
            if box_height < frame_height * 0.018 or box_height > frame_height * 0.74:
                continue
            if box_width / box_height > 1.35:
                continue
            foot_x = int(round((x1 + x2) / 2))
            foot_y = int(round(y2))
            radius = max(4, int(round(box_width * 0.28)))
            left, right = max(0, foot_x - radius), min(frame_width, foot_x + radius + 1)
            top, bottom = max(0, foot_y - radius), min(frame_height, foot_y + radius + 1)
            if left >= right or top >= bottom:
                continue
            green_ratio = float(np.mean(green[top:bottom, left:right] > 0))
            if green_ratio < 0.12:
                continue
            valid_players += 1
            confidence_sum += float(confidences[index])
        return {
            "valid_players": valid_players,
            "raw_players": raw_players,
            "confidence_sum": round(confidence_sum, 4),
        }

    def _activate_model(
        self,
        label: str,
        path: Path,
        model: Any,
        scores: dict[str, Any],
        reason: str,
    ) -> None:
        self.model_path = path
        names = getattr(model, "names", {}) or {}
        if isinstance(names, list):
            names = dict(enumerate(names))
        class_names = {
            str(class_name).lower()
            for class_name in names.values()
        }
        is_specialized = {"ball", "goalkeeper", "player"}.issubset(class_names)
        self.model_mode = (
            "football-specialized-yolo"
            if is_specialized
            else "balanced-yolo-with-football-guards"
        )
        self.active_image_size = (
            max(640, settings.MATCH_ANALYSIS_IMAGE_SIZE)
            if is_specialized
            else 640
        )
        self.model_selection = {
            "strategy": "automatic_preview_v1",
            "selected": label,
            "reason": reason,
            "preview_image_size": 640,
            "analysis_image_size": self.active_image_size,
            "candidates": scores,
        }

    def _load_model(self, path: Path) -> Any:
        cache_key = str(path)
        if cache_key in self.models:
            return self.models[cache_key]
        from ultralytics import YOLO

        model = YOLO(str(path))
        self.models[cache_key] = model
        return model

    def _select_pitch_model_for_video(
        self,
        capture: cv2.VideoCapture,
        start_frame: int,
        source_frames: int,
    ) -> Any | None:
        if not self.pitch_model_candidates:
            self.pitch_model_selection = {
                "strategy": "unavailable",
                "selected": None,
                "reason": "no_pitch_model_found",
                "candidates": {},
            }
            return None

        from ultralytics import YOLO

        preview_frames = self._pitch_preview_frames(
            capture,
            start_frame,
            source_frames,
        )
        if not preview_frames:
            selected_label, selected_path = self.pitch_model_candidates[0]
            self.pitch_model = YOLO(str(selected_path), task="pose")
            self.pitch_model_path = selected_path
            self.pitch_model_selection = {
                "strategy": "fallback_order",
                "selected": selected_label,
                "reason": "preview_unavailable",
                "candidates": {},
            }
            return self.pitch_model

        scores: dict[str, Any] = {}
        selected: tuple[str, Path] | None = None
        selected_rank: tuple[float, ...] | None = None
        for label, path in self.pitch_model_candidates:
            candidate_model = None
            try:
                candidate_model = YOLO(str(path), task="pose")
                score = self._pitch_model_preview_score(
                    candidate_model,
                    preview_frames,
                )
                scores[label] = {
                    "path": str(path),
                    **score,
                }
                rank = self._pitch_preview_rank(score)
                if selected_rank is None or rank > selected_rank:
                    selected = (label, path)
                    selected_rank = rank
            except (ImportError, OSError, RuntimeError, ValueError) as exc:
                scores[label] = {
                    "path": str(path),
                    "error": f"{type(exc).__name__}: {exc}",
                }
            finally:
                if candidate_model is not None:
                    del candidate_model
                gc.collect()

        capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        if selected is None:
            self.pitch_model_selection = {
                "strategy": "multi_frame_geometry_gate_v1",
                "selected": None,
                "reason": "all_candidates_failed",
                "candidates": scores,
            }
            return None

        selected_label, selected_path = selected
        self.pitch_model = YOLO(str(selected_path), task="pose")
        self.pitch_model_path = selected_path
        self.pitch_model_selection = {
            "strategy": "multi_frame_geometry_gate_v1",
            "selected": selected_label,
            "reason": "best_wide_view_geometry_and_reprojection",
            "preview_frames": [index for index, _ in preview_frames],
            "candidates": scores,
        }
        return self.pitch_model

    def _pitch_preview_frames(
        self,
        capture: cv2.VideoCapture,
        start_frame: int,
        source_frames: int,
    ) -> list[tuple[int, np.ndarray]]:
        original_position = int(capture.get(cv2.CAP_PROP_POS_FRAMES) or start_frame)
        available = (
            max(1, source_frames - start_frame)
            if source_frames > 0
            else 241
        )
        span = max(0, min(240, available - 1))
        offsets = sorted({0, span // 2, span})
        previews: list[tuple[int, np.ndarray]] = []
        for offset in offsets:
            frame_index = start_frame + offset
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, frame = capture.read()
            if ok and frame is not None:
                previews.append((frame_index, frame))
        capture.set(cv2.CAP_PROP_POS_FRAMES, original_position)
        return previews

    def _pitch_model_preview_score(
        self,
        model: Any,
        frames: list[tuple[int, np.ndarray]],
    ) -> dict[str, Any]:
        geometry = PitchRadar(None)
        targets = geometry._pitch_vertices()
        visible_total = 0
        valid_homographies = 0
        wide_views = 0
        inlier_ratios: list[float] = []
        reprojection_errors: list[float] = []
        frame_scores: list[dict[str, Any]] = []

        for frame_index, frame in frames:
            results = model.predict(
                frame,
                imgsz=max(960, settings.YOLO_IMAGE_SIZE),
                device=settings.YOLO_DEVICE,
                verbose=False,
            )
            frame_score: dict[str, Any] = {
                "frame": frame_index,
                "visible_keypoints": 0,
                "valid_homography": False,
            }
            if not results or results[0].keypoints is None:
                frame_scores.append(frame_score)
                continue
            keypoints = results[0].keypoints
            source = keypoints.xy.cpu().numpy()
            if source.ndim == 3:
                if source.shape[0] == 0:
                    frame_scores.append(frame_score)
                    continue
                source = source[0]
            if source.ndim != 2 or source.shape[0] == 0 or source.shape[1] < 2:
                frame_scores.append(frame_score)
                continue
            confidence = keypoints.conf
            confidence_values = (
                np.ones(len(source), dtype=np.float32)
                if confidence is None
                else confidence.cpu().numpy()
            )
            if confidence_values.ndim == 2:
                confidence_values = confidence_values[0]
            count = min(len(source), len(targets), len(confidence_values))
            source = source[:count].astype(np.float32)
            target = targets[:count].astype(np.float32)
            confidence_values = confidence_values[:count]
            visible = (
                (confidence_values >= 0.34)
                & (source[:, 0] > 1)
                & (source[:, 1] > 1)
            )
            visible_count = int(np.count_nonzero(visible))
            visible_total += visible_count
            frame_score["visible_keypoints"] = visible_count
            if visible_count < 4:
                frame_scores.append(frame_score)
                continue

            homography, inlier_mask = cv2.findHomography(
                source[visible],
                target[visible],
                cv2.RANSAC,
                320.0,
            )
            if homography is None or not np.all(np.isfinite(homography)):
                frame_scores.append(frame_score)
                continue
            inliers = (
                inlier_mask.reshape(-1).astype(bool)
                if inlier_mask is not None
                else np.ones(visible_count, dtype=bool)
            )
            inlier_count = int(np.count_nonzero(inliers))
            if inlier_count < 4:
                frame_scores.append(frame_score)
                continue
            errors = geometry._reprojection_errors(
                source[visible][inliers],
                target[visible][inliers],
                homography,
            )
            error = float(np.median(errors))
            inlier_ratio = inlier_count / max(1, visible_count)
            visible_target = target[visible]
            span_x = float(np.ptp(visible_target[:, 0]))
            span_y = float(np.ptp(visible_target[:, 1]))
            hull_area = float(cv2.contourArea(cv2.convexHull(source[visible])))
            frame_area = float(max(1, frame.shape[0] * frame.shape[1]))
            is_wide_view = (
                visible_count >= 8
                and span_x >= PITCH_LENGTH_CM * 0.50
                and span_y >= PITCH_WIDTH_CM * 0.40
                and hull_area >= frame_area * 0.045
            )
            valid_homographies += 1
            wide_views += int(is_wide_view)
            inlier_ratios.append(inlier_ratio)
            reprojection_errors.append(error)
            frame_score.update(
                {
                    "valid_homography": True,
                    "wide_view": is_wide_view,
                    "inliers": inlier_count,
                    "inlier_ratio": round(inlier_ratio, 4),
                    "median_reprojection_error_cm": round(error, 2),
                }
            )
            frame_scores.append(frame_score)

        return {
            "wide_view_frames": wide_views,
            "valid_homographies": valid_homographies,
            "visible_keypoints_total": visible_total,
            "mean_inlier_ratio": (
                round(float(np.mean(inlier_ratios)), 4)
                if inlier_ratios
                else 0.0
            ),
            "median_reprojection_error_cm": (
                round(float(np.median(reprojection_errors)), 2)
                if reprojection_errors
                else None
            ),
            "frames": frame_scores,
        }

    def _pitch_preview_rank(self, score: dict[str, Any]) -> tuple[float, ...]:
        error = score.get("median_reprojection_error_cm")
        return (
            float(score.get("wide_view_frames", 0)),
            float(score.get("valid_homographies", 0)),
            float(score.get("mean_inlier_ratio", 0.0)),
            -float(error if error is not None else 1_000_000.0),
            float(score.get("visible_keypoints_total", 0)),
        )

    def _resolve_model_path(self) -> Path:
        specialized = self._resolve_asset_path(settings.MATCH_ANALYSIS_PLAYER_MODEL_PATH)
        if specialized is not None:
            return specialized
        fallback = self._resolve_asset_path(
            settings.MATCH_ANALYSIS_PLAYER_MODEL_FALLBACK_PATH
        )
        if fallback is not None:
            return fallback
        configured = self._resolve_asset_path(settings.YOLO_MODEL_PATH)
        if configured is not None:
            return configured
        return Path("yolo11n.pt")

    def _unique_model_paths(
        self,
        values: list[tuple[str, Path | None]],
    ) -> list[tuple[str, Path]]:
        unique: list[tuple[str, Path]] = []
        seen: set[str] = set()
        for label, path in values:
            if path is None:
                continue
            normalized = str(path.resolve())
            if normalized in seen:
                continue
            seen.add(normalized)
            unique.append((label, path))
        return unique

    def _resolve_asset_path(self, value: str) -> Path | None:
        configured = Path(value)
        candidates = [
            configured,
            Path("/app") / configured,
            Path(__file__).resolve().parents[2] / configured,
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate
        return None

    def _resolve_tracker_config(self) -> str:
        configured = Path(settings.MATCH_ANALYSIS_TRACKER)
        if configured.exists():
            return str(configured)
        app_relative = Path("/app") / configured
        if app_relative.exists():
            return str(app_relative)
        return settings.MATCH_ANALYSIS_TRACKER

    def _load_kit_reference_config(
        self,
        team_context: dict[str, Any],
    ) -> dict[str, Any]:
        references = team_context.get("kit_references") or {}
        labels = {
            int(team): str(label)
            for team, label in (team_context.get("team_labels") or {}).items()
            if str(team).isdigit() and label
        }
        palettes: dict[int, list[tuple[int, int, int]]] = {}
        goalkeeper_palettes: dict[int, list[tuple[int, int, int]]] = {}
        summary: dict[str, Any] = {
            "source": "online_appearance_clustering",
            "teams": {},
        }

        for team in (1, 2):
            object_name = (
                references.get(f"team_{team}_selected")
                or references.get(f"team_{team}_primary")
                or references.get(f"team_{team}_alternate")
            )
            team_summary: dict[str, Any] = {
                "team": team,
                "label": labels.get(team, f"Team {team}"),
                "object_name": object_name,
                "status": "missing",
                "palette_bgr": [],
            }
            if object_name:
                try:
                    image = self._load_reference_image(object_name)
                    palette = self._extract_reference_palette(image)
                    if palette:
                        palettes[team] = palette
                        team_summary["status"] = "ready"
                        team_summary["palette_bgr"] = [
                            list(color)
                            for color in palette
                        ]
                    else:
                        team_summary["status"] = "no_reliable_colors"
                except Exception as exc:
                    team_summary["status"] = "failed"
                    team_summary["error"] = str(exc)
            summary["teams"][str(team)] = team_summary

            goalkeeper_object = references.get(f"team_{team}_goalkeeper")
            goalkeeper_summary: dict[str, Any] = {
                "object_name": goalkeeper_object,
                "status": "missing",
                "palette_bgr": [],
            }
            if goalkeeper_object:
                try:
                    goalkeeper_image = self._load_reference_image(goalkeeper_object)
                    goalkeeper_palette = self._extract_reference_palette(goalkeeper_image)
                    if goalkeeper_palette:
                        goalkeeper_palettes[team] = goalkeeper_palette
                        goalkeeper_summary["status"] = "ready"
                        goalkeeper_summary["palette_bgr"] = [
                            list(color) for color in goalkeeper_palette
                        ]
                    else:
                        goalkeeper_summary["status"] = "no_reliable_colors"
                except Exception as exc:
                    goalkeeper_summary["status"] = "failed"
                    goalkeeper_summary["error"] = str(exc)
            team_summary["goalkeeper"] = goalkeeper_summary

        if palettes:
            summary["source"] = "stored_kit_images_with_online_adaptation"
        summary["seeded_teams"] = sorted(palettes)
        return {
            "palettes": palettes,
            "goalkeeper_palettes": goalkeeper_palettes,
            "team_labels": labels,
            "summary": summary,
        }

    def _load_reference_image(self, object_name: str) -> np.ndarray:
        response = client.get_object(BUCKET_NAME, object_name)
        try:
            raw = response.read()
        finally:
            response.close()
            response.release_conn()
        image = cv2.imdecode(
            np.frombuffer(raw, dtype=np.uint8),
            cv2.IMREAD_UNCHANGED,
        )
        if image is None:
            raise ValueError("Could not decode stored kit image")
        return image

    def _extract_reference_palette(
        self,
        image: np.ndarray,
    ) -> list[tuple[int, int, int]]:
        alpha_mask: np.ndarray | None = None
        if image.ndim == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        elif image.shape[2] == 4:
            alpha_mask = image[:, :, 3] >= 24
            image = image[:, :, :3]

        height, width = image.shape[:2]
        inset_y = max(0, int(height * 0.04))
        inset_x = max(0, int(width * 0.04))
        image = image[
            inset_y : max(inset_y + 1, height - inset_y),
            inset_x : max(inset_x + 1, width - inset_x),
        ]
        if alpha_mask is not None:
            alpha_mask = alpha_mask[
                inset_y : max(inset_y + 1, height - inset_y),
                inset_x : max(inset_x + 1, width - inset_x),
            ]
        image = cv2.resize(image, (180, 180), interpolation=cv2.INTER_AREA)
        if alpha_mask is not None:
            alpha_mask = cv2.resize(
                alpha_mask.astype(np.uint8),
                (180, 180),
                interpolation=cv2.INTER_NEAREST,
            ).astype(bool)

        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        usable = ~(
            (hsv[:, :, 1] < 20)
            & (hsv[:, :, 2] > 238)
        )
        if alpha_mask is not None:
            usable &= alpha_mask
        pixels = image[usable].reshape(-1, 3).astype(np.float32)
        if len(pixels) < 64:
            pixels = image.reshape(-1, 3).astype(np.float32)

        cluster_count = min(4, max(1, len(pixels) // 64))
        cv2.setRNGSeed(42)
        _, labels, centers = cv2.kmeans(
            pixels,
            cluster_count,
            None,
            (
                cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
                30,
                0.5,
            ),
            5,
            cv2.KMEANS_PP_CENTERS,
        )
        counts = np.bincount(labels.reshape(-1), minlength=cluster_count)
        ordered = sorted(
            zip(centers, counts),
            key=lambda item: int(item[1]),
            reverse=True,
        )
        palette: list[tuple[int, int, int]] = []
        for center, count in ordered:
            if int(count) / max(1, len(pixels)) < 0.025:
                continue
            color = tuple(int(round(channel)) for channel in center)
            if any(
                float(np.linalg.norm(np.array(color) - np.array(existing))) < 24.0
                for existing in palette
            ):
                continue
            palette.append(color)
        return palette[:4]

    def _transcode_for_browser(self, source_path: Path, output_path: Path) -> str:
        ffmpeg = shutil.which("ffmpeg")
        if ffmpeg is None:
            source_path.replace(output_path)
            return "mjpg"
        command = [
            ffmpeg,
            "-y",
            "-i",
            str(source_path),
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "24",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(output_path),
        ]
        completed = subprocess.run(command, capture_output=True, text=True, check=False)
        if completed.returncode != 0 or not output_path.exists() or output_path.stat().st_size <= 1024:
            source_path.replace(output_path)
            return "mjpg"
        return "h264"

    def _download_video(self, bucket: str, object_name: str, local_path: Path) -> None:
        response = client.get_object(bucket, object_name)
        try:
            with local_path.open("wb") as file:
                for chunk in response.stream(1024 * 1024):
                    file.write(chunk)
        finally:
            response.close()
            response.release_conn()

    def _put_file(self, bucket: str, object_name: str, path: Path, content_type: str) -> None:
        with path.open("rb") as file:
            client.put_object(
                bucket,
                object_name,
                file,
                length=path.stat().st_size,
                content_type=content_type,
            )

    def _put_json(self, bucket: str, object_name: str, payload: dict[str, Any]) -> None:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        client.put_object(bucket, object_name, io.BytesIO(data), length=len(data), content_type="application/json")

    def _put_bytes(
        self,
        bucket: str,
        object_name: str,
        data: bytes,
        content_type: str,
    ) -> None:
        client.put_object(
            bucket,
            object_name,
            io.BytesIO(data),
            length=len(data),
            content_type=content_type,
        )
