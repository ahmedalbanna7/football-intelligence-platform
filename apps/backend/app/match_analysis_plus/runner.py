from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter
from typing import Any
import colorsys
import io
import json
import shutil
import subprocess

import cv2
import numpy as np

from app.core.config import settings
from app.services.minio_client import client


PERSON_ALIASES = {"person", "player", "goalkeeper"}
BALL_ALIASES = {"sports ball", "ball"}
TEAM_DISPLAY_COLORS = {
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
VISUAL_LAYER_SCHEMA_VERSION = 1
VISUAL_LAYER_SAMPLE_RATE_HZ = 6.0
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


class PlayerValidityFilter:
    """Reject obvious field fixtures before they can receive a stable identity."""

    def __init__(self) -> None:
        self.raw_seen = 0
        self.kept = 0
        self.rejected_implausible_shape = 0
        self.rejected_field_fixture = 0
        self.rejected_sparse_foreground = 0

    def filter(
        self,
        players: list[AnalysisObject],
        frame: np.ndarray,
    ) -> list[AnalysisObject]:
        kept: list[AnalysisObject] = []
        for player in players:
            self.raw_seen += 1
            width = max(1.0, player.bbox[2] - player.bbox[0])
            height = max(1.0, player.bbox[3] - player.bbox[1])
            aspect_ratio = width / height
            if aspect_ratio < 0.105 or aspect_ratio > 1.18:
                self.rejected_implausible_shape += 1
                continue
            if self._looks_like_thin_field_fixture(frame, player.bbox, aspect_ratio):
                self.rejected_field_fixture += 1
                continue
            kept.append(player)
            self.kept += 1
        return kept

    def summary(self) -> dict[str, Any]:
        return {
            "engine": "player_validity_filter_v1",
            "raw_player_detections": self.raw_seen,
            "kept_player_detections": self.kept,
            "rejected_implausible_shape": self.rejected_implausible_shape,
            "rejected_field_fixtures": self.rejected_field_fixture,
            "rejected_sparse_foreground": self.rejected_sparse_foreground,
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


class TrackIdStabilizer:
    def __init__(
        self,
        max_gap_frames: int = 240,
        confirmation_hits: int = 4,
        hidden_hold_frames: int = 18,
    ) -> None:
        self.max_gap_frames = max_gap_frames
        self.confirmation_hits = confirmation_hits
        self.hidden_hold_frames = hidden_hold_frames
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
        return {
            "engine": "identity_isolation_stabilizer_v4_observation_only",
            "raw_track_ids_seen": raw_count,
            "stable_tracks_count": len(confirmed_tracks),
            "internal_track_candidates": len(self.tracks),
            "tentative_tracks_count": len(self.tracks) - len(confirmed_tracks),
            "max_gap_frames": self.max_gap_frames,
            "confirmation_hits": self.confirmation_hits,
            "hidden_hold_frames": self.hidden_hold_frames,
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
            "tracks_with_multiple_raw_ids": sum(1 for count in raw_ids_per_stable.values() if count > 1),
            "max_raw_ids_per_stable_track": max(raw_ids_per_stable.values(), default=0),
            "avg_raw_ids_per_stable_track": round(raw_count / stable_count, 3),
            "fragmentation_reduction_percent": round(max(0, raw_count - len(confirmed_tracks)) * 100 / max(raw_count, 1), 2),
            "raw_ids_per_stable_track": raw_ids_per_stable,
            "identity_locked_tracks": sum(1 for state in confirmed_tracks.values() if state.identity_locked),
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
    ) -> bool:
        if not crowded:
            return False
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
        required_margin = 0.90 if severe_overlap else 0.48
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
        same_raw = self.raw_to_stable.get(raw_id) == state.stable_id
        trusted_visual = visual_reliable and visual_quality >= 0.80
        bootstrap_raw = same_raw and not trusted_visual and state.reliable_hits < 4

        family_mismatch = self._color_family_mismatch(state.jersey_family, jersey_family)
        if family_mismatch and family_confidence >= 0.70:
            self.rejected_color_family_mismatches += 1
            if (
                state.identity_locked
                and trusted_visual
                and color_similarity < 0.50
                and appearance < 0.72
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

        far_gate = max_distance * (1.4 if gap <= 4 else 1.9)
        if (
            center_distance > far_gate
            and foot_distance > far_gate
            and appearance < 0.82
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
            if self._raw_id_identity_mismatch(state, player, frame_index, appearance_hist, jersey_color):
                self.raw_id_identity_mismatch_ignores += 1
            else:
                raw_bonus = 0.72

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
        min_score = 3.9 if gap <= 4 else 4.2
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
            bbox_height=self._bbox_height(player.bbox),
            depth_proxy=self._depth_proxy(player.bbox),
            last_reliable_frame=frame_index if appearance_hist is not None else 0,
            reliable_hits=1 if appearance_hist is not None else 0,
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
    ) -> None:
        state = self.tracks[stable_id]
        new_center = self._center(player.bbox)
        new_foot = self._foot(player.bbox)
        new_depth = self._depth_proxy(player.bbox)
        frame_delta = max(frame_index - state.last_frame, 1)
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
        if not visual_reliable:
            state.occlusion_hits += 1
        if visual_reliable and appearance_hist is not None:
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
        if visual_reliable and jersey_color is not None:
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

    def _max_center_distance(self, bbox_a: list[float], bbox_b: list[float], gap: int) -> float:
        size = max(self._bbox_size(bbox_a), self._bbox_size(bbox_b), 1.0)
        return max(65.0, min(320.0, size * 1.55 + min(gap, 45) * 5.0))

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
        torso = person[6:36, 8:24]
        if torso.size == 0:
            return None, None

        hsv = cv2.cvtColor(torso, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, np.array([0, 38, 45]), np.array([179, 255, 255]))
        if int(np.count_nonzero(mask)) < max(12, int(mask.size * 0.08)):
            mask = None

        region_descriptors = [
            self._region_hist(person[0:20, 7:25]),
            self._region_hist(torso),
            self._region_hist(person[34:61, 4:28]),
        ]
        gray = cv2.cvtColor(person, cv2.COLOR_BGR2GRAY)
        gradients = self._gradient_descriptor(gray)
        appearance = self._normalize_hist(
            np.concatenate(
                [
                    region_descriptors[0] * 0.85,
                    region_descriptors[1] * 1.35,
                    region_descriptors[2] * 1.0,
                    gradients * 0.55,
                ]
            ).astype(np.float32)
        )
        if mask is not None:
            pixels = torso[mask > 0].reshape(-1, 3).astype(np.float32)
        else:
            pixels = torso.reshape(-1, 3).astype(np.float32)
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
        gallery_best = max(
            self._appearance_similarity(candidate, reference)
            for reference in state.appearance_gallery
        )
        return max(aggregate, gallery_best * 0.96)

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
        return float(np.clip(history * 0.42 + appearance * 0.28 + continuity * 0.30 + lock_bonus, 0.0, 1.0))

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
    """Keep two stable kit-color anchors and vote per stable player identity."""

    def __init__(self) -> None:
        self.anchors: dict[int, tuple[int, int, int]] = {}
        self.track_votes: dict[int, dict[int, float]] = {}
        self.observations = 0
        self.anchor_initializations = 0

    def update(
        self,
        players: list[AnalysisObject],
        track_states: dict[int, StableTrackState],
        team_by_track: dict[int, int],
    ) -> None:
        colors: dict[int, tuple[int, int, int]] = {}
        for player in players:
            state = track_states.get(player.track_id)
            if state is None or state.jersey_color is None:
                continue
            colors[player.track_id] = state.jersey_color
            self.observations += 1

        if not colors:
            return
        created_second_anchor = self._ensure_anchors(list(colors.values()))
        if created_second_anchor:
            self.track_votes.clear()

        assignments: dict[int, int] = {}
        for track_id, color in colors.items():
            team = self._nearest_team(color)
            assignments[track_id] = team
            votes = self.track_votes.setdefault(track_id, {1: 0.0, 2: 0.0})
            votes[1] *= 0.86
            votes[2] *= 0.86
            votes[team] += 1.0
            team_by_track[track_id] = max(votes, key=votes.get)

        if len(self.anchors) == 2:
            self._update_anchors(colors, assignments)

    def summary(self) -> dict[str, Any]:
        return {
            "engine": "stable_kit_color_classifier_v2",
            "kit_anchors_bgr": {
                str(team): list(color)
                for team, color in sorted(self.anchors.items())
            },
            "classified_tracks": len(self.track_votes),
            "color_observations": self.observations,
            "anchor_initializations": self.anchor_initializations,
        }

    def _ensure_anchors(self, colors: list[tuple[int, int, int]]) -> bool:
        if not self.anchors:
            if len(colors) == 1:
                self.anchors[1] = colors[0]
                self.anchor_initializations += 1
                return False
            best_pair: tuple[tuple[int, int, int], tuple[int, int, int]] | None = None
            best_distance = -1.0
            for first_index, first in enumerate(colors):
                for second in colors[first_index + 1 :]:
                    distance = self._color_distance(first, second)
                    if distance > best_distance:
                        best_distance = distance
                        best_pair = (first, second)
            if best_pair is not None and best_distance >= 0.30:
                self.anchors = {1: best_pair[0], 2: best_pair[1]}
                self.anchor_initializations += 2
                return True
            self.anchors[1] = colors[0]
            self.anchor_initializations += 1
            return False

        if len(self.anchors) == 1:
            anchor = self.anchors[1]
            candidate = max(colors, key=lambda color: self._color_distance(anchor, color))
            if self._color_distance(anchor, candidate) >= 0.30:
                self.anchors[2] = candidate
                self.anchor_initializations += 1
                return True
        return False

    def _nearest_team(self, color: tuple[int, int, int]) -> int:
        return min(
            self.anchors,
            key=lambda team: self._color_distance(color, self.anchors[team]),
        )

    def _update_anchors(
        self,
        colors: dict[int, tuple[int, int, int]],
        assignments: dict[int, int],
    ) -> None:
        for team in (1, 2):
            samples = [
                colors[track_id]
                for track_id, assigned_team in assignments.items()
                if assigned_team == team
            ]
            if not samples:
                continue
            sample = tuple(
                int(value)
                for value in np.median(np.array(samples, dtype=np.float32), axis=0)
            )
            anchor = self.anchors[team]
            self.anchors[team] = tuple(
                int(round(anchor[index] * 0.97 + sample[index] * 0.03))
                for index in range(3)
            )

    def _color_distance(
        self,
        first: tuple[int, int, int],
        second: tuple[int, int, int],
    ) -> float:
        pixels = np.array([[list(first), list(second)]], dtype=np.uint8)
        hsv = cv2.cvtColor(pixels, cv2.COLOR_BGR2HSV)[0]
        hue_gap = abs(float(hsv[0, 0]) - float(hsv[1, 0]))
        hue_gap = min(hue_gap, 180.0 - hue_gap) / 90.0
        saturation_gap = abs(float(hsv[0, 1]) - float(hsv[1, 1])) / 255.0
        value_gap = abs(float(hsv[0, 2]) - float(hsv[1, 2])) / 255.0
        both_colored = hsv[0, 1] >= 42 and hsv[1, 1] >= 42
        if both_colored:
            return hue_gap * 0.72 + saturation_gap * 0.18 + value_gap * 0.10
        return hue_gap * 0.10 + saturation_gap * 0.48 + value_gap * 0.42


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
        self.pitch_stabilized_observations = 0

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
            candidate = self._observe(
                frame_index=frame_index,
                image_center=center,
                pitch_center=pitch_center,
                raw_track_id=ball.raw_track_id,
                aspect=width / height,
                frame_width=frame_width,
                near_player=near_player,
            )

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
            "engine": "motion_confirmed_ball_filter_v3",
            "raw_ball_observations": self.raw_seen,
            "kept_ball_observations": self.kept,
            "filtered_static_candidates": self.filtered_static,
            "suppressed_tentative_observations": self.suppressed_tentative,
            "static_hits_threshold": self.static_hits,
            "pitch_stabilized_observations": self.pitch_stabilized_observations,
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


class PitchRadar:
    def __init__(self, model: Any | None, stride: int = 12) -> None:
        self.model = model
        self.stride = max(1, stride)
        self.homography: np.ndarray | None = None
        self.calibration_mode: str | None = None
        self.last_calibrated_frame = -1
        self.attempts = 0
        self.successes = 0
        self.assisted_calibrations = 0
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
        self.visual_marker_observations = 0
        self.visual_marker_tracks: list[dict[str, Any]] = []
        self.next_visual_marker_id = 1
        self.rendered_frames = 0
        self.last_visible_keypoints = 0
        self.last_inliers = 0
        self.last_reprojection_error_cm: float | None = None
        self.last_target_span_cm: tuple[float, float] = (0.0, 0.0)
        self.last_player_valid_ratio: float | None = None
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
            self._track_camera_motion(frame, frame_index, players)
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
                        )
                        self.last_line_alignment_score = round(line_score, 4)
                        self.goal_geometry_calibrations += 1
                        return
                    self.goal_geometry_rejections += 1

            if self.homography is not None:
                return
            if self.model is None or frame_index % self.stride != 0:
                return

            results = self.model.predict(
                frame,
                imgsz=settings.YOLO_IMAGE_SIZE,
                device=settings.YOLO_DEVICE,
                verbose=False,
            )
            if not results or results[0].keypoints is None:
                return
            keypoints = results[0].keypoints
            source = keypoints.xy.cpu().numpy()
            if source.ndim == 3:
                source = source[0]
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
                (confidence_values >= 0.42)
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
            if not is_wide_view:
                self.rejected_local_calibrations += 1
                return

            homography, _ = cv2.findHomography(
                source[visible],
                target[visible],
                0,
            )
            if homography is None or not np.all(np.isfinite(homography)):
                self.rejected_geometry += 1
                return
            errors = self._reprojection_errors(
                source[visible],
                target[visible],
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
            if (
                not np.isfinite(error)
                or error > 260.0
                or p90_error > 480.0
                or not player_ok
                or line_score < 0.42
            ):
                self.rejected_geometry += 1
                return
            self._accept_homography(
                homography,
                frame_index,
                error,
                self.last_visible_keypoints,
                "wide_view_keypoints",
                player_ratio,
            )
            self.last_line_alignment_score = round(line_score, 4)
            self.wide_view_calibrations += 1
        except (AttributeError, IndexError, RuntimeError, TypeError, ValueError, cv2.error):
            self.errors += 1

    def _accept_homography(
        self,
        homography: np.ndarray,
        frame_index: int,
        error: float,
        inliers: int,
        mode: str,
        player_ratio: float,
    ) -> None:
        self.homography = homography / homography[2, 2]
        self.calibration_mode = mode
        self.last_calibrated_frame = frame_index
        self.last_reprojection_error_cm = round(error, 2)
        self.last_inliers = inliers
        self.last_player_valid_ratio = round(player_ratio, 4)
        self.successes += 1

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
            self.homography is None
            or previous_gray is None
            or previous_mask is None
            or previous_scale is None
            or abs(previous_scale - current_scale) > 1e-6
        ):
            return

        self.camera_tracking_attempts += 1
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
        self.last_camera_inliers = inlier_count
        self.last_camera_inlier_ratio = round(inlier_ratio, 4)
        self.last_camera_reprojection_error_px = round(reprojection_error, 3)

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

        center_y = PITCH_WIDTH_CM / 2
        penalty_half = PENALTY_AREA_WIDTH_CM / 2
        goal_half = GOAL_AREA_WIDTH_CM / 2
        segments = [
            ((PITCH_LENGTH_CM, center_y - penalty_half), (PITCH_LENGTH_CM, center_y + penalty_half)),
            ((PITCH_LENGTH_CM - PENALTY_AREA_LENGTH_CM, center_y - penalty_half), (PITCH_LENGTH_CM - PENALTY_AREA_LENGTH_CM, center_y + penalty_half)),
            ((PITCH_LENGTH_CM - PENALTY_AREA_LENGTH_CM, center_y - penalty_half), (PITCH_LENGTH_CM, center_y - penalty_half)),
            ((PITCH_LENGTH_CM - PENALTY_AREA_LENGTH_CM, center_y + penalty_half), (PITCH_LENGTH_CM, center_y + penalty_half)),
            ((PITCH_LENGTH_CM - GOAL_AREA_LENGTH_CM, center_y - goal_half), (PITCH_LENGTH_CM - GOAL_AREA_LENGTH_CM, center_y + goal_half)),
            ((PITCH_LENGTH_CM - GOAL_AREA_LENGTH_CM, center_y - goal_half), (PITCH_LENGTH_CM, center_y - goal_half)),
            ((PITCH_LENGTH_CM - GOAL_AREA_LENGTH_CM, center_y + goal_half), (PITCH_LENGTH_CM, center_y + goal_half)),
        ]
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
        return float(np.mean(scores[: min(5, len(scores))]))

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
            self.homography is None
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
            pitch_xy = self.transform_point(
                ((ball.bbox[0] + ball.bbox[2]) / 2, (ball.bbox[1] + ball.bbox[3]) / 2)
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

    def summary(self) -> dict[str, Any]:
        return {
            "engine": "metric_pitch_geometry_radar_v3",
            "model_available": self.model is not None,
            "calibration_mode": self.calibration_mode,
            "calibration_attempts": self.attempts,
            "successful_calibrations": self.successes,
            "assisted_calibrations": self.assisted_calibrations,
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
            },
            "projection_model": "planar_homography_with_camera_motion_compensation",
            "ground_plane_3d": {"z_cm": 0.0},
            "errors": self.errors,
        }

    def transform_point(self, point: tuple[float, float]) -> tuple[float, float] | None:
        if self.homography is None:
            return None
        transformed = cv2.perspectiveTransform(
            np.array(point, dtype=np.float32).reshape(1, 1, 2),
            self.homography,
        ).reshape(2)
        x_cm, y_cm = float(transformed[0]), float(transformed[1])
        if not np.isfinite(x_cm) or not np.isfinite(y_cm):
            return None
        if x_cm < -250 or x_cm > PITCH_LENGTH_CM + 250:
            return None
        if y_cm < -250 or y_cm > PITCH_WIDTH_CM + 250:
            return None
        return (
            min(PITCH_LENGTH_CM, max(0.0, x_cm)),
            min(PITCH_WIDTH_CM, max(0.0, y_cm)),
        )

    def pitch_to_video_matrix(self) -> np.ndarray | None:
        """Return the current metric-pitch to source-video projection."""
        if self.homography is None:
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
        self.model = None
        self.pitch_model = None
        self.model_path = self._resolve_model_path()
        self.pitch_model_path = self._resolve_asset_path(settings.MATCH_ANALYSIS_PITCH_MODEL_PATH)
        self.model_mode = "unloaded"

    def run(
        self,
        run_id: int,
        match_id: int,
        bucket: str,
        object_name: str,
        artifact_prefix: str,
        mode: str = "FULL_ANALYSIS",
        max_frames: int = 450,
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

            self._download_video(bucket, object_name, input_path)
            summary = self._process_video(
                input_path=input_path,
                raw_output_path=raw_output_path,
                output_path=output_path,
                thumbnail_path=thumbnail_path,
                mode=normalized_mode,
                max_frames=max_frames,
            )

            visual_layers_payload = summary.pop("_visual_layers_payload", None)

            output_object = f"{artifact_prefix}/output.mp4"
            summary_object = f"{artifact_prefix}/summary.json"
            thumbnail_object = f"{artifact_prefix}/thumbnail.jpg"
            visual_layers_object = f"{artifact_prefix}/visual_layers.json"
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
            return payload

    def _process_video(
        self,
        input_path: Path,
        raw_output_path: Path,
        output_path: Path,
        thumbnail_path: Path,
        mode: str,
        max_frames: int,
    ) -> dict[str, Any]:
        start = perf_counter()
        model = self._load_model()
        pitch_model = self._load_pitch_model()
        capture = cv2.VideoCapture(str(input_path))
        if not capture.isOpened():
            raise ValueError("Could not open uploaded video for match analysis")

        fps = capture.get(cv2.CAP_PROP_FPS) or 24.0
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 1280)
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 720)
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
        ball_control: list[int] = []
        class_counts: dict[str, int] = {}
        confidence_values: list[float] = []
        player_filter = PlayerValidityFilter()
        track_stabilizer = TrackIdStabilizer()
        team_classifier = TeamColorClassifier()
        ball_filter = BallStaticFilter()
        radar = PitchRadar(pitch_model, settings.MATCH_ANALYSIS_RADAR_STRIDE)
        frames_processed = 0
        detections_count = 0
        raw_detections_count = 0

        while max_frames <= 0 or frames_processed < max_frames:
            ok, frame = capture.read()
            if not ok:
                break

            raw_objects = self._detect_and_track(model, frame, mode)
            raw_detections_count += len(raw_objects)
            raw_players = player_filter.filter(
                [item for item in raw_objects if item.class_name == "player"],
                frame,
            )
            raw_balls = [item for item in raw_objects if item.class_name == "ball"]
            players = track_stabilizer.update(frames_processed, raw_players, frame)
            balls = ball_filter.filter(
                frames_processed,
                raw_balls,
                players,
                width,
                pitch_transform=radar.transform_point,
            )
            radar.update(
                frame,
                frames_processed,
                players=players,
                static_markers=ball_filter.static_marker_centers(frames_processed),
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
                if item.confidence is not None:
                    confidence_values.append(item.confidence)

            team_classifier.update(players, track_stabilizer.tracks, team_by_track)
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
            current_control = self._ball_control(
                players,
                balls,
                team_by_track,
                ball_control,
                pitch_transform=radar.transform_point,
            )
            if current_control is not None:
                ball_control.append(current_control)

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
                players=players,
                balls=balls,
                team_by_track=team_by_track,
            )
            if frames_processed == 0:
                cv2.imwrite(str(thumbnail_path), annotated)
            writer.write(annotated)
            frames_processed += 1

        capture.release()
        writer.release()
        if frames_processed == 0:
            raise ValueError("No frames were processed by match analysis")
        output_codec = self._transcode_for_browser(raw_output_path, output_path)

        tracks: list[dict[str, Any]] = []
        for track_id in sorted(track_frames):
            state = track_stabilizer.tracks.get(track_id)
            video_samples = track_video_samples.get(track_id, [])
            pitch_samples = track_pitch_samples.get(track_id, [])
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
                }
            )
        team_1 = sum(1 for item in ball_control if item == 1)
        team_2 = sum(1 for item in ball_control if item == 2)
        total_control = max(team_1 + team_2, 1)
        elapsed_ms = round((perf_counter() - start) * 1000, 2)
        processing_fps = round(frames_processed / max(elapsed_ms / 1000, 0.001), 3)
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
        )

        return {
            "status": "ok",
            "engine": "match_analysis_plus",
            "model": str(self.model_path),
            "model_mode": self.model_mode,
            "pitch_model": str(self.pitch_model_path) if self.pitch_model_path is not None else None,
            "tracker": settings.MATCH_ANALYSIS_TRACKER,
            "output_codec": output_codec,
            "output_content_type": "video/mp4",
            "frames_processed": frames_processed,
            "max_frames": max_frames,
            "fps": round(float(fps), 3),
            "processing_fps": processing_fps,
            "resolution": [width, height],
            "detections_count": detections_count,
            "raw_detections_count": raw_detections_count,
            "class_counts": class_counts,
            "confidence": {
                "avg": round(float(np.mean(confidence_values)), 4) if confidence_values else None,
                "min": round(float(np.min(confidence_values)), 4) if confidence_values else None,
                "max": round(float(np.max(confidence_values)), 4) if confidence_values else None,
            },
            "tracks_count": len(tracks),
            "raw_tracks_count": len(track_stabilizer.raw_ids_seen),
            "player_filter": player_filter.summary(),
            "team_classifier": team_classifier.summary(),
            "id_stabilizer": track_stabilizer.summary(),
            "ball_filter": ball_filter.summary(),
            "radar": radar.summary(),
            "metric_tracking": {
                "coordinate_system": "pitch_centimeters",
                "ground_plane_z_cm": 0.0,
                "trajectory_sample_rate_hz": VISUAL_LAYER_SAMPLE_RATE_HZ,
                "heatmap_ready": radar.homography is not None,
            },
            "tracks": tracks[:250],
            "_visual_layers_payload": visual_layers_payload,
            "team_ball_control": {
                "team_1_percent": round(team_1 * 100 / total_control, 2),
                "team_2_percent": round(team_2 * 100 / total_control, 2),
            },
            "notes": [
                "sports-main source is vendored in apps/match-analysis-worker/sports-main",
                "every run executes player, ball, tracking, team classification, and pitch radar analysis",
                "field-fixture filtering runs before stable player identity assignment",
                "distance and speed use metric ground-plane coordinates only after validated pitch calibration",
            ],
            "elapsed_ms": elapsed_ms,
        }

    def _detect_and_track(self, model: Any, frame: np.ndarray, mode: str) -> list[AnalysisObject]:
        classes = self._target_class_ids(model)
        results = model.track(
            frame,
            persist=True,
            conf=max(settings.YOLO_CONFIDENCE, 0.25),
            imgsz=settings.YOLO_IMAGE_SIZE,
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
                )
            )
        return objects

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
    ) -> dict[str, Any]:
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
                    "color": self._track_visual_color(track_id),
                    "frames": track_frames.get(track_id, 0),
                    "first_frame": int(video_path[0][0]),
                    "last_frame": int(video_path[-1][0]),
                    "video_path": video_path,
                    "pitch_path": pitch_path,
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
            self._draw_triangle(frame, ball.bbox, (0, 255, 255))
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
        text_color = (15, 15, 15) if team == 1 else (255, 255, 255)
        cv2.ellipse(frame, (center_x, y2), (max(10, (x2 - x1) // 2), 8), 0, -45, 235, color, 2)
        scale = self._font_scale(frame, 0.5)
        small = self._font_scale(frame, 0.38)
        thickness = self._thickness(frame)
        label = str(player.track_id)
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

    def _draw_triangle(self, frame: np.ndarray, bbox: list[float], color: tuple[int, int, int]) -> None:
        x = int((bbox[0] + bbox[2]) / 2)
        y = int(bbox[1])
        points = np.array([[x, y], [x - 9, y - 18], [x + 9, y - 18]])
        cv2.drawContours(frame, [points], 0, color, cv2.FILLED)
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

    def _target_class_ids(self, model: Any) -> list[int]:
        names = getattr(model, "names", {}) or {}
        if isinstance(names, list):
            names = dict(enumerate(names))
        class_ids = [
            int(class_id)
            for class_id, class_name in names.items()
            if self._map_class_name(str(class_name).lower()) is not None
        ]
        return sorted(class_ids)

    def _load_model(self) -> Any:
        if self.model is not None:
            return self.model
        from ultralytics import YOLO

        self.model = YOLO(str(self.model_path))
        class_names = {
            str(class_name).lower()
            for class_name in (getattr(self.model, "names", {}) or {}).values()
        }
        self.model_mode = (
            "football-specialized-yolo"
            if {"ball", "goalkeeper", "player"}.issubset(class_names)
            else "balanced-yolo-with-football-guards"
        )
        return self.model

    def _load_pitch_model(self) -> Any | None:
        if self.pitch_model is not None:
            return self.pitch_model
        if self.pitch_model_path is None or not self.pitch_model_path.exists():
            return None
        from ultralytics import YOLO

        self.pitch_model = YOLO(str(self.pitch_model_path))
        return self.pitch_model

    def _resolve_model_path(self) -> Path:
        specialized = self._resolve_asset_path(settings.MATCH_ANALYSIS_PLAYER_MODEL_PATH)
        if specialized is not None:
            return specialized
        configured = self._resolve_asset_path(settings.YOLO_MODEL_PATH)
        if configured is not None:
            return configured
        return Path("yolo11n.pt")

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
            data = file.read()
        client.put_object(bucket, object_name, io.BytesIO(data), length=len(data), content_type=content_type)

    def _put_json(self, bucket: str, object_name: str, payload: dict[str, Any]) -> None:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        client.put_object(bucket, object_name, io.BytesIO(data), length=len(data), content_type="application/json")
