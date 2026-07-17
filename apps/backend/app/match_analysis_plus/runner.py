from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter
from typing import Any
import io
import json
import shutil
import subprocess

import cv2
import numpy as np

from app.core.config import settings
from app.services.minio_client import client


PERSON_ALIASES = {"person", "player", "goalkeeper", "referee"}
BALL_ALIASES = {"sports ball", "ball"}


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

@dataclass
class BallStaticCandidate:
    center: tuple[float, float]
    first_frame: int
    last_frame: int
    hits: int = 1
    rejected_hits: int = 0


class BallStaticFilter:
    def __init__(self, cell_size: int = 24, static_hits: int = 5) -> None:
        self.cell_size = cell_size
        self.static_hits = static_hits
        self.candidates: dict[tuple[int, int], BallStaticCandidate] = {}
        self.raw_seen = 0
        self.kept = 0
        self.filtered_static = 0

    def filter(
        self,
        frame_index: int,
        balls: list[AnalysisObject],
        players: list[AnalysisObject],
        frame_width: int,
    ) -> list[AnalysisObject]:
        kept: list[AnalysisObject] = []
        for ball in balls:
            self.raw_seen += 1
            center = self._center(ball.bbox)
            if self._near_player_foot(center, players, frame_width):
                kept.append(ball)
                self.kept += 1
                self._remember(frame_index, center)
                continue

            candidate = self._remember(frame_index, center)
            if candidate.hits >= self.static_hits:
                candidate.rejected_hits += 1
                self.filtered_static += 1
                continue

            kept.append(ball)
            self.kept += 1
        return kept

    def summary(self) -> dict[str, Any]:
        return {
            "engine": "static_field_marker_filter_v1",
            "raw_ball_observations": self.raw_seen,
            "kept_ball_observations": self.kept,
            "filtered_static_candidates": self.filtered_static,
            "static_hits_threshold": self.static_hits,
        }

    def _remember(self, frame_index: int, center: tuple[float, float]) -> BallStaticCandidate:
        key = (round(center[0] / self.cell_size), round(center[1] / self.cell_size))
        candidate = self.candidates.get(key)
        if candidate is None:
            candidate = BallStaticCandidate(center=center, first_frame=frame_index, last_frame=frame_index)
            self.candidates[key] = candidate
            return candidate
        candidate.hits += 1
        candidate.last_frame = frame_index
        alpha = 1.0 / candidate.hits
        candidate.center = (
            candidate.center[0] * (1.0 - alpha) + center[0] * alpha,
            candidate.center[1] * (1.0 - alpha) + center[1] * alpha,
        )
        return candidate

    def _near_player_foot(
        self,
        center: tuple[float, float],
        players: list[AnalysisObject],
        frame_width: int,
    ) -> bool:
        threshold = max(42.0, frame_width * 0.032)
        for player in players:
            foot = ((player.bbox[0] + player.bbox[2]) / 2, player.bbox[3])
            if float(np.hypot(foot[0] - center[0], foot[1] - center[1])) <= threshold:
                return True
        return False

    def _center(self, bbox: list[float]) -> tuple[float, float]:
        return ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)


class MatchAnalysisPlusRunner:
    supported_modes = {
        "PLAYER_DETECTION",
        "BALL_DETECTION",
        "PLAYER_TRACKING",
        "TEAM_CLASSIFICATION",
        "RADAR",
    }

    def __init__(self) -> None:
        self.model = None
        self.model_path = self._resolve_model_path()

    def run(
        self,
        run_id: int,
        match_id: int,
        bucket: str,
        object_name: str,
        artifact_prefix: str,
        mode: str = "PLAYER_TRACKING",
        max_frames: int = 450,
    ) -> dict[str, Any]:
        normalized_mode = (mode or "PLAYER_TRACKING").upper()
        if normalized_mode not in self.supported_modes:
            raise ValueError(f"Unsupported match analysis mode: {mode}")

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

            output_object = f"{artifact_prefix}/output.mp4"
            summary_object = f"{artifact_prefix}/summary.json"
            thumbnail_object = f"{artifact_prefix}/thumbnail.jpg"
            self._put_file(bucket, output_object, output_path, "video/mp4")
            if thumbnail_path.exists():
                self._put_file(bucket, thumbnail_object, thumbnail_path, "image/jpeg")

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
        team_by_track: dict[int, int] = {}
        team_colors: dict[int, tuple[int, int, int]] = {}
        ball_control: list[int] = []
        class_counts: dict[str, int] = {}
        confidence_values: list[float] = []
        track_stabilizer = TrackIdStabilizer()
        ball_filter = BallStaticFilter()
        frames_processed = 0
        detections_count = 0
        raw_detections_count = 0

        while max_frames <= 0 or frames_processed < max_frames:
            ok, frame = capture.read()
            if not ok:
                break

            raw_objects = self._detect_and_track(model, frame, mode)
            raw_detections_count += len(raw_objects)
            raw_players = [item for item in raw_objects if item.class_name == "player"]
            raw_balls = [item for item in raw_objects if item.class_name == "ball"]
            players = track_stabilizer.update(frames_processed, raw_players, frame)
            balls = ball_filter.filter(frames_processed, raw_balls, players, width)
            objects = players + balls
            detected_objects = [item for item in objects if not item.is_predicted]
            detections_count += len(detected_objects)
            for item in detected_objects:
                class_counts[item.class_name] = class_counts.get(item.class_name, 0) + 1
                if item.confidence is not None:
                    confidence_values.append(item.confidence)

            self._assign_teams(frame, players, team_by_track, team_colors)
            self._update_movement(
                players=players,
                frame_index=frames_processed,
                fps=fps,
                frame_width=width,
                last_positions=last_positions,
                track_distance=track_distance,
                track_speed=track_speed,
                track_frames=track_frames,
            )
            current_control = self._ball_control(players, balls, team_by_track, ball_control)
            if current_control is not None:
                ball_control.append(current_control)

            annotated = frame.copy()
            self._draw_overlay(
                annotated,
                players=players,
                balls=balls,
                team_by_track=team_by_track,
                team_colors=team_colors,
                track_distance=track_distance,
                track_speed=track_speed,
                ball_control=ball_control,
                mode=mode,
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
            tracks.append(
                {
                    "track_id": track_id,
                    "team": team_by_track.get(track_id),
                    "frames": track_frames.get(track_id, 0),
                    "distance_m": round(track_distance.get(track_id, 0.0), 2),
                    "last_speed_kmh": round(track_speed.get(track_id, 0.0), 2),
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

        return {
            "status": "ok",
            "engine": "match_analysis_plus",
            "model": str(self.model_path),
            "model_mode": "sports-main-light-yolo",
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
            "id_stabilizer": track_stabilizer.summary(),
            "ball_filter": ball_filter.summary(),
            "tracks": tracks[:250],
            "team_ball_control": {
                "team_1_percent": round(team_1 * 100 / total_control, 2),
                "team_2_percent": round(team_2 * 100 / total_control, 2),
            },
            "notes": [
                "sports-main source is vendored in apps/match-analysis-worker/sports-main",
                "specialized sports-main models are optional; this run uses the local YOLO model by default",
            ],
            "elapsed_ms": elapsed_ms,
        }

    def _detect_and_track(self, model: Any, frame: np.ndarray, mode: str) -> list[AnalysisObject]:
        if mode == "BALL_DETECTION":
            classes = [32]
        elif mode == "PLAYER_DETECTION":
            classes = [0]
        else:
            classes = [0, 32]

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

    def _assign_teams(
        self,
        frame: np.ndarray,
        players: list[AnalysisObject],
        team_by_track: dict[int, int],
        team_colors: dict[int, tuple[int, int, int]],
    ) -> None:
        for player in players:
            if player.is_predicted and player.track_id not in team_by_track:
                continue
            if player.track_id in team_by_track:
                continue
            color = self._dominant_jersey_color(frame, player.bbox)
            if color is None:
                team_by_track[player.track_id] = 1
                continue
            if not team_colors:
                team_colors[1] = color
                team_by_track[player.track_id] = 1
                continue
            if len(team_colors) == 1:
                distance = np.linalg.norm(np.array(team_colors[1]) - np.array(color))
                if distance > 80:
                    team_colors[2] = color
                    team_by_track[player.track_id] = 2
                else:
                    team_by_track[player.track_id] = 1
                continue
            distances = {
                team: np.linalg.norm(np.array(team_color) - np.array(color))
                for team, team_color in team_colors.items()
            }
            team_by_track[player.track_id] = min(distances, key=distances.get)

    def _dominant_jersey_color(
        self,
        frame: np.ndarray,
        bbox: list[float],
    ) -> tuple[int, int, int] | None:
        height, width = frame.shape[:2]
        x1, y1, x2, y2 = [int(round(value)) for value in bbox]
        x1, x2 = max(0, x1), min(width, x2)
        y1, y2 = max(0, y1), min(height, y2)
        if x2 <= x1 or y2 <= y1:
            return None
        crop = frame[y1:y2, x1:x2]
        torso = crop[: max(1, crop.shape[0] // 2), :]
        if torso.size == 0:
            return None
        pixels = torso.reshape(-1, 3).astype(np.float32)
        if len(pixels) < 6:
            return None
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
        _, labels, centers = cv2.kmeans(pixels, 2, None, criteria, 3, cv2.KMEANS_PP_CENTERS)
        counts = np.bincount(labels.flatten())
        center = centers[int(np.argmax(counts))]
        return tuple(int(value) for value in center)

    def _update_movement(
        self,
        players: list[AnalysisObject],
        frame_index: int,
        fps: float,
        frame_width: int,
        last_positions: dict[int, tuple[float, float, int]],
        track_distance: dict[int, float],
        track_speed: dict[int, float],
        track_frames: dict[int, int],
    ) -> None:
        meters_per_pixel = 68.0 / max(frame_width, 1)
        for player in players:
            if player.is_predicted:
                continue
            foot = ((player.bbox[0] + player.bbox[2]) / 2, player.bbox[3])
            previous = last_positions.get(player.track_id)
            track_frames[player.track_id] = track_frames.get(player.track_id, 0) + 1
            if previous is not None:
                distance = float(np.hypot(foot[0] - previous[0], foot[1] - previous[1]) * meters_per_pixel)
                frame_delta = max(frame_index - previous[2], 1)
                if distance < 15:
                    track_distance[player.track_id] = track_distance.get(player.track_id, 0.0) + distance
                    track_speed[player.track_id] = distance / (frame_delta / fps) * 3.6
            last_positions[player.track_id] = (foot[0], foot[1], frame_index)

    def _ball_control(
        self,
        players: list[AnalysisObject],
        balls: list[AnalysisObject],
        team_by_track: dict[int, int],
        ball_control: list[int],
    ) -> int | None:
        players = [player for player in players if not player.is_predicted]
        if not players or not balls:
            return ball_control[-1] if ball_control else None
        ball = balls[0]
        ball_center = ((ball.bbox[0] + ball.bbox[2]) / 2, (ball.bbox[1] + ball.bbox[3]) / 2)
        nearest: tuple[float, AnalysisObject] | None = None
        for player in players:
            foot = ((player.bbox[0] + player.bbox[2]) / 2, player.bbox[3])
            distance = float(np.hypot(foot[0] - ball_center[0], foot[1] - ball_center[1]))
            if nearest is None or distance < nearest[0]:
                nearest = (distance, player)
        if nearest is None or nearest[0] > 90:
            return ball_control[-1] if ball_control else None
        return team_by_track.get(nearest[1].track_id, 1)

    def _draw_overlay(
        self,
        frame: np.ndarray,
        players: list[AnalysisObject],
        balls: list[AnalysisObject],
        team_by_track: dict[int, int],
        team_colors: dict[int, tuple[int, int, int]],
        track_distance: dict[int, float],
        track_speed: dict[int, float],
        ball_control: list[int],
        mode: str,
    ) -> None:
        for player in players:
            team = team_by_track.get(player.track_id, 1)
            color = team_colors.get(team, (80, 220, 80) if team == 2 else (240, 240, 240))
            self._draw_player(frame, player, color, track_distance, track_speed)
        for ball in balls:
            self._draw_triangle(frame, ball.bbox, (0, 255, 255))
        self._draw_header(frame, mode)
        self._draw_ball_control(frame, ball_control)

    def _draw_player(
        self,
        frame: np.ndarray,
        player: AnalysisObject,
        color: tuple[int, int, int],
        track_distance: dict[int, float],
        track_speed: dict[int, float],
    ) -> None:
        x1, y1, x2, y2 = [int(round(value)) for value in player.bbox]
        center_x = int((x1 + x2) / 2)
        cv2.ellipse(frame, (center_x, y2), (max(10, (x2 - x1) // 2), 8), 0, -45, 235, color, 2)
        scale = self._font_scale(frame, 0.5)
        small = self._font_scale(frame, 0.38)
        thickness = self._thickness(frame)
        cv2.rectangle(frame, (center_x - 14, y2 + 4), (center_x + 14, y2 + 21), (245, 250, 245), cv2.FILLED)
        cv2.putText(frame, str(player.track_id), (center_x - 9, y2 + 18), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thickness)
        distance = track_distance.get(player.track_id)
        speed = track_speed.get(player.track_id)
        if distance is not None:
            cv2.putText(frame, f"{distance:.1f}m", (center_x - 17, y2 + 36), cv2.FONT_HERSHEY_SIMPLEX, small, (0, 0, 0), thickness)
        if speed is not None:
            cv2.putText(frame, f"{speed:.1f}km/h", (center_x - 24, y2 + 50), cv2.FONT_HERSHEY_SIMPLEX, small, (0, 0, 0), thickness)

    def _draw_triangle(self, frame: np.ndarray, bbox: list[float], color: tuple[int, int, int]) -> None:
        x = int((bbox[0] + bbox[2]) / 2)
        y = int(bbox[1])
        points = np.array([[x, y], [x - 9, y - 18], [x + 9, y - 18]])
        cv2.drawContours(frame, [points], 0, color, cv2.FILLED)
        cv2.drawContours(frame, [points], 0, (0, 0, 0), 1)

    def _draw_header(self, frame: np.ndarray, mode: str) -> None:
        scale = self._font_scale(frame, 0.48)
        thickness = self._thickness(frame)
        cv2.putText(frame, f"Match Analysis +  {mode}", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), thickness + 2)
        cv2.putText(frame, f"Match Analysis +  {mode}", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thickness)

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

    def _load_model(self) -> Any:
        if self.model is not None:
            return self.model
        from ultralytics import YOLO

        self.model = YOLO(str(self.model_path))
        return self.model

    def _resolve_model_path(self) -> Path:
        configured = Path(settings.YOLO_MODEL_PATH)
        if configured.exists():
            return configured
        return Path("yolo11n.pt")

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
