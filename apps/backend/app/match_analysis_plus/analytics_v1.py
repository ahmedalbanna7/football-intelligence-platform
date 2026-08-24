from __future__ import annotations

from collections import defaultdict
import math
from typing import Any

import numpy as np


PITCH_LENGTH_CM = 10500.0
PITCH_WIDTH_CM = 6800.0
ANALYTICS_ROLES = {"player", "goalkeeper"}
MAX_PLAYER_SPEED_KMH = 45.0


class AnalyticsRealV1:
    """Build metric football analytics only from quality-gated pitch coordinates."""

    def build(
        self,
        layers: dict[str, Any],
        possession: dict[str, Any],
        pitch_gate: dict[str, Any],
        ball_gate: dict[str, Any],
        team_identity: dict[str, Any],
        team_context: dict[str, Any],
    ) -> dict[str, Any]:
        fps = float(layers.get("fps") or 25.0)
        metric_ready = bool(pitch_gate.get("metric_outputs_verified"))
        ball_ready = bool(ball_gate.get("possession_ready"))
        selected_teams = self._selected_teams(team_context)
        tracks = [
            track
            for track in layers.get("tracks", [])
            if int(track.get("team") or 0) in selected_teams
            and str(track.get("role_name") or "player") in ANALYTICS_ROLES
        ]

        pressure = self._pressure_metrics(tracks, fps) if metric_ready else {}
        players = [
            self._player_metrics(track, fps, pressure.get(int(track["track_id"]), {}), metric_ready)
            for track in tracks
        ]
        formations = self._formation_metrics(tracks, team_context) if metric_ready else {}
        space_control = self._space_control(tracks) if metric_ready else self._blocked("pitch_quality_gate")
        passing = self._passing_candidates(possession, selected_teams) if ball_ready else self._blocked("ball_quality_gate")
        teams = self._team_metrics(
            players,
            possession,
            formations,
            space_control,
            team_context,
            selected_teams,
            ball_ready,
        )

        failed_conditions: list[str] = []
        if not metric_ready:
            failed_conditions.append("pitch_quality_gate")
        if team_identity.get("quality_gate", {}).get("status") != "passed":
            failed_conditions.append("team_identity_gate")
        return {
            "schema_version": "analytics_real_v1",
            "engine": "metric_canonical_football_analytics_v1",
            "status": "ready" if not failed_conditions else "partial",
            "analysis_scope": team_context.get("analysis_scope") or "both_teams_full",
            "selected_teams": sorted(selected_teams),
            "quality_gate": {
                "status": "passed" if not failed_conditions else "partial",
                "failed_conditions": failed_conditions,
                "metric_outputs_released": metric_ready,
                "ball_outputs_released": ball_ready,
                "team_identity_status": team_identity.get("quality_gate", {}).get("status", "needs_review"),
            },
            "coordinate_system": "pitch_centimeters",
            "players": players,
            "teams": teams,
            "passing_candidates": passing,
            "pressure": {
                "status": "ready" if metric_ready else "blocked",
                "distance_threshold_m": 3.0,
                "tracks": pressure if metric_ready else {},
            },
            "formations": formations if metric_ready else self._blocked("pitch_quality_gate"),
            "space_control": space_control,
            "possession": possession if ball_ready else {
                "status": "blocked",
                "reason": "ball_quality_gate",
                "unverified_preview": possession,
            },
            "roster_linkage": {
                "status": "prepared",
                "teams": team_context.get("rosters") or {},
                "automatic_name_assignment": False,
                "next_identity_source": "manual_assignment_or_jersey_ocr",
            },
        }

    def _selected_teams(self, context: dict[str, Any]) -> set[int]:
        scope = str(context.get("analysis_scope") or "both_teams_full").lower()
        if scope in {"none", "no_analysis", "disabled"}:
            return set()
        if "both" in scope or scope in {"full", "all"}:
            selected = {1, 2}
        elif any(value in scope for value in ("opponent", "another", "second")):
            selected = {2}
        elif any(value in scope for value in ("my_team", "primary", "first")):
            selected = {1}
        else:
            selected = {1, 2}
        if not bool(context.get("analyze_primary_players", True)):
            selected.discard(1)
        if not bool(context.get("analyze_opponent_players", True)):
            selected.discard(2)
        return selected

    def _player_metrics(
        self,
        track: dict[str, Any],
        fps: float,
        pressure: dict[str, Any],
        metric_ready: bool,
    ) -> dict[str, Any]:
        path = track.get("pitch_path") or []
        distance_m = 0.0
        speeds: list[float] = []
        rejected_steps = 0
        if metric_ready:
            for previous, current in zip(path, path[1:]):
                frame_gap = max(1, int(current[0]) - int(previous[0]))
                distance = math.hypot(float(current[1]) - float(previous[1]), float(current[2]) - float(previous[2])) / 100.0
                speed = distance / (frame_gap / max(fps, 1e-6)) * 3.6
                if speed > MAX_PLAYER_SPEED_KMH or distance > 15.0:
                    rejected_steps += 1
                    continue
                if distance >= 0.02:
                    distance_m += distance
                    speeds.append(speed)
        return {
            "track_id": int(track["track_id"]),
            "team": int(track.get("team") or 0),
            "role_name": str(track.get("role_name") or "player"),
            "player_id": track.get("player_id"),
            "player_name": track.get("player_name"),
            "identity_confidence": track.get("identity_confidence"),
            "team_confidence": track.get("team_confidence"),
            "distance_m": round(distance_m, 2) if metric_ready else None,
            "average_speed_kmh": round(float(np.mean(speeds)), 2) if speeds else (0.0 if metric_ready else None),
            "max_speed_kmh": round(max(speeds), 2) if speeds else (0.0 if metric_ready else None),
            "metric_steps": len(speeds),
            "rejected_metric_steps": rejected_steps,
            "heatmap": self._heatmap(path) if metric_ready else [],
            "heatmap_samples": len(path) if metric_ready else 0,
            "pressure": pressure if metric_ready else self._blocked("pitch_quality_gate"),
        }

    def _heatmap(self, path: list[list[int]], columns: int = 16, rows: int = 10) -> dict[str, Any]:
        grid = [[0 for _ in range(columns)] for _ in range(rows)]
        for _, x_cm, y_cm in path:
            column = min(columns - 1, max(0, int(float(x_cm) / PITCH_LENGTH_CM * columns)))
            row = min(rows - 1, max(0, int(float(y_cm) / PITCH_WIDTH_CM * rows)))
            grid[row][column] += 1
        maximum = max((max(row) for row in grid), default=0)
        return {
            "columns": columns,
            "rows": rows,
            "samples": len(path),
            "max_bin": maximum,
            "grid": grid,
        }

    def _frame_positions(self, tracks: list[dict[str, Any]]) -> dict[int, list[tuple[int, int, float, float]]]:
        positions: dict[int, list[tuple[int, int, float, float]]] = defaultdict(list)
        for track in tracks:
            track_id = int(track["track_id"])
            team = int(track.get("team") or 0)
            for frame, x_cm, y_cm in track.get("pitch_path") or []:
                positions[int(frame)].append((track_id, team, float(x_cm), float(y_cm)))
        return positions

    def _pressure_metrics(self, tracks: list[dict[str, Any]], fps: float) -> dict[int, dict[str, Any]]:
        samples: dict[int, list[float]] = defaultdict(list)
        under_pressure: dict[int, int] = defaultdict(int)
        for frame_players in self._frame_positions(tracks).values():
            for track_id, team, x_cm, y_cm in frame_players:
                opponents = [item for item in frame_players if item[1] in {1, 2} and item[1] != team]
                if not opponents:
                    continue
                nearest_m = min(math.hypot(x_cm - item[2], y_cm - item[3]) / 100.0 for item in opponents)
                samples[track_id].append(nearest_m)
                if nearest_m <= 3.0:
                    under_pressure[track_id] += 1
        result: dict[int, dict[str, Any]] = {}
        for track_id, distances in samples.items():
            result[track_id] = {
                "samples": len(distances),
                "under_pressure_samples": under_pressure[track_id],
                "under_pressure_seconds": round(under_pressure[track_id] / max(6.0, min(fps, 6.0)), 2),
                "average_nearest_opponent_m": round(float(np.mean(distances)), 2),
                "minimum_nearest_opponent_m": round(min(distances), 2),
                "pressure_index": round(float(np.mean([max(0.0, 1.0 - distance / 5.0) for distance in distances])), 4),
            }
        return result

    def _formation_metrics(self, tracks: list[dict[str, Any]], context: dict[str, Any]) -> dict[str, Any]:
        by_team: dict[int, list[dict[str, float]]] = defaultdict(list)
        for frame_players in self._frame_positions(tracks).values():
            for team in (1, 2):
                points = np.array([[item[2], item[3]] for item in frame_players if item[1] == team], dtype=np.float32)
                if len(points) < 3:
                    continue
                by_team[team].append({
                    "centroid_x_cm": float(np.mean(points[:, 0])),
                    "centroid_y_cm": float(np.mean(points[:, 1])),
                    "team_length_m": float((np.max(points[:, 0]) - np.min(points[:, 0])) / 100.0),
                    "team_width_m": float((np.max(points[:, 1]) - np.min(points[:, 1])) / 100.0),
                })
        result: dict[str, Any] = {}
        configured = context.get("formations") or {}
        for team, snapshots in by_team.items():
            result[str(team)] = {
                "team": team,
                "configured_formation": configured.get(str(team)),
                "snapshots": len(snapshots),
                "average_centroid_cm": {
                    "x": round(float(np.mean([item["centroid_x_cm"] for item in snapshots])), 2),
                    "y": round(float(np.mean([item["centroid_y_cm"] for item in snapshots])), 2),
                },
                "average_team_length_m": round(float(np.mean([item["team_length_m"] for item in snapshots])), 2),
                "average_team_width_m": round(float(np.mean([item["team_width_m"] for item in snapshots])), 2),
            }
        return {"status": "ready", "teams": result}

    def _space_control(self, tracks: list[dict[str, Any]]) -> dict[str, Any]:
        frames = self._frame_positions(tracks)
        frame_ids = sorted(frames)
        if len(frame_ids) > 600:
            stride = max(1, len(frame_ids) // 600)
            frame_ids = frame_ids[::stride]
        grid_x, grid_y = np.meshgrid(
            np.linspace(0, PITCH_LENGTH_CM, 16),
            np.linspace(0, PITCH_WIDTH_CM, 10),
        )
        grid = np.column_stack((grid_x.ravel(), grid_y.ravel()))
        controlled = {1: 0, 2: 0}
        valid_frames = 0
        for frame in frame_ids:
            players = frames[frame]
            team_points = {
                team: np.array([[item[2], item[3]] for item in players if item[1] == team], dtype=np.float32)
                for team in (1, 2)
            }
            if min(len(team_points[1]), len(team_points[2])) < 3:
                continue
            distances = {
                team: np.min(np.linalg.norm(grid[:, None, :] - points[None, :, :], axis=2), axis=1)
                for team, points in team_points.items()
            }
            controlled[1] += int(np.sum(distances[1] <= distances[2]))
            controlled[2] += int(np.sum(distances[2] < distances[1]))
            valid_frames += 1
        total = max(controlled[1] + controlled[2], 1)
        return {
            "status": "ready" if valid_frames else "insufficient_players",
            "engine": "nearest_player_pitch_grid_v1",
            "grid": [16, 10],
            "sampled_frames": valid_frames,
            "team_1_percent": round(controlled[1] * 100.0 / total, 2),
            "team_2_percent": round(controlled[2] * 100.0 / total, 2),
        }

    def _passing_candidates(self, possession: dict[str, Any], selected_teams: set[int]) -> dict[str, Any]:
        events = [
            event
            for event in possession.get("events", [])
            if int(event.get("to_team") or 0) in selected_teams
            and event.get("type") in {"completed_pass", "turnover", "possession_change"}
        ]
        return {
            "status": "ready",
            "events": events,
            "completed_passes": sum(1 for event in events if event.get("type") == "completed_pass"),
            "turnovers": sum(1 for event in events if event.get("type") == "turnover"),
        }

    def _team_metrics(
        self,
        players: list[dict[str, Any]],
        possession: dict[str, Any],
        formations: dict[str, Any],
        space_control: dict[str, Any],
        context: dict[str, Any],
        selected_teams: set[int],
        ball_ready: bool,
    ) -> dict[str, Any]:
        labels = context.get("team_labels") or {}
        result: dict[str, Any] = {}
        for team in sorted(selected_teams):
            team_players = [player for player in players if player["team"] == team]
            possession_value = possession.get(f"team_{team}_percent") if ball_ready else None
            result[str(team)] = {
                "team": team,
                "name": labels.get(str(team), f"Team {team}"),
                "players_count": len(team_players),
                "total_distance_m": round(sum(float(player.get("distance_m") or 0.0) for player in team_players), 2),
                "average_speed_kmh": round(float(np.mean([player.get("average_speed_kmh") or 0.0 for player in team_players])), 2) if team_players else 0.0,
                "possession_percent": possession_value,
                "formation": (formations.get("teams") or {}).get(str(team)),
                "space_control_percent": space_control.get(f"team_{team}_percent") if space_control.get("status") == "ready" else None,
            }
        return result

    def _blocked(self, reason: str) -> dict[str, Any]:
        return {"status": "blocked", "reason": reason}
