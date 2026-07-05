from typing import Any

from app.ai.tactical_identity_layer.identity_confidence import weighted_average
from app.ai.tactical_identity_layer.lineup_context import LineupContext
from app.ai.tactical_identity_layer.schemas import IdentityCandidate
from app.ai.tactical_identity_layer.schemas import TrackIdentityResolution


class IdentityResolver:
    score_weights = {
        "team_match": 0.20,
        "zone_match": 0.25,
        "active_player": 0.20,
        "jersey_number": 0.25,
        "position_continuity": 0.10,
    }

    def resolve(
        self,
        tracks: list[dict[str, Any]],
        match_context: dict[str, Any],
    ) -> list[TrackIdentityResolution]:
        lineup_context = LineupContext(match_context)
        active_players = lineup_context.active_players_at(minute=0)

        return [
            self._resolve_track(track, active_players)
            for track in tracks
            if track.get("class_name") == "player"
        ]

    def _resolve_track(
        self,
        track: dict[str, Any],
        active_players: list[dict[str, Any]],
    ) -> TrackIdentityResolution:
        candidates = [
            self._score_candidate(track, player)
            for player in active_players
        ]
        candidates = sorted(
            candidates,
            key=lambda candidate: candidate.score,
            reverse=True,
        )

        best = candidates[0] if candidates else None
        confidence = best.score if best is not None else None
        resolved_player_id = (
            best.player_id
            if best is not None and best.score >= 0.45
            else None
        )
        resolved_player = (
            {
                "player_id": best.player_id,
                "name": best.name,
                "jersey_number": best.jersey_number,
                "zone": best.zone,
            }
            if best is not None and resolved_player_id is not None
            else None
        )

        return TrackIdentityResolution(
            track_id=int(track.get("track_id")),
            team_context=track.get("team_context"),
            team_id=best.team_id if best is not None else None,
            resolved_player_id=resolved_player_id,
            resolved_player=resolved_player,
            confidence=confidence,
            zone=best.zone if best is not None else None,
            candidates=candidates[:5],
            track_snapshot={
                "recognized_shirt_number": track.get("recognized_shirt_number"),
                "shirt_number_confidence": track.get("shirt_number_confidence"),
                "team_assignment_confidence": track.get("team_assignment_confidence"),
                "team_context": track.get("team_context"),
                "team_label": track.get("team_label"),
                "team_confidence": track.get("team_confidence"),
                "kit_match_score": track.get("kit_match_score"),
                "observation_count": len(track.get("frames", [])),
                "crop_samples": track.get("crop_samples", [])[:3],
            },
        )

    def _score_candidate(
        self,
        track: dict[str, Any],
        player: dict[str, Any],
    ) -> IdentityCandidate:
        expected_zones = player.get("expected_zones") or []
        starting_zone = player.get("starting_zone")
        recognized_number = track.get("recognized_shirt_number")
        shirt_number = player.get("jersey_number")

        reasons = {
            "team_match": self._team_match_score(track, player),
            "zone_match": self._zone_match_score(starting_zone, expected_zones),
            "active_player": 1.0,
            "jersey_number": self._jersey_number_score(
                recognized_number,
                shirt_number,
                track.get("shirt_number_confidence"),
            ),
            "position_continuity": 0.5,
        }

        return IdentityCandidate(
            player_id=int(player.get("player_id")),
            team_id=player.get("team_id"),
            name=player.get("player_name"),
            jersey_number=shirt_number,
            zone=starting_zone,
            score=weighted_average(self.score_weights, reasons),
            reasons=reasons,
        )

    def _team_match_score(
        self,
        track: dict[str, Any],
        player: dict[str, Any],
    ) -> float:
        team_context = track.get("team_context")
        team_confidence = (
            track.get("team_assignment_confidence")
            or track.get("team_confidence")
        )
        player_team_context = player.get("team_context")
        if player_team_context is None:
            return team_confidence or 0.65
        if team_context == player_team_context:
            return team_confidence or 0.9
        return 0.2

    def _zone_match_score(
        self,
        starting_zone: str | None,
        expected_zones: list[str],
    ) -> float:
        if starting_zone is None:
            return 0.35
        if not expected_zones:
            return 0.6
        if starting_zone in expected_zones:
            return 1.0
        return 0.45

    def _jersey_number_score(
        self,
        recognized_number: int | None,
        shirt_number: int | None,
        confidence: float | None,
    ) -> float | None:
        if recognized_number is None or shirt_number is None:
            return None
        if recognized_number == shirt_number:
            return confidence or 0.85
        return 0.05
