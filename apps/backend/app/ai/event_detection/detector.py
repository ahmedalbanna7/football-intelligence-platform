from typing import Any


class EventDetectionEngine:
    def detect(
        self,
        tracks: list[dict[str, Any]],
        match_context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        context = match_context or {}
        events: list[dict[str, Any]] = []
        players = [
            track
            for track in tracks
            if track.get("class_name") == "player"
        ]
        balls = [
            track
            for track in tracks
            if track.get("class_name") == "ball"
        ]

        for player in players:
            events.append(self._player_seen_event(player, context))

        for ball in balls:
            events.append(self._ball_seen_event(ball, context))

        if players and balls:
            events.append(self._possession_candidate_event(players, balls[0], context))

        if len(players) >= 2 and balls:
            events.append(self._possible_pass_event(players, balls[0], context))

        if tracks:
            events.append(self._movement_summary_event(tracks, context))

        return {
            "status": "ok",
            "data": {
                "events": events,
                "events_count": len(events),
                "tracks_received": len(tracks),
            },
            "meta": {
                "engine": "event_detection_stub_v2",
                "match_context": context,
                "event_types": [
                    "ball_seen",
                    "player_seen",
                    "possible_pass",
                    "possession_candidate",
                    "movement_summary",
                ],
            },
        }

    def _player_seen_event(
        self,
        track: dict[str, Any],
        context: dict[str, Any],
    ) -> dict[str, Any]:
        frames = track.get("frames", [])
        return {
            "type": "player_seen",
            "track_id": track.get("track_id"),
            "object_key": track.get("object_key"),
            "team_context": self._infer_team_context(track, context),
            "first_frame": frames[0].get("frame_index") if frames else None,
            "last_frame": frames[-1].get("frame_index") if frames else None,
            "confidence": self._average_confidence(frames),
        }

    def _ball_seen_event(
        self,
        track: dict[str, Any],
        context: dict[str, Any],
    ) -> dict[str, Any]:
        frames = track.get("frames", [])
        return {
            "type": "ball_seen",
            "track_id": track.get("track_id"),
            "object_key": track.get("object_key"),
            "first_frame": frames[0].get("frame_index") if frames else None,
            "last_frame": frames[-1].get("frame_index") if frames else None,
            "confidence": self._average_confidence(frames),
            "analysis_scope": context.get("analysis_scope"),
        }

    def _possession_candidate_event(
        self,
        players: list[dict[str, Any]],
        ball: dict[str, Any],
        context: dict[str, Any],
    ) -> dict[str, Any]:
        ball_position = self._last_center(ball)
        nearest_player = min(
            players,
            key=lambda player: self._distance(
                self._last_center(player),
                ball_position,
            ),
        )
        return {
            "type": "possession_candidate",
            "player_track_id": nearest_player.get("track_id"),
            "player_object_key": nearest_player.get("object_key"),
            "team_context": self._infer_team_context(nearest_player, context),
            "ball_track_id": ball.get("track_id"),
            "distance_to_ball": round(
                self._distance(self._last_center(nearest_player), ball_position),
                2,
            ),
        }

    def _possible_pass_event(
        self,
        players: list[dict[str, Any]],
        ball: dict[str, Any],
        context: dict[str, Any],
    ) -> dict[str, Any]:
        ordered_players = sorted(players, key=lambda player: player.get("track_id") or 0)
        source = ordered_players[0]
        target = ordered_players[1]
        return {
            "type": "possible_pass",
            "source_track_id": source.get("track_id"),
            "target_track_id": target.get("track_id"),
            "ball_track_id": ball.get("track_id"),
            "source_team_context": self._infer_team_context(source, context),
            "target_team_context": self._infer_team_context(target, context),
            "confidence": 0.42,
            "note": "Stub pass candidate based on available player and ball tracks.",
        }

    def _movement_summary_event(
        self,
        tracks: list[dict[str, Any]],
        context: dict[str, Any],
    ) -> dict[str, Any]:
        moving_tracks = [
            track
            for track in tracks
            if len(track.get("frames", [])) >= 2
        ]
        return {
            "type": "movement_summary",
            "tracks_count": len(tracks),
            "moving_tracks_count": len(moving_tracks),
            "analysis_scope": context.get("analysis_scope"),
            "match_type": context.get("match_type"),
        }

    def _infer_team_context(
        self,
        track: dict[str, Any],
        context: dict[str, Any],
    ) -> str:
        if track.get("team_context"):
            return track["team_context"]
        object_key = track.get("object_key") or ""
        if context.get("match_type") in {"internal_scrimmage", "academy_match"}:
            return "club_internal"
        if object_key.endswith("left"):
            return "primary_team"
        if object_key.endswith("right"):
            return "opponent_team"
        return "unknown"

    def _average_confidence(self, frames: list[dict[str, Any]]) -> float | None:
        values = [
            frame.get("confidence")
            for frame in frames
            if frame.get("confidence") is not None
        ]
        if not values:
            return None
        return round(sum(values) / len(values), 4)

    def _last_center(self, track: dict[str, Any]) -> list[float]:
        frames = track.get("frames", [])
        if not frames:
            return [0, 0]
        return frames[-1].get("center") or [0, 0]

    def _distance(
        self,
        first: list[float],
        second: list[float],
    ) -> float:
        dx = first[0] - second[0]
        dy = first[1] - second[1]
        return (dx**2 + dy**2) ** 0.5
