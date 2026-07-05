from typing import Any


class AnalyticsEngine:
    def analyze(
        self,
        tracks: list[dict[str, Any]],
        match_context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        context = match_context or {}
        players = [
            self._analyze_track(track, context)
            for track in tracks
            if track.get("class_name") == "player"
        ]
        ball_tracks = [
            self._analyze_track(track, context)
            for track in tracks
            if track.get("class_name") == "ball"
        ]

        return {
            "status": "ok",
            "data": {
                "players": players,
                "players_by_team": {
                    "primary_team": [
                        player
                        for player in players
                        if player.get("team_context") == "primary_team"
                    ],
                    "opponent_team": [
                        player
                        for player in players
                        if player.get("team_context") == "opponent_team"
                    ],
                    "club_internal": [
                        player
                        for player in players
                        if player.get("team_context") == "club_internal"
                    ],
                },
                "ball": ball_tracks[0] if ball_tracks else None,
                "team_aggregates": self._team_aggregates(players),
                "team": {
                    "players_detected": len(players),
                    "tracks_received": len(tracks),
                    "total_distance": round(sum(player.get("distance", 0) for player in players), 2),
                    "average_speed": self._average(
                        [player.get("average_speed", 0) for player in players]
                    ),
                },
                "tracks_received": len(tracks),
            },
            "meta": {
                "engine": "analytics_real_v1_pixels",
                "metrics": [
                    "position",
                    "distance",
                    "speed",
                    "max_speed",
                    "acceleration",
                    "heatmap",
                    "movement_summary",
                ],
                "match_context": context,
                "units": {
                    "distance": "pixels",
                    "speed": "pixels_per_second",
                },
            },
        }

    def _analyze_track(
        self,
        track: dict[str, Any],
        context: dict[str, Any],
    ) -> dict[str, Any]:
        frames = track.get("frames", [])
        centers = [
            frame.get("center") or frame.get("bbox_center")
            for frame in frames
            if (frame.get("center") or frame.get("bbox_center")) is not None
        ]
        timestamps = [
            frame.get("timestamp_seconds")
            for frame in frames
            if frame.get("timestamp_seconds") is not None
        ]

        distance = 0.0
        segment_speeds = []
        for index, (previous, current) in enumerate(zip(centers, centers[1:])):
            dx = current[0] - previous[0]
            dy = current[1] - previous[1]
            segment_distance = (dx**2 + dy**2) ** 0.5
            distance += segment_distance
            if len(timestamps) > index + 1:
                dt = timestamps[index + 1] - timestamps[index]
                if dt > 0:
                    segment_speeds.append(segment_distance / dt)

        duration = timestamps[-1] - timestamps[0] if len(timestamps) >= 2 else 0
        average_speed = distance / duration if duration > 0 else 0
        accelerations = [
            (current - previous) / (timestamps[index + 2] - timestamps[index + 1])
            for index, (previous, current) in enumerate(zip(segment_speeds, segment_speeds[1:]))
            if len(timestamps) > index + 2 and timestamps[index + 2] > timestamps[index + 1]
        ]

        return {
            "track_id": track.get("track_id"),
            "object_key": track.get("object_key"),
            "class_name": track.get("class_name"),
            "team_context": self._infer_team_context(track, context),
            "player_name": track.get("resolved_player", {}).get("name") if isinstance(track.get("resolved_player"), dict) else None,
            "recognized_shirt_number": track.get("recognized_shirt_number"),
            "positions_count": len(centers),
            "first_position": centers[0] if centers else None,
            "last_position": centers[-1] if centers else None,
            "distance": round(distance, 2),
            "average_speed": round(average_speed, 2),
            "max_speed": round(max(segment_speeds), 2) if segment_speeds else 0,
            "average_acceleration": round(sum(accelerations) / len(accelerations), 2) if accelerations else 0,
            "heatmap": self._heatmap(centers),
            "movement_summary": self._movement_summary(centers, distance, average_speed),
        }

    def _heatmap(self, centers: list[list[float]], bins_x: int = 6, bins_y: int = 4) -> list[dict[str, Any]]:
        if not centers:
            return []
        max_x = max(center[0] for center in centers) or 1
        max_y = max(center[1] for center in centers) or 1
        buckets: dict[tuple[int, int], int] = {}
        for x, y in centers:
            bx = min(bins_x - 1, max(0, int((x / max_x) * bins_x)))
            by = min(bins_y - 1, max(0, int((y / max_y) * bins_y)))
            buckets[(bx, by)] = buckets.get((bx, by), 0) + 1
        total = len(centers)
        return [
            {
                "x_bin": x,
                "y_bin": y,
                "count": count,
                "share": round(count / total, 4),
            }
            for (x, y), count in sorted(buckets.items())
        ]

    def _movement_summary(
        self,
        centers: list[list[float]],
        distance: float,
        average_speed: float,
    ) -> dict[str, Any]:
        if len(centers) < 2:
            return {
                "trend": "insufficient_data",
                "distance_level": "none",
                "speed_level": "none",
            }
        dx = centers[-1][0] - centers[0][0]
        dy = centers[-1][1] - centers[0][1]
        horizontal = "right" if dx > 20 else "left" if dx < -20 else "stable"
        vertical = "down" if dy > 20 else "up" if dy < -20 else "stable"
        return {
            "trend": f"{horizontal}_{vertical}",
            "distance_level": "high" if distance > 900 else "medium" if distance > 300 else "low",
            "speed_level": "high" if average_speed > 90 else "medium" if average_speed > 30 else "low",
        }

    def _team_aggregates(self, players: list[dict[str, Any]]) -> dict[str, Any]:
        aggregates: dict[str, Any] = {}
        for team_context in {"primary_team", "opponent_team", "club_internal", "unknown"}:
            team_players = [
                player for player in players
                if player.get("team_context") == team_context
            ]
            if not team_players:
                continue
            aggregates[team_context] = {
                "players_count": len(team_players),
                "total_distance": round(sum(player.get("distance", 0) for player in team_players), 2),
                "average_speed": self._average([player.get("average_speed", 0) for player in team_players]),
                "max_speed": max(player.get("max_speed", 0) for player in team_players),
            }
        return aggregates

    def _average(self, values: list[float]) -> float:
        values = [value for value in values if value is not None]
        return round(sum(values) / len(values), 2) if values else 0

    def _infer_team_context(
        self,
        track: dict[str, Any],
        context: dict[str, Any],
    ) -> str:
        if track.get("team_context"):
            return track["team_context"]
        if context.get("match_type") in {"internal_scrimmage", "academy_match"}:
            return "club_internal"
        object_key = track.get("object_key") or ""
        if object_key.endswith("left"):
            return "primary_team"
        if object_key.endswith("right"):
            return "opponent_team"
        return "unknown"
