from typing import Any


class ReportEngine:
    def build_json_report(self, context: dict[str, Any]) -> dict[str, Any]:
        match_context = context.get("match_context", {})
        frames = context.get("frames", {})
        detections = context.get("detections", {})
        tracks = context.get("tracks", {})
        team_assignment = context.get("team_assignment", {})
        crops = context.get("crops", {})
        tactical_identity = context.get("tactical_identity", {})
        events = context.get("events", {})
        analytics = context.get("analytics", {})
        artifacts = context.get("artifacts", {})
        llm = context.get("llm", {})

        video_meta = frames.get("meta", {})
        detections_data = detections.get("data", {})
        tracks_data = tracks.get("data", {})
        team_assignment_data = team_assignment.get("data", {})
        crops_data = crops.get("data", {})
        tactical_identity_data = tactical_identity.get("data", {})
        events_data = events.get("data", {})
        analytics_data = analytics.get("data", {})
        artifacts_data = artifacts.get("data", {})
        llm_data = llm.get("data", {})
        targets = match_context.get("analysis_targets", {})

        primary_players = analytics_data.get("players_by_team", {}).get(
            "primary_team",
            [],
        )
        opponent_players = analytics_data.get("players_by_team", {}).get(
            "opponent_team",
            [],
        )
        if not targets.get("include_primary_players", True):
            primary_players = []
        if not targets.get("include_opponent_players", True):
            opponent_players = []

        return {
            "status": "ok",
            "data": {
                "report_type": "json",
                "summary": {
                    "coach_summary": llm_data.get("coach_summary"),
                    "match_type": match_context.get("match_type"),
                    "analysis_scope": match_context.get("analysis_scope"),
                    "primary_team_name": match_context.get("primary_team_name"),
                    "opponent_team_name": match_context.get("opponent_team_name"),
                },
                "video": {
                    "fps": video_meta.get("fps"),
                    "duration_seconds": video_meta.get("duration_seconds"),
                    "frame_count": video_meta.get("frame_count"),
                    "resolution": video_meta.get("resolution"),
                    "sampled_frames_count": video_meta.get("sampled_frames_count"),
                },
                "counts": {
                    "detections": detections_data.get("detections_count", 0),
                    "tracks": tracks_data.get("tracks_count", 0),
                    "team_assigned_tracks": team_assignment_data.get("tracks_count", 0),
                    "identity_assignments": tactical_identity_data.get(
                        "assignments_count",
                        0,
                    ),
                    "identity_resolved": tactical_identity_data.get(
                        "resolved_count",
                        0,
                    ),
                    "events": events_data.get("events_count", 0),
                    "players_detected": analytics_data.get("team", {}).get(
                        "players_detected",
                        0,
                    ),
                    "track_observations": artifacts_data.get(
                        "track_observations_count",
                        0,
                    ),
                    "player_crops": crops_data.get("crops_count", 0),
                    "jersey_crops": crops_data.get("jersey_crops_count", 0),
                },
                "yolo_tracking": {
                    "detector": detections.get("meta", {}).get("engine"),
                    "model": detections.get("meta", {}).get("model"),
                    "person_detections_count": detections_data.get(
                        "raw_class_counts",
                        {},
                    ).get("person", 0),
                    "ball_detections_count": detections_data.get(
                        "raw_class_counts",
                        {},
                    ).get("sports ball", 0),
                    "tracks_count": tracks_data.get("tracks_count", 0),
                    "track_observations_count": artifacts_data.get(
                        "track_observations_count",
                        0,
                    ),
                    "artifacts": artifacts_data.get("artifacts", {}),
                    "crops": {
                        "status": crops.get("status"),
                        "crops_count": crops_data.get("crops_count", 0),
                        "jersey_crops_count": crops_data.get(
                            "jersey_crops_count",
                            0,
                        ),
                        "crops_prefix": crops_data.get("crops_prefix"),
                    },
                    "sample_detections_count": len(
                        artifacts_data.get("sample_detections", [])
                    ),
                    "sample_track_observations_count": len(
                        artifacts_data.get("sample_track_observations", [])
                    ),
                    "processing_warnings": self._processing_warnings(
                        detections_data,
                        tracks_data,
                        tactical_identity_data,
                    ),
                },
                "teams": {
                    "kit_reference_analysis": team_assignment_data.get(
                        "kit_reference_analysis",
                        {},
                    ),
                    "primary_team": primary_players,
                    "opponent_team": opponent_players,
                    "club_internal": analytics_data.get("players_by_team", {}).get(
                        "club_internal",
                        [],
                    ),
                    "player_analysis_targets": {
                        "include_primary_players": targets.get(
                            "include_primary_players",
                            True,
                        ),
                        "include_opponent_players": targets.get(
                            "include_opponent_players",
                            True,
                        ),
                    },
                },
                "charts": {
                    "team_distance": [
                        {
                            "team_context": team_context,
                            "total_distance": values.get("total_distance", 0),
                            "players_count": values.get("players_count", 0),
                        }
                        for team_context, values in analytics_data.get(
                            "team_aggregates",
                            {},
                        ).items()
                    ],
                    "player_speed": [
                        {
                            "track_id": player.get("track_id"),
                            "team_context": player.get("team_context"),
                            "shirt_number": player.get("recognized_shirt_number"),
                            "average_speed": player.get("average_speed"),
                            "max_speed": player.get("max_speed"),
                        }
                        for player in analytics_data.get("players", [])
                    ],
                },
                "heatmaps": {
                    "players": [
                        {
                            "track_id": player.get("track_id"),
                            "team_context": player.get("team_context"),
                            "heatmap": player.get("heatmap", []),
                        }
                        for player in analytics_data.get("players", [])
                    ],
                },
                "events": events_data.get("events", []),
                "identity": {
                    "assignments": tactical_identity_data.get("assignments", []),
                    "engine": tactical_identity.get("meta", {}).get("engine"),
                },
                "coach": llm_data,
                "debug_sections_available": [
                    key
                    for key in context.keys()
                ],
            },
            "meta": {
                "engine": "report_v2_compact",
                "future_outputs": ["pdf", "charts", "heatmaps", "highlights"],
            },
        }

    def _processing_warnings(
        self,
        detections_data: dict[str, Any],
        tracks_data: dict[str, Any],
        tactical_identity_data: dict[str, Any],
    ) -> list[str]:
        warnings: list[str] = []
        raw_counts = detections_data.get("raw_class_counts", {})
        if raw_counts.get("sports ball", 0) == 0:
            warnings.append("sports_ball_not_detected")
        if tracks_data.get("tracks_count", 0) > 80:
            warnings.append("tracking_is_tracklet_level_not_final_player_identity")
        if tactical_identity_data.get("resolved_count", 0) == 0:
            warnings.append("no_players_resolved_add_lineup_or_identity_inputs")
        return warnings
