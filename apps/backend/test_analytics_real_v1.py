import unittest

from app.match_analysis_plus.analytics_v1 import AnalyticsRealV1


class AnalyticsRealV1Tests(unittest.TestCase):
    def setUp(self) -> None:
        self.engine = AnalyticsRealV1()
        self.layers = {
            "fps": 25,
            "tracks": [
                {
                    "track_id": 1,
                    "team": 1,
                    "role_name": "player",
                    "pitch_path": [[0, 1000, 1000], [25, 1100, 1000], [50, 1200, 1000]],
                },
                {
                    "track_id": 2,
                    "team": 2,
                    "role_name": "player",
                    "pitch_path": [[0, 1250, 1000], [25, 1350, 1000], [50, 1450, 1000]],
                },
                {
                    "track_id": 3,
                    "team": 0,
                    "role_name": "referee",
                    "pitch_path": [[0, 1150, 1000], [25, 1200, 1000]],
                },
            ],
        }

    def test_metric_outputs_are_blocked_when_pitch_gate_fails(self) -> None:
        result = self.engine.build(
            self.layers,
            {},
            {"metric_outputs_verified": False},
            {"possession_ready": False},
            {"quality_gate": {"status": "passed"}},
            {"analysis_scope": "both_teams_full"},
        )
        self.assertIsNone(result["players"][0]["distance_m"])
        self.assertEqual(result["space_control"]["status"], "blocked")

    def test_scope_and_roles_filter_player_analytics(self) -> None:
        result = self.engine.build(
            self.layers,
            {"team_1_percent": 55.0, "team_2_percent": 45.0, "events": []},
            {"metric_outputs_verified": True},
            {"possession_ready": True},
            {"quality_gate": {"status": "passed"}},
            {
                "analysis_scope": "my_team",
                "analyze_primary_players": True,
                "analyze_opponent_players": True,
                "team_labels": {"1": "Primary"},
            },
        )
        self.assertEqual([player["track_id"] for player in result["players"]], [1])
        self.assertAlmostEqual(result["players"][0]["distance_m"], 2.0)
        self.assertEqual(result["players"][0]["heatmap"]["samples"], 3)
        self.assertEqual(result["teams"]["1"]["possession_percent"], 55.0)


if __name__ == "__main__":
    unittest.main()
