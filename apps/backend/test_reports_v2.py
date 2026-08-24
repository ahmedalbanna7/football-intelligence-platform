import unittest

import cv2
import numpy as np

from app.match_analysis_plus.reports_v2 import ReportsV2Builder


class ReportsV2Tests(unittest.TestCase):
    def setUp(self) -> None:
        self.builder = ReportsV2Builder()
        self.analytics = {
            "analysis_scope": "both_teams_full",
            "quality_gate": {"status": "passed"},
            "teams": {
                "1": {"team": 1, "name": "Team 1", "players_count": 1, "total_distance_m": 120.0, "average_speed_kmh": 6.2, "possession_percent": 55.0, "space_control_percent": 52.0},
                "2": {"team": 2, "name": "Team 2", "players_count": 1, "total_distance_m": 110.0, "average_speed_kmh": 5.8, "possession_percent": 45.0, "space_control_percent": 48.0},
            },
            "players": [{"track_id": 1, "team": 1, "distance_m": 120.0, "average_speed_kmh": 6.2, "max_speed_kmh": 19.0, "heatmap": {"samples": 2, "grid": [[0, 1], [1, 0]]}}],
            "possession": {},
            "passing_candidates": {},
        }
        self.summary = {"frames_processed": 250, "fps": 25, "source_start_frame": 0, "source_end_frame": 249}

    def test_builds_json_png_and_pdf_artifacts(self) -> None:
        report = self.builder.build(self.analytics, self.summary, {})
        chart = self.builder.team_chart_png(report)
        heatmaps = self.builder.heatmap_atlas_png(report)
        pdf = self.builder.pdf(report)
        self.assertEqual(report["schema_version"], "reports_v2")
        self.assertIsNotNone(cv2.imdecode(np.frombuffer(chart, np.uint8), cv2.IMREAD_COLOR))
        self.assertIsNotNone(cv2.imdecode(np.frombuffer(heatmaps, np.uint8), cv2.IMREAD_COLOR))
        self.assertTrue(pdf.startswith(b"%PDF-1.4"))
        self.assertIn(b"%%EOF", pdf)

    def test_compare_keeps_quality_context(self) -> None:
        report = self.builder.build(self.analytics, self.summary, {})
        compared = self.builder.compare([(1, 10, report), (2, 20, report)])
        self.assertEqual(compared["runs_count"], 2)
        self.assertEqual(compared["runs"][0]["run_id"], 10)


if __name__ == "__main__":
    unittest.main()
