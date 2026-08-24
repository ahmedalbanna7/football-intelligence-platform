import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from app.match_analysis_plus.runner import (
    AnalysisObject,
    MatchAnalysisPlusRunner,
    StableTrackState,
    TeamColorClassifier,
)


class TeamIdentityV3Tests(unittest.TestCase):
    def test_shadow_change_preserves_colored_kit_distance(self) -> None:
        classifier = TeamColorClassifier()
        same_hue_shadow = classifier._color_distance((30, 80, 220), (12, 36, 95))
        different_hue = classifier._color_distance((30, 80, 220), (220, 80, 30))
        self.assertLess(same_hue_shadow, different_hue)

    def test_goalkeeper_uses_separate_reference(self) -> None:
        classifier = TeamColorClassifier(
            reference_palettes_bgr={1: [(20, 30, 220)], 2: [(220, 40, 20)]},
            goalkeeper_reference_palettes_bgr={1: [(30, 220, 220)], 2: [(220, 30, 220)]},
        )
        state = StableTrackState(
            stable_id=9,
            bbox=[0, 0, 20, 60],
            center=(10, 30),
            foot=(10, 60),
            velocity=(0, 0),
            foot_velocity=(0, 0),
            last_frame=1,
            raw_ids_seen={9},
            jersey_color=(30, 220, 220),
            role_name="goalkeeper",
        )
        teams: dict[int, int] = {}
        classifier.update(
            [AnalysisObject(9, "player", [0, 0, 20, 60], role_name="goalkeeper")],
            {9: state},
            teams,
        )
        self.assertEqual(teams[9], 1)
        self.assertEqual(classifier.assignment_sources[9], "goalkeeper_kit_reference")


class PerformancePipelineTests(unittest.TestCase):
    def test_detection_cache_round_trip_and_source_frame_offset(self) -> None:
        runner = MatchAnalysisPlusRunner()
        item = AnalysisObject(7, "player", [1, 2, 30, 80], 0.91, 11, False, "player")
        with TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "detections.jsonl"
            path.write_text(
                json.dumps({"frame": 0, "source_frame": 100, "objects": [runner._serialize_analysis_object(item)]}) + "\n",
                encoding="utf-8",
            )
            cached = runner._load_detection_cache(path, expected_start_frame=100)
        restored = runner._deserialize_analysis_object(cached[0][0])
        self.assertEqual(restored.track_id, 7)
        self.assertEqual(restored.raw_track_id, 11)
        self.assertEqual(restored.bbox, [1.0, 2.0, 30.0, 80.0])

    def test_performance_summary_reports_cache_savings(self) -> None:
        runner = MatchAnalysisPlusRunner()
        summary = runner._performance_summary(100, 5.0, 80, 20, 4.0, 2.0)
        self.assertEqual(summary["yolo_skipped_frames"], 80)
        self.assertEqual(summary["stateful_tracking_batch_size"], 1)
        self.assertTrue(summary["detection_cache_reusable"])


if __name__ == "__main__":
    unittest.main()
