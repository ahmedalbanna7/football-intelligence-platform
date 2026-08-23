import unittest
from types import SimpleNamespace
from unittest.mock import Mock

from app.tracking_quality.metrics import evaluate_release_suite, evaluate_tracking
from app.tracking_quality.service import TrackingQualityService


def _payload(rows: list[tuple[int, str, list[float]]], prediction: bool) -> dict:
    identity_key = "track_id" if prediction else "identity_id"
    payload = {
        "observations": [
            {
                "frame": frame,
                identity_key: identity,
                "bbox": bbox,
            }
            for frame, identity, bbox in rows
        ]
    }
    if not prediction:
        payload["verification"] = {
            "status": "verified",
            "annotator": "tracking-quality-test",
        }
    return payload


class TrackingQualityMetricTests(unittest.TestCase):
    def setUp(self) -> None:
        self.ground_truth_rows = [
            (0, "a", [10, 10, 30, 60]),
            (0, "b", [80, 10, 100, 60]),
            (1, "a", [14, 10, 34, 60]),
            (1, "b", [76, 10, 96, 60]),
            (2, "a", [18, 10, 38, 60]),
            (2, "b", [72, 10, 92, 60]),
        ]

    def test_perfect_tracking_scores_one_hundred(self) -> None:
        predictions = [
            (frame, "1" if identity == "a" else "2", bbox)
            for frame, identity, bbox in self.ground_truth_rows
        ]

        metrics = evaluate_tracking(
            _payload(predictions, prediction=True),
            _payload(self.ground_truth_rows, prediction=False),
        )

        self.assertEqual(100.0, metrics["idf1"])
        self.assertEqual(100.0, metrics["hota"])
        self.assertEqual(0, metrics["id_switches"])
        self.assertEqual(0, metrics["fragmentation"])

    def test_identity_swap_is_measured_as_switches(self) -> None:
        predictions = []
        for frame, identity, bbox in self.ground_truth_rows:
            if frame < 2:
                prediction_id = "1" if identity == "a" else "2"
            else:
                prediction_id = "2" if identity == "a" else "1"
            predictions.append((frame, prediction_id, bbox))

        metrics = evaluate_tracking(
            _payload(predictions, prediction=True),
            _payload(self.ground_truth_rows, prediction=False),
        )

        self.assertEqual(2, metrics["id_switches"])
        self.assertLess(metrics["idf1"], 100.0)
        self.assertLess(metrics["hota"], 100.0)

    def test_missing_middle_observation_creates_fragmentation(self) -> None:
        predictions = [
            (frame, "1" if identity == "a" else "2", bbox)
            for frame, identity, bbox in self.ground_truth_rows
            if not (frame == 1 and identity == "a")
        ]

        metrics = evaluate_tracking(
            _payload(predictions, prediction=True),
            _payload(self.ground_truth_rows, prediction=False),
        )

        self.assertEqual(1, metrics["fragmentation"])
        self.assertLess(metrics["idf1"], 100.0)

    def test_predictions_outside_selected_clip_are_ignored(self) -> None:
        predictions = [
            (frame, "1" if identity == "a" else "2", bbox)
            for frame, identity, bbox in self.ground_truth_rows
        ]
        predictions.append((99, "unrelated", [0, 0, 20, 50]))

        metrics = evaluate_tracking(
            _payload(predictions, prediction=True),
            _payload(self.ground_truth_rows, prediction=False),
        )

        self.assertEqual(100.0, metrics["idf1"])
        self.assertEqual(3, metrics["evaluated_frames"])
        self.assertEqual("annotated_frames_only", metrics["evaluation_scope"])

    def test_source_frame_coordinates_align_offset_runs(self) -> None:
        predictions = _payload([(0, "1", [10, 10, 30, 60])], prediction=True)
        predictions["observations"][0]["source_frame"] = 500
        ground_truth = _payload([(20, "a", [10, 10, 30, 60])], prediction=False)
        ground_truth["observations"][0]["source_frame"] = 500
        ground_truth["clips"] = [
            {
                "start_frame": 20,
                "end_frame": 20,
                "source_start_frame": 500,
                "source_end_frame": 500,
                "scenario": "reentry",
                "camera_style": "tactical",
                "critical": True,
            }
        ]

        metrics = evaluate_tracking(predictions, ground_truth)

        self.assertEqual(100.0, metrics["idf1"])
        self.assertEqual("source", metrics["frame_coordinate_space"])
        self.assertEqual(500, metrics["clips"][0]["start_frame"])

    def test_ground_truth_draft_preserves_original_source_frames(self) -> None:
        service = TrackingQualityService()
        service._get_predictions = Mock(
            return_value={
                "observations": [
                    {
                        "frame": 0,
                        "source_frame": 90000,
                        "track_id": 7,
                        "raw_track_id": 12,
                        "bbox": [10, 20, 40, 90],
                        "team": 1,
                        "role_name": "player",
                    },
                    {
                        "frame": 10,
                        "source_frame": 90010,
                        "track_id": 7,
                        "raw_track_id": 12,
                        "bbox": [12, 20, 42, 90],
                        "team": 1,
                        "role_name": "player",
                    },
                ]
            }
        )
        service._put_json = Mock()
        assessment = SimpleNamespace(predictions_object="matches/11/run/quality/predictions.jsonl")
        db = Mock()
        db.query.return_value.filter.return_value.first.return_value = assessment
        run = SimpleNamespace(id=57, match_id=11, summary_json={})

        result = service.build_ground_truth_draft(
            db,
            run,
            start_frame=0,
            end_frame=10,
            sample_every_frames=10,
            track_ids=[7],
            scenario="crossing",
            camera_style="tactical",
            critical=True,
        )

        ground_truth = result["ground_truth"]
        self.assertEqual(90000, ground_truth["clips"][0]["source_start_frame"])
        self.assertEqual(90010, ground_truth["clips"][0]["source_end_frame"])
        self.assertEqual(90000, ground_truth["frames"][0]["objects"][0]["source_frame"])

    def test_unverified_ground_truth_is_rejected(self) -> None:
        ground_truth = _payload(self.ground_truth_rows, prediction=False)
        ground_truth["verification"]["status"] = "draft"

        with self.assertRaisesRegex(ValueError, "manually reviewed"):
            evaluate_tracking(
                _payload(self.ground_truth_rows, prediction=True),
                ground_truth,
            )

    def test_selected_identity_coverage_ignores_other_people_in_same_frame(self) -> None:
        ground_truth = _payload(
            [row for row in self.ground_truth_rows if row[1] == "a"],
            prediction=False,
        )
        ground_truth["coverage"] = "selected_identities"
        predictions = [
            (frame, "1" if identity == "a" else "unrelated", bbox)
            for frame, identity, bbox in self.ground_truth_rows
        ]

        metrics = evaluate_tracking(
            _payload(predictions, prediction=True),
            ground_truth,
        )

        self.assertEqual(100.0, metrics["idf1"])
        self.assertEqual("selected_identities", metrics["coverage"])

    def test_release_gate_reports_per_clip_and_passes_complete_suite(self) -> None:
        predictions = _payload(
            [
                (frame, "1" if identity == "a" else "2", bbox)
                for frame, identity, bbox in self.ground_truth_rows
            ],
            prediction=True,
        )
        ground_truth = _payload(self.ground_truth_rows, prediction=False)
        for observation in predictions["observations"]:
            observation["team"] = 1 if observation["track_id"] == "1" else 2
        for observation in ground_truth["observations"]:
            observation["team"] = 1 if observation["identity_id"] == "a" else 2
        ground_truth["clips"] = [
            {"start_frame": 0, "end_frame": 0, "scenario": "crossing", "camera_style": "tactical", "critical": True},
            {"start_frame": 1, "end_frame": 1, "scenario": "crowding", "camera_style": "tactical", "critical": True},
            {"start_frame": 2, "end_frame": 2, "scenario": "reentry", "camera_style": "close_or_moving", "critical": True},
        ]

        metrics = evaluate_tracking(predictions, ground_truth)

        self.assertEqual(3, len(metrics["clips"]))
        self.assertEqual(0, metrics["cross_team"]["cross_team_switches"])
        self.assertEqual("passed", metrics["release_gate"]["status"])

    def test_release_gate_blocks_cross_team_identity_reuse(self) -> None:
        ground_truth = _payload(self.ground_truth_rows, prediction=False)
        predictions = _payload(
            [(frame, "shared", bbox) for frame, _, bbox in self.ground_truth_rows],
            prediction=True,
        )
        for observation in ground_truth["observations"]:
            observation["team"] = 1 if observation["identity_id"] == "a" else 2
        ground_truth["clips"] = [
            {"start_frame": 0, "end_frame": 2, "scenario": "crossing", "camera_style": "tactical", "critical": True}
        ]

        metrics = evaluate_tracking(predictions, ground_truth)

        self.assertEqual(1, metrics["cross_team"]["cross_team_switches"])
        condition = next(
            item for item in metrics["release_gate"]["conditions"]
            if item["code"] == "cross_team_switches"
        )
        self.assertFalse(condition["passed"])

    def test_identity_metadata_supplies_verified_team_labels(self) -> None:
        ground_truth = _payload(self.ground_truth_rows, prediction=False)
        ground_truth["identities"] = {
            "a": {"team": 1, "role_name": "player"},
            "b": {"team": 2, "role_name": "player"},
        }
        for observation in ground_truth["observations"]:
            observation["team"] = None
            observation["role_name"] = None
        predictions = _payload(
            [
                (frame, "1" if identity == "a" else "2", bbox)
                for frame, identity, bbox in self.ground_truth_rows
            ],
            prediction=True,
        )

        metrics = evaluate_tracking(predictions, ground_truth)

        self.assertEqual("measured", metrics["cross_team"]["status"])
        self.assertEqual(0, metrics["cross_team"]["cross_team_switches"])

    def test_release_gate_requires_camera_and_scenario_coverage(self) -> None:
        predictions = _payload(
            [
                (frame, "1" if identity == "a" else "2", bbox)
                for frame, identity, bbox in self.ground_truth_rows
            ],
            prediction=True,
        )
        ground_truth = _payload(self.ground_truth_rows, prediction=False)
        for observation in ground_truth["observations"]:
            observation["team"] = 1 if observation["identity_id"] == "a" else 2
        ground_truth["clips"] = [
            {"start_frame": 0, "end_frame": 2, "scenario": "crossing", "camera_style": "tactical", "critical": True}
        ]

        metrics = evaluate_tracking(predictions, ground_truth)

        self.assertEqual("blocked", metrics["release_gate"]["status"])
        missing = {
            item["code"]: item.get("missing", [])
            for item in metrics["release_gate"]["conditions"]
        }
        self.assertIn("crowding", missing["scenario_coverage"])
        self.assertIn("close_or_moving", missing["camera_coverage"])

    def test_release_suite_combines_multiple_camera_runs(self) -> None:
        base = {
            "idf1": 98.0,
            "hota": 94.0,
            "fragmentation": 0,
            "cross_team": {"status": "measured", "cross_team_switches": 0},
            "release_gate": {"unresolved_fragments": 0},
        }
        suite = evaluate_release_suite(
            [
                {
                    **base,
                    "clips": [
                        {"scenario": "crossing", "camera_style": "tactical", "critical": True, "id_switches": 0},
                        {"scenario": "crowding", "camera_style": "tactical", "critical": True, "id_switches": 0},
                    ],
                },
                {
                    **base,
                    "clips": [
                        {"scenario": "reentry", "camera_style": "close_or_moving", "critical": True, "id_switches": 0}
                    ],
                },
            ]
        )

        self.assertEqual("passed", suite["status"])
        self.assertEqual(2, suite["cases_count"])


if __name__ == "__main__":
    unittest.main()
