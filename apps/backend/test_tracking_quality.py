import unittest

from app.tracking_quality.metrics import evaluate_tracking


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


if __name__ == "__main__":
    unittest.main()
