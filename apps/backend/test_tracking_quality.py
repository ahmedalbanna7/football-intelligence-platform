import unittest

from app.tracking_quality.metrics import evaluate_tracking


def _payload(rows: list[tuple[int, str, list[float]]], prediction: bool) -> dict:
    identity_key = "track_id" if prediction else "identity_id"
    return {
        "observations": [
            {
                "frame": frame,
                identity_key: identity,
                "bbox": bbox,
            }
            for frame, identity, bbox in rows
        ]
    }


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


if __name__ == "__main__":
    unittest.main()
