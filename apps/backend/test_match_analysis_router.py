import unittest
from types import SimpleNamespace

from app.modules.match_analysis_plus.router import resolve_player_detection_mode


class _FakeSession:
    def __init__(self, runs):
        self.runs = {run.id: run for run in runs}

    def get(self, _model, run_id):
        return self.runs.get(run_id)


def _run(run_id, summary=None, config=None):
    return SimpleNamespace(
        id=run_id,
        summary_json=summary or {},
        analysis_config_json=config or {},
    )


class PlayerDetectionModeProvenanceTests(unittest.TestCase):
    def test_recovers_specialized_detector_through_nested_caches(self):
        original = _run(
            73,
            summary={"model_mode": "football-specialized-yolo"},
        )
        first_cache = _run(
            82,
            summary={"model_mode": "cached-football-detections"},
            config={
                "reuse_run_id": 73,
                "reuse_model_mode": "football-specialized-yolo",
            },
        )
        second_cache = _run(
            94,
            summary={
                "model_mode": "cached-football-detections",
                "player_detection_mode": "cached-football-detections",
            },
            config={
                "reuse_run_id": 82,
                "reuse_model_mode": "cached-football-detections",
            },
        )

        mode = resolve_player_detection_mode(
            _FakeSession([original, first_cache, second_cache]),
            second_cache,
        )

        self.assertEqual(mode, "football-specialized-yolo")

    def test_preserves_non_cached_detector_mode(self):
        source = _run(12, summary={"model_mode": "generic-yolo"})

        mode = resolve_player_detection_mode(_FakeSession([source]), source)

        self.assertEqual(mode, "generic-yolo")

    def test_stops_on_cache_cycle(self):
        first = _run(
            1,
            summary={"model_mode": "cached-football-detections"},
            config={"reuse_run_id": 2},
        )
        second = _run(
            2,
            summary={"model_mode": "cached-football-detections"},
            config={"reuse_run_id": 1},
        )

        mode = resolve_player_detection_mode(_FakeSession([first, second]), first)

        self.assertEqual(mode, "cached-football-detections")


if __name__ == "__main__":
    unittest.main()
