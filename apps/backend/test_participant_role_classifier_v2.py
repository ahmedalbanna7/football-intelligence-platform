import unittest
from types import SimpleNamespace

import numpy as np

from app.match_analysis_plus.runner import AnalysisObject, ParticipantRoleClassifierV2


class _Radar:
    def __init__(self, inside: bool = True, pitch_point=(5200.0, 3400.0)) -> None:
        self.inside = inside
        self.pitch_point = pitch_point

    def is_reliable(self, threshold: float = 0.58) -> bool:
        return True

    def contains_image_point(self, point, margin_cm: float = 0.0) -> bool:
        return self.inside

    def transform_point(self, point):
        return self.pitch_point if self.inside else None

    def playing_surface_mask(self, frame):
        return np.full(frame.shape[:2], 255 if self.inside else 0, dtype=np.uint8)


def _player(role: str = "player") -> AnalysisObject:
    return AnalysisObject(
        track_id=1,
        raw_track_id=1,
        class_name="player",
        bbox=[100.0, 100.0, 150.0, 230.0],
        confidence=0.95,
        role_name=role,
    )


def _stable(role: str = "player") -> SimpleNamespace:
    return SimpleNamespace(
        role_votes={role: 20.0},
        role_name=role,
        role_locked=False,
    )


class ParticipantRoleClassifierV2Tests(unittest.TestCase):
    def setUp(self) -> None:
        self.frame = np.full((360, 640, 3), (40, 130, 40), dtype=np.uint8)

    def _run(self, role: str, radar: _Radar, team: int | None, confidence: float, frames: int = 24):
        resolver = ParticipantRoleClassifierV2()
        player = _player(role)
        stable = _stable(role)
        team_map = {1: team} if team is not None else {}
        classifier = SimpleNamespace(track_confidence={1: confidence})
        for _ in range(frames):
            resolver.update([player], {1: stable}, team_map, classifier, radar, self.frame)
        return resolver, player, stable

    def test_team_player_role_locks(self) -> None:
        resolver, player, stable = self._run("player", _Radar(), 1, 0.95)

        state = resolver.get(1)
        self.assertEqual("player", player.role_name)
        self.assertEqual("player", stable.role_name)
        self.assertTrue(state.locked)
        self.assertGreaterEqual(state.confidence, 0.84)

    def test_specialized_referee_role_locks_without_team(self) -> None:
        resolver, player, _, = self._run("referee", _Radar(), 0, 1.0)

        state = resolver.get(1)
        self.assertEqual("referee", player.role_name)
        self.assertTrue(state.locked)

    def test_touchline_official_locks_as_assistant_referee(self) -> None:
        resolver, player, _ = self._run(
            "referee",
            _Radar(pitch_point=(5200.0, 90.0)),
            0,
            1.0,
        )

        state = resolver.get(1)
        self.assertEqual("assistant_referee", player.role_name)
        self.assertTrue(state.locked)
        self.assertIn("touchline_history", state.evidence)

    def test_goalkeeper_near_goal_locks(self) -> None:
        resolver, player, _ = self._run(
            "goalkeeper",
            _Radar(pitch_point=(450.0, 3400.0)),
            1,
            0.75,
        )

        state = resolver.get(1)
        self.assertEqual("goalkeeper", player.role_name)
        self.assertTrue(state.locked)

    def test_geometry_and_kit_outlier_find_goalkeeper_without_detector_label(self) -> None:
        resolver = ParticipantRoleClassifierV2()
        player = _player("player")
        stable = _stable("player")
        classifier = SimpleNamespace(
            track_confidence={1: 0.82},
            kit_outlier_score=lambda _team, _state: 0.78,
        )
        radar = _Radar(pitch_point=(520.0, 3400.0))

        for _ in range(32):
            resolver.update(
                [player],
                {1: stable},
                {1: 1},
                classifier,
                radar,
                self.frame,
            )

        state = resolver.get(1)
        self.assertEqual("goalkeeper", player.role_name)
        self.assertTrue(state.locked)
        self.assertIn("persistent_goalkeeper_zone", state.evidence)
        self.assertIn("outfield_kit_outlier", state.evidence)

    def test_matching_outfield_kit_does_not_turn_defender_into_goalkeeper(self) -> None:
        resolver = ParticipantRoleClassifierV2()
        player = _player("player")
        stable = _stable("player")
        classifier = SimpleNamespace(
            track_confidence={1: 0.95},
            kit_outlier_score=lambda _team, _state: 0.04,
        )
        radar = _Radar(pitch_point=(620.0, 3400.0))

        for _ in range(48):
            resolver.update(
                [player],
                {1: stable},
                {1: 1},
                classifier,
                radar,
                self.frame,
            )

        self.assertEqual("player", player.role_name)
        self.assertTrue(resolver.get(1).locked)

    def test_outside_person_becomes_staff_and_locks(self) -> None:
        resolver, player, _ = self._run("player", _Radar(inside=False), None, 0.0)

        state = resolver.get(1)
        self.assertEqual("staff_outside_pitch", player.role_name)
        self.assertTrue(state.locked)
        self.assertIn("outside_pitch_history", state.evidence)

    def test_locked_role_rejects_later_detector_noise(self) -> None:
        resolver, player, stable = self._run("player", _Radar(), 1, 0.95)
        stable.role_votes = {"referee": 100.0}
        for _ in range(90):
            resolver.update(
                [player],
                {1: stable},
                {1: 0},
                SimpleNamespace(track_confidence={1: 1.0}),
                _Radar(),
                self.frame,
            )

        self.assertEqual("player", resolver.get(1).role_name)
        self.assertGreater(resolver.prevented_role_changes, 0)


if __name__ == "__main__":
    unittest.main()
