import unittest

import cv2
import numpy as np

from app.match_analysis_plus.runner import AnalysisObject, TrackIdStabilizer


def _player(raw_id: int, x: float, y: float, color_role: str = "player") -> AnalysisObject:
    return AnalysisObject(
        track_id=raw_id,
        raw_track_id=raw_id,
        class_name="player",
        bbox=[x, y, x + 36.0, y + 96.0],
        confidence=0.9,
        role_name=color_role,
    )


def _frame(players: list[tuple[AnalysisObject, tuple[int, int, int]]]) -> np.ndarray:
    frame = np.zeros((760, 960, 3), dtype=np.uint8)
    frame[:, :] = (42, 128, 42)
    for player, color in players:
        x1, y1, x2, y2 = [int(value) for value in player.bbox]
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, -1)
    return frame


class TrackIdStabilizerV2Tests(unittest.TestCase):
    def _confirm_single(self, color: tuple[int, int, int] = (240, 240, 240)) -> TrackIdStabilizer:
        stabilizer = TrackIdStabilizer()
        for frame_index in range(6):
            player = _player(1, 100 + frame_index * 3, 120)
            stabilizer.update(frame_index, [player], _frame([(player, color)]))
        self.assertTrue(stabilizer.tracks[1].confirmed)
        return stabilizer

    def test_impossible_same_kit_jump_cannot_steal_identity(self) -> None:
        stabilizer = self._confirm_single((40, 220, 80))
        jumped = _player(90, 680, 560)

        stabilizer.update(6, [jumped], _frame([(jumped, (40, 220, 80))]))

        self.assertEqual(5, stabilizer.tracks[1].last_frame)
        self.assertEqual(2, stabilizer.raw_to_stable[90])
        self.assertGreater(stabilizer.rejected_hard_motion_jumps, 0)

    def test_local_native_track_keeps_stable_identity(self) -> None:
        stabilizer = self._confirm_single()
        player = _player(1, 119, 121)

        output = stabilizer.update(6, [player], _frame([(player, (240, 240, 240))]))

        self.assertEqual([1], [item.track_id for item in output])
        self.assertEqual(6, stabilizer.tracks[1].last_frame)

    def test_summary_surfaces_motion_gate_quality(self) -> None:
        stabilizer = self._confirm_single()

        summary = stabilizer.summary()

        self.assertIn("max_accepted_motion_gate_ratio", summary)
        self.assertEqual(0, summary["tracks_over_motion_gate"])

    def test_team_color_blocks_cross_team_raw_id_swap(self) -> None:
        stabilizer = TrackIdStabilizer()
        for frame_index in range(10):
            blue = _player(1, 100 + frame_index * 5, 160)
            white = _player(2, 420 - frame_index * 5, 160)
            stabilizer.update(
                frame_index,
                [blue, white],
                _frame([(blue, (220, 70, 30)), (white, (245, 245, 245))]),
            )

        blue = _player(2, 152, 160)
        white = _player(1, 368, 160)
        output = stabilizer.update(
            10,
            [blue, white],
            _frame([(blue, (220, 70, 30)), (white, (245, 245, 245))]),
        )
        by_raw = {item.raw_track_id: item.track_id for item in output}

        self.assertEqual(1, by_raw[2])
        self.assertEqual(2, by_raw[1])

    def test_same_team_crossing_uses_trajectory_not_swapped_raw_ids(self) -> None:
        stabilizer = TrackIdStabilizer()
        kit = (40, 220, 80)
        for frame_index in range(10):
            right_mover = _player(1, 100 + frame_index * 6, 180)
            left_mover = _player(2, 300 - frame_index * 6, 180)
            stabilizer.update(
                frame_index,
                [right_mover, left_mover],
                _frame([(right_mover, kit), (left_mover, kit)]),
            )

        for frame_index in range(10, 26):
            right_mover = _player(2, 100 + frame_index * 6, 180)
            left_mover = _player(1, 300 - frame_index * 6, 180)
            stabilizer.update(
                frame_index,
                [right_mover, left_mover],
                _frame([(right_mover, kit), (left_mover, kit)]),
            )

        self.assertGreater(stabilizer.tracks[1].foot[0], stabilizer.tracks[2].foot[0])
        self.assertIn(2, stabilizer.tracks[1].raw_ids_seen)
        self.assertIn(1, stabilizer.tracks[2].raw_ids_seen)

    def test_gradual_camera_pan_does_not_fragment_native_identity(self) -> None:
        stabilizer = TrackIdStabilizer()
        output = []
        for frame_index in range(18):
            player = _player(1, 100 + frame_index * 9, 170 + frame_index * 2)
            output = stabilizer.update(
                frame_index,
                [player],
                _frame([(player, (245, 245, 245))]),
            )

        self.assertEqual([1], [item.track_id for item in output])
        self.assertEqual(1, len(stabilizer.tracks))

    def test_transient_referee_labels_do_not_reclassify_player(self) -> None:
        stabilizer = self._confirm_single()
        for frame_index in range(6, 10):
            player = _player(1, 119 + frame_index, 120, "referee")
            stabilizer.update(frame_index, [player], _frame([(player, (240, 240, 240))]))

        self.assertEqual("player", stabilizer.tracks[1].role_name)

    def test_consistent_referee_track_is_promoted_and_locked(self) -> None:
        stabilizer = TrackIdStabilizer()
        for frame_index in range(20):
            referee = _player(1, 120 + frame_index * 2, 140, "referee")
            stabilizer.update(
                frame_index,
                [referee],
                _frame([(referee, (30, 30, 30))]),
            )

        self.assertEqual("referee", stabilizer.tracks[1].role_name)
        self.assertTrue(stabilizer.tracks[1].role_locked)

    def test_player_and_referee_keep_identities_when_raw_ids_swap(self) -> None:
        stabilizer = TrackIdStabilizer()
        for frame_index in range(20):
            player = _player(1, 100 + frame_index * 5, 180, "player")
            referee = _player(2, 360 - frame_index * 5, 180, "referee")
            stabilizer.update(
                frame_index,
                [player, referee],
                _frame([(player, (40, 220, 80)), (referee, (25, 25, 25))]),
            )

        for frame_index in range(20, 30):
            player = _player(2, 100 + frame_index * 5, 180, "player")
            referee = _player(1, 360 - frame_index * 5, 180, "referee")
            output = stabilizer.update(
                frame_index,
                [player, referee],
                _frame([(player, (40, 220, 80)), (referee, (25, 25, 25))]),
            )

        by_role = {item.role_name: item.track_id for item in output}
        self.assertEqual(1, by_role["player"])
        self.assertEqual(2, by_role["referee"])

    def test_two_referees_cross_without_exchanging_stable_ids(self) -> None:
        stabilizer = TrackIdStabilizer()
        kit = (25, 25, 25)
        for frame_index in range(20):
            right_mover = _player(1, 90 + frame_index * 5, 200, "referee")
            left_mover = _player(2, 350 - frame_index * 5, 200, "referee")
            stabilizer.update(
                frame_index,
                [right_mover, left_mover],
                _frame([(right_mover, kit), (left_mover, kit)]),
            )

        for frame_index in range(20, 32):
            right_mover = _player(2, 90 + frame_index * 5, 200, "referee")
            left_mover = _player(1, 350 - frame_index * 5, 200, "referee")
            stabilizer.update(
                frame_index,
                [right_mover, left_mover],
                _frame([(right_mover, kit), (left_mover, kit)]),
            )

        self.assertGreater(stabilizer.tracks[1].foot[0], stabilizer.tracks[2].foot[0])
        self.assertGreater(stabilizer.raw_id_motion_conflict_ignores, 0)

    def test_long_gap_native_reid_preserves_identity(self) -> None:
        stabilizer = self._confirm_single((40, 220, 80))
        reentry = _player(1, 130, 120)

        output = stabilizer.update(50, [reentry], _frame([(reentry, (40, 220, 80))]))

        self.assertEqual([1], [item.track_id for item in output])
        self.assertEqual(1, len(stabilizer.tracks))
        self.assertEqual(1, stabilizer.tracks[1].native_reid_reentries)

    def test_long_gap_new_raw_id_starts_new_identity(self) -> None:
        stabilizer = self._confirm_single((40, 220, 80))
        reentry = _player(91, 130, 120)

        stabilizer.update(50, [reentry], _frame([(reentry, (40, 220, 80))]))

        self.assertEqual(5, stabilizer.tracks[1].last_frame)
        self.assertEqual(2, stabilizer.raw_to_stable[91])
        self.assertGreater(stabilizer.rejected_long_gap_reentries, 0)


if __name__ == "__main__":
    unittest.main()
