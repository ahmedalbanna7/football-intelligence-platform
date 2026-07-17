import unittest

import cv2
import numpy as np

from app.match_analysis_plus.runner import AnalysisObject, TrackIdStabilizer


FRAME_HEIGHT = 360
FRAME_WIDTH = 640


def _player(
    x: float,
    y: float,
    raw_id: int,
    height: float = 120,
) -> AnalysisObject:
    return AnalysisObject(
        track_id=raw_id,
        raw_track_id=raw_id,
        class_name="player",
        bbox=[x, y, x + 44, y + height],
        confidence=0.9,
    )


def _draw_player(
    frame: np.ndarray,
    player: AnalysisObject,
    jersey: tuple[int, int, int],
    shorts: tuple[int, int, int],
    marker: tuple[int, int, int],
) -> None:
    x1, y1, x2, y2 = [int(value) for value in player.bbox]
    height = y2 - y1
    cv2.rectangle(frame, (x1, y1), (x2, y1 + int(height * 0.58)), jersey, cv2.FILLED)
    cv2.rectangle(frame, (x1, y1 + int(height * 0.58)), (x2, y2), shorts, cv2.FILLED)
    cv2.circle(frame, ((x1 + x2) // 2, y1 + 10), 6, marker, cv2.FILLED)


class TrackIdStabilizerTests(unittest.TestCase):
    def _run_crossing(self, same_jersey: bool) -> TrackIdStabilizer:
        stabilizer = TrackIdStabilizer(max_gap_frames=30)
        first_ids: tuple[int, int] | None = None
        last_ids: tuple[int, int] | None = None

        for frame_index in range(24):
            frame = np.full((FRAME_HEIGHT, FRAME_WIDTH, 3), (45, 105, 45), dtype=np.uint8)
            player_a = _player(90 + frame_index * 18, 112, 10 if frame_index < 12 else 20)
            player_b = _player(506 - frame_index * 18, 112, 20 if frame_index < 12 else 10)
            jersey_a = (0, 220, 220)
            jersey_b = jersey_a if same_jersey else (220, 70, 20)
            _draw_player(frame, player_a, jersey_a, (25, 25, 25), (30, 90, 210))
            _draw_player(frame, player_b, jersey_b, (210, 210, 210), (90, 170, 230))

            output = stabilizer.update(frame_index, [player_a, player_b], frame)
            self.assertTrue(all(not item.is_predicted for item in output))
            if len(output) == 2:
                current_ids = (output[0].track_id, output[1].track_id)
                if first_ids is None:
                    first_ids = current_ids
                last_ids = current_ids

        self.assertIsNotNone(first_ids)
        self.assertEqual(first_ids, last_ids)
        self.assertEqual(2, len(stabilizer.tracks))
        self.assertTrue(all(state.identity_locked for state in stabilizer.tracks.values()))
        return stabilizer

    def test_different_jerseys_survive_raw_id_swap_and_overlap(self) -> None:
        stabilizer = self._run_crossing(same_jersey=False)
        self.assertGreater(stabilizer.rejected_color_family_mismatches, 0)
        self.assertGreater(stabilizer.crowded_visual_freezes, 0)

    def test_same_jersey_survives_raw_id_swap_and_overlap(self) -> None:
        stabilizer = self._run_crossing(same_jersey=True)
        self.assertGreater(stabilizer.motion_matches, 0)
        self.assertGreater(stabilizer.crowded_visual_freezes, 0)

    def test_depth_and_ground_trajectory_survive_crossing(self) -> None:
        stabilizer = TrackIdStabilizer(max_gap_frames=30)
        first_ids: tuple[int, int] | None = None
        last_ids: tuple[int, int] | None = None

        for frame_index in range(22):
            frame = np.full((FRAME_HEIGHT, FRAME_WIDTH, 3), (45, 105, 45), dtype=np.uint8)
            height_a = 88 + frame_index * 2.4
            height_b = 138 - frame_index * 2.1
            player_a = _player(170 + frame_index * 9, 82 + frame_index * 4, 31 if frame_index < 11 else 41, height_a)
            player_b = _player(360 - frame_index * 9, 184 - frame_index * 4, 41 if frame_index < 11 else 31, height_b)
            _draw_player(frame, player_a, (0, 220, 220), (30, 30, 30), (40, 100, 210))
            _draw_player(frame, player_b, (0, 220, 220), (215, 215, 215), (100, 180, 230))

            output = stabilizer.update(frame_index, [player_a, player_b], frame)
            self.assertTrue(all(not item.is_predicted for item in output))
            if len(output) == 2:
                current_ids = (output[0].track_id, output[1].track_id)
                if first_ids is None:
                    first_ids = current_ids
                last_ids = current_ids

        self.assertIsNotNone(first_ids)
        self.assertEqual(first_ids, last_ids)
        self.assertEqual(2, len(stabilizer.tracks))
        self.assertTrue(all(state.depth_proxy > 0 for state in stabilizer.tracks.values()))

    def test_identity_is_reacquired_after_temporary_occlusion(self) -> None:
        stabilizer = TrackIdStabilizer(max_gap_frames=30)
        first_ids: tuple[int, int] | None = None
        last_ids: tuple[int, int] | None = None

        for frame_index in range(28):
            frame = np.full((FRAME_HEIGHT, FRAME_WIDTH, 3), (45, 105, 45), dtype=np.uint8)
            player_a = _player(90 + frame_index * 16, 112, 51 if frame_index < 14 else 61)
            player_b = _player(510 - frame_index * 16, 112, 61 if frame_index < 14 else 51)
            _draw_player(frame, player_a, (0, 220, 220), (25, 25, 25), (30, 90, 210))
            _draw_player(frame, player_b, (0, 220, 220), (210, 210, 210), (90, 170, 230))

            visible = [player_a] if 11 <= frame_index <= 14 else [player_a, player_b]
            output = stabilizer.update(frame_index, visible, frame)
            self.assertTrue(all(not item.is_predicted for item in output))
            if len(output) == 2:
                current_ids = (output[0].track_id, output[1].track_id)
                if first_ids is None:
                    first_ids = current_ids
                last_ids = current_ids

        self.assertIsNotNone(first_ids)
        self.assertEqual(first_ids, last_ids)
        self.assertEqual(2, len(stabilizer.tracks))
        self.assertTrue(all(state.identity_locked for state in stabilizer.tracks.values()))

    def test_early_raw_swap_cannot_cross_distinct_jersey_colors(self) -> None:
        stabilizer = TrackIdStabilizer(max_gap_frames=20)
        first_ids: tuple[int, int] | None = None
        last_ids: tuple[int, int] | None = None

        for frame_index in range(12):
            frame = np.full((FRAME_HEIGHT, FRAME_WIDTH, 3), (45, 105, 45), dtype=np.uint8)
            player_a = _player(180 + frame_index * 9, 112, 81 if frame_index < 2 else 91)
            player_b = _player(300 - frame_index * 9, 112, 91 if frame_index < 2 else 81)
            _draw_player(frame, player_a, (0, 220, 220), (25, 25, 25), (30, 90, 210))
            _draw_player(frame, player_b, (220, 70, 20), (210, 210, 210), (90, 170, 230))

            output = stabilizer.update(frame_index, [player_a, player_b], frame)
            self.assertTrue(all(not item.is_predicted for item in output))
            if len(output) == 2:
                current_ids = (output[0].track_id, output[1].track_id)
                if first_ids is None:
                    first_ids = current_ids
                last_ids = current_ids

        self.assertIsNotNone(first_ids)
        self.assertEqual(first_ids, last_ids)
        self.assertEqual(2, len(stabilizer.tracks))

    def test_crowded_bootstrap_keeps_raw_owner_until_appearance_is_visible(self) -> None:
        stabilizer = TrackIdStabilizer(max_gap_frames=20)
        first_ids: tuple[int, int] | None = None
        last_ids: tuple[int, int] | None = None

        for frame_index in range(10):
            frame = np.full((FRAME_HEIGHT, FRAME_WIDTH, 3), (45, 105, 45), dtype=np.uint8)
            separation = frame_index * 10
            player_a = _player(220 - separation, 112, 101)
            player_b = _player(235 + separation, 112, 111)
            _draw_player(frame, player_a, (0, 220, 220), (25, 25, 25), (30, 90, 210))
            _draw_player(frame, player_b, (220, 70, 20), (210, 210, 210), (90, 170, 230))

            output = stabilizer.update(frame_index, [player_a, player_b], frame)
            self.assertTrue(all(not item.is_predicted for item in output))
            if len(output) == 2:
                current_ids = (output[0].track_id, output[1].track_id)
                if first_ids is None:
                    first_ids = current_ids
                last_ids = current_ids

        self.assertIsNotNone(first_ids)
        self.assertEqual(first_ids, last_ids)
        self.assertEqual(2, len(stabilizer.tracks))

    def test_single_merged_detection_does_not_swap_two_tracks(self) -> None:
        stabilizer = TrackIdStabilizer(max_gap_frames=20)
        first_ids: tuple[int, int] | None = None
        last_ids: tuple[int, int] | None = None

        for frame_index in range(16):
            frame = np.full((FRAME_HEIGHT, FRAME_WIDTH, 3), (45, 105, 45), dtype=np.uint8)
            if frame_index < 7:
                x_a = 130 + frame_index * 18
                x_b = 370 - frame_index * 18
            else:
                x_a = 238 + (frame_index - 6) * 18
                x_b = 262 - (frame_index - 6) * 18

            player_a = _player(x_a, 112, 121 if frame_index < 7 else 131)
            player_b = _player(x_b, 112, 131 if frame_index < 7 else 121)
            _draw_player(frame, player_a, (0, 220, 220), (25, 25, 25), (30, 90, 210))
            _draw_player(frame, player_b, (220, 70, 20), (210, 210, 210), (90, 170, 230))

            if frame_index == 6:
                merged = _player(238, 112, 131)
                _draw_player(frame, merged, (0, 220, 220), (25, 25, 25), (30, 90, 210))
                output = stabilizer.update(frame_index, [merged], frame)
                self.assertEqual([], output)
                continue

            output = stabilizer.update(frame_index, [player_a, player_b], frame)
            self.assertTrue(all(not item.is_predicted for item in output))
            if len(output) == 2:
                current_ids = (output[0].track_id, output[1].track_id)
                if first_ids is None:
                    first_ids = current_ids
                last_ids = current_ids

        self.assertIsNotNone(first_ids)
        self.assertEqual(first_ids, last_ids)
        self.assertEqual(2, len(stabilizer.tracks))
        self.assertGreater(stabilizer.prediction_ambiguity_freezes, 0)

    def test_tentative_detection_never_renders_as_a_ghost_track(self) -> None:
        stabilizer = TrackIdStabilizer(max_gap_frames=20)
        frame = np.full((FRAME_HEIGHT, FRAME_WIDTH, 3), (45, 105, 45), dtype=np.uint8)
        player = _player(210, 112, 201)
        _draw_player(frame, player, (0, 220, 220), (25, 25, 25), (30, 90, 210))

        self.assertEqual([], stabilizer.update(0, [player], frame))
        self.assertEqual([], stabilizer.update(1, [player], frame))
        for frame_index in range(2, 10):
            self.assertEqual([], stabilizer.update(frame_index, [], frame))

        summary = stabilizer.summary()
        self.assertEqual(0, summary["stable_tracks_count"])
        self.assertEqual(0, summary["predicted_boxes_rendered"])
        self.assertGreaterEqual(summary["discarded_tentative_tracks"], 1)


if __name__ == "__main__":
    unittest.main()
