import unittest

import cv2
import numpy as np

from app.match_analysis_plus.runner import (
    AnalysisObject,
    BallStaticFilter,
    GOAL_AREA_LENGTH_CM,
    GOAL_AREA_WIDTH_CM,
    MatchAnalysisPlusRunner,
    PENALTY_SPOT_DISTANCE_CM,
    PITCH_LENGTH_CM,
    PITCH_WIDTH_CM,
    PitchRadar,
    PlayerValidityFilter,
    TeamColorClassifier,
    TrackIdStabilizer,
)


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


class PlayerValidityFilterTests(unittest.TestCase):
    def test_corner_flag_is_rejected_before_identity_assignment(self) -> None:
        frame = np.full((FRAME_HEIGHT, FRAME_WIDTH, 3), (45, 105, 45), dtype=np.uint8)
        frame[20:190, 30:150] = (70, 78, 165)
        pole = AnalysisObject(
            track_id=301,
            raw_track_id=301,
            class_name="player",
            bbox=[45, 35, 135, 175],
            confidence=0.61,
        )
        player = _player(240, 100, 302)
        cv2.line(frame, (78, 55), (78, 174), (235, 235, 235), 2)
        cv2.fillPoly(
            frame,
            [np.array([[78, 35], [101, 47], [78, 61]], dtype=np.int32)],
            (20, 45, 230),
        )
        _draw_player(frame, player, (0, 220, 220), (25, 25, 25), (30, 90, 210))

        validity_filter = PlayerValidityFilter()
        output = validity_filter.filter([pole, player], frame)

        self.assertEqual([302], [item.track_id for item in output])
        self.assertEqual(1, validity_filter.summary()["rejected_field_fixtures"])

    def test_football_model_classes_include_players_goalkeepers_and_ball(self) -> None:
        class Model:
            names = {0: "ball", 1: "goalkeeper", 2: "player", 3: "referee"}

        runner = MatchAnalysisPlusRunner.__new__(MatchAnalysisPlusRunner)
        self.assertEqual([0, 1, 2], runner._target_class_ids(Model()))


class PitchRadarTests(unittest.TestCase):
    def test_goal_area_geometry_maps_to_metric_penalty_end(self) -> None:
        frame = np.full((720, 1280, 3), (45, 112, 45), dtype=np.uint8)
        white = (245, 245, 245)
        cv2.line(frame, (340, 200), (940, 200), white, 8)
        cv2.line(frame, (340, 200), (340, 350), white, 8)
        cv2.line(frame, (940, 200), (940, 350), white, 8)
        cv2.line(frame, (340, 350), (940, 350), white, 8)
        cv2.line(frame, (0, 200), (1279, 200), white, 8)
        cv2.line(frame, (0, 650), (1279, 650), white, 8)
        cv2.line(frame, (520, 60), (520, 200), white, 10)
        cv2.line(frame, (760, 60), (760, 200), white, 10)
        cv2.line(frame, (520, 60), (760, 60), white, 10)
        cv2.ellipse(frame, (640, 500), (16, 7), 0, 0, 360, white, cv2.FILLED)

        radar = PitchRadar(model=None, stride=12)
        result = radar._goal_area_metric_homography(
            frame,
            players=[],
            marker_candidates=[(640.0, 500.0)],
        )

        self.assertIsNotNone(result)
        homography = result[0]  # type: ignore[index]
        transformed = cv2.perspectiveTransform(
            np.float32([[640.0, 500.0], [340.0, 350.0]]).reshape(-1, 1, 2),
            homography,
        ).reshape(-1, 2)
        np.testing.assert_allclose(
            transformed[0],
            [PITCH_LENGTH_CM - PENALTY_SPOT_DISTANCE_CM, PITCH_WIDTH_CM / 2],
            atol=35.0,
        )
        np.testing.assert_allclose(
            transformed[1],
            [
                PITCH_LENGTH_CM - GOAL_AREA_LENGTH_CM,
                PITCH_WIDTH_CM / 2 - GOAL_AREA_WIDTH_CM / 2,
            ],
            atol=55.0,
        )

    def test_radar_renders_observed_players_with_stable_ids(self) -> None:
        radar = PitchRadar(model=None, stride=12)
        radar.homography = np.array(
            [
                [12000.0 / FRAME_WIDTH, 0.0, 0.0],
                [0.0, 7000.0 / FRAME_HEIGHT, 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        radar.last_calibrated_frame = 0
        frame = np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8)
        player_one = _player(145, 90, 7)
        player_two = _player(410, 110, 9)
        ball = AnalysisObject(
            track_id=1,
            raw_track_id=1,
            class_name="ball",
            bbox=[315, 210, 327, 222],
            confidence=0.9,
        )

        radar.draw(
            frame,
            frame_index=1,
            players=[player_one, player_two],
            balls=[ball],
            team_by_track={7: 1, 9: 2},
        )

        self.assertEqual(1, radar.summary()["rendered_frames"])
        self.assertGreater(int(np.count_nonzero(frame)), 1000)

    def test_visual_marker_requires_temporal_stability(self) -> None:
        radar = PitchRadar(model=None, stride=12)

        first = radar._track_visual_markers(
            0,
            [(100.0, 120.0), (300.0, 220.0)],
            FRAME_WIDTH,
        )
        second = radar._track_visual_markers(
            12,
            [(180.0, 120.0), (305.0, 223.0)],
            FRAME_WIDTH,
        )

        self.assertEqual([], first)
        self.assertEqual([(305.0, 223.0)], second)


class BallStaticFilterTests(unittest.TestCase):
    def test_pitch_coordinates_reject_a_marker_despite_camera_motion(self) -> None:
        ball_filter = BallStaticFilter(static_hits=3)
        outputs: list[list[AnalysisObject]] = []
        for frame_index in range(5):
            marker = AnalysisObject(
                track_id=frame_index + 1,
                raw_track_id=frame_index + 1,
                class_name="ball",
                bbox=[100 + frame_index * 18, 170, 132 + frame_index * 18, 184],
                confidence=0.7,
            )
            outputs.append(
                ball_filter.filter(
                    frame_index,
                    [marker],
                    [],
                    FRAME_WIDTH,
                    pitch_transform=lambda _point: (2450.0, 3180.0),
                )
            )

        self.assertTrue(all(output == [] for output in outputs))
        self.assertEqual(5, ball_filter.summary()["filtered_static_candidates"])
        self.assertEqual([(188.0, 177.0)], ball_filter.static_marker_centers(4))

    def test_moving_ball_is_not_rejected_in_pitch_coordinates(self) -> None:
        ball_filter = BallStaticFilter(static_hits=3)
        outputs: list[list[AnalysisObject]] = []
        for frame_index in range(6):
            ball = AnalysisObject(
                track_id=frame_index + 1,
                raw_track_id=frame_index + 1,
                class_name="ball",
                bbox=[100, 170, 116, 186],
                confidence=0.8,
            )
            outputs.append(
                ball_filter.filter(
                    frame_index,
                    [ball],
                    [],
                    FRAME_WIDTH,
                    pitch_transform=lambda _point, index=frame_index: (
                        900.0 + index * 180.0,
                        3000.0,
                    ),
                )
            )

        self.assertEqual([], outputs[0])
        self.assertTrue(all(len(output) == 1 for output in outputs[1:]))
        self.assertEqual(0, ball_filter.summary()["filtered_static_candidates"])

    def test_ball_near_player_is_kept_immediately(self) -> None:
        ball_filter = BallStaticFilter(static_hits=3)
        player = _player(180, 100, 1)
        ball = AnalysisObject(
            track_id=1,
            raw_track_id=1,
            class_name="ball",
            bbox=[198, 214, 214, 230],
            confidence=0.8,
        )

        output = ball_filter.filter(0, [ball], [player], FRAME_WIDTH)

        self.assertEqual([ball], output)


class TeamColorClassifierTests(unittest.TestCase):
    def test_same_kit_tracks_share_a_team_and_distinct_kit_is_separated(self) -> None:
        classifier = TeamColorClassifier()
        yellow_one = _player(100, 100, 1)
        blue = _player(250, 100, 2)
        yellow_two = _player(400, 100, 3)

        class State:
            def __init__(self, jersey_color: tuple[int, int, int]) -> None:
                self.jersey_color = jersey_color

        states = {
            1: State((25, 185, 225)),
            2: State((205, 125, 55)),
            3: State((35, 175, 215)),
        }
        team_by_track: dict[int, int] = {}

        for _ in range(5):
            classifier.update(
                [yellow_one, blue, yellow_two],
                states,  # type: ignore[arg-type]
                team_by_track,
            )

        self.assertEqual(team_by_track[1], team_by_track[3])
        self.assertNotEqual(team_by_track[1], team_by_track[2])
        self.assertEqual(2, len(classifier.summary()["kit_anchors_bgr"]))


if __name__ == "__main__":
    unittest.main()
