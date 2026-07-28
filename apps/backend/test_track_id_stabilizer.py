import unittest

import cv2
import numpy as np

from app.match_analysis_plus.runner import (
    AnalysisObject,
    BallTrackerV2,
    BallStaticFilter,
    GOAL_AREA_LENGTH_CM,
    GOAL_AREA_WIDTH_CM,
    MatchAnalysisPlusRunner,
    PENALTY_SPOT_DISTANCE_CM,
    PITCH_LENGTH_CM,
    PITCH_WIDTH_CM,
    PitchRadar,
    PitchOccupancyFilter,
    PlayerValidityFilter,
    PossessionTracker,
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

    def test_specialized_detector_keeps_small_player_class_before_pitch_mask(self) -> None:
        frame = np.full((FRAME_HEIGHT, FRAME_WIDTH, 3), (45, 105, 45), dtype=np.uint8)
        small_player = AnalysisObject(
            track_id=41,
            raw_track_id=41,
            class_name="player",
            bbox=[300, 80, 308, 112],
            confidence=0.72,
        )

        validity_filter = PlayerValidityFilter()
        output = validity_filter.filter(
            [small_player],
            frame,
            specialized_detector=True,
        )

        self.assertEqual([41], [item.track_id for item in output])
        self.assertEqual(1, validity_filter.summary()["specialized_detector_observations"])

    def test_football_model_classes_include_players_goalkeepers_and_ball(self) -> None:
        class Model:
            names = {0: "ball", 1: "goalkeeper", 2: "player", 3: "referee"}

        runner = MatchAnalysisPlusRunner.__new__(MatchAnalysisPlusRunner)
        self.assertEqual([0, 1, 2, 3], runner._target_class_ids(Model()))
        self.assertEqual(
            [0],
            runner._target_class_ids(Model(), include_players=False, include_ball=True),
        )


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
        radar.calibration_confidence = 1.0
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

    def test_referee_is_excluded_from_team_kit_anchors(self) -> None:
        classifier = TeamColorClassifier()
        yellow = _player(100, 100, 1)
        blue = _player(250, 100, 2)
        referee = _player(400, 100, 3)
        referee.role_name = "referee"

        class State:
            def __init__(
                self,
                jersey_color: tuple[int, int, int],
                role_name: str,
            ) -> None:
                self.jersey_color = jersey_color
                self.role_name = role_name
                self.appearance_hist = None

        states = {
            1: State((25, 185, 225), "player"),
            2: State((205, 125, 55), "player"),
            3: State((15, 15, 15), "referee"),
        }
        team_by_track: dict[int, int] = {}

        classifier.update(
            [yellow, blue, referee],
            states,  # type: ignore[arg-type]
            team_by_track,
        )

        self.assertEqual(0, team_by_track[3])
        self.assertEqual([3], classifier.summary()["official_tracks"])
        self.assertEqual(2, len(classifier.summary()["kit_anchors_bgr"]))

    def test_stored_kit_references_keep_team_orientation(self) -> None:
        classifier = TeamColorClassifier(
            reference_palettes_bgr={
                1: [(20, 190, 225)],
                2: [(205, 125, 55)],
            },
            team_labels={
                1: "Primary club",
                2: "Opponent",
            },
        )
        primary = _player(100, 100, 1)
        opponent = _player(250, 100, 2)

        class State:
            def __init__(self, jersey_color: tuple[int, int, int]) -> None:
                self.jersey_color = jersey_color
                self.role_name = "player"
                self.appearance_hist = None

        states = {
            1: State((25, 185, 220)),
            2: State((200, 130, 60)),
        }
        team_by_track: dict[int, int] = {}

        classifier.update(
            [primary, opponent],
            states,  # type: ignore[arg-type]
            team_by_track,
        )

        self.assertEqual(1, team_by_track[1])
        self.assertEqual(2, team_by_track[2])
        summary = classifier.summary()
        self.assertEqual("stored_kit_images", summary["reference_source"])
        self.assertEqual("Primary club", summary["team_labels"]["1"])

    def test_kit_palette_excludes_white_background(self) -> None:
        runner = MatchAnalysisPlusRunner()
        image = np.full((160, 160, 3), 255, dtype=np.uint8)
        image[20:145, 48:112] = (20, 180, 225)

        palette = runner._extract_reference_palette(image)

        self.assertTrue(palette)
        self.assertLess(
            float(np.linalg.norm(np.array(palette[0]) - np.array((20, 180, 225)))),
            20.0,
        )


class ParticipantRoleTests(unittest.TestCase):
    def test_specialized_model_roles_are_preserved(self) -> None:
        runner = MatchAnalysisPlusRunner()

        self.assertEqual("player", runner._map_class_name("player"))
        self.assertEqual("player", runner._map_class_name("goalkeeper"))
        self.assertEqual("player", runner._map_class_name("referee"))
        self.assertEqual("goalkeeper", runner._map_role_name("goalkeeper"))
        self.assertEqual("referee", runner._map_role_name("referee"))


class VisualLayerArtifactTests(unittest.TestCase):
    def test_visual_layer_payload_keeps_video_and_metric_paths_separate(self) -> None:
        runner = MatchAnalysisPlusRunner()
        payload = runner._build_visual_layers_payload(
            fps=25.0,
            frames_processed=100,
            width=1920,
            height=1080,
            track_frames={7: 80},
            track_video_samples={7: [[0, 500, 700], [5, 520, 705]]},
            track_pitch_samples={
                7: [
                    {"frame": 0, "x": 9050.0, "y": 3100.0, "z": 0.0},
                    {"frame": 5, "x": 9100.0, "y": 3125.0, "z": 0.0},
                ]
            },
            pitch_to_video_samples=[
                [0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
            ],
            team_by_track={7: 2},
        )

        self.assertEqual(2, payload["schema_version"])
        self.assertEqual([1920, 1080], payload["resolution"])
        self.assertEqual(4.0, payload["duration_seconds"])
        self.assertEqual([[0, 500, 700], [5, 520, 705]], payload["tracks"][0]["video_path"])
        self.assertEqual([[0, 9050, 3100], [5, 9100, 3125]], payload["tracks"][0]["pitch_path"])
        self.assertEqual(2, payload["tracks"][0]["team"])

    def test_pitch_projection_is_inverse_of_metric_calibration(self) -> None:
        radar = PitchRadar(None)
        radar.homography = np.array(
            [[5.0, 0.0, 100.0], [0.0, 5.0, 200.0], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
        radar.calibration_confidence = 1.0

        inverse = radar.pitch_to_video_matrix()

        self.assertIsNotNone(inverse)
        projected = cv2.perspectiveTransform(
            np.array([[[600.0, 700.0]]], dtype=np.float32),
            inverse,
        ).reshape(2)
        self.assertTrue(np.allclose(projected, [100.0, 100.0], atol=1e-4))


class PitchCalibrationV2Tests(unittest.TestCase):
    def test_outside_pitch_person_is_rejected_before_stable_tracking(self) -> None:
        radar = PitchRadar(None)
        radar.homography = np.array(
            [
                [PITCH_LENGTH_CM / FRAME_WIDTH, 0.0, 0.0],
                [0.0, PITCH_WIDTH_CM / FRAME_HEIGHT, 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        radar.calibration_confidence = 0.95
        frame = np.full((FRAME_HEIGHT, FRAME_WIDTH, 3), (45, 112, 45), dtype=np.uint8)
        inside = _player(220, 100, 1)
        outside = _player(650, 100, 2)

        output = PitchOccupancyFilter().filter(0, [inside, outside], frame, radar)

        self.assertEqual([1], [item.raw_track_id for item in output])

    def test_frame_confidence_reports_reliable_metric_state(self) -> None:
        radar = PitchRadar(None)
        radar.homography = np.eye(3, dtype=np.float64)
        radar.calibration_confidence = 0.84
        radar.calibration_source = "automatic"
        radar.last_calibrated_frame = 0

        radar.record_frame_confidence(0)

        self.assertTrue(radar.frame_confidence[0]["reliable"])
        self.assertEqual("automatic", radar.frame_confidence[0]["source"])


class BallTrackingV2Tests(unittest.TestCase):
    def test_penalty_spot_is_rejected_even_when_detected_as_ball(self) -> None:
        ball_filter = BallStaticFilter(static_hits=2)
        marker = AnalysisObject(
            track_id=1,
            raw_track_id=1,
            class_name="ball",
            bbox=[100, 100, 118, 116],
            confidence=0.8,
        )

        output = ball_filter.filter(
            0,
            [marker],
            [],
            FRAME_WIDTH,
            pitch_transform=lambda _point: (PENALTY_SPOT_DISTANCE_CM, PITCH_WIDTH_CM / 2),
        )

        self.assertEqual([], output)
        self.assertEqual(1, ball_filter.summary()["penalty_spot_rejections"])

    def test_ball_track_interpolates_short_detector_gap(self) -> None:
        tracker = BallTrackerV2(max_interpolation_frames=3)
        first = AnalysisObject(4, "ball", [100, 100, 114, 114], 0.9, raw_track_id=4)
        second = AnalysisObject(4, "ball", [108, 100, 122, 114], 0.9, raw_track_id=4)

        self.assertEqual([], tracker.update(0, [first], [], FRAME_WIDTH))
        observed = tracker.update(1, [second], [], FRAME_WIDTH)
        predicted = tracker.update(2, [], [], FRAME_WIDTH)

        self.assertEqual(1, observed[0].track_id)
        self.assertFalse(observed[0].is_predicted)
        self.assertTrue(predicted[0].is_predicted)

    def test_ball_track_reacquires_after_a_long_detector_gap(self) -> None:
        tracker = BallTrackerV2(max_interpolation_frames=3)
        first = AnalysisObject(1, "ball", [100, 100, 114, 114], 0.9)
        second = AnalysisObject(2, "ball", [108, 100, 122, 114], 0.9)
        reacquire_first = AnalysisObject(3, "ball", [400, 220, 414, 234], 0.9)
        reacquire_second = AnalysisObject(4, "ball", [408, 220, 422, 234], 0.9)

        self.assertEqual([], tracker.update(0, [first], [], FRAME_WIDTH))
        self.assertEqual(1, len(tracker.update(2, [second], [], FRAME_WIDTH)))
        self.assertEqual([], tracker.update(10, [reacquire_first], [], FRAME_WIDTH))
        reacquired = tracker.update(12, [reacquire_second], [], FRAME_WIDTH)

        self.assertEqual(1, len(reacquired))
        self.assertFalse(reacquired[0].is_predicted)
        self.assertEqual(1, tracker.summary()["expired_track_resets"])
        self.assertEqual(2, tracker.summary()["reinitializations"])

    def test_possession_is_confirmed_for_nearest_player(self) -> None:
        tracker = PossessionTracker(confirmation_frames=1)
        player = _player(180, 100, 7)
        ball = AnalysisObject(1, "ball", [198, 212, 212, 226], 0.9, raw_track_id=1)
        pitch_transform = lambda point: (point[0] * 10.0, point[1] * 10.0)

        owner, team = tracker.update(0, [player], [ball], {7: 2}, pitch_transform)

        self.assertEqual(7, owner)
        self.assertEqual(2, team)


class ModelBundleSelectionTests(unittest.TestCase):
    def test_pitch_gate_prefers_wide_stable_geometry(self) -> None:
        runner = MatchAnalysisPlusRunner()
        narrow = {
            "wide_view_frames": 0,
            "valid_homographies": 3,
            "mean_inlier_ratio": 1.0,
            "median_reprojection_error_cm": 15.0,
            "visible_keypoints_total": 40,
        }
        wide = {
            "wide_view_frames": 1,
            "valid_homographies": 3,
            "mean_inlier_ratio": 0.9,
            "median_reprojection_error_cm": 60.0,
            "visible_keypoints_total": 24,
        }

        self.assertGreater(
            runner._pitch_preview_rank(wide),
            runner._pitch_preview_rank(narrow),
        )

    def test_model_candidates_are_deduplicated_by_resolved_path(self) -> None:
        runner = MatchAnalysisPlusRunner()
        path = runner.general_model_path

        candidates = runner._unique_model_paths(
            [("first", path), ("duplicate", path), ("missing", None)]
        )

        self.assertEqual([("first", path)], candidates)


if __name__ == "__main__":
    unittest.main()
