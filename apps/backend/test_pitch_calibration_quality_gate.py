import unittest

import numpy as np

from app.match_analysis_plus.runner import PitchRadar


class PitchCalibrationQualityGateTests(unittest.TestCase):
    def test_constrained_bootstrap_accepts_strong_partial_wide_view_only_before_lock(self) -> None:
        radar = PitchRadar(model=None)

        self.assertTrue(
            radar._is_constrained_bootstrap_candidate(
                visible_keypoints=7,
                span_x=4515.0,
                span_y=6800.0,
                source_hull_ratio=0.20,
            )
        )

        radar.homography = np.eye(3, dtype=np.float64)
        self.assertFalse(
            radar._is_constrained_bootstrap_candidate(
                visible_keypoints=7,
                span_x=4515.0,
                span_y=6800.0,
                source_hull_ratio=0.20,
            )
        )

    def test_constrained_bootstrap_rejects_narrow_local_geometry(self) -> None:
        radar = PitchRadar(model=None)

        self.assertFalse(
            radar._is_constrained_bootstrap_candidate(
                visible_keypoints=7,
                span_x=2200.0,
                span_y=2600.0,
                source_hull_ratio=0.03,
            )
        )

    def test_quality_gate_passes_stable_metric_sequence(self) -> None:
        radar = PitchRadar(model=None, stride=5)
        radar.homography = np.eye(3, dtype=np.float64)
        radar.calibration_mode = "manual_correspondences"
        radar.calibration_source = "manual"
        radar.calibration_confidence = 0.91
        radar.last_reprojection_error_cm = 35.0
        radar.last_line_alignment_score = 0.84
        radar.frame_confidence = [
            {
                "frame": frame,
                "confidence": 0.91,
                "source": "manual",
                "reliable": True,
            }
            for frame in range(100)
        ]

        gate = radar.quality_gate()

        self.assertEqual("passed", gate["status"])
        self.assertTrue(gate["metric_outputs_verified"])
        self.assertEqual([], gate["failed_conditions"])

    def test_unrecovered_camera_cut_blocks_metric_release(self) -> None:
        radar = PitchRadar(model=None, stride=5)
        radar.homography = np.eye(3, dtype=np.float64)
        radar.calibration_mode = "wide_view_keypoints"
        radar.last_reprojection_error_cm = 60.0
        radar.last_line_alignment_score = 0.72
        radar.camera_cuts = [
            {"frame": 40, "recovered_frame": None, "recovery_frames": None}
        ]
        radar.frame_confidence = [
            {
                "frame": frame,
                "confidence": 0.86,
                "source": "automatic",
                "reliable": True,
            }
            for frame in range(100)
        ]

        gate = radar.quality_gate()

        self.assertEqual("needs_manual_calibration", gate["status"])
        self.assertIn("camera_cut_recovery", gate["failed_conditions"])

    def test_camera_cut_detector_separates_hard_cut_from_small_change(self) -> None:
        radar = PitchRadar(model=None)
        dark = np.zeros((120, 160), dtype=np.uint8)
        almost_dark = np.full((120, 160), 3, dtype=np.uint8)
        bright = np.full((120, 160), 255, dtype=np.uint8)

        self.assertFalse(radar._is_camera_cut(dark, almost_dark))
        self.assertTrue(radar._is_camera_cut(dark, bright))


if __name__ == "__main__":
    unittest.main()
