from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from growing_filament.video_comparison import (
    RegistrationConfig,
    SegmentationConfig,
    compare_with_model,
    connected_components,
    component_to_centerline,
    polyline_metrics,
    segment_mask,
    validate_centerline_rows,
)


class VideoComparisonFixtureTests(unittest.TestCase):
    def setUp(self):
        self.config = SegmentationConfig(
            frame_stride=1,
            min_component_size=5,
            threshold="absolute",
            threshold_value=0.25,
            max_centerline_points=120,
        )

    def _frame(self, points, width=160, height=120, radius=1):
        frame = np.full((height, width), 220, dtype=np.uint8)
        for x, y in points:
            x, y = int(round(x)), int(round(y))
            frame[max(0, y - radius) : y + radius + 1, max(0, x - radius) : x + radius + 1] = 20
        return frame

    def test_straight_fixture_has_ordered_centerline(self):
        frame = self._frame([(x, 60) for x in np.linspace(20, 140, 80)])
        mask, _ = segment_mask(frame, self.config)
        components = connected_components(mask, self.config.min_component_size)
        self.assertEqual(len(components), 1)
        points, flags = component_to_centerline(components[0], frame.shape, self.config)
        self.assertGreater(len(points), 10)
        self.assertLess(abs(polyline_metrics(points)["length_px"] - 120.0), 8.0)
        self.assertNotIn("short_centerline", flags)

    def test_arc_and_sinusoidal_fixtures_produce_finite_metrics(self):
        for kind in ("arc", "sinusoidal"):
            t = np.linspace(0.0, 1.0, 120)
            if kind == "arc":
                points = np.column_stack((80 + 45 * np.cos(np.pi * (t + 0.1)), 60 + 45 * np.sin(np.pi * (t + 0.1))))
            else:
                points = np.column_stack((15 + 130 * t, 60 + 20 * np.sin(2 * np.pi * t)))
            frame = self._frame(points, radius=1)
            mask, _ = segment_mask(frame, self.config)
            components = connected_components(mask, self.config.min_component_size)
            self.assertTrue(components, kind)
            centreline, _ = component_to_centerline(components[0], frame.shape, self.config)
            metrics = polyline_metrics(centreline)
            self.assertTrue(np.isfinite(list(metrics.values())).all(), kind)
            self.assertGreater(metrics["length_px"], metrics["endpoint_distance_px"])

    def test_noise_and_ambiguous_components_are_detectable(self):
        frame = self._frame([(x, 60) for x in np.linspace(20, 100, 60)])
        frame = np.minimum(frame, np.random.default_rng(7).integers(0, 255, frame.shape, dtype=np.uint8))
        # A deterministic second line is intentionally ambiguous, not silently merged.
        frame[20:23, 25:95] = 20
        mask, _ = segment_mask(frame, self.config)
        components = connected_components(mask, self.config.min_component_size)
        self.assertGreaterEqual(len(components), 2)

    def test_growth_and_missing_frames_have_contract_and_censor_signals(self):
        rows = []
        for frame in range(3):
            for point_id, x in enumerate(np.linspace(10 + frame * 3, 60 + frame * 3, 8)):
                rows.append(
                    {
                        "time": frame / 10,
                        "filament_id": "filament-0000",
                        "point_id": point_id,
                        "x": x,
                        "y": 30.0,
                        "quality": 0.9,
                    }
                )
        report = validate_centerline_rows(rows, max_jump_px=20)
        self.assertTrue(report["valid"])
        rows.append(dict(rows[-1]))
        self.assertFalse(validate_centerline_rows(rows)["valid"])

    def test_calibrated_comparison_computes_registered_metrics(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "centerline.csv").write_text(
                "time,filament_id,point_id,x,y,quality,frame,coordinate_system,quality_flags,censor\n"
                "0.0,filament-0000,0,10,20,1,0,pixel,ok,0\n"
                "0.0,filament-0000,1,30,20,1,0,pixel,ok,0\n",
                encoding="utf-8",
            )
            (root / "observation_summary.csv").write_text(
                "frame,time,filament_id,n_points,component_area,length_px,endpoint_distance_px,curvature_mean_px_inv,curvature_max_px_inv,quality,quality_flags,censor\n"
                "0,0.0,filament-0000,2,10,20,20,0,0,1,ok,0\n",
                encoding="utf-8",
            )
            model = root / "model.csv"
            model.write_text("time,point_id,x,y\n0,0,0,0\n0,1,1,0\n", encoding="utf-8")
            result = compare_with_model(
                root,
                model,
                RegistrationConfig(pixel_per_model_unit=20.0, x_offset_px=10.0, y_offset_px=20.0),
                output_dir=root / "comparison",
            )
            row = result["rows"][0]
            self.assertEqual(row["metric_status"], "computed")
            self.assertAlmostEqual(row["endpoint_distance_px"], 0.0)
            self.assertAlmostEqual(row["shape_rmse_px"], 0.0)

    def test_uncalibrated_comparison_suppresses_metrics(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "centerline.csv").write_text(
                "time,filament_id,point_id,x,y,quality,frame,coordinate_system,quality_flags,censor\n"
                "0.0,filament-0000,0,0,0,1,0,pixel,ok,0\n"
                "0.0,filament-0000,1,10,0,1,0,pixel,ok,0\n",
                encoding="utf-8",
            )
            (root / "observation_summary.csv").write_text(
                "frame,time,filament_id,n_points,component_area,length_px,endpoint_distance_px,curvature_mean_px_inv,curvature_max_px_inv,quality,quality_flags,censor\n"
                "0,0.0,filament-0000,2,10,10,10,0,0,1,ok,0\n",
                encoding="utf-8",
            )
            model = root / "model.csv"
            model.write_text("time,point_id,x,y\n0,0,0,0\n0,1,1,0\n", encoding="utf-8")
            result = compare_with_model(root, model, RegistrationConfig(), output_dir=root / "comparison")
            self.assertEqual(result["summary"]["calibration_status"], "not_calibrated_metrics_suppressed")
            self.assertEqual(result["rows"][0]["metric_status"], "not_computed_uncalibrated")
            self.assertEqual(result["rows"][0]["metric_reason"], "pixel_per_model_unit_not_specified")


if __name__ == "__main__":
    unittest.main()
