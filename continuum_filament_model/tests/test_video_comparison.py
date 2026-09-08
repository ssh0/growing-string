from __future__ import annotations

import csv
import json
import shutil
import subprocess
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
    endpoint_correspondence,
    polyline_metrics,
    run_pipeline,
    skeleton_topology,
    segment_mask,
    validate_centerline_rows,
)
from growing_filament.video_comparison import _make_candidate


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

    def test_branch_and_loop_topology_are_censored_signals(self):
        branch = np.asarray(
            [(y, 3) for y in range(1, 6)] + [(3, x) for x in range(1, 6)],
            dtype=float,
        )
        branch_topology = skeleton_topology(branch)
        self.assertGreaterEqual(branch_topology["junction_count"], 1)
        self.assertGreaterEqual(branch_topology["endpoint_count"], 4)
        loop = np.asarray(
            [(1, x) for x in range(1, 5)]
            + [(4, x) for x in range(1, 5)]
            + [(y, 1) for y in range(2, 4)]
            + [(y, 4) for y in range(2, 4)],
            dtype=float,
        )
        loop_topology = skeleton_topology(loop)
        self.assertGreater(loop_topology["cycle_rank"], 0)
        self.assertEqual(loop_topology["endpoint_count"], 0)

    def test_component_truncation_and_boundary_are_censored(self):
        component = np.asarray([(y, 10) for y in range(2, 25)], dtype=int)
        candidate = _make_candidate(
            0, 0.0, "filament-0000", component, (30, 40), self.config,
            False, 1, 3, (10, 1, 20, 28),
        )
        self.assertIn("components_truncated", candidate.flags)
        self.assertIn("roi_clipped", candidate.flags)
        self.assertTrue(candidate.censor)
        image_edge = np.asarray([(y, 0) for y in range(2, 25)], dtype=int)
        edge_candidate = _make_candidate(
            0, 0.0, "filament-0001", image_edge, (30, 40), self.config,
            False, 1, 1, None,
        )
        self.assertIn("out_of_view", edge_candidate.flags)
        self.assertTrue(edge_candidate.censor)

    def test_noise_and_ambiguous_components_are_detectable(self):
        frame = self._frame([(x, 60) for x in np.linspace(20, 100, 60)])
        frame = np.minimum(frame, np.random.default_rng(7).integers(0, 255, frame.shape, dtype=np.uint8))
        # A deterministic second line is intentionally ambiguous, not silently merged.
        frame[20:23, 25:95] = 20
        mask, _ = segment_mask(frame, self.config)
        components = connected_components(mask, self.config.min_component_size)
        self.assertGreaterEqual(len(components), 2)

    @unittest.skipUnless(shutil.which("ffmpeg"), "ffmpeg is required for the synthetic video fixture")
    def test_missing_frame_lineage_and_new_lineage_are_retained(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            video = root / "missing.mp4"
            frames = []
            for x0 in (10, None, 45):
                frame = np.full((64, 64), 220, dtype=np.uint8)
                if x0 is not None:
                    frame[30:33, x0 : x0 + 15] = 20
                frames.append(frame)
            process = subprocess.Popen(
                ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "gray", "-s", "64x64", "-r", "1", "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", str(video)],
                stdin=subprocess.PIPE, stderr=subprocess.PIPE,
            )
            assert process.stdin is not None
            for frame in frames:
                process.stdin.write(frame.tobytes())
            process.stdin.close()
            self.assertEqual(process.wait(), 0)
            if process.stderr is not None:
                process.stderr.close()
            output = root / "output"
            run_pipeline(
                video, output,
                SegmentationConfig(
                    background="none", contrast="none", threshold="absolute", threshold_value=0.5,
                    frame_stride=1, min_component_size=5, max_components=1, max_jump_px=5,
                ),
            )
            with (output / "lineage.csv").open() as handle:
                lineage = list(csv.DictReader(handle))
            statuses = [row["status"] for row in lineage]
            self.assertIn("missing", statuses)
            self.assertIn("new_lineage", statuses)
            with (output / "events.csv").open() as handle:
                events = list(csv.DictReader(handle))
            self.assertTrue(any(row["event"] == "missing" for row in events))

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

    def test_endpoint_order_auto_selects_reverse(self):
        observed = np.asarray([[0.0, 0.0], [10.0, 0.0]])
        model = np.asarray([[10.0, 0.0], [0.0, 0.0]])
        correspondence = endpoint_correspondence(observed, model, "auto")
        self.assertEqual(correspondence["selected_method"], "reverse")
        self.assertAlmostEqual(correspondence["selected_endpoint_distance_px"], 0.0)

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
            self.assertEqual(result["rows"][0]["censor"], 1)

    def test_censored_comparison_has_no_quantitative_metric(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "centerline.csv").write_text(
                "time,filament_id,point_id,x,y,quality,frame,coordinate_system,quality_flags,censor\n"
                "0.0,filament-0000,0,0,0,1,0,pixel,large_jump,1\n"
                "0.0,filament-0000,1,10,0,1,0,pixel,large_jump,1\n", encoding="utf-8"
            )
            (root / "observation_summary.csv").write_text(
                "frame,time,filament_id,n_points,component_area,length_px,endpoint_distance_px,curvature_mean_px_inv,curvature_max_px_inv,quality,quality_flags,censor\n"
                "0,0.0,filament-0000,2,10,10,10,0,0,1,large_jump,1\n", encoding="utf-8"
            )
            model = root / "model.csv"
            model.write_text("time,point_id,x,y\n0,0,0,0\n0,1,1,0\n", encoding="utf-8")
            result = compare_with_model(
                root, model,
                RegistrationConfig(pixel_per_model_unit=10.0),
                output_dir=root / "comparison",
            )
            row = result["rows"][0]
            self.assertEqual(row["metric_status"], "not_computed_censored")
            self.assertIsNone(row["endpoint_distance_px"])
            self.assertEqual(result["summary"]["eligible_rows"], 0)


if __name__ == "__main__":
    unittest.main()
