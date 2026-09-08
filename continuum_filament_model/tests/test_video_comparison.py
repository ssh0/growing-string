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
    sampled_frame_range,
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

    def test_sampled_frame_range_is_mechanical(self):
        self.assertEqual(sampled_frame_range(698, 15), {"count": 47, "first": 0, "last": 690})
        self.assertEqual(sampled_frame_range(698, 15, 3), {"count": 3, "first": 0, "last": 30})

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

    def test_real_mask_topology_backend_parity(self):
        t_mask = np.zeros((35, 35), dtype=bool)
        t_mask[3:8, 3:26] = True
        t_mask[3:25, 12:17] = True
        loop_mask = np.zeros((40, 40), dtype=bool)
        loop_mask[5:30, 5:30] = True
        loop_mask[10:25, 10:25] = False
        t_component = np.argwhere(t_mask)
        loop_component = np.argwhere(loop_mask)
        t_results = []
        loop_results = []
        for backend in ("auto", "numpy"):
            config = SegmentationConfig(min_component_size=1, skeleton_backend=backend)
            t_points, t_flags = component_to_centerline(t_component, t_mask.shape, config)
            loop_points, loop_flags = component_to_centerline(loop_component, loop_mask.shape, config)
            t_results.append((t_flags, len(t_points)))
            loop_results.append((loop_flags, len(loop_points)))
        self.assertEqual(t_results[0][0], t_results[1][0])
        self.assertIn("branched_component", t_results[0][0])
        self.assertNotIn("loop_component", t_results[0][0])
        self.assertEqual(loop_results[0][0], loop_results[1][0])
        self.assertIn("loop_component", loop_results[0][0])
        self.assertTrue(all("branched_component" not in flags for flags, _ in loop_results))
        self.assertTrue(all("loop_component" not in flags for flags, _ in t_results))
        candidates = [
            _make_candidate(0, 0.0, "t", t_component, t_mask.shape, SegmentationConfig(min_component_size=1, skeleton_backend=backend), False, 1, 1, None)
            for backend in ("auto", "numpy")
        ]
        loop_candidates = [
            _make_candidate(0, 0.0, "loop", loop_component, loop_mask.shape, SegmentationConfig(min_component_size=1, skeleton_backend=backend), False, 1, 1, None)
            for backend in ("auto", "numpy")
        ]
        self.assertEqual([(c.censor, c.centerline_exported, c.flags) for c in candidates], [(c.censor, c.centerline_exported, c.flags) for c in candidates[1:]] + [(candidates[0].censor, candidates[0].centerline_exported, candidates[0].flags)])
        self.assertTrue(all(c.censor and c.centerline_exported for c in candidates))
        self.assertTrue(all(c.censor and not c.centerline_exported for c in loop_candidates))

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
    @unittest.skipUnless(shutil.which("ffmpeg"), "ffmpeg is required for the component event fixture")
    def test_component_events_are_one_frame_event_each(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            video = root / "components.mp4"
            frame = np.full((64, 64), 220, dtype=np.uint8)
            frame[8:11, 5:18] = 20
            frame[28:31, 5:18] = 20
            frame[48:51, 5:18] = 20
            process = subprocess.Popen(
                ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "gray", "-s", "64x64", "-r", "1", "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", str(video)],
                stdin=subprocess.PIPE, stderr=subprocess.PIPE,
            )
            assert process.stdin is not None
            process.stdin.write(frame.tobytes())
            process.stdin.close()
            self.assertEqual(process.wait(), 0)
            if process.stderr is not None:
                process.stderr.close()
            output = root / "output"
            run_pipeline(
                video, output,
                SegmentationConfig(background="none", contrast="none", threshold="absolute", threshold_value=0.5, frame_stride=1, min_component_size=5, max_components=2),
            )
            with (output / "events.csv").open() as handle:
                events = list(csv.DictReader(handle))
            self.assertEqual(sum(row["event"] == "ambiguous_components" for row in events), 1)
            self.assertEqual(sum(row["event"] == "components_truncated" for row in events), 1)

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
            for event_name in ("new_lineage", "reconnected_after_missing"):
                matching = [row for row in events if row["event"] == event_name]
                self.assertLessEqual(len(matching), 1, event_name)

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

    def test_leading_missing_frame_stays_in_comparison_population(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "centerline.csv").write_text(
                "time,filament_id,point_id,x,y,quality,frame,coordinate_system,quality_flags,censor\n"
                "1.0,filament-0000,0,0,0,1,1,pixel,ok,0\n"
                "1.0,filament-0000,1,10,0,1,1,pixel,ok,0\n",
                encoding="utf-8",
            )
            (root / "observation_summary.csv").write_text(
                "frame,time,filament_id,n_points,component_area,length_px,endpoint_distance_px,curvature_mean_px_inv,curvature_max_px_inv,quality,quality_flags,censor\n"
                "1,1.0,filament-0000,2,10,10,10,0,0,1,ok,0\n",
                encoding="utf-8",
            )
            (root / "lineage.csv").write_text(
                "frame,time,filament_id,status,censor,details\n"
                "1,1.0,filament-0000,matched,0,\n"
                "2,2.0,filament-0000,missing,1,no_component\n"
                "3,3.0,filament-0000,matched,0,\n",
                encoding="utf-8",
            )
            (root / "events.csv").write_text("frame,time,event,severity,details\n2,2.0,missing,censor,no_component\n", encoding="utf-8")
            (root / "manifest.json").write_text(
                json.dumps({"video": {"fps": 1.0}, "run": {"frame_range": {"first": 0, "last": 3, "stride": 1}}}),
                encoding="utf-8",
            )
            model = root / "model.csv"
            model.write_text("time,point_id,x,y\n0,0,0,0\n0,1,1,0\n", encoding="utf-8")
            result = compare_with_model(root, model, RegistrationConfig(), output_dir=root / "comparison")
            rows = result["rows"]
            self.assertEqual(result["summary"]["population_rows"], 4)
            self.assertEqual(result["summary"]["eligible_rows"] + result["summary"]["excluded_from_metric_denominator"], 4)
            self.assertEqual(rows[0]["frame"], 0)
            self.assertEqual(rows[0]["filament_id"], "unknown")
            self.assertEqual(rows[0]["metric_reason"], "missing_observation_lineage")
            self.assertEqual(rows[0]["censor"], 1)

    def test_comparison_reason_categories_and_denominator_are_distinct(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            centerline_rows = []
            summary_rows = []
            for frame, time_s, flags, censor in [
                (1, 1.0, "new_lineage", 1),
                (2, 2.0, "reconnected_after_missing", 1),
                (3, 3.0, "large_jump", 1),
                (4, 4.0, "ok", 0),
            ]:
                centerline_rows.extend([
                    f"{time_s},filament-0000,0,0,0,0.9,{frame},pixel,{flags},{censor}",
                    f"{time_s},filament-0000,1,10,0,0.9,{frame},pixel,{flags},{censor}",
                ])
                summary_rows.append(f"{frame},{time_s},filament-0000,2,10,10,10,0,0,0.9,{flags},{censor}")
            (root / "centerline.csv").write_text("time,filament_id,point_id,x,y,quality,frame,coordinate_system,quality_flags,censor\n" + "\n".join(centerline_rows) + "\n", encoding="utf-8")
            (root / "observation_summary.csv").write_text("frame,time,filament_id,n_points,component_area,length_px,endpoint_distance_px,curvature_mean_px_inv,curvature_max_px_inv,quality,quality_flags,censor\n" + "\n".join(summary_rows) + "\n", encoding="utf-8")
            (root / "lineage.csv").write_text("frame,time,filament_id,status,censor,details\n0,0.0,unknown,missing_unknown,1,no_component\n1,1.0,filament-0000,new_lineage,1,\n2,2.0,filament-0000,reconnected_after_missing,1,\n3,3.0,filament-0000,matched,1,\n4,4.0,filament-0000,matched,0,\n", encoding="utf-8")
            (root / "manifest.json").write_text(json.dumps({"video": {"fps": 1.0}, "run": {"frame_range": {"first": 0, "last": 4, "stride": 1}}}), encoding="utf-8")
            model = root / "model.csv"
            model.write_text("time,point_id,x,y\n1,0,0,0\n1,1,1,0\n3,0,0,0\n3,1,1,0\n", encoding="utf-8")
            result = compare_with_model(root, model, RegistrationConfig(pixel_per_model_unit=1.0, max_time_error_s=0.01), output_dir=root / "comparison")
            reasons = {row["metric_reason"] for row in result["rows"]}
            self.assertTrue({"missing_observation_lineage", "new_lineage_boundary", "reconnected_after_missing_boundary", "quality_censor_flag", "model_time_unmatched_or_centerline_unavailable"}.issubset(reasons))
            self.assertEqual(result["summary"]["eligible_rows"] + result["summary"]["excluded_from_metric_denominator"], result["summary"]["population_rows"])

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
