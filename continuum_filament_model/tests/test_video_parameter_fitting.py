from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from continuum_filament_model.benchmarks import video_parameter_fitting as fitting


class VideoParameterFittingTests(unittest.TestCase):
    def _write_centerline(
        self,
        directory: Path,
        lengths: list[float],
        *,
        censored: set[int] | None = None,
        widths: list[float] | None = None,
        flags: dict[int, str] | None = None,
    ) -> Path:
        censored = censored or set()
        flags = flags or {}
        path = directory / "centerline.csv"
        fields = ["time", "filament_id", "point_id", "x", "y", "quality", "frame", "quality_flags", "censor"]
        if widths is not None:
            fields.append("width")
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
            writer.writeheader()
            for frame, length in enumerate(lengths):
                # Five points make the length calculation independent of the
                # minimum-point guard while retaining an exact straight fixture.
                for point_id, x in enumerate(np.linspace(0.0, length, 5)):
                    row = {
                        "time": frame * 0.2,
                        "filament_id": "filament-0000",
                        "point_id": point_id,
                        "x": x,
                        "y": 0.0,
                        "quality": 1.0,
                        "frame": frame,
                        "quality_flags": flags.get(frame, "ok"),
                        "censor": int(frame in censored),
                    }
                    if widths is not None:
                        row["width"] = widths[frame]
                    writer.writerow(row)
        return path

    def _write_model(self, directory: Path, *, with_parameters: bool = False) -> Path:
        path = directory / ("model.npz" if with_parameters else "model.csv")
        points = np.asarray([[0.0, 0.0], [1.0, 0.1], [2.0, 0.0]])
        if not with_parameters:
            with path.open("w", encoding="utf-8") as handle:
                handle.write("time,point_id,x,y\n")
                for point_id, (x, y) in enumerate(points):
                    handle.write(f"0,{point_id},{x},{y}\n")
                    handle.write(f"0.2,{point_id},{x},{y}\n")
            return path
        metadata = {"parameters": {"axial_stiffness": 100.0, "bending_stiffness": 0.4, "reference_length": 1.0, "diameter": 0.08}}
        np.savez_compressed(
            path,
            positions=np.concatenate([points, points]),
            position_offsets=np.asarray([0, 3, 6]),
            rest_lengths=np.ones(4),
            rest_offsets=np.asarray([0, 2, 4]),
            times=np.asarray([0.0, 0.2]),
            steps=np.asarray([0, 1]),
            metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
        )
        return path

    def test_exponential_growth_is_deterministic_and_has_confidence_interval(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            lengths = [2.0 * np.exp(0.15 * frame * 0.2) for frame in range(6)]
            centerline = self._write_centerline(root, lengths)
            first = fitting.fit_observation(centerline, config={"coordinate_scale": 1.0})
            second = fitting.fit_observation(centerline, config={"coordinate_scale": 1.0})
            self.assertEqual(first["growth"], second["growth"])
            self.assertEqual(first["growth"]["selected_model"], "exponential")
            self.assertEqual(first["growth"]["selection_criterion"], "normalized_rmse")
            self.assertEqual(first["growth"]["growth_rate_ci_method"], "Huber_IRLS_weighted_covariance_1.96_standard_errors")
            self.assertAlmostEqual(first["growth"]["growth_rate"], 0.15, places=10)
            self.assertEqual(len(first["growth"]["growth_rate_ci95"]), 2)
            self.assertEqual(first["population"]["eligible_rows"], 6)
            self.assertTrue(all(abs(row["residual"]) < 1.0e-9 for row in first["growth_residual_rows"]))

    def test_censored_frames_are_excluded_from_growth_population(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            lengths = [2.0 * np.exp(0.1 * frame * 0.2) for frame in range(6)]
            centerline = self._write_centerline(root, lengths, censored={2}, flags={4: "branched_component"})
            report = fitting.fit_observation(centerline)
            self.assertEqual(report["population"]["population_rows"], 6)
            self.assertEqual(report["population"]["eligible_rows"], 4)
            self.assertEqual(report["population"]["censored_rows"], 2)
            self.assertIn("branched_component", report["population"]["excluded_reason_counts"])
            self.assertIn("quality_censor_flag", report["population"]["excluded_reason_counts"])
            self.assertAlmostEqual(report["growth"]["growth_rate"], 0.1, places=10)

    def test_width_proxy_and_missing_width_are_explicit(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            centerline = self._write_centerline(root, [2.0, 2.1, 2.2], widths=[0.08, 0.10, 0.12])
            report = fitting.fit_observation(centerline, config={"width_scale": 2.0})
            self.assertEqual(report["diameter"]["status"], "proxy_estimate")
            self.assertAlmostEqual(report["diameter"]["diameter_proxy"], 0.2)
            self.assertEqual(report["identification"]["diameter_proxy"]["status"], "proxy_estimate")
            no_width_root = root / "no_width"
            no_width_root.mkdir()
            no_width = self._write_centerline(no_width_root, [2.0, 2.1, 2.2])
            no_width_report = fitting.fit_observation(no_width)
            self.assertEqual(no_width_report["diameter"]["status"], "unidentifiable_no_width_field")

    def test_shape_metrics_use_frechet_and_curvature_with_explicit_calibration(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            centerline = root / "centerline.csv"
            with centerline.open("w", encoding="utf-8") as handle:
                handle.write("time,filament_id,point_id,x,y,quality,frame,quality_flags,censor\n")
                points = [(0.0, 0.0), (1.0, 0.1), (2.0, 0.0)]
                for frame, time_s in enumerate((0.0, 0.2)):
                    for point_id, (x, y) in enumerate(points):
                        handle.write(f"{time_s},filament-0000,{point_id},{x},{y},1,{frame},ok,0\n")
            model = self._write_model(root)
            report = fitting.fit_observation(
                centerline,
                model_paths=[model],
                config={"pixel_per_model_unit": 1.0, "max_time_error_s": 0.01},
            )
            self.assertEqual(report["shape"]["status"], "computed")
            self.assertEqual(report["shape"]["eligible_rows"], 2)
            self.assertAlmostEqual(report["shape"]["selected_candidate"]["frechet_distance_px_median"], 0.0)
            self.assertAlmostEqual(report["shape"]["selected_candidate"]["curvature_mse_px_inv2_median"], 0.0)

    def test_shape_metrics_are_suppressed_without_calibration(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            centerline = root / "centerline.csv"
            with centerline.open("w", encoding="utf-8") as handle:
                handle.write("time,filament_id,point_id,x,y,quality,frame,quality_flags,censor\n")
                for point_id, x in enumerate((0.0, 1.0, 2.0)):
                    handle.write(f"0,filament-0000,{point_id},{x},0,1,0,ok,0\n")
            report = fitting.fit_observation(centerline, model_paths=[self._write_model(root)])
            self.assertEqual(report["shape"]["status"], "no_computed_metrics")
            self.assertEqual(report["shape"]["candidates"][0]["eligible_rows"], 0)
            self.assertEqual(report["frame_rows"][0]["shape_reason"], "pixel_per_model_unit_not_specified")

    def test_parameterized_model_reports_conditional_chi_and_diameter(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            centerline = self._write_centerline(root, [2.0, 2.0, 2.0], widths=[0.08, 0.08, 0.08])
            model = self._write_model(root, with_parameters=True)
            report = fitting.fit_observation(
                centerline,
                model_paths=[model],
                config={"pixel_per_model_unit": 1.0, "width_unit": "pixel"},
            )
            self.assertAlmostEqual(report["identification"]["chi"]["estimate"], 0.001)
            self.assertEqual(report["identification"]["chi"]["status"], "conditional_on_selected_model_trajectory")
            comparison = report["identification"]["diameter_proxy"]["model_comparison"]
            self.assertEqual(comparison["status"], "computed")
            self.assertAlmostEqual(comparison["model_minus_observed"], 0.0)

    def test_manifest_resolution_uses_data_root_and_detects_hash(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            extraction = root / "extraction"
            extraction.mkdir()
            centerline = self._write_centerline(extraction, [2.0, 2.1, 2.2])
            manifest = root / "gray5_manifest.json"
            manifest.write_text(json.dumps({
                "input": {"logical_id": "gray5.mp4"},
                "full_period_run": {"artifacts": {"centerline": {"path": "centerline.csv", "bytes": centerline.stat().st_size, "sha256": fitting.sha256_file(centerline)}}},
            }), encoding="utf-8")
            loaded = fitting.load_observation(manifest, data_root=extraction)
            self.assertEqual(loaded["source"]["status"], "resolved")
            self.assertTrue(loaded["source"]["integrity"]["hash_match"])
            self.assertEqual(loaded["source"]["artifact_integrity"]["centerline"]["status"], "verified")
            self.assertEqual(loaded["population"]["population_rows"], 3)

    def test_declared_centerline_and_companion_hash_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            extraction = root / "extraction"
            extraction.mkdir()
            centerline = self._write_centerline(extraction, [2.0, 2.1, 2.2])
            summary = extraction / "observation_summary.csv"
            summary.write_text("frame,time,filament_id,censor\n0,0,filament-0000,0\n", encoding="utf-8")
            manifest = root / "manifest.json"
            manifest.write_text(json.dumps({
                "input": {"logical_id": "gray5.mp4"},
                "full_period_run": {"artifacts": {
                    "centerline": {"path": "centerline.csv", "bytes": centerline.stat().st_size, "sha256": "wrong"},
                    "observation_summary": {"path": "observation_summary.csv", "bytes": summary.stat().st_size, "sha256": "wrong-summary"},
                }},
            }), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "integrity_mismatch"):
                fitting.load_observation(manifest, data_root=extraction)

            manifest.write_text(json.dumps({
                "input": {"logical_id": "gray5.mp4"},
                "full_period_run": {"artifacts": {
                    "centerline": {"path": "centerline.csv", "bytes": centerline.stat().st_size, "sha256": fitting.sha256_file(centerline)},
                    "observation_summary": {"path": "observation_summary.csv", "bytes": summary.stat().st_size, "sha256": "wrong-summary"},
                }},
            }), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "integrity_mismatch: observation_summary"):
                fitting.load_observation(manifest, data_root=extraction)

    def test_leading_unknown_frame_is_retained_in_selected_population(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            extraction = root / "extraction"
            extraction.mkdir()
            centerline = extraction / "centerline.csv"
            with centerline.open("w", encoding="utf-8") as handle:
                handle.write("time,filament_id,point_id,x,y,quality,frame,quality_flags,censor\n")
                for point_id, x in enumerate((0.0, 1.0, 2.0)):
                    handle.write(f"1,filament-0000,{point_id},{x},0,1,1,ok,0\n")
            (extraction / "lineage.csv").write_text(
                "frame,time,filament_id,status,censor,details\n1,1,filament-0000,matched,0,\n",
                encoding="utf-8",
            )
            (extraction / "metadata.json").write_text(json.dumps({
                "video": {"fps": 1.0}, "run": {"frame_range": {"first": 0, "last": 1, "stride": 1}},
            }), encoding="utf-8")
            manifest = root / "manifest.json"
            manifest.write_text(json.dumps({
                "input": {"logical_id": "gray5.mp4"},
                "selected_filament_id": "filament-0000",
                "full_period_run": {"artifacts": {"centerline": {"path": "centerline.csv"}}},
            }), encoding="utf-8")
            loaded = fitting.load_observation(manifest, data_root=extraction)
            records = loaded["records"]
            self.assertEqual(loaded["population"]["population_rows"], 2)
            self.assertEqual(records[0]["filament_id"], "unknown")
            self.assertFalse(records[0]["eligible"])
            self.assertEqual(records[0]["exclusion_reason"], "missing_observation_lineage")

    def test_diameter_model_comparison_requires_declared_common_units(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            centerline = self._write_centerline(root, [2.0, 2.0, 2.0], widths=[0.08, 0.08, 0.08])
            report = fitting.fit_observation(
                centerline,
                model_paths=[self._write_model(root, with_parameters=True)],
                config={"pixel_per_model_unit": 1.0},
            )
            comparison = report["identification"]["diameter_proxy"]["model_comparison"]
            self.assertEqual(comparison["status"], "not_computed_unit_mismatch")
            self.assertIsNone(comparison["model_minus_observed"])

    def test_run_suite_writes_compact_summary_and_reproducibility_manifest(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            centerline = self._write_centerline(root, [2.0, 2.1, 2.2])
            output = root / "results"
            compact = fitting.run_suite([centerline], output, config={"coordinate_scale": 1.0})
            self.assertEqual(len(compact["reports"]), 1)
            self.assertTrue((output / "compact_summary.json").is_file())
            self.assertTrue((output / "fit_summary.csv").is_file())
            self.assertTrue((output / "reproducibility_manifest.json").is_file())
            self.assertTrue((output / "centerline_frame_fits.csv").is_file())
            fields = (output / "centerline_frame_fits.csv").read_text(encoding="utf-8").splitlines()[0]
            self.assertIn("growth_residual", fields)


if __name__ == "__main__":
    unittest.main()
