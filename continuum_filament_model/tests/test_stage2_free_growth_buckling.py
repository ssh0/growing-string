from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from continuum_filament_model.benchmarks.free_growth_buckling_stage2 import (
    load_config,
    load_config_from_mapping,
    run_suite,
)


class Stage2FreeGrowthBucklingTest(unittest.TestCase):
    def _config(self) -> dict:
        return {
            "base": {
                "n_nodes": 5,
                "dt": 0.001,
                "t_end": 0.004,
                "max_report_rows": 16,
            },
            "fixtures": [
                {"name": "fixture", "overrides": {"growth_rate": 0.02, "amplitude": 0.02}},
            ],
            "refinement": {
                "base_fixture": "fixture",
                "n_nodes": [5, 7],
                "dt_values": [0.001, 0.0005],
            },
            "replicates": {
                "fixtures": ["fixture"],
                "seeds": [17, 23],
                "noise_fraction": 0.1,
            },
            "video_model_case": "fixture",
        }

    def test_default_config_has_independent_axial_and_drag_contrasts(self):
        config = load_config(
            Path(__file__).resolve().parents[1] / "benchmarks" / "configs" / "stage2_free_free.json"
        )
        conditions = config["contrast_conditions"]
        self.assertEqual({item["factor"] for item in conditions}, {"axial_stiffness", "drag_density"})
        self.assertEqual(len(conditions), 4)
        self.assertNotIn("output_policy", config)
        fixture = next(item for item in config["fixtures"] if item["name"] == "fast_growth_low_bend")
        baseline = dict(config["base"])
        baseline.update(fixture["overrides"])
        controlled = ("growth_rate", "bending_stiffness", "amplitude", "n_nodes", "dt")
        for condition in conditions:
            effective = dict(baseline)
            effective.update(condition["overrides"])
            self.assertEqual(condition["base_fixture"], "fast_growth_low_bend")
            self.assertEqual(set(condition["overrides"]), {condition["factor"]})
            for key in controlled:
                self.assertEqual(effective[key], baseline[key])

    def test_suite_keeps_free_free_diagnostics_and_replicates_separate(self):
        config = load_config_from_mapping(self._config())
        with tempfile.TemporaryDirectory() as directory:
            report = run_suite(config, Path(directory))
            self.assertEqual(report["boundary"], "free/free")
            self.assertFalse(report["contact_enabled"])
            self.assertEqual(report["deterministic_fixture_count"], 1)
            self.assertEqual(report["numerical_refinement_count"], 4)
            self.assertEqual(len(report["deterministic_fixtures"]), 1)
            self.assertEqual(len(report["numerical_refinements"]), 4)
            self.assertTrue(all(row["run_kind"] == "deterministic_fixture" for row in report["deterministic_fixtures"]))
            self.assertTrue(all(row["run_kind"] == "numerical_refinement" for row in report["numerical_refinements"]))
            self.assertEqual(report["exploratory_replicate_count"], 2)
            self.assertFalse(any(row["phase_boundary_claim"] for row in report["records"]))
            deterministic = next(row for row in report["records"] if row["run_kind"] == "deterministic_fixture")
            replicate = next(row for row in report["records"] if row["run_kind"] == "exploratory_replicate")
            self.assertIsNone(deterministic["seed"])
            self.assertEqual(replicate["trial"], 1)
            run_summary = next(item for item in report["results"] if item["run_name"] == "fixture")
            self.assertGreaterEqual(len(run_summary["endpoint_trajectory"]), 2)
            self.assertEqual(run_summary["accepted_steps"], 4)
            observable = run_summary["observables"][0]
            for key in (
                "reference_length",
                "contour_length",
                "axial_force_compression_proxy",
                "endpoint_force_residual_norm_max",
                "endpoint_moment_residual_norm_max",
                "max_transverse_amplitude",
                "rms_curvature",
                "first_mode_fraction",
                "energy_total",
                "growth_work_cumulative",
                "dissipation_cumulative",
                "accepted_dt",
            ):
                self.assertIn(key, observable)
            self.assertIn("event_sequence_hash", run_summary["events"])
            self.assertIn("dimensionless_groups", run_summary)
            self.assertGreater(run_summary["dimensionless_groups"]["G_s"], 0.0)
            self.assertIn("chi", run_summary["dimensionless_groups"])
            self.assertNotIn("bending_to_axial_ratio", run_summary["dimensionless_groups"])
            persisted = json.loads((Path(directory) / "compact_summary.json").read_text(encoding="utf-8"))
            summary_fields = (Path(directory) / "summary.csv").read_text(encoding="utf-8").splitlines()[0].split(",")
            self.assertIn("G_b", summary_fields)
            self.assertNotIn("growth_bending_number", summary_fields)
            persisted_run = next(item for item in persisted["results"] if item["run_name"] == "fixture")
            self.assertNotIn("observables", persisted_run)
            self.assertLessEqual(len(persisted_run["endpoint_trajectory"]), 32)
            self.assertGreaterEqual(len(persisted_run["diagnostic_observations"]), 2)
            self.assertLessEqual(len(persisted_run["diagnostic_observations"]), 3)
            self.assertTrue((Path(directory) / "compact_manifest.json").is_file())

    def test_axial_and_drag_contrasts_preserve_other_experiment_axes(self):
        config = self._config()
        config["contrast_conditions"] = [
            {
                "name": "soft_axial",
                "base_fixture": "fixture",
                "factor": "axial_stiffness",
                "overrides": {"axial_stiffness": 2.0},
            },
            {
                "name": "high_drag",
                "base_fixture": "fixture",
                "factor": "drag_density",
                "overrides": {"drag_density": 2.0},
            },
        ]
        with tempfile.TemporaryDirectory() as directory:
            report = run_suite(config, Path(directory))
            self.assertEqual(report["parameter_contrast_count"], 2)
            contrasts = [row for row in report["records"] if row["run_kind"] == "parameter_contrast"]
            self.assertEqual({row["run_name"] for row in contrasts}, {"soft_axial", "high_drag"})
            for row in contrasts:
                self.assertFalse(row["phase_boundary_claim"])
                self.assertEqual(row["n_nodes"], 5)
                self.assertEqual(row["dt"], 0.001)
                self.assertEqual(row["growth_rate"], 0.02)
                self.assertEqual(row["bending_stiffness"], 0.1)
                result = next(item for item in report["results"] if item["run_name"] == row["run_name"])
                self.assertEqual(result["contrast_factor"], row["contrast_factor"])
                self.assertEqual(result["contrast_value"], row["contrast_value"])
                self.assertEqual(result["provenance"]["base_fixture"], "fixture")
                self.assertIn("chi", result["dimensionless_groups"])
                self.assertIn("G_b", result["dimensionless_groups"])
                self.assertFalse(result["contact_enabled"])

    def test_run_names_cannot_escape_output_directory(self):
        config = self._config()
        config["fixtures"][0]["name"] = "../escape"
        with self.assertRaises(ValueError):
            load_config_from_mapping(config)

    def test_failed_zero_progress_run_is_serializable_and_unresolved(self):
        config = self._config()
        config["base"]["max_displacement_fraction"] = 1.0e-12
        config["base"]["max_retries"] = 0
        with tempfile.TemporaryDirectory() as directory:
            report = run_suite(config, Path(directory), video_path=Path(directory) / "missing.mp4")
            result = next(item for item in report["results"] if item["run_name"] == "fixture")
            self.assertEqual(result["classification"]["label"], "numerically-unresolved")
            self.assertEqual(result["classification"]["unresolved_reason_category"], "numerical_nonconvergence")
            self.assertEqual(len(result["observables"]), 1)
            self.assertEqual(report["video_comparison"]["status"], "input_missing")
            self.assertEqual(report["video_comparison"]["numerical_unresolved"]["status"], "numerically_unresolved")
            self.assertEqual(report["video_comparison"]["numerical_unresolved"]["category"], "numerical_nonconvergence")

    def test_tiny_positive_end_time_executes_one_step(self):
        config = self._config()
        config["base"]["t_end"] = 5.0e-13
        with tempfile.TemporaryDirectory() as directory:
            report = run_suite(config, Path(directory))
        result = next(item for item in report["results"] if item["run_name"] == "fixture")
        self.assertEqual(result["accepted_steps"], 1)
        self.assertEqual(len(result["observables"]), 2)
        self.assertGreater(result["observables"][-1]["time"], 0.0)

    def test_zero_amplitude_has_no_dominant_mode(self):
        config = self._config()
        config["base"]["amplitude"] = 0.0
        config["fixtures"][0]["overrides"]["amplitude"] = 0.0
        with tempfile.TemporaryDirectory() as directory:
            report = run_suite(config, Path(directory))
        result = next(item for item in report["results"] if item["run_name"] == "fixture")
        self.assertIsNone(result["observables"][0]["dominant_mode"])
        self.assertEqual(result["observables"][0]["first_mode_fraction"], 0.0)

    def test_single_interior_replicate_noise_stays_bounded(self):
        config = self._config()
        config["base"]["n_nodes"] = 3
        config["refinement"]["n_nodes"] = [3, 5]
        with tempfile.TemporaryDirectory() as directory:
            report = run_suite(config, Path(directory))
            replicate = next(
                item for item in report["results"] if item["run_kind"] == "exploratory_replicate"
            )
            initial = replicate["observables"][0]
            self.assertLess(initial["initial_transverse_amplitude"], 0.1)
            self.assertLess(initial["contour_length"], 3.0)
            self.assertFalse(replicate["contact_enabled"])

    def test_failed_run_preserves_all_accepted_partial_states(self):
        from continuum_filament_model.benchmarks import free_growth_buckling_stage2 as stage2
        from growing_filament.model import ModelError, OverdampedGrowingFilament

        class FailingAfterTwoAcceptedSteps(OverdampedGrowingFilament):
            def step(self, dt=None):
                if self.accepted_steps >= 2:
                    raise ModelError("forced partial-trajectory failure")
                return super().step(dt)

        config = self._config()
        config["base"]["growth_rate"] = 0.02
        with tempfile.TemporaryDirectory() as directory, patch.object(
            stage2, "OverdampedGrowingFilament", FailingAfterTwoAcceptedSteps
        ):
            report = run_suite(config, Path(directory))
        result = next(item for item in report["results"] if item["run_name"] == "fixture")
        self.assertIsNotNone(result["failure_reason"])
        self.assertEqual(result["accepted_steps"], 2)
        self.assertEqual(len(result["observables"]), 3)
        self.assertEqual([row["step"] for row in result["observables"]], [0, 1, 2])
        self.assertTrue(all(row["time"] < next_row["time"] for row, next_row in zip(result["observables"], result["observables"][1:])))

    def test_calibrated_comparison_reports_model_inadequacy_candidates_separately(self):
        from continuum_filament_model.benchmarks import free_growth_buckling_stage2 as stage2

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "output"
            video = root / "input.mp4"
            video.write_bytes(b"video")
            extraction = {
                "manifest": {
                    "input": {"metadata": {"path": str(video.resolve())}},
                    "processed_frames": 2,
                    "candidate_count": 2,
                    "candidate_censor_count": 1,
                    "selected_filament_id": "filament-0000",
                },
                "validation": {"valid": True},
                "summary_rows": [{"frame": 0}, {"frame": 1}],
            }
            comparison = {
                "rows": [
                    {
                        "frame": 0,
                        "time": 0.0,
                        "filament_id": "filament-0000",
                        "metric_status": "computed",
                        "censor": 0,
                        "shape_rmse_px": 8.0,
                        "model_length_px": 100.0,
                        "length_difference_px": 4.0,
                    },
                    {
                        "frame": 1,
                        "time": 1.0,
                        "filament_id": "filament-0000",
                        "metric_status": "not_computed_censored",
                        "censor": 1,
                        "shape_rmse_px": 30.0,
                        "model_length_px": 100.0,
                        "length_difference_px": 30.0,
                    },
                ],
                "summary": {
                    "calibration_status": "calibrated",
                    "eligible_rows": 1,
                    "censored_rows": 1,
                    "excluded_from_metric_denominator": 1,
                    "observation_dir": str((output / "_video_artifacts").resolve()),
                    "model_path": str((output / "_runs" / "fixture" / "trajectory.npz").resolve()),
                },
            }
            with patch.object(stage2, "run_pipeline", return_value=extraction), patch.object(
                stage2, "compare_with_model", return_value=comparison
            ):
                record = stage2.run_video_comparison(
                    video,
                    output,
                    output / "_runs" / "fixture" / "trajectory.npz",
                    self._config(),
                    registration={"pixel_per_model_unit": 10.0, "time_scale": 1.0, "time_offset": 0.0},
                )
            self.assertEqual(record["holdout"]["status"], "calibrated_comparison")
            self.assertTrue(record["data_quality"]["censor"])
            self.assertEqual(record["model_inadequacy"]["status"], "candidate")
            self.assertEqual(record["model_inadequacy"]["candidate_count"], 1)
            self.assertIn("shape_rmse_px_above_threshold", record["model_inadequacy"]["candidates"][0]["reasons"])
            self.assertEqual(record["model_inadequacy"]["thresholds"]["shape_rmse_px"], 5.0)

    def test_invalid_centerline_validation_suppresses_calibrated_comparison(self):
        from continuum_filament_model.benchmarks import free_growth_buckling_stage2 as stage2

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "output"
            video = root / "input.mp4"
            video.write_bytes(b"video")
            extraction = {
                "manifest": {
                    "input": {"metadata": {"path": str(video.resolve())}},
                    "processed_frames": 1,
                    "candidate_count": 1,
                    "candidate_censor_count": 0,
                    "selected_filament_id": "filament-0000",
                },
                "validation": {"valid": False, "errors": ["time is not monotonic"]},
                "summary_rows": [{"frame": 0}],
            }
            with patch.object(stage2, "run_pipeline", return_value=extraction), patch.object(
                stage2, "compare_with_model", side_effect=AssertionError("invalid input must not be compared")
            ):
                record = stage2.run_video_comparison(
                    video,
                    output,
                    output / "model.npz",
                    self._config(),
                    registration={"pixel_per_model_unit": 10.0, "time_scale": 1.0, "time_offset": 0.0},
                )
            self.assertEqual(record["holdout"]["status"], "calibrated_but_input_unusable")
            self.assertEqual(record["holdout"]["comparison"]["eligible_rows"], 0)
            self.assertFalse(record["holdout"]["comparison"]["comparison_performed"])
            self.assertEqual(record["model_inadequacy"]["status"], "not_assessed_no_eligible_rows")
            self.assertEqual(record["holdout"]["quantitative_model_overlay"], "suppressed_registered_input_quality")
            self.assertIn("centerline_contract_invalid", record["data_quality"]["reasons"])

    def test_incomplete_decode_suppresses_calibrated_comparison(self):
        from continuum_filament_model.benchmarks import free_growth_buckling_stage2 as stage2

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "output"
            video = root / "input.mp4"
            video.write_bytes(b"video")
            extraction = {
                "manifest": {
                    "input": {"metadata": {"path": str(video.resolve())}},
                    "run": {"frame_range": {"decode_complete": False}},
                    "candidate_censor_count": 0,
                    "selected_filament_id": "filament-0000",
                },
                "validation": {"valid": True},
                "summary_rows": [{"frame": 0}],
            }
            with patch.object(stage2, "run_pipeline", return_value=extraction), patch.object(
                stage2, "compare_with_model", side_effect=AssertionError("incomplete decode must not be compared")
            ):
                record = stage2.run_video_comparison(
                    video,
                    output,
                    output / "model.npz",
                    self._config(),
                    registration={"pixel_per_model_unit": 10.0, "time_scale": 1.0, "time_offset": 0.0},
                )
            self.assertEqual(record["holdout"]["status"], "calibrated_but_input_unusable")
            self.assertEqual(record["model_inadequacy"]["status"], "not_assessed_no_eligible_rows")
            self.assertIn("decode_incomplete", record["data_quality"]["reasons"])
            self.assertEqual(record["extraction"]["decode_complete"], False)

    def test_missing_time_registration_suppresses_model_comparison(self):
        from continuum_filament_model.benchmarks import free_growth_buckling_stage2 as stage2

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "output"
            video = root / "input.mp4"
            video.write_bytes(b"video")
            extraction = {
                "manifest": {
                    "input": {"metadata": {"path": str(video.resolve())}},
                    "processed_frames": 1,
                    "candidate_count": 1,
                    "candidate_censor_count": 0,
                    "selected_filament_id": "filament-0000",
                },
                "validation": {"valid": True},
                "summary_rows": [{"frame": 0}],
            }
            with patch.object(stage2, "run_pipeline", return_value=extraction), patch.object(
                stage2, "compare_with_model", side_effect=AssertionError("comparison must be suppressed")
            ):
                record = stage2.run_video_comparison(
                    video,
                    output,
                    output / "model.npz",
                    self._config(),
                    registration={"pixel_per_model_unit": 10.0},
                )
            self.assertEqual(record["holdout"]["status"], "comparison_only_unregistered")
            self.assertEqual(record["model_inadequacy"]["status"], "not_assessed_unregistered")
            self.assertEqual(record["holdout"]["comparison"]["eligible_rows"], 0)
            self.assertIn("time_scale", record["holdout"]["comparison"]["registration_missing_fields"])
            self.assertIn("time_offset", record["holdout"]["comparison"]["registration_missing_fields"])

    def test_malformed_or_nonfinite_registration_suppresses_comparison(self):
        from continuum_filament_model.benchmarks import free_growth_buckling_stage2 as stage2

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "output"
            video = root / "input.mp4"
            video.write_bytes(b"video")
            extraction = {
                "manifest": {
                    "input": {"metadata": {"path": str(video.resolve())}},
                    "processed_frames": 1,
                    "candidate_count": 1,
                    "candidate_censor_count": 0,
                    "selected_filament_id": "filament-0000",
                },
                "validation": {"valid": True},
                "summary_rows": [{"frame": 0}],
            }
            with patch.object(stage2, "run_pipeline", return_value=extraction), patch.object(
                stage2, "compare_with_model", side_effect=AssertionError("comparison must be suppressed")
            ):
                for registration, field in (
                    ({"pixel_per_model_unit": 10.0, "time_scale": "bad", "time_offset": 0.0}, "time_scale"),
                    ({"pixel_per_model_unit": 10.0, "time_scale": float("nan"), "time_offset": 0.0}, "time_scale"),
                    ({"pixel_per_model_unit": "bad", "time_scale": 1.0, "time_offset": 0.0}, "pixel_per_model_unit"),
                    ({"pixel_per_model_unit": 10.0, "time_scale": 1.0, "time_offset": 0.0, "endpoint_order": []}, "endpoint_order"),
                ):
                    with self.subTest(registration=registration):
                        record = stage2.run_video_comparison(
                            video,
                            output,
                            output / "model.npz",
                            self._config(),
                            registration=registration,
                        )
                        comparison = record["holdout"]["comparison"]
                        self.assertEqual(record["holdout"]["status"], "comparison_only_unregistered")
                        self.assertFalse(comparison["comparison_performed"])
                        self.assertEqual(comparison["eligible_rows"], 0)
                        self.assertTrue(
                            field in comparison["registration_missing_fields"]
                            or field in comparison["registration_invalid_fields"]
                        )

    def test_invalid_max_displacement_fraction_is_rejected_as_configuration_error(self):
        config = self._config()
        config["base"]["max_displacement_fraction"] = 2.0
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):
                run_suite(config, Path(directory))
            self.assertFalse((Path(directory) / "_runs").exists())

    def test_invalid_video_model_case_is_rejected_before_runs(self):
        config = self._config()
        config["video_model_case"] = "not-generated"
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):
                run_suite(config, Path(directory), video_path=Path(directory) / "missing.mp4")
            self.assertFalse((Path(directory) / "_runs").exists())

    def test_video_case_without_refinement_is_numerically_unresolved(self):
        from continuum_filament_model.benchmarks import free_growth_buckling_stage2 as stage2

        config = self._config()
        config["contrast_conditions"] = [
            {
                "name": "soft_axial",
                "base_fixture": "fixture",
                "factor": "axial_stiffness",
                "overrides": {"axial_stiffness": 2.0},
            }
        ]
        config["video_model_case"] = "soft_axial"
        with tempfile.TemporaryDirectory() as directory:
            report = run_suite(config, Path(directory))
        status = stage2._numerical_unresolved_assessment(report["results"], "soft_axial")
        self.assertEqual(status["status"], "numerically_unresolved")
        self.assertIn("video_model_refinement_missing", status["reasons"])

    def test_unsupported_replicate_seed_is_rejected_before_conversion(self):
        for seeds in ([1.5, 2], [-1, 2], [2**32, 2], [True, 2], ["101", 2]):
            with self.subTest(seeds=seeds):
                config = self._config()
                config["replicates"]["seeds"] = list(seeds)
                with tempfile.TemporaryDirectory() as directory:
                    with self.assertRaises(ValueError):
                        run_suite(config, Path(directory))
                    self.assertFalse((Path(directory) / "_runs").exists())

    def test_invalid_refinement_nodes_are_rejected_before_spec_generation(self):
        config = self._config()
        config["refinement"]["n_nodes"] = [5.5, 7]
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):
                run_suite(config, Path(directory))
            self.assertFalse((Path(directory) / "_runs").exists())

    def test_invalid_fixture_override_is_rejected_before_any_run(self):
        config = self._config()
        config["fixtures"][0]["overrides"]["axial_stiffness"] = 0.0
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):
                run_suite(config, Path(directory))
            self.assertFalse((Path(directory) / "_runs").exists())

    def test_generated_run_name_collisions_are_rejected_before_execution(self):
        config = self._config()
        config["fixtures"].append(
            {"name": "fixture_refine_n5_dt0.001", "overrides": {"growth_rate": 0.02, "amplitude": 0.02}}
        )
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):
                run_suite(config, Path(directory))
            self.assertFalse((Path(directory) / "_runs").exists())

    def test_numerical_unresolved_suppresses_model_inadequacy(self):
        from continuum_filament_model.benchmarks import free_growth_buckling_stage2 as stage2

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "output"
            video = root / "input.mp4"
            video.write_bytes(b"video")
            extraction = {
                "manifest": {
                    "input": {"metadata": {"path": str(video.resolve())}},
                    "processed_frames": 1,
                    "candidate_count": 1,
                    "candidate_censor_count": 0,
                    "selected_filament_id": "filament-0000",
                },
                "validation": {"valid": True},
                "summary_rows": [{"frame": 0}],
            }
            comparison = {
                "rows": [{
                    "frame": 0,
                    "time": 0.0,
                    "filament_id": "filament-0000",
                    "metric_status": "computed",
                    "censor": 0,
                    "shape_rmse_px": 20.0,
                    "model_length_px": 100.0,
                    "length_difference_px": 0.0,
                }],
                "summary": {
                    "eligible_rows": 1,
                    "excluded_from_metric_denominator": 0,
                    "observation_dir": str((output / "_video_artifacts").resolve()),
                    "model_path": str((output / "model.npz").resolve()),
                },
            }
            with patch.object(stage2, "run_pipeline", return_value=extraction), patch.object(
                stage2, "compare_with_model", return_value=comparison
            ):
                record = stage2.run_video_comparison(
                    video,
                    output,
                    output / "model.npz",
                    self._config(),
                    registration={"pixel_per_model_unit": 10.0, "time_scale": 1.0, "time_offset": 0.0},
                    numerical_status={
                        "status": "numerically_unresolved",
                        "category": "numerical_nonconvergence",
                        "reasons": ["selected_model_failure"],
                    },
                )
            self.assertEqual(record["holdout"]["status"], "numerical_unresolved")
            self.assertEqual(record["model_inadequacy"]["status"], "not_assessed_numerical_unresolved")
            self.assertEqual(record["numerical_unresolved"]["reasons"], ["selected_model_failure"])

    def test_video_failure_categories_are_stable_and_redacted(self):
        from continuum_filament_model.benchmarks import free_growth_buckling_stage2 as stage2

        cases = (
            (RuntimeError("required executable is not installed: ffmpeg"), "missing_ffmpeg", "execution_environment"),
            (RuntimeError("command failed (ffprobe…): Invalid data found when processing input"), "decode_or_corrupt_input", "input_quality"),
            (ValueError("unsupported format"), "invalid_format", "input_quality"),
            (RuntimeError("analysis failed"), "analysis_failure", "analysis"),
            (ImportError("No module named 'imageio'"), "missing_dependency", "execution_environment"),
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            video = root / "input.mp4"
            video.write_bytes(b"video")
            for index, (error, category, domain) in enumerate(cases):
                output = root / f"output-{index}"
                with patch.object(stage2, "run_pipeline", side_effect=error):
                    record = stage2.run_video_comparison(video, output, output / "model.npz", self._config())
                self.assertEqual(record["failure"], {"category": category, "domain": domain})
                self.assertEqual(record["error"], category)
                self.assertIsNone(record["numerical_unresolved"]["category"])
                self.assertNotIn(str(root.resolve()), json.dumps(record))

    def test_video_pipeline_failure_preserves_numerical_status(self):
        from continuum_filament_model.benchmarks import free_growth_buckling_stage2 as stage2

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            video = root / "input.mp4"
            video.write_bytes(b"video")
            numerical_status = {
                "status": "numerically_unresolved",
                "category": "numerical_nonconvergence",
                "reasons": ["selected_model_failure"],
            }
            with patch.object(stage2, "run_pipeline", side_effect=RuntimeError("analysis failed")):
                record = stage2.run_video_comparison(
                    video,
                    root / "output",
                    root / "model.npz",
                    self._config(),
                    numerical_status=numerical_status,
                )
            self.assertEqual(record["numerical_unresolved"], numerical_status)
            self.assertEqual(record["failure"]["domain"], "analysis")

    def test_video_failure_hashing_is_best_effort(self):
        from continuum_filament_model.benchmarks import free_growth_buckling_stage2 as stage2

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            video = root / "input.mp4"
            video.write_bytes(b"video")
            with patch.object(stage2, "run_pipeline", side_effect=RuntimeError("analysis failed")), patch.object(
                stage2, "sha256_file", side_effect=OSError("input became unreadable")
            ):
                record = stage2.run_video_comparison(video, root / "output", root / "model.npz", self._config())
            self.assertEqual(record["failure"], {"category": "analysis_failure", "domain": "analysis"})
            self.assertIsNone(record["input_sha256"])
            self.assertEqual(record["status"], "unusable")

    def test_missing_video_is_censored_without_substitution_or_fit(self):
        config = load_config_from_mapping(self._config())
        with tempfile.TemporaryDirectory() as directory:
            report = run_suite(
                config,
                Path(directory),
                video_path=Path(directory) / "missing.mp4",
            )
            video = report["video_comparison"]
            self.assertEqual(video["status"], "input_missing")
            self.assertTrue(video["data_quality"]["censor"])
            self.assertEqual(video["quantitative_fitting"], "suppressed")
            self.assertFalse(video["legacy_video_substitution"])
            self.assertEqual(video["calibration"]["runs"], [])
            self.assertEqual(video["holdout"]["runs"], [])
            trajectory = next(item for item in report["results"] if item.get("trajectory_path"))
            self.assertFalse(Path(trajectory["trajectory_path"]).is_absolute())
            manifest = json.loads((Path(directory) / "compact_manifest.json").read_text(encoding="utf-8"))
            self.assertIn("video_comparison_manifest.json", manifest["artifacts"])

    def test_video_compact_record_normalizes_external_paths(self):
        from continuum_filament_model.benchmarks import free_growth_buckling_stage2 as stage2

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "output"
            video = root / "source" / "input.mp4"
            video.parent.mkdir()
            video.write_bytes(b"video")
            artifact_dir = output / "_video_artifacts"
            model_path = output / "_runs" / "fixture" / "trajectory.npz"
            extraction = {
                "manifest": {
                    "input": {
                        "metadata": {"path": str(video.resolve()), "width": 10, "height": 10},
                    },
                    "processed_frames": 1,
                    "candidate_count": 1,
                    "candidate_censor_count": 0,
                    "selected_filament_id": "filament-0000",
                },
                "validation": {"valid": True},
                "summary_rows": [{"frame": 0}],
            }
            comparison = {
                "summary": {
                    "observation_dir": str(artifact_dir.resolve()),
                    "model_path": str(model_path.resolve()),
                    "model_logical_id": model_path.name,
                    "rows": 1,
                },
            }
            with patch.object(stage2, "run_pipeline", return_value=extraction), patch.object(
                stage2, "compare_with_model", return_value=comparison
            ), patch.object(stage2, "sha256_file", return_value="hash"):
                record = stage2.run_video_comparison(
                    video,
                    output,
                    model_path,
                    self._config(),
                    registration={"pixel_per_model_unit": 10.0, "time_scale": 1.0, "time_offset": 0.0},
                )
            serialized = json.dumps(record)
            persisted = (output / "video_comparison_manifest.json").read_text(encoding="utf-8")
            self.assertNotIn(str(root.resolve()), serialized)
            self.assertNotIn(str(root.resolve()), persisted)
            self.assertEqual(record["video_metadata"]["path"], video.name)
            self.assertEqual(record["holdout"]["comparison"]["observation_dir"], "_video_artifacts")
            self.assertEqual(
                record["holdout"]["comparison"]["model_path"],
                "_runs/fixture/trajectory.npz",
            )


if __name__ == "__main__":
    unittest.main()
