from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

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
            self.assertEqual(report["deterministic_fixture_count"], 5)
            self.assertEqual(report["exploratory_replicate_count"], 2)
            self.assertFalse(any(row["phase_boundary_claim"] for row in report["records"]))
            deterministic = next(row for row in report["records"] if row["run_kind"] == "deterministic_fixture")
            replicate = next(row for row in report["records"] if row["run_kind"] == "exploratory_replicate")
            self.assertIsNone(deterministic["seed"])
            self.assertEqual(replicate["trial"], 1)
            run_summary = next(item for item in report["results"] if item["run_name"] == "fixture")
            self.assertGreaterEqual(len(run_summary["endpoint_trajectory"]), 2)
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


if __name__ == "__main__":
    unittest.main()
