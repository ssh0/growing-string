from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from continuum_filament_model.benchmarks.free_growth_buckling_stage2 import (
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
            self.assertTrue((Path(directory) / "compact_manifest.json").is_file())

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
