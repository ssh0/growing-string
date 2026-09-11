import csv
import json
import tempfile
import unittest
from pathlib import Path

from continuum_filament_model.benchmarks.free_free_convergence_gate import (
    GateError,
    load_config,
    load_config_from_mapping,
    run_gate,
)


class FreeFreeConvergenceGateTests(unittest.TestCase):
    def _config(self):
        return {
            "base": {
                "n_nodes": 5,
                "dt": 0.001,
                "t_end": 0.004,
                "max_retries": 1,
            },
            "representatives": [
                {"name": "straight", "role": "straight", "overrides": {"growth_rate": 0.0}},
                {"name": "boundary", "role": "boundary-near", "overrides": {"growth_rate": 0.01}},
                {"name": "buckled", "role": "buckled-candidate", "overrides": {"growth_rate": 0.02}},
            ],
            "controls": [],
            "temporal_refinement": {"n_nodes": 5, "dt_values": [0.001, 0.0005, 0.00025]},
            "spatial_refinement": {"n_nodes": [3, 5, 7], "dt": 0.001},
            "contrasts": [
                {"name": "growth", "base": "buckled", "factor": "growth_rate", "value": 0.03},
                {"name": "axial", "base": "buckled", "factor": "axial_stiffness", "value": 2.0},
                {"name": "bend", "base": "buckled", "factor": "bending_stiffness", "value": 0.2},
                {"name": "drag", "base": "buckled", "factor": "drag_density", "value": 2.0},
                {"name": "imperfection", "base": "buckled", "factor": "amplitude", "value": 0.04},
            ],
            "sensitivity": {"base": "buckled", "seeds": [17], "amplitude_factors": [1.0], "noise_fraction": 0.1},
        }

    def test_default_config_has_three_refinements_and_all_explicit_contrasts(self):
        config = load_config(Path(__file__).resolve().parents[1] / "benchmarks" / "configs" / "free_free_convergence_gate.json")
        self.assertEqual(len(config["temporal_refinement"]["dt_values"]), 3)
        self.assertEqual(len(config["spatial_refinement"]["n_nodes"]), 3)
        self.assertEqual(
            {item["factor"] for item in config["contrasts"]},
            {"growth_rate", "axial_stiffness", "bending_stiffness", "drag_density", "amplitude"},
        )
        self.assertEqual(config["base"].get("contact_stiffness", 0.0), 0.0)
        self.assertEqual(config["base"].get("diameter", 0.0), 0.0)

    def test_gate_separates_populations_and_records_mechanical_audit(self):
        raw_config = self._config()
        raw_config["output_policy"] = {"max_metrics_rows": 3}
        config = load_config_from_mapping(raw_config)
        with tempfile.TemporaryDirectory() as directory:
            report = run_gate(config, Path(directory))
            self.assertEqual(report["boundary"], "free/free")
            self.assertFalse(report["contact_enabled"])
            self.assertEqual(report["deterministic_fixture_population"]["temporal_count"], 9)
            self.assertEqual(report["deterministic_fixture_population"]["spatial_count"], 9)
            self.assertEqual(report["parameter_contrast_population"]["run_count"], 5)
            self.assertEqual(report["sensitivity_replicate_population"]["run_count"], 1)
            self.assertEqual(len(report["convergence"]), 6)
            self.assertTrue(all("morphology_status" in item and "mechanics_status" in item for item in report["convergence"]))
            self.assertTrue(all(item["status"] in {"resolved", "numerically-unresolved"} for item in report["convergence"]))

            temporal = json.loads((Path(directory) / "compact_summary.json").read_text(encoding="utf-8"))["deterministic_fixture_population"]["runs"]
            row = next(item for item in temporal if item["run_kind"] == "deterministic_fixture")
            for key in (
                "requested_dt", "accepted_dt_min", "rejected_trials", "event_count",
                "G_b", "G_s", "chi", "onset_time", "peak_transverse_amplitude",
                "peak_curvature_rms", "mode_spectrum", "mode_fractions", "energy_final",
                "reference_length_final", "growth_reference_energy_change_cumulative",
                "growth_work_cumulative", "dissipation_estimate_cumulative",
                "endpoint_force_residual_final", "endpoint_moment_residual_final",
                "total_length_final", "failure_reason_codes",
            ):
                self.assertIn(key, row)
            self.assertEqual(row["seed"], None)
            self.assertTrue((Path(directory) / "_runs").exists())
            with (Path(directory) / "temporal_runs.csv").open(newline="", encoding="utf-8") as stream:
                fields = next(csv.reader(stream))
            self.assertIn("growth_reference_energy_change_cumulative", fields)
            self.assertIn("mode_spectrum", fields)
            metrics = Path(directory) / "_runs" / "straight__temporal_dt0.001" / "metrics.csv"
            self.assertLessEqual(len(metrics.read_text(encoding="utf-8").splitlines()) - 1, 3)

    def test_unstable_case_is_preserved_as_numerically_unresolved(self):
        config = self._config()
        config["base"]["max_displacement_fraction"] = 1.0e-12
        config["base"]["max_retries"] = 0
        with tempfile.TemporaryDirectory() as directory:
            report = run_gate(config, Path(directory))
            runs = report["deterministic_fixture_population"]["runs"]
            straight = next(item for item in runs if item["representative"] == "straight")
            self.assertEqual(straight["label"], "numerically-unresolved")
            self.assertTrue(straight["failure_reason_codes"])
            self.assertTrue(any(item["status"] == "numerically-unresolved" for item in report["convergence"]))

    def test_contact_or_fixed_boundary_settings_are_rejected(self):
        config = self._config()
        config["base"]["contact_stiffness"] = 1.0
        with self.assertRaises(GateError):
            load_config_from_mapping(config)
        config = self._config()
        config["base"]["fixed_left"] = True
        with self.assertRaises(GateError):
            load_config_from_mapping(config)

    def test_missing_parameter_contrast_is_rejected(self):
        config = self._config()
        config["contrasts"] = config["contrasts"][:-1]
        with self.assertRaises(GateError):
            load_config_from_mapping(config)


if __name__ == "__main__":
    unittest.main()
