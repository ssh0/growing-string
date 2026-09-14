import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from continuum_filament_model.benchmarks.free_free_convergence_gate import (
    CaseSpec,
    GateError,
    _compare_refinement,
    _initial_state,
    _mode_observables,
    _run_case,
    load_config,
    load_config_from_mapping,
    run_gate,
)
from growing_filament.model import FilamentState


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
            "spatial_refinement": {"n_nodes": [3, 5, 7], "dt": 0.00025},
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
            self.assertEqual(report["schema_version"], "continuum-filament-free-free-convergence-gate-8")
            self.assertFalse(report["contact_enabled"])
            self.assertEqual(report["deterministic_fixture_population"]["temporal_count"], 9)
            self.assertEqual(report["deterministic_fixture_population"]["spatial_count"], 9)
            self.assertEqual(report["parameter_contrast_population"]["run_count"], 5)
            self.assertEqual(report["sensitivity_replicate_population"]["run_count"], 1)
            finest_dt = min(config["temporal_refinement"]["dt_values"])
            self.assertTrue(all(run["requested_dt"] == finest_dt for run in report["parameter_contrast_population"]["runs"]))
            self.assertTrue(all(run["requested_dt"] == finest_dt for run in report["sensitivity_replicate_population"]["runs"]))
            for population_name in ("parameter_contrast_population", "sensitivity_replicate_population"):
                population = report[population_name]
                self.assertEqual(population["numerical_contract"]["status"], "numerically-unresolved")
                self.assertEqual(population["numerical_contract"]["reason_codes"], ["not_refined_across_time_or_space"])
                self.assertTrue(all(run["numerical_status"] == "numerically-unresolved" for run in population["runs"]))
                self.assertTrue(all(run["numerical_reason_codes"] == ["not_refined_across_time_or_space"] for run in population["runs"]))
            self.assertEqual(len(report["convergence"]), 6)
            self.assertTrue(all("morphology_status" in item and "mechanics_status" in item for item in report["convergence"]))
            self.assertTrue(all(item["status"] in {"resolved", "numerically-unresolved"} for item in report["convergence"]))
            self.assertIn("endpoint_shear_residual_final", report["convergence"][0]["audit"])
            self.assertIn("mechanical_balance_residual_cumulative", report["convergence"][0]["audit"])
            self.assertIn("mechanical_balance_residual_max_abs", report["convergence"][0]["audit"])
            self.assertIn("accepted_dt_values", report["convergence"][0]["audit"])
            self.assertIn("remesh_energy_jump_cumulative", report["convergence"][0]["audit"])
            self.assertIn("event_sequence_hash", report["convergence"][0]["audit"])
            self.assertIn("python_version", report["convergence"][0]["audit"])
            self.assertIn("numpy_version", report["convergence"][0]["audit"])
            self.assertIn("rejection_reason_counts", report["convergence"][0]["audit"])

            temporal = json.loads((Path(directory) / "compact_summary.json").read_text(encoding="utf-8"))["deterministic_fixture_population"]["runs"]
            row = next(item for item in temporal if item["run_kind"] == "deterministic_fixture")
            for key in (
                "requested_dt", "accepted_dt_min", "accepted_dt_mean", "accepted_dt_values", "rejected_trials", "rejection_reason_counts", "event_count",
                "G_b", "G_s", "chi", "onset_time", "peak_transverse_amplitude",
                "peak_curvature_rms", "mode_spectrum", "mode_fractions", "energy_initial", "energy_final", "energy_span",
                "energy_stretch_initial", "energy_stretch_final", "energy_stretch_span",
                "energy_bend_initial", "energy_bend_final", "energy_bend_span",
                "mechanical_energy_change_cumulative", "reference_length_final", "growth_work_cumulative",
                "dissipation_estimate_cumulative",
                "endpoint_force_residual_final", "endpoint_moment_residual_final",
                "endpoint_shear_residual_final", "mechanical_balance_residual_cumulative", "mechanical_balance_residual_max_abs",
                "remesh_energy_jump_cumulative", "remesh_occurred", "event_sequence_hash",
                "initial_state_hash", "canonical_state_hash", "input_hash", "git_revision",
                "python_version", "numpy_version",
                "total_length_final", "failure_reason_codes", "numerical_status", "numerical_reason_codes",
            ):
                self.assertIn(key, row)
            self.assertEqual(row["seed"], None)
            self.assertTrue((Path(directory) / "_runs").exists())
            with (Path(directory) / "temporal_runs.csv").open(newline="", encoding="utf-8") as stream:
                fields = next(csv.reader(stream))
            self.assertIn("growth_work_cumulative", fields)
            self.assertIn("accepted_dt_mean", fields)
            self.assertIn("accepted_dt_values", fields)
            self.assertIn("mechanical_balance_residual_cumulative", fields)
            self.assertIn("remesh_energy_jump_cumulative", fields)
            self.assertIn("event_sequence_hash", fields)
            self.assertIn("endpoint_shear_residual_final", fields)
            self.assertNotIn("growth_reference_energy_change_cumulative", fields)
            self.assertIn("mode_spectrum", fields)
            contrast_run = next(iter(report["parameter_contrast_population"]["runs"]))
            for artifact_name in ("summary.json", "manifest.json"):
                artifact = json.loads((Path(directory) / "_runs" / contrast_run["run_name"] / artifact_name).read_text(encoding="utf-8"))
                self.assertEqual(artifact["numerical_status"], "numerically-unresolved")
                self.assertEqual(artifact["numerical_reason_codes"], ["not_refined_across_time_or_space"])
            metrics = Path(directory) / "_runs" / "straight__temporal_001" / "metrics.csv"
            self.assertLessEqual(len(metrics.read_text(encoding="utf-8").splitlines()) - 1, 3)

    def test_spatial_refinement_requires_finest_temporal_dt(self):
        config = self._config()
        config["spatial_refinement"]["dt"] = 0.001
        with self.assertRaises(GateError):
            load_config_from_mapping(config)

    def test_balance_residual_refinement_uses_residual_scale(self):
        tolerances = load_config_from_mapping(self._config())["tolerances"]

        def run(name, dt, balance):
            return {
                "run_name": name,
                "failure_reason": None,
                "classification": {"label": "sub-threshold-or-relaxing", "onset_time": None, "peak_amplitude": 1.0, "peak_curvature_rms": 1.0},
                "morphology": {"peak_mode_fractions": {"1": 1.0}},
                "mechanical": {
                    "energy_final": 1.0,
                    "growth_work_cumulative": 1.0,
                    "dissipation_estimate_cumulative": 1.0,
                    "endpoint_force_residual_final": 1.0,
                    "endpoint_moment_residual_final": 1.0,
                    "endpoint_shear_residual_final": 1.0,
                    "total_length_final": 1.0,
                    "mechanical_balance_residual_cumulative": balance,
                    "mechanical_balance_residual_max_abs": abs(balance),
                    "remesh_occurred": False,
                },
                "effective_values": {"dt": dt},
                "accepted_dt_min": dt,
                "accepted_dt_max": dt,
                "accepted_dt_mean": dt,
                "accepted_dt_values": [dt],
                "rejected_trials": 0,
                "event_count": 0,
                "provenance": {},
            }

        result = _compare_refinement(
            [run("coarse", 0.004, 0.0073), run("middle", 0.002, 0.00365), run("fine", 0.001, 0.00182)],
            tolerances,
            "temporal",
            "fine",
        )
        self.assertEqual(result["mechanics_status"], "numerically-unresolved")
        self.assertIn("mechanical_balance_residual_cumulative_out_of_tolerance", result["mechanics_reason_codes"])

    def test_temporal_refinement_rejects_collapsed_accepted_dt(self):
        tolerances = load_config_from_mapping(self._config())["tolerances"]

        def run(name, requested_dt):
            return {
                "run_name": name,
                "failure_reason": None,
                "classification": {"label": "sub-threshold-or-relaxing", "onset_time": None, "peak_amplitude": 1.0, "peak_curvature_rms": 1.0},
                "morphology": {"peak_mode_fractions": {"1": 1.0}},
                "mechanical": {
                    "energy_initial": 1.0,
                    "energy_final": 1.0,
                    "energy_span": 0.0,
                    "energy_stretch_initial": 1.0,
                    "energy_stretch_final": 1.0,
                    "energy_stretch_span": 0.0,
                    "energy_bend_initial": 1.0,
                    "energy_bend_final": 1.0,
                    "energy_bend_span": 0.0,
                    "mechanical_energy_change_cumulative": 0.0,
                    "growth_work_cumulative": 1.0,
                    "dissipation_estimate_cumulative": 1.0,
                    "endpoint_force_residual_final": 1.0,
                    "endpoint_moment_residual_final": 1.0,
                    "endpoint_shear_residual_final": 1.0,
                    "total_length_final": 1.0,
                    "mechanical_balance_residual_cumulative": 0.0,
                    "mechanical_balance_residual_max_abs": 0.0,
                    "remesh_occurred": False,
                },
                "effective_values": {"dt": requested_dt},
                "accepted_dt_min": 0.001,
                "accepted_dt_max": 0.001,
                "accepted_dt_mean": 0.001,
                "accepted_dt_values": [0.001],
                "rejected_trials": 1,
                "event_count": 1,
                "events": {"rejection_reason_counts": {"displacement_exceeded": 1}},
                "provenance": {},
            }

        result = _compare_refinement(
            [run("coarse", 0.004), run("middle", 0.002), run("fine", 0.001)],
            tolerances,
            "temporal",
            "fine",
        )
        self.assertIn("accepted_dt_not_distinct", result["mechanics_reason_codes"])
        self.assertEqual(result["status"], "numerically-unresolved")

    def test_temporal_refinement_requires_sufficient_t_end(self):
        config = self._config()
        config["base"]["t_end"] = 0.0001
        with self.assertRaises(GateError):
            load_config_from_mapping(config)

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

    def test_perturbation_changes_manifest_input_hash(self):
        config = load_config_from_mapping(self._config())
        first = CaseSpec("hash_first", "sensitivity_replicate", {}, refinement_axis="sensitivity", seed=101)
        second = CaseSpec("hash_second", "sensitivity_replicate", {}, refinement_axis="sensitivity", seed=101)
        with tempfile.TemporaryDirectory() as directory:
            first_result = _run_case(
                first,
                config["base"],
                Path(directory),
                None,
                n_nodes=5,
                dt=0.00025,
                sensitivity={"amplitude_factor": 0.5, "noise_fraction": 0.1},
            )
            second_result = _run_case(
                second,
                config["base"],
                Path(directory),
                None,
                n_nodes=5,
                dt=0.00025,
                sensitivity={"amplitude_factor": 1.0, "noise_fraction": 0.1},
            )
            self.assertNotEqual(first_result["provenance"]["input_hash"], second_result["provenance"]["input_hash"])

    def test_initial_reference_lengths_are_independent_of_perturbation(self):
        config = load_config_from_mapping(self._config())["base"]
        small, _ = _initial_state(config, amplitude_factor=0.5)
        large, _ = _initial_state(config, amplitude_factor=2.0)
        expected = config["length"] / (config["n_nodes"] - 1)
        np.testing.assert_allclose(small.rest_lengths, expected)
        np.testing.assert_allclose(large.rest_lengths, expected)
        self.assertFalse(np.allclose(small.positions, large.positions))

    def test_mode_observables_are_rigid_rotation_invariant(self):
        positions = np.array([[0.0, 0.0], [0.5, 0.1], [1.0, 0.3], [1.5, 0.1], [2.0, 0.0]])
        rest_lengths = np.linalg.norm(np.diff(positions, axis=0), axis=1)
        state = FilamentState(positions, rest_lengths)
        angle = 0.37
        rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
        rotated = FilamentState(positions @ rotation.T, rest_lengths)

        reference = _mode_observables(state)
        actual = _mode_observables(rotated)
        self.assertAlmostEqual(reference[0], actual[0])
        np.testing.assert_allclose(reference[1], actual[1], atol=1.0e-15)
        np.testing.assert_allclose(reference[2], actual[2], atol=1.0e-15)

    def test_contact_or_fixed_boundary_settings_are_rejected(self):
        for key, value in (("contact_stiffness", 1.0), ("diameter", 0.1), ("fixed_left", True), ("fixed_right", True)):
            config = self._config()
            config["representatives"][0]["overrides"][key] = value
            with self.assertRaises(GateError):
                load_config_from_mapping(config)

    def test_initial_geometry_failure_is_retained_in_events_and_manifest(self):
        positions = np.array([[0.0, 0.0], [1.0, 1.0], [0.0, 1.0], [1.0, 0.0]])
        state = FilamentState(positions, np.linalg.norm(np.diff(positions, axis=0), axis=1))
        config = load_config_from_mapping(self._config())
        spec = CaseSpec("invalid_initial_geometry", "test", {})
        with tempfile.TemporaryDirectory() as directory:
            with patch(
                "continuum_filament_model.benchmarks.free_free_convergence_gate._initial_state",
                return_value=(state, {"mode": "test"}),
            ):
                _run_case(spec, config["base"], Path(directory), None)
            events = json.loads((Path(directory) / "_runs" / spec.name / "events.json").read_text(encoding="utf-8"))
            manifest = json.loads((Path(directory) / "_runs" / spec.name / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(events["event_count"], 1)
            self.assertEqual(events["failure_event"]["reason"], "initial_crossing")
            self.assertEqual(manifest["events"][0]["reason"], "initial_crossing")

    def test_cross_population_generated_name_collisions_are_rejected(self):
        config = self._config()
        config["representatives"][0]["name"] = "control__x"
        config["controls"] = [{"name": "x__temporal_001", "overrides": {}}]
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(GateError):
                run_gate(config, Path(directory))

    def test_duplicate_control_or_contrast_names_are_rejected(self):
        config = self._config()
        config["controls"] = [
            {"name": "duplicate", "overrides": {}},
            {"name": "duplicate", "overrides": {}},
        ]
        with self.assertRaises(GateError):
            load_config_from_mapping(config)
        config = self._config()
        config["contrasts"][1]["name"] = config["contrasts"][0]["name"]
        with self.assertRaises(GateError):
            load_config_from_mapping(config)

    def test_missing_parameter_contrast_is_rejected(self):
        config = self._config()
        config["contrasts"] = config["contrasts"][:-1]
        with self.assertRaises(GateError):
            load_config_from_mapping(config)


if __name__ == "__main__":
    unittest.main()
