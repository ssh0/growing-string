from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from continuum_filament_model.benchmarks.contact_buckling_benchmark import (
    CaseSpec,
    contact_metrics,
    dimensionless_groups,
    initial_state,
    run_benchmark,
    run_case,
)
from growing_filament.model import FilamentState


class ContactBucklingBenchmarkTests(unittest.TestCase):
    def _config(self):
        return {
            "length": 2.0,
            "n_nodes": 7,
            "axial_stiffness": 100.0,
            "bending_stiffness": 0.1,
            "drag_density": 1.0,
            "growth_rate": 0.0,
            "contact_stiffness": 10.0,
            "diameter": 0.35,
            "dt": 0.0005,
            "t_end": 0.001,
            "a_max_factor": 8.0,
            "initial_shape": "u",
            "fixed_left": True,
            "fixed_right": True,
            "reject_crossing": True,
            "max_retries": 12,
            "dt_min": 1.0e-10,
            "max_displacement_fraction": 0.25,
        }

    def test_contact_metrics_match_finite_radius_definition(self):
        positions = np.asarray(
            [[-0.5, 2.0], [-0.5, 0.0], [0.5, 0.0], [0.5, 1.0]],
            dtype=float,
        )
        state = FilamentState(positions, [2.0, 1.0, 1.0])
        metrics = contact_metrics(state, 1.2)
        self.assertEqual(metrics["active_contact_pairs"], 1)
        self.assertEqual(metrics["finite_radius_contact_pairs"], [[0, 2]])
        self.assertAlmostEqual(metrics["contact_length"], 1.0)
        self.assertAlmostEqual(metrics["max_penetration"], 0.2)
        self.assertAlmostEqual(metrics["penetration_ratio"], 1.0 / 6.0)

    def test_dimensionless_axes_include_chi_and_contact_ratio(self):
        config = self._config()
        groups = dimensionless_groups(config)
        self.assertAlmostEqual(groups["chi"], 0.1 / (100.0 * 2.0**2))
        self.assertAlmostEqual(groups["Pi_c"], 10.0 * 0.35**2 / 0.1)
        self.assertGreater(groups["growth_number_G_b"], -1.0)

    def test_case_is_deterministic_and_reports_contact_energy_work(self):
        config = self._config()
        first = run_case(CaseSpec("deterministic", {}, "test"), config, git_revision="test-revision")
        second = run_case(CaseSpec("deterministic", {}, "test"), config, git_revision="test-revision")
        self.assertIsNone(first["failure_reason"])
        self.assertEqual(first["classification"], second["classification"])
        self.assertEqual(first["metrics_rows"], second["metrics_rows"])
        self.assertEqual(
            first["manifest"]["canonical_state_hash"],
            second["manifest"]["canonical_state_hash"],
        )
        self.assertGreaterEqual(first["onset"]["contact_time"], 0.0)
        self.assertIn("energy_contact", first["final_metrics"])
        self.assertIn("growth_work", first["final_metrics"])
        self.assertIn("dissipation_work", first["final_metrics"])

    def test_benchmark_writes_compact_outputs_without_trajectory(self):
        config = {
            "schema_version": "test",
            "base": self._config(),
            "cases": [
                {"name": "representative", "group": "representative", "overrides": {}},
                {"name": "dt_fine", "group": "convergence", "overrides": {"dt": 0.00025}},
            ],
            "phase_grid": {},
            "convergence": {},
            "output_policy": {"save_trajectory": False},
        }
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            suite = run_benchmark(config, output, git_revision="test-revision")
            self.assertEqual(len(suite["results"]), 2)
            for name in (
                "summary.csv",
                "summary.json",
                "convergence_summary.csv",
                "convergence_summary.json",
                "metrics.csv",
                "phase_map.png",
                "compact_manifest.json",
                "suite.json",
            ):
                self.assertTrue((output / name).is_file(), name)
            self.assertFalse(list(output.glob("*.npz")))
            manifest = json.loads((output / "compact_manifest.json").read_text())
            self.assertEqual(manifest["git_revision"], "test-revision")
            self.assertEqual(manifest["case_order"], ["representative", "dt_fine"])

    def test_compact_manifest_replays_identically(self):
        config = {
            "schema_version": "test",
            "base": self._config(),
            "cases": [{"name": "replay", "group": "test", "overrides": {}}],
            "phase_grid": {},
            "convergence": {},
        }
        with tempfile.TemporaryDirectory() as directory:
            first_dir = Path(directory) / "first"
            second_dir = Path(directory) / "second"
            run_benchmark(config, first_dir, git_revision="test-revision")
            run_benchmark(config, second_dir, git_revision="test-revision")
            first = json.loads((first_dir / "compact_manifest.json").read_text())
            second = json.loads((second_dir / "compact_manifest.json").read_text())
            self.assertEqual(first, second)
            self.assertEqual(
                (first_dir / "summary.csv").read_bytes(),
                (second_dir / "summary.csv").read_bytes(),
            )


if __name__ == "__main__":
    unittest.main()
