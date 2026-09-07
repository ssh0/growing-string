import json
import tempfile
import unittest
from pathlib import Path

from continuum_filament_model.benchmarks.buckling_benchmark import (
    CaseSpec,
    dimensionless_groups,
    initial_perturbed_state,
    run_case,
)


class BucklingBenchmarkTests(unittest.TestCase):
    def _config(self):
        return {
            "length": 2.0,
            "n_nodes": 5,
            "axial_stiffness": 100.0,
            "bending_stiffness": 0.1,
            "drag_density": 1.0,
            "growth_rate": 0.0,
            "amplitude": 0.01,
            "dt": 0.0005,
            "t_end": 0.002,
            "a_max_factor": 2.0,
            "contact_stiffness": 0.0,
            "diameter": 0.0,
            "fixed_left": True,
            "fixed_right": True,
            "reject_crossing": True,
        }

    def test_fixture_and_dimensionless_groups_are_deterministic(self):
        config = self._config()
        state = initial_perturbed_state(config)
        self.assertAlmostEqual(state.positions[0, 1], 0.0)
        self.assertAlmostEqual(state.positions[-1, 1], 0.0)
        self.assertGreater(state.positions[2, 1], 0.0)
        groups = dimensionless_groups(config)
        self.assertAlmostEqual(groups["G_b"], 0.0)
        self.assertGreater(groups["tau_b"], 0.0)
        self.assertEqual(groups["diameter_over_L"], 0.0)

    def test_case_writes_manifest_metrics_and_noncontact_conditions(self):
        config = self._config()
        with tempfile.TemporaryDirectory() as directory:
            summary = run_case(
                CaseSpec("test_case", {}),
                config,
                Path(directory),
                git_revision="test-revision",
            )
            case_dir = Path(directory) / "test_case"
            self.assertIsNone(summary["failure_reason"])
            self.assertFalse(summary["contact_enabled"])
            self.assertTrue((case_dir / "manifest.json").is_file())
            self.assertTrue((case_dir / "metrics.csv").is_file())
            self.assertTrue((case_dir / "events.json").is_file())
            manifest = json.loads((case_dir / "manifest.json").read_text())
            self.assertEqual(manifest["git_revision"], "test-revision")
            self.assertEqual(manifest["metadata"]["physical_conditions"]["contact_stiffness"], 0.0)
            self.assertEqual(manifest["metadata"]["physical_conditions"]["diameter"], 0.0)
            self.assertGreaterEqual(manifest["event_count"], 1)
            self.assertIn(summary["classification"]["label"], {"straight", "buckled-single", "unresolved"})


if __name__ == "__main__":
    unittest.main()
