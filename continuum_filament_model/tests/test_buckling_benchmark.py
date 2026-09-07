import json
import tempfile
import unittest
from pathlib import Path

from continuum_filament_model.benchmarks.buckling_benchmark import (
    CaseSpec,
    dimensionless_groups,
    initial_perturbed_state,
    run_case,
    trial_summary,
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
            "trial_noise_fraction": 0.05,
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
        seeded_a = initial_perturbed_state(config, seed=11)
        seeded_b = initial_perturbed_state(config, seed=11)
        seeded_c = initial_perturbed_state(config, seed=22)
        self.assertTrue((seeded_a.positions == seeded_b.positions).all())
        self.assertFalse((seeded_a.positions == seeded_c.positions).all())

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

    def test_seed_and_trial_identity_are_saved_and_aggregated(self):
        config = self._config()
        with tempfile.TemporaryDirectory() as directory:
            first = run_case(
                CaseSpec("trial_01", {"trial_noise_fraction": 0.05}, seed=11, trial=1, base_name="fast_growth"),
                config,
                Path(directory),
                git_revision="test-revision",
            )
            second = run_case(
                CaseSpec("trial_02", {"trial_noise_fraction": 0.05}, seed=22, trial=2, base_name="fast_growth"),
                config,
                Path(directory),
                git_revision="test-revision",
            )
            manifest = json.loads((Path(directory) / "trial_01" / "manifest.json").read_text())
            self.assertEqual(manifest["seed"], 11)
            self.assertEqual(manifest["trial"], 1)
            self.assertEqual(manifest["base_case"], "fast_growth")
            aggregate = trial_summary([first, second])
            self.assertEqual(aggregate[0]["n_trials"], 2)
            self.assertEqual(aggregate[0]["seeds"], [11, 22])
            self.assertIn("onset_time_mean", aggregate[0])


if __name__ == "__main__":
    unittest.main()
