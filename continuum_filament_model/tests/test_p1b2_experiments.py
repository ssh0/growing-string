import json
import tempfile
import unittest
from pathlib import Path

from continuum_filament_model.benchmarks.p1b2_experiments import (
    aggregate_trials,
    dimensionless_groups,
    load_config,
    target_overrides,
)


class P1B2ExperimentTests(unittest.TestCase):
    def _base(self):
        return {
            "length": 2.0,
            "n_nodes": 7,
            "axial_stiffness": 100.0,
            "bending_stiffness": 0.1,
            "drag_density": 1.0,
            "growth_rate": 0.1,
            "amplitude": 0.02,
            "dt": 0.001,
            "t_end": 0.1,
            "a_max_factor": 2.0,
        }

    def _summary(self, label, onset, peak, seed):
        return {
            "trial": seed,
            "seed": seed,
            "classification": {
                "label": label,
                "onset_time": onset,
                "peak_max_transverse_displacement": peak,
            },
            "failure_reason": None,
            "peak_observables": {
                "first_mode_fraction": 0.9,
                "max_curvature": 0.2,
                "energy_total": 0.01,
            },
        }

    def test_target_coordinates_round_trip(self):
        base = self._base()
        overrides = target_overrides(base, 0.25, 0.00025)
        config = {**base, **overrides}
        groups = dimensionless_groups(config)
        self.assertAlmostEqual(groups["G_b"], 0.25, places=10)
        self.assertAlmostEqual(groups["chi"], 0.00025, places=12)

    def test_trial_summary_keeps_denominator_and_iqr(self):
        aggregate = aggregate_trials([
            self._summary("straight", None, 0.02, 101),
            self._summary("buckled-single", 0.1, 0.08, 202),
            self._summary("buckled-single", 0.12, 0.10, 303),
            self._summary("unresolved", None, None, 404),
        ])
        self.assertEqual(aggregate["n_trials"], 4)
        self.assertEqual(aggregate["denominator"], 4)
        self.assertEqual(aggregate["regime"], "numerically-unresolved")
        self.assertEqual(aggregate["label_counts"]["buckled-single"], 2)
        self.assertEqual(aggregate["statistics"]["onset_time"]["missing_count"], 2)
        self.assertAlmostEqual(aggregate["statistics"]["onset_time"]["iqr"], 0.01)

    def test_config_requires_five_grid_seeds(self):
        raw = {
            "base": self._base(),
            "pilot": [{"name": "p", "overrides": {}}],
            "grid": {"G_b": [0.1, 0.2, 0.3], "chi": [0.1, 0.2, 0.3], "seeds": [1, 2, 3, 4]},
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            path.write_text(json.dumps(raw), encoding="utf-8")
            with self.assertRaises(ValueError):
                load_config(path)


if __name__ == "__main__":
    unittest.main()
