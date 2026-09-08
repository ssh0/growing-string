from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from continuum_filament_model.benchmarks.linear_mode_convergence import (
    _finite_difference_hessian,
    _linearization,
    _run_growth_case,
)
from growing_filament.model import FilamentState, ModelError


class P0BLinearModeConvergenceTest(unittest.TestCase):
    def test_filament_state_rejects_nonfinite_time_and_noninteger_step(self):
        positions = [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]]
        for bad_time in (
            np.nan,
            np.inf,
            -np.inf,
            True,
            [0.0],
            np.asarray([0.0]),
            np.asarray(0.0),
        ):
            with self.subTest(time=bad_time):
                with self.assertRaisesRegex(ModelError, "time"):
                    FilamentState(positions, [1.0, 1.0], time=bad_time)
        for bad_step in (
            np.nan,
            np.inf,
            -np.inf,
            1.5,
            True,
            [3],
            np.asarray([3]),
            np.asarray(3),
        ):
            with self.subTest(step=bad_step):
                with self.assertRaisesRegex(ModelError, "step"):
                    FilamentState(positions, [1.0, 1.0], step=bad_step)
        state = FilamentState(positions, [1.0, 1.0], time=0.25, step=3)
        integer_float_state = FilamentState(positions, [1.0, 1.0], time=np.float64(0.25), step=3.0)
        self.assertEqual(state.time, 0.25)
        self.assertEqual(state.step, 3)
        self.assertEqual(integer_float_state.step, 3.0)

    def test_discrete_linearized_hessian_matches_force_gradient(self):
        reference = _linearization(2.0, 7, 0.1, 1.0)
        numerical = _finite_difference_hessian(2.0, 7, 100.0, 0.1, 1.0, 1.0e-7)
        np.testing.assert_allclose(numerical, reference["hessian"], rtol=1.0e-7, atol=1.0e-9)
        numerical_values = np.linalg.eigvalsh(numerical / reference["drag"][0, 0])
        np.testing.assert_allclose(
            numerical_values,
            reference["eigenvalues"],
            rtol=1.0e-7,
            atol=1.0e-9,
        )

    def test_growth_free_compact_accounting_keeps_energy_gate_and_remesh_separate(self):
        growth = {
            "length": 2.0,
            "axial_stiffness": 100.0,
            "bending_stiffness": 0.1,
            "drag_density": 1.0,
            "amplitude": 0.02,
            "t_end": 0.01,
            "a_max_factor": 2.0,
            "max_retries": 4,
            "max_displacement_fraction": 0.25,
        }
        representative = {"name": "growth-free-control", "growth_rate": 0.0}
        with tempfile.TemporaryDirectory() as temporary:
            temporary_path = Path(temporary)
            summary = _run_growth_case(
                growth,
                representative,
                n_nodes=5,
                dt=1.0e-4,
                output=Path(temporary),
                run_name="growth-free-regression",
            )
            run_dir = temporary_path / "growth" / "growth-free-regression"
            with (run_dir / "metrics.csv").open(newline="", encoding="utf-8") as stream:
                metrics = list(csv.DictReader(stream))
            stored_summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
        required_fields = {
            "accepted_dt_actual",
            "mechanical_energy_change_step",
            "energy_balance_residual_step",
            "mechanical_change_plus_dissipation_step",
            "rejected_trials_step",
            "event_count_step",
            "remesh_jumps_step",
        }
        self.assertTrue(required_fields.issubset(metrics[0].keys()))
        self.assertTrue(any(row["accepted_dt_actual"] for row in metrics[1:]))
        self.assertTrue(required_fields.issubset(stored_summary["metrics_schema"]))
        self.assertEqual(stored_summary["accepted_dt_count"], len(metrics) - 1)
        self.assertIn("energy_balance_residual_max_abs", stored_summary)
        self.assertTrue(summary["growth_free_energy_nonincrease"])
        self.assertTrue(summary["fixed_mesh_observed"])
        self.assertEqual(summary["remesh_jumps"], 0)
        self.assertIn("growth_reference_energy_change", summary)
        self.assertIn("dissipation_euler_estimate", summary)
        self.assertIn("rejected_trials", summary)
        self.assertIn("event_count", summary)


if __name__ == "__main__":
    unittest.main()
