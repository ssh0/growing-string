from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from continuum_filament_model.benchmarks.linear_mode_convergence import (
    _convergence_status,
    _finite_difference_hessian,
    _linearization,
    _run_growth_case,
)
from growing_filament.model import (
    FilamentState,
    ModelError,
    ModelParameters,
    OverdampedGrowingFilament,
)


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

    def test_morphology_gate_reports_energy_and_adaptive_dt_as_audit_only(self):
        base = {
            "run": "coarse",
            "failure_reason": None,
            "fixed_mesh_observed": True,
            "classification": {
                "label": "straight",
                "onset_time": None,
                "peak_max_transverse_displacement": 0.02,
            },
            "peak_A1_over_L": 0.01,
            "dt_requested": 0.001,
            "accepted_dt_min": 0.0005,
            "accepted_dt_max": 0.001,
            "accepted_dt_mean": 0.00075,
            "accepted_dt_count": 10,
            "rejected_trials": 2,
            "event_count": 13,
            "energy_final": 1.0,
            "dissipation_euler_estimate": 2.0,
        }
        fine = {**base, "run": "fine", "accepted_dt_min": 0.001, "accepted_dt_mean": 0.001, "rejected_trials": 0, "event_count": 11, "energy_final": 0.01, "dissipation_euler_estimate": 0.1}
        result = _convergence_status(
            [base, fine],
            {
                "onset_time_relative": 0.1,
                "peak_transverse_relative": 0.15,
                "peak_transverse_absolute_fraction_of_length": 0.002,
                "a1_over_length_relative": 0.15,
            },
            "fine",
        )
        self.assertEqual(result["status"], "morphology-converged")
        self.assertEqual(result["scope"], "morphology-only")
        self.assertFalse(result["energy_and_dissipation_gated"])
        self.assertTrue(result["audit"]["adaptive_dt_warning"])
        self.assertGreater(result["audit"]["energy_final_relative_span"], 1.0)

    def test_numeric_strings_fail_validation_and_first_step_with_model_error(self):
        positions = [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]]
        for field, value in (("time", "0"), ("step", "3")):
            with self.subTest(field=field):
                kwargs = {field: value}
                with self.assertRaises(ModelError):
                    FilamentState(positions, [1.0, 1.0], **kwargs)

        model = OverdampedGrowingFilament(
            FilamentState(positions, [1.0, 1.0]),
            ModelParameters(
                axial_stiffness=2.0,
                bending_stiffness=1.0,
                drag_density=1.0,
                dt=1.0e-4,
                t_end=1.0e-3,
                a_max=2.0,
                fixed_left=True,
                fixed_right=True,
            ),
        )
        for field, value in (("time", "0"), ("step", "3")):
            with self.subTest(mutated_field=field):
                setattr(model.state, field, value)
                with self.assertRaises(ModelError):
                    model.step()
                setattr(model.state, field, 0.0 if field == "time" else 0)

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
