from __future__ import annotations

import unittest
from dataclasses import replace

import numpy as np

from continuum_filament_model.benchmarks.time_integration import (
    run_fixed_mesh_diagnostic,
    run_time_convergence_benchmark,
)
from growing_filament.model import (
    FilamentState,
    ModelError,
    ModelParameters,
    OverdampedGrowingFilament,
)


class TimeIntegrationContractTest(unittest.TestCase):
    def test_manufactured_three_node_explicit_euler_is_first_order(self):
        records = run_time_convergence_benchmark()

        self.assertEqual([record["dt"] for record in records], [0.02, 0.01, 0.005])
        self.assertEqual([record["n_nodes"] for record in records], [3, 3, 3])
        for record in records:
            self.assertEqual(record["rejected_steps"], 0)
            np.testing.assert_allclose(record["accepted_dts"], record["dt"])
            self.assertGreater(record["error"], 0.0)
            self.assertTrue(np.isfinite(record["error"]))

        observed_orders = [record["observed_order"] for record in records[:-1]]
        self.assertTrue(
            all(0.9 < order < 1.1 for order in observed_orders),
            msg=(
                "manufactured-solution orders were not consistently first-order: "
                f"{observed_orders}"
            ),
        )
        self.assertLess(records[2]["error"], records[1]["error"])
        self.assertLess(records[1]["error"], records[0]["error"])

    def test_fixed_mesh_diagnostic_accepts_only_finite_non_increasing_energy(self):
        for requested_dt in (1.0e-2, 5.0e-3):
            diagnostic = run_fixed_mesh_diagnostic(requested_dt)
            energies = np.asarray(diagnostic["energies"], dtype=float)
            times = np.asarray(diagnostic["times"], dtype=float)
            accepted_dts = np.asarray(diagnostic["accepted_dts"], dtype=float)

            self.assertTrue(np.isfinite(energies).all())
            self.assertTrue(np.isfinite(times).all())
            self.assertTrue(np.isfinite(accepted_dts).all())
            self.assertTrue(np.all(accepted_dts > 0.0))
            self.assertTrue(all(diagnostic["finite_positions"]))
            self.assertEqual(len(energies) - 1, len(accepted_dts))
            self.assertEqual(set(diagnostic["node_counts"]), {4})
            tolerance = 1.0e-12 * np.maximum(1.0, np.abs(energies[:-1]))
            self.assertTrue(
                np.all(np.diff(energies) <= tolerance),
                msg=f"accepted fixed-mesh energy increased for dt={requested_dt}: {energies}",
            )

        dt_01 = run_fixed_mesh_diagnostic(1.0e-2)
        self.assertIn(1.0e-2, dt_01["rejected_dts"])
        self.assertTrue(
            any("growth-free trial energy increased" in reason
                for reason in dt_01["rejection_reasons"])
        )

    def test_growth_trial_may_increase_energy_and_is_still_accepted(self):
        model = OverdampedGrowingFilament(
            FilamentState(
                [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]],
                [1.0, 1.0],
            ),
            ModelParameters(
                axial_stiffness=10.0,
                bending_stiffness=1.0,
                drag_density=1.0,
                growth_rate=1.0,
                reference_length=1.0,
                dt=1.0e-2,
                a_max=2.0,
                fixed_left=True,
                fixed_right=True,
            ),
        )
        before = model.energy()
        state = model.step()

        self.assertEqual(model.rejected_steps, 0)
        self.assertEqual(model.accepted_dts, [1.0e-2])
        self.assertGreater(model.energy(), before)
        self.assertEqual(state.n_nodes, 3)

    def test_model_parameters_reject_all_non_finite_numeric_inputs(self):
        fields = (
            "axial_stiffness",
            "bending_stiffness",
            "drag_density",
            "contact_stiffness",
            "diameter",
            "growth_rate",
            "reference_length",
            "dt",
            "t_end",
            "a_max",
            "dt_min",
            "max_retries",
            "max_displacement_fraction",
            "energy_tolerance",
        )
        for field in fields:
            for bad_value in (np.nan, np.inf, -np.inf):
                with self.subTest(field=field, bad_value=bad_value):
                    parameters = replace(ModelParameters(), **{field: bad_value})
                    with self.assertRaisesRegex(ModelError, field):
                        parameters.validate()


if __name__ == "__main__":
    unittest.main()
