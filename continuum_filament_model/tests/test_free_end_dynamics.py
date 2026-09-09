from __future__ import annotations

import unittest

import numpy as np

from continuum_filament_model.benchmarks.free_end_benchmark import run_benchmark
from growing_filament.model import (
    FilamentState,
    ModelParameters,
    OverdampedGrowingFilament,
    straight_state,
)


class FreeEndDynamicsTest(unittest.TestCase):
    def test_straight_free_free_equilibrium_has_natural_residuals_zero(self):
        model = OverdampedGrowingFilament(
            straight_state(5),
            ModelParameters(reference_length=1.0, dt=1.0e-3, t_end=2.0e-3),
        )
        diagnostics = model.endpoint_diagnostics()

        self.assertEqual(diagnostics["boundary_condition"], {"left": "free", "right": "free"})
        self.assertFalse(diagnostics["contact_enabled"])
        np.testing.assert_allclose(diagnostics["net_force"], [0.0, 0.0], atol=1.0e-14)
        self.assertEqual(diagnostics["net_torque_about_left_endpoint"], 0.0)
        for side in ("left", "right"):
            endpoint = diagnostics[side]
            np.testing.assert_allclose(endpoint["force_residual"], [0.0, 0.0], atol=1.0e-14)
            self.assertAlmostEqual(endpoint["bending_moment"], 0.0, places=14)
            self.assertAlmostEqual(endpoint["shear_equivalent_residual"], 0.0, places=14)
            np.testing.assert_allclose(endpoint["contact_force"], [0.0, 0.0], atol=1.0e-14)

        initial = model.state.positions.copy()
        model.step()
        np.testing.assert_allclose(model.state.positions, initial, atol=1.0e-14)

    def test_endpoint_diagnostics_are_rigid_translation_and_rotation_covariant(self):
        state = FilamentState(
            [[0.0, 0.0], [1.0, 0.3], [2.0, 0.0], [3.0, -0.2]],
            np.linalg.norm(np.diff([[0.0, 0.0], [1.0, 0.3], [2.0, 0.0], [3.0, -0.2]], axis=0), axis=1),
        )
        params = ModelParameters(reference_length=1.0, reject_crossing=False)
        model = OverdampedGrowingFilament(state, params)
        translated = state.positions + np.asarray([3.0, -2.0])
        translated_diagnostics = model.endpoint_diagnostics(translated, state.rest_lengths)
        original = model.endpoint_diagnostics()
        for side in ("left", "right"):
            np.testing.assert_allclose(
                translated_diagnostics[side]["force_residual"],
                original[side]["force_residual"],
                atol=1.0e-12,
            )
            self.assertAlmostEqual(
                translated_diagnostics[side]["bending_moment"],
                original[side]["bending_moment"],
                places=12,
            )

        angle = 0.47
        rotation = np.asarray(
            [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
        )
        rotated = state.positions @ rotation.T
        rotated_diagnostics = model.endpoint_diagnostics(rotated, state.rest_lengths)
        for side in ("left", "right"):
            np.testing.assert_allclose(
                rotated_diagnostics[side]["force_residual"],
                np.asarray(original[side]["force_residual"]) @ rotation.T,
                rtol=1.0e-12,
                atol=1.0e-12,
            )
            self.assertAlmostEqual(
                rotated_diagnostics[side]["force_residual_norm"],
                original[side]["force_residual_norm"],
                places=12,
            )
            self.assertAlmostEqual(
                rotated_diagnostics[side]["bending_moment"],
                original[side]["bending_moment"],
                places=12,
            )

    def test_free_end_bending_relaxation_moves_endpoints_without_anchor(self):
        positions = np.asarray(
            [[0.0, 0.0], [1.0, 0.4], [2.0, 0.4], [3.0, 0.0]],
            dtype=float,
        )
        rest_lengths = np.linalg.norm(np.diff(positions, axis=0), axis=1)
        common = dict(
            axial_stiffness=50.0,
            bending_stiffness=1.0,
            drag_density=1.0,
            growth_rate=0.0,
            reference_length=1.0,
            dt=1.0e-5,
            t_end=5.0e-3,
            a_max=2.0,
            reject_crossing=False,
        )
        free = OverdampedGrowingFilament(FilamentState(positions, rest_lengths), ModelParameters(**common))
        fixed = OverdampedGrowingFilament(
            FilamentState(positions, rest_lengths),
            ModelParameters(**common, fixed_left=True, fixed_right=True),
        )
        free_initial_energy = free.energy()
        free_initial_endpoints = free.state.positions[[0, -1]].copy()
        free.run()
        fixed.run()

        self.assertLess(free.energy(), free_initial_energy)
        self.assertGreater(
            float(np.linalg.norm(free.state.positions[[0, -1]] - free_initial_endpoints)),
            0.0,
        )
        np.testing.assert_allclose(fixed.state.positions[[0, -1]], positions[[0, -1]])
        self.assertGreater(free.accepted_steps, 0)
        self.assertEqual(free.rejected_steps, 0)

    def test_uniform_growth_has_first_step_free_end_response_and_no_contact_constraint(self):
        dt = 1.0e-4
        growth_rate = 0.5
        axial_stiffness = 4.0
        model = OverdampedGrowingFilament(
            straight_state(3),
            ModelParameters(
                axial_stiffness=axial_stiffness,
                bending_stiffness=1.0,
                drag_density=1.0,
                growth_rate=growth_rate,
                reference_length=1.0,
                dt=dt,
                a_max=2.0,
            ),
        )
        before = model.state.positions.copy()
        grown = np.exp(growth_rate * dt)
        expected_endpoint_speed = 2.0 * axial_stiffness * (grown - 1.0) / grown**2
        model.step()

        displacement = model.state.positions - before
        self.assertLess(displacement[0, 0], 0.0)
        self.assertGreater(displacement[-1, 0], 0.0)
        # The left endpoint is pulled left and the right endpoint right.  The
        # sign convention in the model's force is checked explicitly here.
        self.assertAlmostEqual(
            float(np.linalg.norm(displacement[0])),
            dt * expected_endpoint_speed,
            delta=1.0e-10,
        )
        self.assertAlmostEqual(float(displacement[1, 0]), 0.0, places=14)
        self.assertFalse(model.endpoint_diagnostics()["contact_enabled"])
        np.testing.assert_allclose(model.endpoint_diagnostics()["left"]["contact_force"], [0.0, 0.0])

    def test_benchmark_reports_free_free_control_refinement_and_endpoint_trajectory(self):
        report = run_benchmark(
            {
                "n_nodes": [5, 7],
                "dt_values": [2.0e-4, 1.0e-4],
                "t_end": 2.0e-3,
                "max_report_rows": 100,
            }
        )
        self.assertEqual(len(report["records"]), 16)
        self.assertTrue(report["refinement"]["time"])
        self.assertTrue(report["refinement"]["space"])
        for record in report["records"]:
            self.assertFalse(record["contact_enabled"])
            self.assertGreaterEqual(len(record["endpoint_trajectory"]), 2)
            self.assertEqual(record["endpoint_trajectory"][0]["time"], 0.0)

        growth_free = next(
            row for row in report["records"]
            if row["case"] == "growth" and row["boundary"] == "free_free" and row["n_nodes"] == 5 and row["dt"] == 1.0e-4
        )
        growth_fixed = next(
            row for row in report["records"]
            if row["case"] == "growth" and row["boundary"] == "fixed_fixed" and row["n_nodes"] == 5 and row["dt"] == 1.0e-4
        )
        self.assertGreater(growth_free["endpoint_distance_change"], 0.0)
        self.assertEqual(growth_fixed["endpoint_distance_change"], 0.0)


if __name__ == "__main__":
    unittest.main()
