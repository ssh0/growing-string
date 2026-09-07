from __future__ import annotations

import unittest

import numpy as np

from growing_filament.model import (
    FilamentState,
    ModelParameters,
    OverdampedGrowingFilament,
    grow_reference_lengths,
    has_nonlocal_intersection,
    remesh,
    segments_intersect,
    straight_state,
)


class ModelTest(unittest.TestCase):
    def test_straight_unstretched_chain_has_zero_energy_and_force(self):
        state = straight_state(5, spacing=1.0)
        model = OverdampedGrowingFilament(state, ModelParameters(reference_length=1.0))
        np.testing.assert_allclose(model.energy(), 0.0, atol=1.0e-14)
        np.testing.assert_allclose(model.forces(), 0.0, atol=1.0e-14)

    def test_energy_and_force_are_translation_invariant(self):
        state = FilamentState(
            [[0.0, 0.0], [1.0, 0.2], [2.0, 0.0]],
            [1.0, 1.0],
        )
        model = OverdampedGrowingFilament(
            state,
            ModelParameters(reference_length=1.0, reject_crossing=False),
        )
        shifted = state.positions + np.asarray([4.0, -3.0])
        shifted_model_energy = model.energy(shifted, state.rest_lengths)
        np.testing.assert_allclose(shifted_model_energy, model.energy())
        np.testing.assert_allclose(
            model.forces(shifted, state.rest_lengths), model.forces()
        )

    def test_stretch_bend_contact_forces_match_finite_difference_energy_gradient(self):
        positions = np.asarray(
            [[0.0, 0.0], [1.2, 0.3], [2.0, 0.0], [1.1, -0.2]],
            dtype=float,
        )
        rest_lengths = np.ones(3)
        model = OverdampedGrowingFilament(
            FilamentState(positions, rest_lengths),
            ModelParameters(
                axial_stiffness=7.0,
                bending_stiffness=1.3,
                contact_stiffness=2.2,
                diameter=1.0,
                reference_length=1.0,
                reject_crossing=False,
            ),
        )

        components = model.energy_components()
        for name in ("stretch", "bend", "contact"):
            self.assertGreater(
                components[name], 0.0,
                msg=f"finite-difference fixture must activate {name} energy",
            )

        analytic_force = model.forces()
        # The perturbations stay well inside the smooth contact-overlap region.
        # A 1e-4--1e-6 sweep checks the central-difference truncation range;
        # 1e-7 relative / 2e-8 absolute covers its observed round-off error.
        for epsilon in (1.0e-4, 1.0e-5, 1.0e-6):
            finite_difference_force = np.zeros_like(positions)
            for node in range(len(positions)):
                for coordinate in range(2):
                    plus = positions.copy()
                    minus = positions.copy()
                    plus[node, coordinate] += epsilon
                    minus[node, coordinate] -= epsilon
                    finite_difference_force[node, coordinate] = -(
                        model.energy(plus, rest_lengths)
                        - model.energy(minus, rest_lengths)
                    ) / (2.0 * epsilon)

            np.testing.assert_allclose(
                analytic_force,
                finite_difference_force,
                rtol=1.0e-7,
                atol=2.0e-8,
                err_msg=(
                    "stretch+bend+contact analytic force disagrees with "
                    f"negative finite-difference energy gradient at epsilon={epsilon}"
                ),
            )

    def test_energy_and_force_are_invariant_under_rigid_rotation(self):
        state = FilamentState(
            [[0.0, 0.0], [1.2, 0.3], [2.0, 0.0], [1.1, -0.2]],
            [1.0, 1.0, 1.0],
        )
        model = OverdampedGrowingFilament(
            state,
            ModelParameters(
                axial_stiffness=7.0,
                bending_stiffness=1.3,
                contact_stiffness=2.2,
                diameter=1.0,
                reference_length=1.0,
                reject_crossing=False,
            ),
        )
        angle = 0.731
        rotation = np.asarray(
            [
                [np.cos(angle), -np.sin(angle)],
                [np.sin(angle), np.cos(angle)],
            ]
        )
        rotated_positions = state.positions @ rotation.T

        np.testing.assert_allclose(
            model.energy(rotated_positions, state.rest_lengths),
            model.energy(),
            rtol=1.0e-12,
            atol=1.0e-12,
            err_msg="rigid rotation changed stretch+bend+contact energy",
        )
        np.testing.assert_allclose(
            model.forces(rotated_positions, state.rest_lengths),
            model.forces() @ rotation.T,
            rtol=1.0e-12,
            atol=1.0e-12,
            err_msg="rigid rotation did not rotate the total force covariantly",
        )

    def test_nonadjacent_node_contact_energy_and_force_obey_threshold_and_action_reaction(self):
        positions = np.asarray(
            [[0.0, 0.0], [0.4, 0.7], [0.8, 0.0]],
            dtype=float,
        )
        rest_lengths = np.linalg.norm(np.diff(positions, axis=0), axis=1)
        common = dict(
            axial_stiffness=4.0,
            bending_stiffness=0.8,
            diameter=1.0,
            reference_length=1.0,
            reject_crossing=False,
        )
        contact_model = OverdampedGrowingFilament(
            FilamentState(positions, rest_lengths),
            ModelParameters(contact_stiffness=3.0, **common),
        )
        no_contact_model = OverdampedGrowingFilament(
            FilamentState(positions, rest_lengths),
            ModelParameters(contact_stiffness=0.0, **common),
        )

        overlap = 1.0 - 0.8
        expected_energy = 0.5 * 3.0 * overlap**2
        np.testing.assert_allclose(
            contact_model.energy_components()["contact"],
            expected_energy,
            err_msg="non-adjacent node contact energy used the wrong threshold law",
        )

        contact_force = contact_model.forces() - no_contact_model.forces()
        expected_force = np.asarray([[-0.6, 0.0], [0.0, 0.0], [0.6, 0.0]])
        np.testing.assert_allclose(
            contact_force,
            expected_force,
            atol=1.0e-12,
            err_msg="non-adjacent node contact force has the wrong magnitude or direction",
        )
        np.testing.assert_allclose(
            contact_force[0],
            -contact_force[2],
            atol=1.0e-12,
            err_msg="contact force does not satisfy pairwise action-reaction",
        )
        np.testing.assert_allclose(
            contact_force[1],
            0.0,
            atol=1.0e-12,
            err_msg="adjacent middle node received a spurious contact contribution",
        )

    def test_nonadjacent_node_contact_is_zero_at_and_outside_diameter(self):
        common = dict(
            axial_stiffness=4.0,
            bending_stiffness=0.8,
            contact_stiffness=3.0,
            diameter=1.0,
            reference_length=1.0,
            reject_crossing=False,
        )
        for label, positions in (
            (
                "exact threshold",
                np.asarray([[0.0, 0.0], [0.4, 0.7], [1.0, 0.0]]),
            ),
            (
                "outside threshold",
                np.asarray([[0.0, 0.0], [0.4, 0.7], [1.2, 0.0]]),
            ),
        ):
            rest_lengths = np.linalg.norm(np.diff(positions, axis=0), axis=1)
            contact_model = OverdampedGrowingFilament(
                FilamentState(positions, rest_lengths),
                ModelParameters(**common),
            )
            no_contact_model = OverdampedGrowingFilament(
                FilamentState(positions, rest_lengths),
                ModelParameters(**{**common, "contact_stiffness": 0.0}),
            )
            contact_force = contact_model.forces() - no_contact_model.forces()

            self.assertEqual(
                contact_model.energy_components()["contact"],
                0.0,
                msg=f"contact energy was nonzero at {label}",
            )
            np.testing.assert_allclose(
                contact_force,
                0.0,
                atol=1.0e-12,
                err_msg=f"contact force was nonzero at {label}",
            )

    def test_growth_is_exponential(self):
        result = grow_reference_lengths(np.ones(2), 0.5, 0.2)
        np.testing.assert_allclose(result, np.exp(0.1))

    def test_remesh_preserves_total_reference_length_and_endpoints(self):
        positions = np.asarray([[0.0, 0.0], [2.0, 0.0], [3.0, 0.0]])
        rest = np.asarray([2.0, 1.0])
        new_positions, new_rest = remesh(positions, rest, a_max=1.0)
        self.assertEqual(len(new_rest), 3)
        self.assertAlmostEqual(float(np.sum(new_rest)), float(np.sum(rest)))
        np.testing.assert_allclose(new_positions[[0, -1]], positions[[0, -1]])

    def test_remesh_recursively_splits_each_child(self):
        positions = np.asarray([[0.0, 0.0], [8.0, 0.0], [9.0, 0.0]])
        rest = np.asarray([8.0, 1.0])

        new_positions, new_rest = remesh(positions, rest, a_max=1.0)

        expected_positions = np.column_stack(
            (np.arange(10, dtype=float), np.zeros(10)),
        )
        self.assertEqual(len(new_rest), 9)
        np.testing.assert_allclose(new_positions, expected_positions)
        np.testing.assert_allclose(new_positions[[0, -1]], positions[[0, -1]])
        self.assertAlmostEqual(float(np.sum(new_rest)), float(np.sum(rest)))
        self.assertLessEqual(float(np.max(new_rest)), 1.0)

    def test_nonlocal_intersection_detects_crossing_and_not_non_crossing_chain(self):
        non_crossing = np.asarray(
            [[0.0, 0.0], [2.0, 2.0], [0.0, 2.0], [2.0, 3.0]],
            dtype=float,
        )
        crossing = np.asarray(
            [[0.0, 0.0], [2.0, 2.0], [0.0, 2.0], [2.0, 0.0]],
            dtype=float,
        )

        self.assertFalse(
            segments_intersect(
                non_crossing[0], non_crossing[1], non_crossing[2], non_crossing[3]
            ),
            msg="non-crossing non-adjacent segments were classified as intersecting",
        )
        self.assertTrue(
            segments_intersect(
                crossing[0], crossing[1], crossing[2], crossing[3]
            ),
            msg="crossing non-adjacent segments were not classified as intersecting",
        )
        self.assertFalse(
            has_nonlocal_intersection(non_crossing),
            msg="non-crossing chain was classified as self-intersecting",
        )
        self.assertTrue(
            has_nonlocal_intersection(crossing),
            msg="crossing chain was not classified as self-intersecting",
        )

    def test_crossing_trial_is_rejected_without_mutating_state(self):
        positions = np.asarray(
            [[0.0, 0.0], [1.0, 1.0], [0.0, 2.0], [-1.0, 1.1]],
            dtype=float,
        )
        rest_lengths = np.asarray([np.sqrt(2.0), np.sqrt(2.0), 2.0])
        params = ModelParameters(
            axial_stiffness=1.0,
            bending_stiffness=1.0,
            drag_density=1.0,
            dt=0.32,
            a_max=10.0,
            max_displacement_fraction=1.0,
            max_retries=0,
            reject_crossing=True,
        )
        model = OverdampedGrowingFilament(
            FilamentState(positions, rest_lengths), params
        )
        before = model.state.copy()

        node_weights = np.empty(4)
        node_weights[0] = rest_lengths[0] / 2.0
        node_weights[-1] = rest_lengths[-1] / 2.0
        node_weights[1:-1] = 0.5 * (rest_lengths[:-1] + rest_lengths[1:])
        velocities = model.forces() / node_weights[:, None]
        trial_positions = positions + params.dt * velocities
        self.assertFalse(
            has_nonlocal_intersection(positions),
            msg="crossing-rejection fixture starts in an invalid self-intersecting state",
        )
        self.assertTrue(
            has_nonlocal_intersection(trial_positions),
            msg="constructed trial did not cross non-adjacent segments",
        )
        self.assertLessEqual(
            float(np.max(np.linalg.norm(params.dt * velocities, axis=1))),
            params.max_displacement_fraction * float(np.min(rest_lengths)),
            msg="crossing fixture also violates the displacement limiter",
        )

        with self.assertRaisesRegex(RuntimeError, "failed to find an accepted step"):
            model.step()

        self.assertEqual(model.rejected_steps, 1)
        self.assertEqual(model.accepted_steps, 0)
        np.testing.assert_array_equal(
            model.state.positions,
            before.positions,
            err_msg="rejected crossing trial mutated node positions",
        )
        np.testing.assert_array_equal(
            model.state.rest_lengths,
            before.rest_lengths,
            err_msg="rejected crossing trial mutated reference lengths",
        )
        self.assertEqual(model.state.time, before.time)
        self.assertEqual(model.state.step, before.step)

    def test_growth_free_overdamped_steps_are_finite_and_non_increasing_in_energy(self):
        initial_rest_lengths = np.ones(3)
        model = OverdampedGrowingFilament(
            FilamentState(
                [[0.0, 0.0], [1.0, 0.4], [2.0, 0.4], [3.0, 0.0]],
                initial_rest_lengths,
            ),
            ModelParameters(
                axial_stiffness=100.0,
                bending_stiffness=1.0,
                drag_density=1.0,
                growth_rate=0.0,
                reference_length=1.0,
                dt=1.0e-5,
                a_max=2.0,
                fixed_left=True,
                fixed_right=True,
            ),
        )
        previous_energy = model.energy()
        self.assertTrue(np.isfinite(previous_energy), "initial energy is not finite")

        for step_number in range(50):
            state = model.step()
            current_energy = model.energy()
            tolerance = 1.0e-12 * max(1.0, abs(previous_energy))
            self.assertTrue(
                np.isfinite(current_energy),
                msg=f"accepted growth-free step {step_number + 1} produced non-finite energy",
            )
            self.assertTrue(
                np.isfinite(state.positions).all(),
                msg=f"accepted growth-free step {step_number + 1} produced non-finite positions",
            )
            self.assertLessEqual(
                current_energy,
                previous_energy + tolerance,
                msg=(
                    f"growth-free overdamped step {step_number + 1} increased "
                    f"energy: {previous_energy} -> {current_energy}"
                ),
            )
            np.testing.assert_array_equal(
                state.rest_lengths,
                initial_rest_lengths,
                err_msg=f"growth-free step {step_number + 1} changed reference lengths",
            )
            previous_energy = current_energy

        self.assertEqual(model.accepted_steps, 50)
        self.assertEqual(model.rejected_steps, 0)

    def test_small_step_reduces_energy_of_bent_fixed_chain(self):
        state = FilamentState(
            [[0.0, 0.0], [1.0, 0.4], [2.0, 0.4], [3.0, 0.0]],
            [1.0, 1.0, 1.0],
        )
        params = ModelParameters(
            reference_length=1.0,
            bending_stiffness=1.0,
            drag_density=1.0,
            dt=1.0e-6,
            a_max=2.0,
            fixed_left=True,
            fixed_right=True,
        )
        model = OverdampedGrowingFilament(state, params)
        before = model.energy()
        model.step()
        after = model.energy()
        self.assertLess(after, before)


if __name__ == "__main__":
    unittest.main()
