from __future__ import annotations

import unittest

import numpy as np

from growing_filament.model import (
    FilamentState,
    ModelParameters,
    OverdampedGrowingFilament,
    grow_reference_lengths,
    remesh,
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
