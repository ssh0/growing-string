from __future__ import annotations

import unittest

import numpy as np

from growing_filament.geometry import nonlocal_segment_contacts
from growing_filament.model import (
    FilamentState,
    ModelParameters,
    OverdampedGrowingFilament,
)


class SegmentPenaltyContactForceTest(unittest.TestCase):
    """Conservative finite-radius segment-contact regression fixtures."""

    def _u_fixture(self, contact_stiffness: float = 20.0, diameter: float = 1.2):
        # Segment 0 and segment 2 are parallel, non-adjacent, and overlap in
        # projection.  Their closest-point parameters are s=0.75 and u=0.5,
        # which makes the non-uniform shape-function scatter observable.
        positions = np.asarray(
            [[-0.5, 2.0], [-0.5, 0.0], [0.5, 0.0], [0.5, 1.0]],
            dtype=float,
        )
        rest_lengths = np.asarray([2.0, 1.0, 1.0], dtype=float)
        parameters = ModelParameters(
            axial_stiffness=1.0,
            bending_stiffness=0.1,
            contact_stiffness=contact_stiffness,
            diameter=diameter,
            reference_length=1.0,
            reject_crossing=False,
        )
        return positions, rest_lengths, parameters

    def _contact_force(self, positions, rest_lengths, parameters):
        contact_model = OverdampedGrowingFilament(
            FilamentState(positions, rest_lengths), parameters
        )
        no_contact = OverdampedGrowingFilament(
            FilamentState(positions, rest_lengths),
            ModelParameters(
                axial_stiffness=parameters.axial_stiffness,
                bending_stiffness=parameters.bending_stiffness,
                drag_density=parameters.drag_density,
                contact_stiffness=0.0,
                diameter=parameters.diameter,
                reference_length=parameters.reference_length,
                reject_crossing=False,
            ),
        )
        return contact_model, contact_model.forces() - no_contact.forces()

    def test_bilinear_scatter_matches_finite_difference_energy_gradient(self):
        positions, rest_lengths, parameters = self._u_fixture()
        contacts = nonlocal_segment_contacts(positions, parameters.diameter)
        self.assertEqual(len(contacts), 1)
        contact = contacts[0]
        self.assertAlmostEqual(contact.distance, 1.0)
        self.assertAlmostEqual(contact.penetration, 0.2)
        self.assertAlmostEqual(contact.parameter_i, 0.75)
        self.assertAlmostEqual(contact.parameter_j, 0.5)

        model, contact_force = self._contact_force(
            positions, rest_lengths, parameters
        )
        # E_c = 1/2 * 20 * 0.2^2.  The unit normal points from the right
        # segment to the left segment, so the pair force is (-4, 0).  The
        # (1-s, s) and (1-u, u) scatter gives (-1, -3, +2, +2).
        self.assertAlmostEqual(model.energy_components()["contact"], 0.4)
        np.testing.assert_allclose(
            contact_force,
            [[-1.0, 0.0], [-3.0, 0.0], [2.0, 0.0], [2.0, 0.0]],
            atol=1.0e-12,
        )

        finite_difference_force = np.zeros_like(positions)
        epsilon = 1.0e-6
        for node in range(len(positions)):
            for coordinate in range(2):
                plus = positions.copy()
                minus = positions.copy()
                plus[node, coordinate] += epsilon
                minus[node, coordinate] -= epsilon
                finite_difference_force[node, coordinate] = -(
                    model.energy_components(plus, rest_lengths)["contact"]
                    - model.energy_components(minus, rest_lengths)["contact"]
                ) / (2.0 * epsilon)
        np.testing.assert_allclose(
            contact_force,
            finite_difference_force,
            rtol=3.0e-6,
            atol=1.0e-5,
            err_msg="segment penalty force is not -dE_contact/dr",
        )

    def test_segment_contact_obeys_action_reaction_and_rigid_invariance(self):
        positions, rest_lengths, parameters = self._u_fixture()
        model, contact_force = self._contact_force(
            positions, rest_lengths, parameters
        )
        np.testing.assert_allclose(
            np.sum(contact_force, axis=0),
            [0.0, 0.0],
            atol=1.0e-12,
            err_msg="segment contact has nonzero net internal force",
        )

        angle = 0.61
        rotation = np.asarray(
            [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
        )
        translation = np.asarray([4.0, -3.0])
        transformed_positions = positions @ rotation.T + translation
        transformed_model, transformed_force = self._contact_force(
            transformed_positions, rest_lengths, parameters
        )
        np.testing.assert_allclose(
            transformed_model.energy_components()["contact"],
            model.energy_components()["contact"],
            rtol=1.0e-12,
            atol=1.0e-12,
        )
        np.testing.assert_allclose(
            transformed_force,
            contact_force @ rotation.T,
            rtol=1.0e-12,
            atol=1.0e-12,
            err_msg="rigid rotation did not rotate segment contact forces",
        )

    def test_segment_contact_is_zero_at_and_outside_the_diameter(self):
        positions, rest_lengths, _ = self._u_fixture()
        for diameter in (1.0, 0.9):
            with self.subTest(diameter=diameter):
                parameters = self._u_fixture(diameter=diameter)[2]
                model, contact_force = self._contact_force(
                    positions, rest_lengths, parameters
                )
                self.assertEqual(model.energy_components()["contact"], 0.0)
                np.testing.assert_allclose(contact_force, 0.0, atol=1.0e-12)

    def test_controlled_u_fixture_separates_without_excessive_penetration(self):
        positions, rest_lengths, _ = self._u_fixture()
        parameters = ModelParameters(
            axial_stiffness=1.0,
            bending_stiffness=1.0e-6,
            drag_density=1.0,
            contact_stiffness=20.0,
            diameter=1.2,
            reference_length=1.0,
            dt=1.0e-4,
            t_end=1.0e-4,
            a_max=3.0,
            fixed_left=True,
            fixed_right=True,
            max_displacement_fraction=0.25,
            reject_crossing=True,
        )
        model = OverdampedGrowingFilament(
            FilamentState(positions, rest_lengths), parameters
        )
        before = nonlocal_segment_contacts(model.state.positions, parameters.diameter)[0]
        state_after = model.step()
        after = nonlocal_segment_contacts(state_after.positions, parameters.diameter)[0]

        self.assertGreater(before.penetration, 0.0)
        self.assertLess(after.penetration, before.penetration)
        self.assertGreater(after.distance, before.distance)
        self.assertLess(after.penetration, 0.2)
        np.testing.assert_allclose(
            state_after.positions[[0, -1]], positions[[0, -1]], atol=0.0
        )
        self.assertEqual(model.rejected_steps, 0)
        self.assertEqual(model.accepted_steps, 1)


if __name__ == "__main__":
    unittest.main()
