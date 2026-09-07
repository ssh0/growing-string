from __future__ import annotations

import unittest

import numpy as np

from continuum_filament_model.benchmarks.remesh_convergence import (
    run_resolution_benchmark,
    run_short_growth_benchmark,
)
from growing_filament.model import (
    FilamentState,
    ModelParameters,
    OverdampedGrowingFilament,
    remesh,
)
from growing_filament.observables import (
    arc_length_weighted_center_of_mass,
    arc_length_weighted_radius_of_gyration,
    radius_of_gyration,
)


class RemeshContractTest(unittest.TestCase):
    def test_remesh_preserves_endpoints_reference_length_and_contour_length(self):
        fixtures = (
            (
                "straight",
                np.asarray([[0.0, 0.0], [2.0, 0.0], [5.0, 0.0]]),
                np.asarray([2.0, 3.0]),
            ),
            (
                "polyline",
                np.asarray([[0.0, 0.0], [2.0, 1.0], [5.0, 0.0]]),
                np.asarray([2.0, 3.0]),
            ),
        )
        for name, positions, rest_lengths in fixtures:
            before_contour = float(
                np.sum(np.linalg.norm(np.diff(positions, axis=0), axis=1))
            )
            remeshed_positions, remeshed_rest = remesh(
                positions,
                rest_lengths,
                a_max=0.75,
            )
            self.assertGreater(
                len(remeshed_rest), len(rest_lengths), msg=f"{name} was not split"
            )
            np.testing.assert_allclose(
                remeshed_positions[[0, -1]],
                positions[[0, -1]],
                rtol=0.0,
                atol=2.0e-15,
                err_msg=f"{name} endpoints changed",
            )
            np.testing.assert_allclose(
                np.sum(remeshed_rest),
                np.sum(rest_lengths),
                rtol=0.0,
                atol=2.0e-15,
                err_msg=f"{name} total reference length changed",
            )
            after_contour = float(
                np.sum(np.linalg.norm(np.diff(remeshed_positions, axis=0), axis=1))
            )
            self.assertAlmostEqual(
                after_contour,
                before_contour,
                places=14,
                msg=f"{name} geometric contour length changed",
            )

    def test_arc_length_weighted_observables_are_invariant_under_midpoint_split(self):
        positions = np.asarray(
            [[0.0, 0.0], [2.0, 1.0], [5.0, 0.0]],
            dtype=float,
        )
        rest_lengths = np.asarray([2.0, 3.0])
        state = FilamentState(positions, rest_lengths)
        remeshed_positions, remeshed_rest = remesh(positions, rest_lengths, 0.75)
        remeshed_state = FilamentState(remeshed_positions, remeshed_rest)

        np.testing.assert_allclose(
            arc_length_weighted_center_of_mass(remeshed_state),
            arc_length_weighted_center_of_mass(state),
            rtol=0.0,
            atol=2.0e-15,
        )
        np.testing.assert_allclose(
            arc_length_weighted_radius_of_gyration(remeshed_state),
            arc_length_weighted_radius_of_gyration(state),
            rtol=0.0,
            atol=2.0e-15,
        )
        self.assertNotAlmostEqual(
            radius_of_gyration(remeshed_state),
            radius_of_gyration(state),
            places=8,
            msg="legacy node-average Rg unexpectedly became the arc-length observable",
        )

    def test_bending_contract_for_a_polygonal_corner_is_explicit(self):
        positions = np.asarray(
            [[0.0, 0.0], [1.0, 1.0], [2.0, 0.0]],
            dtype=float,
        )
        rest_lengths = np.full(2, np.sqrt(2.0))
        parameters = ModelParameters(
            bending_stiffness=2.0,
            reference_length=1.0,
            reject_crossing=False,
        )
        model = OverdampedGrowingFilament(
            FilamentState(positions, rest_lengths), parameters
        )
        refined_positions, refined_rest = remesh(positions, rest_lengths, 0.75)
        refined_model = OverdampedGrowingFilament(
            FilamentState(refined_positions, refined_rest), parameters
        )

        # A fixed polygonal corner is a curvature singularity.  Both adjacent
        # dual cells halve, so the local-arc-length contract predicts 2x energy;
        # smooth-curve resolution tests, not this singular fixture, are used for
        # convergence claims.
        base_bend = model.energy_components()["bend"]
        refined_bend = refined_model.energy_components()["bend"]
        self.assertGreater(base_bend, 0.0)
        np.testing.assert_allclose(refined_bend, 2.0 * base_bend, rtol=1.0e-14)

        # Nodal forces are not a remesh-invariant array: the same generalized
        # bending force is redistributed to the newly independent nodes.  They
        # remain conservative and have zero net force in either representation.
        base_forces = model.forces()
        refined_forces = refined_model.forces()
        self.assertGreater(np.linalg.norm(base_forces[0]), 0.0)
        self.assertGreater(np.linalg.norm(refined_forces[1]), 0.0)
        np.testing.assert_allclose(np.sum(base_forces, axis=0), 0.0, atol=1.0e-14)
        np.testing.assert_allclose(np.sum(refined_forces, axis=0), 0.0, atol=1.0e-14)

    def test_refined_bending_force_is_the_energy_gradient(self):
        positions = np.asarray(
            [[0.0, 0.0], [1.0, 0.7], [2.0, 0.1], [3.0, 0.8]],
            dtype=float,
        )
        rest_lengths = np.asarray([1.0, 1.0, 1.0])
        remeshed_positions, remeshed_rest = remesh(positions, rest_lengths, 0.6)
        model = OverdampedGrowingFilament(
            FilamentState(remeshed_positions, remeshed_rest),
            ModelParameters(
                axial_stiffness=1.0,
                bending_stiffness=1.7,
                reference_length=1.0,
                reject_crossing=False,
            ),
        )
        epsilon = 1.0e-6
        finite_difference_force = np.zeros_like(remeshed_positions)
        for node in range(len(remeshed_positions)):
            for coordinate in range(2):
                plus = remeshed_positions.copy()
                minus = remeshed_positions.copy()
                plus[node, coordinate] += epsilon
                minus[node, coordinate] -= epsilon
                finite_difference_force[node, coordinate] = -(
                    model.energy(plus, remeshed_rest)
                    - model.energy(minus, remeshed_rest)
                ) / (2.0 * epsilon)
        np.testing.assert_allclose(
            model.forces(),
            finite_difference_force,
            rtol=2.0e-8,
            atol=2.0e-9,
        )

    def test_smooth_a_max_refinement_has_predefined_convergence_trend(self):
        records = run_resolution_benchmark()
        self.assertEqual([record["a_max"] for record in records], [2.0, 1.0, 0.5, 0.25])
        self.assertEqual(
            [record["n_nodes"] for record in records],
            sorted(record["n_nodes"] for record in records),
        )

        for record in records:
            self.assertLessEqual(record["max_rest_length"], record["a_max"])
            np.testing.assert_allclose(
                record["reference_length"],
                record["contour_length"],
                rtol=0.0,
                atol=2.0e-14,
            )
        contour_lengths = [record["contour_length"] for record in records]
        bend_energies = [record["bend_energy"] for record in records]
        radius_errors = [record["arc_radius_error"] for record in records]
        force_residuals = [record["interior_force_residual"] for record in records]

        # Chordal contour length and local-arc bending energy approach their
        # smooth-curve limits from below; the arc-length Rg error and interior
        # force residual decrease under refinement.
        self.assertTrue(all(b > a for a, b in zip(contour_lengths, contour_lengths[1:])))
        self.assertTrue(all(b > a for a, b in zip(bend_energies, bend_energies[1:])))
        self.assertTrue(all(b < a for a, b in zip(radius_errors, radius_errors[1:])))
        self.assertTrue(all(b < a for a, b in zip(force_residuals, force_residuals[1:])))

    def test_short_growth_remesh_benchmark_is_reproducible(self):
        first = run_short_growth_benchmark()
        second = run_short_growth_benchmark()
        self.assertEqual(first, second)
        self.assertEqual(first["node_counts"], [4, 7, 7, 7, 7])
        self.assertEqual(first["remesh_step_flags"], [1, 0, 0, 0])
        self.assertEqual(first["accepted_steps"], 4)
        self.assertEqual(first["rejected_steps"], 0)
        np.testing.assert_allclose(
            first["reference_lengths"],
            [3.0 * np.exp(0.1 * step) for step in range(5)],
            rtol=0.0,
            atol=2.0e-15,
        )
        np.testing.assert_allclose(first["contour_lengths"], 3.0, atol=2.0e-15)


if __name__ == "__main__":
    unittest.main()
