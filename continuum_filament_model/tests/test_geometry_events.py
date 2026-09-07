from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from growing_filament.geometry import (
    find_swept_nonlocal_intersection,
    geometry_diagnostics,
    initial_geometry_diagnostic,
    segment_closest_points,
)
from growing_filament.io import load_trajectory_metadata, save_trajectory
from growing_filament.model import (
    FilamentState,
    ModelError,
    ModelParameters,
    OverdampedGrowingFilament,
)
from growing_filament.reproducibility import compare_reproducibility


class GeometryEventsTest(unittest.TestCase):
    def test_nonlocal_segment_closest_points_and_contact_are_separate_from_node_contact(self):
        positions = np.asarray(
            [[-1.0, 0.0], [1.0, 0.0], [0.0, 0.5], [0.0, 1.5]],
        )
        distance, point_i, point_j, parameter_i, parameter_j = segment_closest_points(
            positions[0], positions[1], positions[2], positions[3]
        )
        self.assertAlmostEqual(distance, 0.5)
        np.testing.assert_allclose(point_i, [0.0, 0.0])
        np.testing.assert_allclose(point_j, [0.0, 0.5])
        self.assertEqual(parameter_i, 0.5)
        self.assertEqual(parameter_j, 0.0)

        diagnostic = geometry_diagnostics(positions, contact_distance=1.0)
        self.assertAlmostEqual(diagnostic["min_nonlocal_distance"], 0.5)
        self.assertEqual(diagnostic["contact_pairs"], [[0, 2]])
        self.assertEqual(diagnostic["intersection_pairs"], [])

        model = OverdampedGrowingFilament(
            FilamentState(positions, np.linalg.norm(np.diff(positions, axis=0), axis=1)),
            ModelParameters(
                contact_stiffness=0.0,
                diameter=1.0,
                reference_length=1.0,
                reject_crossing=True,
            ),
        )
        contact_events = [event for event in model.events if event["reason"] == "contact"]
        self.assertEqual(len(contact_events), 1)
        self.assertIn("contact_pairs", contact_events[0])
        self.assertEqual(contact_events[0]["diagnostic_event_type"], "finite_radius_gap_contact")
        self.assertEqual(contact_events[0]["centerline_intersections"], [])

    def test_initial_crossing_is_rejected_even_when_trial_crossing_switch_is_disabled(self):
        crossing = FilamentState(
            [[0.0, 0.0], [2.0, 2.0], [0.0, 2.0], [2.0, 0.0]],
            [np.sqrt(8.0), 2.0, np.sqrt(8.0)],
        )
        diagnostic = initial_geometry_diagnostic(crossing.positions)
        self.assertFalse(diagnostic["valid"])
        self.assertEqual(diagnostic["reason"], "initial_crossing")
        with self.assertRaisesRegex(ModelError, "initial_crossing") as context:
            OverdampedGrowingFilament(
                crossing,
                ModelParameters(reject_crossing=False),
            )
        self.assertIsNotNone(context.exception.event)
        self.assertEqual(context.exception.event["reason"], "initial_crossing")
        self.assertEqual(context.exception.event["diagnostic_event_type"], "centerline_intersection")
        self.assertEqual(len(context.exception.event["centerline_intersections"]), 1)
        self.assertFalse(context.exception.event["accepted"])

    def test_swept_crossing_detects_intermediate_counterexample(self):
        # At t=0 and t=1 the horizontal segment (0, 1) and vertical segment
        # (2, 3) are disjoint.  The latter translates through the former and
        # first touches it at t=0.29.
        start = np.asarray(
            [[-1.0, 0.0], [1.0, 0.0], [-2.16, -0.25], [-2.16, 0.25]],
        )
        end = np.asarray(
            [[-1.0, 0.0], [1.0, 0.0], [1.84, -0.25], [1.84, 0.25]],
        )
        self.assertEqual(initial_geometry_diagnostic(start)["intersection_pairs"], [])
        self.assertEqual(initial_geometry_diagnostic(end)["intersection_pairs"], [])
        crossing = find_swept_nonlocal_intersection(start, end)
        self.assertIsNotNone(crossing)
        assert crossing is not None
        self.assertEqual((crossing.segment_i, crossing.segment_j), (0, 2))
        self.assertAlmostEqual(crossing.normalized_time, 0.29, places=10)

        class PrescribedTrial(OverdampedGrowingFilament):
            def forces(self, positions, rest_lengths):
                weights = np.empty(len(rest_lengths) + 1)
                weights[0] = rest_lengths[0] / 2.0
                weights[-1] = rest_lengths[-1] / 2.0
                weights[1:-1] = 0.5 * (rest_lengths[:-1] + rest_lengths[1:])
                return (end - start) * weights[:, None]

        model = PrescribedTrial(
            FilamentState(start, [10.0, 10.0, 10.0]),
            ModelParameters(
                dt=1.0,
                a_max=100.0,
                max_displacement_fraction=1.0,
                max_retries=0,
                reject_crossing=True,
            ),
        )
        with self.assertRaisesRegex(RuntimeError, "swept"):
            model.step()
        attempt = [event for event in model.events if event["event_type"] == "step_attempt"][-1]
        self.assertEqual(attempt["reason"], "crossing_rejection")
        self.assertFalse(attempt["accepted"])
        self.assertAlmostEqual(attempt["swept_crossing"]["normalized_time"], 0.29, places=10)
        np.testing.assert_array_equal(model.state.positions, start)

    def test_events_include_rejection_reason_dt_state_and_energy_summaries(self):
        state = FilamentState(
            [[0.0, 0.0], [1.0, 0.4], [2.0, 0.4], [3.0, 0.0]],
            [1.0, 1.0, 1.0],
        )
        model = OverdampedGrowingFilament(
            state,
            ModelParameters(
                axial_stiffness=100.0,
                bending_stiffness=1.0,
                dt=1.0e-2,
                a_max=2.0,
                fixed_left=True,
                fixed_right=True,
                max_retries=0,
            ),
        )
        with self.assertRaisesRegex(RuntimeError, "growth-free trial energy increased"):
            model.step()
        attempt = [event for event in model.events if event["event_type"] == "step_attempt"][-1]
        self.assertEqual(attempt["reason"], "energy_increased")
        self.assertFalse(attempt["accepted"])
        self.assertEqual(attempt["requested_dt"], 1.0e-2)
        self.assertIsNone(attempt["accepted_dt"])
        self.assertIn("min_segment_length", attempt["state_trial"])
        self.assertIn("min_nonlocal_distance", attempt["state_trial"])
        self.assertIn("max_displacement", attempt)
        self.assertIn("energy_before", attempt)
        self.assertIn("energy_trial", attempt)
        self.assertEqual(model.rejected_steps, 1)
        self.assertEqual(model.accepted_steps, 0)

    def test_events_distinguish_displacement_excess_and_nonfinite_trials(self):
        class LargeTrial(OverdampedGrowingFilament):
            def forces(self, positions, rest_lengths):
                return np.full_like(positions, 10.0)

        large = LargeTrial(
            FilamentState([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]], [1.0, 1.0]),
            ModelParameters(dt=1.0, max_retries=0, a_max=2.0),
        )
        with self.assertRaises(RuntimeError):
            large.step()
        self.assertEqual(large.events[-1]["reason"], "displacement_exceeded")

        class NonfiniteTrial(OverdampedGrowingFilament):
            def forces(self, positions, rest_lengths):
                return np.full_like(positions, np.nan)

        nonfinite = NonfiniteTrial(
            FilamentState([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]], [1.0, 1.0]),
            ModelParameters(dt=1.0, max_retries=0, a_max=2.0),
        )
        with self.assertRaises(RuntimeError):
            nonfinite.step()
        self.assertEqual(nonfinite.events[-1]["reason"], "nonfinite")

    def _run_fixture(self) -> tuple[OverdampedGrowingFilament, list[FilamentState]]:
        model = OverdampedGrowingFilament(
            FilamentState(
                [[0.0, 0.0], [1.0, 0.2], [2.0, 0.0]],
                [1.0, 1.0],
            ),
            ModelParameters(
                axial_stiffness=4.0,
                bending_stiffness=0.5,
                dt=1.0e-5,
                t_end=2.0e-5,
                a_max=2.0,
                fixed_left=True,
                fixed_right=True,
            ),
        )
        return model, model.run()

    def test_replay_manifest_matches_event_sequence_and_canonical_state_hash(self):
        first_model, first_trajectory = self._run_fixture()
        second_model, second_trajectory = self._run_fixture()
        first = first_model.run_manifest(metadata={"fixture": "gate3"})
        second = second_model.run_manifest(metadata={"fixture": "gate3"})
        comparison = compare_reproducibility(first, second)
        self.assertTrue(comparison["match"], comparison["differences"])
        self.assertEqual(first["canonical_state_hash"], second["canonical_state_hash"])
        self.assertEqual(first["events"], second["events"])
        self.assertEqual(first["accepted_steps"], 2)
        self.assertEqual(first["rejected_steps"], 0)

        with tempfile.TemporaryDirectory() as directory:
            path = save_trajectory(
                Path(directory) / "fixture.npz",
                first_trajectory,
                first_model.parameters,
                metadata={"fixture": "gate3"},
                events=first_model.events,
                manifest=first,
            )
            saved = load_trajectory_metadata(path)
        self.assertEqual(saved["manifest"], first)
        self.assertEqual(saved["events"], first["events"])


if __name__ == "__main__":
    unittest.main()
