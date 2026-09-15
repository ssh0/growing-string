from __future__ import annotations

import unittest

import numpy as np

from continuum_filament_model.benchmarks.contact_folding_validation import initial_state
from growing_filament.geometry import (
    initial_geometry_diagnostic,
    segments_intersect,
)
from growing_filament.model import (
    FilamentState,
    ModelError,
    ModelParameters,
    OverdampedGrowingFilament,
    remesh_with_lineage,
)


class SegmentIntersectionTest(unittest.TestCase):
    @staticmethod
    def _subdivided_sine_fixture() -> tuple[np.ndarray, np.ndarray, tuple[str, ...]]:
        base = initial_state(
            {
                "length": 4.0,
                "n_nodes": 9,
                "amplitude": 0.55,
                "rest_length_factor": 0.75,
                "initial_shape": "sine",
            }
        )
        return remesh_with_lineage(
            base.positions,
            base.rest_lengths,
            base.segment_lineage,
            0.2,
        )

    def test_exact_midpoint_subdivision_does_not_create_crossing(self):
        positions, rest_lengths, lineage = self._subdivided_sine_fixture()

        self.assertEqual(positions.shape, (25, 2))
        self.assertEqual(rest_lengths.shape, (24,))
        self.assertFalse(
            segments_intersect(positions[21], positions[22], positions[23], positions[24])
        )
        diagnostic = initial_geometry_diagnostic(positions, contact_distance=0.5)
        self.assertTrue(diagnostic["valid"])
        self.assertEqual(diagnostic["intersection_pairs"], [])
        self.assertIn([21, 23], diagnostic["contact_pairs"])
        self.assertEqual(diagnostic["centerline_intersections"], [])
        OverdampedGrowingFilament(
            FilamentState(positions, rest_lengths, segment_lineage=lineage),
            ModelParameters(diameter=0.5, reject_crossing=True),
        )

    def test_genuine_crossing_remains_rejected(self):
        crossing = np.asarray(
            [[0.0, 0.0], [2.0, 2.0], [0.0, 2.0], [2.0, 0.0]],
        )
        self.assertTrue(segments_intersect(crossing[0], crossing[1], crossing[2], crossing[3]))
        diagnostic = initial_geometry_diagnostic(crossing)
        self.assertFalse(diagnostic["valid"])
        self.assertEqual(diagnostic["intersection_pairs"], [[0, 2]])
        with self.assertRaisesRegex(ModelError, "initial_crossing"):
            OverdampedGrowingFilament(
                FilamentState(crossing, np.linalg.norm(np.diff(crossing, axis=0), axis=1)),
                ModelParameters(reject_crossing=True),
            )

    def test_collinear_disjoint_and_overlap_semantics(self):
        disjoint = (
            np.asarray([0.0, 0.0]),
            np.asarray([1.0, 0.0]),
            np.asarray([2.0, 0.0]),
            np.asarray([3.0, 0.0]),
        )
        overlap = (
            np.asarray([0.0, 0.0]),
            np.asarray([3.0, 0.0]),
            np.asarray([1.0, 0.0]),
            np.asarray([4.0, 0.0]),
        )
        self.assertFalse(segments_intersect(*disjoint))
        self.assertTrue(segments_intersect(*overlap))
        self.assertFalse(segments_intersect(disjoint[1], disjoint[0], disjoint[2], disjoint[3]))
        self.assertTrue(segments_intersect(overlap[1], overlap[0], overlap[3], overlap[2]))

    def test_endpoint_touch_and_endpoint_order_symmetry(self):
        first = np.asarray([0.0, 0.0])
        second = np.asarray([2.0, 0.0])
        third = np.asarray([2.0, 0.0])
        fourth = np.asarray([2.0, 2.0])
        endpoint_variants = (
            (first, second, third, fourth),
            (second, first, third, fourth),
            (first, second, fourth, third),
            (second, first, fourth, third),
            (third, fourth, first, second),
            (fourth, third, first, second),
            (third, fourth, second, first),
            (fourth, third, second, first),
        )
        for variant in endpoint_variants:
            with self.subTest(variant=variant):
                self.assertTrue(segments_intersect(*variant))

    def test_shallow_and_coordinate_scaled_crossings_remain_detected(self):
        shallow = (
            np.asarray([0.0, 0.0]),
            np.asarray([1.0, 1.0e-13]),
            np.asarray([0.0, 1.0e-13]),
            np.asarray([1.0, 0.0]),
        )
        self.assertTrue(segments_intersect(*shallow))

        unit_crossing = np.asarray(
            [[0.0, 0.0], [1.0, 1.0], [0.0, 1.0], [1.0, 0.0]],
        )
        for scale in (1.0e-6, 1.0, 1.0e6):
            with self.subTest(scale=scale):
                scaled = unit_crossing * scale
                self.assertTrue(segments_intersect(scaled[0], scaled[1], scaled[2], scaled[3]))

    def test_subdivision_false_positive_stays_false_across_coordinate_scales(self):
        positions, _, _ = self._subdivided_sine_fixture()
        for scale in (1.0e-6, 1.0, 1.0e6):
            with self.subTest(scale=scale):
                scaled = positions * scale
                self.assertFalse(segments_intersect(scaled[21], scaled[22], scaled[23], scaled[24]))


if __name__ == "__main__":
    unittest.main()
