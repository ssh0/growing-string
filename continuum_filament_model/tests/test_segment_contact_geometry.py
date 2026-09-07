from __future__ import annotations

import unittest

import numpy as np

from growing_filament.geometry import (
    ContactDiagnosticType,
    NormalStatus,
    SegmentFeature,
    geometry_diagnostics,
    nonlocal_segment_contacts,
    segment_contact_geometry,
)
from growing_filament.model import remesh


class SegmentContactGeometryTest(unittest.TestCase):
    def test_straight_no_contact_has_analytic_gap_and_normal(self):
        result = segment_contact_geometry(
            [-1.0, 0.0],
            [1.0, 0.0],
            [2.0, 1.0],
            [4.0, 1.0],
            diameter=0.5,
            segment_i=3,
            segment_j=7,
        )

        self.assertEqual((result.segment_i, result.segment_j), (3, 7))
        self.assertAlmostEqual(result.distance, np.sqrt(2.0))
        self.assertAlmostEqual(result.gap, np.sqrt(2.0) - 0.5)
        self.assertEqual(result.penetration, 0.0)
        self.assertEqual(result.feature, SegmentFeature.ENDPOINT_ENDPOINT)
        self.assertEqual(result.normal_status, NormalStatus.DEFINED)
        np.testing.assert_allclose(result.normal, [-1.0 / np.sqrt(2.0), -1.0 / np.sqrt(2.0)])
        self.assertEqual(result.diagnostic_type, ContactDiagnosticType.NO_CONTACT)
        self.assertFalse(result.is_contact)

    def test_parallel_projection_overlap_is_deterministic(self):
        result = segment_contact_geometry(
            [0.0, 0.0],
            [2.0, 0.0],
            [0.5, 2.0],
            [1.5, 2.0],
            diameter=1.0,
        )

        self.assertAlmostEqual(result.distance, 2.0)
        self.assertAlmostEqual(result.parameter_i, 0.5)
        self.assertAlmostEqual(result.parameter_j, 0.5)
        np.testing.assert_allclose(result.point_i, [1.0, 0.0])
        np.testing.assert_allclose(result.point_j, [1.0, 2.0])
        self.assertEqual(result.feature, SegmentFeature.PARALLEL_OVERLAP)
        self.assertEqual(result.normal_status, NormalStatus.DEFINED)
        np.testing.assert_allclose(result.normal, [0.0, -1.0])

    def test_endpoint_interior_features_are_not_swapped(self):
        endpoint_on_first = segment_contact_geometry(
            [0.0, 0.0],
            [0.0, 1.0],
            [-1.0, 1.5],
            [1.0, 1.5],
            diameter=0.25,
        )
        self.assertEqual(endpoint_on_first.feature, SegmentFeature.ENDPOINT_INTERIOR)
        self.assertAlmostEqual(endpoint_on_first.parameter_i, 1.0)
        self.assertAlmostEqual(endpoint_on_first.parameter_j, 0.5)
        np.testing.assert_allclose(endpoint_on_first.point_i, [0.0, 1.0])
        np.testing.assert_allclose(endpoint_on_first.point_j, [0.0, 1.5])

        endpoint_on_second = segment_contact_geometry(
            [-1.0, 1.5],
            [1.0, 1.5],
            [0.0, 0.0],
            [0.0, 1.0],
            diameter=0.25,
        )
        self.assertEqual(endpoint_on_second.feature, SegmentFeature.INTERIOR_ENDPOINT)

    def test_near_parallel_and_threshold_sides_have_no_fabricated_normal(self):
        first = np.asarray([[0.0, 0.0], [10.0, 0.0]])
        second = np.asarray([[2.0, 1.0], [8.0, 1.0 + 1.0e-10]])
        result = segment_contact_geometry(*first, *second, diameter=1.0)
        self.assertTrue(np.isfinite(result.distance))
        self.assertGreater(result.distance, 0.0)
        self.assertEqual(result.normal_status, NormalStatus.DEFINED)
        self.assertIsNotNone(result.normal)
        self.assertAlmostEqual(np.linalg.norm(result.normal), 1.0)

        at_threshold = segment_contact_geometry(
            [0.0, 0.0], [2.0, 0.0], [0.5, 1.0], [1.5, 1.0], diameter=1.0
        )
        above = segment_contact_geometry(
            [0.0, 0.0], [2.0, 0.0], [0.5, 1.0 + 1.0e-9], [1.5, 1.0 + 1.0e-9], diameter=1.0
        )
        below = segment_contact_geometry(
            [0.0, 0.0], [2.0, 0.0], [0.5, 1.0 - 1.0e-9], [1.5, 1.0 - 1.0e-9], diameter=1.0
        )
        self.assertEqual(at_threshold.gap, 0.0)
        self.assertTrue(at_threshold.is_contact)
        self.assertFalse(above.is_contact)
        self.assertGreater(above.gap, 0.0)
        self.assertTrue(below.is_contact)
        self.assertAlmostEqual(below.penetration, 1.0e-9)

    def test_zero_distance_crossing_and_collinear_overlap_do_not_choose_normal(self):
        crossing = segment_contact_geometry(
            [-1.0, 0.0], [1.0, 0.0], [0.0, -1.0], [0.0, 1.0], diameter=0.4
        )
        self.assertEqual(crossing.distance, 0.0)
        self.assertEqual(crossing.feature, SegmentFeature.INTERIOR_INTERIOR)
        self.assertEqual(crossing.normal_status, NormalStatus.UNDEFINED_ZERO_DISTANCE)
        self.assertIsNone(crossing.normal)
        self.assertEqual(crossing.diagnostic_type, ContactDiagnosticType.CENTERLINE_INTERSECTION)
        self.assertEqual(crossing.penetration, 0.4)

        overlap = segment_contact_geometry(
            [0.0, 0.0], [2.0, 0.0], [1.0, 0.0], [3.0, 0.0], diameter=0.4
        )
        self.assertEqual(overlap.distance, 0.0)
        self.assertEqual(overlap.feature, SegmentFeature.COLLINEAR_OVERLAP)
        self.assertEqual(overlap.normal_status, NormalStatus.UNDEFINED_ZERO_DISTANCE)
        self.assertIsNone(overlap.normal)
        self.assertAlmostEqual(overlap.parameter_i, 0.75)
        self.assertAlmostEqual(overlap.parameter_j, 0.25)

    def test_centerline_intersection_and_finite_radius_gap_are_separate_diagnostics(self):
        crossing = geometry_diagnostics(
            np.asarray([[-1.0, 0.0], [1.0, 0.0], [0.0, -1.0], [0.0, 1.0]]),
            contact_distance=0.5,
        )
        self.assertEqual(len(crossing["centerline_intersections"]), 1)
        self.assertEqual(crossing["finite_radius_contacts"], [])
        self.assertEqual(crossing["segment_contacts"][0]["diagnostic_type"], "centerline_intersection")

        gap_contact = geometry_diagnostics(
            np.asarray([[-1.0, 0.0], [1.0, 0.0], [0.0, 0.4], [0.0, 1.4]]),
            contact_distance=0.5,
        )
        self.assertEqual(gap_contact["intersection_pairs"], [])
        self.assertEqual(len(gap_contact["finite_radius_contacts"]), 1)
        self.assertEqual(
            gap_contact["finite_radius_contacts"][0]["diagnostic_type"],
            "finite_radius_gap_contact",
        )

    def test_rigid_transform_preserves_geometry_and_rotates_normal(self):
        original = segment_contact_geometry(
            [-1.0, 0.0], [1.0, 0.0], [0.25, 1.5], [0.25, 2.5], diameter=0.75
        )
        angle = 0.731
        rotation = np.asarray(
            [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
        )
        translation = np.asarray([4.0, -3.0])
        transformed = segment_contact_geometry(
            rotation @ np.asarray([-1.0, 0.0]) + translation,
            rotation @ np.asarray([1.0, 0.0]) + translation,
            rotation @ np.asarray([0.25, 1.5]) + translation,
            rotation @ np.asarray([0.25, 2.5]) + translation,
            diameter=0.75,
        )

        self.assertAlmostEqual(original.distance, transformed.distance)
        self.assertAlmostEqual(original.gap, transformed.gap)
        self.assertAlmostEqual(original.penetration, transformed.penetration)
        self.assertEqual(original.feature, transformed.feature)
        self.assertEqual(original.normal_status, transformed.normal_status)
        np.testing.assert_allclose(transformed.point_i, rotation @ original.point_i + translation)
        np.testing.assert_allclose(transformed.point_j, rotation @ original.point_j + translation)
        np.testing.assert_allclose(transformed.normal, rotation @ original.normal)

    def test_nonlocal_pair_exclusion_and_remesh_diagnostic(self):
        adjacent_only = np.asarray([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]])
        self.assertEqual(nonlocal_segment_contacts(adjacent_only, diameter=2.0), ())

        positions = np.asarray(
            [[-1.0, 0.0], [1.0, 0.0], [1.0, 2.0], [-1.0, 2.0]]
        )
        rest_lengths = np.asarray([2.0, 2.0, 2.0])
        before = nonlocal_segment_contacts(positions, diameter=1.1)
        refined_positions, refined_lengths = remesh(positions, rest_lengths, a_max=1.0)
        after = nonlocal_segment_contacts(refined_positions, diameter=1.1)

        self.assertTrue(before)
        self.assertTrue(after)
        self.assertAlmostEqual(min(item.distance for item in before), 2.0)
        # Current-array non-adjacency is intentionally re-evaluated after
        # remeshing; newly separated children of formerly adjacent segments
        # can become eligible pairs.  The original long-range separation must
        # nevertheless remain diagnosable.
        self.assertTrue(any(np.isclose(item.distance, 2.0) for item in after))
        self.assertTrue(all(item.segment_j >= item.segment_i + 2 for item in after))
        self.assertTrue(np.all(refined_lengths <= 1.0))


if __name__ == "__main__":
    unittest.main()
