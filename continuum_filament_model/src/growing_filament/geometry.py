"""Finite-radius and swept geometry diagnostics for open polylines.

The simulator still uses node-level soft contact as its only contact force.  The
functions in this module are deliberately force-free diagnostics: they measure
non-local segment separation and detect topological intersections without
adding segment repulsion, friction, or adhesion.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

Array = np.ndarray
_GEOMETRY_EPS = 1.0e-12


@dataclass(frozen=True)
class SegmentDistance:
    """Closest-point result for one non-local segment pair."""

    segment_i: int
    segment_j: int
    distance: float
    point_i: tuple[float, float]
    point_j: tuple[float, float]
    parameter_i: float
    parameter_j: float

    def as_dict(self) -> dict[str, object]:
        return {
            "segment_i": self.segment_i,
            "segment_j": self.segment_j,
            "distance": self.distance,
            "point_i": list(self.point_i),
            "point_j": list(self.point_j),
            "parameter_i": self.parameter_i,
            "parameter_j": self.parameter_j,
        }


@dataclass(frozen=True)
class SweptIntersection:
    """First detected intersection during a linear endpoint trial."""

    segment_i: int
    segment_j: int
    normalized_time: float
    point: tuple[float, float]

    def as_dict(self) -> dict[str, object]:
        return {
            "segment_i": self.segment_i,
            "segment_j": self.segment_j,
            "normalized_time": self.normalized_time,
            "point": list(self.point),
        }


def _validate_positions(positions: Array) -> Array:
    values = np.asarray(positions, dtype=float)
    if values.ndim != 2 or values.shape[1] != 2 or values.shape[0] < 2:
        raise ValueError("positions must have shape (N, 2), with N >= 2")
    if not np.isfinite(values).all():
        raise ValueError("positions contain non-finite values")
    return values


def _cross2(a: Array, b: Array) -> float:
    return float(a[0] * b[1] - a[1] * b[0])


def _on_segment(a: Array, b: Array, p: Array, eps: float = _GEOMETRY_EPS) -> bool:
    return (
        min(a[0], b[0]) - eps <= p[0] <= max(a[0], b[0]) + eps
        and min(a[1], b[1]) - eps <= p[1] <= max(a[1], b[1]) + eps
    )


def segments_intersect(
    a: Array,
    b: Array,
    c: Array,
    d: Array,
    eps: float = _GEOMETRY_EPS,
) -> bool:
    """Return whether two closed 2-D segments intersect."""

    a, b, c, d = (np.asarray(value, dtype=float) for value in (a, b, c, d))
    ab = b - a
    ac = c - a
    ad = d - a
    cd = d - c
    ca = a - c
    cb = b - c
    o1 = _cross2(ab, ac)
    o2 = _cross2(ab, ad)
    o3 = _cross2(cd, ca)
    o4 = _cross2(cd, cb)

    if abs(o1) <= eps and _on_segment(a, b, c, eps):
        return True
    if abs(o2) <= eps and _on_segment(a, b, d, eps):
        return True
    if abs(o3) <= eps and _on_segment(c, d, a, eps):
        return True
    if abs(o4) <= eps and _on_segment(c, d, b, eps):
        return True
    return (o1 > 0.0) != (o2 > 0.0) and (o3 > 0.0) != (o4 > 0.0)


def segment_closest_points(
    a: Array,
    b: Array,
    c: Array,
    d: Array,
    eps: float = _GEOMETRY_EPS,
) -> tuple[float, Array, Array, float, float]:
    """Return distance, closest points, and segment parameters.

    The returned points are ``a + parameter_i * (b-a)`` and
    ``c + parameter_j * (d-c)``.  Degenerate geometric segments are rejected
    because they are not valid filament segments.
    """

    a, b, c, d = (np.asarray(value, dtype=float) for value in (a, b, c, d))
    if any(value.shape != (2,) for value in (a, b, c, d)):
        raise ValueError("segment endpoints must have shape (2,)")
    if not np.isfinite(np.concatenate((a, b, c, d))).all():
        raise ValueError("segment endpoints contain non-finite values")
    u = b - a
    v = d - c
    w = a - c
    uu = float(np.dot(u, u))
    vv = float(np.dot(v, v))
    if uu <= eps * eps or vv <= eps * eps:
        raise ValueError("zero-length geometric segment")

    uv = float(np.dot(u, v))
    uw = float(np.dot(u, w))
    vw = float(np.dot(v, w))
    denominator = uu * vv - uv * uv

    if denominator <= eps * max(uu * vv, 1.0):
        # Nearly parallel segments: minimizing over either endpoint and the
        # opposing segment is stable and also handles collinear overlap.
        candidates = [
            (0.0, float(np.clip(np.dot(a - c, v) / vv, 0.0, 1.0))),
            (1.0, float(np.clip(np.dot(b - c, v) / vv, 0.0, 1.0))),
            (float(np.clip(np.dot(c - a, u) / uu, 0.0, 1.0)), 0.0),
            (float(np.clip(np.dot(d - a, u) / uu, 0.0, 1.0)), 1.0),
        ]
    else:
        parameter_i = (uv * vw - vv * uw) / denominator
        parameter_j = (uu * vw - uv * uw) / denominator
        candidates = [(float(np.clip(parameter_i, 0.0, 1.0)), float(np.clip(parameter_j, 0.0, 1.0)))]
        # Clamping one parameter can move the optimum to the other segment's
        # endpoint.  Enumerating all four endpoint projections is inexpensive
        # for the small diagnostic pair counts used by this prototype.
        for parameter_i_candidate in (0.0, 1.0):
            point = a + parameter_i_candidate * u
            candidates.append(
                (parameter_i_candidate, float(np.clip(np.dot(point - c, v) / vv, 0.0, 1.0)))
            )
        for parameter_j_candidate in (0.0, 1.0):
            point = c + parameter_j_candidate * v
            candidates.append(
                (float(np.clip(np.dot(point - a, u) / uu, 0.0, 1.0)), parameter_j_candidate)
            )

    parameter_i, parameter_j = min(
        candidates,
        key=lambda pair: float(np.sum((a + pair[0] * u - c - pair[1] * v) ** 2)),
    )
    point_i = a + parameter_i * u
    point_j = c + parameter_j * v
    distance = float(np.linalg.norm(point_i - point_j))
    return distance, point_i, point_j, float(parameter_i), float(parameter_j)


def nonlocal_segment_distances(positions: Array) -> tuple[SegmentDistance, ...]:
    """Return closest-point diagnostics for all non-adjacent segment pairs."""

    values = _validate_positions(positions)
    result: list[SegmentDistance] = []
    for i in range(len(values) - 1):
        for j in range(i + 2, len(values) - 1):
            distance, point_i, point_j, parameter_i, parameter_j = segment_closest_points(
                values[i], values[i + 1], values[j], values[j + 1]
            )
            result.append(
                SegmentDistance(
                    i,
                    j,
                    distance,
                    (float(point_i[0]), float(point_i[1])),
                    (float(point_j[0]), float(point_j[1])),
                    parameter_i,
                    parameter_j,
                )
            )
    return tuple(result)


def closest_nonlocal_segment_pair(positions: Array) -> Optional[SegmentDistance]:
    distances = nonlocal_segment_distances(positions)
    return min(distances, key=lambda value: value.distance) if distances else None


def minimum_nonlocal_segment_distance(positions: Array) -> float:
    """Return the minimum non-local segment distance, or ``inf`` if absent."""

    closest = closest_nonlocal_segment_pair(positions)
    return float(closest.distance) if closest is not None else float("inf")


def nonlocal_intersection_pairs(positions: Array) -> tuple[tuple[int, int], ...]:
    """Return all intersecting non-adjacent segment index pairs."""

    values = _validate_positions(positions)
    pairs: list[tuple[int, int]] = []
    for i in range(len(values) - 1):
        for j in range(i + 2, len(values) - 1):
            if segments_intersect(values[i], values[i + 1], values[j], values[j + 1]):
                pairs.append((i, j))
    return tuple(pairs)


def has_nonlocal_intersection(positions: Array) -> bool:
    return bool(nonlocal_intersection_pairs(positions))


def initial_geometry_diagnostic(
    positions: Array,
    contact_distance: float = 0.0,
) -> dict[str, object]:
    """Summarize initial finite-radius geometry without applying a force law."""

    values = _validate_positions(positions)
    if not np.isfinite(contact_distance) or contact_distance < 0.0:
        raise ValueError("contact_distance must be finite and non-negative")
    segment_lengths = np.linalg.norm(np.diff(values, axis=0), axis=1)
    distances = nonlocal_segment_distances(values)
    closest = min(distances, key=lambda value: value.distance) if distances else None
    intersections = nonlocal_intersection_pairs(values)
    contacts = tuple(
        (value.segment_i, value.segment_j)
        for value in distances
        if contact_distance > 0.0 and value.distance <= contact_distance + _GEOMETRY_EPS
    )
    return {
        "valid": not bool(intersections),
        "reason": "initial_crossing" if intersections else (
            "contact" if contacts else "initial_geometry_valid"
        ),
        "min_segment_length": float(np.min(segment_lengths)),
        "min_nonlocal_distance": float(closest.distance) if closest else float("inf"),
        "closest_nonlocal_pair": closest.as_dict() if closest else None,
        "intersection_pairs": [list(pair) for pair in intersections],
        "contact_pairs": [list(pair) for pair in contacts],
    }


def _orientation_polynomial(
    a0: Array,
    a1: Array,
    b0: Array,
    b1: Array,
    c0: Array,
    c1: Array,
) -> np.ndarray:
    """Coefficients (constant first) of orient(a(t), b(t), c(t))."""

    u0 = b0 - a0
    u1 = (b1 - a1) - u0
    v0 = c0 - a0
    v1 = (c1 - a1) - v0
    return np.asarray(
        [_cross2(u0, v0), _cross2(u1, v0) + _cross2(u0, v1), _cross2(u1, v1)],
        dtype=float,
    )


def _polynomial_roots_in_unit_interval(coefficients: Array, eps: float) -> list[float]:
    values = np.asarray(coefficients, dtype=float)
    scale = max(float(np.max(np.abs(values))), 1.0)
    trimmed = values.copy()
    while len(trimmed) > 1 and abs(trimmed[-1]) <= eps * scale:
        trimmed = trimmed[:-1]
    if len(trimmed) <= 1:
        return []
    roots = np.roots(trimmed[::-1])
    result: list[float] = []
    for root in roots:
        if abs(float(np.imag(root))) <= 100.0 * eps:
            value = float(np.real(root))
            if -100.0 * eps <= value <= 1.0 + 100.0 * eps:
                result.append(float(np.clip(value, 0.0, 1.0)))
    return result


def _endpoint_coincidence_roots(
    first_start: Array,
    first_end: Array,
    second_start: Array,
    second_end: Array,
    eps: float,
) -> list[float]:
    """Return times at which two moving endpoints can coincide."""

    roots: list[float] = []
    first_delta = first_end - first_start
    second_delta = second_end - second_start
    difference_start = first_start - second_start
    difference_delta = first_delta - second_delta
    for coordinate in range(2):
        if abs(difference_delta[coordinate]) <= eps:
            continue
        time = -difference_start[coordinate] / difference_delta[coordinate]
        if -100.0 * eps <= time <= 1.0 + 100.0 * eps:
            roots.append(float(np.clip(time, 0.0, 1.0)))
    return roots


def _interpolated_positions(start: Array, end: Array, time: float) -> Array:
    return start + time * (end - start)


def find_swept_nonlocal_intersection(
    start_positions: Array,
    end_positions: Array,
    eps: float = _GEOMETRY_EPS,
) -> Optional[SweptIntersection]:
    """Find a non-local segment intersection during a linear endpoint trial.

    Endpoint orientations are quadratic polynomials in normalized time.  Their
    roots partition ``[0, 1]`` into intervals where the intersection predicate
    is constant for non-degenerate configurations.  Each interval and root is
    checked, so an intersection that is absent at both endpoints is not hidden
    by an end-state-only test.
    """

    start = _validate_positions(start_positions)
    end = _validate_positions(end_positions)
    if start.shape != end.shape:
        raise ValueError("start_positions and end_positions must have the same shape")

    n_segments = len(start) - 1
    for i in range(n_segments):
        for j in range(i + 2, n_segments):
            a0, a1 = start[i], end[i]
            b0, b1 = start[i + 1], end[i + 1]
            c0, c1 = start[j], end[j]
            d0, d1 = start[j + 1], end[j + 1]
            polynomials = (
                _orientation_polynomial(a0, a1, b0, b1, c0, c1),
                _orientation_polynomial(a0, a1, b0, b1, d0, d1),
                _orientation_polynomial(c0, c1, d0, d1, a0, a1),
                _orientation_polynomial(c0, c1, d0, d1, b0, b1),
            )
            candidates = {0.0, 1.0}
            for polynomial in polynomials:
                candidates.update(_polynomial_roots_in_unit_interval(polynomial, eps))
            # If a pair remains collinear for an interval, orientation roots
            # alone do not reveal when endpoint projections start/stop
            # overlapping.  Endpoint-coincidence roots cover that degenerate
            # case without changing the generic quadratic-root path.
            for first, first_end in ((a0, a1), (b0, b1)):
                for second, second_end in ((c0, c1), (d0, d1)):
                    candidates.update(
                        _endpoint_coincidence_roots(
                            first,
                            first_end,
                            second,
                            second_end,
                            eps,
                        )
                    )
            times = sorted(candidates)

            def intersection_at(time: float) -> bool:
                positions = _interpolated_positions(start, end, time)
                return segments_intersect(
                    positions[i], positions[i + 1], positions[j], positions[j + 1], eps
                )

            for time in times:
                if intersection_at(time):
                    positions = _interpolated_positions(start, end, time)
                    point = 0.25 * (
                        positions[i] + positions[i + 1] + positions[j] + positions[j + 1]
                    )
                    return SweptIntersection(i, j, float(time), (float(point[0]), float(point[1])))

            for left, right in zip(times, times[1:]):
                if right - left <= eps:
                    continue
                middle = 0.5 * (left + right)
                if not intersection_at(middle):
                    continue
                # Refine the first true point in this interval.  The exact
                # contact can be a root, so keep the returned time conservative.
                low, high = left, middle
                for _ in range(60):
                    if high - low <= 1.0e-12:
                        break
                    candidate = 0.5 * (low + high)
                    if intersection_at(candidate):
                        high = candidate
                    else:
                        low = candidate
                time = high
                positions = _interpolated_positions(start, end, time)
                point = 0.25 * (
                    positions[i] + positions[i + 1] + positions[j] + positions[j + 1]
                )
                return SweptIntersection(i, j, float(time), (float(point[0]), float(point[1])))
    return None


def has_swept_nonlocal_intersection(start_positions: Array, end_positions: Array) -> bool:
    return find_swept_nonlocal_intersection(start_positions, end_positions) is not None


def geometry_diagnostics(
    positions: Array,
    contact_distance: float = 0.0,
) -> dict[str, object]:
    """Return JSON-friendly finite-radius geometry measurements."""

    values = _validate_positions(positions)
    initial = initial_geometry_diagnostic(values, contact_distance=contact_distance)
    return {
        "min_segment_length": initial["min_segment_length"],
        "min_nonlocal_distance": initial["min_nonlocal_distance"],
        "closest_nonlocal_pair": initial["closest_nonlocal_pair"],
        "intersection_pairs": initial["intersection_pairs"],
        "contact_pairs": initial["contact_pairs"],
    }


# Short aliases make the diagnostic API discoverable without changing the
# explicit names used in the model and tests.
segment_segment_closest_points = segment_closest_points
swept_nonlocal_crossing = find_swept_nonlocal_intersection
