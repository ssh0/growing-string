"""Finite-radius and swept geometry diagnostics for open polylines.

The simulator still uses node-level soft contact as its only contact force.  The
functions in this module are deliberately force-free diagnostics: they measure
non-local segment separation and detect topological intersections without
adding segment repulsion, friction, or adhesion.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional

import numpy as np

Array = np.ndarray
_GEOMETRY_EPS = 1.0e-12


class SegmentFeature(str, Enum):
    """Feature pair attaining the segment-to-segment minimum.

    The names describe the closest-point parameters, not a force law.  A
    collinear overlap has a non-unique set of closest points and is therefore
    reported explicitly instead of being silently classified as an arbitrary
    endpoint/interior pair.  ``parallel_overlap`` has the analogous meaning
    for parallel, distinct centerlines with overlapping projections.
    """

    ENDPOINT_ENDPOINT = "endpoint_endpoint"
    ENDPOINT_INTERIOR = "endpoint_interior"
    INTERIOR_ENDPOINT = "interior_endpoint"
    INTERIOR_INTERIOR = "interior_interior"
    COLLINEAR_OVERLAP = "collinear_overlap"
    PARALLEL_OVERLAP = "parallel_overlap"


class NormalStatus(str, Enum):
    """Whether a unique separation normal is available."""

    DEFINED = "defined"
    UNDEFINED_ZERO_DISTANCE = "undefined_zero_distance"


class ContactDiagnosticType(str, Enum):
    """Mutually exclusive diagnostic classification for one segment pair."""

    NO_CONTACT = "no_contact"
    FINITE_RADIUS_GAP_CONTACT = "finite_radius_gap_contact"
    CENTERLINE_INTERSECTION = "centerline_intersection"


@dataclass(frozen=True)
class SegmentDistance:
    """Closest-point result for one non-local segment pair.

    This is the historical centerline-only API.  Use
    :func:`segment_contact_geometry` when a finite diameter, gap, penetration,
    feature, or normal status is required.
    """

    segment_i: int
    segment_j: int
    distance: float
    point_i: tuple[float, float]
    point_j: tuple[float, float]
    parameter_i: float
    parameter_j: float

    @property
    def closest_point_i(self) -> tuple[float, float]:
        return self.point_i

    @property
    def closest_point_j(self) -> tuple[float, float]:
        return self.point_j

    @property
    def closest_parameter_i(self) -> float:
        return self.parameter_i

    @property
    def closest_parameter_j(self) -> float:
        return self.parameter_j

    @property
    def closest_points(self) -> tuple[tuple[float, float], tuple[float, float]]:
        return self.point_i, self.point_j

    @property
    def parameters(self) -> tuple[float, float]:
        return self.parameter_i, self.parameter_j

    def as_contact(
        self,
        diameter: float,
        *,
        feature: SegmentFeature = SegmentFeature.INTERIOR_INTERIOR,
        normal_status: Optional[NormalStatus] = None,
    ) -> "SegmentContactGeometry":
        """Promote a centerline result to the finite-radius contract.

        Prefer :func:`segment_contact_geometry` when the feature and normal
        status must be inferred from the original endpoints.  This adapter is
        provided so existing callers that cache ``SegmentDistance`` values can
        adopt the new contract without changing their data flow.
        """

        diameter = _validate_diameter(diameter)
        gap = float(self.distance - diameter)
        penetration = max(0.0, float(-gap))
        if normal_status is None:
            normal_status = (
                NormalStatus.DEFINED
                if self.distance > _GEOMETRY_EPS
                else NormalStatus.UNDEFINED_ZERO_DISTANCE
            )
        else:
            normal_status = NormalStatus(normal_status)
        if self.distance <= _GEOMETRY_EPS and normal_status is NormalStatus.DEFINED:
            raise ValueError("zero-distance geometry cannot have a defined normal")
        normal: Optional[tuple[float, float]] = None
        if normal_status is NormalStatus.DEFINED:
            delta = np.asarray(self.point_i) - np.asarray(self.point_j)
            normal = (float(delta[0] / self.distance), float(delta[1] / self.distance))
        diagnostic_type = (
            ContactDiagnosticType.CENTERLINE_INTERSECTION
            if self.distance <= _GEOMETRY_EPS
            else (
                ContactDiagnosticType.FINITE_RADIUS_GAP_CONTACT
                if gap <= 0.0
                else ContactDiagnosticType.NO_CONTACT
            )
        )
        return SegmentContactGeometry(
            segment_i=self.segment_i,
            segment_j=self.segment_j,
            distance=float(self.distance),
            gap=gap,
            penetration=penetration,
            point_i=self.point_i,
            point_j=self.point_j,
            parameter_i=float(self.parameter_i),
            parameter_j=float(self.parameter_j),
            diameter=diameter,
            feature=feature,
            normal=normal,
            normal_status=normal_status,
            centerline_intersection=(diagnostic_type is ContactDiagnosticType.CENTERLINE_INTERSECTION),
            diagnostic_type=diagnostic_type,
        )

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
class SegmentContactGeometry:
    """Method-independent finite-radius geometry for one segment pair.

    ``diameter`` is the exclusion diameter used by the contract, so
    ``gap = distance - diameter`` and ``penetration = max(0, -gap)``.  A
    normal is returned only for strictly positive separation.  At zero
    distance (crossing or overlap), ``normal`` is ``None`` and
    ``normal_status`` records that no arbitrary direction was selected.
    """

    segment_i: int
    segment_j: int
    distance: float
    gap: float
    penetration: float
    point_i: tuple[float, float]
    point_j: tuple[float, float]
    parameter_i: float
    parameter_j: float
    diameter: float
    feature: SegmentFeature
    normal: Optional[tuple[float, float]]
    normal_status: NormalStatus
    centerline_intersection: bool
    diagnostic_type: ContactDiagnosticType

    @property
    def closest_point_i(self) -> tuple[float, float]:
        return self.point_i

    @property
    def closest_point_j(self) -> tuple[float, float]:
        return self.point_j

    @property
    def closest_parameter_i(self) -> float:
        return self.parameter_i

    @property
    def closest_parameter_j(self) -> float:
        return self.parameter_j

    @property
    def closest_points(self) -> tuple[tuple[float, float], tuple[float, float]]:
        return self.point_i, self.point_j

    @property
    def parameters(self) -> tuple[float, float]:
        return self.parameter_i, self.parameter_j

    @property
    def normal_vector(self) -> Optional[tuple[float, float]]:
        return self.normal

    @property
    def feature_type(self) -> SegmentFeature:
        return self.feature

    @property
    def is_contact(self) -> bool:
        """Whether the inclusive zero-gap contact threshold is met."""

        return self.gap <= 0.0

    @property
    def is_finite_radius_contact(self) -> bool:
        return self.diagnostic_type is ContactDiagnosticType.FINITE_RADIUS_GAP_CONTACT

    @property
    def centerline_crossing(self) -> bool:
        return self.centerline_intersection

    def as_dict(self) -> dict[str, object]:
        feature = self.feature.value
        normal_status = self.normal_status.value
        diagnostic_type = self.diagnostic_type.value
        return {
            "segment_i": self.segment_i,
            "segment_j": self.segment_j,
            "distance": float(self.distance),
            "gap": float(self.gap),
            "penetration": float(self.penetration),
            "diameter": float(self.diameter),
            "point_i": list(self.point_i),
            "point_j": list(self.point_j),
            "closest_point_i": list(self.point_i),
            "closest_point_j": list(self.point_j),
            "closest_points": [list(self.point_i), list(self.point_j)],
            "parameter_i": float(self.parameter_i),
            "parameter_j": float(self.parameter_j),
            "closest_parameter_i": float(self.parameter_i),
            "closest_parameter_j": float(self.parameter_j),
            "parameters": [float(self.parameter_i), float(self.parameter_j)],
            "feature": feature,
            "feature_type": feature,
            "normal": None if self.normal is None else list(self.normal),
            "normal_status": normal_status,
            "centerline_intersection": bool(self.centerline_intersection),
            "centerline_crossing": bool(self.centerline_intersection),
            "diagnostic_type": diagnostic_type,
            "is_contact": bool(self.is_contact),
        }


# Short noun aliases for callers that prefer concise annotations.
SegmentContact = SegmentContactGeometry
FeatureType = SegmentFeature


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


# Name the topological predicate explicitly for contact consumers.  The
# historical ``segments_intersect`` spelling remains the implementation API.
centerline_intersection = segments_intersect


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


def _validate_diameter(diameter: float) -> float:
    try:
        value = float(diameter)
    except (TypeError, ValueError) as exc:
        raise ValueError("diameter must be finite and non-negative") from exc
    if not np.isfinite(value) or value < 0.0:
        raise ValueError("diameter must be finite and non-negative")
    return value


def _is_collinear(a: Array, b: Array, c: Array, d: Array, eps: float) -> bool:
    u = b - a
    v = d - c
    w = c - a
    uu = float(np.linalg.norm(u))
    vv = float(np.linalg.norm(v))
    return (
        abs(_cross2(u, v)) <= eps * max(uu * vv, 1.0)
        and abs(_cross2(u, w)) <= eps * max(uu * float(np.linalg.norm(w)), 1.0)
    )


def _projected_interval_on_first(
    a: Array,
    b: Array,
    c: Array,
    d: Array,
) -> tuple[float, float]:
    u = b - a
    uu = float(np.dot(u, u))
    values = (float(np.dot(c - a, u) / uu), float(np.dot(d - a, u) / uu))
    return min(values), max(values)


def _collinear_overlap_parameters(
    a: Array,
    b: Array,
    c: Array,
    d: Array,
    eps: float,
) -> Optional[tuple[float, float]]:
    """Return deterministic midpoint parameters for a positive collinear overlap."""

    if not _is_collinear(a, b, c, d, eps):
        return None
    first_low, first_high = _projected_interval_on_first(a, b, c, d)
    overlap_low = max(0.0, first_low)
    overlap_high = min(1.0, first_high)
    if overlap_high - overlap_low <= eps:
        return None
    parameter_i = 0.5 * (overlap_low + overlap_high)
    v = d - c
    vv = float(np.dot(v, v))
    point = a + parameter_i * (b - a)
    parameter_j = float(np.clip(np.dot(point - c, v) / vv, 0.0, 1.0))
    return float(parameter_i), parameter_j


def _parallel_overlap_parameters(
    a: Array,
    b: Array,
    c: Array,
    d: Array,
    eps: float,
) -> Optional[tuple[float, float]]:
    """Return parameters for an exact/near-exact parallel projection overlap."""

    u = b - a
    v = d - c
    uu = float(np.dot(u, u))
    vv = float(np.dot(v, v))
    if abs(_cross2(u, v)) > eps * max(np.sqrt(uu * vv), 1.0):
        return None
    first_low, first_high = _projected_interval_on_first(a, b, c, d)
    overlap_low = max(0.0, first_low)
    overlap_high = min(1.0, first_high)
    if overlap_high - overlap_low <= eps:
        return None
    parameter_i = 0.5 * (overlap_low + overlap_high)
    point = a + parameter_i * u
    parameter_j = float(np.clip(np.dot(point - c, v) / vv, 0.0, 1.0))
    return float(parameter_i), parameter_j


def _parameter_is_endpoint(parameter: float, eps: float) -> bool:
    return parameter <= eps or parameter >= 1.0 - eps


def _feature_for_parameters(
    parameter_i: float,
    parameter_j: float,
    *,
    collinear_overlap: bool,
    parallel_overlap: bool,
    eps: float,
) -> SegmentFeature:
    if collinear_overlap:
        return SegmentFeature.COLLINEAR_OVERLAP
    if parallel_overlap:
        return SegmentFeature.PARALLEL_OVERLAP
    i_endpoint = _parameter_is_endpoint(parameter_i, eps)
    j_endpoint = _parameter_is_endpoint(parameter_j, eps)
    if i_endpoint and j_endpoint:
        return SegmentFeature.ENDPOINT_ENDPOINT
    if i_endpoint:
        return SegmentFeature.ENDPOINT_INTERIOR
    if j_endpoint:
        return SegmentFeature.INTERIOR_ENDPOINT
    return SegmentFeature.INTERIOR_INTERIOR


def segment_contact_geometry(
    a: Array,
    b: Array,
    c: Array,
    d: Array,
    diameter: float,
    *,
    segment_i: int = 0,
    segment_j: int = 1,
    eps: float = _GEOMETRY_EPS,
) -> SegmentContactGeometry:
    """Return the finite-radius geometry contract for one segment pair.

    The centerline calculation is independent of any contact response.  The
    returned normal points from the closest point on segment ``j`` to the
    closest point on segment ``i``.  When the centerline distance is zero, the
    direction is not unique for crossing/overlapping segments, so ``normal``
    is ``None`` and ``normal_status`` is ``undefined_zero_distance``.
    """

    if not isinstance(segment_i, (int, np.integer)) or not isinstance(segment_j, (int, np.integer)):
        raise ValueError("segment indices must be integers")
    if segment_i < 0 or segment_j < 0:
        raise ValueError("segment indices must be non-negative")
    diameter_value = _validate_diameter(diameter)
    a, b, c, d = (np.asarray(value, dtype=float) for value in (a, b, c, d))
    # The historical helper performs the complete shape/finite/degeneracy
    # validation and remains the numerical source of the closest pair.
    distance, point_i, point_j, parameter_i, parameter_j = segment_closest_points(
        a, b, c, d, eps=eps
    )

    collinear_overlap_parameters = _collinear_overlap_parameters(a, b, c, d, eps)
    parallel_overlap_parameters = _parallel_overlap_parameters(a, b, c, d, eps)
    if collinear_overlap_parameters is not None:
        parameter_i, parameter_j = collinear_overlap_parameters
        point_i = a + parameter_i * (b - a)
        point_j = c + parameter_j * (d - c)
        distance = float(np.linalg.norm(point_i - point_j))
    elif parallel_overlap_parameters is not None:
        # For parallel distinct centerlines there is a continuum of
        # minimizers.  Midpoint projection makes the diagnostic deterministic
        # without implying a preferred force application point.
        parameter_i, parameter_j = parallel_overlap_parameters
        point_i = a + parameter_i * (b - a)
        point_j = c + parameter_j * (d - c)
        distance = float(np.linalg.norm(point_i - point_j))

    collinear_overlap = collinear_overlap_parameters is not None
    parallel_overlap = (
        parallel_overlap_parameters is not None and not collinear_overlap
    )
    feature = _feature_for_parameters(
        parameter_i,
        parameter_j,
        collinear_overlap=collinear_overlap,
        parallel_overlap=parallel_overlap,
        eps=eps,
    )
    centerline_intersection = segments_intersect(a, b, c, d, eps=eps)
    gap = float(distance - diameter_value)
    penetration = max(0.0, float(diameter_value - distance))
    if distance <= eps:
        normal_status = NormalStatus.UNDEFINED_ZERO_DISTANCE
        normal = None
    else:
        normal_status = NormalStatus.DEFINED
        delta = point_i - point_j
        normal = (
            float(delta[0] / distance),
            float(delta[1] / distance),
        )
    if centerline_intersection:
        diagnostic_type = ContactDiagnosticType.CENTERLINE_INTERSECTION
    elif gap <= 0.0:
        diagnostic_type = ContactDiagnosticType.FINITE_RADIUS_GAP_CONTACT
    else:
        diagnostic_type = ContactDiagnosticType.NO_CONTACT
    return SegmentContactGeometry(
        segment_i=int(segment_i),
        segment_j=int(segment_j),
        distance=float(distance),
        gap=gap,
        penetration=penetration,
        point_i=(float(point_i[0]), float(point_i[1])),
        point_j=(float(point_j[0]), float(point_j[1])),
        parameter_i=float(parameter_i),
        parameter_j=float(parameter_j),
        diameter=diameter_value,
        feature=feature,
        normal=normal,
        normal_status=normal_status,
        centerline_intersection=centerline_intersection,
        diagnostic_type=diagnostic_type,
    )


# Explicit aliases keep the contract discoverable for callers that use either
# ``contact`` or ``geometry`` in their naming conventions.
segment_segment_contact_geometry = segment_contact_geometry
segment_contact = segment_contact_geometry


def nonlocal_segment_contacts(
    positions: Array,
    diameter: float,
    *,
    eps: float = _GEOMETRY_EPS,
) -> tuple[SegmentContactGeometry, ...]:
    """Return finite-radius geometry for all non-adjacent segment pairs.

    Segment indices are zero-based indices into this exact ``positions``
    snapshot.  Adjacent pairs (``j == i + 1``) are excluded, while pairs
    separated by at least one segment (``j >= i + 2``) are included, matching
    :func:`nonlocal_segment_distances` and the existing crossing contract.
    """

    values = _validate_positions(positions)
    _validate_diameter(diameter)
    result: list[SegmentContactGeometry] = []
    for i in range(len(values) - 1):
        for j in range(i + 2, len(values) - 1):
            result.append(
                segment_contact_geometry(
                    values[i],
                    values[i + 1],
                    values[j],
                    values[j + 1],
                    diameter,
                    segment_i=i,
                    segment_j=j,
                    eps=eps,
                )
            )
    return tuple(result)


nonlocal_segment_contact_geometry = nonlocal_segment_contacts
finite_radius_segment_contacts = nonlocal_segment_contacts


def closest_nonlocal_segment_contact(
    positions: Array,
    diameter: float,
    *,
    eps: float = _GEOMETRY_EPS,
) -> Optional[SegmentContactGeometry]:
    contacts = nonlocal_segment_contacts(positions, diameter, eps=eps)
    return min(contacts, key=lambda value: value.distance) if contacts else None


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
    *,
    include_all_segment_contacts: bool = True,
) -> dict[str, object]:
    """Summarize initial finite-radius geometry without applying a force law."""

    values = _validate_positions(positions)
    if not np.isfinite(contact_distance) or contact_distance < 0.0:
        raise ValueError("contact_distance must be finite and non-negative")
    segment_lengths = np.linalg.norm(np.diff(values, axis=0), axis=1)
    distances = nonlocal_segment_distances(values)
    closest = min(distances, key=lambda value: value.distance) if distances else None
    contacts = nonlocal_segment_contacts(values, contact_distance)
    intersections = tuple(
        (value.segment_i, value.segment_j)
        for value in contacts
        if value.centerline_intersection
    )
    # Keep the historical ``contact_pairs`` semantics (positive diameter and
    # inclusive threshold), while exposing explicit classifications for new
    # contact consumers.  A centerline intersection takes precedence over a
    # finite-radius gap contact in ``diagnostic_type``.
    contact_pairs = tuple(
        (value.segment_i, value.segment_j)
        for value in contacts
        if contact_distance > 0.0 and value.is_contact
    )
    finite_radius_contacts = tuple(
        value for value in contacts if value.is_finite_radius_contact
    )
    centerline_contacts = tuple(
        value for value in contacts if value.centerline_intersection
    )
    serialized_contacts = (
        contacts
        if include_all_segment_contacts
        else tuple(
            value for value in contacts
            if value.is_contact or value.centerline_intersection
        )
    )
    return {
        "valid": not bool(intersections),
        "reason": "initial_crossing" if intersections else (
            "contact" if contact_pairs else "initial_geometry_valid"
        ),
        "min_segment_length": float(np.min(segment_lengths)),
        "min_nonlocal_distance": float(closest.distance) if closest else float("inf"),
        "closest_nonlocal_pair": closest.as_dict() if closest else None,
        "intersection_pairs": [list(pair) for pair in intersections],
        "contact_pairs": [list(pair) for pair in contact_pairs],
        "segment_contacts": [value.as_dict() for value in serialized_contacts],
        "finite_radius_contacts": [value.as_dict() for value in finite_radius_contacts],
        "centerline_intersections": [value.as_dict() for value in centerline_contacts],
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
    *,
    include_all_segment_contacts: bool = True,
) -> dict[str, object]:
    """Return JSON-friendly finite-radius geometry measurements."""

    values = _validate_positions(positions)
    initial = initial_geometry_diagnostic(
        values,
        contact_distance=contact_distance,
        include_all_segment_contacts=include_all_segment_contacts,
    )
    return {
        "min_segment_length": initial["min_segment_length"],
        "min_nonlocal_distance": initial["min_nonlocal_distance"],
        "closest_nonlocal_pair": initial["closest_nonlocal_pair"],
        "intersection_pairs": initial["intersection_pairs"],
        "contact_pairs": initial["contact_pairs"],
        "segment_contacts": initial["segment_contacts"],
        "finite_radius_contacts": initial["finite_radius_contacts"],
        "centerline_intersections": initial["centerline_intersections"],
    }


# Short aliases make the diagnostic API discoverable without changing the
# explicit names used in the model and tests.
segment_segment_closest_points = segment_closest_points
swept_nonlocal_crossing = find_swept_nonlocal_intersection
