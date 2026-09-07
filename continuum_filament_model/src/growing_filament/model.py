"""Energy-based prototype for a growing semiflexible filament.

The implementation deliberately starts with a small, explicit model:

* two-dimensional open filament;
* overdamped dynamics on an isotropic dissipative substrate;
* axial stretching and discrete bending energies;
* optional node-level soft contact plus segment-intersection rejection;
* exponential growth of the local reference lengths;
* midpoint remeshing when a reference segment becomes too long.

This is a research prototype, not a validated reproduction of the legacy
model or of the bacterial experiment. See ``notes/model_spec.md`` and
``notes/validation_plan.md`` before using it for scientific claims.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np

from .geometry import (
    SegmentDistance,
    SweptIntersection,
    find_swept_nonlocal_intersection,
    geometry_diagnostics,
    has_nonlocal_intersection,
    has_swept_nonlocal_intersection,
    initial_geometry_diagnostic,
    minimum_nonlocal_segment_distance,
    nonlocal_intersection_pairs,
    nonlocal_segment_distances,
    segment_closest_points,
    segments_intersect,
)

Array = np.ndarray


class ModelError(ValueError):
    """Raised when a state or model parameter violates an invariant.

    Geometry-validation failures may expose a JSON-compatible ``event``
    attribute so callers can retain the rejected initial-condition record even
    though no simulator instance is created.
    """

    def __init__(self, message: str, event: Optional[dict[str, object]] = None):
        super().__init__(message)
        self.event = event


@dataclass(frozen=True)
class ModelParameters:
    """Physical and numerical parameters for the prototype.

    ``axial_stiffness`` corresponds to the continuum-like quantity ``EA``.
    ``bending_stiffness`` corresponds to ``EI``. ``drag_density`` is the
    substrate drag per reference length; node drag is derived from the local
    reference lengths at every step.
    """

    axial_stiffness: float = 100.0
    bending_stiffness: float = 1.0
    drag_density: float = 1.0
    contact_stiffness: float = 0.0
    diameter: float = 0.0
    growth_rate: float = 0.0
    reference_length: float = 1.0
    dt: float = 1.0e-4
    t_end: float = 1.0
    a_max: float = 1.5
    dt_min: float = 1.0e-10
    max_retries: int = 12
    max_displacement_fraction: float = 0.25
    # Relative scale for the growth-free energy acceptance test.  Growth can
    # inject energy, so this condition is deliberately disabled when
    # ``growth_rate`` is non-zero.
    energy_tolerance: float = 1.0e-12
    fixed_left: bool = False
    fixed_right: bool = False
    reject_crossing: bool = True

    def validate(self) -> None:
        numeric = {
            "axial_stiffness": self.axial_stiffness,
            "bending_stiffness": self.bending_stiffness,
            "drag_density": self.drag_density,
            "contact_stiffness": self.contact_stiffness,
            "diameter": self.diameter,
            "growth_rate": self.growth_rate,
            "reference_length": self.reference_length,
            "dt": self.dt,
            "t_end": self.t_end,
            "a_max": self.a_max,
            "dt_min": self.dt_min,
            "max_retries": self.max_retries,
            "max_displacement_fraction": self.max_displacement_fraction,
            "energy_tolerance": self.energy_tolerance,
        }
        for name, value in numeric.items():
            try:
                finite = bool(np.isfinite(value))
            except (TypeError, ValueError):
                finite = False
            if not finite:
                raise ModelError(f"{name} must be finite: {value!r}")

        positive = {
            "axial_stiffness": self.axial_stiffness,
            "bending_stiffness": self.bending_stiffness,
            "drag_density": self.drag_density,
            "reference_length": self.reference_length,
            "dt": self.dt,
            "t_end": self.t_end,
            "a_max": self.a_max,
            "dt_min": self.dt_min,
        }
        for name, value in positive.items():
            if value <= 0.0:
                raise ModelError(f"{name} must be positive: {value}")
        if self.contact_stiffness < 0.0:
            raise ModelError("contact_stiffness must be non-negative")
        if self.diameter < 0.0:
            raise ModelError("diameter must be non-negative")
        if self.growth_rate < 0.0:
            raise ModelError("growth_rate must be non-negative in this prototype")
        if self.max_retries < 0:
            raise ModelError("max_retries must be non-negative")
        if int(self.max_retries) != self.max_retries:
            raise ModelError("max_retries must be an integer")
        if not 0.0 < self.max_displacement_fraction <= 1.0:
            raise ModelError("max_displacement_fraction must be in (0, 1]")
        if self.energy_tolerance < 0.0:
            raise ModelError("energy_tolerance must be non-negative")


@dataclass
class FilamentState:
    """State of an open filament.

    ``positions`` has shape ``(N, 2)`` and ``rest_lengths`` has shape
    ``(N-1,)``. The model intentionally stores reference lengths separately
    from current geometric lengths.
    """

    positions: Array
    rest_lengths: Array
    time: float = 0.0
    step: int = 0

    def __post_init__(self) -> None:
        self.positions = np.asarray(self.positions, dtype=float).copy()
        self.rest_lengths = np.asarray(self.rest_lengths, dtype=float).copy()
        self.validate()

    @property
    def n_nodes(self) -> int:
        return int(self.positions.shape[0])

    @property
    def n_segments(self) -> int:
        return self.n_nodes - 1

    def copy(self) -> "FilamentState":
        return FilamentState(
            self.positions.copy(),
            self.rest_lengths.copy(),
            time=self.time,
            step=self.step,
        )

    def validate(self, eps: float = 1.0e-12) -> None:
        if self.positions.ndim != 2 or self.positions.shape[1] != 2:
            raise ModelError("positions must have shape (N, 2)")
        if self.n_nodes < 3:
            raise ModelError("at least three nodes are required")
        if self.rest_lengths.shape != (self.n_segments,):
            raise ModelError("rest_lengths must have shape (N-1,)")
        if not np.isfinite(self.positions).all():
            raise ModelError("positions contain non-finite values")
        if not np.isfinite(self.rest_lengths).all():
            raise ModelError("rest_lengths contain non-finite values")
        if np.any(self.rest_lengths <= eps):
            raise ModelError("all rest lengths must be positive")
        edge_lengths = np.linalg.norm(np.diff(self.positions, axis=0), axis=1)
        if np.any(edge_lengths <= eps):
            raise ModelError("zero-length geometric segment")


def straight_state(n_nodes: int, spacing: float = 1.0) -> FilamentState:
    """Create a straight, equally spaced open filament."""

    if n_nodes < 3:
        raise ModelError("n_nodes must be at least 3")
    if spacing <= 0.0:
        raise ModelError("spacing must be positive")
    positions = np.column_stack((np.arange(n_nodes) * spacing,
                                 np.zeros(n_nodes)))
    return FilamentState(positions, np.full(n_nodes - 1, spacing))


def grow_reference_lengths(rest_lengths: Array, growth_rate: float, dt: float) -> Array:
    """Apply local exponential reference-length growth for one time step."""

    if not np.isfinite(growth_rate) or not np.isfinite(dt):
        raise ModelError("growth_rate and dt must be finite")
    if growth_rate < 0.0 or dt < 0.0:
        raise ModelError("growth_rate and dt must be non-negative")
    values = np.asarray(rest_lengths, dtype=float)
    if not np.isfinite(values).all():
        raise ModelError("rest_lengths contain non-finite values")
    result = values * np.exp(growth_rate * dt)
    if not np.isfinite(result).all():
        raise ModelError("reference-length growth produced non-finite values")
    return result


def remesh(positions: Array, rest_lengths: Array, a_max: float) -> Tuple[Array, Array]:
    """Split long reference segments at their geometric midpoint.

    This operation is numerical remeshing, not physical material addition.
    It preserves total reference length and the endpoints.
    """

    if a_max <= 0.0:
        raise ModelError("a_max must be positive")
    p = np.asarray(positions, dtype=float).copy()
    a = np.asarray(rest_lengths, dtype=float).copy()
    if p.ndim != 2 or p.shape[1] != 2 or a.shape != (len(p) - 1,):
        raise ModelError("invalid remeshing shapes")
    if not np.isfinite(p).all() or not np.isfinite(a).all():
        raise ModelError("remeshing inputs contain non-finite values")

    # A segment can require several splits when a deliberately small a_max is
    # used. The loop is finite because every split halves the reference length.
    i = 0
    while i < len(a):
        if a[i] <= a_max:
            i += 1
            continue
        midpoint = 0.5 * (p[i] + p[i + 1])
        old = a[i]
        p = np.insert(p, i + 1, midpoint, axis=0)
        a[i] = old / 2.0
        a = np.insert(a, i + 1, old / 2.0)
        # Revisit the left child; either child may still exceed a_max.
    return p, a


EVENT_SCHEMA_VERSION = "continuum-filament-events-1"


def _node_contact_pairs(positions: Array, diameter: float) -> list[list[int]]:
    if diameter <= 0.0:
        return []
    pairs: list[list[int]] = []
    for i in range(len(positions)):
        for j in range(i + 2, len(positions)):
            if float(np.linalg.norm(positions[i] - positions[j])) <= diameter + 1.0e-12:
                pairs.append([i, j])
    return pairs


def _state_summary(
    positions: Array,
    rest_lengths: Array,
    diameter: float = 0.0,
    max_displacement: Optional[float] = None,
) -> Dict[str, object]:
    """Return a JSON-friendly state and finite-radius diagnostic summary."""

    p = np.asarray(positions, dtype=float)
    a = np.asarray(rest_lengths, dtype=float)
    finite = bool(np.isfinite(p).all() and np.isfinite(a).all())
    summary: Dict[str, object] = {
        "finite": finite,
        "n_nodes": int(len(p)) if p.ndim >= 1 else 0,
        "n_segments": int(len(a)) if a.ndim >= 1 else 0,
        "max_displacement": (
            None if max_displacement is None else float(max_displacement)
        ),
    }
    if not finite or p.ndim != 2 or p.shape[1] != 2 or len(a) != len(p) - 1:
        summary.update(
            {
                "min_segment_length": None,
                "min_nonlocal_distance": None,
                "closest_nonlocal_pair": None,
                "intersection_pairs": [],
                "contact_pairs": [],
                "node_contact_pairs": [],
                "reference_length": None,
                "contour_length": None,
            }
        )
        return summary
    lengths = np.linalg.norm(np.diff(p, axis=0), axis=1)
    min_segment_length = float(np.min(lengths)) if len(lengths) else None
    try:
        geometry = geometry_diagnostics(p, contact_distance=diameter)
        min_nonlocal_distance = geometry["min_nonlocal_distance"]
        if not np.isfinite(float(min_nonlocal_distance)):
            min_nonlocal_distance = None
        summary.update(
            {
                **geometry,
                "node_contact_pairs": _node_contact_pairs(p, diameter),
                "min_segment_length": min_segment_length,
                "min_nonlocal_distance": min_nonlocal_distance,
                "reference_length": float(np.sum(a)),
                "contour_length": float(np.sum(lengths)),
            }
        )
    except (ValueError, FloatingPointError):
        summary.update(
            {
                "min_segment_length": min_segment_length,
                "min_nonlocal_distance": None,
                "closest_nonlocal_pair": None,
                "intersection_pairs": [],
                "contact_pairs": [],
                "node_contact_pairs": [],
                "reference_length": float(np.sum(a)),
                "contour_length": float(np.sum(lengths)),
            }
        )
    return summary


def _node_weights(rest_lengths: Array) -> Array:
    """Assign reference-length weights to nodes for substrate drag."""

    weights = np.empty(len(rest_lengths) + 1, dtype=float)
    weights[0] = 0.5 * rest_lengths[0]
    weights[-1] = 0.5 * rest_lengths[-1]
    if len(rest_lengths) > 1:
        weights[1:-1] = 0.5 * (rest_lengths[:-1] + rest_lengths[1:])
    return weights


def _bending_energy_and_forces(
    positions: Array,
    rest_lengths: Array,
    bending_stiffness: float,
) -> Tuple[float, Array]:
    """Return local-arc-length bending energy and its conservative force.

    ``t_i`` is the unit tangent of geometric segment ``i`` and ``h_i`` is
    the reference-length dual cell around interior node ``i``.  The bending
    energy is ``EI / 2 * sum(|t_i - t_{i-1}|^2 / h_i)``.  Keeping the dual
    cell in reference coordinates makes growth and the bending discretization
    use the same local material measure, while the unit tangent keeps a
    straight, stretched segment free of spurious bending energy.
    """

    forces = np.zeros_like(positions)
    if len(positions) < 3:
        return 0.0, forces

    edges = np.diff(positions, axis=0)
    geometric_lengths = np.linalg.norm(edges, axis=1)
    if np.any(geometric_lengths <= 1.0e-12):
        raise ModelError("zero-length geometric segment")
    tangents = edges / geometric_lengths[:, None]
    local_reference_lengths = 0.5 * (rest_lengths[:-1] + rest_lengths[1:])
    tangent_jumps = tangents[1:] - tangents[:-1]
    coefficients = bending_stiffness / local_reference_lengths
    bending = 0.5 * float(
        np.sum(coefficients * np.sum(tangent_jumps * tangent_jumps, axis=1))
    )

    identity = np.eye(positions.shape[1])
    for i, (jump, coefficient) in enumerate(
        zip(tangent_jumps, coefficients), start=1
    ):
        previous_tangent = tangents[i - 1]
        next_tangent = tangents[i]
        previous_projection = identity - np.outer(previous_tangent, previous_tangent)
        next_projection = identity - np.outer(next_tangent, next_tangent)
        previous_gradient = previous_projection @ jump / geometric_lengths[i - 1]
        next_gradient = next_projection @ jump / geometric_lengths[i]
        forces[i - 1] -= coefficient * previous_gradient
        forces[i] += coefficient * (previous_gradient + next_gradient)
        forces[i + 1] -= coefficient * next_gradient
    return bending, forces


class OverdampedGrowingFilament:
    """Energy-based overdamped simulator for a single open filament."""

    def __init__(self, state: FilamentState, parameters: ModelParameters):
        parameters.validate()
        state.validate()
        initial_geometry = initial_geometry_diagnostic(
            state.positions,
            contact_distance=parameters.diameter,
        )
        # Initial topology is a hard invariant.  ``reject_crossing=False`` is
        # retained for force/energy benchmarks, but it must not be used to
        # admit an already self-intersecting filament.
        if not bool(initial_geometry["valid"]):
            pairs = initial_geometry["intersection_pairs"]
            event = {
                "schema_version": EVENT_SCHEMA_VERSION,
                "event_type": "initialization",
                "reason": "initial_crossing",
                "accepted": False,
                "requested_dt": None,
                "trial_dt": None,
                "accepted_dt": None,
                "time_before": float(state.time),
                "time_after": float(state.time),
                "step_before": int(state.step),
                "step_after": int(state.step),
                "state_before": None,
                "state_trial": _state_summary(
                    state.positions,
                    state.rest_lengths,
                    parameters.diameter,
                ),
                "state_after": None,
                "energy_before": None,
                "energy_trial": None,
                "energy_after": None,
                "geometry": initial_geometry,
                "detail": f"initial non-local segment intersection for pairs {pairs}",
            }
            raise ModelError(
                "initial_crossing: initial non-local segment intersection "
                f"detected for pairs {pairs}",
                event=event,
            )
        self.parameters = parameters
        self.state = state.copy()
        self.left_anchor = self.state.positions[0].copy()
        self.right_anchor = self.state.positions[-1].copy()
        self.initial_state = self.state.copy()
        self.initial_geometry = initial_geometry
        self.accepted_steps = 0
        self.rejected_steps = 0
        self.accepted_dts: list[float] = []
        self.rejected_dts: list[float] = []
        self.rejection_reasons: list[str] = []
        self.events: list[dict[str, object]] = []
        self._last_swept_intersection: Optional[SweptIntersection] = None
        self._append_event(
            {
                "event_type": "initialization",
                "reason": "initial_geometry_valid",
                "accepted": True,
                "requested_dt": None,
                "trial_dt": None,
                "accepted_dt": None,
                "time_before": None,
                "time_after": float(self.state.time),
                "step_before": None,
                "step_after": int(self.state.step),
                "state_before": None,
                "state_trial": _state_summary(
                    self.state.positions,
                    self.state.rest_lengths,
                    self.parameters.diameter,
                ),
                "state_after": _state_summary(
                    self.state.positions,
                    self.state.rest_lengths,
                    self.parameters.diameter,
                ),
                "energy_before": None,
                "energy_trial": float(self.energy()),
                "energy_after": float(self.energy()),
                "energy_components": self.energy_components(),
                "detail": "initial geometry is finite and non-intersecting",
            }
        )
        initial_node_contact_pairs = _node_contact_pairs(
            self.state.positions,
            self.parameters.diameter,
        )
        if initial_geometry["contact_pairs"] or initial_node_contact_pairs:
            self._append_event(
                {
                    "event_type": "geometry_diagnostic",
                    "reason": "contact",
                    "accepted": True,
                    "requested_dt": None,
                    "trial_dt": None,
                    "accepted_dt": None,
                    "time_before": float(self.state.time),
                    "time_after": float(self.state.time),
                    "step_before": int(self.state.step),
                    "step_after": int(self.state.step),
                    "state_before": _state_summary(
                        self.state.positions,
                        self.state.rest_lengths,
                        self.parameters.diameter,
                    ),
                    "state_trial": None,
                    "state_after": _state_summary(
                        self.state.positions,
                        self.state.rest_lengths,
                        self.parameters.diameter,
                    ),
                    "energy_before": float(self.energy()),
                    "energy_trial": None,
                    "energy_after": float(self.energy()),
                    "contact_pairs": initial_geometry["contact_pairs"],
                    "node_contact_pairs": initial_node_contact_pairs,
                    "detail": "contact diagnostic only; node contact force and segment geometry are recorded separately",
                }
            )

    def _append_event(self, event: dict[str, object]) -> None:
        event = dict(event)
        event.setdefault("schema_version", EVENT_SCHEMA_VERSION)
        event["event_index"] = len(self.events)
        self.events.append(event)

    @property
    def event_log(self) -> list[dict[str, object]]:
        """Return a copy of the structured attempt and geometry event log."""

        return [dict(event) for event in self.events]

    @classmethod
    def from_straight(
        cls,
        n_nodes: int,
        spacing: float = 1.0,
        parameters: Optional[ModelParameters] = None,
    ) -> "OverdampedGrowingFilament":
        params = parameters or ModelParameters(reference_length=spacing)
        if parameters is not None and parameters.reference_length <= 0.0:
            raise ModelError("reference_length must be positive")
        return cls(straight_state(n_nodes, spacing), params)

    def energy_components(self, positions: Optional[Array] = None,
                          rest_lengths: Optional[Array] = None) -> Dict[str, float]:
        p = self.state.positions if positions is None else np.asarray(positions, dtype=float)
        a = self.state.rest_lengths if rest_lengths is None else np.asarray(rest_lengths, dtype=float)
        if p.shape != (len(a) + 1, 2):
            raise ModelError("incompatible positions and rest_lengths")

        lengths = np.linalg.norm(np.diff(p, axis=0), axis=1)
        if np.any(lengths <= 1.0e-12):
            raise ModelError("zero-length geometric segment")
        axial = 0.5 * self.parameters.axial_stiffness * np.sum(
            (lengths - a) ** 2 / a
        )

        bending, _ = _bending_energy_and_forces(
            p,
            a,
            self.parameters.bending_stiffness,
        )

        contact = 0.0
        if self.parameters.contact_stiffness > 0.0 and self.parameters.diameter > 0.0:
            for i in range(len(p)):
                for j in range(i + 2, len(p)):
                    distance = float(np.linalg.norm(p[i] - p[j]))
                    overlap = self.parameters.diameter - distance
                    if overlap > 0.0:
                        contact += 0.5 * self.parameters.contact_stiffness * overlap ** 2
        return {"stretch": float(axial), "bend": float(bending), "contact": float(contact)}

    def energy(self, positions: Optional[Array] = None,
               rest_lengths: Optional[Array] = None) -> float:
        return float(sum(self.energy_components(positions, rest_lengths).values()))

    def forces(self, positions: Optional[Array] = None,
               rest_lengths: Optional[Array] = None) -> Array:
        """Return ``-dE/dr`` for the current geometry."""

        p = self.state.positions if positions is None else np.asarray(positions, dtype=float)
        a = self.state.rest_lengths if rest_lengths is None else np.asarray(rest_lengths, dtype=float)
        if p.shape != (len(a) + 1, 2):
            raise ModelError("incompatible positions and rest_lengths")
        forces = np.zeros_like(p)

        edges = np.diff(p, axis=0)
        lengths = np.linalg.norm(edges, axis=1)
        if np.any(lengths <= 1.0e-12):
            raise ModelError("zero-length geometric segment")
        unit_edges = edges / lengths[:, None]
        edge_force = self.parameters.axial_stiffness * (lengths - a) / a
        for i, force_vector in enumerate(edge_force[:, None] * unit_edges):
            forces[i] += force_vector
            forces[i + 1] -= force_vector

        _, bending_forces = _bending_energy_and_forces(
            p,
            a,
            self.parameters.bending_stiffness,
        )
        forces += bending_forces

        if self.parameters.contact_stiffness > 0.0 and self.parameters.diameter > 0.0:
            for i in range(len(p)):
                for j in range(i + 2, len(p)):
                    delta = p[i] - p[j]
                    distance = float(np.linalg.norm(delta))
                    overlap = self.parameters.diameter - distance
                    if overlap > 0.0 and distance > 1.0e-12:
                        force = self.parameters.contact_stiffness * overlap * delta / distance
                        forces[i] += force
                        forces[j] -= force
        return forces

    def _apply_boundary_conditions(self, positions: Array, velocities: Array) -> None:
        if self.parameters.fixed_left:
            positions[0] = self.left_anchor
            velocities[0] = 0.0
        if self.parameters.fixed_right:
            positions[-1] = self.right_anchor
            velocities[-1] = 0.0

    def _trial_rejection_reason(
        self,
        positions: Array,
        dt: float,
        velocities: Array,
        rest_lengths: Array,
        start_positions: Optional[Array] = None,
    ) -> Optional[str]:
        self._last_swept_intersection = None
        if not np.isfinite(rest_lengths).all():
            return "trial reference lengths are non-finite"
        if not np.isfinite(positions).all() or not np.isfinite(velocities).all():
            return "trial positions or velocities are non-finite"
        local_scale = max(float(np.min(rest_lengths)), 1.0e-12)
        max_displacement = float(np.max(np.linalg.norm(dt * velocities, axis=1)))
        if max_displacement > self.parameters.max_displacement_fraction * local_scale:
            return "trial displacement exceeds max_displacement_fraction"
        if self.parameters.reject_crossing:
            if start_positions is not None and np.shape(start_positions) == np.shape(positions):
                self._last_swept_intersection = find_swept_nonlocal_intersection(
                    start_positions,
                    positions,
                )
                if self._last_swept_intersection is not None:
                    return (
                        "trial swept non-local segment crossing at normalized time "
                        f"{self._last_swept_intersection.normalized_time:.17g}"
                    )
            if has_nonlocal_intersection(positions):
                return "trial contains a non-local segment intersection"
        return None

    @staticmethod
    def _event_reason(rejection_reason: Optional[str]) -> str:
        if rejection_reason is None:
            return "accepted"
        if "non-finite" in rejection_reason:
            return "nonfinite"
        if "displacement exceeds" in rejection_reason:
            return "displacement_exceeded"
        if "crossing" in rejection_reason or "intersection" in rejection_reason:
            return "crossing_rejection"
        if "energy increased" in rejection_reason:
            return "energy_increased"
        return "invalid_trial"

    def _trial_is_valid(self, positions: Array, dt: float,
                        velocities: Array, rest_lengths: Array) -> bool:
        """Keep the pre-Gate-2 boolean helper for callers of the prototype API."""

        return self._trial_rejection_reason(
            positions, dt, velocities, rest_lengths
        ) is None

    def step(self, dt: Optional[float] = None) -> FilamentState:
        """Advance one accepted step and record every trial as a structured event.

        For a growth-free trial, explicit Euler is accepted only when its
        total energy does not increase beyond ``energy_tolerance``.  Growth
        changes the reference lengths and can inject energy, so this check is
        intentionally not applied when ``growth_rate`` is non-zero.
        """

        requested_dt = self.parameters.dt if dt is None else float(dt)
        if not np.isfinite(requested_dt) or requested_dt <= 0.0:
            raise ModelError("dt must be finite and positive")
        trial_dt = requested_dt
        previous_state = self.state.copy()
        previous_energy = self.energy()
        previous_components = self.energy_components()
        if not np.isfinite(previous_energy):
            raise ModelError("current energy is non-finite")
        last_rejection_reason = "no trial was evaluated"

        for _ in range(self.parameters.max_retries + 1):
            grown = grow_reference_lengths(
                self.state.rest_lengths,
                self.parameters.growth_rate,
                trial_dt,
            )
            positions, rest_lengths = remesh(
                self.state.positions,
                grown,
                self.parameters.a_max,
            )
            forces = self.forces(positions, rest_lengths)
            drag = self.parameters.drag_density * _node_weights(rest_lengths)
            velocities = forces / drag[:, None]
            trial_displacement = trial_dt * velocities
            trial_positions = positions + trial_displacement
            self._apply_boundary_conditions(trial_positions, velocities)
            actual_displacement = trial_positions - positions
            max_displacement = (
                float(np.max(np.linalg.norm(actual_displacement, axis=1)))
                if np.isfinite(actual_displacement).all()
                else None
            )
            trial_summary = _state_summary(
                trial_positions,
                rest_lengths,
                self.parameters.diameter,
                max_displacement=max_displacement,
            )

            rejection_reason = self._trial_rejection_reason(
                trial_positions,
                trial_dt,
                velocities,
                rest_lengths,
                start_positions=positions,
            )
            trial_energy: Optional[float] = None
            trial_components: Optional[Dict[str, float]] = None
            if rejection_reason is None:
                try:
                    trial_components = self.energy_components(trial_positions, rest_lengths)
                    trial_energy = float(sum(trial_components.values()))
                except (ModelError, ValueError, FloatingPointError) as exc:
                    rejection_reason = f"invalid trial energy: {exc}"
                else:
                    if not np.isfinite(trial_energy):
                        rejection_reason = "trial energy is non-finite"
                    elif self.parameters.growth_rate == 0.0:
                        tolerance = self.parameters.energy_tolerance * max(
                            1.0, abs(previous_energy)
                        )
                        if trial_energy > previous_energy + tolerance:
                            rejection_reason = (
                                "growth-free trial energy increased: "
                                f"{previous_energy:.17g} -> {trial_energy:.17g}"
                            )

            event_base: dict[str, object] = {
                "event_type": "step_attempt",
                "reason": self._event_reason(rejection_reason),
                "accepted": rejection_reason is None,
                "requested_dt": float(requested_dt),
                "trial_dt": float(trial_dt),
                "accepted_dt": float(trial_dt) if rejection_reason is None else None,
                "time_before": float(previous_state.time),
                "time_after": (
                    float(previous_state.time + trial_dt)
                    if rejection_reason is None
                    else float(previous_state.time)
                ),
                "step_before": int(previous_state.step),
                "step_after": (
                    int(previous_state.step + 1)
                    if rejection_reason is None
                    else int(previous_state.step)
                ),
                "state_before": _state_summary(
                    previous_state.positions,
                    previous_state.rest_lengths,
                    self.parameters.diameter,
                ),
                "state_trial": trial_summary,
                "state_after": None,
                "energy_before": float(previous_energy),
                "energy_trial": trial_energy,
                "energy_after": None,
                "energy_components_before": previous_components,
                "energy_components_trial": trial_components,
                "max_displacement": max_displacement,
                "swept_crossing": (
                    self._last_swept_intersection.as_dict()
                    if self._last_swept_intersection is not None
                    else None
                ),
                "detail": rejection_reason,
            }

            if rejection_reason is None:
                self.state = FilamentState(
                    trial_positions,
                    rest_lengths,
                    time=self.state.time + trial_dt,
                    step=self.state.step + 1,
                )
                event_base["state_after"] = _state_summary(
                    self.state.positions,
                    self.state.rest_lengths,
                    self.parameters.diameter,
                    max_displacement=max_displacement,
                )
                event_base["energy_after"] = trial_energy
                self._append_event(event_base)
                contact_pairs = trial_summary.get("contact_pairs", [])
                node_contact_pairs = trial_summary.get("node_contact_pairs", [])
                if contact_pairs or node_contact_pairs:
                    self._append_event(
                        {
                            "event_type": "geometry_diagnostic",
                            "reason": "contact",
                            "accepted": True,
                            "requested_dt": float(requested_dt),
                            "trial_dt": float(trial_dt),
                            "accepted_dt": float(trial_dt),
                            "time_before": float(previous_state.time),
                            "time_after": float(self.state.time),
                            "step_before": int(previous_state.step),
                            "step_after": int(self.state.step),
                            "state_before": event_base["state_before"],
                            "state_trial": trial_summary,
                            "state_after": event_base["state_after"],
                            "energy_before": float(previous_energy),
                            "energy_trial": trial_energy,
                            "energy_after": trial_energy,
                            "contact_pairs": contact_pairs,
                            "node_contact_pairs": node_contact_pairs,
                            "detail": "contact diagnostic only; node contact force and segment geometry are recorded separately",
                        }
                    )
                self.accepted_steps += 1
                self.accepted_dts.append(float(trial_dt))
                return self.state.copy()

            event_base["state_after"] = event_base["state_before"]
            event_base["energy_after"] = float(previous_energy)
            self._append_event(event_base)
            contact_pairs = trial_summary.get("contact_pairs", [])
            node_contact_pairs = trial_summary.get("node_contact_pairs", [])
            if contact_pairs or node_contact_pairs:
                self._append_event(
                    {
                        "event_type": "geometry_diagnostic",
                        "reason": "contact",
                        "accepted": None,
                        "requested_dt": float(requested_dt),
                        "trial_dt": float(trial_dt),
                        "accepted_dt": None,
                        "time_before": float(previous_state.time),
                        "time_after": float(previous_state.time),
                        "step_before": int(previous_state.step),
                        "step_after": int(previous_state.step),
                        "state_before": event_base["state_before"],
                        "state_trial": trial_summary,
                        "state_after": event_base["state_after"],
                        "energy_before": float(previous_energy),
                        "energy_trial": trial_energy,
                        "energy_after": float(previous_energy),
                        "contact_pairs": contact_pairs,
                        "node_contact_pairs": node_contact_pairs,
                        "detail": "contact diagnostic only; node contact force and segment geometry are recorded separately",
                    }
                )
            last_rejection_reason = rejection_reason
            self.rejected_steps += 1
            self.rejected_dts.append(float(trial_dt))
            self.rejection_reasons.append(rejection_reason)
            trial_dt *= 0.5
            if trial_dt < self.parameters.dt_min:
                break

        raise RuntimeError(
            "failed to find an accepted step; "
            f"requested_dt={requested_dt}, dt_min={self.parameters.dt_min}, "
            f"last_rejection_reason={last_rejection_reason}"
        )

    def run_manifest(
        self,
        metadata: Optional[Dict[str, object]] = None,
        input_data: object = None,
    ) -> dict[str, object]:
        """Return a canonical manifest for the current deterministic run."""

        from .reproducibility import build_manifest

        return build_manifest(
            self.parameters,
            self.initial_state,
            final_state=self.state,
            events=self.events,
            metadata=metadata,
            input_data=input_data,
        )

    def run(self, t_end: Optional[float] = None) -> list[FilamentState]:
        """Run until ``t_end`` and return copies of all accepted states."""

        target = self.parameters.t_end if t_end is None else float(t_end)
        if target < self.state.time:
            raise ModelError("t_end must not be earlier than current time")
        trajectory = [self.state.copy()]
        while self.state.time < target - 1.0e-15:
            self.step(min(self.parameters.dt, target - self.state.time))
            trajectory.append(self.state.copy())
        return trajectory
