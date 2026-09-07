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

Array = np.ndarray


class ModelError(ValueError):
    """Raised when a state or model parameter violates an invariant."""


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
    fixed_left: bool = False
    fixed_right: bool = False
    reject_crossing: bool = True

    def validate(self) -> None:
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
        if not 0.0 < self.max_displacement_fraction <= 1.0:
            raise ModelError("max_displacement_fraction must be in (0, 1]")


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

    if growth_rate < 0.0 or dt < 0.0:
        raise ModelError("growth_rate and dt must be non-negative")
    return np.asarray(rest_lengths, dtype=float) * np.exp(growth_rate * dt)


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


def _cross2(a: Array, b: Array) -> float:
    return float(a[0] * b[1] - a[1] * b[0])


def _on_segment(a: Array, b: Array, p: Array, eps: float = 1.0e-12) -> bool:
    return (
        min(a[0], b[0]) - eps <= p[0] <= max(a[0], b[0]) + eps
        and min(a[1], b[1]) - eps <= p[1] <= max(a[1], b[1]) + eps
    )


def segments_intersect(a: Array, b: Array, c: Array, d: Array,
                       eps: float = 1.0e-12) -> bool:
    """Return whether two closed 2D segments intersect."""

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


def has_nonlocal_intersection(positions: Array) -> bool:
    """Detect intersections between non-adjacent open-chain segments."""

    n_segments = len(positions) - 1
    for i in range(n_segments):
        for j in range(i + 2, n_segments):
            # Segments sharing a node are local neighbors and are excluded.
            if j == i + 1:
                continue
            if segments_intersect(
                positions[i], positions[i + 1],
                positions[j], positions[j + 1],
            ):
                return True
    return False


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
        self.parameters = parameters
        self.state = state.copy()
        self.left_anchor = self.state.positions[0].copy()
        self.right_anchor = self.state.positions[-1].copy()
        self.accepted_steps = 0
        self.rejected_steps = 0

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

    def _trial_is_valid(self, positions: Array, dt: float,
                        velocities: Array, rest_lengths: Array) -> bool:
        if not np.isfinite(positions).all() or not np.isfinite(velocities).all():
            return False
        local_scale = max(float(np.min(rest_lengths)), 1.0e-12)
        if float(np.max(np.linalg.norm(dt * velocities, axis=1))) > \
                self.parameters.max_displacement_fraction * local_scale:
            return False
        if self.parameters.reject_crossing and has_nonlocal_intersection(positions):
            return False
        return True

    def step(self, dt: Optional[float] = None) -> FilamentState:
        """Advance one accepted step, halving ``dt`` after invalid trials."""

        requested_dt = self.parameters.dt if dt is None else float(dt)
        if requested_dt <= 0.0:
            raise ModelError("dt must be positive")
        trial_dt = requested_dt

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
            trial_positions = positions + trial_dt * velocities
            self._apply_boundary_conditions(trial_positions, velocities)

            if self._trial_is_valid(trial_positions, trial_dt, velocities, rest_lengths):
                self.state = FilamentState(
                    trial_positions,
                    rest_lengths,
                    time=self.state.time + trial_dt,
                    step=self.state.step + 1,
                )
                self.accepted_steps += 1
                return self.state.copy()

            self.rejected_steps += 1
            trial_dt *= 0.5
            if trial_dt < self.parameters.dt_min:
                break

        raise RuntimeError(
            "failed to find an accepted step; "
            f"requested_dt={requested_dt}, dt_min={self.parameters.dt_min}"
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
