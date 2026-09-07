"""Small deterministic P1A remeshing and resolution benchmarks.

The smooth-shape part samples the same analytic curve independently at each
``a_max``.  This is intentional: repeatedly splitting a polygon preserves its
corners, whose continuum squared-curvature energy is singular.  The short
run separately exercises the production midpoint-remeshing path.
"""

from __future__ import annotations

import json
from typing import Any, Iterable

import numpy as np

from growing_filament.model import (
    FilamentState,
    ModelParameters,
    OverdampedGrowingFilament,
    straight_state,
)
from growing_filament.observables import (
    arc_length_weighted_radius_of_gyration,
    contour_length,
)

DEFAULT_A_MAX = (2.0, 1.0, 0.5, 0.25)


def _parabola(x: np.ndarray) -> np.ndarray:
    return np.column_stack((x, 0.03 * x * x + 0.1 * x))


def smooth_parabola_state(a_max: float) -> FilamentState:
    """Sample one fixed smooth parabola with a target arc spacing."""

    if a_max <= 0.0:
        raise ValueError("a_max must be positive")
    dense_x = np.linspace(-5.0, 5.0, 20_001)
    dense_positions = _parabola(dense_x)
    dense_lengths = np.linalg.norm(np.diff(dense_positions, axis=0), axis=1)
    dense_arc = np.concatenate(([0.0], np.cumsum(dense_lengths)))
    n_segments = int(np.ceil(float(dense_arc[-1]) / a_max))
    target_arc = np.linspace(0.0, float(dense_arc[-1]), n_segments + 1)
    x = np.interp(target_arc, dense_arc, dense_x)
    positions = _parabola(x)
    rest_lengths = np.linalg.norm(np.diff(positions, axis=0), axis=1)
    return FilamentState(positions, rest_lengths)


def _resolution_model(state: FilamentState, a_max: float) -> OverdampedGrowingFilament:
    return OverdampedGrowingFilament(
        state,
        ModelParameters(
            axial_stiffness=100.0,
            bending_stiffness=1.0,
            drag_density=1.0,
            reference_length=1.0,
            a_max=a_max,
            reject_crossing=False,
        ),
    )


def _interior_force_residual(model: OverdampedGrowingFilament) -> float:
    """Measure the interior force residual away from endpoint cells.

    The first and last two nodes are omitted because the open-chain bending
    stencil represents boundary generalized forces separately.  The remaining
    nodal force norm should decrease for the smooth parabola as resolution is
    refined.
    """

    forces = model.forces()
    interior = forces[2:-2]
    if len(interior) == 0:
        return 0.0
    return float(np.linalg.norm(interior))


def run_resolution_benchmark(
    a_max_values: Iterable[float] = DEFAULT_A_MAX,
) -> list[dict[str, Any]]:
    """Return measurements for a reproducible smooth-geometry refinement."""

    values = tuple(float(value) for value in a_max_values)
    if not values or any(value <= 0.0 for value in values):
        raise ValueError("a_max_values must contain positive values")
    reference_state = smooth_parabola_state(min(values) / 8.0)
    reference_radius = arc_length_weighted_radius_of_gyration(reference_state)
    records: list[dict[str, Any]] = []
    for a_max in values:
        state = smooth_parabola_state(a_max)
        model = _resolution_model(state, a_max)
        records.append(
            {
                "a_max": a_max,
                "n_nodes": state.n_nodes,
                "reference_length": float(np.sum(state.rest_lengths)),
                "max_rest_length": float(np.max(state.rest_lengths)),
                "contour_length": contour_length(state),
                "bend_energy": model.energy_components()["bend"],
                "arc_length_weighted_radius_of_gyration": arc_length_weighted_radius_of_gyration(state),
                "arc_radius_error": abs(
                    arc_length_weighted_radius_of_gyration(state) - reference_radius
                ),
                "interior_force_residual": _interior_force_residual(model),
            }
        )
    return records


def run_short_growth_benchmark() -> dict[str, Any]:
    """Exercise deterministic growth-triggered midpoint remeshing."""

    model = OverdampedGrowingFilament(
        straight_state(4, spacing=1.0),
        ModelParameters(
            axial_stiffness=5.0,
            bending_stiffness=1.0,
            drag_density=1.0,
            growth_rate=10.0,
            reference_length=1.0,
            dt=0.01,
            t_end=0.04,
            a_max=1.05,
            fixed_left=True,
            fixed_right=True,
            reject_crossing=False,
        ),
    )
    trajectory = model.run()
    node_counts = [state.n_nodes for state in trajectory]
    return {
        "node_counts": node_counts,
        "remesh_step_flags": [
            int(after > before)
            for before, after in zip(node_counts, node_counts[1:])
        ],
        "reference_lengths": [
            float(np.sum(state.rest_lengths)) for state in trajectory
        ],
        "contour_lengths": [contour_length(state) for state in trajectory],
        "accepted_steps": model.accepted_steps,
        "rejected_steps": model.rejected_steps,
    }


def main() -> None:
    print(
        json.dumps(
            {
                "resolution": run_resolution_benchmark(),
                "short_growth": run_short_growth_benchmark(),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
