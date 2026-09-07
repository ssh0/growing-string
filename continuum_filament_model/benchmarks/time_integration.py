"""Small deterministic Gate 2 time-integration diagnostics.

The manufactured solution deliberately uses three collinear nodes, fixed
endpoints, and no growth.  The middle node then obeys a scalar linear
first-order ODE while the mesh remains fixed, so the explicit-Euler time
error can be measured without mixing in remeshing error.
"""

from __future__ import annotations

import json
from typing import Any, Iterable

import numpy as np

from growing_filament.model import (
    FilamentState,
    ModelParameters,
    OverdampedGrowingFilament,
)

DEFAULT_TIME_STEPS = (2.0e-2, 1.0e-2, 5.0e-3)
MANUFACTURED_T_END = 2.0e-1
MANUFACTURED_INITIAL_MIDDLE_X = 1.25
MANUFACTURED_SPACING = 1.0


def manufactured_three_node_state(
    middle_x: float = MANUFACTURED_INITIAL_MIDDLE_X,
) -> FilamentState:
    """Return the three-node fixed-mesh manufactured-solution state."""

    return FilamentState(
        [[0.0, 0.0], [middle_x, 0.0], [2.0 * MANUFACTURED_SPACING, 0.0]],
        [MANUFACTURED_SPACING, MANUFACTURED_SPACING],
    )


def manufactured_three_node_parameters(dt: float) -> ModelParameters:
    """Return parameters for the scalar axial-relaxation manufactured ODE."""

    return ModelParameters(
        axial_stiffness=2.0,
        bending_stiffness=1.0,
        drag_density=1.0,
        growth_rate=0.0,
        reference_length=MANUFACTURED_SPACING,
        dt=dt,
        t_end=MANUFACTURED_T_END,
        # The reference length is one and a_max is two: no remeshing can occur.
        a_max=2.0,
        fixed_left=True,
        fixed_right=True,
        reject_crossing=False,
    )


def manufactured_middle_x(time: float) -> float:
    """Exact middle-node x coordinate for the manufactured ODE.

    With fixed endpoints at 0 and 2, unit rest lengths, EA=2, and unit drag,
    the middle node obeys ``dx/dt = -4 (x - 1)``.
    """

    return MANUFACTURED_SPACING + (
        MANUFACTURED_INITIAL_MIDDLE_X - MANUFACTURED_SPACING
    ) * np.exp(-4.0 * time)


def run_manufactured_solution(
    dt: float,
    t_end: float = MANUFACTURED_T_END,
) -> dict[str, Any]:
    """Integrate one resolution and return its error and diagnostics."""

    model = OverdampedGrowingFilament(
        manufactured_three_node_state(),
        manufactured_three_node_parameters(dt),
    )
    trajectory = model.run(t_end=t_end)
    final_state = trajectory[-1]
    exact_middle_x = manufactured_middle_x(final_state.time)
    numerical_middle_x = float(final_state.positions[1, 0])
    return {
        "dt": float(dt),
        "t_end": float(final_state.time),
        "n_nodes": final_state.n_nodes,
        "numerical_middle_x": numerical_middle_x,
        "exact_middle_x": float(exact_middle_x),
        "error": abs(numerical_middle_x - exact_middle_x),
        "accepted_dts": list(model.accepted_dts),
        "rejected_dts": list(model.rejected_dts),
        "accepted_steps": model.accepted_steps,
        "rejected_steps": model.rejected_steps,
    }


def run_time_convergence_benchmark(
    dt_values: Iterable[float] = DEFAULT_TIME_STEPS,
    t_end: float = MANUFACTURED_T_END,
) -> list[dict[str, Any]]:
    """Return manufactured-solution measurements at several time steps."""

    values = tuple(float(dt) for dt in dt_values)
    if len(values) < 3 or any(dt <= 0.0 for dt in values):
        raise ValueError("dt_values must contain at least three positive values")
    records = [run_manufactured_solution(dt, t_end=t_end) for dt in values]
    for coarse, fine in zip(records, records[1:]):
        coarse["observed_order"] = float(
            np.log(coarse["error"] / fine["error"])
            / np.log(coarse["dt"] / fine["dt"])
        )
    return records


def run_fixed_mesh_diagnostic(
    dt: float,
    t_end: float = 5.0e-2,
) -> dict[str, Any]:
    """Record energies and accepted ``dt`` values for a fixed mesh.

    This is intentionally a compact diagnostic return value, not a general
    event schema.  The fixture is the bent four-node growth-free case that
    exposed the original ``dt=0.01`` explicit-Euler energy increase.
    """

    model = OverdampedGrowingFilament(
        FilamentState(
            [[0.0, 0.0], [1.0, 0.4], [2.0, 0.4], [3.0, 0.0]],
            [1.0, 1.0, 1.0],
        ),
        ModelParameters(
            axial_stiffness=100.0,
            bending_stiffness=1.0,
            drag_density=1.0,
            growth_rate=0.0,
            reference_length=1.0,
            dt=dt,
            t_end=t_end,
            a_max=2.0,
            fixed_left=True,
            fixed_right=True,
        ),
    )
    times = [float(model.state.time)]
    energies = [float(model.energy())]
    node_counts = [model.state.n_nodes]
    finite_positions = [bool(np.isfinite(model.state.positions).all())]
    accepted_dts: list[float] = []
    while model.state.time < t_end - 1.0e-15:
        state = model.step(min(dt, t_end - model.state.time))
        accepted_dts.append(float(model.accepted_dts[-1]))
        times.append(float(state.time))
        energies.append(float(model.energy()))
        node_counts.append(state.n_nodes)
        finite_positions.append(bool(np.isfinite(state.positions).all()))

    return {
        "requested_dt": float(dt),
        "times": times,
        "energies": energies,
        "accepted_dts": accepted_dts,
        "node_counts": node_counts,
        "finite_positions": finite_positions,
        "accepted_steps": model.accepted_steps,
        "rejected_steps": model.rejected_steps,
        "rejected_dts": list(model.rejected_dts),
        "rejection_reasons": list(model.rejection_reasons),
    }


def main() -> None:
    print(
        json.dumps(
            {
                "time_convergence": run_time_convergence_benchmark(),
                "fixed_mesh_dt_0_01": run_fixed_mesh_diagnostic(1.0e-2),
                "fixed_mesh_dt_0_005": run_fixed_mesh_diagnostic(5.0e-3),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
