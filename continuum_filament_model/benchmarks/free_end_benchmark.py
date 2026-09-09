"""Bounded free/free endpoint dynamics benchmark.

This runner exercises the natural endpoint implementation without enabling
contact.  It reports endpoint trajectories, discrete boundary residuals, and
accepted-Euler work diagnostics for free/free and fixed/fixed runs.  The
fixed/fixed rows are a control, not an experimental-reproduction claim.

Only compact JSON/CSV summaries are written.  Full node trajectories are not
written by this benchmark.

Example::

    PYTHONPATH=continuum_filament_model/src \\
      python continuum_filament_model/benchmarks/free_end_benchmark.py \\
      --output /tmp/growing-string-free-end
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

if __package__ in {None, ""}:  # pragma: no cover - direct-file entry point
    _HERE = Path(__file__).resolve()
    _SRC = _HERE.parents[1] / "src"
    if str(_SRC) not in sys.path:
        sys.path.insert(0, str(_SRC))

from growing_filament.model import (  # noqa: E402
    FilamentState,
    ModelParameters,
    OverdampedGrowingFilament,
    _node_weights,
    grow_reference_lengths,
)


SCHEMA_VERSION = "continuum-filament-free-end-dynamics-1"
DEFAULT_CONFIG: dict[str, Any] = {
    "length": 4.0,
    "n_nodes": [5, 9],
    "dt_values": [2.0e-4, 1.0e-4],
    "t_end": 2.0e-2,
    "axial_stiffness": 20.0,
    "bending_stiffness": 0.2,
    "drag_density": 1.0,
    "growth_rate": 0.2,
    "bend_amplitude": 0.15,
    "a_max_factor": 4.0,
    "max_report_rows": 1000,
}


class BenchmarkError(ValueError):
    """Invalid free-end benchmark configuration or result."""


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return [_jsonable(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, float):
        if not np.isfinite(value):
            raise BenchmarkError(
                "benchmark result contains a non-finite float"
            )
        return float(value)
    return value


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            _jsonable(value), ensure_ascii=False, indent=2, sort_keys=True
        )
        + "\n",
        encoding="utf-8",
    )


def validate_config(config: Mapping[str, Any]) -> None:
    for key in (
        "length",
        "t_end",
        "axial_stiffness",
        "bending_stiffness",
        "drag_density",
        "bend_amplitude",
        "a_max_factor",
    ):
        try:
            value = float(config[key])
        except (KeyError, TypeError, ValueError) as exc:
            raise BenchmarkError(f"{key} must be a finite number") from exc
        if not np.isfinite(value) or value <= 0.0:
            raise BenchmarkError(f"{key} must be positive and finite")
    if float(config["growth_rate"]) < 0.0:
        raise BenchmarkError("growth_rate must be non-negative")
    nodes = tuple(int(value) for value in config.get("n_nodes", ()))
    if len(nodes) < 2 or any(value < 3 for value in nodes):
        raise BenchmarkError("n_nodes must contain at least two values >= 3")
    dts = tuple(float(value) for value in config.get("dt_values", ()))
    if len(dts) < 2 or any(
        value <= 0.0 or not np.isfinite(value) for value in dts
    ):
        raise BenchmarkError(
            "dt_values must contain at least two positive finite values"
        )
    max_rows = int(config.get("max_report_rows", 0))
    if max_rows < 2:
        raise BenchmarkError("max_report_rows must be at least 2")


def _initial_state(
    case: str, n_nodes: int, length: float, amplitude: float
) -> FilamentState:
    x = np.linspace(0.0, length, n_nodes)
    if case == "growth":
        y = np.zeros_like(x)
    elif case == "bending_relaxation":
        y = amplitude * np.sin(np.pi * x / length)
    else:
        raise BenchmarkError(f"unknown case: {case}")
    positions = np.column_stack((x, y))
    rest_lengths = np.linalg.norm(np.diff(positions, axis=0), axis=1)
    return FilamentState(positions, rest_lengths)


def _endpoint_row(
    model: OverdampedGrowingFilament, state: FilamentState
) -> dict[str, Any]:
    diagnostics = model.endpoint_diagnostics(
        state.positions, state.rest_lengths
    )
    return {
        "time": float(state.time),
        "step": int(state.step),
        "n_nodes": int(state.n_nodes),
        "left": list(state.positions[0]),
        "right": list(state.positions[-1]),
        "endpoint_distance": float(
            np.linalg.norm(state.positions[-1] - state.positions[0])
        ),
        "endpoint_force_residual_norm_max": float(
            diagnostics["endpoint_force_residual_norm_max"]
        ),
        "moment_residual_norm_max": float(
            diagnostics["moment_residual_norm_max"]
        ),
        "energy": float(model.energy(state.positions, state.rest_lengths)),
    }


def _work_diagnostics(
    model: OverdampedGrowingFilament,
    trajectory: Iterable[FilamentState],
) -> tuple[list[dict[str, Any]], dict[str, float]]:
    """Return compact endpoint rows plus growth-work/dissipation accounting."""

    states = list(trajectory)
    rows = [_endpoint_row(model, state) for state in states]
    growth_work = 0.0
    dissipation_work = 0.0
    balance_residual = 0.0
    for before, after in zip(states, states[1:]):
        dt = float(after.time - before.time)
        if dt <= 0.0:
            raise BenchmarkError("trajectory times must increase")
        energy_before = model.energy(before.positions, before.rest_lengths)
        grown = grow_reference_lengths(
            before.rest_lengths,
            model.parameters.growth_rate,
            dt,
        )
        growth_increment = (
            model.energy(before.positions, grown) - energy_before
        )
        growth_work += float(growth_increment)
        if before.positions.shape == after.positions.shape:
            velocity = (after.positions - before.positions) / dt
            gamma = model.parameters.drag_density * _node_weights(
                before.rest_lengths
            )
            dissipation_increment = dt * float(
                np.sum(gamma[:, None] * velocity * velocity)
            )
            dissipation_work += dissipation_increment
        else:
            # This benchmark chooses a_max so remeshing should not occur.  Do
            # not silently interpret a variable-node transition as work.
            raise BenchmarkError(
                "unexpected remeshing in bounded free-end benchmark"
            )
        energy_after = model.energy(after.positions, after.rest_lengths)
        balance_residual += float(
            energy_after
            - energy_before
            - growth_increment
            + dissipation_increment
        )
    initial_energy = model.energy(states[0].positions, states[0].rest_lengths)
    final_energy = model.energy(states[-1].positions, states[-1].rest_lengths)
    return rows, {
        "initial_energy": float(initial_energy),
        "final_energy": float(final_energy),
        "growth_work": float(growth_work),
        "dissipation_work": float(dissipation_work),
        "mechanical_balance_residual": float(balance_residual),
    }


def run_case(
    config: Mapping[str, Any],
    case: str,
    boundary: str,
    n_nodes: int,
    dt: float,
) -> dict[str, Any]:
    length = float(config["length"])
    state = _initial_state(
        case, n_nodes, length, float(config["bend_amplitude"])
    )
    spacing = length / (n_nodes - 1)
    params = ModelParameters(
        axial_stiffness=float(config["axial_stiffness"]),
        bending_stiffness=float(config["bending_stiffness"]),
        drag_density=float(config["drag_density"]),
        contact_stiffness=0.0,
        diameter=0.0,
        growth_rate=float(config["growth_rate"]) if case == "growth" else 0.0,
        reference_length=spacing,
        dt=float(dt),
        t_end=float(config["t_end"]),
        a_max=float(config["a_max_factor"]) * spacing,
        fixed_left=boundary == "fixed_fixed",
        fixed_right=boundary == "fixed_fixed",
        max_displacement_fraction=0.5,
        max_retries=20,
        reject_crossing=True,
    )
    model = OverdampedGrowingFilament(state, params)
    trajectory = model.run()
    rows, work = _work_diagnostics(model, trajectory)
    if len(rows) > int(config["max_report_rows"]):
        raise BenchmarkError(
            f"endpoint report exceeded max_report_rows="
            f"{config['max_report_rows']}"
        )
    initial = rows[0]
    final = rows[-1]
    initial_span = float(initial["endpoint_distance"])
    final_span = float(final["endpoint_distance"])
    final_diagnostics = model.endpoint_diagnostics(
        trajectory[-1].positions,
        trajectory[-1].rest_lengths,
    )
    return {
        "case": case,
        "boundary": boundary,
        "n_nodes": int(n_nodes),
        "dt": float(dt),
        "t_end": float(trajectory[-1].time),
        "contact_enabled": bool(final_diagnostics["contact_enabled"]),
        "accepted_steps": int(model.accepted_steps),
        "rejected_steps": int(model.rejected_steps),
        "initial_endpoint_distance": initial_span,
        "final_endpoint_distance": final_span,
        "endpoint_distance_change": final_span - initial_span,
        "endpoint_displacement_left": float(
            np.linalg.norm(
                trajectory[-1].positions[0] - trajectory[0].positions[0]
            )
        ),
        "endpoint_displacement_right": float(
            np.linalg.norm(
                trajectory[-1].positions[-1] - trajectory[0].positions[-1]
            )
        ),
        "final_endpoint_force_residual_norm_max": float(
            final_diagnostics["endpoint_force_residual_norm_max"]
        ),
        "final_moment_residual_norm_max": float(
            final_diagnostics["moment_residual_norm_max"]
        ),
        "max_endpoint_force_residual_norm": float(
            max(row["endpoint_force_residual_norm_max"] for row in rows)
        ),
        "max_moment_residual_norm": float(
            max(row["moment_residual_norm_max"] for row in rows)
        ),
        **work,
        "endpoint_trajectory": rows,
    }


def _relative_change(left: float, right: float, scale: float = 1.0) -> float:
    return float(
        abs(left - right) / max(abs(left), abs(right), scale * 1.0e-12)
    )


def _refinement_reports(
    records: list[dict[str, Any]], config: Mapping[str, Any]
) -> dict[str, list[dict[str, Any]]]:
    reports: dict[str, list[dict[str, Any]]] = {"time": [], "space": []}
    for case in ("growth", "bending_relaxation"):
        for boundary in ("free_free", "fixed_fixed"):
            for n_nodes in tuple(int(value) for value in config["n_nodes"]):
                matching = sorted(
                    (
                        row
                        for row in records
                        if row["case"] == case
                        and row["boundary"] == boundary
                        and row["n_nodes"] == n_nodes
                    ),
                    key=lambda row: row["dt"],
                )
                for coarse, fine in zip(matching, matching[1:]):
                    reports["time"].append(
                        {
                            "case": case,
                            "boundary": boundary,
                            "n_nodes": n_nodes,
                            "coarse_dt": coarse["dt"],
                            "fine_dt": fine["dt"],
                            "relative_final_energy_change": _relative_change(
                                coarse["final_energy"], fine["final_energy"]
                            ),
                            # The long schema key is intentionally stable.
                            "relative_endpoint_distance_change": _relative_change(  # noqa: E501
                                coarse["final_endpoint_distance"],
                                fine["final_endpoint_distance"],
                                scale=float(config["length"]),
                            ),
                        }
                    )
            for dt in tuple(float(value) for value in config["dt_values"]):
                matching = sorted(
                    (
                        row
                        for row in records
                        if row["case"] == case
                        and row["boundary"] == boundary
                        and row["dt"] == dt
                    ),
                    key=lambda row: row["n_nodes"],
                )
                for coarse, fine in zip(matching, matching[1:]):
                    reports["space"].append(
                        {
                            "case": case,
                            "boundary": boundary,
                            "coarse_n_nodes": coarse["n_nodes"],
                            "fine_n_nodes": fine["n_nodes"],
                            "dt": dt,
                            "relative_final_energy_change": _relative_change(
                                coarse["final_energy"], fine["final_energy"]
                            ),
                            # The long schema key is intentionally stable.
                            "relative_endpoint_distance_change": _relative_change(  # noqa: E501
                                coarse["final_endpoint_distance"],
                                fine["final_endpoint_distance"],
                                scale=float(config["length"]),
                            ),
                        }
                    )
    return reports


def run_benchmark(config: Mapping[str, Any] | None = None) -> dict[str, Any]:
    effective = dict(DEFAULT_CONFIG)
    effective.update(dict(config or {}))
    validate_config(effective)
    records = [
        run_case(effective, case, boundary, int(n_nodes), float(dt))
        for case in ("growth", "bending_relaxation")
        for boundary in ("free_free", "fixed_fixed")
        for n_nodes in tuple(int(value) for value in effective["n_nodes"])
        for dt in tuple(float(value) for value in effective["dt_values"])
    ]
    return {
        "schema_version": SCHEMA_VERSION,
        "scope": {
            "primary_boundary": "free_free",
            "control_boundary": "fixed_fixed",
            "contact_enabled": False,
            "excluded": [
                "friction",
                "adhesion",
                "anisotropic_drag",
                "localized_growth",
                "contact_solver",
            ],
            "trajectory_policy": (
                "endpoint coordinates and residuals only; "
                "no full node trajectory is written"
            ),
        },
        "acceptance_criteria": {
            "straight_free_free_equilibrium": (
                "endpoint force and bending-moment residuals are zero "
                "within numerical tolerance"
            ),
            "free_free_growth": (
                "uniform growth moves both free endpoints; "
                "fixed_fixed control keeps endpoint positions fixed"
            ),
            "free_free_relaxation": (
                "growth-free bending relaxation lowers energy without "
                "imposing endpoint anchors"
            ),
            "invariance": (
                "translation and rotation preserve endpoint residual "
                "norms and rotate residual vectors covariantly"
            ),
            "refinement": (
                "time and space comparisons are reported separately; no "
                "experimental claim is made from this benchmark"
            ),
        },
        "configuration": effective,
        "records": records,
        "refinement": _refinement_reports(records, effective),
    }


def _write_csv(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        "case",
        "boundary",
        "n_nodes",
        "dt",
        "t_end",
        "accepted_steps",
        "rejected_steps",
        "initial_endpoint_distance",
        "final_endpoint_distance",
        "endpoint_distance_change",
        "endpoint_displacement_left",
        "endpoint_displacement_right",
        "final_endpoint_force_residual_norm_max",
        "final_moment_residual_norm_max",
        "max_endpoint_force_residual_norm",
        "max_moment_residual_norm",
        "initial_energy",
        "final_energy",
        "growth_work",
        "dissipation_work",
        "mechanical_balance_residual",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for record in records:
            writer.writerow({column: record[column] for column in columns})


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, help="optional JSON config overriding defaults"
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="optional directory for compact summary.json/summary.csv",
    )
    args = parser.parse_args(argv)
    config: dict[str, Any] = {}
    if args.config is not None:
        config = json.loads(args.config.read_text(encoding="utf-8"))
    report = run_benchmark(config)
    if args.output is not None:
        args.output.mkdir(parents=True, exist_ok=True)
        _write_json(args.output / "summary.json", report)
        _write_csv(args.output / "summary.csv", report["records"])
    print(
        json.dumps(
            _jsonable(report), ensure_ascii=False, indent=2, sort_keys=True
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
