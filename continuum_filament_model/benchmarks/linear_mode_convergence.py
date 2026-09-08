"""P0-B linear-mode and growth convergence gates.

The benchmark has two deliberately separate parts:

* ``linear_mode`` linearizes the existing non-contact energy gradient about a
  straight, fixed-endpoint state.  The analytic reference is the *discrete*
  Hessian of the tangent-difference bending energy with endpoint positions
  fixed and endpoint tangents free.  A finite-difference Hessian of the public
  ``forces`` API, mode shapes, and a small-amplitude decay run are compared
  for several meshes and time steps.
* ``growth`` runs the three P1B.2 representative fixtures at fixed mesh for
  ``dt``, ``dt/2``, ``dt/4`` and at three spatial resolutions.  It records
  reference-length energy change, an Euler estimate of substrate dissipation,
  remesh energy jump, rejected trials, and event counts as separate quantities.

No contact force, friction, adhesion, experiment fit, critical value, or
universality claim is made here.  The output is intended for a temporary run
 directory.  Only the compact JSON/CSV summary should be copied into the
 repository.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

if __package__ in {None, ""}:  # pragma: no cover - direct-file entry point
    _HERE = Path(__file__).resolve()
    _SRC = _HERE.parents[1] / "src"
    _BENCHMARKS = _HERE.parent
    for _path in (_SRC, _BENCHMARKS):
        if str(_path) not in sys.path:
            sys.path.insert(0, str(_path))
    from buckling_benchmark import (  # type: ignore  # noqa: E402
        _classification,
        _metrics,
        _mode_coefficients,
        _transverse_coordinates,
        dimensionless_groups,
        initial_perturbed_state,
    )
else:  # pragma: no cover - exercised when imported as a package
    from .buckling_benchmark import (
        _classification,
        _metrics,
        _mode_coefficients,
        _transverse_coordinates,
        dimensionless_groups,
        initial_perturbed_state,
    )

from growing_filament.model import (  # noqa: E402
    FilamentState,
    ModelError,
    ModelParameters,
    OverdampedGrowingFilament,
    _node_weights,
    grow_reference_lengths,
    remesh,
    straight_state,
)
from growing_filament.reproducibility import (  # noqa: E402
    canonical_json_bytes,
    detect_git_revision,
)


SCHEMA_VERSION = "continuum-filament-p0b-linear-mode-convergence-2"
DEFAULT_CONFIG: dict[str, Any] = {
    "linear_mode": {
        "length": 2.0,
        "n_nodes": [5, 9, 17],
        "dt_values": [2.0e-4, 1.0e-4, 5.0e-5],
        "t_end": 0.1,
        "axial_stiffness": 100.0,
        "bending_stiffness": 0.1,
        "drag_density": 1.0,
        "amplitude_fraction": 1.0e-5,
        "hessian_fd_epsilon": 1.0e-7,
        "n_modes": 3,
        "a_max_factor": 2.0,
        "decay_rate_relative_tolerance": 0.01,
    },
    "growth": {
        "length": 2.0,
        "axial_stiffness": 100.0,
        "bending_stiffness": 0.1,
        "drag_density": 1.0,
        "n_nodes_temporal": 9,
        "n_nodes_spatial": [5, 7, 9],
        "dt": 1.0e-3,
        "dt_factors": [1.0, 0.5, 0.25],
        "t_end": 0.2,
        "amplitude": 0.02,
        "a_max_factor": 2.0,
        "max_retries": 12,
        "max_displacement_fraction": 0.25,
        "gate_scope": "morphology-only",
        "representatives": [
            {"name": "straight", "growth_rate": 0.05},
            {"name": "boundary-near", "growth_rate": 0.15},
            {"name": "buckled-candidate", "growth_rate": 0.2},
            {"name": "growth-free-control", "growth_rate": 0.0, "control_only": True},
        ],
        "tolerances": {
            "onset_time_relative": 0.10,
            "peak_transverse_relative": 0.15,
            "peak_transverse_absolute_fraction_of_length": 0.002,
            "a1_over_length_relative": 0.15,
        },
    },
}


class BenchmarkError(ValueError):
    """Invalid P0-B benchmark configuration."""


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
        return float(value) if math.isfinite(value) else None
    return value


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json_bytes(_jsonable(value)) + b"\n")


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(_jsonable(value))).hexdigest()


def _merge_config(config: Mapping[str, Any] | None) -> dict[str, Any]:
    result = json.loads(json.dumps(DEFAULT_CONFIG))
    if config is None:
        return result
    if not isinstance(config, Mapping):
        raise BenchmarkError("config root must be an object")
    for section in ("linear_mode", "growth"):
        if section in config:
            if not isinstance(config[section], Mapping):
                raise BenchmarkError(f"{section} must be an object")
            result[section].update(dict(config[section]))
    if "growth" in config and isinstance(config["growth"], Mapping):
        tolerances = config["growth"].get("tolerances")
        if tolerances is not None:
            if not isinstance(tolerances, Mapping):
                raise BenchmarkError("growth.tolerances must be an object")
            result["growth"]["tolerances"].update(dict(tolerances))
    return result


def load_config(path: Path | None) -> dict[str, Any]:
    raw = None if path is None else json.loads(path.read_text(encoding="utf-8"))
    return _merge_config(raw)


def raw_config_sha256(path: Path | None) -> str | None:
    """Return the hash of the exact JSON file supplied by the caller."""

    if path is None:
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _positive(value: Any, name: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise BenchmarkError(f"{name} must be finite and positive") from exc
    if not math.isfinite(number) or number <= 0.0:
        raise BenchmarkError(f"{name} must be finite and positive")
    return number


def _validate_config(config: Mapping[str, Any]) -> None:
    linear = config["linear_mode"]
    growth = config["growth"]
    for section, keys in {
        "linear_mode": ("length", "t_end", "axial_stiffness", "bending_stiffness", "drag_density", "amplitude_fraction", "hessian_fd_epsilon", "a_max_factor", "decay_rate_relative_tolerance"),
        "growth": ("length", "axial_stiffness", "bending_stiffness", "drag_density", "dt", "t_end", "amplitude", "a_max_factor"),
    }.items():
        values = config[section]
        for key in keys:
            _positive(values.get(key), f"{section}.{key}")
    for name, nodes in (("linear_mode.n_nodes", linear["n_nodes"]), ("growth.n_nodes_spatial", growth["n_nodes_spatial"]),):
        if len(nodes) != 3:
            raise BenchmarkError(f"{name} must contain exactly three resolutions")
        if any(int(node) != node or int(node) < 3 for node in nodes):
            raise BenchmarkError(f"{name} must contain integers >= 3")
    if int(growth["n_nodes_temporal"]) != growth["n_nodes_temporal"] or int(growth["n_nodes_temporal"]) < 3:
        raise BenchmarkError("growth.n_nodes_temporal must be an integer >= 3")
    if len(linear["dt_values"]) != 3 or any(_positive(dt, "linear_mode.dt_values") <= 0.0 for dt in linear["dt_values"]):
        raise BenchmarkError("linear_mode.dt_values must contain three positive values")
    if len(growth["dt_factors"]) != 3 or any(_positive(dt, "growth.dt_factors") <= 0.0 for dt in growth["dt_factors"]):
        raise BenchmarkError("growth.dt_factors must contain three positive values")
    if int(linear["n_modes"]) < 1:
        raise BenchmarkError("linear_mode.n_modes must be positive")
    if growth.get("gate_scope") != "morphology-only":
        raise BenchmarkError("growth.gate_scope must be 'morphology-only'")
    reps = growth.get("representatives")
    if not isinstance(reps, Sequence) or not reps:
        raise BenchmarkError("growth.representatives must be non-empty")
    for rep in reps:
        if not isinstance(rep, Mapping) or not rep.get("name"):
            raise BenchmarkError("each growth representative needs name")
        rate = float(rep.get("growth_rate"))
        if not math.isfinite(rate) or rate < 0.0:
            raise BenchmarkError("growth_rate must be finite and non-negative")
    for key in ("onset_time_relative", "peak_transverse_relative", "a1_over_length_relative", "peak_transverse_absolute_fraction_of_length"):
        _positive(growth["tolerances"].get(key), f"growth.tolerances.{key}")


def _linearization(length: float, n_nodes: int, bending_stiffness: float, drag_density: float) -> dict[str, Any]:
    """Return the analytic discrete transverse Hessian and drag spectrum.

    For ``h=L/(N-1)`` and endpoint values ``y_0=y_{N-1}=0``, the small-slope
    tangent-difference energy is ``EI/(2 h**3) ||D y||²``.  ``D`` contains
    the second differences at interior nodes.  This is the fixed endpoint,
    free tangent discretization used by the production energy, not a clamped
    slope boundary condition.
    """

    h = float(length) / (int(n_nodes) - 1)
    n_free = int(n_nodes) - 2
    second_difference = np.zeros((n_free, n_free), dtype=float)
    for row in range(n_free):
        second_difference[row, row] = -2.0
        if row > 0:
            second_difference[row, row - 1] = 1.0
        if row + 1 < n_free:
            second_difference[row, row + 1] = 1.0
    hessian = float(bending_stiffness) / h**3 * (second_difference.T @ second_difference)
    drag = float(drag_density) * h * np.eye(n_free)
    inv_sqrt_drag = np.eye(n_free) / math.sqrt(float(drag[0, 0]))
    eigenvalues, eigenvectors = np.linalg.eigh(inv_sqrt_drag @ hessian @ inv_sqrt_drag)
    # Columns are mass-normalized because inv_sqrt_drag is scalar here.
    modes = inv_sqrt_drag @ eigenvectors
    return {
        "h": h,
        "hessian": hessian,
        "drag": drag,
        "eigenvalues": eigenvalues,
        "modes": modes,
        "second_difference": second_difference,
    }


def _finite_difference_hessian(
    length: float,
    n_nodes: int,
    axial_stiffness: float,
    bending_stiffness: float,
    drag_density: float,
    epsilon: float,
) -> np.ndarray:
    h = length / (n_nodes - 1)
    state = straight_state(n_nodes, h)
    params = ModelParameters(
        axial_stiffness=axial_stiffness,
        bending_stiffness=bending_stiffness,
        drag_density=drag_density,
        contact_stiffness=0.0,
        diameter=0.0,
        growth_rate=0.0,
        reference_length=h,
        dt=1.0e-6,
        t_end=1.0e-3,
        a_max=2.0 * h,
        fixed_left=True,
        fixed_right=True,
        reject_crossing=True,
    )
    model = OverdampedGrowingFilament(state, params)
    free = np.arange(1, n_nodes - 1)
    hessian = np.zeros((len(free), len(free)), dtype=float)
    for column, node in enumerate(free):
        plus = state.positions.copy()
        minus = state.positions.copy()
        plus[node, 1] += epsilon
        minus[node, 1] -= epsilon
        force_plus = model.forces(plus, state.rest_lengths)[free, 1]
        force_minus = model.forces(minus, state.rest_lengths)[free, 1]
        hessian[:, column] = -(force_plus - force_minus) / (2.0 * epsilon)
    return 0.5 * (hessian + hessian.T)


def _mode_error(numerical: np.ndarray, analytic: np.ndarray, drag: np.ndarray) -> tuple[float, float]:
    numerical = numerical / math.sqrt(float(numerical @ drag @ numerical))
    analytic = analytic / math.sqrt(float(analytic @ drag @ analytic))
    overlap = abs(float(numerical @ drag @ analytic))
    error_plus = float(np.linalg.norm(numerical - analytic))
    error_minus = float(np.linalg.norm(numerical + analytic))
    return min(error_plus, error_minus), overlap


def _linear_decay_record(
    linear: Mapping[str, Any],
    n_nodes: int,
    dt: float,
    analytic_rate: float,
    analytic_mode: np.ndarray,
) -> dict[str, Any]:
    length = float(linear["length"])
    h = length / (n_nodes - 1)
    drag_scalar = float(linear["drag_density"]) * h
    mode = analytic_mode / math.sqrt(float(analytic_mode @ (drag_scalar * np.eye(len(analytic_mode))) @ analytic_mode))
    amplitude = float(linear["amplitude_fraction"]) * length
    positions = np.column_stack((np.linspace(0.0, length, n_nodes), np.zeros(n_nodes)))
    positions[1:-1, 1] = amplitude * mode / max(float(np.max(np.abs(mode))), 1.0e-15)
    state = FilamentState(positions, np.full(n_nodes - 1, h))
    params = ModelParameters(
        axial_stiffness=float(linear["axial_stiffness"]),
        bending_stiffness=float(linear["bending_stiffness"]),
        drag_density=float(linear["drag_density"]),
        contact_stiffness=0.0,
        diameter=0.0,
        growth_rate=0.0,
        reference_length=h,
        dt=float(dt),
        t_end=float(linear["t_end"]),
        a_max=float(linear["a_max_factor"]) * h,
        fixed_left=True,
        fixed_right=True,
        reject_crossing=True,
    )
    model = OverdampedGrowingFilament(state, params)
    times = [0.0]
    amplitudes = [abs(float(np.dot(mode, drag_scalar * state.positions[1:-1, 1])))]
    try:
        while model.state.time < float(linear["t_end"]) - 1.0e-15:
            model.step(min(float(dt), float(linear["t_end"]) - model.state.time))
            times.append(float(model.state.time))
            amplitudes.append(abs(float(np.dot(mode, drag_scalar * model.state.positions[1:-1, 1]))))
    except (ModelError, RuntimeError, ValueError, FloatingPointError) as exc:
        return {
            "n_nodes": n_nodes,
            "dt": float(dt),
            "decay_failure": f"{type(exc).__name__}: {exc}",
            "accepted_steps": model.accepted_steps,
            "rejected_steps": model.rejected_steps,
        }
    times_array = np.asarray(times, dtype=float)
    amplitudes_array = np.asarray(amplitudes, dtype=float)
    valid = np.isfinite(times_array) & np.isfinite(amplitudes_array) & (amplitudes_array > 0.0)
    slope = float(np.polyfit(times_array[valid], np.log(amplitudes_array[valid]), 1)[0])
    measured_rate = -slope
    relative_error = float(abs(measured_rate - analytic_rate) / analytic_rate)
    tolerance = float(linear["decay_rate_relative_tolerance"])
    return {
        "n_nodes": n_nodes,
        "dt": float(dt),
        "decay_failure": None,
        "accepted_steps": int(model.accepted_steps),
        "rejected_steps": int(model.rejected_steps),
        "decay_rate_analytic": float(analytic_rate),
        "decay_rate_measured": float(measured_rate),
        "decay_rate_relative_error": relative_error,
        "decay_rate_relative_tolerance": tolerance,
        "decay_rate_status": "pass" if relative_error <= tolerance else "fail",
        "tau_mode_measured": float(1.0 / measured_rate),
        "accepted_dt_min": float(min(model.accepted_dts)),
        "accepted_dt_max": float(max(model.accepted_dts)),
    }


def run_linear_modes(linear: Mapping[str, Any], output: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for raw_n in linear["n_nodes"]:
        n_nodes = int(raw_n)
        reference = _linearization(
            float(linear["length"]),
            n_nodes,
            float(linear["bending_stiffness"]),
            float(linear["drag_density"]),
        )
        numerical_hessian = _finite_difference_hessian(
            float(linear["length"]),
            n_nodes,
            float(linear["axial_stiffness"]),
            float(linear["bending_stiffness"]),
            float(linear["drag_density"]),
            float(linear["hessian_fd_epsilon"]),
        )
        h = float(reference["h"])
        drag = reference["drag"]
        inv_sqrt_drag = np.eye(n_nodes - 2) / math.sqrt(float(drag[0, 0]))
        numerical_eigenvalues, numerical_eigenvectors = np.linalg.eigh(
            inv_sqrt_drag @ numerical_hessian @ inv_sqrt_drag
        )
        n_modes = min(int(linear["n_modes"]), n_nodes - 2)
        continuum_tau_b = float(linear["drag_density"]) * float(linear["length"]) ** 4 / (
            float(linear["bending_stiffness"]) * np.pi**4
        )
        for mode_index in range(n_modes):
            mode_error, overlap = _mode_error(
                numerical_eigenvectors[:, mode_index],
                reference["modes"][:, mode_index],
                drag,
            )
            records.append(
                {
                    "record_type": "linear_eigenmode",
                    "n_nodes": n_nodes,
                    "mode": mode_index + 1,
                    "mesh_dx_over_L": float(h / float(linear["length"])),
                    "eigenvalue_analytic": float(reference["eigenvalues"][mode_index]),
                    "eigenvalue_numerical_hessian": float(numerical_eigenvalues[mode_index]),
                    "eigenvalue_relative_error": float(
                        abs(numerical_eigenvalues[mode_index] - reference["eigenvalues"][mode_index])
                        / max(abs(reference["eigenvalues"][mode_index]), 1.0e-15)
                    ),
                    "mode_shape_l2_error": mode_error,
                    "mode_shape_mass_overlap": overlap,
                    "tau_b_definition": continuum_tau_b,
                    "tau_b_definition_is_not_assumed_true": True,
                    "boundary_condition": "endpoint positions fixed; endpoint tangents free",
                    "contact_stiffness": 0.0,
                    "diameter": 0.0,
                }
            )
        for raw_dt in linear["dt_values"]:
            decay = _linear_decay_record(
                linear,
                n_nodes,
                float(raw_dt),
                float(reference["eigenvalues"][0]),
                reference["modes"][:, 0],
            )
            records.append(
                {
                    "record_type": "linear_decay",
                    "n_nodes": n_nodes,
                    "mode": 1,
                    "mesh_dx_over_L": float(h / float(linear["length"])),
                    "eigenvalue_analytic": float(reference["eigenvalues"][0]),
                    "eigenvalue_numerical_hessian": float(numerical_eigenvalues[0]),
                    "eigenvalue_relative_error": float(
                        abs(numerical_eigenvalues[0] - reference["eigenvalues"][0])
                        / max(abs(reference["eigenvalues"][0]), 1.0e-15)
                    ),
                    "tau_b_definition": continuum_tau_b,
                    "tau_b_definition_is_not_assumed_true": True,
                    "boundary_condition": "endpoint positions fixed; endpoint tangents free",
                    "contact_stiffness": 0.0,
                    "diameter": 0.0,
                    **decay,
                }
            )
    return records


def _growth_metrics(
    state: FilamentState,
    simulator: OverdampedGrowingFilament,
    anchors: np.ndarray,
    coefficients_count: int = 6,
) -> dict[str, Any]:
    row = _metrics(state, simulator, anchors)
    coefficients = _mode_coefficients(state, anchors, max_mode=coefficients_count)
    row["A1_over_L"] = float(abs(coefficients[0]) / np.linalg.norm(anchors[1] - anchors[0]))
    row["mode_spectrum_l2"] = float(np.linalg.norm(coefficients))
    for index, coefficient in enumerate(coefficients, start=1):
        row[f"mode_{index}_amplitude"] = float(coefficient)
        row[f"mode_{index}_abs_amplitude"] = float(abs(coefficient))
    row["mode_spectrum"] = [float(value) for value in coefficients]
    return row


def _union_fieldnames(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    """Preserve first-seen order while retaining every per-step column."""

    fields: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fields.append(key)
    return fields


def _run_growth_case(
    growth: Mapping[str, Any],
    representative: Mapping[str, Any],
    n_nodes: int,
    dt: float,
    output: Path,
    run_name: str,
) -> dict[str, Any]:
    length = float(growth["length"])
    spacing = length / (n_nodes - 1)
    config = {
        "length": length,
        "n_nodes": int(n_nodes),
        "axial_stiffness": float(growth["axial_stiffness"]),
        "bending_stiffness": float(growth["bending_stiffness"]),
        "drag_density": float(growth["drag_density"]),
        "growth_rate": float(representative["growth_rate"]),
        "amplitude": float(growth["amplitude"]),
        "dt": float(dt),
        "t_end": float(growth["t_end"]),
        "a_max_factor": float(growth["a_max_factor"]),
        "a_max": float(growth["a_max_factor"]) * spacing,
        "contact_stiffness": 0.0,
        "diameter": 0.0,
        "fixed_left": True,
        "fixed_right": True,
        "reject_crossing": True,
        "max_retries": int(growth["max_retries"]),
        "max_displacement_fraction": float(growth["max_displacement_fraction"]),
    }
    state = initial_perturbed_state(config, seed=None)
    anchors = np.asarray([state.positions[0], state.positions[-1]], dtype=float)
    params = ModelParameters(
        axial_stiffness=config["axial_stiffness"],
        bending_stiffness=config["bending_stiffness"],
        drag_density=config["drag_density"],
        contact_stiffness=0.0,
        diameter=0.0,
        growth_rate=config["growth_rate"],
        reference_length=spacing,
        dt=config["dt"],
        t_end=config["t_end"],
        a_max=config["a_max"],
        max_retries=config["max_retries"],
        max_displacement_fraction=config["max_displacement_fraction"],
        fixed_left=True,
        fixed_right=True,
        reject_crossing=True,
    )
    run_dir = output / "growth" / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    rejection_reasons: Counter[str] = Counter()
    failure_reason: str | None = None
    remesh_jumps = 0
    remesh_energy_jump_total = 0.0
    growth_energy_change_total = 0.0
    dissipation_total = 0.0
    try:
        simulator = OverdampedGrowingFilament(state, params)
        initial = _growth_metrics(simulator.state, simulator, anchors)
        initial.update(
            {
                "growth_energy_change_step": 0.0,
                "growth_energy_change_cumulative": 0.0,
                "dissipation_step_estimate": 0.0,
                "dissipation_cumulative_estimate": 0.0,
                "remesh_energy_jump_step": 0.0,
                "remesh_energy_jump_cumulative": 0.0,
                "total_energy_change_step": 0.0,
                "rejected_trials_cumulative": 0,
                "event_count_cumulative": len(simulator.events),
                "remesh_jumps_cumulative": 0,
            }
        )
        rows.append(initial)
        while simulator.state.time < config["t_end"] - 1.0e-15:
            before = simulator.state.copy()
            energy_before = simulator.energy()
            rejected_before = simulator.rejected_steps
            events_before = len(simulator.events)
            requested_dt = min(config["dt"], config["t_end"] - before.time)
            simulator.step(requested_dt)
            accepted_dt = float(simulator.accepted_dts[-1])
            after = simulator.state.copy()
            grown_rest = grow_reference_lengths(before.rest_lengths, config["growth_rate"], accepted_dt)
            remeshed_positions, remeshed_rest = remesh(before.positions, grown_rest, config["a_max"])
            energy_after_growth_same_geometry = simulator.energy(before.positions, grown_rest)
            energy_after_remesh = simulator.energy(remeshed_positions, remeshed_rest)
            remesh_occurred = len(remeshed_rest) != len(before.rest_lengths)
            remesh_jumps += int(remesh_occurred)
            remesh_jump = float(energy_after_remesh - energy_after_growth_same_geometry)
            remesh_energy_jump_total += remesh_jump
            growth_change = float(energy_after_growth_same_geometry - energy_before)
            growth_energy_change_total += growth_change
            forces = simulator.forces(remeshed_positions, remeshed_rest)
            drag = config["drag_density"] * _node_weights(remeshed_rest)
            velocities = forces / drag[:, None]
            velocities[0] = 0.0
            velocities[-1] = 0.0
            dissipation = float(accepted_dt * np.sum(drag[:, None] * velocities * velocities))
            dissipation_total += dissipation
            total_change = float(simulator.energy(after.positions, after.rest_lengths) - energy_before)
            mechanical_change = float(total_change - growth_change - remesh_jump)
            row = _growth_metrics(after, simulator, anchors)
            row.update(
                {
                    "growth_energy_change_step": growth_change,
                    "growth_energy_change_cumulative": growth_energy_change_total,
                    "dissipation_step_estimate": dissipation,
                    "dissipation_cumulative_estimate": dissipation_total,
                    "remesh_energy_jump_step": remesh_jump,
                    "remesh_energy_jump_cumulative": remesh_energy_jump_total,
                    "total_energy_change_step": total_change,
                    "mechanical_energy_change_step": mechanical_change,
                    "energy_balance_residual_step": float(total_change - growth_change - remesh_jump - mechanical_change),
                    "mechanical_change_plus_dissipation_step": float(mechanical_change + dissipation),
                    "accepted_dt_actual": accepted_dt,
                    "rejected_trials_step": simulator.rejected_steps - rejected_before,
                    "rejected_trials_cumulative": simulator.rejected_steps,
                    "event_count_step": len(simulator.events) - events_before,
                    "event_count_cumulative": len(simulator.events),
                    "remesh_jumps_step": int(remesh_occurred),
                    "remesh_jumps_cumulative": remesh_jumps,
                }
            )
            rows.append(row)
        rejection_reasons.update(str(reason) for reason in simulator.rejection_reasons)
        classification = _classification(rows, config, None)
        peak_row = max(rows, key=lambda row: float(row["max_transverse_displacement"]))
        total_energy = np.asarray([float(row["energy_total"]) for row in rows])
        growth_free_nonincrease = bool(
            config["growth_rate"] != 0.0
            or np.all(np.diff(total_energy) <= 1.0e-12 * np.maximum(1.0, np.abs(total_energy[:-1])))
        )
        accepted_dt_values = [
            float(row["accepted_dt_actual"])
            for row in rows
            if row.get("accepted_dt_actual") is not None
        ]
        mechanical_changes = [
            float(row["mechanical_energy_change_step"])
            for row in rows
            if row.get("mechanical_energy_change_step") is not None
        ]
        balance_residuals = [
            abs(float(row["energy_balance_residual_step"]))
            for row in rows
            if row.get("energy_balance_residual_step") is not None
        ]
        metrics_fieldnames = _union_fieldnames(rows)
        summary = {
            "schema_version": SCHEMA_VERSION,
            "record_type": "growth_run",
            "run": run_name,
            "representative": str(representative["name"]),
            "n_nodes": n_nodes,
            "dt_requested": float(dt),
            "growth_rate": config["growth_rate"],
            "dimensionless_groups": dimensionless_groups(config),
            "fixed_mesh_expected": True,
            "fixed_mesh_observed": remesh_jumps == 0,
            "classification": classification,
            "growth_free_energy_nonincrease": growth_free_nonincrease,
            "accepted_steps": int(simulator.accepted_steps),
            "rejected_trials": int(simulator.rejected_steps),
            "rejection_reason_counts": dict(sorted(rejection_reasons.items())),
            "event_count": len(simulator.events),
            "remesh_jumps": remesh_jumps,
            "growth_reference_energy_change": growth_energy_change_total,
            "dissipation_euler_estimate": dissipation_total,
            "remesh_energy_jump": remesh_energy_jump_total,
            "initial_A1_over_L": rows[0]["A1_over_L"],
            "peak_A1_over_L": max(row["A1_over_L"] for row in rows),
            "peak_mode_spectrum": peak_row["mode_spectrum"],
            "peak_transverse": classification.get("peak_max_transverse_displacement"),
            "peak_curvature_rms": max(row["rms_curvature"] for row in rows),
            "energy_initial": rows[0]["energy_total"],
            "energy_final": rows[-1]["energy_total"],
            "energy_min": float(np.min(total_energy)),
            "energy_max": float(np.max(total_energy)),
            "energy_span_relative_to_initial": float(
                (np.max(total_energy) - np.min(total_energy))
                / max(abs(float(rows[0]["energy_total"])), 1.0e-15)
            ),
            "mechanical_energy_change_cumulative": float(sum(mechanical_changes)),
            "energy_balance_residual_max_abs": float(max(balance_residuals, default=0.0)),
            "accepted_dt_count": len(accepted_dt_values),
            "accepted_dt_min": min(accepted_dt_values) if accepted_dt_values else None,
            "accepted_dt_max": max(accepted_dt_values) if accepted_dt_values else None,
            "accepted_dt_mean": float(np.mean(accepted_dt_values)) if accepted_dt_values else None,
            "metrics_schema": metrics_fieldnames,
            "metrics_rows": len(rows),
            "failure_reason": None,
            "boundary_condition": "endpoint positions fixed; endpoint tangents free",
            "contact_stiffness": 0.0,
            "diameter": 0.0,
            "growth_energy_definition": "E(r_before, a_grown) - E(r_before, a_before); discrete fixed-geometry reference-length energy change, not a claim of a complete continuum growth-work derivation",
            "dissipation_definition": "dt * sum_i Gamma_i |v_i|^2 evaluated at the accepted Euler trial; diagnostic estimate",
            "remesh_definition": "E(r_after_remesh, a_after_remesh) - E(r_before, a_grown), kept separate from growth and dissipation",
        }
    except (ModelError, RuntimeError, ValueError, FloatingPointError) as exc:
        failure_reason = f"{type(exc).__name__}: {exc}"
        simulator = locals().get("simulator")
        if simulator is None:
            summary = {
                "schema_version": SCHEMA_VERSION,
                "record_type": "growth_run",
                "run": run_name,
                "representative": str(representative["name"]),
                "n_nodes": n_nodes,
                "dt_requested": float(dt),
                "growth_rate": config["growth_rate"],
                "classification": {"label": "unresolved", "failure_reason": failure_reason},
                "failure_reason": failure_reason,
            }
        else:
            summary = {
                "schema_version": SCHEMA_VERSION,
                "record_type": "growth_run",
                "run": run_name,
                "representative": str(representative["name"]),
                "n_nodes": n_nodes,
                "dt_requested": float(dt),
                "growth_rate": config["growth_rate"],
                "classification": {"label": "unresolved", "failure_reason": failure_reason},
                "accepted_steps": int(simulator.accepted_steps),
                "rejected_trials": int(simulator.rejected_steps),
                "event_count": len(simulator.events),
                "remesh_jumps": remesh_jumps,
                "failure_reason": failure_reason,
            }
    _write_json(run_dir / "summary.json", summary)
    if rows:
        fields = summary.get("metrics_schema", _union_fieldnames(rows))
        with (run_dir / "metrics.csv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="raise", lineterminator="\n")
            writer.writeheader()
            for row in rows:
                writer.writerow(_jsonable({field: row.get(field) for field in fields}))
    return summary


def _convergence_status(
    runs: Sequence[Mapping[str, Any]],
    tolerance: Mapping[str, Any],
    reference_key: str,
) -> dict[str, Any]:
    """Evaluate morphology only and expose adaptive-dt audit diagnostics.

    The growth energy/work diagnostics are deliberately not pass criteria:
    the current explicit Euler runner may adapt the accepted step after trial
    rejection, so a morphology pass is not a full growth-relaxation
    convergence claim.
    """

    base = {
        "scope": "morphology-only",
        "energy_and_dissipation_gated": False,
        "run_count": len(runs),
    }
    if not runs or any(run.get("failure_reason") for run in runs):
        return {**base, "status": "numerically-unresolved", "reason": "failed run"}
    if any(not run.get("fixed_mesh_observed", True) for run in runs):
        return {
            **base,
            "status": "numerically-unresolved",
            "reason": "unexpected remesh in fixed-mesh gate",
        }
    reference = next(run for run in runs if run["run"] == reference_key)
    labels = [str(run["classification"].get("label")) for run in runs]
    accepted_dt_ranges = [
        {
            "run": run["run"],
            "requested_dt": run.get("dt_requested"),
            "accepted_dt_min": run.get("accepted_dt_min"),
            "accepted_dt_max": run.get("accepted_dt_max"),
            "accepted_dt_mean": run.get("accepted_dt_mean"),
            "accepted_dt_count": run.get("accepted_dt_count"),
            "rejected_trials": run.get("rejected_trials"),
            "event_count": run.get("event_count"),
            "energy_final": run.get("energy_final"),
            "dissipation_euler_estimate": run.get("dissipation_euler_estimate"),
        }
        for run in runs
    ]
    dt_ratios = [
        float(item["accepted_dt_min"]) / float(item["requested_dt"])
        for item in accepted_dt_ranges
        if item["accepted_dt_min"] is not None and float(item["requested_dt"]) > 0.0
    ]
    energy_values = [
        float(run["energy_final"])
        for run in runs
        if run.get("energy_final") is not None
    ]
    dissipation_values = [
        float(run["dissipation_euler_estimate"])
        for run in runs
        if run.get("dissipation_euler_estimate") is not None
    ]
    adaptive_dt_warning = bool(dt_ratios and min(dt_ratios) < 0.999999)
    audit = {
        "accepted_dt_ranges": accepted_dt_ranges,
        "accepted_dt_min_over_requested_min": min(dt_ratios) if dt_ratios else None,
        "accepted_dt_max_over_requested_max": max(dt_ratios) if dt_ratios else None,
        "adaptive_dt_warning": adaptive_dt_warning,
        "energy_final_values": energy_values,
        "energy_final_relative_span": (
            (max(energy_values) - min(energy_values)) / max(min(abs(value) for value in energy_values), 1.0e-15)
            if energy_values
            else None
        ),
        "dissipation_euler_estimate_values": dissipation_values,
        "rejected_trials_values": [int(run.get("rejected_trials", 0)) for run in runs],
        "event_count_values": [int(run.get("event_count", 0)) for run in runs],
    }
    if len(set(labels)) != 1 or labels[0] == "unresolved":
        return {
            **base,
            "status": "numerically-unresolved",
            "reason": "classification disagreement or unresolved fixture",
            "labels": labels,
            "reference_run": reference_key,
            "audit": audit,
        }
    ref_class = reference["classification"]
    ref_peak = ref_class.get("peak_max_transverse_displacement")
    ref_onset = ref_class.get("onset_time")
    ref_a1 = float(reference.get("peak_A1_over_L", np.nan))
    failures: list[str] = []
    for run in runs:
        classification = run["classification"]
        onset = classification.get("onset_time")
        if (onset is None) != (ref_onset is None):
            failures.append(f"{run['run']}: onset presence differs")
        elif onset is not None and ref_onset is not None:
            if abs(float(onset) - float(ref_onset)) / max(abs(float(ref_onset)), 1.0e-15) > float(tolerance["onset_time_relative"]):
                failures.append(f"{run['run']}: onset tolerance")
        peak = classification.get("peak_max_transverse_displacement")
        if peak is None or ref_peak is None:
            failures.append(f"{run['run']}: peak missing")
        elif abs(float(peak) - float(ref_peak)) / max(abs(float(ref_peak)), float(tolerance["peak_transverse_absolute_fraction_of_length"])) > float(tolerance["peak_transverse_relative"]):
            failures.append(f"{run['run']}: peak tolerance")
        a1 = float(run.get("peak_A1_over_L", np.nan))
        if not math.isfinite(a1) or abs(a1 - ref_a1) / max(abs(ref_a1), 1.0e-12) > float(tolerance["a1_over_length_relative"]):
            failures.append(f"{run['run']}: A1/L tolerance")
    return {
        **base,
        "status": "morphology-converged" if not failures else "numerically-unresolved",
        "reason": None if not failures else "; ".join(failures),
        "labels": labels,
        "reference_run": reference_key,
        "audit": audit,
    }


def _directory_bytes(root: Path) -> int:
    return sum(path.stat().st_size for path in root.rglob("*") if path.is_file())


def _write_compact_csv(path: Path, records: Sequence[Mapping[str, Any]]) -> None:
    fields = sorted({key for record in records for key in record.keys()})
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(_jsonable(record) for record in records)


def run_benchmark(
    config: Mapping[str, Any],
    output: Path,
    *,
    raw_config_sha256_value: str | None = None,
    raw_config_path: str | None = None,
) -> dict[str, Any]:
    _validate_config(config)
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    revision = detect_git_revision(Path.cwd())
    planned_linear_runs = len(config["linear_mode"]["n_nodes"]) * len(config["linear_mode"]["dt_values"])
    planned_growth_runs = sum(
        1 if bool(rep.get("control_only", False)) else 6
        for rep in config["growth"]["representatives"]
    )
    linear_records = run_linear_modes(config["linear_mode"], output)
    growth_records: list[dict[str, Any]] = []
    convergence: list[dict[str, Any]] = []
    growth = config["growth"]
    for representative in growth["representatives"]:
        name = str(representative["name"])
        if bool(representative.get("control_only", False)):
            control_name = f"{name}__control_n{int(growth['n_nodes_temporal'])}__dt{float(growth['dt']):.8g}"
            control = _run_growth_case(
                growth,
                representative,
                int(growth["n_nodes_temporal"]),
                float(growth["dt"]),
                output,
                control_name,
            )
            control["refinement"] = "control"
            control["dt_factor"] = 1.0
            growth_records.append(control)
            continue
        temporal_runs = []
        for factor in growth["dt_factors"]:
            dt = float(growth["dt"]) * float(factor)
            run_name = f"{name}__temporal_n{int(growth['n_nodes_temporal'])}__dt{dt:.8g}"
            record = _run_growth_case(growth, representative, int(growth["n_nodes_temporal"]), dt, output, run_name)
            record["refinement"] = "temporal"
            record["dt_factor"] = float(factor)
            growth_records.append(record)
            temporal_runs.append(record)
        temporal_reference = min(temporal_runs, key=lambda run: float(run["dt_requested"]))["run"]
        convergence.append({
            "representative": name,
            "refinement": "temporal",
            **_convergence_status(temporal_runs, growth["tolerances"], temporal_reference),
        })
        spatial_runs = []
        for raw_n in growth["n_nodes_spatial"]:
            n_nodes = int(raw_n)
            run_name = f"{name}__spatial_n{n_nodes}__dt{float(growth['dt']):.8g}"
            record = _run_growth_case(growth, representative, n_nodes, float(growth["dt"]), output, run_name)
            record["refinement"] = "spatial"
            record["dt_factor"] = 1.0
            growth_records.append(record)
            spatial_runs.append(record)
        spatial_reference = max(spatial_runs, key=lambda run: int(run["n_nodes"]))["run"]
        convergence.append({
            "representative": name,
            "refinement": "spatial",
            **_convergence_status(spatial_runs, growth["tolerances"], spatial_reference),
        })
    elapsed_seconds = time.perf_counter() - started
    effective_hash = _sha256_json(config)
    compact = {
        "schema_version": SCHEMA_VERSION,
        "benchmark": "P0-B non-contact linear-mode and growth numerical gates",
        "git_revision": revision,
        "source_revision": revision,
        "execution_revision": revision,
        "config_sha256": effective_hash,
        "effective_config_sha256": effective_hash,
        "raw_config_sha256": raw_config_sha256_value,
        "config_hash_algorithm": "SHA-256",
        "raw_config_path": raw_config_path,
        "provenance": {
            "source_revision": revision,
            "execution_revision": revision,
            "raw_config_sha256": raw_config_sha256_value,
            "effective_config_sha256": effective_hash,
            "hash_algorithm": "SHA-256",
            "python_version": platform.python_version(),
            "numpy_version": np.__version__,
            "platform": platform.platform(),
            "elapsed_seconds": float(elapsed_seconds),
            "planned_linear_run_count": planned_linear_runs,
            "planned_growth_run_count": planned_growth_runs,
            "actual_linear_run_count": len(config["linear_mode"]["n_nodes"]) * len(config["linear_mode"]["dt_values"]),
            "actual_growth_run_count": len(growth_records),
            "actual_run_count": len(config["linear_mode"]["n_nodes"]) * len(config["linear_mode"]["dt_values"]) + len(growth_records),
        },
        "boundary_condition": "endpoint positions fixed; endpoint tangents free; not a clamped-end condition",
        "non_contact_fixture": {"contact_stiffness": 0.0, "diameter": 0.0, "initial_nonintersection_required": True},
        "linear_mode": {
            "records": linear_records,
            "hessian_reference": "discrete tangent-difference Hessian K=(EI/h^3)D^T D with fixed endpoint positions and free endpoint tangents",
            "drag_reference": "Gamma=zeta*h I on free interior nodes",
            "tau_b_definition": "zeta*L^4/(EI*pi^4)",
            "tau_b_status": "definition only; measured tau_mode is reported separately and tau_b is not assumed to be the true discrete mode time",
            "decay_rate_gate": {
                "scope": "linear diagnostic threshold",
                "relative_tolerance": float(config["linear_mode"]["decay_rate_relative_tolerance"]),
                "status_field": "decay_rate_status",
            },
        },
        "growth": {
            "records": growth_records,
            "convergence": convergence,
            "gate_scope": growth["gate_scope"],
            "energy_accounting": {
                "growth_reference_energy_change": "E(r_before,a_grown)-E(r_before,a_before), fixed-geometry discrete diagnostic",
                "dissipation_euler_estimate": "dt*sum Gamma_i|v_i|^2 at accepted Euler trial",
                "remesh_energy_jump": "E(r_after_remesh,a_after_remesh)-E(r_before,a_grown)",
                "rejected_trials_and_events": "simulator counters and event counts, not energy terms",
            },
        },
        "p1b2_relation": {
            "unresolved_regions_preserved": True,
            "decision_rule": "Any classification disagreement, tolerance failure, failed run, or unexpected remesh is numerically-unresolved; no unresolved P1B.2 region is promoted to resolved.",
            "not_claimed": ["critical buckling boundary", "critical value", "universality", "phase transition", "experimental fit"],
        },
        "acceptance_gates": {
            "growth_free_energy_nonincrease": "required for growth_rate=0 records",
            "linear_hessian_and_mode": "finite-difference Hessian eigenvalue/mode overlap against the discrete reference",
            "linear_decay_rate": "diagnostic pass/fail threshold on measured-vs-discrete first-mode rate",
            "growth_temporal": "morphology-only dt, dt/2, dt/4 comparison; adaptive accepted dt/rejections and energy diagnostics are reported, not gated",
            "growth_spatial": "morphology-only three-resolution comparison; adaptive accepted dt/rejections and energy diagnostics are reported, not gated",
        },
    }
    compact_json_path = output / "compact_summary.json"
    compact_csv_path = output / "compact_summary.csv"
    effective_config_path = output / "effective_config.json"
    _write_json(compact_json_path, compact)
    _write_compact_csv(compact_csv_path, [*linear_records, *growth_records, *convergence])
    _write_json(effective_config_path, config)
    # Measure the temporary run tree before the final compact JSON rewrite.
    # The measurement point is explicit because the JSON contains its own
    # provenance fields and therefore cannot contain a self-referential exact
    # byte count.
    compact["provenance"].update(
        {
            "temporary_output_bytes_before_final_summary_write": _directory_bytes(output),
            "compact_json_bytes_before_final_summary_write": compact_json_path.stat().st_size,
            "compact_csv_bytes": compact_csv_path.stat().st_size,
            "effective_config_bytes": effective_config_path.stat().st_size,
            "measurement_note": "byte counts are measured immediately before the final compact JSON provenance write",
        }
    )
    _write_json(compact_json_path, compact)
    return compact


def _cli() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = _cli()
    try:
        config = load_config(args.config)
        summary = run_benchmark(
            config,
            args.output,
            raw_config_sha256_value=raw_config_sha256(args.config),
            raw_config_path=None if args.config is None else str(args.config),
        )
    except (BenchmarkError, OSError, json.JSONDecodeError) as exc:
        print(f"P0-B benchmark configuration/output error: {exc}", file=sys.stderr)
        return 2
    linear_failures = [
        record for record in summary["linear_mode"]["records"]
        if record.get("record_type") == "linear_decay"
        and (record.get("decay_failure") or record.get("decay_rate_status") == "fail")
    ]
    unresolved = [
        row for row in summary["growth"]["convergence"]
        if row.get("status") == "numerically-unresolved"
    ]
    print(json.dumps({"output": str(args.output), "linear_decay_failures": len(linear_failures), "growth_unresolved_gates": len(unresolved)}, sort_keys=True))
    return 1 if linear_failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
