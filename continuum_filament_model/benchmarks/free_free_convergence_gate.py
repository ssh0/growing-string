"""Focused free/free non-contact mechanics convergence gate.

This runner is an audit harness around :class:`OverdampedGrowingFilament`.
It deliberately does not add a force law: the only active physics are uniform
reference-length growth, stretching, discrete bending, and isotropic substrate
drag.  Contact, friction, adhesion, and folding are rejected by configuration
and are never used by this gate.

The gate keeps three decisions separate:

* morphology: onset, peak transverse amplitude, curvature and mode shape;
* mechanics: energy, reference-length growth work, dissipation and residuals;
* overall: resolved only when both populations converge.

A failed run or a disagreement between refinement levels is retained as
``numerically-unresolved`` with reason codes.  Per-step metrics remain in the
caller-provided temporary directory; repository-facing output is compact and
provenance-bearing.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

if __package__ in {None, ""}:  # pragma: no cover - direct-file entry point
    _HERE = Path(__file__).resolve()
    _ROOT = _HERE.parents[1]
    for _path in (_ROOT, _ROOT / "src"):
        if str(_path) not in sys.path:
            sys.path.insert(0, str(_path))
    from benchmarks.buckling_benchmark import dimensionless_groups  # type: ignore  # noqa: E402
else:  # pragma: no cover - imported package path
    # This is the single authoritative definition for G_b, G_s, and chi.
    from .buckling_benchmark import dimensionless_groups  # noqa: E402

from growing_filament.model import (  # noqa: E402
    FilamentState,
    ModelError,
    ModelParameters,
    OverdampedGrowingFilament,
    _node_weights,
    grow_reference_lengths,
    remesh,
)
from growing_filament.observables import contour_length, discrete_curvature  # noqa: E402
from growing_filament.reproducibility import (  # noqa: E402
    build_manifest,
    canonical_json_bytes,
    detect_git_revision,
    event_sequence_hash,
)

SCHEMA_VERSION = "continuum-filament-free-free-convergence-gate-1"
_SAFE_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*\Z")

DEFAULT_CONFIG: dict[str, Any] = {
    "schema_version": SCHEMA_VERSION,
    "base": {
        "length": 2.0,
        "n_nodes": 9,
        "axial_stiffness": 5.0,
        "bending_stiffness": 0.02,
        "drag_density": 1.0,
        "growth_rate": 0.5,
        "amplitude": 0.02,
        "dt": 0.004,
        "t_end": 1.5,
        "a_max_factor": 8.0,
        "max_displacement_fraction": 1.0,
        "max_retries": 8,
        "dt_min": 1.0e-10,
    },
    "representatives": [
        {"name": "straight", "role": "straight", "overrides": {"growth_rate": 0.1}},
        {"name": "boundary_near", "role": "boundary-near", "overrides": {"growth_rate": 0.3}},
        {"name": "buckled", "role": "buckled-candidate", "overrides": {"growth_rate": 0.5}},
    ],
    "controls": [
        {"name": "growth_free_control", "role": "growth-free-control", "overrides": {"growth_rate": 0.0}},
    ],
    "temporal_refinement": {
        "n_nodes": 9,
        "dt_values": [0.004, 0.002, 0.001],
    },
    "spatial_refinement": {
        "n_nodes": [5, 9, 13],
        "dt": 0.004,
    },
    "contrasts": [
        {"name": "lower_growth", "base": "buckled", "factor": "growth_rate", "value": 0.3},
        {"name": "lower_axial_stiffness", "base": "buckled", "factor": "axial_stiffness", "value": 2.5},
        {"name": "higher_bending_stiffness", "base": "buckled", "factor": "bending_stiffness", "value": 0.04},
        {"name": "higher_drag_density", "base": "buckled", "factor": "drag_density", "value": 2.0},
        {"name": "larger_initial_imperfection", "base": "buckled", "factor": "amplitude", "value": 0.04},
    ],
    "sensitivity": {
        "base": "buckled",
        "seeds": [101, 202, 303],
        "amplitude_factors": [0.75, 1.0, 1.25],
        "noise_fraction": 0.10,
    },
    "tolerances": {
        "onset_time_relative": 0.15,
        "peak_amplitude_relative": 0.20,
        "curvature_rms_relative": 0.25,
        "mode_fraction_absolute": 0.15,
        "energy_relative": 0.25,
        "growth_work_relative": 0.25,
        "dissipation_relative": 0.25,
        "endpoint_force_relative": 0.35,
        "endpoint_moment_relative": 0.35,
        "total_length_relative": 0.10,
        "mechanical_balance_relative": 0.35,
        "absolute_floor_fraction_of_length": 0.002,
    },
    "output_policy": {
        "max_metrics_rows": 512,
    },
}


class GateError(ValueError):
    """Invalid convergence-gate configuration."""


@dataclass(frozen=True)
class CaseSpec:
    name: str
    kind: str
    overrides: dict[str, Any]
    role: str | None = None
    representative: str | None = None
    refinement_axis: str | None = None
    seed: int | None = None
    perturbation: dict[str, Any] | None = None
    contrast_factor: str | None = None
    contrast_value: float | None = None


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
        if not math.isfinite(value):
            raise GateError(f"non-finite value cannot be serialized: {value!r}")
        return float(value)
    return value


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json_bytes(_jsonable(value)) + b"\n")


def _csv_value(value: Any) -> Any:
    if isinstance(value, (Mapping, list, tuple, np.ndarray)):
        return json.dumps(_jsonable(value), sort_keys=True, ensure_ascii=False)
    return _jsonable(value)


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    names = list(fields or (rows[0].keys() if rows else []))
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=names, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _csv_value(row.get(key)) for key in names})


def _safe_name(name: str, context: str = "case") -> str:
    if not _SAFE_NAME.fullmatch(name):
        raise GateError(f"{context} name must be a safe path component")
    return name


def _finite_positive(value: Any, context: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise GateError(f"{context} must be positive and finite") from exc
    if not math.isfinite(number) or number <= 0.0:
        raise GateError(f"{context} must be positive and finite")
    return number


def _integer_at_least(value: Any, minimum: int, context: str) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise GateError(f"{context} must be an integer >= {minimum}")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise GateError(f"{context} must be an integer >= {minimum}") from exc
    if not math.isfinite(number) or not number.is_integer() or int(number) < minimum:
        raise GateError(f"{context} must be an integer >= {minimum}")
    return int(number)


def _seed(value: Any, context: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise GateError(f"{context} must be an integer seed")
    result = int(value)
    if result < 0 or result >= 2**32:
        raise GateError(f"{context} must be in [0, 2**32)")
    return result


def _merged_config(config: Mapping[str, Any] | None) -> dict[str, Any]:
    result = json.loads(json.dumps(DEFAULT_CONFIG))
    if config is None:
        return result
    if not isinstance(config, Mapping):
        raise GateError("configuration root must be an object")
    for key, value in config.items():
        if isinstance(value, Mapping) and isinstance(result.get(key), Mapping):
            merged = dict(result[key])
            merged.update(dict(value))
            result[key] = merged
        else:
            result[key] = value
    for section in ("temporal_refinement", "spatial_refinement", "sensitivity", "output_policy", "tolerances"):
        if section in config and isinstance(config[section], Mapping):
            result[section] = {**DEFAULT_CONFIG[section], **dict(config[section])}
    # Lists are intentional replacement values, not concatenations.
    for section in ("representatives", "controls", "contrasts"):
        if section in config:
            result[section] = config[section]
    return result


def _validate_scope(settings: Mapping[str, Any], context: str) -> None:
    for key in ("contact_stiffness", "diameter"):
        try:
            value = float(settings.get(key, 0.0))
        except (TypeError, ValueError) as exc:
            raise GateError(f"{context}.{key} must be zero for the free/free gate") from exc
        if not math.isfinite(value) or value != 0.0:
            raise GateError(f"{context}.{key} must be zero for the free/free gate")
    for key in ("fixed_left", "fixed_right"):
        if bool(settings.get(key, False)):
            raise GateError(f"{context}.{key} must be false for the free/free gate")


def _validate_config(config: Mapping[str, Any]) -> dict[str, Any]:
    base = dict(config.get("base", {}))
    _validate_scope(base, "base")
    for key in ("length", "axial_stiffness", "bending_stiffness", "drag_density", "dt", "t_end", "a_max_factor"):
        _finite_positive(base.get(key), f"base.{key}")
    _integer_at_least(base.get("n_nodes"), 3, "base.n_nodes")
    for key in ("max_displacement_fraction",):
        value = _finite_positive(base.get(key), f"base.{key}")
        if value > 1.0:
            raise GateError("base.max_displacement_fraction must be <= 1")
    if float(base.get("growth_rate", 0.0)) < 0.0 or not math.isfinite(float(base.get("growth_rate", 0.0))):
        raise GateError("base.growth_rate must be finite and non-negative")
    _integer_at_least(base.get("max_retries"), 0, "base.max_retries")
    _finite_positive(base.get("dt_min"), "base.dt_min")
    representatives = config.get("representatives")
    if not isinstance(representatives, list) or len(representatives) < 3:
        raise GateError("at least three deterministic representatives are required")
    names: set[str] = set()
    for index, item in enumerate(representatives):
        if not isinstance(item, Mapping) or not item.get("name"):
            raise GateError(f"representatives[{index}] requires name")
        name = _safe_name(str(item["name"]), "representative")
        if name in names:
            raise GateError(f"duplicate representative: {name}")
        names.add(name)
        if not isinstance(item.get("overrides", {}), Mapping):
            raise GateError(f"representatives[{index}].overrides must be an object")
        _validate_scope(item.get("overrides", {}), f"representatives[{index}].overrides")
    controls = config.get("controls", [])
    if not isinstance(controls, list):
        raise GateError("controls must be a list")
    for index, item in enumerate(controls):
        if not isinstance(item, Mapping) or not item.get("name"):
            raise GateError(f"controls[{index}] requires name")
        _safe_name(str(item["name"]), "control")
        if not isinstance(item.get("overrides", {}), Mapping):
            raise GateError(f"controls[{index}].overrides must be an object")
        _validate_scope(item.get("overrides", {}), f"controls[{index}].overrides")

    temporal = dict(config.get("temporal_refinement", {}))
    spatial = dict(config.get("spatial_refinement", {}))
    temporal_dts = [float(value) for value in temporal.get("dt_values", [])]
    if len(temporal_dts) < 3 or len(set(temporal_dts)) < 3 or any(not math.isfinite(value) or value <= 0.0 for value in temporal_dts):
        raise GateError("temporal_refinement.dt_values requires three distinct positive values")
    _integer_at_least(temporal.get("n_nodes"), 3, "temporal_refinement.n_nodes")
    spatial_nodes = [_integer_at_least(value, 3, "spatial_refinement.n_nodes") for value in spatial.get("n_nodes", [])]
    if len(spatial_nodes) < 3 or len(set(spatial_nodes)) < 3:
        raise GateError("spatial_refinement.n_nodes requires three distinct resolutions")
    _finite_positive(spatial.get("dt"), "spatial_refinement.dt")

    factors = {"growth_rate", "axial_stiffness", "bending_stiffness", "drag_density", "amplitude"}
    contrasts = config.get("contrasts", [])
    if not isinstance(contrasts, list):
        raise GateError("contrasts must be a list")
    contrast_factors: set[str] = set()
    for index, item in enumerate(contrasts):
        if not isinstance(item, Mapping) or not item.get("name"):
            raise GateError(f"contrasts[{index}] requires name")
        _safe_name(str(item["name"]), "contrast")
        factor = str(item.get("factor", ""))
        if factor not in factors:
            raise GateError(f"contrast factor must be one of {sorted(factors)}")
        contrast_factors.add(factor)
        if not item.get("base") in names:
            raise GateError(f"contrast {item['name']} references unknown representative")
        value = float(item.get("value"))
        if not math.isfinite(value) or value <= 0.0 or (factor == "growth_rate" and value < 0.0):
            raise GateError(f"contrast {item['name']} has invalid value")
    missing_factors = factors - contrast_factors
    if missing_factors:
        raise GateError(f"explicit contrasts are missing: {sorted(missing_factors)}")

    sensitivity = dict(config.get("sensitivity", {}))
    if sensitivity.get("base") not in names:
        raise GateError("sensitivity.base must reference a representative")
    seeds = [_seed(value, f"sensitivity.seeds[{index}]") for index, value in enumerate(sensitivity.get("seeds", []))]
    if not seeds or len(set(seeds)) != len(seeds):
        raise GateError("sensitivity requires unique seeds")
    perturbations = [float(value) for value in sensitivity.get("amplitude_factors", [])]
    if not perturbations or any(not math.isfinite(value) or value <= 0.0 for value in perturbations):
        raise GateError("sensitivity.amplitude_factors must be positive and finite")
    if not 0.0 <= float(sensitivity.get("noise_fraction", 0.0)) <= 1.0:
        raise GateError("sensitivity.noise_fraction must be in [0, 1]")
    if not isinstance(config.get("tolerances", {}), Mapping):
        raise GateError("tolerances must be an object")
    for key, value in config["tolerances"].items():
        _finite_positive(value, f"tolerances.{key}")
    output_policy = config.get("output_policy", {})
    _integer_at_least(output_policy.get("max_metrics_rows", 512), 3, "output_policy.max_metrics_rows")
    return _jsonable(dict(config))


def load_config(path: Path | None = None) -> dict[str, Any]:
    raw = None if path is None else json.loads(path.read_text(encoding="utf-8"))
    return _validate_config(_merged_config(raw))


def load_config_from_mapping(config: Mapping[str, Any]) -> dict[str, Any]:
    return _validate_config(_merged_config(config))


def _representative_map(config: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    return {str(item["name"]): item for item in config["representatives"]}


def _effective(base: Mapping[str, Any], spec: CaseSpec, *, n_nodes: int | None = None, dt: float | None = None) -> dict[str, Any]:
    value = dict(base)
    value.update(spec.overrides)
    if n_nodes is not None:
        value["n_nodes"] = n_nodes
    if dt is not None:
        value["dt"] = dt
    value["n_nodes"] = _integer_at_least(value["n_nodes"], 3, f"case {spec.name}.n_nodes")
    value["length"] = _finite_positive(value["length"], f"case {spec.name}.length")
    for key in ("axial_stiffness", "bending_stiffness", "drag_density", "dt", "t_end", "a_max_factor", "dt_min"):
        value[key] = _finite_positive(value[key], f"case {spec.name}.{key}")
    value["growth_rate"] = float(value.get("growth_rate", 0.0))
    value["amplitude"] = float(value.get("amplitude", 0.0))
    if value["growth_rate"] < 0.0 or not math.isfinite(value["growth_rate"]):
        raise GateError(f"case {spec.name}.growth_rate must be finite and non-negative")
    if value["amplitude"] < 0.0 or not math.isfinite(value["amplitude"]):
        raise GateError(f"case {spec.name}.amplitude must be finite and non-negative")
    value["max_displacement_fraction"] = float(value.get("max_displacement_fraction", 1.0))
    value["max_retries"] = _integer_at_least(value.get("max_retries", 0), 0, f"case {spec.name}.max_retries")
    value["seed"] = spec.seed
    value["run_kind"] = spec.kind
    value["role"] = spec.role
    value["representative"] = spec.representative or spec.name
    value["refinement_axis"] = spec.refinement_axis
    value["contrast_factor"] = spec.contrast_factor
    value["contrast_value"] = spec.contrast_value
    _validate_scope(value, f"case {spec.name}")
    value["contact_stiffness"] = 0.0
    value["diameter"] = 0.0
    value["fixed_left"] = False
    value["fixed_right"] = False
    value["reject_crossing"] = True
    value["a_max"] = value["a_max_factor"] * value["length"] / (value["n_nodes"] - 1)
    return value


def _initial_state(config: Mapping[str, Any], *, seed: int | None = None, amplitude_factor: float = 1.0, noise_fraction: float = 0.0) -> tuple[FilamentState, dict[str, Any]]:
    length = float(config["length"])
    n_nodes = int(config["n_nodes"])
    amplitude = float(config.get("amplitude", 0.0)) * float(amplitude_factor)
    x = np.linspace(0.0, length, n_nodes)
    y = amplitude * np.sin(np.pi * x / length)
    perturbation_mode = "deterministic_sine"
    if seed is not None and noise_fraction > 0.0 and n_nodes > 3 and amplitude > 0.0:
        rng = np.random.default_rng(seed)
        noise = rng.normal(size=n_nodes - 2)
        scale = float(np.std(noise))
        if scale > 1.0e-15:
            y[1:-1] += amplitude * noise_fraction * noise / scale
        perturbation_mode = "sine_plus_seeded_node_noise"
    positions = np.column_stack((x, y))
    rest_lengths = np.linalg.norm(np.diff(positions, axis=0), axis=1)
    state = FilamentState(positions, rest_lengths)
    metadata = {
        "mode": perturbation_mode,
        "seed": seed,
        "amplitude_base": float(config.get("amplitude", 0.0)),
        "amplitude_factor": float(amplitude_factor),
        "amplitude_effective": amplitude,
        "noise_fraction": float(noise_fraction),
        "initial_condition_is_perturbation_only": True,
    }
    return state, metadata


def _mode_observables(state: FilamentState, max_mode: int = 6) -> tuple[float, np.ndarray, np.ndarray]:
    points = state.positions
    chord = points[-1] - points[0]
    chord_length = float(np.linalg.norm(chord))
    if not math.isfinite(chord_length) or chord_length <= 1.0e-12:
        raise GateError("zero endpoint chord in mode observable")
    tangent = chord / chord_length
    normal = np.asarray([-tangent[1], tangent[0]], dtype=float)
    transverse = (points - points[0]) @ normal
    lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
    total = float(np.sum(lengths))
    if total <= 1.0e-12:
        raise GateError("zero contour length in mode observable")
    u = np.concatenate(([0.0], np.cumsum(lengths))) / total
    gauss_x, gauss_w = np.polynomial.legendre.leggauss(5)
    coefficients = np.zeros(max_mode, dtype=float)
    for index in range(len(lengths)):
        u0, u1 = u[index], u[index + 1]
        local_u = 0.5 * (u1 - u0) * gauss_x + 0.5 * (u1 + u0)
        local_y = transverse[index] + (transverse[index + 1] - transverse[index]) * (local_u - u0) / max(u1 - u0, 1.0e-15)
        for mode in range(1, max_mode + 1):
            coefficients[mode - 1] += 0.5 * (u1 - u0) * float(np.sum(gauss_w * local_y * np.sin(mode * np.pi * local_u)))
    coefficients *= 2.0
    power = coefficients * coefficients
    mode_total = float(np.sum(power))
    fractions = power / mode_total if mode_total > 1.0e-30 else np.zeros_like(power)
    return float(np.max(np.abs(transverse))), coefficients, fractions


def _metric(model: OverdampedGrowingFilament, state: FilamentState, initial_amplitude: float, perturbation: Mapping[str, Any], *, requested_dt: float | None = None, accepted_dt: float | None = None) -> dict[str, Any]:
    maximum, spectrum, fractions = _mode_observables(state)
    curvature = discrete_curvature(state)
    components = model.energy_components(state.positions, state.rest_lengths)
    diagnostics = model.endpoint_diagnostics(state.positions, state.rest_lengths)
    length = contour_length(state)
    return {
        "time": float(state.time),
        "step": int(state.step),
        "n_nodes": int(state.n_nodes),
        "requested_dt": requested_dt,
        "accepted_dt": accepted_dt,
        "reference_length": float(np.sum(state.rest_lengths)),
        "total_length": float(length),
        "contour_length": float(length),
        "endpoint_distance": float(np.linalg.norm(state.positions[-1] - state.positions[0])),
        "max_transverse_amplitude": maximum,
        "initial_transverse_amplitude": float(initial_amplitude),
        "rms_curvature": float(np.sqrt(np.mean(curvature * curvature))) if len(curvature) else 0.0,
        "mode_spectrum": [float(value) for value in spectrum],
        "mode_fractions": {str(index + 1): float(value) for index, value in enumerate(fractions)},
        "first_mode_fraction": float(fractions[0]) if len(fractions) else 0.0,
        "dominant_mode": int(np.argmax(fractions) + 1) if float(np.sum(fractions)) > 1.0e-30 else None,
        "energy_total": float(sum(components.values())),
        "energy_stretch": float(components["stretch"]),
        "energy_bend": float(components["bend"]),
        "energy_contact": float(components["contact"]),
        "endpoint_force_residual_norm_max": float(diagnostics["endpoint_force_residual_norm_max"]),
        "endpoint_force_residual_left_norm": float(diagnostics["left"]["force_residual_norm"]),
        "endpoint_force_residual_right_norm": float(diagnostics["right"]["force_residual_norm"]),
        "endpoint_moment_residual_norm_max": float(diagnostics["moment_residual_norm_max"]),
        "endpoint_moment_residual_left": float(diagnostics["left"]["bending_moment"]),
        "endpoint_moment_residual_right": float(diagnostics["right"]["bending_moment"]),
        "endpoint_shear_residual_norm_max": float(max(abs(float(diagnostics["left"]["shear_equivalent_residual"])), abs(float(diagnostics["right"]["shear_equivalent_residual"])))),
        "growth_reference_energy_change_step": 0.0,
        "growth_reference_energy_change_cumulative": 0.0,
        "growth_work_step": 0.0,
        "growth_work_cumulative": 0.0,
        "dissipation_estimate_step": 0.0,
        "dissipation_estimate_cumulative": 0.0,
        "remesh_energy_jump_step": 0.0,
        "remesh_energy_jump_cumulative": 0.0,
        "mechanical_energy_change_step": 0.0,
        "mechanical_balance_residual_step": 0.0,
        "mechanical_balance_residual_cumulative": 0.0,
        "rejected_trials_step": 0,
        "rejected_trials_cumulative": 0,
        "event_count_step": 0,
        "event_count_cumulative": len(model.events),
        "remesh_occurred": False,
        "initial_condition": dict(perturbation),
    }


def _downsample_rows(rows: Sequence[Mapping[str, Any]], maximum: int) -> list[dict[str, Any]]:
    """Bound external per-run metrics without changing gate calculations."""

    if len(rows) <= maximum:
        return [dict(row) for row in rows]
    if maximum < 3:
        raise GateError("maximum metrics rows must be at least three")
    indices = set(np.linspace(0, len(rows) - 1, maximum, dtype=int).tolist())
    peak = max(range(len(rows)), key=lambda index: float(rows[index]["max_transverse_amplitude"]))
    if peak not in indices:
        removable = sorted(index for index in indices if index not in {0, len(rows) - 1})
        if removable:
            indices.remove(removable[-1])
        else:
            indices.remove(max(indices))
        indices.add(peak)
    return [dict(rows[index]) for index in sorted(indices)]


def _failure_codes(message: str) -> list[str]:
    lower = message.lower()
    codes: list[str] = []
    if "failed to find an accepted step" in lower:
        codes.append("step_not_accepted")
    if "non-finite" in lower or "nonfinite" in lower:
        codes.append("nonfinite_state")
    if "displacement" in lower:
        codes.append("displacement_exceeded")
    if "crossing" in lower or "intersection" in lower:
        codes.append("crossing_rejection")
    if "initial" in lower:
        codes.append("initial_geometry_invalid")
    return codes or ["run_failure"]


def _run_case(spec: CaseSpec, base: Mapping[str, Any], output: Path, revision: str | None, *, n_nodes: int | None = None, dt: float | None = None, sensitivity: Mapping[str, Any] | None = None) -> dict[str, Any]:
    _safe_name(spec.name)
    effective = _effective(base, spec, n_nodes=n_nodes, dt=dt)
    perturbation = dict(sensitivity or {"mode": "deterministic_sine", "seed": spec.seed, "amplitude_factor": 1.0, "noise_fraction": 0.0})
    seed = spec.seed
    amplitude_factor = float(perturbation.get("amplitude_factor", 1.0))
    noise_fraction = float(perturbation.get("noise_fraction", 0.0))
    state, initial_metadata = _initial_state(effective, seed=seed, amplitude_factor=amplitude_factor, noise_fraction=noise_fraction)
    initial_amplitude = _mode_observables(state)[0]
    parameters = ModelParameters(
        axial_stiffness=float(effective["axial_stiffness"]),
        bending_stiffness=float(effective["bending_stiffness"]),
        drag_density=float(effective["drag_density"]),
        contact_stiffness=0.0,
        diameter=0.0,
        growth_rate=float(effective["growth_rate"]),
        reference_length=float(effective["length"]) / (int(effective["n_nodes"]) - 1),
        dt=float(effective["dt"]),
        t_end=float(effective["t_end"]),
        a_max=float(effective["a_max"]),
        dt_min=float(effective["dt_min"]),
        max_retries=int(effective["max_retries"]),
        max_displacement_fraction=float(effective["max_displacement_fraction"]),
        fixed_left=False,
        fixed_right=False,
        reject_crossing=True,
    )
    groups = dict(dimensionless_groups(effective))
    rows: list[dict[str, Any]] = []
    failure_reason: str | None = None
    failure_codes: list[str] = []
    failure_event: dict[str, Any] | None = None
    simulator: OverdampedGrowingFilament | None = None
    growth_cumulative = 0.0
    dissipation_cumulative = 0.0
    remesh_cumulative = 0.0
    balance_cumulative = 0.0
    try:
        simulator = OverdampedGrowingFilament(state, parameters)
        rows.append(_metric(simulator, state, initial_amplitude, initial_metadata))
        end_tolerance = max(1.0e-15, 1.0e-12 * max(1.0, abs(parameters.t_end)))
        while simulator.state.time < parameters.t_end - end_tolerance or (simulator.accepted_steps == 0 and simulator.state.time < parameters.t_end):
            before = simulator.state.copy()
            energy_before = simulator.energy(before.positions, before.rest_lengths)
            rejected_before = simulator.rejected_steps
            events_before = len(simulator.events)
            requested = min(float(parameters.dt), float(parameters.t_end - before.time))
            simulator.step(requested)
            after = simulator.state.copy()
            accepted = float(simulator.accepted_dts[-1])
            grown = grow_reference_lengths(before.rest_lengths, parameters.growth_rate, accepted)
            remeshed_positions, remeshed_lengths = remesh(before.positions, grown, parameters.a_max)
            energy_after_growth = simulator.energy(before.positions, grown)
            energy_after_remesh = simulator.energy(remeshed_positions, remeshed_lengths)
            growth_change = float(energy_after_growth - energy_before)
            remesh_jump = float(energy_after_remesh - energy_after_growth)
            if remeshed_positions.shape == after.positions.shape:
                velocity = (after.positions - remeshed_positions) / accepted
                gamma = parameters.drag_density * _node_weights(after.rest_lengths)
                dissipation = float(accepted * np.sum(gamma[:, None] * velocity * velocity))
            else:  # Defensive path for a future remesher with different topology.
                dissipation = None
            energy_change = float(simulator.energy(after.positions, after.rest_lengths) - energy_before)
            balance = None if dissipation is None else float(energy_change - growth_change - remesh_jump + dissipation)
            growth_cumulative += growth_change
            remesh_cumulative += remesh_jump
            if dissipation is not None:
                dissipation_cumulative += dissipation
            if balance is not None:
                balance_cumulative += balance
            row = _metric(
                simulator,
                after,
                initial_amplitude,
                initial_metadata,
                requested_dt=requested,
                accepted_dt=accepted,
            )
            row.update({
                "growth_reference_energy_change_step": growth_change,
                "growth_reference_energy_change_cumulative": growth_cumulative,
                "growth_work_step": growth_change,
                "growth_work_cumulative": growth_cumulative,
                "dissipation_estimate_step": dissipation,
                "dissipation_estimate_cumulative": dissipation_cumulative,
                "remesh_energy_jump_step": remesh_jump,
                "remesh_energy_jump_cumulative": remesh_cumulative,
                "mechanical_energy_change_step": energy_change - growth_change - remesh_jump,
                "mechanical_balance_residual_step": balance,
                "mechanical_balance_residual_cumulative": balance_cumulative,
                "rejected_trials_step": int(simulator.rejected_steps - rejected_before),
                "rejected_trials_cumulative": int(simulator.rejected_steps),
                "event_count_step": int(len(simulator.events) - events_before),
                "event_count_cumulative": len(simulator.events),
                "remesh_occurred": bool(remeshed_positions.shape != before.positions.shape),
            })
            rows.append(row)
    except (ModelError, RuntimeError, ValueError, FloatingPointError) as exc:
        failure_reason = f"{type(exc).__name__}: {exc}"
        failure_codes = _failure_codes(str(exc))
        event = getattr(exc, "event", None)
        if isinstance(event, Mapping):
            failure_event = dict(event)
    if simulator is None:
        failure_events = [failure_event] if failure_event is not None else []
        compact_events = {
            "event_sequence_hash": event_sequence_hash(failure_events) if failure_events else None,
            "event_count": len(failure_events),
            "accepted_steps": 0,
            "rejected_trials": 0,
            "rejection_reason_counts": {},
            "contact_enabled": False,
            "contact_event_count": 0,
            "failure_event": failure_event,
        }
        final_state = state
    else:
        rejection_counts = Counter(str(value) for value in simulator.rejection_reasons)
        compact_events = {
            "event_sequence_hash": event_sequence_hash(simulator.event_log),
            "event_count": len(simulator.events),
            "accepted_steps": int(simulator.accepted_steps),
            "rejected_trials": int(simulator.rejected_steps),
            "rejection_reason_counts": dict(sorted(rejection_counts.items())),
            "accepted_dt_min": float(min(simulator.accepted_dts)) if simulator.accepted_dts else None,
            "accepted_dt_max": float(max(simulator.accepted_dts)) if simulator.accepted_dts else None,
            "accepted_dt_mean": float(np.mean(simulator.accepted_dts)) if simulator.accepted_dts else None,
            "requested_dt": float(parameters.dt),
            "contact_enabled": False,
            "contact_event_count": 0,
        }
        final_state = simulator.state
    classification = _classify(rows, effective, failure_reason)
    mechanical = _mechanical_summary(rows)
    morphology = _morphology_summary(rows, classification)
    metadata = {
        "benchmark": "free_free_noncontact_mechanics_convergence_gate",
        "boundary": "free/free",
        "physical_scope": ["uniform_reference_length_growth", "stretching", "discrete_bending", "isotropic_substrate_drag"],
        "excluded": ["contact", "friction", "adhesion", "folding", "localized_growth", "parameter_identification", "gray5_physics_fit"],
        "initial_condition": initial_metadata,
        "dimensionless_groups": groups,
        "failure_reason": failure_reason,
        "failure_reason_codes": failure_codes,
    }
    manifest = build_manifest(parameters, state, final_state=final_state, events=(simulator.event_log if simulator else failure_events), metadata=metadata, input_data=effective, git_revision=revision)
    manifest.update({"run_name": spec.name, "run_kind": spec.kind, "seed": seed, "perturbation": initial_metadata, "refinement_axis": spec.refinement_axis})
    run_dir = output / "_runs" / spec.name
    run_dir.mkdir(parents=True, exist_ok=True)
    _write_json(run_dir / "manifest.json", manifest)
    _write_json(run_dir / "events.json", compact_events)
    metrics_rows_to_write = _downsample_rows(rows, int(effective.get("max_metrics_rows", 512))) if rows else []
    if rows:
        _write_csv(run_dir / "metrics.csv", metrics_rows_to_write)
    if simulator is not None:
        _write_json(run_dir / "summary.json", _compact_run_summary(spec, effective, groups, classification, morphology, mechanical, compact_events, failure_reason, failure_codes, initial_metadata, manifest, rows))
    else:
        _write_json(run_dir / "summary.json", {"run_name": spec.name, "failure_reason": failure_reason, "failure_reason_codes": failure_codes, "classification": classification})
    result = _compact_run_summary(spec, effective, groups, classification, morphology, mechanical, compact_events, failure_reason, failure_codes, initial_metadata, manifest, rows)
    result["metrics_rows_total"] = len(rows)
    result["metrics_rows_saved"] = len(metrics_rows_to_write)
    result["metrics_path"] = str((run_dir / "metrics.csv").relative_to(output)) if rows else None
    result["manifest_path"] = str((run_dir / "manifest.json").relative_to(output))
    return result


def _classify(rows: Sequence[Mapping[str, Any]], config: Mapping[str, Any], failure_reason: str | None) -> dict[str, Any]:
    if failure_reason or not rows:
        return {"label": "numerically-unresolved", "onset_time": None, "peak_time": None, "peak_amplitude": None, "unresolved_reason_category": "numerical_nonconvergence"}
    initial = float(rows[0]["max_transverse_amplitude"])
    threshold = max(3.0 * initial, 0.005 * float(config["length"]))
    onset = next((row for row in rows[1:] if float(row["max_transverse_amplitude"]) > threshold), None)
    peak = max(rows, key=lambda row: float(row["max_transverse_amplitude"]))
    label = "buckling-candidate" if onset is not None and float(peak["time"]) > float(onset["time"]) else "sub-threshold-or-relaxing"
    return {
        "label": label,
        "onset_time": None if onset is None else float(onset["time"]),
        "onset_definition": "first accepted observation with max_transverse_amplitude > max(3*initial_amplitude, 0.005*L); morphology label only",
        "threshold_amplitude": float(threshold),
        "peak_time": float(peak["time"]),
        "peak_amplitude": float(peak["max_transverse_amplitude"]),
        "peak_curvature_rms": float(peak["rms_curvature"]),
        "peak_mode_fractions": dict(peak["mode_fractions"]),
        "post_onset_observed": bool(onset is not None and float(peak["time"]) > float(onset["time"])),
        "unresolved_reason_category": None,
    }


def _morphology_summary(rows: Sequence[Mapping[str, Any]], classification: Mapping[str, Any]) -> dict[str, Any]:
    peak = max(rows, key=lambda row: float(row["max_transverse_amplitude"])) if rows else {}
    return {
        "label": classification.get("label"),
        "onset_time": classification.get("onset_time"),
        "peak_time": classification.get("peak_time"),
        "peak_transverse_amplitude": classification.get("peak_amplitude"),
        "peak_curvature_rms": classification.get("peak_curvature_rms"),
        "peak_mode_spectrum": peak.get("mode_spectrum"),
        "peak_mode_fractions": peak.get("mode_fractions"),
        "peak_dominant_mode": peak.get("dominant_mode"),
        "initial_transverse_amplitude": rows[0].get("initial_transverse_amplitude") if rows else None,
    }


def _mechanical_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"energy_initial": None, "energy_final": None, "growth_work_cumulative": None, "dissipation_estimate_cumulative": None, "mechanical_balance_residual_cumulative": None}
    final = rows[-1]
    energy = [float(row["energy_total"]) for row in rows]
    balance = [abs(float(row["mechanical_balance_residual_step"])) for row in rows if row.get("mechanical_balance_residual_step") is not None]
    return {
        "energy_initial": float(rows[0]["energy_total"]),
        "energy_final": float(final["energy_total"]),
        "energy_min": float(min(energy)),
        "energy_max": float(max(energy)),
        "energy_span": float(max(energy) - min(energy)),
        "reference_length_initial": float(rows[0]["reference_length"]),
        "reference_length_final": float(final["reference_length"]),
        "total_length_initial": float(rows[0]["total_length"]),
        "total_length_final": float(final["total_length"]),
        "growth_reference_energy_change_cumulative": float(final["growth_reference_energy_change_cumulative"]),
        "growth_work_cumulative": float(final["growth_work_cumulative"]),
        "dissipation_estimate_cumulative": float(final["dissipation_estimate_cumulative"]),
        "remesh_energy_jump_cumulative": float(final["remesh_energy_jump_cumulative"]),
        "mechanical_energy_change_cumulative": float(sum(float(row["mechanical_energy_change_step"]) for row in rows[1:])),
        "mechanical_balance_residual_cumulative": float(final["mechanical_balance_residual_cumulative"]),
        "mechanical_balance_residual_max_abs": float(max(balance, default=0.0)),
        "endpoint_force_residual_final": float(final["endpoint_force_residual_norm_max"]),
        "endpoint_moment_residual_final": float(final["endpoint_moment_residual_norm_max"]),
        "endpoint_shear_residual_final": float(final["endpoint_shear_residual_norm_max"]),
        "endpoint_force_residual_max": float(max(float(row["endpoint_force_residual_norm_max"]) for row in rows)),
        "endpoint_moment_residual_max": float(max(float(row["endpoint_moment_residual_norm_max"]) for row in rows)),
        "total_length_max": float(max(float(row["total_length"]) for row in rows)),
        "remesh_occurred": bool(any(bool(row["remesh_occurred"]) for row in rows)),
    }


def _compact_run_summary(spec: CaseSpec, effective: Mapping[str, Any], groups: Mapping[str, Any], classification: Mapping[str, Any], morphology: Mapping[str, Any], mechanical: Mapping[str, Any], events: Mapping[str, Any], failure_reason: str | None, failure_codes: Sequence[str], initial_condition: Mapping[str, Any], manifest: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    accepted_values = [float(row["accepted_dt"]) for row in rows if row.get("accepted_dt") is not None]
    requested_values = [float(row["requested_dt"]) for row in rows if row.get("requested_dt") is not None]
    return {
        "schema_version": SCHEMA_VERSION,
        "run_name": spec.name,
        "run_kind": spec.kind,
        "role": spec.role,
        "representative": spec.representative or spec.name,
        "refinement_axis": spec.refinement_axis,
        "contrast_factor": spec.contrast_factor,
        "contrast_value": spec.contrast_value,
        "boundary": "free/free",
        "contact_enabled": False,
        "effective_config": dict(effective),
        "effective_values": {key: effective.get(key) for key in ("growth_rate", "axial_stiffness", "bending_stiffness", "drag_density", "amplitude", "n_nodes", "dt", "t_end")},
        "dimensionless_groups": dict(groups),
        "classification": dict(classification),
        "morphology": dict(morphology),
        "mechanical": dict(mechanical),
        "accepted_dt_values": sorted(set(accepted_values)),
        "requested_dt_values": sorted(set(requested_values)),
        "accepted_dt_min": min(accepted_values) if accepted_values else None,
        "accepted_dt_max": max(accepted_values) if accepted_values else None,
        "accepted_dt_mean": float(np.mean(accepted_values)) if accepted_values else None,
        "rejected_trials": int(events.get("rejected_trials", 0)),
        "event_count": int(events.get("event_count", 0)),
        "events": dict(events),
        "initial_condition": dict(initial_condition),
        "seed": spec.seed,
        "perturbation": dict(initial_condition),
        "failure_reason": failure_reason,
        "failure_reason_codes": list(failure_codes),
        "provenance": {
            "git_revision": manifest.get("git_revision"),
            "input_hash": manifest.get("input_hash"),
            "initial_state_hash": manifest.get("initial_state_hash"),
            "canonical_state_hash": manifest.get("canonical_state_hash"),
            "event_sequence_hash": events.get("event_sequence_hash"),
            "python_version": manifest.get("python_version"),
            "numpy_version": manifest.get("numpy_version"),
        },
        "metrics_rows": len(rows),
    }


def _relative(value: Any, reference: Any, floor: float) -> float | None:
    if value is None or reference is None:
        return None
    try:
        a, b = float(value), float(reference)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(a) or not math.isfinite(b):
        return None
    return abs(a - b) / max(abs(b), floor)


def _compare_refinement(runs: Sequence[Mapping[str, Any]], tolerances: Mapping[str, Any], axis: str, reference_name: str) -> dict[str, Any]:
    base = {"axis": axis, "run_count": len(runs), "reference_run": reference_name, "morphology_status": "numerically-unresolved", "mechanics_status": "numerically-unresolved", "status": "numerically-unresolved", "morphology_reason_codes": [], "mechanics_reason_codes": [], "audit": {}}
    if not runs:
        base["morphology_reason_codes"] = ["missing_refinement_runs"]
        base["mechanics_reason_codes"] = ["missing_refinement_runs"]
        return base
    reference = next((run for run in runs if run["run_name"] == reference_name), runs[-1])
    run_failures = [run for run in runs if run.get("failure_reason") or run.get("classification", {}).get("label") == "numerically-unresolved"]
    morph_reasons: set[str] = set()
    mech_reasons: set[str] = set()
    if run_failures:
        morph_reasons.add("run_numerically_unresolved")
        mech_reasons.add("run_numerically_unresolved")
    if any(bool(run.get("mechanical", {}).get("remesh_occurred")) for run in runs):
        morph_reasons.add("remesh_occurred")
        mech_reasons.add("remesh_occurred")
    labels = [str(run.get("classification", {}).get("label")) for run in runs]
    if len(set(labels)) != 1 or labels[0] == "numerically-unresolved":
        morph_reasons.add("classification_disagreement")
    ref_class = reference.get("classification", {})
    ref_onset = ref_class.get("onset_time")
    ref_morph = reference.get("morphology", {})
    length_floor = float(tolerances["absolute_floor_fraction_of_length"])
    for run in runs:
        cls = run.get("classification", {})
        onset = cls.get("onset_time")
        if (onset is None) != (ref_onset is None):
            morph_reasons.add("onset_presence_disagreement")
        elif onset is not None and _relative(onset, ref_onset, 1.0e-12) > float(tolerances["onset_time_relative"]):
            morph_reasons.add("onset_time_out_of_tolerance")
        if _relative(cls.get("peak_amplitude"), ref_class.get("peak_amplitude"), length_floor) is None or _relative(cls.get("peak_amplitude"), ref_class.get("peak_amplitude"), length_floor) > float(tolerances["peak_amplitude_relative"]):
            morph_reasons.add("peak_amplitude_out_of_tolerance")
        if _relative(cls.get("peak_curvature_rms"), ref_class.get("peak_curvature_rms"), 1.0e-12) is None or _relative(cls.get("peak_curvature_rms"), ref_class.get("peak_curvature_rms"), 1.0e-12) > float(tolerances["curvature_rms_relative"]):
            morph_reasons.add("curvature_rms_out_of_tolerance")
        fractions = (run.get("morphology", {}).get("peak_mode_fractions") or {})
        ref_fractions = (ref_morph.get("peak_mode_fractions") or {})
        for key in set(fractions) | set(ref_fractions):
            if abs(float(fractions.get(key, 0.0)) - float(ref_fractions.get(key, 0.0))) > float(tolerances["mode_fraction_absolute"]):
                morph_reasons.add("mode_fraction_out_of_tolerance")

    mechanical_fields = (
        ("energy_final", "energy_relative"),
        ("growth_work_cumulative", "growth_work_relative"),
        ("dissipation_estimate_cumulative", "dissipation_relative"),
        ("endpoint_force_residual_final", "endpoint_force_relative"),
        ("endpoint_moment_residual_final", "endpoint_moment_relative"),
        ("total_length_final", "total_length_relative"),
        ("mechanical_balance_residual_cumulative", "mechanical_balance_relative"),
    )
    ref_mechanical = reference.get("mechanical", {})
    floors = {
        "energy_final": 1.0e-12,
        "growth_work_cumulative": 1.0e-12,
        "dissipation_estimate_cumulative": 1.0e-12,
        "endpoint_force_residual_final": 1.0e-12,
        "endpoint_moment_residual_final": 1.0e-12,
        "total_length_final": 1.0,
        # The balance residual is an absolute diagnostic with energy-scale
        # units; normalise it against an O(1) scale rather than treating a
        # near-zero residual as a relative quantity with a tiny denominator.
        "mechanical_balance_residual_cumulative": 1.0,
    }
    for run in runs:
        current = run.get("mechanical", {})
        for field, tolerance_key in mechanical_fields:
            difference = _relative(current.get(field), ref_mechanical.get(field), floors[field])
            if difference is None or difference > float(tolerances[tolerance_key]):
                mech_reasons.add(f"{field}_out_of_tolerance")
    accepted = [run.get("accepted_dt_min") for run in runs]
    rejected = [int(run.get("rejected_trials", 0)) for run in runs]
    base["audit"] = {
        "requested_dt": [run.get("effective_values", {}).get("dt") for run in runs],
        "accepted_dt_min": accepted,
        "accepted_dt_max": [run.get("accepted_dt_max") for run in runs],
        "accepted_dt_mean": [run.get("accepted_dt_mean") for run in runs],
        "rejected_trials": rejected,
        "event_counts": [int(run.get("event_count", 0)) for run in runs],
        "energy_final": [run.get("mechanical", {}).get("energy_final") for run in runs],
        "growth_work_cumulative": [run.get("mechanical", {}).get("growth_work_cumulative") for run in runs],
        "dissipation_estimate_cumulative": [run.get("mechanical", {}).get("dissipation_estimate_cumulative") for run in runs],
        "endpoint_force_residual_final": [run.get("mechanical", {}).get("endpoint_force_residual_final") for run in runs],
        "endpoint_moment_residual_final": [run.get("mechanical", {}).get("endpoint_moment_residual_final") for run in runs],
        "total_length_final": [run.get("mechanical", {}).get("total_length_final") for run in runs],
    }
    base["morphology_reason_codes"] = sorted(morph_reasons)
    base["mechanics_reason_codes"] = sorted(mech_reasons)
    base["morphology_status"] = "morphology-converged" if not morph_reasons else "numerically-unresolved"
    base["mechanics_status"] = "mechanics-converged" if not mech_reasons else "numerically-unresolved"
    base["status"] = "resolved" if not morph_reasons and not mech_reasons else "numerically-unresolved"
    base["reference_metrics"] = {"classification": ref_class, "morphology": ref_morph, "mechanical": ref_mechanical}
    return base


def _specs_for_representative(config: Mapping[str, Any], representative: Mapping[str, Any], axis: str, *, n_nodes: int | None = None, dt: float | None = None) -> CaseSpec:
    return CaseSpec(str(representative["name"]), "deterministic_fixture", dict(representative.get("overrides", {})), role=str(representative.get("role", representative["name"])), representative=str(representative["name"]), refinement_axis=axis)


def _run_deterministic(config: Mapping[str, Any], output: Path, revision: str | None) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    base = config["base"]
    representatives = list(config["representatives"])
    temporal_all: list[dict[str, Any]] = []
    spatial_all: list[dict[str, Any]] = []
    convergence: list[dict[str, Any]] = []
    temporal_cfg = config["temporal_refinement"]
    spatial_cfg = config["spatial_refinement"]
    for item in representatives:
        name = str(item["name"])
        temporal_runs: list[dict[str, Any]] = []
        for index, raw_dt in enumerate(temporal_cfg["dt_values"], start=1):
            dt = float(raw_dt)
            spec = _specs_for_representative(config, item, "temporal")
            spec = CaseSpec(f"{name}__temporal_dt{dt:.8g}", spec.kind, spec.overrides, spec.role, spec.representative, "temporal")
            run = _run_case(spec, base, output, revision, n_nodes=int(temporal_cfg["n_nodes"]), dt=dt)
            temporal_runs.append(run)
            temporal_all.append(run)
        temporal_reference = min(temporal_runs, key=lambda run: float(run["effective_values"]["dt"]))
        convergence.append({"representative": name, "role": item.get("role"), **_compare_refinement(temporal_runs, config["tolerances"], "temporal", temporal_reference["run_name"])})
        spatial_runs: list[dict[str, Any]] = []
        for raw_n in spatial_cfg["n_nodes"]:
            n_nodes = int(raw_n)
            spec = _specs_for_representative(config, item, "spatial")
            spec = CaseSpec(f"{name}__spatial_n{n_nodes}", spec.kind, spec.overrides, spec.role, spec.representative, "spatial")
            run = _run_case(spec, base, output, revision, n_nodes=n_nodes, dt=float(spatial_cfg["dt"]))
            spatial_runs.append(run)
            spatial_all.append(run)
        spatial_reference = max(spatial_runs, key=lambda run: int(run["effective_values"]["n_nodes"]))
        convergence.append({"representative": name, "role": item.get("role"), **_compare_refinement(spatial_runs, config["tolerances"], "spatial", spatial_reference["run_name"])})
    return temporal_all, spatial_all, convergence


def _run_controls(config: Mapping[str, Any], output: Path, revision: str | None) -> list[dict[str, Any]]:
    base = config["base"]
    n_nodes = int(config["temporal_refinement"]["n_nodes"])
    dt = float(config["temporal_refinement"]["dt_values"][-1])
    records: list[dict[str, Any]] = []
    for item in config.get("controls", []):
        spec = CaseSpec(f"control__{item['name']}", "deterministic_control", dict(item.get("overrides", {})), role=str(item.get("role", item["name"])), representative=str(item["name"]), refinement_axis="control")
        records.append(_run_case(spec, base, output, revision, n_nodes=n_nodes, dt=dt))
    return records


def _run_contrasts(config: Mapping[str, Any], output: Path, revision: str | None) -> list[dict[str, Any]]:
    representatives = _representative_map(config)
    n_nodes = int(config["temporal_refinement"]["n_nodes"])
    dt = float(config["temporal_refinement"]["dt_values"][-1])
    records: list[dict[str, Any]] = []
    for item in config.get("contrasts", []):
        base_rep = representatives[str(item["base"])]
        factor = str(item["factor"])
        overrides = dict(base_rep.get("overrides", {}))
        overrides[factor] = float(item["value"])
        spec = CaseSpec(
            f"contrast__{item['name']}", "parameter_contrast", overrides,
            role=f"contrast:{factor}", representative=str(item["base"]), refinement_axis="contrast",
            contrast_factor=factor, contrast_value=float(item["value"]),
        )
        records.append(_run_case(spec, config["base"], output, revision, n_nodes=n_nodes, dt=dt))
    return records


def _run_sensitivity(config: Mapping[str, Any], output: Path, revision: str | None) -> list[dict[str, Any]]:
    representatives = _representative_map(config)
    sensitivity = config["sensitivity"]
    item = representatives[str(sensitivity["base"])]
    n_nodes = int(config["temporal_refinement"]["n_nodes"])
    dt = float(config["temporal_refinement"]["dt_values"][-1])
    records: list[dict[str, Any]] = []
    for seed in sensitivity["seeds"]:
        for amplitude_factor in sensitivity["amplitude_factors"]:
            token = f"seed{int(seed)}_amp{float(amplitude_factor):.8g}"
            spec = CaseSpec(
                f"sensitivity__{item['name']}__{token}", "sensitivity_replicate", dict(item.get("overrides", {})),
                role="initial-condition-sensitivity", representative=str(item["name"]), refinement_axis="sensitivity", seed=int(seed),
                perturbation={"amplitude_factor": float(amplitude_factor), "noise_fraction": float(sensitivity["noise_fraction"])},
            )
            records.append(_run_case(spec, config["base"], output, revision, n_nodes=n_nodes, dt=dt, sensitivity=spec.perturbation))
    return records


def _compact_rows(records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "run_name": item.get("run_name"),
            "run_kind": item.get("run_kind"),
            "role": item.get("role"),
            "representative": item.get("representative"),
            "refinement_axis": item.get("refinement_axis"),
            "seed": item.get("seed"),
            "perturbation": item.get("perturbation"),
            "growth_rate": item.get("effective_values", {}).get("growth_rate"),
            "axial_stiffness": item.get("effective_values", {}).get("axial_stiffness"),
            "bending_stiffness": item.get("effective_values", {}).get("bending_stiffness"),
            "drag_density": item.get("effective_values", {}).get("drag_density"),
            "amplitude": item.get("effective_values", {}).get("amplitude"),
            "n_nodes": item.get("effective_values", {}).get("n_nodes"),
            "requested_dt": item.get("effective_values", {}).get("dt"),
            "accepted_dt_min": item.get("accepted_dt_min"),
            "accepted_dt_max": item.get("accepted_dt_max"),
            "rejected_trials": item.get("rejected_trials"),
            "event_count": item.get("event_count"),
            "G_b": item.get("dimensionless_groups", {}).get("G_b"),
            "G_s": item.get("dimensionless_groups", {}).get("G_s"),
            "chi": item.get("dimensionless_groups", {}).get("chi"),
            "label": item.get("classification", {}).get("label"),
            "onset_time": item.get("classification", {}).get("onset_time"),
            "peak_transverse_amplitude": item.get("morphology", {}).get("peak_transverse_amplitude"),
            "peak_curvature_rms": item.get("morphology", {}).get("peak_curvature_rms"),
            "mode_spectrum": item.get("morphology", {}).get("peak_mode_spectrum"),
            "mode_fractions": item.get("morphology", {}).get("peak_mode_fractions"),
            "energy_initial": item.get("mechanical", {}).get("energy_initial"),
            "energy_final": item.get("mechanical", {}).get("energy_final"),
            "reference_length_final": item.get("mechanical", {}).get("reference_length_final"),
            "total_length_final": item.get("mechanical", {}).get("total_length_final"),
            "growth_reference_energy_change_cumulative": item.get("mechanical", {}).get("growth_reference_energy_change_cumulative"),
            "growth_work_cumulative": item.get("mechanical", {}).get("growth_work_cumulative"),
            "dissipation_estimate_cumulative": item.get("mechanical", {}).get("dissipation_estimate_cumulative"),
            "endpoint_force_residual_final": item.get("mechanical", {}).get("endpoint_force_residual_final"),
            "endpoint_moment_residual_final": item.get("mechanical", {}).get("endpoint_moment_residual_final"),
            "total_length_final": item.get("mechanical", {}).get("total_length_final"),
            "failure_reason": item.get("failure_reason"),
            "failure_reason_codes": item.get("failure_reason_codes"),
        }
        for item in records
    ]


def run_gate(config: Mapping[str, Any] | None, output: Path) -> dict[str, Any]:
    effective = load_config_from_mapping(config) if config is not None else load_config()
    effective["base"]["max_metrics_rows"] = int(effective.get("output_policy", {}).get("max_metrics_rows", 512))
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    revision = detect_git_revision(Path(__file__).resolve().parents[2])
    temporal, spatial, convergence = _run_deterministic(effective, output, revision)
    controls = _run_controls(effective, output, revision)
    contrasts = _run_contrasts(effective, output, revision)
    sensitivity = _run_sensitivity(effective, output, revision)
    all_runs = [*temporal, *spatial, *controls, *contrasts, *sensitivity]
    temporal_rows = _compact_rows(temporal)
    spatial_rows = _compact_rows(spatial)
    _write_csv(output / "temporal_runs.csv", temporal_rows)
    _write_csv(output / "spatial_runs.csv", spatial_rows)
    _write_csv(output / "controls.csv", _compact_rows(controls))
    _write_csv(output / "contrasts.csv", _compact_rows(contrasts))
    _write_csv(output / "sensitivity_replicates.csv", _compact_rows(sensitivity))
    _write_json(output / "convergence_summary.json", {"schema_version": SCHEMA_VERSION, "tolerances": effective["tolerances"], "representatives": convergence})
    _write_csv(output / "convergence_summary.csv", convergence)
    _write_json(output / "effective_config.json", effective)
    config_hash = hashlib.sha256(canonical_json_bytes(effective)).hexdigest()
    overall = "resolved" if convergence and all(item["status"] == "resolved" for item in convergence) else "numerically-unresolved"
    summary = {
        "schema_version": SCHEMA_VERSION,
        "benchmark": "focused free/free non-contact mechanics convergence gate",
        "source_revision": revision,
        "config_sha256": config_hash,
        "boundary": "free/free",
        "contact_enabled": False,
        "physical_scope": ["uniform_reference_length_growth", "stretching", "discrete_bending", "isotropic_substrate_drag"],
        "excluded": ["contact", "friction", "adhesion", "folding", "gray5_physics_fit", "parameter_identification"],
        "dimensionless_group_definition": dimensionless_groups(effective["base"])["definition"],
        "deterministic_fixture_population": {"run_count": len(temporal) + len(spatial), "temporal_count": len(temporal), "spatial_count": len(spatial), "runs": temporal_rows + spatial_rows},
        "control_population": {"run_count": len(controls), "runs": _compact_rows(controls)},
        "parameter_contrast_population": {"run_count": len(contrasts), "factors": sorted({str(item.get("contrast_factor")) for item in contrasts}), "runs": _compact_rows(contrasts)},
        "sensitivity_replicate_population": {"run_count": len(sensitivity), "seeds": sorted({item.get("seed") for item in sensitivity}), "runs": _compact_rows(sensitivity)},
        "convergence": convergence,
        "overall_status": overall,
        "unresolved_policy": "Any run failure, unresolved morphology, or mechanics/work/energy disagreement remains numerically-unresolved; morphology convergence never promotes mechanics convergence.",
        "large_artifact_policy": "per-run metrics, events, and manifests remain under _runs in the caller-provided output; only compact summaries are repository candidates",
    }
    _write_json(output / "compact_summary.json", summary)
    artifact_paths = [output / name for name in ("compact_summary.json", "convergence_summary.json", "convergence_summary.csv", "temporal_runs.csv", "spatial_runs.csv", "controls.csv", "contrasts.csv", "sensitivity_replicates.csv", "effective_config.json")]
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "benchmark": summary["benchmark"],
        "source_revision": revision,
        "config_sha256": config_hash,
        "run_count": len(all_runs),
        "deterministic_fixture_count": len(temporal) + len(spatial),
        "control_count": len(controls),
        "contrast_count": len(contrasts),
        "sensitivity_replicate_count": len(sensitivity),
        "boundary": "free/free",
        "contact_enabled": False,
        "overall_status": overall,
        "artifacts": {path.name: {"bytes": path.stat().st_size, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()} for path in artifact_paths},
    }
    _write_json(output / "compact_manifest.json", manifest)
    return summary


def _cli() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = _cli()
    run_gate(load_config(args.config), args.output)
    print(json.dumps({"output": str(args.output.resolve())}, ensure_ascii=False))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
