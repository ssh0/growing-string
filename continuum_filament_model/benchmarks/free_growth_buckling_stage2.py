"""Stage 2 free/free growth--relaxation--buckling exploration.

This runner is intentionally an experiment harness around the existing
``OverdampedGrowingFilament`` API.  It does not add contact, friction, or a
second solver.  The default design varies growth rate, bending/axial ratio, substrate drag,
initial imperfection, time step, spatial resolution, and seeded initial
imperfections.  Deterministic fixtures, parameter contrasts, and exploratory
replicates are kept in separate tables; labels are morphology observations, not
phase boundaries.

Large model trajectories and the raw video-analysis artifacts belong in a
caller-provided temporary directory.  The suite summary contains only
endpoint trajectories and sampled scalar diagnostics so it can be reviewed or
committed as a compact provenance-bearing result.
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
    _SRC = _ROOT / "src"
    for _path in (_ROOT, _SRC):
        if str(_path) not in sys.path:
            sys.path.insert(0, str(_path))

if __package__ in {None, ""}:  # pragma: no cover - direct-file entry point
    from benchmarks.buckling_benchmark import dimensionless_groups as _shared_dimensionless_groups  # noqa: E402
else:
    from .buckling_benchmark import dimensionless_groups as _shared_dimensionless_groups  # noqa: E402

from growing_filament.io import save_trajectory  # noqa: E402
from growing_filament.model import (  # noqa: E402
    FilamentState,
    ModelError,
    ModelParameters,
    OverdampedGrowingFilament,
    _node_weights,
    grow_reference_lengths,
)
from growing_filament.observables import contour_length, discrete_curvature  # noqa: E402
from growing_filament.reproducibility import (  # noqa: E402
    build_manifest,
    canonical_json_bytes,
    detect_git_revision,
    event_sequence_hash,
)
from growing_filament.video_comparison import (  # noqa: E402
    RegistrationConfig,
    SegmentationConfig,
    compare_with_model,
    run_pipeline,
    sha256_file,
)

SCHEMA_VERSION = "continuum-filament-stage2-free-free-1"
MODEL_INADEQUACY_THRESHOLDS: dict[str, float] = {
    "shape_rmse_px": 5.0,
    "relative_length_error": 0.25,
}
DEFAULT_VIDEO_CONFIG: dict[str, Any] = {
    "polarity": "dark",
    "background": "local_median",
    "threshold": "absolute",
    "threshold_value": 0.8,
    "frame_stride": 15,
    "min_component_size": 30,
    "max_components": 1,
    "roi": [280, 150, 420, 320],
}
DEFAULT_CONFIG: dict[str, Any] = {
    "schema_version": SCHEMA_VERSION,
    "base": {
        "length": 2.0,
        "n_nodes": 9,
        "axial_stiffness": 5.0,
        "bending_stiffness": 0.1,
        "drag_density": 1.0,
        "growth_rate": 0.05,
        "amplitude": 0.02,
        "rest_length_mode": "geometric_initial",
        "rest_length_factor": 1.0,
        "dt": 0.002,
        "t_end": 4.0,
        "a_max_factor": 8.0,
        "max_displacement_fraction": 1.0,
        "max_retries": 8,
        "dt_min": 1.0e-10,
        "max_report_rows": 256,
    },
    "fixtures": [
        {"name": "slow_growth_low_bend", "overrides": {"growth_rate": 0.05, "bending_stiffness": 0.02, "amplitude": 0.02}},
        {"name": "fast_growth_low_bend", "overrides": {"growth_rate": 0.20, "bending_stiffness": 0.02, "amplitude": 0.02}},
        {"name": "fast_growth_high_bend", "overrides": {"growth_rate": 0.20, "bending_stiffness": 0.10, "amplitude": 0.02}},
        {"name": "fast_growth_large_imperfection", "overrides": {"growth_rate": 0.20, "bending_stiffness": 0.02, "amplitude": 0.05}},
        {"name": "fast_growth_small_imperfection", "overrides": {"growth_rate": 0.20, "bending_stiffness": 0.02, "amplitude": 0.01}},
    ],
    "contrast_conditions": [
        {"name": "fast_growth_low_bend_soft_axial", "base_fixture": "fast_growth_low_bend", "factor": "axial_stiffness", "overrides": {"axial_stiffness": 2.5}},
        {"name": "fast_growth_low_bend_stiff_axial", "base_fixture": "fast_growth_low_bend", "factor": "axial_stiffness", "overrides": {"axial_stiffness": 10.0}},
        {"name": "fast_growth_low_bend_low_drag", "base_fixture": "fast_growth_low_bend", "factor": "drag_density", "overrides": {"drag_density": 0.5}},
        {"name": "fast_growth_low_bend_high_drag", "base_fixture": "fast_growth_low_bend", "factor": "drag_density", "overrides": {"drag_density": 2.0}},
    ],
    "refinement": {
        "base_fixture": "fast_growth_low_bend",
        "n_nodes": [9, 13],
        "dt_values": [0.002, 0.001],
    },
    "replicates": {
        "fixtures": ["slow_growth_low_bend", "fast_growth_low_bend", "fast_growth_high_bend"],
        "seeds": [101, 202, 303],
        "noise_fraction": 0.25,
    },
    "video": DEFAULT_VIDEO_CONFIG,
}


class Stage2Error(ValueError):
    """Invalid Stage 2 configuration or result."""


_SAFE_RUN_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*\Z")


def _validate_run_name(name: str, context: str = "run") -> str:
    if not _SAFE_RUN_NAME.fullmatch(name):
        raise Stage2Error(f"{context} name must be a safe single path component")
    return name


@dataclass(frozen=True)
class RunSpec:
    name: str
    kind: str
    overrides: dict[str, Any]
    seed: int | None = None
    trial: int = 0
    base_fixture: str | None = None
    contrast_factor: str | None = None
    contrast_value: float | None = None


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
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
            raise Stage2Error(f"non-finite result cannot be serialized: {value!r}")
        return float(value)
    return value


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json_bytes(value) + b"\n")


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(fields or (rows[0].keys() if rows else []))
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow(_jsonable(row))


def _merge_config(config: Mapping[str, Any] | None) -> dict[str, Any]:
    value = json.loads(json.dumps(DEFAULT_CONFIG))
    if config:
        for key, item in config.items():
            if isinstance(item, Mapping) and isinstance(value.get(key), Mapping):
                merged = dict(value[key])
                merged.update(dict(item))
                value[key] = merged
            else:
                value[key] = item
        if "fixtures" in config and "contrast_conditions" not in config:
            value["contrast_conditions"] = []
    return value


def _validate_config(config: dict[str, Any]) -> dict[str, Any]:
    base = dict(config["base"])
    required = ("length", "n_nodes", "axial_stiffness", "bending_stiffness", "drag_density", "dt", "t_end", "a_max_factor")
    for key in required:
        if key not in base or not math.isfinite(float(base[key])):
            raise Stage2Error(f"base.{key} must be finite")
    if int(base["n_nodes"]) < 3 or int(base["n_nodes"]) != base["n_nodes"]:
        raise Stage2Error("base.n_nodes must be an integer >= 3")
    for key in ("length", "axial_stiffness", "bending_stiffness", "drag_density", "dt", "t_end", "a_max_factor"):
        if float(base[key]) <= 0.0:
            raise Stage2Error(f"base.{key} must be positive")
    if base.get("rest_length_mode", "geometric_initial") not in {"geometric_initial", "projected_spacing"}:
        raise Stage2Error("base.rest_length_mode must be geometric_initial or projected_spacing")
    if float(base.get("max_displacement_fraction", 1.0)) <= 0.0:
        raise Stage2Error("base.max_displacement_fraction must be positive")
    fixtures = config.get("fixtures", [])
    if not isinstance(fixtures, list) or not fixtures:
        raise Stage2Error("at least one deterministic fixture is required")
    names: list[str] = []
    for item in fixtures:
        if not isinstance(item, Mapping) or not item.get("name"):
            raise Stage2Error("every fixture requires a name")
        name = str(item["name"])
        _validate_run_name(name, "fixture")
        names.append(name)
    if len(names) != len(set(names)):
        raise Stage2Error("fixture names must be unique")
    contrast_conditions = config.get("contrast_conditions", [])
    if not isinstance(contrast_conditions, list):
        raise Stage2Error("contrast_conditions must be a list")
    fixture_names = set(names)
    condition_names: set[str] = set()
    allowed_factors = {"axial_stiffness", "drag_density"}
    for condition in contrast_conditions:
        if not isinstance(condition, Mapping):
            raise Stage2Error("contrast conditions require objects")
        name = str(condition.get("name", ""))
        base_fixture = str(condition.get("base_fixture", ""))
        factor = str(condition.get("factor", ""))
        overrides = condition.get("overrides")
        if not name or name in fixture_names or name in condition_names:
            raise Stage2Error("contrast condition names must be unique and not reuse fixture names")
        _validate_run_name(name, "contrast condition")
        if base_fixture not in fixture_names:
            raise Stage2Error(f"contrast condition references unknown fixture: {base_fixture}")
        if factor not in allowed_factors or not isinstance(overrides, Mapping) or set(overrides) != {factor}:
            raise Stage2Error("contrast conditions must vary exactly axial_stiffness or drag_density")
        value = float(overrides[factor])
        if not math.isfinite(value) or value <= 0.0:
            raise Stage2Error(f"contrast {name}.{factor} must be positive and finite")
        condition_names.add(name)
    refinement = config.get("refinement", {})
    if len(refinement.get("n_nodes", [])) < 2 or len(refinement.get("dt_values", [])) < 2:
        raise Stage2Error("refinement requires at least two spatial resolutions and two time steps")
    rep = config.get("replicates", {})
    seeds = [int(seed) for seed in rep.get("seeds", [])]
    if len(seeds) != len(set(seeds)):
        raise Stage2Error("replicate seeds must be unique")
    if not 0.0 <= float(rep.get("noise_fraction", 0.0)) <= 1.0:
        raise Stage2Error("replicates.noise_fraction must be in [0, 1]")
    return config


def load_config(path: Path | None) -> dict[str, Any]:
    raw = {} if path is None else json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        raise Stage2Error("Stage 2 config root must be an object")
    return _validate_config(_merge_config(raw))


def _fixture_specs(config: Mapping[str, Any]) -> list[RunSpec]:
    result: list[RunSpec] = []
    for item in config["fixtures"]:
        if not isinstance(item, Mapping) or not item.get("name"):
            raise Stage2Error("fixture entries require name")
        result.append(RunSpec(str(item["name"]), "deterministic_fixture", dict(item.get("overrides", {}))))
    return result


def _contrast_specs(config: Mapping[str, Any], fixtures: Sequence[RunSpec]) -> list[RunSpec]:
    by_name = {item.name: item for item in fixtures}
    result: list[RunSpec] = []
    controlled = ("growth_rate", "bending_stiffness", "amplitude", "n_nodes", "dt")
    for item in config.get("contrast_conditions", []):
        name = str(item["name"])
        base_fixture = str(item["base_fixture"])
        factor = str(item["factor"])
        value = float(item["overrides"][factor])
        base_overrides = dict(by_name[base_fixture].overrides)
        overrides = {**base_overrides, factor: value}
        baseline = dict(config["base"])
        baseline.update(base_overrides)
        effective = dict(baseline)
        effective.update({factor: value})
        for key in controlled:
            if effective.get(key) != baseline.get(key):
                raise Stage2Error(f"contrast {name} changes controlled parameter {key}")
        result.append(RunSpec(
            name,
            "parameter_contrast",
            overrides,
            base_fixture=base_fixture,
            contrast_factor=factor,
            contrast_value=value,
        ))
    return result


def _all_specs(config: Mapping[str, Any]) -> list[RunSpec]:
    fixtures = _fixture_specs(config)
    result = list(fixtures)
    result.extend(_contrast_specs(config, fixtures))
    by_name = {item.name: item for item in fixtures}
    refinement = config["refinement"]
    base_name = str(refinement["base_fixture"])
    if base_name not in by_name:
        raise Stage2Error(f"refinement references unknown fixture: {base_name}")
    for n_nodes in refinement["n_nodes"]:
        for dt in refinement["dt_values"]:
            result.append(RunSpec(
                f"{base_name}_refine_n{int(n_nodes)}_dt{float(dt):.8g}",
                "numerical_refinement",
                {**by_name[base_name].overrides, "n_nodes": int(n_nodes), "dt": float(dt)},
                base_fixture=base_name,
            ))
    rep = config["replicates"]
    for fixture_name in rep["fixtures"]:
        if fixture_name not in by_name:
            raise Stage2Error(f"replicates references unknown fixture: {fixture_name}")
        for trial, seed in enumerate(rep["seeds"], start=1):
            result.append(RunSpec(
                f"{fixture_name}_replicate_{trial:02d}_seed{int(seed)}",
                "exploratory_replicate",
                {**by_name[fixture_name].overrides, "noise_fraction": float(rep["noise_fraction"])},
                seed=int(seed),
                trial=trial,
                base_fixture=fixture_name,
            ))
    return result


def _effective(base: Mapping[str, Any], spec: RunSpec) -> dict[str, Any]:
    value = dict(base)
    value.update(spec.overrides)
    value["n_nodes"] = int(value["n_nodes"])
    value["dt"] = float(value["dt"])
    value["t_end"] = float(value["t_end"])
    value["seed"] = spec.seed
    value["trial"] = spec.trial
    value["run_kind"] = spec.kind
    value["base_fixture"] = spec.base_fixture or spec.name
    value["contrast_factor"] = spec.contrast_factor
    value["contrast_value"] = spec.contrast_value
    if value["n_nodes"] < 3 or value["dt"] <= 0.0 or value["t_end"] <= 0.0:
        raise Stage2Error(f"invalid run values for {spec.name}")
    return value


def _initial_state(config: Mapping[str, Any], seed: int | None) -> FilamentState:
    length = float(config["length"])
    n_nodes = int(config["n_nodes"])
    amplitude = float(config.get("amplitude", 0.0))
    x = np.linspace(0.0, length, n_nodes)
    y = amplitude * np.sin(np.pi * x / length)
    noise_fraction = float(config.get("noise_fraction", 0.0))
    if seed is not None and noise_fraction and n_nodes - 2 >= 2:
        rng = np.random.default_rng(int(seed))
        noise = rng.normal(size=n_nodes - 2)
        noise /= float(np.std(noise))
        y[1:-1] += amplitude * noise_fraction * noise
    positions = np.column_stack((x, y))
    projected = length / (n_nodes - 1)
    if config.get("rest_length_mode", "geometric_initial") == "projected_spacing":
        rest_lengths = np.full(n_nodes - 1, projected * float(config.get("rest_length_factor", 1.0)))
    else:
        rest_lengths = np.linalg.norm(np.diff(positions, axis=0), axis=1) * float(config.get("rest_length_factor", 1.0))
    return FilamentState(positions, rest_lengths)


def _transverse_and_modes(state: FilamentState, max_mode: int = 6) -> tuple[float, np.ndarray, np.ndarray]:
    points = state.positions
    chord = points[-1] - points[0]
    chord_length = float(np.linalg.norm(chord))
    if chord_length <= 1.0e-12:
        raise Stage2Error("endpoint chord collapsed")
    tangent = chord / chord_length
    normal = np.asarray([-tangent[1], tangent[0]])
    transverse = (points - points[0]) @ normal
    lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
    total = float(np.sum(lengths))
    if total <= 1.0e-12:
        raise Stage2Error("zero contour length")
    cumulative = np.concatenate(([0.0], np.cumsum(lengths))) / total
    gauss_x, gauss_w = np.polynomial.legendre.leggauss(5)
    coefficients = np.zeros(max_mode, dtype=float)
    for index, segment_length in enumerate(lengths):
        u0, u1 = cumulative[index], cumulative[index + 1]
        local_u = 0.5 * (u1 - u0) * gauss_x + 0.5 * (u1 + u0)
        local_y = transverse[index] + (transverse[index + 1] - transverse[index]) * (local_u - u0) / max(u1 - u0, 1.0e-15)
        for mode in range(1, max_mode + 1):
            coefficients[mode - 1] += 0.5 * (u1 - u0) * float(np.sum(gauss_w * local_y * np.sin(mode * np.pi * local_u)))
    return float(np.max(np.abs(transverse))), coefficients * 2.0, transverse


def _metric_row(model: OverdampedGrowingFilament, state: FilamentState, initial_amp: float) -> dict[str, Any]:
    max_transverse, coefficients, _ = _transverse_and_modes(state)
    mode_power = np.square(coefficients)
    mode_total = float(np.sum(mode_power))
    curvature = discrete_curvature(state)
    components = model.energy_components(state.positions, state.rest_lengths)
    diagnostics = model.endpoint_diagnostics(state.positions, state.rest_lengths)
    lengths = np.linalg.norm(np.diff(state.positions, axis=0), axis=1)
    axial_force = model.parameters.axial_stiffness * (lengths - state.rest_lengths) / state.rest_lengths
    mode_fractions = mode_power / mode_total if mode_total > 1.0e-30 else np.zeros_like(mode_power)
    return {
        "time": float(state.time),
        "step": int(state.step),
        "n_nodes": int(state.n_nodes),
        "left_x": float(state.positions[0, 0]),
        "left_y": float(state.positions[0, 1]),
        "right_x": float(state.positions[-1, 0]),
        "right_y": float(state.positions[-1, 1]),
        "endpoint_distance": float(np.linalg.norm(state.positions[-1] - state.positions[0])),
        "contour_length": contour_length(state),
        "reference_length": float(np.sum(state.rest_lengths)),
        "max_transverse_amplitude": float(max_transverse),
        "initial_transverse_amplitude": float(initial_amp),
        "rms_curvature": float(np.sqrt(np.mean(curvature * curvature))) if len(curvature) else 0.0,
        "max_curvature": float(np.max(curvature)) if len(curvature) else 0.0,
        "axial_force_min": float(np.min(axial_force)),
        "axial_force_max": float(np.max(axial_force)),
        "axial_force_compression_proxy": float(max(0.0, -float(np.min(axial_force)))),
        "axial_force_abs_max": float(np.max(np.abs(axial_force))),
        "endpoint_force_residual_left_norm": float(diagnostics["left"]["force_residual_norm"]),
        "endpoint_force_residual_right_norm": float(diagnostics["right"]["force_residual_norm"]),
        "endpoint_force_residual_norm_max": float(diagnostics["endpoint_force_residual_norm_max"]),
        "endpoint_moment_residual_left": float(diagnostics["left"]["bending_moment"]),
        "endpoint_moment_residual_right": float(diagnostics["right"]["bending_moment"]),
        "endpoint_moment_residual_norm_max": float(diagnostics["moment_residual_norm_max"]),
        "endpoint_shear_residual_norm_max": float(max(abs(float(diagnostics["left"]["shear_equivalent_residual"])), abs(float(diagnostics["right"]["shear_equivalent_residual"])))),
        "energy_total": float(sum(components.values())),
        "energy_stretch": float(components["stretch"]),
        "energy_bend": float(components["bend"]),
        "energy_contact": float(components["contact"]),
        "mode_fractions": {str(index + 1): float(value) for index, value in enumerate(mode_fractions)},
        "first_mode_fraction": float(mode_fractions[0]) if len(mode_fractions) else 0.0,
        "dominant_mode": int(np.argmax(mode_fractions) + 1) if len(mode_fractions) else None,
        "growth_work_increment": 0.0,
        "growth_work_cumulative": 0.0,
        "dissipation_increment": 0.0,
        "dissipation_cumulative": 0.0,
        "mechanical_balance_residual_increment": 0.0,
        "mechanical_balance_residual_cumulative": 0.0,
        "accepted_dt": None,
    }


def _sample_rows(rows: Sequence[Mapping[str, Any]], maximum: int) -> list[dict[str, Any]]:
    if len(rows) <= maximum:
        return [dict(row) for row in rows]
    indices = set(np.linspace(0, len(rows) - 1, maximum, dtype=int).tolist())
    for key in ("max_transverse_amplitude", "axial_force_compression_proxy"):
        indices.add(max(range(len(rows)), key=lambda i: float(rows[i][key])))
    return [dict(rows[index]) for index in sorted(indices)]


def _compact_events(events: Sequence[Mapping[str, Any]], model: OverdampedGrowingFilament) -> dict[str, Any]:
    rejections = [event for event in events if event.get("event_type") == "step_attempt" and event.get("accepted") is False]
    rejection_counts = Counter(str(event.get("reason")) for event in rejections)
    geometry_counts = Counter(str(event.get("diagnostic_event_type")) for event in events if event.get("event_type") == "geometry_diagnostic")
    return {
        "event_schema_version": events[0].get("schema_version") if events else None,
        "event_sequence_hash": event_sequence_hash(events),
        "event_count": len(events),
        "accepted_steps": int(model.accepted_steps),
        "rejected_steps": int(model.rejected_steps),
        "rejection_reason_counts": dict(sorted(rejection_counts.items())),
        "geometry_event_counts": dict(sorted(geometry_counts.items())),
        "accepted_dt_min": float(min(model.accepted_dts)) if model.accepted_dts else None,
        "accepted_dt_max": float(max(model.accepted_dts)) if model.accepted_dts else None,
        "rejected_dt_min": float(min(model.rejected_dts)) if model.rejected_dts else None,
        "rejected_dt_max": float(max(model.rejected_dts)) if model.rejected_dts else None,
        "rejected_trials": [
            {
                "time_before": event.get("time_before"),
                "requested_dt": event.get("requested_dt"),
                "trial_dt": event.get("trial_dt"),
                "reason": event.get("reason"),
            }
            for event in rejections[:64]
        ],
        "contact_enabled": False,
        "contact_event_count": int(sum(geometry_counts.values())),
    }


def _apply_work_diagnostics(rows: list[dict[str, Any]], trajectory: Sequence[FilamentState], model: OverdampedGrowingFilament) -> None:
    growth_cumulative = 0.0
    dissipation_cumulative = 0.0
    balance_cumulative = 0.0
    for index, (before, after) in enumerate(zip(trajectory, trajectory[1:]), start=1):
        dt = float(after.time - before.time)
        if dt <= 0.0:
            rows[index]["growth_work_increment"] = None
            rows[index]["dissipation_increment"] = None
            rows[index]["mechanical_balance_residual_increment"] = None
            rows[index]["accepted_dt"] = None
            continue
        energy_before = model.energy(before.positions, before.rest_lengths)
        grown = grow_reference_lengths(before.rest_lengths, model.parameters.growth_rate, dt)
        growth_increment = float(model.energy(before.positions, grown) - energy_before)
        if before.positions.shape != after.positions.shape:
            dissipation_increment: float | None = None
            balance_increment: float | None = None
        else:
            velocity = (after.positions - before.positions) / dt
            gamma = model.parameters.drag_density * _node_weights(before.rest_lengths)
            dissipation_increment = float(dt * np.sum(gamma[:, None] * velocity * velocity))
            energy_after = model.energy(after.positions, after.rest_lengths)
            balance_increment = float(energy_after - energy_before - growth_increment + dissipation_increment)
        growth_cumulative += growth_increment
        if dissipation_increment is not None:
            dissipation_cumulative += dissipation_increment
        if balance_increment is not None:
            balance_cumulative += balance_increment
        rows[index]["growth_work_increment"] = growth_increment
        rows[index]["growth_work_cumulative"] = growth_cumulative
        rows[index]["dissipation_increment"] = dissipation_increment
        rows[index]["dissipation_cumulative"] = dissipation_cumulative
        rows[index]["mechanical_balance_residual_increment"] = balance_increment
        rows[index]["mechanical_balance_residual_cumulative"] = balance_cumulative
        rows[index]["accepted_dt"] = dt


def _classify(rows: Sequence[Mapping[str, Any]], config: Mapping[str, Any], failure_reason: str | None = None) -> dict[str, Any]:
    if failure_reason is not None:
        return {
            "label": "numerically-unresolved",
            "onset_time": None,
            "onset_definition": "run failed before a complete accepted trajectory was available",
            "unresolved_reason_category": "numerical_nonconvergence",
        }
    if not rows:
        return {
            "label": "numerically-unresolved",
            "onset_time": None,
            "onset_definition": "no accepted observations",
            "unresolved_reason_category": "numerical_nonconvergence",
        }
    initial = float(rows[0]["max_transverse_amplitude"])
    threshold = max(3.0 * initial, 0.005 * float(config["length"]))
    onset_row = next((row for row in rows[1:] if float(row["max_transverse_amplitude"]) > threshold), None)
    peak = max(rows, key=lambda row: float(row["max_transverse_amplitude"]))
    post_onset = onset_row is not None and float(peak["time"]) > float(onset_row["time"])
    label = "buckling-candidate" if onset_row is not None and post_onset else "sub-threshold-or-relaxing"
    return {
        "label": label,
        "unresolved_reason_category": None,
        "onset_time": None if onset_row is None else float(onset_row["time"]),
        "onset_definition": "first accepted observation with max_transverse_amplitude > max(3*initial_amplitude, 0.005*length); morphology label only",
        "threshold_max_transverse_amplitude": float(threshold),
        "peak_time": float(peak["time"]),
        "peak_max_transverse_amplitude": float(peak["max_transverse_amplitude"]),
        "peak_first_mode_fraction": float(peak["first_mode_fraction"]),
        "peak_rms_curvature": float(peak["rms_curvature"]),
        "post_onset_observed": bool(post_onset),
        "phase_boundary_claim": False,
    }


def _dimensionless_groups(config: Mapping[str, Any]) -> dict[str, float | str]:
    groups = dict(_shared_dimensionless_groups(config))
    tau_b = float(groups["tau_b"])
    tau_s = float(groups["tau_s"])
    length = float(config["length"])
    groups.update({
        "dt_over_tau_s": float(float(config["dt"]) / tau_s),
        "t_end_over_tau_b": float(float(config["t_end"]) / tau_b),
        "initial_amplitude_over_L": float(float(config.get("amplitude", 0.0)) / length),
    })
    return groups


def run_case(spec: RunSpec, base_config: Mapping[str, Any], output: Path, revision: str | None, save_trajectory_file: bool = False) -> dict[str, Any]:
    _validate_run_name(spec.name)
    config = _effective(base_config, spec)
    groups = _dimensionless_groups(config)
    state = _initial_state(config, spec.seed)
    spacing = float(config["length"]) / (int(config["n_nodes"]) - 1)
    params = ModelParameters(
        axial_stiffness=float(config["axial_stiffness"]),
        bending_stiffness=float(config["bending_stiffness"]),
        drag_density=float(config["drag_density"]),
        contact_stiffness=0.0,
        diameter=0.0,
        growth_rate=float(config.get("growth_rate", 0.0)),
        reference_length=spacing,
        dt=float(config["dt"]),
        t_end=float(config["t_end"]),
        a_max=float(config["a_max_factor"]) * spacing,
        dt_min=float(config.get("dt_min", 1.0e-10)),
        max_retries=int(config.get("max_retries", 8)),
        max_displacement_fraction=float(config.get("max_displacement_fraction", 1.0)),
        fixed_left=False,
        fixed_right=False,
        reject_crossing=True,
    )
    initial_amp = float(np.max(np.abs(_transverse_and_modes(state)[2])))
    simulator: OverdampedGrowingFilament | None = None
    trajectory: list[FilamentState] = [state.copy()]
    failure_reason: str | None = None
    try:
        simulator = OverdampedGrowingFilament(state, params)
        trajectory = simulator.run()
    except (ModelError, RuntimeError, ValueError, FloatingPointError) as exc:
        failure_reason = f"{type(exc).__name__}: {exc}"
        if simulator is not None:
            initial_state = simulator.initial_state.copy()
            current_state = simulator.state.copy()
            trajectory = [initial_state]
            if current_state.step != initial_state.step or current_state.time > initial_state.time + 1.0e-15:
                trajectory.append(current_state)
    if simulator is None:
        # Keep the schema usable when initialization itself fails.
        rows: list[dict[str, Any]] = []
        compact_events: dict[str, Any] = {"event_sequence_hash": None, "event_count": 0, "accepted_steps": 0, "rejected_steps": 0, "contact_enabled": False}
        final_state = state
    else:
        rows = [_metric_row(simulator, item, initial_amp) for item in trajectory]
        _apply_work_diagnostics(rows, trajectory, simulator)
        compact_events = _compact_events(simulator.event_log, simulator)
        final_state = simulator.state
    classification = _classify(rows, config, failure_reason)
    metadata = {
        "benchmark": "stage2_free_free_growth_relaxation_buckling",
        "run_kind": spec.kind,
        "base_fixture": spec.base_fixture or spec.name,
        "contrast_factor": spec.contrast_factor,
        "contrast_value": spec.contrast_value,
        "seed": spec.seed,
        "trial": spec.trial,
        "boundary": "free/free",
        "contact_enabled": False,
        "physical_scope": ["uniform_reference_length_growth", "stretching", "discrete_bending", "isotropic_substrate_drag"],
        "excluded": ["segment_contact", "node_contact", "friction", "adhesion", "folding", "localized_growth", "parameter_identification"],
        "axial_force_proxy_definition": "EA*(geometric_segment_length-rest_length)/rest_length; compression proxy=max(0,-min(segment force))",
        "mode_fraction_definition": "squared piecewise-linear arc-length sine coefficients divided by sum of modes 1..6",
        "growth_work_definition": "E(r_before,a_before grown)-E(r_before,a_before)",
        "dissipation_definition": "accepted dt * sum_i(zeta*w_i*|v_i|^2), Euler trajectory estimate",
        "dimensionless_groups": groups,
        "contrast_controlled_parameters": ["growth_rate", "bending_stiffness", "amplitude", "n_nodes", "dt"],
        "failure_reason": failure_reason,
    }
    manifest = build_manifest(params, state, final_state=final_state, events=(simulator.event_log if simulator else []), metadata=metadata, input_data=config, git_revision=revision)
    manifest["run_name"] = spec.name
    manifest["run_kind"] = spec.kind
    manifest["seed"] = spec.seed
    manifest["trial"] = spec.trial
    manifest["full_event_sequence_hash"] = compact_events.get("event_sequence_hash")
    stored_rows = _sample_rows(rows, int(config.get("max_report_rows", 256)))
    endpoint_trajectory = [
        {key: row[key] for key in ("time", "step", "left_x", "left_y", "right_x", "right_y", "endpoint_distance", "reference_length", "contour_length")}
        for row in stored_rows
    ]
    result = {
        "schema_version": SCHEMA_VERSION,
        "run_name": spec.name,
        "run_kind": spec.kind,
        "base_fixture": spec.base_fixture or spec.name,
        "contrast_factor": spec.contrast_factor,
        "contrast_value": spec.contrast_value,
        "seed": spec.seed,
        "trial": spec.trial,
        "effective_config": config,
        "boundary": "free/free",
        "contact_enabled": False,
        "failure_reason": failure_reason,
        "classification": classification,
        "dimensionless_groups": groups,
        "accepted_steps": int(simulator.accepted_steps) if simulator else 0,
        "rejected_steps": int(simulator.rejected_steps) if simulator else 0,
        "accepted_dt_values": sorted({float(value) for value in (simulator.accepted_dts if simulator else [])}),
        "rejected_dt_values": sorted({float(value) for value in (simulator.rejected_dts if simulator else [])}),
        "events": compact_events,
        "provenance": {
            "git_revision": revision,
            "manifest_schema_version": manifest.get("manifest_schema_version"),
            "input_hash": manifest.get("input_hash"),
            "initial_state_hash": manifest.get("initial_state_hash"),
            "canonical_state_hash": manifest.get("canonical_state_hash"),
            "event_sequence_hash": compact_events.get("event_sequence_hash"),
            "python_version": manifest.get("python_version"),
            "numpy_version": manifest.get("numpy_version"),
            "run_kind": spec.kind,
            "base_fixture": spec.base_fixture or spec.name,
            "contrast_factor": spec.contrast_factor,
            "contrast_value": spec.contrast_value,
            "dimensionless_groups": groups,
        },
        "endpoint_trajectory": endpoint_trajectory,
        "observables": stored_rows,
        "metrics_rows_total": len(rows),
        "metrics_rows_saved": len(stored_rows),
        "trajectory_saved": False,
        "parameter_identification": "suppressed; exploratory no-contact comparison only",
    }
    run_dir = output / "_runs" / spec.name
    run_dir.mkdir(parents=True, exist_ok=True)
    _write_json(run_dir / "summary.json", result)
    _write_json(run_dir / "events.json", compact_events)
    _write_json(run_dir / "manifest.json", manifest)
    if rows:
        _write_csv(run_dir / "metrics.csv", rows)
    if save_trajectory_file and simulator is not None:
        save_trajectory(run_dir / "trajectory.npz", trajectory, params, metadata=metadata, events=simulator.event_log, manifest=manifest, input_data=config)
        result["trajectory_saved"] = True
        result["trajectory_path"] = str((run_dir / "trajectory.npz").relative_to(output))
        _write_json(run_dir / "summary.json", result)
    return result


def _summary_row(result: Mapping[str, Any]) -> dict[str, Any]:
    config = result["effective_config"]
    groups = result["dimensionless_groups"]
    classification = result["classification"]
    peak = max(result["observables"], key=lambda row: float(row["max_transverse_amplitude"])) if result["observables"] else {}
    final = result["observables"][-1] if result["observables"] else {}
    return {
        "run_name": result["run_name"],
        "run_kind": result["run_kind"],
        "base_fixture": result["base_fixture"],
        "contrast_factor": result.get("contrast_factor"),
        "contrast_value": result.get("contrast_value"),
        "seed": result["seed"],
        "trial": result["trial"],
        "growth_rate": config.get("growth_rate"),
        "bending_stiffness": config.get("bending_stiffness"),
        "axial_stiffness": config.get("axial_stiffness"),
        "drag_density": config.get("drag_density"),
        "tau_b": groups["tau_b"],
        "tau_s": groups["tau_s"],
        "G_b": groups["G_b"],
        "G_s": groups["G_s"],
        "chi": groups["chi"],
        "growth_bending_number": groups["G_b"],
        "dt_over_tau_b": groups["dt_over_tau_b"],
        "dt_over_tau_s": groups["dt_over_tau_s"],
        "mesh_ratio_initial_dx_over_L": groups["mesh_ratio_initial_dx_over_L"],
        "initial_amplitude_over_L": groups["initial_amplitude_over_L"],
        "n_nodes": config.get("n_nodes"),
        "dt": config.get("dt"),
        "t_end": config.get("t_end"),
        "initial_amplitude": result["observables"][0]["initial_transverse_amplitude"] if result["observables"] else None,
        "label": classification.get("label"),
        "onset_time": classification.get("onset_time"),
        "peak_time": classification.get("peak_time"),
        "peak_amplitude": classification.get("peak_max_transverse_amplitude"),
        "peak_first_mode_fraction": classification.get("peak_first_mode_fraction"),
        "peak_rms_curvature": classification.get("peak_rms_curvature"),
        "final_endpoint_distance": final.get("endpoint_distance"),
        "final_reference_length": final.get("reference_length"),
        "final_contour_length": final.get("contour_length"),
        "final_compression_proxy": final.get("axial_force_compression_proxy"),
        "accepted_steps": result["accepted_steps"],
        "rejected_steps": result["rejected_steps"],
        "failure_reason": result["failure_reason"],
        "unresolved_reason_category": classification.get("unresolved_reason_category"),
        "phase_boundary_claim": False,
    }


def _output_relative_path(path: str | Path, output: Path) -> str:
    candidate = Path(path).expanduser()
    try:
        return candidate.resolve().relative_to(output.expanduser().resolve()).as_posix()
    except (OSError, ValueError):
        return candidate.name


def _compact_video_metadata(metadata: Mapping[str, Any] | None, logical_id: str) -> dict[str, Any]:
    value = dict(metadata or {})
    if "path" in value:
        value["path"] = logical_id
    return value


def _compact_comparison_summary(summary: Mapping[str, Any], output: Path, artifact_dir: Path, model_path: Path) -> dict[str, Any]:
    value = dict(summary)
    if "observation_dir" in value:
        value["observation_dir"] = _output_relative_path(artifact_dir, output)
    if "model_path" in value:
        value["model_path"] = _output_relative_path(model_path, output)
    return value


NUMERICAL_REFINEMENT_THRESHOLDS: dict[str, float] = {
    "onset_time_abs": 0.1,
    "peak_amplitude_relative": 0.2,
}


def _numerical_unresolved_assessment(results: Sequence[Mapping[str, Any]], video_case: str) -> dict[str, Any]:
    selected = next((result for result in results if result.get("run_name") == video_case), None)
    refinements = [
        result for result in results
        if result.get("run_kind") == "numerical_refinement" and result.get("base_fixture") == video_case
    ]
    reasons: set[str] = set()
    if selected is None:
        reasons.add("video_model_case_missing")
    else:
        classification = selected.get("classification", {})
        if selected.get("failure_reason") is not None:
            reasons.add("selected_model_failure")
        if classification.get("unresolved_reason_category") == "numerical_nonconvergence" or classification.get("label") == "numerically-unresolved":
            reasons.add("selected_model_numerically_unresolved")
    for result in refinements:
        classification = result.get("classification", {})
        if result.get("failure_reason") is not None:
            reasons.add("refinement_run_failure")
        if classification.get("unresolved_reason_category") == "numerical_nonconvergence" or classification.get("label") == "numerically-unresolved":
            reasons.add("refinement_numerically_unresolved")
    if selected is not None:
        baseline = selected.get("classification", {})
        baseline_onset = baseline.get("onset_time")
        baseline_peak = baseline.get("peak_max_transverse_amplitude")
        baseline_label = baseline.get("label")
        for result in refinements:
            classification = result.get("classification", {})
            refinement_onset = classification.get("onset_time")
            if (baseline_onset is None) != (refinement_onset is None) or (
                baseline_onset is not None
                and refinement_onset is not None
                and abs(float(baseline_onset) - float(refinement_onset)) > NUMERICAL_REFINEMENT_THRESHOLDS["onset_time_abs"]
            ):
                reasons.add("refinement_onset_disagreement")
            if baseline_label != classification.get("label"):
                reasons.add("refinement_classification_disagreement")
            refinement_peak = classification.get("peak_max_transverse_amplitude")
            if baseline_peak is not None and refinement_peak is not None:
                scale = max(abs(float(baseline_peak)), abs(float(refinement_peak)), 1.0e-15)
                if abs(float(baseline_peak) - float(refinement_peak)) / scale > NUMERICAL_REFINEMENT_THRESHOLDS["peak_amplitude_relative"]:
                    reasons.add("refinement_peak_disagreement")
    return {
        "status": "numerically_unresolved" if reasons else "numerically_resolved",
        "category": "numerical_nonconvergence" if reasons else None,
        "reasons": sorted(reasons),
        "model_run": video_case,
        "refinement_runs": [result.get("run_name") for result in refinements],
        "thresholds": dict(NUMERICAL_REFINEMENT_THRESHOLDS),
    }


def _model_inadequacy_assessment(
    rows: Sequence[Mapping[str, Any]],
    registration: RegistrationConfig,
    registration_explicit: bool,
    numerical_status: Mapping[str, Any],
) -> dict[str, Any]:
    thresholds = dict(MODEL_INADEQUACY_THRESHOLDS)
    conditions = [
        "registration.calibrated",
        "metric_status == computed",
        "shape_rmse_px > 5.0 or abs(length_difference_px) / model_length_px > 0.25",
    ]
    result: dict[str, Any] = {
        "status": "not_assessed_unregistered",
        "category": "model_inadequacy",
        "thresholds": thresholds,
        "conditions": conditions,
        "candidate_count": 0,
        "candidates": [],
        "scope": "eligible calibrated comparison rows only; input quality/censor and numerical status remain separate",
    }
    if numerical_status.get("status") == "numerically_unresolved":
        result["status"] = "not_assessed_numerical_unresolved"
        return result
    if not registration_explicit:
        return result
    if not registration.calibrated:
        result["status"] = "not_assessed_uncalibrated"
        return result
    eligible = [row for row in rows if row.get("metric_status") == "computed" and not row.get("censor")]
    if not eligible:
        result["status"] = "not_assessed_no_eligible_rows"
        return result
    candidates: list[dict[str, Any]] = []
    for row in eligible:
        reasons: list[str] = []
        shape_rmse = row.get("shape_rmse_px")
        if shape_rmse is not None and float(shape_rmse) > thresholds["shape_rmse_px"]:
            reasons.append("shape_rmse_px_above_threshold")
        model_length = row.get("model_length_px")
        length_difference = row.get("length_difference_px")
        relative_length_error = None
        if model_length is not None and length_difference is not None and float(model_length) > 0.0:
            relative_length_error = abs(float(length_difference)) / float(model_length)
            if relative_length_error > thresholds["relative_length_error"]:
                reasons.append("relative_length_error_above_threshold")
        if reasons:
            candidates.append({
                "frame": row.get("frame"),
                "time": row.get("time"),
                "filament_id": row.get("filament_id"),
                "shape_rmse_px": None if shape_rmse is None else float(shape_rmse),
                "relative_length_error": relative_length_error,
                "reasons": reasons,
            })
    result["status"] = "candidate" if candidates else "no_candidate"
    result["candidate_count"] = len(candidates)
    result["candidates"] = candidates[:64]
    result["candidate_rows_truncated"] = len(candidates) > 64
    return result


def _video_failure_category(exc: BaseException) -> tuple[str, str]:
    message = str(exc).lower()
    if "required executable is not installed" in message and ("ffmpeg" in message or "ffprobe" in message):
        return "missing_ffmpeg", "execution_environment"
    if any(token in message for token in ("unsupported format", "invalid format", "not a valid video")):
        return "invalid_format", "input_quality"
    if any(token in message for token in ("decode", "corrupt", "invalid data", "moov atom not found", "video metadata failed", "command failed")):
        return "decode_or_corrupt_input", "input_quality"
    return "analysis_failure", "analysis"


def _holdout_status(
    usable: bool,
    registration: RegistrationConfig,
    registration_explicit: bool,
    summary: Mapping[str, Any],
    reasons: Sequence[str],
    numerical_status: Mapping[str, Any],
) -> str:
    if numerical_status.get("status") == "numerically_unresolved":
        return "numerical_unresolved"
    if not registration_explicit:
        return "comparison_only_unregistered" if usable else "censored"
    if not registration.calibrated:
        return "comparison_only_uncalibrated" if usable else "censored"
    if int(summary.get("eligible_rows", 0)) > 0:
        return "calibrated_comparison"
    if not usable:
        return "calibrated_but_input_unusable"
    if reasons or int(summary.get("excluded_from_metric_denominator", 0)) > 0:
        return "calibrated_but_censored"
    return "calibrated_no_eligible_rows"


def _unregistered_comparison_summary(
    extraction: Mapping[str, Any],
    model_path: Path,
    registration: Mapping[str, Any],
    missing_fields: Sequence[str],
) -> dict[str, Any]:
    population_rows = len(extraction.get("summary_rows", []))
    return {
        "schema_version": SCHEMA_VERSION,
        "calibration_status": "registration_incomplete_metrics_suppressed",
        "registration": dict(registration),
        "registration_missing_fields": list(missing_fields),
        "model_logical_id": model_path.name,
        "rows": population_rows,
        "eligible_rows": 0,
        "computed_rows": 0,
        "censored_rows": population_rows,
        "population_rows": population_rows,
        "excluded_from_metric_denominator": population_rows,
        "excluded_from_metric_reasons": ["explicit_pixel_and_time_registration_required"],
        "comparison_performed": False,
    }


def run_video_comparison(
    video_path: str | Path,
    output: Path,
    model_path: Path,
    config: Mapping[str, Any],
    registration: Mapping[str, Any] | None = None,
    numerical_status: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Run the existing video pipeline and return a compact QC-only record.

    No registration is inferred.  When it is absent, the pipeline's observed
    length/curvature/endpoint and shape comparison rows remain available for
    review, but quantitative pixel/model metrics and parameter fitting stay
    censored.  Calibration and holdout are explicit separate records.
    """

    artifact_dir = output / "_video_artifacts"
    compact_path = output / "video_comparison_manifest.json"
    source = Path(video_path).expanduser()
    if not source.is_file():
        record = {
            "schema_version": SCHEMA_VERSION,
            "status": "input_missing",
            "input_logical_id": source.name,
            "input_sha256": None,
            "data_quality": {"usable": False, "censor": True, "reasons": ["input_missing"]},
            "calibration": {"status": "not_run_input_missing", "runs": []},
            "holdout": {"status": "not_run_input_missing", "runs": []},
            "model_inadequacy": {
                "status": "not_assessed_input_missing",
                "category": "model_inadequacy",
                "thresholds": dict(MODEL_INADEQUACY_THRESHOLDS),
                "candidates": [],
            },
            "numerical_unresolved": {
                "status": "not_assessed_input_missing",
                "category": None,
                "reasons": [],
            },
            "quantitative_fitting": "suppressed",
            "legacy_video_substitution": False,
        }
        _write_json(compact_path, record)
        return record
    try:
        video_cfg = dict(DEFAULT_VIDEO_CONFIG)
        video_cfg.update(dict(config.get("video", {})))
        extraction = run_pipeline(source, artifact_dir, SegmentationConfig.from_mapping(video_cfg), command_line=["stage2", "extract", "${INPUT_VIDEO}", "${OUTPUT_DIR}"])
        registration_mapping = dict(registration or {})
        required_registration_fields = ("pixel_per_model_unit", "time_scale", "time_offset")
        missing_registration_fields = [
            field for field in required_registration_fields
            if field not in registration_mapping or registration_mapping[field] is None
        ]
        registration_explicit = not missing_registration_fields
        registration_value = RegistrationConfig.from_mapping(registration_mapping)
        if registration_explicit:
            comparison = compare_with_model(artifact_dir, model_path, registration_value, output_dir=artifact_dir)
            comparison_summary = comparison["summary"]
            comparison_rows = comparison.get("rows", [])
        else:
            comparison = {"rows": []}
            comparison_summary = _unregistered_comparison_summary(
                extraction, model_path, registration_mapping, missing_registration_fields
            )
            comparison_rows = []
        video_manifest = extraction["manifest"]
        validation = extraction["validation"]
        usable = bool(validation.get("valid")) and bool(extraction.get("summary_rows"))
        reasons: list[str] = []
        if not validation.get("valid"):
            reasons.append("centerline_contract_invalid")
        if not extraction.get("summary_rows"):
            reasons.append("no_centerline_candidates")
        if int(video_manifest.get("candidate_censor_count", 0)) > 0:
            reasons.append("candidate_quality_censor_present")
        numerical_status_value = dict(numerical_status or {
            "status": "not_assessed_no_suite_context",
            "category": None,
            "reasons": [],
        })
        calibration_status = "configured_not_fitted" if registration_explicit else "not_registered_metrics_suppressed"
        model_inadequacy = _model_inadequacy_assessment(
            comparison_rows, registration_value, registration_explicit, numerical_status_value
        )
        record = {
            "schema_version": SCHEMA_VERSION,
            "status": "extracted" if usable else "unusable",
            "input_logical_id": source.name,
            "input_sha256": sha256_file(source),
            "input_bytes": source.stat().st_size,
            "video_metadata": _compact_video_metadata(
                video_manifest.get("input", {}).get("metadata"), source.name
            ),
            "extraction": {
                "processed_frames": video_manifest.get("processed_frames"),
                "candidate_count": video_manifest.get("candidate_count"),
                "candidate_censor_count": video_manifest.get("candidate_censor_count"),
                "selected_filament_id": video_manifest.get("selected_filament_id"),
                "validation": validation,
                "artifact_dir_external": True,
            },
            "data_quality": {"usable": usable, "censor": bool(reasons), "reasons": reasons},
            "calibration": {
                "status": calibration_status,
                "runs": [],
                "parameter_identification": "suppressed; registration is not an inferred fit",
            },
            "holdout": {
                "status": _holdout_status(
                    usable,
                    registration_value,
                    registration_explicit,
                    comparison_summary,
                    reasons,
                    numerical_status_value,
                ),
                "model_logical_id": model_path.name,
                "comparison": _compact_comparison_summary(
                    comparison_summary, output, artifact_dir, model_path
                ),
                "supported_observables": ["observed_length_px", "observed_endpoint_distance_px", "observed_curvature_mean_px_inv", "observed_curvature_max_px_inv", "temporal_frame_coverage"],
                "quantitative_model_overlay": (
                    "available_for_eligible_registered_rows"
                    if registration_explicit and int(comparison_summary.get("eligible_rows", 0)) > 0
                    else "suppressed_without_explicit_registration"
                ),
            },
            "model_inadequacy": model_inadequacy,
            "numerical_unresolved": numerical_status_value,
            "quantitative_fitting": "suppressed",
            "lineage_censor_preserved": True,
            "legacy_video_substitution": False,
        }
    except (OSError, RuntimeError, ValueError, KeyError) as exc:
        failure_category, failure_domain = _video_failure_category(exc)
        record = {
            "schema_version": SCHEMA_VERSION,
            "status": "unusable",
            "input_logical_id": source.name,
            "input_sha256": sha256_file(source),
            "data_quality": {
                "usable": False,
                "censor": True,
                "reasons": [failure_category] if failure_domain == "input_quality" else ["pipeline_failure"],
            },
            "failure": {"category": failure_category, "domain": failure_domain},
            "calibration": {"status": "not_run_pipeline_error", "runs": []},
            "holdout": {"status": "not_run_pipeline_error", "runs": []},
            "model_inadequacy": {
                "status": "not_assessed_pipeline_error",
                "category": "model_inadequacy",
                "thresholds": dict(MODEL_INADEQUACY_THRESHOLDS),
                "candidates": [],
            },
            "numerical_unresolved": {
                "status": "not_assessed_pipeline_error",
                "category": None,
                "reasons": [],
            },
            "quantitative_fitting": "suppressed",
            "legacy_video_substitution": False,
            "error": failure_category,
        }
    _write_json(compact_path, record)
    return record


def run_suite(config: Mapping[str, Any] | None, output: Path, *, video_path: str | Path | None = None, registration: Mapping[str, Any] | None = None) -> dict[str, Any]:
    effective = load_config(None) if config is None else load_config_from_mapping(config)
    output.mkdir(parents=True, exist_ok=True)
    revision = detect_git_revision(Path(__file__).resolve().parents[2])
    specs = _all_specs(effective)
    video_case = str(effective.get("video_model_case", "fast_growth_low_bend"))
    results: list[dict[str, Any]] = []
    for spec in specs:
        results.append(run_case(spec, effective["base"], output, revision, save_trajectory_file=bool(video_path and spec.name == video_case)))
    rows = [_summary_row(result) for result in results]
    _write_csv(output / "summary.csv", rows)
    deterministic = [row for row in rows if row["run_kind"] in {"deterministic_fixture", "numerical_refinement"}]
    contrasts = [row for row in rows if row["run_kind"] == "parameter_contrast"]
    replicates = [row for row in rows if row["run_kind"] == "exploratory_replicate"]
    summary = {
        "schema_version": SCHEMA_VERSION,
        "benchmark": "Stage 2 free/free growth-induced buckling exploration",
        "source_revision": revision,
        "config_sha256": hashlib.sha256(canonical_json_bytes(effective)).hexdigest(),
        "boundary": "free/free",
        "contact_enabled": False,
        "deterministic_fixture_count": len(deterministic),
        "exploratory_replicate_count": len(replicates),
        "parameter_contrast_count": len(contrasts),
        "controlled_contrast_parameters": ["growth_rate", "bending_stiffness", "amplitude", "n_nodes", "dt"],
        "dimensionless_group_definition": _dimensionless_groups(effective["base"])["definition"],
        "records": rows,
        "deterministic_fixtures": deterministic,
        "parameter_contrasts": contrasts,
        "exploratory_replicates": replicates,
        "separation_rule": "deterministic fixtures/refinement, parameter contrasts, and seeded exploratory replicates remain separate populations",
        "classification_scope": "onset and morphology labels only; no phase-boundary claim",
        "unresolved_categories": ["input_quality", "censor", "model_inadequacy", "numerical_nonconvergence"],
        "results": results,
        "video_comparison": None,
        "large_artifacts": "_runs and _video_artifacts are external-style artifacts; commit only compact summary/manifest files",
    }
    if video_path is not None:
        numerical_status = _numerical_unresolved_assessment(results, video_case)
        selected = next((result for result in results if result["run_name"] == video_case), None)
        trajectory_value = selected.get("trajectory_path") if selected is not None else None
        if not trajectory_value:
            raise Stage2Error(f"video_model_case has no saved trajectory: {video_case}")
        trajectory_path = Path(str(trajectory_value))
        if trajectory_path.is_absolute():
            raise Stage2Error("video trajectory path must be output-relative")
        video_record = run_video_comparison(
            video_path,
            output,
            output / trajectory_path,
            effective,
            registration,
            numerical_status=numerical_status,
        )
        video_record["source_revision"] = revision
        video_record["config_sha256"] = summary["config_sha256"]
        summary["video_comparison"] = video_record
        _write_json(output / "video_comparison_manifest.json", video_record)
    _write_json(output / "compact_summary.json", summary)
    _write_json(output / "effective_config.json", effective)
    artifact_paths = [output / "summary.csv", output / "compact_summary.json", output / "effective_config.json"]
    if video_path is not None:
        artifact_paths.append(output / "video_comparison_manifest.json")
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "benchmark": "stage2_free_free_growth_relaxation_buckling",
        "source_revision": revision,
        "config_sha256": summary["config_sha256"],
        "run_count": len(results),
        "run_names": [result["run_name"] for result in results],
        "deterministic_fixture_count": len(deterministic),
        "exploratory_replicate_count": len(replicates),
        "parameter_contrast_count": len(contrasts),
        "controlled_contrast_parameters": ["growth_rate", "bending_stiffness", "amplitude", "n_nodes", "dt"],
        "dimensionless_group_definition": _dimensionless_groups(effective["base"])["definition"],
        "contact_enabled": False,
        "phase_boundary_claim": False,
        "artifacts": {path.name: {"bytes": path.stat().st_size, "sha256": sha256_file(path)} for path in artifact_paths},
        "video_comparison": summary["video_comparison"],
    }
    _write_json(output / "compact_manifest.json", manifest)
    return summary


def load_config_from_mapping(value: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize and validate a configuration supplied by Python callers."""

    if not isinstance(value, Mapping):
        raise Stage2Error("Stage 2 config must be a mapping")
    return _validate_config(_merge_config(value))


def _cli() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--video", type=Path)
    parser.add_argument("--registration", type=Path)
    return parser.parse_args()


def main() -> int:
    args = _cli()
    config = load_config(args.config)
    registration = json.loads(args.registration.read_text(encoding="utf-8")) if args.registration else None
    run_suite(config, args.output, video_path=args.video, registration=registration)
    print(json.dumps({"output": str(args.output.resolve()), "video": bool(args.video)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
