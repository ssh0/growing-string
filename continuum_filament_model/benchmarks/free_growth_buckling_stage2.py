"""Stage 2 free/free growth--relaxation--buckling exploration.

This runner is intentionally an experiment harness around the existing
``OverdampedGrowingFilament`` API.  It does not add contact, friction, or a
second solver.  The default design varies growth rate, bending/axial ratio,
initial imperfection, time step, spatial resolution, and seeded initial
imperfections.  Deterministic fixtures and exploratory replicates are kept in
separate tables; labels are morphology observations, not phase boundaries.

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
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

if __package__ in {None, ""}:  # pragma: no cover - direct-file entry point
    _HERE = Path(__file__).resolve()
    _SRC = _HERE.parents[1] / "src"
    if str(_SRC) not in sys.path:
        sys.path.insert(0, str(_SRC))

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
    "output_policy": {
        "save_trajectories": False,
        "max_output_bytes": 120_000_000,
    },
}


class Stage2Error(ValueError):
    """Invalid Stage 2 configuration or result."""


@dataclass(frozen=True)
class RunSpec:
    name: str
    kind: str
    overrides: dict[str, Any]
    seed: int | None = None
    trial: int = 0
    base_fixture: str | None = None


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
    if not fixtures:
        raise Stage2Error("at least one deterministic fixture is required")
    names = [str(item.get("name")) for item in fixtures if isinstance(item, Mapping) and item.get("name")]
    if len(names) != len(set(names)):
        raise Stage2Error("fixture names must be unique")
    if len(names) != len(fixtures):
        raise Stage2Error("every fixture requires a name")
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


def _all_specs(config: Mapping[str, Any]) -> list[RunSpec]:
    fixtures = _fixture_specs(config)
    result = list(fixtures)
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
    if seed is not None and noise_fraction:
        rng = np.random.default_rng(int(seed))
        noise = rng.normal(size=n_nodes - 2)
        noise /= max(float(np.std(noise)), 1.0e-15)
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
    forces = model.forces(state.positions, state.rest_lengths)
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


def run_case(spec: RunSpec, base_config: Mapping[str, Any], output: Path, revision: str | None, save_trajectory_file: bool = False) -> dict[str, Any]:
    config = _effective(base_config, spec)
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
            trajectory = [simulator.initial_state.copy(), simulator.state.copy()]
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
        "seed": spec.seed,
        "trial": spec.trial,
        "effective_config": config,
        "boundary": "free/free",
        "contact_enabled": False,
        "failure_reason": failure_reason,
        "classification": classification,
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
        result["trajectory_path"] = str((run_dir / "trajectory.npz").resolve())
        _write_json(run_dir / "summary.json", result)
    return result


def _summary_row(result: Mapping[str, Any]) -> dict[str, Any]:
    config = result["effective_config"]
    groups = {
        "tau_b": float(config["drag_density"] * config["length"] ** 4 / (config["bending_stiffness"] * np.pi**4)),
        "growth_bending_number": float(config["growth_rate"] * config["drag_density"] * config["length"] ** 4 / (config["bending_stiffness"] * np.pi**4)),
        "bending_to_axial_ratio": float(config["bending_stiffness"] / (config["axial_stiffness"] * config["length"] ** 2)),
    }
    classification = result["classification"]
    peak = max(result["observables"], key=lambda row: float(row["max_transverse_amplitude"])) if result["observables"] else {}
    final = result["observables"][-1] if result["observables"] else {}
    return {
        "run_name": result["run_name"],
        "run_kind": result["run_kind"],
        "base_fixture": result["base_fixture"],
        "seed": result["seed"],
        "trial": result["trial"],
        "growth_rate": config.get("growth_rate"),
        "bending_stiffness": config.get("bending_stiffness"),
        "axial_stiffness": config.get("axial_stiffness"),
        "bending_to_axial_ratio": groups["bending_to_axial_ratio"],
        "growth_bending_number": groups["growth_bending_number"],
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


def run_video_comparison(video_path: str | Path, output: Path, model_path: Path, config: Mapping[str, Any], registration: Mapping[str, Any] | None = None) -> dict[str, Any]:
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
            "quantitative_fitting": "suppressed",
            "legacy_video_substitution": False,
        }
        _write_json(compact_path, record)
        return record
    try:
        video_cfg = dict(DEFAULT_VIDEO_CONFIG)
        video_cfg.update(dict(config.get("video", {})))
        extraction = run_pipeline(source, artifact_dir, SegmentationConfig.from_mapping(video_cfg), command_line=["stage2", "extract", "${INPUT_VIDEO}", "${OUTPUT_DIR}"])
        registration_value = RegistrationConfig.from_mapping(registration)
        comparison = compare_with_model(artifact_dir, model_path, registration_value, output_dir=artifact_dir)
        video_manifest = extraction["manifest"]
        validation = extraction["validation"]
        comparison_summary = comparison["summary"]
        usable = bool(validation.get("valid")) and bool(extraction.get("summary_rows"))
        reasons: list[str] = []
        if not validation.get("valid"):
            reasons.append("centerline_contract_invalid")
        if not extraction.get("summary_rows"):
            reasons.append("no_centerline_candidates")
        if int(video_manifest.get("candidate_censor_count", 0)) > 0:
            reasons.append("candidate_quality_censor_present")
        calibration_status = "configured_not_fitted" if registration_value.calibrated else "not_available_metrics_suppressed"
        record = {
            "schema_version": SCHEMA_VERSION,
            "status": "extracted" if usable else "unusable",
            "input_logical_id": source.name,
            "input_sha256": sha256_file(source),
            "input_bytes": source.stat().st_size,
            "video_metadata": video_manifest.get("input", {}).get("metadata"),
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
                "status": "comparison_only_uncalibrated" if usable else "censored",
                "model_logical_id": model_path.name,
                "comparison": comparison_summary,
                "supported_observables": ["observed_length_px", "observed_endpoint_distance_px", "observed_curvature_mean_px_inv", "observed_curvature_max_px_inv", "temporal_frame_coverage"],
                "quantitative_model_overlay": "suppressed_without_pixel_per_model_unit_or_uncensored_centerline",
            },
            "quantitative_fitting": "suppressed",
            "lineage_censor_preserved": True,
            "legacy_video_substitution": False,
        }
    except (OSError, RuntimeError, ValueError, KeyError) as exc:
        record = {
            "schema_version": SCHEMA_VERSION,
            "status": "unusable",
            "input_logical_id": source.name,
            "input_sha256": sha256_file(source),
            "data_quality": {"usable": False, "censor": True, "reasons": [f"pipeline_error:{type(exc).__name__}"]},
            "calibration": {"status": "not_run_pipeline_error", "runs": []},
            "holdout": {"status": "not_run_pipeline_error", "runs": []},
            "quantitative_fitting": "suppressed",
            "legacy_video_substitution": False,
            "error": str(exc),
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
    deterministic = [row for row in rows if row["run_kind"] != "exploratory_replicate"]
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
        "records": rows,
        "deterministic_fixtures": deterministic,
        "exploratory_replicates": replicates,
        "separation_rule": "deterministic_fixture and numerical_refinement rows are never pooled with seeded exploratory_replicate rows",
        "classification_scope": "onset and morphology labels only; no phase-boundary claim",
        "unresolved_categories": ["input_quality", "censor", "model_inadequacy", "numerical_nonconvergence"],
        "results": results,
        "video_comparison": None,
        "large_artifacts": "_runs and _video_artifacts are external-style artifacts; commit only compact summary/manifest files",
    }
    if video_path is not None:
        selected = next((result for result in results if result["run_name"] == video_case), None)
        if selected is None or not selected.get("trajectory_path"):
            raise Stage2Error(f"video_model_case has no saved trajectory: {video_case}")
        summary["video_comparison"] = run_video_comparison(video_path, output, Path(selected["trajectory_path"]), effective, registration)
    _write_json(output / "compact_summary.json", summary)
    _write_json(output / "effective_config.json", effective)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "benchmark": "stage2_free_free_growth_relaxation_buckling",
        "source_revision": revision,
        "config_sha256": summary["config_sha256"],
        "run_count": len(results),
        "run_names": [result["run_name"] for result in results],
        "deterministic_fixture_count": len(deterministic),
        "exploratory_replicate_count": len(replicates),
        "contact_enabled": False,
        "phase_boundary_claim": False,
        "artifacts": {path.name: {"bytes": path.stat().st_size, "sha256": sha256_file(path)} for path in (output / "summary.csv", output / "compact_summary.json", output / "effective_config.json")},
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
