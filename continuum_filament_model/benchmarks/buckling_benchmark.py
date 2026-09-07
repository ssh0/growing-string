"""Non-contact growth--relaxation and buckling benchmark runner.

This module is deliberately a thin experiment harness around the public
``growing_filament`` API.  It does not reimplement energy, force, growth, or
remeshing equations.  Results are intended as a reproducible mechanical
regime map (straight / buckled-single / unresolved), not as evidence of a
phase transition.

Run from the repository root, for example::

    PYTHONPATH=continuum_filament_model/src \
      python continuum_filament_model/benchmarks/buckling_benchmark.py \
      --output continuum_filament_model/results/p1b-smoke

The default suite is intentionally small.  Use ``--config`` to provide a
JSON object with ``base`` and ``cases`` keys; each case has a ``name`` and an
optional ``overrides`` object.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

# Permit the documented direct-file command as well as ``python -m`` from the
# repository root.  The benchmark still imports the installed-in-tree public
# library; it does not copy its physical equations.
if __package__ in {None, ""}:
    _HERE = Path(__file__).resolve()
    _SRC = _HERE.parents[1] / "src"
    if str(_SRC) not in sys.path:
        sys.path.insert(0, str(_SRC))

from growing_filament.model import (  # noqa: E402
    FilamentState,
    ModelError,
    ModelParameters,
    OverdampedGrowingFilament,
)
from growing_filament.observables import (  # noqa: E402
    arc_length_weighted_radius_of_gyration,
    contour_length,
    discrete_curvature,
    radius_of_gyration,
)
from growing_filament.reproducibility import (  # noqa: E402
    build_manifest,
    canonical_json_bytes,
    detect_git_revision,
)
from growing_filament.io import save_trajectory  # noqa: E402


SCHEMA_VERSION = "continuum-filament-p1b-buckling-1"


@dataclass(frozen=True)
class CaseSpec:
    """One deterministic benchmark case."""

    name: str
    overrides: dict[str, Any]


# These values are small enough for local validation while still separating
# the bending and growth time scales.  They are not fitted to experiment.
DEFAULT_BASE_CONFIG: dict[str, Any] = {
    "length": 2.0,
    "n_nodes": 9,
    "axial_stiffness": 100.0,
    "bending_stiffness": 0.1,
    "drag_density": 1.0,
    "growth_rate": 0.0,
    "amplitude": 0.02,
    "dt": 0.001,
    "t_end": 0.2,
    "a_max_factor": 2.0,
    "contact_stiffness": 0.0,
    "diameter": 0.0,
    "fixed_left": True,
    "fixed_right": True,
    "reject_crossing": True,
    "max_displacement_fraction": 0.25,
    "max_retries": 12,
    "dt_min": 1.0e-10,
}

DEFAULT_CASES: tuple[CaseSpec, ...] = (
    CaseSpec("growth_free_calibration", {"growth_rate": 0.0, "t_end": 0.1}),
    CaseSpec("slow_growth", {"growth_rate": 0.01, "t_end": 0.2}),
    CaseSpec("fast_growth", {"growth_rate": 0.2, "t_end": 0.2}),
    CaseSpec("low_EI", {"bending_stiffness": 0.05, "growth_rate": 0.2, "t_end": 0.2}),
    CaseSpec("high_EI", {"bending_stiffness": 0.2, "growth_rate": 0.2, "t_end": 0.2}),
    CaseSpec("small_perturbation", {"amplitude": 0.01, "growth_rate": 0.2, "t_end": 0.2}),
    CaseSpec("large_perturbation", {"amplitude": 0.04, "growth_rate": 0.2, "t_end": 0.2}),
)

DEFAULT_SENSITIVITY: tuple[CaseSpec, ...] = (
    CaseSpec("dt_half", {"dt_factor": 0.5, "growth_rate": 0.2, "t_end": 0.2}),
    CaseSpec("spatial_refined", {"n_nodes": 17, "growth_rate": 0.2, "t_end": 0.2}),
    CaseSpec("amplitude_half", {"amplitude": 0.01, "growth_rate": 0.2, "t_end": 0.2}),
)


class BenchmarkError(ValueError):
    """Invalid benchmark configuration."""


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
            return None
        return float(value)
    return value


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json_bytes(value) + b"\n")


def _as_float(config: Mapping[str, Any], key: str) -> float:
    try:
        return float(config[key])
    except (KeyError, TypeError, ValueError) as exc:
        raise BenchmarkError(f"{key} must be a finite number") from exc


def validate_case_config(config: Mapping[str, Any]) -> None:
    required = (
        "length",
        "n_nodes",
        "axial_stiffness",
        "bending_stiffness",
        "drag_density",
        "growth_rate",
        "amplitude",
        "dt",
        "t_end",
        "a_max_factor",
    )
    for key in required:
        if key not in config:
            raise BenchmarkError(f"missing configuration key: {key}")
        value = _as_float(config, key)
        if not math.isfinite(value):
            raise BenchmarkError(f"{key} must be finite")
    if int(config["n_nodes"]) != config["n_nodes"] or int(config["n_nodes"]) < 3:
        raise BenchmarkError("n_nodes must be an integer >= 3")
    for key in (
        "length",
        "axial_stiffness",
        "bending_stiffness",
        "drag_density",
        "amplitude",
        "dt",
        "t_end",
        "a_max_factor",
    ):
        if float(config[key]) <= 0.0:
            raise BenchmarkError(f"{key} must be positive")
    if float(config["growth_rate"]) < 0.0:
        raise BenchmarkError("growth_rate must be non-negative")
    if config.get("contact_stiffness", 0.0) != 0.0 or config.get("diameter", 0.0) != 0.0:
        raise BenchmarkError(
            "P1B non-contact benchmark requires contact_stiffness=0 and diameter=0"
        )
    if not bool(config.get("fixed_left", True)) or not bool(config.get("fixed_right", True)):
        raise BenchmarkError("P1B fixture requires both endpoints fixed")
    if not bool(config.get("reject_crossing", True)):
        raise BenchmarkError("P1B fixture requires reject_crossing=true")


def _effective_case(base: Mapping[str, Any], spec: CaseSpec) -> dict[str, Any]:
    config = dict(base)
    overrides = dict(spec.overrides)
    dt_factor = float(overrides.pop("dt_factor", 1.0))
    if dt_factor <= 0.0 or not math.isfinite(dt_factor):
        raise BenchmarkError(f"{spec.name}: dt_factor must be positive and finite")
    config.update(overrides)
    config["dt"] = float(config.get("dt", DEFAULT_BASE_CONFIG["dt"])) * dt_factor
    config["n_nodes"] = int(config["n_nodes"])
    config["contact_stiffness"] = 0.0
    config["diameter"] = 0.0
    config["fixed_left"] = True
    config["fixed_right"] = True
    config["reject_crossing"] = True
    validate_case_config(config)
    return config


def load_suite_config(path: Path | None) -> tuple[dict[str, Any], list[CaseSpec], list[CaseSpec]]:
    if path is None:
        return dict(DEFAULT_BASE_CONFIG), list(DEFAULT_CASES), list(DEFAULT_SENSITIVITY)
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        raise BenchmarkError("config root must be a JSON object")
    base = dict(DEFAULT_BASE_CONFIG)
    base.update(dict(raw.get("base", {})))

    def parse_specs(key: str, default: Sequence[CaseSpec]) -> list[CaseSpec]:
        values = raw.get(key, default)
        if values is None:
            return []
        specs: list[CaseSpec] = []
        for item in values:
            if not isinstance(item, Mapping) or "name" not in item:
                raise BenchmarkError(f"{key} entries require name and overrides")
            specs.append(CaseSpec(str(item["name"]), dict(item.get("overrides", {}))))
        return specs

    return base, parse_specs("cases", DEFAULT_CASES), parse_specs("sensitivity", DEFAULT_SENSITIVITY)


def initial_perturbed_state(config: Mapping[str, Any]) -> FilamentState:
    """Create ``y(x)=A sin(pi*x/L)`` with an unstressed initial polyline."""

    length = float(config["length"])
    n_nodes = int(config["n_nodes"])
    amplitude = float(config["amplitude"])
    x = np.linspace(0.0, length, n_nodes)
    y = amplitude * np.sin(np.pi * x / length)
    positions = np.column_stack((x, y))
    rest_lengths = np.linalg.norm(np.diff(positions, axis=0), axis=1)
    return FilamentState(positions, rest_lengths)


def _transverse_coordinates(state: FilamentState, anchors: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    chord = np.asarray(anchors[1] - anchors[0], dtype=float)
    chord_length = float(np.linalg.norm(chord))
    if chord_length <= 1.0e-12:
        raise BenchmarkError("fixed endpoint chord must be non-zero")
    tangent = chord / chord_length
    normal = np.array([-tangent[1], tangent[0]])
    relative = state.positions - anchors[0]
    return relative @ tangent, relative @ normal


def _mode_coefficients(state: FilamentState, anchors: np.ndarray, max_mode: int = 6) -> np.ndarray:
    """Project the piecewise-linear transverse shape onto sine modes."""

    _, transverse = _transverse_coordinates(state, anchors)
    edges = np.diff(state.positions, axis=0)
    lengths = np.linalg.norm(edges, axis=1)
    total = float(np.sum(lengths))
    if total <= 1.0e-12:
        return np.full(max_mode, np.nan)
    cumulative = np.concatenate(([0.0], np.cumsum(lengths))) / total
    # Five-point Gauss integration is deterministic and exact enough for the
    # smooth fixture; it also handles remeshed, variable-node trajectories.
    gauss_x, gauss_w = np.polynomial.legendre.leggauss(5)
    values = np.zeros(max_mode, dtype=float)
    for i, segment_length in enumerate(lengths):
        u0, u1 = cumulative[i], cumulative[i + 1]
        local_u = 0.5 * (u1 - u0) * gauss_x + 0.5 * (u1 + u0)
        local_y = transverse[i] + (transverse[i + 1] - transverse[i]) * (
            (local_u - u0) / max(u1 - u0, 1.0e-15)
        )
        for mode in range(1, max_mode + 1):
            values[mode - 1] += 0.5 * (u1 - u0) * float(
                np.sum(gauss_w * local_y * np.sin(mode * np.pi * local_u))
            )
    return 2.0 * values


def _metrics(state: FilamentState, simulator: OverdampedGrowingFilament, anchors: np.ndarray) -> dict[str, Any]:
    _, transverse = _transverse_coordinates(state, anchors)
    coefficients = _mode_coefficients(state, anchors)
    mode_norm = float(np.sqrt(np.sum(coefficients * coefficients)))
    curvature = discrete_curvature(state)
    components = simulator.energy_components(state.positions, state.rest_lengths)
    force = simulator.forces(state.positions, state.rest_lengths)
    reaction_left = (-force[0]).tolist() if simulator.parameters.fixed_left else None
    reaction_right = (-force[-1]).tolist() if simulator.parameters.fixed_right else None
    return {
        "time": float(state.time),
        "step": int(state.step),
        "n_nodes": int(state.n_nodes),
        "reference_length": float(np.sum(state.rest_lengths)),
        "contour_length": contour_length(state),
        "endpoint_distance": float(np.linalg.norm(state.positions[-1] - state.positions[0])),
        "radius_of_gyration": radius_of_gyration(state),
        "arc_length_weighted_radius_of_gyration": arc_length_weighted_radius_of_gyration(state),
        "max_transverse_displacement": float(np.max(np.abs(transverse))),
        "first_mode_amplitude": float(coefficients[0]),
        "first_mode_abs_amplitude": float(abs(coefficients[0])),
        "mode_norm": mode_norm,
        "first_mode_fraction": float(abs(coefficients[0]) / mode_norm) if mode_norm > 1.0e-15 else 1.0,
        "dominant_mode": int(np.argmax(np.abs(coefficients)) + 1) if np.isfinite(coefficients).all() else None,
        "max_curvature": float(np.max(curvature)) if len(curvature) else 0.0,
        "rms_curvature": float(np.sqrt(np.mean(curvature * curvature))) if len(curvature) else 0.0,
        "energy_total": float(sum(components.values())),
        "energy_stretch": float(components["stretch"]),
        "energy_bend": float(components["bend"]),
        "energy_contact": float(components["contact"]),
        "fixed_endpoint_reaction_left_x": None if reaction_left is None else float(reaction_left[0]),
        "fixed_endpoint_reaction_left_y": None if reaction_left is None else float(reaction_left[1]),
        "fixed_endpoint_reaction_right_x": None if reaction_right is None else float(reaction_right[0]),
        "fixed_endpoint_reaction_right_y": None if reaction_right is None else float(reaction_right[1]),
    }


def dimensionless_groups(config: Mapping[str, Any]) -> dict[str, float | str]:
    length = float(config["length"])
    ei = float(config["bending_stiffness"])
    ea = float(config["axial_stiffness"])
    zeta = float(config["drag_density"])
    growth = float(config["growth_rate"])
    tau_b = zeta * length**4 / (ei * np.pi**4)
    tau_s = zeta * length**2 / ea
    return {
        "length_scale": length,
        "tau_b": float(tau_b),
        "tau_s": float(tau_s),
        "G_b": float(growth * tau_b),
        "G_s": float(growth * tau_s),
        "bending_to_stretching": float(ei / (ea * length**2)),
        "mesh_ratio_initial_dx_over_L": float((length / (int(config["n_nodes"]) - 1)) / length),
        "dt_over_tau_b": float(float(config["dt"]) / tau_b),
        "diameter_over_L": 0.0,
        "contact_stiffness": 0.0,
        "definition": "tau_b=zeta*L^4/(EI*pi^4), first sine bending mode; tau_s=zeta*L^2/EA",
    }


def _classification(
    rows: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
    failure_reason: str | None,
) -> dict[str, Any]:
    if not rows or failure_reason is not None:
        return {
            "label": "unresolved",
            "failure_reason": failure_reason or "no accepted trajectory",
            "onset_time": None,
            "dominant_mode": None,
            "threshold": None,
        }
    initial_amp = abs(float(rows[0]["first_mode_abs_amplitude"]))
    length = float(config["length"])
    threshold = max(3.0 * initial_amp, 0.005 * length)
    candidates = [row for row in rows if float(row["max_transverse_displacement"]) > threshold]
    onset = candidates[0] if candidates else None
    peak = max(rows, key=lambda row: float(row["max_transverse_displacement"]))
    final = rows[-1]
    mode_dominant = int(peak["dominant_mode"]) if peak["dominant_mode"] is not None else None
    single_fraction = float(peak["first_mode_fraction"])
    if onset is None:
        label = "straight"
    elif mode_dominant == 1 and single_fraction >= 0.70:
        label = "buckled-single"
    else:
        label = "unresolved"
    return {
        "label": label,
        "failure_reason": None,
        "onset_time": None if onset is None else float(onset["time"]),
        "dominant_mode_at_peak": mode_dominant,
        "single_mode_fraction_at_peak": single_fraction,
        "threshold_max_transverse_displacement": threshold,
        "peak_max_transverse_displacement": float(peak["max_transverse_displacement"]),
        "peak_first_mode_amplitude": float(peak["first_mode_abs_amplitude"]),
        "final_max_transverse_displacement": float(final["max_transverse_displacement"]),
        "initial_first_mode_amplitude": initial_amp,
    }


def _write_metrics_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        path.write_text("\n", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(_jsonable(row) for row in rows)


def _plot_case(path: Path, rows: Sequence[Mapping[str, Any]], classification: Mapping[str, Any], config: Mapping[str, Any]) -> str:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - depends on optional local package
        return f"plot unavailable: {type(exc).__name__}: {exc}"
    if not rows:
        return "plot skipped: no trajectory rows"
    times = [row["time"] for row in rows]
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)
    axes[0, 0].plot(times, [row["max_transverse_displacement"] for row in rows], label="max |y|")
    axes[0, 0].plot(times, [row["first_mode_abs_amplitude"] for row in rows], label="|A1|")
    axes[0, 0].set(xlabel="time", ylabel="displacement", title="buckling indicators")
    axes[0, 0].legend()
    axes[0, 1].plot(times, [row["energy_total"] for row in rows], label="total")
    axes[0, 1].plot(times, [row["energy_bend"] for row in rows], label="bend")
    axes[0, 1].plot(times, [row["energy_stretch"] for row in rows], label="stretch")
    axes[0, 1].set(xlabel="time", ylabel="energy", title="energy")
    axes[0, 1].legend()
    axes[1, 0].plot(times, [row["max_curvature"] for row in rows])
    axes[1, 0].set(xlabel="time", ylabel="max curvature", title="curvature")
    axes[1, 1].plot(times, [row["endpoint_distance"] for row in rows], label="endpoint distance")
    axes[1, 1].plot(times, [row["reference_length"] for row in rows], label="reference length")
    axes[1, 1].set(xlabel="time", ylabel="length", title=str(classification.get("label")))
    axes[1, 1].legend()
    fig.suptitle(
        f"{config.get('name', 'case')} | G_b={dimensionless_groups(config)['G_b']:.4g}"
    )
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return "generated"


def run_case(case: CaseSpec, base_config: Mapping[str, Any], output_root: Path, git_revision: str | None) -> dict[str, Any]:
    config = _effective_case(base_config, case)
    config["name"] = case.name
    config["a_max"] = (float(config["length"]) / (int(config["n_nodes"]) - 1)) * float(config["a_max_factor"])
    params = ModelParameters(
        axial_stiffness=float(config["axial_stiffness"]),
        bending_stiffness=float(config["bending_stiffness"]),
        drag_density=float(config["drag_density"]),
        contact_stiffness=0.0,
        diameter=0.0,
        growth_rate=float(config["growth_rate"]),
        reference_length=float(config["length"]) / (int(config["n_nodes"]) - 1),
        dt=float(config["dt"]),
        t_end=float(config["t_end"]),
        a_max=float(config["a_max"]),
        dt_min=float(config.get("dt_min", 1.0e-10)),
        max_retries=int(config.get("max_retries", 12)),
        max_displacement_fraction=float(config.get("max_displacement_fraction", 0.25)),
        fixed_left=True,
        fixed_right=True,
        reject_crossing=True,
    )
    state = initial_perturbed_state(config)
    anchors = np.asarray([state.positions[0], state.positions[-1]], dtype=float)
    case_dir = output_root / case.name
    case_dir.mkdir(parents=True, exist_ok=True)
    effective = {
        "schema_version": SCHEMA_VERSION,
        "case": case.name,
        "requested_overrides": case.overrides,
        "effective_config": config,
        "physical_conditions": {
            "contact_stiffness": 0.0,
            "diameter": 0.0,
            "initial_geometry": "deterministic sine perturbation; non-crossing validated by library",
            "boundary": "both endpoint positions fixed",
        },
    }
    _write_json(case_dir / "config.json", effective)
    _write_json(case_dir / "dimensionless.json", dimensionless_groups(config))

    simulator: OverdampedGrowingFilament | None = None
    trajectory: list[FilamentState] = [state.copy()]
    rows: list[dict[str, Any]] = []
    failure_reason: str | None = None
    try:
        simulator = OverdampedGrowingFilament(state, params)
        trajectory = simulator.run()
        rows = [_metrics(item, simulator, anchors) for item in trajectory]
    except (ModelError, RuntimeError, ValueError, FloatingPointError) as exc:
        failure_reason = f"{type(exc).__name__}: {exc}"
        if simulator is not None:
            trajectory = [simulator.initial_state.copy()] + [simulator.state.copy()]
            rows = [_metrics(item, simulator, anchors) for item in trajectory]
        else:
            rows = []

    if simulator is not None:
        events = simulator.event_log
        final_state = simulator.state
    else:
        events = []
        final_state = state
    metadata = {
        "benchmark": "p1b_noncontact_growth_relaxation_buckling",
        "case": case.name,
        "classification_scope": "mechanical regime map; not a phase-transition claim",
        "physical_conditions": effective["physical_conditions"],
        "dimensionless_groups": dimensionless_groups(config),
        "failure_reason": failure_reason,
        "fixed_endpoint_reaction_limit": (
            "Reported values are the negative of the library's unconstrained endpoint force. "
            "The API has no separate reaction solver; values are a diagnostic proxy, not a new force law."
        ),
    }
    manifest = build_manifest(
        params,
        state,
        final_state=final_state,
        events=events,
        metadata=metadata,
        input_data=effective,
        git_revision=git_revision,
    )
    classification = _classification(rows, config, failure_reason)
    rejection_counts = dict(Counter(str(event.get("reason")) for event in events if event.get("accepted") is False))
    summary = {
        "schema_version": SCHEMA_VERSION,
        "case": case.name,
        "effective_config": config,
        "dimensionless_groups": dimensionless_groups(config),
        "classification": classification,
        "failure_reason": failure_reason,
        "accepted_steps": int(simulator.accepted_steps) if simulator else 0,
        "rejected_steps": int(simulator.rejected_steps) if simulator else 0,
        "rejection_reason_counts": rejection_counts,
        "event_count": len(events),
        "contact_enabled": False,
        "initial_noncontact_required": True,
        "fixed_endpoint_reaction": metadata["fixed_endpoint_reaction_limit"],
        "initial_observables": rows[0] if rows else None,
        "final_observables": rows[-1] if rows else None,
        "peak_observables": max(rows, key=lambda row: row["max_transverse_displacement"]) if rows else None,
    }
    _write_metrics_csv(case_dir / "metrics.csv", rows)
    _write_json(case_dir / "events.json", events)
    _write_json(case_dir / "manifest.json", manifest)
    _write_json(case_dir / "summary.json", summary)
    save_trajectory(
        case_dir / "trajectory.npz",
        trajectory,
        params,
        metadata=metadata,
        events=events,
        manifest=manifest,
        input_data=effective,
    )
    summary["plot"] = _plot_case(case_dir / "overview.png", rows, classification, config)
    _write_json(case_dir / "summary.json", summary)
    return summary


def write_suite_csv(path: Path, summaries: Sequence[Mapping[str, Any]]) -> None:
    fields = [
        "case",
        "label",
        "growth_rate",
        "bending_stiffness",
        "amplitude",
        "n_nodes",
        "dt",
        "G_b",
        "G_s",
        "onset_time",
        "peak_max_transverse_displacement",
        "peak_first_mode_amplitude",
        "dominant_mode_at_peak",
        "single_mode_fraction_at_peak",
        "accepted_steps",
        "rejected_steps",
        "failure_reason",
    ]
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for summary in summaries:
            config = summary["effective_config"]
            groups = summary["dimensionless_groups"]
            classification = summary["classification"]
            writer.writerow(
                {
                    "case": summary["case"],
                    "label": classification.get("label"),
                    "growth_rate": config["growth_rate"],
                    "bending_stiffness": config["bending_stiffness"],
                    "amplitude": config["amplitude"],
                    "n_nodes": config["n_nodes"],
                    "dt": config["dt"],
                    "G_b": groups["G_b"],
                    "G_s": groups["G_s"],
                    "onset_time": classification.get("onset_time"),
                    "peak_max_transverse_displacement": classification.get("peak_max_transverse_displacement"),
                    "peak_first_mode_amplitude": classification.get("peak_first_mode_amplitude"),
                    "dominant_mode_at_peak": classification.get("dominant_mode_at_peak"),
                    "single_mode_fraction_at_peak": classification.get("single_mode_fraction_at_peak"),
                    "accepted_steps": summary["accepted_steps"],
                    "rejected_steps": summary["rejected_steps"],
                    "failure_reason": summary["failure_reason"],
                }
            )


def run_suite(
    base_config: Mapping[str, Any],
    cases: Sequence[CaseSpec],
    sensitivities: Sequence[CaseSpec],
    output: Path,
    selected_case: str | None = None,
    include_sensitivity: bool = True,
) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    revision = detect_git_revision(Path.cwd())
    selected = [case for case in cases if selected_case is None or case.name == selected_case]
    if selected_case is not None and not selected:
        raise BenchmarkError(f"unknown case: {selected_case}")
    if include_sensitivity and selected_case is None:
        selected = selected + list(sensitivities)
    summaries = [run_case(case, base_config, output, revision) for case in selected]
    write_suite_csv(output / "summary.csv", summaries)
    suite = {
        "schema_version": SCHEMA_VERSION,
        "benchmark": "P1B non-contact growth-relaxation buckling",
        "git_revision": revision,
        "contact_enabled": False,
        "diameter": 0.0,
        "initial_noncontact_required": True,
        "classification": "straight / buckled-single / unresolved mechanical regimes; no phase-transition claim",
        "tau_b_definition": "drag_density * length_scale^4 / (bending_stiffness * pi^4)",
        "cases": [summary["case"] for summary in summaries],
        "results": summaries,
        "unresolved_items": [
            "segment contact force, adhesion, friction, and folding are not implemented",
            "experimental fitting and triangular-lattice comparison are out of scope",
            "fixed-end reactions are force diagnostics because the library has no reaction API",
            "no universality or phase-transition claim is made",
        ],
    }
    _write_json(output / "suite.json", suite)
    return suite


def _cli() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, help="JSON suite config")
    parser.add_argument("--output", type=Path, required=True, help="output directory")
    parser.add_argument("--case", help="run one named case without sensitivities")
    parser.add_argument("--no-sensitivity", action="store_true", help="omit dt/space/amplitude sensitivity cases")
    parser.add_argument("--growth-rate", type=float, help="override growth_rate for all selected cases")
    parser.add_argument("--bending-stiffness", type=float, help="override bending_stiffness for all selected cases")
    parser.add_argument("--dt", type=float, help="override dt for all selected cases")
    parser.add_argument("--amplitude", type=float, help="override perturbation amplitude for all selected cases")
    parser.add_argument("--n-nodes", type=int, help="override spatial resolution for all selected cases")
    return parser.parse_args()


def main() -> int:
    args = _cli()
    try:
        base, cases, sensitivities = load_suite_config(args.config)
        cli_overrides = {
            key: value
            for key, value in {
                "growth_rate": args.growth_rate,
                "bending_stiffness": args.bending_stiffness,
                "dt": args.dt,
                "amplitude": args.amplitude,
                "n_nodes": args.n_nodes,
            }.items()
            if value is not None
        }
        base.update(cli_overrides)
        suite = run_suite(
            base,
            cases,
            sensitivities,
            args.output,
            selected_case=args.case,
            include_sensitivity=not args.no_sensitivity,
        )
    except (BenchmarkError, OSError, json.JSONDecodeError) as exc:
        print(f"benchmark configuration/output error: {exc}", file=sys.stderr)
        return 2
    failures = [
        result for result in suite["results"] if result.get("failure_reason") is not None
    ]
    print(json.dumps({"output": str(args.output), "cases": len(suite["results"]), "failures": len(failures)}, sort_keys=True))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
