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


SCHEMA_VERSION = "continuum-filament-p1b-buckling-2"


@dataclass(frozen=True)
class CaseSpec:
    """One benchmark case, optionally one seeded stochastic trial."""

    name: str
    overrides: dict[str, Any]
    seed: int | None = None
    trial: int = 0
    base_name: str | None = None


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
    # Seeded trials add a small initial imperfection around the deterministic
    # sine fixture.  This is an input perturbation, not a new force law.
    "trial_noise_fraction": 0.05,
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

DEFAULT_TRIAL_DESIGN: dict[str, Any] = {
    "cases": ["slow_growth", "fast_growth", "high_EI"],
    "seeds": [11, 22, 33],
    "noise_fraction": 0.05,
}


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
    noise_fraction = float(config.get("trial_noise_fraction", 0.0))
    if not math.isfinite(noise_fraction) or not 0.0 <= noise_fraction <= 0.5:
        raise BenchmarkError("trial_noise_fraction must be finite and in [0, 0.5]")
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
    config["seed"] = spec.seed
    config["trial"] = int(spec.trial)
    config["base_name"] = spec.base_name or spec.name
    config["initial_perturbation_mode"] = (
        "sine_plus_seeded_node_noise" if spec.seed is not None else "deterministic_sine"
    )
    validate_case_config(config)
    return config


def load_suite_config(path: Path | None) -> tuple[
    dict[str, Any], list[CaseSpec], list[CaseSpec], dict[str, Any]
]:
    if path is None:
        return (
            dict(DEFAULT_BASE_CONFIG),
            list(DEFAULT_CASES),
            list(DEFAULT_SENSITIVITY),
            dict(DEFAULT_TRIAL_DESIGN),
        )
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
            specs.append(
                CaseSpec(
                    str(item["name"]),
                    dict(item.get("overrides", {})),
                    seed=(None if item.get("seed") is None else int(item["seed"])),
                    trial=int(item.get("trial", 0)),
                    base_name=(None if item.get("base_name") is None else str(item["base_name"])),
                )
            )
        return specs

    trial_design = dict(DEFAULT_TRIAL_DESIGN)
    trial_design.update(dict(raw.get("trial_design", {})))
    return (
        base,
        parse_specs("cases", DEFAULT_CASES),
        parse_specs("sensitivity", DEFAULT_SENSITIVITY),
        trial_design,
    )


def initial_perturbed_state(config: Mapping[str, Any], seed: int | None = None) -> FilamentState:
    """Create a deterministic sine fixture or a seeded small imperfection."""

    length = float(config["length"])
    n_nodes = int(config["n_nodes"])
    amplitude = float(config["amplitude"])
    x = np.linspace(0.0, length, n_nodes)
    y = amplitude * np.sin(np.pi * x / length)
    if seed is not None:
        rng = np.random.default_rng(int(seed))
        noise = rng.normal(size=n_nodes - 2)
        scale = float(config.get("trial_noise_fraction", 0.0))
        noise_scale = max(float(np.std(noise)), 1.0e-15)
        y[1:-1] += amplitude * scale * noise / noise_scale
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
    chi = float(ei / (ea * length**2))
    return {
        "length_scale": length,
        "tau_b": float(tau_b),
        "tau_s": float(tau_s),
        "G_b": float(growth * tau_b),
        "G_s": float(growth * tau_s),
        # ``chi`` is the P1B.2 map coordinate.  Keep the historical name as
        # an alias so P1B.1 consumers remain compatible.
        "chi": chi,
        "bending_to_stretching": chi,
        "mesh_ratio_initial_dx_over_L": float((length / (int(config["n_nodes"]) - 1)) / length),
        "dt_over_tau_b": float(float(config["dt"]) / tau_b),
        "diameter_over_L": 0.0,
        "contact_stiffness": 0.0,
        "definition": "tau_b=zeta*L^4/(EI*pi^4), first sine bending mode; tau_s=zeta*L^2/EA; chi=EI/(EA*L^2)",
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


def _compact_event_log(events: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Keep rejection diagnostics and a count summary without a trajectory.

    Full events contain repeated geometry diagnostics for every accepted and
    rejected attempt.  P1B.2 needs the rejection reasons and the manifest's
    deterministic event fingerprint, not a second copy of every accepted
    state.  Accepted-step counts are retained in the synthetic summary event
    and are also written to the manifest from the simulator counters.
    """

    result: list[dict[str, Any]] = []
    rejection_counts: Counter[str] = Counter()
    accepted_steps = 0
    rejected_steps = 0
    for event in events:
        event_type = event.get("event_type")
        accepted = event.get("accepted")
        if event_type == "step_attempt" and accepted is True:
            accepted_steps += 1
            continue
        if event_type == "step_attempt" and accepted is False:
            rejected_steps += 1
            rejection_counts[str(event.get("reason"))] += 1
            continue
        if event_type != "initialization":
            continue
        compact: dict[str, Any] = {
            key: event.get(key)
            for key in (
                "event_type",
                "reason",
                "accepted",
                "time_before",
                "time_after",
                "step_before",
                "step_after",
                "energy_trial",
                "energy_after",
            )
            if key in event
        }
        if event.get("detail") is not None:
            compact["detail"] = str(event["detail"])[:240]
        result.append(compact)
    result.append({
        "event_type": "step_attempt_summary",
        "accepted_steps": accepted_steps,
        "rejected_steps": rejected_steps,
        "rejection_reason_counts": dict(sorted(rejection_counts.items())),
    })
    return result


def _sample_metrics(rows: Sequence[Mapping[str, Any]], maximum: int | None) -> list[Mapping[str, Any]]:
    """Downsample only the stored time-series CSV; classification uses all rows."""

    if maximum is None or maximum <= 0 or len(rows) <= maximum:
        return list(rows)
    indices = set(np.linspace(0, len(rows) - 1, maximum, dtype=int).tolist())
    peak_index = max(range(len(rows)), key=lambda index: float(rows[index]["max_transverse_displacement"]))
    indices.add(peak_index)
    return [rows[index] for index in sorted(indices)]


def _write_metrics_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        path.write_text("\n", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=fieldnames,
            extrasaction="ignore",
            lineterminator="\n",
        )
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


def run_case(
    case: CaseSpec,
    base_config: Mapping[str, Any],
    output_root: Path,
    git_revision: str | None,
    *,
    save_trajectory_file: bool = True,
    write_plot: bool = True,
    compact_events: bool = False,
    metrics_max_rows: int | None = None,
) -> dict[str, Any]:
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
    state = initial_perturbed_state(config, seed=case.seed)
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
            "initial_geometry": (
                "deterministic sine perturbation; non-crossing validated by library"
                if case.seed is None
                else "sine perturbation plus seeded small node noise; non-crossing validated by library"
            ),
            "boundary": "both endpoint positions fixed",
        },
        "trial_design": {
            "base_case": case.base_name or case.name,
            "trial": int(case.trial),
            "seed": case.seed,
            "noise_fraction": float(config.get("trial_noise_fraction", 0.0)),
            "distribution": config.get(
                "trial_noise_distribution",
                "numpy.default_rng(seed).normal(0, 1) on interior nodes, sample-standardized",
            ),
            "amplitude_scale": float(config.get("amplitude", 0.0)) * float(config.get("trial_noise_fraction", 0.0)),
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
        raw_events = simulator.event_log
        events = _compact_event_log(raw_events) if compact_events else raw_events
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
        "trial_design": effective["trial_design"],
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
    # Keep trial identity easy to query without discarding the generic
    # manifest metadata used by the existing reproducibility API.
    manifest["seed"] = case.seed
    manifest["trial"] = int(case.trial)
    manifest["base_case"] = case.base_name or case.name
    if compact_events and simulator is not None:
        manifest["accepted_steps"] = int(simulator.accepted_steps)
        manifest["rejected_steps"] = int(simulator.rejected_steps)
        manifest["event_log_compaction"] = "rejection_attempts_plus_step_summary"
    classification = _classification(rows, config, failure_reason)
    rejection_counts = dict(Counter(str(event.get("reason")) for event in events if event.get("accepted") is False))
    summary = {
        "schema_version": SCHEMA_VERSION,
        "case": case.name,
        "base_case": case.base_name or case.name,
        "trial": int(case.trial),
        "seed": case.seed,
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
    stored_rows = _sample_metrics(rows, metrics_max_rows)
    _write_metrics_csv(case_dir / "metrics.csv", stored_rows)
    _write_json(case_dir / "events.json", events)
    _write_json(case_dir / "manifest.json", manifest)
    _write_json(case_dir / "summary.json", summary)
    if save_trajectory_file:
        save_trajectory(
            case_dir / "trajectory.npz",
            trajectory,
            params,
            metadata=metadata,
            events=events,
            manifest=manifest,
            input_data=effective,
        )
    summary["trajectory_saved"] = bool(save_trajectory_file)
    summary["plot"] = (
        _plot_case(case_dir / "overview.png", rows, classification, config)
        if write_plot
        else "skipped by compact experiment output policy"
    )
    summary["events_compacted"] = bool(compact_events)
    summary["metrics_rows_total"] = len(rows)
    summary["metrics_rows_saved"] = len(stored_rows)
    _write_json(case_dir / "summary.json", summary)
    return summary


def write_suite_csv(path: Path, summaries: Sequence[Mapping[str, Any]]) -> None:
    fields = [
        "case",
        "base_case",
        "trial",
        "seed",
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
        writer = csv.DictWriter(
            stream,
            fieldnames=fields,
            extrasaction="ignore",
            lineterminator="\n",
        )
        writer.writeheader()
        for summary in summaries:
            config = summary["effective_config"]
            groups = summary["dimensionless_groups"]
            classification = summary["classification"]
            writer.writerow(
                {
                    "case": summary["case"],
                    "base_case": summary.get("base_case", summary["case"]),
                    "trial": summary.get("trial", 0),
                    "seed": summary.get("seed"),
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


def _mean_std(values: Sequence[float]) -> tuple[float | None, float | None]:
    if not values:
        return None, None
    array = np.asarray(values, dtype=float)
    return float(np.mean(array)), float(np.std(array, ddof=1)) if len(array) > 1 else 0.0


def trial_summary(summaries: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Aggregate seeded trials without mixing them with deterministic fixtures."""

    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for summary in summaries:
        if int(summary.get("trial", 0)) > 0:
            grouped.setdefault(str(summary["base_case"]), []).append(summary)
    results: list[dict[str, Any]] = []
    for base_case, values in grouped.items():
        labels = Counter(str(item["classification"].get("label")) for item in values)
        onsets = [
            float(item["classification"]["onset_time"])
            for item in values
            if item["classification"].get("onset_time") is not None
        ]
        peaks = [
            float(item["classification"]["peak_max_transverse_displacement"])
            for item in values
            if item["classification"].get("peak_max_transverse_displacement") is not None
        ]
        onset_mean, onset_std = _mean_std(onsets)
        peak_mean, peak_std = _mean_std(peaks)
        representative = values[0]
        results.append(
            {
                "base_case": base_case,
                "n_trials": len(values),
                "seeds": [item.get("seed") for item in values],
                "label_counts": dict(sorted(labels.items())),
                "onset_time_mean": onset_mean,
                "onset_time_std": onset_std,
                "peak_max_transverse_mean": peak_mean,
                "peak_max_transverse_std": peak_std,
                "G_b": representative["dimensionless_groups"]["G_b"],
                "G_s": representative["dimensionless_groups"]["G_s"],
                "bending_stiffness": representative["effective_config"]["bending_stiffness"],
                "failure_count": sum(1 for item in values if item.get("failure_reason")),
            }
        )
    return results


def question_comparison_rows(
    summaries: Sequence[Mapping[str, Any]],
    trials: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Create machine-readable evidence tables for each research question."""

    deterministic = {
        str(item["case"]): item for item in summaries if int(item.get("trial", 0)) == 0
    }
    trial_by_case = {str(item["base_case"]): item for item in trials}
    rows: list[dict[str, Any]] = []

    def add_deterministic(question: str, comparison: str, names: Sequence[str]) -> None:
        for name in names:
            item = deterministic.get(name)
            if item is None:
                continue
            classification = item["classification"]
            rows.append(
                {
                    "question": question,
                    "comparison": comparison,
                    "case": name,
                    "source": "deterministic_fixture",
                    "label_or_counts": classification.get("label"),
                    "G_b": item["dimensionless_groups"]["G_b"],
                    "G_s": item["dimensionless_groups"]["G_s"],
                    "bending_stiffness": item["effective_config"]["bending_stiffness"],
                    "n_trials": 1,
                    "onset_time_mean": classification.get("onset_time"),
                    "onset_time_std": 0.0,
                    "peak_max_transverse_mean": classification.get("peak_max_transverse_displacement"),
                    "peak_max_transverse_std": 0.0,
                    "failure_count": int(item.get("failure_reason") is not None),
                }
            )

    def add_trial(question: str, comparison: str, names: Sequence[str]) -> None:
        for name in names:
            item = trial_by_case.get(name)
            if item is None:
                continue
            rows.append(
                {
                    "question": question,
                    "comparison": comparison,
                    "case": name,
                    "source": "seeded_trials",
                    "label_or_counts": json.dumps(item["label_counts"], sort_keys=True),
                    "G_b": item["G_b"],
                    "G_s": item["G_s"],
                    "bending_stiffness": item["bending_stiffness"],
                    "n_trials": item["n_trials"],
                    "onset_time_mean": item["onset_time_mean"],
                    "onset_time_std": item["onset_time_std"],
                    "peak_max_transverse_mean": item["peak_max_transverse_mean"],
                    "peak_max_transverse_std": item["peak_max_transverse_std"],
                    "failure_count": item["failure_count"],
                }
            )

    add_deterministic(
        "growth_rate_vs_bending_relaxation",
        "G_b growth comparison",
        ("growth_free_calibration", "slow_growth", "fast_growth"),
    )
    add_deterministic(
        "rigidity_and_initial_amplitude",
        "EI and deterministic perturbation comparison",
        ("low_EI", "high_EI", "small_perturbation", "large_perturbation"),
    )
    add_deterministic(
        "numerical_sensitivity",
        "dt / spatial resolution / amplitude sensitivity",
        ("dt_half", "spatial_refined", "amplitude_half"),
    )
    add_trial(
        "seeded_variability",
        "seeded initial-imperfection trial distribution",
        tuple(trial_by_case),
    )
    return rows


def write_question_comparison_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fields = [
        "question", "comparison", "case", "source", "label_or_counts", "G_b", "G_s",
        "bending_stiffness", "n_trials", "onset_time_mean", "onset_time_std",
        "peak_max_transverse_mean", "peak_max_transverse_std", "failure_count",
    ]
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=fields,
            extrasaction="ignore",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(_jsonable(row) for row in rows)


def _plot_question_comparison(path: Path, rows: Sequence[Mapping[str, Any]]) -> str:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - optional local dependency
        return f"plot unavailable: {type(exc).__name__}: {exc}"
    if not rows:
        return "plot skipped: no comparison rows"
    colors = {"straight": "tab:blue", "buckled-single": "tab:orange", "unresolved": "tab:red"}
    deterministic = [row for row in rows if row["source"] == "deterministic_fixture"]
    seeded = [row for row in rows if row["source"] == "seeded_trials"]
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    for row in deterministic:
        label = str(row["label_or_counts"])
        axes[0, 0].scatter(row["G_b"], row["peak_max_transverse_mean"], color=colors.get(label, "black"), label=label)
        axes[0, 1].scatter(row["bending_stiffness"], row["peak_max_transverse_mean"], color=colors.get(label, "black"), label=label)
    for row in seeded:
        axes[0, 0].errorbar(row["G_b"], row["peak_max_transverse_mean"], yerr=row["peak_max_transverse_std"], fmt="o", color="black", capsize=3)
        axes[0, 1].errorbar(row["bending_stiffness"], row["peak_max_transverse_mean"], yerr=row["peak_max_transverse_std"], fmt="o", color="black", capsize=3)
    axes[0, 0].set(xlabel="G_b", ylabel="peak max |y|", title="growth / bending relaxation")
    axes[0, 1].set(xlabel="EI", ylabel="peak max |y|", title="rigidity comparison")
    axes[0, 0].legend(loc="best", fontsize="small")
    sensitivity = [row for row in rows if row["question"] == "numerical_sensitivity"]
    if sensitivity:
        labels = [str(row["case"]) for row in sensitivity]
        values = [row["peak_max_transverse_mean"] for row in sensitivity]
        axes[1, 0].bar(labels, values, color="tab:green")
        axes[1, 0].tick_params(axis="x", rotation=30)
    axes[1, 0].set(ylabel="peak max |y|", title="sensitivity indicators")
    if seeded:
        labels = [str(row["case"]) for row in seeded]
        means = [row["onset_time_mean"] or 0.0 for row in seeded]
        errors = [row["onset_time_std"] or 0.0 for row in seeded]
        axes[1, 1].bar(labels, means, yerr=errors, color="tab:purple", capsize=3)
        axes[1, 1].tick_params(axis="x", rotation=30)
    axes[1, 1].set(ylabel="onset time mean ± std", title="seeded trial variability")
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return "generated"


def _seeded_trial_cases(
    cases: Sequence[CaseSpec], trial_design: Mapping[str, Any]
) -> list[CaseSpec]:
    by_name = {case.name: case for case in cases}
    names = [str(name) for name in trial_design.get("cases", [])]
    seeds = [int(seed) for seed in trial_design.get("seeds", [])]
    noise_fraction = float(trial_design.get("noise_fraction", 0.05))
    if len(set(seeds)) != len(seeds):
        raise BenchmarkError("trial_design seeds must be unique")
    if not 0.0 <= noise_fraction <= 0.5:
        raise BenchmarkError("trial_design noise_fraction must be in [0, 0.5]")
    result: list[CaseSpec] = []
    for base_name in names:
        if base_name not in by_name:
            raise BenchmarkError(f"trial_design references unknown case: {base_name}")
        base = by_name[base_name]
        for trial, seed in enumerate(seeds, start=1):
            overrides = dict(base.overrides)
            overrides["trial_noise_fraction"] = noise_fraction
            result.append(
                CaseSpec(
                    f"{base_name}_trial_{trial:02d}_seed_{seed}",
                    overrides,
                    seed=seed,
                    trial=trial,
                    base_name=base_name,
                )
            )
    return result


def run_suite(
    base_config: Mapping[str, Any],
    cases: Sequence[CaseSpec],
    sensitivities: Sequence[CaseSpec],
    output: Path,
    selected_case: str | None = None,
    include_sensitivity: bool = True,
    trial_design: Mapping[str, Any] | None = None,
    include_trials: bool = True,
) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    revision = detect_git_revision(Path.cwd())
    selected = [case for case in cases if selected_case is None or case.name == selected_case]
    if selected_case is not None and not selected:
        raise BenchmarkError(f"unknown case: {selected_case}")
    if include_sensitivity and selected_case is None:
        selected = selected + list(sensitivities)
    trial_design_value = dict(trial_design or {})
    if include_trials and selected_case is None and trial_design_value:
        selected = selected + _seeded_trial_cases(cases, trial_design_value)
    summaries = [run_case(case, base_config, output, revision) for case in selected]
    write_suite_csv(output / "summary.csv", summaries)
    trials = trial_summary(summaries)
    comparisons = question_comparison_rows(summaries, trials)
    write_question_comparison_csv(output / "question_comparison.csv", comparisons)
    comparison_plot = _plot_question_comparison(output / "question_comparison.png", comparisons)
    _write_json(output / "trial_summary.json", trials)
    _write_json(output / "question_comparison.json", comparisons)
    suite = {
        "schema_version": SCHEMA_VERSION,
        "benchmark": "P1B non-contact growth-relaxation buckling",
        "git_revision": revision,
        "contact_enabled": False,
        "diameter": 0.0,
        "initial_noncontact_required": True,
        "classification": "straight / buckled-single / unresolved mechanical regimes; no phase-transition claim",
        "tau_b_definition": "drag_density * length_scale^4 / (bending_stiffness * pi^4)",
        "trial_design": trial_design_value,
        "cases": [summary["case"] for summary in summaries],
        "results": summaries,
        "trial_summary": trials,
        "question_comparison": comparisons,
        "question_comparison_plot": comparison_plot,
        "unresolved_items": [
            "seeded trials vary only the deterministic initial imperfection; they are not an experimental noise model",
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
    parser.add_argument("--no-trials", action="store_true", help="omit seeded trial cases")
    parser.add_argument("--growth-rate", type=float, help="override growth_rate for all selected cases")
    parser.add_argument("--bending-stiffness", type=float, help="override bending_stiffness for all selected cases")
    parser.add_argument("--dt", type=float, help="override dt for all selected cases")
    parser.add_argument("--amplitude", type=float, help="override perturbation amplitude for all selected cases")
    parser.add_argument("--n-nodes", type=int, help="override spatial resolution for all selected cases")
    return parser.parse_args()


def main() -> int:
    args = _cli()
    try:
        base, cases, sensitivities, trial_design = load_suite_config(args.config)
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
            trial_design=trial_design,
            include_trials=not args.no_trials,
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
