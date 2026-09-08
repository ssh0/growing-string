"""Finite-radius contact buckling and folding benchmark.

The runner is intentionally an experiment harness around the public
``growing_filament`` model.  It measures finite-radius segment contacts while
keeping the penalty law, growth rule, remeshing, and step acceptance in the
model implementation.  The default output is compact: no trajectory arrays
or videos are written.

Example::

    PYTHONPATH=continuum_filament_model/src \\
      python continuum_filament_model/benchmarks/contact_buckling_benchmark.py \\
      --config continuum_filament_model/benchmarks/configs/p2_contact_buckling.json \\
      --output continuum_filament_model/results/contact_buckling

The results are numerical diagnostics, not a phase-transition or experimental
validation claim.  In particular, ``contact_length`` is the conservative
active-pair support estimator documented in ``notes/contact_buckling_coiling.md``.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

if __package__ in {None, ""}:
    _HERE = Path(__file__).resolve()
    _SRC = _HERE.parents[1] / "src"
    if str(_SRC) not in sys.path:
        sys.path.insert(0, str(_SRC))

from growing_filament.geometry import nonlocal_segment_contacts  # noqa: E402
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
    sha256_hex,
)


SCHEMA_VERSION = "continuum-filament-p2-contact-buckling-1"


@dataclass(frozen=True)
class CaseSpec:
    """One deterministic benchmark case."""

    name: str
    overrides: dict[str, Any]
    group: str = "phase_map"


class BenchmarkError(ValueError):
    """Invalid contact-buckling benchmark input."""


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
    path.write_bytes(canonical_json_bytes(value) + b"\n")


def _finite_float(config: Mapping[str, Any], key: str) -> float:
    try:
        value = float(config[key])
    except (KeyError, TypeError, ValueError) as exc:
        raise BenchmarkError(f"{key} must be a finite number") from exc
    if not math.isfinite(value):
        raise BenchmarkError(f"{key} must be finite")
    return value


def validate_case_config(config: Mapping[str, Any]) -> None:
    required = (
        "length",
        "n_nodes",
        "axial_stiffness",
        "bending_stiffness",
        "drag_density",
        "growth_rate",
        "contact_stiffness",
        "diameter",
        "dt",
        "t_end",
        "a_max_factor",
    )
    for key in required:
        _finite_float(config, key)
    n_nodes = _finite_float(config, "n_nodes")
    if int(n_nodes) != n_nodes or n_nodes < 3:
        raise BenchmarkError("n_nodes must be an integer >= 3")
    for key in (
        "length",
        "axial_stiffness",
        "bending_stiffness",
        "drag_density",
        "dt",
        "t_end",
        "a_max_factor",
    ):
        if _finite_float(config, key) <= 0.0:
            raise BenchmarkError(f"{key} must be positive")
    for key in ("growth_rate", "contact_stiffness", "diameter"):
        if _finite_float(config, key) < 0.0:
            raise BenchmarkError(f"{key} must be non-negative")
    if not 0.0 < float(config.get("max_displacement_fraction", 0.25)) <= 1.0:
        raise BenchmarkError("max_displacement_fraction must be in (0, 1]")
    if int(config.get("max_retries", 12)) != config.get("max_retries", 12):
        raise BenchmarkError("max_retries must be an integer")
    shape = str(config.get("initial_shape", "u"))
    if shape not in {"u", "s", "sine", "straight"}:
        raise BenchmarkError("initial_shape must be one of u, s, sine, straight")
    if not bool(config.get("fixed_left", True)):
        raise BenchmarkError("the benchmark currently requires a fixed left endpoint")
    if not bool(config.get("reject_crossing", True)):
        raise BenchmarkError("reject_crossing=true is required for contact diagnostics")


def load_config(path: Path) -> dict[str, Any]:
    """Load and lightly normalize a P2 JSON configuration."""

    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        raise BenchmarkError("config root must be a JSON object")
    base = dict(raw.get("base", {}))
    defaults = {
        "length": 2.0,
        "n_nodes": 9,
        "axial_stiffness": 100.0,
        "bending_stiffness": 0.1,
        "drag_density": 1.0,
        "growth_rate": 0.1,
        "contact_stiffness": 10.0,
        "diameter": 0.30,
        "dt": 0.0005,
        "t_end": 0.01,
        "a_max_factor": 8.0,
        "initial_shape": "u",
        "fixed_left": True,
        "fixed_right": True,
        "reject_crossing": True,
        "max_retries": 12,
        "dt_min": 1.0e-10,
        "max_displacement_fraction": 0.25,
        "buckling_threshold_fraction": 0.02,
        "fold_curvature_threshold": 1.0e-8,
        "convergence_tolerance": 0.1,
    }
    effective = dict(defaults)
    effective.update(base)
    cases = []
    for item in raw.get("cases", []):
        if not isinstance(item, Mapping) or "name" not in item:
            raise BenchmarkError("cases entries require name")
        cases.append(
            {
                "name": str(item["name"]),
                "overrides": dict(item.get("overrides", {})),
                "group": str(item.get("group", "phase_map")),
            }
        )
    phase_grid = dict(raw.get("phase_grid", {}))
    convergence = dict(raw.get("convergence", {}))
    output_policy = dict(raw.get("output_policy", {}))
    return {
        "schema_version": str(raw.get("schema_version", SCHEMA_VERSION)),
        "base": effective,
        "cases": cases,
        "phase_grid": phase_grid,
        "convergence": convergence,
        "output_policy": output_policy,
    }


def _grid_cases(base: Mapping[str, Any], phase_grid: Mapping[str, Any]) -> list[CaseSpec]:
    """Expand named parameter axes into deterministic phase-map cases."""

    axes = [
        key for key in ("growth_rate", "contact_stiffness", "diameter", "bending_stiffness")
        if key in phase_grid and phase_grid[key]
    ]
    if not axes:
        return []
    values = [list(phase_grid[key]) for key in axes]
    result: list[CaseSpec] = []
    for combo in itertools.product(*values):
        overrides = dict(zip(axes, combo))
        tokens = [f"{key}={float(value):g}" for key, value in overrides.items()]
        result.append(CaseSpec("grid_" + "_".join(tokens), overrides, "phase_map"))
    return result


def _convergence_cases(convergence: Mapping[str, Any]) -> list[CaseSpec]:
    result: list[CaseSpec] = []
    for item in convergence.get("cases", []):
        if not isinstance(item, Mapping) or "name" not in item:
            raise BenchmarkError("convergence cases entries require name")
        result.append(
            CaseSpec(
                str(item["name"]),
                dict(item.get("overrides", {})),
                str(item.get("group", "convergence")),
            )
        )
    return result


def case_specs(config: Mapping[str, Any]) -> list[CaseSpec]:
    """Return explicit, phase-grid, and convergence cases in stable order."""

    result = [
        CaseSpec(str(item["name"]), dict(item.get("overrides", {})), str(item.get("group", "phase_map")))
        for item in config.get("cases", [])
    ]
    result.extend(_grid_cases(config["base"], config.get("phase_grid", {})))
    result.extend(_convergence_cases(config.get("convergence", {})))
    names = [case.name for case in result]
    if len(set(names)) != len(names):
        raise BenchmarkError("case names must be unique")
    return result


def initial_state(config: Mapping[str, Any]) -> FilamentState:
    """Create the deterministic sine, S, or semicircular U fixture."""

    length = float(config["length"])
    n_nodes = int(config["n_nodes"])
    u = np.linspace(0.0, 1.0, n_nodes)
    shape = str(config.get("initial_shape", "u"))
    amplitude = float(config.get("amplitude", 0.04 * length))
    if shape == "u":
        radius = length / np.pi
        theta = np.linspace(np.pi, 0.0, n_nodes)
        positions = np.column_stack((radius * np.cos(theta), -radius * np.sin(theta)))
    elif shape == "s":
        positions = np.column_stack((length * u, amplitude * np.sin(2.0 * np.pi * u)))
    elif shape == "straight":
        positions = np.column_stack((length * u, np.zeros_like(u)))
    else:
        positions = np.column_stack((length * u, amplitude * np.sin(np.pi * u)))
    rest_lengths = np.linalg.norm(np.diff(positions, axis=0), axis=1)
    return FilamentState(positions, rest_lengths)


def _effective_case(base: Mapping[str, Any], case: CaseSpec) -> dict[str, Any]:
    result = dict(base)
    result.update(case.overrides)
    result["name"] = case.name
    result["group"] = case.group
    result["n_nodes"] = int(result["n_nodes"])
    validate_case_config(result)
    return result


def dimensionless_groups(config: Mapping[str, Any]) -> dict[str, float | str]:
    """Return P2 axes and the corresponding P1B time-scale groups."""

    length = float(config["length"])
    ei = float(config["bending_stiffness"])
    ea = float(config["axial_stiffness"])
    zeta = float(config["drag_density"])
    growth = float(config["growth_rate"])
    diameter = float(config["diameter"])
    contact_stiffness = float(config["contact_stiffness"])
    tau_b = zeta * length**4 / (ei * np.pi**4)
    tau_s = zeta * length**2 / ea
    chi = ei / (ea * length**2)
    pi_c = contact_stiffness * diameter**2 / ei if diameter > 0.0 else 0.0
    return {
        "length_scale": length,
        "tau_b": float(tau_b),
        "tau_s": float(tau_s),
        # Keep the P1B field names alongside descriptive P2 aliases so
        # existing summary consumers can compare the two benchmarks directly.
        "G_b": float(growth * tau_b),
        "G_s": float(growth * tau_s),
        "growth_number_G_b": float(growth * tau_b),
        "growth_number_G_s": float(growth * tau_s),
        "chi": float(chi),
        "contact_stiffness": contact_stiffness,
        "diameter": diameter,
        "diameter_over_L": diameter / length,
        "Pi_c": float(pi_c),
        "dt_over_tau_b": float(float(config["dt"]) / tau_b),
        "definition": "chi=EI/(EA L^2), Pi_c=k_c D^2/EI, G_b=g*zeta*L^4/(EI*pi^4)",
    }


def _signed_curvature(state: FilamentState) -> np.ndarray:
    edges = np.diff(state.positions, axis=0)
    lengths = np.linalg.norm(edges, axis=1)
    if len(edges) < 2:
        return np.empty(0, dtype=float)
    tangents = edges / lengths[:, None]
    cross = tangents[:-1, 0] * tangents[1:, 1] - tangents[:-1, 1] * tangents[1:, 0]
    local = 0.5 * (state.rest_lengths[:-1] + state.rest_lengths[1:])
    return cross / np.maximum(local, 1.0e-15)


def contact_metrics(state: FilamentState, diameter: float) -> dict[str, Any]:
    """Measure active finite-radius segment contacts for one state.

    ``contact_length`` is the sum of ``min(length_i, length_j)`` over active
    pairs.  It is a conservative contact-support estimator: it avoids
    claiming a continuous contact interval from one closest-point sample and
    is invariant to the ordering of the pair.  ``max_penetration`` is the
    penalty overlap ``max(0, D-d)`` and not a post-processed distance.
    """

    if diameter <= 0.0:
        return {
            "contact_length": 0.0,
            "active_contact_pairs": 0,
            "max_penetration": 0.0,
            "penetration_ratio": 0.0,
            "finite_radius_contact_pairs": [],
            "centerline_intersection_pairs": [],
        }
    lengths = np.linalg.norm(np.diff(state.positions, axis=0), axis=1)
    contacts = nonlocal_segment_contacts(state.positions, float(diameter))
    active = [contact for contact in contacts if contact.is_contact]
    finite = [contact for contact in active if not contact.centerline_intersection]
    penetration = max((float(contact.penetration) for contact in finite), default=0.0)
    support = sum(
        min(float(lengths[contact.segment_i]), float(lengths[contact.segment_j]))
        for contact in finite
    )
    return {
        "contact_length": float(support),
        "active_contact_pairs": int(len(active)),
        "max_penetration": penetration,
        "penetration_ratio": float(penetration / diameter) if diameter > 0.0 else 0.0,
        "finite_radius_contact_pairs": [
            [int(contact.segment_i), int(contact.segment_j)] for contact in finite
        ],
        "centerline_intersection_pairs": [
            [int(contact.segment_i), int(contact.segment_j)]
            for contact in active
            if contact.centerline_intersection
        ],
    }


def _contact_components(pairs: Sequence[Sequence[int]]) -> int:
    parent: dict[int, int] = {}

    def find(value: int) -> int:
        parent.setdefault(value, value)
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    def union(left: int, right: int) -> None:
        root_left, root_right = find(left), find(right)
        if root_left != root_right:
            parent[root_right] = root_left

    for left, right in pairs:
        union(int(left), int(right))
    return len({find(value) for value in parent})


def _shape_metrics(state: FilamentState, config: Mapping[str, Any]) -> dict[str, Any]:
    start, end = state.positions[0], state.positions[-1]
    chord = end - start
    chord_length = float(np.linalg.norm(chord))
    tangent = chord / max(chord_length, 1.0e-15)
    normal = np.asarray([-tangent[1], tangent[0]])
    transverse = (state.positions - start) @ normal
    signed = _signed_curvature(state)
    threshold = float(config.get("fold_curvature_threshold", 1.0e-8))
    signs = np.sign(signed[np.abs(signed) >= threshold])
    fold_count = int(np.sum(signs[1:] * signs[:-1] < 0.0)) if len(signs) > 1 else 0
    return {
        "max_transverse_displacement": float(np.max(np.abs(transverse))),
        "max_curvature": float(np.max(discrete_curvature(state))) if len(discrete_curvature(state)) else 0.0,
        "rms_curvature": float(np.sqrt(np.mean(signed * signed))) if len(signed) else 0.0,
        "fold_count": fold_count,
        "endpoint_distance": chord_length,
    }


def _periodicity_metrics(state: FilamentState) -> tuple[float | None, float]:
    """Estimate dominant folding wavelength and spectral periodicity."""

    signed = _signed_curvature(state)
    if len(signed) < 3:
        return None, 0.0
    edge_lengths = np.linalg.norm(np.diff(state.positions, axis=0), axis=1)
    arc_nodes = np.concatenate(([0.0], np.cumsum(edge_lengths)))
    # Signed curvature is defined at interior nodes, not at segment centers.
    centers = arc_nodes[1:-1]
    sample_u = np.linspace(float(centers[0]), float(centers[-1]), 64)
    values = np.interp(sample_u, centers, signed)
    values -= float(np.mean(values))
    spectrum = np.abs(np.fft.rfft(values))[1:]
    if not np.any(spectrum > 1.0e-12):
        return None, 0.0
    index = int(np.argmax(spectrum)) + 1
    total = float(np.sum(spectrum))
    wavelength = float(np.sum(edge_lengths) / index)
    periodicity = float(spectrum[index - 1] / total) if total > 0.0 else 0.0
    return wavelength, periodicity


def _node_drag(rest_lengths: np.ndarray, drag_density: float) -> np.ndarray:
    weights = np.empty(len(rest_lengths) + 1, dtype=float)
    weights[0] = 0.5 * rest_lengths[0]
    weights[-1] = 0.5 * rest_lengths[-1]
    if len(rest_lengths) > 1:
        weights[1:-1] = 0.5 * (rest_lengths[:-1] + rest_lengths[1:])
    return drag_density * weights


def _work_rows(
    trajectory: Sequence[FilamentState],
    simulator: OverdampedGrowingFilament,
) -> list[dict[str, float]]:
    """Add discrete energy/work accounting to accepted trajectory rows.

    For fixed node count, ``W_diss = dt sum Gamma_i |v_i|^2`` is the accepted
    explicit-Euler estimate.  A remesh changes node identity, so its interval
    is marked ``remesh_interval`` and contributes no fabricated velocity; the
    residual ``delta_E + W_diss`` is the growth-work diagnostic.
    """

    cumulative_dissipation = 0.0
    cumulative_growth = 0.0
    result = []
    for index, state in enumerate(trajectory):
        components = simulator.energy_components(state.positions, state.rest_lengths)
        total = float(sum(components.values()))
        dissipation = 0.0
        remesh_interval = False
        if index > 0:
            previous = trajectory[index - 1]
            dt = float(state.time - previous.time)
            remesh_interval = previous.n_nodes != state.n_nodes
            if not remesh_interval and dt > 0.0:
                velocity = (state.positions - previous.positions) / dt
                gamma = _node_drag(state.rest_lengths, simulator.parameters.drag_density)
                dissipation = float(dt * np.sum(gamma[:, None] * velocity * velocity))
            previous_components = simulator.energy_components(
                previous.positions, previous.rest_lengths
            )
            delta_energy = total - float(sum(previous_components.values()))
            growth_work = delta_energy + dissipation
            cumulative_dissipation += dissipation
            cumulative_growth += growth_work
        result.append(
            {
                "energy_stretch": float(components["stretch"]),
                "energy_bend": float(components["bend"]),
                "energy_contact": float(components["contact"]),
                "energy_total": total,
                "dissipation_work_step": dissipation,
                "growth_work_step": 0.0 if index == 0 else float(growth_work),
                "dissipation_work": cumulative_dissipation,
                "growth_work": cumulative_growth,
                "remesh_interval": bool(remesh_interval),
            }
        )
    return result


def _row(
    state: FilamentState,
    simulator: OverdampedGrowingFilament,
    config: Mapping[str, Any],
    work: Mapping[str, Any],
) -> dict[str, Any]:
    contacts = contact_metrics(state, float(config["diameter"]))
    shape = _shape_metrics(state, config)
    wavelength, periodicity = _periodicity_metrics(state)
    self_loop_count = _contact_components(contacts["finite_radius_contact_pairs"])
    return {
        "time": float(state.time),
        "step": int(state.step),
        "n_nodes": int(state.n_nodes),
        "reference_length": float(np.sum(state.rest_lengths)),
        "contour_length": contour_length(state),
        "radius_of_gyration": radius_of_gyration(state),
        "arc_length_weighted_radius_of_gyration": arc_length_weighted_radius_of_gyration(state),
        **shape,
        **contacts,
        "self_loop_count": int(self_loop_count),
        "fold_wavelength": wavelength,
        "fold_periodicity": periodicity,
        **work,
    }


def _onset(rows: Sequence[Mapping[str, Any]], config: Mapping[str, Any]) -> dict[str, float | None]:
    if not rows:
        return {"buckling_time": None, "contact_time": None}
    length = float(config["length"])
    initial_shape = float(rows[0]["max_transverse_displacement"])
    buckling_threshold = max(
        3.0 * initial_shape,
        float(config.get("buckling_threshold_fraction", 0.02)) * length,
    )
    buckled = next(
        (row for row in rows if float(row["max_transverse_displacement"]) > buckling_threshold),
        None,
    )
    contact = next(
        (row for row in rows if int(row["active_contact_pairs"]) > 0),
        None,
    )
    return {
        "buckling_time": None if buckled is None else float(buckled["time"]),
        "contact_time": None if contact is None else float(contact["time"]),
    }


def _classification(rows: Sequence[Mapping[str, Any]], failure: str | None) -> str:
    if failure or not rows:
        return "unresolved"
    if any(int(row["active_contact_pairs"]) > 0 for row in rows):
        if any(int(row["fold_count"]) > 0 for row in rows):
            return "folding-contact"
        return "self-contact"
    return "buckled-no-contact" if rows[-1]["max_transverse_displacement"] > rows[0]["max_transverse_displacement"] * 3.0 else "straight-no-contact"


def run_case(
    case: CaseSpec,
    base_config: Mapping[str, Any],
    *,
    git_revision: str | None = None,
) -> dict[str, Any]:
    """Run one case and return a compact, JSON-compatible summary."""

    config = _effective_case(base_config, case)
    state = initial_state(config)
    rest_reference = float(np.mean(state.rest_lengths))
    params = ModelParameters(
        axial_stiffness=float(config["axial_stiffness"]),
        bending_stiffness=float(config["bending_stiffness"]),
        drag_density=float(config["drag_density"]),
        contact_stiffness=float(config["contact_stiffness"]),
        diameter=float(config["diameter"]),
        growth_rate=float(config["growth_rate"]),
        reference_length=rest_reference,
        dt=float(config["dt"]),
        t_end=float(config["t_end"]),
        a_max=float(np.max(state.rest_lengths)) * float(config["a_max_factor"]),
        dt_min=float(config.get("dt_min", 1.0e-10)),
        max_retries=int(config.get("max_retries", 12)),
        max_displacement_fraction=float(config.get("max_displacement_fraction", 0.25)),
        fixed_left=bool(config.get("fixed_left", True)),
        fixed_right=bool(config.get("fixed_right", True)),
        reject_crossing=True,
    )
    failure: str | None = None
    simulator: OverdampedGrowingFilament | None = None
    trajectory: list[FilamentState] = [state.copy()]
    try:
        simulator = OverdampedGrowingFilament(state, params)
        trajectory = simulator.run()
    except (ModelError, RuntimeError, ValueError, FloatingPointError) as exc:
        failure = f"{type(exc).__name__}: {exc}"
    if simulator is None:
        # Build a diagnostic simulator only when initialization succeeded; an
        # initial crossing has no valid force/energy trajectory to summarize.
        rows: list[dict[str, Any]] = []
        final_state = state
        events: list[dict[str, Any]] = []
    else:
        work = _work_rows(trajectory, simulator)
        rows = [_row(item, simulator, config, work[index]) for index, item in enumerate(trajectory)]
        final_state = simulator.state
        events = simulator.event_log
    onsets = _onset(rows, config)
    groups = dimensionless_groups(config)
    metadata = {
        "benchmark": "P2 finite-radius contact buckling and folding",
        "case": case.name,
        "group": case.group,
        "physical_conditions": {
            "contact_stiffness": float(config["contact_stiffness"]),
            "diameter": float(config["diameter"]),
            "boundary": "fixed endpoint positions",
            "initial_shape": config.get("initial_shape", "u"),
        },
        "measurement_definitions": {
            "contact_length": "sum min(segment lengths) over active finite-radius pairs",
            "max_penetration": "max(0, diameter - centerline distance)",
            "dissipation_work": "accepted Euler dt*sum Gamma_i*|v_i|^2 estimate",
            "growth_work": "delta total energy plus discrete dissipation estimate",
        },
        "failure_reason": failure,
    }
    manifest = build_manifest(
        params,
        state,
        final_state=final_state,
        events=events,
        metadata=metadata,
        input_data=config,
        git_revision=git_revision,
    )
    compact_events = [
        {
            "event_type": event.get("event_type"),
            "accepted": event.get("accepted"),
            "reason": event.get("reason"),
            "time_after": event.get("time_after"),
        }
        for event in events
        if event.get("event_type") == "step_attempt" and event.get("accepted") is False
    ]
    manifest["events"] = compact_events
    manifest["event_count_total"] = len(events)
    peak = max(rows, key=lambda row: float(row["max_penetration"])) if rows else None
    periodicity_peak = max(rows, key=lambda row: float(row["fold_periodicity"])) if rows else None
    return {
        "schema_version": SCHEMA_VERSION,
        "case": case.name,
        "group": case.group,
        "effective_config": config,
        "dimensionless_groups": groups,
        "classification": _classification(rows, failure),
        "failure_reason": failure,
        "accepted_steps": int(simulator.accepted_steps) if simulator else 0,
        "rejected_steps": int(simulator.rejected_steps) if simulator else 0,
        "event_count": len(events),
        "onset": onsets,
        "initial_metrics": rows[0] if rows else None,
        "final_metrics": rows[-1] if rows else None,
        "peak_contact_metrics": peak,
        "max_penetration": max((float(row["max_penetration"]) for row in rows), default=0.0),
        "max_penetration_ratio": max((float(row["penetration_ratio"]) for row in rows), default=0.0),
        "max_contact_length": max((float(row["contact_length"]) for row in rows), default=0.0),
        "max_active_contact_pairs": max((int(row["active_contact_pairs"]) for row in rows), default=0),
        "max_fold_count": max((int(row["fold_count"]) for row in rows), default=0),
        "max_self_loop_count": max((int(row["self_loop_count"]) for row in rows), default=0),
        "dominant_fold_wavelength": None if periodicity_peak is None else periodicity_peak["fold_wavelength"],
        "max_fold_periodicity": max((float(row["fold_periodicity"]) for row in rows), default=0.0),
        "max_radius_of_gyration": max((float(row["radius_of_gyration"]) for row in rows), default=0.0),
        "max_curvature": max((float(row["max_curvature"]) for row in rows), default=0.0),
        "final_growth_work": float(rows[-1]["growth_work"]) if rows else 0.0,
        "final_dissipation_work": float(rows[-1]["dissipation_work"]) if rows else 0.0,
        "metrics_rows": rows,
        "manifest": manifest,
    }


def _summary_row(result: Mapping[str, Any]) -> dict[str, Any]:
    config = result["effective_config"]
    groups = result["dimensionless_groups"]
    onset = result["onset"]
    return {
        "case": result["case"],
        "group": result["group"],
        "classification": result["classification"],
        "chi": groups["chi"],
        "growth_rate": config["growth_rate"],
        "G_b": groups["G_b"],
        "G_s": groups["G_s"],
        "growth_number_G_b": groups["growth_number_G_b"],
        "growth_number_G_s": groups["growth_number_G_s"],
        "contact_stiffness": config["contact_stiffness"],
        "diameter": config["diameter"],
        "Pi_c": groups["Pi_c"],
        "dt": config["dt"],
        "buckling_time": onset["buckling_time"],
        "contact_time": onset["contact_time"],
        "max_contact_length": result["max_contact_length"],
        "max_active_contact_pairs": result["max_active_contact_pairs"],
        "max_penetration": result["max_penetration"],
        "max_penetration_ratio": result["max_penetration_ratio"],
        "max_radius_of_gyration": result["max_radius_of_gyration"],
        "max_curvature": result["max_curvature"],
        "max_fold_count": result["max_fold_count"],
        "max_self_loop_count": result["max_self_loop_count"],
        "dominant_fold_wavelength": result["dominant_fold_wavelength"],
        "max_fold_periodicity": result["max_fold_periodicity"],
        "final_growth_work": result["final_growth_work"],
        "final_dissipation_work": result["final_dissipation_work"],
        "accepted_steps": result["accepted_steps"],
        "rejected_steps": result["rejected_steps"],
        "failure_reason": result["failure_reason"],
    }


def _convergence_rows(
    results: Sequence[Mapping[str, Any]],
    tolerance: float,
) -> list[dict[str, Any]]:
    """Compare paired dt and contact-stiffness refinement cases.

    The output is deliberately an assessment table rather than a claim that a
    coarse/fine pair is converged.  A failed or missing pair is retained as
    ``converged=false`` so an incomplete validation cannot disappear from the
    compact result.
    """

    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for result in results:
        group = str(result.get("group", ""))
        if group in {"dt_convergence", "kc_convergence"}:
            grouped.setdefault(group, []).append(result)
    rows: list[dict[str, Any]] = []
    for group, values in sorted(grouped.items()):
        if len(values) < 2:
            rows.append({
                "group": group,
                "axis": "dt" if group == "dt_convergence" else "contact_stiffness",
                "case_coarse_or_low": values[0]["case"] if values else None,
                "case_fine_or_high": None,
                "value_coarse_or_low": None,
                "value_fine_or_high": None,
                "penetration_ratio_coarse_or_low": None,
                "penetration_ratio_fine_or_high": None,
                "relative_penetration_difference": None,
                "classification_match": False,
                "criterion": "paired cases required",
                "converged": False,
            })
            continue
        axis = "dt" if group == "dt_convergence" else "contact_stiffness"
        ordered = sorted(values, key=lambda item: float(item["effective_config"][axis]))
        low, high = ordered[0], ordered[-1]
        if axis == "dt":
            fine, coarse = low, high
            criterion = "relative penetration difference <= convergence_tolerance and classification match"
            converged = (
                low["failure_reason"] is None
                and high["failure_reason"] is None
                and low["classification"] == high["classification"]
                and abs(float(low["max_penetration_ratio"]) - float(high["max_penetration_ratio"]))
                / max(abs(float(low["max_penetration_ratio"])), 1.0e-15)
                <= tolerance
            )
            first, second = coarse, fine
        else:
            low, high = ordered[0], ordered[-1]
            criterion = "higher contact stiffness does not increase maximum penetration ratio"
            converged = (
                low["failure_reason"] is None
                and high["failure_reason"] is None
                and float(high["max_penetration_ratio"]) <= float(low["max_penetration_ratio"]) + 1.0e-12
            )
            first, second = low, high
        first_penetration = float(first["max_penetration_ratio"])
        second_penetration = float(second["max_penetration_ratio"])
        relative_difference = abs(first_penetration - second_penetration) / max(abs(second_penetration), 1.0e-15)
        rows.append({
            "group": group,
            "axis": axis,
            "case_coarse_or_low": first["case"],
            "case_fine_or_high": second["case"],
            "value_coarse_or_low": first["effective_config"][axis],
            "value_fine_or_high": second["effective_config"][axis],
            "penetration_ratio_coarse_or_low": first_penetration,
            "penetration_ratio_fine_or_high": second_penetration,
            "relative_penetration_difference": relative_difference,
            "classification_match": bool(first["classification"] == second["classification"]),
            "criterion": criterion,
            "converged": bool(converged),
        })
    return rows


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("\n", encoding="utf-8")
        return
    fields = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(_jsonable(row) for row in rows)


def _write_phase_plot(path: Path, rows: Sequence[Mapping[str, Any]]) -> str:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - environment dependent
        return f"plot unavailable: {type(exc).__name__}: {exc}"
    if not rows:
        return "plot skipped: no rows"
    labels = [str(row["case"]) for row in rows]
    x = np.asarray([float(row["Pi_c"]) for row in rows])
    y = np.asarray([float(row["max_penetration_ratio"]) for row in rows])
    color_values = np.asarray([float(row["max_contact_length"]) for row in rows])
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
    scatter = axes[0].scatter(x, y, c=color_values, cmap="viridis", s=55)
    axes[0].set(xlabel=r"$\Pi_c=k_cD^2/EI$", ylabel="max penetration / D", title="contact penalty map")
    fig.colorbar(scatter, ax=axes[0], label="max contact support length")
    axes[1].bar(np.arange(len(rows)), [float(row["max_active_contact_pairs"]) for row in rows])
    axes[1].set_xticks(np.arange(len(rows)), labels, rotation=55, ha="right", fontsize=7)
    axes[1].set(ylabel="active contact pairs", title="contact onset / folding cases")
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return "generated"


def _write_case_timeseries(path: Path, results: Sequence[Mapping[str, Any]]) -> None:
    rows: list[dict[str, Any]] = []
    for result in results:
        for row in result.get("metrics_rows", []):
            rows.append({"case": result["case"], **row})
    _write_csv(path, rows)


def run_benchmark(
    config: Mapping[str, Any],
    output: Path,
    *,
    git_revision: str | None = None,
) -> dict[str, Any]:
    """Run the configured phase map and write compact reproducibility files."""

    output.mkdir(parents=True, exist_ok=True)
    base = dict(config["base"])
    specs = case_specs(config)
    if not specs:
        raise BenchmarkError("configuration has no cases or phase_grid")
    revision = git_revision if git_revision is not None else detect_git_revision(Path.cwd())
    results = [run_case(spec, base, git_revision=revision) for spec in specs]
    rows = [_summary_row(result) for result in results]
    convergence_rows = _convergence_rows(
        results,
        float(base.get("convergence_tolerance", 0.1)),
    )
    _write_csv(output / "summary.csv", rows)
    _write_json(output / "summary.json", rows)
    _write_csv(output / "convergence_summary.csv", convergence_rows)
    _write_json(output / "convergence_summary.json", convergence_rows)
    _write_case_timeseries(output / "metrics.csv", results)
    plot_status = _write_phase_plot(output / "phase_map.png", rows)
    input_config = _jsonable(config)
    config_hash = sha256_hex(canonical_json_bytes(input_config))
    case_manifests = {
        str(result["case"]): {
            "input_hash": result["manifest"].get("input_hash"),
            "initial_state_hash": result["manifest"].get("initial_state_hash"),
            "canonical_state_hash": result["manifest"].get("canonical_state_hash"),
            "event_count_total": result["manifest"].get("event_count_total", result["event_count"]),
        }
        for result in results
    }
    compact_manifest = {
        "manifest_schema_version": "continuum-filament-p2-compact-manifest-1",
        "benchmark_schema_version": SCHEMA_VERSION,
        "benchmark": "finite-radius contact buckling and folding",
        "git_revision": revision,
        "config_hash": config_hash,
        "config_source": "caller-provided JSON-compatible configuration",
        "case_order": [result["case"] for result in results],
        "case_manifests": case_manifests,
        "convergence_summary": convergence_rows,
        "output_policy": {
            "trajectory_arrays": False,
            "videos": False,
            "stored_time_series": "compact CSV metrics only",
        },
        "measurement_limits": [
            "penalty contact permits finite penetration and is stiffness/time-step dependent",
            "contact_length is an active-pair support estimator, not a continuous contact integral",
            "centerline crossings are rejected by the model and are not resolved by an arbitrary normal",
            "growth work and dissipation are discrete Euler diagnostics; remesh intervals are flagged",
        ],
    }
    _write_json(output / "compact_manifest.json", compact_manifest)
    suite = {
        "schema_version": SCHEMA_VERSION,
        "benchmark": "finite-radius contact buckling and folding",
        "git_revision": revision,
        "config_hash": config_hash,
        "plot": plot_status,
        "convergence_summary": convergence_rows,
        "results": [
            {key: value for key, value in result.items() if key not in {"metrics_rows", "manifest"}}
            for result in results
        ],
        "compact_manifest": compact_manifest,
    }
    _write_json(output / "suite.json", suite)
    return suite


def _cli() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = _cli()
    try:
        config = load_config(args.config)
        suite = run_benchmark(config, args.output)
    except (BenchmarkError, OSError, json.JSONDecodeError, ValueError) as exc:
        print(f"contact benchmark error: {exc}", file=sys.stderr)
        return 2
    failures = [item for item in suite["results"] if item.get("failure_reason") is not None]
    print(json.dumps({"output": str(args.output), "cases": len(suite["results"]), "failures": len(failures)}, sort_keys=True))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
