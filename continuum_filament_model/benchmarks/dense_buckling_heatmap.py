"""Dense non-contact buckling map with compact waveform diagnostics.

The runner extends the existing P1B benchmark measurement layer without
changing the continuum solver.  Each ``(G_b, chi)`` cell is a deterministic
fixture (the initial sine imperfection is not a stochastic trial).  Results
contain continuous observables, mode decomposition, an explicit unresolved
reason, and only three coordinate snapshots per selected representative.

Example::

    PYTHONPATH=continuum_filament_model/src \
      python continuum_filament_model/benchmarks/dense_buckling_heatmap.py \
      --config continuum_filament_model/benchmarks/configs/p1b2_dense_heatmap.json \
      --output continuum_filament_model/results/presentation_data/dense_buckling

The output is deliberately bounded.  Full trajectories, event logs, and
per-cell plots are not written.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

if __package__ in {None, ""}:
    _HERE = Path(__file__).resolve()
    _SRC = _HERE.parents[1] / "src"
    if str(_SRC) not in sys.path:
        sys.path.insert(0, str(_SRC))
    if str(_HERE.parent) not in sys.path:
        sys.path.insert(0, str(_HERE.parent))
    from buckling_benchmark import (  # type: ignore  # noqa: E402
        _classification,
        _metrics,
        _mode_coefficients,
        _transverse_coordinates,
        dimensionless_groups,
        initial_perturbed_state,
        validate_case_config,
    )
    from p1b2_experiments import target_overrides  # type: ignore  # noqa: E402
else:
    from .buckling_benchmark import (  # noqa: E402
        _classification,
        _metrics,
        _mode_coefficients,
        _transverse_coordinates,
        dimensionless_groups,
        initial_perturbed_state,
        validate_case_config,
    )
    from .p1b2_experiments import target_overrides  # noqa: E402

from growing_filament.model import (  # noqa: E402
    FilamentState,
    ModelError,
    ModelParameters,
    OverdampedGrowingFilament,
)
from growing_filament.reproducibility import (  # noqa: E402
    canonical_json_bytes,
    detect_git_revision,
)

SCHEMA_VERSION = "continuum-filament-dense-buckling-1"


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


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(fields), extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(_jsonable(row) for row in rows)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_config(path: Path) -> dict[str, Any]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        raise ValueError("dense heatmap config root must be an object")
    base = dict(raw.get("base", {}))
    defaults = {
        "length": 2.0,
        "n_nodes": 7,
        "axial_stiffness": 100.0,
        "bending_stiffness": 0.1,
        "drag_density": 1.0,
        "growth_rate": 0.1,
        "amplitude": 0.02,
        "dt": 0.002,
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
        "trial_noise_fraction": 0.0,
    }
    effective = {**defaults, **base}
    effective["n_nodes"] = int(effective["n_nodes"])
    validate_case_config(effective)
    axes = dict(raw.get("axes", {}))
    gb_values = [float(value) for value in axes.get("G_b", [])]
    chi_values = [float(value) for value in axes.get("chi", [])]
    if len(gb_values) < 7 or len(chi_values) < 7:
        raise ValueError("dense heatmap requires at least 7 values on each axis")
    if any(value <= 0.0 or not math.isfinite(value) for value in gb_values + chi_values):
        raise ValueError("G_b and chi axes must contain positive finite values")
    if len(gb_values) * len(chi_values) > int(raw.get("output_policy", {}).get("max_conditions", 80)):
        raise ValueError("dense heatmap exceeds output_policy.max_conditions")
    if len(set(gb_values)) != len(gb_values) or len(set(chi_values)) != len(chi_values):
        raise ValueError("dense heatmap axes must be unique")
    output_policy = {
        "max_conditions": 80,
        "max_output_bytes": 8_000_000,
        **dict(raw.get("output_policy", {})),
    }
    return {
        "schema_version": str(raw.get("schema_version", SCHEMA_VERSION)),
        "base": effective,
        "axes": {"G_b": gb_values, "chi": chi_values},
        "snapshot_roles": list(raw.get("snapshot_roles", ["straight", "single_buckling", "higher_mode", "boundary_near"])),
        "output_policy": output_policy,
    }


def _node_drag(rest_lengths: np.ndarray, drag_density: float) -> np.ndarray:
    weights = np.empty(len(rest_lengths) + 1, dtype=float)
    weights[0] = 0.5 * rest_lengths[0]
    weights[-1] = 0.5 * rest_lengths[-1]
    if len(rest_lengths) > 1:
        weights[1:-1] = 0.5 * (rest_lengths[:-1] + rest_lengths[1:])
    return drag_density * weights


def _work_and_rows(
    trajectory: Sequence[FilamentState],
    simulator: OverdampedGrowingFilament,
    anchors: np.ndarray,
) -> tuple[list[dict[str, Any]], float]:
    """Measure states and a conservative accepted-Euler dissipation estimate."""

    rows: list[dict[str, Any]] = []
    cumulative_dissipation = 0.0
    for index, state in enumerate(trajectory):
        row = _metrics(state, simulator, anchors)
        dissipation_step = 0.0
        remesh_interval = False
        if index > 0:
            previous = trajectory[index - 1]
            remesh_interval = previous.n_nodes != state.n_nodes
            dt = float(state.time - previous.time)
            if not remesh_interval and dt > 0.0:
                velocity = (state.positions - previous.positions) / dt
                gamma = _node_drag(previous.rest_lengths, simulator.parameters.drag_density)
                dissipation_step = float(dt * np.sum(gamma[:, None] * velocity * velocity))
                cumulative_dissipation += dissipation_step
        row.update({
            "dissipation_energy_step": dissipation_step,
            "dissipation_energy": cumulative_dissipation,
            "remesh_interval": bool(remesh_interval),
        })
        rows.append(row)
    return rows, cumulative_dissipation


def _mode_detail(state: FilamentState, anchors: np.ndarray, max_mode: int = 6) -> dict[str, Any]:
    coefficients = np.asarray(_mode_coefficients(state, anchors, max_mode=max_mode), dtype=float)
    absolute = np.abs(coefficients)
    norm = float(np.linalg.norm(coefficients))
    fractions = absolute / norm if norm > 1.0e-15 else np.zeros_like(absolute)
    return {
        "mode_amplitudes": coefficients.tolist(),
        "mode_abs_amplitudes": absolute.tolist(),
        "mode_fractions": fractions.tolist(),
        "dominant_mode": int(np.argmax(absolute) + 1) if len(absolute) else None,
        "first_mode_fraction": float(fractions[0]) if len(fractions) else None,
        "higher_mode_fraction": float(np.sqrt(np.sum(fractions[1:] ** 2))) if len(fractions) > 1 else 0.0,
    }


def _waveform_classification(
    rows: Sequence[Mapping[str, Any]],
    mode: Mapping[str, Any],
    config: Mapping[str, Any],
    failure_reason: str | None,
) -> dict[str, Any]:
    if failure_reason:
        return {"label": "numerical_failure", "reason": "solver_failure", "detail": failure_reason}
    if not rows:
        return {"label": "numerical_failure", "reason": "no_accepted_trajectory", "detail": None}
    peak = max(rows, key=lambda row: float(row["max_transverse_displacement"]))
    threshold = max(3.0 * abs(float(rows[0]["first_mode_abs_amplitude"])), 0.005 * float(config["length"]))
    amplitude = float(peak["max_transverse_displacement"])
    dominant = mode.get("dominant_mode")
    f1 = float(mode.get("first_mode_fraction") or 0.0)
    higher = float(mode.get("higher_mode_fraction") or 0.0)
    curvature = float(peak.get("max_curvature", 0.0))
    rms = float(peak.get("rms_curvature", 0.0))
    concentration = curvature / max(rms, 1.0e-15)
    if amplitude <= threshold:
        return {"label": "no_onset", "reason": "no_onset", "detail": "peak transverse displacement did not exceed onset threshold"}
    # A dominant n>=2 mode is the direct explanation for f1 < 70%; a large
    # max/RMS curvature ratio is retained as a separate localised-buckling
    # signal so the two unresolved mechanisms are not conflated.
    higher_mode = dominant is not None and int(dominant) >= 2 and higher >= 0.30
    localized = concentration >= 3.0
    if higher_mode and localized:
        reason = "higher_mode_wave_and_localized_buckling"
    elif higher_mode:
        reason = "higher_mode_wave"
    elif localized:
        reason = "localized_buckling"
    elif f1 < 0.70:
        reason = "mixed_mode"
    else:
        reason = "single_mode"
    return {
        "label": "higher_mode_wave" if higher_mode else ("localized_buckling" if localized else ("single_mode" if f1 >= 0.70 else "mixed_mode")),
        "reason": reason,
        "detail": {
            "dominant_mode": dominant,
            "first_mode_fraction": f1,
            "higher_mode_fraction": higher,
            "curvature_concentration_max_over_rms": concentration,
            "onset_threshold": threshold,
        },
    }


def _run_cell(base: Mapping[str, Any], gb: float, chi: float) -> tuple[dict[str, Any], list[FilamentState]]:
    overrides = target_overrides(base, gb, chi)
    config = dict(base)
    config.update(overrides)
    config["name"] = f"gb_{gb:.8g}_chi_{chi:.8g}"
    config["a_max"] = (float(config["length"]) / (int(config["n_nodes"]) - 1)) * float(config["a_max_factor"])
    state = initial_perturbed_state(config)
    anchors = np.asarray([state.positions[0], state.positions[-1]], dtype=float)
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
    failure_reason: str | None = None
    simulator: OverdampedGrowingFilament | None = None
    trajectory: list[FilamentState] = [state.copy()]
    try:
        simulator = OverdampedGrowingFilament(state, params)
        trajectory = simulator.run()
    except (ModelError, RuntimeError, ValueError, FloatingPointError) as exc:
        failure_reason = f"{type(exc).__name__}: {exc}"
    if simulator is None:
        rows: list[dict[str, Any]] = []
        classification = {"label": "unresolved", "failure_reason": failure_reason}
        mode = {"mode_amplitudes": [], "mode_fractions": [], "dominant_mode": None, "first_mode_fraction": None, "higher_mode_fraction": None}
        waveform = _waveform_classification(rows, mode, config, failure_reason)
        return {
            "cell": config["name"], "G_b": float(gb), "chi": float(chi),
            "actual_G_b": dimensionless_groups(config)["G_b"], "actual_chi": dimensionless_groups(config)["chi"],
            "classification": classification, "waveform_classification": waveform,
            "failure_reason": failure_reason, "onset_time": None, "A_max_over_L": None,
            "first_mode_fraction": None, "curvature_rms": None, "dissipation_energy": 0.0,
            "dominant_mode": None, "mode_spectrum": mode, "accepted_steps": 0, "rejected_steps": 0,
            "metrics_rows": [],
        }, trajectory
    rows, dissipation = _work_and_rows(trajectory, simulator, anchors)
    classification = _classification(rows, config, failure_reason)
    peak_index = max(range(len(rows)), key=lambda index: float(rows[index]["max_transverse_displacement"]))
    peak = rows[peak_index]
    mode = _mode_detail(trajectory[peak_index], anchors)
    waveform = _waveform_classification(rows, mode, config, failure_reason)
    onset_time = classification.get("onset_time")
    _, transverse = _transverse_coordinates(trajectory[peak_index], anchors)
    # Include both the existing mechanical label and the morphology label.  A
    # single-mode morphology is the resolved-buckling candidate; the other
    # labels explain why a low first-mode fraction remains unresolved.
    return {
        "cell": config["name"], "G_b": float(gb), "chi": float(chi),
        "actual_G_b": dimensionless_groups(config)["G_b"], "actual_chi": dimensionless_groups(config)["chi"],
        "classification": classification, "waveform_classification": waveform,
        "failure_reason": failure_reason, "onset_time": onset_time,
        "A_max_over_L": float(np.max(np.abs(transverse)) / float(config["length"])),
        "first_mode_fraction": mode["first_mode_fraction"],
        "curvature_rms": float(peak["rms_curvature"]),
        "dissipation_energy": float(dissipation),
        "dominant_mode": mode["dominant_mode"], "mode_spectrum": mode,
        "accepted_steps": int(simulator.accepted_steps), "rejected_steps": int(simulator.rejected_steps),
        "peak_time": float(peak["time"]), "peak_step": int(peak["step"]),
        "peak_max_transverse_displacement": float(peak["max_transverse_displacement"]),
        "peak_max_curvature": float(peak["max_curvature"]),
        "metrics_rows": rows,
    }, trajectory


def _snapshot(state: FilamentState, label: str) -> dict[str, Any]:
    return {
        "label": label,
        "time": float(state.time),
        "step": int(state.step),
        "x": [float(value) for value in state.positions[:, 0]],
        "y": [float(value) for value in state.positions[:, 1]],
        "n_nodes": int(state.n_nodes),
    }


def _select_representatives(records: Sequence[Mapping[str, Any]], requested: Sequence[str]) -> dict[str, str | None]:
    def first(predicate):
        for record in records:
            if predicate(record):
                return str(record["cell"])
        return None

    selected: dict[str, str | None] = {}
    selected["straight"] = first(lambda item: item["waveform_classification"]["reason"] == "no_onset")
    selected["single_buckling"] = first(lambda item: item["waveform_classification"]["reason"] == "single_mode")
    selected["higher_mode"] = first(lambda item: "higher_mode" in str(item["waveform_classification"]["reason"]))
    selected["boundary_near"] = first(lambda item: item["classification"].get("label") == "unresolved")
    # Guarantee a compact representative artifact even when a particular axis
    # range has no member of one morphology class.
    fallback = str(records[len(records) // 2]["cell"]) if records else None
    for role in requested:
        selected.setdefault(str(role), fallback)
    for role, cell in list(selected.items()):
        if cell is None:
            selected[role] = fallback
    return selected


def run_experiment(config: Mapping[str, Any], output: Path) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    trajectories: dict[str, list[FilamentState]] = {}
    for gb in config["axes"]["G_b"]:
        for chi in config["axes"]["chi"]:
            record, trajectory = _run_cell(config["base"], float(gb), float(chi))
            records.append(record)
            trajectories[str(record["cell"])] = trajectory
    rows = []
    for record in records:
        classification = record["classification"]
        waveform = record["waveform_classification"]
        rows.append({
            "cell": record["cell"], "G_b": record["G_b"], "chi": record["chi"],
            "actual_G_b": record["actual_G_b"], "actual_chi": record["actual_chi"],
            "label": classification.get("label"), "waveform_class": waveform.get("label"),
            "unresolved_reason": waveform.get("reason"), "dominant_mode": record["dominant_mode"],
            "A_max_over_L": record["A_max_over_L"], "first_mode_fraction": record["first_mode_fraction"],
            "curvature_rms": record["curvature_rms"], "onset_time": record["onset_time"],
            "dissipation_energy": record["dissipation_energy"],
            "accepted_steps": record["accepted_steps"], "rejected_steps": record["rejected_steps"],
            "failure_reason": record["failure_reason"],
        })
    fields = list(rows[0].keys()) if rows else []
    _write_csv(output / "heatmap.csv", rows, fields)
    _write_json(output / "heatmap.json", {
        "schema_version": SCHEMA_VERSION,
        "axes": config["axes"],
        "measurement_definitions": {
            "A_max_over_L": "maximum absolute transverse displacement at the peak state divided by configured L",
            "first_mode_fraction": "|A1| / sqrt(sum_n |An|^2), with n=1..6 sine projections",
            "curvature_rms": "RMS of the existing discrete curvature observable at the peak state",
            "onset_time": "first accepted state above max(3 initial |A1|, 0.005 L)",
            "dissipation_energy": "accepted explicit-Euler estimate sum dt*Gamma_i*|v_i|^2; remesh intervals omitted",
        },
        "unresolved_decomposition": {
            "higher_mode_wave": "dominant mode n>=2 and higher-mode fraction >= 0.30",
            "localized_buckling": "peak max curvature / curvature RMS >= 3.0",
            "mixed_mode": "onset with first-mode fraction < 0.70 but neither specific signal alone",
            "no_onset": "peak transverse displacement did not exceed onset threshold",
        },
        "rows": rows,
        "records": [{key: value for key, value in record.items() if key != "metrics_rows"} for record in records],
    })
    selected = _select_representatives(records, config.get("snapshot_roles", []))
    snapshots: dict[str, Any] = {"schema_version": SCHEMA_VERSION, "representatives": selected, "cases": {}}
    for role, cell in selected.items():
        if cell is None or cell not in trajectories:
            continue
        trajectory = trajectories[cell]
        if not trajectory:
            snapshots["cases"][role] = {"cell": cell, "snapshots": [], "status": "no_trajectory"}
            continue
        targets = (0.0, float(trajectory[-1].time) / 2.0, float(trajectory[-1].time))
        chosen = []
        used: set[int] = set()
        for label, target in zip(("t0", "t_mid", "t_end"), targets):
            index = min((i for i in range(len(trajectory)) if i not in used), key=lambda i: abs(float(trajectory[i].time) - target), default=0)
            used.add(index)
            chosen.append(_snapshot(trajectory[index], label))
        snapshots["cases"][role] = {"cell": cell, "snapshots": chosen}
    _write_json(output / "snapshots.json", snapshots)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "benchmark": "dense non-contact buckling heatmap",
        "source_revision": detect_git_revision(Path.cwd()),
        "config": _jsonable(config),
        "condition_count": len(records),
        "axes": config["axes"],
        "artifacts": {},
        "output_policy": config["output_policy"],
    }
    for path in (output / "heatmap.csv", output / "heatmap.json", output / "snapshots.json"):
        manifest["artifacts"][path.name] = {"bytes": path.stat().st_size, "sha256": _sha256(path)}
    manifest["output_bytes"] = sum(item["bytes"] for item in manifest["artifacts"].values())
    manifest["within_budget"] = manifest["output_bytes"] <= int(config["output_policy"]["max_output_bytes"])
    _write_json(output / "manifest.json", manifest)
    if not manifest["within_budget"]:
        raise ValueError("dense heatmap output exceeds configured byte budget")
    return {"schema_version": SCHEMA_VERSION, "condition_count": len(records), "output": str(output), "representatives": selected, "manifest": manifest}


def _cli() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = _cli()
    try:
        config = load_config(args.config)
        result = run_experiment(config, args.output)
    except (OSError, ValueError, json.JSONDecodeError, ModelError) as exc:
        print(f"dense heatmap error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({"output": str(args.output), "conditions": result["condition_count"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
