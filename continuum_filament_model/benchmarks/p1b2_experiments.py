"""P1B.2 convergence, regime-map, and seeded robustness experiments.

This is an experiment harness, not a second filament solver.  Every run is
constructed through :mod:`buckling_benchmark` and the public
``growing_filament`` API.  The compact output policy deliberately omits per-run
trajectory archives and plots while retaining the per-run configuration,
metrics, events, summary, and manifest.

Example::

    PYTHONPATH=continuum_filament_model/src \
      python continuum_filament_model/benchmarks/p1b2_experiments.py \
      --config continuum_filament_model/benchmarks/configs/p1b2_noncontact.json \
      --output continuum_filament_model/results/p1b2

The output is a small, bounded non-contact experiment.  It must not be read as
an experiment-noise model: only the initial imperfection is seeded.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from collections import Counter
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
        CaseSpec,
        BenchmarkError,
        _jsonable,
        _write_json,
        dimensionless_groups,
        run_case,
    )
else:
    from .buckling_benchmark import (  # noqa: E402
        CaseSpec,
        BenchmarkError,
        _jsonable,
        _write_json,
        dimensionless_groups,
        run_case,
    )


SCHEMA_VERSION = "continuum-filament-p1b2-1"
DEFAULT_TOLERANCES: dict[str, float] = {
    "onset_time_relative": 0.10,
    "peak_max_transverse_relative": 0.10,
    "peak_max_transverse_absolute_fraction_of_length": 0.002,
}
DEFAULT_OUTPUT_POLICY: dict[str, Any] = {
    "save_trajectories": False,
    "save_per_run_plots": False,
    "max_runs": 100,
    "max_output_bytes": 120_000_000,
}


# ---------------------------------------------------------------------------
# Small serialization and tabulation helpers


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = list(rows[0].keys()) if rows else []
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=list(fields),
            extrasaction="ignore",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(_jsonable(row) for row in rows)


def _safe_token(value: float) -> str:
    return f"{float(value):.8g}".replace("-", "m").replace(".", "p")


def _stats(values: Sequence[float], denominator: int) -> dict[str, Any]:
    finite = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    if not finite:
        return {
            "count": 0,
            "missing_count": int(denominator),
            "median": None,
            "iqr": None,
            "mean": None,
            "std": None,
        }
    array = np.asarray(finite, dtype=float)
    q25, q75 = np.percentile(array, [25.0, 75.0])
    return {
        "count": int(len(array)),
        "missing_count": int(max(0, denominator - len(array))),
        "median": float(np.median(array)),
        "iqr": float(q75 - q25),
        "mean": float(np.mean(array)),
        "std": float(np.std(array, ddof=1)) if len(array) > 1 else 0.0,
    }


def _run_path(summary: Mapping[str, Any], root: Path) -> str:
    """Return a stable relative path for a run summary/manifest pair."""

    # ``run_case`` stores the case name below the caller-supplied root.  The
    # path is supplied by callers in metadata, so this fallback remains useful
    # for compact indexes that only have summary fields.
    return str(root / str(summary["case"]))


def _trial_metric_values(summary: Mapping[str, Any]) -> dict[str, float | None]:
    classification = summary.get("classification", {})
    peak = summary.get("peak_observables") or {}
    return {
        "onset_time": classification.get("onset_time"),
        "peak_max_transverse_displacement": classification.get("peak_max_transverse_displacement"),
        "peak_first_mode_fraction": peak.get("first_mode_fraction"),
        "peak_max_curvature": peak.get("max_curvature"),
        "peak_energy_total": peak.get("energy_total"),
    }


def aggregate_trials(summaries: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Aggregate seeded runs without folding in the deterministic fixture."""

    values = [item for item in summaries if int(item.get("trial", 0)) > 0]
    denominator = len(values)
    labels = Counter(str(item.get("classification", {}).get("label", "unresolved")) for item in values)
    reasons: Counter[str] = Counter()
    for item in values:
        if labels.get("unresolved", 0) and item.get("classification", {}).get("label") == "unresolved":
            reason = item.get("failure_reason") or item.get("classification", {}).get("failure_reason")
            reasons[str(reason or "classification_rule_unresolved")] += 1
    metric_values: dict[str, list[float]] = {}
    for item in values:
        for key, value in _trial_metric_values(item).items():
            if value is not None and math.isfinite(float(value)):
                metric_values.setdefault(key, []).append(float(value))
    statistics = {
        key: _stats(metric_values.get(key, []), denominator)
        for key in (
            "onset_time",
            "peak_max_transverse_displacement",
            "peak_first_mode_fraction",
            "peak_max_curvature",
            "peak_energy_total",
        )
    }
    fractions = {
        label: (float(labels.get(label, 0)) / denominator if denominator else None)
        for label in ("straight", "buckled-single", "unresolved")
    }
    if labels.get("unresolved", 0) > 0:
        regime = "numerically-unresolved"
    elif labels.get("straight", 0) == denominator and denominator:
        regime = "resolved-straight"
    elif labels.get("buckled-single", 0) == denominator and denominator:
        regime = "resolved-buckled"
    elif denominator:
        regime = "trial-mixed"
    else:
        regime = "numerically-unresolved"
    return {
        "n_trials": denominator,
        "denominator": denominator,
        "seeds": [item.get("seed") for item in values],
        "label_counts": {label: int(labels.get(label, 0)) for label in ("straight", "buckled-single", "unresolved")},
        "label_fractions": fractions,
        "regime": regime,
        "unresolved_reasons": dict(sorted(reasons.items())),
        "statistics": statistics,
    }


def _flatten_trial_row(cell_id: str, groups: Mapping[str, Any], aggregate: Mapping[str, Any]) -> dict[str, Any]:
    stats = aggregate["statistics"]
    row: dict[str, Any] = {
        "cell": cell_id,
        "G_b": groups.get("G_b"),
        "chi": groups.get("chi", groups.get("bending_to_stretching")),
        "regime": aggregate.get("regime"),
        "n_trials": aggregate.get("n_trials"),
        "denominator": aggregate.get("denominator"),
        "straight_count": aggregate["label_counts"].get("straight", 0),
        "buckled_single_count": aggregate["label_counts"].get("buckled-single", 0),
        "unresolved_count": aggregate["label_counts"].get("unresolved", 0),
        "straight_fraction": aggregate["label_fractions"].get("straight"),
        "buckled_single_fraction": aggregate["label_fractions"].get("buckled-single"),
        "unresolved_fraction": aggregate["label_fractions"].get("unresolved"),
        "unresolved_reasons": json.dumps(aggregate.get("unresolved_reasons", {}), sort_keys=True),
    }
    for metric, values in stats.items():
        prefix = metric.replace("peak_max_transverse_displacement", "peak_max_transverse").replace("peak_first_mode_fraction", "peak_first_mode_fraction")
        for name in ("count", "missing_count", "median", "iqr", "mean", "std"):
            row[f"{prefix}_{name}"] = values.get(name)
    return row


# ---------------------------------------------------------------------------
# Configuration and target construction


def load_config(path: Path) -> dict[str, Any]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        raise BenchmarkError("P1B.2 config root must be an object")
    config = dict(raw)
    config.setdefault("tolerances", dict(DEFAULT_TOLERANCES))
    tolerances = dict(DEFAULT_TOLERANCES)
    tolerances.update(dict(config["tolerances"]))
    config["tolerances"] = tolerances
    policy = dict(DEFAULT_OUTPUT_POLICY)
    policy.update(dict(config.get("output_policy", {})))
    config["output_policy"] = policy
    base = dict(config.get("base", {}))
    required = ("length", "n_nodes", "axial_stiffness", "bending_stiffness", "drag_density", "amplitude", "dt", "t_end", "a_max_factor")
    missing = [key for key in required if key not in base]
    if missing:
        raise BenchmarkError(f"P1B.2 base is missing: {', '.join(missing)}")
    base.update({"contact_stiffness": 0.0, "diameter": 0.0, "fixed_left": True, "fixed_right": True, "reject_crossing": True})
    config["base"] = base
    seeds = list(config.get("grid", {}).get("seeds", []))
    if len(seeds) < 5:
        raise BenchmarkError("P1B.2 requires at least five distinct grid seeds")
    if len(set(int(seed) for seed in seeds)) != len(seeds):
        raise BenchmarkError("grid seeds must be unique")
    if int(policy["max_runs"]) <= 0:
        raise BenchmarkError("output_policy.max_runs must be positive")
    return config


def target_overrides(base: Mapping[str, Any], target_gb: float, target_chi: float, *, length: float | None = None) -> dict[str, Any]:
    """Convert target ``(G_b, chi)`` coordinates to physical parameters."""

    L = float(base["length"] if length is None else length)
    ea = float(base["axial_stiffness"])
    zeta = float(base["drag_density"])
    chi = float(target_chi)
    gb = float(target_gb)
    ei = chi * ea * L * L
    tau_b = zeta * L**4 / (ei * np.pi**4)
    if ei <= 0.0 or not math.isfinite(tau_b):
        raise BenchmarkError("target chi produced an invalid bending stiffness")
    return {
        "length": L,
        "bending_stiffness": ei,
        "growth_rate": gb / tau_b,
        "target_G_b": gb,
        "target_chi": chi,
    }


def _pilot_specs(config: Mapping[str, Any]) -> list[tuple[CaseSpec, str]]:
    specs: list[tuple[CaseSpec, str]] = []
    for item in config.get("pilot", []):
        if not isinstance(item, Mapping) or "name" not in item:
            raise BenchmarkError("pilot entries require name")
        overrides = dict(item.get("overrides", {}))
        role = str(item.get("role", item["name"]))
        specs.append((CaseSpec(str(item["name"]), overrides), role))
    if not specs:
        raise BenchmarkError("P1B.2 requires pilot cases")
    return specs


def _compact_run(case: CaseSpec, base: Mapping[str, Any], output_root: Path, revision: str | None) -> dict[str, Any]:
    return run_case(
        case,
        base,
        output_root,
        revision,
        save_trajectory_file=False,
        write_plot=False,
        compact_events=True,
        metrics_max_rows=64,
    )


def _write_manifest_index(path: Path, records: Sequence[Mapping[str, Any]]) -> None:
    rows = []
    for record in records:
        row = {
            "run_id": record.get("run_id"),
            "section": record.get("section"),
            "cell": record.get("cell"),
            "case": record.get("case"),
            "trial": record.get("trial"),
            "seed": record.get("seed"),
            "manifest": record.get("manifest"),
            "initial_state_hash": record.get("initial_state_hash"),
            "canonical_state_hash": record.get("canonical_state_hash"),
        }
        rows.append(row)
    _write_json(path.with_suffix(".json"), rows)
    _write_csv(path.with_suffix(".csv"), rows)


def _collect_manifest_index(output: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for manifest_path in sorted(output.rglob("manifest.json")):
        if manifest_path.parent == output:
            continue
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        relative = manifest_path.relative_to(output)
        parts = relative.parts
        section = parts[0] if parts else "unknown"
        rows.append({
            "run_id": str(relative.parent),
            "section": section,
            "cell": parts[1] if section == "grid" and len(parts) > 1 else None,
            "case": parts[-2] if len(parts) >= 2 else None,
            "trial": manifest.get("trial", 0),
            "seed": manifest.get("seed"),
            "manifest": str(relative),
            "initial_state_hash": manifest.get("initial_state_hash"),
            "canonical_state_hash": manifest.get("canonical_state_hash"),
        })
    return rows


# ---------------------------------------------------------------------------
# Pilot and convergence


def run_pilot(config: Mapping[str, Any], output: Path, revision: str | None) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    root = output / "pilot"
    summaries: list[dict[str, Any]] = []
    records: list[dict[str, Any]] = []
    for spec, role in _pilot_specs(config):
        spec = CaseSpec(spec.name, {**spec.overrides, "pilot_role": role})
        summary = _compact_run(spec, config["base"], root, revision)
        summaries.append(summary)
        manifest_path = root / spec.name / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        records.append({
            "name": spec.name,
            "role": role,
            "label": summary["classification"].get("label"),
            "onset_time": summary["classification"].get("onset_time"),
            "peak_max_transverse_displacement": summary["classification"].get("peak_max_transverse_displacement"),
            "G_b": summary["dimensionless_groups"].get("G_b"),
            "chi": summary["dimensionless_groups"].get("chi"),
            "manifest": str(manifest_path.relative_to(output)),
            "initial_state_hash": manifest.get("initial_state_hash"),
        })
    _write_csv(output / "pilot_summary.csv", records)
    _write_json(output / "pilot_summary.json", {"selection_rule": "pilot cases are executed before convergence; roles are recorded without relabeling the measured classification", "cases": records})
    _plot_pilot(output / "pilot_summary.png", records)
    _write_manifest_index(output / "pilot_manifest_index", [
        {**record, "section": "pilot", "case": record["name"], "run_id": f"pilot/{record['name']}", "trial": 0, "seed": None, "canonical_state_hash": None, "manifest": record["manifest"]}
        for record in records
    ])
    return summaries, {"cases": records}


def _convergence_points(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    pilots = {spec.name: (spec, role) for spec, role in _pilot_specs(config)}
    points = config.get("convergence", [])
    if not points:
        raise BenchmarkError("P1B.2 requires convergence representative points")
    result: list[dict[str, Any]] = []
    for item in points:
        if not isinstance(item, Mapping) or "name" not in item or "pilot" not in item:
            raise BenchmarkError("convergence entries require name and pilot")
        pilot_name = str(item["pilot"])
        if pilot_name not in pilots:
            raise BenchmarkError(f"convergence references unknown pilot: {pilot_name}")
        n_nodes = [int(value) for value in item.get("n_nodes", [])]
        dt_values = [float(value) for value in item.get("dt_values", [])]
        if len(n_nodes) < 3 or len(set(n_nodes)) < 3:
            raise BenchmarkError(f"convergence point {item['name']} requires three spatial resolutions")
        if len(dt_values) < 2 or not any(abs(a - 0.5 * b) <= 1.0e-14 * max(1.0, abs(a), abs(b)) for a in dt_values for b in dt_values):
            raise BenchmarkError(f"convergence point {item['name']} requires dt and its half")
        base_spec, role = pilots[pilot_name]
        overrides = dict(base_spec.overrides)
        overrides.update(dict(item.get("overrides", {})))
        result.append({"name": str(item["name"]), "pilot": pilot_name, "pilot_role": role, "overrides": overrides, "n_nodes": n_nodes, "dt_values": dt_values})
    return result


def _relative_difference(value: float | None, reference: float | None, absolute_floor: float) -> float | None:
    if value is None or reference is None:
        return None
    return abs(float(value) - float(reference)) / max(abs(float(reference)), absolute_floor)


def run_convergence(config: Mapping[str, Any], output: Path, revision: str | None) -> dict[str, Any]:
    root = output / "convergence"
    run_rows: list[dict[str, Any]] = []
    point_results: list[dict[str, Any]] = []
    tolerances = dict(config["tolerances"])
    for point in _convergence_points(config):
        point_root = root / point["name"]
        summaries: list[dict[str, Any]] = []
        for n_nodes in point["n_nodes"]:
            for dt in point["dt_values"]:
                name = f"n{n_nodes:03d}_dt_{_safe_token(dt)}"
                case = CaseSpec(name, {**point["overrides"], "n_nodes": n_nodes, "dt": dt, "convergence_point": point["name"]})
                summary = _compact_run(case, config["base"], point_root, revision)
                summaries.append(summary)
                classification = summary["classification"]
                run_rows.append({
                    "representative": point["name"],
                    "pilot": point["pilot"],
                    "pilot_role": point["pilot_role"],
                    "case": name,
                    "n_nodes": n_nodes,
                    "dt": dt,
                    "label": classification.get("label"),
                    "onset_time": classification.get("onset_time"),
                    "peak_max_transverse_displacement": classification.get("peak_max_transverse_displacement"),
                    "G_b": summary["dimensionless_groups"].get("G_b"),
                    "chi": summary["dimensionless_groups"].get("chi"),
                    "failure_reason": summary.get("failure_reason"),
                    "manifest": str((point_root / name / "manifest.json").relative_to(output)),
                })
        reference = max(summaries, key=lambda item: (int(item["effective_config"]["n_nodes"]), -float(item["effective_config"]["dt"])))
        ref_class = reference["classification"]
        labels = [str(item["classification"].get("label")) for item in summaries]
        class_stable = len(set(labels)) == 1 and labels[0] != "unresolved"
        peaks = [item["classification"].get("peak_max_transverse_displacement") for item in summaries]
        peak_ref = ref_class.get("peak_max_transverse_displacement")
        abs_floor = float(tolerances["peak_max_transverse_absolute_fraction_of_length"]) * float(config["base"]["length"])
        peak_diffs = [_relative_difference(value, peak_ref, abs_floor) for value in peaks]
        finite_peak_diffs = [value for value in peak_diffs if value is not None]
        peak_max_diff = max(finite_peak_diffs) if finite_peak_diffs else None
        onsets = [item["classification"].get("onset_time") for item in summaries]
        onset_ref = ref_class.get("onset_time")
        onset_diffs = [_relative_difference(value, onset_ref, 1.0e-12) for value in onsets]
        onset_missing_mismatch = any(value is None for value in onsets) != all(value is None for value in onsets)
        finite_onset_diffs = [value for value in onset_diffs if value is not None]
        onset_max_diff = max(finite_onset_diffs) if finite_onset_diffs else 0.0 if not onset_missing_mismatch else None
        onset_stable = not onset_missing_mismatch and (onset_max_diff is None or onset_max_diff <= float(tolerances["onset_time_relative"]))
        peak_stable = peak_max_diff is not None and peak_max_diff <= float(tolerances["peak_max_transverse_relative"])
        converged = bool(class_stable and onset_stable and peak_stable)
        result = {
            "representative": point["name"],
            "pilot": point["pilot"],
            "pilot_role": point["pilot_role"],
            "pilot_selection_recorded": True,
            "n_runs": len(summaries),
            "n_nodes_values": point["n_nodes"],
            "dt_values": point["dt_values"],
            "labels": labels,
            "classification_stable": class_stable,
            "onset_time_max_relative_difference": onset_max_diff,
            "onset_time_within_tolerance": onset_stable,
            "peak_max_transverse_max_relative_difference": peak_max_diff,
            "peak_max_transverse_within_tolerance": peak_stable,
            "tolerances": tolerances,
            "status": "converged" if converged else "numerically-unresolved",
            "reference_case": reference["case"],
            "reference_label": ref_class.get("label"),
            "reference_onset_time": onset_ref,
            "reference_peak_max_transverse_displacement": peak_ref,
            "runs": run_rows[-len(summaries):],
        }
        point_results.append(result)
    summary_rows = []
    for result in point_results:
        summary_rows.append({
            "representative": result["representative"],
            "pilot": result["pilot"],
            "pilot_role": result["pilot_role"],
            "n_runs": result["n_runs"],
            "n_nodes_values": json.dumps(result["n_nodes_values"]),
            "dt_values": json.dumps(result["dt_values"]),
            "labels": json.dumps(result["labels"]),
            "classification_stable": result["classification_stable"],
            "onset_time_max_relative_difference": result["onset_time_max_relative_difference"],
            "onset_time_within_tolerance": result["onset_time_within_tolerance"],
            "peak_max_transverse_max_relative_difference": result["peak_max_transverse_max_relative_difference"],
            "peak_max_transverse_within_tolerance": result["peak_max_transverse_within_tolerance"],
            "status": result["status"],
            "reference_case": result["reference_case"],
            "reference_label": result["reference_label"],
            "reference_onset_time": result["reference_onset_time"],
            "reference_peak_max_transverse_displacement": result["reference_peak_max_transverse_displacement"],
        })
    _write_csv(output / "convergence_runs.csv", run_rows)
    _write_csv(output / "convergence_summary.csv", summary_rows)
    _write_json(output / "convergence_summary.json", {"schema_version": SCHEMA_VERSION, "tolerances": tolerances, "representatives": point_results})
    _plot_convergence(output / "convergence_summary.png", run_rows, point_results)
    return {"runs": run_rows, "representatives": point_results}


# ---------------------------------------------------------------------------
# 3x3 regime map and seeded trials


def run_grid(config: Mapping[str, Any], output: Path, revision: str | None) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    grid = config["grid"]
    gb_values = [float(value) for value in grid.get("G_b", [])]
    chi_values = [float(value) for value in grid.get("chi", [])]
    if len(gb_values) < 3 or len(chi_values) < 3:
        raise BenchmarkError("P1B.2 regime grid requires at least 3x3 G_b x chi values")
    seeds = [int(value) for value in grid["seeds"]]
    grid_base = dict(config["base"])
    if "n_nodes" in grid:
        grid_base["n_nodes"] = int(grid["n_nodes"])
    if "dt" in grid:
        grid_base["dt"] = float(grid["dt"])
    if "t_end" in grid:
        grid_base["t_end"] = float(grid["t_end"])
    all_summaries: list[dict[str, Any]] = []
    map_rows: list[dict[str, Any]] = []
    trial_rows: list[dict[str, Any]] = []
    for i, gb in enumerate(gb_values, start=1):
        for j, chi in enumerate(chi_values, start=1):
            cell = f"gb_{i:02d}_chi_{j:02d}"
            overrides = target_overrides(grid_base, gb, chi)
            cell_root = output / "grid" / cell
            deterministic = _compact_run(CaseSpec("deterministic", overrides), grid_base, cell_root, revision)
            all_summaries.append(deterministic)
            trial_summaries: list[dict[str, Any]] = []
            for trial, seed in enumerate(seeds, start=1):
                trial_overrides = {**overrides, "trial_noise_fraction": float(grid.get("noise_fraction", 0.05)), "grid_cell": cell}
                trial_case = CaseSpec(f"trial_{trial:02d}_seed_{seed}", trial_overrides, seed=seed, trial=trial, base_name=cell)
                trial_summary = _compact_run(trial_case, grid_base, cell_root, revision)
                trial_summaries.append(trial_summary)
                all_summaries.append(trial_summary)
            aggregate = aggregate_trials(trial_summaries)
            groups = deterministic["dimensionless_groups"]
            row = _flatten_trial_row(cell, groups, aggregate)
            row.update({
                "grid_index_gb": i,
                "grid_index_chi": j,
                "target_G_b": gb,
                "target_chi": chi,
                "actual_G_b": groups.get("G_b"),
                "actual_chi": groups.get("chi"),
                "deterministic_label": deterministic["classification"].get("label"),
                "deterministic_onset_time": deterministic["classification"].get("onset_time"),
                "deterministic_peak_max_transverse_displacement": deterministic["classification"].get("peak_max_transverse_displacement"),
                "deterministic_failure_reason": deterministic.get("failure_reason"),
                "deterministic_manifest": str((cell_root / "deterministic" / "manifest.json").relative_to(output)),
            })
            map_rows.append(row)
            trial_rows.append({**row, "aggregate": aggregate})
    _write_csv(output / "regime_map.csv", map_rows)
    _write_json(output / "regime_map.json", {
        "schema_version": SCHEMA_VERSION,
        "axes": {"G_b": gb_values, "chi": chi_values},
        "classification": {
            "resolved_straight": "all seeded trials are straight",
            "resolved_buckled": "all seeded trials are buckled-single",
            "trial_mixed": "at least one straight and one buckled-single, with no unresolved trial",
            "numerically_unresolved": "at least one unresolved/failing seeded trial or missing denominator",
        },
        "rows": trial_rows,
    })
    _write_csv(output / "trial_summary.csv", map_rows)
    _write_json(output / "trial_summary.json", {"schema_version": SCHEMA_VERSION, "cells": trial_rows})
    _plot_regime_map(output / "regime_map.png", map_rows)
    return all_summaries, map_rows, trial_rows


# ---------------------------------------------------------------------------
# Optional two-length comparison


def run_size_comparison(config: Mapping[str, Any], output: Path, revision: str | None) -> list[dict[str, Any]]:
    size = config.get("size_comparison")
    if not size or not bool(size.get("enabled", True)):
        return []
    lengths = [float(value) for value in size.get("lengths", [])]
    points = list(size.get("points", []))
    seeds = [int(value) for value in size.get("seeds", [])]
    if not lengths or not points:
        raise BenchmarkError("size_comparison requires lengths and points")
    base = dict(config["base"])
    rows: list[dict[str, Any]] = []
    for length in lengths:
        sized_base = dict(base)
        sized_base["length"] = length
        sized_base["n_nodes"] = int(size.get("n_nodes", base["n_nodes"]))
        sized_base["dt"] = float(size.get("dt", base["dt"]))
        sized_base["t_end"] = float(size.get("t_end", base["t_end"]))
        for index, point in enumerate(points, start=1):
            name = str(point.get("name", f"point_{index:02d}"))
            gb = float(point["G_b"])
            chi = float(point["chi"])
            overrides = target_overrides(sized_base, gb, chi)
            root = output / "size_comparison" / f"L_{_safe_token(length)}" / name
            deterministic = _compact_run(CaseSpec("deterministic", overrides), sized_base, root, revision)
            trial_summaries: list[dict[str, Any]] = []
            for trial, seed in enumerate(seeds, start=1):
                trial_case = CaseSpec(
                    f"trial_{trial:02d}_seed_{seed}",
                    {**overrides, "trial_noise_fraction": float(size.get("noise_fraction", 0.05))},
                    seed=seed,
                    trial=trial,
                    base_name=name,
                )
                trial_summaries.append(_compact_run(trial_case, sized_base, root, revision))
            aggregate = aggregate_trials(trial_summaries)
            rows.append({
                "length": length,
                "name": name,
                "target_G_b": gb,
                "target_chi": chi,
                "deterministic_label": deterministic["classification"].get("label"),
                "deterministic_onset_time": deterministic["classification"].get("onset_time"),
                "deterministic_peak_max_transverse_displacement": deterministic["classification"].get("peak_max_transverse_displacement"),
                "trial_regime": aggregate["regime"],
                "trial_label_counts": json.dumps(aggregate["label_counts"], sort_keys=True),
                "trial_statistics": json.dumps(aggregate["statistics"], sort_keys=True),
                "n_trials": aggregate["n_trials"],
                "denominator": aggregate["denominator"],
            })
    _write_csv(output / "size_comparison.csv", rows)
    _write_json(output / "size_comparison.json", {
        "schema_version": SCHEMA_VERSION,
        "description": "same target G_b and chi at a second initial length; non-contact deterministic fixture plus seeded initial imperfection trials",
        "rows": rows,
    })
    _plot_size_comparison(output / "size_comparison.png", rows)
    return rows


# ---------------------------------------------------------------------------
# Plots (optional matplotlib dependency; data artifacts remain authoritative)


def _matplotlib():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        return plt
    except Exception:
        return None


def _plot_pilot(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    plt = _matplotlib()
    if plt is None or not rows:
        return
    fig, ax = plt.subplots(figsize=(7, 4))
    for row in rows:
        ax.scatter(row["G_b"], row["peak_max_transverse_displacement"], label=f"{row['role']}: {row['label']}")
    ax.set(xlabel="G_b", ylabel="peak max |y|", title="P1B.2 pilot representative points")
    ax.legend(fontsize="small")
    fig.savefig(path, dpi=140)
    plt.close(fig)


def _plot_convergence(path: Path, runs: Sequence[Mapping[str, Any]], points: Sequence[Mapping[str, Any]]) -> None:
    plt = _matplotlib()
    if plt is None or not runs:
        return
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    for representative in [str(point["representative"]) for point in points]:
        values = [row for row in runs if row["representative"] == representative]
        values.sort(key=lambda row: (row["n_nodes"], row["dt"]))
        x = np.arange(len(values))
        axes[0].plot(x, [row["peak_max_transverse_displacement"] or np.nan for row in values], "o-", label=representative)
        axes[1].plot(x, [row["onset_time"] or np.nan for row in values], "o-", label=representative)
        axes[0].set_xticks(x, [f"n={row['n_nodes']}\ndt={row['dt']:.1g}" for row in values], rotation=35, ha="right", fontsize="small")
        axes[1].set_xticks(x, [f"n={row['n_nodes']}\ndt={row['dt']:.1g}" for row in values], rotation=35, ha="right", fontsize="small")
    axes[0].set(ylabel="peak max |y|", title="spatial / dt sensitivity")
    axes[1].set(ylabel="onset time", title="buckling onset sensitivity")
    axes[0].legend(fontsize="small")
    fig.savefig(path, dpi=140)
    plt.close(fig)


def _plot_regime_map(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    plt = _matplotlib()
    if plt is None or not rows:
        return
    colors = {"resolved-straight": "tab:blue", "resolved-buckled": "tab:orange", "trial-mixed": "tab:purple", "numerically-unresolved": "tab:red"}
    fig, ax = plt.subplots(figsize=(7, 5))
    for row in rows:
        ax.scatter(row["G_b"], row["chi"], s=850, marker="s", color=colors.get(row["regime"], "black"), edgecolor="black")
        ax.text(row["G_b"], row["chi"], f"{row['straight_count']}/{row['buckled_single_count']}/{row['unresolved_count']}", ha="center", va="center", fontsize=8)
    ax.set(xlabel="G_b = g tau_b", ylabel="chi = EI/(EA L^2)", title="P1B.2 non-contact regime map\ncounts: straight / buckled-single / unresolved")
    ax.set_yscale("log")
    handles = [plt.Line2D([0], [0], marker="s", color="w", markerfacecolor=color, markeredgecolor="black", label=label, markersize=10) for label, color in colors.items()]
    ax.legend(handles=handles, fontsize="small", loc="best")
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_size_comparison(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    plt = _matplotlib()
    if plt is None or not rows:
        return
    fig, ax = plt.subplots(figsize=(8, 4))
    labels = [f"L={row['length']}\n{row['name']}" for row in rows]
    values = [row["deterministic_peak_max_transverse_displacement"] or 0.0 for row in rows]
    ax.bar(labels, values, color="tab:green")
    ax.set(ylabel="deterministic peak max |y|", title="same target (G_b, chi) at a second length")
    ax.tick_params(axis="x", rotation=25)
    fig.savefig(path, dpi=140)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Orchestration and CLI


def _planned_runs(config: Mapping[str, Any], mode: str) -> int:
    count = 0
    if mode in {"full", "pilot"}:
        count += len(list(_pilot_specs(config)))
    if mode in {"full", "convergence"}:
        count += sum(len(item.get("n_nodes", [])) * len(item.get("dt_values", [])) for item in config.get("convergence", []))
    if mode in {"full", "grid"}:
        count += len(config.get("grid", {}).get("G_b", [])) * len(config.get("grid", {}).get("chi", [])) * (1 + len(config.get("grid", {}).get("seeds", [])))
    if mode == "full" and config.get("size_comparison", {}).get("enabled", True):
        size = config.get("size_comparison", {})
        count += len(size.get("lengths", [])) * len(size.get("points", [])) * (1 + len(size.get("seeds", [])))
    return count


def _directory_size(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def run_experiment(config: Mapping[str, Any], output: Path, mode: str = "full") -> dict[str, Any]:
    if mode not in {"full", "pilot", "convergence", "grid"}:
        raise BenchmarkError(f"unknown P1B.2 mode: {mode}")
    planned = _planned_runs(config, mode)
    policy = dict(config["output_policy"])
    if planned > int(policy["max_runs"]):
        raise BenchmarkError(f"planned run count {planned} exceeds max_runs={policy['max_runs']}")
    output.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    revision = None
    # detect_git_revision is intentionally obtained through the existing runner
    # in each case; this avoids a second revision implementation here.
    from growing_filament.reproducibility import detect_git_revision
    revision = detect_git_revision(Path.cwd())
    result: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "mode": mode,
        "git_revision": revision,
        "planned_runs": planned,
        "output_policy": policy,
        "tolerances": config["tolerances"],
        "deterministic_fixture_and_seeded_trials_are_separate": True,
        "seed_scope": "numpy.default_rng(seed) initial imperfection only; no stochastic dynamics or experimental noise model",
    }
    if mode in {"full", "pilot"}:
        _, result["pilot"] = run_pilot(config, output, revision)
    if mode in {"full", "convergence"}:
        result["convergence"] = run_convergence(config, output, revision)
    if mode in {"full", "grid"}:
        _, result["regime_map"], result["trial_summary"] = run_grid(config, output, revision)
    if mode == "full":
        result["size_comparison"] = run_size_comparison(config, output, revision)
    result["elapsed_seconds"] = float(time.monotonic() - started)
    manifest_rows = _collect_manifest_index(output)
    _write_manifest_index(output / "manifest_index", manifest_rows)
    result["manifest_count"] = len(manifest_rows)
    result["output_bytes"] = _directory_size(output)
    result["output_within_budget"] = result["output_bytes"] <= int(policy["max_output_bytes"])
    if not result["output_within_budget"]:
        raise BenchmarkError(f"P1B.2 output exceeds max_output_bytes={policy['max_output_bytes']}")
    _write_json(output / "experiment_summary.json", result)
    _write_json(output / "effective_config.json", config)
    return result


def _cli() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True, help="P1B.2 JSON experiment config")
    parser.add_argument("--output", type=Path, required=True, help="bounded output directory")
    parser.add_argument("--mode", choices=("full", "pilot", "convergence", "grid"), default="full")
    return parser.parse_args()


def main() -> int:
    args = _cli()
    try:
        config = load_config(args.config)
        result = run_experiment(config, args.output, mode=args.mode)
    except (BenchmarkError, OSError, json.JSONDecodeError, ValueError) as exc:
        print(f"P1B.2 configuration/output error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({
        "output": str(args.output),
        "mode": args.mode,
        "planned_runs": result["planned_runs"],
        "elapsed_seconds": result["elapsed_seconds"],
        "output_bytes": result["output_bytes"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
