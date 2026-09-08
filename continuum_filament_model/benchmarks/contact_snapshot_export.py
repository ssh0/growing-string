"""Export compact contact/folding geometry snapshots for presentation data.

The cases are built with the P2 contact benchmark's public runner inputs and
use the same model forces and finite-radius geometry.  Only ``t0``, ``t_mid``
and ``t_end`` coordinates are retained; each active finite-radius pair carries
its closest point, normal, penetration, and penalty normal force.
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
    from contact_buckling_benchmark import (  # type: ignore  # noqa: E402
        _shape_metrics,
        contact_metrics,
        initial_state,
        validate_case_config,
    )
else:
    from .contact_buckling_benchmark import (  # noqa: E402
        _shape_metrics,
        contact_metrics,
        initial_state,
        validate_case_config,
    )

from growing_filament.geometry import nonlocal_segment_contacts  # noqa: E402
from growing_filament.model import FilamentState, ModelError, ModelParameters, OverdampedGrowingFilament  # noqa: E402
from growing_filament.reproducibility import canonical_json_bytes, detect_git_revision  # noqa: E402

SCHEMA_VERSION = "continuum-filament-contact-snapshots-1"
DEFAULT_CASES: tuple[dict[str, Any], ...] = (
    {
        "name": "u_self_contact",
        "role": "U-shaped self-contact",
        "length": 2.0,
        "n_nodes": 7,
        "axial_stiffness": 100.0,
        "bending_stiffness": 0.1,
        "drag_density": 1.0,
        "growth_rate": 0.05,
        "contact_stiffness": 10.0,
        "diameter": 0.35,
        "dt": 0.001,
        "t_end": 0.02,
        "a_max_factor": 8.0,
        "initial_shape": "u",
        "amplitude": 0.04,
    },
    {
        "name": "s_contact_folding",
        "role": "S-shaped contact/folding",
        "length": 2.0,
        "n_nodes": 9,
        "axial_stiffness": 100.0,
        "bending_stiffness": 0.1,
        "drag_density": 1.0,
        "growth_rate": 0.05,
        "contact_stiffness": 10.0,
        "diameter": 0.30,
        "dt": 0.0005,
        "t_end": 0.02,
        "a_max_factor": 4.0,
        "initial_shape": "s",
        "amplitude": 0.20,
    },
)


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
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(_jsonable(row) for row in rows)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_cases(path: Path | None) -> list[dict[str, Any]]:
    if path is None:
        return [dict(item) for item in DEFAULT_CASES]
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping) or not isinstance(raw.get("cases"), list):
        raise ValueError("contact snapshot config requires a cases list")
    return [dict(item) for item in raw["cases"]]


def _contact_force_rows(state: FilamentState, config: Mapping[str, Any]) -> list[dict[str, Any]]:
    diameter = float(config["diameter"])
    stiffness = float(config["contact_stiffness"])
    positions = state.positions
    result: list[dict[str, Any]] = []
    for contact in nonlocal_segment_contacts(positions, diameter):
        if not contact.is_contact or contact.centerline_intersection or contact.normal is None:
            continue
        normal = np.asarray(contact.normal, dtype=float)
        penetration = float(contact.penetration)
        force = stiffness * penetration * normal
        point_i = (1.0 - contact.parameter_i) * positions[contact.segment_i] + contact.parameter_i * positions[contact.segment_i + 1]
        point_j = (1.0 - contact.parameter_j) * positions[contact.segment_j] + contact.parameter_j * positions[contact.segment_j + 1]
        point = 0.5 * (point_i + point_j)
        result.append({
            "segment_i": int(contact.segment_i),
            "segment_j": int(contact.segment_j),
            "distance": float(contact.distance),
            "penetration": penetration,
            "contact_point": point.tolist(),
            "closest_point_i": point_i.tolist(),
            "closest_point_j": point_j.tolist(),
            "normal": normal.tolist(),
            "normal_force": force.tolist(),
            "normal_force_magnitude": float(np.linalg.norm(force)),
        })
    return result


def _snapshot(state: FilamentState, label: str, config: Mapping[str, Any]) -> dict[str, Any]:
    contacts = _contact_force_rows(state, config)
    return {
        "label": label,
        "time": float(state.time),
        "step": int(state.step),
        "n_nodes": int(state.n_nodes),
        "x": state.positions[:, 0].astype(float).tolist(),
        "y": state.positions[:, 1].astype(float).tolist(),
        "contacts": contacts,
        "active_contact_pairs": len(contacts),
        "shape_metrics": _shape_metrics(state, config),
    }


def _run_case(config: Mapping[str, Any]) -> dict[str, Any]:
    effective = dict(config)
    effective.setdefault("fixed_left", True)
    effective.setdefault("fixed_right", True)
    effective.setdefault("reject_crossing", True)
    effective.setdefault("max_retries", 12)
    effective.setdefault("dt_min", 1.0e-10)
    effective.setdefault("max_displacement_fraction", 0.25)
    effective.setdefault("buckling_threshold_fraction", 0.02)
    effective.setdefault("fold_curvature_threshold", 1.0e-8)
    validate_case_config(effective)
    state = initial_state(effective)
    params = ModelParameters(
        axial_stiffness=float(effective["axial_stiffness"]),
        bending_stiffness=float(effective["bending_stiffness"]),
        drag_density=float(effective["drag_density"]),
        contact_stiffness=float(effective["contact_stiffness"]),
        diameter=float(effective["diameter"]),
        growth_rate=float(effective["growth_rate"]),
        reference_length=float(np.mean(state.rest_lengths)),
        dt=float(effective["dt"]),
        t_end=float(effective["t_end"]),
        a_max=float(np.max(state.rest_lengths)) * float(effective["a_max_factor"]),
        dt_min=float(effective["dt_min"]),
        max_retries=int(effective["max_retries"]),
        max_displacement_fraction=float(effective["max_displacement_fraction"]),
        fixed_left=True,
        fixed_right=True,
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
    snapshots: list[dict[str, Any]] = []
    if trajectory:
        targets = (0.0, float(trajectory[-1].time) / 2.0, float(trajectory[-1].time))
        used: set[int] = set()
        for label, target in zip(("t0", "t_mid", "t_end"), targets):
            index = min((i for i in range(len(trajectory)) if i not in used), key=lambda i: abs(float(trajectory[i].time) - target), default=0)
            used.add(index)
            snapshots.append(_snapshot(trajectory[index], label, effective))
    all_contacts = [_contact_force_rows(item, effective) for item in trajectory]
    max_contacts = max((len(item) for item in all_contacts), default=0)
    return {
        "name": str(effective["name"]),
        "role": str(effective.get("role", effective["name"])),
        "effective_config": effective,
        "failure_reason": failure,
        "classification": (
            "folding-contact" if any(int(_shape_metrics(item, effective)["fold_count"]) > 0 and contacts for item, contacts in zip(trajectory, all_contacts))
            else "self-contact" if max_contacts else "folding-no-contact" if any(int(_shape_metrics(item, effective)["fold_count"]) > 0 for item in trajectory)
            else "no-contact"
        ),
        "max_active_contact_pairs": max_contacts,
        "max_penetration": max((float(contact["penetration"]) for contacts in all_contacts for contact in contacts), default=0.0),
        "snapshots": snapshots,
        "accepted_steps": int(simulator.accepted_steps) if simulator else 0,
        "rejected_steps": int(simulator.rejected_steps) if simulator else 0,
    }


def run_export(cases: Sequence[Mapping[str, Any]], output: Path) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    results = [_run_case(case) for case in cases]
    rows = []
    for result in results:
        for snapshot in result["snapshots"]:
            rows.append({
                "case": result["name"], "role": result["role"], "label": snapshot["label"],
                "time": snapshot["time"], "step": snapshot["step"], "n_nodes": snapshot["n_nodes"],
                "active_contact_pairs": snapshot["active_contact_pairs"],
                "max_penetration": max((float(item["penetration"]) for item in snapshot["contacts"]), default=0.0),
                "classification": result["classification"], "failure_reason": result["failure_reason"],
            })
    _write_csv(output / "contact_snapshots.csv", rows, list(rows[0].keys()) if rows else [])
    payload = {
        "schema_version": SCHEMA_VERSION,
        "measurement_definitions": {
            "contact_point": "midpoint of the two segment closest points",
            "normal": "unit vector from segment j to segment i returned by nonlocal_segment_contacts",
            "normal_force": "k_c * penetration * normal; centerline intersections are excluded",
            "snapshot_times": "nearest accepted states to t=0, t_end/2, and t_end",
        },
        "cases": results,
    }
    _write_json(output / "contact_snapshots.json", payload)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "source_revision": detect_git_revision(Path.cwd()),
        "case_order": [item["name"] for item in results],
        "artifacts": {},
    }
    for path in (output / "contact_snapshots.csv", output / "contact_snapshots.json"):
        manifest["artifacts"][path.name] = {"bytes": path.stat().st_size, "sha256": _sha256(path)}
    manifest["output_bytes"] = sum(item["bytes"] for item in manifest["artifacts"].values())
    _write_json(output / "manifest.json", manifest)
    return {"cases": len(results), "output": str(output), "manifest": manifest}


def _cli() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = _cli()
    try:
        result = run_export(load_cases(args.config), args.output)
    except (OSError, ValueError, json.JSONDecodeError, ModelError) as exc:
        print(f"contact snapshot export error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({"output": result["output"], "cases": result["cases"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
