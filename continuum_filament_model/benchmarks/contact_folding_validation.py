"""C1 frictionless finite-radius contact/folding validation harness.

This runner is intentionally separate from the historical P2 contact-buckling
suite.  C1 is the segment-contact law with the legacy node penalty disabled.
It records compact contact lineage and refinement diagnostics; it never writes
trajectories or videos and never treats finite penalty as a non-penetration
proof.
"""

from __future__ import annotations

import argparse
import csv
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


SCHEMA_VERSION = "continuum-filament-contact-folding-c1-1"


@dataclass(frozen=True)
class CaseSpec:
    name: str
    group: str
    overrides: dict[str, Any]
    expected_contact: bool = False


class ValidationError(ValueError):
    """Invalid C1 validation input."""


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


def _positive(config: Mapping[str, Any], key: str) -> float:
    try:
        value = float(config[key])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValidationError(f"{key} must be finite") from exc
    if not math.isfinite(value) or value <= 0.0:
        raise ValidationError(f"{key} must be positive")
    return value


def validate_case_config(config: Mapping[str, Any]) -> None:
    for key in (
        "length",
        "axial_stiffness",
        "bending_stiffness",
        "drag_density",
        "dt",
        "t_end",
        "a_max_factor",
        "rest_length_factor",
    ):
        _positive(config, key)
    try:
        n_nodes = int(config["n_nodes"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValidationError("n_nodes must be an integer >= 3") from exc
    if n_nodes < 3 or float(config["n_nodes"]) != n_nodes:
        raise ValidationError("n_nodes must be an integer >= 3")
    for key in ("growth_rate", "contact_stiffness", "diameter"):
        try:
            value = float(config[key])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValidationError(f"{key} must be finite") from exc
        if not math.isfinite(value) or value < 0.0:
            raise ValidationError(f"{key} must be non-negative and finite")
    if not 0.0 < float(config.get("max_displacement_fraction", 0.25)) <= 1.0:
        raise ValidationError("max_displacement_fraction must be in (0, 1]")
    if bool(config.get("enable_legacy_node_contact", False)):
        raise ValidationError("C1 validation requires legacy node contact disabled")
    if bool(config.get("reject_crossing", True)) is not True:
        raise ValidationError("reject_crossing=true is required")
    boundary = str(config.get("boundary", "free/free"))
    if boundary != "free/free":
        raise ValidationError("boundary must be free/free")
    if str(config.get("initial_shape", "sine")) not in {"sine", "u"}:
        raise ValidationError("initial_shape must be sine or u")


def load_config(path: Path) -> dict[str, Any]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        raise ValidationError("config root must be an object")
    defaults = {
        "length": 4.0,
        "n_nodes": 17,
        "axial_stiffness": 20.0,
        "bending_stiffness": 0.03,
        "drag_density": 1.0,
        "growth_rate": 0.25,
        "contact_stiffness": 25.0,
        "diameter": 0.22,
        "dt": 0.002,
        "t_end": 1.0,
        "a_max_factor": 2.0,
        "rest_length_factor": 1.03,
        "amplitude": 0.55,
        "initial_shape": "sine",
        "boundary": "free/free",
        "reject_crossing": True,
        "enable_legacy_node_contact": False,
        "dt_min": 1.0e-10,
        "max_retries": 12,
        "max_displacement_fraction": 0.25,
        "convergence_tolerance": 0.20,
        "expected_contact": False,
    }
    base = dict(defaults)
    base.update(dict(raw.get("base", {})))
    cases: list[CaseSpec] = []
    for item in raw.get("cases", []):
        if not isinstance(item, Mapping) or "name" not in item:
            raise ValidationError("each case requires name")
        overrides = dict(item.get("overrides", {}))
        expected = bool(
            item.get(
                "expected_contact",
                overrides.get("expected_contact", base.get("expected_contact", False)),
            )
        )
        cases.append(
            CaseSpec(str(item["name"]), str(item.get("group", "primary")), overrides, expected)
        )
    refinements = dict(raw.get("refinements", {}))
    output_policy = dict(raw.get("output_policy", {}))
    return {
        "schema_version": str(raw.get("schema_version", SCHEMA_VERSION)),
        "base": base,
        "cases": cases,
        "refinements": refinements,
        "output_policy": output_policy,
    }


def _effective_case(base: Mapping[str, Any], case: CaseSpec) -> dict[str, Any]:
    config = dict(base)
    config.update(case.overrides)
    config["name"] = case.name
    config["group"] = case.group
    config["expected_contact"] = case.expected_contact
    validate_case_config(config)
    return config


def initial_state(config: Mapping[str, Any]) -> FilamentState:
    """Create a smooth, non-crossing fixture with an explicit compression knob."""

    length = float(config["length"])
    n_nodes = int(config["n_nodes"])
    u = np.linspace(0.0, 1.0, n_nodes)
    amplitude = float(config.get("amplitude", 0.55))
    shape = str(config.get("initial_shape", "sine"))
    if shape == "u":
        radius = length / np.pi
        theta = np.linspace(np.pi, 0.0, n_nodes)
        positions = np.column_stack((radius * np.cos(theta), -radius * np.sin(theta)))
    else:
        positions = np.column_stack((length * u, amplitude * np.sin(2.0 * np.pi * u)))
    geometric_lengths = np.linalg.norm(np.diff(positions, axis=0), axis=1)
    rest_lengths = geometric_lengths * float(config.get("rest_length_factor", 1.0))
    return FilamentState(positions, rest_lengths)


def _signed_curvature(state: FilamentState) -> np.ndarray:
    edges = np.diff(state.positions, axis=0)
    lengths = np.linalg.norm(edges, axis=1)
    if len(edges) < 2:
        return np.empty(0, dtype=float)
    tangents = edges / lengths[:, None]
    cross = tangents[:-1, 0] * tangents[1:, 1] - tangents[:-1, 1] * tangents[1:, 0]
    local = 0.5 * (state.rest_lengths[:-1] + state.rest_lengths[1:])
    return cross / np.maximum(local, 1.0e-15)


def _fold_metrics(state: FilamentState, threshold: float) -> dict[str, Any]:
    signed = _signed_curvature(state)
    absolute = np.abs(signed)
    active = absolute >= threshold
    signs = np.sign(signed[active])
    fold_count = int(np.sum(signs[1:] * signs[:-1] < 0.0)) if len(signs) > 1 else 0
    edges = np.linalg.norm(np.diff(state.positions, axis=0), axis=1)
    arc = np.concatenate(([0.0], np.cumsum(edges)))
    candidate_positions: list[float] = []
    for index in range(1, len(absolute) - 1):
        if (
            absolute[index] >= threshold
            and absolute[index] >= absolute[index - 1]
            and absolute[index] >= absolute[index + 1]
        ):
            candidate_positions.append(float(arc[index + 1]))
    spacing = None
    if len(candidate_positions) >= 2:
        spacing = float(np.median(np.diff(candidate_positions)))
    mean_abs = float(np.mean(absolute)) if len(absolute) else 0.0
    max_abs = float(np.max(absolute)) if len(absolute) else 0.0
    return {
        "max_curvature": float(np.max(discrete_curvature(state))) if state.n_nodes >= 3 else 0.0,
        "curvature_rms": float(np.sqrt(np.mean(signed * signed))) if len(signed) else 0.0,
        "curvature_concentration": max_abs / mean_abs if mean_abs > 1.0e-15 else 0.0,
        "fold_count_proxy": fold_count,
        "fold_spacing": spacing,
        "fold_proxy_limit": "curvature sign changes and local peaks; not a topological fold count",
    }


def _root_lineage(label: str) -> str:
    return str(label).split(".", 1)[0]


def _canonical_pair(left: str, right: str) -> tuple[str, str]:
    return tuple(sorted((str(left), str(right))))


def _pair_records(
    state: FilamentState,
    diameter: float,
    contact_stiffness: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    contacts = nonlocal_segment_contacts(state.positions, diameter)
    active: list[dict[str, Any]] = []
    all_crossings: list[dict[str, Any]] = []
    lineage = state.segment_lineage or tuple(str(i) for i in range(state.n_segments))
    for contact in contacts:
        if contact.centerline_intersection:
            all_crossings.append(contact.as_dict())
            continue
        if not contact.is_finite_radius_contact:
            continue
        pair = _canonical_pair(lineage[contact.segment_i], lineage[contact.segment_j])
        normal = None if contact.normal is None else np.asarray(contact.normal, dtype=float)
        force = (
            np.zeros(2, dtype=float)
            if normal is None
            else contact_stiffness * contact.penetration * normal
        )
        active.append(
            {
                "pair": list(pair),
                "root_pair": list(_canonical_pair(_root_lineage(pair[0]), _root_lineage(pair[1]))),
                "segment_indices": [int(contact.segment_i), int(contact.segment_j)],
                "feature": contact.feature.value,
                "distance": float(contact.distance),
                "gap": float(contact.gap),
                "penetration": float(contact.penetration),
                "normal": None if normal is None else normal.tolist(),
                "force_vector": force.tolist(),
                "force_magnitude": float(np.linalg.norm(force)),
                "point_i": list(contact.point_i),
                "point_j": list(contact.point_j),
            }
        )
    return active, all_crossings


def _lineage_contact_observables(
    state: FilamentState,
    previous: FilamentState | None,
    active: Sequence[Mapping[str, Any]],
    previous_active: Sequence[Mapping[str, Any]],
    dt: float,
    episode_durations: Mapping[tuple[str, str], float],
) -> tuple[dict[str, Any], dict[tuple[str, str], float]]:
    previous_by_pair = {tuple(item["pair"]): item for item in previous_active}
    current_pairs = {tuple(item["pair"]) for item in active}
    previous_pairs = set(previous_by_pair)
    remesh_transition = previous is not None and previous.n_nodes != state.n_nodes
    detached = (
        [] if remesh_transition else [list(pair) for pair in sorted(previous_pairs - current_pairs)]
    )
    next_episode_durations: dict[tuple[str, str], float] = {}
    residence_step = 0.0
    slip_step = 0.0
    work_step = 0.0
    lineage_resets = 0
    for item in active:
        pair = tuple(item["pair"])
        old = previous_by_pair.get(pair)
        continuing = (
            old is not None
            and previous is not None
            and not remesh_transition
            and tuple(old["root_pair"]) == tuple(item["root_pair"])
            and old["feature"] == item["feature"]
        )
        if not continuing:
            if remesh_transition and tuple(item["root_pair"]) in {
                tuple(value["root_pair"]) for value in previous_active
            }:
                lineage_resets += 1
        else:
            duration = float(episode_durations.get(pair, 0.0)) + max(dt, 0.0)
            next_episode_durations[pair] = duration
            residence_step = max(residence_step, max(dt, 0.0))
            previous_relative = np.asarray(old["point_i"], dtype=float) - np.asarray(
                old["point_j"], dtype=float
            )
            current_relative = np.asarray(item["point_i"], dtype=float) - np.asarray(
                item["point_j"], dtype=float
            )
            normal = (
                np.asarray(item["normal"], dtype=float)
                if item["normal"] is not None
                else np.zeros(2)
            )
            tangent = np.asarray([-normal[1], normal[0]])
            relative_delta = current_relative - previous_relative
            slip_step += abs(float(np.dot(relative_delta, tangent)))
            work_step += float(
                np.dot(np.asarray(item["force_vector"], dtype=float), relative_delta)
            )
    return (
        {
            "contact_residence_step": residence_step,
            "contact_residence_time": max(next_episode_durations.values(), default=0.0),
            "relative_tangential_slip_step": slip_step,
            "contact_work_step": work_step,
            "detached_pairs": detached,
            "lineage_reset_count": lineage_resets,
            "remesh_contact_transition": remesh_transition,
        },
        next_episode_durations,
    )


def _energy_total(components: Mapping[str, float]) -> float:
    return float(components["stretch"] + components["bend"] + components["contact"])


def _metrics_rows(
    trajectory: Sequence[FilamentState],
    simulator: OverdampedGrowingFilament,
    config: Mapping[str, Any],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    previous_active: list[dict[str, Any]] = []
    episode_durations: dict[tuple[str, str], float] = {}
    cumulative_slip = 0.0
    cumulative_work = 0.0
    initial = trajectory[0]
    initial_endpoints = initial.positions[[0, -1]].copy()
    for index, state in enumerate(trajectory):
        previous = trajectory[index - 1] if index else None
        dt = float(state.time - previous.time) if previous is not None else 0.0
        active, crossings = _pair_records(
            state, float(config["diameter"]), float(config["contact_stiffness"])
        )
        lineage_obs, episode_durations = _lineage_contact_observables(
            state, previous, active, previous_active, dt, episode_durations
        )
        cumulative_slip += float(lineage_obs["relative_tangential_slip_step"])
        cumulative_work += float(lineage_obs["contact_work_step"])
        root_pairs = {tuple(item["root_pair"]) for item in active}
        contacts = nonlocal_segment_contacts(state.positions, float(config["diameter"]))
        finite_contacts = [item for item in contacts if item.is_finite_radius_contact]
        min_gap = min((float(item.gap) for item in contacts), default=float("inf"))
        max_penetration = max((float(item.penetration) for item in finite_contacts), default=0.0)
        contact_force_norm = float(sum(float(item["force_magnitude"]) for item in active))
        contact_force = simulator.contact_force_components(state.positions, state.rest_lengths)[
            "segment_c1"
        ]
        action_reaction = float(np.linalg.norm(np.sum(contact_force, axis=0)))
        endpoint_displacement = state.positions[[0, -1]] - initial_endpoints
        components = simulator.energy_components(state.positions, state.rest_lengths)
        contact_components = simulator.contact_energy_components(
            state.positions, state.rest_lengths
        )
        fold = _fold_metrics(state, float(config.get("fold_curvature_threshold", 1.0e-8)))
        rows.append(
            {
                "time": float(state.time),
                "step": int(state.step),
                "requested_dt": float(config["dt"]),
                "accepted_dt": None if previous is None else dt,
                "n_nodes": int(state.n_nodes),
                "remeshed_since_previous": bool(
                    previous is not None and previous.n_nodes != state.n_nodes
                ),
                "segment_lineage": list(state.segment_lineage or ()),
                "active_contact_pairs": len(active),
                "active_contact_pair_set": [item["pair"] for item in active],
                "active_contact_root_pair_set": [item["root_pair"] for item in active],
                "active_contact_features": [item["feature"] for item in active],
                "contact_records": active,
                "crossing_pairs": crossings,
                "min_gap": None if not math.isfinite(min_gap) else min_gap,
                "max_penetration": max_penetration,
                "max_penetration_ratio": max_penetration / float(config["diameter"])
                if float(config["diameter"]) > 0
                else 0.0,
                "contact_normal": active[0]["normal"] if active else None,
                "contact_force_norm": contact_force_norm,
                "contact_action_reaction_residual": action_reaction,
                "contact_work_step": float(lineage_obs["contact_work_step"]),
                "contact_work": cumulative_work,
                "contact_residence_step": float(lineage_obs["contact_residence_step"]),
                "contact_residence_time": float(lineage_obs["contact_residence_time"]),
                "relative_tangential_slip_step": float(
                    lineage_obs["relative_tangential_slip_step"]
                ),
                "relative_tangential_slip": cumulative_slip,
                "detached_pairs": lineage_obs["detached_pairs"],
                "lineage_reset_count": int(lineage_obs["lineage_reset_count"]),
                "remesh_contact_transition": bool(lineage_obs["remesh_contact_transition"]),
                "endpoint_motion_left": endpoint_displacement[0].tolist(),
                "endpoint_motion_right": endpoint_displacement[1].tolist(),
                "endpoint_motion_max": float(np.max(np.linalg.norm(endpoint_displacement, axis=1))),
                "energy_stretch": float(components["stretch"]),
                "energy_bend": float(components["bend"]),
                "energy_contact": float(components["contact"]),
                "energy_contact_segment_c1": float(contact_components["segment_c1"]),
                "energy_contact_node_legacy": float(contact_components["node_legacy"]),
                "energy_total": _energy_total(components),
                "radius_of_gyration": radius_of_gyration(state),
                "arc_length_weighted_radius_of_gyration": arc_length_weighted_radius_of_gyration(
                    state
                ),
                "contour_length": contour_length(state),
                "active_root_pair_count": len(root_pairs),
                **fold,
                "contact_identity_status": "lineage-aware-current-array-IDs",
                "contact_identity_limit": "root pairs preserve split ancestry; they do not establish material contact history",
            }
        )
        previous_active = active
    return rows


def _onset(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    contact_row = next((row for row in rows if int(row["active_contact_pairs"]) > 0), None)
    detach_row = next((row for row in rows if row["detached_pairs"]), None)
    return {
        "contact_onset_time": None if contact_row is None else float(contact_row["time"]),
        "contact_onset_step": None if contact_row is None else int(contact_row["step"]),
        "detachment_first_time": None if detach_row is None else float(detach_row["time"]),
    }


def _contact_sequence(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    previous: dict[tuple[str, str], tuple[tuple[str, str], str, int]] = {}
    next_episode: dict[tuple[str, str], int] = {}
    sequence: list[dict[str, Any]] = []
    for row in rows:
        time = float(row["time"])
        step = int(row["step"])
        if bool(row["remesh_contact_transition"]):
            sequence.append(
                {
                    "event": "remesh_boundary",
                    "time": time,
                    "step": step,
                    "n_nodes": int(row["n_nodes"]),
                    "pair": None,
                    "root_pair": None,
                    "feature": None,
                    "episode": None,
                }
            )
            previous = {}
        current: dict[tuple[str, str], tuple[tuple[str, str], str, int]] = {}
        records = sorted(
            row["contact_records"],
            key=lambda item: (tuple(item["pair"]), str(item["feature"])),
        )
        current_pairs = {tuple(item["pair"]) for item in records}
        for pair, (root_pair, feature, episode) in sorted(previous.items()):
            if pair not in current_pairs:
                sequence.append(
                    {
                        "event": "contact_detachment",
                        "time": time,
                        "step": step,
                        "n_nodes": int(row["n_nodes"]),
                        "pair": list(pair),
                        "root_pair": list(root_pair),
                        "feature": feature,
                        "episode": episode,
                    }
                )
        for item in records:
            pair = tuple(item["pair"])
            root_pair = tuple(item["root_pair"])
            feature = str(item["feature"])
            old = previous.get(pair)
            continuing = old is not None and old[:2] == (root_pair, feature)
            if continuing:
                episode = old[2]
            else:
                if old is not None:
                    sequence.append(
                        {
                            "event": "contact_detachment",
                            "time": time,
                            "step": step,
                            "n_nodes": int(row["n_nodes"]),
                            "pair": list(pair),
                            "root_pair": list(old[0]),
                            "feature": old[1],
                            "episode": old[2],
                        }
                    )
                episode = next_episode.get(pair, -1) + 1
                next_episode[pair] = episode
                sequence.append(
                    {
                        "event": "contact_onset",
                        "time": time,
                        "step": step,
                        "n_nodes": int(row["n_nodes"]),
                        "pair": list(pair),
                        "root_pair": list(root_pair),
                        "feature": feature,
                        "episode": episode,
                    }
                )
            current[pair] = (root_pair, feature, episode)
        previous = current
    return sequence


def _contact_sequence_signature(
    sequence: Sequence[Mapping[str, Any]],
) -> tuple[tuple[Any, ...], ...]:
    return tuple(
        (
            str(item["event"]),
            None if item["time"] is None else float(item["time"]),
            None if item["step"] is None else int(item["step"]),
            None if item["n_nodes"] is None else int(item["n_nodes"]),
            None if item["pair"] is None else tuple(item["pair"]),
            None if item["root_pair"] is None else tuple(item["root_pair"]),
            None if item["feature"] is None else str(item["feature"]),
            None if item["episode"] is None else int(item["episode"]),
        )
        for item in sequence
    )


def _event_counts(simulator: OverdampedGrowingFilament) -> dict[str, int]:
    counts: dict[str, int] = {}
    for reason in simulator.rejection_reasons:
        key = str(reason)
        if "crossing" in key or "intersection" in key:
            key = "crossing_rejection"
        elif "displacement" in key:
            key = "displacement_exceeded"
        elif "non-finite" in key:
            key = "nonfinite"
        else:
            key = "other_rejection"
        counts[key] = counts.get(key, 0) + 1
    return counts


def _accepted_dt_collapsed(simulator: OverdampedGrowingFilament) -> bool:
    for event in simulator.event_log:
        if event.get("event_type") != "step_attempt" or event.get("accepted") is not True:
            continue
        requested = event.get("requested_dt")
        accepted = event.get("accepted_dt")
        if requested is None or accepted is None:
            continue
        if float(accepted) < 0.25 * float(requested):
            return True
    return False


def run_case(
    case: CaseSpec, base_config: Mapping[str, Any], *, git_revision: str | None = None
) -> dict[str, Any]:
    config = _effective_case(base_config, case)
    initial = initial_state(config)
    initial_active, _ = _pair_records(
        initial, float(config["diameter"]), float(config["contact_stiffness"])
    )
    initial_contacts = nonlocal_segment_contacts(initial.positions, float(config["diameter"]))
    initial_min_gap = min((float(item.gap) for item in initial_contacts), default=float("inf"))
    initial_contact_violation = bool(initial_active) or initial_min_gap < 0.0
    initial_contact_control = case.group == "initial_contact_control"
    params = ModelParameters(
        axial_stiffness=float(config["axial_stiffness"]),
        bending_stiffness=float(config["bending_stiffness"]),
        drag_density=float(config["drag_density"]),
        contact_stiffness=float(config["contact_stiffness"]),
        diameter=float(config["diameter"]),
        growth_rate=float(config["growth_rate"]),
        reference_length=float(np.mean(initial.rest_lengths)),
        dt=float(config["dt"]),
        t_end=float(config["t_end"]),
        a_max=float(np.max(initial.rest_lengths)) * float(config["a_max_factor"]),
        dt_min=float(config.get("dt_min", 1.0e-10)),
        max_retries=int(config.get("max_retries", 12)),
        max_displacement_fraction=float(config.get("max_displacement_fraction", 0.25)),
        reject_crossing=True,
        enable_legacy_node_contact=False,
    )
    failure: str | None = None
    failure_event: dict[str, Any] | None = None
    simulator: OverdampedGrowingFilament | None = None
    trajectory: list[FilamentState] = [initial.copy()]
    try:
        simulator = OverdampedGrowingFilament(initial, params)
        trajectory = simulator.run()
    except ModelError as exc:
        failure = f"{type(exc).__name__}: {exc}"
        failure_event = None if exc.event is None else dict(exc.event)
    except (RuntimeError, ValueError, FloatingPointError) as exc:
        failure = f"{type(exc).__name__}: {exc}"
    if simulator is None:
        rows: list[dict[str, Any]] = []
        final = initial
        events = [] if failure_event is None else [failure_event]
        rejected = 0
        accepted = 0
        rejection_counts: dict[str, int] = {}
    else:
        if failure:
            trajectory = simulator.accepted_trajectory
        rows = _metrics_rows(trajectory, simulator, config)
        final = simulator.state
        events = simulator.event_log
        rejected = int(simulator.rejected_steps)
        accepted = int(simulator.accepted_steps)
        rejection_counts = _event_counts(simulator)
    onset = _onset(rows)
    observed_contact = onset["contact_onset_time"] is not None
    expected = bool(config.get("expected_contact", False))
    numerical_reasons: list[str] = []
    if failure:
        numerical_reasons.append("solver_failure")
    if expected and not observed_contact:
        numerical_reasons.append("expected_contact_not_observed")
    if initial_contact_control:
        if not initial_active:
            numerical_reasons.append("initial_contact_control_not_observed")
    elif initial_contact_violation:
        numerical_reasons.append("initial_contact_violation")
    if rows and any(row["crossing_pairs"] for row in rows):
        numerical_reasons.append("centerline_crossing_guard_observed")
    if simulator is not None and _accepted_dt_collapsed(simulator):
        numerical_reasons.append("accepted_dt_collapsed_below_requested")
    numerical_status = "numerically-unresolved" if numerical_reasons else "resolved"
    contact_sequence = _contact_sequence(rows)
    metadata = {
        "benchmark": "C1 frictionless finite-radius segment contact/folding validation",
        "case": case.name,
        "group": case.group,
        "contact_law": "C1 segment penalty only",
        "legacy_node_contact": "disabled",
        "friction": "disabled",
        "adhesion": "disabled",
        "contact_history": "disabled; lineage is diagnostic only",
        "nested_alternatives": {
            "C1": "active baseline",
            "friction": "not implemented; add only after defined C1 failure and independent observation",
            "adhesion": "not implemented; add only after defined C1 failure and independent observation",
            "history_dependent": "not implemented; add only after defined C1 failure and independent observation",
            "identical_initial_conditions": "same case config is the comparison contract",
        },
        "expected_contact": expected,
        "measurement_limits": {
            "finite_penalty": "penetration is finite and stiffness/time-step dependent; not a hard non-penetration proof",
            "fold_count_proxy": "curvature sign changes/local peaks, not a topological fold count",
            "fold_spacing": "median spacing of resolved curvature peaks; undefined for short/nonperiodic shapes",
            "slip": "relative tangential closest-point displacement; diagnostic, no friction force",
        },
        "failure_reason": failure,
    }
    manifest = build_manifest(
        params,
        initial,
        final_state=final,
        events=events,
        metadata=metadata,
        input_data=config,
        git_revision=git_revision,
    )
    accepted_dt_values = (
        [] if simulator is None else [float(value) for value in simulator.accepted_dts]
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "case": case.name,
        "group": case.group,
        "expected_contact": expected,
        "effective_config": config,
        "initial_condition": "initial-contact-control"
        if initial_contact_control
        else "non-contact-required",
        "initial_active_contact_pairs": len(initial_active),
        "initial_min_gap": None if not math.isfinite(initial_min_gap) else initial_min_gap,
        "classification": (
            "initial-contact-control"
            if initial_contact_control
            else ("contact-observed" if observed_contact else "no-contact-observed")
        ),
        "numerical_status": numerical_status,
        "numerical_reason_codes": numerical_reasons,
        "failure_reason": failure,
        "accepted_steps": accepted,
        "rejected_steps": rejected,
        "requested_dt": float(config["dt"]),
        "accepted_dt_values": accepted_dt_values,
        "accepted_dt_min": min(accepted_dt_values) if accepted_dt_values else None,
        "accepted_dt_max": max(accepted_dt_values) if accepted_dt_values else None,
        "accepted_dt_mean": float(np.mean(accepted_dt_values)) if accepted_dt_values else None,
        "rejection_reason_counts": rejection_counts,
        "event_count": len(events),
        "onset": onset,
        "initial_metrics": rows[0] if rows else None,
        "final_metrics": rows[-1] if rows else None,
        "max_penetration_ratio": max(
            (float(row["max_penetration_ratio"]) for row in rows), default=0.0
        ),
        "max_contact_residence_time": max(
            (float(row["contact_residence_time"]) for row in rows), default=0.0
        ),
        "max_relative_tangential_slip": max(
            (float(row["relative_tangential_slip"]) for row in rows), default=0.0
        ),
        "max_endpoint_motion": max(
            (float(row["endpoint_motion_max"]) for row in rows), default=0.0
        ),
        "max_curvature_concentration": max(
            (float(row["curvature_concentration"]) for row in rows), default=0.0
        ),
        "max_fold_count_proxy": max((int(row["fold_count_proxy"]) for row in rows), default=0),
        "fold_spacing_final": None if not rows else rows[-1]["fold_spacing"],
        "contact_root_sequence": contact_sequence,
        "crossing_guard_rejections": rejection_counts.get("crossing_rejection", 0),
        "ccd_contract": "swept centerline-crossing guard only; finite-radius CCD and hard non-penetration are not implemented",
        "metrics_rows": rows,
        "manifest": manifest,
    }


def _summary_row(result: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "case": result["case"],
        "group": result["group"],
        "expected_contact": result["expected_contact"],
        "classification": result["classification"],
        "numerical_status": result["numerical_status"],
        "numerical_reason_codes": json.dumps(result["numerical_reason_codes"], sort_keys=True),
        "requested_dt": result["requested_dt"],
        "accepted_dt_min": result["accepted_dt_min"],
        "accepted_dt_mean": result["accepted_dt_mean"],
        "accepted_dt_max": result["accepted_dt_max"],
        "accepted_steps": result["accepted_steps"],
        "rejected_steps": result["rejected_steps"],
        "contact_onset_time": result["onset"]["contact_onset_time"],
        "detachment_first_time": result["onset"]["detachment_first_time"],
        "max_penetration_ratio": result["max_penetration_ratio"],
        "max_contact_residence_time": result["max_contact_residence_time"],
        "max_relative_tangential_slip": result["max_relative_tangential_slip"],
        "max_endpoint_motion": result["max_endpoint_motion"],
        "max_curvature_concentration": result["max_curvature_concentration"],
        "max_fold_count_proxy": result["max_fold_count_proxy"],
        "fold_spacing_final": result["fold_spacing_final"],
        "contact_root_sequence": json.dumps(result["contact_root_sequence"], sort_keys=True),
        "crossing_guard_rejections": result["crossing_guard_rejections"],
        "failure_reason": result["failure_reason"],
    }


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("\n", encoding="utf-8")
        return
    fields = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=fields, extrasaction="ignore", lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(_jsonable(row) for row in rows)


def _relative_difference(left: float | None, right: float | None) -> float | None:
    if left is None or right is None:
        return None
    return abs(float(left) - float(right)) / max(abs(float(right)), 1.0e-12)


def _refinement_summary(
    results: Sequence[Mapping[str, Any]], config: Mapping[str, Any]
) -> list[dict[str, Any]]:
    """Assess requested temporal/spatial/kc refinement without overclaiming."""

    rows: list[dict[str, Any]] = []
    tolerance = float(config["base"].get("convergence_tolerance", 0.2))
    for axis, group, value_key in (
        ("dt", "temporal_refinement", "dt"),
        ("n_nodes", "spatial_refinement", "n_nodes"),
        ("contact_stiffness", "contact_stiffness_refinement", "contact_stiffness"),
    ):
        values = [result for result in results if result["group"] == group]
        if len(values) < 2:
            rows.append(
                {
                    "axis": axis,
                    "status": "numerically-unresolved",
                    "reason_codes": ["paired_refinement_cases_required"],
                }
            )
            continue
        ordered = sorted(values, key=lambda result: float(result["effective_config"][value_key]))
        coarse, fine = ordered[0], ordered[-1]
        reasons: list[str] = []
        if coarse["failure_reason"] or fine["failure_reason"]:
            reasons.append("solver_failure")
        if coarse["numerical_status"] != "resolved" or fine["numerical_status"] != "resolved":
            reasons.append("member_case_numerically_unresolved")
        if axis == "dt":
            accepted = [result["accepted_dt_mean"] for result in ordered]
            if len({None if value is None else round(float(value), 15) for value in accepted}) < 2:
                reasons.append("accepted_dt_not_distinct")
        sequence_match = _contact_sequence_signature(
            coarse["contact_root_sequence"]
        ) == _contact_sequence_signature(fine["contact_root_sequence"])
        if not sequence_match:
            reasons.append("contact_sequence_changed")
        penetration_difference = _relative_difference(
            coarse["max_penetration_ratio"], fine["max_penetration_ratio"]
        )
        residence_difference = _relative_difference(
            coarse["max_contact_residence_time"], fine["max_contact_residence_time"]
        )
        if penetration_difference is None or penetration_difference > tolerance:
            reasons.append("penetration_not_within_tolerance")
        if residence_difference is None or residence_difference > tolerance:
            reasons.append("contact_residence_not_within_tolerance")
        if (
            axis == "contact_stiffness"
            and float(fine["max_penetration_ratio"])
            > float(coarse["max_penetration_ratio"]) + 1.0e-12
        ):
            reasons.append("higher_stiffness_increased_penetration")
        rows.append(
            {
                "axis": axis,
                "coarse_or_low_case": coarse["case"],
                "fine_or_high_case": fine["case"],
                "coarse_or_low_value": coarse["effective_config"][value_key],
                "fine_or_high_value": fine["effective_config"][value_key],
                "penetration_relative_difference": penetration_difference,
                "residence_relative_difference": residence_difference,
                "contact_sequence_match": sequence_match,
                "status": "resolved" if not reasons else "numerically-unresolved",
                "reason_codes": reasons,
                "nonpenetration_claim": "not-a-proof; finite penalty remains penetrable",
            }
        )
    return rows


def run_benchmark(
    config: Mapping[str, Any], output: Path, *, git_revision: str | None = None
) -> dict[str, Any]:
    raw_cases = list(config.get("cases", []))
    if not raw_cases:
        raise ValidationError("at least one case is required")
    cases = [
        item
        if isinstance(item, CaseSpec)
        else CaseSpec(
            str(item["name"]),
            str(item.get("group", "primary")),
            dict(item.get("overrides", {})),
            bool(
                item.get(
                    "expected_contact",
                    item.get("overrides", {}).get(
                        "expected_contact", config["base"].get("expected_contact", False)
                    ),
                )
            ),
        )
        for item in raw_cases
    ]
    output.mkdir(parents=True, exist_ok=True)
    revision = git_revision if git_revision is not None else detect_git_revision(Path.cwd())
    results = [run_case(case, config["base"], git_revision=revision) for case in cases]
    summary = [_summary_row(result) for result in results]
    refinement = _refinement_summary(results, config)
    metrics: list[dict[str, Any]] = []
    for result in results:
        for row in result["metrics_rows"]:
            metrics.append({"case": result["case"], **row})
    config_hash = sha256_hex(canonical_json_bytes(_jsonable(config)))
    compact_manifest = {
        "manifest_schema_version": "continuum-filament-contact-folding-compact-1",
        "benchmark_schema_version": SCHEMA_VERSION,
        "benchmark": "C1 frictionless finite-radius contact/folding validation",
        "contact_law": "segment penalty C1 only",
        "legacy_node_contact": "disabled",
        "friction": "disabled",
        "adhesion": "disabled",
        "contact_history": "disabled; lineage diagnostic only",
        "nested_alternatives": {
            "C1": "active baseline",
            "friction": "not implemented",
            "adhesion": "not implemented",
            "history_dependent": "not implemented",
            "identical_initial_conditions": "same case config is the comparison contract",
        },
        "git_revision": revision,
        "config_hash": config_hash,
        "case_order": [result["case"] for result in results],
        "case_manifests": {
            result["case"]: {
                "input_hash": result["manifest"].get("input_hash"),
                "initial_state_hash": result["manifest"].get("initial_state_hash"),
                "canonical_state_hash": result["manifest"].get("canonical_state_hash"),
                "event_count": result["event_count"],
            }
            for result in results
        },
        "refinement": refinement,
        "output_policy": {
            "trajectory_arrays": False,
            "videos": False,
            "stored_time_series": "compact metrics CSV with JSON contact records",
        },
        "limitations": [
            "finite penalty penetration is not a hard non-penetration proof",
            "swept centerline crossing guard is not finite-radius continuous-time CCD",
            "lineage IDs track remesh descendants but do not add contact history",
            "fold spacing/count are morphology proxies and do not establish experimental folding reproduction",
            "centerline alone does not infer filament diameter",
        ],
    }
    suite = {
        "schema_version": SCHEMA_VERSION,
        "benchmark": "C1 frictionless finite-radius contact/folding validation",
        "git_revision": revision,
        "config_hash": config_hash,
        "refinement": refinement,
        "results": [
            {key: value for key, value in result.items() if key not in {"metrics_rows", "manifest"}}
            for result in results
        ],
        "compact_manifest": compact_manifest,
    }
    _write_csv(output / "summary.csv", summary)
    _write_json(output / "summary.json", summary)
    _write_csv(output / "metrics.csv", metrics)
    _write_csv(output / "refinement_summary.csv", refinement)
    _write_json(output / "refinement_summary.json", refinement)
    _write_json(output / "compact_manifest.json", compact_manifest)
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
    except (ValidationError, OSError, json.JSONDecodeError, ValueError) as exc:
        print(f"contact folding validation error: {exc}", file=sys.stderr)
        return 2
    unresolved = [item for item in suite["results"] if item["numerical_status"] != "resolved"]
    print(
        json.dumps(
            {
                "output": str(args.output),
                "cases": len(suite["results"]),
                "numerically_unresolved": len(unresolved),
            },
            sort_keys=True,
        )
    )
    # An unresolved refinement is a recorded validation result, not a CLI
    # execution failure.  Consumers must inspect numerical_status explicitly.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
