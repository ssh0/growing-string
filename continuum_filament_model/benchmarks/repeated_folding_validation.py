"""Long-duration repeated-contact validation for the C1 baseline.

The runner observes the existing frictionless finite-radius segment penalty
without adding friction, adhesion, or contact history.  It stores accepted
states as compact rows and turns contact transitions into episode records.
Trajectories and videos are deliberately not written by this module.
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
    _BENCHMARKS = _HERE.parent
    _SRC = _HERE.parents[1] / "src"
    for _path in (_BENCHMARKS, _SRC):
        if str(_path) not in sys.path:
            sys.path.insert(0, str(_path))

try:
    from .contact_folding_validation import (  # type: ignore[import-not-found]
        _fold_metrics,
        _jsonable,
        _pair_records,
        initial_state as _c1_initial_state,
    )
except ImportError:  # direct ``python benchmarks/repeated_folding_validation.py``
    from contact_folding_validation import (
        _fold_metrics,
        _jsonable,
        _pair_records,
        initial_state as _c1_initial_state,
    )

from growing_filament.geometry import nonlocal_segment_contacts
from growing_filament.model import (
    FilamentState,
    ModelError,
    ModelParameters,
    OverdampedGrowingFilament,
)
from growing_filament.reproducibility import (
    build_manifest,
    canonical_json_bytes,
    detect_git_revision,
    sha256_hex,
)


SCHEMA_VERSION = "continuum-filament-repeated-folding-c1-1"


@dataclass(frozen=True)
class CaseSpec:
    """One deterministic case or explicitly separated shape-only sensitivity case."""

    name: str
    group: str
    overrides: dict[str, Any]
    expected_contact: bool = False
    population: str = "deterministic"
    refinement_axis: str | None = None
    refinement_family: str | None = None
    refinement_role: str | None = None
    expected_repeated: bool = False


class ValidationError(ValueError):
    """Invalid repeated-folding validation input."""


def _strict_bool(value: Any, key: str) -> bool:
    if type(value) is not bool:
        raise ValidationError(f"{key} must be boolean")
    return value


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json_bytes(_jsonable(value)) + b"\n")


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
    if str(config.get("boundary", "free/free")) != "free/free":
        raise ValidationError("boundary must be free/free")
    if bool(config.get("enable_legacy_node_contact", False)):
        raise ValidationError("C1 validation requires legacy node contact disabled")
    if bool(config.get("friction", False)) or bool(config.get("adhesion", False)):
        raise ValidationError("friction and adhesion are not active in the C1 runner")
    if bool(config.get("contact_history", False)):
        raise ValidationError("contact history is not active in the C1 runner")
    _strict_bool(config.get("expected_contact", False), "expected_contact")
    _strict_bool(config.get("expected_repeated", False), "expected_repeated")
    if _strict_bool(config.get("reject_crossing", True), "reject_crossing") is not True:
        raise ValidationError("reject_crossing=true is required")
    if str(config.get("initial_shape", "sine")) not in {"sine", "u"}:
        raise ValidationError("initial_shape must be sine or u")


def _defaults() -> dict[str, Any]:
    return {
        "length": 4.0,
        "n_nodes": 9,
        "axial_stiffness": 5.0,
        "bending_stiffness": 0.03,
        "drag_density": 1.0,
        "growth_rate": 1.0,
        "contact_stiffness": 2.0,
        "diameter": 0.5,
        "dt": 0.02,
        "t_end": 2.4,
        "a_max_factor": 8.0,
        "rest_length_factor": 0.75,
        "amplitude": 0.55,
        "initial_shape": "sine",
        "boundary": "free/free",
        "reject_crossing": True,
        "enable_legacy_node_contact": False,
        "friction": False,
        "adhesion": False,
        "contact_history": False,
        "dt_min": 1.0e-8,
        "max_retries": 4,
        "max_displacement_fraction": 0.5,
        "fold_curvature_threshold": 1.0e-8,
        "episode_time_tolerance": 0.08,
        "penetration_tolerance": 0.30,
        "residence_tolerance": 0.30,
        "fold_tolerance": 0.50,
        "fold_count_tolerance": 1,
        "remesh_boundary_tolerance": 1,
        "contact_identity_tolerance": 0.08,
        "expected_contact": False,
        "expected_repeated": False,
    }


def load_config(path: Path) -> dict[str, Any]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        raise ValidationError("config root must be an object")
    base = _defaults()
    base.update(dict(raw.get("base", {})))
    cases: list[CaseSpec] = []
    for item in raw.get("cases", []):
        if not isinstance(item, Mapping) or "name" not in item:
            raise ValidationError("each case requires name")
        overrides = dict(item.get("overrides", {}))
        expected = _strict_bool(
            item.get(
                "expected_contact",
                overrides.get("expected_contact", base.get("expected_contact", False)),
            ),
            "expected_contact",
        )
        expected_repeated = _strict_bool(
            item.get(
                "expected_repeated",
                overrides.get("expected_repeated", base.get("expected_repeated", False)),
            ),
            "expected_repeated",
        )
        cases.append(
            CaseSpec(
                str(item["name"]),
                str(item.get("group", "primary")),
                overrides,
                expected,
                str(item.get("population", "deterministic")),
                None if item.get("refinement_axis") is None else str(item["refinement_axis"]),
                None if item.get("refinement_family") is None else str(item["refinement_family"]),
                None if item.get("refinement_role") is None else str(item["refinement_role"]),
                expected_repeated,
            )
        )
    return {
        "schema_version": str(raw.get("schema_version", SCHEMA_VERSION)),
        "base": base,
        "cases": cases,
        "refinement_pairs": list(raw.get("refinement_pairs", [])),
        "output_policy": dict(raw.get("output_policy", {})),
    }


def _effective_case(base: Mapping[str, Any], case: CaseSpec) -> dict[str, Any]:
    config = dict(base)
    config.update(case.overrides)
    config.update(
        {
            "name": case.name,
            "group": case.group,
            "population": case.population,
            "refinement_axis": case.refinement_axis,
            "refinement_family": case.refinement_family,
            "refinement_role": case.refinement_role,
            "expected_repeated": _strict_bool(case.expected_repeated, "expected_repeated"),
            "expected_contact": _strict_bool(case.expected_contact, "expected_contact"),
        }
    )
    validate_case_config(config)
    if case.population not in {"deterministic", "shape_only_sensitivity"}:
        raise ValidationError(f"unsupported population: {case.population}")
    if case.population == "shape_only_sensitivity":
        allowed = {"initial_shape", "amplitude", "rest_length_factor"}
        unexpected = set(case.overrides) - allowed
        if unexpected:
            raise ValidationError(
                "shape-only sensitivity may vary only initial_shape/amplitude/rest_length_factor: "
                f"{sorted(unexpected)}"
            )
    return config


def _initial_state(config: Mapping[str, Any]) -> FilamentState:
    """Use the C1 fixture contract; shape sensitivity changes only that fixture."""

    return _c1_initial_state(config)


def _event_counts(simulator: OverdampedGrowingFilament) -> dict[str, int]:
    counts: dict[str, int] = {}
    for reason in simulator.rejection_reasons:
        text = str(reason)
        if "crossing" in text or "intersection" in text:
            key = "crossing_rejection"
        elif "displacement" in text:
            key = "displacement_exceeded"
        elif "non-finite" in text:
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
        if (
            requested is not None
            and accepted is not None
            and float(accepted) < 0.25 * float(requested)
        ):
            return True
    return False


def _normalized_contact_positions(
    record: Mapping[str, Any], state: FilamentState
) -> tuple[float, float] | None:
    indices = record.get("segment_indices")
    if indices is None or len(indices) != 2:
        return None
    try:
        segment_indices = tuple(int(index) for index in indices)
    except (TypeError, ValueError):
        return None
    rest_lengths = np.asarray(state.rest_lengths, dtype=float)
    if (
        not np.isfinite(rest_lengths).all()
        or np.sum(rest_lengths) <= 0.0
        or any(index < 0 or index >= len(rest_lengths) for index in segment_indices)
    ):
        return None
    cumulative = np.concatenate(([0.0], np.cumsum(rest_lengths)))
    total = float(cumulative[-1])
    return tuple(
        float((cumulative[index] + 0.5 * rest_lengths[index]) / total)
        for index in segment_indices
    )


def _metrics_rows(
    trajectory: Sequence[FilamentState],
    simulator: OverdampedGrowingFilament,
    config: Mapping[str, Any],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    initial = trajectory[0]
    initial_endpoints = initial.positions[[0, -1]].copy()
    for index, state in enumerate(trajectory):
        previous = trajectory[index - 1] if index else None
        active, crossings = _pair_records(
            state, float(config["diameter"]), float(config["contact_stiffness"])
        )
        for record in active:
            normalized_positions = _normalized_contact_positions(record, state)
            if normalized_positions is not None:
                record["contact_identity"] = normalized_positions
        contacts = nonlocal_segment_contacts(state.positions, float(config["diameter"]))
        finite_contacts = [item for item in contacts if item.is_finite_radius_contact]
        min_gap = min((float(item.gap) for item in contacts), default=float("inf"))
        max_penetration = max((float(item.penetration) for item in finite_contacts), default=0.0)
        contact_force = simulator.contact_force_components(state.positions, state.rest_lengths)[
            "segment_c1"
        ]
        endpoint_displacement = state.positions[[0, -1]] - initial_endpoints
        components = simulator.energy_components(state.positions, state.rest_lengths)
        contact_components = simulator.contact_energy_components(
            state.positions, state.rest_lengths
        )
        fold = _fold_metrics(state, float(config.get("fold_curvature_threshold", 1.0e-8)))
        rows.append(
            {
                "raw_state_index": index,
                "time": float(state.time),
                "step": int(state.step),
                "requested_dt": float(config["dt"]),
                "accepted_dt": None if previous is None else float(state.time - previous.time),
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
                if float(config["diameter"]) > 0.0
                else 0.0,
                "contact_force_norm": float(sum(item["force_magnitude"] for item in active)),
                "contact_action_reaction_residual": float(
                    np.linalg.norm(np.sum(contact_force, axis=0))
                ),
                "endpoint_motion_left": endpoint_displacement[0].tolist(),
                "endpoint_motion_right": endpoint_displacement[1].tolist(),
                "endpoint_motion_max": float(np.max(np.linalg.norm(endpoint_displacement, axis=1))),
                "energy_stretch": float(components["stretch"]),
                "energy_bend": float(components["bend"]),
                "energy_contact": float(components["contact"]),
                "energy_contact_segment_c1": float(contact_components["segment_c1"]),
                "energy_contact_node_legacy": float(contact_components["node_legacy"]),
                "energy_total": float(sum(components.values())),
                **fold,
            }
        )
    return rows


def _episode_event(
    event: str,
    row: Mapping[str, Any],
    episode: Mapping[str, Any] | None = None,
    *,
    reason: str | None = None,
) -> dict[str, Any]:
    return {
        "event": event,
        "reason": reason,
        "time": float(row["time"]),
        "step": int(row["step"]),
        "n_nodes": int(row["n_nodes"]),
        "raw_state_index": int(row["raw_state_index"]),
        "remesh_boundary": bool(row["remeshed_since_previous"]),
        "episode": None if episode is None else int(episode["episode_id"]),
        "pair": None if episode is None else list(episode["pair"]),
        "root_pair": None if episode is None else list(episode["root_pair"]),
        "feature": None if episode is None else episode["feature"],
        "contact_identity": None
        if episode is None
        else episode.get("contact_identity"),
    }


def _normalized_contact_identity(
    sequence: Sequence[Mapping[str, Any]],
) -> tuple[tuple[str, tuple[Any, ...] | None, str | None], ...]:
    return tuple(
        (
            str(event["event"]),
            (
                tuple(float(value) for value in event["contact_identity"])
                if event.get("contact_identity") is not None
                else (
                    None
                    if event.get("root_pair") is None
                    else tuple(str(value) for value in event["root_pair"])
                )
            ),
            None if event.get("feature") is None else str(event["feature"]),
        )
        for event in sequence
        if event["event"] not in {"active_continuation", "remesh_boundary"}
    )


def _contact_identity_match(
    left: Sequence[tuple[str, tuple[Any, ...] | None, str | None]],
    right: Sequence[tuple[str, tuple[Any, ...] | None, str | None]],
    tolerance: float,
) -> bool:
    if len(left) != len(right):
        return False
    for left_item, right_item in zip(left, right):
        if left_item[0] != right_item[0] or left_item[2] != right_item[2]:
            return False
        left_identity, right_identity = left_item[1], right_item[1]
        if left_identity is None or right_identity is None:
            if left_identity != right_identity:
                return False
            continue
        if len(left_identity) != len(right_identity):
            return False
        if all(isinstance(value, (int, float)) for value in left_identity + right_identity):
            if any(
                abs(float(left_value) - float(right_value)) > tolerance
                for left_value, right_value in zip(left_identity, right_identity)
            ):
                return False
        elif left_identity != right_identity:
            return False
    return True


def _episode_tracker(rows: Sequence[Mapping[str, Any]], diameter: float) -> dict[str, Any]:
    """Build transitions while retaining raw accepted-state coordinates.

    Exact segment IDs are useful within one run, but they are not used as the
    refinement signature: adaptive remeshing legitimately changes them.
    """

    sequence: list[dict[str, Any]] = []
    episodes: list[dict[str, Any]] = []
    active: dict[tuple[tuple[str, str], str], dict[str, Any]] = {}
    detached_keys: set[tuple[tuple[str, str], str]] = set()
    previous_row: Mapping[str, Any] | None = None
    cumulative_slip = 0.0

    def close_episode(episode: dict[str, Any], row: Mapping[str, Any], reason: str) -> None:
        if episode.get("detachment_time") is not None or episode.get("remesh_closed"):
            return
        if reason == "detachment":
            episode["detachment_time"] = float(row["time"])
            episode["detachment_step"] = int(row["step"])
            episode["detachment_n_nodes"] = int(row["n_nodes"])
            episode["censored_at_end"] = False
        elif reason == "remesh_boundary":
            episode["remesh_closed"] = True
            episode["remesh_boundary_time"] = float(row["time"])
            episode["remesh_boundary_step"] = int(row["step"])
            episode["censored_at_end"] = True
        elif reason == "end_censored":
            episode["censored_at_end"] = True
        else:
            episode["censored_at_end"] = False
        episode["observed_until_time"] = float(row["time"])
        episode["residence_duration"] = max(
            0.0, float(episode["observed_until_time"]) - float(episode["onset_time"])
        )
        episode["close_reason"] = reason

    for row in rows:
        current_records = sorted(
            row["contact_records"],
            key=lambda item: (tuple(item["pair"]), str(item["feature"])),
        )
        current: dict[tuple[tuple[str, str], str], Mapping[str, Any]] = {
            (tuple(item["pair"]), str(item["feature"])): item for item in current_records
        }
        if bool(row["remeshed_since_previous"]):
            sequence.append(_episode_event("remesh_boundary", row, reason="n_nodes_changed"))
            detached_keys.clear()
            for episode in list(active.values()):
                close_episode(episode, row, "remesh_boundary")
            active.clear()

        old_keys = set(active)
        current_keys = set(current)
        for key in sorted(old_keys - current_keys):
            old_episode = active[key]
            has_feature_change = any(current_key[0] == key[0] for current_key in current_keys)
            if has_feature_change:
                continue
            episode = active.pop(key)
            close_episode(episode, row, "detachment")
            detached_keys.add(key)
            sequence.append(_episode_event("contact_detachment", row, episode))

        for key, record in current.items():
            pair, feature = key
            old = active.get(key)
            if old is not None:
                old_record = old["last_record"]
                sequence.append(_episode_event("active_continuation", row, old))
                if old_record.get("normal") is not None and record.get("normal") is not None:
                    previous_relative = np.asarray(old_record["point_i"]) - np.asarray(
                        old_record["point_j"]
                    )
                    current_relative = np.asarray(record["point_i"]) - np.asarray(record["point_j"])
                    normal = np.asarray(record["normal"], dtype=float)
                    tangent = np.asarray([-normal[1], normal[0]])
                    slip = abs(float(np.dot(current_relative - previous_relative, tangent)))
                    old["relative_tangential_slip"] += slip
                    cumulative_slip += slip
                old["max_penetration_ratio"] = max(
                    float(old["max_penetration_ratio"]),
                    float(record["penetration"]) / max(diameter, 1.0e-15),
                )
                old["sample_count"] += 1
                old["last_record"] = record
                old["last_time"] = float(row["time"])
                continue

            same_pair = [
                value
                for value in active.values()
                if value["pair"] == pair
                and (tuple(value["pair"]), str(value["feature"])) not in current_keys
            ]
            same_root = [
                value
                for value in active.values()
                if value["root_pair"] == tuple(record["root_pair"])
                and (tuple(value["pair"]), str(value["feature"])) not in current_keys
            ]
            if same_pair:
                old = same_pair[0]
                close_episode(old, row, "feature_change")
                sequence.append(
                    _episode_event(
                        "feature_change", row, old, reason=f"{old['feature']}->{feature}"
                    )
                )
                old_key = (tuple(old["pair"]), str(old["feature"]))
                detached_keys.discard(old_key)
                active.pop(old_key, None)
            elif same_root:
                old = same_root[0]
                close_episode(old, row, "pair_change")
                sequence.append(
                    _episode_event(
                        "pair_change", row, old, reason="root_pair_active_with_new_segment_pair"
                    )
                )
                old_key = (tuple(old["pair"]), str(old["feature"]))
                detached_keys.discard(old_key)
                active.pop(old_key, None)

            episode_id = len(episodes)
            episode = {
                "episode_id": episode_id,
                "pair": pair,
                "root_pair": tuple(record["root_pair"]),
                "feature": feature,
                "contact_identity": record.get("contact_identity"),
                "onset_time": float(row["time"]),
                "onset_step": int(row["step"]),
                "onset_n_nodes": int(row["n_nodes"]),
                "detachment_time": None,
                "detachment_step": None,
                "detachment_n_nodes": None,
                "observed_until_time": float(row["time"]),
                "residence_duration": 0.0,
                "censored_at_end": True,
                "remesh_closed": False,
                "close_reason": None,
                "sample_count": 1,
                "max_penetration_ratio": float(record["penetration"]) / max(diameter, 1.0e-15),
                "relative_tangential_slip": 0.0,
                "last_time": float(row["time"]),
                "last_record": record,
            }
            episodes.append(episode)
            is_recontact = (
                key in detached_keys
                and previous_row is not None
                and not bool(row["remeshed_since_previous"])
            )
            sequence.append(
                _episode_event("recontact" if is_recontact else "contact_onset", row, episode)
            )
            active[key] = episode

        previous_row = row

    if rows:
        final_row = rows[-1]
        for episode in active.values():
            close_episode(episode, final_row, "end_censored")

    for episode in episodes:
        episode.pop("last_record", None)
        episode["pair"] = list(episode["pair"])
        episode["root_pair"] = list(episode["root_pair"])

    onset_times = [float(episode["onset_time"]) for episode in episodes]
    detachment_times = [
        float(episode["detachment_time"])
        for episode in episodes
        if episode["detachment_time"] is not None
    ]
    recontacts = [event for event in sequence if event["event"] == "recontact"]
    signature = {
        "contact_observed": bool(episodes),
        "episode_count": len(episodes),
        "detachment_count": sum(event["event"] == "contact_detachment" for event in sequence),
        "recontact_count": len(recontacts),
        "feature_change_count": sum(event["event"] == "feature_change" for event in sequence),
        "pair_change_count": sum(event["event"] == "pair_change" for event in sequence),
        "remesh_boundary_count": sum(event["event"] == "remesh_boundary" for event in sequence),
        "censored_episode_count": sum(bool(episode["censored_at_end"]) for episode in episodes),
        "repeated_episode_signature": bool(len(episodes) >= 2 and recontacts),
        "contact_identity_pattern": _normalized_contact_identity(sequence),
        "censor_pattern": tuple(bool(episode["censored_at_end"]) for episode in episodes),
        "event_pattern": tuple(
            event["event"] for event in sequence if event["event"] != "active_continuation"
        ),
    }
    onset_intervals = [right - left for left, right in zip(onset_times, onset_times[1:])]
    detachment_intervals = [
        right - left for left, right in zip(detachment_times, detachment_times[1:])
    ]
    return {
        "episodes": episodes,
        "sequence": sequence,
        "signature": signature,
        "onset_intervals": onset_intervals,
        "detachment_intervals": detachment_intervals,
        "max_residence_duration": max(
            (float(episode["residence_duration"]) for episode in episodes), default=0.0
        ),
        "total_residence_duration": sum(
            float(episode["residence_duration"]) for episode in episodes
        ),
        "cumulative_relative_tangential_slip": cumulative_slip,
        "onset_times": onset_times,
    }


def _fold_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {
            "max_fold_count_proxy": 0,
            "fold_spacing_proxy": None,
            "fold_period_proxy": None,
            "max_curvature_concentration": 0.0,
            "fold_event_times": [],
            "fold_proxy_limit": "curvature sign changes/local peaks; not an experimental fold count",
        }
    spacing = [float(row["fold_spacing"]) for row in rows if row["fold_spacing"] is not None]
    event_times: list[float] = []
    previous_count = int(rows[0]["fold_count_proxy"])
    for row in rows[1:]:
        count = int(row["fold_count_proxy"])
        if count > previous_count:
            event_times.append(float(row["time"]))
        previous_count = max(previous_count, count)
    periods = np.diff(event_times).tolist() if len(event_times) >= 2 else []
    return {
        "max_fold_count_proxy": max(int(row["fold_count_proxy"]) for row in rows),
        "fold_spacing_proxy": None if not spacing else float(np.median(spacing)),
        "fold_period_proxy": None if not periods else float(np.median(periods)),
        "max_curvature_concentration": max(float(row["curvature_concentration"]) for row in rows),
        "fold_event_times": event_times,
        "fold_proxy_limit": "curvature sign changes/local peaks; not an experimental fold count",
    }


def _relative_difference(left: float | None, right: float | None) -> float | None:
    if left is None or right is None:
        return None
    return abs(float(left) - float(right)) / max(abs(float(right)), 1.0e-12)


def _time_match(left: Sequence[float], right: Sequence[float], tolerance: float) -> bool:
    return len(left) == len(right) and all(abs(a - b) <= tolerance for a, b in zip(left, right))


def _refinement_case_pairs(
    results: Sequence[Mapping[str, Any]], config: Mapping[str, Any]
) -> list[dict[str, Any]]:
    explicit = config.get("refinement_pairs", [])
    if explicit:
        return [dict(item) for item in explicit if isinstance(item, Mapping)]
    grouped: dict[tuple[str, str, str], list[Mapping[str, Any]]] = {}
    for result in results:
        effective = result["effective_config"]
        if effective.get("population") != "deterministic":
            continue
        axis = effective.get("refinement_axis")
        family = effective.get("refinement_family")
        if axis and family:
            grouped.setdefault((str(axis), str(family), "deterministic"), []).append(result)
    pairs = []
    for (axis, family, population), values in sorted(grouped.items()):
        if len(values) >= 2:
            ordered = sorted(
                values,
                key=lambda item: float(
                    item["effective_config"].get(
                        {
                            "temporal": "dt",
                            "spatial": "n_nodes",
                            "contact_stiffness": "contact_stiffness",
                        }.get(axis, "dt"),
                        0.0,
                    )
                ),
            )
            pairs.append(
                {
                    "name": f"{family}_{axis}",
                    "axis": axis,
                    "family": family,
                    "population": population,
                    "cases": [ordered[0]["case"], ordered[-1]["case"]],
                }
            )
    return pairs


def _compare_refinement(
    results: Sequence[Mapping[str, Any]], config: Mapping[str, Any]
) -> list[dict[str, Any]]:
    base = config["base"]
    output: list[dict[str, Any]] = []
    by_name = {result["case"]: result for result in results}
    for pair_spec in _refinement_case_pairs(results, config):
        names = list(pair_spec.get("cases", []))
        row: dict[str, Any] = {
            "name": pair_spec.get("name"),
            "axis": pair_spec.get("axis"),
            "family": pair_spec.get("family"),
            "population": pair_spec.get("population", "deterministic"),
            "cases": names,
            "status": "numerically-unresolved",
            "sequence_status": "numerically-unresolved",
            "penetration_status": "numerically-unresolved",
            "residence_status": "numerically-unresolved",
            "fold_status": "numerically-unresolved",
            "reason_codes": [],
        }
        if len(names) != 2 or any(name not in by_name for name in names):
            row["reason_codes"].append("paired_refinement_cases_required")
            output.append(row)
            continue
        left, right = (by_name[names[0]], by_name[names[1]])
        tolerance = dict(pair_spec.get("tolerances", {}))
        time_tolerance = float(tolerance.get("time", base.get("episode_time_tolerance", 0.08)))
        penetration_tolerance = float(
            tolerance.get("penetration", base.get("penetration_tolerance", 0.30))
        )
        residence_tolerance = float(
            tolerance.get("residence", base.get("residence_tolerance", 0.30))
        )
        fold_tolerance = float(tolerance.get("fold", base.get("fold_tolerance", 0.50)))
        fold_count_tolerance = int(tolerance.get("fold_count", base.get("fold_count_tolerance", 1)))
        remesh_tolerance = int(
            tolerance.get("remesh_boundaries", base.get("remesh_boundary_tolerance", 1))
        )
        identity_tolerance = float(
            tolerance.get("identity", base.get("contact_identity_tolerance", 0.08))
        )
        left_sig = dict(left["episode_signature"])
        right_sig = dict(right["episode_signature"])
        numerical_match = (
            left["numerical_status"] == "resolved"
            and right["numerical_status"] == "resolved"
        )
        if not numerical_match:
            row["reason_codes"].append("paired_case_numerically_unresolved")
        structural_keys = (
            "contact_observed",
            "episode_count",
            "detachment_count",
            "recontact_count",
            "feature_change_count",
            "pair_change_count",
            "censored_episode_count",
            "censor_pattern",
        )
        structure_match = all(left_sig[key] == right_sig[key] for key in structural_keys)
        left_event_pattern = tuple(
            str(event)
            for event in left_sig.get("event_pattern", ())
            if str(event) != "remesh_boundary"
        )
        right_event_pattern = tuple(
            str(event)
            for event in right_sig.get("event_pattern", ())
            if str(event) != "remesh_boundary"
        )
        event_pattern_match = left_event_pattern == right_event_pattern
        left_identity_pattern = _normalized_contact_identity(left["episode_sequence"])
        right_identity_pattern = _normalized_contact_identity(right["episode_sequence"])
        identity_match = _contact_identity_match(
            left_identity_pattern, right_identity_pattern, identity_tolerance
        )
        left_events = [
            event
            for event in left["episode_sequence"]
            if event["event"] not in {"active_continuation", "remesh_boundary"}
        ]
        right_events = [
            event
            for event in right["episode_sequence"]
            if event["event"] not in {"active_continuation", "remesh_boundary"}
        ]
        event_time_match = (
            len(left_events) == len(right_events)
            and all(
                left_event["event"] == right_event["event"]
                and abs(float(left_event["time"]) - float(right_event["time"]))
                <= time_tolerance
                for left_event, right_event in zip(left_events, right_events)
            )
        )
        remesh_match = (
            abs(left_sig["remesh_boundary_count"] - right_sig["remesh_boundary_count"])
            <= remesh_tolerance
        )
        time_match = _time_match(
            left["episode_onset_times"], right["episode_onset_times"], time_tolerance
        )
        if not structure_match or not event_pattern_match:
            row["reason_codes"].append("episode_structure_changed")
        if not identity_match:
            row["reason_codes"].append("episode_contact_identity_changed")
        if not remesh_match:
            row["reason_codes"].append("remesh_boundary_count_changed_beyond_tolerance")
        if not time_match or not event_time_match:
            row["reason_codes"].append("episode_times_outside_tolerance")
        row["sequence_status"] = (
            "resolved"
            if numerical_match
            and structure_match
            and event_pattern_match
            and identity_match
            and remesh_match
            and time_match
            and event_time_match
            else "numerically-unresolved"
        )

        penetration_difference = _relative_difference(
            left["max_penetration_ratio"], right["max_penetration_ratio"]
        )
        residence_difference = _relative_difference(
            left["max_residence_duration"], right["max_residence_duration"]
        )
        row["penetration_relative_difference"] = penetration_difference
        row["residence_relative_difference"] = residence_difference
        row["penetration_status"] = (
            "resolved"
            if numerical_match
            and penetration_difference is not None
            and penetration_difference <= penetration_tolerance
            else "numerically-unresolved"
        )
        row["residence_status"] = (
            "resolved"
            if numerical_match
            and residence_difference is not None
            and residence_difference <= residence_tolerance
            else "numerically-unresolved"
        )
        if row["penetration_status"] != "resolved" and numerical_match:
            row["reason_codes"].append("penetration_outside_tolerance")
        if row["residence_status"] != "resolved" and numerical_match:
            row["reason_codes"].append("residence_outside_tolerance")

        left_fold = left["fold_summary"]
        right_fold = right["fold_summary"]
        count_match = (
            abs(int(left_fold["max_fold_count_proxy"]) - int(right_fold["max_fold_count_proxy"]))
            <= fold_count_tolerance
        )
        spacing_difference = _relative_difference(
            left_fold["fold_spacing_proxy"], right_fold["fold_spacing_proxy"]
        )
        period_difference = _relative_difference(
            left_fold["fold_period_proxy"], right_fold["fold_period_proxy"]
        )
        concentration_difference = _relative_difference(
            left_fold["max_curvature_concentration"],
            right_fold["max_curvature_concentration"],
        )
        spacing_match = (
            left_fold["fold_spacing_proxy"] is None and right_fold["fold_spacing_proxy"] is None
        ) or (spacing_difference is not None and spacing_difference <= fold_tolerance)
        period_match = (
            left_fold["fold_period_proxy"] is None and right_fold["fold_period_proxy"] is None
        ) or (period_difference is not None and period_difference <= fold_tolerance)
        concentration_match = (
            concentration_difference is not None and concentration_difference <= fold_tolerance
        )
        optional_fold_match = spacing_match and period_match and concentration_match
        row["fold_spacing_relative_difference"] = spacing_difference
        row["fold_period_relative_difference"] = period_difference
        row["curvature_concentration_relative_difference"] = concentration_difference
        row["fold_status"] = (
            "resolved"
            if numerical_match and count_match and optional_fold_match
            else "numerically-unresolved"
        )
        if row["fold_status"] != "resolved" and numerical_match:
            row["reason_codes"].append("fold_proxy_outside_tolerance")
        row["status"] = (
            "resolved"
            if all(
                row[key] == "resolved"
                for key in (
                    "sequence_status",
                    "penetration_status",
                    "residence_status",
                    "fold_status",
                )
            )
            and left["numerical_status"] == right["numerical_status"] == "resolved"
            else "numerically-unresolved"
        )
        row["tolerances"] = {
            "time": time_tolerance,
            "penetration": penetration_tolerance,
            "residence": residence_tolerance,
            "fold": fold_tolerance,
            "fold_count": fold_count_tolerance,
            "remesh_boundaries": remesh_tolerance,
            "identity": identity_tolerance,
            "discretization": "remesh boundary count and episode structure are compared with explicit tolerances; exact timestamps and segment IDs are not required",
        }
        output.append(row)
    return output


def run_case(
    case: CaseSpec, base_config: Mapping[str, Any], *, git_revision: str | None = None
) -> dict[str, Any]:
    config = _effective_case(base_config, case)
    initial = _initial_state(config)
    initial_active, _ = _pair_records(
        initial, float(config["diameter"]), float(config["contact_stiffness"])
    )
    initial_contacts = nonlocal_segment_contacts(initial.positions, float(config["diameter"]))
    initial_min_gap = min((float(item.gap) for item in initial_contacts), default=float("inf"))
    initial_contact_violation = bool(initial_active) or initial_min_gap < 0.0
    initial_control = case.group == "initial_contact_control"
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
        rejected = accepted = 0
        rejection_counts: dict[str, int] = {}
        dt_collapsed = False
    else:
        if failure:
            trajectory = simulator.accepted_trajectory
        rows = _metrics_rows(trajectory, simulator, config)
        final = simulator.state
        events = simulator.event_log
        rejected = int(simulator.rejected_steps)
        accepted = int(simulator.accepted_steps)
        rejection_counts = _event_counts(simulator)
        dt_collapsed = _accepted_dt_collapsed(simulator)
    tracker = _episode_tracker(rows, float(config["diameter"]))
    observed_contact = bool(tracker["episodes"])
    repeated_episode_signature = bool(tracker["signature"]["repeated_episode_signature"])
    if repeated_episode_signature:
        evidence_classification = "repeated-folding"
    elif observed_contact:
        evidence_classification = "single-proxy/contact"
    else:
        evidence_classification = "no-contact"
    reasons: list[str] = []
    if failure:
        reasons.append("solver_failure")
    if config["expected_contact"] and not observed_contact:
        reasons.append("expected_contact_not_observed")
    if not config["expected_contact"] and observed_contact:
        reasons.append("unexpected_contact_observed")
    if config["expected_repeated"] and not repeated_episode_signature:
        reasons.append("repeated_folding_not_observed")
    if initial_control:
        if not initial_active:
            reasons.append("initial_contact_control_not_observed")
    elif initial_contact_violation:
        reasons.append("initial_contact_violation")
    crossing_rejection_in_events = any(
        event.get("event_type") == "step_attempt"
        and event.get("accepted") is False
        and event.get("reason") == "crossing_rejection"
        for event in events
    )
    if (
        crossing_rejection_in_events
        or rejection_counts.get("crossing_rejection", 0) > 0
        or (rows and any(row["crossing_pairs"] for row in rows))
    ):
        reasons.append("centerline_crossing_guard_observed")
    if dt_collapsed:
        reasons.append("accepted_dt_collapsed_below_requested")
    numerical_status = "numerically-unresolved" if reasons else "resolved"
    fold_summary = _fold_summary(rows)
    metadata = {
        "benchmark": "long-duration repeated contact/folding validation",
        "runner_schema_version": SCHEMA_VERSION,
        "case": case.name,
        "group": case.group,
        "population": case.population,
        "contact_law": "C1 segment penalty only",
        "legacy_node_contact": "disabled",
        "friction": "disabled",
        "adhesion": "disabled",
        "contact_history": "disabled; lineage and episodes are diagnostics only",
        "initial_contact_control": initial_control,
        "expected_repeated": bool(config["expected_repeated"]),
        "evidence_classification": evidence_classification,
        "measurement_limits": {
            "finite_penalty": "penetration is finite and stiffness/time-step dependent; not hard non-penetration",
            "fold_proxy": "curvature peaks/sign changes are not an experimental or topological fold count",
            "episode_identity": "lineage supports remesh diagnostics but does not implement contact history",
            "gray5": "not used for quantitative contact validation; input-QC/exploratory context only",
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
        "population": case.population,
        "effective_config": config,
        "expected_contact": bool(config["expected_contact"]),
        "initial_condition": "initial-contact-control"
        if initial_control
        else "non-contact-required",
        "initial_active_contact_pairs": len(initial_active),
        "initial_min_gap": None if not math.isfinite(initial_min_gap) else initial_min_gap,
        "classification": "initial-contact-control"
        if initial_control
        else ("contact-observed" if observed_contact else "no-contact-observed"),
        "evidence_classification": evidence_classification,
        "expected_repeated": bool(config["expected_repeated"]),
        "repeated_episode_signature": repeated_episode_signature,
        "numerical_status": numerical_status,
        "numerical_reason_codes": reasons,
        "failure_reason": failure,
        "accepted_steps": accepted,
        "rejected_steps": rejected,
        "requested_dt": float(config["dt"]),
        "accepted_dt_values": accepted_dt_values,
        "accepted_dt_min": min(accepted_dt_values) if accepted_dt_values else None,
        "accepted_dt_max": max(accepted_dt_values) if accepted_dt_values else None,
        "accepted_dt_mean": float(np.mean(accepted_dt_values)) if accepted_dt_values else None,
        "rejection_reason_counts": rejection_counts,
        "crossing_guard_rejections": rejection_counts.get("crossing_rejection", 0),
        "event_count": len(events),
        "episodes": tracker["episodes"],
        "episode_sequence": tracker["sequence"],
        "episode_signature": tracker["signature"],
        "episode_onset_times": tracker["onset_times"],
        "onset_intervals": tracker["onset_intervals"],
        "detachment_intervals": tracker["detachment_intervals"],
        "episode_count": len(tracker["episodes"]),
        "recontact_count": tracker["signature"]["recontact_count"],
        "max_residence_duration": tracker["max_residence_duration"],
        "total_residence_duration": tracker["total_residence_duration"],
        "cumulative_relative_tangential_slip": tracker["cumulative_relative_tangential_slip"],
        "max_penetration_ratio": max(
            (float(row["max_penetration_ratio"]) for row in rows), default=0.0
        ),
        "max_endpoint_motion": max(
            (float(row["endpoint_motion_max"]) for row in rows), default=0.0
        ),
        "max_curvature_concentration": max(
            (float(row["curvature_concentration"]) for row in rows), default=0.0
        ),
        "fold_summary": fold_summary,
        "metrics_rows": rows,
        "manifest": manifest,
        "ccd_contract": "swept centerline crossing guard only; finite-radius CCD and hard non-penetration are not implemented",
    }


def _summary_row(result: Mapping[str, Any]) -> dict[str, Any]:
    fold = result["fold_summary"]
    return {
        "case": result["case"],
        "group": result["group"],
        "population": result["population"],
        "classification": result["classification"],
        "evidence_classification": result["evidence_classification"],
        "expected_repeated": result["expected_repeated"],
        "repeated_episode_signature": result["repeated_episode_signature"],
        "numerical_status": result["numerical_status"],
        "numerical_reason_codes": json.dumps(result["numerical_reason_codes"], sort_keys=True),
        "requested_dt": result["requested_dt"],
        "accepted_dt_min": result["accepted_dt_min"],
        "accepted_dt_mean": result["accepted_dt_mean"],
        "accepted_dt_max": result["accepted_dt_max"],
        "accepted_steps": result["accepted_steps"],
        "rejected_steps": result["rejected_steps"],
        "episode_count": result["episode_count"],
        "recontact_count": result["recontact_count"],
        "detachment_count": result["episode_signature"]["detachment_count"],
        "remesh_boundary_count": result["episode_signature"]["remesh_boundary_count"],
        "max_residence_duration": result["max_residence_duration"],
        "total_residence_duration": result["total_residence_duration"],
        "onset_intervals": json.dumps(result["onset_intervals"]),
        "detachment_intervals": json.dumps(result["detachment_intervals"]),
        "max_penetration_ratio": result["max_penetration_ratio"],
        "cumulative_relative_tangential_slip": result["cumulative_relative_tangential_slip"],
        "max_endpoint_motion": result["max_endpoint_motion"],
        "max_curvature_concentration": result["max_curvature_concentration"],
        "max_fold_count_proxy": fold["max_fold_count_proxy"],
        "fold_spacing_proxy": fold["fold_spacing_proxy"],
        "fold_period_proxy": fold["fold_period_proxy"],
        "fold_curvature_concentration": fold["max_curvature_concentration"],
        "crossing_guard_rejections": result["crossing_guard_rejections"],
        "episode_signature": json.dumps(result["episode_signature"], sort_keys=True),
        "failure_reason": result["failure_reason"],
    }


def _csv_cell(value: Any) -> Any:
    normalized = _jsonable(value)
    if isinstance(normalized, (Mapping, list, tuple)):
        return json.dumps(normalized, sort_keys=True, separators=(",", ":"))
    return normalized


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
        writer.writerows(
            {field: _csv_cell(row.get(field)) for field in fields} for row in rows
        )


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
            _strict_bool(
                item.get(
                    "expected_contact",
                    item.get("overrides", {}).get(
                        "expected_contact", config["base"].get("expected_contact", False)
                    ),
                ),
                "expected_contact",
            ),
            str(item.get("population", "deterministic")),
            None if item.get("refinement_axis") is None else str(item["refinement_axis"]),
            None if item.get("refinement_family") is None else str(item["refinement_family"]),
            None if item.get("refinement_role") is None else str(item["refinement_role"]),
            _strict_bool(
                item.get(
                    "expected_repeated",
                    item.get("overrides", {}).get(
                        "expected_repeated", config["base"].get("expected_repeated", False)
                    ),
                ),
                "expected_repeated",
            ),
        )
        for item in raw_cases
    ]
    output.mkdir(parents=True, exist_ok=True)
    revision = git_revision if git_revision is not None else detect_git_revision(Path.cwd())
    results = [run_case(case, config["base"], git_revision=revision) for case in cases]
    refinement = _compare_refinement(results, config)
    summary = [_summary_row(result) for result in results]
    metrics = [
        {"case": result["case"], **row} for result in results for row in result["metrics_rows"]
    ]
    config_hash = sha256_hex(canonical_json_bytes(_jsonable(config)))
    compact_manifest = {
        "manifest_schema_version": "continuum-filament-repeated-folding-compact-1",
        "benchmark_schema_version": SCHEMA_VERSION,
        "benchmark": "long-duration repeated contact/folding validation",
        "git_revision": revision,
        "config_hash": config_hash,
        "contact_law": "C1 frictionless finite-radius segment penalty only",
        "legacy_node_contact": "disabled",
        "friction": "disabled",
        "adhesion": "disabled",
        "contact_history": "disabled; episode tracking is diagnostic only",
        "gray5": "input-QC/exploratory context only; not quantitative contact validation",
        "case_order": [result["case"] for result in results],
        "case_manifests": {
            result["case"]: {
                "population": result["population"],
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
            "stored_time_series": "compact accepted-state metrics CSV with JSON contact records",
        },
        "limitations": [
            "finite penalty penetration is not a hard non-penetration proof",
            "episode identity is not a contact-history law",
            "fold spacing/period/count are morphology proxies, not experimental fold counts",
            "exact episode timestamps and current segment IDs are not refinement equality criteria",
        ],
    }
    suite = {
        "schema_version": SCHEMA_VERSION,
        "benchmark": "long-duration repeated contact/folding validation",
        "git_revision": revision,
        "config_hash": config_hash,
        "refinement": refinement,
        "population_counts": {
            population: sum(result["population"] == population for result in results)
            for population in ("deterministic", "shape_only_sensitivity")
        },
        "results": [
            {key: value for key, value in result.items() if key not in {"metrics_rows", "manifest"}}
            for result in results
        ],
        "compact_manifest": compact_manifest,
    }
    _write_csv(output / "summary.csv", summary)
    _write_json(output / "summary.json", summary)
    _write_csv(output / "metrics.csv", metrics)
    _write_json(output / "refinement_summary.json", refinement)
    _write_csv(output / "refinement_summary.csv", refinement)
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
        print(f"repeated folding validation error: {exc}", file=sys.stderr)
        return 2
    unresolved = [item for item in suite["results"] if item["numerical_status"] != "resolved"]
    unresolved_refinement = [item for item in suite["refinement"] if item["status"] != "resolved"]
    print(
        json.dumps(
            {
                "output": str(args.output),
                "cases": len(suite["results"]),
                "numerically_unresolved_cases": len(unresolved),
                "numerically_unresolved_refinements": len(unresolved_refinement),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
