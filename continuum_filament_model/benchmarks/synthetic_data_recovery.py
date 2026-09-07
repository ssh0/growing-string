"""Generate and analyse bounded raw-like synthetic filament observations.

The runner is an observation-contract and identifiability check, not an
experiment substitute.  It deliberately keeps generated CSV/JSON artifacts in
an output directory supplied by the caller; only compact aggregate summaries
are intended for version control.

Example::

    TMP_DIR=$(mktemp -d /tmp/growing-string-synthetic.XXXXXX)
    PYTHONPATH=continuum_filament_model/src \
      python continuum_filament_model/benchmarks/synthetic_data_recovery.py \
      --config continuum_filament_model/benchmarks/configs/synthetic_data_recovery.json \
      --output "$TMP_DIR"

The mechanical reference quantities used by the fixtures are recorded in the
raw metadata.  The sinusoidal fixture uses the same first-mode convention as
P1B, while configuration conversion and dimensionless groups are obtained from
P1B/P1B.2 helpers rather than reimplementing the core filament solver.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

if __package__ in {None, ""}:
    _HERE = Path(__file__).resolve()
    _SRC = _HERE.parents[1] / "src"
    if str(_SRC) not in sys.path:
        sys.path.insert(0, str(_SRC))
    if str(_HERE.parent) not in sys.path:
        sys.path.insert(0, str(_HERE.parent))
    from buckling_benchmark import (  # type: ignore  # noqa: E402
        dimensionless_groups,
        initial_perturbed_state,
    )
    from p1b2_experiments import target_overrides  # type: ignore  # noqa: E402
else:  # pragma: no cover - package execution is useful to downstream callers
    from .buckling_benchmark import dimensionless_groups, initial_perturbed_state
    from .p1b2_experiments import target_overrides


SCHEMA_VERSION = "continuum-filament-synthetic-recovery-1"
ANALYSIS_REVISION = "synthetic-recovery-v1"
RAW_FIELDS = (
    "time",
    "frame",
    "filament_id",
    "point_id",
    "x",
    "y",
    "width",
    "quality",
    "missing",
    "flags",
)


class SyntheticRecoveryError(ValueError):
    """Invalid synthetic-data configuration or failed acceptance check."""


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


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        _jsonable(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_canonical_json_bytes(value) + b"\n")


def _safe_token(value: Any) -> str:
    return str(value).replace("-", "m").replace(".", "p")


def _write_csv_bytes(rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> bytes:
    from io import StringIO

    stream = StringIO()
    writer = csv.DictWriter(
        stream,
        fieldnames=list(fields),
        extrasaction="ignore",
        lineterminator="\n",
    )
    writer.writeheader()
    for row in rows:
        encoded: dict[str, Any] = {}
        for field in fields:
            value = row.get(field)
            if value is None or (isinstance(value, float) and not math.isfinite(value)):
                encoded[field] = ""
            elif isinstance(value, bool):
                encoded[field] = "true" if value else "false"
            else:
                encoded[field] = value
        writer.writerow(encoded)
    return stream.getvalue().encode("utf-8")


def _as_float(value: Any, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise SyntheticRecoveryError(f"{name} must be numeric") from exc
    if not math.isfinite(result):
        raise SyntheticRecoveryError(f"{name} must be finite")
    return result


def _validate_config(config: Mapping[str, Any]) -> None:
    base = config.get("base")
    truth = config.get("truth")
    fixtures = config.get("fixtures")
    scenarios = config.get("scenarios")
    if not isinstance(base, Mapping) or not isinstance(truth, Mapping):
        raise SyntheticRecoveryError("config requires object-valued base and truth")
    if not isinstance(fixtures, Sequence) or not fixtures:
        raise SyntheticRecoveryError("config requires non-empty fixtures")
    if not isinstance(scenarios, Sequence) or not scenarios:
        raise SyntheticRecoveryError("config requires non-empty scenarios")
    for key in ("n_points", "n_frames"):
        value = int(base.get(key, 0))
        if value < 5:
            raise SyntheticRecoveryError(f"base.{key} must be >= 5")
    if _as_float(base.get("pixel_size_m_per_px"), "pixel_size_m_per_px") <= 0.0:
        raise SyntheticRecoveryError("pixel_size_m_per_px must be positive")
    if _as_float(base.get("frame_interval"), "frame_interval") <= 0.0:
        raise SyntheticRecoveryError("frame_interval must be positive")
    for key in ("length", "EA", "EI", "drag", "diameter"):
        if _as_float(truth.get(key), f"truth.{key}") <= 0.0:
            raise SyntheticRecoveryError(f"truth.{key} must be positive")
    fixture_names = set()
    for fixture in fixtures:
        if not isinstance(fixture, Mapping) or "name" not in fixture:
            raise SyntheticRecoveryError("fixture entries require name")
        name = str(fixture["name"])
        if name in fixture_names:
            raise SyntheticRecoveryError(f"duplicate fixture: {name}")
        fixture_names.add(name)
        if name not in {"straight", "sinusoidal_buckling", "arc", "growth"}:
            raise SyntheticRecoveryError(f"unknown fixture: {name}")
    seeds = config.get("seeds", [])
    if len(seeds) < 2 or len(set(int(seed) for seed in seeds)) != len(seeds):
        raise SyntheticRecoveryError("seeds must contain at least two distinct integers")
    for scenario in scenarios:
        if not isinstance(scenario, Mapping) or "name" not in scenario:
            raise SyntheticRecoveryError("scenario entries require name")
        noise = scenario.get("noise", {})
        if not isinstance(noise, Mapping):
            raise SyntheticRecoveryError("scenario.noise must be an object")
        for key in ("localization_std_px", "width_noise_px", "blur_alpha", "scale_error"):
            value = _as_float(noise.get(key, 0.0), f"noise.{key}")
            if key == "blur_alpha" and not 0.0 <= value < 1.0:
                raise SyntheticRecoveryError("blur_alpha must be in [0, 1)")
            if key == "scale_error" and value <= -1.0:
                raise SyntheticRecoveryError("scale_error must be greater than -1")
            if key in {"localization_std_px", "width_noise_px"} and value < 0.0:
                raise SyntheticRecoveryError(f"{key} must be non-negative")


def _fixture_lookup(config: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    return {str(item["name"]): item for item in config["fixtures"]}


def _api_reference(
    fixture: str,
    truth: Mapping[str, Any],
    base: Mapping[str, Any],
    amplitude: float,
) -> dict[str, Any]:
    """Call the existing P1B/P1B.2 conversion APIs for audit metadata."""

    length = float(truth["length"])
    ea = float(truth["EA"])
    ei = float(truth["EI"])
    drag = float(truth["drag"])
    growth = float(truth.get("g", 0.0))
    n_points = int(base["n_points"])
    p1b_config = {
        "length": length,
        "n_nodes": n_points,
        "axial_stiffness": ea,
        "bending_stiffness": ei,
        "drag_density": drag,
        "growth_rate": growth,
        "amplitude": amplitude,
        "dt": float(base["frame_interval"]),
        "t_end": float(base["frame_interval"]) * (int(base["n_frames"]) - 1),
        "a_max_factor": 2.0,
    }
    groups = dimensionless_groups(p1b_config)
    target = target_overrides(
        p1b_config,
        target_gb=float(groups["G_b"]),
        target_chi=float(groups["chi"]),
    )
    initial = initial_perturbed_state(p1b_config, seed=None)
    return {
        "p1b_dimensionless_groups": groups,
        "p1b2_target_conversion": target,
        "p1b_initial_state_hash": _sha256(
            _canonical_json_bytes(
                {
                    "positions": initial.positions,
                    "rest_lengths": initial.rest_lengths,
                }
            )
        ),
        "fixture_api_scope": (
            "P1B initial sine fixture and dimensionless groups; "
            "P1B.2 target_overrides conversion; no core physics copied"
        ),
        "fixture": fixture,
    }


def _truth_positions(
    fixture: str,
    time: float,
    truth: Mapping[str, Any],
    base: Mapping[str, Any],
    fixture_config: Mapping[str, Any],
) -> np.ndarray:
    length = float(truth["length"])
    s = np.linspace(0.0, 1.0, int(base["n_points"]))
    if fixture == "straight":
        current_length = length
        return np.column_stack((current_length * s, np.zeros_like(s)))
    if fixture == "growth":
        current_length = length * math.exp(float(truth["g"]) * time)
        return np.column_stack((current_length * s, np.zeros_like(s)))
    if fixture == "sinusoidal_buckling":
        amplitude = float(fixture_config.get("amplitude", 0.12))
        rate = float(truth["EI"]) * math.pi**4 / (float(truth["drag"]) * length**4)
        y = amplitude * math.exp(-rate * time) * np.sin(math.pi * s)
        # This is exactly the deterministic P1B sine convention at t=0.
        return np.column_stack((length * s, y))
    if fixture == "arc":
        radius = float(fixture_config.get("radius", 3.0))
        angle = float(fixture_config.get("angle", 1.0))
        theta = (s - 0.5) * angle
        return np.column_stack((radius * np.sin(theta), radius * (1.0 - np.cos(theta))))
    raise SyntheticRecoveryError(f"unsupported fixture: {fixture}")


def _event(
    event_type: str,
    frame: int | None,
    detail: str,
    affected_points: Sequence[int] = (),
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "event_type": event_type,
        "frame": frame,
        "affected_points": [int(value) for value in affected_points],
        "detail": detail,
    }


def generate_raw_run(
    fixture: str,
    scenario: Mapping[str, Any],
    seed: int,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    """Generate a deterministic raw-like CSV payload and contract metadata."""

    base = config["base"]
    truth = config["truth"]
    fixture_config = _fixture_lookup(config)[fixture]
    n_frames = int(base["n_frames"])
    frame_interval = float(base["frame_interval"])
    true_pixel_size = float(base["pixel_size_m_per_px"])
    noise = dict(scenario.get("noise", {}))
    rng = np.random.default_rng(int(seed))
    rows: list[dict[str, Any]] = []
    events: list[dict[str, Any]] = []
    frame_drops = {int(value) for value in noise.get("frame_drop", [])}
    missing_by_frame = {
        int(key): [int(point) for point in value]
        for key, value in dict(noise.get("missing_points", {})).items()
    }
    order_frame = noise.get("point_order_error_frame")
    order_frame = None if order_frame is None else int(order_frame)
    scale_error = float(noise.get("scale_error", 0.0))
    blur_alpha = float(noise.get("blur_alpha", 0.0))
    localization_std = float(noise.get("localization_std_px", 0.0))
    width_noise = float(noise.get("width_noise_px", 0.0))
    width_variation = float(noise.get("width_variation_fraction", 0.0))
    if scale_error:
        events.append(_event("pixel_scale_mismatch", None, f"coordinate scale multiplied by 1+{scale_error}"))
    if localization_std:
        events.append(_event("localization_noise", None, f"Gaussian noise std={localization_std} px"))
    if blur_alpha:
        events.append(_event("blur", None, f"neighbor smoothing alpha={blur_alpha}"))
    if width_variation:
        events.append(_event("width_variation", None, f"fraction={width_variation}"))

    for frame in range(n_frames):
        time = frame * frame_interval
        truth_positions = _truth_positions(fixture, time, truth, base, fixture_config)
        if frame in frame_drops:
            events.append(_event("frame_drop", frame, "frame omitted from raw centerline"))
            continue
        points_px = truth_positions / true_pixel_size
        if scale_error:
            points_px = points_px * (1.0 + scale_error)
        if blur_alpha:
            blurred = points_px.copy()
            blurred[1:-1] = (1.0 - blur_alpha) * points_px[1:-1] + blur_alpha * 0.5 * (
                points_px[:-2] + points_px[2:]
            )
            points_px = blurred
        if localization_std:
            points_px = points_px + rng.normal(0.0, localization_std, size=points_px.shape)
        missing_points = set(missing_by_frame.get(frame, []))
        if missing_points:
            events.append(_event("missing_centerline", frame, "point rows omitted/flagged", sorted(missing_points)))
        if frame == order_frame:
            events.append(_event("point_order_error", frame, "two point identifiers swapped", [12, 13]))
        if frame in missing_by_frame:
            # Keep all source rows in the raw-like input, but mark missing
            # values explicitly so the contract checker can distinguish them
            # from a parser failure.
            pass
        point_ids = list(range(len(points_px)))
        if frame == order_frame and len(point_ids) > 14:
            point_ids[12], point_ids[13] = point_ids[13], point_ids[12]
        width_base_px = float(truth["diameter"]) / true_pixel_size
        for source_point, point_id in enumerate(point_ids):
            is_missing = source_point in missing_points
            width = width_base_px * (
                1.0 + width_variation * math.sin(2.0 * math.pi * source_point / max(len(points_px) - 1, 1))
            )
            if width_noise:
                width += float(rng.normal(0.0, width_noise))
            row: dict[str, Any] = {
                "time": time,
                "frame": frame,
                "filament_id": "synthetic-001",
                "point_id": point_id,
                "x": None if is_missing else float(points_px[source_point, 0]),
                "y": None if is_missing else float(points_px[source_point, 1]),
                "width": None if is_missing else float(width),
                "quality": 0.0 if is_missing else float(max(0.05, 1.0 - 0.25 * blur_alpha)),
                "missing": is_missing,
                "flags": "missing" if is_missing else ("point_order_error" if frame == order_frame else ""),
            }
            rows.append(row)
    events.sort(key=lambda item: (-1 if item["frame"] is None else int(item["frame"]), item["event_type"]))
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "data_contract": "continuum-filament-experiment-data-contract-1",
        "source": "synthetic_raw_like",
        "image_id": f"synthetic://{fixture}/{scenario['name']}/seed-{seed}",
        "filament_id": "synthetic-001",
        "units": {"time": "s", "coordinate": "pixel", "width": "pixel"},
        "frame_interval": frame_interval,
        "expected_frames": list(range(n_frames)),
        "pixel_size_m_per_px": true_pixel_size,
        "analysis_pixel_size_m_per_px": true_pixel_size,
        "temperature": None,
        "nutrient_condition": None,
        "agar_condition": None,
        "filament_diameter_method": "synthetic width field; median is a proxy only",
        "centerline_extraction_method": "synthetic analytic fixture + bounded corruption",
        "centerline_extraction_revision": ANALYSIS_REVISION,
        "missing_data_policy": "missing rows are retained with missing=true and null coordinates",
        "excluded_frames": sorted(frame_drops),
        "initial_shape": {
            "kind": fixture,
            "parameters": _jsonable(fixture_config),
            "initial_time": 0.0,
        },
        "truth_parameters": {
            "g": float(truth["g"]),
            "EI": float(truth["EI"]),
            "EA": float(truth["EA"]),
            "drag": float(truth["drag"]),
            "diameter": float(truth["diameter"]),
        },
        "fixture": fixture,
        "scenario": scenario["name"],
        "seed": int(seed),
        "noise_config": _jsonable(noise),
        "api_reference": _api_reference(
            fixture,
            truth,
            base,
            float(fixture_config.get("amplitude", 0.12)),
        ),
    }
    csv_bytes = _write_csv_bytes(rows, RAW_FIELDS)
    return {
        "rows": rows,
        "events": events,
        "metadata": metadata,
        "csv_bytes": csv_bytes,
        "raw_synthetic_hash": _sha256(csv_bytes),
    }


def _read_csv_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", newline="") as stream:
        for raw in csv.DictReader(stream):
            def optional_float(name: str) -> float | None:
                value = raw.get(name, "")
                return None if value in (None, "") else float(value)

            rows.append(
                {
                    "time": float(raw["time"]),
                    "frame": int(raw["frame"]),
                    "filament_id": raw["filament_id"],
                    "point_id": int(raw["point_id"]),
                    "x": optional_float("x"),
                    "y": optional_float("y"),
                    "width": optional_float("width"),
                    "quality": float(raw["quality"]),
                    "missing": raw["missing"].lower() == "true",
                    "flags": raw.get("flags", ""),
                }
            )
    return rows


def _arc_resample(points: np.ndarray, n_points: int) -> tuple[np.ndarray, float]:
    differences = np.diff(points, axis=0)
    lengths = np.linalg.norm(differences, axis=1)
    if len(lengths) == 0 or not np.isfinite(lengths).all() or np.any(lengths <= 1.0e-12):
        raise SyntheticRecoveryError("degenerate centerline cannot be arc-length reparameterized")
    total = float(np.sum(lengths))
    cumulative = np.concatenate(([0.0], np.cumsum(lengths)))
    target = np.linspace(0.0, total, n_points)
    result = np.column_stack(
        [np.interp(target, cumulative, points[:, axis]) for axis in range(points.shape[1])]
    )
    return result, total


def _curvature(points: np.ndarray) -> np.ndarray:
    if len(points) < 3:
        return np.empty(0, dtype=float)
    before = points[:-2]
    center = points[1:-1]
    after = points[2:]
    first = center - before
    second = after - center
    first_length = np.linalg.norm(first, axis=1)
    second_length = np.linalg.norm(second, axis=1)
    tangent_before = first / first_length[:, None]
    tangent_after = second / second_length[:, None]
    angles = np.arctan2(
        tangent_before[:, 0] * tangent_after[:, 1] - tangent_before[:, 1] * tangent_after[:, 0],
        np.sum(tangent_before * tangent_after, axis=1),
    )
    return angles / (0.5 * (first_length + second_length))


def _fit_slope(times: Sequence[float], values: Sequence[float]) -> float | None:
    if len(times) < 3:
        return None
    x = np.asarray(times, dtype=float)
    y = np.asarray(values, dtype=float)
    valid = np.isfinite(x) & np.isfinite(y)
    if np.count_nonzero(valid) < 3:
        return None
    x = x[valid]
    y = y[valid]
    if float(np.ptp(x)) <= 1.0e-12:
        return None
    return float(np.polyfit(x, y, 1)[0])


def _point_order_diagnostic(points: np.ndarray) -> tuple[bool, float]:
    lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
    if len(lengths) < 3 or not np.isfinite(lengths).all():
        return True, float("inf")
    median = float(np.median(lengths))
    if median <= 1.0e-12:
        return True, float("inf")
    score = max(float(np.max(lengths)) / median, median / max(float(np.min(lengths)), 1.0e-12))
    return bool(score > 1.8), score


def _projection_amplitude(points: np.ndarray) -> float:
    start, end = points[0], points[-1]
    chord = end - start
    chord_length = float(np.linalg.norm(chord))
    if chord_length <= 1.0e-12:
        return float("nan")
    tangent = chord / chord_length
    normal = np.asarray([-tangent[1], tangent[0]])
    transverse = (points - start) @ normal
    u = np.linspace(0.0, 1.0, len(points))
    integrate = getattr(np, "trapezoid", np.trapz)
    return float(2.0 * integrate(transverse * np.sin(math.pi * u), u))


def _contract_checks(rows: Sequence[Mapping[str, Any]], metadata: Mapping[str, Any]) -> dict[str, Any]:
    issues: list[str] = []
    times = [float(row["time"]) for row in rows]
    if any(right < left for left, right in zip(times, times[1:])):
        issues.append("time_not_monotonic")
    expected_frames = {int(value) for value in metadata.get("expected_frames", [])}
    observed_frames = {int(row["frame"]) for row in rows}
    missing_frames = sorted(expected_frames - observed_frames)
    if missing_frames:
        issues.append("frame_drop")
    by_frame: dict[int, list[Mapping[str, Any]]] = {}
    for row in rows:
        by_frame.setdefault(int(row["frame"]), []).append(row)
        if not bool(row["missing"]):
            if row["x"] is None or row["y"] is None or not np.isfinite([row["x"], row["y"]]).all():
                issues.append("nonfinite_coordinate")
    duplicate_point_ids = []
    for frame, frame_rows in by_frame.items():
        point_ids = [int(row["point_id"]) for row in frame_rows]
        if len(point_ids) != len(set(point_ids)):
            duplicate_point_ids.append(frame)
    if duplicate_point_ids:
        issues.append("duplicate_point_id")
    if not metadata.get("pixel_size_m_per_px"):
        issues.append("missing_pixel_scale")
    return {
        "valid": not issues,
        "issues": sorted(set(issues)),
        "missing_frames": missing_frames,
        "observed_frame_count": len(observed_frames),
        "row_count": len(rows),
    }


def analyse_raw_run(
    raw_path: Path,
    metadata: Mapping[str, Any],
    raw_events: Sequence[Mapping[str, Any]],
    fixture: str,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate, arc-length reparameterize, and measure a raw-like CSV."""

    rows = _read_csv_rows(raw_path)
    checks = _contract_checks(rows, metadata)
    analysis_scale = float(metadata.get("analysis_pixel_size_m_per_px", metadata["pixel_size_m_per_px"]))
    quality_threshold = float(config["base"].get("quality_threshold", 0.5))
    n_resample = int(config["base"].get("resample_points", 31))
    by_frame: dict[int, list[Mapping[str, Any]]] = {}
    for row in rows:
        by_frame.setdefault(int(row["frame"]), []).append(row)
    frame_rows: list[dict[str, Any]] = []
    event_types = Counter(str(item["event_type"]) for item in raw_events)
    for frame in [int(value) for value in metadata.get("expected_frames", [])]:
        candidates = by_frame.get(frame, [])
        valid_rows = [
            row
            for row in candidates
            if not row["missing"]
            and float(row["quality"]) >= quality_threshold
            and row["x"] is not None
            and row["y"] is not None
        ]
        result: dict[str, Any] = {
            "frame": frame,
            "time": frame * float(metadata["frame_interval"]),
            "n_rows": len(candidates),
            "n_valid_points": len(valid_rows),
            "valid": False,
            "censor_reason": None,
            "point_order_error": False,
            "point_order_score": None,
            "length": None,
            "mean_abs_curvature": None,
            "max_abs_curvature": None,
            "mean_width": None,
            "mean_quality": None,
            "amplitude": None,
            "resampled_positions": None,
        }
        if not candidates:
            result["censor_reason"] = "frame_drop"
            frame_rows.append(result)
            continue
        has_missing_rows = any(bool(row["missing"]) for row in candidates)
        if len(valid_rows) < int(config["base"].get("min_points", 7)) or has_missing_rows:
            result["censor_reason"] = "insufficient_points_or_missing"
            frame_rows.append(result)
            continue
        ordered = sorted(valid_rows, key=lambda row: int(row["point_id"]))
        points = np.asarray([[float(row["x"]), float(row["y"])] for row in ordered], dtype=float)
        points *= analysis_scale
        order_error, order_score = _point_order_diagnostic(points)
        result["point_order_error"] = order_error
        result["point_order_score"] = order_score
        if order_error:
            result["censor_reason"] = "point_order_error"
            frame_rows.append(result)
            continue
        try:
            resampled, length = _arc_resample(points, n_resample)
        except SyntheticRecoveryError:
            result["censor_reason"] = "arc_length_failed"
            frame_rows.append(result)
            continue
        curvature = _curvature(resampled)
        widths = [float(row["width"]) * analysis_scale for row in ordered if row["width"] is not None]
        result.update(
            {
                "valid": True,
                "length": length,
                "mean_abs_curvature": float(np.mean(np.abs(curvature))) if len(curvature) else 0.0,
                "max_abs_curvature": float(np.max(np.abs(curvature))) if len(curvature) else 0.0,
                "mean_width": float(np.mean(widths)) if widths else None,
                "mean_quality": float(np.mean([float(row["quality"]) for row in ordered])),
                "amplitude": _projection_amplitude(resampled),
                "resampled_positions": resampled,
            }
        )
        frame_rows.append(result)
    valid_frames = [row for row in frame_rows if row["valid"]]
    velocities: list[float] = []
    for previous, current in zip(valid_frames, valid_frames[1:]):
        dt = float(current["time"]) - float(previous["time"])
        if dt > 0.0:
            velocities.append(float(np.mean(np.linalg.norm(
                current["resampled_positions"] - previous["resampled_positions"], axis=1
            )) / dt))
    for row in frame_rows:
        row.pop("resampled_positions", None)
    truth = config["truth"]
    recovered: dict[str, Any] = {
        "growth_rate": None,
        "EI_over_drag": None,
        "diameter_proxy": None,
        "velocity_median": float(np.median(velocities)) if velocities else None,
    }
    g_truth = float(truth["g"])
    if fixture == "growth":
        growth_rows = [row for row in valid_frames if row["length"] and row["length"] > 0.0]
        slope = _fit_slope(
            [float(row["time"]) for row in growth_rows],
            [math.log(float(row["length"])) for row in growth_rows],
        )
        recovered["growth_rate"] = slope
    if fixture == "sinusoidal_buckling":
        amplitude_rows = [row for row in valid_frames if row["amplitude"] is not None and abs(float(row["amplitude"])) > 1.0e-12]
        slope = _fit_slope(
            [float(row["time"]) for row in amplitude_rows],
            [math.log(abs(float(row["amplitude"]))) for row in amplitude_rows],
        )
        if slope is not None and amplitude_rows:
            observed_length = float(amplitude_rows[0]["length"])
            recovered["EI_over_drag"] = float(-slope * observed_length**4 / math.pi**4)
    width_rows = [row for row in valid_frames if row["mean_width"] is not None]
    if width_rows:
        recovered["diameter_proxy"] = float(np.median([float(row["mean_width"]) for row in width_rows]))
    truth_identifiable = {
        "growth_rate": g_truth if fixture == "growth" else None,
        "EI_over_drag": float(truth["EI"]) / float(truth["drag"]) if fixture == "sinusoidal_buckling" else None,
        "diameter_proxy": float(truth["diameter"]),
    }
    errors: dict[str, float | None] = {}
    for key, truth_value in truth_identifiable.items():
        estimate = recovered.get(key)
        if truth_value is None or estimate is None:
            errors[key] = None
        else:
            errors[key] = abs(float(estimate) - float(truth_value)) / max(abs(float(truth_value)), 1.0e-12)
    detected_events = list(raw_events)
    if any(row["point_order_error"] for row in frame_rows):
        detected_events.append(_event("detected_point_order_error", None, "frame QC rejected a non-monotone segment sequence"))
    if any(row["censor_reason"] == "frame_drop" for row in frame_rows):
        detected_events.append(_event("detected_frame_drop", None, "expected frame absent from raw input"))
    if any(row["censor_reason"] == "insufficient_points_or_missing" for row in frame_rows):
        detected_events.append(_event("detected_missing_points", None, "frame censored by point/quality threshold"))
    if any(event["event_type"] == "pixel_scale_mismatch" for event in raw_events):
        detected_events.append(_event("detected_pixel_scale_mismatch", None, "synthetic calibration perturbation is declared in metadata"))
    if any(event["event_type"] == "blur" for event in raw_events):
        detected_events.append(_event("detected_blur", None, "blur declaration is retained; no deblurring is applied"))
    if any(event["event_type"] == "width_variation" for event in raw_events):
        detected_events.append(_event("detected_width_variation", None, "width variation is summarized as a diameter proxy, not fitted away"))
    detected_types = Counter(str(item["event_type"]) for item in detected_events)
    noise = config["_current_noise"]
    degradations = []
    if float(noise.get("localization_std_px", 0.0)) > 0.0:
        degradations.append("localization_noise")
    if float(noise.get("blur_alpha", 0.0)) > 0.0:
        degradations.append("blur")
    if noise.get("frame_drop"):
        degradations.append("frame_drop")
    if noise.get("missing_points"):
        degradations.append("missing_points")
    if float(noise.get("scale_error", 0.0)) != 0.0:
        degradations.append("pixel_scale_mismatch")
    if noise.get("point_order_error_frame") is not None:
        degradations.append("point_order_error")
    status = "valid"
    if "point_order_error" in degradations:
        status = "censored"
    elif "pixel_scale_mismatch" in degradations:
        status = "biased"
    elif degradations:
        status = "degraded"
    return {
        "fixture": fixture,
        "scenario": metadata["scenario"],
        "seed": int(metadata["seed"]),
        "contract_checks": checks,
        "frame_count_expected": len(metadata.get("expected_frames", [])),
        "frame_count_observed": len({int(row["frame"]) for row in rows}),
        "valid_frame_count": len(valid_frames),
        "censored_frame_count": sum(1 for row in frame_rows if row["censor_reason"]),
        "frame_rows": frame_rows,
        "raw_event_types": dict(sorted(event_types.items())),
        "detected_event_types": dict(sorted(detected_types.items())),
        "degradations": degradations,
        "status": status,
        "recovered": recovered,
        "truth_identifiable": truth_identifiable,
        "relative_errors": errors,
        "unidentifiable": {
            "EA": "centerline time series without calibrated axial force/extension response cannot identify EA",
            "EI": "passive shape relaxation identifies EI/drag, not EI alone",
            "drag": "passive shape relaxation identifies EI/drag, not drag alone",
            "true_reaction": "centerline observations do not observe constrained endpoint reaction force",
            "contact_law": "these fixtures contain no contact-force observation and cannot identify a contact law",
        },
        "analysis_revision": ANALYSIS_REVISION,
    }


def _run_one(
    fixture: str,
    scenario: Mapping[str, Any],
    seed: int,
    config: Mapping[str, Any],
    output: Path,
) -> dict[str, Any]:
    run_id = f"{fixture}/{scenario['name']}/seed-{_safe_token(seed)}"
    run_dir = output / "runs" / run_id
    raw = generate_raw_run(fixture, scenario, seed, config)
    run_dir.mkdir(parents=True, exist_ok=True)
    raw_path = run_dir / "centerline.csv"
    raw_path.write_bytes(raw["csv_bytes"])
    _write_json(run_dir / "events.json", raw["events"])
    _write_json(run_dir / "metadata.json", raw["metadata"])
    # The analyser reads back the serialized CSV to test the actual raw-like
    # contract rather than analysing the in-memory generator objects.
    analysis_config = dict(config)
    analysis_config["_current_noise"] = dict(scenario.get("noise", {}))
    analysis = analyse_raw_run(raw_path, raw["metadata"], raw["events"], fixture, analysis_config)
    analysis_bytes = _canonical_json_bytes(analysis)
    result_hash = _sha256(analysis_bytes)
    manifest = {
        "manifest_schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "fixture": fixture,
        "scenario": scenario["name"],
        "seed": int(seed),
        "raw_synthetic_hash": raw["raw_synthetic_hash"],
        "analysis_result_hash": result_hash,
        "analysis_revision": ANALYSIS_REVISION,
        "git_revision": config.get("git_revision"),
        "truth_config": config["truth"],
        "noise_config": scenario.get("noise", {}),
        "contract_checks": analysis["contract_checks"],
        "status": analysis["status"],
        "identified_quantities": analysis["recovered"],
        "unidentifiable": analysis["unidentifiable"],
        "raw_event_types": analysis["raw_event_types"],
        "detected_event_types": analysis["detected_event_types"],
    }
    _write_json(run_dir / "analysis.json", analysis)
    _write_json(run_dir / "manifest.json", manifest)
    return {
        "run_id": run_id,
        "fixture": fixture,
        "scenario": scenario["name"],
        "seed": int(seed),
        "status": analysis["status"],
        "raw_synthetic_hash": raw["raw_synthetic_hash"],
        "analysis_result_hash": result_hash,
        "analysis_revision": ANALYSIS_REVISION,
        "valid_frame_count": analysis["valid_frame_count"],
        "censored_frame_count": analysis["censored_frame_count"],
        "frame_count_expected": analysis["frame_count_expected"],
        "frame_count_observed": analysis["frame_count_observed"],
        "degradations": analysis["degradations"],
        "detected_event_types": analysis["detected_event_types"],
        "recovered": analysis["recovered"],
        "truth_identifiable": analysis["truth_identifiable"],
        "relative_errors": analysis["relative_errors"],
        "unidentifiable": analysis["unidentifiable"],
        "manifest_path": str((run_dir / "manifest.json").relative_to(output)),
    }


def _summarize_distribution(rows: Sequence[Mapping[str, Any]], key: str) -> dict[str, Any]:
    values = [float(row["relative_errors"][key]) for row in rows if row["relative_errors"].get(key) is not None]
    if not values:
        return {"count": 0, "mean": None, "std": None, "min": None, "max": None}
    array = np.asarray(values, dtype=float)
    return {
        "count": int(len(array)),
        "mean": float(np.mean(array)),
        "std": float(np.std(array, ddof=1)) if len(array) > 1 else 0.0,
        "min": float(np.min(array)),
        "max": float(np.max(array)),
    }


def _acceptance_checks(rows: Sequence[Mapping[str, Any]], config: Mapping[str, Any], replay: Mapping[str, Any]) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    zero_rows = [row for row in rows if row["scenario"] == "clean_zero_noise"]
    tolerance = dict(config.get("acceptance", {}))
    zero_ok = True
    failures: list[str] = []
    for row in zero_rows:
        for key, limit in {
            "growth_rate": float(tolerance.get("growth_relative_error", 0.03)),
            "EI_over_drag": float(tolerance.get("EI_over_drag_relative_error", 0.05)),
            "diameter_proxy": float(tolerance.get("diameter_relative_error", 0.03)),
        }.items():
            error = row["relative_errors"].get(key)
            if error is not None and float(error) > limit:
                zero_ok = False
                failures.append(f"{row['run_id']}:{key}={error}>{limit}")
    checks.append({"name": "noise_zero_truth_recovery", "passed": zero_ok, "detail": failures or "identifiable fixture values within tolerance"})
    checks.append({"name": "same_seed_replay_hash_match", "passed": bool(replay.get("match")), "detail": replay})
    order_rows = [row for row in rows if "point_order_error" in row["degradations"]]
    order_ok = bool(order_rows) and all(row["status"] == "censored" and row["detected_event_types"].get("detected_point_order_error", 0) > 0 for row in order_rows)
    checks.append({"name": "point_order_error_not_silent", "passed": order_ok, "detail": {"runs": [row["run_id"] for row in order_rows]}})
    scale_rows = [row for row in rows if "pixel_scale_mismatch" in row["degradations"]]
    scale_ok = bool(scale_rows) and all(row["status"] == "biased" and row["detected_event_types"].get("detected_pixel_scale_mismatch", 0) > 0 for row in scale_rows)
    checks.append({"name": "scale_mismatch_not_silent", "passed": scale_ok, "detail": {"runs": [row["run_id"] for row in scale_rows]}})
    missing_rows = [row for row in rows if "missing_points" in row["degradations"]]
    missing_ok = bool(missing_rows) and all(row["censored_frame_count"] > 0 and row["detected_event_types"].get("detected_missing_points", 0) > 0 for row in missing_rows)
    checks.append({"name": "missing_points_are_censored", "passed": missing_ok, "detail": {"runs": [row["run_id"] for row in missing_rows]}})
    drop_rows = [row for row in rows if "frame_drop" in row["degradations"]]
    drop_ok = bool(drop_rows) and all(row["frame_count_observed"] < row["frame_count_expected"] and row["detected_event_types"].get("detected_frame_drop", 0) > 0 for row in drop_rows)
    checks.append({"name": "frame_drop_is_detected", "passed": drop_ok, "detail": {"runs": [row["run_id"] for row in drop_rows]}})
    noise_rows = [row for row in rows if row["scenario"] == "localization_noise"]
    clean_growth = [row for row in zero_rows if row["fixture"] == "growth" and row["relative_errors"].get("growth_rate") is not None]
    noisy_growth = [row for row in noise_rows if row["fixture"] == "growth" and row["relative_errors"].get("growth_rate") is not None]
    noise_distribution_ok = bool(noisy_growth) and _summarize_distribution(noisy_growth, "growth_rate")["mean"] >= _summarize_distribution(clean_growth, "growth_rate")["mean"]
    checks.append({"name": "observation_noise_distribution_reported", "passed": noise_distribution_ok, "detail": {"clean": _summarize_distribution(clean_growth, "growth_rate"), "noise": _summarize_distribution(noisy_growth, "growth_rate")}})
    return checks


def run_suite(config: Mapping[str, Any], output: Path) -> dict[str, Any]:
    _validate_config(config)
    output.mkdir(parents=True, exist_ok=True)
    fixture_names = [str(item["name"]) for item in config["fixtures"]]
    scenario_list = list(config["scenarios"])
    seeds = [int(value) for value in config["seeds"]]
    rows: list[dict[str, Any]] = []
    for scenario in scenario_list:
        selected = [str(value) for value in scenario.get("fixtures", fixture_names)]
        for fixture in selected:
            if fixture not in fixture_names:
                raise SyntheticRecoveryError(f"scenario references unknown fixture: {fixture}")
            for seed in seeds:
                rows.append(_run_one(fixture, scenario, seed, config, output))
    # Replay is deliberately a separate check: same seed must match exactly;
    # different seeds are retained as a distribution and are not expected to
    # produce identical raw or analysis hashes.
    replay_scenario = next(item for item in scenario_list if item["name"] == "localization_noise")
    replay_a = _run_one("sinusoidal_buckling", replay_scenario, seeds[0], config, output / "replay-a")
    replay_b = _run_one("sinusoidal_buckling", replay_scenario, seeds[0], config, output / "replay-b")
    replay = {
        "same_seed": int(seeds[0]),
        "match": replay_a["raw_synthetic_hash"] == replay_b["raw_synthetic_hash"] and replay_a["analysis_result_hash"] == replay_b["analysis_result_hash"],
        "raw_synthetic_hash_a": replay_a["raw_synthetic_hash"],
        "raw_synthetic_hash_b": replay_b["raw_synthetic_hash"],
        "analysis_result_hash_a": replay_a["analysis_result_hash"],
        "analysis_result_hash_b": replay_b["analysis_result_hash"],
        "different_seed_hashes": sorted({row["raw_synthetic_hash"] for row in rows if row["scenario"] == "localization_noise" and row["fixture"] == "sinusoidal_buckling"}),
    }
    acceptance = _acceptance_checks(rows, config, replay)
    if not all(bool(item["passed"]) for item in acceptance):
        failed = [item["name"] for item in acceptance if not item["passed"]]
        raise SyntheticRecoveryError(f"synthetic recovery acceptance failed: {', '.join(failed)}")
    distribution: dict[str, Any] = {}
    for scenario in sorted({str(row["scenario"]) for row in rows}):
        for fixture in sorted({str(row["fixture"]) for row in rows if row["scenario"] == scenario}):
            group = [row for row in rows if row["scenario"] == scenario and row["fixture"] == fixture]
            distribution[f"{fixture}/{scenario}"] = {
                "n_runs": len(group),
                "seeds": [row["seed"] for row in group],
                "status_counts": dict(sorted(Counter(str(row["status"]) for row in group).items())),
                "valid_frame_count": _summarize_distribution(
                    [{"relative_errors": {"value": row["valid_frame_count"]}} for row in group], "value"
                ),
                "growth_rate_error": _summarize_distribution(group, "growth_rate"),
                "EI_over_drag_error": _summarize_distribution(group, "EI_over_drag"),
                "diameter_error": _summarize_distribution(group, "diameter_proxy"),
            }
    compact = {
        "schema_version": SCHEMA_VERSION,
        "analysis_revision": ANALYSIS_REVISION,
        "git_revision": config.get("git_revision"),
        "truth_config": config["truth"],
        "noise_scenarios": [item for item in scenario_list],
        "fixture_names": fixture_names,
        "seed_policy": {
            "same_seed_replay": replay,
            "multiple_seed_results_are_distribution_not_replay": True,
            "seeds": seeds,
        },
        "run_count": len(rows),
        "rows": rows,
        "distribution": distribution,
        "acceptance_checks": acceptance,
        "claims_boundary": {
            "can_say": [
                "the raw-like contract can represent centerline, time/frame, width, quality, events, metadata, and explicit missingness",
                "arc-length reparameterization, curvature, velocity, length, and event QC are deterministic on these bounded fixtures",
                "g and EI/drag can be recovered only for fixtures that expose those observables under the declared calibration",
                "observation noise, frame drops, missing points, blur, and scale/order corruption are visible as degraded, biased, or censored outcomes",
            ],
            "cannot_say": [
                "synthetic recovery is not an experimental result or proof of model validity",
                "EA cannot be recovered from centerline-only passive observations without calibrated axial response/force data",
                "EI and drag are not separately identifiable from passive shape relaxation; only EI/drag is estimated",
                "true endpoint reactions and contact laws cannot be recovered from these inputs",
            ],
        },
    }
    _write_json(output / "compact_summary.json", compact)
    compact_rows = []
    for row in rows:
        compact_rows.append({
            key: row[key]
            for key in (
                "run_id", "fixture", "scenario", "seed", "status", "raw_synthetic_hash",
                "analysis_result_hash", "valid_frame_count", "censored_frame_count",
                "frame_count_expected", "frame_count_observed", "degradations",
                "detected_event_types", "relative_errors", "manifest_path",
            )
        })
    _write_json(output / "compact_runs.json", compact_rows)
    return compact


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if not isinstance(config, Mapping):
        raise SystemExit("config root must be an object")
    # The revision is captured in the manifest, but callers can override it
    # when replaying a released analysis bundle.
    config = dict(config)
    config["git_revision"] = config.get("git_revision") or _git_revision()
    run_suite(config, args.output)
    print(json.dumps({"output": str(args.output), "schema_version": SCHEMA_VERSION}, sort_keys=True))
    return 0


def _git_revision() -> str | None:
    import subprocess

    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True, timeout=2.0
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return completed.stdout.strip() or None


if __name__ == "__main__":  # pragma: no cover - exercised by the documented runner command
    raise SystemExit(main())
