"""Registration-independent morphology comparison for growing filaments.

The comparison in this module intentionally does not use a pixel/model-unit
registration or a video/model clock registration.  Observation centre-lines
remain in pixel coordinates and model centre-lines remain in model
coordinates; both are reduced by their own current contour length and aligned
by the data-derived growth progress ``q``.

This is a morphology-only comparison.  It preserves observation quality,
lineage, and censoring, and never turns a shape difference into a
``model_inadequacy`` assessment.
"""

from __future__ import annotations

import csv
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .video_comparison import (
    ALLOWED_LINEAGE_STATUSES,
    _choose_filament,
    _file_record,
    _read_csv_rows,
    canonical_json,
    load_model_output,
    sha256_file,
    sha256_text,
    validate_centerline_rows,
)
from .reproducibility import detect_git_revision


SCHEMA_VERSION = "continuum-filament-scale-free-shape-0.1"
MODE_COUNT = 6


@dataclass(frozen=True)
class ScaleFreeConfig:
    """Numerical and eligibility settings for the morphology comparison."""

    sample_points: int = 80
    min_points: int = 2
    min_length: float = 1.0e-9
    min_quality: float = 0.20
    max_progress_error: float = 0.05
    monotonic_tolerance: float = 1.0e-8
    min_growth_span_relative: float = 1.0e-8

    def __post_init__(self) -> None:
        if self.sample_points < 8:
            raise ValueError("sample_points must be at least 8")
        if self.min_points < 2:
            raise ValueError("min_points must be at least 2")
        if not math.isfinite(self.min_length) or self.min_length <= 0.0:
            raise ValueError("min_length must be positive and finite")
        if not 0.0 <= self.min_quality <= 1.0:
            raise ValueError("min_quality must be in [0, 1]")
        if self.max_progress_error < 0.0 or not math.isfinite(self.max_progress_error):
            raise ValueError("max_progress_error must be finite and non-negative")
        if self.monotonic_tolerance < 0.0 or not math.isfinite(self.monotonic_tolerance):
            raise ValueError("monotonic_tolerance must be finite and non-negative")
        if self.min_growth_span_relative <= 0.0 or not math.isfinite(self.min_growth_span_relative):
            raise ValueError("min_growth_span_relative must be positive and finite")

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any] | None) -> "ScaleFreeConfig":
        return cls(**dict(value or {}))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class _Frame:
    index: int
    time_s: float | None
    points: np.ndarray | None
    length: float | None
    quality: float | None = None
    quality_flags: str = "ok"
    lineage_status: str = "observed"
    censor: bool = False
    q: float | None = None
    source: str = ""


@dataclass(frozen=True)
class _Progress:
    status: str
    initial_length: float | None
    final_length: float | None
    growth_span: float | None
    valid_length_count: int
    nonmonotonic_decrease_count: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        for row in rows:
            clean: dict[str, Any] = {}
            for field in fields:
                value = row.get(field, "")
                if isinstance(value, (dict, list, tuple)):
                    value = canonical_json(value)
                elif isinstance(value, np.generic):
                    value = value.item()
                clean[field] = value
            writer.writerow(clean)


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(canonical_json(value) + "\n", encoding="utf-8")


def _float_or_none(value: Any) -> float | None:
    if isinstance(value, bool) or value is None or value == "":
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _bool_value(value: Any) -> bool:
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes"}:
            return True
        if normalized in {"0", "false", "no"}:
            return False
        raise ValueError("invalid boolean value")
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and value in {0, 1}:
        return bool(value)
    raise ValueError("invalid boolean value")


def _length(points: np.ndarray) -> float:
    if len(points) < 2:
        return 0.0
    return float(np.sum(np.linalg.norm(np.diff(points, axis=0), axis=1)))


def _finite_points(points: np.ndarray | None) -> bool:
    return bool(points is not None and points.ndim == 2 and points.shape[1] == 2 and len(points) >= 2 and np.isfinite(points).all())


def _resample(points: np.ndarray, count: int) -> np.ndarray:
    values = np.asarray(points, dtype=float)
    if len(values) == 0:
        return np.zeros((count, 2), dtype=float)
    if len(values) == 1:
        return np.repeat(values, count, axis=0)
    distances = np.concatenate(([0.0], np.cumsum(np.linalg.norm(np.diff(values, axis=0), axis=1))))
    keep = np.concatenate(([True], np.diff(distances) > 1.0e-15))
    distances = distances[keep]
    values = values[keep]
    if len(values) == 1 or distances[-1] <= 1.0e-15:
        return np.repeat(values[:1], count, axis=0)
    target = np.linspace(0.0, float(distances[-1]), count)
    return np.column_stack([np.interp(target, distances, values[:, axis]) for axis in range(2)])


def _arc_length_radius_of_gyration(points: np.ndarray, total_length: float) -> float:
    starts = points[:-1]
    ends = points[1:]
    lengths = np.linalg.norm(ends - starts, axis=1)
    if total_length <= 0.0:
        return float("nan")
    center = np.sum(lengths[:, None] * 0.5 * (starts + ends), axis=0) / total_length
    second = np.sum(
        lengths
        * (
            np.sum(starts * starts, axis=1)
            + np.sum(starts * ends, axis=1)
            + np.sum(ends * ends, axis=1)
        )
        / 3.0
    ) / total_length
    return float(math.sqrt(max(float(second - np.dot(center, center)), 0.0)))


def _peak_deflection(points: np.ndarray, total_length: float) -> float:
    chord = points[-1] - points[0]
    chord_length = float(np.linalg.norm(chord))
    if chord_length <= 1.0e-15:
        distances = np.linalg.norm(points - points[0], axis=1)
    else:
        relative = points - points[0]
        distances = np.abs(relative[:, 0] * chord[1] - relative[:, 1] * chord[0]) / chord_length
    return float(np.max(distances) / total_length)


def _curvature_rms_times_length(points: np.ndarray, total_length: float) -> float:
    if len(points) < 3:
        return 0.0
    sampled = _resample(points, 256)
    segments = np.diff(sampled, axis=0)
    lengths = np.linalg.norm(segments, axis=1)
    values: list[float] = []
    for index in range(1, len(sampled) - 1):
        left = segments[index - 1]
        right = segments[index]
        left_norm = float(lengths[index - 1])
        right_norm = float(lengths[index])
        if left_norm <= 1.0e-15 or right_norm <= 1.0e-15:
            continue
        cosine = float(np.clip(np.dot(left, right) / (left_norm * right_norm), -1.0, 1.0))
        angle = math.acos(cosine)
        values.append(angle / max(0.5 * (left_norm + right_norm), 1.0e-15))
    return float(math.sqrt(np.mean(np.square(values))) * total_length) if values else 0.0


def _mode_fractions(points: np.ndarray, total_length: float, sample_points: int) -> np.ndarray:
    if total_length <= 0.0:
        return np.zeros(MODE_COUNT, dtype=float)
    chord = points[-1] - points[0]
    chord_length = float(np.linalg.norm(chord))
    if chord_length <= 1.0e-15:
        return np.zeros(MODE_COUNT, dtype=float)
    tangent = chord / chord_length
    normal = np.asarray([-tangent[1], tangent[0]])
    sampled = _resample(points, max(8, int(sample_points)))
    u = np.linspace(0.0, 1.0, len(sampled))
    transverse = (sampled - points[0]) @ normal / total_length
    integrate = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
    coefficients = np.asarray(
        [2.0 * integrate(transverse * np.sin(mode * np.pi * u), u) for mode in range(1, MODE_COUNT + 1)],
        dtype=float,
    )
    power = coefficients * coefficients
    total = float(np.sum(power))
    return power / total if total > 1.0e-30 else np.zeros(MODE_COUNT, dtype=float)


def shape_observables(points: np.ndarray, *, sample_points: int = 80, min_length: float = 1.0e-9) -> dict[str, Any] | None:
    """Return dimensionless polyline observables or ``None`` for invalid input."""

    values = np.asarray(points, dtype=float)
    if not _finite_points(values):
        return None
    total_length = _length(values)
    if not math.isfinite(total_length) or total_length <= min_length:
        return None
    endpoint = float(np.linalg.norm(values[-1] - values[0]))
    fractions = _mode_fractions(values, total_length, sample_points)
    return {
        "normalized_endpoint_distance": endpoint / total_length,
        "normalized_radius_of_gyration": _arc_length_radius_of_gyration(values, total_length) / total_length,
        "normalized_peak_deflection": _peak_deflection(values, total_length),
        "curvature_rms_times_length": _curvature_rms_times_length(values, total_length),
        "mode_fractions": [float(value) for value in fractions],
        "length": total_length,
        "point_count": int(len(values)),
    }


def _procrustes_distance(first: np.ndarray, second: np.ndarray) -> float:
    """Return rotation/translation-invariant RMS distance for equal arc samples."""

    a = np.asarray(first, dtype=float)
    b = np.asarray(second, dtype=float)
    if len(a) != len(b) or len(a) < 2:
        return float("nan")
    a = a - np.mean(a, axis=0)
    b = b - np.mean(b, axis=0)
    # The curves have already been normalized by their own contour length.
    # Do not normalize their centred RMS again: that would erase legitimate
    # differences in deflection amplitude.
    a_complex = a[:, 0] + 1j * a[:, 1]
    b_complex = b[:, 0] + 1j * b[:, 1]
    cross = np.vdot(b_complex, a_complex)
    if abs(cross) <= 1.0e-15:
        rotated = b
    else:
        phase = cross / abs(cross)
        rotated_complex = b_complex * phase
        rotated = np.column_stack((rotated_complex.real, rotated_complex.imag))
    return float(np.sqrt(np.mean(np.sum((a - rotated) ** 2, axis=1))))


def normalized_shape_distance(first: np.ndarray, second: np.ndarray, *, sample_points: int = 80, min_length: float = 1.0e-9) -> tuple[float | None, str | None]:
    """Compare two shapes after independent length normalization.

    Translation, rotation, and endpoint orientation are nuisance degrees of
    freedom.  Reflection is not removed because it changes a handed shape.
    """

    if not _finite_points(first) or not _finite_points(second):
        return None, None
    first_length = _length(first)
    second_length = _length(second)
    if first_length <= min_length or second_length <= min_length:
        return None, None
    first_sampled = _resample(first, sample_points) / first_length
    second_sampled = _resample(second, sample_points) / second_length
    forward = _procrustes_distance(first_sampled, second_sampled)
    reverse = _procrustes_distance(first_sampled, second_sampled[::-1])
    if not math.isfinite(forward) and not math.isfinite(reverse):
        return None, None
    if forward <= reverse:
        return forward, "forward"
    return reverse, "reverse"


def _progress_eligible(frame: _Frame, config: ScaleFreeConfig) -> bool:
    if (
        not _finite_points(frame.points)
        or frame.length is None
        or not math.isfinite(frame.length)
        or frame.length <= config.min_length
        or frame.censor
    ):
        return False
    if frame.source == "observation":
        return (
            frame.quality is not None
            and config.min_quality <= frame.quality <= 1.0
            and frame.lineage_status in ALLOWED_LINEAGE_STATUSES
        )
    return True


def _progress(frames: Sequence[_Frame], config: ScaleFreeConfig) -> _Progress:
    lengths = [float(frame.length) for frame in frames if _progress_eligible(frame, config)]
    if len(lengths) < 2:
        return _Progress("insufficient_length_observations", None, None, None, len(lengths), 0)
    initial = float(lengths[0])
    final = float(lengths[-1])
    span = final - initial
    relative_floor = config.min_growth_span_relative * max(abs(initial), config.min_length)
    decreases = sum(1 for before, after in zip(lengths, lengths[1:]) if after < before - config.monotonic_tolerance * max(abs(initial), 1.0))
    if abs(span) <= relative_floor:
        return _Progress("zero_growth_span", initial, final, span, len(lengths), decreases)
    if decreases:
        return _Progress("non_monotonic_lengths", initial, final, span, len(lengths), decreases)
    return _Progress("ok", initial, final, span, len(lengths), decreases)


def _assign_progress(frames: Sequence[_Frame], progress: _Progress, config: ScaleFreeConfig) -> None:
    if progress.status != "ok" or progress.growth_span is None or abs(progress.growth_span) <= config.min_length:
        return
    for frame in frames:
        if _progress_eligible(frame, config):
            frame.q = float((frame.length - progress.initial_length) / progress.growth_span)  # type: ignore[operator]


def _validate_frame_keys(rows: Sequence[Mapping[str, Any]], artifact: str) -> dict[str, Any]:
    errors: list[str] = []
    seen_frame_keys: set[tuple[int, str]] = set()
    seen_centerline_keys: set[tuple[int, str, int]] = set()
    frame_times: dict[tuple[int, str], float] = {}
    time_frames: dict[tuple[float, str], int] = {}
    for line_number, row in enumerate(rows, start=2):
        try:
            raw_frame = row["frame"]
            if raw_frame in (None, ""):
                raise ValueError("empty frame")
            frame = int(raw_frame)
            if frame < 0:
                raise ValueError("negative frame")
            filament = str(row.get("filament_id", "")).strip()
            if not filament:
                raise ValueError("empty filament_id")
            raw_time = row["time"]
            if raw_time in (None, ""):
                raise ValueError("empty time")
            time_s = float(raw_time)
            if not math.isfinite(time_s):
                raise ValueError("non-finite time")
        except (KeyError, TypeError, ValueError):
            errors.append(f"{artifact} line {line_number}: invalid frame/time key")
            continue
        frame_key = (frame, filament)
        if artifact == "centerline":
            try:
                point_id = int(row["point_id"])
            except (KeyError, TypeError, ValueError):
                errors.append(f"{artifact} line {line_number}: invalid point_id")
            else:
                point_key = (frame, filament, point_id)
                if point_key in seen_centerline_keys:
                    errors.append(f"{artifact} line {line_number}: duplicate frame/filament/point_id key")
                seen_centerline_keys.add(point_key)
        else:
            if frame_key in seen_frame_keys:
                errors.append(f"{artifact} line {line_number}: duplicate frame/filament key")
            seen_frame_keys.add(frame_key)
        previous_time = frame_times.get(frame_key)
        if previous_time is not None and abs(previous_time - time_s) > 1.0e-12:
            errors.append(f"{artifact} line {line_number}: frame maps to multiple times")
        frame_times[frame_key] = time_s
        time_key = (time_s, filament)
        previous_frame = time_frames.get(time_key)
        if previous_frame is not None and previous_frame != frame:
            errors.append(f"{artifact} line {line_number}: time maps to multiple frames")
        time_frames[time_key] = frame
        if "censor" not in row or row["censor"] in (None, ""):
            errors.append(f"{artifact} line {line_number}: missing censor value")
        else:
            try:
                _bool_value(row["censor"])
            except ValueError:
                errors.append(f"{artifact} line {line_number}: invalid censor value")
        if artifact in {"summary", "centerline"}:
            try:
                quality = float(row["quality"])
                if not math.isfinite(quality) or not 0.0 <= quality <= 1.0:
                    raise ValueError("quality outside [0,1]")
            except (KeyError, TypeError, ValueError):
                errors.append(f"{artifact} line {line_number}: invalid quality value")
    return {"valid": not errors, "errors": errors, "row_count": len(rows)}


def _index_frame_rows(rows: Sequence[Mapping[str, Any]]) -> dict[tuple[int, str], Mapping[str, Any]]:
    indexed: dict[tuple[int, str], Mapping[str, Any]] = {}
    for row in rows:
        try:
            key = (int(row["frame"]), str(row.get("filament_id", "")))
        except (KeyError, TypeError, ValueError):
            continue
        indexed[key] = row
    return indexed


def _read_observation_csv(path: Path, artifact: str) -> tuple[list[dict[str, str]], dict[str, Any]]:
    if not path.is_file():
        return [], {"valid": False, "errors": [f"missing_{artifact}_artifact"], "present": False, "row_count": 0}
    try:
        rows = _read_csv_rows(path)
    except (OSError, UnicodeError, csv.Error) as exc:
        return [], {"valid": False, "errors": [f"{artifact}_read_failed:{type(exc).__name__}"], "present": True, "row_count": 0}
    return rows, {"valid": True, "errors": [], "present": True, "row_count": len(rows)}


def _validate_manifest_artifacts(observation_dir: Path, manifest: Mapping[str, Any]) -> dict[str, Any]:
    artifacts = manifest.get("artifacts")
    errors: list[str] = []
    allowed = {
        "centerline", "centerline.csv", "observation_summary", "observation_summary.csv",
        "lineage", "lineage.csv", "metadata", "metadata.json", "events", "events.csv",
    }
    if not isinstance(artifacts, Mapping):
        return {"valid": False, "errors": ["artifacts_not_mapping"], "warnings": []}
    canonical = {
        "centerline": ("centerline", "centerline.csv"),
        "summary": ("observation_summary", "observation_summary.csv"),
        "lineage": ("lineage", "lineage.csv"),
    }
    for logical_name, (artifact_key, canonical_name) in canonical.items():
        record = artifacts.get(artifact_key)
        if record is None:
            record = artifacts.get(canonical_name)
        if not isinstance(record, Mapping):
            errors.append(f"artifact {logical_name}: record_missing")
            continue
        if record.get("path") != canonical_name:
            errors.append(f"artifact {logical_name}: noncanonical_path")
        if not isinstance(record.get("sha256"), str) or not record.get("sha256"):
            errors.append(f"artifact {logical_name}: hash_missing")
    for artifact_name, record in artifacts.items():
        if artifact_name not in allowed:
            errors.append(f"artifact {artifact_name}: unknown_artifact")
            continue
        if not isinstance(record, Mapping):
            errors.append(f"artifact {artifact_name}: record_not_mapping")
            continue
        artifact_path = record.get("path")
        if not isinstance(artifact_path, str) or not artifact_path:
            errors.append(f"artifact {artifact_name}: path_missing")
            continue
        path_value = Path(artifact_path)
        if path_value.is_absolute() or ".." in path_value.parts:
            errors.append(f"artifact {artifact_name}: path_outside_observation")
            continue
        canonical_path_by_name = {
            "centerline": "centerline.csv", "centerline.csv": "centerline.csv",
            "observation_summary": "observation_summary.csv", "observation_summary.csv": "observation_summary.csv",
            "lineage": "lineage.csv", "lineage.csv": "lineage.csv",
            "metadata": "metadata.json", "metadata.json": "metadata.json",
            "events": "events.csv", "events.csv": "events.csv",
        }
        expected_path = canonical_path_by_name[artifact_name]
        if artifact_name in canonical_path_by_name and artifact_path != expected_path:
            errors.append(f"artifact {artifact_name}: noncanonical_path")
            continue
        path = observation_dir / path_value
        if path.is_symlink():
            errors.append(f"artifact {artifact_name}: symlink_not_allowed")
            continue
        try:
            path.resolve().relative_to(observation_dir.resolve())
        except ValueError:
            errors.append(f"artifact {artifact_name}: path_outside_observation")
            continue
        if not path.is_file():
            errors.append(f"artifact {artifact_name}: file_missing")
            continue
        if record.get("bytes") is not None and record.get("bytes") != path.stat().st_size:
            errors.append(f"artifact {artifact_name}: byte_count_mismatch")
        expected_hash = record.get("sha256")
        if expected_hash is not None:
            try:
                actual_hash = sha256_file(path)
            except OSError:
                errors.append(f"artifact {artifact_name}: hash_read_failed")
            else:
                if actual_hash != expected_hash:
                    errors.append(f"artifact {artifact_name}: sha256_mismatch")
    return {"valid": not errors, "errors": errors, "warnings": []}


def _validate_npz_source(path: Path) -> dict[str, Any]:
    errors: list[str] = []
    frame_count = 0
    try:
        with np.load(path, allow_pickle=False) as archive:
            positions = np.asarray(archive["positions"])
            times = np.asarray(archive["times"])
            offsets = np.asarray(archive["position_offsets"])
        if positions.ndim != 2 or positions.shape[1] != 2:
            errors.append("positions_must_be_n_by_two")
        if times.ndim != 1:
            errors.append("times_must_be_one_dimensional")
        frame_count = int(len(times)) if times.ndim == 1 else 0
        if offsets.ndim != 1 or len(offsets) != frame_count + 1:
            errors.append("position_offsets_length_mismatch")
        else:
            if not np.isfinite(offsets).all():
                errors.append("position_offsets_non_finite")
            if not np.equal(offsets, np.floor(offsets)).all():
                errors.append("position_offsets_non_integer")
            if len(offsets) == 0 or int(offsets[0]) != 0:
                errors.append("position_offsets_must_start_at_zero")
            if np.any(np.diff(offsets) < 0):
                errors.append("position_offsets_must_be_monotonic")
            if int(offsets[-1]) != len(positions):
                errors.append("position_offsets_end_mismatch")
            if np.any(offsets < 0) or np.any(offsets > len(positions)):
                errors.append("position_offsets_out_of_bounds")
    except (OSError, EOFError, KeyError, TypeError, ValueError):
        errors.append("npz_trajectory_arrays_unreadable")
    return {"valid": not errors, "errors": errors, "warnings": [], "source_frame_count": frame_count}


def _validate_lineage_rows(
    rows: Sequence[Mapping[str, Any]],
    summary_rows: Sequence[Mapping[str, Any]],
    centerline_rows: Sequence[Mapping[str, Any]],
    artifact_present: bool,
) -> dict[str, Any]:
    errors: list[str] = []
    required = {"frame", "filament_id", "status", "censor"}
    if not artifact_present:
        errors.append("missing_lineage_artifact")
    if artifact_present and not rows:
        errors.append("empty_lineage_artifact")
    seen: set[tuple[int, str]] = set()
    for line_number, row in enumerate(rows, start=2):
        missing = sorted(required.difference(row))
        if missing:
            errors.append(f"line {line_number}: missing lineage columns {','.join(missing)}")
            continue
        try:
            key = (int(row["frame"]), str(row["filament_id"]))
            if not str(row["status"]):
                raise ValueError("empty status")
            _bool_value(row["censor"])
        except (TypeError, ValueError):
            errors.append(f"line {line_number}: invalid lineage value")
            continue
        if key in seen:
            errors.append(f"line {line_number}: duplicate lineage key")
        seen.add(key)
    expected_keys = set(_index_frame_rows(summary_rows)) | set(_index_frame_rows(centerline_rows))
    lineage_keys = set(_index_frame_rows(rows))
    for frame, filament in sorted(expected_keys - lineage_keys):
        errors.append(f"missing lineage for frame={frame},filament={filament}")
    return {
        "valid": not errors,
        "errors": errors,
        "warnings": [],
        "artifact_present": artifact_present,
        "row_count": len(rows),
    }


def _validate_observation_consistency(
    summary_rows: Sequence[Mapping[str, Any]],
    centerline_rows: Sequence[Mapping[str, Any]],
    lineage_rows: Sequence[Mapping[str, Any]],
    processed_frames: Sequence[int],
) -> dict[str, Any]:
    errors: list[str] = []
    summary_by_key = _index_frame_rows(summary_rows)
    lineage_by_key = _index_frame_rows(lineage_rows)
    centerline_by_key: dict[tuple[int, str], list[Mapping[str, Any]]] = {}
    for row in centerline_rows:
        try:
            key = (int(row["frame"]), str(row.get("filament_id", "")))
        except (KeyError, TypeError, ValueError):
            continue
        centerline_by_key.setdefault(key, []).append(row)
    summary_keys = set(summary_by_key)
    centerline_keys = set(centerline_by_key)
    for key in sorted(centerline_keys - summary_keys):
        errors.append(f"centerline key missing from summary: frame={key[0]},filament={key[1]}")
    allowed_lineage_keys = summary_keys | centerline_keys
    known_filaments = {filament for _, filament in allowed_lineage_keys}
    processed_set = set(processed_frames)
    for frame, filament in sorted(set(lineage_by_key) - allowed_lineage_keys):
        if frame not in processed_set or (known_filaments and filament not in known_filaments and filament != "unknown"):
            errors.append(f"phantom lineage key: frame={frame},filament={filament}")
    for key in sorted(summary_keys):
        summary = summary_by_key[key]
        lineage = lineage_by_key.get(key)
        try:
            summary_time = float(summary["time"])
            summary_quality = float(summary["quality"])
            summary_censor = _bool_value(summary["censor"])
            summary_length = float(summary["length_px"])
            if not math.isfinite(summary_length) or summary_length <= 0.0:
                raise ValueError("invalid summary length")
        except (KeyError, TypeError, ValueError):
            errors.append(f"invalid summary metadata for frame={key[0]},filament={key[1]}")
            continue
        if lineage is not None:
            try:
                if abs(float(lineage["time"]) - summary_time) > 5.0e-9:
                    errors.append(f"lineage time mismatch for frame={key[0]},filament={key[1]}")
                if _bool_value(lineage["censor"]) != summary_censor:
                    errors.append(f"lineage censor mismatch for frame={key[0]},filament={key[1]}")
            except (KeyError, TypeError, ValueError):
                errors.append(f"invalid lineage metadata for frame={key[0]},filament={key[1]}")
        try:
            exported_value = str(summary["centerline_exported"]).strip()
            if exported_value not in {"0", "1"}:
                raise ValueError("centerline_exported must be 0 or 1")
            exported = int(exported_value)
            expected_points = int(summary["n_points"])
            if expected_points < 0:
                raise ValueError("n_points must be non-negative")
            actual_points = len(centerline_by_key.get(key, []))
            if exported == 1 and (key not in centerline_by_key or actual_points != expected_points):
                errors.append(f"centerline count mismatch for frame={key[0]},filament={key[1]}")
            if exported == 0 and key in centerline_by_key:
                errors.append(f"unexpected centerline for frame={key[0]},filament={key[1]}")
        except (KeyError, TypeError, ValueError):
            errors.append(f"invalid centerline summary metadata for frame={key[0]},filament={key[1]}")
        if key in centerline_by_key:
            point_rows: list[Mapping[str, Any]] = []
            for row in centerline_by_key[key]:
                try:
                    int(row["point_id"])
                except (KeyError, TypeError, ValueError):
                    errors.append(f"invalid point_id for frame={key[0]},filament={key[1]}")
                else:
                    point_rows.append(row)
            point_rows.sort(key=lambda row: int(row["point_id"]))
            try:
                points = np.asarray([[float(row["x"]), float(row["y"])] for row in point_rows], dtype=float)
                centerline_length = _length(points)
                tolerance = max(1.0e-6, 1.0e-6 * max(summary_length, centerline_length))
                if not math.isfinite(centerline_length) or abs(centerline_length - summary_length) > tolerance:
                    errors.append(f"length mismatch for frame={key[0]},filament={key[1]}")
            except (KeyError, TypeError, ValueError):
                errors.append(f"invalid centerline geometry for frame={key[0]},filament={key[1]}")
            for row in centerline_by_key[key]:
                try:
                    if abs(float(row["time"]) - summary_time) > 5.0e-9:
                        errors.append(f"time mismatch for frame={key[0]},filament={key[1]}")
                    if abs(float(row["quality"]) - summary_quality) > 5.0e-9:
                        errors.append(f"quality mismatch for frame={key[0]},filament={key[1]}")
                    if _bool_value(row["censor"]) != summary_censor:
                        errors.append(f"censor mismatch for frame={key[0]},filament={key[1]}")
                except (KeyError, TypeError, ValueError):
                    errors.append(f"invalid centerline metadata for frame={key[0]},filament={key[1]}")
    return {"valid": not errors, "errors": errors, "warnings": []}


def _observation_frames(observation_dir: Path, filament_id: str | None) -> tuple[list[_Frame], str | None, dict[str, Any]]:
    summary_rows, summary_artifact = _read_observation_csv(observation_dir / "observation_summary.csv", "summary")
    centerline_rows, centerline_artifact = _read_observation_csv(observation_dir / "centerline.csv", "centerline")
    lineage_path = observation_dir / "lineage.csv"
    lineage_artifact_present = lineage_path.is_file()
    lineage_rows, lineage_artifact = _read_observation_csv(lineage_path, "lineage")
    manifest_path = observation_dir / "manifest.json"
    manifest_load_errors: list[str] = []
    if manifest_path.exists():
        try:
            loaded_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if not isinstance(loaded_manifest, Mapping):
                raise ValueError("manifest must be an object")
            manifest = loaded_manifest
        except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
            manifest = {}
            manifest_load_errors.append(f"invalid_manifest_json:{type(exc).__name__}")
    else:
        manifest = {}
    frame_key_validation = {
        artifact: _validate_frame_keys(rows, artifact)
        for artifact, rows in (("centerline", centerline_rows), ("summary", summary_rows), ("lineage", lineage_rows))
    }
    frame_keys_valid = all(item["valid"] for item in frame_key_validation.values())
    lineage_validation = _validate_lineage_rows(lineage_rows, summary_rows, centerline_rows, lineage_artifact_present)
    manifest_structure_errors: list[str] = []
    segmentation_config = manifest.get("segmentation_config", {})
    if not isinstance(segmentation_config, Mapping):
        manifest_structure_errors.append("segmentation_config_not_mapping")
        segmentation_config = {}
    run_section = manifest.get("run", {})
    if not isinstance(run_section, Mapping):
        manifest_structure_errors.append("run_not_mapping")
        run_section = {}
    video_section = manifest.get("video", {})
    if not isinstance(video_section, Mapping):
        manifest_structure_errors.append("video_not_mapping")
        video_section = {}
    try:
        max_jump_px = float(segmentation_config.get("max_jump_px", 80.0))
        if not math.isfinite(max_jump_px) or max_jump_px <= 0.0:
            raise ValueError("max_jump_px must be positive and finite")
    except (TypeError, ValueError):
        manifest_structure_errors.append("invalid_max_jump_px")
        max_jump_px = 80.0
    centerline_validation = validate_centerline_rows(centerline_rows, max_jump_px=max_jump_px)
    manifest_validation = manifest.get("validation")
    if not isinstance(manifest_validation, Mapping):
        manifest_validation = {"valid": False, "errors": ["missing_manifest_validation"]}
    elif (
        not isinstance(manifest_validation.get("errors"), list)
        or not isinstance(manifest_validation.get("warnings"), list)
        or manifest_validation.get("errors")
    ):
        manifest_structure_errors.append("invalid_manifest_validation_shape")
        manifest_validation = {"valid": False, "errors": ["invalid_manifest_validation_shape"]}
    manifest_artifact_validation = _validate_manifest_artifacts(observation_dir, manifest)
    if not manifest_artifact_validation.get("valid", False):
        manifest_structure_errors.extend(manifest_artifact_validation.get("errors", []))
    manifest_input = manifest.get("input", {})
    if not isinstance(manifest_input, Mapping):
        manifest_structure_errors.append("input_not_mapping")
    manifest_errors = manifest_load_errors + manifest_structure_errors
    if manifest_errors:
        manifest_validation = {
            "valid": False,
            "errors": manifest_errors + list(manifest_validation.get("errors", [])),
        }
    manifest_validation_valid = manifest_validation.get("valid") is True
    selected = filament_id or _choose_filament(summary_rows) or _choose_filament(lineage_rows)
    summary_by_key = _index_frame_rows(summary_rows)
    lineage_by_key = _index_frame_rows(lineage_rows)
    grouped: dict[tuple[int, str], list[tuple[int, float, float]]] = {}
    for row in centerline_rows:
        if selected is None or row.get("filament_id") != selected:
            continue
        try:
            grouped.setdefault((int(row["frame"]), selected), []).append((int(row["point_id"]), float(row["x"]), float(row["y"])))
        except (KeyError, TypeError, ValueError):
            continue
    frame_range = run_section.get("frame_range", {})
    processed_frames: list[int] = []
    if frame_range is not None and not isinstance(frame_range, Mapping):
        manifest_structure_errors.append("frame_range_not_mapping")
        frame_range = {}
    if isinstance(frame_range, Mapping):
        if frame_range.get("decode_complete") is not True:
            manifest_structure_errors.append("decode_incomplete")
        first = frame_range.get("first")
        last = frame_range.get("last")
        stride = frame_range.get("stride", 1)
        if (first is None) != (last is None):
            manifest_structure_errors.append("incomplete_frame_range")
        elif first is not None:
            try:
                first_float = float(first)
                last_float = float(last)
                stride_float = float(stride)
                if not all(math.isfinite(value) and value.is_integer() for value in (first_float, last_float, stride_float)):
                    raise ValueError("frame range values must be finite integers")
                first_int = int(first_float)
                last_int = int(last_float)
                stride_int = int(stride_float)
                if stride_int < 1 or last_int < first_int:
                    raise ValueError("invalid frame range")
                processed_frames = list(range(first_int, last_int + 1, stride_int))
                if frame_range.get("count") != len(processed_frames):
                    raise ValueError("frame range count mismatch")
            except (TypeError, ValueError):
                manifest_structure_errors.append("invalid_frame_range")
    artifact_frame_set: set[int] = set()
    for rows in (summary_rows, centerline_rows, lineage_rows):
        for row in rows:
            try:
                artifact_frame_set.add(int(row["frame"]))
            except (KeyError, TypeError, ValueError):
                continue
    if set(processed_frames) != artifact_frame_set:
        manifest_structure_errors.append("frame_range_artifact_frame_mismatch")
    consistency_validation = _validate_observation_consistency(summary_rows, centerline_rows, lineage_rows, processed_frames)
    if manifest_structure_errors:
        manifest_validation = {
            "valid": False,
            "errors": manifest_structure_errors + list(manifest_validation.get("errors", [])),
        }
        manifest_validation_valid = False
    keys: set[tuple[int, str]] = set()
    if selected is not None:
        keys.update(key for key in summary_by_key if key[1] == selected)
        keys.update(key for key in lineage_by_key if key[1] == selected)
        for row in centerline_rows:
            if row.get("filament_id") != selected:
                continue
            try:
                keys.add((int(row["frame"]), str(row.get("filament_id", ""))))
            except (KeyError, TypeError, ValueError):
                continue
    for frame in processed_frames:
        selected_key = (frame, selected) if selected is not None else (frame, "unknown")
        if selected_key not in keys:
            keys.add((frame, "unknown"))
    input_metadata = manifest_input.get("metadata", {}) if isinstance(manifest_input, Mapping) else {}
    fps_value = video_section.get("fps")
    if fps_value is None and isinstance(input_metadata, Mapping):
        fps_value = input_metadata.get("fps")
    fps = _float_or_none(fps_value)
    if fps is None or fps <= 0.0:
        manifest_structure_errors.append("missing_or_invalid_fps")
        fps = None
        manifest_validation = {
            "valid": False,
            "errors": ["missing_or_invalid_fps"] + list(manifest_validation.get("errors", [])),
        }
        manifest_validation_valid = False
    contract_valid = (
        manifest_validation_valid
        and bool(centerline_validation.get("valid"))
        and frame_keys_valid
        and bool(lineage_validation.get("valid"))
        and all(item.get("valid", False) for item in (summary_artifact, centerline_artifact, lineage_artifact))
        and bool(consistency_validation.get("valid"))
    )
    frames: list[_Frame] = []
    for frame_index, key_filament in sorted(keys):
        row = summary_by_key.get((frame_index, key_filament))
        lineage = lineage_by_key.get((frame_index, key_filament))
        if row is None and lineage is None and key_filament == "unknown":
            frames.append(_Frame(frame_index, frame_index / fps if fps is not None else None, None, None, None, "missing_observation", "missing_unknown", True, source="observation"))
            continue
        source = row or lineage or {}
        points_rows = sorted(grouped.get((frame_index, selected), []))
        points = np.asarray([[x, y] for _, x, y in points_rows], dtype=float) if points_rows else None
        raw_length = _float_or_none((row or {}).get("length_px"))
        quality = _float_or_none((row or {}).get("quality"))
        flags = str((row or {}).get("quality_flags", "ok") or "ok")
        lineage_status = str((lineage or {}).get("status", "missing_lineage"))
        try:
            censor = _bool_value((row or {}).get("censor", "0")) or _bool_value((lineage or {}).get("censor", "0"))
        except ValueError:
            censor = True
        if lineage is None:
            censor = True
        if lineage_status not in ALLOWED_LINEAGE_STATUSES:
            if lineage_status not in flags.split(";"):
                flags = ";".join(part for part in (flags, lineage_status) if part)
            censor = True
        time_s = _float_or_none(source.get("time"))
        frames.append(_Frame(frame_index, time_s, points, raw_length, quality, flags, lineage_status, censor, source="observation"))
    return frames, selected, {
        "manifest": manifest,
        "manifest_validation": manifest_validation,
        "manifest_validation_valid": manifest_validation_valid,
        "centerline_validation": centerline_validation,
        "frame_key_validation": frame_key_validation,
        "lineage_validation": lineage_validation,
        "consistency_validation": consistency_validation,
        "summary_artifact": summary_artifact,
        "centerline_artifact": centerline_artifact,
        "lineage_artifact": lineage_artifact,
        "manifest_artifact_validation": manifest_artifact_validation,
        "frame_keys_valid": frame_keys_valid,
        "contract_valid": contract_valid,
        "summary_count": len(summary_rows),
        "lineage_count": len(lineage_rows),
    }


def _validate_model_frames(frames: Sequence[_Frame], config: ScaleFreeConfig) -> dict[str, Any]:
    errors: list[str] = []
    valid_count = 0
    if not frames:
        errors.append("no_model_centerline_frames")
    previous_time: float | None = None
    for frame in frames:
        frame_errors: list[str] = []
        if frame.time_s is None or not math.isfinite(frame.time_s):
            frame_errors.append("non-finite time")
        points = frame.points
        if points is None or points.ndim != 2 or points.shape[1] != 2:
            frame_errors.append("centerline must be an array of x,y points")
        elif len(points) < config.min_points:
            frame_errors.append(f"centerline has fewer than {config.min_points} points")
        elif not np.isfinite(points).all():
            frame_errors.append("centerline has non-finite coordinates")
        else:
            length = _length(points)
            if not math.isfinite(length) or length <= config.min_length:
                frame_errors.append("centerline is degenerate")
        if previous_time is not None and frame.time_s is not None and frame.time_s < previous_time - 1.0e-12:
            frame_errors.append("time is not monotonic")
        if frame_errors:
            errors.extend(f"frame {frame.index}: {error}" for error in frame_errors)
        else:
            valid_count += 1
        if frame.time_s is not None and math.isfinite(frame.time_s):
            previous_time = frame.time_s
    return {
        "valid": not errors,
        "errors": errors,
        "warnings": [],
        "frame_count": len(frames),
        "valid_frame_count": valid_count,
        "invalid_frame_count": len(frames) - valid_count,
    }


def _read_json_model_frames(path: Path) -> tuple[list[_Frame], dict[str, Any]]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(value, dict) and isinstance(value.get("trajectory"), list):
        source_frames = value["trajectory"]
    elif isinstance(value, dict) and isinstance(value.get("frames"), list):
        source_frames = value["frames"]
    else:
        return [], {"valid": False, "errors": ["no_model_centerline_frames"], "warnings": [], "source_frame_count": 0}
    frames: list[_Frame] = []
    for index, source_frame in enumerate(source_frames):
        if not isinstance(source_frame, Mapping):
            frames.append(_Frame(index, None, None, None, source="model"))
            continue
        if "time" not in source_frame and "time_s" not in source_frame:
            time_s = None
        else:
            raw_time = source_frame.get("time", source_frame.get("time_s"))
            try:
                time_s = float(raw_time)
            except (TypeError, ValueError):
                time_s = None
        if time_s is None:
            raw_points = source_frame.get("points", source_frame.get("positions", []))
        else:
            raw_points = source_frame.get("points", source_frame.get("positions", []))
        try:
            points = np.asarray(raw_points, dtype=float)
        except (TypeError, ValueError):
            points = None
        length = _length(points) if points is not None and points.ndim == 2 and points.shape[1] == 2 and len(points) >= 2 else None
        frames.append(_Frame(index, time_s, points, length, source="model"))
    return frames, {"valid": True, "errors": [], "warnings": [], "source_frame_count": len(source_frames)}


def _validate_model_csv_source(path: Path) -> dict[str, Any]:
    required = {"time", "x", "y"}
    errors: list[str] = []
    row_count = 0
    point_ids_by_time: dict[float, list[int]] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames = set(reader.fieldnames or [])
        missing_columns = sorted(required.difference(fieldnames))
        if missing_columns:
            errors.append(f"missing required columns {','.join(missing_columns)}")
        for line_number, row in enumerate(reader, start=2):
            row_count += 1
            missing_values = sorted(field for field in required if field not in row or row[field] in (None, ""))
            if missing_values:
                errors.append(f"line {line_number}: missing required values {','.join(missing_values)}")
                continue
            try:
                time_s = float(row["time"])
                values = [time_s, float(row["x"]), float(row["y"])]
            except (TypeError, ValueError):
                errors.append(f"line {line_number}: invalid time or coordinate")
                continue
            if not np.isfinite(values).all():
                errors.append(f"line {line_number}: non-finite time or coordinate")
            if "point_id" in fieldnames:
                try:
                    point_id = int(row["point_id"])
                except (TypeError, ValueError):
                    errors.append(f"line {line_number}: invalid point_id")
                else:
                    point_ids_by_time.setdefault(time_s, []).append(point_id)
    if "point_id" in fieldnames:
        for time_s, point_ids in point_ids_by_time.items():
            expected = list(range(len(point_ids)))
            if point_ids != expected:
                errors.append(f"time {time_s}: point_id must be ordered and contiguous from zero")
    return {
        "valid": not errors,
        "errors": errors,
        "warnings": [],
        "source_row_count": row_count,
    }


def _validate_model_scope(model_path: Path, metadata: Mapping[str, Any] | None) -> dict[str, Any]:
    if model_path.suffix.lower() != ".npz" or metadata is None:
        return {"valid": False, "errors": ["stage2_scope_metadata_required"], "warnings": []}
    raw_metadata = metadata.get("metadata")
    raw_manifest = metadata.get("manifest")
    errors: list[str] = []
    sources: list[Mapping[str, Any]] = []
    if isinstance(raw_metadata, Mapping):
        sources.append(raw_metadata)
    else:
        errors.append("model_metadata_not_mapping")
    if isinstance(raw_manifest, Mapping):
        if "metadata" in raw_manifest:
            if isinstance(raw_manifest["metadata"], Mapping):
                sources.append(raw_manifest["metadata"])
            else:
                errors.append("model_manifest_metadata_not_mapping")
    else:
        errors.append("model_manifest_not_mapping")
    scope: dict[str, Any] = {}
    for key in ("benchmark", "boundary", "contact_enabled", "physical_scope", "run_kind", "sensitivity_protocol"):
        values = [source[key] for source in sources if key in source]
        if not values:
            continue
        first_value = values[0]
        normalized_first = set(first_value) if key == "physical_scope" and isinstance(first_value, list) else first_value
        for value in values[1:]:
            normalized_value = set(value) if key == "physical_scope" and isinstance(value, list) else value
            if normalized_value != normalized_first:
                errors.append(f"model_scope_metadata_mismatch:{key}")
        scope[key] = first_value
    if scope.get("benchmark") != "stage2_free_free_growth_relaxation_buckling":
        errors.append("unsupported_stage2_benchmark")
    run_kind = scope.get("run_kind")
    baseline_kinds = {"deterministic_fixture", "numerical_refinement", "parameter_contrast"}
    population = "stage2_deterministic"
    if run_kind not in baseline_kinds and run_kind != "initial_condition_sensitivity":
        errors.append("unsupported_model_run_kind")
    if run_kind == "initial_condition_sensitivity":
        population = "initial_condition_sensitivity"
        protocol = scope.get("sensitivity_protocol")
        if not isinstance(protocol, Mapping):
            errors.append("missing_sensitivity_protocol")
        else:
            if protocol.get("parameter") != "initial_condition":
                errors.append("invalid_sensitivity_parameter")
            if not protocol.get("perturbation_range"):
                errors.append("missing_sensitivity_perturbation_range")
            if not isinstance(protocol.get("metrics"), list) or not protocol.get("metrics"):
                errors.append("missing_sensitivity_metrics")
            if not isinstance(protocol.get("acceptance_criteria"), Mapping) or not protocol.get("acceptance_criteria"):
                errors.append("missing_sensitivity_acceptance_criteria")
    if scope.get("boundary") != "free/free":
        errors.append("model_boundary_is_not_free_free")
    if scope.get("contact_enabled") is not False:
        errors.append("model_contact_scope_not_disabled")
    required_scope = {
        "uniform_reference_length_growth",
        "stretching",
        "discrete_bending",
        "isotropic_substrate_drag",
    }
    physical_scope = scope.get("physical_scope")
    if not isinstance(physical_scope, list) or set(physical_scope) != required_scope:
        errors.append("model_physical_scope_not_exact_stage2_allowlist")
    return {
        "valid": not errors,
        "errors": errors,
        "warnings": [],
        "scope": scope,
        "population": population,
        "sensitivity_protocol": scope.get("sensitivity_protocol"),
    }


def _model_frames(model_path: Path, config: ScaleFreeConfig) -> tuple[list[_Frame], dict[str, Any]]:
    provenance: dict[str, Any] = {
        "logical_id": model_path.name,
        "sha256": sha256_file(model_path),
        "bytes": model_path.stat().st_size,
    }
    source_validation: dict[str, Any] = {"valid": True, "errors": [], "warnings": [], "source_row_count": None}
    scope_validation: dict[str, Any] = {"valid": False, "errors": ["stage2_scope_metadata_required"], "warnings": []}
    try:
        if model_path.suffix.lower() == ".json":
            frames, source_validation = _read_json_model_frames(model_path)
        else:
            if model_path.suffix.lower() == ".npz":
                source_validation = _validate_npz_source(model_path)
            if model_path.suffix.lower() == ".csv":
                try:
                    source_validation = _validate_model_csv_source(model_path)
                except OSError as exc:
                    source_validation = {
                        "valid": False,
                        "errors": [f"model_csv_read_failed:{type(exc).__name__}"],
                        "warnings": [],
                        "source_row_count": 0,
                    }
            loaded = load_model_output(model_path)
            frames = []
            for index, frame in enumerate(loaded):
                points = np.asarray(frame.points, dtype=float)
                length = _length(points) if points.ndim == 2 and points.shape[1] == 2 and len(points) >= 2 else None
                frames.append(_Frame(index, float(frame.time_s), points, length, source="model"))
    except (OSError, EOFError, KeyError, TypeError, ValueError, IndexError) as exc:
        provenance["validation"] = {
            "valid": False,
            "errors": list(source_validation["errors"]) + [f"model_load_failed:{type(exc).__name__}"],
            "warnings": [],
            "frame_count": 0,
            "valid_frame_count": 0,
            "invalid_frame_count": 0,
            "source_row_count": source_validation.get("source_row_count"),
            "source_frame_count": source_validation.get("source_frame_count"),
            "scope": scope_validation,
        }
        provenance["population"] = "deterministic_or_unclassified"
        return [], provenance
    frame_validation = _validate_model_frames(frames, config)
    validation_errors = list(source_validation["errors"]) + list(frame_validation["errors"])
    provenance["validation"] = {
        **frame_validation,
        "valid": not validation_errors,
        "errors": validation_errors,
        "source_row_count": source_validation.get("source_row_count"),
        "source_frame_count": source_validation.get("source_frame_count"),
    }
    if model_path.suffix.lower() == ".npz":
        try:
            with np.load(model_path, allow_pickle=False) as archive:
                raw = archive["metadata_json"]
                metadata = json.loads(str(raw.item() if raw.ndim == 0 else raw.tolist()))
            scope_validation = _validate_model_scope(model_path, metadata if isinstance(metadata, Mapping) else None)
            if not isinstance(metadata, Mapping):
                raise TypeError("model metadata must be an object")
            model_metadata = metadata.get("metadata") if isinstance(metadata.get("metadata"), Mapping) else {}
            model_manifest = metadata.get("manifest") if isinstance(metadata.get("manifest"), Mapping) else {}
            provenance.update(
                {
                    "run_kind": model_metadata.get("run_kind") or model_manifest.get("run_kind"),
                    "base_fixture": model_metadata.get("base_fixture") or model_manifest.get("base_fixture"),
                    "seed": model_metadata.get("seed"),
                    "trial": model_metadata.get("trial"),
                    "source_revision": model_manifest.get("git_revision") or model_manifest.get("source_revision"),
                    "input_hash": model_manifest.get("input_hash"),
                    "initial_state_hash": model_manifest.get("initial_state_hash"),
                    "canonical_state_hash": model_manifest.get("canonical_state_hash"),
                    "event_sequence_hash": model_manifest.get("event_sequence_hash"),
                    "failure_reason": model_metadata.get("failure_reason") or model_manifest.get("failure_reason"),
                }
            )
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
            provenance["metadata_status"] = "unavailable"
    provenance["scope_validation"] = scope_validation
    failure_reason = provenance.get("failure_reason")
    if failure_reason not in (None, ""):
        validation_errors.append("model_failure_reason_present")
    validation_errors.extend(scope_validation.get("errors", []))
    provenance["validation"]["scope"] = scope_validation
    provenance["validation"]["valid"] = not validation_errors
    provenance["validation"]["errors"] = validation_errors
    provenance["population"] = scope_validation.get("population") if scope_validation.get("valid") else "unclassified"
    provenance["sensitivity_protocol"] = scope_validation.get("sensitivity_protocol")
    return frames, provenance


def _coverage(frames: Sequence[_Frame]) -> dict[str, Any]:
    present = [frame for frame in frames if frame.index is not None]
    if not present:
        return {"first_frame": None, "last_frame": None, "count": 0, "time_first_s": None, "time_last_s": None, "time_coverage_is_physical": False}
    times = [frame.time_s for frame in present if frame.time_s is not None]
    return {
        "first_frame": min(frame.index for frame in present),
        "last_frame": max(frame.index for frame in present),
        "count": len(present),
        "time_first_s": min(times) if times else None,
        "time_last_s": max(times) if times else None,
        "time_coverage_is_physical": False,
    }


def _feature_row(prefix: str, features: Mapping[str, Any] | None) -> dict[str, Any]:
    result: dict[str, Any] = {}
    names = (
        "normalized_endpoint_distance",
        "normalized_radius_of_gyration",
        "normalized_peak_deflection",
        "curvature_rms_times_length",
    )
    for name in names:
        result[f"{prefix}_{name}"] = None if features is None else features.get(name)
    fractions = [None] * MODE_COUNT if features is None else features.get("mode_fractions", [None] * MODE_COUNT)
    for index in range(MODE_COUNT):
        result[f"{prefix}_mode_{index + 1}_fraction"] = fractions[index] if index < len(fractions) else None
    return result


def _compact_observation_provenance(observation_dir: Path, manifest: Mapping[str, Any]) -> dict[str, Any]:
    source_value = manifest.get("input")
    source = source_value if isinstance(source_value, Mapping) else {}
    artifacts_value = manifest.get("artifacts")
    artifacts = artifacts_value if isinstance(artifacts_value, Mapping) else {}
    return {
        "logical_id": source.get("logical_id") or observation_dir.name,
        "sha256": source.get("sha256"),
        "bytes": source.get("bytes"),
        "manifest_artifact_id": "manifest.json",
        "artifacts": {
            str(key): {"logical_id": value.get("path", str(key)), "sha256": value.get("sha256"), "bytes": value.get("bytes")}
            for key, value in artifacts.items()
            if isinstance(value, Mapping)
        },
    }


def scale_free_shape_comparison(
    observation_dir: str | Path,
    model_path: str | Path,
    *,
    output_dir: str | Path | None = None,
    filament_id: str | None = None,
    config: ScaleFreeConfig | Mapping[str, Any] | None = None,
    source_revision: str | None = None,
    external_artifact_ids: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Compare observation and model morphology without absolute registration.

    ``observation_dir`` must be the output of :func:`run_pipeline`; ``model_path``
    may be the existing ``trajectory.npz``, centreline CSV, or JSON trajectory
    format.  No coordinate conversion or time conversion is performed.
    """

    cfg = config if isinstance(config, ScaleFreeConfig) else ScaleFreeConfig.from_mapping(config)
    obs_dir = Path(observation_dir).expanduser().resolve()
    model_file = Path(model_path).expanduser().resolve()
    out_dir = Path(output_dir).expanduser().resolve() if output_dir is not None else obs_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    observations, selected, observation_info = _observation_frames(obs_dir, filament_id)
    models, model_provenance = _model_frames(model_file, cfg) if model_file.is_file() else (
        [],
        {
            "logical_id": model_file.name,
            "sha256": None,
            "bytes": None,
            "population": "unknown",
            "validation": {
                "valid": False,
                "errors": ["model_file_missing"],
                "warnings": [],
                "frame_count": 0,
                "valid_frame_count": 0,
                "invalid_frame_count": 0,
            },
        },
    )
    observation_manifest = observation_info.get("manifest") or {}
    observation_contract_valid = bool(observation_info.get("contract_valid"))
    model_validation = model_provenance.get("validation") or {}
    model_contract_valid = bool(models) and bool(model_validation.get("valid"))
    observation_progress = _progress(observations if observation_contract_valid else [], cfg)
    model_progress = _progress(models if model_contract_valid else [], cfg)
    _assign_progress(observations, observation_progress, cfg)
    _assign_progress(models, model_progress, cfg)
    observation_features: dict[int, dict[str, Any] | None] = {}
    for frame in observations:
        eligible = observation_contract_valid and _progress_eligible(frame, cfg)
        observation_features[frame.index] = shape_observables(frame.points, sample_points=cfg.sample_points, min_length=cfg.min_length) if eligible else None  # type: ignore[arg-type]
    model_features: dict[int, dict[str, Any] | None] = {
        frame.index: shape_observables(frame.points, sample_points=cfg.sample_points, min_length=cfg.min_length) if _finite_points(frame.points) else None  # type: ignore[arg-type]
        for frame in models
    }
    progress_alignment_possible = observation_contract_valid and model_contract_valid and observation_progress.status == "ok" and model_progress.status == "ok"
    model_by_observation: dict[int, tuple[_Frame, float]] = {}
    if progress_alignment_possible:
        usable_models = [frame for frame in models if frame.q is not None and model_features.get(frame.index) is not None]
        for observation in observations:
            if observation.q is None or observation_features.get(observation.index) is None or not usable_models:
                continue
            model = min(usable_models, key=lambda item: abs(float(item.q) - float(observation.q)))
            error = abs(float(model.q) - float(observation.q))
            if error <= cfg.max_progress_error:
                model_by_observation[observation.index] = (model, error)
    output_rows: list[dict[str, Any]] = []
    for observation in observations:
        model_match = model_by_observation.get(observation.index)
        model = model_match[0] if model_match is not None else None
        model_features_value = model_features.get(model.index) if model is not None else None
        obs_features = observation_features.get(observation.index)
        row: dict[str, Any] = {
            "observation_frame": observation.index,
            "observation_time_s": observation.time_s,
            "filament_id": selected or "unknown",
            "observation_length_px": observation.length,
            "observation_q": observation.q,
            "observation_quality": observation.quality,
            "observation_quality_flags": observation.quality_flags,
            "lineage_status": observation.lineage_status,
            "observation_censor": int(observation.censor),
            "model_frame": model.index if model is not None else None,
            "model_time_s": model.time_s if model is not None else None,
            "model_length": model.length if model is not None else None,
            "model_q": model.q if model is not None else None,
            "progress_error": model_match[1] if model_match is not None else None,
            "progress_match_method": "nearest_growth_progress" if model_match is not None else "no_match",
            "comparison_censor": int(observation.censor or obs_features is None or model_match is None),
            "eligibility_status": "eligible" if obs_features is not None else "not_eligible",
            "metric_reason": "" if obs_features is not None and model_match is not None else (
                "observation_censored_or_quality" if obs_features is None and observation.censor else
                "observation_centerline_unavailable" if obs_features is None else
                "growth_progress_unavailable_or_unmatched"
            ),
            "normalized_shape_distance": None,
            "shape_distance_orientation": None,
        }
        row.update(_feature_row("observation", obs_features))
        row.update(_feature_row("model", model_features_value))
        if obs_features is not None and model is not None and model_features_value is not None and progress_alignment_possible:
            distance, orientation = normalized_shape_distance(observation.points, model.points, sample_points=cfg.sample_points, min_length=cfg.min_length)  # type: ignore[arg-type]
            row["normalized_shape_distance"] = distance
            row["shape_distance_orientation"] = orientation
            if distance is None:
                row["comparison_censor"] = 1
                row["metric_reason"] = "shape_distance_undefined"
        output_rows.append(row)

    eligible_count = sum(row["eligibility_status"] == "eligible" for row in output_rows)
    compared_count = sum(row["normalized_shape_distance"] is not None for row in output_rows)
    reasons: list[str] = []
    if selected is None:
        reasons.append("no_selected_filament")
    if not observation_info.get("manifest_validation_valid", False):
        reasons.append("observation_manifest_validation_invalid")
    if not (observation_info.get("centerline_validation") or {}).get("valid", False):
        reasons.append("observation_centerline_contract_invalid")
    if not observation_info.get("frame_keys_valid", False):
        reasons.append("observation_frame_key_invalid")
    if not (observation_info.get("lineage_validation") or {}).get("valid", False):
        reasons.append("observation_lineage_invalid")
    if not (observation_info.get("consistency_validation") or {}).get("valid", False):
        reasons.append("observation_artifact_inconsistent")
    if not all(
        (observation_info.get(name) or {}).get("valid", False)
        for name in ("summary_artifact", "centerline_artifact", "lineage_artifact")
    ):
        reasons.append("observation_artifact_unavailable")
    if not (observation_info.get("manifest_artifact_validation") or {}).get("valid", False):
        reasons.append("observation_artifact_hash_invalid")
    if not observations:
        reasons.append("no_observation_frames")
    if not models:
        reasons.append("model_centerline_unavailable")
    if not model_contract_valid and "model_file_missing" not in model_validation.get("errors", []):
        reasons.append("model_centerline_contract_invalid")
    if observation_progress.status != "ok":
        reasons.append(f"observation_{observation_progress.status}")
    if model_progress.status != "ok":
        reasons.append(f"model_{model_progress.status}")
    if observations and eligible_count == 0:
        reasons.append("no_eligible_observation_centerlines")
    if progress_alignment_possible and eligible_count > 0 and compared_count == 0:
        reasons.append("no_valid_growth_progress_matches")
    status = "computed" if compared_count > 0 else "input_quality_comparison_unavailable"
    if not observation_contract_valid:
        status = "input_quality_invalid_observation_contract"
    elif not observations or selected is None:
        status = "input_quality_no_centerline"
    elif not models:
        status = "model_centerline_unavailable"
    elif not model_contract_valid:
        status = "input_quality_invalid_model_contract"
    elif observation_progress.status == "zero_growth_span" or model_progress.status == "zero_growth_span":
        status = "growth_progress_undefined_zero_span"
    elif observation_progress.status == "non_monotonic_lengths" or model_progress.status == "non_monotonic_lengths":
        status = "growth_progress_undefined_non_monotonic"
    elif eligible_count == 0:
        status = "input_quality_no_eligible_centerline"
    elif compared_count == 0:
        status = "no_valid_growth_progress_matches"

    config_hash = sha256_text(canonical_json(cfg.to_dict()))
    revision = source_revision or detect_git_revision(Path(__file__).resolve().parents[2])
    input_provenance = _compact_observation_provenance(obs_dir, observation_manifest)
    if external_artifact_ids:
        input_provenance["external_artifact_ids"] = dict(external_artifact_ids)
    coverage = {
        "observation": _coverage(observations),
        "model": _coverage(models),
        "frame_fraction_is_not_physical_time": True,
    }
    censored_count = sum(int(row["comparison_censor"]) for row in output_rows)
    compact: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "comparison_mode": "scale_free_shape",
        "status": status,
        "input_quality": {
            "usable": compared_count > 0,
            "censor": bool(reasons) or censored_count > 0,
            "reasons": sorted(set(reasons)),
            "diagnostic": ";".join(sorted(set(reasons))) if reasons else "ok",
        },
        "registration": {
            "status": "not_required_not_inferred",
            "pixel_per_model_unit": None,
            "time_scale": None,
            "time_offset": None,
            "physical_time_alignment": False,
        },
        "filament_id": selected,
        "config": cfg.to_dict(),
        "config_sha256": config_hash,
        "source_revision": revision,
        "observation_provenance": input_provenance,
        "model_provenance": model_provenance,
        "observation_progress": observation_progress.to_dict(),
        "model_progress": model_progress.to_dict(),
        "model_validation": model_validation,
        "video_alignment": {
            "status": "shape_alignment_only",
            "fit_performed": compared_count > 0,
            "fit_components": ["translation", "rotation", "endpoint_orientation"],
            "absolute_registration_used": False,
            "distinguished_from_initial_condition_sensitivity": True,
        },
        "initial_condition_sensitivity": {
            "population": model_provenance.get("population"),
            "protocol": model_provenance.get("sensitivity_protocol"),
            "metrics": (
                (model_provenance.get("sensitivity_protocol") or {}).get("metrics", [])
                if model_provenance.get("population") == "initial_condition_sensitivity" else []
            ),
            "acceptance_criteria": (
                (model_provenance.get("sensitivity_protocol") or {}).get("acceptance_criteria")
                if model_provenance.get("population") == "initial_condition_sensitivity" else None
            ),
        },
        "observation_validation": {
            "manifest": observation_info.get("manifest_validation"),
            "centerline": observation_info.get("centerline_validation"),
            "frame_keys": observation_info.get("frame_key_validation"),
            "lineage": observation_info.get("lineage_validation"),
            "consistency": observation_info.get("consistency_validation"),
            "artifacts": {
                "summary": observation_info.get("summary_artifact"),
                "centerline": observation_info.get("centerline_artifact"),
                "lineage": observation_info.get("lineage_artifact"),
            },
            "manifest_artifacts": observation_info.get("manifest_artifact_validation"),
            "contract_valid": observation_contract_valid,
        },
        "progress_coordinate": "q=(L-L_initial)/(L_final-L_initial); nearest matching only; no interpolation",
        "spatial_normalization": "independent current contour length L; shape samples parameterized by s/L",
        "coverage": coverage,
        "rows": len(output_rows),
        "eligible_observation_rows": eligible_count,
        "compared_rows": compared_count,
        "censored_rows": censored_count,
        "excluded_from_comparison_denominator": len(output_rows) - compared_count,
        "model_population": model_provenance.get("population"),
        "model_run_kind": model_provenance.get("run_kind"),
        "parameter_identification": "suppressed",
        "model_inadequacy": "not_assessed_in_scale_free_morphology_mode",
        "quantitative_physical_fit": "suppressed",
        "limitations": [
            "pixel and model coordinates are normalized independently by current contour length",
            "video/model time registration is not used; time fields are coverage metadata only",
            "censored, missing, low-quality, new-lineage, and reconnected frames do not contribute shape distance",
            "growth progress is not defined for zero-span or non-monotonic length records",
        ],
    }
    csv_fields = [
        "observation_frame", "observation_time_s", "filament_id", "observation_length_px", "observation_q",
        "observation_quality", "observation_quality_flags", "lineage_status", "observation_censor",
        "model_frame", "model_time_s", "model_length", "model_q", "progress_error", "progress_match_method",
        "comparison_censor", "eligibility_status", "metric_reason", "normalized_shape_distance",
        "shape_distance_orientation",
    ]
    for prefix in ("observation", "model"):
        csv_fields.extend([
            f"{prefix}_normalized_endpoint_distance",
            f"{prefix}_normalized_radius_of_gyration",
            f"{prefix}_normalized_peak_deflection",
            f"{prefix}_curvature_rms_times_length",
            *[f"{prefix}_mode_{index + 1}_fraction" for index in range(MODE_COUNT)],
        ])
    _write_csv(out_dir / "scale_free_comparison.csv", output_rows, csv_fields)
    _write_json(out_dir / "scale_free_comparison.json", compact)
    artifacts = {
        "comparison_csv": _file_record(out_dir / "scale_free_comparison.csv"),
        "comparison_json": _file_record(out_dir / "scale_free_comparison.json"),
    }
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "comparison_mode": "scale_free_shape",
        "input_logical_id": input_provenance.get("logical_id"),
        "input_sha256": input_provenance.get("sha256"),
        "model_logical_id": model_provenance.get("logical_id"),
        "model_sha256": model_provenance.get("sha256"),
        "source_revision": revision,
        "config_sha256": config_hash,
        "model_population": model_provenance.get("population"),
        "video_alignment": compact["video_alignment"],
        "initial_condition_sensitivity": compact["initial_condition_sensitivity"],
        "status": status,
        "eligible_observation_rows": eligible_count,
        "compared_rows": compared_count,
        "censored_rows": compact["censored_rows"],
        "artifacts": artifacts,
        "external_artifact_ids": dict(external_artifact_ids or {}),
        "registration": compact["registration"],
        "spatial_normalization": compact["spatial_normalization"],
        "progress_coordinate": compact["progress_coordinate"],
    }
    _write_json(out_dir / "scale_free_comparison_manifest.json", manifest)
    compact["artifacts"] = artifacts
    compact["manifest_logical_id"] = "scale_free_comparison_manifest.json"
    return {"summary": compact, "rows": output_rows, "manifest": manifest}


__all__ = [
    "MODE_COUNT",
    "SCHEMA_VERSION",
    "ScaleFreeConfig",
    "normalized_shape_distance",
    "scale_free_shape_comparison",
    "shape_observables",
]
