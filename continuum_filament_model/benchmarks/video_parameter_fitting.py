"""Fit growth and conditional mechanical parameters to extracted video data.

This runner is the observation-side bridge between the video extraction
contract (PR #16) and the synthetic recovery protocol (PR #12).  It does not
silently turn a censored or uncalibrated frame into a measurement:

* contour-length growth is fitted with deterministic Huber regression in both
  log-length (exponential) and length (linear) coordinates;
* confidence intervals and frame residuals are reported for the selected fit;
* shape metrics are computed only when an explicit pixel/model calibration is
  supplied and a model trajectory is available;
* ``chi`` is a conditional value of the selected model trajectory, not a free
  estimate from a single observed movie; and
* ``diameter_proxy`` is estimated only from an explicit width field.

The committed video manifests intentionally contain hashes and compact run
metadata, not the extracted centerline files.  ``--data-root`` points at the
uncommitted extraction directory when those files are available.  Running the
CLI without that directory produces an auditable ``centerline_not_found``
summary rather than a fabricated estimate.

Example::

    PYTHONPATH=continuum_filament_model/src \\
      python continuum_filament_model/benchmarks/video_parameter_fitting.py \\
      --manifest continuum_filament_model/results/video_comparison/gray5_manifest.json \\
      --manifest continuum_filament_model/results/video_comparison/original_manifest.json \\
      --data-root /path/to/extraction/output \\
      --model /path/to/trajectory.npz \\
      --pixel-per-model-unit 12 \\
      --output continuum_filament_model/results/video_parameter_fitting
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

if __package__ in {None, ""}:
    _HERE = Path(__file__).resolve()
    _SRC = _HERE.parents[1] / "src"
    if str(_SRC) not in sys.path:
        sys.path.insert(0, str(_SRC))

from growing_filament.video_comparison import (  # noqa: E402
    endpoint_correspondence,
    load_model_output,
)

SCHEMA_VERSION = "continuum-filament-video-parameter-fitting-1"
ANALYSIS_REVISION = "video-parameter-fitting-v1"
RECOVERY_PROTOCOL = "continuum_filament_model/benchmarks/synthetic_data_recovery.py"
_CENSOR_FLAGS = {
    "ambiguous_components",
    "branched_component",
    "components_truncated",
    "disconnected_skeleton",
    "large_jump",
    "loop_component",
    "low_quality",
    "new_lineage",
    "out_of_view",
    "reconnected_after_missing",
    "roi_clipped",
    "short_centerline",
    "skeleton_loss",
}


class VideoParameterFittingError(ValueError):
    """Raised when an observation source violates the fitting contract."""


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return [_jsonable(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def canonical_json(value: Any) -> str:
    return json.dumps(_jsonable(value), ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(canonical_json(value) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        for row in rows:
            clean: dict[str, Any] = {}
            for field in fields:
                value = row.get(field)
                if isinstance(value, float) and not math.isfinite(value):
                    value = ""
                clean[field] = value
            writer.writerow(clean)


def _float(value: Any, default: float | None = None) -> float | None:
    if value in (None, "", "null"):
        return default
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def _int(value: Any, default: int | None = None) -> int | None:
    if value in (None, "", "null"):
        return default
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _truthy(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _safe_name(value: str) -> str:
    token = "".join(char if char.isalnum() or char in "-_" else "_" for char in value)
    return token.strip("_") or "observation"


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


# ---------------------------------------------------------------------------
# Observation source resolution and censoring


def _artifact_entry(manifest: Mapping[str, Any], name: str) -> Mapping[str, Any] | None:
    for container in (
        manifest.get("full_period_run", {}).get("artifacts", {}),
        manifest.get("artifacts", {}),
    ):
        value = container.get(name)
        if isinstance(value, Mapping):
            return value
    return None


def _candidate_roots(source: Path, data_root: Path | None) -> list[Path]:
    roots: list[Path] = []
    if data_root is not None:
        roots.extend([data_root, data_root / "full_period", data_root / source.stem])
    roots.extend([source.parent, source.parent / "full_period", source.parent / source.stem])
    unique: list[Path] = []
    seen: set[str] = set()
    for root in roots:
        resolved = root.expanduser().resolve()
        if str(resolved) not in seen:
            unique.append(resolved)
            seen.add(str(resolved))
    return unique


def resolve_observation_source(source: str | Path, data_root: str | Path | None = None) -> dict[str, Any]:
    """Resolve a centerline CSV and companion artifacts from a manifest.

    A compact PR #16 result manifest stores artifact names and hashes but not
    the local extraction directory.  Resolution therefore remains explicit;
    no recursive search outside ``data_root`` is performed.
    """

    source_path = Path(source).expanduser().resolve()
    root = Path(data_root).expanduser().resolve() if data_root is not None else None
    if not source_path.is_file():
        raise FileNotFoundError(source_path)
    if source_path.suffix.lower() == ".csv":
        manifest: dict[str, Any] = {}
        manifest_path: Path | None = None
        artifact_name = source_path.name
    else:
        manifest = json.loads(source_path.read_text(encoding="utf-8"))
        if not isinstance(manifest, Mapping):
            raise VideoParameterFittingError(f"manifest root must be an object: {source_path}")
        manifest_path = source_path
        centerline_entry = _artifact_entry(manifest, "centerline")
        artifact_name = str((centerline_entry or {}).get("path", "centerline.csv"))
    artifact_file: Path | None = None
    roots = _candidate_roots(source_path, root)
    direct_artifact = Path(artifact_name)
    if direct_artifact.is_absolute() and direct_artifact.is_file():
        artifact_file = direct_artifact.resolve()
    else:
        for candidate_root in roots:
            candidate = candidate_root / artifact_name
            if candidate.is_file():
                artifact_file = candidate
                break
    artifact_entry = _artifact_entry(manifest, "centerline")
    companion: dict[str, Path] = {}
    for name, fallback in (
        ("observation_summary", "observation_summary.csv"),
        ("lineage", "lineage.csv"),
        ("events", "events.csv"),
        ("metadata", "metadata.json"),
    ):
        entry = _artifact_entry(manifest, name)
        artifact = str((entry or {}).get("path", fallback))
        for candidate_root in roots:
            candidate = candidate_root / artifact
            if candidate.is_file():
                companion[name] = candidate.resolve()
                break
    # A caller may provide the parent of per-video extraction directories.
    # Add a logical-id-derived candidate after reading the manifest, while
    # keeping resolution explicitly bounded to the supplied roots.
    logical_id_hint = str(
        manifest.get("input", {}).get("logical_id")
        or manifest.get("observation_logical_id")
        or manifest.get("video_logical_id")
        or source_path.stem
    )
    if root is not None:
        for logical_root in (root / Path(logical_id_hint).stem, root / logical_id_hint):
            if logical_root.resolve() not in roots:
                roots.append(logical_root.resolve())
    # Re-run companion lookup with the logical-id-derived candidates.
    if artifact_file is None and not direct_artifact.is_absolute():
        for candidate_root in roots:
            candidate = candidate_root / artifact_name
            if candidate.is_file():
                artifact_file = candidate
                break
    for name, fallback in (
        ("observation_summary", "observation_summary.csv"),
        ("lineage", "lineage.csv"),
        ("events", "events.csv"),
        ("metadata", "metadata.json"),
    ):
        if name in companion:
            continue
        entry = _artifact_entry(manifest, name)
        artifact = str((entry or {}).get("path", fallback))
        for candidate_root in roots:
            candidate = candidate_root / artifact
            if candidate.is_file():
                companion[name] = candidate.resolve()
                break
    integrity: dict[str, Any] = {
        "declared_sha256": (artifact_entry or {}).get("sha256"),
        "declared_bytes": (artifact_entry or {}).get("bytes"),
        "observed_sha256": sha256_file(artifact_file) if artifact_file else None,
        "observed_bytes": artifact_file.stat().st_size if artifact_file else None,
    }
    integrity["match"] = (
        artifact_file is not None
        and (integrity["declared_sha256"] is None or integrity["declared_sha256"] == integrity["observed_sha256"])
        and (integrity["declared_bytes"] is None or int(integrity["declared_bytes"]) == integrity["observed_bytes"])
    ) if artifact_file is not None else None
    logical_id = str(
        manifest.get("input", {}).get("logical_id")
        or manifest.get("observation_logical_id")
        or manifest.get("video_logical_id")
        or source_path.stem
    )
    selected = (
        manifest.get("metadata", {}).get("selected_filament_id")
        or manifest.get("selected_filament_id")
        or manifest.get("comparison", {}).get("summary", {}).get("selected_lineage_id")
        or manifest.get("comparison", {}).get("summary", {}).get("filament_id")
    )
    return {
        "source": str(source_path),
        "manifest_path": str(manifest_path) if manifest_path else None,
        "manifest_sha256": sha256_file(manifest_path) if manifest_path else None,
        "logical_id": logical_id,
        "manifest": _jsonable(manifest),
        "centerline_path": str(artifact_file) if artifact_file else None,
        "companion_paths": {name: str(path) for name, path in companion.items()},
        "declared_centerline": _jsonable(artifact_entry or {}),
        "integrity": integrity,
        "selected_filament_id": str(selected) if selected else None,
        "status": "resolved" if artifact_file else "centerline_not_found",
    }


def _metadata_value(source: Mapping[str, Any], key: str, default: Any = None) -> Any:
    metadata = source.get("manifest", {}).get("metadata", {})
    if isinstance(metadata, Mapping) and key in metadata:
        return metadata[key]
    return source.get("manifest", {}).get(key, default)


def _flags(value: Any) -> list[str]:
    return [part for part in str(value or "").split(";") if part and part != "ok"]


def _choose_filament(centerline: Sequence[Mapping[str, Any]], summary: Sequence[Mapping[str, Any]], source: Mapping[str, Any], requested: str | None) -> str | None:
    selected = requested or source.get("selected_filament_id")
    if selected:
        return str(selected)
    counts: Counter[str] = Counter()
    for row in list(centerline) + list(summary):
        if row.get("filament_id") not in (None, ""):
            counts[str(row["filament_id"])] += 1
    return sorted(counts, key=lambda item: (-counts[item], item))[0] if counts else None


def _observation_rows(
    source: Mapping[str, Any],
    *,
    filament_id: str | None = None,
    quality_threshold: float = 0.2,
    coordinate_scale: float = 1.0,
    width_scale: float | None = None,
    min_points: int = 3,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    centerline_path = source.get("centerline_path")
    if not centerline_path:
        return [], {"status": "centerline_not_found", "selected_filament_id": filament_id}
    centerline = _read_csv(Path(centerline_path))
    companions = source.get("companion_paths", {})
    summary = _read_csv(Path(companions["observation_summary"])) if "observation_summary" in companions else []
    lineage = _read_csv(Path(companions["lineage"])) if "lineage" in companions else []
    chosen = _choose_filament(centerline, summary, source, filament_id)
    grouped: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
    for row in centerline:
        current = str(row.get("filament_id", ""))
        if chosen is not None and current != chosen:
            continue
        frame = _int(row.get("frame"))
        time_s = _float(row.get("time"))
        if frame is None:
            frame = int(round((time_s or 0.0) * 1_000_000))
        if time_s is None:
            time_s = float(frame)
        x, y = _float(row.get("x")), _float(row.get("y"))
        if x is None or y is None or not np.isfinite([x, y]).all():
            continue
        grouped[(frame, current)].append({
            "point_id": _int(row.get("point_id"), len(grouped[(frame, current)])),
            "x": x,
            "y": y,
            "quality": _float(row.get("quality"), 0.0),
            "time": time_s,
            "width": _float(row.get("width", row.get("diameter", row.get("thickness")))),
            "flags": _flags(row.get("quality_flags", row.get("flags", ""))),
            "censor": _truthy(row.get("censor", False)),
        })
    summary_by_key: dict[tuple[int, str], Mapping[str, Any]] = {}
    for row in summary:
        current = str(row.get("filament_id", ""))
        if chosen is None or current == chosen:
            frame = _int(row.get("frame"))
            if frame is not None:
                summary_by_key[(frame, current)] = row
    lineage_by_key: dict[tuple[int, str], Mapping[str, Any]] = {}
    for row in lineage:
        current = str(row.get("filament_id", ""))
        if chosen is None or current == chosen:
            frame = _int(row.get("frame"))
            if frame is not None:
                lineage_by_key[(frame, current)] = row
    keys = set(grouped) | set(summary_by_key) | set(lineage_by_key)
    records: list[dict[str, Any]] = []
    for frame, current in sorted(keys):
        points_rows = sorted(grouped.get((frame, current), []), key=lambda row: int(row["point_id"]))
        summary_row = summary_by_key.get((frame, current), {})
        lineage_row = lineage_by_key.get((frame, current), {})
        time_s = _float(summary_row.get("time"), _float(lineage_row.get("time"), points_rows[0]["time"] if points_rows else float(frame)))
        points = np.asarray([[row["x"], row["y"]] for row in points_rows], dtype=float)
        quality_values = [float(row["quality"]) for row in points_rows if row["quality"] is not None]
        quality = _float(summary_row.get("quality"), float(np.mean(quality_values)) if quality_values else None)
        flags = set(_flags(summary_row.get("quality_flags", "")))
        for row in points_rows:
            flags.update(row["flags"])
        status = str(lineage_row.get("status", "observed"))
        if status not in {"observed", "matched", "initial_lineage"}:
            flags.add(status)
        censored = _truthy(summary_row.get("censor", False)) or _truthy(lineage_row.get("censor", False)) or any(row["censor"] for row in points_rows)
        censor_flags = sorted(flags & _CENSOR_FLAGS)
        if censor_flags:
            reason = censor_flags[0]
            censored = True
        elif not points_rows:
            reason = "missing_observation_lineage"
            censored = True
        elif len(points_rows) < min_points:
            reason = "insufficient_centerline_points"
            censored = True
        elif quality is not None and quality < quality_threshold:
            reason = "quality_below_threshold"
            censored = True
        else:
            reason = "quality_censor_flag" if censored else None
        if reason:
            censored = True
        length_px = _polyline_length(points) if len(points) >= 2 else _float(summary_row.get("length_px"))
        widths = [float(row["width"]) for row in points_rows if row["width"] is not None and math.isfinite(float(row["width"]))]
        scale = coordinate_scale if width_scale is None else width_scale
        records.append({
            "frame": int(frame),
            "time": float(time_s if time_s is not None else frame),
            "filament_id": current,
            "points": points,
            "length_px": length_px,
            "length": None if length_px is None else float(length_px) * coordinate_scale,
            "width": float(np.median(widths) * scale) if widths else None,
            "quality": quality,
            "quality_flags": sorted(flags),
            "lineage_status": status,
            "censor": int(censored),
            "eligible": not censored,
            "exclusion_reason": reason,
        })
    records.sort(key=lambda row: (row["time"], row["frame"]))
    return records, {
        "status": "resolved",
        "selected_filament_id": chosen,
        "centerline_rows": len(centerline),
        "summary_rows": len(summary),
        "lineage_rows": len(lineage),
        "population_rows": len(records),
        "eligible_rows": sum(bool(row["eligible"]) for row in records),
        "censored_rows": sum(not bool(row["eligible"]) for row in records),
        "excluded_reason_counts": dict(sorted(Counter(row["exclusion_reason"] for row in records if row["exclusion_reason"]).items())),
    }


def load_observation(
    source: str | Path,
    *,
    data_root: str | Path | None = None,
    filament_id: str | None = None,
    quality_threshold: float = 0.2,
    coordinate_scale: float = 1.0,
    width_scale: float | None = None,
    min_points: int = 3,
) -> dict[str, Any]:
    """Load a manifest/centerline and return censored frame records."""

    resolved = resolve_observation_source(source, data_root)
    records, population = _observation_rows(
        resolved,
        filament_id=filament_id,
        quality_threshold=quality_threshold,
        coordinate_scale=coordinate_scale,
        width_scale=width_scale,
        min_points=min_points,
    )
    return {"source": resolved, "records": records, "population": population}


# ---------------------------------------------------------------------------
# Deterministic robust regression and geometric metrics


def _polyline_length(points: np.ndarray) -> float:
    return float(np.sum(np.linalg.norm(np.diff(points, axis=0), axis=1))) if len(points) >= 2 else 0.0


def _resample(points: np.ndarray, n: int = 80) -> np.ndarray:
    if len(points) == 0:
        return np.empty((0, 2), dtype=float)
    if len(points) == 1:
        return np.repeat(points, n, axis=0)
    distance = np.concatenate(([0.0], np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))))
    if distance[-1] <= 1.0e-12:
        return np.repeat(points[:1], n, axis=0)
    target = np.linspace(0.0, float(distance[-1]), n)
    return np.column_stack([np.interp(target, distance, points[:, axis]) for axis in range(2)])


def _curvature(points: np.ndarray) -> np.ndarray:
    if len(points) < 3:
        return np.empty(0, dtype=float)
    before, center, after = points[:-2], points[1:-1], points[2:]
    first, second = center - before, after - center
    first_length = np.linalg.norm(first, axis=1)
    second_length = np.linalg.norm(second, axis=1)
    if np.any(first_length <= 1.0e-12) or np.any(second_length <= 1.0e-12):
        return np.empty(0, dtype=float)
    tangent_before = first / first_length[:, None]
    tangent_after = second / second_length[:, None]
    angle = np.arctan2(
        tangent_before[:, 0] * tangent_after[:, 1] - tangent_before[:, 1] * tangent_after[:, 0],
        np.sum(tangent_before * tangent_after, axis=1),
    )
    return angle / (0.5 * (first_length + second_length))


def discrete_frechet(first: np.ndarray, second: np.ndarray) -> float:
    """Return the discrete Frechet distance for two ordered polylines."""

    if len(first) == 0 or len(second) == 0:
        return float("nan")
    distances = np.linalg.norm(first[:, None, :] - second[None, :, :], axis=2)
    dynamic = np.empty_like(distances)
    dynamic[0, 0] = distances[0, 0]
    for i in range(1, len(first)):
        dynamic[i, 0] = max(dynamic[i - 1, 0], distances[i, 0])
    for j in range(1, len(second)):
        dynamic[0, j] = max(dynamic[0, j - 1], distances[0, j])
    for i in range(1, len(first)):
        for j in range(1, len(second)):
            dynamic[i, j] = max(distances[i, j], min(dynamic[i - 1, j], dynamic[i - 1, j - 1], dynamic[i, j - 1]))
    return float(dynamic[-1, -1])


def _huber_fit(x: Sequence[float], y: Sequence[float], delta: float = 1.345) -> dict[str, Any]:
    """Deterministic Huber IRLS fit of ``y = intercept + slope*x``."""

    x_array, y_array = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    valid = np.isfinite(x_array) & np.isfinite(y_array)
    x_array, y_array = x_array[valid], y_array[valid]
    if len(x_array) < 3 or np.ptp(x_array) <= 1.0e-12:
        return {"status": "insufficient_data", "n": int(len(x_array))}
    design = np.column_stack((np.ones(len(x_array)), x_array))
    beta = np.linalg.lstsq(design, y_array, rcond=None)[0]
    weights = np.ones(len(x_array))
    for _ in range(50):
        updated = np.linalg.lstsq(design * np.sqrt(weights)[:, None], y_array * np.sqrt(weights), rcond=None)[0]
        residual = y_array - design @ updated
        scale = 1.4826 * float(np.median(np.abs(residual - np.median(residual))))
        scale = max(scale, float(np.std(residual)), 1.0e-12)
        ratio = np.abs(residual) / (delta * scale)
        new_weights = np.where(ratio <= 1.0, 1.0, 1.0 / np.maximum(ratio, 1.0e-12))
        if np.max(np.abs(updated - beta)) <= 1.0e-12 * max(1.0, float(np.max(np.abs(beta)))):
            beta = updated
            weights = new_weights
            break
        beta, weights = updated, new_weights
    residual = y_array - design @ beta
    weighted_sse = float(np.sum(weights * residual * residual))
    dof = max(len(x_array) - 2, 1)
    sigma2 = weighted_sse / dof
    information = design.T @ (weights[:, None] * design)
    covariance = sigma2 * np.linalg.pinv(information)
    standard_error = np.sqrt(np.maximum(np.diag(covariance), 0.0))
    ci = [[float(beta[index] - 1.96 * standard_error[index]), float(beta[index] + 1.96 * standard_error[index])] for index in range(2)]
    rss = float(np.sum(residual * residual))
    variance = max(rss / len(x_array), 1.0e-30)
    aic = float(len(x_array) * (math.log(2.0 * math.pi * variance) + 1.0) + 4.0)
    return {
        "status": "ok",
        "n": int(len(x_array)),
        "intercept": float(beta[0]),
        "slope": float(beta[1]),
        "intercept_ci95": ci[0],
        "slope_ci95": ci[1],
        "standard_error": [float(value) for value in standard_error],
        "residual_sse": rss,
        "rmse": float(math.sqrt(rss / len(x_array))),
        "aic": aic,
        "predicted": (design @ beta).tolist(),
        "residuals": residual.tolist(),
        "x": x_array.tolist(),
        "y": y_array.tolist(),
    }


def _growth_fit(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    eligible = [row for row in records if row.get("eligible") and row.get("length") is not None and float(row["length"]) > 0.0]
    times = [float(row["time"]) for row in eligible]
    lengths = [float(row["length"]) for row in eligible]
    exponential = _huber_fit(times, np.log(lengths))
    linear = _huber_fit(times, lengths)
    result: dict[str, Any] = {
        "status": "insufficient_data",
        "eligible_frames": len(eligible),
        "exponential": exponential,
        "linear": linear,
        "selected_model": None,
        "growth_rate": None,
        "growth_rate_ci95": None,
        "initial_length": None,
        "residuals": [],
    }
    if exponential.get("status") != "ok" and linear.get("status") != "ok":
        return result
    if exponential.get("status") == "ok" and linear.get("status") == "ok":
        # Compare normalized residuals because the response transforms have
        # different units.  AIC remains in each model's native likelihood.
        exp_score = float(exponential["rmse"])
        lin_score = float(linear["rmse"]) / max(float(np.median(lengths)), 1.0e-12)
        selected = "exponential" if exp_score <= lin_score else "linear"
    else:
        selected = "exponential" if exponential.get("status") == "ok" else "linear"
    if selected == "exponential":
        fit = exponential
        result.update({
            "status": "ok",
            "selected_model": selected,
            "growth_rate": fit["slope"],
            "growth_rate_ci95": fit["slope_ci95"],
            "initial_length": math.exp(float(fit["intercept"])),
        })
        result["initial_length_ci95"] = [math.exp(float(fit["intercept_ci95"][0])), math.exp(float(fit["intercept_ci95"][1]))]
        prediction = np.exp(np.asarray(fit["predicted"], dtype=float))
        result["residuals"] = [
            {"frame": int(row["frame"]), "time": float(row["time"]), "length": float(row["length"]), "predicted_length": float(pred), "residual": float(row["length"] - pred), "selected_model": selected}
            for row, pred in zip(eligible, prediction)
        ]
    else:
        fit = linear
        intercept, slope = float(fit["intercept"]), float(fit["slope"])
        growth = slope / intercept if intercept > 1.0e-12 else None
        if growth is not None:
            covariance = np.asarray(fit["standard_error"], dtype=float)
            # Conservative delta-method interval when linear growth is chosen.
            growth_se = math.sqrt((covariance[1] / intercept) ** 2 + (slope * covariance[0] / intercept**2) ** 2)
            growth_ci = [growth - 1.96 * growth_se, growth + 1.96 * growth_se]
        else:
            growth_ci = None
        result.update({
            "status": "ok",
            "selected_model": selected,
            "growth_rate": growth,
            "growth_rate_ci95": growth_ci,
            "initial_length": intercept,
            "initial_length_ci95": fit["intercept_ci95"],
        })
        result["residuals"] = [
            {"frame": int(row["frame"]), "time": float(row["time"]), "length": float(row["length"]), "predicted_length": float(pred), "residual": float(row["length"] - pred), "selected_model": selected}
            for row, pred in zip(eligible, fit["predicted"])
        ]
    return result


def _diameter_fit(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    values = np.asarray([float(row["width"]) for row in records if row.get("eligible") and row.get("width") is not None], dtype=float)
    if len(values) == 0:
        return {"status": "unidentifiable_no_width_field", "n": 0, "diameter_proxy": None, "ci95": None}
    median = float(np.median(values))
    mad_scale = 1.4826 * float(np.median(np.abs(values - median)))
    standard_error = mad_scale / math.sqrt(len(values)) if len(values) > 1 else 0.0
    return {
        "status": "proxy_estimate",
        "n": int(len(values)),
        "diameter_proxy": median,
        "ci95": [median - 1.96 * standard_error, median + 1.96 * standard_error],
        "median_absolute_deviation": float(np.median(np.abs(values - median))),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
    }


def _model_parameters(path: Path) -> dict[str, Any]:
    value: dict[str, Any] = {}
    if path.suffix.lower() == ".npz":
        try:
            with np.load(path, allow_pickle=False) as archive:
                raw = archive.get("metadata_json")
                if raw is not None:
                    decoded = raw.item() if raw.ndim == 0 else raw.tolist()
                    metadata = json.loads(str(decoded))
                    if isinstance(metadata, Mapping):
                        value = dict(metadata.get("parameters", {}))
        except (OSError, ValueError, KeyError, json.JSONDecodeError):
            value = {}
    if not value:
        for sidecar in (path.with_suffix(path.suffix + ".json"), path.with_suffix(".json")):
            if sidecar.is_file():
                try:
                    decoded = json.loads(sidecar.read_text(encoding="utf-8"))
                    if isinstance(decoded, Mapping):
                        value = dict(decoded.get("parameters", decoded))
                        break
                except (OSError, json.JSONDecodeError):
                    pass
    return value


def _chi_from_parameters(parameters: Mapping[str, Any]) -> float | None:
    ea = _float(parameters.get("axial_stiffness", parameters.get("EA")))
    ei = _float(parameters.get("bending_stiffness", parameters.get("EI")))
    length = _float(parameters.get("reference_length", parameters.get("length")))
    if ea is None or ei is None or length is None or ea <= 0.0 or length <= 0.0:
        return None
    return float(ei / (ea * length * length))


def _registration_points(points: np.ndarray, registration: Mapping[str, Any]) -> np.ndarray:
    scale = _float(registration.get("pixel_per_model_unit"))
    if scale is None or scale <= 0.0:
        raise VideoParameterFittingError("pixel_per_model_unit is required for shape metrics")
    angle = math.radians(float(registration.get("rotation_deg", 0.0)))
    rotation = np.asarray([[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]])
    offset = np.asarray([float(registration.get("x_offset_px", 0.0)), float(registration.get("y_offset_px", 0.0))])
    return np.asarray(points, dtype=float) @ rotation.T * scale + offset


def _shape_candidate(
    records: Sequence[Mapping[str, Any]],
    model_path: Path,
    registration: Mapping[str, Any],
    *,
    resample_points: int = 80,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    frames = load_model_output(model_path)
    time_scale = float(registration.get("time_scale", 1.0))
    time_offset = float(registration.get("time_offset", 0.0))
    max_time_error = float(registration.get("max_time_error_s", 0.20))
    per_frame: list[dict[str, Any]] = []
    for record in records:
        row: dict[str, Any] = {
            "frame": int(record["frame"]),
            "time": float(record["time"]),
            "candidate_id": model_path.name,
            "eligible": int(bool(record["eligible"])),
            "frechet_distance_px": None,
            "curvature_mse_px_inv2": None,
            "shape_status": "excluded_observation_censor",
            "shape_reason": record.get("exclusion_reason") or "quality_censor_flag",
        }
        if not record["eligible"]:
            per_frame.append(row)
            continue
        model_time = (float(record["time"]) - time_offset) / time_scale
        if not frames:
            row.update(shape_status="not_computed_model_centerline_unavailable", shape_reason="model_centerline_unavailable")
            per_frame.append(row)
            continue
        model_frame = min(frames, key=lambda value: abs(value.time_s - model_time))
        time_error = abs(model_frame.time_s - model_time)
        row["model_time"] = float(model_frame.time_s)
        row["time_error_s"] = float(time_error)
        if time_error > max_time_error or len(record["points"]) < 2 or len(model_frame.points) < 2:
            row.update(shape_status="not_computed_time_or_centerline", shape_reason="model_time_unmatched_or_centerline_unavailable")
            per_frame.append(row)
            continue
        try:
            model_pixels = _registration_points(model_frame.points, registration)
        except (ValueError, TypeError, VideoParameterFittingError):
            row.update(shape_status="not_computed_uncalibrated", shape_reason="pixel_per_model_unit_not_specified")
            per_frame.append(row)
            continue
        correspondence = endpoint_correspondence(record["points"], model_pixels, str(registration.get("endpoint_order", "auto")))
        selected_model = np.asarray(correspondence["model_points"], dtype=float)
        observed_sample = _resample(np.asarray(record["points"], dtype=float), resample_points)
        model_sample = _resample(selected_model, resample_points)
        observed_curvature = _curvature(observed_sample)
        model_curvature = _curvature(model_sample)
        if len(observed_curvature) == 0 or len(model_curvature) == 0:
            row.update(shape_status="not_computed_degenerate_geometry", shape_reason="curvature_unavailable")
            per_frame.append(row)
            continue
        n = min(len(observed_curvature), len(model_curvature))
        frechet = discrete_frechet(observed_sample, model_sample)
        curvature_mse = float(np.mean((observed_curvature[:n] - model_curvature[:n]) ** 2))
        row.update(
            frechet_distance_px=float(frechet),
            curvature_mse_px_inv2=curvature_mse,
            endpoint_distance_px=float(correspondence["selected_endpoint_distance_px"]),
            shape_status="computed",
            shape_reason="",
        )
        per_frame.append(row)
    computed = [row for row in per_frame if row["shape_status"] == "computed"]
    if computed:
        losses = [
            float(row["frechet_distance_px"]) / max(float(next(item["length_px"] for item in records if item["frame"] == row["frame"] and item["length_px"] is not None)), 1.0e-12)
            + math.sqrt(float(row["curvature_mse_px_inv2"])) * max(float(next(item["length_px"] for item in records if item["frame"] == row["frame"] and item["length_px"] is not None)), 1.0e-12)
            for row in computed
        ]
        summary: dict[str, Any] = {
            "candidate_id": model_path.name,
            "status": "computed",
            "eligible_rows": len(computed),
            "population_rows": len(per_frame),
            "excluded_rows": len(per_frame) - len(computed),
            "frechet_distance_px_median": float(np.median([row["frechet_distance_px"] for row in computed])),
            "curvature_mse_px_inv2_median": float(np.median([row["curvature_mse_px_inv2"] for row in computed])),
            "normalized_shape_loss_median": float(np.median(losses)),
        }
    else:
        summary = {
            "candidate_id": model_path.name,
            "status": "no_computed_metrics",
            "eligible_rows": 0,
            "population_rows": len(per_frame),
            "excluded_rows": len(per_frame),
            "frechet_distance_px_median": None,
            "curvature_mse_px_inv2_median": None,
            "normalized_shape_loss_median": None,
        }
    parameters = _model_parameters(model_path)
    summary.update({
        "path": str(model_path.resolve()),
        "sha256": sha256_file(model_path),
        "parameters": parameters,
        "chi": _chi_from_parameters(parameters),
        "diameter": _float(parameters.get("diameter")),
    })
    return summary, per_frame


def fit_observation(
    source: str | Path,
    *,
    data_root: str | Path | None = None,
    model_paths: Sequence[str | Path] = (),
    config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Fit one manifest/centerline source and return a compact report."""

    values = dict(config or {})
    coordinate_scale = float(values.get("coordinate_scale", 1.0))
    width_scale = values.get("width_scale")
    width_scale = None if width_scale in (None, "") else float(width_scale)
    if coordinate_scale <= 0.0 or (width_scale is not None and width_scale <= 0.0):
        raise VideoParameterFittingError("coordinate_scale and width_scale must be positive")
    observation = load_observation(
        source,
        data_root=data_root,
        filament_id=values.get("filament_id"),
        quality_threshold=float(values.get("quality_threshold", 0.2)),
        coordinate_scale=coordinate_scale,
        width_scale=width_scale,
        min_points=int(values.get("min_points", 3)),
    )
    resolved = observation["source"]
    records = observation["records"]
    growth = _growth_fit(records)
    diameter = _diameter_fit(records)
    registration = {
        "pixel_per_model_unit": values.get("pixel_per_model_unit"),
        "x_offset_px": float(values.get("x_offset_px", 0.0)),
        "y_offset_px": float(values.get("y_offset_px", 0.0)),
        "rotation_deg": float(values.get("rotation_deg", 0.0)),
        "time_scale": float(values.get("time_scale", 1.0)),
        "time_offset": float(values.get("time_offset", 0.0)),
        "max_time_error_s": float(values.get("max_time_error_s", 0.20)),
        "endpoint_order": str(values.get("endpoint_order", "auto")),
    }
    shape_candidates: list[dict[str, Any]] = []
    shape_rows: list[dict[str, Any]] = []
    if model_paths:
        for model_path_value in sorted({str(Path(path).expanduser().resolve()) for path in model_paths}):
            model_path = Path(model_path_value)
            candidate, rows = _shape_candidate(
                records,
                model_path,
                registration,
                resample_points=int(values.get("resample_points", 80)),
            )
            shape_candidates.append(candidate)
            shape_rows.extend(rows)
        fitted = [candidate for candidate in shape_candidates if candidate.get("status") == "computed"]
        selected = min(fitted, key=lambda candidate: (float(candidate["normalized_shape_loss_median"]), str(candidate["candidate_id"]))) if fitted else None
        shape = {
            "status": "computed" if selected else "no_computed_metrics",
            "registration": registration,
            "candidate_count": len(shape_candidates),
            "eligible_rows": int(selected["eligible_rows"]) if selected else 0,
            "population_rows": len(records),
            "selected_candidate_id": selected["candidate_id"] if selected else None,
            "selected_candidate": selected,
            "candidates": shape_candidates,
        }
    else:
        shape = {
            "status": "not_requested",
            "registration": registration,
            "candidate_count": 0,
            "eligible_rows": 0,
            "population_rows": len(records),
            "selected_candidate_id": None,
            "selected_candidate": None,
            "candidates": [],
        }
    selected_candidate = shape.get("selected_candidate") or {}
    selected_chi = selected_candidate.get("chi")
    selected_model_diameter = selected_candidate.get("diameter")
    report = {
        "schema_version": SCHEMA_VERSION,
        "analysis_revision": ANALYSIS_REVISION,
        "upstream_recovery_protocol": RECOVERY_PROTOCOL,
        "logical_id": resolved["logical_id"],
        "source": {
            "manifest_path": resolved.get("manifest_path"),
            "manifest_sha256": resolved.get("manifest_sha256"),
            "centerline_path": resolved.get("centerline_path"),
            "centerline_integrity": resolved.get("integrity"),
            "status": resolved.get("status"),
            "selected_filament_id": observation["population"].get("selected_filament_id"),
        },
        "population": observation["population"],
        "growth": growth,
        "diameter": diameter,
        "shape": shape,
        "identification": {
            "growth_rate": {
                "estimate": growth.get("growth_rate"),
                "ci95": growth.get("growth_rate_ci95"),
                "status": "identified_from_eligible_contour_length" if growth.get("status") == "ok" else growth.get("status"),
            },
            "chi": {
                "estimate": selected_chi,
                "status": "conditional_on_selected_model_trajectory" if selected_chi is not None else "unidentifiable_from_single_centerline_without_parameterized_model_grid",
                "definition": "EI/(EA*L^2)",
            },
            "diameter_proxy": {
                "estimate": diameter.get("diameter_proxy"),
                "ci95": diameter.get("ci95"),
                "status": diameter.get("status"),
                "model_diameter": selected_model_diameter,
                "model_minus_observed": None if selected_model_diameter is None or diameter.get("diameter_proxy") is None else float(selected_model_diameter) - float(diameter["diameter_proxy"]),
            },
        },
        "limitations": [
            "censored frames (branch, loop, out-of-view, missing, reconnected, and quality flags) are excluded from fitted denominators",
            "confidence intervals quantify the declared regression residual model and do not include calibration or biological replicate uncertainty",
            "chi is not identifiable from one passive centerline time series without a parameterized model comparison or force/extension data",
            "diameter_proxy is a width-field proxy and is not evidence that the finite-radius contact law has been identified",
            "a successful shape metric is not a validation of the physical model or of a single biological lineage",
        ],
        "frame_rows": shape_rows,
        "observation_frame_rows": [
            {
                "frame": int(row["frame"]),
                "time": float(row["time"]),
                "length": row.get("length"),
                "length_px": row.get("length_px"),
                "width": row.get("width"),
                "quality": row.get("quality"),
                "eligible": int(bool(row.get("eligible"))),
                "censor": int(bool(row.get("censor"))),
                "exclusion_reason": row.get("exclusion_reason"),
            }
            for row in records
        ],
        "growth_residual_rows": growth.get("residuals", []),
    }
    return report


def _git_revision() -> str | None:
    try:
        completed = subprocess.run(["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True, timeout=2.0)
    except (OSError, subprocess.SubprocessError):
        return None
    return completed.stdout.strip() or None


def run_suite(
    sources: Sequence[str | Path],
    output: str | Path,
    *,
    data_root: str | Path | None = None,
    model_paths: Sequence[str | Path] = (),
    config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Analyse all sources and write compact JSON/CSV artifacts."""

    destination = Path(output).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    reports: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    for source in sources:
        try:
            reports.append(fit_observation(source, data_root=data_root, model_paths=model_paths, config=config))
        except (OSError, ValueError, VideoParameterFittingError, json.JSONDecodeError) as exc:
            errors.append({"source": str(Path(source).expanduser().resolve()), "error": f"{type(exc).__name__}: {exc}"})
    summary_rows: list[dict[str, Any]] = []
    artifact_rows: list[dict[str, Any]] = []
    for report in reports:
        logical_id = str(report["logical_id"])
        token = _safe_name(Path(logical_id).stem)
        frame_path = destination / f"{token}_frame_fits.csv"
        rows = [dict(row) for row in report.get("observation_frame_rows", [])]
        selected_candidate_id = report.get("shape", {}).get("selected_candidate_id")
        shape_by_frame = {
            int(row["frame"]): row
            for row in report.get("frame_rows", [])
            if selected_candidate_id is None or row.get("candidate_id") == selected_candidate_id
        }
        residual_by_frame = {int(row["frame"]): row for row in report.get("growth_residual_rows", [])}
        for row in rows:
            shape_row = shape_by_frame.get(int(row["frame"]), {})
            residual = residual_by_frame.get(int(row["frame"]), {})
            row.update({
                "candidate_id": shape_row.get("candidate_id"),
                "shape_status": shape_row.get("shape_status"),
                "shape_reason": shape_row.get("shape_reason"),
                "model_time": shape_row.get("model_time"),
                "time_error_s": shape_row.get("time_error_s"),
                "frechet_distance_px": shape_row.get("frechet_distance_px"),
                "endpoint_distance_px": shape_row.get("endpoint_distance_px"),
                "curvature_mse_px_inv2": shape_row.get("curvature_mse_px_inv2"),
                "growth_model": report["growth"].get("selected_model"),
                "observed_length": row.get("length"),
                "growth_residual": residual.get("residual"),
                "growth_predicted_length": residual.get("predicted_length"),
            })
        # A growth residual can exist only for an eligible length row, but retain
        # it even if an unusual input omitted that row from the population table.
        known_frames = {int(row["frame"]) for row in rows}
        for residual in report.get("growth_residual_rows", []):
            if int(residual["frame"]) not in known_frames:
                rows.append({
                    "frame": residual["frame"], "time": residual["time"],
                    "observed_length": residual.get("length"),
                    "growth_residual": residual["residual"],
                    "growth_predicted_length": residual["predicted_length"],
                    "growth_model": residual["selected_model"],
                })
        fields = [
            "frame", "time", "candidate_id", "eligible", "shape_status", "shape_reason",
            "model_time", "time_error_s", "frechet_distance_px", "endpoint_distance_px",
            "curvature_mse_px_inv2", "observed_length", "growth_model", "growth_predicted_length", "growth_residual",
        ]
        _write_csv(frame_path, rows, fields)
        report["artifacts"] = {"frame_fits": {"path": frame_path.name, "bytes": frame_path.stat().st_size, "sha256": sha256_file(frame_path)}}
        summary_rows.append({
            "logical_id": logical_id,
            "status": report["source"]["status"],
            "population_rows": report["population"].get("population_rows", 0),
            "eligible_rows": report["population"].get("eligible_rows", 0),
            "censored_rows": report["population"].get("censored_rows", 0),
            "growth_rate": report["growth"].get("growth_rate"),
            "growth_rate_ci95": canonical_json(report["growth"].get("growth_rate_ci95")),
            "growth_model": report["growth"].get("selected_model"),
            "diameter_proxy": report["diameter"].get("diameter_proxy"),
            "chi": report["identification"]["chi"].get("estimate"),
            "shape_status": report["shape"].get("status"),
            "shape_candidate": report["shape"].get("selected_candidate_id"),
            "frame_fits": frame_path.name,
        })
        artifact_rows.append({"logical_id": logical_id, **report["artifacts"]["frame_fits"]})
    config_value = dict(config or {})
    config_hash = sha256_bytes(canonical_json(config_value).encode("utf-8"))
    compact = {
        "schema_version": SCHEMA_VERSION,
        "analysis_revision": ANALYSIS_REVISION,
        "upstream_recovery_protocol": RECOVERY_PROTOCOL,
        "git_revision": _git_revision(),
        "config": config_value,
        "config_sha256": config_hash,
        "data_root": str(Path(data_root).expanduser().resolve()) if data_root is not None else None,
        "model_paths": [str(Path(path).expanduser().resolve()) for path in model_paths],
        "source_count": len(sources),
        "reports": reports,
        "errors": errors,
        "summary_rows": summary_rows,
        "claims_boundary": {
            "can_say": [
                "the declared eligible-frame population was fit deterministically",
                "growth estimates and residual confidence intervals are available when at least three eligible lengths exist",
                "shape residuals are available only with explicit pixel/model registration and a model centerline",
            ],
            "cannot_say": [
                "a recovered parameter is a true material constant without calibration and independent validation",
                "chi is identified from one trajectory without a parameterized candidate model comparison",
                "censored video frames are missing at random or may be treated as ordinary observations",
            ],
        },
    }
    summary_path = destination / "compact_summary.json"
    summary_csv_path = destination / "fit_summary.csv"
    manifest_path = destination / "reproducibility_manifest.json"
    _write_json(summary_path, compact)
    _write_csv(
        summary_csv_path,
        summary_rows,
        (
            "logical_id", "status", "population_rows", "eligible_rows", "censored_rows",
            "growth_rate", "growth_rate_ci95", "growth_model", "diameter_proxy", "chi",
            "shape_status", "shape_candidate", "frame_fits",
        ),
    )
    reproducibility = {
        "schema_version": SCHEMA_VERSION,
        "analysis_revision": ANALYSIS_REVISION,
        "upstream_recovery_protocol": RECOVERY_PROTOCOL,
        "config_sha256": config_hash,
        "sources": [
            {
                "source": str(Path(source).expanduser().resolve()),
                "sha256": sha256_file(source) if Path(source).is_file() else None,
            }
            for source in sources
        ],
        "model_paths": [
            {"path": str(Path(path).expanduser().resolve()), "sha256": sha256_file(path) if Path(path).is_file() else None}
            for path in model_paths
        ],
        "output_artifacts": {
            "compact_summary": {"path": summary_path.name, "bytes": summary_path.stat().st_size, "sha256": sha256_file(summary_path)},
            "fit_summary": {"path": summary_csv_path.name, "bytes": summary_csv_path.stat().st_size, "sha256": sha256_file(summary_csv_path)},
            "frame_fits": artifact_rows,
        },
        "determinism": "no timestamps, random seeds, or generated videos are written; same inputs/config produce the same JSON/CSV values",
    }
    _write_json(manifest_path, reproducibility)
    return compact


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", action="append", type=Path, dest="sources", help="PR #16 manifest; repeatable")
    parser.add_argument("--centerline", action="append", type=Path, dest="centerlines", help="centerline.csv; repeatable")
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--model", action="append", type=Path, default=[])
    parser.add_argument("--output", type=Path, default=Path("continuum_filament_model/results/video_parameter_fitting"))
    parser.add_argument("--filament-id")
    parser.add_argument("--quality-threshold", type=float, default=0.2)
    parser.add_argument("--coordinate-scale", type=float, default=1.0, help="length units per observed pixel")
    parser.add_argument("--width-scale", type=float, help="width units per width-field unit; defaults to coordinate-scale")
    parser.add_argument("--min-points", type=int, default=3)
    parser.add_argument("--pixel-per-model-unit", type=float)
    parser.add_argument("--x-offset-px", type=float, default=0.0)
    parser.add_argument("--y-offset-px", type=float, default=0.0)
    parser.add_argument("--rotation-deg", type=float, default=0.0)
    parser.add_argument("--time-scale", type=float, default=1.0)
    parser.add_argument("--time-offset", type=float, default=0.0)
    parser.add_argument("--max-time-error-s", type=float, default=0.20)
    parser.add_argument("--endpoint-order", choices=("auto", "forward", "reverse"), default="auto")
    args = parser.parse_args(argv)
    sources = list(args.sources or []) + list(args.centerlines or [])
    if not sources:
        sources = [
            Path("continuum_filament_model/results/video_comparison/gray5_manifest.json"),
            Path("continuum_filament_model/results/video_comparison/original_manifest.json"),
        ]
    config = {
        "filament_id": args.filament_id,
        "quality_threshold": args.quality_threshold,
        "coordinate_scale": args.coordinate_scale,
        "width_scale": args.width_scale,
        "min_points": args.min_points,
        "pixel_per_model_unit": args.pixel_per_model_unit,
        "x_offset_px": args.x_offset_px,
        "y_offset_px": args.y_offset_px,
        "rotation_deg": args.rotation_deg,
        "time_scale": args.time_scale,
        "time_offset": args.time_offset,
        "max_time_error_s": args.max_time_error_s,
        "endpoint_order": args.endpoint_order,
    }
    compact = run_suite(sources, args.output, data_root=args.data_root, model_paths=args.model, config=config)
    print(json.dumps({"output": str(Path(args.output).resolve()), "reports": len(compact["reports"]), "errors": len(compact["errors"])}, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised by the documented CLI
    raise SystemExit(main())
