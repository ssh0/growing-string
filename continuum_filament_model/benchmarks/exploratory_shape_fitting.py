"""探索的な高解像度フィラメント形状フィッティング。

このスクリプトは、抽出済みの ``gray5`` 中心線を力学モデルへ厳密に同定する
ものではない。観察中心線を滑らかな初期形状として用い、節点数を増やした
過減衰・成長フィラメントを ``G_b``、``chi``、モデル成長時間の grid で走らせ、
同じ画像座標系へ登録した形状を Fréchet 距離と曲率プロファイル RMSE で比較する。
結果は、数値的な探索範囲と censor 状態を含む探索的な比較資料として出力する。

リポジトリルートからの実行例::

    PYTHONPATH=continuum_filament_model/src \
      python continuum_filament_model/benchmarks/exploratory_shape_fitting.py \
      --output continuum_filament_model/results/presentation_data/exploratory_fitting

観察中心線がライセンス等の理由で存在しない場合は、観察データを補完せず、
``FileNotFoundError`` で停止する。
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

if __package__ in {None, ""}:
    _HERE = Path(__file__).resolve()
    _SRC = _HERE.parents[1] / "src"
    if str(_SRC) not in sys.path:
        sys.path.insert(0, str(_SRC))

from growing_filament.model import (  # noqa: E402
    FilamentState,
    ModelParameters,
    OverdampedGrowingFilament,
)

SCHEMA_VERSION = "continuum-filament-exploratory-shape-fitting-2"
DEFAULT_CENTERLINE = Path("continuum_filament_model/results/presentation_data/video_gray5/centerline.csv")
DEFAULT_OUTPUT = Path("continuum_filament_model/results/presentation_data/exploratory_fitting")
DEFAULT_TARGET_TIMES = (1.0, 3.0)
DEFAULT_GB_GRID = (0.2, 0.775, 1.35, 1.925, 2.5)
DEFAULT_CHI_GRID = (0.0001, 0.0003, 0.0006, 0.001)
DEFAULT_GROWTH_TIMES = (0.02, 0.05)
DEFAULT_RESOLUTION_VALUES = (25, 30, 35)
FEATURE_NAMES = (
    "deflection_ratio",
    "slack_ratio",
    "curvature_mean_abs_L",
    "curvature_rms_L",
    "curvature_max_L",
    "mode_1_abs",
    "mode_2_over_mode_1",
    "mode_3_over_mode_1",
)
FEATURE_LABELS = (
    "Amax/chord",
    "L/chord - 1",
    "mean(|k|)L",
    "RMS(k)L",
    "max(|k|)L",
    "mode 1",
    "mode 2/mode 1",
    "mode 3/mode 1",
)
# Scales keep the dimensionless feature residuals comparable without allowing
# the pixel-coordinate residual to dominate the shape objective.
FEATURE_SCALES = {
    "deflection_ratio": 0.25,
    "slack_ratio": 0.25,
    "curvature_mean_abs_L": 1.0,
    "curvature_rms_L": 1.0,
    "curvature_max_L": 2.0,
    "mode_1_abs": 0.25,
    "mode_2_over_mode_1": 1.0,
    "mode_3_over_mode_1": 1.0,
}


@dataclass(frozen=True)
class ObservationFrame:
    frame: int
    time: float
    filament_id: str
    points: np.ndarray
    censor: bool
    quality: float
    quality_flags: str


@dataclass
class CandidateResult:
    G_b: float
    chi: float
    growth_time: float
    score: float
    status: str
    frame_metrics: list[dict[str, Any]]
    temporal_features: dict[str, Any]
    trajectory: list[FilamentState]
    failure_reason: str | None = None


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


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_jsonable(value), ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _logical_path(path: Path) -> str:
    try:
        return str(path.relative_to(Path.cwd()))
    except ValueError:
        return str(path)


def _read_centerline(path: Path) -> list[ObservationFrame]:
    if not path.is_file():
        raise FileNotFoundError(path)
    grouped: dict[tuple[int, str], list[dict[str, str]]] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            try:
                key = (int(row["frame"]), str(row["filament_id"]))
                point = (float(row["x"]), float(row["y"]))
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"invalid centerline row in {path}: {row}") from exc
            if not np.isfinite(point).all():
                continue
            row = dict(row)
            row["_point"] = point  # type: ignore[assignment]
            grouped.setdefault(key, []).append(row)
    frames: list[ObservationFrame] = []
    for (frame, filament_id), rows in sorted(grouped.items()):
        rows.sort(key=lambda row: int(row.get("point_id", 0)))
        if len(rows) < 3:
            continue
        points = np.asarray([row["_point"] for row in rows], dtype=float)
        flags = ";".join(sorted({flag for row in rows for flag in str(row.get("quality_flags", "")).split(";") if flag and flag != "ok"}))
        try:
            time = float(rows[0]["time"])
            quality = float(np.mean([float(row.get("quality", 1.0)) for row in rows]))
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"invalid centerline metadata in {path}: frame={frame}") from exc
        censor = any(str(row.get("censor", "0")).strip().lower() in {"1", "true", "yes"} for row in rows)
        frames.append(ObservationFrame(frame, time, filament_id, points, censor, quality, flags or "ok"))
    if not frames:
        raise ValueError(f"no centerline with at least three points found: {path}")
    return frames


def select_observations(
    frames: Sequence[ObservationFrame],
    target_times: Sequence[float] = DEFAULT_TARGET_TIMES,
    *,
    filament_id: str | None = None,
    max_time_error: float = 0.51,
) -> list[ObservationFrame]:
    """Select nearest representative frames without silently dropping censor flags."""

    if not target_times:
        raise ValueError("target_times must not be empty")
    candidates = [item for item in frames if filament_id is None or item.filament_id == filament_id]
    if not candidates:
        raise ValueError(f"filament_id not found: {filament_id}")
    selected: list[ObservationFrame] = []
    selected_keys: set[tuple[int, str]] = set()
    for target in target_times:
        if not math.isfinite(float(target)):
            raise ValueError("target_times must be finite")
        item = min(candidates, key=lambda value: abs(value.time - float(target)))
        if abs(item.time - float(target)) > max_time_error:
            raise ValueError(f"no observation within {max_time_error}s of t={target}")
        key = (item.frame, item.filament_id)
        if key not in selected_keys:
            selected.append(item)
            selected_keys.add(key)
    selected.sort(key=lambda item: item.time)
    return selected


def resample_polyline(points: np.ndarray, n: int = 100) -> np.ndarray:
    """Resample an ordered polyline uniformly by arc length."""

    values = np.asarray(points, dtype=float)
    if values.ndim != 2 or values.shape[1] != 2 or len(values) < 2:
        raise ValueError("points must have shape (N, 2), N>=2")
    if n < 2:
        raise ValueError("n must be at least 2")
    distance = np.concatenate(([0.0], np.cumsum(np.linalg.norm(np.diff(values, axis=0), axis=1))))
    if distance[-1] <= 1.0e-12:
        raise ValueError("polyline has zero contour length")
    target = np.linspace(0.0, float(distance[-1]), n)
    return np.column_stack([np.interp(target, distance, values[:, axis]) for axis in range(2)])


def discrete_frechet(first: np.ndarray, second: np.ndarray) -> float:
    """Return the discrete Fréchet distance for two ordered polylines."""

    left, right = np.asarray(first, dtype=float), np.asarray(second, dtype=float)
    if left.ndim != 2 or right.ndim != 2 or left.shape[1:] != right.shape[1:] or len(left) == 0 or len(right) == 0:
        raise ValueError("both polylines must be non-empty arrays with shape (N, 2)")
    distances = np.linalg.norm(left[:, None, :] - right[None, :, :], axis=2)
    dynamic = np.empty_like(distances)
    dynamic[0, 0] = distances[0, 0]
    for i in range(1, len(left)):
        dynamic[i, 0] = max(dynamic[i - 1, 0], distances[i, 0])
    for j in range(1, len(right)):
        dynamic[0, j] = max(dynamic[0, j - 1], distances[0, j])
    for i in range(1, len(left)):
        for j in range(1, len(right)):
            dynamic[i, j] = max(
                distances[i, j],
                min(dynamic[i - 1, j], dynamic[i - 1, j - 1], dynamic[i, j - 1]),
            )
    return float(dynamic[-1, -1])


def curvature_profile(points: np.ndarray, n: int = 100) -> np.ndarray:
    """Return signed discrete curvature on an arc-length-resampled polyline."""

    sampled = resample_polyline(points, n)
    first = sampled[1:-1] - sampled[:-2]
    second = sampled[2:] - sampled[1:-1]
    first_length = np.linalg.norm(first, axis=1)
    second_length = np.linalg.norm(second, axis=1)
    if np.any(first_length <= 1.0e-12) or np.any(second_length <= 1.0e-12):
        raise ValueError("curvature is undefined for a zero-length local segment")
    cross = first[:, 0] * second[:, 1] - first[:, 1] * second[:, 0]
    dot = np.sum(first * second, axis=1)
    angle = np.arctan2(cross, dot)
    return angle / (0.5 * (first_length + second_length))


def curvature_rmse(first: np.ndarray, second: np.ndarray, n: int = 100) -> float:
    """Compare signed curvature profiles after independent arc-length sampling."""

    left, right = curvature_profile(first, n), curvature_profile(second, n)
    return float(np.sqrt(np.mean((left - right) ** 2)))


def shape_features(points: np.ndarray) -> dict[str, float]:
    """Extract dimensionless shape features that are robust to translation and scale.

    The feature vector intentionally describes morphology rather than pixel-wise
    correspondence: endpoint-normalised deflection, contour slack, curvature
    statistics multiplied by contour length, and low-order transverse modes.
    """

    # Feature extraction uses a light arc-length smoothing pass so one-pixel
    # skeleton stair steps do not dominate curvature statistics.
    values = _smooth_polyline(_rotate_to_chord(resample_polyline(points, 160)), passes=2)
    contour = _contour_length(values)
    chord = float(np.linalg.norm(values[-1] - values[0]))
    if contour <= 1.0e-12 or chord <= 1.0e-12:
        raise ValueError("shape features require a non-degenerate polyline")
    curvature = curvature_profile(values, n=100)
    transverse = values[:, 1]
    u = np.linspace(0.0, 1.0, len(transverse))
    transverse_normalised = transverse / contour
    integrate = getattr(np, "trapezoid", np.trapz)
    modes = np.asarray(
        [2.0 * integrate(transverse_normalised * np.sin(mode * np.pi * u), u) for mode in (1, 2, 3)],
        dtype=float,
    )
    # A finite floor avoids unstable mode ratios when the first mode is
    # numerically absent.  Such a candidate is still visible in the feature
    # residual rather than producing inf/NaN.
    mode_1 = max(abs(float(modes[0])), 1.0e-6)
    return {
        "deflection_ratio": float(np.max(np.abs(transverse)) / chord),
        "slack_ratio": float(contour / chord - 1.0),
        "curvature_mean_abs_L": float(np.mean(np.abs(curvature)) * contour),
        "curvature_rms_L": float(np.sqrt(np.mean(curvature * curvature)) * contour),
        "curvature_max_L": float(np.max(np.abs(curvature)) * contour),
        "mode_1_abs": float(abs(modes[0])),
        "mode_2_over_mode_1": float(abs(modes[1]) / mode_1),
        "mode_3_over_mode_1": float(abs(modes[2]) / mode_1),
    }


def weighted_feature_loss(observed: Mapping[str, float], model: Mapping[str, float]) -> tuple[float, dict[str, float]]:
    """Return mean squared residual in the declared dimensionless feature scales."""

    residuals: dict[str, float] = {}
    for name in FEATURE_NAMES:
        scale = float(FEATURE_SCALES[name])
        residuals[name] = (float(model[name]) - float(observed[name])) / scale
    return float(np.mean(np.asarray(list(residuals.values()), dtype=float) ** 2)), residuals


def _growth_rate(times: Sequence[float], lengths: Sequence[float]) -> float | None:
    """Estimate log contour-length growth rate from at least two time points."""

    if len(times) < 2 or len(lengths) < 2:
        return None
    time_values = np.asarray(times, dtype=float)
    length_values = np.asarray(lengths, dtype=float)
    valid = np.isfinite(time_values) & np.isfinite(length_values) & (length_values > 0.0)
    if np.count_nonzero(valid) < 2 or np.ptp(time_values[valid]) <= 1.0e-12:
        return None
    slope = np.polyfit(time_values[valid], np.log(length_values[valid]), 1)[0]
    return float(slope) if math.isfinite(float(slope)) else None


def _contour_length(points: np.ndarray) -> float:
    return float(np.sum(np.linalg.norm(np.diff(np.asarray(points, dtype=float), axis=0), axis=1)))


def _rotate_to_chord(points: np.ndarray) -> np.ndarray:
    values = np.asarray(points, dtype=float) - np.asarray(points[0], dtype=float)
    chord = values[-1]
    if np.linalg.norm(chord) <= 1.0e-12:
        raise ValueError("polyline endpoints must be distinct")
    angle = math.atan2(float(chord[1]), float(chord[0]))
    rotation = np.asarray([[math.cos(-angle), -math.sin(-angle)], [math.sin(-angle), math.cos(-angle)]])
    return values @ rotation.T


def _smooth_polyline(points: np.ndarray, passes: int = 2) -> np.ndarray:
    """Remove pixel stair-step noise while preserving both endpoints."""

    values = np.asarray(points, dtype=float).copy()
    kernel = np.asarray([1.0, 4.0, 6.0, 4.0, 1.0]) / 16.0
    for _ in range(max(0, passes)):
        padded = np.pad(values, ((2, 2), (0, 0)), mode="edge")
        filtered = np.column_stack([np.convolve(padded[:, axis], kernel, mode="valid") for axis in range(2)])
        filtered[0] = values[0]
        filtered[-1] = values[-1]
        values = filtered
    return values


def canonical_observation(points: np.ndarray, n_nodes: int, model_length: float = 2.0) -> np.ndarray:
    """Create a smooth, endpoint-oriented model initial shape."""

    if model_length <= 0.0:
        raise ValueError("model_length must be positive")
    dense = _rotate_to_chord(resample_polyline(points, max(4 * n_nodes, 80)))
    dense = _smooth_polyline(dense)
    dense = dense / _contour_length(dense) * model_length
    sampled = resample_polyline(dense, n_nodes)
    sampled = sampled / _contour_length(sampled) * model_length
    return sampled


def _register_model_to_observation(model_points: np.ndarray, observed_points: np.ndarray) -> np.ndarray:
    """Use one global scale and endpoint direction; do not warp the model."""

    model = np.asarray(model_points, dtype=float)
    observed = np.asarray(observed_points, dtype=float)
    model_length = _contour_length(model)
    observed_length = _contour_length(observed)
    if model_length <= 1.0e-12 or observed_length <= 1.0e-12:
        raise ValueError("cannot register a zero-length polyline")
    model_origin = model[0]
    observed_origin = observed[0]
    model_chord = model[-1] - model_origin
    observed_chord = observed[-1] - observed_origin
    if np.linalg.norm(model_chord) <= 1.0e-12 or np.linalg.norm(observed_chord) <= 1.0e-12:
        raise ValueError("cannot register a polyline with a zero chord")
    angle = math.atan2(float(observed_chord[1]), float(observed_chord[0])) - math.atan2(float(model_chord[1]), float(model_chord[0]))
    rotation = np.asarray([[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]])
    return (model - model_origin) @ rotation.T * (observed_length / model_length) + observed_origin


def compare_shape(model_points: np.ndarray, observed_points: np.ndarray) -> dict[str, Any]:
    """Return registered geometry, Fréchet distance, curvature RMSE, and loss."""

    observed = np.asarray(observed_points, dtype=float)
    forward = _register_model_to_observation(model_points, observed)
    reverse = _register_model_to_observation(np.asarray(model_points)[::-1], observed)

    def evaluate(registered: np.ndarray) -> tuple[float, float, float]:
        observed_sample = resample_polyline(observed, 100)
        model_sample = resample_polyline(registered, 100)
        frechet = discrete_frechet(observed_sample, model_sample)
        curvature = curvature_rmse(observed_sample, model_sample, n=100)
        length = max(_contour_length(observed), 1.0e-12)
        return frechet, curvature, frechet / length + curvature * length

    forward_metrics = evaluate(forward)
    reverse_metrics = evaluate(reverse)
    registered, values = (forward, forward_metrics) if forward_metrics[2] <= reverse_metrics[2] else (reverse, reverse_metrics)
    observed_features = shape_features(observed)
    model_features = shape_features(registered)
    feature_loss, feature_residuals = weighted_feature_loss(observed_features, model_features)
    return {
        "model_points_registered": registered,
        "frechet_distance_px": float(values[0]),
        "curvature_rmse_px_inv": float(values[1]),
        "normalized_shape_loss": float(values[2]),
        "feature_loss": feature_loss,
        "observed_features": observed_features,
        "model_features": model_features,
        "feature_residuals": feature_residuals,
        "observed_length_px": _contour_length(observed),
        "selected_orientation": "forward" if values is forward_metrics else "reverse",
    }


def _make_parameters(
    *,
    G_b: float,
    chi: float,
    model_length: float,
    reference_lengths: np.ndarray,
    dt: float,
    t_end: float,
    max_retries: int,
) -> ModelParameters:
    if G_b <= 0.0 or chi <= 0.0:
        raise ValueError("G_b and chi must be positive")
    # EA=1 is a numerically gentle nondimensionalisation.  chi remains exactly
    # EI/(EA L^2), and g is selected from the documented G_b definition.
    axial_stiffness = 1.0
    drag_density = 1.0
    bending_stiffness = chi * axial_stiffness * model_length**2
    tau_b = drag_density * model_length**4 / (bending_stiffness * math.pi**4)
    growth_rate = G_b / tau_b
    return ModelParameters(
        axial_stiffness=axial_stiffness,
        bending_stiffness=bending_stiffness,
        drag_density=drag_density,
        growth_rate=growth_rate,
        reference_length=float(np.mean(reference_lengths)),
        dt=dt,
        t_end=max(t_end, 0.0),
        a_max=4.0 * float(np.mean(reference_lengths)),
        max_retries=max_retries,
        fixed_left=True,
        fixed_right=True,
        reject_crossing=True,
    )


def _simulate_candidate(
    initial_state: FilamentState,
    *,
    G_b: float,
    chi: float,
    model_length: float,
    max_time: float,
    dt: float,
    max_retries: int,
) -> list[FilamentState]:
    if max_time <= 0.0:
        return [initial_state.copy()]
    params = _make_parameters(
        G_b=G_b,
        chi=chi,
        model_length=model_length,
        reference_lengths=initial_state.rest_lengths,
        dt=dt,
        t_end=max_time,
        max_retries=max_retries,
    )
    simulator = OverdampedGrowingFilament(initial_state.copy(), params)
    return simulator.run(max_time)


def _state_at_time(trajectory: Sequence[FilamentState], target: float) -> FilamentState:
    if not trajectory:
        raise ValueError("empty trajectory")
    return min(trajectory, key=lambda state: abs(float(state.time) - float(target))).copy()


def _grid_values(values: Iterable[float], *, minimum: float, maximum: float) -> tuple[float, ...]:
    result = sorted({float(value) for value in values if minimum <= float(value) <= maximum})
    if not result:
        raise ValueError("search grid is empty")
    return tuple(result)


def _refined_values(best: float, *, minimum: float, maximum: float, logarithmic: bool = False) -> tuple[float, ...]:
    if logarithmic:
        values = np.asarray([best / math.sqrt(2.0), best, best * math.sqrt(2.0)], dtype=float)
    else:
        delta = max(abs(best) * 0.25, 0.01)
        values = np.asarray([best - delta, best, best + delta], dtype=float)
    return _grid_values(values, minimum=minimum, maximum=maximum)


def _evaluate_candidate(
    initial_state: FilamentState,
    observations: Sequence[ObservationFrame],
    *,
    G_b: float,
    chi: float,
    growth_time: float,
    model_length: float,
    dt: float,
    max_retries: int,
) -> CandidateResult:
    try:
        trajectory = _simulate_candidate(
            initial_state,
            G_b=G_b,
            chi=chi,
            model_length=model_length,
            max_time=max(float(growth_time), 0.0),
            dt=dt,
            max_retries=max_retries,
        )
    except (RuntimeError, ValueError, FloatingPointError) as exc:
        return CandidateResult(
            G_b, chi, growth_time, float("inf"), "simulation_failed", [], {}, [],
            f"{type(exc).__name__}: {exc}",
        )

    anchor_time = observations[0].time
    final_elapsed = max(observations[-1].time - anchor_time, 1.0e-12)
    frame_metrics: list[dict[str, Any]] = []
    fit_losses: list[float] = []
    geometric_losses: list[float] = []
    observed_lengths: list[float] = []
    model_lengths: list[float] = []
    for index, observation in enumerate(observations):
        fraction = max(0.0, float(observation.time - anchor_time) / final_elapsed)
        requested_model_time = float(growth_time) * fraction
        state = _state_at_time(trajectory, requested_model_time)
        comparison = compare_shape(state.positions, observation.points)
        observed_lengths.append(_contour_length(observation.points))
        model_lengths.append(_contour_length(state.positions))
        metric = {
            "frame": observation.frame,
            "time": observation.time,
            "filament_id": observation.filament_id,
            "censor": observation.censor,
            "quality": observation.quality,
            "quality_flags": observation.quality_flags,
            "requested_model_time": requested_model_time,
            "model_time": float(state.time),
            "n_model_nodes": state.n_nodes,
            "frechet_distance_px": comparison["frechet_distance_px"],
            "curvature_rmse_px_inv": comparison["curvature_rmse_px_inv"],
            "normalized_shape_loss": comparison["normalized_shape_loss"],
            "feature_loss": comparison["feature_loss"],
            "observed_features": comparison["observed_features"],
            "model_features": comparison["model_features"],
            "feature_residuals": comparison["feature_residuals"],
            "selected_orientation": comparison["selected_orientation"],
        }
        frame_metrics.append(metric)
        if index > 0 or len(observations) == 1:
            fit_losses.append(float(comparison["feature_loss"]))
            geometric_losses.append(float(comparison["frechet_distance_px"]) / max(float(comparison["observed_length_px"]), 1.0e-12))

    observed_growth = _growth_rate([item.time for item in observations], observed_lengths)
    model_growth = _growth_rate([item.time for item in observations], model_lengths)
    temporal_loss = 0.0
    temporal_residual = None
    if observed_growth is not None and model_growth is not None:
        growth_scale = max(abs(observed_growth), 0.1)
        temporal_residual = (model_growth - observed_growth) / growth_scale
        temporal_loss = float(temporal_residual * temporal_residual)
    temporal_features = {
        "observed_growth_rate": observed_growth,
        "model_growth_rate": model_growth,
        "normalised_growth_rate_residual": temporal_residual,
        "growth_rate_loss": temporal_loss,
        "time_unit": "selected observation time",
    }
    # Dimensionless feature morphology is primary.  A bounded Fréchet term
    # remains in the objective so a feature-only winner cannot visibly miss the
    # ordered centerline; the raw pixel scale is removed by observed length.
    visual_weight = 50.0
    score = (
        float(np.median(fit_losses) + temporal_loss + visual_weight * np.median(geometric_losses))
        if fit_losses else float("inf")
    )
    return CandidateResult(G_b, chi, growth_time, score, "computed", frame_metrics, temporal_features, trajectory)


def _candidate_row(result: CandidateResult, phase: str) -> dict[str, Any]:
    last = result.frame_metrics[-1] if result.frame_metrics else {}
    return {
        "phase": phase,
        "G_b": result.G_b,
        "chi": result.chi,
        "growth_time": result.growth_time,
        "status": result.status,
        "score": result.score,
        "target_frechet_distance_px": last.get("frechet_distance_px"),
        "target_curvature_rmse_px_inv": last.get("curvature_rmse_px_inv"),
        "target_feature_loss": last.get("feature_loss"),
        "observed_growth_rate": result.temporal_features.get("observed_growth_rate"),
        "model_growth_rate": result.temporal_features.get("model_growth_rate"),
        "growth_rate_loss": result.temporal_features.get("growth_rate_loss"),
        "failure_reason": result.failure_reason,
    }


def _write_candidate_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fields = [
        "phase", "G_b", "chi", "growth_time", "status", "score",
        "target_frechet_distance_px", "target_curvature_rmse_px_inv", "target_feature_loss",
        "observed_growth_rate", "model_growth_rate", "growth_rate_loss", "failure_reason",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(_jsonable(row) for row in rows)


def _write_resolution_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fields = [
        "n_nodes", "status", "G_b", "chi", "growth_time", "score",
        "frechet_distance_px", "curvature_rmse_px_inv", "feature_loss",
        "observed_growth_rate", "model_growth_rate", "failure_reason",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(_jsonable(row) for row in rows)


def _write_trajectory(path: Path, trajectory: Sequence[FilamentState], metadata: Mapping[str, Any]) -> None:
    positions: list[np.ndarray] = []
    rest_lengths: list[np.ndarray] = []
    position_offsets = [0]
    rest_offsets = [0]
    for state in trajectory:
        positions.append(np.asarray(state.positions, dtype=float))
        rest_lengths.append(np.asarray(state.rest_lengths, dtype=float))
        position_offsets.append(position_offsets[-1] + len(state.positions))
        rest_offsets.append(rest_offsets[-1] + len(state.rest_lengths))
    np.savez_compressed(
        path,
        times=np.asarray([state.time for state in trajectory], dtype=float),
        steps=np.asarray([state.step for state in trajectory], dtype=int),
        positions=np.concatenate(positions, axis=0),
        position_offsets=np.asarray(position_offsets, dtype=int),
        rest_lengths=np.concatenate(rest_lengths, axis=0),
        rest_offsets=np.asarray(rest_offsets, dtype=int),
        metadata_json=np.asarray(json.dumps(_jsonable(metadata), sort_keys=True)),
    )


def _write_plot(
    path: Path,
    observations: Sequence[ObservationFrame],
    best: CandidateResult,
    baseline: Mapping[str, Any],
    candidate_rows: Sequence[Mapping[str, Any]],
) -> None:
    import matplotlib.pyplot as plt

    target_index = len(observations) - 1
    target_observation = observations[target_index]
    censor_label = "CENSORED exploratory overlay" if all(item.censor for item in observations) else "exploratory overlay"
    target_metric = best.frame_metrics[target_index]
    target_state = _state_at_time(best.trajectory, float(target_metric["requested_model_time"]))
    target_comparison = compare_shape(target_state.positions, target_observation.points)
    target_registered = target_comparison["model_points_registered"]
    seed_observation = observations[0]
    seed_comparison = compare_shape(best.trajectory[0].positions, seed_observation.points)
    seed_registered = seed_comparison["model_points_registered"]

    figure, axes = plt.subplots(3, 2, figsize=(14, 14), constrained_layout=True)
    axes[0, 0].plot(seed_observation.points[:, 0], seed_observation.points[:, 1], "-", color="tab:blue", linewidth=2.5, label="observation seed")
    axes[0, 0].plot(seed_registered[:, 0], seed_registered[:, 1], "--", color="tab:orange", linewidth=2.0, label="model seed")
    axes[0, 0].set_title(f"causal seed: t={seed_observation.time:g}s / n={best.trajectory[0].n_nodes}")
    axes[0, 1].plot(target_observation.points[:, 0], target_observation.points[:, 1], "-", color="tab:blue", linewidth=2.5, label="observation")
    axes[0, 1].plot(target_registered[:, 0], target_registered[:, 1], "--", color="tab:orange", linewidth=2.2, label="best model")
    axes[0, 1].set_title(
        f"shape: target t={target_observation.time:g}s / G_b={best.G_b:.4g}, chi={best.chi:.4g}\n"
        f"Frechet={target_metric['frechet_distance_px']:.3g}px, feature loss={target_metric['feature_loss']:.3g}"
    )
    for axis in axes[0]:
        axis.set_aspect("equal", adjustable="datalim")
        axis.set_xlabel("x [pixel]")
        axis.set_ylabel("y [pixel]")
        axis.grid(alpha=0.22)
        axis.legend(fontsize=8)

    observed_features = target_metric["observed_features"]
    model_features = target_metric["model_features"]
    feature_x = np.arange(len(FEATURE_NAMES))
    width = 0.38
    axes[1, 0].bar(feature_x - width / 2, [observed_features[name] for name in FEATURE_NAMES], width, label="observation", color="tab:blue")
    axes[1, 0].bar(feature_x + width / 2, [model_features[name] for name in FEATURE_NAMES], width, label="model", color="tab:orange")
    axes[1, 0].set_xticks(feature_x, FEATURE_LABELS, rotation=35, ha="right", fontsize=8)
    axes[1, 0].set_title("dimensionless feature vector at target")
    axes[1, 0].set_ylabel("feature value")
    axes[1, 0].grid(axis="y", alpha=0.22)
    axes[1, 0].legend(fontsize=8)

    frame_times = [float(metric["time"]) for metric in best.frame_metrics]
    for name, color in (("deflection_ratio", "tab:purple"), ("slack_ratio", "tab:green"), ("curvature_rms_L", "tab:red")):
        axes[1, 1].plot(frame_times, [metric["observed_features"][name] for metric in best.frame_metrics], "o-", color=color, label=f"obs {name}")
        axes[1, 1].plot(frame_times, [metric["model_features"][name] for metric in best.frame_metrics], "--", color=color, alpha=0.7, label=f"model {name}")
    axes[1, 1].set_title("feature time development")
    axes[1, 1].set_xlabel("observation time [s]")
    axes[1, 1].set_ylabel("dimensionless feature")
    axes[1, 1].grid(alpha=0.22)
    axes[1, 1].legend(fontsize=7, ncol=2)

    observed_curvature = curvature_profile(target_observation.points, n=100)
    model_curvature = curvature_profile(target_registered, n=100)
    axes[2, 0].plot(np.linspace(0.0, 1.0, len(observed_curvature)), observed_curvature, color="tab:blue", label="observation")
    axes[2, 0].plot(np.linspace(0.0, 1.0, len(model_curvature)), model_curvature, "--", color="tab:orange", label="model")
    axes[2, 0].set_title("target curvature profile")
    axes[2, 0].set_xlabel("normalized arc length")
    axes[2, 0].set_ylabel(r"signed $\kappa$ [pixel$^{-1}$]")
    axes[2, 0].grid(alpha=0.22)
    axes[2, 0].legend(fontsize=8)

    computed = [row for row in candidate_rows if row.get("status") == "computed" and math.isfinite(float(row["score"]))]
    scatter = axes[2, 1].scatter(
        [float(row["G_b"]) for row in computed],
        [float(row["chi"]) for row in computed],
        c=[float(row["score"]) for row in computed],
        cmap="viridis",
        s=55,
        edgecolors="black",
        linewidths=0.25,
    )
    axes[2, 1].scatter([best.G_b], [best.chi], marker="*", s=220, color="red", edgecolors="black", label="best")
    axes[2, 1].set_yscale("log")
    axes[2, 1].set_xlabel(r"$G_b$")
    axes[2, 1].set_ylabel(r"$\chi$")
    axes[2, 1].set_title(f"feature loss search / straight baseline={baseline['feature_loss']:.3g}")
    axes[2, 1].grid(alpha=0.22)
    axes[2, 1].legend(fontsize=8)
    figure.colorbar(scatter, ax=axes[2, 1], label="weighted feature loss")
    figure.suptitle(f"gray5 {censor_label}: robust features and continuum model", fontsize=14)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def run_exploratory_fit(
    centerline_path: str | Path = DEFAULT_CENTERLINE,
    output_dir: str | Path = DEFAULT_OUTPUT,
    *,
    target_times: Sequence[float] = DEFAULT_TARGET_TIMES,
    filament_id: str | None = None,
    n_nodes: int = 25,
    resolution_values: Sequence[int] = DEFAULT_RESOLUTION_VALUES,
    model_length: float = 2.0,
    coarse_gb: Sequence[float] = DEFAULT_GB_GRID,
    coarse_chi: Sequence[float] = DEFAULT_CHI_GRID,
    growth_times: Sequence[float] = DEFAULT_GROWTH_TIMES,
    refine: bool = True,
    dt: float = 0.01,
    max_retries: int = 4,
    write_plot: bool = True,
) -> dict[str, Any]:
    """Run coarse/refined ``G_b``–``chi``–growth-time exploration and persist artifacts."""

    if not 5 <= int(n_nodes) <= 100:
        raise ValueError("n_nodes must be between 5 and 100")
    resolution_grid = tuple(sorted({int(value) for value in resolution_values}))
    if not resolution_grid or any(value < 5 or value > 100 for value in resolution_grid):
        raise ValueError("resolution_values must contain integers between 5 and 100")
    if dt <= 0.0:
        raise ValueError("dt must be positive")
    centerline = Path(centerline_path).expanduser().resolve()
    output = Path(output_dir).expanduser().resolve()
    all_frames = _read_centerline(centerline)
    observations = select_observations(all_frames, target_times, filament_id=filament_id)
    if len(observations) > 1 and observations[-1].time <= observations[0].time:
        raise ValueError("selected observation times must increase")

    # Use the earliest selected representative as the causal initial state;
    # later observations are evaluated only after forward model evolution.
    initial_shape_frame = observations[0]
    initial_positions = canonical_observation(initial_shape_frame.points, int(n_nodes), model_length)
    initial_state = FilamentState(initial_positions, np.linalg.norm(np.diff(initial_positions, axis=0), axis=1))
    gb_values = _grid_values(coarse_gb, minimum=0.2, maximum=2.5)
    chi_values = _grid_values(coarse_chi, minimum=1.0e-6, maximum=1.0)
    time_values = _grid_values(growth_times, minimum=0.0, maximum=10.0)
    candidate_results: list[CandidateResult] = []
    candidate_rows: list[dict[str, Any]] = []

    def explore(gb_grid: Sequence[float], chi_grid: Sequence[float], time_grid: Sequence[float], phase: str) -> None:
        for gb in gb_grid:
            for chi in chi_grid:
                for growth_time in time_grid:
                    result = _evaluate_candidate(
                        initial_state,
                        observations,
                        G_b=float(gb),
                        chi=float(chi),
                        growth_time=float(growth_time),
                        model_length=model_length,
                        dt=dt,
                        max_retries=max_retries,
                    )
                    candidate_results.append(result)
                    candidate_rows.append(_candidate_row(result, phase))

    explore(gb_values, chi_values, time_values, "coarse")
    valid = [result for result in candidate_results if result.status == "computed" and math.isfinite(result.score)]
    if not valid:
        raise RuntimeError("all exploratory shape-fitting candidates failed")
    best = min(valid, key=lambda result: result.score)
    refinement_spec: dict[str, Any] | None = None
    if refine:
        refined_gb = _refined_values(best.G_b, minimum=0.2, maximum=2.5)
        refined_chi = _refined_values(best.chi, minimum=1.0e-6, maximum=1.0, logarithmic=True)
        refined_time = _refined_values(best.growth_time, minimum=0.0, maximum=10.0)
        refinement_spec = {"G_b": refined_gb, "chi": refined_chi, "growth_time": refined_time}
        explore(refined_gb, refined_chi, refined_time, "refined")
        best = min(
            (result for result in candidate_results if result.status == "computed" and math.isfinite(result.score)),
            key=lambda result: result.score,
        )

    # Mesh sensitivity uses the same selected best parameters at each mesh.
    # It is intentionally not a second independent parameter refit; the
    # artifact therefore cannot be misread as resolution-converged inference.
    resolution_rows: list[dict[str, Any]] = []
    for resolution in resolution_grid:
        if resolution == int(n_nodes):
            resolution_result = best
        else:
            resolution_positions = canonical_observation(observations[0].points, resolution, model_length)
            resolution_state = FilamentState(
                resolution_positions,
                np.linalg.norm(np.diff(resolution_positions, axis=0), axis=1),
            )
            resolution_result = _evaluate_candidate(
                resolution_state,
                observations,
                G_b=best.G_b,
                chi=best.chi,
                growth_time=best.growth_time,
                model_length=model_length,
                dt=dt,
                max_retries=max_retries,
            )
        if resolution_result.status == "computed":
            resolution_target = resolution_result.frame_metrics[-1]
            resolution_rows.append({
                "n_nodes": resolution,
                "status": resolution_result.status,
                "G_b": best.G_b,
                "chi": best.chi,
                "growth_time": best.growth_time,
                "score": resolution_result.score,
                "frechet_distance_px": resolution_target.get("frechet_distance_px"),
                "curvature_rmse_px_inv": resolution_target.get("curvature_rmse_px_inv"),
                "feature_loss": resolution_target.get("feature_loss"),
                "observed_growth_rate": resolution_result.temporal_features.get("observed_growth_rate"),
                "model_growth_rate": resolution_result.temporal_features.get("model_growth_rate"),
                "failure_reason": None,
            })
        else:
            resolution_rows.append({
                "n_nodes": resolution,
                "status": resolution_result.status,
                "G_b": best.G_b,
                "chi": best.chi,
                "growth_time": best.growth_time,
                "score": None,
                "frechet_distance_px": None,
                "curvature_rmse_px_inv": None,
                "feature_loss": None,
                "observed_growth_rate": None,
                "model_growth_rate": None,
                "failure_reason": resolution_result.failure_reason,
            })

    target_observation = observations[-1]
    straight = np.column_stack((np.linspace(0.0, model_length, int(n_nodes)), np.zeros(int(n_nodes))))
    baseline = compare_shape(straight, target_observation.points)
    target_metric = best.frame_metrics[-1]
    baseline_frechet = float(baseline["frechet_distance_px"])
    best_frechet = float(target_metric["frechet_distance_px"])
    improvement = baseline_frechet - best_frechet
    improvement_percent = 100.0 * improvement / max(baseline_frechet, 1.0e-12)
    baseline_feature_loss = float(baseline["feature_loss"])
    best_feature_loss = float(target_metric["feature_loss"])
    feature_improvement = baseline_feature_loss - best_feature_loss
    feature_improvement_percent = 100.0 * feature_improvement / max(baseline_feature_loss, 1.0e-12)

    output.mkdir(parents=True, exist_ok=True)
    trajectory_path = output / "best_fit_trajectory.npz"
    _write_trajectory(
        trajectory_path,
        best.trajectory,
        {
            "schema_version": SCHEMA_VERSION,
            "G_b": best.G_b,
            "chi": best.chi,
            "growth_time": best.growth_time,
            "n_nodes_initial": n_nodes,
            "model_length": model_length,
        },
    )
    candidates_path = output / "candidate_scores.csv"
    _write_candidate_csv(candidates_path, candidate_rows)
    resolution_path = output / "resolution_sensitivity.csv"
    _write_resolution_csv(resolution_path, resolution_rows)
    resolution_json_path = output / "resolution_sensitivity.json"
    _write_json(resolution_json_path, {
        "schema_version": SCHEMA_VERSION,
        "mode": "fixed_best_parameters_mesh_sensitivity",
        "parameters": {"G_b": best.G_b, "chi": best.chi, "growth_time": best.growth_time},
        "rows": resolution_rows,
        "interpretation": "This is a mesh sensitivity check at fixed parameters, not an independent optimum at each resolution.",
    })
    plot_path = output / "best_fit_comparison.png"
    if write_plot:
        _write_plot(plot_path, observations, best, baseline, candidate_rows)

    censored_count = sum(bool(item.censor) for item in observations)
    summary: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "computed",
        "input": {
            "centerline": _logical_path(centerline),
            "selected_filament_id": observations[0].filament_id,
            "initial_shape_frame": {
                "frame": initial_shape_frame.frame,
                "time": initial_shape_frame.time,
                "reason": "earliest selected representative frame used as causal model seed",
            },
            "selected_frames": [
                {
                    "frame": item.frame,
                    "time": item.time,
                    "filament_id": item.filament_id,
                    "n_points": len(item.points),
                    "censor": item.censor,
                    "quality": item.quality,
                    "quality_flags": item.quality_flags,
                }
                for item in observations
            ],
        },
        "data_quality": {
            "selected_frame_count": len(observations),
            "censored_frame_count": censored_count,
            "all_selected_frames_censored": censored_count == len(observations),
            "interpretation": "censored exploratory overlay; not a material-constant identification",
        },
        "search": {
            "n_nodes": n_nodes,
            "resolution_values": resolution_grid,
            "resolution_sensitivity_mode": "fixed_best_parameters_mesh_sensitivity",
            "model_length": model_length,
            "coarse_G_b": gb_values,
            "coarse_chi": chi_values,
            "growth_times": time_values,
            "refinement": refinement_spec,
            "candidate_count": len(candidate_results),
            "visual_frechet_weight": 50.0,
            "dt": dt,
            "model_nondimensionalisation": "EA=1, zeta=1, EI=chi*EA*L^2; growth_rate=G_b/tau_b",
        },
        "resolution_sensitivity": resolution_rows,
        "best_fit": {
            "G_b": best.G_b,
            "chi": best.chi,
            "growth_time": best.growth_time,
            "normalized_shape_loss": best.score,
            "target_frame": target_metric,
            "frame_metrics": best.frame_metrics,
            "temporal_features": best.temporal_features,
            "feature_names": FEATURE_NAMES,
            "feature_scales": FEATURE_SCALES,
        },
        "baseline": {
            "type": "straight_centerline",
            "target_frechet_distance_px": baseline_frechet,
            "target_curvature_rmse_px_inv": baseline["curvature_rmse_px_inv"],
            "feature_loss": baseline_feature_loss,
            "observed_features": baseline["observed_features"],
            "model_features": baseline["model_features"],
            "normalized_shape_loss": baseline["normalized_shape_loss"],
        },
        "improvement": {
            "frechet_distance_px_reduction": improvement,
            "frechet_distance_percent": improvement_percent,
            "feature_loss_reduction": feature_improvement,
            "feature_loss_percent": feature_improvement_percent,
            "reference": "straight_centerline baseline on the selected target frame",
        },
        "artifacts": {
            "comparison_plot": _logical_path(plot_path),
            "candidate_scores": _logical_path(candidates_path),
            "resolution_sensitivity_csv": _logical_path(resolution_path),
            "resolution_sensitivity_json": _logical_path(resolution_json_path),
            "best_fit_trajectory": _logical_path(trajectory_path),
        },
        "limitations": [
            "The anchor observation is smoothed and used as the model initial geometry; this is an exploratory shape comparison, not an independent parameter identification.",
            "Pixel-to-physical calibration is not inferred. Each overlay uses one global similarity registration to the selected observation frame.",
            "All selected frames retain their source censor and quality flags; a low geometric loss does not establish tracking validity or physical model validity.",
            "The reported Fréchet improvement is relative to a straight-centerline baseline, not a holdout or uncertainty-calibrated comparison.",
        ],
    }
    _write_json(output / "summary.json", summary)
    return summary


def _parse_float_list(values: Sequence[str] | None, default: Sequence[float]) -> tuple[float, ...]:
    if values is None:
        return tuple(float(value) for value in default)
    return tuple(float(value) for value in values)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--centerline", type=Path, default=DEFAULT_CENTERLINE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--target-times", nargs="+", help="representative observation times in seconds")
    parser.add_argument("--filament-id")
    parser.add_argument("--n-nodes", type=int, default=25)
    parser.add_argument("--resolution-values", nargs="+", help="mesh values for fixed-parameter sensitivity check")
    parser.add_argument("--model-length", type=float, default=2.0)
    parser.add_argument("--coarse-gb", nargs="+")
    parser.add_argument("--coarse-chi", nargs="+")
    parser.add_argument("--growth-times", nargs="+")
    parser.add_argument("--no-refine", action="store_true")
    parser.add_argument("--dt", type=float, default=0.01)
    parser.add_argument("--max-retries", type=int, default=4)
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args(argv)
    try:
        summary = run_exploratory_fit(
            args.centerline,
            args.output,
            target_times=_parse_float_list(args.target_times, DEFAULT_TARGET_TIMES),
            filament_id=args.filament_id,
            n_nodes=args.n_nodes,
            resolution_values=(
                tuple(int(value) for value in args.resolution_values)
                if args.resolution_values is not None else DEFAULT_RESOLUTION_VALUES
            ),
            model_length=args.model_length,
            coarse_gb=_parse_float_list(args.coarse_gb, DEFAULT_GB_GRID),
            coarse_chi=_parse_float_list(args.coarse_chi, DEFAULT_CHI_GRID),
            growth_times=_parse_float_list(args.growth_times, DEFAULT_GROWTH_TIMES),
            refine=not args.no_refine,
            dt=args.dt,
            max_retries=args.max_retries,
            write_plot=not args.no_plot,
        )
    except (OSError, RuntimeError, ValueError, FloatingPointError) as exc:
        print(f"exploratory shape fitting error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({"output": str(args.output), "status": summary["status"], "best_fit": summary["best_fit"]}, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
