"""Video-to-centreline comparison utilities.

This module deliberately lives beside, rather than inside, the continuum
filament solver.  It is an observation/visualisation pipeline: it does not
change or reimplement the model equations.  The implementation uses ffmpeg
when available and has a pure NumPy connected-component and skeleton fallback
so an OpenCV/NumPy ABI mismatch does not prevent analysis.

The public entry points are :func:`run_pipeline`, :func:`validate_centerline_csv`,
and :func:`compare_with_model`.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
from collections import deque
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

import numpy as np

SCHEMA_VERSION = "continuum-filament-observation-0.1"


# ---------------------------------------------------------------------------
# Configuration and deterministic serialization


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def canonical_json(value: Any) -> str:
    return json.dumps(_jsonable(value), ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_file(path: str | Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _parse_roi(value: Any) -> tuple[int, int, int, int] | None:
    if value is None or value == "" or value == []:
        return None
    if isinstance(value, str):
        value = [int(float(part.strip())) for part in value.split(",")]
    if len(value) != 4:
        raise ValueError("ROI must be [x0, y0, x1, y1]")
    x0, y0, x1, y1 = (int(v) for v in value)
    if x1 <= x0 or y1 <= y0:
        raise ValueError("ROI must have x1>x0 and y1>y0")
    return x0, y0, x1, y1


@dataclass(frozen=True)
class SegmentationConfig:
    """Image-to-mask settings.  Threshold values are in [0, 1] after contrast."""

    polarity: str = "dark"  # dark filament on bright background, or bright
    background: str = "median"  # none, median, local_median, or scalar
    background_value: float = 0.0
    contrast: str = "percentile"  # none or percentile
    contrast_low: float = 1.0
    contrast_high: float = 99.0
    threshold: str = "otsu"  # otsu, absolute, percentile
    threshold_value: float = 0.45
    threshold_percentile: float = 12.0
    roi: tuple[int, int, int, int] | None = None
    frame_stride: int = 5
    min_component_size: int = 12
    max_components: int = 8
    max_centerline_points: int = 240
    skeleton: bool = True
    max_jump_px: float = 80.0
    min_quality: float = 0.20
    boundary_margin_px: int = 2
    output_budget_bytes: int | None = None

    def __post_init__(self) -> None:
        if self.polarity not in {"dark", "bright"}:
            raise ValueError("polarity must be dark or bright")
        if self.background not in {"none", "median", "local_median", "scalar"}:
            raise ValueError("background must be none, median, local_median, or scalar")
        if self.contrast not in {"none", "percentile"}:
            raise ValueError("contrast must be none or percentile")
        if self.threshold not in {"otsu", "absolute", "percentile"}:
            raise ValueError("threshold must be otsu, absolute, or percentile")
        if self.frame_stride < 1 or self.min_component_size < 1:
            raise ValueError("frame_stride and min_component_size must be positive")
        if self.max_components < 1 or self.max_centerline_points < 2:
            raise ValueError("max_components and max_centerline_points are too small")
        if self.boundary_margin_px < 0:
            raise ValueError("boundary_margin_px must be non-negative")
        if self.output_budget_bytes is not None and self.output_budget_bytes <= 0:
            raise ValueError("output_budget_bytes must be positive when specified")
        _parse_roi(self.roi)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any] | None) -> "SegmentationConfig":
        values = dict(value or {})
        if "roi" in values:
            values["roi"] = _parse_roi(values["roi"])
        return cls(**values)

    def to_dict(self) -> dict[str, Any]:
        return _jsonable(asdict(self))


@dataclass(frozen=True)
class RegistrationConfig:
    """Time and spatial registration between model units and image pixels.

    ``model_time = (observed_time - time_offset) / time_scale``.
    ``pixel = pixel_per_model_unit * R(rotation) @ model + (x_offset, y_offset)``.
    A missing spatial scale intentionally disables quantitative metrics.
    """

    pixel_per_model_unit: float | None = None
    x_offset_px: float = 0.0
    y_offset_px: float = 0.0
    rotation_deg: float = 0.0
    time_scale: float = 1.0
    time_offset: float = 0.0
    max_time_error_s: float = 0.20
    endpoint_order: str = "auto"  # auto, forward, or reverse

    def __post_init__(self) -> None:
        if self.pixel_per_model_unit is not None and self.pixel_per_model_unit <= 0:
            raise ValueError("pixel_per_model_unit must be positive")
        if self.time_scale <= 0 or self.max_time_error_s < 0:
            raise ValueError("time_scale must be positive and max_time_error_s non-negative")
        if self.endpoint_order not in {"auto", "forward", "reverse"}:
            raise ValueError("endpoint_order must be auto, forward, or reverse")

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any] | None) -> "RegistrationConfig":
        return cls(**dict(value or {}))

    def to_dict(self) -> dict[str, Any]:
        return _jsonable(asdict(self))

    @property
    def calibrated(self) -> bool:
        return self.pixel_per_model_unit is not None

    def model_time(self, observed_time: float) -> float:
        return (float(observed_time) - self.time_offset) / self.time_scale

    def model_to_pixel(self, points: np.ndarray) -> np.ndarray:
        if not self.calibrated:
            raise ValueError("pixel_per_model_unit is required for model-to-pixel registration")
        angle = math.radians(self.rotation_deg)
        rotation = np.asarray(
            [[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]],
            dtype=float,
        )
        return np.asarray(points, dtype=float) @ rotation.T * float(self.pixel_per_model_unit) + np.asarray(
            [self.x_offset_px, self.y_offset_px], dtype=float
        )


@dataclass(frozen=True)
class VideoMetadata:
    path: str
    width: int
    height: int
    fps: float
    frame_count: int | None
    duration_s: float | None
    codec: str | None = None
    pixel_format: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return _jsonable(asdict(self))


@dataclass
class CenterlineCandidate:
    frame_index: int
    time_s: float
    filament_id: str
    points: np.ndarray  # x,y in pixel coordinates
    component_area: int
    centroid: np.ndarray
    quality: float
    flags: list[str] = field(default_factory=list)
    censor: bool = False
    topology: dict[str, int] = field(default_factory=dict)
    component_count_total: int = 1
    centerline_exported: bool = True

    def summary(self) -> dict[str, Any]:
        metrics = polyline_metrics(self.points)
        return {
            "frame": self.frame_index,
            "time": self.time_s,
            "filament_id": self.filament_id,
            "n_points": int(len(self.points)) if self.centerline_exported else 0,
            "centerline_exported": int(self.centerline_exported),
            "component_area": self.component_area,
            "component_count_total": self.component_count_total,
            "endpoint_count": int(self.topology.get("endpoint_count", 0)),
            "junction_count": int(self.topology.get("junction_count", 0)),
            "cycle_rank": int(self.topology.get("cycle_rank", 0)),
            "length_px": metrics["length_px"],
            "endpoint_distance_px": metrics["endpoint_distance_px"],
            "curvature_mean_px_inv": metrics["curvature_mean_px_inv"],
            "curvature_max_px_inv": metrics["curvature_max_px_inv"],
            "quality": self.quality,
            "quality_flags": ";".join(self.flags) if self.flags else "ok",
            "censor": int(self.censor),
        }


# ---------------------------------------------------------------------------
# Video reading without OpenCV


def _run_command(command: Sequence[str]) -> str:
    try:
        completed = subprocess.run(command, check=True, capture_output=True, text=True)
    except FileNotFoundError as exc:
        raise RuntimeError(f"required executable is not installed: {command[0]}") from exc
    except subprocess.CalledProcessError as exc:
        stderr = (exc.stderr or "").strip()
        raise RuntimeError(f"command failed ({' '.join(command[:3])}…): {stderr}") from exc
    return completed.stdout


def _rational(value: str | None, default: float = 0.0) -> float:
    if not value or value in {"N/A", "0/0"}:
        return default
    try:
        if "/" in value:
            numerator, denominator = value.split("/", 1)
            return float(numerator) / float(denominator)
        return float(value)
    except (ValueError, ZeroDivisionError):
        return default


def probe_video(path: str | Path) -> VideoMetadata:
    """Read stable metadata through ffprobe, with imageio as a fallback."""

    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    try:
        raw = _run_command(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=width,height,r_frame_rate,avg_frame_rate,nb_frames,duration,codec_name,pix_fmt",
                "-show_entries",
                "format=duration",
                "-of",
                "json",
                str(source),
            ]
        )
        data = json.loads(raw)
        stream = (data.get("streams") or [{}])[0]
        fmt = data.get("format") or {}
        fps = _rational(stream.get("avg_frame_rate"), _rational(stream.get("r_frame_rate"), 0.0))
        duration_raw = stream.get("duration") or fmt.get("duration")
        duration = float(duration_raw) if duration_raw not in (None, "N/A") else None
        count_raw = stream.get("nb_frames")
        count = int(count_raw) if count_raw not in (None, "N/A") else None
        if count is None and duration is not None and fps > 0:
            count = int(round(duration * fps))
        return VideoMetadata(
            path=str(source),
            width=int(stream.get("width") or 0),
            height=int(stream.get("height") or 0),
            fps=float(fps),
            frame_count=count,
            duration_s=duration,
            codec=stream.get("codec_name"),
            pixel_format=stream.get("pix_fmt"),
        )
    except Exception as ffprobe_error:
        try:
            import imageio.v3 as iio  # optional fallback already present in the environment

            meta = dict(iio.immeta(source, plugin="ffmpeg"))
            size = tuple(meta.get("size") or (0, 0))
            fps = float(meta.get("fps") or 0.0)
            duration = float(meta["duration"]) if meta.get("duration") else None
            count = int(meta["nframes"]) if meta.get("nframes") not in (None, float("inf")) else None
            if count is None and duration and fps:
                count = int(round(duration * fps))
            return VideoMetadata(str(source), int(size[0]), int(size[1]), fps, count, duration)
        except Exception as fallback_error:
            raise RuntimeError(
                f"video metadata failed with ffprobe ({ffprobe_error}) and imageio ({fallback_error})"
            ) from fallback_error


def _iter_video_frames_imageio(
    path: str | Path,
    metadata: VideoMetadata,
    frame_stride: int,
    max_frames: int | None,
) -> Iterator[tuple[int, float, np.ndarray]]:
    """Optional imageio fallback used when the ffmpeg executable is unavailable."""

    import imageio.v3 as iio

    yielded = 0
    for index, raw in enumerate(iio.imiter(path, plugin="ffmpeg")):
        if index % frame_stride:
            continue
        array = np.asarray(raw)
        if array.ndim == 3:
            array = np.asarray(np.dot(array[..., :3], [0.299, 0.587, 0.114]), dtype=np.uint8)
        image = np.asarray(array, dtype=np.uint8)
        seconds = index / metadata.fps if metadata.fps > 0 else float(index)
        yield index, float(seconds), image
        yielded += 1
        if max_frames is not None and yielded >= max_frames:
            break


def iter_video_frames(
    path: str | Path,
    metadata: VideoMetadata | None = None,
    frame_stride: int = 1,
    max_frames: int | None = None,
) -> Iterator[tuple[int, float, np.ndarray]]:
    """Yield ``(frame_index, seconds, grayscale_uint8)`` using ffmpeg rawvideo."""

    if frame_stride < 1:
        raise ValueError("frame_stride must be positive")
    meta = metadata or probe_video(path)
    if shutil.which("ffmpeg") is None:
        yield from _iter_video_frames_imageio(path, meta, frame_stride, max_frames)
        return
    if meta.width <= 0 or meta.height <= 0:
        raise ValueError("video metadata has no frame dimensions")
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(path),
        "-f",
        "rawvideo",
        "-pix_fmt",
        "gray",
        "-vsync",
        "0",
        "-",
    ]
    process: subprocess.Popen[bytes] | None = None
    try:
        try:
            process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        except FileNotFoundError:
            yield from _iter_video_frames_imageio(path, meta, frame_stride, max_frames)
            return
        bytes_per_frame = meta.width * meta.height
        index = 0
        yielded = 0
        while True:
            assert process.stdout is not None
            raw = process.stdout.read(bytes_per_frame)
            if len(raw) != bytes_per_frame:
                break
            if index % frame_stride == 0:
                image = np.frombuffer(raw, dtype=np.uint8).reshape(meta.height, meta.width).copy()
                seconds = index / meta.fps if meta.fps > 0 else float(index)
                yield index, float(seconds), image
                yielded += 1
                if max_frames is not None and yielded >= max_frames:
                    break
            index += 1
        if max_frames is None or yielded < max_frames:
            stderr = process.stderr.read().decode("utf-8", errors="replace") if process.stderr else ""
            return_code = process.wait()
            if return_code != 0:
                raise RuntimeError(f"ffmpeg frame decode failed: {stderr.strip()}")
    finally:
        if process is not None and process.poll() is None:
            process.kill()
            process.wait()
        if process is not None:
            if process.stdout is not None:
                process.stdout.close()
            if process.stderr is not None:
                process.stderr.close()


# ---------------------------------------------------------------------------
# Segmentation and centreline extraction


def _normalise_image(image: np.ndarray, config: SegmentationConfig) -> np.ndarray:
    values = np.asarray(image, dtype=float)
    if config.background == "median":
        background = float(np.median(values))
        if config.polarity == "dark":
            values = background - values
        else:
            values = values - background
    elif config.background == "local_median":
        try:
            from scipy.ndimage import median_filter  # type: ignore

            background = median_filter(values, size=31, mode="reflect")
        except Exception:
            background = float(np.median(values))
        if config.polarity == "dark":
            values = background - values
        else:
            values = values - background
    elif config.background == "scalar":
        if config.polarity == "dark":
            values = float(config.background_value) - values
        else:
            values = values - float(config.background_value)
    elif config.polarity == "dark":
        values = values.max() - values
    values = np.maximum(values, 0.0)
    if config.contrast == "percentile":
        lo, hi = np.percentile(values, [config.contrast_low, config.contrast_high])
        if hi > lo:
            values = (values - lo) / (hi - lo)
        else:
            values = np.zeros_like(values)
    else:
        max_value = float(np.max(values))
        if max_value > 0:
            values = values / max_value
    return np.clip(values, 0.0, 1.0)


def _otsu(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    if values.size == 0 or float(np.max(values)) <= float(np.min(values)):
        return float(np.median(values)) if values.size else 0.5
    histogram, edges = np.histogram(values, bins=256, range=(0.0, 1.0))
    probabilities = histogram.astype(float) / max(float(values.size), 1.0)
    cumulative = np.cumsum(probabilities)
    means = np.cumsum(probabilities * np.arange(256, dtype=float))
    total_mean = means[-1]
    denominator = cumulative * (1.0 - cumulative)
    between = np.zeros_like(denominator)
    valid = denominator > 1.0e-15
    between[valid] = (total_mean * cumulative[valid] - means[valid]) ** 2 / denominator[valid]
    return float((edges[int(np.argmax(between))] + edges[int(np.argmax(between)) + 1]) * 0.5)


def segment_mask(image: np.ndarray, config: SegmentationConfig) -> tuple[np.ndarray, dict[str, Any]]:
    """Return a binary mask and threshold diagnostics; coordinates remain pixels."""

    score = _normalise_image(image, config)
    roi = _parse_roi(config.roi)
    if roi is None:
        roi_mask = np.ones(score.shape, dtype=bool)
    else:
        x0, y0, x1, y1 = roi
        roi_mask = np.zeros(score.shape, dtype=bool)
        roi_mask[max(y0, 0) : min(y1, score.shape[0]), max(x0, 0) : min(x1, score.shape[1])] = True
    values = score[roi_mask]
    if config.threshold == "otsu":
        threshold = _otsu(values)
    elif config.threshold == "percentile":
        threshold = float(np.percentile(values, config.threshold_percentile)) if values.size else 1.0
    else:
        threshold = float(config.threshold_value)
    mask = (score >= threshold) & roi_mask
    return mask, {
        "threshold": threshold,
        "foreground_fraction": float(np.mean(mask)),
        "roi": roi,
        "score_min": float(np.min(score)) if score.size else 0.0,
        "score_max": float(np.max(score)) if score.size else 0.0,
    }


def connected_components(mask: np.ndarray, min_size: int = 1) -> list[np.ndarray]:
    """Return 8-connected ``(y,x)`` arrays without requiring OpenCV/skimage."""

    binary = np.asarray(mask, dtype=bool)
    try:
        from skimage.measure import label, regionprops  # type: ignore

        labels = label(binary, connectivity=2)
        components = []
        for region in regionprops(labels):
            if region.area >= min_size:
                components.append(np.argwhere(labels == region.label))
        components.sort(key=len, reverse=True)
        return components
    except Exception:
        # This path is intentionally simple and dependency-free.  It is slower
        # than skimage for large masks but is safe under binary ABI failures.
        height, width = binary.shape
        visited = np.zeros_like(binary)
        components: list[np.ndarray] = []
        neighbours = [
            (dy, dx)
            for dy in (-1, 0, 1)
            for dx in (-1, 0, 1)
            if (dy, dx) != (0, 0)
        ]
        for y in range(height):
            for x in range(width):
                if not binary[y, x] or visited[y, x]:
                    continue
                queue = deque([(y, x)])
                visited[y, x] = True
                pixels: list[tuple[int, int]] = []
                while queue:
                    cy, cx = queue.popleft()
                    pixels.append((cy, cx))
                    for dy, dx in neighbours:
                        ny, nx = cy + dy, cx + dx
                        if 0 <= ny < height and 0 <= nx < width and binary[ny, nx] and not visited[ny, nx]:
                            visited[ny, nx] = True
                            queue.append((ny, nx))
                if len(pixels) >= min_size:
                    components.append(np.asarray(pixels, dtype=int))
        components.sort(key=len, reverse=True)
        return components


def _fallback_skeleton(mask: np.ndarray) -> np.ndarray:
    """A dependency-free thinning approximation, adequate for line-like fixtures."""

    current = np.asarray(mask, dtype=bool).copy()
    # Iterative boundary erosion preserves the medial-ish core.  It is not
    # claimed to be a topology-preserving skeleton; skimage is preferred.
    for _ in range(max(1, min(current.shape) // 3)):
        if not np.any(current):
            break
        neighbour_count = np.zeros_like(current, dtype=np.uint8)
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if not dy and not dx:
                    continue
                shifted = np.zeros_like(current)
                y0, y1 = max(0, dy), min(current.shape[0], current.shape[0] + dy)
                x0, x1 = max(0, dx), min(current.shape[1], current.shape[1] + dx)
                shifted[y0:y1, x0:x1] = current[y0 - dy : y1 - dy, x0 - dx : x1 - dx]
                neighbour_count += shifted
        boundary = current & (neighbour_count < 8)
        if not np.any(boundary):
            break
        candidate = current & ~boundary
        if np.any(candidate):
            current = candidate
        else:
            break
    return current


def _skeleton_coordinates(component: np.ndarray, shape: tuple[int, int], use_skeleton: bool) -> np.ndarray:
    y0, x0 = np.min(component, axis=0)
    y1, x1 = np.max(component, axis=0)
    local = np.zeros((int(y1 - y0 + 3), int(x1 - x0 + 3)), dtype=bool)
    local[component[:, 0] - y0 + 1, component[:, 1] - x0 + 1] = True
    skeleton = local
    if use_skeleton:
        try:
            from skimage.morphology import skeletonize  # type: ignore

            skeleton = skeletonize(local)
        except Exception:
            skeleton = _fallback_skeleton(local)
    coords = np.argwhere(skeleton)
    if len(coords) == 0:
        coords = np.argwhere(local)
    coords = coords.astype(float)
    coords[:, 0] += float(y0 - 1)
    coords[:, 1] += float(x0 - 1)
    return coords


def _ordered_skeleton(coords_yx: np.ndarray) -> np.ndarray:
    """Order skeleton pixels by the longest endpoint-to-endpoint graph path."""

    coords = [tuple(int(round(v)) for v in row) for row in coords_yx]
    if len(coords) < 2:
        return np.asarray(coords_yx[:, ::-1], dtype=float)
    index = {point: i for i, point in enumerate(coords)}
    graph: list[list[int]] = [[] for _ in coords]
    for i, (y, x) in enumerate(coords):
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if not dy and not dx:
                    continue
                j = index.get((y + dy, x + dx))
                if j is not None:
                    graph[i].append(j)
    endpoints = [i for i, neighbours in enumerate(graph) if len(neighbours) == 1]
    if not endpoints:
        center = np.mean(np.asarray(coords, dtype=float), axis=0)
        angles = np.arctan2(np.asarray(coords)[:, 0] - center[0], np.asarray(coords)[:, 1] - center[1])
        order = np.argsort(angles)
        return np.asarray(coords, dtype=float)[order][:, ::-1]

    best_path: list[int] = []
    for start in endpoints:
        queue = deque([start])
        parent = {start: -1}
        while queue:
            current = queue.popleft()
            for neighbour in graph[current]:
                if neighbour not in parent:
                    parent[neighbour] = current
                    queue.append(neighbour)
        for finish in endpoints:
            if finish not in parent:
                continue
            path: list[int] = []
            current = finish
            while current >= 0:
                path.append(current)
                current = parent[current]
            path.reverse()
            if len(path) > len(best_path):
                best_path = path
    if not best_path:
        best_path = list(range(len(coords)))
    # Append branch pixels in their deterministic coordinate order only when
    # the path is very short; the main path is the comparison candidate.
    return np.asarray([coords[i] for i in best_path], dtype=float)[:, ::-1]


def skeleton_topology(coords_yx: np.ndarray) -> dict[str, int]:
    """Summarise skeleton graph topology without assuming an open curve.

    ``cycle_rank = E - V + C`` detects loops, while degree-one and degree-three
    pixels detect endpoints and branches.  A loop is never silently exported as
    an open centerline candidate: callers mark it censored.
    """

    coords = [tuple(int(round(v)) for v in row) for row in np.asarray(coords_yx)]
    if not coords:
        return {
            "skeleton_vertices": 0,
            "skeleton_edges": 0,
            "skeleton_components": 0,
            "endpoint_count": 0,
            "junction_count": 0,
            "cycle_rank": 0,
        }
    index = {point: i for i, point in enumerate(coords)}
    graph: list[list[int]] = [[] for _ in coords]
    for i, (y, x) in enumerate(coords):
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if not dy and not dx:
                    continue
                j = index.get((y + dy, x + dx))
                if j is not None:
                    graph[i].append(j)
    visited: set[int] = set()
    component_count = 0
    for start in range(len(coords)):
        if start in visited:
            continue
        component_count += 1
        queue = [start]
        visited.add(start)
        while queue:
            current = queue.pop()
            for neighbour in graph[current]:
                if neighbour not in visited:
                    visited.add(neighbour)
                    queue.append(neighbour)
    vertices = len(coords)
    edges = sum(len(neighbours) for neighbours in graph) // 2
    cycle_rank = max(0, edges - vertices + component_count)
    return {
        "skeleton_vertices": vertices,
        "skeleton_edges": edges,
        "skeleton_components": component_count,
        "endpoint_count": sum(len(neighbours) == 1 for neighbours in graph),
        "junction_count": sum(len(neighbours) >= 3 for neighbours in graph),
        "cycle_rank": cycle_rank,
    }


def resample_polyline(points_xy: np.ndarray, max_points: int = 240) -> np.ndarray:
    points = np.asarray(points_xy, dtype=float)
    if len(points) <= max_points:
        return points.copy()
    segment_lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
    distance = np.concatenate(([0.0], np.cumsum(segment_lengths)))
    target = np.linspace(0.0, float(distance[-1]), max_points)
    result = np.column_stack(
        [np.interp(target, distance, points[:, axis]) for axis in range(points.shape[1])]
    )
    return result


def component_to_centerline(
    component_yx: np.ndarray,
    image_shape: tuple[int, int],
    config: SegmentationConfig,
) -> tuple[np.ndarray, list[str]]:
    skeleton = _skeleton_coordinates(component_yx, image_shape, config.skeleton)
    topology = skeleton_topology(skeleton)
    points = _ordered_skeleton(skeleton)
    flags: list[str] = []
    if topology["junction_count"] > 0:
        flags.append("branched_component")
    if topology["cycle_rank"] > 0 or topology["endpoint_count"] == 0:
        flags.append("loop_component")
    if topology["skeleton_components"] > 1:
        flags.append("disconnected_skeleton")
    if len(points) < 3:
        flags.append("short_centerline")
    if len(component_yx) > 0 and len(points) < max(2, int(math.sqrt(len(component_yx)) / 2)):
        flags.append("skeleton_loss")
    return resample_polyline(points, config.max_centerline_points), flags


def polyline_metrics(points_xy: np.ndarray) -> dict[str, float]:
    points = np.asarray(points_xy, dtype=float)
    if len(points) < 2:
        return {
            "length_px": 0.0,
            "endpoint_distance_px": 0.0,
            "curvature_mean_px_inv": float("nan"),
            "curvature_max_px_inv": float("nan"),
        }
    vectors = np.diff(points, axis=0)
    lengths = np.linalg.norm(vectors, axis=1)
    total = float(np.sum(lengths))
    endpoint_distance = float(np.linalg.norm(points[-1] - points[0]))
    curvature: list[float] = []
    for i in range(1, len(points) - 1):
        left, right = vectors[i - 1], vectors[i]
        left_norm, right_norm = np.linalg.norm(left), np.linalg.norm(right)
        if left_norm <= 1e-12 or right_norm <= 1e-12:
            continue
        cosine = float(np.clip(np.dot(left, right) / (left_norm * right_norm), -1.0, 1.0))
        angle = math.acos(cosine)
        curvature.append(angle / max((left_norm + right_norm) * 0.5, 1e-12))
    return {
        "length_px": total,
        "endpoint_distance_px": endpoint_distance,
        "curvature_mean_px_inv": float(np.mean(curvature)) if curvature else 0.0,
        "curvature_max_px_inv": float(np.max(curvature)) if curvature else 0.0,
    }


# ---------------------------------------------------------------------------
# Frame pipeline, contract validation, and deterministic artifacts


class _TrackManager:
    def __init__(self, max_jump_px: float) -> None:
        self.max_jump_px = max_jump_px
        self.next_id = 0
        self.last: dict[str, np.ndarray] = {}
        self.last_order: dict[str, int] = {}

    def assign(
        self,
        centroids: Sequence[np.ndarray],
        frame_index: int,
        observation_order: int,
    ) -> tuple[list[str], list[bool], list[str]]:
        names = list(self.last)
        pairs: list[tuple[float, int, str]] = []
        for index, centroid in enumerate(centroids):
            for name in names:
                if self.last_order[name] >= observation_order:
                    continue
                distance = float(np.linalg.norm(centroid - self.last[name]))
                pairs.append((distance, index, name))
        pairs.sort()
        assigned: dict[int, str] = {}
        used: set[str] = set()
        jumps = [False] * len(centroids)
        statuses = ["new_lineage"] * len(centroids)
        for distance, index, name in pairs:
            if index in assigned or name in used:
                continue
            if distance <= self.max_jump_px:
                assigned[index] = name
                used.add(name)
                jumps[index] = distance > self.max_jump_px * 0.5
                statuses[index] = (
                    "reconnected_after_missing"
                    if self.last_order[name] < observation_order - 1
                    else "matched"
                )
        for index, centroid in enumerate(centroids):
            if index not in assigned:
                assigned[index] = f"filament-{self.next_id:04d}"
                self.next_id += 1
                statuses[index] = "initial_lineage" if observation_order == 0 else "new_lineage"
        result = [assigned[i] for i in range(len(centroids))]
        for name, centroid in zip(result, centroids):
            self.last[name] = np.asarray(centroid, dtype=float)
            self.last_order[name] = observation_order
        return result, jumps, statuses


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames), extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        for row in rows:
            clean = {name: _jsonable(row.get(name, "")) for name in fieldnames}
            writer.writerow(clean)


def _candidate_rows(candidate: CenterlineCandidate) -> list[dict[str, Any]]:
    if not candidate.centerline_exported:
        return []
    rows = []
    for point_id, (x, y) in enumerate(candidate.points):
        rows.append(
            {
                "time": f"{candidate.time_s:.9f}",
                "filament_id": candidate.filament_id,
                "point_id": point_id,
                "x": f"{float(x):.9f}",
                "y": f"{float(y):.9f}",
                "quality": f"{candidate.quality:.9f}",
                "frame": candidate.frame_index,
                "coordinate_system": "pixel",
                "quality_flags": ";".join(candidate.flags) if candidate.flags else "ok",
                "censor": int(candidate.censor),
            }
        )
    return rows


def _make_candidate(
    frame_index: int,
    time_s: float,
    filament_id: str,
    component: np.ndarray,
    image_shape: tuple[int, int],
    config: SegmentationConfig,
    track_jump: bool,
    component_count: int,
    component_total: int,
    roi: tuple[int, int, int, int] | None,
) -> CenterlineCandidate:
    points, flags = component_to_centerline(component, image_shape, config)
    centroid_yx = np.mean(component, axis=0) if len(component) else np.asarray([np.nan, np.nan])
    centroid = centroid_yx[::-1].astype(float)
    area_score = min(1.0, len(component) / max(config.min_component_size * 10.0, 1.0))
    line_score = min(1.0, len(points) / 20.0)
    quality = float(np.clip(0.5 * area_score + 0.5 * line_score, 0.0, 1.0))
    if track_jump:
        flags.append("large_jump")
    if component_total > 1:
        flags.append("ambiguous_components")
    if component_total > component_count:
        flags.append("components_truncated")
    margin = max(0, int(config.boundary_margin_px))
    component_y, component_x = component[:, 0], component[:, 1]
    height, width = image_shape
    image_boundary = bool(
        np.any(component_x <= margin)
        or np.any(component_x >= width - 1 - margin)
        or np.any(component_y <= margin)
        or np.any(component_y >= height - 1 - margin)
    )
    endpoint_boundary = bool(
        len(points) >= 2
        and (
            np.any(points[0] <= [margin, margin])
            or np.any(points[0] >= [width - 1 - margin, height - 1 - margin])
            or np.any(points[-1] <= [margin, margin])
            or np.any(points[-1] >= [width - 1 - margin, height - 1 - margin])
        )
    )
    if image_boundary or endpoint_boundary:
        flags.append("out_of_view")
    if roi is not None:
        x0, y0, x1, y1 = roi
        roi_boundary = bool(
            np.any(component_x <= x0 + margin)
            or np.any(component_x >= x1 - 1 - margin)
            or np.any(component_y <= y0 + margin)
            or np.any(component_y >= y1 - 1 - margin)
            or (
                len(points) >= 2
                and (
                    np.any(points[0] <= [x0 + margin, y0 + margin])
                    or np.any(points[0] >= [x1 - 1 - margin, y1 - 1 - margin])
                    or np.any(points[-1] <= [x0 + margin, y0 + margin])
                    or np.any(points[-1] >= [x1 - 1 - margin, y1 - 1 - margin])
                )
            )
        )
        if roi_boundary:
            flags.append("roi_clipped")
    if quality < config.min_quality:
        flags.append("low_quality")
    censor = any(
        flag in flags
        for flag in {
            "short_centerline", "skeleton_loss", "large_jump", "ambiguous_components",
            "components_truncated", "branched_component", "loop_component", "disconnected_skeleton",
            "out_of_view", "roi_clipped", "low_quality",
        }
    )
    topology = skeleton_topology(_skeleton_coordinates(component, image_shape, config.skeleton))
    return CenterlineCandidate(
        frame_index, time_s, filament_id, points, len(component), centroid, quality, flags,
        censor, topology, component_total, "loop_component" not in flags,
    )


def validate_centerline_rows(rows: Sequence[Mapping[str, Any]], max_jump_px: float = 80.0) -> dict[str, Any]:
    """Validate the contract and return errors/warnings without modifying rows."""

    errors: list[str] = []
    warnings: list[str] = []
    seen: set[tuple[str, float, int]] = set()
    by_filament: dict[str, list[tuple[float, int, float, float]]] = {}
    by_frame: dict[tuple[str, float], list[tuple[int, float, float]]] = {}
    previous_time: float | None = None
    required_columns = {"time", "filament_id", "point_id", "x", "y", "quality"}
    for line_number, row in enumerate(rows, start=2):
        missing = sorted(required_columns.difference(row))
        if missing:
            errors.append(f"line {line_number}: missing required columns {','.join(missing)}")
            continue
        try:
            time_s = float(row["time"])
            point_id = int(row["point_id"])
            x = float(row["x"])
            y = float(row["y"])
            quality = float(row["quality"])
        except (KeyError, TypeError, ValueError) as exc:
            errors.append(f"line {line_number}: invalid required value ({exc})")
            continue
        filament = str(row.get("filament_id", ""))
        if not np.isfinite([time_s, x, y, quality]).all():
            errors.append(f"line {line_number}: non-finite time, coordinate, or quality")
        if not 0.0 <= quality <= 1.0:
            warnings.append(f"line {line_number}: quality is outside [0,1]")
        if previous_time is not None and time_s < previous_time - 1e-12:
            errors.append(f"line {line_number}: time is not monotonic")
        previous_time = time_s
        key = (filament, time_s, point_id)
        if key in seen:
            errors.append(f"line {line_number}: duplicate point_id within time/filament")
        seen.add(key)
        by_filament.setdefault(filament, []).append((time_s, point_id, x, y))
        by_frame.setdefault((filament, time_s), []).append((point_id, x, y))
    for filament, values in by_filament.items():
        values.sort(key=lambda item: (item[0], item[1]))
        for (frame_filament, frame_time), frame_values in sorted(by_frame.items()):
            if frame_filament != filament:
                continue
            point_ids = [item[0] for item in sorted(frame_values)]
            if point_ids != list(range(len(point_ids))):
                errors.append(f"filament {filament}: point_id is not strictly ordered at time {frame_time}")
        summaries = []
        for frame_time in sorted({item[0] for item in values}):
            frame_values = by_frame[(filament, frame_time)]
            summaries.append((frame_time, float(np.mean([item[1] for item in frame_values])), float(np.mean([item[2] for item in frame_values]))))
        previous: tuple[float, float, float] | None = None
        for frame_time, centroid_x, centroid_y in summaries:
            if previous is not None:
                jump = math.hypot(centroid_x - previous[1], centroid_y - previous[2])
                if jump > max_jump_px:
                    warnings.append(f"filament {filament}: centroid jump {jump:.3f}px near time {frame_time}")
            previous = (frame_time, centroid_x, centroid_y)
    return {
        "valid": not errors,
        "errors": errors,
        "warnings": warnings,
        "n_rows": len(rows),
        "n_filaments": len(by_filament),
        "time_monotonic": not any("time is not monotonic" in error for error in errors),
        "finite_coordinates": not any("non-finite" in error for error in errors),
    }


def validate_centerline_csv(path: str | Path, max_jump_px: float = 80.0) -> dict[str, Any]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return validate_centerline_rows(list(csv.DictReader(handle)), max_jump_px=max_jump_px)


def _choose_filament(summary_rows: Sequence[Mapping[str, Any]]) -> str | None:
    counts: dict[str, int] = {}
    for row in summary_rows:
        name = str(row.get("filament_id", ""))
        counts[name] = counts.get(name, 0) + 1
    if not counts:
        return None
    return sorted(counts, key=lambda name: (-counts[name], name))[0]


def _file_record(path: str | Path) -> dict[str, Any]:
    value = Path(path)
    return {"path": value.name, "bytes": value.stat().st_size, "sha256": sha256_file(value)}


def _runtime_capabilities() -> dict[str, Any]:
    result: dict[str, Any] = {
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "opencv_used": False,
    }
    optional_modules = ("imageio", "PIL", "marimo", "skimage")
    for module_name in optional_modules:
        try:
            module = __import__(module_name)
            result[module_name] = getattr(module, "__version__", "installed")
        except Exception as exc:
            result[module_name] = f"unavailable:{type(exc).__name__}"
    for executable in ("ffmpeg", "ffprobe"):
        result[executable] = {"available": shutil.which(executable) is not None}
        if shutil.which(executable):
            try:
                result[executable]["version"] = _run_command([executable, "-version"]).splitlines()[0]
            except Exception as exc:
                result[executable]["version_error"] = f"{type(exc).__name__}: {exc}"
    if shutil.which("ffmpeg"):
        try:
            encoders = _run_command(["ffmpeg", "-hide_banner", "-encoders"])
            result["libx264"] = {"available": "libx264" in encoders}
        except Exception as exc:
            result["libx264"] = {"available": False, "error": f"{type(exc).__name__}: {exc}"}
    else:
        result["libx264"] = {"available": False}
    return result


def run_pipeline(
    video_path: str | Path,
    output_dir: str | Path,
    config: SegmentationConfig | Mapping[str, Any] | None = None,
    *,
    max_frames: int | None = None,
    command_line: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Segment a video and write contract, QC, metadata, and manifest artifacts."""

    cfg = config if isinstance(config, SegmentationConfig) else SegmentationConfig.from_mapping(config)
    source = Path(video_path).expanduser().resolve()
    destination = Path(output_dir).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    metadata = probe_video(source)
    tracker = _TrackManager(cfg.max_jump_px)
    previous_points: dict[str, np.ndarray] = {}
    candidates: list[CenterlineCandidate] = []
    events: list[dict[str, Any]] = []
    lineage_rows: list[dict[str, Any]] = []
    threshold_values: list[float] = []
    component_counts: list[int] = []
    processed_frame_indices: list[int] = []
    processed_times: list[float] = []
    for observation_order, (frame_index, time_s, frame) in enumerate(iter_video_frames(source, metadata, cfg.frame_stride, max_frames)):
        processed_frame_indices.append(frame_index)
        processed_times.append(time_s)
        mask, diagnostics = segment_mask(frame, cfg)
        threshold_values.append(float(diagnostics["threshold"]))
        all_components = connected_components(mask, cfg.min_component_size)
        component_total = len(all_components)
        components = all_components[: cfg.max_components]
        component_counts.append(component_total)
        if component_total > cfg.max_components:
            events.append({"frame": frame_index, "time": time_s, "event": "components_truncated", "severity": "censor", "details": f"total={component_total};kept={len(components)}"})
        if not components:
            events.append({"frame": frame_index, "time": time_s, "event": "missing", "severity": "censor", "details": "no_component"})
            for name in sorted(tracker.last):
                lineage_rows.append({"frame": frame_index, "time": time_s, "filament_id": name, "status": "missing", "censor": 1, "details": "no_component"})
            continue
        centroids = [np.mean(component, axis=0)[::-1] for component in components]
        previous_track_ids = set(tracker.last)
        filament_ids, jumps, track_statuses = tracker.assign(centroids, frame_index, observation_order)
        assigned_track_ids = set(filament_ids)
        for name in sorted(previous_track_ids - assigned_track_ids):
            lineage_rows.append({"frame": frame_index, "time": time_s, "filament_id": name, "status": "missing", "censor": 1, "details": "component_not_assigned"})
            events.append({"frame": frame_index, "time": time_s, "event": "missing", "severity": "censor", "details": name})
        if component_total > 1:
            events.append({"frame": frame_index, "time": time_s, "event": "ambiguous_components", "severity": "warning", "details": str(component_total)})
        for component, filament_id, jump, track_status in zip(components, filament_ids, jumps, track_statuses):
            candidate = _make_candidate(
                frame_index,
                time_s,
                filament_id,
                component,
                frame.shape,
                cfg,
                jump,
                len(components),
                component_total,
                _parse_roi(cfg.roi),
            )
            if track_status == "reconnected_after_missing":
                candidate.flags.append("reconnected_after_missing")
                candidate.censor = True
                events.append({"frame": frame_index, "time": time_s, "event": "reconnected_after_missing", "severity": "censor", "details": filament_id})
            elif track_status == "new_lineage":
                candidate.flags.append("new_lineage")
                candidate.censor = True
                events.append({"frame": frame_index, "time": time_s, "event": "new_lineage", "severity": "censor", "details": filament_id})
            lineage_rows.append({"frame": frame_index, "time": time_s, "filament_id": filament_id, "status": track_status, "censor": int(candidate.censor), "details": ""})
            previous = previous_points.get(filament_id)
            if previous is not None and len(previous) >= 2 and len(candidate.points) >= 2:
                direct = float(np.linalg.norm(candidate.points[0] - previous[0]) + np.linalg.norm(candidate.points[-1] - previous[-1]))
                reversed_distance = float(np.linalg.norm(candidate.points[-1] - previous[0]) + np.linalg.norm(candidate.points[0] - previous[-1]))
                if reversed_distance < direct:
                    candidate.points = candidate.points[::-1].copy()
                    candidate.flags.append("orientation_reversed")
            previous_points[filament_id] = candidate.points.copy()
            candidates.append(candidate)
            for flag in candidate.flags:
                severity = "censor" if candidate.censor else "warning"
                events.append({"frame": frame_index, "time": time_s, "event": flag, "severity": severity, "details": filament_id})
    centerline_rows = [row for candidate in candidates for row in _candidate_rows(candidate)]
    summary_rows = [candidate.summary() for candidate in candidates]
    validation = validate_centerline_rows(centerline_rows, max_jump_px=cfg.max_jump_px)
    if not validation["valid"]:
        events.append({"frame": -1, "time": "", "event": "contract_invalid", "severity": "error", "details": ";".join(validation["errors"])})
    selected_filament = _choose_filament(summary_rows)
    centerline_fields = [
        "time", "filament_id", "point_id", "x", "y", "quality", "frame",
        "coordinate_system", "quality_flags", "censor",
    ]
    summary_fields = [
        "frame", "time", "filament_id", "n_points", "centerline_exported", "component_area",
        "component_count_total", "endpoint_count", "junction_count", "cycle_rank", "length_px",
        "endpoint_distance_px", "curvature_mean_px_inv", "curvature_max_px_inv", "quality",
        "quality_flags", "censor",
    ]
    lineage_fields = ["frame", "time", "filament_id", "status", "censor", "details"]
    _write_csv(destination / "centerline.csv", centerline_rows, centerline_fields)
    _write_csv(destination / "observation_summary.csv", summary_rows, summary_fields)
    _write_csv(destination / "events.csv", events, ["frame", "time", "event", "severity", "details"])
    _write_csv(destination / "lineage.csv", lineage_rows, lineage_fields)
    coverage = "full_period" if max_frames is None else "partial_max_frames"
    frame_range = {
        "first": min(processed_frame_indices) if processed_frame_indices else None,
        "last": max(processed_frame_indices) if processed_frame_indices else None,
        "count": len(processed_frame_indices),
        "stride": cfg.frame_stride,
        "coverage": coverage,
    }
    command_value = []
    for token in command_line or []:
        text = str(token)
        if text == str(source):
            text = "${INPUT_VIDEO}"
        else:
            try:
                is_output_path = Path(text).expanduser().resolve() == destination
            except (OSError, RuntimeError):
                is_output_path = False
            if text == str(destination) or is_output_path:
                text = "${OUTPUT_DIR}"
        command_value.append(text)
    config_hash = sha256_text(canonical_json(cfg.to_dict()))
    metadata_json = {
        "schema_version": SCHEMA_VERSION,
        "video": metadata.to_dict(),
        "input_logical_id": source.name,
        "coordinate_system": "pixel",
        "time_unit": "s",
        "data_contract": ["time", "filament_id", "point_id", "x", "y", "quality"],
        "segmentation": cfg.to_dict(),
        "selected_filament_id": selected_filament,
        "quality_definition": "deterministic area/centerline-point score; flags and censor are authoritative",
        "calibration": {"pixel_per_model_unit": None, "status": "not_specified"},
        "run": {"max_frames": max_frames, "frame_range": frame_range, "coverage": coverage},
        "validation": validation,
        "limitations": [
            "frame-local segmentation candidates are not proof of a single biological filament lineage",
            "pixel coordinates are not physical coordinates",
            "quality/censor flags identify intervals requiring review",
        ],
    }
    (destination / "metadata.json").write_text(canonical_json(metadata_json) + "\n", encoding="utf-8")
    artifact_paths = [
        destination / "metadata.json", destination / "centerline.csv", destination / "observation_summary.csv",
        destination / "events.csv", destination / "lineage.csv",
    ]
    artifacts = {path.stem: _file_record(path) for path in artifact_paths}
    artifact_bytes = sum(item["bytes"] for item in artifacts.values())
    budget_status = "not_configured"
    if cfg.output_budget_bytes is not None:
        budget_status = "within_budget" if artifact_bytes <= cfg.output_budget_bytes else "exceeded"
    runtime = _runtime_capabilities()
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "input": {
            "logical_id": source.name,
            "path": str(source),
            "sha256": sha256_file(source),
            "bytes": source.stat().st_size,
            "metadata": metadata.to_dict(),
        },
        "segmentation_config": cfg.to_dict(),
        "config_sha256": config_hash,
        "command": command_value,
        "command_sha256": sha256_text(canonical_json(command_value)),
        "run": {"max_frames": max_frames, "frame_range": frame_range, "coverage": coverage},
        "validation": validation,
        "selected_filament_id": selected_filament,
        "processed_frames": len(processed_frame_indices),
        "candidate_count": len(candidates),
        "candidate_censor_count": sum(int(candidate.censor) for candidate in candidates),
        "threshold_summary": {
            "min": min(threshold_values) if threshold_values else None,
            "max": max(threshold_values) if threshold_values else None,
            "mean": float(np.mean(threshold_values)) if threshold_values else None,
        },
        "component_count_summary": {
            "max": max(component_counts) if component_counts else 0,
            "mean": float(np.mean(component_counts)) if component_counts else 0.0,
        },
        "runtime": runtime,
        "artifacts": artifacts,
        "output_budget": {"bytes": cfg.output_budget_bytes, "artifact_bytes": artifact_bytes, "status": budget_status},
        "comparison_artifacts": {},
    }
    (destination / "manifest.json").write_text(canonical_json(manifest) + "\n", encoding="utf-8")
    return {
        "output_dir": str(destination),
        "metadata": metadata.to_dict(),
        "manifest": manifest,
        "validation": validation,
        "candidates": candidates,
        "summary_rows": summary_rows,
        "events": events,
    }


# ---------------------------------------------------------------------------
# Model loading, comparison, and visualisation


@dataclass
class ModelFrame:
    time_s: float
    points: np.ndarray
    source: str
    metrics: dict[str, float]


@dataclass(frozen=True)
class ModelMatch:
    frame: ModelFrame
    time_error_s: float
    method: str
    tolerance_s: float


def _model_polyline_metrics(points: np.ndarray) -> dict[str, float]:
    values = polyline_metrics(points)
    return {
        "length_model": values["length_px"],
        "endpoint_distance_model": values["endpoint_distance_px"],
        "curvature_mean_model_inv": values["curvature_mean_px_inv"],
        "curvature_max_model_inv": values["curvature_max_px_inv"],
    }


def load_model_output(path: str | Path) -> list[ModelFrame]:
    """Load the existing trajectory serializer, a centreline CSV, or JSON.

    A metrics-only summary is accepted as a model source but has no centerline;
    callers then receive a clear ``model_centerline_unavailable`` reason.
    """

    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(source)
    if source.suffix.lower() == ".npz":
        with np.load(source, allow_pickle=False) as archive:
            times = np.asarray(archive["times"], dtype=float)
            positions = np.asarray(archive["positions"], dtype=float)
            offsets = np.asarray(archive["position_offsets"], dtype=int)
        return [
            ModelFrame(float(time), positions[offsets[i] : offsets[i + 1]], str(source), _model_polyline_metrics(positions[offsets[i] : offsets[i + 1]]))
            for i, time in enumerate(times)
        ]
    if source.suffix.lower() == ".csv":
        with source.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        groups: dict[float, list[tuple[int, float, float]]] = {}
        for row in rows:
            if not {"time", "x", "y"}.issubset(row):
                continue
            time_s = float(row["time"])
            point_id = int(row.get("point_id", len(groups.get(time_s, []))))
            groups.setdefault(time_s, []).append((point_id, float(row["x"]), float(row["y"])))
        result = []
        for time_s in sorted(groups):
            points = np.asarray([[x, y] for _, x, y in sorted(groups[time_s])], dtype=float)
            result.append(ModelFrame(time_s, points, str(source), _model_polyline_metrics(points)))
        return result
    value = json.loads(source.read_text(encoding="utf-8"))
    if isinstance(value, dict) and isinstance(value.get("trajectory"), list):
        frames = value["trajectory"]
    elif isinstance(value, dict) and isinstance(value.get("frames"), list):
        frames = value["frames"]
    else:
        # Metrics-only summaries are still useful for a report.
        return []
    result = []
    for frame in frames:
        time_s = float(frame.get("time", frame.get("time_s", 0.0)))
        points = np.asarray(frame.get("points", frame.get("positions", [])), dtype=float)
        if points.ndim == 2 and points.shape[1] == 2 and len(points):
            result.append(ModelFrame(time_s, points, str(source), _model_polyline_metrics(points)))
    return result


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def match_model_frame(frames: Sequence[ModelFrame], time_s: float, max_error: float) -> ModelMatch | None:
    """Match, rather than interpolate, the nearest model frame within tolerance."""

    if not frames:
        return None
    index = min(range(len(frames)), key=lambda i: abs(frames[i].time_s - time_s))
    error = abs(float(frames[index].time_s) - float(time_s))
    if error > max_error:
        return None
    return ModelMatch(frames[index], error, "nearest_frame", float(max_error))


def endpoint_correspondence(
    observed_points: np.ndarray,
    model_points_px: np.ndarray,
    order: str = "auto",
) -> dict[str, Any]:
    """Evaluate both endpoint orientations and select an explicit correspondence."""

    if len(observed_points) < 2 or len(model_points_px) < 2:
        return {
            "selected_method": None,
            "forward_endpoint_distance_px": None,
            "reverse_endpoint_distance_px": None,
            "selected_endpoint_distance_px": None,
            "model_points": model_points_px,
        }
    forward = float(
        0.5
        * (
            np.linalg.norm(observed_points[0] - model_points_px[0])
            + np.linalg.norm(observed_points[-1] - model_points_px[-1])
        )
    )
    reverse = float(
        0.5
        * (
            np.linalg.norm(observed_points[0] - model_points_px[-1])
            + np.linalg.norm(observed_points[-1] - model_points_px[0])
        )
    )
    if order == "forward":
        selected = "forward"
    elif order == "reverse":
        selected = "reverse"
    else:
        selected = "forward" if forward <= reverse else "reverse"
    return {
        "selected_method": selected,
        "forward_endpoint_distance_px": forward,
        "reverse_endpoint_distance_px": reverse,
        "selected_endpoint_distance_px": forward if selected == "forward" else reverse,
        "model_points": model_points_px if selected == "forward" else model_points_px[::-1].copy(),
    }


def _resample_for_rmse(points: np.ndarray, n: int = 80) -> np.ndarray:
    if len(points) == n:
        return points
    if len(points) < 2:
        return np.repeat(points[:1], n, axis=0) if len(points) else np.zeros((n, 2))
    distances = np.concatenate(([0.0], np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))))
    target = np.linspace(0.0, float(distances[-1]), n)
    return np.column_stack([np.interp(target, distances, points[:, axis]) for axis in range(2)])


def compare_with_model(
    observation_dir: str | Path,
    model_path: str | Path,
    registration: RegistrationConfig | Mapping[str, Any] | None = None,
    *,
    filament_id: str | None = None,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Compare observed pixel geometry and model geometry without implicit calibration."""

    obs_dir = Path(observation_dir)
    out_dir = Path(output_dir) if output_dir is not None else obs_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    registration_value = registration if isinstance(registration, RegistrationConfig) else RegistrationConfig.from_mapping(registration)
    summary_rows = _read_csv_rows(obs_dir / "observation_summary.csv")
    centerline_rows = _read_csv_rows(obs_dir / "centerline.csv")
    chosen = filament_id or _choose_filament(summary_rows)
    grouped: dict[tuple[int, str], list[tuple[int, float, float]]] = {}
    for row in centerline_rows:
        if chosen is not None and row.get("filament_id") != chosen:
            continue
        grouped.setdefault((int(row["frame"]), row["filament_id"]), []).append((int(row["point_id"]), float(row["x"]), float(row["y"])))
    model_frames = load_model_output(model_path)
    output_rows: list[dict[str, Any]] = []
    for row in summary_rows:
        if chosen is not None and row.get("filament_id") != chosen:
            continue
        observed_time = float(row["time"])
        registered_time = registration_value.model_time(observed_time)
        model_match = match_model_frame(model_frames, registered_time, registration_value.max_time_error_s)
        model_frame = model_match.frame if model_match is not None else None
        time_error_s = model_match.time_error_s if model_match is not None else None
        flags = [flag for flag in str(row.get("quality_flags", "")).split(";") if flag and flag != "ok"]
        censored = bool(int(row.get("censor", "0")))
        result: dict[str, Any] = {
            "frame": int(row["frame"]),
            "time": observed_time,
            "filament_id": row.get("filament_id", ""),
            "observed_length_px": float(row["length_px"]),
            "observed_endpoint_distance_px": float(row["endpoint_distance_px"]),
            "observed_curvature_mean_px_inv": float(row["curvature_mean_px_inv"]),
            "observed_curvature_max_px_inv": float(row["curvature_max_px_inv"]),
            "quality": float(row["quality"]),
            "quality_flags": ";".join(flags) if flags else "ok",
            "censor": int(censored),
            "registered_model_time": registered_time,
            "model_time": model_frame.time_s if model_frame is not None else None,
            "time_error_s": time_error_s,
            "time_match_method": model_match.method if model_match is not None else "no_match",
            "time_tolerance_s": registration_value.max_time_error_s,
            "model_length": model_frame.metrics["length_model"] if model_frame is not None else None,
            "model_endpoint_distance": model_frame.metrics["endpoint_distance_model"] if model_frame is not None else None,
            "model_curvature_mean": model_frame.metrics["curvature_mean_model_inv"] if model_frame is not None else None,
            "model_curvature_max": model_frame.metrics["curvature_max_model_inv"] if model_frame is not None else None,
            "model_length_px": None,
            "model_endpoint_distance_px": None,
            "endpoint_correspondence_method": None,
            "forward_endpoint_distance_px": None,
            "reverse_endpoint_distance_px": None,
            "selected_endpoint_distance_px": None,
            "endpoint_distance_px": None,
            "shape_rmse_px": None,
            "length_difference_px": None,
            "curvature_difference_model_units": None,
            "metric_status": "not_computed_uncalibrated",
            "metric_reason": "pixel_per_model_unit_not_specified",
        }
        points_rows = sorted(grouped.get((int(row["frame"]), row.get("filament_id", "")), []))
        observed_points = np.asarray([[x, y] for _, x, y in points_rows], dtype=float)
        if model_frame is None:
            result["metric_status"] = "not_computed_model_unmatched"
            result["metric_reason"] = "model_time_unmatched_or_centerline_unavailable"
            result["censor"] = 1
            if "model_time_unmatched" not in flags:
                result["quality_flags"] = ";".join(flags + ["model_time_unmatched"]) if flags else "model_time_unmatched"
        elif censored:
            result["metric_status"] = "not_computed_censored"
            result["metric_reason"] = "quality_censor_flag"
            result["censor"] = 1
        elif registration_value.calibrated and len(observed_points) >= 2:
            model_pixels = registration_value.model_to_pixel(model_frame.points)
            correspondence = endpoint_correspondence(observed_points, model_pixels, registration_value.endpoint_order)
            selected_model_pixels = correspondence["model_points"]
            observed_resampled = _resample_for_rmse(observed_points)
            model_resampled = _resample_for_rmse(selected_model_pixels)
            result.update(
                {
                    "model_length_px": model_frame.metrics["length_model"] * float(registration_value.pixel_per_model_unit),
                    "model_endpoint_distance_px": model_frame.metrics["endpoint_distance_model"] * float(registration_value.pixel_per_model_unit),
                    "endpoint_correspondence_method": correspondence["selected_method"],
                    "forward_endpoint_distance_px": correspondence["forward_endpoint_distance_px"],
                    "reverse_endpoint_distance_px": correspondence["reverse_endpoint_distance_px"],
                    "selected_endpoint_distance_px": correspondence["selected_endpoint_distance_px"],
                    "endpoint_distance_px": correspondence["selected_endpoint_distance_px"],
                    "shape_rmse_px": float(np.sqrt(np.mean(np.sum((observed_resampled - model_resampled) ** 2, axis=1)))),
                    "length_difference_px": float(row["length_px"]) - model_frame.metrics["length_model"] * float(registration_value.pixel_per_model_unit),
                    "curvature_difference_model_units": float(row["curvature_mean_px_inv"]) * float(registration_value.pixel_per_model_unit) - model_frame.metrics["curvature_mean_model_inv"],
                    "metric_status": "computed",
                    "metric_reason": "",
                }
            )
        elif model_frame is not None:
            result["metric_reason"] = "pixel_per_model_unit_not_specified" if not registration_value.calibrated else "observed_centerline_missing"
            result["censor"] = 1
            if not registration_value.calibrated:
                result["quality_flags"] = ";".join(flags + ["uncalibrated_comparison"]) if flags else "uncalibrated_comparison"
        output_rows.append(result)
    fields = [
        "frame", "time", "filament_id", "observed_length_px", "observed_endpoint_distance_px",
        "observed_curvature_mean_px_inv", "observed_curvature_max_px_inv", "quality", "quality_flags", "censor",
        "registered_model_time", "model_time", "time_error_s", "time_match_method", "time_tolerance_s",
        "model_length", "model_endpoint_distance", "model_curvature_mean", "model_curvature_max",
        "model_length_px", "model_endpoint_distance_px", "endpoint_correspondence_method",
        "forward_endpoint_distance_px", "reverse_endpoint_distance_px", "selected_endpoint_distance_px",
        "endpoint_distance_px", "shape_rmse_px", "length_difference_px", "curvature_difference_model_units",
        "metric_status", "metric_reason",
    ]
    _write_csv(out_dir / "comparison.csv", output_rows, fields)
    calibration_status = "calibrated" if registration_value.calibrated else "not_calibrated_metrics_suppressed"
    compact = {
        "schema_version": SCHEMA_VERSION,
        "observation_dir": str(obs_dir.resolve()),
        "model_logical_id": Path(model_path).name,
        "model_path": str(Path(model_path).resolve()),
        "model_sha256": sha256_file(model_path),
        "model_bytes": Path(model_path).stat().st_size,
        "registration": registration_value.to_dict(),
        "calibration_status": calibration_status,
        "filament_id": chosen,
        "rows": len(output_rows),
        "eligible_rows": sum(row["metric_status"] == "computed" for row in output_rows),
        "computed_rows": sum(row["metric_status"] == "computed" for row in output_rows),
        "censored_rows": sum(bool(row["censor"]) for row in output_rows),
        "excluded_from_metric_denominator": sum(row["metric_status"] != "computed" for row in output_rows),
        "not_computed_reason_counts": {
            reason: sum(row["metric_reason"] == reason for row in output_rows)
            for reason in sorted({row["metric_reason"] for row in output_rows if row["metric_reason"]})
        },
        "limitations": [
            "registration is configuration, not an inferred calibration",
            "a selected candidate is not proof of single-filament tracking",
        ],
    }
    (out_dir / "comparison.json").write_text(canonical_json(compact) + "\n", encoding="utf-8")
    comparison_artifacts = {
        "comparison_csv": _file_record(out_dir / "comparison.csv"),
        "comparison_json": _file_record(out_dir / "comparison.json"),
    }
    comparison_manifest = {
        "schema_version": SCHEMA_VERSION,
        "observation_logical_id": obs_dir.name,
        "model_logical_id": Path(model_path).name,
        "registration": registration_value.to_dict(),
        "calibration_status": calibration_status,
        "artifacts": comparison_artifacts,
        "eligible_rows": compact["eligible_rows"],
        "excluded_from_metric_denominator": compact["excluded_from_metric_denominator"],
    }
    (out_dir / "comparison_manifest.json").write_text(canonical_json(comparison_manifest) + "\n", encoding="utf-8")
    return {"rows": output_rows, "summary": compact, "model_frames": model_frames, "filament_id": chosen}


def _draw_line(draw: Any, points: np.ndarray, colour: tuple[int, int, int], width: int = 2) -> None:
    if len(points) >= 2:
        draw.line([tuple(map(float, point)) for point in points], fill=colour, width=width, joint="curve")


def _model_canvas_points(points: np.ndarray, width: int, height: int, bounds: tuple[float, float, float, float] | None) -> np.ndarray:
    if bounds is None or len(points) == 0:
        return points
    x0, y0, x1, y1 = bounds
    scale = min((width - 50) / max(x1 - x0, 1e-12), (height - 50) / max(y1 - y0, 1e-12))
    return np.column_stack(((points[:, 0] - x0) * scale + 25, (points[:, 1] - y0) * scale + 25))


def render_comparison(
    video_path: str | Path,
    observation_dir: str | Path,
    model_path: str | Path,
    output_dir: str | Path,
    registration: RegistrationConfig | Mapping[str, Any] | None = None,
    *,
    max_video_frames: int | None = None,
    representative_count: int = 6,
    output_budget_bytes: int | None = None,
) -> dict[str, Any]:
    """Render side-by-side frames/video; never writes them into the repository by default."""

    from PIL import Image, ImageDraw

    obs_dir = Path(observation_dir)
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    registration_value = registration if isinstance(registration, RegistrationConfig) else RegistrationConfig.from_mapping(registration)
    comparison = compare_with_model(obs_dir, model_path, registration_value, output_dir=destination)
    rows_by_frame = {int(row["frame"]): row for row in comparison["rows"]}
    point_rows = _read_csv_rows(obs_dir / "centerline.csv")
    points_by_frame: dict[int, np.ndarray] = {}
    for frame in sorted({int(row["frame"]) for row in point_rows}):
        values = [row for row in point_rows if int(row["frame"]) == frame and row.get("filament_id") == comparison["filament_id"]]
        values.sort(key=lambda row: int(row["point_id"]))
        points_by_frame[frame] = np.asarray([[float(row["x"]), float(row["y"])] for row in values])
    model_frames: list[ModelFrame] = comparison["model_frames"]
    all_model_points = np.concatenate([frame.points for frame in model_frames if len(frame.points)], axis=0) if model_frames and any(len(frame.points) for frame in model_frames) else np.empty((0, 2))
    bounds = None
    if len(all_model_points):
        bounds = (float(np.min(all_model_points[:, 0])), float(np.min(all_model_points[:, 1])), float(np.max(all_model_points[:, 0])), float(np.max(all_model_points[:, 1])))
    meta = probe_video(video_path)
    observation_metadata = json.loads((obs_dir / "metadata.json").read_text(encoding="utf-8"))
    observation_config = SegmentationConfig.from_mapping(observation_metadata["segmentation"])
    frame_images = destination / "frames"
    frame_images.mkdir(exist_ok=True)
    representative_targets: set[int] = set()
    candidate_frames = sorted(rows_by_frame)
    if candidate_frames:
        representative_indices = np.linspace(
            0, len(candidate_frames) - 1, min(representative_count, len(candidate_frames)), dtype=int
        )
        representative_targets.update(candidate_frames[int(index)] for index in representative_indices)
    output_video = destination / "comparison.mp4"
    runtime = _runtime_capabilities()
    if not runtime["ffmpeg"]["available"]:
        raise RuntimeError("comparison video requires ffmpeg; run `ffmpeg -version` to inspect the installation")
    if not runtime["libx264"]["available"]:
        raise RuntimeError("comparison video requires the libx264 encoder; run `ffmpeg -hide_banner -encoders | grep 264`")
    ffmpeg_command = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
        "-s", f"{meta.width * 2}x{meta.height}", "-r", f"{meta.fps / max(1, observation_config.frame_stride):.9f}", "-i", "-",
        "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p", str(output_video),
    ]
    process = subprocess.Popen(ffmpeg_command, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    written = 0
    try:
        for frame_index, time_s, frame in iter_video_frames(video_path, meta, observation_config.frame_stride, max_video_frames):
            base = Image.fromarray(frame, mode="L").convert("RGB")
            left = base.copy()
            left_draw = ImageDraw.Draw(left)
            observed_points = points_by_frame.get(frame_index)
            if observed_points is not None:
                _draw_line(left_draw, observed_points, (255, 50, 30), 3)
            row = rows_by_frame.get(frame_index)
            model_match = match_model_frame(
                model_frames,
                registration_value.model_time(time_s),
                registration_value.max_time_error_s,
            )
            model_frame = model_match.frame if model_match is not None else None
            if model_frame is not None and registration_value.calibrated:
                _draw_line(left_draw, registration_value.model_to_pixel(model_frame.points), (40, 220, 255), 2)
            left_draw.text((8, 8), f"observed pixel  frame={frame_index}  t={time_s:.3f}s", fill=(255, 255, 0))
            right = Image.new("RGB", (meta.width, meta.height), (245, 245, 245))
            right_draw = ImageDraw.Draw(right)
            if model_frame is not None:
                _draw_line(right_draw, _model_canvas_points(model_frame.points, meta.width, meta.height, bounds), (30, 100, 220), 3)
            right_draw.text(
                (8, 8),
                "model unit" + (" + registered" if registration_value.calibrated else " (uncalibrated)"),
                fill=(10, 10, 10),
            )
            if row is not None:
                text = f"Lobs={row['observed_length_px']:.1f}px  q={row['quality']}  censor={row['censor']}"
                right_draw.text((8, 28), text, fill=(10, 10, 10))
                right_draw.text((8, 48), str(row["metric_status"]), fill=(10, 10, 10))
                right_draw.text((8, 68), f"time_error={row.get('time_error_s', '')} method={row.get('time_match_method', '')}", fill=(10, 10, 10))
                right_draw.text((8, 88), f"reason={row.get('metric_reason', '')}", fill=(10, 10, 10))
            combined = np.asarray(Image.fromarray(np.hstack((np.asarray(left), np.asarray(right)))))
            assert process.stdin is not None
            process.stdin.write(combined.tobytes())
            written += 1
            if frame_index in representative_targets:
                Image.fromarray(combined).save(frame_images / f"frame_{frame_index:06d}.png")
    finally:
        if process.stdin is not None:
            process.stdin.close()
        stderr = process.stderr.read().decode("utf-8", errors="replace") if process.stderr else ""
        return_code = process.wait()
        if return_code != 0:
            # A build without libx264 may still have another encoder.  Retry is
            # intentionally explicit rather than silently producing no video.
            raise RuntimeError(f"comparison video encoding failed: {stderr.strip()}")
    frame_artifacts = {
        path.name: _file_record(path)
        for path in sorted(frame_images.glob("*.png"))
    }
    rendered_artifacts = {
        "comparison_video": _file_record(output_video),
        "comparison_csv": _file_record(destination / "comparison.csv"),
        "comparison_json": _file_record(destination / "comparison.json"),
        "comparison_manifest": _file_record(destination / "comparison_manifest.json"),
        "representative_frames": frame_artifacts,
    }
    rendered_bytes = sum(item["bytes"] for key, item in rendered_artifacts.items() if key != "representative_frames") + sum(item["bytes"] for item in frame_artifacts.values())
    budget_status = "not_configured" if output_budget_bytes is None else ("within_budget" if rendered_bytes <= output_budget_bytes else "exceeded")
    render_manifest = {
        "schema_version": SCHEMA_VERSION,
        "video_logical_id": Path(video_path).name,
        "video": str(Path(video_path).resolve()),
        "frames_written": written,
        "max_video_frames": max_video_frames,
        "coverage": "full_period" if max_video_frames is None else "partial_max_frames",
        "output_video": str(output_video),
        "representative_frames": sorted(representative_targets),
        "calibration_status": comparison["summary"]["calibration_status"],
        "runtime": runtime,
        "artifacts": rendered_artifacts,
        "output_budget": {"bytes": output_budget_bytes, "artifact_bytes": rendered_bytes, "status": budget_status},
    }
    (destination / "render_manifest.json").write_text(canonical_json(render_manifest) + "\n", encoding="utf-8")
    return {"output_dir": str(destination), "comparison": comparison, "render_manifest": render_manifest}


__all__ = [
    "SCHEMA_VERSION",
    "SegmentationConfig",
    "RegistrationConfig",
    "VideoMetadata",
    "CenterlineCandidate",
    "probe_video",
    "iter_video_frames",
    "segment_mask",
    "connected_components",
    "component_to_centerline",
    "skeleton_topology",
    "polyline_metrics",
    "validate_centerline_rows",
    "validate_centerline_csv",
    "run_pipeline",
    "ModelFrame",
    "ModelMatch",
    "load_model_output",
    "match_model_frame",
    "endpoint_correspondence",
    "compare_with_model",
    "render_comparison",
    "sha256_file",
]
