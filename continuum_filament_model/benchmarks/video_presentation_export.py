"""Create compact presentation data from an observed filament video.

The extractor delegates segmentation and lineage handling to
``growing_filament.video_comparison.run_pipeline``.  It never imports or uses
OpenCV.  For an available input it additionally exports representative raw
centerline coordinates and an arc-length curvature profile.  When the
requested source is absent (as in a clean checkout without the licensed
observation asset), it writes an explicit ``input_missing`` manifest rather
than fabricating observation data.
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
else:
    _HERE = Path(__file__).resolve()

from growing_filament.reproducibility import detect_git_revision  # noqa: E402
from growing_filament.video_comparison import (  # noqa: E402
    RegistrationConfig,
    SegmentationConfig,
    canonical_json,
    compare_with_model,
    run_pipeline,
)

SCHEMA_VERSION = "continuum-filament-video-presentation-1"


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(canonical_json(value) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _logical_video_id(source: Path) -> str:
    return "img/gray5.mp4" if source.name == "gray5.mp4" else source.name


def _normalise_pipeline_artifacts(output: Path, logical_id: str, source_revision: str | None) -> dict[str, Any]:
    """Remove local absolute input paths from the generated pipeline artifacts."""

    metadata_path = output / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    video_metadata = dict(metadata.get("video", {}))
    video_metadata["path"] = logical_id
    metadata["video"] = video_metadata
    metadata["input_logical_id"] = logical_id
    metadata["source_revision"] = source_revision
    _write_json(metadata_path, metadata)

    manifest_path = output / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["source_revision"] = source_revision
    manifest["input"] = dict(manifest.get("input", {}))
    manifest["input"]["logical_id"] = logical_id
    manifest["input"]["path"] = logical_id
    manifest["input"]["metadata"] = video_metadata
    for record in manifest.get("artifacts", {}).values():
        artifact_path = output / str(record.get("path", ""))
        if artifact_path.is_file():
            record["bytes"] = artifact_path.stat().st_size
            record["sha256"] = _sha256(artifact_path)
    _write_json(manifest_path, manifest)
    return manifest


def _normalise_comparison_paths(output: Path) -> None:
    """Keep optional model-comparison JSON portable as well."""

    path = output / "comparison.json"
    if not path.is_file():
        return
    value = json.loads(path.read_text(encoding="utf-8"))
    value["observation_dir"] = "continuum_filament_model/results/presentation_data/video_gray5"
    if value.get("model_path"):
        value["model_path"] = Path(str(value["model_path"])).name
    _write_json(path, value)


def _curvature_profile(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    points = np.asarray(points, dtype=float)
    if len(points) < 3:
        return np.zeros(len(points), dtype=float), np.zeros(len(points), dtype=float)
    vectors = np.diff(points, axis=0)
    lengths = np.linalg.norm(vectors, axis=1)
    distance = np.concatenate(([0.0], np.cumsum(lengths)))
    curvature = np.zeros(len(points), dtype=float)
    for index in range(1, len(points) - 1):
        left, right = vectors[index - 1], vectors[index]
        left_norm, right_norm = np.linalg.norm(left), np.linalg.norm(right)
        if left_norm <= 1.0e-12 or right_norm <= 1.0e-12:
            continue
        cross = float(left[0] * right[1] - left[1] * right[0])
        dot = float(np.dot(left, right))
        angle = math.atan2(abs(cross), dot)
        curvature[index] = angle / max(0.5 * (left_norm + right_norm), 1.0e-12)
    return distance, curvature


def _read_centerline(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _presentation_records(root: Path, result: Mapping[str, Any], representative_count: int) -> dict[str, Any]:
    centerline_path = root / "centerline.csv"
    summary_path = root / "observation_summary.csv"
    rows = _read_centerline(centerline_path)
    grouped: dict[tuple[int, str], list[dict[str, Any]]] = {}
    for row in rows:
        key = (int(row["frame"]), str(row["filament_id"]))
        grouped.setdefault(key, []).append(row)
    profile_rows: list[dict[str, Any]] = []
    frame_records: list[dict[str, Any]] = []
    ordered_keys = sorted(grouped, key=lambda key: (key[0], key[1]))
    if ordered_keys:
        indexes = np.linspace(0, len(ordered_keys) - 1, min(representative_count, len(ordered_keys)), dtype=int)
        selected = [ordered_keys[int(index)] for index in sorted(set(indexes))]
    else:
        selected = []
    for frame, filament_id in ordered_keys:
        group = sorted(grouped[(frame, filament_id)], key=lambda item: int(item["point_id"]))
        points = np.asarray([[float(item["x"]), float(item["y"])] for item in group], dtype=float)
        distance, curvature = _curvature_profile(points)
        time_s = float(group[0]["time"])
        for item, arc, value in zip(group, distance, curvature):
            profile_rows.append({
                "frame": frame, "time": time_s, "filament_id": filament_id,
                "point_id": int(item["point_id"]), "arc_length_px": float(arc),
                "x": float(item["x"]), "y": float(item["y"]),
                "curvature_px_inv": float(value), "quality": float(item["quality"]),
                "censor": int(item.get("censor", 0)),
            })
        if (frame, filament_id) in selected:
            frame_records.append({
                "frame": frame, "time": time_s, "filament_id": filament_id,
                "x": points[:, 0].tolist(), "y": points[:, 1].tolist(),
                "arc_length_px": distance.tolist(), "curvature_px_inv": curvature.tolist(),
                "n_points": len(points), "length_px": float(distance[-1]) if len(distance) else 0.0,
                "quality": float(np.mean([float(item["quality"]) for item in group])),
                "censor": int(any(int(item.get("censor", 0)) for item in group)),
            })
    summary_rows: list[dict[str, Any]] = []
    with summary_path.open(newline="", encoding="utf-8") as handle:
        summary_rows = list(csv.DictReader(handle))
    length_timeseries = [
        {
            "frame": int(item["frame"]), "time": float(item["time"]),
            "filament_id": item["filament_id"], "length_px": float(item["length_px"]),
            "curvature_mean_px_inv": float(item["curvature_mean_px_inv"]),
            "curvature_max_px_inv": float(item["curvature_max_px_inv"]),
            "quality": float(item["quality"]), "censor": int(item["censor"]),
        }
        for item in summary_rows
    ]
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "extracted" if bool(rows) else "extracted_no_centerline",
        "raw_centerline_available": bool(rows),
        "selected_filament_id": result["manifest"].get("selected_filament_id"),
        "representative_frames": frame_records,
        "length_timeseries": length_timeseries,
        "curvature_profile": profile_rows,
        "validation": result["validation"],
    }


def run_export(
    video: Path,
    output: Path,
    config: SegmentationConfig | None = None,
    *,
    max_frames: int | None = None,
    representative_count: int = 6,
    model_path: Path | None = None,
    registration: Mapping[str, Any] | None = None,
    filament_id: str | None = None,
) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    source = video.expanduser().resolve()
    logical_id = _logical_video_id(source)
    source_revision = detect_git_revision(Path.cwd())
    cfg = config or SegmentationConfig(
        polarity="dark", background="local_median", contrast="percentile",
        threshold="absolute", threshold_value=0.8, frame_stride=15,
        min_component_size=30, max_components=1, max_centerline_points=200,
        output_budget_bytes=100 * 1024 * 1024,
    )
    if not source.is_file():
        payload = {
            "schema_version": SCHEMA_VERSION,
            "source_revision": source_revision,
            "status": "input_missing",
            "raw_centerline_available": False,
            "input": {"logical_id": logical_id, "path": logical_id},
            "reason": "img/gray5.mp4 is not available at extraction time; no observation rows were fabricated",
            "backend": {"opencv_used": False, "requested": "PIL/imageio/skimage-compatible existing pipeline"},
        }
        _write_json(output / "video_presentation.json", payload)
        manifest = {"schema_version": SCHEMA_VERSION, "source_revision": source_revision, "status": "input_missing", "input": payload["input"], "artifacts": {}}
        path = output / "video_presentation.json"
        manifest["artifacts"][path.name] = {"bytes": path.stat().st_size, "sha256": _sha256(path)}
        _write_json(output / "manifest.json", manifest)
        return payload
    result = run_pipeline(source, output, cfg, max_frames=max_frames, command_line=["video_presentation_export", str(source), str(output)])
    result["manifest"] = _normalise_pipeline_artifacts(output, logical_id, source_revision)
    payload = _presentation_records(output, result, representative_count)
    payload["source_revision"] = source_revision
    payload["input"] = result["manifest"]["input"]
    payload["segmentation_config"] = cfg.to_dict()
    if model_path is not None:
        comparison = compare_with_model(
            output,
            model_path,
            RegistrationConfig.from_mapping(registration),
            filament_id=filament_id,
            output_dir=output,
        )
        _normalise_comparison_paths(output)
        payload["status"] = "extracted_compared"
        payload["comparison"] = comparison["summary"]
    _write_json(output / "video_presentation.json", payload)
    compact_paths = [output / name for name in ("centerline.csv", "observation_summary.csv", "events.csv", "lineage.csv", "metadata.json", "video_presentation.json", "comparison.csv", "comparison.json", "comparison_manifest.json") if (output / name).is_file()]
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "source_revision": source_revision,
        "status": payload["status"],
        "input": result["manifest"]["input"],
        "selected_filament_id": result["manifest"].get("selected_filament_id"),
        "processed_frames": result["manifest"].get("processed_frames"),
        "opencv_used": False,
        "artifacts": {path.name: {"bytes": path.stat().st_size, "sha256": _sha256(path)} for path in compact_paths},
    }
    # Keep the full pipeline manifest generated by run_pipeline intact.  The
    # presentation manifest is a separate compact index so its artifact hashes
    # do not self-reference a file while it is being rewritten.
    _write_json(output / "presentation_manifest.json", manifest)
    return payload


def _cli() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, default=Path("img/gray5.mp4"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--representative-count", type=int, default=6)
    parser.add_argument("--model", type=Path, help="optional model trajectory/centerline for direct comparison")
    parser.add_argument("--registration", type=Path, help="optional JSON registration for pixel/model-unit metrics")
    parser.add_argument("--filament-id")
    return parser.parse_args()


def main() -> int:
    args = _cli()
    try:
        registration = json.loads(args.registration.read_text(encoding="utf-8")) if args.registration else None
        result = run_export(
            args.video,
            args.output,
            max_frames=args.max_frames,
            representative_count=args.representative_count,
            model_path=args.model,
            registration=registration,
            filament_id=args.filament_id,
        )
    except (OSError, RuntimeError, ValueError, json.JSONDecodeError) as exc:
        print(f"video presentation export error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({"output": str(args.output), "status": result["status"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
