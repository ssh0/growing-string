"""Bounded registration-independent comparison of Stage 2 runs and gray5.

This runner reuses the existing free/free Stage 2 solver and video extraction
pipeline.  It writes large trajectories and raw extraction artifacts below the
caller output directory, while root-level summary files are compact.  The
comparison itself is morphology-only: no pixel/model scale, clock mapping,
parameter fit, or model-inadequacy decision is created here.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

if __package__ in {None, ""}:  # pragma: no cover - direct-file entry point
    _HERE = Path(__file__).resolve()
    _ROOT = _HERE.parents[1]
    _SRC = _ROOT / "src"
    for _path in (_ROOT, _SRC):
        if str(_path) not in sys.path:
            sys.path.insert(0, str(_path))

if __package__ in {None, ""}:  # pragma: no cover - direct-file entry point
    from benchmarks.free_growth_buckling_stage2 import (  # noqa: E402
        _all_specs,
        load_config,
        load_config_from_mapping,
        run_case,
    )
else:
    from .free_growth_buckling_stage2 import (  # noqa: E402
        _all_specs,
        load_config,
        load_config_from_mapping,
        run_case,
    )
from growing_filament.reproducibility import canonical_json_bytes, detect_git_revision  # noqa: E402
from growing_filament.scale_free_comparison import (  # noqa: E402
    SCHEMA_VERSION,
    ScaleFreeConfig,
    scale_free_shape_comparison,
)
from growing_filament.video_comparison import (  # noqa: E402
    SegmentationConfig,
    run_pipeline,
    sha256_file,
)


RUNNER_SCHEMA_VERSION = "continuum-filament-scale-free-runner-0.1"
DEFAULT_VIDEO = Path(__file__).resolve().parents[2] / "img" / "gray5.mp4"
DEFAULT_CASES = ("fast_growth_low_bend", "fast_growth_high_bend")


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _compact_input(source: Path) -> dict[str, Any]:
    if not source.is_file():
        return {"logical_id": source.name, "sha256": None, "bytes": None, "status": "missing"}
    return {"logical_id": source.name, "sha256": sha256_file(source), "bytes": source.stat().st_size, "status": "available"}


def run_bounded_comparison(
    config: Mapping[str, Any] | None,
    output: str | Path,
    *,
    video_path: str | Path = DEFAULT_VIDEO,
    shape_config: ScaleFreeConfig | Mapping[str, Any] | None = None,
    max_frames: int | None = None,
) -> dict[str, Any]:
    """Run the intended deterministic Stage 2 cases against gray5.

    A missing or unusable video still produces a compact input-quality result;
    no substitute video or inferred registration is used.
    """

    effective = load_config_from_mapping(config) if config is not None else load_config(None)
    destination = Path(output).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    source = Path(video_path).expanduser().resolve()
    revision = detect_git_revision(Path(__file__).resolve().parents[2])
    config_hash = __import__("hashlib").sha256(canonical_json_bytes(effective)).hexdigest()
    all_specs = _all_specs(effective)
    specs = {spec.name: spec for spec in all_specs}
    configured_cases = effective.get("scale_free_cases")
    if isinstance(configured_cases, list):
        case_names = [str(name) for name in configured_cases if str(name) in specs]
    else:
        case_names = [name for name in DEFAULT_CASES if name in specs]
        if not case_names:
            case_names = [spec.name for spec in all_specs if spec.kind == "deterministic_fixture"][:2]
    model_records: list[dict[str, Any]] = []
    for case_name in case_names:
        spec = specs[case_name]
        result = run_case(spec, effective["base"], destination, revision, save_trajectory_file=True)
        trajectory_value = result.get("trajectory_path")
        model_path = destination / str(trajectory_value) if trajectory_value else None
        model_record: dict[str, Any] = {
            "run_name": case_name,
            "run_kind": result.get("run_kind"),
            "base_fixture": result.get("base_fixture"),
            "seed": result.get("seed"),
            "trial": result.get("trial"),
            "model_logical_id": model_path.name if model_path is not None else None,
            "model_sha256": sha256_file(model_path) if model_path is not None and model_path.is_file() else None,
            "model_path_external": True,
            "model_failure_reason": result.get("failure_reason"),
            "comparison": None,
        }
        model_records.append(model_record)

    artifact_dir = destination / "_video_artifacts"
    extraction: dict[str, Any]
    try:
        video_config = SegmentationConfig.from_mapping(dict(effective.get("video", {})))
        if not source.is_file():
            raise FileNotFoundError(source)
        extracted = run_pipeline(
            source,
            artifact_dir,
            video_config,
            max_frames=max_frames,
            command_line=["scale-free", "extract", "${INPUT_VIDEO}", "${OUTPUT_DIR}"],
        )
        manifest = extracted["manifest"]
        extraction = {
            "status": "extracted" if manifest.get("validation", {}).get("valid") and manifest.get("run", {}).get("frame_range", {}).get("decode_complete", True) else "unusable",
            "processed_frames": manifest.get("processed_frames"),
            "candidate_count": manifest.get("candidate_count"),
            "candidate_censor_count": manifest.get("candidate_censor_count"),
            "selected_filament_id": manifest.get("selected_filament_id"),
            "validation": manifest.get("validation"),
            "decode_complete": manifest.get("run", {}).get("frame_range", {}).get("decode_complete"),
            "external_artifact_ids": {
                "video_manifest": {"logical_id": "manifest.json", "sha256": sha256_file(artifact_dir / "manifest.json") if (artifact_dir / "manifest.json").is_file() else None},
                "centerline": {"logical_id": "centerline.csv", "sha256": manifest.get("artifacts", {}).get("centerline", {}).get("sha256")},
                "lineage": {"logical_id": "lineage.csv", "sha256": manifest.get("artifacts", {}).get("lineage", {}).get("sha256")},
            },
        }
    except FileNotFoundError:
        extraction = {
            "status": "input_quality_missing_video",
            "processed_frames": 0,
            "candidate_count": 0,
            "candidate_censor_count": 0,
            "selected_filament_id": None,
            "validation": {"valid": False, "errors": ["input_missing"]},
            "decode_complete": False,
            "external_artifact_ids": {},
        }
    except (OSError, RuntimeError, ValueError, KeyError, ImportError) as exc:
        extraction = {
            "status": "input_quality_pipeline_failure",
            "processed_frames": 0,
            "candidate_count": 0,
            "candidate_censor_count": 0,
            "selected_filament_id": None,
            "validation": {"valid": False, "errors": [type(exc).__name__]},
            "decode_complete": False,
            "external_artifact_ids": {},
        }

    shape_cfg = shape_config if isinstance(shape_config, ScaleFreeConfig) else ScaleFreeConfig.from_mapping(shape_config)
    if extraction["status"] == "extracted":
        for record in model_records:
            model_path = destination / "_runs" / record["run_name"] / "trajectory.npz" if record.get("model_logical_id") else None
            if model_path is None or not model_path.is_file():
                record["comparison"] = {
                    "status": "model_centerline_unavailable",
                    "input_quality": {"usable": False, "reasons": ["model_centerline_unavailable"]},
                    "compared_rows": 0,
                }
                continue
            case_dir = destination / "_scale_free" / record["run_name"]
            comparison = scale_free_shape_comparison(
                artifact_dir,
                model_path,
                output_dir=case_dir,
                config=shape_cfg,
                source_revision=revision,
                external_artifact_ids={
                    "video": source.name,
                    "video_manifest": "_video_artifacts/manifest.json",
                    "model_run": record["run_name"],
                },
            )
            record["comparison"] = comparison["summary"]
            record["comparison_artifact_ids"] = comparison["manifest"].get("artifacts", {})
    else:
        for record in model_records:
            record["comparison"] = {
                "status": "input_quality_comparison_suppressed",
                "input_quality": {"usable": False, "reasons": [extraction["status"]]},
                "compared_rows": 0,
            }

    comparison_rows = []
    for record in model_records:
        comparison = record.get("comparison") or {}
        comparison_rows.append({
            "run_name": record["run_name"],
            "run_kind": record["run_kind"],
            "status": comparison.get("status"),
            "eligible_observation_rows": comparison.get("eligible_observation_rows", 0),
            "compared_rows": comparison.get("compared_rows", 0),
            "censored_rows": comparison.get("censored_rows", 0),
            "model_sha256": record.get("model_sha256"),
            "input_quality_reasons": ";".join((comparison.get("input_quality") or {}).get("reasons", [])),
        })
    report: dict[str, Any] = {
        "schema_version": RUNNER_SCHEMA_VERSION,
        "comparison_schema_version": SCHEMA_VERSION,
        "comparison_mode": "scale_free_shape",
        "source_revision": revision,
        "config_sha256": config_hash,
        "shape_config_sha256": __import__("hashlib").sha256(canonical_json_bytes(shape_cfg.to_dict())).hexdigest(),
        "video": _compact_input(source),
        "extraction": extraction,
        "model_runs": model_records,
        "summary_csv": "summary.csv",
        "artifact_policy": "trajectories, raw extraction, and per-case scale-free CSV are external-style artifacts; root summaries are compact",
        "registration": {"status": "not_required_not_inferred", "pixel_per_model_unit": None, "time_scale": None, "time_offset": None},
        "parameter_identification": "suppressed",
        "model_inadequacy": "not_assessed_in_scale_free_morphology_mode",
        "physical_time_alignment": False,
    }
    _write_json(destination / "compact_summary.json", report)
    _write_csv(
        destination / "summary.csv",
        comparison_rows,
        ["run_name", "run_kind", "status", "eligible_observation_rows", "compared_rows", "censored_rows", "model_sha256", "input_quality_reasons"],
    )
    manifest = {
        "schema_version": RUNNER_SCHEMA_VERSION,
        "comparison_mode": "scale_free_shape",
        "source_revision": revision,
        "config_sha256": config_hash,
        "shape_config_sha256": report["shape_config_sha256"],
        "video": report["video"],
        "model_run_names": [record["run_name"] for record in model_records],
        "model_artifacts": {
            record["run_name"]: {
                "logical_id": record.get("model_logical_id"),
                "sha256": record.get("model_sha256"),
            }
            for record in model_records
        },
        "external_artifact_ids": extraction.get("external_artifact_ids", {}),
        "registration": report["registration"],
        "parameter_identification": "suppressed",
        "model_inadequacy": "not_assessed_in_scale_free_morphology_mode",
    }
    _write_json(destination / "compact_manifest.json", manifest)
    return report


def _cli() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--video", type=Path, default=DEFAULT_VIDEO)
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--max-progress-error", type=float)
    return parser.parse_args()


def main() -> int:
    args = _cli()
    shape_config = {"max_progress_error": args.max_progress_error} if args.max_progress_error is not None else None
    report = run_bounded_comparison(
        load_config(args.config),
        args.output,
        video_path=args.video,
        shape_config=shape_config,
        max_frames=args.max_frames,
    )
    print(json.dumps({"output": str(args.output.resolve()), "status": report["extraction"]["status"], "model_runs": len(report["model_runs"])}, ensure_ascii=False))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
