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
import hashlib
import json
import shutil
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

if __package__ in {None, ""}:  # pragma: no cover - direct-file entry point
    _HERE = Path(__file__).resolve()
    _ROOT = _HERE.parents[1]
    _SRC = _ROOT / "src"
    for _path in (_ROOT, _SRC):
        if str(_path) not in sys.path:
            sys.path.insert(0, str(_path))

if __package__ in {None, ""}:  # pragma: no cover - direct-file entry point
    from benchmarks.free_growth_buckling_stage2 import (  # noqa: E402
        RunSpec,
        _all_specs,
        load_config,
        load_config_from_mapping,
        run_case,
    )
else:
    from .free_growth_buckling_stage2 import (  # noqa: E402
        RunSpec,
        _all_specs,
        load_config,
        load_config_from_mapping,
        run_case,
    )
from growing_filament.io import load_trajectory_metadata  # noqa: E402
from growing_filament.reproducibility import canonical_json_bytes, detect_git_revision  # noqa: E402
from growing_filament.scale_free_comparison import (  # noqa: E402
    SCHEMA_VERSION,
    ScaleFreeConfig,
    _trajectory_sha256,
    scale_free_shape_comparison,
    shape_observables,
)
from growing_filament.video_comparison import (  # noqa: E402
    SegmentationConfig,
    canonical_json,
    load_model_output,
    run_pipeline,
    sha256_file,
    sha256_text,
)


RUNNER_SCHEMA_VERSION = "continuum-filament-scale-free-runner-0.1"
DEFAULT_VIDEO = Path(__file__).resolve().parents[2] / "img" / "gray5.mp4"
DEFAULT_CASES = ("fast_growth_low_bend", "fast_growth_high_bend")
_SENSITIVITY_METRICS = frozenset({"normalized_endpoint_distance"})


def _trajectory_arrays(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {
            name: np.asarray(archive[name])
            for name in ("positions", "position_offsets", "rest_lengths", "rest_offsets", "times", "steps")
        }


def _rewrite_sensitivity_metadata(path: Path, protocol: Mapping[str, Any]) -> None:
    arrays = _trajectory_arrays(path)
    metadata = load_trajectory_metadata(path)
    metadata["metadata"] = dict(metadata.get("metadata", {}))
    metadata["metadata"]["sensitivity_protocol"] = dict(protocol)
    metadata["manifest"] = dict(metadata.get("manifest", {}))
    metadata["manifest"]["metadata"] = dict(metadata["manifest"].get("metadata", {}))
    metadata["manifest"]["metadata"]["sensitivity_protocol"] = dict(protocol)
    temporary = path.with_name(f".{path.stem}.sensitivity.npz")
    np.savez_compressed(
        temporary,
        **arrays,
        metadata_json=np.asarray(json.dumps(metadata, sort_keys=True, allow_nan=False)),
    )
    temporary.replace(path)


def _sensitivity_member_record(
    artifact: Path,
    member_id: str,
    perturbation: float,
    baseline_run_id: str,
    metrics: Sequence[str],
) -> dict[str, Any]:
    metadata = load_trajectory_metadata(artifact)
    manifest = metadata["manifest"]
    frames = load_model_output(artifact)
    if not frames:
        raise ValueError(f"sensitivity member has no trajectory: {artifact}")
    features = shape_observables(frames[-1].points, sample_points=80)
    if features is None:
        raise ValueError(f"sensitivity member has no finite final shape: {artifact}")
    metric_values = {metric: float(features[metric]) for metric in metrics}
    artifact_hash = sha256_file(artifact)
    return {
        "id": member_id,
        "perturbation_value": float(perturbation),
        "perturbation_vector": [float(perturbation)],
        "perturbation_norm": abs(float(perturbation)),
        "artifact_path": str(Path("members") / artifact.name),
        "trajectory_sha256": artifact_hash,
        "provenance": {
            "seed": metadata.get("metadata", {}).get("seed"),
            "trajectory_sha256": artifact_hash,
            "initial_state_hash": manifest.get("initial_state_hash"),
            "baseline_run_id": baseline_run_id,
        },
        "result": {
            "metrics": metric_values,
            "metrics_sha256": sha256_text(canonical_json(metric_values)),
        },
    }


def _run_initial_condition_sensitivity(
    config: Mapping[str, Any],
    specs: Sequence[RunSpec],
    output: Path,
    revision: str | None,
) -> dict[str, Any]:
    if not specs:
        raise ValueError("initial condition sensitivity population is empty")
    settings = config["initial_condition_sensitivity"]
    metrics = [str(metric) for metric in settings["metrics"]]
    unsupported_metrics = [metric for metric in metrics if metric not in _SENSITIVITY_METRICS]
    if unsupported_metrics:
        raise ValueError("unsupported sensitivity metrics: " + ",".join(unsupported_metrics))
    base_name = str(settings["base_fixture"])
    outer_name = f"{base_name}_initial_condition_sensitivity"
    baseline_spec = next(spec for spec in specs if abs(float(spec.perturbation_value or 0.0)) <= 1.0e-12)
    baseline_id = "baseline"
    member_output = output / "_sensitivity_members"
    outer_dir = output / "_runs" / outer_name
    members_dir = outer_dir / "members"
    members_dir.mkdir(parents=True, exist_ok=True)
    member_records: list[dict[str, Any]] = []
    for index, spec in enumerate(specs):
        member_result = run_case(spec, config["base"], member_output, revision, save_trajectory_file=True)
        trajectory_value = member_result.get("trajectory_path")
        if not trajectory_value:
            raise ValueError(f"sensitivity member trajectory unavailable: {spec.name}")
        source = member_output / str(trajectory_value)
        member_id = baseline_id if spec is baseline_spec else f"member_{index:02d}"
        destination = members_dir / f"{member_id}.npz"
        shutil.copyfile(source, destination)
        member_records.append(_sensitivity_member_record(
            destination,
            member_id,
            float(spec.perturbation_value),
            baseline_id,
            metrics,
        ))
    outer_spec = RunSpec(
        outer_name,
        "initial_condition_sensitivity",
        dict(baseline_spec.overrides),
        base_fixture=base_name,
    )
    outer_result = run_case(outer_spec, config["base"], output, revision, save_trajectory_file=True)
    outer_value = outer_result.get("trajectory_path")
    if not outer_value:
        raise ValueError("initial condition sensitivity outer trajectory unavailable")
    outer_path = output / str(outer_value)
    outer_arrays = _trajectory_arrays(outer_path)
    aggregate_metrics: dict[str, Any] = {"member_count": len(member_records)}
    acceptance = dict(settings["acceptance_criteria"])
    accepted = True
    for metric in metrics:
        values = [float(member["result"]["metrics"][metric]) for member in member_records]
        delta = max(values) - min(values)
        aggregate_metrics[f"max_{metric}_delta"] = delta
        threshold = float(acceptance.get(f"max_{metric}_delta", acceptance.get("max_metric_delta")))
        accepted = accepted and delta <= threshold
    protocol = {
        "parameter": "initial_condition",
        "outer_run_id": outer_name,
        "outer_member_id": baseline_id,
        "outer_member_sha256": next(member["trajectory_sha256"] for member in member_records if member["id"] == baseline_id),
        "outer_trajectory_sha256": _trajectory_sha256(**outer_arrays),
        "baseline_run_id": baseline_id,
        "baseline_initial_state_hash": next(member["provenance"]["initial_state_hash"] for member in member_records if member["id"] == baseline_id),
        "perturbation_range": {
            "min": min(float(spec.perturbation_value) for spec in specs),
            "max": max(float(spec.perturbation_value) for spec in specs),
        },
        "members": member_records,
        "metrics": metrics,
        "aggregate_metrics": aggregate_metrics,
        "acceptance_criteria": acceptance,
        "accepted": accepted,
    }
    _rewrite_sensitivity_metadata(outer_path, protocol)
    return {
        "run_name": outer_name,
        "run_kind": "initial_condition_sensitivity",
        "model_population": "initial_condition_sensitivity",
        "base_fixture": base_name,
        "seed": None,
        "trial": 0,
        "model_logical_id": outer_path.name,
        "model_sha256": sha256_file(outer_path),
        "model_path_external": True,
        "model_failure_reason": outer_result.get("failure_reason"),
        "sensitivity_protocol": protocol,
        "comparison": None,
    }


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
    """Run the bounded Stage 2 populations against gray5.

    A missing or unusable video still produces a compact input-quality result;
    no substitute video or inferred registration is used.
    """

    effective = load_config_from_mapping(config) if config is not None else load_config(None)
    destination = Path(output).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    source = Path(video_path).expanduser().resolve()
    revision = detect_git_revision(Path(__file__).resolve().parents[2])
    config_hash = hashlib.sha256(canonical_json_bytes(effective)).hexdigest()
    all_specs = _all_specs(effective, include_initial_condition_sensitivity=True)
    specs = {spec.name: spec for spec in all_specs if spec.population == "stage2_deterministic"}
    sensitivity_specs = [spec for spec in all_specs if spec.population == "initial_condition_sensitivity"]
    configured_cases = effective.get("scale_free_cases")
    configuration_errors: list[str] = []
    if isinstance(configured_cases, list):
        requested_cases = [str(name) for name in configured_cases]
        if len(requested_cases) != len(set(requested_cases)):
            configuration_errors.append("duplicate_scale_free_cases")
        unknown_cases = [name for name in requested_cases if name not in specs]
        if unknown_cases:
            configuration_errors.append("unknown_scale_free_cases:" + ",".join(unknown_cases))
        case_names = [name for name in requested_cases if name in specs]
    elif "scale_free_cases" in effective:
        configuration_errors.append("scale_free_cases_must_be_list")
        case_names = []
    else:
        case_names = [name for name in DEFAULT_CASES if name in specs]
        if not case_names:
            case_names = [spec.name for spec in all_specs if spec.population == "stage2_deterministic" and spec.kind == "deterministic_fixture"][:2]
    # The bounded follow-up is intentionally narrower than the general Stage 2
    # harness: only deterministic baseline fixtures and verified sensitivity
    # artifacts are eligible.  Refinements, contrasts, and exploratory
    # replicates must not be presented as this population.
    unsupported_cases = [name for name in case_names if specs[name].kind not in {"deterministic_fixture"}]
    if unsupported_cases:
        configuration_errors.append("unsupported_scale_free_cases:" + ",".join(unsupported_cases))
    case_names = [name for name in case_names if name not in unsupported_cases]
    if not case_names:
        configuration_errors.append("no_scale_free_cases_selected")
    model_records: list[dict[str, Any]] = []
    for case_name in ([] if configuration_errors else case_names):
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
    if not configuration_errors:
        model_records.append(_run_initial_condition_sensitivity(effective, sensitivity_specs, destination, revision))

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

    if configuration_errors:
        extraction = dict(extraction)
        extraction["status"] = "input_quality_invalid_configuration"
        extraction["configuration_errors"] = configuration_errors
        extraction["comparison_suppressed"] = True
    if isinstance(shape_config, ScaleFreeConfig):
        shape_cfg = shape_config
    else:
        shape_values = dict(shape_config or {})
        # This bounded follow-up is the explicitly exploratory candidate-input
        # comparison.  Direct library callers retain strict validated-centerline
        # defaults unless they opt in through ScaleFreeConfig.
        shape_values.setdefault("allow_censored_candidates", True)
        shape_cfg = ScaleFreeConfig.from_mapping(shape_values)
    input_status = "candidate_input_exploratory" if shape_cfg.allow_censored_candidates else "validated_centerline"
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
                    "model_sha256": record.get("model_sha256"),
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
            "model_population": comparison.get("model_population", record.get("model_population")),
            "input_status": comparison.get("input_status", input_status),
            "status": comparison.get("status"),
            "comparison_suppressed": comparison.get("comparison_suppressed", comparison.get("compared_rows", 0) == 0),
            "eligible_observation_rows": comparison.get("eligible_observation_rows", 0),
            "candidate_computed_rows": (comparison.get("candidate_input") or {}).get("candidate_computed_rows", 0),
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
        "shape_config_sha256": hashlib.sha256(canonical_json_bytes(shape_cfg.to_dict())).hexdigest(),
        "input_status": input_status,
        "comparison_suppressed": any(
            (record.get("comparison") or {}).get("compared_rows", 0) == 0
            for record in model_records
        ) or bool(configuration_errors) or extraction.get("status") != "extracted",
        "video": _compact_input(source),
        "extraction": extraction,
        "configuration_errors": configuration_errors,
        "model_runs": model_records,
        "model_populations": {
            "stage2_deterministic": [record["run_name"] for record in model_records if record.get("run_kind") == "deterministic_fixture"],
            "initial_condition_sensitivity": [record["run_name"] for record in model_records if record.get("run_kind") == "initial_condition_sensitivity"],
        },
        "summary_csv": "summary.csv",
        "artifact_policy": "trajectories, raw extraction, and per-case scale-free CSV are external-style artifacts; root summaries are compact",
        "registration": {"status": "not_required_not_inferred", "pixel_per_model_unit": None, "time_scale": None, "time_offset": None},
        "parameter_identification": "suppressed",
        "model_inadequacy": "not_assessed_in_scale_free_morphology_mode",
        "model_inadequacy_assessment": "suppressed",
        "physical_conclusions": "suppressed",
        "physical_time_alignment": False,
    }
    _write_json(destination / "compact_summary.json", report)
    _write_csv(
        destination / "summary.csv",
        comparison_rows,
        [
            "run_name", "run_kind", "model_population", "input_status", "status", "comparison_suppressed",
            "eligible_observation_rows", "candidate_computed_rows", "compared_rows", "censored_rows",
            "model_sha256", "input_quality_reasons",
        ],
    )
    manifest = {
        "schema_version": RUNNER_SCHEMA_VERSION,
        "comparison_mode": "scale_free_shape",
        "input_status": input_status,
        "source_revision": revision,
        "config_sha256": config_hash,
        "shape_config_sha256": report["shape_config_sha256"],
        "video": report["video"],
        "model_run_names": [record["run_name"] for record in model_records],
        "model_populations": report["model_populations"],
        "configuration_errors": configuration_errors,
        "comparison_suppressed": bool(configuration_errors) or extraction.get("status") != "extracted" or any(
            (record.get("comparison") or {}).get("status", "").startswith("input_quality_")
            or (record.get("comparison") or {}).get("compared_rows", 0) == 0
            for record in model_records
        ),
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
        "model_inadequacy": "suppressed",
        "physical_conclusions": "suppressed",
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
