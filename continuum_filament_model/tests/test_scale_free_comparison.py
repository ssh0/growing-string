from __future__ import annotations

import csv
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from continuum_filament_model.video_compare import build_parser
from growing_filament.scale_free_comparison import (
    ScaleFreeConfig,
    normalized_shape_distance,
    scale_free_shape_comparison,
    shape_observables,
)


class ScaleFreeShapeComparisonTests(unittest.TestCase):
    def _write_observation(self, root: Path, lengths: list[float], *, censored: set[int] | None = None, flags: dict[int, str] | None = None) -> Path:
        root.mkdir(parents=True, exist_ok=True)
        censored = censored or set()
        flags = flags or {}
        rows = []
        centerline = []
        for frame, length in enumerate(lengths):
            is_censored = frame in censored
            quality_flags = flags.get(frame, "ok")
            rows.append(
                {
                    "frame": frame,
                    "time": 100.0 + frame * 7.0,
                    "filament_id": "filament-0000",
                    "n_points": 3,
                    "centerline_exported": 1,
                    "length_px": length,
                    "quality": 0.9,
                    "quality_flags": quality_flags,
                    "censor": int(is_censored),
                }
            )
            for point_id, x in enumerate((0.0, length * 0.5, length)):
                centerline.append(
                    {
                        "frame": frame,
                        "time": 100.0 + frame * 7.0,
                        "filament_id": "filament-0000",
                        "point_id": point_id,
                        "x": x,
                        "y": 0.0,
                        "quality": 0.9,
                        "coordinate_system": "pixel",
                        "quality_flags": quality_flags,
                        "censor": int(is_censored),
                    }
                )
        with (root / "observation_summary.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        with (root / "centerline.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(centerline[0]))
            writer.writeheader()
            writer.writerows(centerline)
        with (root / "lineage.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["frame", "time", "filament_id", "status", "censor", "details"])
            writer.writeheader()
            for row in rows:
                writer.writerow({
                    "frame": row["frame"],
                    "time": row["time"],
                    "filament_id": row["filament_id"],
                    "status": "matched" if row["frame"] else "initial_lineage",
                    "censor": row["censor"],
                    "details": "",
                })
        (root / "manifest.json").write_text(
            json.dumps(
                {
                    "input": {"logical_id": "synthetic-observation.mp4", "sha256": "video-hash", "bytes": 12},
                    "artifacts": {
                        name: {
                            "path": name,
                            "sha256": hashlib.sha256((root / name).read_bytes()).hexdigest(),
                            "bytes": (root / name).stat().st_size,
                        }
                        for name in ("centerline.csv", "observation_summary.csv", "lineage.csv")
                    },
                    "video": {"fps": 2.0},
                    "run": {"frame_range": {"first": 0, "last": len(lengths) - 1, "count": len(lengths), "stride": 1, "decode_complete": True}},
                    "validation": {"valid": True, "errors": [], "warnings": []},
                }
            ),
            encoding="utf-8",
        )
        return root

    def _write_model(self, root: Path, lengths: list[float]) -> Path:
        model = root / "model.npz"
        positions = []
        offsets = [0]
        for length in lengths:
            positions.extend([[0.0, 0.0], [length * 0.5, 0.0], [length, 0.0]])
            offsets.append(offsets[-1] + 3)
        metadata = {
            "parameters": {
                "axial_stiffness": 10.0,
                "bending_stiffness": 1.0,
                "drag_density": 1.0,
                "contact_stiffness": 0.0,
                "diameter": 0.0,
                "growth_rate": 0.02,
                "reference_length": 1.0,
                "dt": 0.001,
                "t_end": 0.002,
                "a_max": 1.0,
                "dt_min": 1.0e-10,
                "max_retries": 8,
                "max_displacement_fraction": 1.0,
                "energy_tolerance": 1.0e-9,
                "reject_crossing": True,
                "fixed_left": False,
                "fixed_right": False,
            },
            "metadata": {
                "benchmark": "stage2_free_free_growth_relaxation_buckling",
                "boundary": "free/free",
                "contact_enabled": False,
                "physical_scope": [
                    "uniform_reference_length_growth",
                    "stretching",
                    "discrete_bending",
                    "isotropic_substrate_drag",
                ],
                "run_kind": "deterministic_fixture",
            },
            "manifest": {"parameters": {
                "axial_stiffness": 10.0,
                "bending_stiffness": 1.0,
                "drag_density": 1.0,
                "contact_stiffness": 0.0,
                "diameter": 0.0,
                "growth_rate": 0.02,
                "reference_length": 1.0,
                "dt": 0.001,
                "t_end": 0.002,
                "a_max": 1.0,
                "dt_min": 1.0e-10,
                "max_retries": 8,
                "max_displacement_fraction": 1.0,
                "energy_tolerance": 1.0e-9,
                "reject_crossing": True,
                "fixed_left": False,
                "fixed_right": False,
            }},
        }
        np.savez(
            model,
            positions=np.asarray(positions, dtype=float),
            position_offsets=np.asarray(offsets, dtype=int),
            times=np.asarray([1000.0 + index * 11.0 for index in range(len(lengths))], dtype=float),
            metadata_json=np.asarray(json.dumps(metadata)),
        )
        return model

    def _write_model_csv(self, root: Path, lengths: list[float]) -> Path:
        model = root / "model.csv"
        with model.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["time", "point_id", "x", "y"])
            for index, length in enumerate(lengths):
                writer.writerow([1000.0 + index * 11.0, 0, 0.0, 0.0])
                writer.writerow([1000.0 + index * 11.0, 1, length * 0.5, 0.0])
                writer.writerow([1000.0 + index * 11.0, 2, length, 0.0])
        return model

    def test_curvature_rms_is_invariant_to_straight_segment_subdivision(self):
        coarse = np.asarray([[0.0, 0.0], [2.0, 0.0], [2.0, 2.0]])
        subdivided = np.asarray([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [2.0, 1.0], [2.0, 2.0]])
        coarse_features = shape_observables(coarse)
        subdivided_features = shape_observables(subdivided)
        self.assertAlmostEqual(
            coarse_features["curvature_rms_times_length"],
            subdivided_features["curvature_rms_times_length"],
            places=10,
        )

    def test_mode_fractions_are_invariant_to_straight_segment_subdivision(self):
        coarse = shape_observables(np.asarray([[0.0, 0.0], [2.0, 0.0], [2.0, 2.0]]))
        subdivided = shape_observables(np.asarray([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [2.0, 1.0], [2.0, 2.0]]))
        for first, second in zip(coarse["mode_fractions"], subdivided["mode_fractions"]):
            self.assertAlmostEqual(first, second, places=10)

    def test_independent_pixel_and_model_rescaling_cancels(self):
        t = np.linspace(0.0, 1.0, 41)
        shape = np.column_stack((t, 0.25 * np.sin(2.0 * np.pi * t)))
        first = shape_observables(shape)
        second = shape_observables(shape * 37.0)
        self.assertIsNotNone(first)
        self.assertIsNotNone(second)
        for key in ("normalized_endpoint_distance", "normalized_radius_of_gyration", "normalized_peak_deflection", "curvature_rms_times_length"):
            self.assertAlmostEqual(first[key], second[key], places=10)
        distance_a, _ = normalized_shape_distance(shape * 11.0, shape * 0.03)
        distance_b, _ = normalized_shape_distance(shape * 2.0, shape * 19.0)
        self.assertAlmostEqual(distance_a or 0.0, distance_b or 0.0, places=10)

    def test_translation_and_rotation_invariance(self):
        t = np.linspace(0.0, 1.0, 31)
        shape = np.column_stack((t, 0.18 * np.sin(np.pi * t)))
        angle = np.deg2rad(47.0)
        rotation = np.asarray([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
        transformed = shape @ rotation.T + np.asarray([13.0, -8.0])
        base = shape_observables(shape)
        moved = shape_observables(transformed * 4.0)
        for key in ("normalized_endpoint_distance", "normalized_radius_of_gyration", "normalized_peak_deflection", "curvature_rms_times_length"):
            self.assertAlmostEqual(base[key], moved[key], places=10)
        distance, orientation = normalized_shape_distance(shape, transformed * 4.0)
        self.assertAlmostEqual(distance or 1.0, 0.0, places=10)
        self.assertIn(orientation, {"forward", "reverse"})

    def test_growth_progress_alignment_does_not_use_time_registration(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            observation = self._write_observation(root / "observation", [10.0, 15.0, 20.0])
            model = self._write_model(root, [1.0, 1.5, 2.0])
            output = root / "comparison"
            result = scale_free_shape_comparison(observation, model, output_dir=output, source_revision="revision-test")
            self.assertEqual(result["summary"]["status"], "computed")
            self.assertEqual(result["summary"]["registration"]["status"], "not_required_not_inferred")
            self.assertIn("s/L", result["summary"]["spatial_normalization"])
            self.assertEqual(result["summary"]["compared_rows"], 3)
            self.assertTrue(all(row["progress_match_method"] == "nearest_growth_progress" for row in result["rows"]))
            self.assertEqual([row["observation_q"] for row in result["rows"]], [0.0, 0.5, 1.0])
            self.assertFalse(result["summary"]["coverage"]["frame_fraction_is_not_physical_time"] is False)
            self.assertEqual(result["summary"]["source_revision"], "revision-test")
            self.assertTrue(result["summary"]["model_provenance"]["sha256"])
            self.assertTrue((output / "scale_free_comparison_manifest.json").is_file())

    def test_censor_is_preserved_and_excluded_from_shape_distance(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            observation = self._write_observation(root / "observation", [10.0, 15.0, 20.0], censored={1}, flags={1: "large_jump"})
            model = self._write_model(root, [1.0, 1.5, 2.0])
            result = scale_free_shape_comparison(observation, model, output_dir=root / "comparison")
            self.assertEqual(result["summary"]["compared_rows"], 2)
            censored = result["rows"][1]
            self.assertEqual(censored["observation_censor"], 1)
            self.assertEqual(censored["comparison_censor"], 1)
            self.assertIsNone(censored["observation_normalized_endpoint_distance"])
            self.assertIsNone(censored["normalized_shape_distance"])
            self.assertEqual(censored["lineage_status"], "matched")

    def test_invalid_observation_contract_is_unavailable_but_retains_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            observation = self._write_observation(root / "observation", [10.0, 20.0])
            centerline = observation / "centerline.csv"
            with centerline.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            rows.append(dict(rows[-1]))
            with centerline.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            manifest_path = observation / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["validation"] = {"valid": False, "errors": ["duplicate point_id"]}
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            model = self._write_model(root, [1.0, 2.0])
            result = scale_free_shape_comparison(observation, model, output_dir=root / "comparison")
            self.assertEqual(result["summary"]["status"], "input_quality_invalid_observation_contract")
            self.assertEqual(result["summary"]["compared_rows"], 0)
            self.assertEqual(result["summary"]["rows"], 2)
            self.assertEqual(result["summary"]["coverage"]["observation"]["count"], 2)
            self.assertIn("observation_manifest_validation_invalid", result["summary"]["input_quality"]["reasons"])
            self.assertIn("observation_centerline_contract_invalid", result["summary"]["input_quality"]["reasons"])
            self.assertFalse(result["summary"]["observation_validation"]["contract_valid"])

    def test_progress_uses_only_valid_uncensored_observation_lengths(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            observation = self._write_observation(root / "observation", [10.0, 20.0, 100.0], censored={2})
            model = self._write_model(root, [1.0, 2.0, 3.0])
            result = scale_free_shape_comparison(observation, model, output_dir=root / "comparison")
            self.assertEqual(result["summary"]["status"], "computed")
            self.assertEqual([row["observation_q"] for row in result["rows"]], [0.0, 1.0, None])
            self.assertEqual([row["model_frame"] for row in result["rows"]], [0, 2, None])
            self.assertEqual(result["summary"]["observation_progress"]["valid_length_count"], 2)
            self.assertEqual(result["summary"]["compared_rows"], 2)

    def test_malformed_model_csv_is_not_silently_skipped(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            observation = self._write_observation(root / "observation", [10.0, 20.0, 30.0])
            model = root / "model.csv"
            with model.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle)
                writer.writerow(["time", "point_id", "y"])
                for index, length in enumerate((1.0, 2.0, 3.0)):
                    writer.writerow([1000.0 + index * 11.0, 0, 0.0])
                    writer.writerow([1000.0 + index * 11.0, 1, length * 0.5])
                    writer.writerow([1000.0 + index * 11.0, 2, length])
            result = scale_free_shape_comparison(observation, model, output_dir=root / "comparison")
            self.assertEqual(result["summary"]["status"], "model_centerline_unavailable")
            self.assertEqual(result["summary"]["compared_rows"], 0)
            self.assertFalse(result["summary"]["model_validation"]["valid"])
            self.assertIn("missing required columns x", result["summary"]["model_validation"]["errors"])
            self.assertIn("model_centerline_contract_invalid", result["summary"]["input_quality"]["reasons"])

    def test_model_csv_rejects_duplicate_point_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            observation = self._write_observation(root / "observation", [10.0, 20.0, 30.0])
            model = self._write_model_csv(root, [1.0, 2.0, 3.0])
            with model.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            rows[1]["point_id"] = "0"
            with model.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            result = scale_free_shape_comparison(observation, model, output_dir=root / "comparison")
            self.assertEqual(result["summary"]["status"], "input_quality_invalid_model_contract")
            self.assertEqual(result["summary"]["compared_rows"], 0)
            self.assertTrue(any("point_id must be ordered" in error for error in result["summary"]["model_validation"]["errors"]))

    def test_model_scope_metadata_is_required(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            observation = self._write_observation(root / "observation", [10.0, 20.0, 30.0])
            model = self._write_model_csv(root, [1.0, 2.0, 3.0])
            result = scale_free_shape_comparison(observation, model, output_dir=root / "comparison")
            self.assertEqual(result["summary"]["status"], "input_quality_invalid_model_contract")
            self.assertEqual(result["summary"]["compared_rows"], 0)
            self.assertIn("stage2_scope_metadata_required", result["summary"]["model_validation"]["errors"])

    def test_invalid_json_model_frame_is_not_dropped(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            observation = self._write_observation(root / "observation", [10.0, 20.0, 30.0])
            model = root / "model.json"
            model.write_text(
                json.dumps(
                    {
                        "trajectory": [
                            {"time": 1000.0, "points": [[0.0, 0.0], [1.0, 0.0]]},
                            {"time": 1001.0, "points": [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]},
                            {"time": 1002.0, "points": [[0.0, 0.0], [2.0, 0.0]]},
                        ]
                    }
                ),
                encoding="utf-8",
            )
            result = scale_free_shape_comparison(observation, model, output_dir=root / "comparison")
            self.assertEqual(result["summary"]["status"], "input_quality_invalid_model_contract")
            self.assertEqual(result["summary"]["compared_rows"], 0)
            self.assertEqual(result["summary"]["model_validation"]["frame_count"], 3)
            self.assertEqual(result["summary"]["model_validation"]["invalid_frame_count"], 1)
            self.assertEqual(result["summary"]["coverage"]["model"]["count"], 3)

    def test_missing_json_model_time_is_not_inferred(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            observation = self._write_observation(root / "observation", [10.0, 20.0])
            model = root / "model.json"
            model.write_text(
                json.dumps({"trajectory": [{"points": [[0.0, 0.0], [1.0, 0.0]]}, {"time": 1.0, "points": [[0.0, 0.0], [2.0, 0.0]]}]}),
                encoding="utf-8",
            )
            result = scale_free_shape_comparison(observation, model, output_dir=root / "comparison")
            self.assertEqual(result["summary"]["status"], "input_quality_invalid_model_contract")
            self.assertEqual(result["summary"]["compared_rows"], 0)
            self.assertEqual(result["summary"]["model_validation"]["invalid_frame_count"], 1)
            self.assertEqual(result["summary"]["coverage"]["model"]["count"], 2)

    def test_invalid_npz_offsets_are_unavailable(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            observation = self._write_observation(root / "observation", [10.0, 20.0])
            model = self._write_model(root, [1.0, 2.0])
            with np.load(model, allow_pickle=False) as archive:
                positions = archive["positions"]
                times = archive["times"]
                metadata_json = archive["metadata_json"]
            np.savez(model, positions=positions[:5], position_offsets=np.asarray([0, 2, 6]), times=times, metadata_json=metadata_json)
            result = scale_free_shape_comparison(observation, model, output_dir=root / "comparison")
            self.assertEqual(result["summary"]["status"], "input_quality_invalid_model_contract")
            self.assertEqual(result["summary"]["compared_rows"], 0)
            self.assertIn("position_offsets_end_mismatch", result["summary"]["model_validation"]["errors"])

    def test_invalid_nested_model_metadata_is_unavailable(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            observation = self._write_observation(root / "observation", [10.0, 20.0])
            model = self._write_model(root, [1.0, 2.0])
            with np.load(model, allow_pickle=False) as archive:
                positions = archive["positions"]
                offsets = archive["position_offsets"]
                times = archive["times"]
            np.savez(model, positions=positions, position_offsets=offsets, times=times, metadata_json=np.asarray(json.dumps({"metadata": [], "manifest": {}})))
            result = scale_free_shape_comparison(observation, model, output_dir=root / "comparison")
            self.assertEqual(result["summary"]["status"], "input_quality_invalid_model_contract")
            self.assertIn("model_metadata_not_mapping", result["summary"]["model_validation"]["errors"])

    def test_exploratory_replicate_is_not_deterministic_stage2(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            observation = self._write_observation(root / "observation", [10.0, 20.0])
            model = self._write_model(root, [1.0, 2.0])
            with np.load(model, allow_pickle=False) as archive:
                positions = archive["positions"]
                offsets = archive["position_offsets"]
                times = archive["times"]
                metadata = json.loads(str(archive["metadata_json"].item()))
            metadata["metadata"]["run_kind"] = "exploratory_replicate"
            np.savez(model, positions=positions, position_offsets=offsets, times=times, metadata_json=np.asarray(json.dumps(metadata)))
            result = scale_free_shape_comparison(observation, model, output_dir=root / "comparison")
            self.assertEqual(result["summary"]["status"], "input_quality_invalid_model_contract")
            self.assertIn("unsupported_model_run_kind", result["summary"]["model_validation"]["errors"])

    def test_initial_condition_sensitivity_protocol_is_labeled(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            observation = self._write_observation(root / "observation", [10.0, 20.0])
            model = self._write_model(root, [1.0, 2.0])
            with np.load(model, allow_pickle=False) as archive:
                positions = archive["positions"]
                offsets = archive["position_offsets"]
                times = archive["times"]
                metadata = json.loads(str(archive["metadata_json"].item()))
            baseline_artifact = root / "member-baseline.bin"
            plus_artifact = root / "member-plus.bin"
            baseline_artifact.write_bytes(b"baseline-member")
            plus_artifact.write_bytes(b"plus-member")
            baseline_hash = hashlib.sha256(baseline_artifact.read_bytes()).hexdigest()
            plus_hash = hashlib.sha256(plus_artifact.read_bytes()).hexdigest()
            baseline_metrics = {"normalized_shape_distance": 0.0}
            plus_metrics = {"normalized_shape_distance": 0.1}
            metadata["metadata"].update(
                {
                    "run_kind": "initial_condition_sensitivity",
                    "sensitivity_protocol": {
                        "parameter": "initial_condition",
                        "perturbation_range": {"min": -0.1, "max": 0.1},
                        "members": [
                            {
                                "id": "baseline",
                                "perturbation_value": 0.0,
                                "artifact_path": "member-baseline.bin",
                                "trajectory_sha256": baseline_hash,
                                "provenance": {"seed": 0, "trajectory_sha256": baseline_hash},
                                "result": {"metrics": baseline_metrics, "metrics_sha256": hashlib.sha256(json.dumps(baseline_metrics, sort_keys=True, separators=(",", ":")).encode()).hexdigest()},
                            },
                            {
                                "id": "plus",
                                "perturbation_value": 0.1,
                                "artifact_path": "member-plus.bin",
                                "trajectory_sha256": plus_hash,
                                "provenance": {"seed": 1, "trajectory_sha256": plus_hash},
                                "result": {"metrics": plus_metrics, "metrics_sha256": hashlib.sha256(json.dumps(plus_metrics, sort_keys=True, separators=(",", ":")).encode()).hexdigest()},
                            },
                        ],
                        "metrics": ["normalized_shape_distance"],
                        "aggregate_metrics": {"member_count": 2, "max_normalized_shape_distance_delta": 0.1},
                        "accepted": True,
                        "acceptance_criteria": {"max_metric_delta": 0.2},
                    },
                }
            )
            np.savez(model, positions=positions, position_offsets=offsets, times=times, metadata_json=np.asarray(json.dumps(metadata)))
            result = scale_free_shape_comparison(observation, model, output_dir=root / "comparison")
            self.assertEqual(result["summary"]["model_population"], "initial_condition_sensitivity")
            self.assertEqual(result["summary"]["initial_condition_sensitivity"]["metrics"], ["normalized_shape_distance"])
            self.assertTrue(result["summary"]["video_alignment"]["distinguished_from_initial_condition_sensitivity"])

    def test_invalid_model_contract_is_unavailable_but_retains_coverage(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            observation = self._write_observation(root / "observation", [10.0, 20.0, 30.0])
            model = self._write_model_csv(root, [1.0, 2.0, 3.0])
            with model.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            rows[4]["x"] = "nan"
            with model.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            result = scale_free_shape_comparison(observation, model, output_dir=root / "comparison")
            self.assertEqual(result["summary"]["status"], "input_quality_invalid_model_contract")
            self.assertEqual(result["summary"]["compared_rows"], 0)
            self.assertFalse(result["summary"]["model_validation"]["valid"])
            self.assertEqual(result["summary"]["model_validation"]["invalid_frame_count"], 1)
            self.assertEqual(result["summary"]["coverage"]["model"]["count"], 3)
            self.assertIn("model_centerline_contract_invalid", result["summary"]["input_quality"]["reasons"])

    def test_scale_free_cli_options_are_not_accepted_by_extract(self):
        parser = build_parser()
        args = parser.parse_args([
            "scale-free",
            "--output", "observation",
            "--model", "model.csv",
            "--shape-config", "shape.json",
            "--max-progress-error", "0.1",
        ])
        self.assertEqual(args.shape_config, "shape.json")
        self.assertEqual(args.max_progress_error, 0.1)
        self.assertFalse(hasattr(args, "video"))
        self.assertFalse(hasattr(args, "config"))
        self.assertFalse(hasattr(args, "registration"))
        self.assertFalse(hasattr(args, "max_frames"))
        self.assertFalse(hasattr(args, "representative_count"))
        with self.assertRaises(SystemExit):
            parser.parse_args(["extract", "--video", "video.mp4", "--output", "observation", "--shape-config", "shape.json"])

    def test_duplicate_observation_summary_key_is_unavailable(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            observation = self._write_observation(root / "observation", [10.0, 20.0, 30.0])
            summary = observation / "observation_summary.csv"
            with summary.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            rows.append(dict(rows[1]))
            with summary.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            model = self._write_model(root, [1.0, 2.0, 3.0])
            result = scale_free_shape_comparison(observation, model, output_dir=root / "comparison")
            self.assertEqual(result["summary"]["status"], "input_quality_invalid_observation_contract")
            self.assertEqual(result["summary"]["compared_rows"], 0)
            self.assertFalse(result["summary"]["observation_validation"]["frame_keys"]["summary"]["valid"])

    def test_invalid_censor_value_is_not_inferred(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            observation = self._write_observation(root / "observation", [10.0, 20.0, 30.0])
            summary = observation / "observation_summary.csv"
            with summary.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            rows[1]["censor"] = "corrupt"
            with summary.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            model = self._write_model(root, [1.0, 2.0, 3.0])
            result = scale_free_shape_comparison(observation, model, output_dir=root / "comparison")
            self.assertEqual(result["summary"]["status"], "input_quality_invalid_observation_contract")
            self.assertEqual(result["summary"]["compared_rows"], 0)
            self.assertEqual(result["rows"][1]["observation_censor"], 1)

    def test_quality_above_one_is_not_eligible(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            observation = self._write_observation(root / "observation", [10.0, 20.0, 30.0])
            summary = observation / "observation_summary.csv"
            with summary.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            rows[1]["quality"] = "2.0"
            with summary.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            model = self._write_model(root, [1.0, 2.0, 3.0])
            result = scale_free_shape_comparison(observation, model, output_dir=root / "comparison")
            self.assertEqual(result["summary"]["status"], "input_quality_invalid_observation_contract")
            self.assertEqual(result["summary"]["compared_rows"], 0)

    def test_malformed_manifest_structure_is_unavailable_with_coverage(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            observation = self._write_observation(root / "observation", [10.0, 20.0, 30.0])
            (observation / "manifest.json").write_text(
                json.dumps({"validation": {"valid": True}, "segmentation_config": [], "run": {"frame_range": []}, "video": {"fps": "bad"}}),
                encoding="utf-8",
            )
            model = self._write_model(root, [1.0, 2.0, 3.0])
            result = scale_free_shape_comparison(observation, model, output_dir=root / "comparison")
            self.assertEqual(result["summary"]["status"], "input_quality_invalid_observation_contract")
            self.assertEqual(result["summary"]["coverage"]["observation"]["count"], 3)
            self.assertFalse(result["summary"]["observation_validation"]["manifest"]["valid"])

    def test_missing_observation_artifact_is_unavailable_with_coverage(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            observation = self._write_observation(root / "observation", [10.0, 20.0, 30.0])
            (observation / "centerline.csv").unlink()
            model = self._write_model(root, [1.0, 2.0, 3.0])
            result = scale_free_shape_comparison(observation, model, output_dir=root / "comparison")
            self.assertEqual(result["summary"]["status"], "input_quality_invalid_observation_contract")
            self.assertEqual(result["summary"]["coverage"]["observation"]["count"], 3)
            self.assertFalse(result["summary"]["observation_validation"]["artifacts"]["centerline"]["valid"])

    def test_malformed_manifest_is_unavailable_with_coverage(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            observation = self._write_observation(root / "observation", [10.0, 20.0, 30.0])
            (observation / "manifest.json").write_text("{", encoding="utf-8")
            model = self._write_model(root, [1.0, 2.0, 3.0])
            result = scale_free_shape_comparison(observation, model, output_dir=root / "comparison")
            self.assertEqual(result["summary"]["status"], "input_quality_invalid_observation_contract")
            self.assertEqual(result["summary"]["compared_rows"], 0)
            self.assertEqual(result["summary"]["coverage"]["observation"]["count"], 3)
            self.assertFalse(result["summary"]["observation_validation"]["manifest"]["valid"])

    def test_observation_metadata_mismatch_is_unavailable(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            observation = self._write_observation(root / "observation", [10.0, 20.0, 30.0])
            centerline = observation / "centerline.csv"
            with centerline.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            rows[0]["time"] = "200.0"
            with centerline.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            model = self._write_model(root, [1.0, 2.0, 3.0])
            result = scale_free_shape_comparison(observation, model, output_dir=root / "comparison")
            self.assertEqual(result["summary"]["status"], "input_quality_invalid_observation_contract")
            self.assertEqual(result["summary"]["compared_rows"], 0)
            self.assertFalse(result["summary"]["observation_validation"]["consistency"]["valid"])

    def test_phantom_lineage_is_unavailable(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            observation = self._write_observation(root / "observation", [10.0, 20.0, 30.0])
            lineage = observation / "lineage.csv"
            with lineage.open("a", newline="", encoding="utf-8") as handle:
                handle.write("99,999,phantom,matched,0,\n")
            model = self._write_model(root, [1.0, 2.0, 3.0])
            result = scale_free_shape_comparison(observation, model, output_dir=root / "comparison")
            self.assertEqual(result["summary"]["status"], "input_quality_invalid_observation_contract")
            self.assertEqual(result["summary"]["compared_rows"], 0)
            self.assertFalse(result["summary"]["observation_validation"]["consistency"]["valid"])

    def test_observation_summary_length_mismatch_is_unavailable(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            observation = self._write_observation(root / "observation", [10.0, 20.0, 30.0])
            centerline = observation / "centerline.csv"
            with centerline.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            for row in rows:
                if row["frame"] == "1":
                    row["x"] = str(float(row["x"]) * 5.0)
            with centerline.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            model = self._write_model(root, [1.0, 2.0, 3.0])
            result = scale_free_shape_comparison(observation, model, output_dir=root / "comparison")
            self.assertEqual(result["summary"]["status"], "input_quality_invalid_observation_contract")
            self.assertIn("length mismatch", " ".join(result["summary"]["observation_validation"]["consistency"]["errors"]))

    def test_observation_artifact_hash_mismatch_is_unavailable(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            observation = self._write_observation(root / "observation", [10.0, 20.0, 30.0])
            centerline = observation / "centerline.csv"
            centerline.write_text(centerline.read_text(encoding="utf-8") + "", encoding="utf-8")
            with centerline.open("a", encoding="utf-8") as handle:
                handle.write("\n")
            model = self._write_model(root, [1.0, 2.0, 3.0])
            result = scale_free_shape_comparison(observation, model, output_dir=root / "comparison")
            self.assertEqual(result["summary"]["status"], "input_quality_invalid_observation_contract")
            self.assertFalse(result["summary"]["observation_validation"]["manifest_artifacts"]["valid"])

    def test_manifest_nested_shape_is_unavailable_with_coverage(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            observation = self._write_observation(root / "observation", [10.0, 20.0, 30.0])
            manifest_path = observation / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["validation"] = {"valid": True, "errors": None, "warnings": []}
            manifest["input"] = []
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            model = self._write_model(root, [1.0, 2.0, 3.0])
            result = scale_free_shape_comparison(observation, model, output_dir=root / "comparison")
            self.assertEqual(result["summary"]["status"], "input_quality_invalid_observation_contract")
            self.assertEqual(result["summary"]["coverage"]["observation"]["count"], 3)

    def test_summary_lineage_mismatch_without_centerline_is_unavailable(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            observation = self._write_observation(root / "observation", [10.0, 20.0, 30.0])
            (observation / "centerline.csv").unlink()
            lineage = observation / "lineage.csv"
            with lineage.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            rows[1]["time"] = "999.0"
            with lineage.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            model = self._write_model(root, [1.0, 2.0, 3.0])
            result = scale_free_shape_comparison(observation, model, output_dir=root / "comparison")
            self.assertEqual(result["summary"]["status"], "input_quality_invalid_observation_contract")
            self.assertFalse(result["summary"]["observation_validation"]["consistency"]["valid"])

    def test_invalid_observation_frame_key_is_unavailable(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            observation = self._write_observation(root / "observation", [10.0, 20.0, 30.0])
            centerline = observation / "centerline.csv"
            with centerline.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            rows[3]["frame"] = "bad"
            with centerline.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            model = self._write_model(root, [1.0, 2.0, 3.0])
            result = scale_free_shape_comparison(observation, model, output_dir=root / "comparison")
            self.assertEqual(result["summary"]["status"], "input_quality_invalid_observation_contract")
            self.assertEqual(result["summary"]["compared_rows"], 0)
            self.assertEqual(result["summary"]["coverage"]["observation"]["count"], 3)
            self.assertFalse(result["summary"]["observation_validation"]["frame_keys"]["centerline"]["valid"])
            self.assertIn("observation_frame_key_invalid", result["summary"]["input_quality"]["reasons"])

    def test_missing_lineage_is_unavailable_without_inferred_lineage(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            observation = self._write_observation(root / "observation", [10.0, 20.0, 30.0])
            (observation / "lineage.csv").unlink()
            model = self._write_model(root, [1.0, 2.0, 3.0])
            result = scale_free_shape_comparison(observation, model, output_dir=root / "comparison")
            self.assertEqual(result["summary"]["status"], "input_quality_invalid_observation_contract")
            self.assertEqual(result["summary"]["compared_rows"], 0)
            self.assertEqual(result["summary"]["coverage"]["observation"]["count"], 3)
            self.assertFalse(result["summary"]["observation_validation"]["lineage"]["valid"])
            self.assertTrue(all(row["lineage_status"] == "missing_lineage" for row in result["rows"]))
            self.assertTrue(all(row["observation_censor"] == 1 for row in result["rows"]))
            self.assertIn("observation_lineage_invalid", result["summary"]["input_quality"]["reasons"])

    def test_lineage_and_processed_coverage_survive_empty_summary(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            observation = self._write_observation(root / "observation", [10.0, 20.0, 30.0])
            (observation / "observation_summary.csv").write_text(
                "frame,time,filament_id,n_points,centerline_exported,component_area,component_count_total,endpoint_count,junction_count,cycle_rank,length_px,endpoint_distance_px,curvature_mean_px_inv,curvature_max_px_inv,quality,quality_flags,censor\n",
                encoding="utf-8",
            )
            (observation / "centerline.csv").write_text(
                "time,filament_id,point_id,x,y,quality,frame,coordinate_system,quality_flags,censor\n",
                encoding="utf-8",
            )
            model = self._write_model(root, [1.0, 2.0, 3.0])
            result = scale_free_shape_comparison(observation, model, output_dir=root / "comparison")
            self.assertEqual(result["summary"]["rows"], 3)
            self.assertEqual(result["summary"]["censored_rows"], 3)
            self.assertEqual(result["summary"]["coverage"]["observation"]["count"], 3)
            self.assertEqual([row["lineage_status"] for row in result["rows"]], ["initial_lineage", "matched", "matched"])
            self.assertTrue(all(row["comparison_censor"] == 1 for row in result["rows"]))

    def test_zero_growth_is_explicitly_unavailable(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            observation = self._write_observation(root / "observation", [10.0, 10.0, 10.0])
            model = self._write_model(root, [1.0, 1.5, 2.0])
            result = scale_free_shape_comparison(observation, model, output_dir=root / "comparison")
            self.assertEqual(result["summary"]["status"], "growth_progress_undefined_zero_span")
            self.assertIn("observation_zero_growth_span", result["summary"]["input_quality"]["reasons"])
            self.assertEqual(result["summary"]["compared_rows"], 0)

    def test_non_monotonic_length_is_not_silently_aligned(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            observation = self._write_observation(root / "observation", [10.0, 15.0, 12.0])
            model = self._write_model(root, [1.0, 1.5, 2.0])
            result = scale_free_shape_comparison(observation, model, output_dir=root / "comparison")
            self.assertEqual(result["summary"]["status"], "growth_progress_undefined_non_monotonic")
            self.assertEqual(result["summary"]["compared_rows"], 0)
            self.assertEqual(result["summary"]["observation_progress"]["nonmonotonic_decrease_count"], 1)

    def test_schema_does_not_emit_model_inadequacy_conclusion(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            observation = self._write_observation(root / "observation", [10.0, 20.0])
            model = self._write_model(root, [1.0, 2.0])
            result = scale_free_shape_comparison(observation, model, output_dir=root / "comparison")
            self.assertEqual(result["summary"]["model_inadequacy"], "not_assessed_in_scale_free_morphology_mode")
            self.assertEqual(result["summary"]["parameter_identification"], "suppressed")
            self.assertEqual(result["manifest"]["comparison_mode"], "scale_free_shape")
            self.assertIn("model_population", result["manifest"])


if __name__ == "__main__":
    unittest.main()
