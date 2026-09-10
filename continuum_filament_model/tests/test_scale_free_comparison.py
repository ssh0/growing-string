from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

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
                    "artifacts": {"centerline": {"path": "centerline.csv", "sha256": "centerline-hash", "bytes": 1}},
                    "video": {"fps": 2.0},
                    "run": {"frame_range": {"first": 0, "last": len(lengths) - 1, "stride": 1}},
                }
            ),
            encoding="utf-8",
        )
        return root

    def _write_model(self, root: Path, lengths: list[float]) -> Path:
        model = root / "model.csv"
        with model.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["time", "point_id", "x", "y"])
            for index, length in enumerate(lengths):
                writer.writerow([1000.0 + index * 11.0, 0, 0.0, 0.0])
                writer.writerow([1000.0 + index * 11.0, 1, length * 0.5, 0.0])
                writer.writerow([1000.0 + index * 11.0, 2, length, 0.0])
        return model

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
