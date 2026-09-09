from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np

from continuum_filament_model.benchmarks import exploratory_shape_fitting as fitting


class ExploratoryShapeFittingTests(unittest.TestCase):
    def test_discrete_frechet_and_curvature_are_zero_for_identical_polyline(self):
        points = np.column_stack((np.linspace(0.0, 1.0, 12), 0.12 * np.sin(np.linspace(0.0, np.pi, 12))))
        self.assertAlmostEqual(fitting.discrete_frechet(points, points), 0.0)
        self.assertAlmostEqual(fitting.curvature_rmse(points, points), 0.0)

    def test_canonical_observation_preserves_order_and_requested_resolution(self):
        points = np.asarray([[4.0, 8.0], [5.0, 7.0], [7.0, 7.0], [8.0, 6.0]])
        nine = 9
        canonical = fitting.canonical_observation(points, n_nodes=nine, model_length=2.0)
        self.assertEqual(canonical.shape, (nine, 2))
        self.assertGreater(np.linalg.norm(canonical[-1] - canonical[0]), 0.0)
        self.assertAlmostEqual(
            float(np.sum(np.linalg.norm(np.diff(canonical, axis=0), axis=1))),
            2.0,
            places=3,
        )

    def test_dimensionless_features_are_invariant_to_translation_rotation_and_scale(self):
        base = np.column_stack((np.linspace(0.0, 1.0, 20), 0.15 * np.sin(np.linspace(0.0, np.pi, 20))))
        angle = 0.7
        rotation = np.asarray([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
        transformed = base @ rotation.T * 4.0 + np.asarray([8.0, -3.0])
        first = fitting.shape_features(base)
        second = fitting.shape_features(transformed)
        for name in fitting.FEATURE_NAMES:
            self.assertAlmostEqual(first[name], second[name], places=6, msg=name)

    def test_select_observations_keeps_censor_and_quality_flags(self):
        frames = [
            fitting.ObservationFrame(0, 1.0, "f0", np.asarray([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]]), True, 0.5, "branched_component"),
            fitting.ObservationFrame(1, 3.0, "f0", np.asarray([[0.0, 0.0], [1.0, 0.2], [2.0, 0.0]]), False, 0.9, "ok"),
        ]
        selected = fitting.select_observations(frames, (1.1, 2.9))
        self.assertEqual([item.frame for item in selected], [0, 1])
        self.assertTrue(selected[0].censor)
        self.assertEqual(selected[0].quality_flags, "branched_component")

    def test_small_exploration_writes_summary_and_trajectory(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            centerline = root / "centerline.csv"
            points = np.column_stack((np.linspace(0.0, 10.0, 12), 0.4 * np.sin(np.linspace(0.0, np.pi, 12))))
            with centerline.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=["time", "filament_id", "point_id", "x", "y", "quality", "frame", "quality_flags", "censor"],
                    lineterminator="\n",
                )
                writer.writeheader()
                for frame, time_s in enumerate((0.0, 0.1)):
                    for point_id, (x, y) in enumerate(points):
                        writer.writerow({
                            "time": time_s,
                            "filament_id": "f0",
                            "point_id": point_id,
                            "x": x,
                            "y": y,
                            "quality": 1.0,
                            "frame": frame,
                            "quality_flags": "ok",
                            "censor": 0,
                        })
            output = root / "output"
            summary = fitting.run_exploratory_fit(
                centerline,
                output,
                target_times=(0.0, 0.1),
                n_nodes=7,
                coarse_gb=(0.2,),
                coarse_chi=(0.001,),
                growth_times=(0.005,),
                refine=False,
                dt=0.001,
                write_plot=False,
            )
            self.assertEqual(summary["status"], "computed")
            self.assertEqual(summary["search"]["candidate_count"], 1)
            self.assertEqual(summary["search"]["n_nodes"], 7)
            self.assertIn("observed_features", summary["best_fit"]["target_frame"])
            self.assertIn("temporal_features", summary["best_fit"])
            self.assertTrue((output / "summary.json").is_file())
            self.assertTrue((output / "candidate_scores.csv").is_file())
            self.assertTrue((output / "best_fit_trajectory.npz").is_file())
            self.assertGreaterEqual(summary["improvement"]["frechet_distance_px_reduction"], 0.0)


if __name__ == "__main__":
    unittest.main()
