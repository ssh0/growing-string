from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from continuum_filament_model.benchmarks.scale_free_shape_comparison import run_bounded_comparison


class ScaleFreeBenchmarkTests(unittest.TestCase):
    def test_missing_video_keeps_deterministic_run_separate_and_censored(self):
        config = {
            "base": {
                "n_nodes": 5,
                "dt": 0.001,
                "t_end": 0.004,
                "max_report_rows": 8,
            },
            "fixtures": [
                {"name": "fast_growth_low_bend", "overrides": {"growth_rate": 0.02, "amplitude": 0.02}},
                {"name": "fast_growth_high_bend", "overrides": {"growth_rate": 0.02, "amplitude": 0.02}},
            ],
            "refinement": {"base_fixture": "fast_growth_low_bend", "n_nodes": [5, 7], "dt_values": [0.001, 0.0005]},
            "replicates": {"fixtures": ["fast_growth_low_bend"], "seeds": [17], "noise_fraction": 0.1},
            "video_model_case": "fast_growth_low_bend",
            "video": {"background": "none", "contrast": "none", "threshold": "absolute", "threshold_value": 0.5, "frame_stride": 1, "min_component_size": 2, "max_components": 1},
        }
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "result"
            report = run_bounded_comparison(
                config,
                output,
                video_path=Path(directory) / "missing-gray5.mp4",
            )
            self.assertEqual(report["extraction"]["status"], "input_quality_missing_video")
            self.assertEqual(
                [row["run_name"] for row in report["model_runs"]],
                [
                    "fast_growth_low_bend",
                    "fast_growth_high_bend",
                    "fast_growth_low_bend_initial_condition_sensitivity",
                ],
            )
            self.assertEqual(report["model_populations"]["stage2_deterministic"], ["fast_growth_low_bend", "fast_growth_high_bend"])
            sensitivity_names = report["model_populations"]["initial_condition_sensitivity"]
            self.assertEqual(sensitivity_names, ["fast_growth_low_bend_initial_condition_sensitivity"])
            sensitivity = next(row for row in report["model_runs"] if row["run_name"] == sensitivity_names[0])
            self.assertEqual(sensitivity["run_kind"], "initial_condition_sensitivity")
            self.assertEqual(len(sensitivity["sensitivity_protocol"]["members"]), 3)
            for member in sensitivity["sensitivity_protocol"]["members"]:
                self.assertTrue((output / "_runs" / sensitivity_names[0] / member["artifact_path"]).is_file())
            self.assertEqual(report["registration"]["status"], "not_required_not_inferred")
            self.assertEqual(report["model_inadequacy"], "not_assessed_in_scale_free_morphology_mode")
            persisted = json.loads((output / "compact_summary.json").read_text(encoding="utf-8"))
            self.assertEqual(persisted["model_runs"][0]["run_kind"], "deterministic_fixture")
            self.assertEqual(persisted["model_populations"]["initial_condition_sensitivity"], sensitivity_names)
            self.assertTrue((output / "compact_manifest.json").is_file())
            self.assertTrue((output / "summary.csv").is_file())


if __name__ == "__main__":
    unittest.main()
