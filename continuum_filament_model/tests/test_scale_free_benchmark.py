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
                {"name": "fixture", "overrides": {"growth_rate": 0.02, "amplitude": 0.02}},
            ],
            "refinement": {"base_fixture": "fixture", "n_nodes": [5, 7], "dt_values": [0.001, 0.0005]},
            "replicates": {"fixtures": ["fixture"], "seeds": [17], "noise_fraction": 0.1},
            "video_model_case": "fixture",
            "video": {"background": "none", "contrast": "none", "threshold": "absolute", "threshold_value": 0.5, "frame_stride": 1, "min_component_size": 2, "max_components": 1},
        }
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "result"
            report = run_bounded_comparison(
                config,
                output,
                video_path=Path(directory) / "missing-gray5.mp4",
                model_cases=("fixture",),
            )
            self.assertEqual(report["extraction"]["status"], "input_quality_missing_video")
            self.assertEqual([row["run_name"] for row in report["deterministic_model_runs"]], ["fixture"])
            self.assertEqual(report["replicate_model_runs"], [])
            self.assertEqual(report["registration"]["status"], "not_required_not_inferred")
            self.assertEqual(report["model_inadequacy"], "not_assessed_in_scale_free_morphology_mode")
            persisted = json.loads((output / "compact_summary.json").read_text(encoding="utf-8"))
            self.assertEqual(persisted["model_runs"][0]["population"], "deterministic_or_unclassified")
            self.assertTrue((output / "compact_manifest.json").is_file())
            self.assertTrue((output / "summary.csv").is_file())


if __name__ == "__main__":
    unittest.main()
