from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from continuum_filament_model.benchmarks.dense_buckling_heatmap import _run_cell, load_config
from continuum_filament_model.benchmarks.video_presentation_export import run_export


class PresentationDataArtifactTests(unittest.TestCase):
    ROOT = Path(__file__).resolve().parents[1]
    PRESENTATION = ROOT / "results" / "presentation_data"

    @staticmethod
    def _sha256(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()

    def test_dense_heatmap_has_continuous_observables_and_unresolved_decomposition(self):
        root = self.PRESENTATION / "dense_buckling"
        heatmap = json.loads((root / "heatmap.json").read_text(encoding="utf-8"))
        rows = heatmap["rows"]
        self.assertEqual(len(rows), 56)
        self.assertEqual(len(heatmap["axes"]["G_b"]), 7)
        self.assertEqual(len(heatmap["axes"]["chi"]), 8)
        required = {
            "A_max_over_L", "first_mode_fraction", "curvature_rms", "onset_time",
            "dissipation_energy", "dominant_mode", "unresolved_reason",
        }
        self.assertTrue(all(required <= set(row) for row in rows))
        unresolved = [record for record in heatmap["records"] if record["classification"]["label"] == "unresolved"]
        self.assertTrue(unresolved)
        self.assertTrue(all(item["waveform_classification"]["reason"] for item in unresolved))
        for item in unresolved:
            spectrum = item["mode_spectrum"]["mode_fractions"]
            self.assertAlmostEqual(float(np.linalg.norm(spectrum)), 1.0, places=10)
            self.assertIsNotNone(item["dominant_mode"])

    def test_dense_cell_is_deterministic_for_same_configuration(self):
        config = load_config(self.ROOT / "benchmarks" / "configs" / "p1b2_dense_heatmap.json")
        base = dict(config["base"])
        base["t_end"] = 0.004
        first, first_trajectory = _run_cell(base, 0.2, 0.00025)
        second, second_trajectory = _run_cell(base, 0.2, 0.00025)
        self.assertEqual(first, second)
        self.assertEqual(len(first_trajectory), len(second_trajectory))
        for left, right in zip(first_trajectory, second_trajectory):
            np.testing.assert_array_equal(left.positions, right.positions)
            np.testing.assert_array_equal(left.rest_lengths, right.rest_lengths)
            self.assertEqual(left.time, right.time)
            self.assertEqual(left.step, right.step)

    def test_dense_representative_snapshots_are_bounded_and_complete(self):
        root = self.PRESENTATION / "dense_buckling"
        payload = json.loads((root / "snapshots.json").read_text(encoding="utf-8"))
        for role in ("straight", "single_buckling", "higher_mode", "boundary_near"):
            self.assertIn(role, payload["representatives"])
            case = payload["cases"][role]
            self.assertEqual(len(case["snapshots"]), 3)
            for snapshot in case["snapshots"]:
                self.assertEqual(len(snapshot["x"]), len(snapshot["y"]))
                self.assertEqual(len(snapshot["x"]), snapshot["n_nodes"])
                self.assertTrue(np.isfinite(snapshot["x"]).all())
                self.assertTrue(np.isfinite(snapshot["y"]).all())
        self.assertNotEqual(payload["representatives"]["higher_mode"], payload["representatives"]["boundary_near"])

    def test_contact_snapshots_include_geometry_and_normal_force(self):
        root = self.PRESENTATION / "contact_snapshots"
        payload = json.loads((root / "contact_snapshots.json").read_text(encoding="utf-8"))
        self.assertEqual(len(payload["cases"]), 2)
        for case in payload["cases"]:
            self.assertEqual(len(case["snapshots"]), 3)
            self.assertGreater(case["max_active_contact_pairs"], 0)
            self.assertIn(case["classification"], {"self-contact", "folding-contact"})
            for snapshot in case["snapshots"]:
                for contact in snapshot["contacts"]:
                    self.assertEqual(len(contact["contact_point"]), 2)
                    self.assertAlmostEqual(np.linalg.norm(contact["normal"]), 1.0, places=10)
                    self.assertAlmostEqual(
                        np.linalg.norm(contact["normal_force"]),
                        contact["normal_force_magnitude"],
                        places=10,
                    )

    def test_artifact_hashes_match_committed_compact_files(self):
        for relative in ("dense_buckling/manifest.json", "contact_snapshots/manifest.json", "video_gray5/manifest.json"):
            manifest_path = self.PRESENTATION / relative
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            for name, artifact in manifest.get("artifacts", {}).items():
                artifact_path = manifest_path.parent / name
                self.assertTrue(artifact_path.is_file(), str(artifact_path))
                self.assertEqual(artifact["bytes"], artifact_path.stat().st_size)
                self.assertEqual(artifact["sha256"], self._sha256(artifact_path))

    def test_missing_video_export_is_explicit_and_deterministic(self):
        with tempfile.TemporaryDirectory() as directory:
            first = Path(directory) / "first"
            second = Path(directory) / "second"
            missing = Path(directory) / "gray5.mp4"
            first_result = run_export(missing, first)
            second_result = run_export(missing, second)
            self.assertEqual(first_result["status"], "input_missing")
            self.assertFalse(first_result["raw_centerline_available"])
            self.assertEqual(
                (first / "video_presentation.json").read_bytes(),
                (second / "video_presentation.json").read_bytes(),
            )


if __name__ == "__main__":
    unittest.main()
