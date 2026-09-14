from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from continuum_filament_model.benchmarks.contact_folding_validation import (
    CaseSpec,
    run_benchmark,
    run_case,
)
from growing_filament.model import (
    FilamentState,
    ModelError,
    ModelParameters,
    OverdampedGrowingFilament,
    remesh_with_lineage,
)
from growing_filament.reproducibility import canonical_state_hash


class ContactFoldingValidationTests(unittest.TestCase):
    def _base(self):
        return {
            "length": 4.0,
            "n_nodes": 9,
            "axial_stiffness": 5.0,
            "bending_stiffness": 0.03,
            "drag_density": 1.0,
            "growth_rate": 1.0,
            "contact_stiffness": 2.0,
            "diameter": 0.5,
            "dt": 0.02,
            "t_end": 0.12,
            "a_max_factor": 8.0,
            "rest_length_factor": 0.75,
            "amplitude": 0.55,
            "initial_shape": "sine",
            "boundary": "free/free",
            "reject_crossing": True,
            "enable_legacy_node_contact": False,
            "dt_min": 1.0e-8,
            "max_retries": 4,
            "max_displacement_fraction": 0.5,
            "convergence_tolerance": 0.25,
        }

    def test_c1_can_disable_legacy_node_term_without_changing_legacy_default(self):
        positions = np.asarray([[0.0, 0.0], [0.4, 0.7], [0.8, 0.0]])
        rest = np.linalg.norm(np.diff(positions, axis=0), axis=1)
        legacy = OverdampedGrowingFilament(
            FilamentState(positions, rest),
            ModelParameters(contact_stiffness=3.0, diameter=1.0, reference_length=1.0),
        )
        c1 = OverdampedGrowingFilament(
            FilamentState(positions, rest),
            ModelParameters(
                contact_stiffness=3.0,
                diameter=1.0,
                reference_length=1.0,
                enable_legacy_node_contact=False,
            ),
        )
        self.assertGreater(legacy.contact_energy_components()["node_legacy"], 0.0)
        self.assertEqual(c1.contact_energy_components()["node_legacy"], 0.0)
        self.assertEqual(c1.contact_energy_components()["segment_c1"], 0.0)

    def test_lineage_is_diagnostic_and_does_not_change_canonical_state_hash(self):
        positions = np.asarray([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
        rest = np.asarray([1.0, 1.0])
        first = FilamentState(positions, rest, segment_lineage=("a", "b"))
        second = FilamentState(positions, rest, segment_lineage=("root.0", "root.1"))
        self.assertEqual(canonical_state_hash(first), canonical_state_hash(second))

    def test_lineage_descendants_survive_recursive_remeshing(self):
        positions = np.asarray([[0.0, 0.0], [4.0, 0.0], [5.0, 0.0]])
        rest = np.asarray([4.0, 1.0])
        refined_positions, refined_rest, lineage = remesh_with_lineage(
            positions, rest, ("left", "right"), a_max=1.0
        )
        self.assertEqual(len(lineage), len(refined_rest))
        self.assertTrue(all(label.startswith("left.") for label in lineage[:4]))
        self.assertEqual(lineage[-1], "right")
        self.assertEqual(len(set(lineage)), len(lineage))
        np.testing.assert_allclose(refined_positions[[0, -1]], positions[[0, -1]])

    def test_lineage_remeshing_rejects_invalid_limits_and_colliding_descendants(self):
        positions = np.asarray([[0.0, 0.0], [4.0, 0.0], [5.0, 0.0]])
        rest = np.asarray([4.0, 1.0])
        with self.assertRaises(ModelError):
            remesh_with_lineage(positions, rest, ("left", "right"), a_max=-1.0)
        with self.assertRaises(ModelError):
            remesh_with_lineage(positions, rest, ("x", "x.0"), a_max=2.0)

    def test_dynamic_case_with_initial_contact_is_unresolved(self):
        result = run_case(
            CaseSpec(
                "invalid_dynamic_fixture",
                "primary",
                {"initial_shape": "u"},
                expected_contact=True,
            ),
            self._base(),
            git_revision="test-revision",
        )
        self.assertEqual(result["initial_condition"], "non-contact-required")
        self.assertGreater(result["initial_active_contact_pairs"], 0)
        self.assertIn("initial_contact_violation", result["numerical_reason_codes"])
        self.assertEqual(result["numerical_status"], "numerically-unresolved")

    def test_dynamic_case_records_c1_observables_and_requested_vs_accepted_dt(self):
        base = self._base()
        result = run_case(
            CaseSpec("dynamic", "primary", {}, expected_contact=True),
            base,
            git_revision="test-revision",
        )
        self.assertIsNone(result["failure_reason"])
        self.assertEqual(result["numerical_status"], "resolved")
        self.assertEqual(result["onset"]["contact_onset_time"], 0.04)
        self.assertGreater(result["max_contact_residence_time"], 0.0)
        self.assertGreaterEqual(result["max_relative_tangential_slip"], 0.0)
        self.assertGreater(result["max_endpoint_motion"], 0.0)
        self.assertTrue(
            result["metrics_rows"][-1]["contact_records"]
            or result["onset"]["detachment_first_time"] is not None
        )
        self.assertTrue(
            all(row["energy_contact_node_legacy"] == 0.0 for row in result["metrics_rows"])
        )
        self.assertTrue(
            all(row["contact_action_reaction_residual"] < 1.0e-10 for row in result["metrics_rows"])
        )
        self.assertEqual(result["requested_dt"], 0.02)
        self.assertEqual(result["accepted_dt_mean"], 0.02)
        self.assertIn("not implemented", result["ccd_contract"])
        self.assertEqual(result["manifest"]["metadata"]["legacy_node_contact"], "disabled")

    def test_benchmark_writes_only_compact_contact_outputs(self):
        base = self._base()
        config = {
            "schema_version": "test",
            "base": base,
            "cases": [
                {"name": "dynamic", "group": "primary", "expected_contact": True, "overrides": {}},
                {
                    "name": "control",
                    "group": "initial_no_contact_control",
                    "expected_contact": False,
                    "overrides": {"diameter": 0.1},
                },
            ],
            "refinements": {},
            "output_policy": {"save_trajectory": False, "save_video": False},
        }
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            suite = run_benchmark(config, output, git_revision="test-revision")
            self.assertEqual(len(suite["results"]), 2)
            for name in (
                "summary.csv",
                "summary.json",
                "metrics.csv",
                "refinement_summary.csv",
                "refinement_summary.json",
                "compact_manifest.json",
                "suite.json",
            ):
                self.assertTrue((output / name).is_file(), name)
            self.assertFalse(list(output.glob("*.npz")))
            self.assertFalse(list(output.glob("*.mp4")))


if __name__ == "__main__":
    unittest.main()
