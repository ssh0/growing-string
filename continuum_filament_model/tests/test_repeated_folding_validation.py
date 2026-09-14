from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from continuum_filament_model.benchmarks.repeated_folding_validation import (
    CaseSpec,
    _compare_refinement,
    _episode_tracker,
    run_benchmark,
    run_case,
)


class RepeatedFoldingValidationTests(unittest.TestCase):
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
            "friction": False,
            "adhesion": False,
            "contact_history": False,
            "dt_min": 1.0e-8,
            "max_retries": 4,
            "max_displacement_fraction": 0.5,
        }

    @staticmethod
    def _record(pair=("0", "2"), feature="endpoint_endpoint", penetration=0.1):
        return {
            "pair": list(pair),
            "root_pair": list(pair),
            "feature": feature,
            "point_i": [0.0, 0.0],
            "point_j": [0.0, 0.1],
            "normal": [0.0, -1.0],
            "penetration": penetration,
        }

    def _row(self, time, step, records, n_nodes=3, remeshed=False):
        return {
            "raw_state_index": step,
            "time": time,
            "step": step,
            "n_nodes": n_nodes,
            "remeshed_since_previous": remeshed,
            "contact_records": records,
        }

    def test_episode_tracker_distinguishes_continuation_detachment_recontact_changes_and_remesh(
        self,
    ):
        continued = self._record()
        continued["point_i"] = [0.1, 0.0]
        rows = [
            self._row(0.0, 0, []),
            self._row(0.1, 1, [self._record()]),
            self._row(0.2, 2, [continued]),
            self._row(0.3, 3, []),
            self._row(0.4, 4, [self._record()]),
            self._row(0.5, 5, [self._record(feature="interior_interior")]),
            self._row(0.6, 6, [self._record(("0.0", "2.0"))], n_nodes=5, remeshed=True),
            self._row(0.7, 7, [], n_nodes=5),
        ]
        tracked = _episode_tracker(rows, diameter=0.5)
        events = {item["event"] for item in tracked["sequence"]}
        self.assertTrue(
            {
                "active_continuation",
                "contact_detachment",
                "recontact",
                "feature_change",
                "remesh_boundary",
            }.issubset(events)
        )
        self.assertGreaterEqual(tracked["signature"]["remesh_boundary_count"], 1)
        self.assertGreaterEqual(tracked["signature"]["recontact_count"], 1)
        self.assertGreater(tracked["cumulative_relative_tangential_slip"], 0.0)
        self.assertTrue(all("onset_n_nodes" in episode for episode in tracked["episodes"]))

    def test_long_case_records_c1_episode_metrics_without_legacy_contact(self):
        result = run_case(
            CaseSpec("dynamic", "primary_repeating", {}, expected_contact=True),
            self._base(),
            git_revision="test-revision",
        )
        self.assertEqual(result["numerical_status"], "resolved")
        self.assertGreater(result["episode_count"], 0)
        self.assertIn(
            "active_continuation", {event["event"] for event in result["episode_sequence"]}
        )
        self.assertIn("max_penetration_ratio", result)
        self.assertIn("fold_period_proxy", result["fold_summary"])
        self.assertTrue(
            all(row["energy_contact_node_legacy"] == 0.0 for row in result["metrics_rows"])
        )
        self.assertEqual(result["manifest"]["metadata"]["contact_law"], "C1 segment penalty only")
        self.assertEqual(result["manifest"]["metadata"]["friction"], "disabled")
        self.assertTrue(
            all(
                {"time", "step", "n_nodes", "remeshed_since_previous"}.issubset(row)
                for row in result["metrics_rows"]
            )
        )

    def test_unexpected_contact_marks_noncontact_control_unresolved(self):
        result = run_case(
            CaseSpec("control", "initial_no_contact_control", {}, expected_contact=False),
            self._base(),
            git_revision="test-revision",
        )
        self.assertIn("unexpected_contact_observed", result["numerical_reason_codes"])
        self.assertEqual(result["numerical_status"], "numerically-unresolved")

    def test_refinement_compares_event_order_and_numeric_status(self):
        base = self._base()
        config = {
            "base": base,
            "refinement_pairs": [
                {
                    "name": "temporal",
                    "axis": "temporal",
                    "family": "repeating",
                    "cases": ["left", "right"],
                }
            ],
        }
        signature = {
            "contact_observed": True,
            "episode_count": 2,
            "detachment_count": 1,
            "recontact_count": 1,
            "feature_change_count": 0,
            "pair_change_count": 0,
            "censored_episode_count": 0,
            "remesh_boundary_count": 0,
        }
        left_sequence = [
            {"event": "contact_onset", "time": 0.1},
            {"event": "contact_detachment", "time": 0.2},
            {"event": "recontact", "time": 0.3},
        ]
        right_sequence = [
            {"event": "contact_onset", "time": 0.1},
            {"event": "recontact", "time": 0.2},
            {"event": "contact_detachment", "time": 0.3},
        ]

        def result(name, sequence, numerical_status):
            result_signature = {
                **signature,
                "event_pattern": tuple(item["event"] for item in sequence),
            }
            return {
                "case": name,
                "effective_config": {},
                "episode_signature": result_signature,
                "episode_onset_times": [0.1, 0.3],
                "episode_sequence": sequence,
                "numerical_status": numerical_status,
                "max_penetration_ratio": 0.1,
                "max_residence_duration": 0.2,
                "fold_summary": {
                    "max_fold_count_proxy": 1,
                    "fold_spacing_proxy": 0.5,
                    "fold_period_proxy": 0.4,
                    "max_curvature_concentration": 2.0,
                },
            }

        event_comparison = _compare_refinement(
            [result("left", left_sequence, "resolved"), result("right", right_sequence, "resolved")],
            config,
        )[0]
        self.assertEqual(event_comparison["sequence_status"], "numerically-unresolved")

        comparison = _compare_refinement(
            [
                result("left", left_sequence, "resolved"),
                result("right", right_sequence, "numerically-unresolved"),
            ],
            config,
        )[0]
        self.assertEqual(comparison["sequence_status"], "numerically-unresolved")
        self.assertEqual(comparison["penetration_status"], "numerically-unresolved")
        self.assertEqual(comparison["residence_status"], "numerically-unresolved")
        self.assertEqual(comparison["fold_status"], "numerically-unresolved")
        self.assertIn("paired_case_numerically_unresolved", comparison["reason_codes"])

    def test_refinement_reports_metric_specific_unresolved_status_and_tolerances(self):
        base = self._base()
        config = {
            "base": base,
            "cases": [
                {
                    "name": "coarse",
                    "group": "temporal_refinement",
                    "refinement_axis": "temporal",
                    "refinement_family": "repeating",
                    "expected_contact": True,
                    "overrides": {},
                },
                {
                    "name": "fine",
                    "group": "temporal_refinement",
                    "refinement_axis": "temporal",
                    "refinement_family": "repeating",
                    "expected_contact": True,
                    "overrides": {"dt": 0.01},
                },
            ],
            "refinement_pairs": [
                {
                    "name": "temporal",
                    "axis": "temporal",
                    "family": "repeating",
                    "cases": ["coarse", "fine"],
                    "tolerances": {"time": 0.1, "penetration": 0.5},
                }
            ],
        }
        results = [
            run_case(
                CaseSpec(
                    "coarse",
                    "temporal_refinement",
                    {},
                    True,
                    "deterministic",
                    "temporal",
                    "repeating",
                    "coarse",
                ),
                base,
                git_revision="test",
            ),
            run_case(
                CaseSpec(
                    "fine",
                    "temporal_refinement",
                    {"dt": 0.01},
                    True,
                    "deterministic",
                    "temporal",
                    "repeating",
                    "fine",
                ),
                base,
                git_revision="test",
            ),
        ]
        refinement = _compare_refinement(results, config)
        self.assertEqual(len(refinement), 1)
        self.assertIn(refinement[0]["status"], {"resolved", "numerically-unresolved"})
        self.assertIn("sequence_status", refinement[0])
        self.assertIn("penetration_status", refinement[0])
        self.assertIn("residence_status", refinement[0])
        self.assertIn("fold_status", refinement[0])
        self.assertIn("discretization", refinement[0]["tolerances"])

    def test_benchmark_keeps_shape_sensitivity_separate_and_writes_compact_only(self):
        base = {**self._base(), "diameter": 0.1}
        config = {
            "schema_version": "test",
            "base": base,
            "cases": [
                {
                    "name": "deterministic",
                    "group": "control",
                    "expected_contact": False,
                    "overrides": {},
                },
                {
                    "name": "shape-only",
                    "group": "shape_only_sensitivity",
                    "population": "shape_only_sensitivity",
                    "expected_contact": False,
                    "overrides": {"amplitude": 0.7},
                },
            ],
            "refinement_pairs": [],
        }
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            suite = run_benchmark(config, output, git_revision="test-revision")
            self.assertEqual(
                suite["population_counts"], {"deterministic": 1, "shape_only_sensitivity": 1}
            )
            self.assertEqual(suite["refinement"], [])
            for filename in (
                "summary.csv",
                "metrics.csv",
                "refinement_summary.json",
                "compact_manifest.json",
                "suite.json",
            ):
                self.assertTrue((output / filename).is_file(), filename)
            self.assertFalse(list(output.glob("*.npz")))
            self.assertFalse(list(output.glob("*.mp4")))
            with (output / "metrics.csv").open(newline="", encoding="utf-8") as stream:
                metrics = next(csv.DictReader(stream))
            self.assertIsInstance(json.loads(metrics["contact_records"]), list)
            self.assertIsInstance(json.loads(metrics["segment_lineage"]), list)


if __name__ == "__main__":
    unittest.main()
