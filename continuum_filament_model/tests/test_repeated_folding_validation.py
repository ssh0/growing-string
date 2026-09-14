from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from continuum_filament_model.benchmarks.repeated_folding_validation import (
    CaseSpec,
    _compare_refinement,
    _effective_case,
    _episode_tracker,
    _fold_summary,
    _initial_state,
    run_benchmark,
    run_case,
    validate_case_config,
    ValidationError,
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
    def _record(
        pair=("0", "2"), feature="endpoint_endpoint", penetration=0.1, root_pair=None
    ):
        return {
            "pair": list(pair),
            "root_pair": list(pair if root_pair is None else root_pair),
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
            self._row(0.55, 5, [self._record()]),
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
        self.assertEqual(tracked["signature"]["recontact_count"], 1)
        self.assertGreater(tracked["cumulative_relative_tangential_slip"], 0.0)
        self.assertTrue(all("onset_n_nodes" in episode for episode in tracked["episodes"]))

    def test_episode_tracker_keeps_one_to_many_root_contacts_active(self):
        child = self._record(("0.1", "2.1"), root_pair=("0", "2"))
        tracked = _episode_tracker(
            [
                self._row(0.0, 0, []),
                self._row(0.1, 1, [self._record()]),
                self._row(0.2, 2, [self._record(), child]),
                self._row(0.3, 3, [self._record(), child]),
            ],
            diameter=0.5,
        )
        self.assertEqual(tracked["signature"]["episode_count"], 2)
        self.assertEqual(tracked["signature"]["pair_change_count"], 0)
        self.assertNotIn("pair_change", {event["event"] for event in tracked["sequence"]})

    def test_one_to_many_contact_closes_disappeared_child_and_recontacts(self):
        child = self._record(("0.1", "2.1"), root_pair=("0", "2"))
        tracked = _episode_tracker(
            [
                self._row(0.0, 0, []),
                self._row(0.1, 1, [self._record()]),
                self._row(0.2, 2, [self._record(), child]),
                self._row(0.3, 3, [child]),
                self._row(0.4, 4, [self._record(), child]),
            ],
            diameter=0.5,
        )
        self.assertEqual(tracked["signature"]["detachment_count"], 1)
        self.assertEqual(tracked["signature"]["recontact_count"], 1)
        self.assertEqual(tracked["signature"]["pair_change_count"], 0)
        self.assertEqual(tracked["episodes"][0]["detachment_time"], 0.3)
        self.assertEqual(tracked["episodes"][2]["close_reason"], "end_censored")

    def test_remesh_boundary_resets_detached_contact_identity(self):
        tracked = _episode_tracker(
            [
                self._row(0.0, 0, []),
                self._row(0.1, 1, [self._record()]),
                self._row(0.2, 2, []),
                self._row(0.3, 3, [], n_nodes=5, remeshed=True),
                self._row(0.4, 4, [self._record()]),
            ],
            diameter=0.5,
        )
        self.assertEqual(tracked["signature"]["recontact_count"], 0)
        self.assertFalse(tracked["signature"]["repeated_episode_signature"])

    def test_episode_tracker_censors_active_episode_at_end(self):
        tracked = _episode_tracker(
            [self._row(0.0, 0, []), self._row(0.1, 1, [self._record()])],
            diameter=0.5,
        )
        self.assertTrue(tracked["episodes"][0]["censored_at_end"])
        self.assertEqual(tracked["signature"]["censored_episode_count"], 1)

    def test_pair_change_is_recorded_when_root_pair_replaces_contact(self):
        replacement = self._record(("0", "3"), root_pair=("0", "2"))
        tracked = _episode_tracker(
            [
                self._row(0.0, 0, []),
                self._row(0.1, 1, [self._record()]),
                self._row(0.2, 2, [replacement]),
                self._row(0.3, 3, [replacement]),
            ],
            diameter=0.5,
        )
        self.assertEqual(tracked["signature"]["pair_change_count"], 1)
        self.assertNotIn("contact_detachment", {event["event"] for event in tracked["sequence"]})

    def test_fold_period_tracks_repeated_count_increases(self):
        rows = [
            {"time": 0.0, "fold_count_proxy": 0, "fold_spacing": None, "curvature_concentration": 1.0},
            {"time": 0.1, "fold_count_proxy": 1, "fold_spacing": None, "curvature_concentration": 1.0},
            {"time": 0.2, "fold_count_proxy": 0, "fold_spacing": None, "curvature_concentration": 1.0},
            {"time": 0.3, "fold_count_proxy": 1, "fold_spacing": None, "curvature_concentration": 1.0},
        ]
        summary = _fold_summary(rows)
        self.assertEqual(summary["fold_event_times"], [0.1, 0.3])
        self.assertAlmostEqual(summary["fold_period_proxy"], 0.2)

    def test_feature_change_back_is_not_recontact(self):
        tracked = _episode_tracker(
            [
                self._row(0.0, 0, []),
                self._row(0.1, 1, [self._record()]),
                self._row(0.2, 2, [self._record(feature="interior_interior")]),
                self._row(0.3, 3, [self._record()]),
            ],
            diameter=0.5,
        )
        self.assertEqual(tracked["signature"]["recontact_count"], 0)
        self.assertFalse(tracked["signature"]["repeated_episode_signature"])

    def test_repeated_contract_and_crossing_guard_configuration(self):
        result = run_case(
            CaseSpec(
                "single",
                "primary_repeating",
                {},
                expected_contact=True,
                expected_repeated=True,
            ),
            {**self._base(), "t_end": 0.02},
            git_revision="test-revision",
        )
        self.assertIn("repeated_folding_not_observed", result["numerical_reason_codes"])
        self.assertNotEqual(result["evidence_classification"], "repeated-folding")
        with self.assertRaises(ValidationError):
            validate_case_config({**self._base(), "reject_crossing": False})
        with self.assertRaises(ValidationError):
            validate_case_config({**self._base(), "reject_crossing": "false"})
        with self.assertRaises(ValidationError):
            validate_case_config({**self._base(), "expected_contact": "false"})
        with self.assertRaises(ValidationError):
            validate_case_config({**self._base(), "expected_repeated": "false"})
        with self.assertRaises(ValidationError):
            validate_case_config({**self._base(), "length": 10**400})
        with self.assertRaises(ValidationError):
            validate_case_config({**self._base(), "n_nodes": 10**400})

    def test_refinement_rejects_unbounded_tolerances(self):
        base = self._base()
        config = {
            "base": base,
            "refinement_pairs": [
                {
                    "name": "temporal",
                    "cases": ["left", "right"],
                    "tolerances": {"identity": float("inf")},
                }
            ],
        }
        results = [
            {
                "case": name,
                "effective_config": {},
                "episode_signature": {
                    "contact_observed": False,
                    "episode_count": 0,
                    "detachment_count": 0,
                    "recontact_count": 0,
                    "feature_change_count": 0,
                    "pair_change_count": 0,
                    "censored_episode_count": 0,
                    "censor_pattern": (),
                    "event_pattern": (),
                },
                "episode_onset_times": [],
                "episode_sequence": [],
                "numerical_status": "resolved",
                "max_penetration_ratio": 0.0,
                "max_residence_duration": 0.0,
                "fold_summary": {
                    "max_fold_count_proxy": 0,
                    "fold_spacing_proxy": None,
                    "fold_period_proxy": None,
                    "max_curvature_concentration": 0.0,
                },
            }
            for name in ("left", "right")
        ]
        with self.assertRaises(ValidationError):
            _compare_refinement(results, config)
        for tolerance in (
            {"identity": 1.0},
            {"time": base["t_end"]},
            {"identity": 10**400},
        ):
            bounded_config = {
                "base": base,
                "refinement_pairs": [
                    {"name": "temporal", "cases": ["left", "right"], "tolerances": tolerance}
                ],
            }
            with self.assertRaises(ValidationError):
                _compare_refinement(results, bounded_config)

        short_results = [
            {**result, "effective_config": {"t_end": 0.1}} for result in results
        ]
        short_config = {
            "base": {**base, "t_end": 2.4},
            "refinement_pairs": [
                {
                    "name": "temporal",
                    "cases": ["left", "right"],
                    "tolerances": {"time": 2.3},
                }
            ],
        }
        with self.assertRaises(ValidationError):
            _compare_refinement(short_results, short_config)

    def test_explicit_refinement_rejects_shape_population_mixing(self):
        results = [
            {
                "case": "deterministic",
                "effective_config": {"population": "deterministic"},
            },
            {
                "case": "shape",
                "effective_config": {"population": "shape_only_sensitivity"},
            },
        ]
        config = {
            "base": self._base(),
            "refinement_pairs": [
                {"name": "mixed", "cases": ["deterministic", "shape"]}
            ],
        }
        with self.assertRaises(ValidationError):
            _compare_refinement(results, config)

        deterministic_results = [
            {"case": name, "effective_config": {"population": "deterministic"}}
            for name in ("left", "right")
        ]
        declared_shape_config = {
            "base": self._base(),
            "refinement_pairs": [
                {
                    "name": "declared-shape",
                    "population": "shape_only_sensitivity",
                    "cases": ["left", "right"],
                }
            ],
        }
        with self.assertRaises(ValidationError):
            _compare_refinement(deterministic_results, declared_shape_config)

        duplicate_config = {
            "base": self._base(),
            "refinement_pairs": [
                {"name": "duplicate", "cases": ["left", "left"]}
            ],
        }
        with self.assertRaises(ValidationError):
            _compare_refinement(deterministic_results, duplicate_config)

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
        self.assertTrue(
            all(
                "contact_identity" in record
                for row in result["metrics_rows"]
                for record in row["contact_records"]
            )
        )
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
            "censor_pattern": (False, False),
            "remesh_boundary_count": 0,
        }
        left_sequence = [
            {
                "event": "contact_onset",
                "time": 0.1,
                "root_pair": ["0", "2"],
                "feature": "endpoint_endpoint",
            },
            {
                "event": "contact_detachment",
                "time": 0.2,
                "root_pair": ["0", "2"],
                "feature": "endpoint_endpoint",
            },
            {
                "event": "recontact",
                "time": 0.3,
                "root_pair": ["0", "2"],
                "feature": "endpoint_endpoint",
            },
        ]
        right_sequence = [
            {
                "event": "contact_onset",
                "time": 0.1,
                "root_pair": ["0", "2"],
                "feature": "endpoint_endpoint",
            },
            {
                "event": "recontact",
                "time": 0.2,
                "root_pair": ["0", "2"],
                "feature": "endpoint_endpoint",
            },
            {
                "event": "contact_detachment",
                "time": 0.3,
                "root_pair": ["0", "2"],
                "feature": "endpoint_endpoint",
            },
        ]

        def result(name, sequence, numerical_status, censor_pattern=(False, False)):
            result_signature = {
                **signature,
                "censor_pattern": censor_pattern,
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

        identity_sequence = [dict(item) for item in left_sequence]
        identity_sequence[0]["root_pair"] = ["1", "3"]
        identity_comparison = _compare_refinement(
            [
                result("left", left_sequence, "resolved"),
                result("right", identity_sequence, "resolved"),
            ],
            config,
        )[0]
        self.assertEqual(identity_comparison["sequence_status"], "numerically-unresolved")
        self.assertIn(
            "episode_contact_identity_changed", identity_comparison["reason_codes"]
        )

        mesh_left = [
            {**item, "contact_identity": [0.25, 0.75]} for item in left_sequence
        ]
        mesh_right = [
            {
                **item,
                "root_pair": ["1", "3"],
                "contact_identity": [0.27, 0.73],
            }
            for item in left_sequence
        ]
        mesh_comparison = _compare_refinement(
            [
                result("left", mesh_left, "resolved"),
                result("right", mesh_right, "resolved"),
            ],
            config,
        )[0]
        self.assertEqual(mesh_comparison["sequence_status"], "resolved")

        censor_comparison = _compare_refinement(
            [
                result("left", left_sequence, "resolved"),
                result("right", left_sequence, "resolved", censor_pattern=(True, False)),
            ],
            config,
        )[0]
        self.assertEqual(censor_comparison["sequence_status"], "numerically-unresolved")

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

    def test_shape_sensitivity_keeps_reference_lengths_fixed(self):
        base = self._base()
        low = _effective_case(
            base,
            CaseSpec(
                "low",
                "shape_only_sensitivity",
                {"amplitude": 0.45},
                expected_contact=False,
                population="shape_only_sensitivity",
            ),
        )
        high = _effective_case(
            base,
            CaseSpec(
                "high",
                "shape_only_sensitivity",
                {"amplitude": 0.70},
                expected_contact=False,
                population="shape_only_sensitivity",
            ),
        )
        self.assertEqual(
            _initial_state(low).rest_lengths.tolist(), _initial_state(high).rest_lengths.tolist()
        )
        with self.assertRaises(ValidationError):
            _effective_case(
                base,
                CaseSpec(
                    "invalid",
                    "shape_only_sensitivity",
                    {"rest_length_factor": 0.8},
                    expected_contact=False,
                    population="shape_only_sensitivity",
                ),
            )

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
