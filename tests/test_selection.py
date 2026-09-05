import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from matchquant.calibration import calibration_diagnostics
from matchquant.io import load_matches
from matchquant.selection import load_candidates, parse_validation_folds, run_nested_selection


class SelectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.matches = load_matches("examples/synthetic_matches.csv")
        cls.folds = parse_validation_folds(
            ["2024-01-03..2024-01-05", "2024-01-06..2024-01-06"]
        )
        cls.candidates = [
            {
                "candidate_order": 0,
                "candidate_name": "frequency_a",
                "model": "frequency",
                "config": {"alpha": 1.0},
            },
            {
                "candidate_order": 1,
                "candidate_name": "elo_b",
                "model": "elo",
                "config": {"k_factor": 20.0},
            },
        ]

    def test_calibration_table_has_stable_shape_finite_values_and_totals(self):
        predictions = pd.DataFrame(
            {
                "actual_result": ["H", "D", "A", "H"],
                "home_win_prob": [0.9, 0.2, 0.1, 0.7],
                "draw_prob": [0.05, 0.6, 0.2, 0.1],
                "away_win_prob": [0.05, 0.2, 0.7, 0.2],
            }
        )
        summary, table = calibration_diagnostics(predictions, n_bins=5)
        self.assertEqual(len(table), 15)
        self.assertEqual(set(table["outcome"]), {"home_win", "draw", "away_win"})
        for outcome in table["outcome"].unique():
            self.assertEqual(
                int(table.loc[table["outcome"].eq(outcome), "sample_count"].sum()),
                len(predictions),
            )
        numeric = table.select_dtypes(include=[np.number]).to_numpy(dtype=float)
        self.assertTrue(np.isfinite(numeric).all())
        self.assertTrue(np.isfinite(summary["multiclass_ece"]))

    def test_final_test_labels_do_not_change_selection(self):
        original = run_nested_selection(
            self.matches,
            self.candidates,
            self.folds,
            "2024-01-07",
            final_test_end="2024-01-10",
        )
        altered_matches = self.matches.copy()
        altered_matches.loc[altered_matches["match_id"].eq("m012"), "home_goals"] = 9
        altered_matches.loc[altered_matches["match_id"].eq("m012"), "away_goals"] = 0
        altered = run_nested_selection(
            altered_matches,
            self.candidates,
            self.folds,
            "2024-01-07",
            final_test_end="2024-01-10",
        )
        self.assertEqual(original["selected_candidate"]["candidate_name"], altered["selected_candidate"]["candidate_name"])
        self.assertEqual(original["ranking"], altered["ranking"])

    def test_same_batch_labels_do_not_change_earlier_final_predictions(self):
        original = run_nested_selection(
            self.matches, self.candidates, self.folds, "2024-01-07", final_test_end="2024-01-10"
        )
        altered_matches = self.matches.copy()
        altered_matches.loc[altered_matches["match_id"].eq("m008"), "home_goals"] = 0
        altered_matches.loc[altered_matches["match_id"].eq("m008"), "away_goals"] = 4
        altered = run_nested_selection(
            altered_matches, self.candidates, self.folds, "2024-01-07", final_test_end="2024-01-10"
        )
        columns = ["home_win_prob", "draw_prob", "away_win_prob"]
        pd.testing.assert_frame_equal(
            original["final_run"].predictions.loc[:1, columns].reset_index(drop=True),
            altered["final_run"].predictions.loc[:1, columns].reset_index(drop=True),
        )

    def test_validation_fold_overlap_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "chronological and non-overlapping"):
            run_nested_selection(
                self.matches,
                self.candidates,
                parse_validation_folds(["2024-01-03..2024-01-06", "2024-01-06..2024-01-06"]),
                "2024-01-07",
            )

    def test_exact_tie_uses_candidate_name(self):
        tied = [
            {"candidate_order": 0, "candidate_name": "zeta", "model": "frequency", "config": {}},
            {"candidate_order": 1, "candidate_name": "alpha", "model": "frequency", "config": {}},
        ]
        result = run_nested_selection(self.matches, tied, self.folds, "2024-01-07")
        self.assertEqual(result["selected_candidate"]["candidate_name"], "alpha")
        self.assertEqual(result["ranking"][0]["candidate_name"], "alpha")

    def test_candidate_config_loads_all_existing_models(self):
        candidates, payload = load_candidates("examples/m3_candidates.json")
        self.assertEqual(payload["schema_version"], "m3-candidate-config-v1")
        self.assertEqual({candidate["model"] for candidate in candidates}, {"frequency", "elo", "poisson"})
        self.assertEqual([candidate["candidate_order"] for candidate in candidates], list(range(6)))

    def test_cli_smoke_writes_timestamped_auditable_run(self):
        with tempfile.TemporaryDirectory() as temporary:
            output_root = Path(temporary) / "runs"
            command = [
                sys.executable,
                "-m",
                "matchquant",
                "select",
                "--input",
                "examples/synthetic_matches.csv",
                "--candidate-config",
                "examples/m3_candidates.json",
                "--validation-fold",
                "2024-01-03..2024-01-05",
                "--validation-fold",
                "2024-01-06..2024-01-06",
                "--final-test-start",
                "2024-01-07",
                "--final-test-end",
                "2024-01-10",
                "--output-root",
                str(output_root),
            ]
            completed = subprocess.run(command, capture_output=True, text=True, check=False)
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertLessEqual(len(completed.stdout.strip().splitlines()), 20)
            run_dirs = list(output_root.glob("m3_select_*"))
            self.assertEqual(len(run_dirs), 1)
            run_dir = run_dirs[0]
            expected = {
                "run_config.json",
                "validation_metrics.json",
                "predictions.csv",
                "final_metrics.json",
                "calibration_reliability.csv",
            }
            self.assertEqual({path.name for path in run_dir.iterdir()}, expected)
            config = json.loads((run_dir / "run_config.json").read_text(encoding="utf-8"))
            self.assertEqual(config["final_test"]["start"], "2024-01-07")
            self.assertFalse(config["selection_rule"]["final_test_metrics_used_for_selection"])
            self.assertEqual(config["data_provenance"]["status"], "missing_manifest")
            self.assertIn("final_test_metrics", config)


if __name__ == "__main__":
    unittest.main()
