import json
import hashlib
import tempfile
import unittest
from pathlib import Path

import pandas as pd
from pandas.testing import assert_frame_equal

from matchquant.fixtures import validate_fixtures
from matchquant.forecast import forecast_fixtures, run_forecast
from matchquant.io import file_sha256, load_matches, write_json
from matchquant.models import FrequencyModel
from matchquant.schema import DataValidationError


class ForecastTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.history_path = Path("examples/synthetic_matches.csv").resolve()
        cls.fixtures_path = Path("examples/synthetic_fixtures.csv").resolve()
        cls.history = load_matches(cls.history_path)
        cls.fixtures = validate_fixtures(pd.read_csv(cls.fixtures_path, dtype="string"))

    def test_date_only_as_of_excludes_same_day_and_future_score_changes(self):
        altered = self.history.copy()
        altered.loc[altered["match_id"].eq("m008"), ["home_goals", "away_goals"]] = [0, 4]
        altered.loc[altered["match_id"].eq("m012"), ["home_goals", "away_goals"]] = [9, 0]
        for model in ("frequency", "elo", "poisson"):
            original = forecast_fixtures(self.history, self.fixtures, model, "2024-01-07")
            changed = forecast_fixtures(altered, self.fixtures, model, "2024-01-07")
            assert_frame_equal(
                original.predictions,
                changed.predictions,
                check_dtype=False,
            )
            self.assertEqual(original.history_rows_used, 7)

    def test_same_batch_fixture_output_is_stable_and_fixture_does_not_update_model(self):
        model = FrequencyModel(alpha=0.0)
        first = forecast_fixtures(self.history, self.fixtures, model, "2024-01-07")
        shuffled = self.fixtures.sample(frac=1, random_state=17).reset_index(drop=True)
        second = forecast_fixtures(self.history, shuffled, "frequency", "2024-01-07", model_config={"alpha": 0.0})
        assert_frame_equal(first.predictions, second.predictions, check_dtype=False)
        self.assertEqual(int(model.counts.sum()), 7)

    def test_exact_time_as_of_uses_strict_prior_batches(self):
        history = pd.DataFrame(
            [
                ["h1", "2024-01-01T09:00:00", "S1", "A", "B", 1, 0],
                ["h2", "2024-01-01T10:00:00", "S1", "C", "D", 0, 2],
                ["h3", "2024-01-01T11:00:00", "S1", "B", "C", 3, 0],
            ],
            columns=["match_id", "date", "season", "home_team", "away_team", "home_goals", "away_goals"],
        )
        fixtures = pd.DataFrame(
            [["f1", "2024-01-01T10:00:00", "S1", "A", "C"]],
            columns=["fixture_id", "date", "season", "home_team", "away_team"],
        )
        altered = history.copy()
        altered.loc[altered["match_id"].eq("h2"), ["home_goals", "away_goals"]] = [4, 0]
        altered.loc[altered["match_id"].eq("h3"), ["home_goals", "away_goals"]] = [0, 4]
        original = forecast_fixtures(history, fixtures, "frequency", "2024-01-01T10:00:00")
        changed = forecast_fixtures(altered, fixtures, "frequency", "2024-01-01T10:00:00")
        self.assertEqual(original.history_rows_used, 1)
        assert_frame_equal(original.predictions, changed.predictions, check_dtype=False)

    def test_exact_as_of_rejects_date_only_rows_on_that_day(self):
        with self.assertRaisesRegex(ValueError, "date-only"):
            forecast_fixtures(self.history, self.fixtures, "frequency", "2024-01-07T12:00:00")

    def test_fixture_before_as_of_is_rejected(self):
        too_early = self.fixtures.copy()
        too_early.loc[0, "date"] = "2024-01-06"
        with self.assertRaisesRegex(DataValidationError, "on or after as_of"):
            forecast_fixtures(self.history, too_early, "frequency", "2024-01-07")

    def test_explicit_model_config_and_required_artifacts(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config_path = root / "poisson.json"
            write_json({"prior_weight": 3.0}, config_path)
            result = run_forecast(
                self.history_path,
                self.fixtures_path,
                "2024-01-07",
                model_name="poisson",
                model_config_path=config_path,
                output_root=root / "runs",
            )
            run_dir = Path(result["run_dir"])
            self.assertEqual(result["selection_evidence_status"], "explicit_model_config")
            self.assertTrue(all((run_dir / name).is_file() for name in ("predictions.csv", "forecast_report.md", "run_config.json")))
            payload = json.loads((run_dir / "run_config.json").read_text(encoding="utf-8"))
            self.assertEqual(payload["history"]["rows_used_before_as_of"], 7)
            self.assertFalse(payload["validation"]["fixture_results_used_for_model_updates"])
            probabilities = result["predictions"][["home_win_prob", "draw_prob", "away_win_prob"]]
            self.assertTrue((probabilities >= 0).all().all())
            self.assertTrue((probabilities.sum(axis=1).sub(1).abs() < 1e-8).all())

    def test_selection_run_config_is_extracted_and_hash_mismatch_is_marked(self):
        candidate_hash = hashlib.sha256(
            json.dumps({"alpha": 1.0}, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            selection_path = root / "m3" / "run_config.json"
            matched = {
                "schema_version": "m3-selection-v1",
                "input_sha256": file_sha256(self.history_path),
                "season_mode": "carry",
                "selected_candidate": {
                    "candidate_name": "frequency_alpha_1",
                    "model": "frequency",
                    "config": {"alpha": 1.0},
                    "configuration_sha256": candidate_hash,
                },
            }
            write_json(matched, selection_path)
            result = run_forecast(
                self.history_path,
                self.fixtures_path,
                "2024-01-07",
                selection_run_config_path=selection_path,
                output_root=root / "matched-runs",
            )
            self.assertEqual(result["selection_evidence_status"], "matched")
            mismatch = dict(matched)
            mismatch["input_sha256"] = "0" * 64
            mismatch_path = root / "mismatch.json"
            write_json(mismatch, mismatch_path)
            mismatch_result = run_forecast(
                self.history_path,
                self.fixtures_path,
                "2024-01-07",
                selection_run_config_path=mismatch_path,
                output_root=root / "mismatch-runs",
            )
            self.assertEqual(mismatch_result["selection_evidence_status"], "history_hash_mismatch")
            payload = json.loads(Path(mismatch_result["run_config_path"]).read_text(encoding="utf-8"))
            self.assertEqual(payload["selection"]["history_hash_status"], "mismatch")
            self.assertIn("history hash", Path(mismatch_result["forecast_report_path"]).read_text(encoding="utf-8"))

    def test_selection_and_explicit_model_modes_are_mutually_exclusive(self):
        with tempfile.TemporaryDirectory() as temporary:
            config_path = Path(temporary) / "model.json"
            write_json({}, config_path)
            with self.assertRaisesRegex(ValueError, "mutually exclusive"):
                run_forecast(
                    self.history_path,
                    self.fixtures_path,
                    "2024-01-07",
                    selection_run_config_path=config_path,
                    model_name="frequency",
                    model_config_path=config_path,
                    output_root=Path(temporary) / "runs",
                )


if __name__ == "__main__":
    unittest.main()
