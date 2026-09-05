import unittest

import pandas as pd

from matchquant.backtest import run_models, run_walk_forward
from matchquant.io import load_matches
from matchquant.metrics import evaluate_predictions


class BacktestTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.matches = load_matches("examples/synthetic_matches.csv")

    def test_all_models_share_test_matches(self):
        runs = run_models(self.matches, ["frequency", "elo", "poisson"], "2024-01-07")
        self.assertEqual({run.test_matches for run in runs.values()}, {5})
        ids = [tuple(run.predictions["match_id"]) for run in runs.values()]
        self.assertTrue(all(value == ids[0] for value in ids[1:]))

    def test_same_day_results_do_not_affect_same_day_predictions(self):
        altered = self.matches.copy()
        altered.loc[altered["match_id"] == "m008", "home_goals"] = 0
        altered.loc[altered["match_id"] == "m008", "away_goals"] = 4
        for model_name in ("frequency", "elo", "poisson"):
            original = run_walk_forward(self.matches, model_name, "2024-01-07")
            changed = run_walk_forward(altered, model_name, "2024-01-07")
            for column in ("home_win_prob", "draw_prob", "away_win_prob"):
                pd.testing.assert_series_equal(
                    original.predictions.loc[:1, column],
                    changed.predictions.loc[:1, column],
                    check_names=False,
                )

    def test_future_score_changes_do_not_change_past_predictions(self):
        altered = self.matches.copy()
        altered.loc[altered["match_id"] == "m012", "home_goals"] = 8
        altered.loc[altered["match_id"] == "m012", "away_goals"] = 0
        original = run_walk_forward(self.matches, "elo", "2024-01-07")
        changed = run_walk_forward(altered, "elo", "2024-01-07")
        pd.testing.assert_frame_equal(
            original.predictions[["match_id", "home_win_prob", "draw_prob", "away_win_prob"]],
            changed.predictions[["match_id", "home_win_prob", "draw_prob", "away_win_prob"]],
        )

    def test_input_row_order_does_not_change_predictions(self):
        shuffled = self.matches.sample(frac=1, random_state=7).reset_index(drop=True)
        original = run_walk_forward(self.matches, "poisson", "2024-01-07")
        changed = run_walk_forward(shuffled, "poisson", "2024-01-07")
        pd.testing.assert_frame_equal(original.predictions, changed.predictions)

    def test_empty_test_set_is_explicit(self):
        with self.assertRaisesRegex(ValueError, "test set is empty"):
            run_walk_forward(self.matches, "frequency", "2030-01-01")

    def test_brier_definition_is_three_class_squared_error_mean(self):
        predictions = pd.DataFrame(
            {
                "season": ["S1", "S1"],
                "actual_result": ["H", "D"],
                "home_win_prob": [0.5, 0.2],
                "draw_prob": [0.3, 0.5],
                "away_win_prob": [0.2, 0.3],
            }
        )
        metrics = evaluate_predictions(predictions)["overall"]
        expected = ((0.5 - 1) ** 2 + 0.3**2 + 0.2**2 + 0.2**2 + (0.5 - 1) ** 2 + 0.3**2) / 2
        self.assertAlmostEqual(metrics["brier_score"], expected)
