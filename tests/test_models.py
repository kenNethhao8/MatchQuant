import unittest

import numpy as np

from matchquant.models import EloModel, FrequencyModel, PoissonModel


class ModelTests(unittest.TestCase):
    def assert_legal(self, prediction):
        probabilities = np.array(
            [prediction["home_win_prob"], prediction["draw_prob"], prediction["away_win_prob"]]
        )
        self.assertTrue(np.all(probabilities >= 0))
        self.assertTrue(np.all(probabilities <= 1))
        self.assertAlmostEqual(float(probabilities.sum()), 1.0, places=12)

    def test_frequency_starts_uniform_and_updates(self):
        model = FrequencyModel()
        self.assertEqual(model.predict("A", "B")["home_win_prob"], 1 / 3)
        model.update("A", "B", 2, 0)
        prediction = model.predict("C", "D")
        self.assert_legal(prediction)
        self.assertGreater(prediction["home_win_prob"], prediction["draw_prob"])

    def test_elo_has_explicit_draw_probability_and_new_team_defaults(self):
        model = EloModel(home_advantage=0)
        prediction = model.predict("new-a", "new-b")
        self.assert_legal(prediction)
        self.assertGreater(prediction["draw_prob"], 0)
        self.assertAlmostEqual(prediction["home_win_prob"], prediction["away_win_prob"], places=12)
        model.update("new-a", "new-b", 1, 1)
        self.assertIn("new-a", model.ratings)

    def test_poisson_exposes_expected_goals(self):
        model = PoissonModel(max_goals=12)
        prediction = model.predict("new-a", "new-b")
        self.assert_legal(prediction)
        self.assertGreater(prediction["expected_home_goals"], 0)
        self.assertGreater(prediction["expected_away_goals"], 0)
        model.update("new-a", "new-b", 4, 0)
        later = model.predict("new-a", "new-b")
        self.assertNotEqual(later["expected_home_goals"], prediction["expected_home_goals"])

    def test_poisson_adapts_high_lambda_grid_to_large_grid_reference(self):
        parameters = {"prior_home_goals": 10.0, "prior_away_goals": 10.0, "prior_weight": 1.0}
        adaptive = PoissonModel(max_goals=10, **parameters)
        reference = PoissonModel(max_goals=80, **parameters)
        adaptive_prediction = adaptive.predict("A", "B")
        reference_prediction = reference.predict("A", "B")
        self.assert_legal(adaptive_prediction)
        for key in ("home_win_prob", "draw_prob", "away_win_prob"):
            self.assertAlmostEqual(adaptive_prediction[key], reference_prediction[key], places=10)
