"""Three small online 1X2 models used by the M1 engine."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

RESULTS = ("H", "D", "A")
MODEL_NAMES = ("frequency", "elo", "poisson")


def result_from_score(home_goals: int, away_goals: int) -> str:
    if home_goals > away_goals:
        return "H"
    if home_goals < away_goals:
        return "A"
    return "D"


def _result_index(result: str) -> int:
    if result not in RESULTS:
        raise ValueError(f"result must be one of {RESULTS}, got {result!r}")
    return RESULTS.index(result)


def _probability_dict(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=float)
    if values.shape != (3,) or not np.all(np.isfinite(values)):
        raise ValueError("model probabilities must be three finite values")
    values = np.clip(values, 0.0, 1.0)
    total = float(values.sum())
    if total <= 0:
        values = np.full(3, 1.0 / 3.0)
    else:
        values = values / total
    return {
        "home_win_prob": float(values[0]),
        "draw_prob": float(values[1]),
        "away_win_prob": float(values[2]),
    }


class FrequencyModel:
    """Smoothed historical 1X2 frequency baseline."""

    name = "frequency"

    def __init__(self, alpha: float = 1.0) -> None:
        if alpha < 0 or not math.isfinite(alpha):
            raise ValueError("frequency alpha must be a finite non-negative number")
        self.alpha = float(alpha)
        self.reset()

    def reset(self) -> None:
        self.counts = np.full(3, self.alpha, dtype=float)

    def predict(self, home_team: str, away_team: str) -> dict[str, float]:
        del home_team, away_team
        if self.counts.sum() == 0:
            return _probability_dict(np.full(3, 1.0 / 3.0))
        return _probability_dict(self.counts / self.counts.sum())

    def update(self, home_team: str, away_team: str, home_goals: int, away_goals: int) -> None:
        del home_team, away_team
        self.counts[_result_index(result_from_score(home_goals, away_goals))] += 1.0


class EloModel:
    """Three-way Elo with an explicit draw probability and draw-aware updates."""

    name = "elo"

    def __init__(
        self,
        initial_rating: float = 1500.0,
        k_factor: float = 20.0,
        home_advantage: float = 50.0,
        rating_scale: float = 400.0,
        draw_base: float = 0.27,
        draw_scale: float = 400.0,
    ) -> None:
        values = (initial_rating, k_factor, home_advantage, rating_scale, draw_base, draw_scale)
        if not all(math.isfinite(float(value)) for value in values):
            raise ValueError("Elo parameters must be finite")
        if k_factor < 0 or rating_scale <= 0 or draw_scale <= 0:
            raise ValueError("Elo requires k_factor >= 0 and positive scales")
        if not 0 <= draw_base < 1:
            raise ValueError("Elo draw_base must be in [0, 1)")
        self.initial_rating = float(initial_rating)
        self.k_factor = float(k_factor)
        self.home_advantage = float(home_advantage)
        self.rating_scale = float(rating_scale)
        self.draw_base = float(draw_base)
        self.draw_scale = float(draw_scale)
        self.reset()

    def reset(self) -> None:
        self.ratings: dict[str, float] = {}

    def _rating(self, team: str) -> float:
        return self.ratings.get(team, self.initial_rating)

    @staticmethod
    def _sigmoid(value: float) -> float:
        if value >= 0:
            return 1.0 / (1.0 + math.exp(-value))
        exp_value = math.exp(value)
        return exp_value / (1.0 + exp_value)

    def _probabilities(self, home_team: str, away_team: str) -> np.ndarray:
        rating_difference = self._rating(home_team) + self.home_advantage - self._rating(away_team)
        draw_probability = self.draw_base * math.exp(-abs(rating_difference) / self.draw_scale)
        decisive_probability = 1.0 - draw_probability
        decisive_home = self._sigmoid(math.log(10.0) * rating_difference / self.rating_scale)
        return np.array(
            [
                decisive_probability * decisive_home,
                draw_probability,
                decisive_probability * (1.0 - decisive_home),
            ],
            dtype=float,
        )

    def predict(self, home_team: str, away_team: str) -> dict[str, float]:
        return _probability_dict(self._probabilities(home_team, away_team))

    def update(self, home_team: str, away_team: str, home_goals: int, away_goals: int) -> None:
        probabilities = self._probabilities(home_team, away_team)
        expected_home_score = probabilities[0] + 0.5 * probabilities[1]
        actual_home_score = 1.0 if home_goals > away_goals else 0.0 if home_goals < away_goals else 0.5
        delta = self.k_factor * (actual_home_score - expected_home_score)
        self.ratings[home_team] = self._rating(home_team) + delta
        self.ratings[away_team] = self._rating(away_team) - delta


@dataclass
class _TeamStats:
    goals_for: float = 0.0
    goals_against: float = 0.0
    matches: int = 0


class PoissonModel:
    """Basic independent Poisson model with online team-rate shrinkage."""

    name = "poisson"

    def __init__(
        self,
        prior_home_goals: float = 1.50,
        prior_away_goals: float = 1.20,
        prior_weight: float = 3.0,
        max_goals: int = 10,
        tail_tolerance: float = 1e-12,
    ) -> None:
        values = (prior_home_goals, prior_away_goals, prior_weight, tail_tolerance)
        if not all(math.isfinite(float(value)) for value in values):
            raise ValueError("Poisson priors must be finite")
        if prior_home_goals <= 0 or prior_away_goals <= 0 or prior_weight <= 0:
            raise ValueError("Poisson priors and prior_weight must be positive")
        if isinstance(max_goals, bool) or int(max_goals) != max_goals or max_goals < 3:
            raise ValueError("Poisson max_goals must be an integer of at least 3")
        if max_goals > 10000:
            raise ValueError("Poisson max_goals must not exceed 10000")
        if not 0 < tail_tolerance < 1:
            raise ValueError("Poisson tail_tolerance must be in (0, 1)")
        self.prior_home_goals = float(prior_home_goals)
        self.prior_away_goals = float(prior_away_goals)
        self.prior_weight = float(prior_weight)
        self.max_goals = int(max_goals)
        self.tail_tolerance = float(tail_tolerance)
        self.reset()

    def reset(self) -> None:
        self.total_home_goals = 0.0
        self.total_away_goals = 0.0
        self.matches = 0
        self.teams: dict[str, _TeamStats] = {}

    def _global_rates(self) -> tuple[float, float]:
        denominator = self.prior_weight + self.matches
        home = (self.prior_weight * self.prior_home_goals + self.total_home_goals) / denominator
        away = (self.prior_weight * self.prior_away_goals + self.total_away_goals) / denominator
        return home, away

    def _team_rate(self, team: str, field: str, prior: float) -> float:
        stats = self.teams.get(team, _TeamStats())
        return (self.prior_weight * prior + getattr(stats, field)) / (
            self.prior_weight + stats.matches
        )

    def expected_goals(self, home_team: str, away_team: str) -> tuple[float, float]:
        global_home, global_away = self._global_rates()
        home_for = self._team_rate(home_team, "goals_for", global_home)
        home_against = self._team_rate(home_team, "goals_against", global_away)
        away_for = self._team_rate(away_team, "goals_for", global_away)
        away_against = self._team_rate(away_team, "goals_against", global_home)
        home_lambda = 0.5 * global_home + 0.25 * home_for + 0.25 * away_against
        away_lambda = 0.5 * global_away + 0.25 * away_for + 0.25 * home_against
        return float(np.clip(home_lambda, 0.05, 10.0)), float(np.clip(away_lambda, 0.05, 10.0))

    @staticmethod
    def _poisson_pmf(rate: float, max_goals: int) -> np.ndarray:
        probabilities = np.empty(max_goals + 1, dtype=float)
        probabilities[0] = math.exp(-rate)
        if max_goals:
            goals = np.arange(1, max_goals + 1, dtype=float)
            probabilities[1:] = probabilities[0] * np.cumprod(rate / goals)
        return probabilities

    def _required_goals(self, rate: float) -> int:
        """Find a grid limit whose one-sided omitted Poisson tail is negligible."""

        grid_limit = self.max_goals
        while True:
            omitted_tail = max(0.0, 1.0 - float(self._poisson_pmf(rate, grid_limit).sum()))
            if omitted_tail <= self.tail_tolerance:
                return grid_limit
            grid_limit = max(grid_limit + 1, int(math.ceil(grid_limit * 1.5)))

    def predict(self, home_team: str, away_team: str) -> dict[str, float]:
        home_lambda, away_lambda = self.expected_goals(home_team, away_team)
        grid_limit = max(self._required_goals(home_lambda), self._required_goals(away_lambda))
        goals = np.arange(grid_limit + 1, dtype=float)
        home_pmf = self._poisson_pmf(home_lambda, grid_limit)
        away_pmf = self._poisson_pmf(away_lambda, grid_limit)
        joint = home_pmf[:, None] * away_pmf[None, :]
        goal_difference = goals[:, None] - goals[None, :]
        probabilities = np.array(
            [
                joint[goal_difference > 0].sum(),
                joint[goal_difference == 0].sum(),
                joint[goal_difference < 0].sum(),
            ],
            dtype=float,
        )
        prediction = _probability_dict(probabilities)
        prediction["expected_home_goals"] = home_lambda
        prediction["expected_away_goals"] = away_lambda
        return prediction

    def update(self, home_team: str, away_team: str, home_goals: int, away_goals: int) -> None:
        self.total_home_goals += float(home_goals)
        self.total_away_goals += float(away_goals)
        self.matches += 1
        home_stats = self.teams.setdefault(home_team, _TeamStats())
        away_stats = self.teams.setdefault(away_team, _TeamStats())
        home_stats.goals_for += float(home_goals)
        home_stats.goals_against += float(away_goals)
        home_stats.matches += 1
        away_stats.goals_for += float(away_goals)
        away_stats.goals_against += float(home_goals)
        away_stats.matches += 1


def create_model(name: str, config: dict[str, Any] | None = None) -> FrequencyModel | EloModel | PoissonModel:
    """Create one named online model from a JSON-compatible configuration."""

    config = {} if config is None else dict(config)
    if name == "frequency":
        return FrequencyModel(**config)
    if name == "elo":
        return EloModel(**config)
    if name == "poisson":
        return PoissonModel(**config)
    raise ValueError(f"unknown model '{name}'; choose from {MODEL_NAMES}")
