"""Metrics for online 1X2 predictions."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .models import RESULTS

PROBABILITY_COLUMNS = ("home_win_prob", "draw_prob", "away_win_prob")


def _score_arrays(frame: pd.DataFrame) -> dict[str, float | int]:
    probabilities = frame.loc[:, PROBABILITY_COLUMNS].to_numpy(dtype=float)
    if not np.all(np.isfinite(probabilities)):
        raise ValueError("prediction probabilities must be finite")
    if np.any(probabilities < 0) or np.any(probabilities > 1):
        raise ValueError("prediction probabilities must be in [0, 1]")
    if not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-8):
        raise ValueError("prediction probabilities must sum to 1 per match")

    result_index = frame["actual_result"].map({result: index for index, result in enumerate(RESULTS)})
    if result_index.isna().any():
        raise ValueError("actual_result must contain only H, D, or A")
    truth = result_index.to_numpy(dtype=int)
    one_hot = np.zeros_like(probabilities)
    one_hot[np.arange(len(frame)), truth] = 1.0
    true_probabilities = np.maximum(probabilities[np.arange(len(frame)), truth], 1e-15)
    predicted = np.argmax(probabilities, axis=1)
    return {
        "n_matches": int(len(frame)),
        "log_loss": float(-np.log(true_probabilities).mean()),
        "brier_score": float(np.square(probabilities - one_hot).sum(axis=1).mean()),
        "accuracy": float(np.mean(predicted == truth)),
    }


def evaluate_predictions(predictions: pd.DataFrame) -> dict[str, object]:
    """Return overall and input-season metrics for a prediction DataFrame."""

    required = {"season", "actual_result", *PROBABILITY_COLUMNS}
    missing = sorted(required.difference(predictions.columns))
    if missing:
        raise ValueError(f"missing prediction columns: {', '.join(missing)}")
    if predictions.empty:
        raise ValueError("cannot evaluate an empty test set")

    by_season: dict[str, dict[str, float | int]] = {}
    for season in sorted(predictions["season"].astype(str).unique()):
        season_frame = predictions.loc[predictions["season"].astype(str).eq(season)]
        by_season[season] = _score_arrays(season_frame)
    return {"overall": _score_arrays(predictions), "by_season": by_season}
