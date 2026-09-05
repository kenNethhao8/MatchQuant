"""Finite reliability and calibration diagnostics for 1X2 probabilities."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .metrics import PROBABILITY_COLUMNS
from .models import RESULTS

OUTCOME_NAMES = ("home_win", "draw", "away_win")


def _validated_arrays(predictions: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    required = {"actual_result", *PROBABILITY_COLUMNS}
    missing = sorted(required.difference(predictions.columns))
    if missing:
        raise ValueError(f"missing prediction columns: {', '.join(missing)}")
    if predictions.empty:
        raise ValueError("cannot diagnose an empty prediction set")
    probabilities = predictions.loc[:, PROBABILITY_COLUMNS].to_numpy(dtype=float)
    if not np.all(np.isfinite(probabilities)):
        raise ValueError("prediction probabilities must be finite")
    if np.any(probabilities < 0) or np.any(probabilities > 1):
        raise ValueError("prediction probabilities must be in [0, 1]")
    if not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-8):
        raise ValueError("prediction probabilities must sum to 1 per match")
    result_index = predictions["actual_result"].map(
        {result: index for index, result in enumerate(RESULTS)}
    )
    if result_index.isna().any():
        raise ValueError("actual_result must contain only H, D, or A")
    return probabilities, result_index.to_numpy(dtype=int)


def reliability_table(predictions: pd.DataFrame, n_bins: int = 10) -> pd.DataFrame:
    """Return one finite row for every outcome and equal-width probability bin.

    Empty bins are retained with count and summary values of zero so the CSV has
    a stable shape and contains no NaN values. ECE excludes those empty bins.
    """

    if isinstance(n_bins, bool) or int(n_bins) != n_bins or n_bins < 2 or n_bins > 100:
        raise ValueError("n_bins must be an integer from 2 through 100")
    n_bins = int(n_bins)
    probabilities, truth = _validated_arrays(predictions)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    rows: list[dict[str, object]] = []
    for outcome_index, outcome in enumerate(OUTCOME_NAMES):
        values = probabilities[:, outcome_index]
        bin_indices = np.minimum((values * n_bins).astype(int), n_bins - 1)
        actual = truth == outcome_index
        for bin_index in range(n_bins):
            selected = bin_indices == bin_index
            count = int(selected.sum())
            rows.append(
                {
                    "outcome": outcome,
                    "bin": bin_index,
                    "bin_lower": float(edges[bin_index]),
                    "bin_upper": float(edges[bin_index + 1]),
                    "sample_count": count,
                    "mean_predicted_probability": float(values[selected].mean()) if count else 0.0,
                    "actual_frequency": float(actual[selected].mean()) if count else 0.0,
                }
            )
    table = pd.DataFrame(rows)
    numeric = table.select_dtypes(include=[np.number]).to_numpy(dtype=float)
    if not np.all(np.isfinite(numeric)):
        raise ValueError("reliability table contains non-finite values")
    return table


def calibration_diagnostics(
    predictions: pd.DataFrame, n_bins: int = 10
) -> tuple[dict[str, object], pd.DataFrame]:
    """Return multiclass ECE and its per-outcome reliability table.

    Multiclass ECE is the equally weighted mean of the three outcome ECEs;
    each outcome ECE weights a bin by its share of all matches.
    """

    table = reliability_table(predictions, n_bins=n_bins)
    n_matches = int(len(predictions))
    per_outcome: dict[str, float] = {}
    for outcome in OUTCOME_NAMES:
        subset = table.loc[table["outcome"].eq(outcome)]
        gaps = np.abs(
            subset["mean_predicted_probability"].to_numpy(dtype=float)
            - subset["actual_frequency"].to_numpy(dtype=float)
        )
        counts = subset["sample_count"].to_numpy(dtype=float)
        per_outcome[outcome] = float(np.sum((counts / n_matches) * gaps))
    ece = float(np.mean(list(per_outcome.values())))
    if not np.isfinite(ece) or not all(np.isfinite(value) for value in per_outcome.values()):
        raise ValueError("calibration diagnostics contain non-finite values")
    return {
        "n_matches": n_matches,
        "n_bins": int(n_bins),
        "multiclass_ece": ece,
        "per_outcome_ece": per_outcome,
    }, table


def multiclass_ece(predictions: pd.DataFrame, n_bins: int = 10) -> float:
    """Return only the scalar multiclass ECE from ``calibration_diagnostics``."""

    summary, _ = calibration_diagnostics(predictions, n_bins=n_bins)
    return float(summary["multiclass_ece"])
