"""Canonical match schema and strict CSV/DataFrame validation."""

from __future__ import annotations

import re
from datetime import date, datetime

import numpy as np
import pandas as pd

REQUIRED_COLUMNS = (
    "match_id",
    "date",
    "season",
    "home_team",
    "away_team",
    "home_goals",
    "away_goals",
)


class DataValidationError(ValueError):
    """Raised when input data cannot be used safely for a backtest."""


def has_explicit_time(value: object) -> bool:
    """Return whether a date value contains a clock, not only a calendar date."""

    if isinstance(value, (datetime, pd.Timestamp)):
        return True
    if isinstance(value, date):
        return False
    text = str(value).strip()
    return bool(re.search(r"[Tt ]\d{1,2}:\d{2}(?::\d{2}(?:\.\d+)?)?", text))


def _sample_indices(mask: pd.Series, limit: int = 5) -> str:
    values = [str(index + 1) for index in mask.index[mask][:limit]]
    suffix = ", ..." if int(mask.sum()) > limit else ""
    return ", ".join(values) + suffix


def _parse_dates(values: pd.Series) -> pd.Series:
    try:
        return pd.to_datetime(values, errors="coerce", format="mixed")
    except (TypeError, ValueError):
        return pd.to_datetime(values, errors="coerce")


def _normalise_text(frame: pd.DataFrame, column: str) -> pd.Series:
    values = frame[column].astype("string").str.strip()
    missing = values.isna() | values.eq("")
    if missing.any():
        raise DataValidationError(
            f"column '{column}' contains missing values at input rows {_sample_indices(missing)}"
        )
    frame[column] = values
    return values


def _normalise_goals(frame: pd.DataFrame, column: str) -> None:
    values = pd.to_numeric(frame[column], errors="coerce")
    numeric = values.to_numpy(dtype=float, na_value=np.nan)
    invalid = values.isna() | ~np.isfinite(numeric) | (values < 0) | (values % 1 != 0)
    if invalid.any():
        raise DataValidationError(
            f"column '{column}' must contain non-negative integer scores; "
            f"invalid input rows {_sample_indices(invalid)}"
        )
    frame[column] = values.astype("int64")


def validate_matches(matches: pd.DataFrame) -> pd.DataFrame:
    """Validate and return a normalized copy of canonical match data.

    Required columns are ``match_id,date,season,home_team,away_team,home_goals,away_goals``.
    Two internal columns are added: ``_timestamp`` and ``_date_only``.
    """

    if not isinstance(matches, pd.DataFrame):
        raise TypeError("matches must be a pandas DataFrame")
    if matches.empty:
        raise DataValidationError("match dataset is empty")

    missing_columns = [column for column in REQUIRED_COLUMNS if column not in matches.columns]
    if missing_columns:
        raise DataValidationError(f"missing required columns: {', '.join(missing_columns)}")

    frame = matches.copy().reset_index(drop=True)
    for column in ("match_id", "season", "home_team", "away_team"):
        _normalise_text(frame, column)

    raw_dates = frame["date"].copy()
    date_missing = raw_dates.isna() | raw_dates.astype("string").str.strip().eq("")
    if date_missing.any():
        raise DataValidationError(
            f"column 'date' contains missing values at input rows {_sample_indices(date_missing)}"
        )
    parsed_dates = _parse_dates(raw_dates)
    invalid_dates = parsed_dates.isna()
    if invalid_dates.any():
        raise DataValidationError(
            f"column 'date' contains invalid dates at input rows {_sample_indices(invalid_dates)}"
        )
    frame["date"] = raw_dates.astype("string").str.strip()
    frame["_timestamp"] = parsed_dates
    frame["_date_only"] = pd.Series(
        [not has_explicit_time(value) for value in raw_dates], dtype="bool"
    )

    _normalise_goals(frame, "home_goals")
    _normalise_goals(frame, "away_goals")

    duplicate_ids = frame["match_id"].duplicated(keep=False)
    if duplicate_ids.any():
        ids = frame.loc[duplicate_ids, "match_id"].drop_duplicates().tolist()[:5]
        raise DataValidationError(f"duplicate match_id values: {ids}")

    same_team = frame["home_team"].str.casefold().eq(frame["away_team"].str.casefold())
    if same_team.any():
        raise DataValidationError(
            f"home_team and away_team cannot be the same; input rows {_sample_indices(same_team)}"
        )

    duplicate_fixture = frame.duplicated(
        subset=["season", "_timestamp", "home_team", "away_team"], keep=False
    )
    if duplicate_fixture.any():
        raise DataValidationError(
            "duplicate fixtures found for the same season, date, home_team, and away_team; "
            f"input rows {_sample_indices(duplicate_fixture)}"
        )

    return frame
