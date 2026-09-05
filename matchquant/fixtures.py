"""Independent schema and validation for pending, score-free fixtures."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .schema import DataValidationError, has_explicit_time

FIXTURE_REQUIRED_COLUMNS = (
    "fixture_id",
    "date",
    "season",
    "home_team",
    "away_team",
)
FIXTURE_COLUMNS = FIXTURE_REQUIRED_COLUMNS
FIXTURE_SCHEMA_VERSION = "m4-pending-fixture-v1"
KNOWN_SCORE_COLUMNS = ("home_goals", "away_goals")


class FixtureValidationError(DataValidationError):
    """Raised when a pending fixture file cannot be used safely."""


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
        raise FixtureValidationError(
            f"column '{column}' contains missing values at input rows {_sample_indices(missing)}"
        )
    frame[column] = values
    return values


def _filled_values(values: pd.Series) -> pd.Series:
    cleaned = values.astype("string").str.strip()
    return ~(cleaned.isna() | cleaned.eq(""))


def validate_fixtures(fixtures: pd.DataFrame) -> pd.DataFrame:
    """Validate and normalize a score-free pending-fixture DataFrame.

    The returned frame retains the five required fixture columns and adds the
    internal ``_timestamp`` and ``_date_only`` fields used by the M1 batch
    chronology.  Goal columns are tolerated only when every value is empty so
    a template exported from another system cannot carry known scores.
    """

    if not isinstance(fixtures, pd.DataFrame):
        raise TypeError("fixtures must be a pandas DataFrame")
    if fixtures.empty:
        raise FixtureValidationError("fixture dataset is empty")

    missing_columns = [
        column for column in FIXTURE_REQUIRED_COLUMNS if column not in fixtures.columns
    ]
    if missing_columns:
        raise FixtureValidationError(
            f"missing required fixture columns: {', '.join(missing_columns)}"
        )

    frame = fixtures.copy().reset_index(drop=True)
    if "match_id" in frame.columns:
        raise FixtureValidationError(
            "fixture CSV must use fixture_id and must not contain canonical match_id"
        )
    for column in ("fixture_id", "season", "home_team", "away_team"):
        _normalise_text(frame, column)

    raw_dates = frame["date"].copy()
    date_missing = raw_dates.isna() | raw_dates.astype("string").str.strip().eq("")
    if date_missing.any():
        raise FixtureValidationError(
            "column 'date' contains missing values at input rows "
            f"{_sample_indices(date_missing)}"
        )
    parsed_dates = _parse_dates(raw_dates)
    invalid_dates = parsed_dates.isna()
    if invalid_dates.any():
        raise FixtureValidationError(
            "column 'date' contains invalid dates at input rows "
            f"{_sample_indices(invalid_dates)}"
        )
    frame["date"] = raw_dates.astype("string").str.strip()
    frame["_timestamp"] = parsed_dates
    frame["_date_only"] = pd.Series(
        [not has_explicit_time(value) for value in raw_dates], dtype="bool"
    )

    for column in KNOWN_SCORE_COLUMNS:
        if column in frame.columns and _filled_values(frame[column]).any():
            filled = _filled_values(frame[column])
            raise FixtureValidationError(
                f"fixture CSV must not contain known scores in column '{column}'; "
                f"input rows {_sample_indices(filled)}"
            )

    duplicate_ids = frame["fixture_id"].duplicated(keep=False)
    if duplicate_ids.any():
        ids = frame.loc[duplicate_ids, "fixture_id"].drop_duplicates().tolist()[:5]
        raise FixtureValidationError(f"duplicate fixture_id values: {ids}")

    same_team = frame["home_team"].str.casefold().eq(frame["away_team"].str.casefold())
    if same_team.any():
        raise FixtureValidationError(
            "home_team and away_team cannot be the same; input rows "
            f"{_sample_indices(same_team)}"
        )

    duplicate_fixture = frame.assign(
        _home_key=frame["home_team"].str.casefold(),
        _away_key=frame["away_team"].str.casefold(),
    ).duplicated(
        subset=["season", "_timestamp", "_home_key", "_away_key"], keep=False
    )
    if duplicate_fixture.any():
        raise FixtureValidationError(
            "duplicate fixtures found for the same season, date, home_team, and away_team; "
            f"input rows {_sample_indices(duplicate_fixture)}"
        )

    return frame


def load_fixtures(path: str | Path) -> pd.DataFrame:
    """Read a UTF-8 pending-fixture CSV and validate its independent schema."""

    fixture_path = Path(path)
    if not fixture_path.is_file():
        raise FileNotFoundError(f"fixture CSV does not exist: {fixture_path}")
    try:
        frame = pd.read_csv(fixture_path, dtype="string", keep_default_na=False)
    except (OSError, ValueError, pd.errors.ParserError) as exc:
        raise FixtureValidationError(
            f"could not read fixture CSV '{fixture_path}': {exc}"
        ) from exc
    return validate_fixtures(frame)


__all__ = [
    "FIXTURE_COLUMNS",
    "FIXTURE_REQUIRED_COLUMNS",
    "FIXTURE_SCHEMA_VERSION",
    "FixtureValidationError",
    "load_fixtures",
    "validate_fixtures",
]
