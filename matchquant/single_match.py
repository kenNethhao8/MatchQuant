"""Explicit, cutoff-safe single-fixture prediction workflow.

This module deliberately reuses MatchQuant's existing chronological factor
engine, Factor Lab, Elo baseline, Ordered Logit, and Dixon-Coles components.
It does not download data or persist fitted model state.
"""

from __future__ import annotations

import copy
import difflib
import hashlib
import json
import math
import re
import uuid
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from .epl import normalize_season
from .factor_engine import (
    ALL_FACTOR_COLUMNS,
    FUNDAMENTAL_FACTOR_COLUMNS,
    ChronologicalFactorEngine,
    build_historical_factor_panel,
)
from .factor_lab import run_factor_lab
from .fixtures import FIXTURE_REQUIRED_COLUMNS, validate_fixtures
from .io import file_sha256
from .quant_models import _clip_probs, fit_and_select_factor_system
from .rich_data import PITCH_PROCESS_COLUMNS, devig_power_probabilities
from .schema import DataValidationError


SINGLE_MATCH_SCHEMA_VERSION = "single-match-report-v1"
PENDING_FIXTURE_ALLOWED_COLUMNS = frozenset(
    {*FIXTURE_REQUIRED_COLUMNS, "referee", "kickoff_time"}
)
MARKET_ALLOWED_FIELDS = frozenset(
    {
        "home_team",
        "away_team",
        "date",
        "observed_at",
        "source",
        "odds_open_h",
        "odds_open_d",
        "odds_open_a",
        "mkt_ah_line",
    }
)
SCORE_COLUMNS = ("home_goals", "away_goals")
STAT_DEFAULTS: dict[str, float] = {
    "ht_home_goals": 0.0,
    "ht_away_goals": 0.0,
    "sh_home_goals": math.nan,
    "sh_away_goals": math.nan,
    "home_shots": 13.0,
    "away_shots": 11.0,
    "home_sot": 4.5,
    "away_sot": 4.0,
    "home_corners": 5.5,
    "away_corners": 4.5,
    "home_fouls": 10.5,
    "away_fouls": 11.0,
    "home_yellow": 1.5,
    "away_yellow": 1.7,
    "home_red": 0.05,
    "away_red": 0.06,
}
COUNT_STAT_COLUMNS = frozenset(PITCH_PROCESS_COLUMNS) - {
    "ht_result",
}
MARKET_NUMERIC_COLUMNS = (
    "odds_open_h",
    "odds_open_d",
    "odds_open_a",
    "odds_close_h",
    "odds_close_d",
    "odds_close_a",
    "mkt_open_prob_home",
    "mkt_open_prob_draw",
    "mkt_open_prob_away",
    "mkt_close_prob_home",
    "mkt_close_prob_draw",
    "mkt_close_prob_away",
    "mkt_steam_home",
    "mkt_steam_draw",
    "mkt_steam_away",
    "mkt_ou25_over_prob",
    "mkt_ah_line",
)
M_OPEN_MARKET_COLUMNS = (
    "mkt_open_prob_home",
    "mkt_open_prob_draw",
    "mkt_open_prob_away",
    "mkt_ah_line",
)
OUTCOME_LABELS = {"H": "主胜", "D": "平局", "A": "客胜"}


class SingleMatchError(DataValidationError):
    """Expected input or model-data error for a single-match report."""


def parse_iso_date(value: object, field_name: str) -> date:
    """Parse a date-only ISO value without adding time-zone semantics."""

    text = str(value).strip()
    if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", text):
        raise SingleMatchError(f"{field_name} must use ISO date-only YYYY-MM-DD")
    try:
        return datetime.strptime(text, "%Y-%m-%d").date()
    except ValueError as exc:
        raise SingleMatchError(f"{field_name} is not a valid calendar date: {text!r}") from exc


def _text(value: object) -> str:
    return " ".join(str(value).split())


def _team_key(value: object) -> str:
    return _text(value).casefold()


def _coerce_finite_column(
    frame: pd.DataFrame,
    column: str,
    *,
    required: bool = False,
    non_negative: bool = False,
    integer: bool = False,
) -> None:
    if column not in frame.columns:
        if required:
            raise SingleMatchError(f"historical input is missing required column '{column}'")
        frame[column] = np.nan
    raw = frame[column].astype("string").str.strip()
    missing = raw.isna() | raw.eq("")
    values = pd.to_numeric(frame[column], errors="coerce").astype("float64")
    numeric = values.to_numpy(dtype=np.float64)
    invalid = (~missing & (values.isna() | ~np.isfinite(numeric)))
    if required:
        invalid |= missing
    if non_negative:
        invalid |= (~missing & values.lt(0))
    if integer:
        invalid |= (~missing & ((values % 1).abs() > 1e-9))
    if invalid.any():
        rows = [str(i + 1) for i in frame.index[invalid][:5]]
        raise SingleMatchError(
            f"historical column '{column}' contains invalid numeric values at rows {', '.join(rows)}"
        )
    frame[column] = values


def _parse_history_dates(values: pd.Series) -> tuple[pd.Series, pd.Series, pd.Series]:
    raw = values.astype("string").str.strip()
    missing = raw.isna() | raw.eq("")
    try:
        parsed = pd.to_datetime(raw, errors="coerce", format="mixed")
    except (TypeError, ValueError):
        parsed = pd.to_datetime(raw, errors="coerce")
    invalid = missing | parsed.isna()
    if invalid.any():
        rows = [str(i + 1) for i in values.index[invalid][:5]]
        raise SingleMatchError(
            f"historical input contains unparseable dates at rows {', '.join(rows)}"
        )
    dates = parsed.map(lambda value: pd.Timestamp(value).date())
    times = parsed.map(lambda value: pd.Timestamp(value).strftime("%H:%M"))
    return parsed, dates, times


def _validate_history(raw_history: pd.DataFrame, as_of: date) -> tuple[pd.DataFrame, int, dict[str, Any]]:
    if raw_history.empty:
        raise SingleMatchError("historical input is empty")
    required_base = (
        "match_id",
        "date",
        "season",
        "home_team",
        "away_team",
        *SCORE_COLUMNS,
    )
    missing_base = [column for column in required_base if column not in raw_history.columns]
    if missing_base:
        raise SingleMatchError(
            f"historical input is missing required columns: {', '.join(missing_base)}"
        )

    # Every date must be readable so the cutoff can be applied safely. Other
    # values are validated only after future and cutoff-day rows are excluded.
    _, parsed_days, parsed_times = _parse_history_dates(raw_history["date"])
    day_mask = parsed_days.map(lambda value: value < as_of)
    excluded_count = int((~day_mask).sum())
    frame = raw_history.loc[day_mask].copy().reset_index(drop=True)
    if frame.empty:
        raise SingleMatchError(f"no completed historical rows exist before as-of date {as_of.isoformat()}")

    frame["date"] = parsed_days.loc[day_mask].map(date.isoformat).reset_index(drop=True)
    if "kickoff_time" not in frame.columns:
        frame["kickoff_time"] = ""
    kickoff = frame["kickoff_time"].astype("string").fillna("").str.strip()
    parsed_time = parsed_times.loc[day_mask].reset_index(drop=True)
    had_source_time = raw_history.loc[day_mask, "date"].astype("string").str.contains(
        r"[Tt ]\d{1,2}:\d{2}", regex=True, na=False
    ).reset_index(drop=True)
    blank_kickoff = kickoff.eq("")
    kickoff = kickoff.where(~(blank_kickoff & had_source_time), parsed_time)
    valid_kickoff = kickoff.eq("") | kickoff.str.fullmatch(r"\d{2}:\d{2}")
    if not valid_kickoff.all():
        raise SingleMatchError("historical kickoff_time values must be blank or HH:MM")
    frame["kickoff_time"] = kickoff

    for column in ("match_id", "season", "home_team", "away_team"):
        values = frame[column].astype("string").fillna("").map(_text)
        if values.eq("").any():
            raise SingleMatchError(f"historical column '{column}' contains empty values")
        frame[column] = values
    normalized_seasons: list[str] = []
    for raw_season in frame["season"]:
        try:
            normalized_seasons.append(normalize_season(str(raw_season)))
        except ValueError as exc:
            raise SingleMatchError(f"invalid historical season {raw_season!r}: {exc}") from exc
    frame["season"] = normalized_seasons
    frame["home_team"] = frame["home_team"].map(_text)
    frame["away_team"] = frame["away_team"].map(_text)
    same_team = frame["home_team"].map(_team_key).eq(frame["away_team"].map(_team_key))
    if same_team.any():
        raise SingleMatchError("historical rows cannot list the same team as home and away")

    required_stats = [column for column in PITCH_PROCESS_COLUMNS if column != "ht_result"]
    missing_stats_columns = [column for column in required_stats if column not in frame.columns]
    if missing_stats_columns:
        raise SingleMatchError(
            "historical input must use the rich match schema; missing technical-stat columns: "
            + ", ".join(missing_stats_columns)
        )
    for column in required_stats:
        _coerce_finite_column(
            frame,
            column,
            required=False,
            non_negative=True,
            integer=column in COUNT_STAT_COLUMNS,
        )
    for column in SCORE_COLUMNS:
        _coerce_finite_column(frame, column, required=True, non_negative=True, integer=True)
        frame[column] = frame[column].astype("int64")

    expected_result = np.where(
        frame["home_goals"] > frame["away_goals"],
        "H",
        np.where(frame["home_goals"] < frame["away_goals"], "A", "D"),
    )
    if "actual_result" in frame.columns:
        result = frame["actual_result"].astype("string").str.strip().str.upper()
        bad_result = result.isna() | ~result.isin(["H", "D", "A"]) | result.ne(expected_result)
        if bad_result.any():
            raise SingleMatchError("historical actual_result values must match the validated full-time scores")
        frame["actual_result"] = result
    else:
        frame["actual_result"] = expected_result

    for column in MARKET_NUMERIC_COLUMNS:
        _coerce_finite_column(frame, column)
    for column in ("odds_open_h", "odds_open_d", "odds_open_a", "odds_close_h", "odds_close_d", "odds_close_a"):
        present = frame[column].notna()
        if (frame.loc[present, column] <= 1.0).any():
            raise SingleMatchError(f"historical decimal odds in '{column}' must be greater than 1")

    open_prob_cols = ["mkt_open_prob_home", "mkt_open_prob_draw", "mkt_open_prob_away"]
    close_prob_cols = ["mkt_close_prob_home", "mkt_close_prob_draw", "mkt_close_prob_away"]
    for prob_cols, odds_cols, label in (
        (open_prob_cols, ["odds_open_h", "odds_open_d", "odds_open_a"], "opening"),
        (close_prob_cols, ["odds_close_h", "odds_close_d", "odds_close_a"], "closing"),
    ):
        present_counts = frame[prob_cols].notna().sum(axis=1)
        partial = present_counts.between(1, 2)
        if partial.any():
            raise SingleMatchError(f"historical {label} market probabilities must be all present or all unavailable")
        missing = present_counts.eq(0)
        if label == "opening" and missing.any():
            odds_present = frame[odds_cols].notna().all(axis=1)
            derive = missing & odds_present
            if derive.any():
                ph, pd_, pa = devig_power_probabilities(
                    frame.loc[derive, odds_cols[0]],
                    frame.loc[derive, odds_cols[1]],
                    frame.loc[derive, odds_cols[2]],
                )
                frame.loc[derive, prob_cols[0]] = ph
                frame.loc[derive, prob_cols[1]] = pd_
                frame.loc[derive, prob_cols[2]] = pa
        present = frame[prob_cols].notna().all(axis=1)
        if present.any():
            values = frame.loc[present, prob_cols].to_numpy(dtype=np.float64)
            if not np.isfinite(values).all() or (values < 0).any() or (values > 1).any():
                raise SingleMatchError(f"historical {label} market probabilities must be finite values in [0, 1]")
            if not np.allclose(values.sum(axis=1), 1.0, atol=1e-3, rtol=0.0):
                raise SingleMatchError(f"historical {label} market probabilities must sum to 1")

    if "referee" not in frame.columns:
        frame["referee"] = "Unknown"
    frame["referee"] = frame["referee"].astype("string").fillna("").str.strip().replace("", "Unknown")

    duplicate_ids = frame["match_id"].duplicated(keep=False)
    duplicate_fixtures = frame.assign(
        _home_key=frame["home_team"].map(_team_key),
        _away_key=frame["away_team"].map(_team_key),
    ).duplicated(subset=["season", "date", "_home_key", "_away_key"], keep=False)
    if duplicate_ids.any():
        raise SingleMatchError("eligible historical rows contain duplicate match_id values")
    if duplicate_fixtures.any():
        raise SingleMatchError("eligible historical rows contain duplicate season/date/team identities")

    stat_columns = list(required_stats)
    missing_stat_rows = frame[stat_columns].isna().any(axis=1)
    missing_stat_count = int(missing_stat_rows.sum())
    missing_stat_fields = {
        column: int(frame[column].isna().sum()) for column in stat_columns if frame[column].isna().any()
    }
    return frame, excluded_count, {
        "eligible_row_count": int(len(frame)),
        "excluded_future_or_cutoff_day_rows": excluded_count,
        "missing_post_match_stat_row_count": missing_stat_count,
        "missing_post_match_stat_field_counts": missing_stat_fields,
    }


def _read_history(path: str | Path) -> pd.DataFrame:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"historical rich input does not exist: {source}")
    try:
        return pd.read_csv(source, dtype="string", keep_default_na=False)
    except (OSError, ValueError, pd.errors.ParserError) as exc:
        raise SingleMatchError(f"could not read historical rich input '{source}': {exc}") from exc


def _canonical_team_index(team_names: Sequence[str]) -> dict[str, list[str]]:
    index: dict[str, list[str]] = {}
    for name in sorted(set(team_names)):
        index.setdefault(_team_key(name), []).append(name)
    return index


def _resolve_team_name(
    requested: object,
    team_names: Sequence[str],
    aliases: Mapping[str, str] | None = None,
) -> str:
    name = _text(requested)
    key = _team_key(name)
    alias_map = aliases or {}
    if key in alias_map:
        name = alias_map[key]
        key = _team_key(name)
    index = _canonical_team_index(team_names)
    candidates = index.get(key, [])
    if len(candidates) == 1:
        return candidates[0]
    if len(candidates) > 1:
        raise SingleMatchError(
            f"team name {str(requested)!r} is ambiguous after whitespace/case normalization: {candidates}"
        )
    suggestions = difflib.get_close_matches(key, list(index), n=3, cutoff=0.45)
    suggested_names = [candidate for suggestion in suggestions for candidate in index[suggestion]]
    hint = f" Suggestions: {', '.join(suggested_names)}." if suggested_names else ""
    raise SingleMatchError(f"unknown team {str(requested)!r} in eligible history.{hint}")


def _load_aliases(path: str | Path | None, team_names: Sequence[str]) -> tuple[dict[str, str], dict[str, Any] | None]:
    if path is None:
        return {}, None
    alias_path = Path(path)
    if not alias_path.is_file():
        raise FileNotFoundError(f"team aliases file does not exist: {alias_path}")
    try:
        payload = json.loads(alias_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SingleMatchError(f"could not read team aliases JSON '{alias_path}': {exc}") from exc
    if not isinstance(payload, dict):
        raise SingleMatchError("team aliases JSON must be an object mapping aliases to canonical team names")
    aliases: dict[str, str] = {}
    for alias, target in payload.items():
        if not isinstance(alias, str) or not _text(alias) or not isinstance(target, str) or not _text(target):
            raise SingleMatchError("each team alias and target must be a non-empty string")
        key = _team_key(alias)
        if key in aliases and _team_key(aliases[key]) != _team_key(target):
            raise SingleMatchError(f"conflicting team aliases normalize to {alias!r}")
        aliases[key] = _resolve_team_name(target, team_names)
    return aliases, {"path": str(alias_path.resolve()), "sha256": file_sha256(alias_path)}


def _new_fixture_id(season: str, fixture_date: date, home_team: str, away_team: str) -> str:
    identity = "\x1f".join((season, fixture_date.isoformat(), _team_key(home_team), _team_key(away_team)))
    digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:20]
    return f"single-{digest}"


def _load_fixture_csv(path: str | Path) -> dict[str, Any]:
    fixture_path = Path(path)
    if not fixture_path.is_file():
        raise FileNotFoundError(f"pending fixture CSV does not exist: {fixture_path}")
    try:
        raw = pd.read_csv(fixture_path, dtype="string", keep_default_na=False)
    except (OSError, ValueError, pd.errors.ParserError) as exc:
        raise SingleMatchError(f"could not read pending fixture CSV '{fixture_path}': {exc}") from exc
    if len(raw) != 1:
        raise SingleMatchError("--fixtures must contain exactly one pending fixture row")
    unsupported = [column for column in raw.columns if column not in PENDING_FIXTURE_ALLOWED_COLUMNS]
    if unsupported:
        raise SingleMatchError(
            "pending fixture contains unsupported fields: "
            + ", ".join(unsupported)
            + "; only fixture_id,date,season,home_team,away_team,referee,kickoff_time are allowed"
        )
    validated = validate_fixtures(raw)
    row = validated.iloc[0]
    return {
        "fixture_id": str(row["fixture_id"]).strip(),
        "date": str(row["date"]).strip(),
        "season": str(row["season"]).strip(),
        "home_team": str(row["home_team"]).strip(),
        "away_team": str(row["away_team"]).strip(),
        "referee": str(row.get("referee", "Unknown")).strip() or "Unknown",
        "kickoff_time": str(row.get("kickoff_time", "")).strip(),
    }


def normalize_fixture(
    *,
    fixture_csv: str | Path | None = None,
    home: str | None = None,
    away: str | None = None,
    fixture_date: str | None = None,
    season: str | None = None,
    as_of: str,
    team_names: Sequence[str],
    aliases: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Validate one explicit fixture and resolve its names against cutoff history."""

    direct_values = (home, away, fixture_date, season)
    if fixture_csv is not None and any(value is not None for value in direct_values):
        raise SingleMatchError("use either --fixtures or all four direct fields; mixed fixture input is not allowed")
    if fixture_csv is None and any(value is None for value in direct_values):
        raise SingleMatchError("direct input requires --home, --away, --date, and --season together")
    fixture = (
        _load_fixture_csv(fixture_csv)
        if fixture_csv is not None
        else {
            "fixture_id": "",
            "date": str(fixture_date),
            "season": str(season),
            "home_team": str(home),
            "away_team": str(away),
            "referee": "Unknown",
            "kickoff_time": "",
        }
    )
    fixture_day = parse_iso_date(fixture["date"], "fixture date")
    as_of_day = parse_iso_date(as_of, "--as-of")
    if fixture_day < as_of_day:
        raise SingleMatchError("fixture date must be on or after the start-of-day --as-of cutoff")
    try:
        normalized_season = normalize_season(fixture["season"])
    except ValueError as exc:
        raise SingleMatchError(f"fixture season is invalid: {exc}") from exc
    resolved_home = _resolve_team_name(fixture["home_team"], team_names, aliases)
    resolved_away = _resolve_team_name(fixture["away_team"], team_names, aliases)
    if _team_key(resolved_home) == _team_key(resolved_away):
        raise SingleMatchError("home and away teams resolve to the same historical team")
    fixture_id = fixture["fixture_id"] or _new_fixture_id(normalized_season, fixture_day, resolved_home, resolved_away)
    return {
        "fixture_id": fixture_id,
        "date": fixture_day.isoformat(),
        "season": normalized_season,
        "home_team": resolved_home,
        "away_team": resolved_away,
        "referee": fixture["referee"] or "Unknown",
        "kickoff_time": fixture["kickoff_time"],
        "requested_home_team": _text(fixture["home_team"]),
        "requested_away_team": _text(fixture["away_team"]),
    }


def _numeric_json_value(value: object, field_name: str, *, greater_than_one: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SingleMatchError(f"market field '{field_name}' must be a JSON number")
    parsed = float(value)
    if not math.isfinite(parsed) or (greater_than_one and parsed <= 1.0):
        extra = " greater than 1" if greater_than_one else " finite"
        raise SingleMatchError(f"market field '{field_name}' must be{extra}")
    return parsed


def _load_market_input(
    path: str | Path | None,
    *,
    fixture: Mapping[str, Any],
    as_of: date,
    team_names: Sequence[str],
    aliases: Mapping[str, str] | None,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    if path is None:
        return None, None
    market_path = Path(path)
    if not market_path.is_file():
        raise FileNotFoundError(f"market JSON does not exist: {market_path}")
    try:
        payload = json.loads(market_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SingleMatchError(f"could not read market JSON '{market_path}': {exc}") from exc
    if not isinstance(payload, dict):
        raise SingleMatchError("market input must be one JSON object")
    extra = sorted(set(payload) - MARKET_ALLOWED_FIELDS)
    required = MARKET_ALLOWED_FIELDS - {"mkt_ah_line"}
    missing = sorted(required - set(payload))
    if extra:
        raise SingleMatchError(f"unsupported market fields: {', '.join(extra)}")
    if missing:
        raise SingleMatchError(f"market input is missing fields: {', '.join(missing)}")
    if _resolve_team_name(payload["home_team"], team_names, aliases) != fixture["home_team"]:
        raise SingleMatchError("market home_team does not match the resolved fixture")
    if _resolve_team_name(payload["away_team"], team_names, aliases) != fixture["away_team"]:
        raise SingleMatchError("market away_team does not match the resolved fixture")
    market_fixture_date = parse_iso_date(payload["date"], "market date")
    observed_at = parse_iso_date(payload["observed_at"], "market observed_at")
    if market_fixture_date.isoformat() != fixture["date"]:
        raise SingleMatchError("market date does not match the requested fixture date")
    if observed_at >= as_of:
        raise SingleMatchError("market observed_at must be strictly before the start of --as-of")
    source = payload["source"]
    if not isinstance(source, str) or not _text(source):
        raise SingleMatchError("market source must be a non-empty description")
    odds = [
        _numeric_json_value(payload[key], key, greater_than_one=True)
        for key in ("odds_open_h", "odds_open_d", "odds_open_a")
    ]
    line = None
    if "mkt_ah_line" in payload:
        line = _numeric_json_value(payload["mkt_ah_line"], "mkt_ah_line")
    p_home, p_draw, p_away = devig_power_probabilities(*[np.array([value]) for value in odds])
    market = {
        "home_team": fixture["home_team"],
        "away_team": fixture["away_team"],
        "date": market_fixture_date.isoformat(),
        "observed_at": observed_at.isoformat(),
        "source": _text(source),
        "odds_open_h": odds[0],
        "odds_open_d": odds[1],
        "odds_open_a": odds[2],
        "mkt_ah_line": line,
        "home_handicap_sign_convention": "主队让球为负；例如 -0.25 表示主队让 0.25 球",
        "probabilities": {
            "home": float(p_home[0]),
            "draw": float(p_draw[0]),
            "away": float(p_away[0]),
        },
    }
    return market, {"path": str(market_path.resolve()), "sha256": file_sha256(market_path)}


def _partition_seasons(eligible: pd.DataFrame) -> tuple[list[str], str, list[str], str]:
    counts = eligible.groupby("season", sort=False).size().to_dict()
    ordered = sorted(counts, key=lambda value: int(value[:4]))
    if not ordered:
        raise SingleMatchError("no eligible historical seasons remain after the cutoff")
    starts = [int(value[:4]) for value in ordered]
    if any(right - left != 1 for left, right in zip(starts, starts[1:])):
        raise SingleMatchError("eligible history skips one or more seasons; chronological carry-forward is unsupported")
    for season in ordered[:-1]:
        if counts[season] != 380:
            raise SingleMatchError(
                f"historical season {season} has {counts[season]} eligible matches; only the latest season may be partial"
            )
    latest = ordered[-1]
    if counts[latest] > 380:
        raise SingleMatchError(f"latest eligible season {latest} has {counts[latest]} rows, more than 380")
    complete = [season for season in ordered if counts[season] == 380]
    for season in complete:
        season_rows = eligible.loc[eligible["season"].eq(season)]
        team_set = set(season_rows["home_team"]) | set(season_rows["away_team"])
        home_keys = season_rows["home_team"].map(_team_key)
        away_keys = season_rows["away_team"].map(_team_key)
        if len(team_set) != 20:
            raise SingleMatchError(
                f"season {season} has {len(team_set)} unique teams; a complete EPL season requires 20"
            )
        if season_rows.assign(_home=home_keys, _away=away_keys).duplicated(
            subset=["_home", "_away"], keep=False
        ).any():
            raise SingleMatchError(f"season {season} repeats a home-away fixture and is not a valid 380-match season")
        home_counts = home_keys.value_counts()
        away_counts = away_keys.value_counts()
        if not (home_counts.eq(19).all() and away_counts.eq(19).all()):
            raise SingleMatchError(
                f"season {season} does not contain 19 home and 19 away fixtures for every team"
            )
    if len(complete) < 3:
        raise SingleMatchError(
            "insufficient complete seasons: at least two training seasons and one validation season "
            "of 380 unique EPL matches each are required"
        )
    train_seasons, validation_season = complete[:-1], complete[-1]
    return train_seasons, validation_season, complete, latest


def _validate_fixture_season(fixture_season: str, current_history_season: str) -> None:
    delta = int(fixture_season[:4]) - int(current_history_season[:4])
    if delta not in (0, 1):
        raise SingleMatchError(
            f"fixture season {fixture_season} must be the current eligible history season "
            f"({current_history_season}) or its immediate successor"
        )


def build_query_features(
    warmed_engine: ChronologicalFactorEngine,
    fixture: Mapping[str, Any],
    *,
    market: Mapping[str, Any] | None = None,
) -> pd.DataFrame:
    """Compute one score-free query row on an independent copy of warmed state."""

    query_engine = copy.deepcopy(warmed_engine)
    if query_engine.current_season is None:
        raise SingleMatchError("historical factor engine has no eligible season state")
    _validate_fixture_season(str(fixture["season"]), str(query_engine.current_season))
    if fixture["season"] != query_engine.current_season:
        query_engine._on_season_change(str(fixture["season"]))
    market_probs = market["probabilities"] if market is not None else None
    row = {
        "match_id": fixture["fixture_id"],
        "season": fixture["season"],
        "date": fixture["date"],
        "kickoff_time": fixture.get("kickoff_time", ""),
        "home_team": fixture["home_team"],
        "away_team": fixture["away_team"],
        "referee": fixture.get("referee", "Unknown"),
        "mkt_open_prob_home": market_probs["home"] if market_probs else np.nan,
        "mkt_open_prob_draw": market_probs["draw"] if market_probs else np.nan,
        "mkt_open_prob_away": market_probs["away"] if market_probs else np.nan,
        "mkt_steam_home": np.nan,
        "mkt_steam_draw": np.nan,
        "mkt_steam_away": np.nan,
        "mkt_ah_line": (
            market.get("mkt_ah_line")
            if market is not None and market.get("mkt_ah_line") is not None
            else np.nan
        ),
    }
    # No goals, result label, or post-match statistics are introduced for this row.
    features = query_engine.compute_pre_match_factors(row)
    return pd.DataFrame([{**row, **features}])


def _team_recent_evidence(eligible: pd.DataFrame, team: str, cutoff: date) -> dict[str, Any]:
    home_rows = eligible.loc[eligible["home_team"].map(_team_key).eq(_team_key(team))].copy()
    away_rows = eligible.loc[eligible["away_team"].map(_team_key).eq(_team_key(team))].copy()
    home_rows["_venue"] = "主"
    away_rows["_venue"] = "客"
    recent = pd.concat([home_rows, away_rows], ignore_index=True)
    recent = recent.sort_values(["date", "kickoff_time", "match_id"], ascending=False, kind="mergesort").head(5)
    records: list[dict[str, Any]] = []
    results = {"W": 0, "D": 0, "L": 0}
    goals_for: list[int] = []
    goals_against: list[int] = []
    sot_values: list[float] = []
    corner_values: list[float] = []
    for row in recent.to_dict(orient="records"):
        is_home = row["_venue"] == "主"
        gf = int(row["home_goals"] if is_home else row["away_goals"])
        ga = int(row["away_goals"] if is_home else row["home_goals"])
        result = "W" if gf > ga else ("L" if gf < ga else "D")
        results[result] += 1
        goals_for.append(gf)
        goals_against.append(ga)
        sot = row.get("home_sot" if is_home else "away_sot")
        corners = row.get("home_corners" if is_home else "away_corners")
        sot_value = float(sot) if pd.notna(sot) and math.isfinite(float(sot)) else None
        corner_value = float(corners) if pd.notna(corners) and math.isfinite(float(corners)) else None
        if sot_value is not None:
            sot_values.append(sot_value)
        if corner_value is not None:
            corner_values.append(corner_value)
        records.append(
            {
                "match_id": str(row["match_id"]),
                "date": str(row["date"]),
                "venue": row["_venue"],
                "opponent": str(row["away_team"] if is_home else row["home_team"]),
                "goals_for": gf,
                "goals_against": ga,
                "result": result,
                "sot_for": sot_value,
                "corners_for": corner_value,
            }
        )
    return {
        "team": team,
        "cutoff_exclusive": cutoff.isoformat(),
        "sample_count": len(records),
        "records": records,
        "record_dates": [record["date"] for record in records],
        "record_count_by_outcome": results,
        "goals_for": {"total": sum(goals_for), "mean": float(np.mean(goals_for)) if goals_for else None},
        "goals_against": {
            "total": sum(goals_against),
            "mean": float(np.mean(goals_against)) if goals_against else None,
        },
        "shots_on_target_for": {
            "valid_sample_count": len(sot_values),
            "mean": float(np.mean(sot_values)) if sot_values else None,
        },
        "corners_for": {
            "valid_sample_count": len(corner_values),
            "mean": float(np.mean(corner_values)) if corner_values else None,
        },
        "missing_stat_count_in_sample": {
            "shots_on_target": len(records) - len(sot_values),
            "corners": len(records) - len(corner_values),
        },
    }


def _probability_dict(values: Sequence[float]) -> dict[str, float]:
    return {"home": float(values[0]), "draw": float(values[1]), "away": float(values[2])}


def _m_open_history_readiness(
    eligible: pd.DataFrame,
    fit_ids: set[str],
) -> tuple[bool, str | None]:
    needed = list(M_OPEN_MARKET_COLUMNS)
    selected = eligible.loc[eligible["match_id"].astype(str).isin(fit_ids)]
    absent_columns = [column for column in needed if column not in selected.columns]
    if absent_columns:
        return False, f"历史拟合样本缺少市场字段：{', '.join(absent_columns)}"
    values = selected[needed].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float64)
    if values.size == 0 or not np.isfinite(values).all():
        return False, "训练或验证样本中存在缺失的赛前初盘概率或亚盘让球值"
    return True, None


def _warning(code: str, message: str) -> dict[str, str]:
    return {"code": code, "message": message}


def _build_result(
    *,
    input_path: Path,
    eligible: pd.DataFrame,
    excluded_count: int,
    history_stats: Mapping[str, Any],
    fixture: Mapping[str, Any],
    as_of: date,
    mode: str,
    max_data_age_days: int,
    alias_source: Mapping[str, Any] | None,
    fixture_source: Mapping[str, Any] | None = None,
    market: Mapping[str, Any] | None,
    market_source: Mapping[str, Any] | None,
) -> dict[str, Any]:
    if mode not in {"research", "replay"}:
        raise SingleMatchError("mode must be 'research' or 'replay'")
    if max_data_age_days < 0:
        raise SingleMatchError("max-data-age-days must be a non-negative integer")
    train_seasons, validation_season, complete_seasons, current_history_season = _partition_seasons(eligible)
    _validate_fixture_season(str(fixture["season"]), current_history_season)

    panel, warmed_engine = build_historical_factor_panel(eligible)
    if str(warmed_engine.current_season) != current_history_season:
        raise SingleMatchError("historical engine season does not match the latest eligible history season")
    train_df = panel.loc[panel["season"].isin(set(train_seasons))].copy()
    validation_df = panel.loc[panel["season"].eq(validation_season)].copy()
    if train_df.empty or validation_df.empty:
        raise SingleMatchError("chronological training or validation partition is empty")

    lab = run_factor_lab(panel, discovery_seasons=train_seasons)
    selected_features = list(lab["selected_fundamental_factors"])
    if not selected_features or any(name not in FUNDAMENTAL_FACTOR_COLUMNS for name in selected_features):
        raise SingleMatchError("Factor Lab returned an invalid or empty Track A feature set")
    track_a, selection_info = fit_and_select_factor_system(
        train_df, validation_df, selected_features, track="fundamental"
    )

    cutoff_day = parse_iso_date(str(fixture["date"]), "fixture date")
    query = build_query_features(warmed_engine, fixture, market=market)
    p_elo = query[["elo_prob_h", "elo_prob_d", "elo_prob_a"]].to_numpy(dtype=np.float64)[0]
    p_ordered = track_a.ordered_model.predict_proba(query)[0]
    p_dc, lambda_home, lambda_away, p_over25 = track_a.dixon_coles_model.predict_proba_and_goals(query)
    p_dc = p_dc[0]
    probabilities = _clip_probs(
        np.asarray(
            [
                track_a.weight_ordered * p_ordered
                + track_a.weight_dc * p_dc
                + track_a.weight_elo * p_elo
            ],
            dtype=np.float64,
        )
    )[0]
    numeric_outputs = np.concatenate(
        [probabilities, p_elo, p_ordered, p_dc, np.asarray([*lambda_home, *lambda_away, *p_over25])]
    )
    if not np.isfinite(numeric_outputs).all():
        raise SingleMatchError("model returned a non-finite single-match output")
    if (probabilities < 0).any() or not math.isclose(float(probabilities.sum()), 1.0, abs_tol=1e-9):
        raise SingleMatchError("model returned invalid 1X2 probabilities")
    if (lambda_home < 0).any() or (lambda_away < 0).any() or not 0.0 <= float(p_over25[0]) <= 1.0:
        raise SingleMatchError("model returned invalid expected goals or over-2.5 probability")

    outcome_order = ("H", "D", "A")
    most_likely_code = outcome_order[int(np.argmax(probabilities))]
    ordered_weight = float(track_a.weight_ordered)
    if ordered_weight > 1e-12:
        ordered_explanation = track_a.ordered_model.explain_row(query.iloc[0].to_dict())
        drivers = [
            {
                "factor": str(item["factor"]),
                "raw_value": float(item["raw_value"]),
                "direction": "提高模型主胜估计" if item["marginal_prob_h_pct"] > 0 else "降低模型主胜估计",
                "local_approx_home_win_contribution_pct_points": float(item["marginal_prob_h_pct"]),
            }
            for item in ordered_explanation["top_factor_contributions"]
            if item["factor"] != "elo_diff"
        ][:3]
        attribution = {
            "available": True,
            "method": "ordered_model_local_first_order_approximation",
            "label": "有序模型分量的局部近似贡献",
            "ordered_component_weight": ordered_weight,
            "contributions": drivers,
            "limitation": "这些局部贡献不是最终集成概率差的加法分解，也不是因果效应。",
        }
    else:
        attribution = {
            "available": False,
            "method": "unavailable_zero_ordered_weight",
            "label": None,
            "ordered_component_weight": ordered_weight,
            "contributions": [],
            "limitation": "有序模型分量的集成权重为零，因此不将其局部贡献当作最终预测解释。",
        }

    fit_ids = set(train_df["match_id"].astype(str)) | set(validation_df["match_id"].astype(str))
    m_open_available = False
    m_open_reason: str | None = "未提供市场输入；本次未运行 M-Open。" if market is None else None
    m_open_probabilities: dict[str, float] | None = None
    m_open_selection: dict[str, Any] | None = None
    if market is not None:
        if market.get("mkt_ah_line") is None:
            m_open_reason = "市场文件未提供有效亚盘让球线；M-Open 需要初盘 1X2 与亚盘输入。"
        elif market is not None:
            ready, m_open_reason = _m_open_history_readiness(eligible, fit_ids)
            if ready:
                m_open_features = selected_features + ["mkt_open_prob_edge", "mkt_ah_supremacy"]
                try:
                    track_m_open, m_open_selection = fit_and_select_factor_system(
                        train_df, validation_df, m_open_features, track="market_open"
                    )
                    m_open_values = np.asarray(track_m_open.predict_proba(query), dtype=np.float64)
                    if (
                        m_open_values.shape != (1, 3)
                        or not np.isfinite(m_open_values).all()
                        or (m_open_values < 0).any()
                        or not math.isclose(float(m_open_values[0].sum()), 1.0, abs_tol=1e-9)
                    ):
                        raise ValueError("M-Open returned invalid 1X2 probabilities")
                    m_open_probabilities = _probability_dict(m_open_values[0])
                    m_open_available = True
                    m_open_reason = None
                except (ValueError, FloatingPointError, np.linalg.LinAlgError) as exc:
                    m_open_reason = f"M-Open 无法拟合：{exc}"

    latest_used_date = max(eligible["date"])
    latest_date = date.fromisoformat(latest_used_date)
    data_age_days = (as_of - latest_date).days
    forecast_horizon_days = (cutoff_day - as_of).days
    state_rows = eligible.loc[eligible["season"].eq(current_history_season)]
    fitting_rows = panel.loc[panel["season"].isin(set(train_seasons + [validation_season]))]
    fitting_cutoff = max(fitting_rows["date"].astype(str))
    recent = {
        "home": _team_recent_evidence(eligible, str(fixture["home_team"]), as_of),
        "away": _team_recent_evidence(eligible, str(fixture["away_team"]), as_of),
    }
    warnings: list[dict[str, str]] = []
    if data_age_days > max_data_age_days:
        warnings.append(
            _warning(
                "stale_history",
                f"最新可用比赛距预测截止日 {data_age_days} 天，超过配置阈值 {max_data_age_days} 天；请复核数据覆盖。",
            )
        )
    missing_stat_rows = int(history_stats.get("missing_post_match_stat_row_count", 0))
    if missing_stat_rows:
        warnings.append(
            _warning(
                "missing_historical_statistics",
                f"{missing_stat_rows} 场 eligible 历史比赛有缺失赛后统计；因子状态使用既有默认值，证据表保留实际有效样本数。",
            )
        )
    if market is not None and not m_open_available:
        warnings.append(_warning("market_track_unavailable", str(m_open_reason)))

    predictions = {
        "elo": _probability_dict(p_elo),
        "ordered_logit": _probability_dict(p_ordered),
        "dixon_coles": _probability_dict(p_dc),
        "track_a": _probability_dict(probabilities),
        "most_likely_outcome_code": most_likely_code,
        "most_likely_outcome_label": OUTCOME_LABELS[most_likely_code],
        "expected_goals": {"home": float(lambda_home[0]), "away": float(lambda_away[0])},
        "probability_total_goals_over_2_5": float(p_over25[0]),
        "track_a_minus_elo_percentage_points": {
            key: float((probabilities[index] - p_elo[index]) * 100.0)
            for index, key in enumerate(("home", "draw", "away"))
        },
        "component_weights": {
            "ordered_logit": float(track_a.weight_ordered),
            "dixon_coles": float(track_a.weight_dc),
            "elo_baseline": float(track_a.weight_elo),
        },
    }
    market_result: dict[str, Any] = {
        "provided": market is not None,
        "input": dict(market) if market is not None else None,
        "track_a_minus_market_percentage_points": None,
        "m_open_available": m_open_available,
        "m_open_unavailable_reason": m_open_reason,
        "m_open_probabilities": m_open_probabilities,
        "m_open_selection": m_open_selection,
    }
    if market is not None:
        market_result["track_a_minus_market_percentage_points"] = {
            key: float((probabilities[index] - market["probabilities"][key]) * 100.0)
            for index, key in enumerate(("home", "draw", "away"))
        }

    raw_features = {
        name: (float(query.iloc[0][name]) if pd.notna(query.iloc[0][name]) and math.isfinite(float(query.iloc[0][name])) else None)
        for name in ALL_FACTOR_COLUMNS
    }
    raw_features.update({
        "elo_prob_h": float(p_elo[0]),
        "elo_prob_d": float(p_elo[1]),
        "elo_prob_a": float(p_elo[2]),
    })
    return {
        "schema_version": SINGLE_MATCH_SCHEMA_VERSION,
        "status": "needs_review" if warnings else "complete",
        "warnings": warnings,
        "fixture": {
            "fixture_id": str(fixture["fixture_id"]),
            "date": str(fixture["date"]),
            "season": str(fixture["season"]),
            "home_team": str(fixture["home_team"]),
            "away_team": str(fixture["away_team"]),
            "requested_home_team": str(fixture.get("requested_home_team", fixture["home_team"])),
            "requested_away_team": str(fixture.get("requested_away_team", fixture["away_team"])),
            "referee": str(fixture.get("referee", "Unknown")),
            "kickoff_time": str(fixture.get("kickoff_time", "")),
            "identity_source": "user_supplied_not_checked_online",
        },
        "temporal": {
            "mode": mode,
            "as_of": as_of.isoformat(),
            "as_of_semantics": "date_only_start_of_day; every same-day historical row is excluded",
            "used_data_cutoff": latest_used_date,
            "data_age_days_at_as_of": data_age_days,
            "max_data_age_days_review_threshold": max_data_age_days,
            "forecast_horizon_days": forecast_horizon_days,
            "eligible_history_row_count": int(len(eligible)),
            "excluded_future_or_cutoff_day_row_count": int(excluded_count),
            "history_state_latest_season": current_history_season,
            "history_state_latest_match_date": max(state_rows["date"]),
            "fitting_cutoff": fitting_cutoff,
            "intervening_unobserved_matches_simulated": False,
        },
        "input_sources": {
            "history": {"path": str(input_path.resolve()), "sha256": file_sha256(input_path)},
            "fixture": dict(fixture_source) if fixture_source is not None else {"type": "direct_arguments"},
            "team_aliases": dict(alias_source) if alias_source is not None else None,
            "market": dict(market_source) if market_source is not None else None,
            "history_validation": dict(history_stats),
        },
        "model": {
            "package": "matchquant",
            "package_version": "2.0.0",
            "model_identifier": "ordered-logit-dixon-coles-elo-track-a-v1",
            "prediction_label": "模型估计概率",
            "selection": {
                "track_a_features": selected_features,
                "track_a_validation_selection": selection_info,
                "factor_lab": {
                    "discovery_seasons": list(train_seasons),
                    "validation_season_excluded_from_discovery": validation_season,
                    "selected_market_factors_not_used_by_track_a": list(lab["selected_market_factors"]),
                    "analyzed_match_count": int(lab["n_matches_analyzed"]),
                },
                "partitions": {
                    "training_seasons": train_seasons,
                    "validation_season": validation_season,
                    "complete_seasons": complete_seasons,
                    "training_row_count": int(len(train_df)),
                    "validation_row_count": int(len(validation_df)),
                    "training_date_start": min(train_df["date"].astype(str)),
                    "training_date_end": max(train_df["date"].astype(str)),
                    "validation_date_start": min(validation_df["date"].astype(str)),
                    "validation_date_end": max(validation_df["date"].astype(str)),
                    "selection_evidence_label": "validation tuning/selection evidence, not a blind test of the final refit",
                },
                "fixed_grid": {
                    "ordered_logit_l2": [0.5, 2.0, 8.0, 25.0],
                    "dixon_coles_l2": [1.0, 5.0, 15.0, 40.0],
                    "blend_weights_step": 0.1,
                    "final_refit_rows": int(len(train_df) + len(validation_df)),
                },
            },
        },
        "predictions": predictions,
        "market_comparison": market_result,
        "features": {
            "raw_inference_values": raw_features,
            "attribution": attribution,
        },
        "recent_evidence": recent,
        "limitations": [
            "比赛日期、赛季与球队由用户提供，未在线核对官方赛程或当前联赛身份。",
            "历史文件的最终版本不能证明每项原始观测在历史时点已经可得；截止日只约束本工具使用的行。",
            "没有伤停、首发、新闻、战术计划或实时数据源；数据截止后发生的比赛不会被模拟。",
            "期望进球是模型均值，不是精确比分承诺；本报告不构成投注建议或未来盈利保证。",
        ],
    }


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if math.isfinite(float(value)) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (pd.Timestamp, datetime, date)):
        return value.isoformat()
    return value


def _unique_run_directory(output_root: str | Path, fixture: Mapping[str, Any]) -> Path:
    root = Path(output_root).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    slug = re.sub(r"[^a-z0-9]+", "-", f"{fixture['home_team']}-{fixture['away_team']}".casefold()).strip("-")[:40]
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    directory = root / f"match_report_{fixture['date']}_{slug}_{stamp}_{uuid.uuid4().hex[:6]}"
    directory.mkdir(exist_ok=False)
    return directory


def run_single_match_report(
    *,
    input_path: str | Path,
    as_of: str,
    output_root: str | Path,
    mode: str = "research",
    fixture_csv: str | Path | None = None,
    home: str | None = None,
    away: str | None = None,
    fixture_date: str | None = None,
    season: str | None = None,
    team_aliases_path: str | Path | None = None,
    market_input_path: str | Path | None = None,
    max_data_age_days: int = 30,
) -> tuple[dict[str, Any], dict[str, str]]:
    """Validate, fit, predict, render, and write the three single-match artifacts."""

    if isinstance(max_data_age_days, bool) or not isinstance(max_data_age_days, int):
        raise SingleMatchError("--max-data-age-days must be a non-negative integer")
    if max_data_age_days < 0:
        raise SingleMatchError("--max-data-age-days must be a non-negative integer")
    as_of_day = parse_iso_date(as_of, "--as-of")
    source = Path(input_path).expanduser().resolve()
    raw_history = _read_history(source)
    eligible, excluded_count, history_stats = _validate_history(raw_history, as_of_day)
    team_names = sorted(set(eligible["home_team"]) | set(eligible["away_team"]))
    aliases, alias_source = _load_aliases(team_aliases_path, team_names)
    fixture = normalize_fixture(
        fixture_csv=fixture_csv,
        home=home,
        away=away,
        fixture_date=fixture_date,
        season=season,
        as_of=as_of,
        team_names=team_names,
        aliases=aliases,
    )
    fixture_source = (
        {
            "type": "score_free_csv",
            "path": str(Path(fixture_csv).expanduser().resolve()),
            "sha256": file_sha256(fixture_csv),
        }
        if fixture_csv is not None
        else {"type": "direct_arguments"}
    )
    market, market_source = _load_market_input(
        market_input_path,
        fixture=fixture,
        as_of=as_of_day,
        team_names=team_names,
        aliases=aliases,
    )
    result = _build_result(
        input_path=source,
        eligible=eligible,
        excluded_count=excluded_count,
        history_stats=history_stats,
        fixture=fixture,
        as_of=as_of_day,
        mode=mode,
        max_data_age_days=max_data_age_days,
        alias_source=alias_source,
        fixture_source=fixture_source,
        market=market,
        market_source=market_source,
    )

    from .match_report import render_report_html, render_report_markdown

    run_directory = _unique_run_directory(output_root, fixture)
    paths = {
        "json": str((run_directory / "match_report.json").resolve()),
        "markdown": str((run_directory / "match_report.md").resolve()),
        "html": str((run_directory / "match_report.html").resolve()),
    }
    result["artifacts"] = paths
    safe_result = _json_safe(result)
    json_text = json.dumps(safe_result, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    markdown = render_report_markdown(safe_result)
    html = render_report_html(safe_result)
    Path(paths["json"]).write_text(json_text, encoding="utf-8")
    Path(paths["markdown"]).write_text(markdown, encoding="utf-8")
    Path(paths["html"]).write_text(html, encoding="utf-8")
    return safe_result, paths


__all__ = [
    "MARKET_ALLOWED_FIELDS",
    "PENDING_FIXTURE_ALLOWED_COLUMNS",
    "SINGLE_MATCH_SCHEMA_VERSION",
    "SingleMatchError",
    "build_query_features",
    "normalize_fixture",
    "parse_iso_date",
    "run_single_match_report",
]
