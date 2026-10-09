"""Pure, bounded calculation of M13A result factors from an eligible timeline."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping

import pandas as pd

from .engine_snapshot_support import configuration_hash
from .models import EloModel
from .schema import DataValidationError

RESULT_FACTOR_MANIFEST_SCHEMA_VERSION = "m13-result-factor-manifest-v1"
RESULT_FACTOR_STATE_SCHEMA_VERSION = "m13-result-factor-state-v1"
RESULT_FACTOR_QUERY_SCHEMA_VERSION = "m13-result-factor-query-v1"
RESULT_FACTOR_STATE_FILENAME = "team_factor_state.json"
RESULT_FACTOR_MANIFEST_FILENAME = "factor_manifest.json"
RESULT_FACTOR_BUILD_REPORT_FILENAME = "factor_build_report.md"
RESULT_FACTOR_QUERY_FILENAME = "factor_query.json"
RESULT_FACTOR_QUERY_CSV_FILENAME = "factor_features.csv"
RESULT_FACTOR_QUERY_REPORT_FILENAME = "factor_query_report.md"
RESULT_FACTOR_SNAPSHOT_MAX_BYTES = 5 * 1024 * 1024
RESULT_FACTOR_WINDOWS = (5, 10)
RESULT_FACTOR_MINIMUM_SAMPLE = 3
RESULT_FACTOR_VENUE_PRIOR_WEIGHT = 5
RESULT_FACTOR_COVERAGE_STATUS = "frozen_snapshot_coverage_not_verified"

ELO_CONFIG: dict[str, int | float] = {
    "initial_rating": 1500,
    "k_factor": 32,
    "home_advantage": 50,
    "rating_scale": 400,
    "draw_base": 0.27,
    "draw_scale": 400,
}
RESULT_FACTOR_CONFIG: dict[str, Any] = {
    "elo": ELO_CONFIG,
    "season_mode": "carry",
    "windows": [5, 10],
    "minimum_sample_for_ready": RESULT_FACTOR_MINIMUM_SAMPLE,
    "venue_prior_weight": RESULT_FACTOR_VENUE_PRIOR_WEIGHT,
}
RESULT_FACTOR_CONFIG_SHA256 = configuration_hash(RESULT_FACTOR_CONFIG)

AS_OF_KEYS = ("value", "timestamp", "date_only")
CONFIG_KEYS = (
    "elo",
    "season_mode",
    "windows",
    "minimum_sample_for_ready",
    "venue_prior_weight",
)
ELO_CONFIG_KEYS = (
    "initial_rating",
    "k_factor",
    "home_advantage",
    "rating_scale",
    "draw_base",
    "draw_scale",
)
LEAGUE_PRIOR_KEYS = ("home", "away")
LEAGUE_SIDE_KEYS = (
    "matches",
    "goals_for_sum",
    "goals_against_sum",
    "goals_for_mean",
    "goals_against_mean",
)
TEAM_STATE_KEYS = (
    "total_appearances",
    "goals_for_sum",
    "goals_against_sum",
    "venue_totals",
    "final_elo",
    "appearances",
)
VENUE_TOTALS_KEYS = ("home", "away")
VENUE_TOTAL_KEYS = ("matches", "goals_for_sum", "goals_against_sum")
APPEARANCE_KEYS = (
    "match_id",
    "batch_time",
    "date",
    "date_only",
    "season",
    "venue",
    "goals_for",
    "goals_against",
    "points",
    "opponent_pre_batch_elo",
)
STATE_KEYS = (
    "schema_version",
    "as_of",
    "used_data_cutoff",
    "rows_used",
    "config",
    "config_sha256",
    "league_priors",
    "teams",
)


def build_result_factor_state(
    eligible_history: pd.DataFrame,
    *,
    as_of: Mapping[str, Any],
    rows_total: int,
) -> dict[str, Any]:
    """Build fixed-window team state; freeze a whole batch before Elo updates."""

    if not isinstance(eligible_history, pd.DataFrame):
        raise TypeError("eligible_history must be a pandas DataFrame")
    if isinstance(rows_total, bool) or not isinstance(rows_total, int) or rows_total < len(eligible_history):
        raise DataValidationError("rows_total must be an integer at least rows_used")
    if set(AS_OF_KEYS) - set(as_of):
        raise DataValidationError("as_of metadata is incomplete")

    elo = EloModel(**ELO_CONFIG)
    teams: dict[str, dict[str, Any]] = {}
    league = {
        "home": {"matches": 0, "goals_for_sum": 0, "goals_against_sum": 0},
        "away": {"matches": 0, "goals_for_sum": 0, "goals_against_sum": 0},
    }

    def team_state(name: str) -> dict[str, Any]:
        if name not in teams:
            teams[name] = {
                "total_appearances": 0,
                "goals_for_sum": 0,
                "goals_against_sum": 0,
                "venue_totals": {
                    "home": {"matches": 0, "goals_for_sum": 0, "goals_against_sum": 0},
                    "away": {"matches": 0, "goals_for_sum": 0, "goals_against_sum": 0},
                },
                "final_elo": float(ELO_CONFIG["initial_rating"]),
                "appearances": [],
            }
        return teams[name]

    for _, batch in eligible_history.groupby("_batch_key", sort=False):
        # Store all pre-batch values before updating any result in this batch.
        frozen: list[dict[str, Any]] = []
        for row in batch.to_dict(orient="records"):
            home = str(row["home_team"])
            away = str(row["away_team"])
            home_goals = int(row["home_goals"])
            away_goals = int(row["away_goals"])
            home_rating = float(elo.ratings.get(home, elo.initial_rating))
            away_rating = float(elo.ratings.get(away, elo.initial_rating))
            batch_time = pd.Timestamp(row["_batch_key"]).isoformat()
            result = 3 if home_goals > away_goals else 1 if home_goals == away_goals else 0
            away_result = 3 if away_goals > home_goals else 1 if home_goals == away_goals else 0
            frozen.append({
            "row": row,
                "home_pre_elo": home_rating,
                "away_pre_elo": away_rating,
                "home_points": result,
                "away_points": away_result,
                "batch_time": batch_time,
            })

        for item in frozen:
            row = item["row"]
            home = str(row["home_team"])
            away = str(row["away_team"])
            home_goals = int(row["home_goals"])
            away_goals = int(row["away_goals"])
            home_team = team_state(home)
            away_team = team_state(away)
            home_appearance = {
                "match_id": str(row["match_id"]),
                "batch_time": item["batch_time"],
                "date": str(row["date"]),
                "date_only": bool(row["_date_only"]),
                "season": str(row["season"]),
                "venue": "home",
                "goals_for": home_goals,
                "goals_against": away_goals,
                "points": item["home_points"],
                "opponent_pre_batch_elo": item["away_pre_elo"],
            }
            away_appearance = {
                "match_id": str(row["match_id"]),
                "batch_time": item["batch_time"],
                "date": str(row["date"]),
                "date_only": bool(row["_date_only"]),
                "season": str(row["season"]),
                "venue": "away",
                "goals_for": away_goals,
                "goals_against": home_goals,
                "points": item["away_points"],
                "opponent_pre_batch_elo": item["home_pre_elo"],
            }
            _accumulate_team(home_team, home_appearance)
            _accumulate_team(away_team, away_appearance)
            league["home"]["matches"] += 1
            league["home"]["goals_for_sum"] += home_goals
            league["home"]["goals_against_sum"] += away_goals
            league["away"]["matches"] += 1
            league["away"]["goals_for_sum"] += away_goals
            league["away"]["goals_against_sum"] += home_goals
            # Reuse the project's Elo formula and stable match-id update order.
            elo.update(home, away, home_goals, away_goals)

    for name, team in teams.items():
        team["final_elo"] = float(elo.ratings.get(name, elo.initial_rating))
    league_priors = {venue: _league_side(values) for venue, values in league.items()}
    used_data_cutoff = str(eligible_history.iloc[-1]["date"]) if not eligible_history.empty else None
    return {
        "schema_version": RESULT_FACTOR_STATE_SCHEMA_VERSION,
        "as_of": {key: as_of[key] for key in AS_OF_KEYS},
        "used_data_cutoff": used_data_cutoff,
        "rows_used": int(len(eligible_history)),
        "config": deepcopy(RESULT_FACTOR_CONFIG),
        "config_sha256": RESULT_FACTOR_CONFIG_SHA256,
        "league_priors": league_priors,
        "teams": teams,
    }


def _accumulate_team(team: dict[str, Any], appearance: dict[str, Any]) -> None:
    team["total_appearances"] += 1
    team["goals_for_sum"] += appearance["goals_for"]
    team["goals_against_sum"] += appearance["goals_against"]
    venue = team["venue_totals"][appearance["venue"]]
    venue["matches"] += 1
    venue["goals_for_sum"] += appearance["goals_for"]
    venue["goals_against_sum"] += appearance["goals_against"]
    appearances = team["appearances"]
    appearances.append(appearance)
    if len(appearances) > 10:
        del appearances[0 : len(appearances) - 10]


def _league_side(values: Mapping[str, int]) -> dict[str, Any]:
    n = values["matches"]
    gf = values["goals_for_sum"]
    ga = values["goals_against_sum"]
    return {
        "matches": n,
        "goals_for_sum": gf,
        "goals_against_sum": ga,
        "goals_for_mean": float(gf / n) if n else None,
        "goals_against_mean": float(ga / n) if n else None,
    }


def team_window_factors(team: Mapping[str, Any] | None, window: int) -> dict[str, Any]:
    if window not in RESULT_FACTOR_WINDOWS:
        raise ValueError(f"unsupported result-factor window: {window}")
    appearances = list(team["appearances"]) if team is not None else []
    window_appearances = appearances[-window:]
    n = len(window_appearances)
    return {
        "n": n,
        "requested_window": window,
        "small_sample": n < RESULT_FACTOR_MINIMUM_SAMPLE,
        "window_complete": n >= window,
        "points_rate": (
            float(sum(item["points"] for item in window_appearances) / (3 * n)) if n else None
        ),
        "goals_for_mean": (
            float(sum(item["goals_for"] for item in window_appearances) / n) if n else None
        ),
        "goals_against_mean": (
            float(sum(item["goals_against"] for item in window_appearances) / n) if n else None
        ),
        "opponent_pre_elo_mean": (
            float(sum(item["opponent_pre_batch_elo"] for item in window_appearances) / n)
            if n else None
        ),
    }


def venue_shrunk_factors(
    team: Mapping[str, Any] | None,
    venue: str,
    league_priors: Mapping[str, Any],
) -> dict[str, Any]:
    if venue not in {"home", "away"}:
        raise ValueError("venue must be 'home' or 'away'")
    prior = league_priors[venue]
    team_venue = team["venue_totals"][venue] if team is not None else {
        "matches": 0, "goals_for_sum": 0, "goals_against_sum": 0,
    }
    n = team_venue["matches"]
    weight = RESULT_FACTOR_VENUE_PRIOR_WEIGHT
    gf_prior = prior["goals_for_mean"]
    ga_prior = prior["goals_against_mean"]
    gf = (
        float((team_venue["goals_for_sum"] + weight * gf_prior) / (n + weight))
        if gf_prior is not None else None
    )
    ga = (
        float((team_venue["goals_against_sum"] + weight * ga_prior) / (n + weight))
        if ga_prior is not None else None
    )
    return {
        "goals_for_shrunk": gf,
        "goals_against_shrunk": ga,
        "sample_n": n,
        "prior_matches": prior["matches"],
        "prior_used": prior["matches"] > 0,
    }


def calendar_rest_days(fixture_date: object, last_match_date: object | None) -> int | None:
    """Calendar-date gap only; this deliberately makes no kickoff-time claim."""

    if last_match_date is None:
        return None
    fixture_day = pd.Timestamp(fixture_date).date()
    last_day = pd.Timestamp(last_match_date).date()
    days = int((fixture_day - last_day).days)
    if days < 0:
        raise DataValidationError("fixture date precedes the team's last eligible appearance")
    return days


def fixture_team_factors(
    state: Mapping[str, Any],
    team_name: str,
    venue: str,
    fixture_date: object,
) -> dict[str, Any]:
    team = state["teams"].get(team_name)
    appearances = list(team["appearances"]) if team is not None else []
    last_date = appearances[-1]["date"] if appearances else None
    rest = calendar_rest_days(fixture_date, last_date)
    result = {
        "unknown_team": team is None,
        "elo_rating": float(team["final_elo"]) if team is not None else float(ELO_CONFIG["initial_rating"]),
        "elo_prior": team is None,
        "total_appearances": int(team["total_appearances"]) if team is not None else 0,
        "recent_5": team_window_factors(team, 5),
        "recent_10": team_window_factors(team, 10),
        "venue": venue_shrunk_factors(team, venue, state["league_priors"]),
        "rest_calendar_days": rest,
        "last_match_date": str(last_date) if last_date is not None else None,
        "unverified_intervening_matches": True,
    }
    return result


RESULT_FACTOR_QUERY_IDENTITY_COLUMNS = (
    "fixture_id", "date", "season", "home_team", "away_team",
)
RESULT_FACTOR_QUERY_VALUE_COLUMNS = (
    "unknown_team", "elo_rating", "elo_prior", "total_appearances",
    "n_5", "requested_window_5", "small_sample_5", "window_complete_5",
    "points_rate_5", "goals_for_mean_5", "goals_against_mean_5", "opponent_pre_elo_mean_5",
    "n_10", "requested_window_10", "small_sample_10", "window_complete_10",
    "points_rate_10", "goals_for_mean_10", "goals_against_mean_10", "opponent_pre_elo_mean_10",
    "venue_goals_for_shrunk", "venue_goals_against_shrunk", "venue_sample_n", "venue_prior_used",
    "rest_calendar_days", "last_match_date", "unverified_intervening_matches",
)
RESULT_FACTOR_QUERY_COLUMNS = (
    *RESULT_FACTOR_QUERY_IDENTITY_COLUMNS,
    *(f"{side}_{column}" for side in ("home", "away") for column in RESULT_FACTOR_QUERY_VALUE_COLUMNS),
)


def flatten_fixture_factors(
    fixture: Mapping[str, Any], home: Mapping[str, Any], away: Mapping[str, Any]
) -> dict[str, Any]:
    row = {column: fixture[column] for column in RESULT_FACTOR_QUERY_IDENTITY_COLUMNS}
    for side, factors in (("home", home), ("away", away)):
        row.update({
            f"{side}_unknown_team": factors["unknown_team"],
            f"{side}_elo_rating": factors["elo_rating"],
            f"{side}_elo_prior": factors["elo_prior"],
            f"{side}_total_appearances": factors["total_appearances"],
        })
        for window in RESULT_FACTOR_WINDOWS:
            feature = factors[f"recent_{window}"]
            suffix = str(window)
            row.update({
                f"{side}_n_{suffix}": feature["n"],
                f"{side}_requested_window_{suffix}": feature["requested_window"],
                f"{side}_small_sample_{suffix}": feature["small_sample"],
                f"{side}_window_complete_{suffix}": feature["window_complete"],
                f"{side}_points_rate_{suffix}": feature["points_rate"],
                f"{side}_goals_for_mean_{suffix}": feature["goals_for_mean"],
                f"{side}_goals_against_mean_{suffix}": feature["goals_against_mean"],
                f"{side}_opponent_pre_elo_mean_{suffix}": feature["opponent_pre_elo_mean"],
            })
        venue = factors["venue"]
        row.update({
            f"{side}_venue_goals_for_shrunk": venue["goals_for_shrunk"],
            f"{side}_venue_goals_against_shrunk": venue["goals_against_shrunk"],
            f"{side}_venue_sample_n": venue["sample_n"],
            f"{side}_venue_prior_used": venue["prior_used"],
            f"{side}_rest_calendar_days": factors["rest_calendar_days"],
            f"{side}_last_match_date": factors["last_match_date"],
            f"{side}_unverified_intervening_matches": factors["unverified_intervening_matches"],
        })
    return row


__all__ = [
    "AS_OF_KEYS", "APPEARANCE_KEYS", "CONFIG_KEYS", "ELO_CONFIG", "ELO_CONFIG_KEYS",
    "LEAGUE_PRIOR_KEYS", "LEAGUE_SIDE_KEYS", "RESULT_FACTOR_BUILD_REPORT_FILENAME",
    "RESULT_FACTOR_CONFIG", "RESULT_FACTOR_CONFIG_SHA256", "RESULT_FACTOR_COVERAGE_STATUS",
    "RESULT_FACTOR_MANIFEST_FILENAME", "RESULT_FACTOR_MANIFEST_SCHEMA_VERSION",
    "RESULT_FACTOR_MINIMUM_SAMPLE", "RESULT_FACTOR_QUERY_COLUMNS", "RESULT_FACTOR_QUERY_CSV_FILENAME",
    "RESULT_FACTOR_QUERY_FILENAME", "RESULT_FACTOR_QUERY_REPORT_FILENAME",
    "RESULT_FACTOR_QUERY_SCHEMA_VERSION", "RESULT_FACTOR_SNAPSHOT_MAX_BYTES", "RESULT_FACTOR_STATE_FILENAME",
    "RESULT_FACTOR_STATE_SCHEMA_VERSION", "RESULT_FACTOR_VENUE_PRIOR_WEIGHT", "RESULT_FACTOR_WINDOWS",
    "STATE_KEYS", "TEAM_STATE_KEYS", "VENUE_TOTALS_KEYS", "VENUE_TOTAL_KEYS",
    "build_result_factor_state", "calendar_rest_days", "fixture_team_factors", "flatten_fixture_factors",
    "team_window_factors", "venue_shrunk_factors",
]
