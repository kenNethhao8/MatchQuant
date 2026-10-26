from __future__ import annotations

import copy
import json
import math
import tempfile
import unittest
from datetime import date, timedelta
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from matchquant.factor_engine import ALL_FACTOR_COLUMNS, FUNDAMENTAL_FACTOR_COLUMNS, build_historical_factor_panel
from matchquant.factor_lab import run_factor_lab
from matchquant.match_report import render_report_html, render_report_markdown
from matchquant.rich_data import convert_raw_epl_season, devig_power_probabilities
from matchquant.single_match import (
    MARKET_ALLOWED_FIELDS,
    SingleMatchError,
    _build_result,
    _load_market_input,
    _load_aliases,
    _partition_seasons,
    _validate_fixture_season,
    _validate_history,
    build_query_features,
    normalize_fixture,
    parse_iso_date,
    run_single_match_report,
)


SEASONS = ("2021-22", "2022-23", "2023-24")
AS_OF = "2024-05-20"
FIXTURE_DATE = "2024-08-18"
FIXTURE_SEASON = "2024-25"
TEAM_NAMES = [f"Club {chr(ord('A') + index)}" for index in range(20)]


def synthetic_rich_history(seasons: tuple[str, ...] = SEASONS) -> pd.DataFrame:
    """Create deterministic three-season, 20-team, 380-match EPL-shaped history."""
    rows = []
    for season in seasons:
        start_year = int(season[:4])
        rotation = list(range(20))
        rounds: list[list[tuple[int, int]]] = []
        for round_number in range(19):
            pairings = [(rotation[i], rotation[-1 - i]) for i in range(10)]
            rounds.append(pairings)
            rotation = [rotation[0], rotation[-1], *rotation[1:-1]]
        for leg in range(2):
            for round_number, pairings in enumerate(rounds):
                match_round = round_number + 19 * leg
                match_date = date(start_year, 8, 1) + timedelta(days=7 * match_round)
                for pair_number, (first, second) in enumerate(pairings):
                    home_idx, away_idx = (first, second)
                    if (round_number + pair_number) % 2:
                        home_idx, away_idx = away_idx, home_idx
                    if leg:
                        home_idx, away_idx = away_idx, home_idx
                    home_goals = (home_idx + match_round + start_year) % 4
                    away_goals = (away_idx + 2 * match_round + start_year) % 3
                    ht_home = min(home_goals, (home_goals + 1) // 2)
                    ht_away = min(away_goals, (away_goals + 1) // 2)
                    p_home, p_draw, p_away = 0.46, 0.28, 0.26
                    row = {
                        "match_id": f"{season}-{match_round:02d}-{pair_number:02d}",
                        "date": match_date.isoformat(),
                        "season": season,
                        "home_team": TEAM_NAMES[home_idx],
                        "away_team": TEAM_NAMES[away_idx],
                        "home_goals": home_goals,
                        "away_goals": away_goals,
                        "actual_result": "H" if home_goals > away_goals else ("A" if home_goals < away_goals else "D"),
                        "referee": f"Referee {pair_number % 5}",
                        "kickoff_time": "15:00",
                        "ht_home_goals": ht_home,
                        "ht_away_goals": ht_away,
                        "ht_result": "H" if ht_home > ht_away else ("A" if ht_home < ht_away else "D"),
                        "sh_home_goals": home_goals - ht_home,
                        "sh_away_goals": away_goals - ht_away,
                        "home_shots": 10 + (home_idx + match_round) % 10,
                        "away_shots": 8 + (away_idx + match_round) % 9,
                        "home_sot": 2 + (home_idx + match_round) % 6,
                        "away_sot": 2 + (away_idx + 2 * match_round) % 5,
                        "home_corners": 3 + (home_idx + match_round) % 8,
                        "away_corners": 2 + (away_idx + match_round) % 7,
                        "home_fouls": 8 + (home_idx + match_round) % 12,
                        "away_fouls": 8 + (away_idx + match_round) % 12,
                        "home_yellow": (home_idx + match_round) % 5,
                        "away_yellow": (away_idx + match_round) % 5,
                        "home_red": 0,
                        "away_red": 0,
                        "odds_open_h": 2.1,
                        "odds_open_d": 3.4,
                        "odds_open_a": 3.7,
                        "odds_close_h": 2.0,
                        "odds_close_d": 3.5,
                        "odds_close_a": 3.8,
                        "mkt_open_prob_home": p_home,
                        "mkt_open_prob_draw": p_draw,
                        "mkt_open_prob_away": p_away,
                        "mkt_close_prob_home": 0.47,
                        "mkt_close_prob_draw": 0.27,
                        "mkt_close_prob_away": 0.26,
                        "mkt_steam_home": 0.01,
                        "mkt_steam_draw": -0.01,
                        "mkt_steam_away": 0.0,
                        "mkt_ou25_over_prob": 0.51,
                        "mkt_ah_line": -0.25,
                    }
                    rows.append(row)
    return pd.DataFrame(rows)


def _fixture(home: str = "Club A", away: str = "Club B") -> dict:
    return {
        "fixture_id": f"{home}-{away}",
        "date": FIXTURE_DATE,
        "season": FIXTURE_SEASON,
        "home_team": home,
        "away_team": away,
        "referee": "Unknown",
        "kickoff_time": "",
        "requested_home_team": home,
        "requested_away_team": away,
    }


class _FakeOrdered:
    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        return np.tile(np.array([[0.52, 0.27, 0.21]]), (len(frame), 1))

    def explain_row(self, row: dict) -> dict:
        return {
            "top_factor_contributions": [
                {"factor": "sot_dominance_diff", "raw_value": row.get("sot_dominance_diff", 0.0), "marginal_prob_h_pct": 1.24},
                {"factor": "corner_pressure_diff", "raw_value": row.get("corner_pressure_diff", 0.0), "marginal_prob_h_pct": -0.42},
                {"factor": "elo_diff", "raw_value": row.get("elo_diff", 0.0), "marginal_prob_h_pct": 0.0},
            ]
        }


class _FakeDC:
    def predict_proba_and_goals(self, frame: pd.DataFrame):
        n = len(frame)
        return (
            np.tile(np.array([[0.48, 0.29, 0.23]]), (n, 1)),
            np.full(n, 1.45),
            np.full(n, 1.12),
            np.full(n, 0.47),
        )


class _FakeSystem:
    def __init__(self) -> None:
        self.weight_ordered = 0.5
        self.weight_dc = 0.3
        self.weight_elo = 0.2
        self.ordered_model = _FakeOrdered()
        self.dixon_coles_model = _FakeDC()

    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        p_elo = frame[["elo_prob_h", "elo_prob_d", "elo_prob_a"]].to_numpy(dtype=float)
        p_ordered = self.ordered_model.predict_proba(frame)
        p_dc = self.dixon_coles_model.predict_proba_and_goals(frame)[0]
        blended = self.weight_ordered * p_ordered + self.weight_dc * p_dc + self.weight_elo * p_elo
        return blended / blended.sum(axis=1, keepdims=True)


def _fake_fit(train, validation, features, *, track="fundamental"):
    return _FakeSystem(), {
        "track": track,
        "selected_ordered_l2": 2.0,
        "selected_dc_l2": 5.0,
        "selected_dixon_coles_rho": -0.05,
        "blend_weights": {"ordered_logit": 0.5, "dixon_coles": 0.3, "elo_baseline": 0.2},
        "validation_log_loss": {
            "elo_baseline": 1.1,
            "ordered_factor_only": 1.08,
            "dixon_coles_factor_only": 1.09,
            "blended_system": 1.07,
        },
    }


def _fake_lab(panel, *, discovery_seasons=None):
    return {
        "selected_fundamental_factors": ["elo_diff", "elo_abs_gap", "expected_tempo", "sot_dominance_diff"],
        "selected_market_factors": ["mkt_open_prob_edge", "mkt_steam_move_diff", "mkt_ah_supremacy"],
        "n_matches_analyzed": int(panel["season"].isin(set(discovery_seasons or [])).sum()),
    }


class SingleMatchBoundaryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.history = synthetic_rich_history()

    def test_iso_date_season_and_fixture_input_validation(self) -> None:
        self.assertEqual(parse_iso_date("2024-05-20", "date"), date(2024, 5, 20))
        for value in ("2024/05/20", "2024-02-30", "tonight"):
            with self.assertRaises(SingleMatchError):
                parse_iso_date(value, "date")
        names = TEAM_NAMES
        with self.assertRaisesRegex(SingleMatchError, "on or after"):
            normalize_fixture(home="Club A", away="Club B", fixture_date="2024-05-19", season=FIXTURE_SEASON, as_of=AS_OF, team_names=names)
        with self.assertRaisesRegex(SingleMatchError, "requires --home"):
            normalize_fixture(home="Club A", away=None, fixture_date=FIXTURE_DATE, season=FIXTURE_SEASON, as_of=AS_OF, team_names=names)
        with self.assertRaises(SingleMatchError):
            normalize_fixture(home="Club A", away="Club B", fixture_date=FIXTURE_DATE, season="2024-26", as_of=AS_OF, team_names=names)
        resolved = normalize_fixture(home="  CLUB   A ", away="Club B", fixture_date=FIXTURE_DATE, season=FIXTURE_SEASON, as_of=AS_OF, team_names=names)
        self.assertEqual(resolved["home_team"], "Club A")
        aliased = normalize_fixture(home="AFC Alpha", away="Club B", fixture_date=FIXTURE_DATE, season=FIXTURE_SEASON, as_of=AS_OF, team_names=names, aliases={"afc alpha": "Club A"})
        self.assertEqual(aliased["home_team"], "Club A")
        with tempfile.TemporaryDirectory() as temporary:
            aliases_path = Path(temporary) / "aliases.json"
            aliases_path.write_text(json.dumps({"AFC Alpha": "Not a Historical Team"}), encoding="utf-8")
            with self.assertRaisesRegex(SingleMatchError, "unknown team"):
                _load_aliases(aliases_path, names)
        with self.assertRaisesRegex(SingleMatchError, "unknown team.*Suggestions"):
            normalize_fixture(home="Club Z", away="Club B", fixture_date=FIXTURE_DATE, season=FIXTURE_SEASON, as_of=AS_OF, team_names=names)
        with self.assertRaisesRegex(SingleMatchError, "same historical team"):
            normalize_fixture(home="Club A", away=" club   a ", fixture_date=FIXTURE_DATE, season=FIXTURE_SEASON, as_of=AS_OF, team_names=names)
        with self.assertRaisesRegex(SingleMatchError, "ambiguous"):
            normalize_fixture(home="Club A", away="Club B", fixture_date=FIXTURE_DATE, season=FIXTURE_SEASON, as_of=AS_OF, team_names=[*names, "club a"])
        with self.assertRaisesRegex(SingleMatchError, "either --fixtures"):
            normalize_fixture(fixture_csv="unused.csv", home="Club A", away=None, fixture_date=None, season=None, as_of=AS_OF, team_names=names)

    def test_fixture_csv_allowlist_one_row_and_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            valid = root / "fixture.csv"
            valid.write_text("fixture_id,date,season,home_team,away_team,referee,kickoff_time\nf-1,2024-08-18,2024-25,Club A,Club B,Referee X,15:00\n", encoding="utf-8")
            normalized = normalize_fixture(fixture_csv=valid, as_of=AS_OF, team_names=TEAM_NAMES)
            self.assertEqual(normalized["fixture_id"], "f-1")
            self.assertEqual(normalized["referee"], "Referee X")
            self.assertEqual(normalized["kickoff_time"], "15:00")
            bad = root / "scored.csv"
            bad.write_text("fixture_id,date,season,home_team,away_team,FTAG\nf-1,2024-08-18,2024-25,Club A,Club B,2\n", encoding="utf-8")
            with self.assertRaisesRegex(SingleMatchError, "unsupported fields"):
                normalize_fixture(fixture_csv=bad, as_of=AS_OF, team_names=TEAM_NAMES)
            multiple = root / "two.csv"
            multiple.write_text("fixture_id,date,season,home_team,away_team\nf-1,2024-08-18,2024-25,Club A,Club B\nf-2,2024-08-18,2024-25,Club C,Club D\n", encoding="utf-8")
            with self.assertRaisesRegex(SingleMatchError, "exactly one"):
                normalize_fixture(fixture_csv=multiple, as_of=AS_OF, team_names=TEAM_NAMES)

    def test_cutoff_excludes_whole_day_and_future_rows_before_score_validation(self) -> None:
        same_day = self.history.iloc[[0]].copy().astype(object)
        same_day.loc[:, "match_id"] = "same-day"
        same_day.loc[:, "date"] = AS_OF
        same_day.loc[:, "home_goals"] = 300
        future = self.history.iloc[[1]].copy().astype(object)
        future.loc[:, "match_id"] = "future"
        future.loc[:, "date"] = "2025-01-01"
        future.loc[:, "home_goals"] = "not-a-score"
        full = pd.concat([self.history, same_day, future], ignore_index=True)
        eligible, excluded, stats = _validate_history(full, date.fromisoformat(AS_OF))
        self.assertEqual(excluded, 2)
        self.assertEqual(len(eligible), len(self.history))
        self.assertEqual(stats["excluded_future_or_cutoff_day_rows"], 2)
        self.assertNotIn("same-day", set(eligible["match_id"]))
        self.assertNotIn("future", set(eligible["match_id"]))

    def test_history_rejects_eligible_duplicate_malformed_and_invalid_scores(self) -> None:
        duplicate = pd.concat([self.history, self.history.iloc[[0]]], ignore_index=True)
        with self.assertRaisesRegex(SingleMatchError, "duplicate match_id"):
            _validate_history(duplicate, date.fromisoformat(AS_OF))
        duplicate_fixture = pd.concat([self.history, self.history.iloc[[0]].copy()], ignore_index=True)
        duplicate_fixture.loc[len(duplicate_fixture) - 1, "match_id"] = "another-id"
        with self.assertRaisesRegex(SingleMatchError, "duplicate season/date/team"):
            _validate_history(duplicate_fixture, date.fromisoformat(AS_OF))
        malformed = self.history.copy().astype(object)
        malformed.loc[0, "home_sot"] = "bad"
        with self.assertRaisesRegex(SingleMatchError, "invalid numeric"):
            _validate_history(malformed, date.fromisoformat(AS_OF))
        bad_score = self.history.copy()
        bad_score.loc[0, "home_goals"] = -1
        with self.assertRaisesRegex(SingleMatchError, "invalid numeric"):
            _validate_history(bad_score, date.fromisoformat(AS_OF))
        bad_date = pd.concat([self.history, self.history.iloc[[0]]], ignore_index=True)
        bad_date.loc[len(bad_date) - 1, "date"] = "not-a-date"
        with self.assertRaisesRegex(SingleMatchError, "unparseable dates"):
            _validate_history(bad_date, date.fromisoformat(AS_OF))

    def test_missing_evidence_statistics_are_counted_and_stay_unavailable(self) -> None:
        partial = self.history.copy().astype(object)
        partial.loc[0, "home_sot"] = ""
        eligible, _, stats = _validate_history(partial, date.fromisoformat(AS_OF))
        self.assertEqual(stats["missing_post_match_stat_row_count"], 1)
        self.assertTrue(pd.isna(eligible.loc[0, "home_sot"]))

    def test_complete_season_gate_and_supported_season_transition(self) -> None:
        train, val, complete, current = _partition_seasons(self.history)
        self.assertEqual(train, ["2021-22", "2022-23"])
        self.assertEqual(val, "2023-24")
        self.assertEqual(len(complete), 3)
        self.assertEqual(current, "2023-24")
        with self.assertRaisesRegex(SingleMatchError, "at least two training"):
            _partition_seasons(self.history[self.history["season"].isin(SEASONS[:2])])
        _validate_fixture_season("2024-25", "2023-24")
        with self.assertRaisesRegex(SingleMatchError, "immediate successor"):
            _validate_fixture_season("2025-26", "2023-24")

    def test_partial_current_season_updates_state_but_not_fit_or_discovery_partitions(self) -> None:
        partial = self.history[self.history["season"].eq("2023-24")].tail(10).copy().astype(object)
        partial["match_id"] = [f"partial-{index}" for index in range(len(partial))]
        partial["season"] = "2024-25"
        partial["date"] = [f"2024-09-{index + 1:02d}" for index in range(len(partial))]
        raw = pd.concat([self.history, partial], ignore_index=True)
        cutoff = date(2024, 10, 1)
        eligible, excluded, stats = _validate_history(raw, cutoff)
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "history.csv"
            raw.to_csv(source, index=False)
            with patch("matchquant.single_match.run_factor_lab", side_effect=_fake_lab) as lab_mock, patch(
                "matchquant.single_match.fit_and_select_factor_system", side_effect=_fake_fit
            ):
                result = _build_result(
                    input_path=source,
                    eligible=eligible,
                    excluded_count=excluded,
                    history_stats=stats,
                    fixture={**_fixture(), "date": "2024-10-04", "season": "2024-25"},
                    as_of=cutoff,
                    mode="research",
                    max_data_age_days=30,
                    alias_source=None,
                    market=None,
                    market_source=None,
                )
        self.assertEqual(result["temporal"]["history_state_latest_season"], "2024-25")
        self.assertEqual(result["model"]["selection"]["partitions"]["training_row_count"], 760)
        self.assertEqual(result["model"]["selection"]["partitions"]["validation_row_count"], 380)
        self.assertEqual(result["model"]["selection"]["factor_lab"]["discovery_seasons"], ["2021-22", "2022-23"])
        self.assertEqual(lab_mock.call_args.kwargs["discovery_seasons"], ["2021-22", "2022-23"])

    def test_actual_factor_lab_selection_isolated_from_open_close_odds_and_steam(self) -> None:
        changed = self.history.copy()
        changed["odds_open_h"] = 8.0
        changed["odds_open_d"] = 2.2
        changed["odds_open_a"] = 1.8
        changed["mkt_open_prob_home"] = 0.20
        changed["mkt_open_prob_draw"] = 0.30
        changed["mkt_open_prob_away"] = 0.50
        changed["odds_close_h"] = 6.0
        changed["mkt_close_prob_home"] = 0.25
        changed["mkt_close_prob_draw"] = 0.30
        changed["mkt_close_prob_away"] = 0.45
        changed["mkt_steam_home"] = 0.05
        changed["mkt_steam_draw"] = -0.02
        changed["mkt_steam_away"] = -0.03
        changed["mkt_ah_line"] = 1.5
        panel_a, _ = build_historical_factor_panel(self.history)
        panel_b, _ = build_historical_factor_panel(changed)
        np.testing.assert_allclose(
            panel_a[list(FUNDAMENTAL_FACTOR_COLUMNS)].to_numpy(dtype=float),
            panel_b[list(FUNDAMENTAL_FACTOR_COLUMNS)].to_numpy(dtype=float),
            rtol=0.0,
            atol=1e-12,
        )
        lab_a = run_factor_lab(panel_a, discovery_seasons=["2021-22", "2022-23"])
        lab_b = run_factor_lab(panel_b, discovery_seasons=["2021-22", "2022-23"])
        self.assertEqual(lab_a["selected_fundamental_factors"], lab_b["selected_fundamental_factors"])

    def test_query_features_are_isolated_and_order_independent(self) -> None:
        eligible, _, _ = _validate_history(self.history, date.fromisoformat(AS_OF))
        _, engine = build_historical_factor_panel(eligible)
        before = (engine.current_season, len(engine.teams), engine.league_matches)
        first = _fixture("Club A", "Club B")
        second = _fixture("Club C", "Club D")
        query_a_1 = build_query_features(engine, first)
        build_query_features(engine, second)
        query_a_2 = build_query_features(engine, first)
        self.assertEqual(before, (engine.current_season, len(engine.teams), engine.league_matches))
        np.testing.assert_allclose(
            query_a_1.loc[0, list(ALL_FACTOR_COLUMNS)].to_numpy(dtype=float),
            query_a_2.loc[0, list(ALL_FACTOR_COLUMNS)].to_numpy(dtype=float),
            rtol=0.0,
            atol=1e-12,
        )
        self.assertNotIn("home_goals", query_a_1.columns)
        self.assertNotIn("actual_result", query_a_1.columns)

    def test_market_input_schema_devig_and_cutoff_validation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "market.json"
            payload = {
                "home_team": "Club A",
                "away_team": "Club B",
                "date": FIXTURE_DATE,
                "observed_at": "2024-05-19",
                "source": "sample user input",
                "odds_open_h": 2.1,
                "odds_open_d": 3.4,
                "odds_open_a": 3.7,
            }
            path.write_text(json.dumps(payload), encoding="utf-8")
            fixture = _fixture()
            market, identity = _load_market_input(path, fixture=fixture, as_of=date.fromisoformat(AS_OF), team_names=TEAM_NAMES, aliases=None)
            self.assertIsNone(market["mkt_ah_line"])
            self.assertTrue(identity["sha256"])
            expected = devig_power_probabilities(np.array([2.1]), np.array([3.4]), np.array([3.7]))
            np.testing.assert_allclose([market["probabilities"]["home"], market["probabilities"]["draw"], market["probabilities"]["away"]], [float(item[0]) for item in expected], atol=1e-12)
            self.assertEqual(set(payload), MARKET_ALLOWED_FIELDS - {"mkt_ah_line"})
            payload["observed_at"] = AS_OF
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(SingleMatchError, "strictly before"):
                _load_market_input(path, fixture=fixture, as_of=date.fromisoformat(AS_OF), team_names=TEAM_NAMES, aliases=None)
            payload["observed_at"] = "2024-05-19"
            payload["odds_open_d"] = 1.0
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(SingleMatchError, "greater than 1"):
                _load_market_input(path, fixture=fixture, as_of=date.fromisoformat(AS_OF), team_names=TEAM_NAMES, aliases=None)
            payload["odds_open_d"] = 3.4
            del payload["odds_open_d"]
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(SingleMatchError, "missing fields.*odds_open_d"):
                _load_market_input(path, fixture=fixture, as_of=date.fromisoformat(AS_OF), team_names=TEAM_NAMES, aliases=None)
            payload["odds_open_d"] = 3.4
            payload["closing_odds"] = 2.0
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(SingleMatchError, "unsupported market fields"):
                _load_market_input(path, fixture=fixture, as_of=date.fromisoformat(AS_OF), team_names=TEAM_NAMES, aliases=None)

    def test_missing_historical_ah_stays_missing_in_rich_converter(self) -> None:
        raw = pd.DataFrame(
            [{"Date": "01/08/2020", "HomeTeam": "Club A", "AwayTeam": "Club B", "FTHG": "1", "FTAG": "0"}]
        )
        converted = convert_raw_epl_season(raw, "2020-21")
        self.assertTrue(pd.isna(converted.loc[0, "mkt_ah_line"]))

    def test_invalid_fixture_fails_before_creating_report_output(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "history.csv"
            output_root = root / "reports"
            self.history.to_csv(source, index=False)
            with self.assertRaisesRegex(SingleMatchError, "unknown team"):
                run_single_match_report(
                    input_path=source,
                    as_of=AS_OF,
                    output_root=output_root,
                    mode="research",
                    home="Typo Team",
                    away="Club B",
                    fixture_date=FIXTURE_DATE,
                    season=FIXTURE_SEASON,
                )
            self.assertFalse(output_root.exists())

    def test_track_a_cutoff_and_market_value_isolation_with_controlled_fit(self) -> None:
        raw_base = self.history.copy()
        cutoff_rows = raw_base.iloc[[0, 1]].copy()
        cutoff_rows.loc[:, "match_id"] = ["cutoff-1", "future-1"]
        cutoff_rows.loc[:, "date"] = [AS_OF, "2025-01-01"]
        cutoff_rows.loc[:, "season"] = ["2023-24", "2024-25"]
        augmented = pd.concat([raw_base, cutoff_rows], ignore_index=True)
        changed = augmented.copy()
        for row in (len(changed) - 2, len(changed) - 1):
            changed.loc[row, "home_goals"] = 0 if changed.loc[row, "home_goals"] != 0 else 4
            changed.loc[row, "home_sot"] = 20

        market_changed = raw_base.copy()
        market_changed["odds_open_h"] = 8.0
        market_changed["odds_open_d"] = 2.2
        market_changed["odds_open_a"] = 1.8
        market_changed["mkt_open_prob_home"] = 0.20
        market_changed["mkt_open_prob_draw"] = 0.30
        market_changed["mkt_open_prob_away"] = 0.50
        market_changed["odds_close_h"] = 6.0
        market_changed["mkt_close_prob_home"] = 0.25
        market_changed["mkt_close_prob_draw"] = 0.30
        market_changed["mkt_close_prob_away"] = 0.45
        market_changed["mkt_steam_home"] = 0.05
        market_changed["mkt_steam_draw"] = -0.02
        market_changed["mkt_steam_away"] = -0.03
        market_changed["mkt_ah_line"] = 1.5

        results = []
        with tempfile.TemporaryDirectory() as temporary:
            with patch("matchquant.single_match.run_factor_lab", side_effect=_fake_lab) as lab_mock, patch(
                "matchquant.single_match.fit_and_select_factor_system", side_effect=_fake_fit
            ):
                shuffled = raw_base.sample(frac=1.0, random_state=17).reset_index(drop=True)
                for name, frame in (("base", raw_base), ("cutoff", augmented), ("changed", changed), ("market", market_changed), ("shuffled", shuffled)):
                    source = Path(temporary) / f"{name}.csv"
                    frame.to_csv(source, index=False)
                    eligible, excluded, stats = _validate_history(frame, date.fromisoformat(AS_OF))
                    results.append(
                        _build_result(
                            input_path=source,
                            eligible=eligible,
                            excluded_count=excluded,
                            history_stats=stats,
                            fixture=_fixture(),
                            as_of=date.fromisoformat(AS_OF),
                            mode="research",
                            max_data_age_days=30,
                            alias_source=None,
                            market=None,
                            market_source=None,
                        )
                    )
        base, cutoff, changed_result, market_result, shuffled_result = results
        for candidate in (cutoff, changed_result, market_result, shuffled_result):
            self.assertEqual(base["predictions"], candidate["predictions"])
            self.assertEqual(base["features"]["attribution"], candidate["features"]["attribution"])
            for factor in ALL_FACTOR_COLUMNS:
                if factor.startswith("mkt_"):
                    continue
                self.assertAlmostEqual(
                    base["features"]["raw_inference_values"][factor],
                    candidate["features"]["raw_inference_values"][factor],
                    delta=1e-12,
                )
            self.assertFalse(any(name.startswith("mkt_") for name in base["model"]["selection"]["track_a_features"]))
        self.assertEqual(cutoff["temporal"]["excluded_future_or_cutoff_day_row_count"], 2)
        self.assertGreaterEqual(lab_mock.call_count, 4)
        for call in lab_mock.call_args_list:
            self.assertEqual(call.kwargs["discovery_seasons"], ["2021-22", "2022-23"])

    def test_optional_market_comparison_survives_when_m_open_lacks_ah_and_runs_when_ready(self) -> None:
        eligible, excluded, stats = _validate_history(self.history, date.fromisoformat(AS_OF))
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            market_path = root / "market.json"
            market_payload = {
                "home_team": "Club A",
                "away_team": "Club B",
                "date": FIXTURE_DATE,
                "observed_at": "2024-05-19",
                "source": "sample user supplied source",
                "odds_open_h": 2.1,
                "odds_open_d": 3.4,
                "odds_open_a": 3.7,
            }
            market_path.write_text(json.dumps(market_payload), encoding="utf-8")
            market, source_identity = _load_market_input(
                market_path,
                fixture=_fixture(),
                as_of=date.fromisoformat(AS_OF),
                team_names=TEAM_NAMES,
                aliases=None,
            )
            history_path = root / "history.csv"
            self.history.to_csv(history_path, index=False)
            with patch("matchquant.single_match.run_factor_lab", side_effect=_fake_lab), patch(
                "matchquant.single_match.fit_and_select_factor_system", side_effect=_fake_fit
            ) as fit_mock:
                no_ah_result = _build_result(
                    input_path=history_path,
                    eligible=eligible,
                    excluded_count=excluded,
                    history_stats=stats,
                    fixture=_fixture(),
                    as_of=date.fromisoformat(AS_OF),
                    mode="research",
                    max_data_age_days=30,
                    alias_source=None,
                    market=market,
                    market_source=source_identity,
                )
                self.assertEqual(fit_mock.call_count, 1)
            self.assertTrue(no_ah_result["market_comparison"]["provided"])
            self.assertFalse(no_ah_result["market_comparison"]["m_open_available"])
            self.assertIn("亚盘", no_ah_result["market_comparison"]["m_open_unavailable_reason"])
            self.assertEqual(no_ah_result["status"], "needs_review")
            market_payload["mkt_ah_line"] = -0.25
            market_path.write_text(json.dumps(market_payload), encoding="utf-8")
            market_with_ah, source_identity = _load_market_input(
                market_path,
                fixture=_fixture(),
                as_of=date.fromisoformat(AS_OF),
                team_names=TEAM_NAMES,
                aliases=None,
            )
            with patch("matchquant.single_match.run_factor_lab", side_effect=_fake_lab), patch(
                "matchquant.single_match.fit_and_select_factor_system", side_effect=_fake_fit
            ) as fit_mock:
                ready_result = _build_result(
                    input_path=history_path,
                    eligible=eligible,
                    excluded_count=excluded,
                    history_stats=stats,
                    fixture=_fixture(),
                    as_of=date.fromisoformat(AS_OF),
                    mode="research",
                    max_data_age_days=30,
                    alias_source=None,
                    market=market_with_ah,
                    market_source=source_identity,
                )
                self.assertEqual(fit_mock.call_count, 2)
                self.assertEqual(fit_mock.call_args_list[1].kwargs["track"], "market_open")
            self.assertTrue(ready_result["market_comparison"]["m_open_available"])
            self.assertIsNotNone(ready_result["market_comparison"]["track_a_minus_market_percentage_points"])

            history_missing_ah = self.history.copy().astype(object)
            history_missing_ah.loc[0, "mkt_ah_line"] = ""
            eligible_missing_ah, excluded_missing_ah, stats_missing_ah = _validate_history(
                history_missing_ah, date.fromisoformat(AS_OF)
            )
            history_missing_path = root / "history_missing_ah.csv"
            history_missing_ah.to_csv(history_missing_path, index=False)
            with patch("matchquant.single_match.run_factor_lab", side_effect=_fake_lab), patch(
                "matchquant.single_match.fit_and_select_factor_system", side_effect=_fake_fit
            ) as fit_mock:
                history_no_ah_result = _build_result(
                    input_path=history_missing_path,
                    eligible=eligible_missing_ah,
                    excluded_count=excluded_missing_ah,
                    history_stats=stats_missing_ah,
                    fixture=_fixture(),
                    as_of=date.fromisoformat(AS_OF),
                    mode="research",
                    max_data_age_days=30,
                    alias_source=None,
                    market=market_with_ah,
                    market_source=source_identity,
                )
                self.assertEqual(fit_mock.call_count, 1)
            self.assertFalse(history_no_ah_result["market_comparison"]["m_open_available"])
            self.assertIn("缺失", history_no_ah_result["market_comparison"]["m_open_unavailable_reason"])

    def test_staleness_needs_review_and_html_escapes_dynamic_text(self) -> None:
        eligible, excluded, stats = _validate_history(self.history, date.fromisoformat(AS_OF))
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "history.csv"
            self.history.to_csv(source, index=False)
            with patch("matchquant.single_match.run_factor_lab", side_effect=_fake_lab), patch(
                "matchquant.single_match.fit_and_select_factor_system", side_effect=_fake_fit
            ):
                result = _build_result(
                    input_path=source,
                    eligible=eligible,
                    excluded_count=excluded,
                    history_stats=stats,
                    fixture=_fixture(),
                    as_of=date.fromisoformat(AS_OF),
                    mode="replay",
                    max_data_age_days=1,
                    alias_source=None,
                    market=None,
                    market_source=None,
                )
            self.assertEqual(result["status"], "needs_review")
            self.assertEqual(result["temporal"]["forecast_horizon_days"], 90)
            rendered = copy.deepcopy(result)
            rendered["fixture"]["home_team"] = "<script>alert(1)</script>"
            rendered["limitations"].append("Source <img src=x onerror=alert(1)>")
            rendered["market_comparison"] = {
                "provided": True,
                "input": {
                    "source": "Book <img src=x onerror=alert(1)>",
                    "observed_at": AS_OF,
                    "probabilities": {"home": 0.45, "draw": 0.28, "away": 0.27},
                },
                "track_a_minus_market_percentage_points": {"home": 1.0, "draw": 2.0, "away": -3.0},
                "m_open_available": False,
                "m_open_unavailable_reason": "AH <missing>",
            }
            document = render_report_html(rendered)
            self.assertNotIn("<script>alert(1)</script>", document)
            self.assertIn("&lt;script&gt;alert(1)&lt;/script&gt;", document)
            self.assertNotIn("<img src=x", document)
            self.assertIn("Book &lt;img src=x onerror=alert(1)&gt;", document)
            markdown = render_report_markdown(rendered)
            self.assertIn("模型判断", markdown)
            self.assertIn("需要复核", markdown)


class SingleMatchEndToEndTests(unittest.TestCase):
    def test_real_model_fit_writes_consistent_three_file_report(self) -> None:
        history = synthetic_rich_history()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            input_csv = root / "synthetic_rich_history.csv"
            output_root = root / "runs"
            history.to_csv(input_csv, index=False)
            result, paths = run_single_match_report(
                input_path=input_csv,
                as_of=AS_OF,
                output_root=output_root,
                mode="replay",
                home="Club A",
                away="Club B",
                fixture_date=FIXTURE_DATE,
                season=FIXTURE_SEASON,
            )
            self.assertEqual(set(paths), {"json", "markdown", "html"})
            for artifact in paths.values():
                self.assertTrue(Path(artifact).is_file())
            saved = json.loads(Path(paths["json"]).read_text(encoding="utf-8"))
            self.assertEqual(saved["schema_version"], "single-match-report-v1")
            self.assertEqual(saved["fixture"]["date"], FIXTURE_DATE)
            self.assertEqual(saved["temporal"]["as_of"], AS_OF)
            self.assertEqual(saved["model"]["selection"]["partitions"]["training_row_count"], 760)
            self.assertEqual(saved["model"]["selection"]["partitions"]["validation_row_count"], 380)
            self.assertEqual(saved["model"]["selection"]["partitions"]["validation_season"], "2023-24")
            p = saved["predictions"]["track_a"]
            probs = [p[key] for key in ("home", "draw", "away")]
            self.assertTrue(all(math.isfinite(value) and value >= 0 for value in probs))
            self.assertAlmostEqual(sum(probs), 1.0, delta=1e-9)
            self.assertTrue(math.isfinite(saved["predictions"]["expected_goals"]["home"]))
            self.assertTrue(0 <= saved["predictions"]["probability_total_goals_over_2_5"] <= 1)
            markdown = Path(paths["markdown"]).read_text(encoding="utf-8")
            html = Path(paths["html"]).read_text(encoding="utf-8")
            self.assertIn("未提供赛前赔率，未计算市场对照", markdown)
            self.assertIn("最近五场可用历史", markdown)
            self.assertIn("模型估计概率", markdown)
            self.assertIn("<!doctype html>", html.lower())
            self.assertNotIn("<script", html.lower())
            self.assertNotIn("https://", html.lower())
            self.assertIn("Track A 基本面", html)
            self.assertEqual(result["artifacts"], paths)
            self.assertNotIn("home_goals", saved["fixture"])
            self.assertIn("validation tuning/selection evidence", saved["model"]["selection"]["partitions"]["selection_evidence_label"])
            if saved["predictions"]["component_weights"]["ordered_logit"] == 0:
                self.assertFalse(saved["features"]["attribution"]["available"])


if __name__ == "__main__":
    unittest.main()
