from __future__ import annotations

import json
import math
import tempfile
import unittest
from datetime import date, timedelta
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from matchquant.single_match import (
    SingleMatchError,
    extract_market_odds_timeline,
    run_single_match_report,
)
from tests.test_single_match import AS_OF, FIXTURE_DATE, FIXTURE_SEASON, SEASONS, TEAM_NAMES, synthetic_rich_history


class MarketAnalysisStage2Tests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.csv_path = self.root / "history.csv"
        synthetic_rich_history().to_csv(self.csv_path, index=False)

        self.home = TEAM_NAMES[0]
        self.away = TEAM_NAMES[1]

        # Standard market input
        self.market_payload = {
            "home_team": self.home,
            "away_team": self.away,
            "date": FIXTURE_DATE,
            "observed_at": "2024-05-19",  # strictly before AS_OF (2024-05-20)
            "source": "Mock Liquid Bookmaker",
            "odds_open_h": 2.10,
            "odds_open_d": 3.40,
            "odds_open_a": 3.60,
            "mkt_ah_line": -0.25,
        }
        self.market_path = self.root / "market.json"
        self.market_path.write_text(json.dumps(self.market_payload), encoding="utf-8")

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_track_a_strict_isolation_bit_exact(self):
        """Red Line 1: Track A probabilities must be 100% bit-exact with and without market input."""
        res_no_market, _ = run_single_match_report(
            input_path=self.csv_path,
            as_of=AS_OF,
            output_root=self.root / "run_no_mkt",
            home=self.home,
            away=self.away,
            fixture_date=FIXTURE_DATE,
            season=FIXTURE_SEASON,
            market_input_path=None,
        )

        res_with_market, _ = run_single_match_report(
            input_path=self.csv_path,
            as_of=AS_OF,
            output_root=self.root / "run_with_mkt",
            home=self.home,
            away=self.away,
            fixture_date=FIXTURE_DATE,
            season=FIXTURE_SEASON,
            market_input_path=self.market_path,
        )

        p_no_mkt = res_no_market["predictions"]["track_a"]
        p_with_mkt = res_with_market["predictions"]["track_a"]

        for outcome in ("home", "draw", "away"):
            diff = abs(p_no_mkt[outcome] - p_with_mkt[outcome])
            self.assertEqual(
                diff,
                0.0,
                f"Track A isolation violation! Outcome '{outcome}' differs: no_mkt={p_no_mkt[outcome]}, with_mkt={p_with_mkt[outcome]}",
            )

        # Expected goals and component predictions must also remain strictly identical
        self.assertEqual(
            res_no_market["predictions"]["expected_goals"],
            res_with_market["predictions"]["expected_goals"],
        )
        self.assertEqual(
            res_no_market["predictions"]["ordered_logit"],
            res_with_market["predictions"]["ordered_logit"],
        )
        self.assertEqual(
            res_no_market["predictions"]["dixon_coles"],
            res_with_market["predictions"]["dixon_coles"],
        )

    def test_market_devig_and_value_edges(self):
        """Test power devigging probabilities and Value Edge (P * Odds - 1) calculation."""
        res, _ = run_single_match_report(
            input_path=self.csv_path,
            as_of=AS_OF,
            output_root=self.root / "run_edges",
            home=self.home,
            away=self.away,
            fixture_date=FIXTURE_DATE,
            season=FIXTURE_SEASON,
            market_input_path=self.market_path,
        )

        mkt_comp = res["market_comparison"]
        self.assertTrue(mkt_comp["provided"])
        self.assertIn("market_open_probabilities", mkt_comp)

        devig = mkt_comp["market_open_probabilities"]
        prob_sum = devig["home"] + devig["draw"] + devig["away"]
        self.assertAlmostEqual(prob_sum, 1.0, places=6)

        edges = mkt_comp["value_edges"]
        self.assertIsNotNone(edges)
        p_track_a = res["predictions"]["track_a"]
        expected_edge_h = round(p_track_a["home"] * self.market_payload["odds_open_h"] - 1.0, 4)
        expected_edge_d = round(p_track_a["draw"] * self.market_payload["odds_open_d"] - 1.0, 4)
        expected_edge_a = round(p_track_a["away"] * self.market_payload["odds_open_a"] - 1.0, 4)

        self.assertEqual(edges["home"], expected_edge_h)
        self.assertEqual(edges["draw"], expected_edge_d)
        self.assertEqual(edges["away"], expected_edge_a)
        self.assertIn("disclaimer", edges)
        self.assertIn("不构成", edges["disclaimer"])

    def test_divergence_alerts_configurable_threshold(self):
        """Test divergence alerts triggering logic and configurable threshold."""
        # Force a market with extreme odds to guarantee divergence
        divergent_payload = dict(self.market_payload)
        divergent_payload["odds_open_h"] = 1.10
        divergent_payload["odds_open_d"] = 10.00
        divergent_payload["odds_open_a"] = 25.00
        div_path = self.root / "divergent_market.json"
        div_path.write_text(json.dumps(divergent_payload), encoding="utf-8")

        res_div, _ = run_single_match_report(
            input_path=self.csv_path,
            as_of=AS_OF,
            output_root=self.root / "run_div",
            home=self.home,
            away=self.away,
            fixture_date=FIXTURE_DATE,
            season=FIXTURE_SEASON,
            market_input_path=div_path,
            divergence_threshold_pct_points=5.0,
        )

        alerts = res_div["market_comparison"]["divergence_alerts"]
        self.assertTrue(alerts["has_divergence"])
        self.assertEqual(alerts["threshold_percentage_points"], 5.0)
        self.assertTrue(len(alerts["alerts"]) > 0)
        first_alert = alerts["alerts"][0]
        self.assertIn("outcome", first_alert)
        self.assertIn("diff_percentage_points", first_alert)
        self.assertIn("note", first_alert)

        # When threshold is set extremely high (e.g. 99.0%), has_divergence must be False
        res_no_div, _ = run_single_match_report(
            input_path=self.csv_path,
            as_of=AS_OF,
            output_root=self.root / "run_no_div",
            home=self.home,
            away=self.away,
            fixture_date=FIXTURE_DATE,
            season=FIXTURE_SEASON,
            market_input_path=div_path,
            divergence_threshold_pct_points=99.0,
        )
        alerts_high = res_no_div["market_comparison"]["divergence_alerts"]
        self.assertFalse(alerts_high["has_divergence"])
        self.assertEqual(len(alerts_high["alerts"]), 0)

    def test_extract_timeline_zero_one_and_multi_snapshots(self):
        """Test Decision Gate B: timeline extraction across 0, 1, and >=2 snapshots."""
        fixture_key = f"{FIXTURE_DATE}_{self.home}_{self.away}".replace(" ", "_")
        snapshot_dir = self.root / "odds_snapshots" / fixture_key
        snapshot_dir.mkdir(parents=True, exist_ok=True)

        fixture = {
            "date": FIXTURE_DATE,
            "home_team": self.home,
            "away_team": self.away,
            "fixture_id": f"single-{fixture_key}",
        }

        # Case 0: Empty snapshot dir
        t0 = extract_market_odds_timeline(
            fixture_id=fixture_key,
            report_root=self.root,
            fixture=fixture,
        )
        self.assertFalse(t0["available"])
        self.assertEqual(t0["snapshot_count"], 0)
        self.assertEqual(t0["status"], "no_snapshots")

        # Case 1: 1 snapshot
        snap1 = {
            "schema_version": "matchquant-odds-snapshot-v1",
            "fixture_id": fixture_key,
            "home_team": self.home,
            "away_team": self.away,
            "date": FIXTURE_DATE,
            "observed_at_utc": "2024-05-18T10:00:00Z",
            "observed_at": "2024-05-18",
            "source": "Bookmaker A",
            "odds_open_h": 2.20,
            "odds_open_d": 3.30,
            "odds_open_a": 3.50,
            "mkt_ah_line": -0.25,
        }
        (snapshot_dir / "20240518T100000Z.json").write_text(json.dumps(snap1), encoding="utf-8")

        t1 = extract_market_odds_timeline(
            fixture_id=fixture_key,
            report_root=self.root,
            fixture=fixture,
        )
        self.assertFalse(t1["available"])
        self.assertEqual(t1["snapshot_count"], 1)
        self.assertEqual(t1["status"], "single_snapshot_baseline")
        self.assertIn("暂无历史变动趋势", t1["message"])
        self.assertIsNone(t1["drift_summary"])

        # Case 2: >=2 snapshots (add 2 more snapshots in non-sorted filenames to test sorting)
        snap2 = dict(snap1)
        snap2["observed_at_utc"] = "2024-05-19T12:00:00Z"
        snap2["observed_at"] = "2024-05-19"
        snap2["odds_open_h"] = 1.95  # Steam move on home: 2.20 -> 1.95
        snap2["odds_open_d"] = 3.45
        snap2["odds_open_a"] = 4.10
        snap2["mkt_ah_line"] = -0.50
        (snapshot_dir / "20240519T120000Z.json").write_text(json.dumps(snap2), encoding="utf-8")

        snap_mid = dict(snap1)
        snap_mid["observed_at_utc"] = "2024-05-18T20:00:00Z"
        snap_mid["odds_open_h"] = 2.05
        (snapshot_dir / "20240518T200000Z.json").write_text(json.dumps(snap_mid), encoding="utf-8")

        t2 = extract_market_odds_timeline(
            fixture_id=fixture_key,
            report_root=self.root,
            fixture=fixture,
        )
        self.assertTrue(t2["available"])
        self.assertEqual(t2["snapshot_count"], 3)
        self.assertEqual(t2["status"], "multi_snapshot_trend")
        self.assertEqual(t2["first_observed_utc"], "2024-05-18T10:00:00Z")
        self.assertEqual(t2["latest_observed_utc"], "2024-05-19T12:00:00Z")

        drift = t2["drift_summary"]
        self.assertIsNotNone(drift)
        self.assertAlmostEqual(drift["odds_drift"]["home"], 1.95 - 2.20, places=4)
        self.assertAlmostEqual(drift["ah_line_drift"], -0.50 - (-0.25), places=4)
        self.assertIn("steam_description", drift)
        self.assertTrue(len(drift["steam_description"]) > 0)

    def test_lookahead_prevention_rejects_future_observed_at(self):
        """Red Line 2: observed_at >= as_of must be rejected to prevent look-ahead bias."""
        future_market = dict(self.market_payload)
        future_market["observed_at"] = AS_OF  # Same as as_of: violates strictly before rule
        bad_path = self.root / "future_market.json"
        bad_path.write_text(json.dumps(future_market), encoding="utf-8")

        with self.assertRaises(SingleMatchError) as ctx:
            run_single_match_report(
                input_path=self.csv_path,
                as_of=AS_OF,
                output_root=self.root / "run_future",
                home=self.home,
                away=self.away,
                fixture_date=FIXTURE_DATE,
                season=FIXTURE_SEASON,
                market_input_path=bad_path,
            )
        self.assertIn("strictly before", str(ctx.exception))

    def test_adapter_analyze_exposes_market_comparison(self):
        """Test adapter integration: execute_adapter_request exposes market_comparison."""
        from matchquant.adapter import execute_adapter_request
        from matchquant.profile import UserProfile

        # Setup mock profile pointing to self.root
        prof = UserProfile(
            data_root=str(self.root / "data"),
            report_root=str(self.root / "runs"),
        )
        prof_path = self.root / "profile.json"
        prof.save(prof_path)

        # Create rich data pointer in data_root
        processed_dir = self.root / "data" / "processed"
        processed_dir.mkdir(parents=True, exist_ok=True)
        rich_path = processed_dir / "epl_rich_canonical.csv"
        can_path = processed_dir / "epl_canonical.csv"
        self.csv_path.replace(rich_path)
        import shutil
        shutil.copyfile(rich_path, can_path)

        pointer = {
            "schema_version": "matchquant-pointer-v1",
            "dataset_version": "2024-test",
            "canonical_sha256": "dummy",
            "rich_sha256": "dummy",
            "latest_completed_match_date": "2024-05-19",
            "seasons": list(SEASONS),
            "completed_matches": 1140,
            "created_at": "2024-05-19T00:00:00Z",
            "canonical_path": str(can_path),
            "rich_path": str(rich_path),
        }
        pointer_file = processed_dir / "epl_current.json"
        pointer_file.write_text(json.dumps(pointer), encoding="utf-8")

        req = {
            "action": "analyze",
            "profile_path": str(prof_path),
            "home": self.home,
            "away": self.away,
            "date": FIXTURE_DATE,
            "season": FIXTURE_SEASON,
            "as_of": AS_OF,
            "market_input_path": str(self.market_path),
        }
        res = execute_adapter_request(req)
        self.assertIn(res["status"], ("ok", "needs_review"))
        self.assertIn("market_comparison", res)
        mkt = res["market_comparison"]
        self.assertTrue(mkt["provided"])
        self.assertIn("market_open_probabilities", mkt)
        self.assertIn("track_a_minus_market_percentage_points", mkt)
        self.assertIn("value_edges", mkt)
        self.assertIn("divergence_alerts", mkt)
        self.assertIn("timeline_analysis", mkt)
