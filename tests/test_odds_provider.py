"""Tests for odds_provider and adapter live odds integration."""

import io
import json
import os
import tempfile
import unittest
from datetime import date
from pathlib import Path
from unittest.mock import MagicMock, patch
import urllib.error

from matchquant.adapter import execute_adapter_request
from matchquant.odds_provider import (
    OddsProviderError,
    fetch_epl_odds,
    get_odds_api_key,
    normalize_team_for_matching,
    snapshot_to_market_input,
)
from matchquant.profile import UserProfile
from matchquant.single_match import MARKET_ALLOWED_FIELDS, _load_market_input


SAMPLE_ODDS_API_FIXTURE = {
    "id": "epl_test_fixture_001",
    "sport_key": "soccer_epl",
    "sport_title": "EPL",
    "commence_time": "2026-10-31T12:30:00Z",
    "home_team": "Chelsea",
    "away_team": "Manchester United",
    "bookmakers": [
        {
            "key": "sport888",
            "title": "888sport",
            "last_update": "2026-10-27T02:10:54Z",
            "markets": [
                {
                    "key": "h2h",
                    "last_update": "2026-10-27T02:10:54Z",
                    "outcomes": [
                        {"name": "Chelsea", "price": 1.83},
                        {"name": "Manchester United", "price": 3.80},
                        {"name": "Draw", "price": 3.75},
                    ],
                }
            ],
        },
        {
            "key": "matchbook",
            "title": "Matchbook",
            "last_update": "2026-10-27T02:11:16Z",
            "markets": [
                {
                    "key": "h2h",
                    "last_update": "2026-10-27T02:11:16Z",
                    "outcomes": [
                        {"name": "Chelsea", "price": 1.88},
                        {"name": "Manchester United", "price": 4.30},
                        {"name": "Draw", "price": 4.10},
                    ],
                },
                {
                    "key": "spreads",
                    "last_update": "2026-10-27T02:11:16Z",
                    "outcomes": [
                        {"name": "Chelsea", "price": 2.10, "point": -0.75},
                        {"name": "Manchester United", "price": 1.87, "point": 0.75},
                    ],
                },
            ],
        },
    ],
}


class OddsProviderTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.output_root = Path(self.temp_dir.name)
        self.profile = UserProfile(
            data_root=str(self.output_root / "data"),
            report_root=str(self.output_root / "runs"),
        )

    def tearDown(self):
        self.temp_dir.cleanup()

    def _mock_response(self, data, status=200):
        body = json.dumps(data).encode("utf-8")
        resp = MagicMock()
        resp.status = status
        resp.read.return_value = body
        resp.__enter__.return_value = resp
        resp.__exit__.return_value = None
        return resp

    @patch("matchquant.odds_provider.urllib.request.urlopen")
    def test_fetch_epl_odds_normal_flow(self, mock_urlopen):
        """Test normal fetch, bookmaker selection, odds parsing, and snapshot persistence."""
        mock_urlopen.return_value = self._mock_response([SAMPLE_ODDS_API_FIXTURE])

        snapshot = fetch_epl_odds(
            home_team="Chelsea",
            away_team="Manchester United",
            fixture_date="2026-10-31",
            api_key="test_mock_key",
            output_dir=self.output_root / "odds_snapshots",
            profile=self.profile,
        )

        self.assertEqual(snapshot["home_team"], "Chelsea")
        self.assertEqual(snapshot["away_team"], "Man United")
        self.assertEqual(snapshot["date"], "2026-10-31")
        self.assertEqual(snapshot["odds_open_h"], 1.83)
        self.assertEqual(snapshot["odds_open_d"], 3.75)
        self.assertEqual(snapshot["odds_open_a"], 3.80)
        # Spreads line from matchbook in the same fixture
        self.assertEqual(snapshot["mkt_ah_line"], -0.75)
        self.assertIn("The Odds API", snapshot["source"])
        self.assertEqual(snapshot["bookmaker"], "sport888")

        # Snapshot file existence and immutability
        snapshot_path = Path(snapshot["snapshot_path"])
        self.assertTrue(snapshot_path.is_file())
        file_data = json.loads(snapshot_path.read_text(encoding="utf-8"))
        self.assertEqual(file_data["odds_open_h"], 1.83)
        self.assertEqual(file_data["contract_note"], snapshot["contract_note"])

    @patch("matchquant.odds_provider.urllib.request.urlopen")
    def test_team_alias_matching_and_chinese_resolution(self, mock_urlopen):
        """Test resolving Chinese aliases ('切尔西', '曼联') to EPL fixtures."""
        mock_urlopen.return_value = self._mock_response([SAMPLE_ODDS_API_FIXTURE])

        snapshot = fetch_epl_odds(
            home_team="切尔西",
            away_team="曼联",
            fixture_date="2026-10-31",
            api_key="test_mock_key",
            output_dir=self.output_root / "odds_snapshots",
            profile=self.profile,
        )

        self.assertEqual(snapshot["home_team"], "Chelsea")
        self.assertEqual(snapshot["away_team"], "Man United")
        self.assertEqual(snapshot["odds_open_h"], 1.83)

    def test_missing_api_key_raises_error(self):
        """Test that missing API key raises clean OddsProviderError."""
        with patch.dict(os.environ, {}, clear=True):
            with patch("matchquant.odds_provider.get_odds_api_key", return_value=None):
                with self.assertRaises(OddsProviderError) as ctx:
                    fetch_epl_odds(
                        home_team="Chelsea",
                        away_team="Man United",
                        fixture_date="2026-10-31",
                        api_key=None,
                        profile=self.profile,
                    )
                self.assertIn("THE_ODDS_API_KEY", str(ctx.exception))

    @patch("matchquant.odds_provider.urllib.request.urlopen")
    def test_http_429_rate_limit_handling(self, mock_urlopen):
        """Test HTTP 429 rate limit is captured and raises clear OddsProviderError."""
        err = urllib.error.HTTPError(
            url="http://test",
            code=429,
            msg="Too Many Requests",
            hdrs={},
            fp=io.BytesIO(b"Rate limit exceeded"),
        )
        mock_urlopen.side_effect = err

        with self.assertRaises(OddsProviderError) as ctx:
            fetch_epl_odds(
                home_team="Chelsea",
                away_team="Man United",
                fixture_date="2026-10-31",
                api_key="test_mock_key",
                profile=self.profile,
                max_retries=0,
            )
        self.assertIn("429", str(ctx.exception))

    @patch("matchquant.odds_provider.urllib.request.urlopen")
    def test_handicap_sign_convention_coverage(self, mock_urlopen):
        """Test handicap sign convention (-0.25, 0.0, +0.5)."""
        test_cases = [-0.5, -0.25, 0.0, 0.25, 0.5]
        for expected_line in test_cases:
            fixture = {
                "id": f"fixture_{expected_line}",
                "commence_time": "2026-10-31T15:00:00Z",
                "home_team": "Chelsea",
                "away_team": "Arsenal",
                "bookmakers": [
                    {
                        "key": "pinnacle",
                        "title": "Pinnacle",
                        "markets": [
                            {
                                "key": "h2h",
                                "outcomes": [
                                    {"name": "Chelsea", "price": 2.10},
                                    {"name": "Arsenal", "price": 3.40},
                                    {"name": "Draw", "price": 3.30},
                                ],
                            },
                            {
                                "key": "spreads",
                                "outcomes": [
                                    {"name": "Chelsea", "price": 1.95, "point": expected_line},
                                    {"name": "Arsenal", "price": 1.95, "point": -expected_line},
                                ],
                            },
                        ],
                    }
                ],
            }
            mock_urlopen.return_value = self._mock_response([fixture])
            snap = fetch_epl_odds(
                home_team="Chelsea",
                away_team="Arsenal",
                fixture_date="2026-10-31",
                api_key="test_mock_key",
                output_dir=self.output_root / "odds_snapshots",
                profile=self.profile,
            )
            self.assertEqual(snap["mkt_ah_line"], expected_line)

    def test_snapshot_to_market_input_strict_contract(self):
        """Test snapshot conversion adheres strictly to MARKET_ALLOWED_FIELDS and validates."""
        snapshot = {
            "schema_version": "matchquant-odds-snapshot-v1",
            "fixture_id": "2026-10-31_Chelsea_Man_United",
            "home_team": "Chelsea",
            "away_team": "Man United",
            "date": "2026-10-31",
            "observed_at_utc": "2026-10-27T02:10:54Z",
            "observed_at": "2026-10-27",
            "source": "The Odds API (888sport)",
            "bookmaker": "sport888",
            "odds_open_h": 1.83,
            "odds_open_d": 3.75,
            "odds_open_a": 3.80,
            "mkt_ah_line": -0.75,
        }

        market_input = snapshot_to_market_input(snapshot)

        # Field set check
        self.assertTrue(set(market_input).issubset(MARKET_ALLOWED_FIELDS))
        self.assertIn("home_team", market_input)
        self.assertIn("odds_open_h", market_input)
        self.assertEqual(market_input["mkt_ah_line"], -0.75)

        # Verify against single_match._load_market_input
        market_file = self.output_root / "market_test.json"
        market_file.write_text(json.dumps(market_input), encoding="utf-8")

        loaded_market, source_meta = _load_market_input(
            market_file,
            fixture={"home_team": "Chelsea", "away_team": "Man United", "date": "2026-10-31"},
            as_of=date(2026, 10, 28),
            team_names=["Chelsea", "Man United"],
            aliases=None,
        )
        self.assertIsNotNone(loaded_market)
        self.assertEqual(loaded_market["odds_open_h"], 1.83)
        self.assertAlmostEqual(
            sum(loaded_market["probabilities"].values()), 1.0, places=4
        )

    @patch("matchquant.adapter.fetch_epl_odds")
    def test_adapter_odds_action(self, mock_fetch):
        """Test adapter action='odds' executes successfully."""
        mock_fetch.return_value = {
            "schema_version": "matchquant-odds-snapshot-v1",
            "fixture_id": "2026-10-31_Chelsea_Man_United",
            "home_team": "Chelsea",
            "away_team": "Man United",
            "date": "2026-10-31",
            "observed_at_utc": "2026-10-27T02:10:54Z",
            "observed_at": "2026-10-27",
            "source": "The Odds API (888sport)",
            "bookmaker": "sport888",
            "bookmaker_title": "888sport",
            "odds_open_h": 1.83,
            "odds_open_d": 3.75,
            "odds_open_a": 3.80,
            "mkt_ah_line": -0.75,
            "snapshot_path": str(self.output_root / "snap.json"),
        }
        # Create the fake snapshot path
        Path(self.output_root / "snap.json").write_text("{}", encoding="utf-8")

        req = {
            "action": "odds",
            "home": "切尔西",
            "away": "曼联",
            "date": "2026-10-31",
            "profile_path": str(self.output_root / "profile.json"),
        }
        res = execute_adapter_request(req)
        self.assertEqual(res["status"], "ok")
        self.assertEqual(res["action"], "odds")
        self.assertIn("market_input", res)
        self.assertEqual(res["market_input"]["odds_open_h"], 1.83)
        self.assertEqual(res["summary"]["bookmaker"], "888sport")

    @patch("matchquant.adapter.fetch_epl_odds")
    @patch("matchquant.adapter.run_single_match_report")
    @patch("matchquant.adapter.check_data_readiness")
    def test_adapter_analyze_with_live_odds_and_default_as_of(
        self, mock_readiness, mock_run_single, mock_fetch
    ):
        """Test analyze with use_live_odds=True defaults as_of to observed_at + 1 day and passes market_input."""
        mock_fetch.return_value = {
            "schema_version": "matchquant-odds-snapshot-v1",
            "fixture_id": "2026-10-31_Chelsea_Man_United",
            "home_team": "Chelsea",
            "away_team": "Man United",
            "date": "2026-10-31",
            "observed_at_utc": "2026-10-27T02:10:54Z",
            "observed_at": "2026-10-27",
            "source": "The Odds API (888sport)",
            "bookmaker": "sport888",
            "bookmaker_title": "888sport",
            "odds_open_h": 1.83,
            "odds_open_d": 3.75,
            "odds_open_a": 3.80,
            "mkt_ah_line": -0.75,
            "snapshot_path": str(self.output_root / "snap.json"),
        }
        Path(self.output_root / "snap.json").write_text("{}", encoding="utf-8")

        fake_rich = self.output_root / "rich.csv"
        fake_rich.write_text("dummy", encoding="utf-8")
        mock_readiness.return_value = {
            "ready": True,
            "status": "ok",
            "rich_path": str(fake_rich),
        }
        fake_json_path = self.output_root / "report.json"
        fake_json_path.write_text("{}", encoding="utf-8")
        mock_run_single.return_value = (
            {
                "status": "ok",
                "run_id": "test_run",
                "predictions": {"track_a": {"home": 0.5, "draw": 0.25, "away": 0.25}},
                "warnings": [],
            },
            {
                "json": str(fake_json_path),
                "markdown": str(self.output_root / "report.md"),
                "html": str(self.output_root / "report.html"),
            },
        )

        req = {
            "action": "analyze",
            "home": "切尔西",
            "away": "曼联",
            "date": "2026-10-31",
            "use_live_odds": True,
            "report_root": str(self.output_root),
        }
        res = execute_adapter_request(req)
        self.assertEqual(res["status"], "ok")
        self.assertIn("live_odds", res)
        self.assertEqual(res["live_odds"]["status"], "ok")
        # Verify as_of was defaulted to 2026-10-28 (observed_at 2026-10-27 + 1 day)
        self.assertEqual(res["temporal"]["as_of"], "2026-10-28")
        # Verify call to run_single_match_report received market_input_path
        call_kwargs = mock_run_single.call_args.kwargs
        self.assertIsNotNone(call_kwargs["market_input_path"])

    @patch("matchquant.adapter.fetch_epl_odds")
    @patch("matchquant.adapter.run_single_match_report")
    @patch("matchquant.adapter.check_data_readiness")
    def test_adapter_analyze_live_odds_graceful_degradation_on_failure(
        self, mock_readiness, mock_run_single, mock_fetch
    ):
        """Test that if live odds fetching fails, adapter degrades to Track A without fatal crash."""
        mock_fetch.side_effect = OddsProviderError("The Odds API network timeout")

        fake_rich = self.output_root / "rich.csv"
        fake_rich.write_text("dummy", encoding="utf-8")
        mock_readiness.return_value = {
            "ready": True,
            "status": "ok",
            "rich_path": str(fake_rich),
        }
        fake_json_path = self.output_root / "report.json"
        fake_json_path.write_text("{}", encoding="utf-8")
        mock_run_single.return_value = (
            {
                "status": "ok",
                "run_id": "test_run",
                "predictions": {"track_a": {"home": 0.5, "draw": 0.25, "away": 0.25}},
                "warnings": [],
            },
            {
                "json": str(fake_json_path),
                "markdown": str(self.output_root / "report.md"),
                "html": str(self.output_root / "report.html"),
            },
        )

        req = {
            "action": "analyze",
            "home": "切尔西",
            "away": "曼联",
            "date": "2026-10-31",
            "use_live_odds": True,
            "report_root": str(self.output_root),
        }
        res = execute_adapter_request(req)
        self.assertEqual(res["status"], "ok")
        self.assertIn("live_odds", res)
        self.assertEqual(res["live_odds"]["status"], "failed")
        # Notice warning recorded
        self.assertTrue(any("实时盘口获取失败" in w for w in res["warnings"]))
        # Single match was run with market_input_path=None
        call_kwargs = mock_run_single.call_args.kwargs
        self.assertIsNone(call_kwargs["market_input_path"])

    @patch("matchquant.adapter.fetch_epl_odds")
    @patch("matchquant.adapter.run_single_match_report")
    @patch("matchquant.adapter.check_data_readiness")
    def test_adapter_analyze_look_ahead_bias_degradation(
        self, mock_readiness, mock_run_single, mock_fetch
    ):
        """Test that if user-specified as_of <= observed_at, market input is rejected to prevent look-ahead bias."""
        mock_fetch.return_value = {
            "schema_version": "matchquant-odds-snapshot-v1",
            "fixture_id": "2026-10-31_Chelsea_Man_United",
            "home_team": "Chelsea",
            "away_team": "Man United",
            "date": "2026-10-31",
            "observed_at_utc": "2026-10-27T02:10:54Z",
            "observed_at": "2026-10-27",
            "source": "The Odds API (888sport)",
            "bookmaker": "sport888",
            "bookmaker_title": "888sport",
            "odds_open_h": 1.83,
            "odds_open_d": 3.75,
            "odds_open_a": 3.80,
            "mkt_ah_line": -0.75,
            "snapshot_path": str(self.output_root / "snap.json"),
        }
        Path(self.output_root / "snap.json").write_text("{}", encoding="utf-8")

        fake_rich = self.output_root / "rich.csv"
        fake_rich.write_text("dummy", encoding="utf-8")
        mock_readiness.return_value = {
            "ready": True,
            "status": "ok",
            "rich_path": str(fake_rich),
        }
        fake_json_path = self.output_root / "report.json"
        fake_json_path.write_text("{}", encoding="utf-8")
        mock_run_single.return_value = (
            {
                "status": "ok",
                "run_id": "test_run",
                "predictions": {"track_a": {"home": 0.5, "draw": 0.25, "away": 0.25}},
                "warnings": [],
            },
            {
                "json": str(fake_json_path),
                "markdown": str(self.output_root / "report.md"),
                "html": str(self.output_root / "report.html"),
            },
        )

        # user specifies as_of = 2026-10-26 (earlier than snapshot observed_at 2026-10-27)
        req = {
            "action": "analyze",
            "home": "切尔西",
            "away": "曼联",
            "date": "2026-10-31",
            "as_of": "2026-10-26",
            "use_live_odds": True,
            "report_root": str(self.output_root),
        }
        res = execute_adapter_request(req)
        self.assertEqual(res["status"], "ok")
        self.assertEqual(res["live_odds"]["status"], "downgraded")
        self.assertTrue(any("防穿越红线" in w for w in res["warnings"]))
        call_kwargs = mock_run_single.call_args.kwargs
        self.assertIsNone(call_kwargs["market_input_path"])


if __name__ == "__main__":
    unittest.main()
