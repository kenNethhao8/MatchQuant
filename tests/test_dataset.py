"""Tests for Layer 2: Dataset versioning, rich conversion, and readiness (T5, T10, T11)."""

import json
import tempfile
import unittest
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from matchquant.dataset import (
    build_dataset_from_manifest,
    check_data_readiness,
    load_current_dataset_pointer,
)
from matchquant.profile import UserProfile
from matchquant.rich_data import convert_raw_epl_season


class DatasetTests(unittest.TestCase):
    def test_t5_missing_technical_stats_and_market_quotes_stay_missing(self) -> None:
        """T5: Missing technical/market fields remain missing (NaN).

        Zero observed shots is distinct from unavailable shots. Missing AH does not become 0.0.
        """
        # CSV with only required score/date/team fields, technical stats completely omitted
        raw_df = pd.DataFrame([
            {
                "Date": "15/08/2024",
                "HomeTeam": "Arsenal",
                "AwayTeam": "Chelsea",
                "FTHG": "2",
                "FTAG": "1",
                "FTR": "H",
                # Note: HS, AS, HST, AST, HC, AC, AHh etc. are completely absent!
            }
        ])
        rich = convert_raw_epl_season(raw_df, "2024-25")

        # Verify missing shots & corners remain NaN rather than 0
        self.assertTrue(pd.isna(rich.loc[0, "home_shots"]))
        self.assertTrue(pd.isna(rich.loc[0, "away_shots"]))
        self.assertTrue(pd.isna(rich.loc[0, "home_sot"]))
        self.assertTrue(pd.isna(rich.loc[0, "home_corners"]))
        self.assertTrue(pd.isna(rich.loc[0, "mkt_ah_line"]))

        # Now test an explicit 0 observed shots in the raw data
        raw_with_zero = pd.DataFrame([
            {
                "Date": "15/08/2024",
                "HomeTeam": "Arsenal",
                "AwayTeam": "Chelsea",
                "FTHG": "0",
                "FTAG": "0",
                "FTR": "D",
                "HS": "0",  # explicitly observed 0 shots
                "AS": "5",
                "HC": "0",  # explicitly observed 0 corners
            }
        ])
        rich_zero = convert_raw_epl_season(raw_with_zero, "2024-25")
        self.assertEqual(rich_zero.loc[0, "home_shots"], 0.0)
        self.assertEqual(rich_zero.loc[0, "away_shots"], 5.0)
        self.assertEqual(rich_zero.loc[0, "home_corners"], 0.0)
        # Missing fields in that same row remain NaN
        self.assertTrue(pd.isna(rich_zero.loc[0, "home_sot"]))
        self.assertTrue(pd.isna(rich_zero.loc[0, "mkt_ah_line"]))

    def test_t10_source_check_age_vs_match_data_age_and_stale_fallback(self) -> None:
        """T10: Source-check age and match-data age are distinct.

        Off-season reported honestly. Refresh failure / stale data without fallback fails;
        with fallback, status is needs_review.
        """
        with tempfile.TemporaryDirectory() as tmp_dir:
            data_root = Path(tmp_dir)
            processed_dir = data_root / "processed"
            processed_dir.mkdir(parents=True, exist_ok=True)

            can_file = processed_dir / "epl_canonical.csv"
            rich_file = processed_dir / "epl_rich_canonical.csv"
            can_file.write_text("match_id,date,season,home_team,away_team,home_goals,away_goals\n1,2024-05-19,2023-24,Arsenal,Everton,2,1\n", encoding="utf-8")
            rich_file.write_text("match_id,date,season,home_team,away_team,home_goals,away_goals\n1,2024-05-19,2023-24,Arsenal,Everton,2,1\n", encoding="utf-8")

            # Simulate pointer where match data cutoff is 2024-05-19, but checked today
            pointer = {
                "schema_version": "m2.1-current-pointer-v1",
                "dataset_version": "test-v1",
                "updated_at": "2024-05-20T10:00:00Z",
                "source_checked_at": "2024-11-01T12:00:00Z",
                "canonical": str(can_file),
                "rich": str(rich_file),
                "data_cutoff": "2024-05-19",
                "canonical_rows": 1,
            }
            (processed_dir / "epl_current.json").write_text(json.dumps(pointer), encoding="utf-8")

            # Case A: As of June 15, 2024 (Summer Off-season)
            readiness_summer = check_data_readiness(data_root, as_of="2024-06-15")
            self.assertTrue(readiness_summer["is_off_season"])
            self.assertTrue(readiness_summer["ready"])
            self.assertIn("off-season", " ".join(readiness_summer["warnings"]))

            # Case B: As of Nov 15, 2024 (In-season, data is 180 days old -> Stale!)
            # Default policy: allow_stale_fallback=False
            readiness_stale_blocked = check_data_readiness(data_root, as_of="2024-11-15", allow_stale_fallback=False)
            self.assertTrue(readiness_stale_blocked["is_stale"])
            self.assertFalse(readiness_stale_blocked["ready"])
            self.assertEqual(readiness_stale_blocked["status"], "stale_review_required")
            self.assertGreater(readiness_stale_blocked["data_age_days"], 150)

            # Case C: Same stale date, but user explicitly authorizes stale fallback
            readiness_fallback = check_data_readiness(data_root, as_of="2024-11-15", allow_stale_fallback=True)
            self.assertTrue(readiness_fallback["is_stale"])
            self.assertTrue(readiness_fallback["ready"])
            self.assertEqual(readiness_fallback["status"], "needs_review")
            self.assertTrue(readiness_fallback["stale_fallback_used"])

    def test_t11_manifest_bound_build_creates_valid_pointer_and_metadata(self) -> None:
        """T11: build_dataset_from_manifest strictly binds to explicit snapshots."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            data_root = Path(tmp_dir)
            raw_dir = data_root / "raw"
            raw_dir.mkdir(parents=True, exist_ok=True)

            from tests.test_sync import _make_epl_season_csv
            s_csv = _make_epl_season_csv("2023-24", match_count=380)
            snap_path = raw_dir / "E0_2023-24.csv"
            snap_path.write_text(s_csv, encoding="utf-8")

            manifest = build_dataset_from_manifest(
                season_snapshots={"2023-24": snap_path},
                data_root=data_root,
            )
            self.assertEqual(manifest["canonical"]["rows"], 380)
            self.assertEqual(manifest["rich"]["rows"], 380)
            self.assertEqual(manifest["data_cutoff"], "2024-05-25")

            pointer = load_current_dataset_pointer(data_root)
            self.assertIsNotNone(pointer)
            self.assertEqual(pointer["canonical_rows"], 380)
            self.assertEqual(pointer["data_cutoff"], "2024-05-25")


if __name__ == "__main__":
    unittest.main()
