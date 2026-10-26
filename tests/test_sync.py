"""Tests for Layer 1: Source synchronization (T1, T2, T3, T4)."""

import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from matchquant.dataset import load_current_dataset_pointer
from matchquant.epl import stable_match_id
from matchquant.io import file_sha256
from matchquant.sync import sync_epl


def _make_epl_season_csv(season: str, match_count: int = 380, pending_count: int = 0) -> str:
    """Generate minimal valid Football-Data E0 CSV content."""
    teams = [
        "Arsenal", "Aston Villa", "Bournemouth", "Brentford", "Brighton",
        "Chelsea", "Crystal Palace", "Everton", "Fulham", "Ipswich",
        "Leicester", "Liverpool", "Man City", "Man United", "Newcastle",
        "Nottingham Forest", "Southampton", "Tottenham", "West Ham", "Wolves"
    ]
    rows = []
    # Generate round robin fixtures
    pairings = []
    for h in teams:
        for a in teams:
            if h != a:
                pairings.append((h, a))

    months = [8, 9, 10, 11, 12, 1, 2, 3, 4, 5]
    start_year = int(season[:4])

    for idx, (h, a) in enumerate(pairings[:match_count]):
        # Match date
        m = months[(idx * len(months)) // max(1, match_count)]
        year = start_year if m >= 8 else start_year + 1
        day = (idx % 25) + 1
        date_str = f"{day:02d}/{m:02d}/{year}"
        rows.append(f"{date_str},{h},{a},1,0,H,Unknown,1,0,H,12,10,4,3,6,4,10,11,1,2,0,0,1.80,3.60,4.20")

    for idx in range(pending_count):
        # Pending match with blank scores
        p_idx = match_count + idx
        h = teams[p_idx % len(teams)]
        a = teams[(p_idx + 1) % len(teams)]
        date_str = f"20/05/{start_year + 1}"
        rows.append(f"{date_str},{h},{a},,,,Unknown,,,,,,,,,,,,,,,,1.90,3.50,4.00")

    header = "Date,HomeTeam,AwayTeam,FTHG,FTAG,FTR,Referee,HTHG,HTAG,HTR,HS,AS,HST,AST,HC,AC,HF,AF,HY,AY,HR,AR,AvgH,AvgD,AvgA\n"
    return header + "\n".join(rows) + "\n"


class SyncTests(unittest.TestCase):
    def test_t1_bootstrap_and_raw_snapshot_immutability(self) -> None:
        """T1: New user bootstrap explicit required seasons; raw files in staging are immutable."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            data_root = Path(tmp_dir)

            def mock_fetcher(season: str, timeout: float) -> bytes:
                return _make_epl_season_csv(season, match_count=380).encode("utf-8")

            # Initial bootstrap sync with apply
            res = sync_epl(
                data_root=data_root,
                seasons=["2024-25"],
                dry_run=False,
                apply=True,
                custom_fetcher=mock_fetcher,
            )
            self.assertEqual(res["status"], "applied")
            self.assertTrue(res["promotion_eligible"])
            self.assertEqual(len(res["seasons_downloaded"]), 1)

            raw_file = data_root / "raw" / "E0_2024-25.csv"
            self.assertTrue(raw_file.is_file())
            initial_hash = file_sha256(raw_file)

            pointer = load_current_dataset_pointer(data_root)
            self.assertIsNotNone(pointer)
            self.assertEqual(pointer["data_cutoff"], "2025-05-25")

            # Second sync for another season: verify 2024-25 raw is cached and untouched
            res2 = sync_epl(
                data_root=data_root,
                seasons=["2024-25"],
                dry_run=False,
                apply=True,
                custom_fetcher=mock_fetcher,
            )
            self.assertIn("2024-25", res2["seasons_cached"])
            self.assertEqual(file_sha256(raw_file), initial_hash)

    def test_t2_dry_run_leaves_pointer_unchanged_and_revisions_block_apply(self) -> None:
        """T2: Dry-run leaves pointer unchanged; revisions to existing records block promotion."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            data_root = Path(tmp_dir)

            def mock_fetcher_v1(season: str, timeout: float) -> bytes:
                return _make_epl_season_csv(season, match_count=380).encode("utf-8")

            # 1. Apply baseline 2024-25
            sync_epl(
                data_root=data_root,
                seasons=["2024-25"],
                dry_run=False,
                apply=True,
                custom_fetcher=mock_fetcher_v1,
            )
            initial_pointer = load_current_dataset_pointer(data_root)
            self.assertIsNotNone(initial_pointer)
            v1_version = initial_pointer["dataset_version"]

            # 2. Dry-run sync: must not alter pointer
            dry_res = sync_epl(
                data_root=data_root,
                seasons=["2024-25"],
                dry_run=True,
                apply=False,
                custom_fetcher=mock_fetcher_v1,
            )
            self.assertEqual(dry_res["action"], "dry_run")
            pointer_after_dry = load_current_dataset_pointer(data_root)
            self.assertEqual(pointer_after_dry["dataset_version"], v1_version)

            # 3. Simulate an upstream revision (e.g. score changed for match 1)
            def mock_fetcher_with_revision(season: str, timeout: float) -> bytes:
                content = _make_epl_season_csv(season, match_count=380)
                # Change the first match score from 1,0 to 2,0
                content_rev = content.replace(",1,0,H,", ",2,0,H,", 1)
                return content_rev.encode("utf-8")

            # Remove cached raw file to force re-fetch into staging
            (data_root / "raw" / "E0_2024-25.csv").unlink()

            rev_res = sync_epl(
                data_root=data_root,
                seasons=["2024-25"],
                dry_run=False,
                apply=True,
                custom_fetcher=mock_fetcher_with_revision,
            )
            # Revisions MUST block promotion!
            self.assertFalse(rev_res["promotion_eligible"])
            self.assertIn("review_required", rev_res["status"])
            self.assertGreater(rev_res["diff_summary"]["total_revisions"], 0)

            # Pointer must remain the previous valid version!
            pointer_after_blocked = load_current_dataset_pointer(data_root)
            self.assertEqual(pointer_after_blocked["dataset_version"], v1_version)

    def test_t3_in_progress_season_handles_pending_and_strict_380_gate(self) -> None:
        """T3: In-progress season declaration excludes pending rows; strict 380 gate applied to completed seasons."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            data_root = Path(tmp_dir)

            # Case A: An incomplete season WITHOUT in-progress declaration fails strict 380 gate
            def mock_incomplete_fetcher(season: str, timeout: float) -> bytes:
                return _make_epl_season_csv(season, match_count=200, pending_count=180).encode("utf-8")

            fail_res = sync_epl(
                data_root=data_root,
                seasons=["2025-26"],
                in_progress_season=None,  # Not declared!
                dry_run=False,
                apply=True,
                custom_fetcher=mock_incomplete_fetcher,
            )
            self.assertFalse(fail_res["promotion_eligible"])
            self.assertTrue(any("380" in err for err in fail_res["promotion_block_reasons"]))

            # Case B: With in-progress season declaration, pending rows are excluded and counted
            success_res = sync_epl(
                data_root=data_root,
                seasons=["2025-26"],
                in_progress_season="2025-26",
                dry_run=False,
                apply=True,
                custom_fetcher=mock_incomplete_fetcher,
            )
            self.assertTrue(success_res["promotion_eligible"])
            self.assertEqual(success_res["status"], "applied")
            self.assertEqual(success_res["diff_summary"]["total_skipped_pending"], 180)

            pointer = load_current_dataset_pointer(data_root)
            self.assertIsNotNone(pointer)
            self.assertEqual(pointer["canonical_rows"], 200)

    def test_t4_cached_seasons_retained_and_snapshot_selection_independent_of_traversal(self) -> None:
        """T4: Old cached seasons are retained when only current season partition is fetched."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            data_root = Path(tmp_dir)

            def mock_multi_season(season: str, timeout: float) -> bytes:
                if season == "2024-25":
                    return _make_epl_season_csv(season, match_count=380).encode("utf-8")
                else:
                    return _make_epl_season_csv(season, match_count=150, pending_count=50).encode("utf-8")

            # Sync 2024-25 first
            sync_epl(
                data_root=data_root,
                seasons=["2024-25"],
                dry_run=False,
                apply=True,
                custom_fetcher=mock_multi_season,
            )

            # Sync with 2024-25 and 2025-26 (in-progress)
            res = sync_epl(
                data_root=data_root,
                seasons=["2024-25", "2025-26"],
                in_progress_season="2025-26",
                dry_run=False,
                apply=True,
                custom_fetcher=mock_multi_season,
            )
            self.assertEqual(res["status"], "applied")
            self.assertIn("2024-25", res["seasons_cached"])
            self.assertIn("2025-26", res["seasons_downloaded"])

            pointer = load_current_dataset_pointer(data_root)
            self.assertEqual(pointer["canonical_rows"], 380 + 150)
            self.assertEqual(pointer["seasons"], ["2024-25", "2025-26"])


if __name__ == "__main__":
    unittest.main()
