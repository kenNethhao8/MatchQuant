"""Tests for Layer 0: UserProfile and environment diagnostics."""

import json
import tempfile
import unittest
from pathlib import Path

from matchquant.profile import (
    DEFAULT_TEAM_ALIASES,
    UserProfile,
    diagnose_environment,
    load_profile,
)


class ProfileTests(unittest.TestCase):
    def test_default_profile_creation_and_alias_resolution(self) -> None:
        prof = UserProfile()
        self.assertEqual(prof.resolve_team_alias("曼城"), "Man City")
        self.assertEqual(prof.resolve_team_alias("维拉"), "Aston Villa")
        self.assertEqual(prof.resolve_team_alias("阿森纳"), "Arsenal")
        self.assertEqual(prof.resolve_team_alias("Chelsea"), "Chelsea")
        self.assertEqual(prof.resolve_team_alias("未知球队"), "未知球队")

    def test_profile_serialization_and_deserialization(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            p_path = Path(tmp_dir) / "profile.json"
            prof = UserProfile(
                data_root=r"E:\Test\data",
                report_root=r"E:\Test\reports",
                in_progress_season="2025-26",
            )
            prof.refresh_policy.auto_refresh = True
            prof.team_aliases["枪手"] = "Arsenal"
            prof.save(p_path)

            loaded = load_profile(p_path)
            self.assertEqual(loaded.data_root, r"E:\Test\data")
            self.assertEqual(loaded.report_root, r"E:\Test\reports")
            self.assertEqual(loaded.in_progress_season, "2025-26")
            self.assertTrue(loaded.refresh_policy.auto_refresh)
            self.assertEqual(loaded.resolve_team_alias("枪手"), "Arsenal")

    def test_profile_validation_and_diagnostics(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            prof = UserProfile(
                data_root=tmp_dir,
                report_root=tmp_dir,
                historical_start_season="2015-16",
            )
            diag = diagnose_environment(prof)
            self.assertEqual(diag["status"], "ok")
            self.assertTrue(diag["data_root_writable"])
            self.assertTrue(diag["report_root_writable"])
            self.assertIn("numpy", diag["packages"])
            self.assertIn("pandas", diag["packages"])
            self.assertIn("scipy", diag["packages"])

    def test_invalid_profile_fields_produce_validation_errors(self) -> None:
        bad_prof = UserProfile(
            data_root="",
            report_root="",
            historical_start_season="invalid-season",
        )
        bad_prof.refresh_policy.source_check_interval_hours = -5
        errors = bad_prof.validate()
        self.assertGreaterEqual(len(errors), 3)


if __name__ == "__main__":
    unittest.main()
