"""Tests for Layer 4: Agent Adapter, Skill installation, and 2027 scenario (T6, T7, T8, T12, T13, T15)."""

import json
import os
import subprocess
import sys
import tempfile
import unittest
from datetime import date, timedelta
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from matchquant.adapter import execute_adapter_request
from matchquant.dataset import build_dataset_from_manifest, load_current_dataset_pointer
from matchquant.profile import UserProfile
from matchquant.skill_installer import install_skill, verify_installed_skill
from tests.test_single_match import TEAM_NAMES, synthetic_rich_history


def _make_rich_csv_for_season(season: str, match_count: int = 380, start_date: date | None = None) -> pd.DataFrame:
    """Generate deterministic rich history for one season."""
    df = synthetic_rich_history((season,))
    if match_count < 380:
        df = df.iloc[:match_count].copy()
    if start_date is not None:
        # Shift dates
        delta = (start_date - pd.to_datetime(df["date"].iloc[0]).date()).days
        df["date"] = pd.to_datetime(df["date"]).map(lambda d: (d.date() + timedelta(days=delta)).isoformat())
    return df


class AdapterTests(unittest.TestCase):
    def setUp(self) -> None:
        sandbox = tempfile.TemporaryDirectory()
        self.addCleanup(sandbox.cleanup)
        self.default_profile_path = Path(sandbox.name) / "default_profile.json"
        environment = patch.dict(os.environ, {"MATCHQUANT_PROFILE_PATH": str(self.default_profile_path)})
        environment.start()
        self.addCleanup(environment.stop)

    def test_t12_adapter_setup_and_readiness_actions(self) -> None:
        """T12: Adapter setup and readiness return structured responses with alias resolution."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            p_path = Path(tmp_dir) / "profile.json"
            prof = UserProfile(data_root=tmp_dir, report_root=tmp_dir)
            prof.save(p_path)

            # 1. Setup action
            setup_res = execute_adapter_request({
                "action": "setup",
                "profile_path": str(p_path),
                "updates": {"auto_refresh": True},
            })
            self.assertEqual(setup_res["status"], "ok")
            self.assertEqual(setup_res["action"], "setup")
            self.assertTrue(setup_res["profile"]["refresh_policy"]["auto_refresh"])
            self.assertEqual(Path(setup_res["saved_profile_path"]), p_path)
            self.assertTrue(json.loads(p_path.read_text(encoding="utf-8"))["refresh_policy"]["auto_refresh"])
            self.assertFalse(self.default_profile_path.exists())

            # 2. Readiness action with alias resolution
            readiness_res = execute_adapter_request({
                "action": "readiness",
                "profile_path": str(p_path),
                "home": "曼城",
                "away": "维拉",
                "date": "2026-05-15",
                "as_of": "2026-05-10",
            })
            self.assertEqual(readiness_res["action"], "readiness")
            self.assertEqual(readiness_res["normalized_fixture"]["resolved_home"], "Man City")
            self.assertEqual(readiness_res["normalized_fixture"]["resolved_away"], "Aston Villa")
            self.assertFalse(readiness_res["ready"])  # No dataset created yet

            # 3. Invalid action produces structured failure
            unknown_res = execute_adapter_request({"action": "invalid_action"})
            self.assertEqual(unknown_res["status"], "failed")
            self.assertIn("unknown adapter action", unknown_res["error"])

    def test_t6_t7_t8_t13_synthetic_2027_end_to_end_and_followup(self) -> None:
        """T6, T7, T8, T13: Synthetic 2027 scenario, cutoff isolation, and follow-up reuse.

        Simulates installation ending in May 2026. Refresh adds missing 2026-27 matches.
        Cutoff 2027-01-10 strictly excludes later results.
        Follow-up reuses the run without refitting.
        """
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            data_root = root / "data"
            report_root = root / "runs"
            raw_dir = data_root / "raw"
            raw_dir.mkdir(parents=True, exist_ok=True)
            report_root.mkdir(parents=True, exist_ok=True)

            # Build initial baseline with 3 complete seasons ending in May 2026: 2023-24, 2024-25, 2025-26
            # Convert synthetic rich history to raw format
            base_seasons = ("2023-24", "2024-25", "2025-26")
            snapshots = {}
            for s in base_seasons:
                df_s = synthetic_rich_history((s,))
                raw_path = raw_dir / f"E0_{s}.csv"
                # Save as E0 raw format
                raw_rows = ["Date,HomeTeam,AwayTeam,FTHG,FTAG,FTR,Referee,HTHG,HTAG,HTR,HS,AS,HST,AST,HC,AC,HF,AF,HY,AY,HR,AR,AvgH,AvgD,AvgA"]
                for _, r in df_s.iterrows():
                    d_obj = pd.to_datetime(r["date"]).strftime("%d/%m/%Y")
                    raw_rows.append(
                        f"{d_obj},{r['home_team']},{r['away_team']},{r['home_goals']},{r['away_goals']},{r['actual_result']},"
                        f"{r['referee']},{r['ht_home_goals']},{r['ht_away_goals']},{r['ht_result']},{r['home_shots']},{r['away_shots']},"
                        f"{r['home_sot']},{r['away_sot']},{r['home_corners']},{r['away_corners']},{r['home_fouls']},{r['away_fouls']},"
                        f"{r['home_yellow']},{r['away_yellow']},{r['home_red']},{r['away_red']},{r['odds_open_h']},{r['odds_open_d']},{r['odds_open_a']}"
                    )
                raw_path.write_text("\n".join(raw_rows) + "\n", encoding="utf-8")
                snapshots[s] = raw_path

            # Initial dataset build ending May 2026
            build_dataset_from_manifest(
                season_snapshots=snapshots,
                data_root=data_root,
            )
            pointer_2026 = load_current_dataset_pointer(data_root)
            self.assertEqual(pointer_2026["data_cutoff"], "2026-04-17")

            # Create profile
            prof = UserProfile(
                data_root=str(data_root),
                report_root=str(report_root),
                in_progress_season="2026-27",
            )
            prof.refresh_policy.auto_refresh = True
            prof.team_aliases["曼城"] = "Club A"
            prof.team_aliases["维拉"] = "Club B"
            p_path = root / "profile.json"
            prof.save(p_path)

            # Synthetic 2026-27 season data: matches from Aug 2026 up to Jan 2027 (and some future Jan 11-14 rows)
            def mock_2027_fetcher(season: str, timeout: float) -> bytes:
                s_df = synthetic_rich_history(("2026-27",))
                raw_rows = ["Date,HomeTeam,AwayTeam,FTHG,FTAG,FTR,Referee,HTHG,HTAG,HTR,HS,AS,HST,AST,HC,AC,HF,AF,HY,AY,HR,AR,AvgH,AvgD,AvgA"]
                for _, r in s_df.iterrows():
                    d_obj = pd.to_datetime(r["date"]).strftime("%d/%m/%Y")
                    raw_rows.append(
                        f"{d_obj},{r['home_team']},{r['away_team']},{r['home_goals']},{r['away_goals']},{r['actual_result']},"
                        f"{r['referee']},{r['ht_home_goals']},{r['ht_away_goals']},{r['ht_result']},{r['home_shots']},{r['away_shots']},"
                        f"{r['home_sot']},{r['away_sot']},{r['home_corners']},{r['away_corners']},{r['home_fouls']},{r['away_fouls']},"
                        f"{r['home_yellow']},{r['away_yellow']},{r['home_red']},{r['away_red']},{r['odds_open_h']},{r['odds_open_d']},{r['odds_open_a']}"
                    )
                return ("\n".join(raw_rows) + "\n").encode("utf-8")

            # Pre-sync 2026-27 into data_root using sync_epl with custom_fetcher
            from matchquant.sync import sync_epl
            sync_epl(
                data_root=data_root,
                seasons=["2023-24", "2024-25", "2025-26", "2026-27"],
                in_progress_season="2026-27",
                dry_run=False,
                apply=True,
                custom_fetcher=mock_2027_fetcher,
            )

            # Analyze 2027 fixture: Man City vs Aston Villa on 2027-01-15 with cutoff 2027-01-10
            analyze_req = {
                "action": "analyze",
                "profile_path": str(p_path),
                "home": "曼城",
                "away": "维拉",
                "date": "2027-01-15",
                "as_of": "2027-01-10",
                "season": "2026-27",
            }
            res = execute_adapter_request(analyze_req)
            self.assertEqual(res["status"], "ok")
            self.assertEqual(res["fixture"]["home"], "Club A")
            self.assertEqual(res["fixture"]["away"], "Club B")
            self.assertEqual(res["temporal"]["as_of"], "2027-01-10")

            # Verify predictions exist and are normalized
            preds = res["predictions"]
            self.assertIn("p_home", preds)
            self.assertIn("p_draw", preds)
            self.assertIn("p_away", preds)
            self.assertAlmostEqual(preds["p_home"] + preds["p_draw"] + preds["p_away"], 1.0, places=4)

            # Verify artifacts were written
            artifacts = res["artifacts"]
            for art_name, art_path in artifacts.items():
                self.assertTrue(Path(art_path).is_file(), f"{art_name} file missing: {art_path}")

            # Verify T7 & T8: Cutoff 2027-01-10 excluded the fixture day and later days
            used_cutoff = res["temporal"]["used_data_cutoff"]
            self.assertLess(used_cutoff, "2027-01-10")

            # T13: Follow-up question reusing existing run
            followup_req = {
                "action": "followup",
                "profile_path": str(p_path),
                "run_id": res["run_id"],
                "run_dir": res["run_dir"],
                "question": "为什么看好主队？给出简要解释",
            }
            f_res = execute_adapter_request(followup_req)
            if f_res.get("status") != "ok":
                self.fail(f"followup failed: {f_res.get('error')}")
            self.assertEqual(f_res["status"], "ok")
            self.assertFalse(f_res["recomputed"])
            self.assertIn("E1", f_res["referenced_evidence_ids"])
            self.assertIn("Club A", f_res["answer"])

    def test_t15_skill_installation_and_cli_smoke(self) -> None:
        """T15: Skill installation into temporary directory and CLI entry points."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            installed_dest = install_skill(tmp_dir)
            verification = verify_installed_skill(installed_dest)
            self.assertEqual(verification["status"], "ok")
            self.assertTrue(verification["has_frontmatter"])

            # CLI smoke: test python -m matchquant.cli agent-adapter --action setup
            cmd = [
                sys.executable,
                "-m",
                "matchquant.cli",
                "agent-adapter",
                "--action",
                "setup",
            ]
            import os
            env = dict(os.environ)
            worktree_dir = str(Path(__file__).resolve().parent.parent)
            env["PYTHONPATH"] = worktree_dir
            env["PYTHONIOENCODING"] = "utf-8"
            completed = subprocess.run(
                cmd, capture_output=True, text=True, encoding="utf-8",
                stdin=subprocess.DEVNULL, env=env, timeout=15
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            out_json = json.loads(completed.stdout)
            self.assertEqual(out_json["action"], "setup")
            self.assertIn(out_json["status"], ("ok", "needs_review"))


if __name__ == "__main__":
    unittest.main()
