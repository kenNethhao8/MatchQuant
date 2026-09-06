import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from matchquant.cli import main as cli_main
from matchquant.epl_local_stage import stage_local_epl_source
from matchquant.io import file_sha256, load_matches, write_json
from matchquant.postmatch_session import run_postmatch_session
from matchquant.postmatch_session_report import write_postmatch_session_report
from matchquant.postmatch_session_support import (
    active_pointer_evidence,
    discover_new_child,
    file_record,
    load_m9_evidence,
)
from matchquant.schema import DataValidationError
from tests.postmatch_session_fixtures import _write_e0_results, build_ready_session


class PostmatchSessionTests(unittest.TestCase):
    def _dry_run(self, paths: dict[str, Path]) -> dict:
        with patch("matchquant.epl_sync_source.urllib.request.urlopen", side_effect=AssertionError("network used")):
            return run_postmatch_session(
                paths["session_manifest"],
                "2024-25",
                results_source_csv=paths["results_source"],
                in_progress_season="2024-25",
                data_root=paths["data_root"],
                output_root=paths["runs_root"],
            )

    def test_source_mode_only_stages_and_dry_runs_without_mutating_pointer(self):
        with tempfile.TemporaryDirectory() as temporary:
            paths = build_ready_session(Path(temporary))
            pointer_before = file_sha256(paths["pointer"])
            result = self._dry_run(paths)
            self.assertEqual(result["status"], "needs_review")
            self.assertEqual(result["stages"]["local_stage"]["status"], "completed")
            self.assertEqual(result["stages"]["sync"]["status"], "dry_run")
            self.assertEqual(result["stages"]["settlement"]["status"], "not_run")
            self.assertTrue(result["apply_eligibility"]["eligible"])
            self.assertEqual(file_sha256(paths["pointer"]), pointer_before)
            self.assertFalse((paths["data_root"] / "processed" / "versions").exists())
            session_dir = Path(result["session_dir"])
            self.assertEqual({item.name for item in session_dir.iterdir()}, {"postmatch_manifest.json", "postmatch_report.md"})

    def test_staging_mode_apply_activates_then_settles_exact_m9_forecast(self):
        with tempfile.TemporaryDirectory() as temporary:
            paths = build_ready_session(Path(temporary))
            dry_run = self._dry_run(paths)
            pointer_before = file_sha256(paths["pointer"])
            with patch("matchquant.epl_sync_source.urllib.request.urlopen", side_effect=AssertionError("network used")):
                applied = run_postmatch_session(
                    paths["session_manifest"],
                    "2024-25",
                    staging_dir=dry_run["staging_dir"],
                    apply=True,
                    in_progress_season="2024-25",
                    data_root=paths["data_root"],
                    output_root=paths["runs_root"],
                )
            self.assertEqual(applied["stages"]["sync"]["status"], "applied")
            self.assertEqual(applied["stages"]["settlement"]["status"], "complete")
            self.assertTrue(Path(applied["settlement_run_dir"]).is_dir())
            self.assertNotEqual(file_sha256(paths["pointer"]), pointer_before)
            self.assertTrue(Path(applied["version_dir"]).is_dir())
            manifest = json.loads(Path(applied["manifest_path"]).read_text(encoding="utf-8"))
            for stage_name in ("local_stage", "sync", "settlement"):
                self.assertTrue(manifest["stages"][stage_name]["artifacts"])

    def test_direct_results_source_cannot_apply_in_the_same_call(self):
        with tempfile.TemporaryDirectory() as temporary:
            paths = build_ready_session(Path(temporary))
            with self.assertRaisesRegex(ValueError, "--apply requires --staging-dir"):
                run_postmatch_session(
                    paths["session_manifest"], "2024-25",
                    results_source_csv=paths["results_source"], apply=True,
                    data_root=paths["data_root"], output_root=paths["runs_root"],
                )

    def test_m9_loader_returns_only_hash_verified_recorded_artifacts(self):
        with tempfile.TemporaryDirectory() as temporary:
            paths = build_ready_session(Path(temporary))
            evidence = load_m9_evidence(paths["session_manifest"])
            self.assertEqual(evidence["status"], "ready")
            self.assertEqual(evidence["history_path"], paths["history"].resolve())
            self.assertEqual(evidence["history_manifest_path"], paths["history_manifest"].resolve())
            self.assertEqual(file_record(paths["history"])["sha256"], evidence["baseline"]["history"]["sha256"])
            self.assertTrue(evidence["forecast_run_dir"].is_dir())
            self.assertEqual(
                file_sha256(evidence["forecast_predictions_path"]),
                evidence["baseline"]["forecast_predictions"]["sha256"],
            )

    def test_active_pointer_requires_both_m9_baseline_hashes(self):
        with tempfile.TemporaryDirectory() as temporary:
            paths = build_ready_session(Path(temporary))
            evidence = load_m9_evidence(paths["session_manifest"])
            matched = active_pointer_evidence(paths["pointer"], evidence["baseline"])
            self.assertEqual(matched["status"], "matched")
            self.assertTrue(matched["canonical_hash_matches_m9"])
            self.assertTrue(matched["processed_manifest_hash_matches_m9"])
            pointer = json.loads(paths["pointer"].read_text(encoding="utf-8"))
            pointer["canonical"] = str(paths["history_manifest"])
            write_json(pointer, paths["pointer"])
            mismatched = active_pointer_evidence(paths["pointer"], evidence["baseline"])
            self.assertEqual(mismatched["status"], "mismatch")
            self.assertFalse(mismatched["canonical_hash_matches_m9"])
            self.assertTrue(mismatched["processed_manifest_hash_matches_m9"])

    def test_tampered_m9_history_blocks_before_local_stage(self):
        with tempfile.TemporaryDirectory() as temporary:
            paths = build_ready_session(Path(temporary))
            manifest_path = paths["session_manifest"]
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["inputs"]["history"]["sha256"] = "0" * 64
            write_json(manifest, manifest_path)
            result = self._dry_run(paths)
            self.assertEqual(result["status"], "blocked")
            self.assertEqual(result["stages"]["sync"]["status"], "not_run")
            self.assertEqual(result["stages"]["settlement"]["status"], "not_run")
            self.assertEqual(result["stages"]["local_stage"]["status"], "not_run")

    def test_tampered_m9_predictions_block_before_local_stage(self):
        with tempfile.TemporaryDirectory() as temporary:
            paths = build_ready_session(Path(temporary))
            manifest = json.loads(paths["session_manifest"].read_text(encoding="utf-8"))
            predictions = Path(manifest["outputs"]["forecast_predictions"]["path"])
            predictions.write_text(predictions.read_text(encoding="utf-8") + "\n", encoding="utf-8")
            result = self._dry_run(paths)
            self.assertEqual(result["status"], "blocked")
            self.assertEqual(result["stages"]["local_stage"]["status"], "not_run")
            self.assertEqual(result["stages"]["sync"]["status"], "not_run")
            self.assertEqual(result["stages"]["settlement"]["status"], "not_run")

    def test_blocked_m9_status_is_preserved_and_stops_all_children(self):
        with tempfile.TemporaryDirectory() as temporary:
            paths = build_ready_session(Path(temporary))
            manifest_path = paths["session_manifest"]
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["status"] = "blocked"
            write_json(manifest, manifest_path)
            result = self._dry_run(paths)
            self.assertEqual(result["status"], "blocked")
            self.assertEqual(result["stages"]["local_stage"]["status"], "not_run")
            self.assertEqual(result["stages"]["sync"]["status"], "not_run")

    def test_malformed_raw_results_retain_m7_and_quality_failure_evidence(self):
        with tempfile.TemporaryDirectory() as temporary:
            paths = build_ready_session(Path(temporary))
            bad_source = Path(temporary) / "input" / "bad_results.csv"
            bad_source.write_text(paths["results_source"].read_text(encoding="utf-8").replace(",FTAG", ""), encoding="utf-8")
            result = run_postmatch_session(
                paths["session_manifest"], "2024-25", results_source_csv=bad_source,
                in_progress_season="2024-25", data_root=paths["data_root"], output_root=paths["runs_root"],
            )
            self.assertEqual(result["status"], "blocked")
            self.assertEqual(result["stages"]["local_stage"]["status"], "completed")
            self.assertEqual(result["stages"]["sync"]["status"], "blocked")
            self.assertEqual(result["stages"]["settlement"]["status"], "not_run")
            quality = Path(result["stages"]["sync"]["artifacts"]["quality_report"]["path"])
            self.assertTrue(quality.is_file())

    def test_revision_candidate_blocks_apply_before_pointer_change_or_m6(self):
        with tempfile.TemporaryDirectory() as temporary:
            paths = build_ready_session(Path(temporary))
            revised_source = _write_e0_results(Path(temporary) / "input" / "revised.csv", load_matches(paths["history"]), revised=True)
            staged = stage_local_epl_source(revised_source, "2024-25", paths["data_root"])
            pointer_before = file_sha256(paths["pointer"])
            result = run_postmatch_session(
                paths["session_manifest"], "2024-25", staging_dir=staged["snapshot_dir"], apply=True,
                in_progress_season="2024-25", data_root=paths["data_root"], output_root=paths["runs_root"],
            )
            self.assertEqual(result["status"], "blocked")
            self.assertEqual(result["stages"]["sync"]["status"], "blocked")
            self.assertEqual(result["stages"]["settlement"]["status"], "not_run")
            self.assertNotEqual(result["stages"]["sync"].get("result_status"), "applied")
            self.assertEqual(file_sha256(paths["pointer"]), pointer_before)

    def test_missing_candidate_blocks_apply_before_pointer_change_or_m6(self):
        with tempfile.TemporaryDirectory() as temporary:
            paths = build_ready_session(Path(temporary))
            missing_source = _write_e0_results(Path(temporary) / "input" / "missing.csv", load_matches(paths["history"]), omit_first=True)
            staged = stage_local_epl_source(missing_source, "2024-25", paths["data_root"])
            pointer_before = file_sha256(paths["pointer"])
            result = run_postmatch_session(paths["session_manifest"], "2024-25", staging_dir=staged["snapshot_dir"], apply=True, in_progress_season="2024-25", data_root=paths["data_root"], output_root=paths["runs_root"])
            self.assertEqual(result["status"], "blocked")
            self.assertEqual(result["stages"]["sync"]["result_status"], "dry-run_apply_blocked")
            self.assertEqual(result["stages"]["settlement"]["status"], "not_run")
            self.assertEqual(file_sha256(paths["pointer"]), pointer_before)

    def test_pointer_mismatch_blocks_apply_without_mutating_pointer(self):
        with tempfile.TemporaryDirectory() as temporary:
            paths = build_ready_session(Path(temporary))
            dry_run = self._dry_run(paths)
            pointer_before = file_sha256(paths["pointer"])
            pointer = json.loads(paths["pointer"].read_text(encoding="utf-8"))
            pointer["canonical_sha256"] = "f" * 64
            write_json(pointer, paths["pointer"])
            mismatched_before = file_sha256(paths["pointer"])
            result = run_postmatch_session(
                paths["session_manifest"], "2024-25", staging_dir=dry_run["staging_dir"], apply=True,
                in_progress_season="2024-25", data_root=paths["data_root"], output_root=paths["runs_root"],
            )
            self.assertEqual(result["status"], "blocked")
            self.assertEqual(result["stages"]["sync"]["result_status"], "pre_apply_pointer_mismatch")
            self.assertEqual(result["stages"]["settlement"]["status"], "not_run")
            self.assertEqual(file_sha256(paths["pointer"]), mismatched_before)
            self.assertNotEqual(pointer_before, mismatched_before)

    def test_m6_failure_preserves_applied_pointer_and_session_evidence(self):
        with tempfile.TemporaryDirectory() as temporary:
            paths = build_ready_session(Path(temporary))
            dry_run = self._dry_run(paths)
            pointer_before = file_sha256(paths["pointer"])
            with patch("matchquant.postmatch_session.settle_forecast_run", side_effect=DataValidationError("forced M6 failure")):
                result = run_postmatch_session(
                    paths["session_manifest"], "2024-25", staging_dir=dry_run["staging_dir"], apply=True,
                    in_progress_season="2024-25", data_root=paths["data_root"], output_root=paths["runs_root"],
                )
            self.assertEqual(result["status"], "blocked")
            self.assertEqual(result["stages"]["sync"]["status"], "applied")
            self.assertEqual(result["stages"]["settlement"]["status"], "blocked")
            self.assertNotEqual(file_sha256(paths["pointer"]), pointer_before)
            self.assertIn("forced M6 failure", result["stages"]["settlement"]["error"]["message"])

    def test_m9_needs_review_is_not_upgraded_after_successful_settlement(self):
        with tempfile.TemporaryDirectory() as temporary:
            paths = build_ready_session(Path(temporary))
            manifest_path = paths["session_manifest"]
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["status"] = "needs_review"
            write_json(manifest, manifest_path)
            dry_run = self._dry_run(paths)
            result = run_postmatch_session(
                paths["session_manifest"], "2024-25", staging_dir=dry_run["staging_dir"], apply=True,
                in_progress_season="2024-25", data_root=paths["data_root"], output_root=paths["runs_root"],
            )
            self.assertEqual(result["stages"]["settlement"]["status"], "complete")
            self.assertEqual(result["status"], "needs_review")

    def test_output_root_guards_run_before_m10_session_creation(self):
        with tempfile.TemporaryDirectory() as temporary:
            paths = build_ready_session(Path(temporary))
            data_child = paths["data_root"] / "m10-output"
            with self.assertRaisesRegex(DataValidationError, "output-root"):
                run_postmatch_session(
                    paths["session_manifest"], "2024-25", results_source_csv=paths["results_source"],
                    data_root=paths["data_root"], output_root=data_child,
                )
            self.assertFalse(data_child.exists())
            m9_dir = paths["session_manifest"].parent
            with self.assertRaisesRegex(DataValidationError, "output-root"):
                run_postmatch_session(
                    paths["session_manifest"], "2024-25", results_source_csv=paths["results_source"],
                    data_root=paths["data_root"], output_root=m9_dir / "child",
                )

    def test_partial_child_discovery_never_binds_arbitrary_last_directory(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            before = set(root.iterdir())
            (root / "m6_settlement_a").mkdir()
            (root / "m6_settlement_b").mkdir()
            bound, evidence = discover_new_child(root, "m6_settlement_", before)
            self.assertIsNone(bound)
            self.assertEqual(evidence["status"], "ambiguous")
            self.assertIsNone(evidence["bound_path"])

    def test_report_renderer_uses_manifest_status_and_disclaimer(self):
        with tempfile.TemporaryDirectory() as temporary:
            report_path = Path(temporary) / "postmatch_report.md"
            write_postmatch_session_report(
                {"status": "blocked", "inputs": {}, "m9": {}, "stages": {}, "limitations": []}, report_path
            )
            report = report_path.read_text(encoding="utf-8")
            self.assertIn("M10 赛后会话报告", report)
            self.assertIn("blocked", report)
            self.assertIn("不声明概率已校准", report)

    def test_cli_source_mode_is_concise_and_returns_review_status(self):
        with tempfile.TemporaryDirectory() as temporary:
            paths = build_ready_session(Path(temporary))
            completed = subprocess.run(
                [
                    sys.executable, "-m", "matchquant", "postmatch-session",
                    "--session-manifest", str(paths["session_manifest"]),
                    "--results-source", str(paths["results_source"]), "--season", "2024-25",
                    "--in-progress-season", "2024-25", "--data-root", str(paths["data_root"]),
                    "--output-root", str(paths["runs_root"]),
                ], capture_output=True, text=True, check=False,
            )
            self.assertEqual(completed.returncode, 2, completed.stderr)
            self.assertIn("MatchQuant M10 post-match session needs_review", completed.stdout)
            self.assertLessEqual(len(completed.stdout.strip().splitlines()), 20)

    def test_cli_staging_apply_smoke_updates_pointer_and_settles_without_network(self):
        with tempfile.TemporaryDirectory() as temporary:
            paths = build_ready_session(Path(temporary))
            dry_run = self._dry_run(paths)
            pointer_before = file_sha256(paths["pointer"])
            versions_root = paths["data_root"] / "processed" / "versions"
            versions_before = set(versions_root.iterdir()) if versions_root.is_dir() else set()
            sessions_before = set(paths["runs_root"].glob("m10_postmatch_session_*"))
            args = [
                "postmatch-session", "--session-manifest", str(paths["session_manifest"]),
                "--staging-dir", str(dry_run["staging_dir"]), "--season", "2024-25", "--apply",
                "--in-progress-season", "2024-25", "--data-root", str(paths["data_root"]),
                "--output-root", str(paths["runs_root"]),
            ]
            with patch("matchquant.epl_sync_source.urllib.request.urlopen", side_effect=AssertionError("network used")), patch("sys.stdout"):
                with self.assertRaises(SystemExit) as exit_info:
                    cli_main(args)
            self.assertEqual(exit_info.exception.code, 0)
            self.assertNotEqual(file_sha256(paths["pointer"]), pointer_before)
            self.assertEqual(len(set(versions_root.iterdir()) - versions_before), 1)
            sessions = set(paths["runs_root"].glob("m10_postmatch_session_*")) - sessions_before
            self.assertEqual(len(sessions), 1)
            manifest = json.loads((next(iter(sessions)) / "postmatch_manifest.json").read_text(encoding="utf-8"))
            settlement = Path(manifest["stages"]["settlement"]["artifacts"]["settlement"]["path"])
            self.assertTrue(settlement.is_file())


if __name__ == "__main__":
    unittest.main()
