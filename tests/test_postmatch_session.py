import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from matchquant.epl_local_stage import stage_local_epl_source
from matchquant.epl import stable_match_id
from matchquant.forecast_session import run_forecast_session
from matchquant.io import file_sha256, load_matches, write_json
from matchquant.postmatch_session import run_postmatch_session
from matchquant.postmatch_session_report import write_postmatch_session_report
from matchquant.postmatch_session_support import (
    active_pointer_evidence,
    discover_new_child,
    file_record,
    load_m9_evidence,
)
from matchquant.schema import DataValidationError, REQUIRED_COLUMNS


EXAMPLES = Path(__file__).resolve().parents[1] / "examples"


def _write_baseline(root: Path) -> dict[str, Path]:
    data_root = root / "data"
    processed = data_root / "processed"
    processed.mkdir(parents=True)
    history = processed / "epl_canonical.csv"
    shutil.copyfile(EXAMPLES / "synthetic_matches.csv", history)
    frame = load_matches(history)
    frame["match_id"] = [stable_match_id(row.season, row.home_team, row.away_team) for row in frame.itertuples()]
    frame.loc[:, list(REQUIRED_COLUMNS)].to_csv(history, index=False)
    frame = load_matches(history)
    history_manifest = processed / "epl_processed_manifest.json"
    write_json(
        {
            "schema_version": "m2.1-processed-manifest-v1",
            "adapter_version": "m2.1-epl-football-data-v1",
            "dataset_version": "m2.1-test-baseline",
            "data_cutoff": str(frame["date"].max()),
            "output": {
                "filename": str(history),
                "sha256": file_sha256(history),
                "rows": len(frame),
                "columns": list(REQUIRED_COLUMNS),
            },
        },
        history_manifest,
    )
    pointer = processed / "epl_current.json"
    write_json(
        {
            "schema_version": "m2.1-current-pointer-v1",
            "dataset_version": "m2.1-test-baseline",
            "canonical": str(history),
            "processed_manifest": str(history_manifest),
            "quality_report": None,
            "diff_report": None,
            "canonical_sha256": file_sha256(history),
        },
        pointer,
    )
    return {"data_root": data_root, "history": history, "history_manifest": history_manifest, "pointer": pointer}


def _write_e0_results(path: Path, history: pd.DataFrame, *, revised: bool = False, omit_first: bool = False) -> Path:
    rows: list[list[object]] = []
    selected = history.loc[history["season"].eq("2024-25")].reset_index(drop=True)
    for index, row in selected.iterrows():
        if omit_first and index == 0:
            continue
        home_goals = int(row["home_goals"]) + (1 if revised and index == 0 else 0)
        rows.append(
            [
                pd.Timestamp(row["date"]).strftime("%d/%m/%Y"),
                row["home_team"],
                row["away_team"],
                home_goals,
                int(row["away_goals"]),
            ]
        )
    rows.append(["11/01/2024", "FutureHome", "FutureAway", 2, 1])
    pd.DataFrame(rows, columns=["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG"]).to_csv(path, index=False)
    return path


def build_ready_session(root: Path) -> dict[str, Path]:
    baseline = _write_baseline(root)
    input_root = root / "input"
    input_root.mkdir()
    fixture_source = input_root / "forecast_fixture.csv"
    fixture_source.write_text(
        "Date,HomeTeam,AwayTeam,FTHG,FTAG\n11/01/2024,FutureHome,FutureAway,,\n",
        encoding="utf-8",
    )
    model_config = input_root / "frequency.json"
    model_config.write_text('{"alpha": 1.0}\n', encoding="utf-8")
    runs_root = root / "runs"
    m9 = run_forecast_session(
        baseline["history"],
        baseline["history_manifest"],
        fixture_source,
        "2024-25",
        "2024-01-10",
        model_name="frequency",
        model_config_path=model_config,
        fixture_data_root=baseline["data_root"],
        output_root=runs_root,
    )
    if m9["status"] != "ready":
        raise AssertionError(f"synthetic M9 setup was not ready: {m9['status']}")
    results_source = _write_e0_results(input_root / "results.csv", load_matches(baseline["history"]))
    return {
        **baseline,
        "runs_root": runs_root,
        "session_manifest": Path(m9["session_manifest_path"]),
        "results_source": results_source,
    }


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


if __name__ == "__main__":
    unittest.main()
