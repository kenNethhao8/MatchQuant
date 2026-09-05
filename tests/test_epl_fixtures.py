import json
import stat
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from matchquant.epl_fixtures import (
    audit_fixture_source_frame,
    compare_fixtures,
    convert_fixture_source_frame,
    prepare_epl_fixtures,
    stable_fixture_id,
)
from matchquant.fixtures import load_fixtures
from matchquant.forecast import run_forecast
from matchquant.io import file_sha256, write_json
from matchquant.schema import DataValidationError


SOURCE_COLUMNS = ["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG"]


def source_frame(rows):
    return pd.DataFrame(rows, columns=SOURCE_COLUMNS)


def write_source(root: Path, rows, name="E0.csv") -> Path:
    path = root / name
    source_frame(rows).to_csv(path, index=False)
    return path


class EplFixtureAdapterTests(unittest.TestCase):
    def test_mapping_classifies_pending_and_completed_and_stable_id_ignores_date(self):
        source = source_frame(
            [
                ["01/08/2025", "Alpha", "Beta", "", ""],
                ["02/08/2025", "Gamma", "Delta", "2", "1"],
            ]
        )
        fixtures, report = convert_fixture_source_frame(source, "2025-26")
        self.assertEqual(report["status"], "ok")
        self.assertEqual(report["pending_rows"], 1)
        self.assertEqual(report["completed_rows"], 1)
        self.assertEqual(fixtures.loc[0, "fixture_id"], stable_fixture_id("2025-26", "Alpha", "Beta"))
        revised = source.copy()
        revised.loc[0, "Date"] = "15/08/2025"
        revised.loc[1, "FTHG"] = ""
        revised.loc[1, "FTAG"] = ""
        changed, _ = convert_fixture_source_frame(revised, "2025-26")
        fixture_id = stable_fixture_id("2025-26", "Alpha", "Beta")
        self.assertEqual(fixtures.loc[fixtures["fixture_id"].eq(fixture_id), "date"].item(), "2025-08-01")
        self.assertEqual(changed.loc[changed["fixture_id"].eq(fixture_id), "date"].item(), "2025-08-15")
        self.assertEqual(list(fixtures.columns), ["fixture_id", "date", "season", "home_team", "away_team"])

    def test_strict_date_and_score_validation_is_explicit(self):
        bad = source_frame(
            [
                ["2025/08/01", "Alpha", "Beta", "", ""],
                ["02/08/2025", "Gamma", "Delta", "1.5", "0"],
                ["03/08/2025", "Epsilon", "Zeta", "2", ""],
            ]
        )
        report = audit_fixture_source_frame(bad, "2025-26")
        self.assertEqual(report["status"], "failed")
        self.assertEqual(report["invalid_date_rows"], 3)
        self.assertEqual(report["partial_score_rows"], 1)
        self.assertEqual(report["invalid_score_rows"], 1)
        self.assertEqual(report["invalid_rows"], 3)
        with self.assertRaisesRegex(DataValidationError, "invalid E0 fixture source"):
            convert_fixture_source_frame(bad, "2025-26")

    def test_duplicate_stable_id_and_conflict_are_not_silently_skipped(self):
        source = source_frame(
            [
                ["01/08/2025", "Alpha", "Beta", "", ""],
                ["04/08/2025", "Alpha", "Beta", "", ""],
            ]
        )
        report = audit_fixture_source_frame(source, "2025-26")
        self.assertEqual(report["duplicate_stable_fixture_ids"], 1)
        self.assertEqual(report["conflicting_fixture_ids"], 1)
        self.assertEqual(report["status"], "failed")
        with self.assertRaises(DataValidationError):
            convert_fixture_source_frame(source, "2025-26")

    def test_partial_invalid_score_is_counted_and_failed(self):
        source = source_frame([["01/08/2025", "Alpha", "Beta", "not-a-score", ""]])
        report = audit_fixture_source_frame(source, "2025-26")
        self.assertEqual(report["partial_score_rows"], 1)
        self.assertEqual(report["invalid_score_rows"], 1)
        self.assertEqual(report["invalid_rows"], 1)
        with self.assertRaises(DataValidationError):
            convert_fixture_source_frame(source, "2025-26")

    def test_missing_columns_and_invalid_team_are_reported(self):
        missing = source_frame([["01/08/2025", "Alpha", "Beta", "", ""]]).drop(columns=["FTAG"])
        report = audit_fixture_source_frame(missing, "2025-26")
        self.assertEqual(report["status"], "failed")
        self.assertIn("FTAG", report["errors"][0])

        same_team = source_frame([["01/08/2025", "Alpha", "alpha", "", ""]])
        report = audit_fixture_source_frame(same_team, "2025-26")
        self.assertEqual(report["invalid_team_rows"], 1)
        self.assertEqual(report["status"], "failed")

    def test_no_pending_creates_evidence_but_no_empty_fixture_csv(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = write_source(root, [["01/08/2025", "Alpha", "Beta", "1", "0"]])
            result = prepare_epl_fixtures(source, "2025-26", root / "data")
            self.assertEqual(result["status"], "no_pending_fixtures")
            self.assertIsNone(result["fixture_path"])
            self.assertTrue(Path(result["manifest_path"]).is_file())
            self.assertTrue(Path(result["quality_report_path"]).is_file())
            self.assertFalse(Path(result["snapshot_dir"], "processed", "epl_fixtures.csv").exists())
            quality = json.loads(Path(result["quality_report_path"]).read_text(encoding="utf-8"))
            self.assertEqual(quality["status"], "no_pending_fixtures")
            self.assertIn("no_pending_fixtures", " ".join(quality["warnings"]))

    def test_failed_snapshot_writes_manifest_quality_and_no_fixture_output(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = write_source(root, [["01/08/2025", "Alpha", "Beta", "1", ""]])
            before = source.read_bytes()
            with self.assertRaisesRegex(DataValidationError, "quality report") as raised:
                prepare_epl_fixtures(source, "2025-26", root / "data")
            message = str(raised.exception)
            self.assertIn("manifest:", message)
            snapshots = list((root / "data" / "fixtures" / "snapshots").iterdir())
            self.assertEqual(len(snapshots), 1)
            snapshot = snapshots[0]
            manifest_path = snapshot / "fixture_snapshot_manifest.json"
            quality_path = snapshot / "reports" / "fixture_quality_report.json"
            self.assertTrue(manifest_path.is_file())
            self.assertTrue(quality_path.is_file())
            self.assertFalse((snapshot / "processed" / "epl_fixtures.csv").exists())
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            quality = json.loads(quality_path.read_text(encoding="utf-8"))
            self.assertEqual(manifest["status"], "failed")
            self.assertEqual(quality["partial_score_rows"], 1)
            self.assertEqual(manifest["source_sha256"], file_sha256(manifest["source_raw_snapshot"]))
            self.assertEqual(source.read_bytes(), before)

    def test_snapshot_is_unique_immutable_and_hashes_evidence(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = write_source(root, [["01/08/2025", "Alpha", "Beta", "", ""]])
            before = source.read_bytes()
            first = prepare_epl_fixtures(source, "2025-26", root / "data", source_url="https://local.invalid/e0.csv")
            second = prepare_epl_fixtures(source, "2025-26", root / "data")
            self.assertNotEqual(first["snapshot_id"], second["snapshot_id"])
            raw = Path(first["manifest"]["source_raw_snapshot"])
            self.assertEqual(raw.read_bytes(), before)
            self.assertEqual(file_sha256(raw), first["source_sha256"])
            self.assertFalse(raw.stat().st_mode & stat.S_IWRITE)
            manifest = json.loads(Path(first["manifest_path"]).read_text(encoding="utf-8"))
            self.assertEqual(manifest["source_sha256"], file_sha256(raw))
            self.assertEqual(manifest["canonical_output_sha256"], file_sha256(first["fixture_path"]))
            self.assertEqual(manifest["canonical_output_row_count"], 1)
            self.assertIsNone(manifest["source_label"])
            self.assertEqual(manifest["source_url"], "https://local.invalid/e0.csv")
            self.assertEqual(source.read_bytes(), before)
            self.assertFalse((root / "data" / "processed" / "epl_current.json").exists())

    def test_baseline_diff_has_all_categories_and_never_changes_baseline(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            baseline = pd.DataFrame(
                [
                    [stable_fixture_id("2025-26", "Alpha", "Beta"), "2025-08-01", "2025-26", "Alpha", "Beta"],
                    [stable_fixture_id("2025-26", "Gamma", "Delta"), "2025-08-02", "2025-26", "Gamma", "Delta"],
                    [stable_fixture_id("2025-26", "Gone", "Team"), "2025-08-03", "2025-26", "Gone", "Team"],
                ],
                columns=["fixture_id", "date", "season", "home_team", "away_team"],
            )
            baseline_path = root / "baseline.csv"
            baseline.to_csv(baseline_path, index=False)
            before = baseline_path.read_bytes()
            source = write_source(
                root,
                [
                    ["01/08/2025", "Alpha", "Beta", "", ""],
                    ["04/08/2025", "Gamma", "Delta", "", ""],
                    ["05/08/2025", "New", "Team", "", ""],
                ],
            )
            result = prepare_epl_fixtures(source, "2025-26", root / "data", baseline_fixtures=baseline_path)
            diff = json.loads(Path(result["diff_report_path"]).read_text(encoding="utf-8"))
            self.assertEqual(diff["counts"], {"added": 1, "unchanged": 1, "revised": 1, "missing_from_candidate": 1})
            self.assertEqual(diff["details"]["revised"][0]["changed_fields"], ["date"])
            self.assertTrue(diff["missing_from_candidate_requires_manual_judgment"])
            self.assertEqual(baseline_path.read_bytes(), before)
            self.assertEqual(compare_fixtures(baseline, load_fixtures(result["fixture_path"]))["counts"]["revised"], 1)

    def test_generated_csv_passes_m4_validator_and_forecast(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = write_source(
                root,
                [["07/01/2024", "A", "B", "", ""], ["08/01/2024", "C", "D", "", ""]],
            )
            result = prepare_epl_fixtures(source, "2024-25", root / "data")
            fixtures = load_fixtures(result["fixture_path"])
            self.assertEqual(len(fixtures), 2)
            config = root / "frequency.json"
            write_json({}, config)
            forecast = run_forecast(
                "examples/synthetic_matches.csv",
                result["fixture_path"],
                "2024-01-07",
                model_name="frequency",
                model_config_path=config,
                output_root=root / "runs",
            )
            self.assertEqual(len(forecast["predictions"]), 2)

    def test_cli_smoke_is_offline_and_concise(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            completed = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "matchquant",
                    "epl-fixtures-prepare",
                    "--source-csv",
                    "examples/synthetic_e0_source.csv",
                    "--season",
                    "2024-25",
                    "--data-root",
                    str(root / "data"),
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertLessEqual(len(completed.stdout.strip().splitlines()), 20)
            self.assertIn("MatchQuant M5 EPL fixture snapshot ok", completed.stdout)
            snapshots = list((root / "data" / "fixtures" / "snapshots").iterdir())
            self.assertEqual(len(snapshots), 1)
            self.assertTrue((snapshots[0] / "processed" / "epl_fixtures.csv").is_file())


if __name__ == "__main__":
    unittest.main()
