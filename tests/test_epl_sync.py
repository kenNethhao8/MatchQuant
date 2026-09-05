import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from matchquant.epl import convert_source_frame
from matchquant.epl_sync import compare_canonical, sync_epl
from matchquant.io import file_sha256, load_matches, write_json
from matchquant.schema import DataValidationError


def source_frame(rows):
    return pd.DataFrame(rows, columns=["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG"])


def complete_source_frame(start="2014-08-01"):
    dates = pd.date_range(start, periods=380, freq="D").strftime("%d/%m/%Y")
    return source_frame([[date, f"Home{index}", f"Away{index}", "1", "0"] for index, date in enumerate(dates)])


def baseline_rows():
    return source_frame(
        [
            ["01/08/2015", "Alpha", "Beta", "1", "0"],
            ["02/08/2015", "Beta", "Gamma", "0", "0"],
            ["03/08/2015", "Gamma", "Alpha", "2", "1"],
        ]
    )


def write_baseline(root: Path, source=None) -> tuple[Path, pd.DataFrame]:
    source = baseline_rows() if source is None else source
    canonical, _ = convert_source_frame(source, "2015-16")
    processed = root / "processed"
    processed.mkdir(parents=True, exist_ok=True)
    canonical_path = processed / "epl_canonical.csv"
    canonical.to_csv(canonical_path, index=False)
    write_json(
        {
            "schema_version": "m2-processed-manifest-v1",
            "adapter_version": "m2-epl-football-data-v1",
            "output": {"filename": str(canonical_path), "sha256": file_sha256(canonical_path), "rows": len(canonical)},
        },
        processed / "epl_processed_manifest.json",
    )
    return canonical_path, canonical


def write_stage(root: Path, source: pd.DataFrame, season: str = "2025-26", name: str = "stage") -> Path:
    stage = root / name
    raw = stage / "raw"
    raw.mkdir(parents=True)
    raw_path = raw / f"E0_{season}.csv"
    source.to_csv(raw_path, index=False)
    write_json(
        {
            "schema_version": "m2.1-source-manifest-v1",
            "adapter_version": "m2.1-epl-football-data-v1",
            "source": "Football-Data.co.uk",
            "competition": "English Premier League",
            "division": "E0",
            "snapshot_id": name,
            "generated_at": "2026-01-01T00:00:00Z",
            "requested_seasons": [season],
            "source_urls": [f"https://example.invalid/{season}/E0.csv"],
            "data_cutoff": None,
            "status": "ok",
            "files": [
                {
                    "season": season,
                    "url": f"https://example.invalid/{season}/E0.csv",
                    "retrieved_at": "2026-01-01T00:00:00Z",
                    "raw_filename": raw_path.name,
                    "sha256": file_sha256(raw_path),
                    "download_status": "ok",
                    "parse_status": "ok",
                }
            ],
        },
        raw / "source_manifest.json",
    )
    return stage


class FakeResponse:
    def __init__(self, body: bytes):
        self.body = body
        self.headers = {"Content-Type": "text/csv"}

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def getcode(self):
        return 200

    def read(self):
        return self.body


class EplSyncTests(unittest.TestCase):
    def test_unfinished_fixture_is_skipped_and_reported(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_baseline(root)
            stage = write_stage(
                root,
                source_frame(
                    [
                        ["01/08/2025", "Delta", "Epsilon", "2", "1"],
                        ["02/08/2025", "Epsilon", "Zeta", "", ""],
                    ]
                ),
            )
            result = sync_epl(staging_dir=stage, data_dir=root, in_progress_season="2025-26")
            self.assertEqual(result["status"], "dry-run")
            self.assertEqual(result["skipped_unfinished"], 1)
            self.assertEqual(result["diff"]["skipped_unfinished"], 1)
            quality = json.loads(Path(result["quality_report_path"]).read_text(encoding="utf-8"))
            diff = json.loads(Path(result["diff_report_path"]).read_text(encoding="utf-8"))
            self.assertEqual(quality["seasons"][0]["skipped_unfinished"][0]["source_row"], 3)
            self.assertEqual(diff["details"]["skipped_unfinished"][0]["reason"], "both full-time score fields are empty")
            self.assertEqual(len(load_matches(result["candidate_path"])), 4)
            self.assertFalse(result["strict_380_gate_applied_by_season"]["2025-26"])

    def test_incomplete_historical_season_fails_without_declaration(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_baseline(root)
            stage = write_stage(root, source_frame([["01/08/2014", "Delta", "Epsilon", "1", "0"]]), season="2014-15")
            with self.assertRaisesRegex(DataValidationError, "validation failed"):
                sync_epl(staging_dir=stage, data_dir=root, apply=True)
            quality = json.loads((stage / "reports" / "epl_quality_report.json").read_text(encoding="utf-8"))
            season = quality["seasons"][0]
            self.assertEqual(season["status"], "failed")
            self.assertIn("incomplete_season", [error["kind"] for error in season["errors"]])
            self.assertTrue(season["strict_380_gate_applied"])
            self.assertEqual(quality["season_policy"]["2014-15"]["match_count_policy"], "strict_complete_season")
            self.assertFalse((root / "processed" / "epl_current.json").exists())
            self.assertFalse((root / "processed" / "versions").exists())

    def test_incomplete_season_requires_explicit_in_progress_declaration(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_baseline(root)
            stage = write_stage(root, source_frame([["01/08/2025", "Delta", "Epsilon", "1", "0"]]))
            with self.assertRaisesRegex(DataValidationError, "validation failed"):
                sync_epl(staging_dir=stage, data_dir=root)
            result = sync_epl(staging_dir=stage, data_dir=root, apply=True, in_progress_season="2025-26")
            self.assertEqual(result["status"], "applied")
            quality = json.loads(Path(result["quality_report_path"]).read_text(encoding="utf-8"))
            policy = quality["season_policy"]["2025-26"]
            self.assertEqual(policy["declared_in_progress_season"], "2025-26")
            self.assertFalse(policy["strict_380_gate_applied"])
            self.assertEqual(policy["match_count_policy"], "in_progress_exception")

    def test_in_progress_declaration_must_be_requested(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_baseline(root)
            stage = write_stage(root, source_frame([["01/08/2025", "Delta", "Epsilon", "1", "0"]]))
            with self.assertRaisesRegex(ValueError, "must be one of requested seasons"):
                sync_epl(staging_dir=stage, data_dir=root, in_progress_season="2024-25")
            self.assertFalse((root / "processed" / "epl_current.json").exists())
            self.assertFalse((root / "processed" / "versions").exists())

    def test_multi_season_gap_fill_applies_policy_per_season(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_baseline(root)
            complete_body = complete_source_frame("2024-08-01").to_csv(index=False).encode("utf-8")
            in_progress_body = source_frame(
                [["01/08/2025", "Delta", "Epsilon", "1", "0"], ["02/08/2025", "Epsilon", "Zeta", "", ""]]
            ).to_csv(index=False).encode("utf-8")
            with patch(
                "matchquant.epl_sync_source.urllib.request.urlopen",
                side_effect=[FakeResponse(complete_body), FakeResponse(in_progress_body)],
            ) as mocked:
                result = sync_epl(
                    seasons=["2024-25", "2025-26"],
                    data_dir=root,
                    in_progress_season="2025-26",
                )
            self.assertEqual(mocked.call_count, 2)
            self.assertEqual(result["status"], "dry-run")
            self.assertEqual(result["diff"]["added"], 381)
            self.assertTrue(result["strict_380_gate_applied_by_season"]["2024-25"])
            self.assertFalse(result["strict_380_gate_applied_by_season"]["2025-26"])
            source_manifest = json.loads((Path(result["staging_dir"]) / "raw" / "source_manifest.json").read_text(encoding="utf-8"))
            for path in (
                Path(result["quality_report_path"]),
                Path(result["diff_report_path"]),
                Path(result["processed_manifest_path"]),
            ):
                evidence = json.loads(path.read_text(encoding="utf-8"))
                self.assertTrue(evidence["season_policy"]["2024-25"]["strict_380_gate_applied"])
                self.assertFalse(evidence["season_policy"]["2025-26"]["strict_380_gate_applied"])
            self.assertTrue(source_manifest["season_policy"]["2024-25"]["strict_380_gate_applied"])
            self.assertFalse(source_manifest["season_policy"]["2025-26"]["strict_380_gate_applied"])

    def test_partial_score_fails_and_writes_quality_report(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_baseline(root)
            stage = write_stage(root, source_frame([["01/08/2025", "Delta", "Epsilon", "", "1"]]))
            with self.assertRaisesRegex(DataValidationError, "validation failed"):
                sync_epl(staging_dir=stage, data_dir=root)
            quality = json.loads((stage / "reports" / "epl_quality_report.json").read_text(encoding="utf-8"))
            self.assertEqual(quality["status"], "failed")
            self.assertEqual(quality["seasons"][0]["errors"][0]["kind"], "partial_score")
            self.assertFalse((root / "processed" / "epl_current.json").exists())

    def test_diff_classifies_added_unchanged_revised_and_missing(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _, baseline = write_baseline(root)
            stage = write_stage(
                root,
                source_frame(
                    [
                        ["01/08/2015", "Alpha", "Beta", "3", "0"],
                        ["02/08/2015", "Beta", "Gamma", "0", "0"],
                        ["04/08/2015", "Delta", "Alpha", "1", "1"],
                    ]
                ),
                season="2015-16",
            )
            result = sync_epl(staging_dir=stage, data_dir=root, in_progress_season="2015-16")
            self.assertEqual(result["diff"], {"added": 1, "unchanged": 1, "revised": 1, "missing_from_candidate": 1, "skipped_unfinished": 0})
            revised = json.loads(Path(result["diff_report_path"]).read_text(encoding="utf-8"))["details"]["revised"][0]
            self.assertEqual(revised["changed_fields"], ["home_goals"])
            display_changed = baseline.copy()
            display_changed.loc[0, "home_team"] = "Alpha FC"
            self.assertEqual(compare_canonical(baseline, display_changed)["counts"]["revised"], 1)

    def test_dry_run_preserves_baseline_and_pointer(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            canonical_path, _ = write_baseline(root)
            before = canonical_path.read_bytes()
            stage = write_stage(root, source_frame([["01/08/2025", "Delta", "Epsilon", "1", "0"]]))
            result = sync_epl(staging_dir=stage, data_dir=root, in_progress_season="2025-26")
            self.assertEqual(result["status"], "dry-run")
            self.assertEqual(canonical_path.read_bytes(), before)
            self.assertFalse((root / "processed" / "epl_current.json").exists())

    def test_apply_creates_version_pointer_and_hashes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            canonical_path, _ = write_baseline(root)
            before = canonical_path.read_bytes()
            stage = write_stage(root, source_frame([["01/08/2025", "Delta", "Epsilon", "1", "0"]]))
            result = sync_epl(staging_dir=stage, data_dir=root, apply=True, in_progress_season="2025-26")
            self.assertEqual(result["status"], "applied")
            version_dir = Path(result["version_dir"])
            pointer_path = Path(result["pointer_path"])
            self.assertTrue(all((version_dir / name).is_file() for name in ("epl_canonical.csv", "epl_processed_manifest.json", "epl_quality_report.json", "epl_sync_diff_report.json")))
            pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
            self.assertEqual(pointer["dataset_version"], result["version_id"])
            self.assertEqual(pointer["canonical_sha256"], file_sha256(pointer["canonical"]))
            manifest = json.loads((version_dir / "epl_processed_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["output"]["sha256"], file_sha256(version_dir / "epl_canonical.csv"))
            self.assertEqual(manifest["quality_report_sha256"], file_sha256(version_dir / "epl_quality_report.json"))
            self.assertEqual(manifest["diff_report_sha256"], file_sha256(version_dir / "epl_sync_diff_report.json"))
            self.assertEqual(canonical_path.read_bytes(), before)

    def test_revision_or_missing_blocks_apply(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_baseline(root)
            stage = write_stage(
                root,
                source_frame([["01/08/2015", "Alpha", "Beta", "3", "0"]]),
                season="2015-16",
            )
            result = sync_epl(staging_dir=stage, data_dir=root, apply=True, in_progress_season="2015-16")
            self.assertEqual(result["status"], "blocked")
            self.assertEqual(set(result["apply_blockers"]), {"revised", "missing_from_candidate"})
            self.assertEqual(json.loads(Path(result["diff_report_path"]).read_text(encoding="utf-8"))["apply"]["status"], "blocked_manual_review")
            self.assertFalse((root / "processed" / "epl_current.json").exists())
            self.assertFalse((root / "processed" / "versions").exists())

    def test_invalid_date_duplicate_and_abnormal_score_fail(self):
        cases = {
            "bad_date": [["2015/08/01", "Delta", "Epsilon", "1", "0"]],
            "duplicate": [["01/08/2025", "Delta", "Epsilon", "1", "0"], ["02/08/2025", "Delta", "Epsilon", "2", "1"]],
            "negative": [["01/08/2025", "Delta", "Epsilon", "-1", "0"]],
        }
        for name, rows in cases.items():
            with self.subTest(case=name), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                write_baseline(root)
                stage = write_stage(root, source_frame(rows), name=name)
                with self.assertRaisesRegex(DataValidationError, "validation failed"):
                    sync_epl(staging_dir=stage, data_dir=root)
                quality = json.loads((stage / "reports" / "epl_quality_report.json").read_text(encoding="utf-8"))
                self.assertEqual(quality["status"], "failed")
                self.assertTrue(quality["seasons"][0]["errors"])

    def test_mock_network_creates_unique_staging_snapshot(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_baseline(root)
            legacy_raw = root / "raw"
            legacy_raw.mkdir()
            legacy_path = legacy_raw / "E0_2025-26.csv"
            legacy_path.write_text("legacy", encoding="utf-8")
            legacy_bytes = legacy_path.read_bytes()
            body = source_frame([["01/08/2025", "Delta", "Epsilon", "1", "0"], ["02/08/2025", "Epsilon", "Zeta", "", ""]]).to_csv(index=False).encode("utf-8")
            with patch("matchquant.epl_sync_source.urllib.request.urlopen", return_value=FakeResponse(body)) as mocked:
                result = sync_epl(seasons=["2025-26"], data_dir=root, timeout=5, in_progress_season="2025-26")
            mocked.assert_called_once()
            staging = Path(result["staging_dir"])
            self.assertTrue(staging.parent.samefile(root / "staging"))
            self.assertTrue(staging.name.startswith("epl_sync_"))
            self.assertEqual((staging / "raw" / "E0_2025-26.csv").read_bytes(), body)
            self.assertEqual(legacy_path.read_bytes(), legacy_bytes)
            self.assertEqual(result["diff"]["skipped_unfinished"], 1)
            source_manifest = json.loads((staging / "raw" / "source_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(source_manifest["declared_in_progress_season"], "2025-26")
            self.assertFalse(source_manifest["season_policy"]["2025-26"]["strict_380_gate_applied"])

    def test_offline_cli_smoke(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_baseline(root)
            stage = write_stage(root, source_frame([["01/08/2025", "Delta", "Epsilon", "1", "0"]]))
            completed = subprocess.run(
                [sys.executable, "-m", "matchquant", "epl-sync", "--staging-dir", str(stage), "--data-dir", str(root), "--in-progress-season", "2025-26"],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertLessEqual(len(completed.stdout.strip().splitlines()), 20)
            self.assertIn("mode=dry-run", completed.stdout)


if __name__ == "__main__":
    unittest.main()
