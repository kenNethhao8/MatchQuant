import json
import os
import stat
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from matchquant.epl import convert_source_frame
from matchquant.epl_local_stage import stage_local_epl_source
from matchquant.epl_sync import sync_epl
from matchquant.io import file_sha256, write_json
from matchquant.schema import DataValidationError


SOURCE_COLUMNS = ["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG"]


def source_frame(rows):
    return pd.DataFrame(rows, columns=SOURCE_COLUMNS)


def write_baseline(root: Path) -> None:
    source = source_frame(
        [
            ["01/08/2015", "Alpha", "Beta", "1", "0"],
            ["02/08/2015", "Beta", "Gamma", "0", "0"],
            ["03/08/2015", "Gamma", "Alpha", "2", "1"],
        ]
    )
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


class LocalEplStageTests(unittest.TestCase):
    def test_valid_snapshot_preserves_bytes_and_records_true_provenance(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "input" / "my_export.csv"
            source.parent.mkdir()
            source.write_bytes(b"Date,HomeTeam,AwayTeam,FTHG,FTAG\r\n01/08/2025,A,B,1,0\r\n")
            before = source.read_bytes()
            result = stage_local_epl_source(
                source,
                "2025-26",
                root / "data",
                source_url="https://example.invalid/export.csv",
                source_label="manual export",
                upstream_retrieved_at="2026-09-05T08:30:00Z",
            )
            manifest = json.loads(Path(result["manifest_path"]).read_text(encoding="utf-8"))
            raw = Path(result["raw_path"])
            entry = manifest["files"][0]
            self.assertEqual(result["status"], "unverified")
            self.assertEqual(raw.read_bytes(), before)
            self.assertEqual(source.read_bytes(), before)
            self.assertEqual(manifest["source_path"], str(source.resolve()))
            self.assertEqual(manifest["source_url"], "https://example.invalid/export.csv")
            self.assertEqual(manifest["source_label"], "manual export")
            self.assertEqual(manifest["upstream_retrieved_at"], "2026-09-05T08:30:00Z")
            self.assertEqual(manifest["upstream_retrieved_at_status"], "user_provided")
            self.assertEqual(entry["retrieved_at"], "2026-09-05T08:30:00Z")
            self.assertEqual(entry["local_archived_at"], manifest["local_archived_at"])
            self.assertEqual(file_sha256(raw), manifest["files"][0]["sha256"])
            self.assertFalse(raw.stat().st_mode & stat.S_IWRITE)
            self.assertFalse((Path(result["snapshot_dir"]) / "processed").exists())
            self.assertFalse((root / "data" / "processed" / "epl_current.json").exists())
            self.assertFalse((root / "data" / "processed" / "versions").exists())

    def test_no_url_does_not_create_fake_url_or_retrieved_time(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "E0.csv"
            source.write_text("Date,HomeTeam,AwayTeam,FTHG,FTAG\n01/08/2025,A,B,1,0\n", encoding="utf-8")
            result = stage_local_epl_source(source, "2025-26", root / "data")
            manifest = json.loads(Path(result["manifest_path"]).read_text(encoding="utf-8"))
            self.assertEqual(manifest["source_urls"], [])
            self.assertIsNone(manifest["upstream_retrieved_at"])
            self.assertEqual(manifest["upstream_retrieved_at_status"], "unknown")
            self.assertIsNone(manifest["files"][0]["retrieved_at"])

    def test_upstream_timestamp_requires_timezone_and_normalizes_to_utc(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "E0.csv"
            source.write_text("Date,HomeTeam,AwayTeam,FTHG,FTAG\n01/08/2025,A,B,1,0\n", encoding="utf-8")
            result = stage_local_epl_source(
                source,
                "2025-26",
                root / "normalized",
                upstream_retrieved_at="2026-09-05T16:30:00+08:00",
            )
            manifest = json.loads(Path(result["manifest_path"]).read_text(encoding="utf-8"))
            self.assertEqual(manifest["upstream_retrieved_at"], "2026-09-05T08:30:00Z")
            self.assertEqual(manifest["files"][0]["retrieved_at"], "2026-09-05T08:30:00Z")
            for value in ("2026-09-05", "2026-09-05T16:30:00", "not-a-timestamp"):
                with self.subTest(value=value), self.assertRaisesRegex(ValueError, "timezone-aware ISO-8601"):
                    stage_local_epl_source(source, "2025-26", root / f"invalid-{value[:4]}", upstream_retrieved_at=value)

    def test_missing_file_leaves_failure_manifest(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "missing.csv"
            with self.assertRaisesRegex(DataValidationError, "manifest:") as raised:
                stage_local_epl_source(source, "2025-26", root / "data")
            manifest_path = Path(str(raised.exception).split("manifest: ", 1)[1])
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(manifest["status"], "failed")
            self.assertEqual(manifest["validation_status"], "failed")
            self.assertIn("does not exist", manifest["error"])
            self.assertEqual(manifest["files"][0]["download_status"], "failed")

    def test_local_stage_sync_chain_is_offline_and_defers_unfinished_rows(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_baseline(root)
            source = root / "local.csv"
            source_frame(
                [
                    ["01/08/2025", "Delta", "Epsilon", "2", "1"],
                    ["02/08/2025", "Epsilon", "Zeta", "", ""],
                ]
            ).to_csv(source, index=False)
            staged = stage_local_epl_source(source, "2025-26", root, source_label="local file")
            with patch("matchquant.epl_sync_source.urllib.request.urlopen", side_effect=AssertionError("network used")):
                result = sync_epl(
                    staging_dir=staged["snapshot_dir"],
                    data_dir=root,
                    in_progress_season="2025-26",
                )
            self.assertEqual(result["status"], "dry-run")
            self.assertEqual(result["skipped_unfinished"], 1)
            self.assertEqual(result["source"], "local-file")
            self.assertEqual(result["source_label"], "local file")
            self.assertEqual(result["source_urls"], [])
            quality = json.loads(Path(result["quality_report_path"]).read_text(encoding="utf-8"))
            diff = json.loads(Path(result["diff_report_path"]).read_text(encoding="utf-8"))
            processed = json.loads(Path(result["processed_manifest_path"]).read_text(encoding="utf-8"))
            self.assertEqual(quality["source_urls"], [])
            self.assertEqual(diff["source_urls"], [])
            self.assertEqual(quality["upstream_retrieved_at"], [])
            self.assertEqual(diff["upstream_retrieved_at"], [])
            self.assertIsNone(processed["inputs"][0]["upstream_retrieved_at"])
            self.assertEqual(processed["inputs"][0]["upstream_retrieved_at_status"], "unknown")
            self.assertEqual(diff["details"]["skipped_unfinished"][0]["reason"], "both full-time score fields are empty")
            self.assertFalse((root / "processed" / "epl_current.json").exists())
            self.assertFalse((root / "processed" / "versions").exists())

    def test_bad_csv_is_unverified_until_epl_sync_writes_quality_failure(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_baseline(root)
            source = root / "bad.csv"
            source.write_text("Date,HomeTeam,AwayTeam,FTHG\n01/08/2025,A,B,1\n", encoding="utf-8")
            staged = stage_local_epl_source(source, "2025-26", root)
            manifest = json.loads(Path(staged["manifest_path"]).read_text(encoding="utf-8"))
            self.assertEqual(manifest["status"], "unverified")
            with self.assertRaisesRegex(DataValidationError, "quality report"):
                sync_epl(staging_dir=staged["snapshot_dir"], data_dir=root)
            quality = json.loads((Path(staged["snapshot_dir"]) / "reports" / "epl_quality_report.json").read_text(encoding="utf-8"))
            self.assertEqual(quality["status"], "failed")
            self.assertIn("missing required source columns", quality["seasons"][0]["errors"][0]["message"])

    def test_tampered_raw_hash_fails_with_evidence(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_baseline(root)
            source = root / "local.csv"
            source.write_text("Date,HomeTeam,AwayTeam,FTHG,FTAG\n01/08/2025,A,B,1,0\n", encoding="utf-8")
            staged = stage_local_epl_source(source, "2025-26", root)
            raw = Path(staged["raw_path"])
            os.chmod(raw, stat.S_IRUSR | stat.S_IWUSR)
            raw.write_bytes(raw.read_bytes() + b"tampered")
            os.chmod(raw, stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
            with self.assertRaisesRegex(DataValidationError, "quality report"):
                sync_epl(staging_dir=staged["snapshot_dir"], data_dir=root)
            quality = json.loads((Path(staged["snapshot_dir"]) / "reports" / "epl_quality_report.json").read_text(encoding="utf-8"))
            self.assertIn("raw SHA-256 mismatch", quality["errors"][0])

    def test_tampered_manifest_hash_fails_with_evidence(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_baseline(root)
            source = root / "local.csv"
            source.write_text("Date,HomeTeam,AwayTeam,FTHG,FTAG\n01/08/2025,A,B,1,0\n", encoding="utf-8")
            staged = stage_local_epl_source(source, "2025-26", root)
            manifest_path = Path(staged["manifest_path"])
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["files"][0]["sha256"] = "0" * 64
            write_json(manifest, manifest_path)
            with self.assertRaisesRegex(DataValidationError, "quality report"):
                sync_epl(staging_dir=staged["snapshot_dir"], data_dir=root)
            quality = json.loads((Path(staged["snapshot_dir"]) / "reports" / "epl_quality_report.json").read_text(encoding="utf-8"))
            self.assertIn("raw SHA-256 mismatch", quality["errors"][0])

    def test_apply_is_still_explicitly_owned_by_epl_sync(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_baseline(root)
            source = root / "local.csv"
            source.write_text("Date,HomeTeam,AwayTeam,FTHG,FTAG\n01/08/2025,A,B,1,0\n", encoding="utf-8")
            staged = stage_local_epl_source(source, "2025-26", root)
            dry_run = sync_epl(staging_dir=staged["snapshot_dir"], data_dir=root, in_progress_season="2025-26")
            self.assertEqual(dry_run["status"], "dry-run")
            self.assertFalse((root / "processed" / "epl_current.json").exists())
            applied = sync_epl(
                staging_dir=staged["snapshot_dir"],
                data_dir=root,
                apply=True,
                in_progress_season="2025-26",
            )
            self.assertEqual(applied["status"], "applied")
            self.assertTrue(Path(applied["pointer_path"]).is_file())
            self.assertTrue(Path(applied["version_dir"]).is_dir())

    def test_sync_preserves_local_provenance_in_reports_and_version_manifest(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_baseline(root)
            source = root / "local.csv"
            source.write_text("Date,HomeTeam,AwayTeam,FTHG,FTAG\n01/08/2025,A,B,1,0\n", encoding="utf-8")
            staged = stage_local_epl_source(
                source,
                "2025-26",
                root,
                source_url="https://example.invalid/local.csv",
                source_label="manual result export",
                upstream_retrieved_at="2026-09-05T16:30:00+08:00",
            )
            dry_run = sync_epl(staging_dir=staged["snapshot_dir"], data_dir=root, in_progress_season="2025-26")
            processed = json.loads(Path(dry_run["processed_manifest_path"]).read_text(encoding="utf-8"))
            quality = json.loads(Path(dry_run["quality_report_path"]).read_text(encoding="utf-8"))
            diff = json.loads(Path(dry_run["diff_report_path"]).read_text(encoding="utf-8"))
            expected_time = "2026-09-05T08:30:00Z"
            for evidence in (processed, quality, diff):
                self.assertEqual(evidence["source_label"], "manual result export")
                self.assertEqual(evidence["source_type"], "local_file")
                self.assertEqual(evidence["upstream_retrieved_at"], [expected_time])
            season = quality["seasons"][0]
            self.assertEqual(season["source_label"], "manual result export")
            self.assertEqual(season["source_type"], "local_file")
            self.assertEqual(season["upstream_retrieved_at"], expected_time)
            input_record = processed["inputs"][0]
            self.assertEqual(input_record["source_label"], "manual result export")
            self.assertEqual(input_record["source_type"], "local_file")
            self.assertEqual(input_record["upstream_retrieved_at"], expected_time)
            applied = sync_epl(staging_dir=staged["snapshot_dir"], data_dir=root, apply=True, in_progress_season="2025-26")
            version = json.loads(Path(applied["version_manifest"]).read_text(encoding="utf-8"))
            self.assertEqual(version["inputs"][0]["source_label"], "manual result export")
            self.assertEqual(version["inputs"][0]["source_type"], "local_file")
            self.assertEqual(version["inputs"][0]["upstream_retrieved_at"], expected_time)

    def test_cli_chain_smoke_is_concise(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_baseline(root)
            stage = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "matchquant",
                    "epl-local-stage",
                    "--source-csv",
                    "examples/synthetic_e0_source.csv",
                    "--season",
                    "2024-25",
                    "--data-dir",
                    str(root),
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(stage.returncode, 0, stage.stderr)
            self.assertLessEqual(len(stage.stdout.strip().splitlines()), 20)
            snapshot = next(line.split("=", 1)[1] for line in stage.stdout.splitlines() if line.startswith("snapshot="))
            sync = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "matchquant",
                    "epl-sync",
                    "--staging-dir",
                    snapshot,
                    "--data-dir",
                    str(root),
                    "--in-progress-season",
                    "2024-25",
                    "--dry-run",
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(sync.returncode, 0, sync.stderr)
            self.assertLessEqual(len(sync.stdout.strip().splitlines()), 20)
            self.assertIn("source=local-file", sync.stdout)
            self.assertNotIn("source_urls=", sync.stdout)


if __name__ == "__main__":
    unittest.main()
