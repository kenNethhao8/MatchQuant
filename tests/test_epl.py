import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from matchquant.epl import (
    audit_source_frame,
    convert_source_frame,
    download_epl,
    prepare_epl,
    stable_match_id,
)
from matchquant.io import file_sha256, load_matches, write_json
from matchquant.schema import DataValidationError, validate_matches


def fixture_source() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Date": ["08/09/2015", "09/10/2015", "10/11/2015"],
            "HomeTeam": ["Alpha", "Beta", "Gamma"],
            "AwayTeam": ["Beta", "Gamma", "Alpha"],
            "FTHG": [2, 0, 1],
            "FTAG": [1, 0, 3],
        }
    )


def complete_fixture_source() -> pd.DataFrame:
    teams = [f"Team{index:02d}" for index in range(20)]
    start = pd.Timestamp("2015-08-01")
    rows = []
    for home_index, home_team in enumerate(teams):
        for away_index, away_team in enumerate(teams):
            if home_index == away_index:
                continue
            rows.append(
                {
                    "Date": (start + pd.Timedelta(days=len(rows))).strftime("%d/%m/%Y"),
                    "HomeTeam": home_team,
                    "AwayTeam": away_team,
                    "FTHG": home_index % 4,
                    "FTAG": away_index % 3,
                }
            )
    return pd.DataFrame(rows)


def write_snapshot(root: Path, source: pd.DataFrame, season: str = "2015-16") -> Path:
    raw_dir = root / "raw"
    raw_dir.mkdir(parents=True)
    raw_path = raw_dir / f"E0_{season}.csv"
    source.to_csv(raw_path, index=False)
    write_json(
        {
            "schema_version": "m2-source-manifest-v1",
            "adapter_version": "m2-epl-football-data-v1",
            "source": "Football-Data.co.uk",
            "competition": "English Premier League",
            "division": "E0",
            "generated_at": "2026-01-01T00:00:00Z",
            "requested_seasons": [season],
            "status": "ok",
            "files": [
                {
                    "season": season,
                    "url": "https://example.invalid/E0.csv",
                    "retrieved_at": "2026-01-01T00:00:00Z",
                    "raw_filename": raw_path.name,
                    "sha256": file_sha256(raw_path),
                    "http_status": 200,
                    "parse_status": "ok",
                    "status": "ok",
                }
            ],
        },
        raw_dir / "source_manifest.json",
    )
    return raw_path


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


class EplAdapterTests(unittest.TestCase):
    def test_mapping_uses_strict_day_month_dates_and_m1_validation(self):
        canonical, report = convert_source_frame(fixture_source(), "2015-16")
        self.assertEqual(canonical["date"].tolist(), ["2015-09-08", "2015-10-09", "2015-11-10"])
        self.assertEqual(canonical["season"].unique().tolist(), ["2015-16"])
        self.assertEqual(canonical["home_goals"].tolist(), [2, 0, 1])
        self.assertEqual(report["invalid_rows"], 0)
        self.assertEqual(report["team_count"], 3)
        self.assertEqual(len(validate_matches(canonical)), 3)

        short_dates = fixture_source()
        short_dates["Date"] = ["08/09/15", "09/10/15", "10/11/15"]
        short_canonical, short_report = convert_source_frame(short_dates, "2015-16")
        self.assertEqual(short_canonical["date"].tolist(), canonical["date"].tolist())
        self.assertEqual(short_report["source_date_format"], "%d/%m/%y")

    def test_stable_id_survives_date_and_score_revisions(self):
        original, _ = convert_source_frame(fixture_source(), "2015-16")
        revised_source = fixture_source()
        revised_source.loc[0, "Date"] = "10/09/2015"
        revised_source.loc[0, "FTHG"] = 9
        revised_source.loc[0, "FTAG"] = 0
        revised, _ = convert_source_frame(revised_source, "2015-16")
        self.assertEqual(original.loc[0, "match_id"], revised.loc[0, "match_id"])
        self.assertEqual(revised.loc[0, "home_team"], "Alpha")
        self.assertEqual(revised.loc[0, "away_team"], "Beta")
        self.assertEqual(stable_match_id("1516", " Alpha ", " Beta"), original.loc[0, "match_id"])
        self.assertNotEqual(stable_match_id("2015-16", "Alpha", "Beta"), stable_match_id("2015-16", "Beta", "Alpha"))
        self.assertNotEqual(stable_match_id("2015-16", "Alpha", "Beta"), stable_match_id("2016-17", "Alpha", "Beta"))

    def test_stable_id_does_not_depend_on_source_row_order(self):
        first, _ = convert_source_frame(fixture_source(), "2015-16")
        second, _ = convert_source_frame(fixture_source().iloc[::-1], "2015-16")
        self.assertEqual(set(first["match_id"]), set(second["match_id"]))

    def test_missing_column_and_illegal_score_are_explicit_errors(self):
        missing = fixture_source().drop(columns=["FTHG"])
        report = audit_source_frame(missing, "2015-16")
        self.assertEqual(report["status"], "failed")
        self.assertIn("FTHG", report["warnings"][0])
        with self.assertRaisesRegex(DataValidationError, "missing required source columns"):
            convert_source_frame(missing, "2015-16")

        illegal = fixture_source()
        illegal["FTHG"] = illegal["FTHG"].astype("string")
        illegal.loc[0, "FTHG"] = "1.5"
        report = audit_source_frame(illegal, "2015-16")
        self.assertEqual(report["invalid_score_rows"], 1)
        with self.assertRaisesRegex(DataValidationError, "invalid EPL source season"):
            convert_source_frame(illegal, "2015-16")

    def test_quality_report_counts_duplicates_and_incomplete(self):
        duplicated = pd.concat([fixture_source(), fixture_source().iloc[[0]]], ignore_index=True)
        report = audit_source_frame(duplicated, "2015-16")
        self.assertEqual(report["duplicate_rows"], 2)
        self.assertEqual(report["status"], "failed")
        self.assertFalse(report["meets_expected_matches"])
        self.assertTrue(any("380" in warning for warning in report["warnings"]))

    def test_download_manifest_records_audit_failure_and_preserves_raw_bytes(self):
        with tempfile.TemporaryDirectory() as temporary:
            body = fixture_source().to_csv(index=False).encode("utf-8")
            with patch("matchquant.epl.urllib.request.urlopen", return_value=FakeResponse(body)):
                with self.assertRaisesRegex(DataValidationError, "download failed"):
                    download_epl(["2015-16"], temporary)
            root = Path(temporary)
            raw_path = root / "raw" / "E0_2015-16.csv"
            manifest = json.loads((root / "raw" / "source_manifest.json").read_text(encoding="utf-8"))
            entry = manifest["files"][0]
            self.assertEqual(raw_path.read_bytes(), body)
            self.assertEqual(entry["status"], "failed")
            self.assertEqual(entry["parse_status"], "ok")
            self.assertEqual(entry["audit_status"], "warning")
            self.assertEqual(entry["audit_match_count"], 3)
            self.assertEqual(entry["source_date_format"], "%d/%m/%Y")
            self.assertFalse(entry["meets_expected_matches"])

    def test_download_does_not_overwrite_an_existing_raw_snapshot(self):
        with tempfile.TemporaryDirectory() as temporary:
            raw_dir = Path(temporary) / "raw"
            raw_dir.mkdir(parents=True)
            raw_path = raw_dir / "E0_2015-16.csv"
            raw_path.write_text("immutable", encoding="utf-8")
            with self.assertRaisesRegex(DataValidationError, "immutable"):
                download_epl(["2015-16"], temporary)
            self.assertEqual(raw_path.read_text(encoding="utf-8"), "immutable")

    def test_prepare_rejects_incomplete_source_without_canonical_or_processed_manifest(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_snapshot(root, fixture_source())
            output = root / "processed" / "epl_canonical.csv"
            processed = root / "processed" / "epl_processed_manifest.json"
            with self.assertRaisesRegex(DataValidationError, "complete-season 380-match gate"):
                prepare_epl(["2015-16"], root, output, root / "reports" / "quality.json", processed)
            self.assertFalse(output.exists())
            self.assertFalse(processed.exists())
            quality = json.loads((root / "reports" / "quality.json").read_text(encoding="utf-8"))
            self.assertEqual(quality["status"], "failed")
            self.assertEqual(quality["seasons"][0]["match_count"], 3)
            self.assertFalse(quality["seasons"][0]["meets_expected_matches"])

    def test_prepare_records_hashes_for_valid_complete_season_offline(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            raw_path = write_snapshot(root, complete_fixture_source())
            result = prepare_epl(["2015-16"], root)
            output = Path(result["output_path"])
            self.assertTrue(output.is_file())
            self.assertEqual(result["status"], "ok")
            processed = json.loads(Path(result["processed_manifest_path"]).read_text(encoding="utf-8"))
            quality = json.loads(Path(result["quality_report_path"]).read_text(encoding="utf-8"))
            self.assertEqual(processed["inputs"][0]["sha256"], file_sha256(raw_path))
            self.assertEqual(processed["output"]["sha256"], file_sha256(output))
            self.assertEqual(processed["output"]["rows"], 380)
            self.assertEqual(quality["seasons"][0]["match_count"], 380)
            self.assertEqual(len(load_matches(output)), 380)


if __name__ == "__main__":
    unittest.main()
