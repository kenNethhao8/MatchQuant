"""Tests for the MatchQuant v2.0 CLI."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from matchquant.cli import build_parser


class CliV2Tests(unittest.TestCase):
    def test_version_flag(self) -> None:
        command = [sys.executable, "-m", "matchquant", "--version"]
        completed = subprocess.run(command, capture_output=True, text=True, check=False)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("matchquant 2.0.0", completed.stdout)

    def test_parser_subcommands_exist(self) -> None:
        parser = build_parser()
        for cmd in ("rich-prepare", "factor-lab", "quant-research", "matchday-radar", "run", "select"):
            parsed = parser.parse_args([cmd] if cmd != "select" else [
                "select", "--input", "x.csv", "--validation-fold", "2022-01-01..2022-06-01", "--final-test-start", "2022-08-01"
            ])
            self.assertEqual(parsed.command, cmd)

    def test_matchday_radar_cli_parser(self) -> None:
        parser = build_parser()
        parsed = parser.parse_args(["matchday-radar", "--input", "test.csv", "--top-n", "5"])
        self.assertEqual(parsed.command, "matchday-radar")
        self.assertEqual(parsed.input, "test.csv")
        self.assertEqual(parsed.top_n, 5)

    def test_single_match_parser_supports_csv_and_direct_inputs(self) -> None:
        parser = build_parser()
        common = [
            "match-report",
            "--input",
            "history.csv",
            "--as-of",
            "2026-05-23",
            "--output-root",
            "runs",
        ]
        csv_args = parser.parse_args([*common, "--fixtures", "fixture.csv", "--mode", "replay"])
        self.assertEqual(csv_args.command, "match-report")
        self.assertEqual(csv_args.fixtures, "fixture.csv")
        direct_args = parser.parse_args(
            [*common, "--home", "Man City", "--away", "Aston Villa", "--date", "2026-05-24", "--season", "2025-26"]
        )
        self.assertEqual(direct_args.home, "Man City")
        self.assertEqual(direct_args.fixture_date, "2026-05-24")
        with self.assertRaises(SystemExit):
            parser.parse_args([*common, "--fixtures", "fixture.csv", "--max-data-age-days", "-1"])

    def test_baseline_run_cli_smoke(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output_dir = Path(temporary) / "run"
            command = [
                sys.executable,
                "-m",
                "matchquant",
                "run",
                "--input",
                "examples/synthetic_matches.csv",
                "--test-start",
                "2024-01-07",
                "--model",
                "all",
                "--output-dir",
                str(output_dir),
            ]
            completed = subprocess.run(command, capture_output=True, text=True, check=False)
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertTrue((output_dir / "predictions.csv").is_file())
            self.assertTrue((output_dir / "metrics.json").is_file())
            metrics = json.loads((output_dir / "metrics.json").read_text(encoding="utf-8"))
            self.assertEqual(set(metrics["models"]), {"frequency", "elo", "poisson"})


if __name__ == "__main__":
    unittest.main()
