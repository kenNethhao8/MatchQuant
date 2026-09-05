import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class CliTests(unittest.TestCase):
    def test_all_model_cli_smoke(self):
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
            self.assertLessEqual(len(completed.stdout.strip().splitlines()), 20)
            self.assertTrue((output_dir / "predictions.csv").is_file())
            self.assertTrue((output_dir / "metrics.json").is_file())
            self.assertTrue((output_dir / "run_config.json").is_file())
            metrics = json.loads((output_dir / "metrics.json").read_text(encoding="utf-8"))
            self.assertEqual(set(metrics["models"]), {"frequency", "elo", "poisson"})

    def test_forecast_cli_smoke_creates_unique_auditable_run(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            model_config = root / "frequency.json"
            model_config.write_text('{"alpha": 1.0}\n', encoding="utf-8")
            output_root = root / "forecast-runs"
            command = [
                sys.executable,
                "-m",
                "matchquant",
                "forecast",
                "--history",
                "examples/synthetic_matches.csv",
                "--fixtures",
                "examples/synthetic_fixtures.csv",
                "--as-of",
                "2024-01-07",
                "--model",
                "frequency",
                "--model-config",
                str(model_config),
                "--output-root",
                str(output_root),
            ]
            completed = subprocess.run(command, capture_output=True, text=True, check=False)
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertLessEqual(len(completed.stdout.strip().splitlines()), 20)
            run_dirs = list(output_root.glob("m4_forecast_*"))
            self.assertEqual(len(run_dirs), 1)
            run_dir = run_dirs[0]
            self.assertEqual(
                {path.name for path in run_dir.iterdir()},
                {"predictions.csv", "forecast_report.md", "run_config.json"},
            )
            config = json.loads((run_dir / "run_config.json").read_text(encoding="utf-8"))
            self.assertEqual(config["schema_version"], "m4-forecast-v1")
            self.assertEqual(config["selection_evidence_status"], "explicit_model_config")

