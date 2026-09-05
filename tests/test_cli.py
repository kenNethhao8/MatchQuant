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

