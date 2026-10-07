import contextlib
import io as io_module
import json
import subprocess
import sys
import tempfile
import unittest
import unittest.mock as mock
from pathlib import Path

from matchquant import launcher
from matchquant.cli import _parser


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


class CliVersionAndDemoTests(unittest.TestCase):
    def test_version_flag_reads_installed_metadata(self):
        command = [sys.executable, "-m", "matchquant", "--version"]
        completed = subprocess.run(command, capture_output=True, text=True, check=False)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        stdout = completed.stdout
        self.assertIn("matchquant 0.1.0", stdout)
        self.assertIn("baseline-v0.1 research", stdout)
        self.assertLessEqual(len(stdout.strip().splitlines()), 3)

    def test_demo_export_cli_succeeds_then_refuses_reexport(self):
        with tempfile.TemporaryDirectory() as temporary:
            target = Path(temporary) / "demo out"
            command = [
                sys.executable,
                "-m",
                "matchquant",
                "demo-export",
                "--output-dir",
                str(target),
            ]
            completed = subprocess.run(command, capture_output=True, text=True, check=False)
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertIn("synthetic=true model=frequency", completed.stdout)
            self.assertIn("bundle_verify=ready", completed.stdout)
            self.assertTrue((target / "demo_manifest.json").is_file())
            self.assertTrue((target / "bundle" / "bundle_manifest.json").is_file())
            self.assertTrue((target / "fixtures.csv").is_file())
            self.assertTrue((target / "results.csv").is_file())
            refused = subprocess.run(command, capture_output=True, text=True, check=False)
            self.assertEqual(refused.returncode, 1, refused.stderr)
            self.assertIn("already exists", refused.stderr)
            self.assertEqual(refused.stdout, "")

    def test_help_text_avoids_network_and_alpha_misreadings(self):
        parser = _parser()
        buffer = io_module.StringIO()
        with contextlib.redirect_stdout(buffer):
            with self.assertRaises(SystemExit) as caught:
                parser.parse_args(["--help"])
        self.assertEqual(caught.exception.code, 0)
        # argparse wraps help text after hyphens; undo those joins before matching.
        text = " ".join(buffer.getvalue().replace("-\n", "-").split())
        self.assertIn("demo-export", text)
        self.assertIn("--version", text)
        self.assertIn("research", text)
        self.assertIn("offline walk-forward backtest", text)
        self.assertIn("not the network", text)
        self.assertNotIn("run an online walk-forward backtest", text)
        self.assertIn("not live data, not an Alpha/factor engine", text)
        self.assertIn("not calibrated probabilities", text)


class LauncherDiagnosticsTests(unittest.TestCase):
    def test_missing_core_dependency_gets_actionable_guidance(self):
        with mock.patch.object(
            launcher,
            "_import_cli",
            side_effect=ModuleNotFoundError("No module named 'pandas'", name="pandas"),
        ):
            with contextlib.redirect_stderr(io_module.StringIO()) as stderr:
                with self.assertRaises(SystemExit) as caught:
                    launcher.main()
        self.assertEqual(caught.exception.code, 1)
        text = stderr.getvalue()
        self.assertIn("pandas", text)
        self.assertIn("venv", text)
        self.assertIn("--no-index --find-links", text)

    def test_unknown_import_error_is_reraised_not_swallowed(self):
        with mock.patch.object(
            launcher,
            "_import_cli",
            side_effect=ModuleNotFoundError("No module named 'someother'", name="someother"),
        ):
            with self.assertRaises(ModuleNotFoundError):
                launcher.main()


if __name__ == "__main__":
    unittest.main()

