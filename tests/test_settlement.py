import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from matchquant.forecast import run_forecast
from matchquant.io import file_sha256, load_matches, write_json
from matchquant.schema import DataValidationError, REQUIRED_COLUMNS
from matchquant.settlement import settle_forecast_run


class SettlementTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.history_path = Path("examples/synthetic_matches.csv").resolve()
        cls.history = load_matches(cls.history_path)

    def _forecast(self, root: Path, as_of: str = "2024-01-07") -> Path:
        root.mkdir(parents=True, exist_ok=True)
        config_path = root / "frequency.json"
        config_path.write_text('{"alpha": 1.0}\n', encoding="utf-8")
        result = run_forecast(
            self.history_path,
            Path("examples/synthetic_fixtures.csv").resolve(),
            as_of,
            model_name="frequency",
            model_config_path=config_path,
            output_root=root / "forecast-runs",
        )
        return Path(result["run_dir"])

    def _write_results(self, path: Path, frame: pd.DataFrame) -> Path:
        frame.loc[:, REQUIRED_COLUMNS].to_csv(path, index=False)
        return path

    def _rows(self, *match_ids: str) -> pd.DataFrame:
        return self.history.loc[self.history["match_id"].isin(match_ids), REQUIRED_COLUMNS].copy()

    def _settle(self, root: Path, results: pd.DataFrame, *, as_of: str = "2024-01-07") -> dict:
        forecast_dir = self._forecast(root, as_of=as_of)
        results_path = self._write_results(root / "results.csv", results)
        return settle_forecast_run(forecast_dir, results_path, output_root=root / "settlement-runs")

    def test_complete_settlement_metrics_and_input_evidence(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            forecast_dir = self._forecast(root)
            results_path = self._write_results(root / "results.csv", self.history)
            forecast_hashes = {
                "config": file_sha256(forecast_dir / "run_config.json"),
                "predictions": file_sha256(forecast_dir / "predictions.csv"),
                "results": file_sha256(results_path),
            }
            result = settle_forecast_run(
                forecast_dir, results_path, output_root=root / "settlement-runs"
            )
            self.assertEqual(result["status"], "complete")
            self.assertEqual(result["counts"]["settled"], 3)
            self.assertEqual(result["metrics"]["coverage"], 1.0)
            self.assertIsNotNone(result["metrics"]["log_loss"])
            self.assertTrue((Path(result["run_dir"]) / "settlement.csv").is_file())
            config = json.loads(Path(result["run_config_path"]).read_text(encoding="utf-8"))
            self.assertEqual(config["forecast_run"]["run_config"]["sha256"], forecast_hashes["config"])
            self.assertEqual(config["forecast_run"]["predictions"]["sha256"], forecast_hashes["predictions"])
            self.assertEqual(config["forecast_run"]["predictions"]["hash_evidence_status"], "verified")
            self.assertEqual(config["results"]["sha256"], forecast_hashes["results"])
            self.assertEqual(file_sha256(forecast_dir / "predictions.csv"), forecast_hashes["predictions"])
            self.assertEqual(file_sha256(forecast_dir / "run_config.json"), forecast_hashes["config"])
            self.assertEqual(file_sha256(results_path), forecast_hashes["results"])

    def test_partial_and_zero_settlement_keep_null_metrics(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            partial = self._settle(root / "partial", self._rows("m008"))
            self.assertEqual(partial["status"], "partial")
            self.assertEqual(partial["counts"]["settled"], 1)
            self.assertEqual(partial["counts"]["unresolved"], 2)
            self.assertAlmostEqual(partial["metrics"]["coverage"], 1 / 3)

            zero = self._settle(root / "zero", self._rows("m001"))
            self.assertEqual(zero["status"], "partial")
            self.assertEqual(zero["counts"]["settled"], 0)
            self.assertEqual(zero["metrics"]["coverage"], 0.0)
            self.assertIsNone(zero["metrics"]["log_loss"])
            metrics = json.loads(Path(zero["metrics_path"]).read_text(encoding="utf-8"))
            self.assertIsNone(metrics["metrics"]["brier_score"])

    def test_duplicate_result_identity_is_ambiguous_and_excluded(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            duplicate = self._rows("m008")
            second = duplicate.copy()
            second.loc[:, "match_id"] = "m008-revision"
            second.loc[:, "date"] = "2024-01-08"
            result = self._settle(root, pd.concat([duplicate, second], ignore_index=True))
            row = result["settlement"].loc[result["settlement"]["fixture_id"].eq("f001")].iloc[0]
            self.assertEqual(result["status"], "needs_review")
            self.assertEqual(row["settlement_status"], "ambiguous")
            self.assertEqual(row["result_match_id"], "")
            self.assertEqual(row["result_match_ids"], "m008|m008-revision")
            self.assertEqual(result["counts"]["settled"], 0)

    def test_unique_rescheduled_result_is_settled_as_date_changed(self):
        with tempfile.TemporaryDirectory() as temporary:
            results = self._rows("m008")
            results.loc[:, "date"] = "2024-01-09"
            result = self._settle(Path(temporary), results)
            row = result["settlement"].loc[result["settlement"]["fixture_id"].eq("f001")].iloc[0]
            self.assertEqual(row["settlement_status"], "settled")
            self.assertEqual(row["date_status"], "date_changed")
            self.assertEqual(row["actual_result"], "H")

    def test_as_of_conflict_and_earlier_reschedule_are_handled_safely(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            before_as_of = self._rows("m008")
            before_as_of.loc[:, "date"] = "2024-01-06"
            result = self._settle(root / "before", before_as_of)
            row = result["settlement"].loc[result["settlement"]["fixture_id"].eq("f001")].iloc[0]
            self.assertEqual(row["settlement_status"], "time_conflict")
            self.assertEqual(row["reason"], "result_before_forecast_as_of")
            self.assertEqual(row["actual_result"], "H")
            self.assertIsNone(result["metrics"]["log_loss"])

            earlier_reschedule = self._rows("m010")
            earlier_reschedule.loc[:, "date"] = "2024-01-07"
            result = self._settle(root / "earlier", earlier_reschedule)
            row = result["settlement"].loc[result["settlement"]["fixture_id"].eq("f003")].iloc[0]
            self.assertEqual(row["settlement_status"], "settled")
            self.assertEqual(row["reason"], "unique_exact_identity_match")
            self.assertEqual(row["date_status"], "date_changed")
            self.assertEqual(row["actual_result"], "D")
            self.assertIsNotNone(result["metrics"]["log_loss"])

    def test_legacy_hash_evidence_is_explicit(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            original = self._forecast(root)
            legacy = root / "legacy-run"
            shutil.copytree(original, legacy)
            config_path = legacy / "run_config.json"
            config = json.loads(config_path.read_text(encoding="utf-8"))
            config.pop("predictions_sha256", None)
            config["outputs"].pop("predictions_sha256", None)
            write_json(config, config_path)
            result = settle_forecast_run(
                legacy,
                self._write_results(root / "results.csv", self.history),
                output_root=root / "settlement-runs",
            )
            self.assertEqual(
                result["config"]["forecast_run"]["predictions"]["hash_evidence_status"],
                "unrecorded_legacy",
            )
            self.assertNotIn("predictions_sha256", config)

    def test_recorded_hash_mismatch_and_invalid_predictions_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            forecast_dir = self._forecast(root)
            predictions_path = forecast_dir / "predictions.csv"
            predictions = pd.read_csv(predictions_path)
            predictions.loc[0, "home_win_prob"] = 0.9
            predictions.to_csv(predictions_path, index=False)
            with self.assertRaisesRegex(DataValidationError, "SHA-256"):
                settle_forecast_run(
                    forecast_dir,
                    self._write_results(root / "results.csv", self.history),
                    output_root=root / "settlement-runs",
                )

    def test_m6_refuses_to_create_outputs_under_active_data_root(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            forecast_dir = self._forecast(root)
            results_path = self._write_results(root / "results.csv", self.history)
            blocked_root = Path(r"D:\Codex\matchquant_data") / "m6-test-output"
            with self.assertRaisesRegex(DataValidationError, "must not write under"):
                settle_forecast_run(forecast_dir, results_path, output_root=blocked_root)

    def test_cli_m4_to_m6_smoke_is_offline_and_concise(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config_path = root / "frequency.json"
            config_path.write_text('{"alpha": 1.0}\n', encoding="utf-8")
            forecast_root = root / "forecast-runs"
            forecast_command = [
                sys.executable, "-m", "matchquant", "forecast",
                "--history", str(self.history_path),
                "--fixtures", str(Path("examples/synthetic_fixtures.csv").resolve()),
                "--as-of", "2024-01-07", "--model", "frequency",
                "--model-config", str(config_path), "--output-root", str(forecast_root),
            ]
            forecast_process = subprocess.run(forecast_command, capture_output=True, text=True, check=False)
            self.assertEqual(forecast_process.returncode, 0, forecast_process.stderr)
            forecast_dir = next(forecast_root.glob("m4_forecast_*"))
            settle_process = subprocess.run(
                [
                    sys.executable, "-m", "matchquant", "settle",
                    "--forecast-run", str(forecast_dir),
                    "--results", str(self.history_path),
                    "--output-root", str(root / "settlement-runs"),
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(settle_process.returncode, 0, settle_process.stderr)
            self.assertLessEqual(len(settle_process.stdout.strip().splitlines()), 20)
            self.assertEqual(len(list((root / "settlement-runs").glob("m6_settlement_*"))), 1)


if __name__ == "__main__":
    unittest.main()
