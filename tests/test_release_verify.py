import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from matchquant.epl_fixtures import prepare_epl_fixtures
from matchquant.forecast import run_forecast
from matchquant.io import file_sha256, load_matches, write_json
from matchquant.release_verify import verify_release
from matchquant.schema import DataValidationError


class ReleaseVerifyTests(unittest.TestCase):
    history_path = Path("examples/synthetic_matches.csv").resolve()
    source_path = Path("examples/synthetic_e0_source.csv").resolve()

    def _build(self, root: Path, *, selection_mismatch: bool = False) -> dict[str, Path]:
        data_root = root / "fixture-data"
        m5 = prepare_epl_fixtures(self.source_path, "2024-25", data_root)
        model_path = root / "frequency.json"
        model_path.write_text('{"alpha": 1.0}\n', encoding="utf-8")
        fixture_path = Path(m5["fixture_path"])
        kwargs = {"model_name": "frequency", "model_config_path": model_path}
        if selection_mismatch:
            selection_path = root / "selection.json"
            write_json(
                {
                    "schema_version": "m3-selection-v1",
                    "input_sha256": "0" * 64,
                    "season_mode": "carry",
                    "validation_folds": [
                        {"fold_id": "validation_1", "start": "2024-01-03", "end": "2024-01-05"},
                        {"fold_id": "validation_2", "start": "2024-01-06", "end": "2024-01-06"},
                    ],
                    "final_test": {"start": "2024-01-07", "end": "2024-01-10"},
                    "selection_rule": {"final_test_metrics_used_for_selection": False},
                    "selected_candidate": {
                        "candidate_name": "frequency_alpha_1",
                        "model": "frequency",
                        "config": {"alpha": 1.0},
                    },
                },
                selection_path,
            )
            kwargs = {"selection_run_config_path": selection_path}
        forecast = run_forecast(
            self.history_path,
            fixture_path,
            "2024-01-07",
            output_root=root / "forecast-runs",
            **kwargs,
        )
        history = load_matches(self.history_path)
        history_manifest = root / "history_manifest.json"
        write_json(
            {
                "schema_version": "m2.1-processed-manifest-v1",
                "dataset_version": "m2.1-test-2026-09-05",
                "data_cutoff": str(history["date"].max()),
                "output": {
                    "filename": str(self.history_path),
                    "sha256": file_sha256(self.history_path),
                    "rows": len(history),
                    "columns": ["match_id", "date", "season", "home_team", "away_team", "home_goals", "away_goals"],
                },
            },
            history_manifest,
        )
        return {
            "forecast": Path(forecast["run_dir"]),
            "history": self.history_path,
            "history_manifest": history_manifest,
            "fixture_manifest": Path(m5["manifest_path"]),
            "fixture_snapshot": Path(m5["snapshot_dir"]),
            "raw": Path(m5["manifest"]["source_raw_snapshot"]),
            "quality": Path(m5["quality_report_path"]),
            "fixture": fixture_path,
        }

    def _verify(self, root: Path, **kwargs) -> tuple[dict, dict[str, Path]]:
        paths = self._build(root, **kwargs)
        result = verify_release(
            paths["forecast"],
            paths["history"],
            paths["history_manifest"],
            paths["fixture_manifest"],
            output_root=root / "m8-runs",
        )
        return result, paths

    def test_complete_m2_m5_m4_chain_is_ready_with_exact_evidence(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            result, paths = self._verify(root)
            self.assertEqual(result["status"], "ready")
            manifest = json.loads(Path(result["release_manifest_path"]).read_text(encoding="utf-8"))
            self.assertEqual(manifest["schema_version"], "m8-release-verify-v1")
            self.assertEqual(manifest["inputs"]["history"]["sha256"], file_sha256(paths["history"]))
            self.assertEqual(manifest["inputs"]["history_manifest"]["dataset_version"], "m2.1-test-2026-09-05")
            self.assertEqual(manifest["inputs"]["history_manifest"]["data_cutoff"], "2024-01-10")
            self.assertEqual(manifest["inputs"]["fixture_snapshot_manifest"]["snapshot_id"], paths["fixture_manifest"].parent.name)
            self.assertEqual(manifest["prediction_hash_evidence"]["evidence_status"], "verified")
            self.assertFalse(manifest["outputs"]["forecast_predictions_copied"])
            self.assertTrue(Path(result["release_report_path"]).is_file())
            report = Path(result["release_report_path"]).read_text(encoding="utf-8")
            self.assertIn("总体状态：`ready`", report)
            self.assertIn("未执行人工审批", report)
            self.assertIn("predictions.csv", report)
            self.assertEqual({path.name for path in Path(result["run_dir"]).iterdir()}, {"release_manifest.json", "release_report.md"})

    def test_history_manifest_output_mismatch_is_blocked(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths = self._build(root)
            payload = json.loads(paths["history_manifest"].read_text(encoding="utf-8"))
            payload["output"]["sha256"] = "0" * 64
            write_json(payload, paths["history_manifest"])
            result = verify_release(paths["forecast"], paths["history"], paths["history_manifest"], paths["fixture_manifest"], output_root=root / "m8-runs")
            self.assertEqual(result["status"], "blocked")
            check = next(item for item in result["checks"] if item["id"] == "history_manifest_output")
            self.assertTrue(any("SHA-256" in reason for reason in check["reasons"]))

    def test_fixture_output_raw_and_quality_mismatches_are_blocked(self):
        for field in ("canonical_output_sha256", "source_sha256", "quality_report_sha256"):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                paths = self._build(root)
                payload = json.loads(paths["fixture_manifest"].read_text(encoding="utf-8"))
                payload[field] = "0" * 64
                write_json(payload, paths["fixture_manifest"])
                result = verify_release(paths["forecast"], paths["history"], paths["history_manifest"], paths["fixture_manifest"], output_root=root / "m8-runs")
                self.assertEqual(result["status"], "blocked")
                self.assertTrue(any(item["status"] == "blocked" and item["id"].startswith("fixture_") for item in result["checks"]))

    def test_legacy_prediction_hash_and_selection_mismatch_need_review(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths = self._build(root)
            config_path = paths["forecast"] / "run_config.json"
            config = json.loads(config_path.read_text(encoding="utf-8"))
            config.pop("predictions_sha256", None)
            config["outputs"].pop("predictions_sha256", None)
            config["outputs"]["prediction_sha256"] = file_sha256(paths["forecast"] / "predictions.csv")
            write_json(config, config_path)
            result = verify_release(paths["forecast"], paths["history"], paths["history_manifest"], paths["fixture_manifest"], output_root=root / "legacy-m8")
            self.assertEqual(result["status"], "needs_review")
            self.assertEqual(result["prediction_hash_evidence"]["evidence_status"], "unrecorded_legacy")

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            result, _ = self._verify(root, selection_mismatch=True)
            self.assertEqual(result["status"], "needs_review")
            selection = next(item for item in result["checks"] if item["id"] == "selection_evidence")
            self.assertTrue(any("mismatch" in reason for reason in selection["reasons"]))

    def test_missing_history_version_is_not_ready(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths = self._build(root)
            payload = json.loads(paths["history_manifest"].read_text(encoding="utf-8"))
            payload.pop("dataset_version")
            write_json(payload, paths["history_manifest"])
            result = verify_release(paths["forecast"], paths["history"], paths["history_manifest"], paths["fixture_manifest"], output_root=root / "m8-runs")
            self.assertEqual(result["status"], "needs_review")
            self.assertIn("dataset_version", " ".join(next(item for item in result["checks"] if item["id"] == "history_manifest_metadata")["reasons"]))

    def test_m4_output_reference_and_malformed_quality_are_blocked(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths = self._build(root)
            config_path = paths["forecast"] / "run_config.json"
            config = json.loads(config_path.read_text(encoding="utf-8"))
            config["outputs"]["predictions"] = str(root / "not-the-run" / "predictions.csv")
            write_json(config, config_path)
            result = verify_release(paths["forecast"], paths["history"], paths["history_manifest"], paths["fixture_manifest"], output_root=root / "m8-runs")
            self.assertEqual(result["status"], "blocked")
            self.assertEqual(next(item for item in result["checks"] if item["id"] == "m4_artifact_reference")["status"], "blocked")

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths = self._build(root)
            paths["quality"].write_text("{not-json", encoding="utf-8")
            result = verify_release(paths["forecast"], paths["history"], paths["history_manifest"], paths["fixture_manifest"], output_root=root / "m8-runs")
            self.assertEqual(result["status"], "blocked")
            self.assertEqual(next(item for item in result["checks"] if item["id"] == "fixture_quality_evidence")["status"], "blocked")

    def test_as_of_fixture_conflict_and_history_rows_conflict_are_blocked(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths = self._build(root)
            config_path = paths["forecast"] / "run_config.json"
            config = json.loads(config_path.read_text(encoding="utf-8"))
            config["history"]["rows_used_before_as_of"] = 999
            write_json(config, config_path)
            result = verify_release(paths["forecast"], paths["history"], paths["history_manifest"], paths["fixture_manifest"], output_root=root / "history-conflict")
            self.assertEqual(result["status"], "blocked")
            self.assertTrue(any("rows_used_before_as_of" in reason for reason in next(item for item in result["checks"] if item["id"] == "time_boundary_crosscheck")["reasons"]))

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths = self._build(root)
            config_path = paths["forecast"] / "run_config.json"
            config = json.loads(config_path.read_text(encoding="utf-8"))
            config["as_of"] = {"value": "2024-01-08", "timestamp": "2024-01-08T00:00:00", "date_only": True}
            write_json(config, config_path)
            result = verify_release(paths["forecast"], paths["history"], paths["history_manifest"], paths["fixture_manifest"], output_root=root / "asof-conflict")
            self.assertEqual(result["status"], "blocked")

    def test_m8_does_not_modify_inputs_and_protects_input_directories(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths = self._build(root)
            before = {name: file_sha256(path) for name, path in paths.items() if path.is_file()}
            with self.assertRaisesRegex(DataValidationError, "input directory"):
                verify_release(paths["forecast"], paths["history"], paths["history_manifest"], paths["fixture_manifest"], output_root=paths["fixture_snapshot"])
            result = verify_release(paths["forecast"], paths["history"], paths["history_manifest"], paths["fixture_manifest"], output_root=root / "m8-runs")
            self.assertEqual(result["status"], "ready")
            after = {name: file_sha256(path) for name, path in paths.items() if path.is_file()}
            self.assertEqual(before, after)
            self.assertFalse((Path(r"D:\Codex\matchquant_data") / "m8_release_test").exists())

    def test_cli_smoke_is_concise_and_creates_one_audit_directory(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths = self._build(root)
            completed = subprocess.run(
                [sys.executable, "-m", "matchquant", "release-verify", "--forecast-run", str(paths["forecast"]), "--history", str(paths["history"]), "--history-manifest", str(paths["history_manifest"]), "--fixture-manifest", str(paths["fixture_manifest"]), "--output-root", str(root / "m8-cli")],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertLessEqual(len(completed.stdout.strip().splitlines()), 20)
            self.assertEqual(len(list((root / "m8-cli").glob("m8_release_*"))), 1)


if __name__ == "__main__":
    unittest.main()
