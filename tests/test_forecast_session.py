import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from matchquant.forecast_session import run_forecast_session
from matchquant.forecast_session_support import discover_new_child
from matchquant.io import file_sha256, load_matches, write_json
from matchquant.schema import DataValidationError, REQUIRED_COLUMNS


class ForecastSessionTests(unittest.TestCase):
    history_path = Path("examples/synthetic_matches.csv").resolve()
    source_template = Path("examples/synthetic_e0_source.csv").resolve()

    def _build_inputs(self, root: Path, completed_date: str = "06/01/2024") -> dict[str, Path]:
        input_root = root / "inputs"
        input_root.mkdir(parents=True, exist_ok=True)
        source = input_root / "local_e0.csv"
        source.write_text(
            self.source_template.read_text(encoding="utf-8").replace(
                "08/01/2024,B,C,1,0", f"{completed_date},B,C,1,0"
            ),
            encoding="utf-8",
        )
        history = self.history_path
        history_manifest = input_root / "history_manifest.json"
        frame = load_matches(history)
        write_json(
            {
                "schema_version": "m2.1-processed-manifest-v1",
                "dataset_version": "m2.1-test-session",
                "data_cutoff": str(frame["date"].max()),
                "output": {
                    "filename": str(history),
                    "sha256": file_sha256(history),
                    "rows": len(frame),
                    "columns": list(REQUIRED_COLUMNS),
                },
            },
            history_manifest,
        )
        model_config = input_root / "frequency.json"
        model_config.write_text('{"alpha": 1.0}\n', encoding="utf-8")
        return {
            "history": history,
            "history_manifest": history_manifest,
            "source": source,
            "model": model_config,
        }

    def _run_explicit(self, root: Path, *, completed_date: str = "06/01/2024") -> tuple[dict, dict[str, Path]]:
        paths = self._build_inputs(root, completed_date=completed_date)
        return self._run_explicit_paths(root, paths)

    def _run_explicit_paths(self, root: Path, paths: dict[str, Path]):
        result = run_forecast_session(
            paths["history"],
            paths["history_manifest"],
            paths["source"],
            "2024-25",
            "2024-01-07",
            model_name="frequency",
            model_config_path=paths["model"],
            fixture_data_root=root / "fixture-data",
            output_root=root / "runs",
        )
        return result, paths

    def _selection_payload(self, history: Path) -> dict[str, object]:
        config = {"alpha": 1.0}
        candidate_hash = hashlib.sha256(
            json.dumps(config, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        return {
            "schema_version": "m3-selection-v1",
            "input_sha256": file_sha256(history),
            "season_mode": "carry",
            "validation_folds": [
                {"fold_id": "validation_1", "start": "2024-01-03", "end": "2024-01-05"},
                {"fold_id": "validation_2", "start": "2024-01-06", "end": "2024-01-06"},
            ],
            "final_test": {"start": "2024-01-07", "end": "2024-01-10"},
            "selection_rule": {
                "primary": "validation pooled average multiclass log loss",
                "final_test_metrics_used_for_selection": False,
            },
            "selected_candidate": {
                "candidate_name": "frequency_alpha_1",
                "model": "frequency",
                "config": config,
                "configuration_sha256": candidate_hash,
            },
        }

    def test_explicit_session_is_ready_and_links_hashed_child_artifacts(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            pointer = root / "fixture-data" / "processed" / "epl_current.json"
            pointer.parent.mkdir(parents=True)
            pointer.write_text('{"version":"untouched"}\n', encoding="utf-8")
            pointer_before = file_sha256(pointer)
            paths = self._build_inputs(root)
            before = {name: file_sha256(path) for name, path in paths.items()}
            result, _ = self._run_explicit_paths(root, paths)

            self.assertEqual(result["status"], "ready")
            session_dir = Path(result["session_dir"])
            self.assertTrue(session_dir.name.startswith("m9_forecast_session_"))
            self.assertEqual(
                {path.name for path in session_dir.iterdir()},
                {"session_manifest.json", "session_report.md"},
            )
            manifest = json.loads(Path(result["session_manifest_path"]).read_text(encoding="utf-8"))
            self.assertEqual(manifest["schema_version"], "m9-forecast-session-v1")
            self.assertEqual(manifest["status"], "ready")
            self.assertEqual(manifest["stages"]["forecast"]["status"], "completed")
            self.assertEqual(manifest["stages"]["release_verification"]["status"], "ready")
            m5_manifest = Path(manifest["stages"]["fixture_snapshot"]["artifacts"]["manifest"]["path"])
            self.assertEqual(manifest["stages"]["fixture_snapshot"]["manifest_path"], str(m5_manifest))
            self.assertEqual(
                manifest["stages"]["fixture_snapshot"]["artifacts"]["manifest"]["sha256"],
                file_sha256(m5_manifest),
            )
            m4_predictions = Path(manifest["stages"]["forecast"]["artifacts"]["predictions"]["path"])
            self.assertEqual(manifest["stages"]["forecast"]["predictions_path"], str(m4_predictions))
            self.assertEqual(
                manifest["stages"]["forecast"]["artifacts"]["predictions"]["sha256"],
                file_sha256(m4_predictions),
            )
            self.assertEqual(
                manifest["outputs"]["forecast_predictions"]["sha256"],
                file_sha256(m4_predictions),
            )
            release_manifest = Path(manifest["stages"]["release_verification"]["artifacts"]["manifest"]["path"])
            self.assertEqual(manifest["stages"]["release_verification"]["release_manifest_path"], str(release_manifest))
            self.assertEqual(
                manifest["stages"]["release_verification"]["artifacts"]["manifest"]["sha256"],
                file_sha256(release_manifest),
            )
            self.assertEqual(
                manifest["outputs"]["release_manifest"]["sha256"],
                file_sha256(release_manifest),
            )
            self.assertEqual(
                manifest["outputs"]["session_report_sha256"],
                file_sha256(result["session_report_path"]),
            )
            report = Path(result["session_report_path"]).read_text(encoding="utf-8")
            self.assertIn("M5 → M4 → M8", report)
            self.assertIn("M7/M2.1", report)
            self.assertIn("M6", report)
            self.assertIn("predictions.csv", report)
            self.assertEqual(file_sha256(pointer), pointer_before)
            after = {name: file_sha256(path) for name, path in paths.items()}
            self.assertEqual(before, after)

    def test_m8_needs_review_is_propagated_without_status_change(self):
        with tempfile.TemporaryDirectory() as temporary:
            result, _ = self._run_explicit(Path(temporary), completed_date="07/01/2024")
            self.assertEqual(result["status"], "needs_review")
            self.assertEqual(result["stages"]["release_verification"]["status"], "needs_review")
            self.assertEqual(result["stages"]["forecast"]["status"], "completed")
            manifest = json.loads(Path(result["session_manifest_path"]).read_text(encoding="utf-8"))
            self.assertEqual(manifest["status"], "needs_review")
            self.assertIn("同日或更晚", Path(result["session_report_path"]).read_text(encoding="utf-8"))

    def test_invalid_fixture_blocks_and_keeps_session_evidence(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths = self._build_inputs(root)
            paths["source"].write_text(
                paths["source"].read_text(encoding="utf-8").replace(",FTHG,FTAG", ",FTHG"),
                encoding="utf-8",
            )
            result = run_forecast_session(
                paths["history"], paths["history_manifest"], paths["source"], "2024-25", "2024-01-07",
                model_name="frequency", model_config_path=paths["model"],
                fixture_data_root=root / "fixture-data", output_root=root / "runs",
            )
            self.assertEqual(result["status"], "blocked")
            self.assertEqual(result["stages"]["fixture_snapshot"]["status"], "blocked")
            self.assertEqual(result["stages"]["forecast"]["status"], "not_run")
            self.assertEqual(result["stages"]["release_verification"]["status"], "not_run")
            self.assertTrue(Path(result["session_manifest_path"]).is_file())
            report = Path(result["session_report_path"]).read_text(encoding="utf-8")
            self.assertIn("M5", report)
            self.assertIn("FTAG", report)
            self.assertTrue(list((root / "fixture-data" / "fixtures" / "snapshots").glob("*/fixture_snapshot_manifest.json")))

    def test_m4_failure_preserves_m5_and_marks_later_stages_not_run(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths = self._build_inputs(root)
            missing_history = root / "inputs" / "missing_history.csv"
            result = run_forecast_session(
                missing_history,
                paths["history_manifest"],
                paths["source"],
                "2024-25",
                "2024-01-07",
                model_name="frequency",
                model_config_path=paths["model"],
                fixture_data_root=root / "fixture-data",
                output_root=root / "runs",
            )
            self.assertEqual(result["status"], "blocked")
            self.assertEqual(result["stages"]["fixture_snapshot"]["status"], "completed")
            self.assertEqual(result["stages"]["forecast"]["status"], "blocked")
            self.assertEqual(result["stages"]["release_verification"]["status"], "not_run")
            self.assertIn("missing_history.csv", Path(result["session_report_path"]).read_text(encoding="utf-8"))

    def test_unexpected_stage_errors_are_not_masked(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths = self._build_inputs(root)
            with patch(
                "matchquant.forecast_session.run_forecast",
                side_effect=RuntimeError("unexpected programming failure"),
            ):
                with self.assertRaisesRegex(RuntimeError, "unexpected programming failure"):
                    run_forecast_session(
                        paths["history"], paths["history_manifest"], paths["source"], "2024-25", "2024-01-07",
                        model_name="frequency", model_config_path=paths["model"],
                        fixture_data_root=root / "fixture-data", output_root=root / "runs",
                    )

    def test_m3_selection_mode_is_explicitly_recorded(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths = self._build_inputs(root)
            selection_path = root / "inputs" / "m3_run_config.json"
            write_json(self._selection_payload(paths["history"]), selection_path)
            result = run_forecast_session(
                paths["history"], paths["history_manifest"], paths["source"], "2024-25", "2024-01-07",
                selection_run_config_path=selection_path,
                fixture_data_root=root / "fixture-data", output_root=root / "selection-runs",
            )
            self.assertEqual(result["status"], "ready")
            manifest = json.loads(Path(result["session_manifest_path"]).read_text(encoding="utf-8"))
            self.assertEqual(manifest["inputs"]["model_selection"]["mode"], "m3_run_config")
            forecast_config = json.loads(Path(result["forecast_run_config_path"]).read_text(encoding="utf-8"))
            self.assertEqual(forecast_config["selection"]["mode"], "m3_run_config")

    def test_output_root_guards_run_before_session_creation(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths = self._build_inputs(root)
            fixture_root = root / "fixture-data"
            with self.assertRaisesRegex(DataValidationError, "output-root"):
                run_forecast_session(
                    paths["history"], paths["history_manifest"], paths["source"], "2024-25", "2024-01-07",
                    model_name="frequency", model_config_path=paths["model"],
                    fixture_data_root=fixture_root, output_root=fixture_root / "runs",
                )
            session_parent = root / "runs"
            session_child = session_parent / "m9_forecast_session_existing"
            session_child.mkdir(parents=True)
            with self.assertRaisesRegex(DataValidationError, "output-root"):
                run_forecast_session(
                    paths["history"], paths["history_manifest"], paths["source"], "2024-25", "2024-01-07",
                    model_name="frequency", model_config_path=paths["model"],
                    fixture_data_root=fixture_root, output_root=session_child,
                )
            self.assertEqual(list(session_parent.iterdir()), [session_child])

    def test_ambiguous_partial_artifacts_are_not_bound(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "runs"
            root.mkdir()
            before = set(root.iterdir())
            (root / "m4_forecast_20240519_a").mkdir()
            (root / "m4_forecast_20240519_b").mkdir()

            bound, discovery = discover_new_child(root, "m4_forecast_", before)

            self.assertIsNone(bound)
            self.assertEqual(discovery["status"], "ambiguous")
            self.assertIsNone(discovery["bound_path"])
            self.assertEqual(len(discovery["candidates"]), 2)

    def test_cli_smoke_is_concise_and_returns_m8_status(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths = self._build_inputs(root)
            completed = subprocess.run(
                [
                    sys.executable, "-m", "matchquant", "forecast-session",
                    "--history", str(paths["history"]),
                    "--history-manifest", str(paths["history_manifest"]),
                    "--fixture-source", str(paths["source"]),
                    "--season", "2024-25", "--as-of", "2024-01-07",
                    "--model", "frequency", "--model-config", str(paths["model"]),
                    "--fixture-data-root", str(root / "fixture-data"),
                    "--output-root", str(root / "runs"),
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertLessEqual(len(completed.stdout.strip().splitlines()), 12)
            self.assertIn("MatchQuant M9 forecast session ready", completed.stdout)


if __name__ == "__main__":
    unittest.main()
