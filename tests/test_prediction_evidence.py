"""B02 tests: the unified M4/M11 prediction-evidence adapter and its M6 settlement bridge."""

import builtins
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd
from pandas.testing import assert_frame_equal

from matchquant.engine_snapshot import build_engine_snapshot, predict_from_engine_snapshot
from matchquant.engine_snapshot_manifest import write_snapshot_manifest
from matchquant.engine_snapshot_support import configuration_hash
from matchquant.forecast import run_forecast
from matchquant.io import file_sha256, load_matches, write_json
from matchquant.prediction_evidence import load_prediction_evidence
from matchquant.schema import DataValidationError, REQUIRED_COLUMNS
from matchquant.settlement import settle_forecast_run


HISTORY_COLUMNS = [
    "match_id",
    "date",
    "season",
    "home_team",
    "away_team",
    "home_goals",
    "away_goals",
]
FIXTURE_COLUMNS = ["fixture_id", "date", "season", "home_team", "away_team"]
MODEL_CONFIGS = {
    "frequency": {"alpha": 1.0},
    "elo": {"k_factor": 20.0, "home_advantage": 50.0, "draw_base": 0.27},
    "poisson": {
        "prior_home_goals": 1.5,
        "prior_away_goals": 1.2,
        "prior_weight": 3.0,
        "max_goals": 10,
        "tail_tolerance": 1e-12,
    },
}
PROBABILITY_COLUMNS = ["home_win_prob", "draw_prob", "away_win_prob"]
IDENTITY_COLUMNS = ["fixture_id", "date", "season", "home_team", "away_team", "model"]


def _history_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            ["h1", "2024-01-01", "2023-24", "A", "B", 2, 0],
            ["h2", "2024-01-01", "2023-24", "C", "D", 1, 1],
            ["h3", "2024-01-02", "2023-24", "B", "C", 0, 1],
            ["h4", "2024-01-03", "2023-24", "D", "A", 1, 2],
            ["h5", "2024-01-04", "2023-24", "A", "C", 1, 1],
            ["h6", "2024-01-05", "2023-24", "B", "D", 3, 1],
            ["h7", "2024-01-06", "2023-24", "C", "A", 0, 2],
            ["h8", "2024-01-07", "2024-25", "A", "B", 1, 0],
            ["h9", "2024-01-08", "2024-25", "C", "D", 2, 2],
        ],
        columns=HISTORY_COLUMNS,
    )


def _fixtures_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            ["f1", "2024-01-07", "2024-25", "A", "B"],
            ["f2", "2024-01-08", "2024-25", "C", "D"],
        ],
        columns=FIXTURE_COLUMNS,
    )


def _late_fixtures_frame() -> pd.DataFrame:
    """Fixtures strictly after the as_of day; usable with an exact-time as_of."""

    return pd.DataFrame(
        [["f2", "2024-01-08", "2024-25", "C", "D"]],
        columns=FIXTURE_COLUMNS,
    )


class PredictionEvidenceTestBase(unittest.TestCase):
    def _write_history_evidence(self, root: Path, frame: pd.DataFrame) -> tuple[Path, Path]:
        root.mkdir(parents=True, exist_ok=True)
        history_path = root / "history.csv"
        manifest_path = root / "history_manifest.json"
        frame.to_csv(history_path, index=False)
        write_json(
            {
                "schema_version": "m2.1-processed-manifest-v1",
                "dataset_version": "m2.1-b02-test",
                "data_cutoff": str(frame["date"].max()),
                "output": {
                    "filename": str(history_path.resolve()),
                    "sha256": file_sha256(history_path),
                    "rows": len(frame),
                    "columns": HISTORY_COLUMNS,
                },
            },
            manifest_path,
        )
        return history_path, manifest_path

    def _m11_run(self, root: Path, model: str, *, as_of: str = "2024-01-07", fixtures: pd.DataFrame | None = None, history: pd.DataFrame | None = None) -> tuple[Path, Path]:
        history_path, manifest_path = self._write_history_evidence(
            root / "evidence", history if history is not None else _history_frame()
        )
        fixtures_path = root / "fixtures.csv"
        (fixtures if fixtures is not None else _fixtures_frame()).to_csv(fixtures_path, index=False)
        built = build_engine_snapshot(
            history_path,
            manifest_path,
            as_of,
            model_names=[model],
            model_configs={model: MODEL_CONFIGS[model]},
            output_root=root / "snapshots",
        )
        predicted = predict_from_engine_snapshot(
            built["snapshot_manifest_path"], fixtures_path, output_root=root / "m11-runs"
        )
        return Path(predicted["run_dir"]), built

    def _m4_run(self, root: Path, model: str, *, as_of: str = "2024-01-07", fixtures: pd.DataFrame | None = None, history: pd.DataFrame | None = None) -> Path:
        history_path, _ = self._write_history_evidence(
            root / "evidence", history if history is not None else _history_frame()
        )
        fixtures_path = root / "fixtures.csv"
        (fixtures if fixtures is not None else _fixtures_frame()).to_csv(fixtures_path, index=False)
        config_path = root / f"{model}.json"
        write_json(MODEL_CONFIGS[model], config_path)
        result = run_forecast(
            history_path,
            fixtures_path,
            as_of,
            model_name=model,
            model_config_path=config_path,
            output_root=root / "m4-runs",
        )
        return Path(result["run_dir"])

    def _results_file(self, root: Path, results: pd.DataFrame) -> Path:
        path = root / "results.csv"
        results.loc[:, REQUIRED_COLUMNS].to_csv(path, index=False)
        return path

    def _rows(self, root: Path, *match_ids: str) -> pd.DataFrame:
        return _history_frame().loc[_history_frame()["match_id"].isin(match_ids), REQUIRED_COLUMNS].copy()

    def _exact_time_history_frame(self) -> pd.DataFrame:
        """History whose as_of-day row carries an explicit time before the as_of clock."""

        frame = _history_frame()
        frame.loc[frame["match_id"].eq("h8"), "date"] = "2024-01-07T11:30:00"
        return frame


class M11SingleModelEquivalenceTests(PredictionEvidenceTestBase):
    def test_m11_matches_m4_predictions_and_settlement_for_every_model(self):
        for model in ("frequency", "elo", "poisson"):
            with self.subTest(model=model):
                with tempfile.TemporaryDirectory() as temporary:
                    root = Path(temporary)
                    m11_run, _ = self._m11_run(root, model)
                    m4_run = self._m4_run(root / "m4-case", model)
                    m11_evidence = load_prediction_evidence(m11_run)
                    self.assertEqual(m11_evidence["producer_schema"], "m11-engine-snapshot-predict-v1")
                    self.assertEqual(m11_evidence["prediction_hash_status"], "verified")
                    self.assertEqual(m11_evidence["model_name"], model)
                    self.assertEqual(
                        m11_evidence["model_config_sha256"],
                        configuration_hash(MODEL_CONFIGS[model]),
                    )
                    self.assertEqual(m11_evidence["origin_evidence"]["kind"], "m11_engine_snapshot")

                    m11_predictions = pd.read_csv(m11_evidence["predictions_path"])
                    m4_predictions = pd.read_csv(m4_run / "predictions.csv")
                    self.assertEqual(list(m11_predictions[IDENTITY_COLUMNS].columns), IDENTITY_COLUMNS)
                    assert_frame_equal(
                        m11_predictions.sort_values("fixture_id").reset_index(drop=True),
                        m4_predictions.sort_values("fixture_id").reset_index(drop=True),
                        check_exact=False,
                        atol=1e-12,
                        rtol=0,
                    )

                    full_results = _history_frame().loc[:, REQUIRED_COLUMNS]
                    results_path = self._results_file(root, full_results)
                    m11_settlement = settle_forecast_run(
                        m11_run, results_path, output_root=root / "settlement-m11"
                    )
                    m4_settlement = settle_forecast_run(
                        m4_run, results_path, output_root=root / "settlement-m4"
                    )
                    self.assertEqual(m11_settlement["status"], m4_settlement["status"])
                    self.assertEqual(m11_settlement["counts"], m4_settlement["counts"])
                    for key in ("log_loss", "brier_score", "accuracy", "coverage"):
                        m11_value = m11_settlement["metrics"][key]
                        m4_value = m4_settlement["metrics"][key]
                        if m11_value is None:
                            self.assertIsNone(m4_value)
                        else:
                            self.assertAlmostEqual(m11_value, m4_value, places=12)
                    assert_frame_equal(
                        m11_settlement["settlement"]
                        .loc[:, IDENTITY_COLUMNS + ["settlement_status", "date_status"]]
                        .sort_values("fixture_id")
                        .reset_index(drop=True),
                        m4_settlement["settlement"]
                        .loc[:, IDENTITY_COLUMNS + ["settlement_status", "date_status"]]
                        .sort_values("fixture_id")
                        .reset_index(drop=True),
                    )
                    config = json.loads(Path(m11_settlement["run_config_path"]).read_text(encoding="utf-8"))
                    self.assertEqual(config["forecast_run"]["producer_schema"], "m11-engine-snapshot-predict-v1")
                    self.assertEqual(config["forecast_run"]["producer_command"], "matchquant engine-snapshot-predict")
                    self.assertEqual(config["forecast_run"]["origin_evidence"]["kind"], "m11_engine_snapshot")

    def test_exact_time_as_of_used_cutoff_on_as_of_day_is_accepted_and_equivalent(self):
        """History rows before an exact-time as_of, on the as_of day, are legitimately used."""

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            exact_history = self._exact_time_history_frame()
            m11_run, built = self._m11_run(
                root,
                "frequency",
                as_of="2024-01-07T12:00:00",
                fixtures=_late_fixtures_frame(),
                history=exact_history,
            )
            manifest = json.loads(Path(built["snapshot_manifest_path"]).read_text(encoding="utf-8"))
            self.assertEqual(manifest["history"]["used_data_cutoff"], "2024-01-07T11:30:00")
            m4_run = self._m4_run(
                root / "m4-case",
                "frequency",
                as_of="2024-01-07T12:00:00",
                fixtures=_late_fixtures_frame(),
                history=exact_history,
            )
            evidence = load_prediction_evidence(m11_run)
            self.assertEqual(evidence["origin_evidence"]["used_data_cutoff"], "2024-01-07T11:30:00")
            results_path = self._results_file(root, _history_frame().loc[:, REQUIRED_COLUMNS])
            m11_settlement = settle_forecast_run(m11_run, results_path, output_root=root / "settlement-m11")
            m4_settlement = settle_forecast_run(m4_run, results_path, output_root=root / "settlement-m4")
            self.assertEqual(m11_settlement["status"], m4_settlement["status"])
            self.assertEqual(m11_settlement["counts"], m4_settlement["counts"])
            self.assertAlmostEqual(
                m11_settlement["metrics"]["log_loss"], m4_settlement["metrics"]["log_loss"], places=12
            )

    def test_m4_branch_keeps_legacy_shape_and_hash_status(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            m4_run = self._m4_run(root, "frequency")
            evidence = load_prediction_evidence(m4_run)
            self.assertEqual(evidence["producer_schema"], "m4-forecast-v1")
            self.assertEqual(evidence["producer_command"], "matchquant forecast")
            self.assertEqual(evidence["prediction_hash_status"], "verified")
            self.assertEqual(evidence["origin_evidence"]["kind"], "m4_run_config")
            self.assertEqual(evidence["origin_evidence"]["history_evidence_status"], "recorded_by_producer_not_reread_by_settlement")

    def test_adapter_dispatch_rejects_unknown_schema(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            m11_run, _ = self._m11_run(root, "frequency")
            config_path = m11_run / "run_config.json"
            config = json.loads(config_path.read_text(encoding="utf-8"))
            config["schema_version"] = "made-up-predict-v9"
            write_json(config, config_path)
            with self.assertRaisesRegex(DataValidationError, "unsupported prediction run schema_version"):
                load_prediction_evidence(m11_run)

    def test_multi_model_m11_run_is_rejected_explicitly(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            history_path, manifest_path = self._write_history_evidence(root / "evidence", _history_frame())
            fixtures_path = root / "fixtures.csv"
            _fixtures_frame().to_csv(fixtures_path, index=False)
            built = build_engine_snapshot(
                history_path,
                manifest_path,
                "2024-01-07",
                model_names=["frequency", "elo"],
                model_configs={name: MODEL_CONFIGS[name] for name in ("frequency", "elo")},
                output_root=root / "snapshots",
            )
            predicted = predict_from_engine_snapshot(
                built["snapshot_manifest_path"], fixtures_path, output_root=root / "m11-runs"
            )
            with self.assertRaisesRegex(DataValidationError, "multi-model M11 prediction runs are not supported"):
                load_prediction_evidence(Path(predicted["run_dir"]))


class M11SettlementSafetyTests(PredictionEvidenceTestBase):
    def test_settlement_succeeds_with_history_disconnected_and_zero_history_or_network_access(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            m11_run, built = self._m11_run(root, "frequency")
            history_path = root / "evidence" / "history.csv"
            manifest_path = root / "evidence" / "history_manifest.json"
            history_hash = file_sha256(history_path)
            history_path.unlink()
            manifest_path.unlink()

            opened_history: list[str] = []
            real_open = builtins.open

            def guarded_open(file, *args, **kwargs):
                text = str(file)
                if "history" in Path(text).name.lower():
                    opened_history.append(text)
                return real_open(file, *args, **kwargs)

            results_path = self._results_file(root, _history_frame().loc[:, REQUIRED_COLUMNS])
            with (
                patch("builtins.open", side_effect=guarded_open),
                patch("urllib.request.urlopen", side_effect=AssertionError("network used")),
            ):
                result = settle_forecast_run(m11_run, results_path, output_root=root / "settlement")
            self.assertEqual(result["status"], "complete")
            self.assertEqual(result["counts"]["settled"], 2)
            self.assertEqual(opened_history, [])

    def test_used_cutoff_at_or_after_date_only_as_of_is_rejected_even_when_manifest_declares_it(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            m11_run, built = self._m11_run(root, "frequency")
            manifest_path = Path(built["snapshot_manifest_path"])
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["history"]["used_data_cutoff"] = "2024-01-07"
            write_snapshot_manifest(manifest, manifest_path)
            config_path = m11_run / "run_config.json"
            config = json.loads(config_path.read_text(encoding="utf-8"))
            config["snapshot"]["manifest_sha256"] = file_sha256(manifest_path)
            config["snapshot"]["manifest_content_sha256"] = json.loads(
                manifest_path.read_text(encoding="utf-8")
            )["manifest_sha256"]
            write_json(config, config_path)
            with self.assertRaisesRegex(DataValidationError, "used_data_cutoff"):
                load_prediction_evidence(m11_run)

    def test_tampered_predictions_file_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            m11_run, _ = self._m11_run(root, "frequency")
            predictions_path = m11_run / "predictions.csv"
            frame = pd.read_csv(predictions_path)
            frame.loc[0, "home_win_prob"] = 0.99
            frame.to_csv(predictions_path, index=False)
            with self.assertRaisesRegex(DataValidationError, "predictions SHA-256 does not match"):
                load_prediction_evidence(m11_run)

    def test_prediction_fixture_identity_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            m11_run, _ = self._m11_run(root, "frequency")
            predictions_path = m11_run / "predictions.csv"
            frame = pd.read_csv(predictions_path, dtype="string", keep_default_na=False)
            frame.loc[0, "home_team"] = "Z"
            frame.to_csv(predictions_path, index=False)
            config_path = m11_run / "run_config.json"
            config = json.loads(config_path.read_text(encoding="utf-8"))
            config["predictions_sha256"] = file_sha256(predictions_path)
            config["outputs"]["predictions_sha256"] = config["predictions_sha256"]
            write_json(config, config_path)
            with self.assertRaisesRegex(DataValidationError, "does not match the recorded source fixture"):
                load_prediction_evidence(m11_run)

    def test_tampered_recorded_fixtures_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            m11_run, _ = self._m11_run(root, "frequency")
            fixtures_path = root / "fixtures.csv"
            frame = pd.read_csv(fixtures_path)
            frame.loc[0, "home_team"] = "Z"
            frame.to_csv(fixtures_path, index=False)
            with self.assertRaisesRegex(DataValidationError, "fixtures SHA-256 does not match"):
                load_prediction_evidence(m11_run)

    def test_recorded_fixtures_with_known_scores_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            m11_run, _ = self._m11_run(root, "frequency")
            fixtures_path = root / "fixtures.csv"
            frame = pd.read_csv(fixtures_path)
            frame["home_goals"] = [1, ""]
            frame.to_csv(fixtures_path, index=False)
            config_path = m11_run / "run_config.json"
            config = json.loads(config_path.read_text(encoding="utf-8"))
            config["fixtures"]["sha256"] = file_sha256(fixtures_path)
            write_json(config, config_path)
            with self.assertRaisesRegex(DataValidationError, "known scores"):
                load_prediction_evidence(m11_run)

    def test_missing_recorded_fixtures_file_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            m11_run, _ = self._m11_run(root, "frequency")
            (root / "fixtures.csv").unlink()
            with self.assertRaisesRegex(FileNotFoundError, "recorded source fixtures CSV does not exist"):
                load_prediction_evidence(m11_run)

    def test_tampered_snapshot_state_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            m11_run, built = self._m11_run(root, "frequency")
            state_path = Path(built["model_states_path"])
            state_path.write_text(
                state_path.read_text(encoding="utf-8").replace("frequency", "frequency2", 1),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(DataValidationError, "SHA-256|hash"):
                load_prediction_evidence(m11_run)

    def test_tampered_snapshot_manifest_content_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            m11_run, built = self._m11_run(root, "frequency")
            manifest_path = Path(built["snapshot_manifest_path"])
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["history"]["rows_used_before_as_of"] = 999
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(DataValidationError, "manifest hash"):
                load_prediction_evidence(m11_run)

    def test_run_as_of_diverging_from_snapshot_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            m11_run, _ = self._m11_run(root, "frequency")
            config_path = m11_run / "run_config.json"
            config = json.loads(config_path.read_text(encoding="utf-8"))
            config["as_of"]["value"] = "2024-01-08"
            config["as_of"]["timestamp"] = "2024-01-08T00:00:00"
            write_json(config, config_path)
            with self.assertRaisesRegex(DataValidationError, "as_of.value does not match"):
                load_prediction_evidence(m11_run)

    def test_run_model_config_diverging_from_snapshot_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            m11_run, _ = self._m11_run(root, "frequency")
            config_path = m11_run / "run_config.json"
            config = json.loads(config_path.read_text(encoding="utf-8"))
            config["models"][0]["config"] = {"alpha": 2.0}
            write_json(config, config_path)
            with self.assertRaisesRegex(DataValidationError, "config does not match the snapshot"):
                load_prediction_evidence(m11_run)

    def test_recorded_state_hash_diverging_from_snapshot_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            m11_run, _ = self._m11_run(root, "frequency")
            config_path = m11_run / "run_config.json"
            config = json.loads(config_path.read_text(encoding="utf-8"))
            config["snapshot"]["state_sha256"] = "0" * 64
            write_json(config, config_path)
            with self.assertRaisesRegex(DataValidationError, "state SHA-256 does not match"):
                load_prediction_evidence(m11_run)

    def test_zero_settlement_keeps_null_metrics_for_m11(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            m11_run, _ = self._m11_run(root, "frequency")
            results_path = self._results_file(root, self._rows(root, "h1"))
            result = settle_forecast_run(m11_run, results_path, output_root=root / "settlement")
            self.assertEqual(result["status"], "partial")
            self.assertEqual(result["counts"]["settled"], 0)
            self.assertEqual(result["metrics"]["coverage"], 0.0)
            self.assertIsNone(result["metrics"]["log_loss"])
            report = Path(result["settlement_report_path"]).read_text(encoding="utf-8")
            self.assertIn("M11 单模型引擎快照预测", report)
            self.assertNotIn("本报告只使用一个已生成的 M4 本地预测快照", report)

    def test_m11_output_root_must_not_overlap_run_or_snapshot_and_inputs_stay_unchanged(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            m11_run, built = self._m11_run(root, "frequency")
            results_path = self._results_file(root, _history_frame().loc[:, REQUIRED_COLUMNS])
            input_hashes = {
                path: file_sha256(path)
                for path in (
                    m11_run / "predictions.csv",
                    m11_run / "run_config.json",
                    Path(built["snapshot_manifest_path"]),
                    Path(built["model_states_path"]),
                    root / "fixtures.csv",
                    results_path,
                )
            }
            with self.assertRaisesRegex(DataValidationError, "M11 snapshot directory"):
                settle_forecast_run(
                    m11_run, results_path, output_root=Path(built["snapshot_manifest_path"]).parent / "out"
                )
            with self.assertRaisesRegex(DataValidationError, "prediction run directory"):
                settle_forecast_run(m11_run, results_path, output_root=m11_run / "out")
            result = settle_forecast_run(m11_run, results_path, output_root=root / "settlement")
            self.assertEqual(result["status"], "complete")
            for path, digest in input_hashes.items():
                self.assertEqual(file_sha256(path), digest, str(path))


if __name__ == "__main__":
    unittest.main()
