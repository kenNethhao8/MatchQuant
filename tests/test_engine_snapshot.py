import contextlib, hashlib, io, json, tempfile, unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd
from pandas.testing import assert_frame_equal

from matchquant.engine_snapshot import build_engine_snapshot, predict_from_engine_snapshot
from matchquant.engine_snapshot_support import export_model_state, import_model_state, load_engine_snapshot
from matchquant.forecast import forecast_fixtures
from matchquant.io import file_sha256, load_matches, write_json
from matchquant.models import create_model
from matchquant.schema import DataValidationError


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
    "elo": {
        "k_factor": 20.0,
        "home_advantage": 50.0,
        "draw_base": 0.27,
    },
    "poisson": {
        "prior_home_goals": 1.5,
        "prior_away_goals": 1.2,
        "prior_weight": 3.0,
        "max_goals": 10,
        "tail_tolerance": 1e-12,
    },
}


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


def _write_history_evidence(root: Path, frame: pd.DataFrame) -> tuple[Path, Path]:
    root.mkdir(parents=True, exist_ok=True)
    history_path = root / "history.csv"
    manifest_path = root / "history_manifest.json"
    frame.to_csv(history_path, index=False)
    write_json(
        {
            "schema_version": "m2.1-processed-manifest-v1",
            "dataset_version": "m2.1-test-snapshot",
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


class EngineSnapshotTests(unittest.TestCase):
    def _case(self, root: Path) -> tuple[Path, Path, Path]:
        history_path, manifest_path = _write_history_evidence(root, _history_frame())
        fixtures_path = root / "fixtures.csv"
        _fixtures_frame().to_csv(fixtures_path, index=False)
        return history_path, manifest_path, fixtures_path

    def _build_one(self, root: Path, model: str = "frequency") -> dict:
        history_path, manifest_path, _ = self._case(root)
        return build_engine_snapshot(
            history_path,
            manifest_path,
            "2024-01-07",
            model_names=[model],
            model_configs={model: MODEL_CONFIGS[model]},
            output_root=root / "snapshots",
        )

    def test_all_model_states_round_trip_through_explicit_json_codecs(self):
        history = load_matches(Path("examples/synthetic_matches.csv"))
        for name, config in MODEL_CONFIGS.items():
            with self.subTest(model=name):
                model = create_model(name, config)
                model.reset()
                for row in history.itertuples(index=False):
                    model.update(row.home_team, row.away_team, row.home_goals, row.away_goals)
                record = export_model_state(model, config)
                serialized = json.loads(json.dumps(record, allow_nan=False))
                restored = import_model_state(serialized)
                self.assertEqual(export_model_state(restored, config), record)
                self.assertEqual(
                    model.predict("A", "B"), restored.predict("A", "B")
                )

    def test_build_writes_only_state_and_manifest_with_verified_hashes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            result = self._build_one(root, "poisson")
            snapshot_dir = Path(result["snapshot_dir"])
            self.assertEqual(
                {path.name for path in snapshot_dir.iterdir()},
                {"model_states.json", "snapshot_manifest.json"},
            )
            self.assertFalse((snapshot_dir / "history.csv").exists())
            manifest = json.loads(Path(result["snapshot_manifest_path"]).read_text(encoding="utf-8"))
            self.assertEqual(manifest["schema_version"], "m11-engine-snapshot-v1")
            self.assertEqual(
                manifest["state"]["sha256"], file_sha256(snapshot_dir / "model_states.json")
            )
            self.assertEqual(
                manifest["manifest_sha256"], result["manifest"]["manifest_sha256"]
            )
            self.assertEqual(manifest["history"]["rows_used_before_as_of"], 7)
            self.assertEqual(manifest["history"]["rows_excluded_at_or_after_as_of"], 2)
            loaded = load_engine_snapshot(result["snapshot_manifest_path"])
            self.assertEqual(loaded["manifest"]["snapshot_id"], snapshot_dir.name)

    def test_future_rows_do_not_change_built_state(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            history_path, manifest_path, _ = self._case(root)
            first = build_engine_snapshot(
                history_path,
                manifest_path,
                "2024-01-07",
                model_names=list(MODEL_CONFIGS),
                model_configs=MODEL_CONFIGS,
                output_root=root / "first",
            )
            changed = _history_frame()
            changed.loc[changed["match_id"].eq("h9"), ["home_goals", "away_goals"]] = [9, 0]
            changed_history, changed_manifest = _write_history_evidence(root / "changed", changed)
            second = build_engine_snapshot(
                changed_history,
                changed_manifest,
                "2024-01-07",
                model_names=list(MODEL_CONFIGS),
                model_configs=MODEL_CONFIGS,
                output_root=root / "second",
            )
            self.assertEqual(
                Path(first["model_states_path"]).read_bytes(),
                Path(second["model_states_path"]).read_bytes(),
            )

    def test_build_requires_manifest_evidence_and_rejects_bad_as_of(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            history_path, manifest_path, _ = self._case(root)
            manifest_path.unlink()
            with self.assertRaises((DataValidationError, FileNotFoundError)):
                build_engine_snapshot(
                    history_path,
                    manifest_path,
                    "2024-01-07",
                    model_names=["frequency"],
                    model_configs={"frequency": MODEL_CONFIGS["frequency"]},
                    output_root=root / "snapshots",
                )
            history_path, manifest_path, _ = self._case(root)
            with self.assertRaises((DataValidationError, ValueError)):
                build_engine_snapshot(
                    history_path,
                    manifest_path,
                    "not-a-date",
                    model_names=["frequency"],
                    model_configs={"frequency": MODEL_CONFIGS["frequency"]},
                    output_root=root / "bad-as-of",
                )

    def test_existing_m3_selection_evidence_is_reused_without_reselection(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            history_path, manifest_path, _ = self._case(root)
            config = MODEL_CONFIGS["frequency"]
            config_hash = hashlib.sha256(
                json.dumps(config, sort_keys=True, separators=(",", ":")).encode("utf-8")
            ).hexdigest()
            selection_path = root / "m3" / "run_config.json"
            write_json(
                {
                    "schema_version": "m3-selection-v1",
                    "input_sha256": file_sha256(history_path),
                    "season_mode": "carry",
                    "validation_folds": [
                        {"fold_id": "validation_1", "start": "2024-01-02", "end": "2024-01-04"},
                        {"fold_id": "validation_2", "start": "2024-01-05", "end": "2024-01-06"},
                    ],
                    "final_test": {"start": "2024-01-07", "end": "2024-01-09"},
                    "selection_rule": {"final_test_metrics_used_for_selection": False},
                    "selected_candidate": {
                        "candidate_name": "frequency_alpha_1",
                        "model": "frequency",
                        "config": config,
                        "configuration_sha256": config_hash,
                    },
                },
                selection_path,
            )
            result = build_engine_snapshot(
                history_path,
                manifest_path,
                "2024-01-07",
                selection_run_config_path=selection_path,
                output_root=root / "snapshots",
            )
            self.assertEqual(result["model_names"], ["frequency"])
            self.assertEqual(result["manifest"]["selection"]["mode"], "m3_run_config")
            self.assertEqual(
                result["manifest"]["models"][0]["config_source"]["mode"],
                "selection_run_config",
            )

    def test_model_state_codec_rejects_unknown_fields(self):
        model = create_model("frequency", MODEL_CONFIGS["frequency"])
        record = export_model_state(model, MODEL_CONFIGS["frequency"])
        record["state"]["opaque"] = 1
        record["state_sha256"] = hashlib.sha256(b"tampered").hexdigest()
        with self.assertRaises(DataValidationError):
            import_model_state(record)

    def test_snapshot_predictions_match_m4_reference_for_every_model(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            history_path, manifest_path, fixtures_path = self._case(root)
            history = load_matches(history_path)
            fixtures = pd.read_csv(fixtures_path, dtype="string")
            for name, config in MODEL_CONFIGS.items():
                with self.subTest(model=name):
                    built = build_engine_snapshot(
                        history_path,
                        manifest_path,
                        "2024-01-07",
                        model_names=[name],
                        model_configs={name: config},
                        output_root=root / f"snapshots-{name}",
                    )
                    predicted = predict_from_engine_snapshot(
                        built["snapshot_manifest_path"],
                        fixtures_path,
                        output_root=root / f"runs-{name}",
                    )
                    reference = forecast_fixtures(
                        history,
                        fixtures,
                        name,
                        "2024-01-07",
                        model_config=config,
                    )
                    actual = predicted["predictions"]
                    assert_frame_equal(
                        actual.reset_index(drop=True),
                        reference.predictions.reset_index(drop=True),
                        check_dtype=False,
                        check_exact=True,
                    )
                    run_config = json.loads(
                        Path(predicted["run_config_path"]).read_text(encoding="utf-8")
                    )
                    self.assertEqual(
                        run_config["snapshot"]["manifest_sha256"],
                        file_sha256(built["snapshot_manifest_path"]),
                    )
                    self.assertEqual(
                        run_config["fixtures"]["sha256"], file_sha256(fixtures_path)
                    )

    def test_prediction_does_not_need_history_after_build(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            history_path, manifest_path, fixtures_path = self._case(root)
            built = self._build_one(root, "frequency")
            history_path.unlink()
            manifest_path.unlink()
            result = predict_from_engine_snapshot(
                built["snapshot_manifest_path"], fixtures_path, output_root=root / "runs"
            )
            self.assertTrue(Path(result["predictions_path"]).is_file())

    def test_invalid_snapshot_and_fixture_inputs_fail_with_explicit_errors(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            history_path, manifest_path, fixtures_path = self._case(root)
            built = build_engine_snapshot(
                history_path,
                manifest_path,
                "2024-01-07",
                model_names=["frequency"],
                model_configs={"frequency": MODEL_CONFIGS["frequency"]},
                output_root=root / "snapshots",
            )
            state_path = Path(built["model_states_path"])
            state_path.write_text(state_path.read_text(encoding="utf-8").replace("frequency", "unknown", 1), encoding="utf-8")
            with self.assertRaisesRegex(DataValidationError, "SHA-256|hash"):
                predict_from_engine_snapshot(
                    built["snapshot_manifest_path"], fixtures_path, output_root=root / "runs"
                )

            built = self._build_one(root, "frequency")
            bad_fixture = root / "bad-fixtures.csv"
            bad = _fixtures_frame()
            bad.loc[0, "date"] = "2024-01-06"
            bad.to_csv(bad_fixture, index=False)
            with self.assertRaisesRegex(DataValidationError, "on or after as_of"):
                predict_from_engine_snapshot(
                    built["snapshot_manifest_path"], bad_fixture, output_root=root / "bad-runs"
                )
            scored = _fixtures_frame()
            scored["home_goals"] = [1, ""]
            scored.to_csv(bad_fixture, index=False)
            with self.assertRaisesRegex(DataValidationError, "known scores"):
                predict_from_engine_snapshot(
                    built["snapshot_manifest_path"], bad_fixture, output_root=root / "scored-runs"
                )

    def test_cli_build_and_predict_are_offline_and_concise(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            history_path, manifest_path, fixtures_path = self._case(root)
            config_path = root / "frequency.json"
            write_json(MODEL_CONFIGS["frequency"], config_path)
            from matchquant.cli import main

            build_output = io.StringIO()
            with patch("urllib.request.urlopen", side_effect=AssertionError("network")), contextlib.redirect_stdout(build_output):
                with self.assertRaises(SystemExit) as build_exit:
                    main(
                        [
                            "engine-snapshot-build",
                            "--history",
                            str(history_path),
                            "--history-manifest",
                            str(manifest_path),
                            "--as-of",
                            "2024-01-07",
                            "--model",
                            "frequency",
                            "--model-config",
                            str(config_path),
                            "--snapshot-root",
                            str(root / "snapshots"),
                        ]
                    )
            self.assertEqual(build_exit.exception.code, 0)
            self.assertLessEqual(len(build_output.getvalue().strip().splitlines()), 20)
            manifest_path_built = next((root / "snapshots").glob("*/snapshot_manifest.json"))

            predict_output = io.StringIO()
            with patch("urllib.request.urlopen", side_effect=AssertionError("network")), contextlib.redirect_stdout(predict_output):
                with self.assertRaises(SystemExit) as predict_exit:
                    main(
                        [
                            "engine-snapshot-predict",
                            "--snapshot-manifest",
                            str(manifest_path_built),
                            "--fixtures",
                            str(fixtures_path),
                            "--output-root",
                            str(root / "runs"),
                        ]
                    )
            self.assertEqual(predict_exit.exception.code, 0)
            self.assertLessEqual(len(predict_output.getvalue().strip().splitlines()), 20)
