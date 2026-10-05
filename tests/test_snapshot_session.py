"""B03 tests: snapshot prediction sessions, read-only run audits, and issue evidence.

The research-side synthetic builder (history, pointer, M12 evidence) and the
B03 runtime window (session / verify / settle) are kept strictly separate: the
access guards only observe the runtime window.
"""

import builtins
import contextlib
import io as io_module
import json
import shutil
import tempfile
import unittest
import urllib.request
from pathlib import Path
from unittest.mock import patch

import pandas as pd
from pandas.testing import assert_frame_equal

from matchquant.cli import main
import matchquant.snapshot_run_verify as snapshot_run_verify_module
from matchquant.engine_snapshot import build_engine_snapshot, predict_from_engine_snapshot
from matchquant.engine_snapshot_status import check_engine_snapshot_status
from matchquant.forecast import run_forecast
from matchquant.io import file_sha256, write_json
from matchquant.schema import DataValidationError, REQUIRED_COLUMNS
from matchquant.settlement import settle_forecast_run
from matchquant.snapshot_audit_support import (
    SESSION_DIR_PREFIX,
    ensure_safe_output_root,
)
from matchquant.snapshot_run_verify import verify_snapshot_run
from matchquant.snapshot_session import run_snapshot_session


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
DATASET_VERSION = "m2.1-b03-test"
FORBIDDEN_PATH_LABELS = ("history.csv", "history_manifest.json", "epl_current.json")


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


class SnapshotSessionTestBase(unittest.TestCase):
    def _write_history_evidence(self, root: Path, frame: pd.DataFrame | None = None) -> tuple[Path, Path]:
        root.mkdir(parents=True, exist_ok=True)
        frame = frame if frame is not None else _history_frame()
        history_path = root / "history.csv"
        manifest_path = root / "history_manifest.json"
        frame.to_csv(history_path, index=False)
        write_json(
            {
                "schema_version": "m2.1-processed-manifest-v1",
                "dataset_version": DATASET_VERSION,
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

    def _write_pointer(self, root: Path, history_path: Path, history_manifest_path: Path) -> Path:
        root.mkdir(parents=True, exist_ok=True)
        pointer_path = root / "epl_current.json"
        write_json(
            {
                "schema_version": "m2.1-current-pointer-v1",
                "dataset_version": DATASET_VERSION,
                "canonical": str(history_path.resolve()),
                "processed_manifest": str(history_manifest_path.resolve()),
                "canonical_sha256": file_sha256(history_path),
            },
            pointer_path,
        )
        return pointer_path

    def _build_snapshot(self, root: Path, model: str, *, as_of: str = "2024-01-07") -> dict:
        history_path, manifest_path = self._write_history_evidence(root / "research")
        return build_engine_snapshot(
            history_path,
            manifest_path,
            as_of,
            model_names=[model],
            model_configs={model: MODEL_CONFIGS[model]},
            output_root=root / "runtime" / "snapshots",
        )

    def _write_fixtures(self, root: Path, frame: pd.DataFrame | None = None) -> Path:
        fixtures_path = root / "runtime" / "inputs" / "fixtures.csv"
        fixtures_path.parent.mkdir(parents=True, exist_ok=True)
        (frame if frame is not None else _fixtures_frame()).to_csv(fixtures_path, index=False)
        return fixtures_path

    def _m11_run(self, root: Path, model: str, *, as_of: str = "2024-01-07") -> tuple[Path, dict, Path]:
        built = self._build_snapshot(root, model, as_of=as_of)
        fixtures_path = self._write_fixtures(root)
        predicted = predict_from_engine_snapshot(
            built["snapshot_manifest_path"], fixtures_path, output_root=root / "runtime" / "runs"
        )
        return Path(predicted["run_dir"]), built, fixtures_path

    def _results_file(self, root: Path) -> Path:
        path = root / "runtime" / "inputs" / "results.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        _history_frame().loc[:, REQUIRED_COLUMNS].to_csv(path, index=False)
        return path

    @contextlib.contextmanager
    def _runtime_guard(self):
        """Observe the B03 runtime window: builtins.open, io.open, urllib, socket."""

        import socket as socket_module

        opened: list[str] = []
        network_calls: list[str] = []
        real_open = builtins.open
        real_io_open = io_module.open
        real_socket = socket_module.socket

        def guarded_open(module_label, file, *args, **kwargs):
            if isinstance(file, (str, Path)):
                opened.append(f"{module_label}:{file}")
            return real_open(file, *args, **kwargs)

        def guarded_socket(*args, **kwargs):
            network_calls.append("socket.socket")
            return real_socket(*args, **kwargs)

        with (
            patch("builtins.open", side_effect=lambda file, *a, **k: guarded_open("builtins", file, *a, **k)),
            patch("io.open", side_effect=lambda file, *a, **k: guarded_open("io", file, *a, **k)),
            patch.object(urllib.request, "urlopen", side_effect=AssertionError("network used")),
            patch("socket.socket", side_effect=guarded_socket),
        ):
            yield opened, network_calls

    def _fresh_issue_evidence(self, root: Path, built: dict) -> Path:
        """Research-side: produce a real M12 fresh record for the snapshot via its pointer."""

        history_path = root / "research" / "history.csv"
        history_manifest_path = root / "research" / "history_manifest.json"
        pointer_path = self._write_pointer(root / "research", history_path, history_manifest_path)
        status = check_engine_snapshot_status(
            built["snapshot_manifest_path"], pointer_path, output_root=root / "research" / "m12_out"
        )
        self.assertEqual(status["status"], "fresh")
        issue_path = root / "runtime" / "inputs" / "issue_fresh.json"
        issue_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(status["status_manifest_path"], issue_path)
        return issue_path

    def _m12_issue_record(self, root: Path, built: dict, pointer_path: Path | None) -> dict:
        status = check_engine_snapshot_status(
            built["snapshot_manifest_path"],
            pointer_path if pointer_path is not None else root / "research" / "missing_pointer.json",
            output_root=root / "research" / "m12_out",
        )
        return status


class SessionClosureTests(SnapshotSessionTestBase):
    def test_session_replay_ready_audit_needs_review_and_settle_closure_for_every_model(self):
        for model in ("frequency", "elo", "poisson"):
            with self.subTest(model=model):
                with tempfile.TemporaryDirectory() as temporary:
                    root = Path(temporary)
                    built = self._build_snapshot(root, model)
                    fixtures_path = self._write_fixtures(root)
                    results_path = self._results_file(root)

                    session = run_snapshot_session(
                        built["snapshot_manifest_path"],
                        fixtures_path,
                        mode="replay",
                        output_root=root / "runtime" / "sessions",
                    )
                    self.assertEqual(session["status"], "ready")
                    self.assertEqual(session["audit_status"], "needs_review")
                    self.assertEqual(session["audit_reasons"], ["issue_evidence_missing"])
                    self.assertTrue(session["predictions_usable"])
                    manifest = json.loads(
                        Path(session["session_manifest_path"]).read_text(encoding="utf-8")
                    )
                    self.assertEqual(manifest["schema_version"], "baseline-snapshot-session-v1")
                    self.assertEqual(manifest["mode"], "replay")
                    self.assertTrue(manifest["inputs_unchanged"])
                    self.assertEqual(
                        Path(manifest["prediction_run"]["run_dir"]).parent,
                        Path(session["session_dir"]),
                    )
                    self.assertEqual(
                        manifest["output_hashes"]["session_report"],
                        file_sha256(Path(session["session_report_path"])),
                    )
                    self.assertIn("output_hashes", manifest)
                    self.assertNotIn("session_manifest_sha256", manifest)
                    audit_payload = json.loads(
                        Path(session["prediction_audit_path"]).read_text(encoding="utf-8")
                    )
                    self.assertEqual(audit_payload["status"], "needs_review")
                    self.assertEqual(audit_payload["schema_version"], "baseline-snapshot-run-audit-v1")
                    self.assertTrue(audit_payload["inputs_unchanged"])
                    self.assertNotIn("audit_json_sha256", audit_payload)
                    self.assertEqual(
                        file_sha256(Path(session["session_manifest_path"])),
                        session["session_manifest_sha256"],
                    )

                    run_dir = Path(session["prediction_run_dir"])
                    m4_config_path = root / "research" / f"{model}.json"
                    write_json(MODEL_CONFIGS[model], m4_config_path)
                    m4_result = run_forecast(
                        root / "research" / "history.csv",
                        fixtures_path,
                        "2024-01-07",
                        model_name=model,
                        model_config_path=m4_config_path,
                        output_root=root / "runtime" / "m4",
                    )
                    m4_run_dir = Path(m4_result["run_dir"])
                    m11_predictions = pd.read_csv(run_dir / "predictions.csv")
                    m4_predictions = pd.read_csv(m4_run_dir / "predictions.csv")
                    assert_frame_equal(
                        m11_predictions.sort_values("fixture_id").reset_index(drop=True),
                        m4_predictions.sort_values("fixture_id").reset_index(drop=True),
                        check_exact=False,
                        atol=1e-12,
                        rtol=0,
                    )
                    m11_settlement = settle_forecast_run(
                        run_dir, results_path, output_root=root / "runtime" / "settle-m11"
                    )
                    m4_settlement = settle_forecast_run(
                        m4_run_dir, results_path, output_root=root / "runtime" / "settle-m4"
                    )
                    self.assertEqual(m11_settlement["status"], "complete")
                    self.assertEqual(m11_settlement["counts"]["settled"], 2)
                    self.assertEqual(m11_settlement["counts"], m4_settlement["counts"])
                    for key in ("log_loss", "brier_score", "accuracy", "coverage"):
                        m11_value = m11_settlement["metrics"][key]
                        m4_value = m4_settlement["metrics"][key]
                        if m11_value is None:
                            self.assertIsNone(m4_value)
                        else:
                            self.assertAlmostEqual(m11_value, m4_value, places=12)

                    input_hashes = {
                        path: file_sha256(path)
                        for path in (
                            Path(built["snapshot_manifest_path"]),
                            Path(built["model_states_path"]),
                            fixtures_path,
                            run_dir / "predictions.csv",
                            run_dir / "run_config.json",
                        )
                    }
                    for path, digest in input_hashes.items():
                        self.assertEqual(file_sha256(path), digest, str(path))

    def test_session_research_is_needs_review_even_with_fresh_issue(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            model = "frequency"
            built = self._build_snapshot(root, model)
            fixtures_path = self._write_fixtures(root)
            issue_path = self._fresh_issue_evidence(root, built)
            session = run_snapshot_session(
                built["snapshot_manifest_path"],
                fixtures_path,
                mode="research",
                output_root=root / "runtime" / "sessions",
                issue_evidence_path=issue_path,
            )
            self.assertEqual(session["status"], "needs_review")
            self.assertEqual(session["audit_status"], "ready")
            self.assertIn("coverage_to_fixture_time_not_proven", session["reasons"])
            manifest = json.loads(Path(session["session_manifest_path"]).read_text(encoding="utf-8"))
            self.assertEqual(manifest["coverage"]["status"], "not_proven")
            self.assertEqual(manifest["audit"]["status"], "ready")
            self.assertTrue(manifest["audit"]["fresh_at_issue"])
            self.assertEqual(manifest["audit"]["active_link_status"], "not_checked_in_runtime")
            self.assertTrue(manifest["predictions_usable"])


class IssueEvidenceTests(SnapshotSessionTestBase):
    def test_fresh_issue_audit_is_ready_and_survives_disconnected_research_files(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            model = "elo"
            run_dir, built, fixtures_path = self._m11_run(root, model)
            results_path = self._results_file(root)
            issue_path = self._fresh_issue_evidence(root, built)
            self.assertTrue(issue_path.is_file())

            shutil.rmtree(root / "research")

            with self._runtime_guard() as (opened, network_calls):
                verification = verify_snapshot_run(
                    run_dir, output_root=root / "runtime" / "audits", issue_evidence_path=issue_path
                )
                session = run_snapshot_session(
                    built["snapshot_manifest_path"],
                    fixtures_path,
                    mode="replay",
                    output_root=root / "runtime" / "sessions",
                    issue_evidence_path=issue_path,
                )
                settlement = settle_forecast_run(
                    run_dir, results_path, output_root=root / "runtime" / "settle"
                )
            self.assertEqual(verification["status"], "ready")
            self.assertEqual(verification["reasons"], [])
            audit_payload = json.loads(
                Path(verification["prediction_audit_path"]).read_text(encoding="utf-8")
            )
            self.assertTrue(audit_payload["issue_evidence"]["fresh_at_issue"])
            self.assertEqual(
                audit_payload["issue_evidence"]["active_link_status"], "not_checked_in_runtime"
            )
            self.assertEqual(audit_payload["issue_evidence"]["record_status"], "fresh")
            self.assertEqual(session["status"], "ready")
            self.assertEqual(session["audit_status"], "ready")
            self.assertEqual(settlement["status"], "complete")
            self.assertEqual(settlement["counts"]["settled"], 2)

            with contextlib.redirect_stdout(io_module.StringIO()):
                with self.assertRaises(SystemExit) as verify_exit:
                    main(
                        [
                            "snapshot-run-verify",
                            "--forecast-run",
                            str(run_dir),
                            "--output-root",
                            str(root / "runtime" / "audits-cli"),
                            "--issue-evidence",
                            str(issue_path),
                        ]
                    )
            self.assertEqual(verify_exit.exception.code, 0)
            with contextlib.redirect_stdout(io_module.StringIO()):
                with self.assertRaises(SystemExit) as session_exit:
                    main(
                        [
                            "snapshot-session",
                            "--snapshot-manifest",
                            str(built["snapshot_manifest_path"]),
                            "--fixtures",
                            str(fixtures_path),
                            "--mode",
                            "replay",
                            "--output-root",
                            str(root / "runtime" / "sessions-cli"),
                            "--issue-evidence",
                            str(issue_path),
                        ]
                    )
            self.assertEqual(session_exit.exception.code, 0)

    def test_stale_and_unverifiable_issue_records_stay_needs_review(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            model = "frequency"
            run_dir, built, _ = self._m11_run(root, model)

            other_history = _history_frame()
            other_history.loc[other_history["match_id"].eq("h1"), "home_goals"] = 9
            other_history_path, other_manifest_path = self._write_history_evidence(
                root / "research" / "other", other_history
            )
            stale_pointer = self._write_pointer(
                root / "research", other_history_path, other_manifest_path
            )
            stale_record = self._m12_issue_record(root, built, stale_pointer)
            self.assertEqual(stale_record["status"], "stale")
            stale_path = root / "runtime" / "inputs" / "issue_stale.json"
            shutil.copyfile(stale_record["status_manifest_path"], stale_path)
            stale_verification = verify_snapshot_run(
                run_dir, output_root=root / "runtime" / "audits-stale", issue_evidence_path=stale_path
            )
            self.assertEqual(stale_verification["status"], "needs_review")
            self.assertEqual(stale_verification["reasons"][0], "issue_evidence_stale")
            self.assertFalse(
                json.loads(Path(stale_verification["prediction_audit_path"]).read_text(encoding="utf-8"))[
                    "issue_evidence"
                ]["fresh_at_issue"]
            )

            missing_pointer_record = self._m12_issue_record(root, built, None)
            self.assertEqual(missing_pointer_record["status"], "unverifiable")
            unverifiable_path = root / "runtime" / "inputs" / "issue_unverifiable.json"
            shutil.copyfile(missing_pointer_record["status_manifest_path"], unverifiable_path)
            unverifiable_verification = verify_snapshot_run(
                run_dir,
                output_root=root / "runtime" / "audits-unverifiable",
                issue_evidence_path=unverifiable_path,
            )
            self.assertEqual(unverifiable_verification["status"], "needs_review")
            self.assertEqual(unverifiable_verification["reasons"][0], "issue_evidence_unverifiable")

            with contextlib.redirect_stdout(io_module.StringIO()):
                with self.assertRaises(SystemExit) as stale_exit:
                    main(
                        [
                            "snapshot-run-verify",
                            "--forecast-run",
                            str(run_dir),
                            "--output-root",
                            str(root / "runtime" / "audits-stale-cli"),
                            "--issue-evidence",
                            str(stale_path),
                        ]
                    )
            self.assertEqual(stale_exit.exception.code, 2)

    def _corrupted_issue_variants(self, root: Path, built: dict, run_dir: Path) -> dict[str, Path]:
        issue_dir = root / "runtime" / "bad-issues"
        issue_dir.mkdir(parents=True, exist_ok=True)
        variants: dict[str, Path] = {}

        malformed = issue_dir / "malformed.json"
        malformed.write_text("{not-json", encoding="utf-8")
        variants["malformed_json"] = malformed

        unknown_schema = issue_dir / "unknown_schema.json"
        write_json({"schema_version": "made-up-status-v9", "status": "fresh"}, unknown_schema)
        variants["unknown_schema"] = unknown_schema

        wrong_target_payload = {
            "schema_version": "m12-engine-snapshot-status-v1",
            "generated_at": "2026-10-05T00:00:00Z",
            "status": "fresh",
            "snapshot": {
                "status": "valid",
                "snapshot_id": "m11_engine_snapshot_elsewhere",
                "sha256": "0" * 64,
                "state_sha256": "0" * 64,
            },
        }
        wrong_target = issue_dir / "wrong_target.json"
        write_json(wrong_target_payload, wrong_target)
        variants["snapshot_target_mismatch"] = wrong_target

        fresh_base = json.loads(self._fresh_issue_evidence(root, built).read_text(encoding="utf-8"))
        future = issue_dir / "future.json"
        future_payload = dict(fresh_base)
        future_payload["generated_at"] = "2099-01-01T00:00:00Z"
        write_json(future_payload, future)
        variants["future_generated_at"] = future

        contradicted = issue_dir / "contradicted.json"
        contradicted_payload = dict(fresh_base)
        contradicted_payload["reasons"] = ["canonical_hash_mismatch"]
        write_json(contradicted_payload, contradicted)
        variants["fresh_with_reasons"] = contradicted

        invalid_pointer = issue_dir / "invalid_pointer.json"
        invalid_pointer_payload = dict(fresh_base)
        invalid_pointer_payload["current_pointer"] = {
            "status": "invalid",
            "dataset_version": DATASET_VERSION,
        }
        write_json(invalid_pointer_payload, invalid_pointer)
        variants["fresh_with_invalid_pointer"] = invalid_pointer

        return variants

    def test_corrupted_issue_evidence_is_blocked_with_exit_1(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            model = "frequency"
            run_dir, built, _ = self._m11_run(root, model)
            variants = self._corrupted_issue_variants(root, built, run_dir)
            for name, issue_path in variants.items():
                with self.subTest(variant=name):
                    with contextlib.redirect_stdout(io_module.StringIO()):
                        with self.assertRaises(SystemExit) as blocked_exit:
                            main(
                                [
                                    "snapshot-run-verify",
                                    "--forecast-run",
                                    str(run_dir),
                                    "--output-root",
                                    str(root / "runtime" / "audits-bad"),
                                    "--issue-evidence",
                                    str(issue_path),
                                ]
                            )
                    self.assertEqual(blocked_exit.exception.code, 1, name)
            audit_dirs = sorted((root / "runtime" / "audits-bad").iterdir())
            self.assertEqual(len(audit_dirs), len(variants))
            newest = json.loads(
                (audit_dirs[-1] / "prediction_audit.json").read_text(encoding="utf-8")
            )
            self.assertEqual(newest["status"], "blocked")
            self.assertFalse(newest["predictions_usable"])

            session = run_snapshot_session(
                built["snapshot_manifest_path"],
                self._write_fixtures(root),
                mode="replay",
                output_root=root / "runtime" / "sessions-bad",
                issue_evidence_path=variants["future_generated_at"],
            )
            self.assertEqual(session["status"], "blocked")
            self.assertFalse(session["predictions_usable"])
            self.assertTrue(Path(session["prediction_audit_path"]).is_file())
            self.assertTrue(Path(session["session_report_path"]).is_file())

    def test_issue_equality_flags_are_recomputed_from_records_not_trusted(self):
        """R1: six evaluator counterexamples plus a fully-synced hash forgery must all
        be blocked at the record internal-consistency gate, never as an unrelated
        hash mismatch of the audited snapshot files."""

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            model = "frequency"
            run_dir, built, _ = self._m11_run(root, model)
            base_payload = json.loads(
                self._fresh_issue_evidence(root, built).read_text(encoding="utf-8")
            )
            other_canonical = str(root / "research" / "some_other_canonical.csv")
            variants: dict[str, tuple[object, str]] = {}

            def canonical_hash(payload: dict) -> None:
                payload["current_pointer"]["canonical"]["sha256"] = "0" * 64
                payload["comparison"]["canonical"]["pointer_sha256"] = "0" * 64

            def canonical_hash_fully_synced(payload: dict) -> None:
                canonical_hash(payload)
                payload["current_pointer"]["canonical"]["recorded_sha256"] = "0" * 64
                payload["current_pointer"]["processed_manifest"]["output_sha256"] = "0" * 64

            def manifest_hash(payload: dict) -> None:
                payload["current_pointer"]["processed_manifest"]["sha256"] = "1" * 64
                payload["comparison"]["processed_manifest"]["pointer_sha256"] = "1" * 64

            def dataset_version(payload: dict) -> None:
                payload["current_pointer"]["dataset_version"] = "m2.1-forged"
                payload["comparison"]["dataset_version"]["pointer"] = "m2.1-forged"
                payload["current_pointer"]["processed_manifest"]["dataset_version"] = "m2.1-forged"

            def canonical_path(payload: dict) -> None:
                payload["current_pointer"]["canonical"]["path"] = other_canonical
                payload["comparison"]["canonical"]["pointer_path"] = other_canonical

            def snapshot_status(payload: dict) -> None:
                payload["snapshot"]["status"] = "invalid"

            def bound_cutoff(payload: dict) -> None:
                payload["snapshot"]["history"]["used_data_cutoff"] = "2099-01-01"

            flag_gate = "contradicts"
            for name, mutator, marker in (
                ("canonical_hash_synced_pointer_and_comparison", canonical_hash, (flag_gate, "recorded_sha256")),
                ("canonical_hash_fully_synced", canonical_hash_fully_synced, (flag_gate,)),
                ("manifest_hash_synced_pointer_and_comparison", manifest_hash, (flag_gate,)),
                ("dataset_version_synced_pointer_and_comparison", dataset_version, (flag_gate,)),
                ("canonical_path_synced_pointer_and_comparison", canonical_path, (flag_gate,)),
                ("snapshot_status_invalid", snapshot_status, ("must be 'valid'",)),
                ("bound_used_cutoff_2099", bound_cutoff, ("strictly loaded snapshot record",)),
            ):
                variants[name] = (mutator, marker)

            issue_dir = root / "runtime" / "inputs" / "recomputed-issues"
            for name, (mutator, marker) in variants.items():
                payload = json.loads(json.dumps(base_payload))
                mutator(payload)
                issue_path = issue_dir / f"{name}.json"
                write_json(payload, issue_path)
                with self.subTest(variant=name):
                    verification = verify_snapshot_run(
                        run_dir,
                        output_root=root / "runtime" / "audits-recomputed",
                        issue_evidence_path=issue_path,
                    )
                    self.assertEqual(verification["status"], "blocked", name)
                    reason = verification["reasons"][0]
                    self.assertTrue(reason.startswith("issue_evidence_invalid:"), reason)
                    self.assertTrue(any(expected in reason for expected in marker), reason)
                    self.assertNotIn("snapshot appears modified or tampered", reason)
                    self.assertNotIn("state SHA-256 does not match the actual state file", reason)
                    self.assertFalse(verification["predictions_usable"])
                    with contextlib.redirect_stderr(io_module.StringIO()), contextlib.redirect_stdout(io_module.StringIO()):
                        with self.assertRaises(SystemExit) as verify_exit:
                            main(
                                [
                                    "snapshot-run-verify",
                                    "--forecast-run",
                                    str(run_dir),
                                    "--output-root",
                                    str(root / "runtime" / "audits-recomputed-cli"),
                                    "--issue-evidence",
                                    str(issue_path),
                                ]
                            )
                    self.assertEqual(verify_exit.exception.code, 1, name)

            session = run_snapshot_session(
                built["snapshot_manifest_path"],
                self._write_fixtures(root),
                mode="replay",
                output_root=root / "runtime" / "sessions-recomputed",
                issue_evidence_path=issue_dir / "canonical_hash_fully_synced.json",
            )
            self.assertEqual(session["status"], "blocked")
            self.assertIn("issue_evidence_invalid", session["reasons"][0])
            with contextlib.redirect_stderr(io_module.StringIO()), contextlib.redirect_stdout(io_module.StringIO()):
                with self.assertRaises(SystemExit) as session_exit:
                    main(
                        [
                            "snapshot-session",
                            "--snapshot-manifest",
                            str(built["snapshot_manifest_path"]),
                            "--fixtures",
                            str(self._write_fixtures(root / "recomputed-session-case")),
                            "--mode",
                            "research",
                            "--output-root",
                            str(root / "runtime" / "sessions-recomputed-cli"),
                            "--issue-evidence",
                            str(issue_dir / "bound_used_cutoff_2099.json"),
                        ]
                    )
            self.assertEqual(session_exit.exception.code, 1)

    def test_issue_metadata_values_and_state_classification_are_validated(self):
        """R1 residual: exact keys alone are not evidence — recorded values must have
        real types and fixed schema values, and stale/unverifiable must follow the
        M12 status machine (pointer-invalid + no comparison is never stale)."""

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            model = "frequency"
            run_dir, built, _ = self._m11_run(root, model)
            base_payload = json.loads(
                self._fresh_issue_evidence(root, built).read_text(encoding="utf-8")
            )

            def unknown_pointer_schema(payload: dict) -> None:
                payload["current_pointer"]["schema_version"] = "invalid-pointer-schema"

            def pointer_bad_sha_type(payload: dict) -> None:
                payload["current_pointer"]["sha256"] = []

            def pointer_missing_path_value(payload: dict) -> None:
                payload["current_pointer"]["path"] = None

            def snapshot_bad_path_type(payload: dict) -> None:
                payload["snapshot"]["path"] = []

            def stale_with_invalid_pointer(payload: dict) -> None:
                payload["status"] = "stale"
                payload["current_pointer"]["status"] = "invalid"
                payload.pop("comparison")
                payload["reasons"] = ["current_pointer_invalid"]

            variants = (
                ("fresh_pointer_unknown_schema", unknown_pointer_schema, "schema_version must be"),
                ("fresh_pointer_bad_sha_type", pointer_bad_sha_type, "must be a 64-character SHA-256"),
                ("fresh_pointer_missing_path_value", pointer_missing_path_value, "current_pointer.path"),
                ("fresh_snapshot_bad_path_type", snapshot_bad_path_type, "snapshot.path"),
                (
                    "stale_with_invalid_pointer_and_no_comparison",
                    stale_with_invalid_pointer,
                    "is unverifiable, not stale",
                ),
            )
            issue_dir = root / "runtime" / "inputs" / "residual-issues"
            for name, mutator, marker in variants:
                payload = json.loads(json.dumps(base_payload))
                mutator(payload)
                issue_path = issue_dir / f"{name}.json"
                write_json(payload, issue_path)
                with self.subTest(variant=name):
                    verification = verify_snapshot_run(
                        run_dir,
                        output_root=root / "runtime" / "audits-residual",
                        issue_evidence_path=issue_path,
                    )
                    self.assertEqual(verification["status"], "blocked", name)
                    reason = verification["reasons"][0]
                    self.assertTrue(reason.startswith("issue_evidence_invalid:"), reason)
                    self.assertIn(marker, reason)
                    self.assertFalse(verification["predictions_usable"])
                    with contextlib.redirect_stderr(io_module.StringIO()), contextlib.redirect_stdout(io_module.StringIO()):
                        with self.assertRaises(SystemExit) as verify_exit:
                            main(
                                [
                                    "snapshot-run-verify",
                                    "--forecast-run",
                                    str(run_dir),
                                    "--output-root",
                                    str(root / "runtime" / "audits-residual-cli"),
                                    "--issue-evidence",
                                    str(issue_path),
                                ]
                            )
                    self.assertEqual(verify_exit.exception.code, 1, name)


class AuditNegativeTests(SnapshotSessionTestBase):
    def test_probability_tamper_with_synced_recorded_hash_is_rejected_by_repredict(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run_dir, built, _ = self._m11_run(root, "frequency")
            predictions_path = run_dir / "predictions.csv"
            frame = pd.read_csv(predictions_path)
            argmax = frame.loc[0, "model_most_likely_result"]
            column_for = {"H": "home_win_prob", "D": "draw_prob", "A": "away_win_prob"}
            frame.loc[0, column_for[argmax]] = frame.loc[0, column_for[argmax]] + 0.02
            for result, column in column_for.items():
                if result != argmax:
                    frame.loc[0, column] = frame.loc[0, column] - 0.01
            frame.to_csv(predictions_path, index=False, encoding="utf-8-sig")
            new_hash = file_sha256(predictions_path)
            config_path = run_dir / "run_config.json"
            config = json.loads(config_path.read_text(encoding="utf-8"))
            config["predictions_sha256"] = new_hash
            config["outputs"]["predictions_sha256"] = new_hash
            write_json(config, config_path)

            verification = verify_snapshot_run(
                run_dir, output_root=root / "runtime" / "audits"
            )
            self.assertEqual(verification["status"], "blocked")
            audit_payload = json.loads(
                Path(verification["prediction_audit_path"]).read_text(encoding="utf-8")
            )
            self.assertTrue(
                any(check["check"] == "repredict_probabilities_match" and check["status"] == "failed"
                    for check in audit_payload["engine_checks"])
            )
            self.assertIn("repredict_mismatch_or_failure", verification["reasons"][0])
            self.assertFalse(audit_payload["predictions_usable"])
            self.assertTrue(audit_payload["inputs_unchanged"])
            self.assertTrue(Path(verification["prediction_audit_report_path"]).is_file())

    def test_verify_rejects_m4_run(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            history_path, _ = self._write_history_evidence(root / "research")
            fixtures_path = self._write_fixtures(root)
            config_path = root / "runtime" / "frequency.json"
            write_json(MODEL_CONFIGS["frequency"], config_path)
            m4_result = run_forecast(
                history_path,
                fixtures_path,
                "2024-01-07",
                model_name="frequency",
                model_config_path=config_path,
                output_root=root / "runtime" / "m4",
            )
            with contextlib.redirect_stdout(io_module.StringIO()):
                with self.assertRaises(SystemExit) as blocked_exit:
                    main(
                        [
                            "snapshot-run-verify",
                            "--forecast-run",
                            str(m4_result["run_dir"]),
                            "--output-root",
                            str(root / "runtime" / "audits"),
                        ]
                    )
            self.assertEqual(blocked_exit.exception.code, 1)
            audit_dirs = list((root / "runtime" / "audits").iterdir())
            payload = json.loads(
                (audit_dirs[0] / "prediction_audit.json").read_text(encoding="utf-8")
            )
            self.assertEqual(payload["status"], "blocked")
            self.assertIn("m4-forecast-v1", payload["reasons"][0])

    def test_verify_blocks_on_tampered_snapshot_state_and_manifest(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run_dir, built, _ = self._m11_run(root, "frequency")
            state_path = Path(built["model_states_path"])
            state_path.write_text(
                state_path.read_text(encoding="utf-8").replace("frequency", "frequency2", 1),
                encoding="utf-8",
            )
            verification = verify_snapshot_run(run_dir, output_root=root / "runtime" / "audits-state")
            self.assertEqual(verification["status"], "blocked")
            self.assertIn("prediction_evidence_load_failed", verification["reasons"][0])

            run_dir2, built2, _ = self._m11_run(root / "case2", "frequency")
            manifest_path = Path(built2["snapshot_manifest_path"])
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["history"]["rows_used_before_as_of"] = 999
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            verification2 = verify_snapshot_run(run_dir2, output_root=root / "case2" / "audits")
            self.assertEqual(verification2["status"], "blocked")

    def test_session_blocks_on_scored_or_before_as_of_fixtures(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            built = self._build_snapshot(root, "frequency")
            scored = _fixtures_frame()
            scored["home_goals"] = [1, ""]
            scored_fixtures = self._write_fixtures(root / "scored-case", scored)
            session = run_snapshot_session(
                built["snapshot_manifest_path"],
                scored_fixtures,
                mode="replay",
                output_root=root / "scored-case" / "runtime" / "sessions",
            )
            self.assertEqual(session["status"], "blocked")
            self.assertFalse(session["predictions_usable"])
            manifest = json.loads(Path(session["session_manifest_path"]).read_text(encoding="utf-8"))
            self.assertEqual(manifest["stages"]["fixtures_load"], "failed")
            self.assertEqual(manifest["stages"]["prediction"], "not_run")
            self.assertTrue(Path(session["prediction_audit_path"]).is_file())

            early = _fixtures_frame()
            early.loc[0, "date"] = "2024-01-06"
            early_fixtures = self._write_fixtures(root / "early-case", early)
            session2 = run_snapshot_session(
                built["snapshot_manifest_path"],
                early_fixtures,
                mode="replay",
                output_root=root / "early-case" / "runtime" / "sessions",
            )
            self.assertEqual(session2["status"], "blocked")
            with contextlib.redirect_stdout(io_module.StringIO()):
                with self.assertRaises(SystemExit) as blocked_exit:
                    main(
                        [
                            "snapshot-session",
                            "--snapshot-manifest",
                            str(built["snapshot_manifest_path"]),
                            "--fixtures",
                            str(early_fixtures),
                            "--mode",
                            "replay",
                            "--output-root",
                            str(root / "early-case" / "runtime" / "sessions-cli"),
                        ]
                    )
            self.assertEqual(blocked_exit.exception.code, 1)

    def test_verify_blocks_on_corrupt_predictions_csv_with_saved_artifacts(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run_dir, built, _ = self._m11_run(root, "frequency")
            predictions_path = run_dir / "predictions.csv"
            frame = pd.read_csv(predictions_path, dtype="string", keep_default_na=False)
            frame.loc[0, "home_win_prob"] = "1.5"
            frame.to_csv(predictions_path, index=False, encoding="utf-8-sig")
            new_hash = file_sha256(predictions_path)
            config_path = run_dir / "run_config.json"
            config = json.loads(config_path.read_text(encoding="utf-8"))
            config["predictions_sha256"] = new_hash
            config["outputs"]["predictions_sha256"] = new_hash
            write_json(config, config_path)
            verification = verify_snapshot_run(run_dir, output_root=root / "runtime" / "audits")
            self.assertEqual(verification["status"], "blocked")
            payload = json.loads(
                Path(verification["prediction_audit_path"]).read_text(encoding="utf-8")
            )
            self.assertFalse(payload["predictions_usable"])
            report = Path(verification["prediction_audit_report_path"]).read_text(encoding="utf-8")
            self.assertIn("blocked", report)

    def test_output_inside_missing_input_paths_is_refused_before_creation(self):
        """R2 residual B: declared input types protect missing runs and missing files
        too — the output guard never relies on the input existing."""

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run_dir, built, fixtures_path = self._m11_run(root, "frequency")
            missing_run = root / "runtime" / "ghost_prediction_run"
            self.assertFalse(missing_run.exists())
            with self.assertRaises((DataValidationError, OSError)):
                verify_snapshot_run(missing_run, output_root=missing_run / "audit_outputs")
            self.assertFalse(missing_run.exists())
            self.assertFalse((missing_run / "audit_outputs").exists())

            inputs_dir = fixtures_path.parent
            with self.assertRaises((DataValidationError, OSError)):
                run_snapshot_session(
                    built["snapshot_manifest_path"],
                    inputs_dir / "missing_fixtures.csv",
                    mode="replay",
                    output_root=inputs_dir / "sessions",
                )
            self.assertFalse((inputs_dir / "sessions").exists())
            self.assertFalse((inputs_dir / "missing_fixtures.csv").exists())

            with contextlib.redirect_stderr(io_module.StringIO()), contextlib.redirect_stdout(io_module.StringIO()):
                with self.assertRaises(SystemExit) as refused_exit:
                    main(
                        [
                            "snapshot-run-verify",
                            "--forecast-run",
                            str(missing_run),
                            "--output-root",
                            str(missing_run / "audit_outputs_cli"),
                        ]
                    )
            self.assertEqual(refused_exit.exception.code, 1)
            self.assertFalse(missing_run.exists())

            missing_run_outside = root / "runtime" / "audits-ghost"
            with contextlib.redirect_stderr(io_module.StringIO()), contextlib.redirect_stdout(io_module.StringIO()):
                with self.assertRaises(SystemExit) as blocked_exit:
                    main(
                        [
                            "snapshot-run-verify",
                            "--forecast-run",
                            str(missing_run),
                            "--output-root",
                            str(missing_run_outside),
                        ]
                    )
            self.assertEqual(blocked_exit.exception.code, 1)
            ghost_audits = list(missing_run_outside.iterdir())
            self.assertEqual(len(ghost_audits), 1)
            payload = json.loads((ghost_audits[0] / "prediction_audit.json").read_text(encoding="utf-8"))
            self.assertEqual(payload["status"], "blocked")
            self.assertIn("run_directory_does_not_exist", payload["reasons"][0])

    def test_unsafe_output_roots_are_refused_without_artifacts(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run_dir, built, fixtures_path = self._m11_run(root, "frequency")
            snapshot_dir = Path(built["snapshot_manifest_path"]).parent
            cases = [
                snapshot_dir / "out",
                snapshot_dir,
                run_dir,
                run_dir / "out",
                fixtures_path.parent / "out",
                Path(r"D:\Codex\matchquant_runs") / "b03-must-not-write",
                Path(r"D:\Codex\matchquant_data") / "b03-must-not-write",
                run_dir.parent / (SESSION_DIR_PREFIX + "existing"),
            ]
            (run_dir.parent / (SESSION_DIR_PREFIX + "existing")).mkdir(parents=True, exist_ok=True)
            for output_root in cases:
                with self.subTest(output_root=str(output_root)):
                    with self.assertRaises((DataValidationError, OSError)):
                        ensure_safe_output_root(
                            output_root,
                            [
                                (run_dir, "dir"),
                                (Path(built["snapshot_manifest_path"]), "file"),
                                (fixtures_path, "file"),
                            ],
                        )
                    before = set(run_dir.parent.iterdir())
                    with self.assertRaises((DataValidationError, OSError)):
                        verify_snapshot_run(run_dir, output_root=output_root)
                    with self.assertRaises((DataValidationError, OSError)):
                        run_snapshot_session(
                            built["snapshot_manifest_path"],
                            fixtures_path,
                            mode="replay",
                            output_root=output_root,
                        )
                    after = set(run_dir.parent.iterdir())
                    self.assertEqual(before, after)
            self.assertFalse((snapshot_dir / "out").exists())
            self.assertFalse((run_dir / "out").exists())


class SessionContractTests(SnapshotSessionTestBase):
    def test_runtime_window_has_zero_history_pointer_and_network_access(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            model = "poisson"
            run_dir, built, fixtures_path = self._m11_run(root, model)
            results_path = self._results_file(root)
            issue_path = self._fresh_issue_evidence(root, built)
            shutil.rmtree(root / "research")
            forbidden = {str(path).lower() for path in (
                root / "research" / "history.csv",
                root / "research" / "history_manifest.json",
                root / "research" / "epl_current.json",
                root / "research" / "m12_out",
            )}

            opened: list[str] = []
            network_calls: list[str] = []
            with self._runtime_guard() as (opened, network_calls):
                session = run_snapshot_session(
                    built["snapshot_manifest_path"],
                    fixtures_path,
                    mode="replay",
                    output_root=root / "runtime" / "sessions",
                    issue_evidence_path=issue_path,
                )
                verification = verify_snapshot_run(
                    run_dir, output_root=root / "runtime" / "audits", issue_evidence_path=issue_path
                )
                settlement = settle_forecast_run(
                    run_dir, results_path, output_root=root / "runtime" / "settle"
                )
            self.assertEqual(session["status"], "ready")
            self.assertEqual(session["audit_status"], "ready")
            self.assertEqual(verification["status"], "ready")
            self.assertEqual(settlement["status"], "complete")
            forbidden_reads = [
                entry for entry in opened
                if any(label in entry.lower() for label in FORBIDDEN_PATH_LABELS)
            ]
            self.assertEqual(forbidden_reads, [])
            self.assertEqual(network_calls, [])

    def test_cli_usage_errors_keep_exit_2(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            stderr = io_module.StringIO()
            with contextlib.redirect_stderr(stderr), contextlib.redirect_stdout(io_module.StringIO()):
                with self.assertRaises(SystemExit) as missing_args:
                    main(["snapshot-session", "--snapshot-manifest", str(root / "x.json")])
            self.assertEqual(missing_args.exception.code, 2)
            with contextlib.redirect_stderr(stderr), contextlib.redirect_stdout(io_module.StringIO()):
                with self.assertRaises(SystemExit) as bad_mode:
                    main(
                        [
                            "snapshot-session",
                            "--snapshot-manifest",
                            str(root / "x.json"),
                            "--fixtures",
                            str(root / "y.csv"),
                            "--mode",
                            "live",
                            "--output-root",
                            str(root / "out"),
                        ]
                    )
            self.assertEqual(bad_mode.exception.code, 2)
            self.assertIn("usage", stderr.getvalue().lower())

    def test_missing_inputs_produce_structured_blocked_artifacts(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run_dir, built, fixtures_path = self._m11_run(root, "frequency")
            stderr = io_module.StringIO()
            with contextlib.redirect_stderr(stderr), contextlib.redirect_stdout(io_module.StringIO()):
                with self.assertRaises(SystemExit) as missing_fixture:
                    main(
                        [
                            "snapshot-session",
                            "--snapshot-manifest",
                            str(built["snapshot_manifest_path"]),
                            "--fixtures",
                            str(root / "runtime" / "inputs" / "nope.csv"),
                            "--mode",
                            "replay",
                            "--output-root",
                            str(root / "runtime" / "sessions"),
                        ]
                    )
            self.assertEqual(missing_fixture.exception.code, 1)
            self.assertNotIn("refused or unwritable", stderr.getvalue())
            sessions = list((root / "runtime" / "sessions").iterdir())
            self.assertEqual(len(sessions), 1)
            manifest = json.loads((sessions[0] / "session_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["status"], "blocked")
            self.assertFalse(manifest["predictions_usable"])
            self.assertEqual(manifest["stages"]["fixtures_load"], "failed")
            self.assertEqual(manifest["stages"]["prediction"], "not_run")
            self.assertTrue((sessions[0] / "session_report.md").is_file())
            self.assertTrue((sessions[0] / "prediction_audit.json").is_file())
            self.assertTrue((sessions[0] / "prediction_audit.md").is_file())
            self.assertIsNone(manifest["inputs"]["fixtures"]["sha256"])

            (run_dir / "predictions.csv").unlink()
            with contextlib.redirect_stderr(io_module.StringIO()), contextlib.redirect_stdout(io_module.StringIO()):
                with self.assertRaises(SystemExit) as blocked_verify:
                    main(
                        [
                            "snapshot-run-verify",
                            "--forecast-run",
                            str(run_dir),
                            "--output-root",
                            str(root / "runtime" / "audits"),
                        ]
                    )
            self.assertEqual(blocked_verify.exception.code, 1)
            audit_dirs = list((root / "runtime" / "audits").iterdir())
            self.assertEqual(len(audit_dirs), 1)
            payload = json.loads((audit_dirs[0] / "prediction_audit.json").read_text(encoding="utf-8"))
            self.assertEqual(payload["status"], "blocked")
            self.assertFalse(payload["predictions_usable"])
            self.assertTrue((audit_dirs[0] / "prediction_audit.md").is_file())

            missing_run = root / "runtime" / "no_such_run"
            with contextlib.redirect_stderr(io_module.StringIO()), contextlib.redirect_stdout(io_module.StringIO()):
                with self.assertRaises(SystemExit) as missing_run_exit:
                    main(
                        [
                            "snapshot-run-verify",
                            "--forecast-run",
                            str(missing_run),
                            "--output-root",
                            str(root / "runtime" / "audits-missing-run"),
                        ]
                    )
            self.assertEqual(missing_run_exit.exception.code, 1)
            missing_run_dirs = list((root / "runtime" / "audits-missing-run").iterdir())
            self.assertEqual(len(missing_run_dirs), 1)
            missing_run_payload = json.loads(
                (missing_run_dirs[0] / "prediction_audit.json").read_text(encoding="utf-8")
            )
            self.assertEqual(missing_run_payload["status"], "blocked")
            self.assertIn("run_directory_does_not_exist", missing_run_payload["reasons"][0])

            missing_issue = root / "runtime" / "inputs" / "nope_issue.json"
            intact_run, _, _ = self._m11_run(root / "missing-issue-case", "frequency")
            with contextlib.redirect_stderr(io_module.StringIO()), contextlib.redirect_stdout(io_module.StringIO()):
                with self.assertRaises(SystemExit) as missing_issue_exit:
                    main(
                        [
                            "snapshot-run-verify",
                            "--forecast-run",
                            str(intact_run),
                            "--output-root",
                            str(root / "runtime" / "audits-missing-issue"),
                            "--issue-evidence",
                            str(missing_issue),
                        ]
                    )
            self.assertEqual(missing_issue_exit.exception.code, 1)
            missing_issue_dirs = list((root / "runtime" / "audits-missing-issue").iterdir())
            self.assertEqual(len(missing_issue_dirs), 1)
            missing_issue_payload = json.loads(
                (missing_issue_dirs[0] / "prediction_audit.json").read_text(encoding="utf-8")
            )
            self.assertEqual(missing_issue_payload["status"], "blocked")
            self.assertIn("issue_evidence_missing_or_unreadable", missing_issue_payload["reasons"][0])
            self.assertFalse(missing_issue_payload["predictions_usable"])
            self.assertTrue((missing_issue_dirs[0] / "prediction_audit.md").is_file())

    def test_verify_missing_issue_and_inventory_io_failure_save_blocked_artifacts(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run_dir, built, _ = self._m11_run(root, "frequency")
            issue_path = self._fresh_issue_evidence(root, built)
            real_file_sha256 = snapshot_run_verify_module.file_sha256

            def failing_issue_hash(path, *args, **kwargs):
                if Path(path) == issue_path:
                    raise PermissionError("simulated issue inventory read failure")
                return real_file_sha256(path, *args, **kwargs)

            with patch(
                "matchquant.snapshot_run_verify.file_sha256", side_effect=failing_issue_hash
            ):
                verification = verify_snapshot_run(
                    run_dir, output_root=root / "runtime" / "audits-io", issue_evidence_path=issue_path
                )
                self.assertEqual(verification["status"], "blocked")
                payload = json.loads(
                    Path(verification["prediction_audit_path"]).read_text(encoding="utf-8")
                )
                self.assertIn("input_read_failed", payload["reasons"][0])
                self.assertIn("PermissionError", payload["reasons"][0])
                with contextlib.redirect_stderr(io_module.StringIO()), contextlib.redirect_stdout(io_module.StringIO()):
                    with self.assertRaises(SystemExit) as blocked_exit:
                        main(
                            [
                                "snapshot-run-verify",
                                "--forecast-run",
                                str(run_dir),
                                "--output-root",
                                str(root / "runtime" / "audits-io-cli"),
                                "--issue-evidence",
                                str(issue_path),
                            ]
                        )
                self.assertEqual(blocked_exit.exception.code, 1)
            issue_entry = payload["inputs"]["issue_evidence"]
            self.assertIsNone(issue_entry["sha256"])
            self.assertEqual(issue_entry["hash_status"], "unavailable")
            self.assertEqual(issue_entry["path"], str(issue_path))
            self.assertFalse(payload["predictions_usable"])
            self.assertEqual(payload["stages"]["input_inventory"], "failed")
            self.assertEqual(payload["stages"]["repredict"], "not_run")
            self.assertTrue(Path(verification["prediction_audit_report_path"]).is_file())

    def test_required_report_write_failure_blocks_session(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            built = self._build_snapshot(root, "frequency")
            fixtures_path = self._write_fixtures(root)

            def failing_report(path, manifest):
                raise PermissionError("simulated required session report write failure")

            with patch(
                "matchquant.snapshot_session.write_snapshot_session_report",
                side_effect=failing_report,
            ):
                session = run_snapshot_session(
                    built["snapshot_manifest_path"],
                    fixtures_path,
                    mode="replay",
                    output_root=root / "runtime" / "sessions",
                )
                self.assertEqual(session["status"], "blocked")
                self.assertFalse(session["predictions_usable"])
                self.assertIn("required_artifact_write_failed", session["reasons"][0])
                self.assertIn("session_report", session["output_write_errors"][0])
                stderr = io_module.StringIO()
                with contextlib.redirect_stderr(stderr), contextlib.redirect_stdout(io_module.StringIO()):
                    with self.assertRaises(SystemExit) as blocked_exit:
                        main(
                            [
                                "snapshot-session",
                                "--snapshot-manifest",
                                str(built["snapshot_manifest_path"]),
                                "--fixtures",
                                str(fixtures_path),
                                "--mode",
                                "replay",
                                "--output-root",
                                str(root / "runtime" / "sessions-cli"),
                            ]
                        )
                self.assertEqual(blocked_exit.exception.code, 1)
            manifest = json.loads(
                Path(session["session_manifest_path"]).read_text(encoding="utf-8")
            )
            self.assertEqual(manifest["status"], "blocked")
            self.assertFalse(manifest["predictions_usable"])
            self.assertEqual(manifest["stages"]["finalize"], "failed")
            self.assertEqual(manifest["stages"]["prediction"], "completed")
            self.assertEqual(manifest["stages"]["audit"], "completed")
            self.assertEqual(manifest["audit"]["status"], "needs_review")
            self.assertTrue(manifest["prediction_run"])
            self.assertIsNone(manifest["output_hashes"]["session_report"])
            self.assertFalse(Path(session["session_report_path"]).exists())
            self.assertTrue(Path(session["session_manifest_path"]).is_file())
            self.assertTrue(Path(session["prediction_audit_path"]).is_file())

    def test_input_files_unchanged_across_blocked_and_successful_audits(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run_dir, built, fixtures_path = self._m11_run(root, "frequency")
            watched = [
                Path(built["snapshot_manifest_path"]),
                Path(built["model_states_path"]),
                fixtures_path,
                run_dir / "run_config.json",
                run_dir / "predictions.csv",
            ]
            before = {path: file_sha256(path) for path in watched}
            tampered_run, _, _ = self._m11_run(root / "tampered-case", "frequency")
            frame = pd.read_csv(tampered_run / "predictions.csv")
            frame.loc[0, "draw_prob"] = frame.loc[0, "draw_prob"] + 0.05
            frame.loc[0, "home_win_prob"] = frame.loc[0, "home_win_prob"] - 0.05
            frame.to_csv(tampered_run / "predictions.csv", index=False)
            verification = verify_snapshot_run(
                tampered_run, output_root=root / "tampered-case" / "audits"
            )
            self.assertEqual(verification["status"], "blocked")
            session = run_snapshot_session(
                built["snapshot_manifest_path"],
                fixtures_path,
                mode="replay",
                output_root=root / "runtime" / "sessions",
            )
            self.assertEqual(session["status"], "ready")
            for path, digest in before.items():
                self.assertEqual(file_sha256(path), digest, str(path))


if __name__ == "__main__":
    unittest.main()
