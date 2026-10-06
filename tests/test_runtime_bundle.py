"""B04 tests: portable runtime bundles, lineage bridge, migration, and tamper gates.

The research-side synthetic builder (history, pointer, M12 evidence, M4
reference) and the B04 runtime window (bundle verify / session / audit / settle)
are kept strictly separate: access guards only observe the runtime window, and
the research root is disconnected before the migrated-bundle closure runs.
"""

import builtins
import contextlib
import io as io_module
import json
import os
import shutil
import stat
import subprocess
import sys
import tempfile
import unittest
import urllib.request
from pathlib import Path
from unittest.mock import patch

import pandas as pd
from pandas.testing import assert_frame_equal

import matchquant.runtime_bundle as runtime_bundle_module
import matchquant.runtime_bundle_validation as runtime_bundle_validation_module
from matchquant.cli import main
from matchquant.engine_snapshot import build_engine_snapshot
from matchquant.engine_snapshot_status import check_engine_snapshot_status
from matchquant.forecast import run_forecast
from matchquant.io import file_sha256, write_json
from matchquant.runtime_bundle import build_runtime_bundle
from matchquant.runtime_bundle_validation import verify_runtime_bundle
from matchquant.schema import DataValidationError, REQUIRED_COLUMNS
from matchquant.settlement import settle_forecast_run
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
DATASET_VERSION = "m2.1-b04-test"
SOURCE_PATH_LABELS = ("research", "history.csv", "history_manifest.json", "epl_current.json", "m12_out")


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


class RuntimeBundleTestBase(unittest.TestCase):
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
            output_root=root / "research" / "snapshots",
        )

    def _write_fixtures(self, root: Path, frame: pd.DataFrame | None = None) -> Path:
        fixtures_path = root / "runtime" / "inputs" / "fixtures.csv"
        fixtures_path.parent.mkdir(parents=True, exist_ok=True)
        (frame if frame is not None else _fixtures_frame()).to_csv(fixtures_path, index=False)
        return fixtures_path

    def _results_file(self, root: Path) -> Path:
        path = root / "runtime" / "inputs" / "results.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        _history_frame().loc[:, REQUIRED_COLUMNS].to_csv(path, index=False)
        return path

    def _fresh_issue_evidence(self, root: Path, built: dict) -> Path:
        """Research side: a real M12 fresh record for the snapshot via its pointer."""

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

    def _stale_issue_evidence(self, root: Path, built: dict) -> Path:
        other_history = _history_frame()
        other_history.loc[other_history["match_id"].eq("h1"), "home_goals"] = 9
        other_history_path, other_manifest_path = self._write_history_evidence(
            root / "research" / "other", other_history
        )
        stale_pointer = self._write_pointer(root / "research", other_history_path, other_manifest_path)
        stale = check_engine_snapshot_status(
            built["snapshot_manifest_path"], stale_pointer, output_root=root / "research" / "m12_out"
        )
        self.assertEqual(stale["status"], "stale")
        stale_path = root / "runtime" / "inputs" / "issue_stale.json"
        stale_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(stale["status_manifest_path"], stale_path)
        return stale_path

    def _build_bundle(self, root: Path, model: str) -> dict:
        built = self._build_snapshot(root, model)
        fixtures_path = self._write_fixtures(root)
        del fixtures_path
        issue_path = self._fresh_issue_evidence(root, built)
        return build_runtime_bundle(
            built["snapshot_manifest_path"], issue_path, output_root=root / "jobs"
        )

    def _bundle_files_digest(self, bundle_root: Path) -> dict[str, str]:
        return {
            entry.relative_to(bundle_root).as_posix(): file_sha256(entry)
            for entry in sorted(bundle_root.rglob("*"))
            if entry.is_file()
        }

    @contextlib.contextmanager
    def _runtime_guard(self):
        """Observe the B04 runtime window: builtins.open, io.open, urllib, socket."""

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

    def _rewrite_bundle_manifest(self, bundle_root: Path, mutate) -> None:
        manifest_path = bundle_root / "bundle_manifest.json"
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        mutate(payload)
        write_json(payload, manifest_path)

    def _sync_bundle_records(self, bundle_root: Path, changed: dict[str, Path]) -> None:
        """Sync the bundle manifest records after attacker changes to payload files.

        ``changed`` maps bundle-relative paths to their modified files. The recorded
        file hashes, byte sizes, and payload-byte totals are all updated so the
        forgery is internally consistent up to the gates under test.
        """

        def mutate(payload):
            total = 0
            for entry in payload["files"]:
                if entry["path"] in changed:
                    entry["sha256"] = file_sha256(changed[entry["path"]])
                    entry["bytes"] = changed[entry["path"]].stat().st_size
                total += entry["bytes"]
            payload["size"]["payload_bytes"] = total

        self._rewrite_bundle_manifest(bundle_root, mutate)


class BundleClosureTests(RuntimeBundleTestBase):
    def test_build_verify_and_migrated_bundle_closure_for_every_model(self):
        for model in ("frequency", "elo", "poisson"):
            with self.subTest(model=model):
                with tempfile.TemporaryDirectory() as temporary:
                    root = Path(temporary)
                    built = self._build_snapshot(root, model)
                    fixtures_path = self._write_fixtures(root)
                    results_path = self._results_file(root)
                    issue_path = self._fresh_issue_evidence(root, built)

                    build = build_runtime_bundle(
                        built["snapshot_manifest_path"], issue_path, output_root=root / "jobs"
                    )
                    self.assertEqual(build["status"], "ready", build["reasons"])
                    self.assertEqual(build["stages"]["self_verify"], "completed")
                    bundle_root = Path(build["bundle_dir"])
                    self.assertEqual(build["limit_bytes"], 5 * 1024 * 1024)
                    self.assertLessEqual(build["total_bytes"], build["limit_bytes"])
                    self.assertTrue(
                        all(not name.endswith((".exe", ".bat", ".py")) for name in
                            (entry["path"] for entry in build["manifest"]["files"]))
                    )
                    manifest_payload = build["manifest"]
                    self.assertEqual(manifest_payload["schema_version"], "baseline-runtime-bundle-v1")
                    self.assertNotIn("bundle_manifest_sha256", manifest_payload)
                    observed_fields = {entry["field"] for entry in manifest_payload["manifest_field_diffs"]["observed"]}
                    self.assertEqual(
                        observed_fields,
                        {"manifest_sha256", "state.path", "outputs.snapshot_dir", "outputs.snapshot_manifest",
                         "outputs.model_states", "outputs.snapshot_report"},
                    )
                    state_path = Path(built["model_states_path"])
                    self.assertEqual(
                        file_sha256(bundle_root / "snapshot" / build["snapshot_id"] / "model_states.json"),
                        file_sha256(state_path),
                    )

                    verification = verify_runtime_bundle(build["bundle_manifest_path"])
                    self.assertEqual(verification["status"], "ready", verification["reasons"])
                    self.assertTrue(verification["issue_validation"]["fresh_at_issue"])
                    self.assertEqual(
                        verification["issue_validation"]["validation_kind"],
                        "bundle_original_issue_via_derived_manifest_lineage",
                    )
                    before_digest = self._bundle_files_digest(bundle_root)

                    # M4 reference is produced on the research side, before the disconnection.
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

                    migrated_root = root / "migrated destination with spaces"
                    migrated_root.mkdir()
                    migrated_bundle = migrated_root / f"{model} runtime bundle"
                    shutil.copytree(bundle_root, migrated_bundle)
                    migrated_manifest = migrated_bundle / "bundle_manifest.json"
                    self.assertNotEqual(str(migrated_bundle), str(bundle_root))

                    # Disconnect every research-side source: snapshot, state, history,
                    # processed manifest, pointer, M12 output, original issue file.
                    shutil.rmtree(root / "research")
                    issue_path.unlink()
                    self.assertFalse((root / "research").exists())

                    reopened = verify_runtime_bundle(migrated_manifest)
                    self.assertEqual(reopened["status"], "ready", reopened["reasons"])

                    with self._runtime_guard() as (opened, network_calls):
                        session = run_snapshot_session(
                            None,
                            fixtures_path,
                            mode="replay",
                            output_root=root / "runtime" / "sessions",
                            bundle_manifest_path=migrated_manifest,
                        )
                        standalone = verify_snapshot_run(
                            Path(session["prediction_run_dir"]),
                            output_root=root / "runtime" / "audits",
                            bundle_manifest_path=migrated_manifest,
                        )
                        settlement = settle_forecast_run(
                            Path(session["prediction_run_dir"]),
                            results_path,
                            output_root=root / "runtime" / "settle",
                        )
                    self.assertEqual(session["status"], "ready", session["reasons"])
                    self.assertEqual(session["audit_status"], "ready")
                    self.assertTrue(session["bundle"]["fresh_at_issue"])
                    self.assertEqual(
                        session["bundle"]["active_link_status"], "not_checked_in_runtime"
                    )
                    self.assertEqual(standalone["status"], "ready", standalone["reasons"])
                    self.assertEqual(settlement["status"], "complete")
                    self.assertEqual(settlement["counts"]["settled"], 2)

                    audit_payload = json.loads(
                        Path(session["prediction_audit_path"]).read_text(encoding="utf-8")
                    )
                    self.assertEqual(
                        audit_payload["bundle"]["validation_kind"],
                        "bundle_original_issue_via_derived_manifest_lineage",
                    )
                    self.assertEqual(
                        audit_payload["issue_evidence"]["record_status"], "fresh"
                    )
                    self.assertTrue(audit_payload["issue_evidence"]["fresh_at_issue"])
                    self.assertEqual(audit_payload["inputs_unchanged"], True)

                    m4_predictions = pd.read_csv(
                        Path(m4_result["run_dir"]) / "predictions.csv"
                    )
                    bundle_predictions = pd.read_csv(
                        Path(session["prediction_run_dir"]) / "predictions.csv"
                    )
                    assert_frame_equal(
                        bundle_predictions.sort_values("fixture_id").reset_index(drop=True),
                        m4_predictions.sort_values("fixture_id").reset_index(drop=True),
                        check_exact=False,
                        atol=1e-12,
                        rtol=0,
                    )
                    for key in ("log_loss", "brier_score", "accuracy"):
                        self.assertIsNotNone(settlement["metrics"][key])
                    self.assertEqual(settlement["metrics"]["coverage"], 1.0)

                    source_reads = [
                        entry for entry in opened
                        if any(label in entry.lower() for label in SOURCE_PATH_LABELS)
                    ]
                    self.assertEqual(source_reads, [])
                    self.assertEqual(network_calls, [])

                    self.assertEqual(self._bundle_files_digest(migrated_bundle), before_digest)

    def test_bundle_research_mode_stays_needs_review_with_cli_exit_2(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            build = self._build_bundle(root, "frequency")
            fixtures_path = self._write_fixtures(root)
            session = run_snapshot_session(
                None,
                fixtures_path,
                mode="research",
                output_root=root / "runtime" / "sessions",
                bundle_manifest_path=build["bundle_manifest_path"],
            )
            self.assertEqual(session["status"], "needs_review")
            self.assertIn("coverage_to_fixture_time_not_proven", session["reasons"])
            self.assertEqual(session["audit_status"], "ready")
            self.assertTrue(session["predictions_usable"])
            stderr = io_module.StringIO()
            with contextlib.redirect_stderr(stderr), contextlib.redirect_stdout(io_module.StringIO()):
                with self.assertRaises(SystemExit) as research_exit:
                    main(
                        [
                            "snapshot-session",
                            "--bundle-manifest",
                            str(build["bundle_manifest_path"]),
                            "--fixtures",
                            str(fixtures_path),
                            "--mode",
                            "research",
                            "--output-root",
                            str(root / "runtime" / "sessions-cli"),
                        ]
                    )
            self.assertEqual(research_exit.exception.code, 2)

    def test_unseen_team_prior_annotation_does_not_change_probabilities(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            model = "elo"
            built = self._build_snapshot(root, model)
            fixtures_path = self._write_fixtures(root)
            issue_path = self._fresh_issue_evidence(root, built)
            build = build_runtime_bundle(
                built["snapshot_manifest_path"], issue_path, output_root=root / "jobs"
            )
            self.assertEqual(build["status"], "ready")
            unknown_fixtures = _fixtures_frame()
            unknown_fixtures.loc[0, "home_team"] = "Everton United FC"
            unknown_fixtures_path = self._write_fixtures(root / "unknown-case", unknown_fixtures)
            session = run_snapshot_session(
                None,
                unknown_fixtures_path,
                mode="replay",
                output_root=root / "runtime" / "sessions",
                bundle_manifest_path=build["bundle_manifest_path"],
            )
            self.assertEqual(session["status"], "ready")
            annotations = session["team_annotations"]
            self.assertEqual(annotations["unseen_home_teams"], ["Everton United FC"])
            self.assertEqual(annotations["unseen_away_teams"], [])
            self.assertEqual(
                annotations["unseen_team_prior"]["initial_rating"],
                MODEL_CONFIGS[model].get("initial_rating", 1500.0),
            )
            m4_config_path = root / "research" / "elo.json"
            write_json(MODEL_CONFIGS[model], m4_config_path)
            m4_result = run_forecast(
                root / "research" / "history.csv",
                unknown_fixtures_path,
                "2024-01-07",
                model_name=model,
                model_config_path=m4_config_path,
                output_root=root / "runtime" / "m4",
            )
            bundle_predictions = pd.read_csv(
                Path(session["prediction_run_dir"]) / "predictions.csv"
            )
            m4_predictions = pd.read_csv(
                Path(m4_result["run_dir"]) / "predictions.csv"
            )
            assert_frame_equal(
                bundle_predictions.sort_values("fixture_id").reset_index(drop=True),
                m4_predictions.sort_values("fixture_id").reset_index(drop=True),
                check_exact=False,
                atol=1e-12,
                rtol=0,
            )

    def test_frequency_bundle_declares_global_frequency_not_unseen_teams(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            build = self._build_bundle(root, "frequency")
            fixtures_path = self._write_fixtures(root)
            session = run_snapshot_session(
                None,
                fixtures_path,
                mode="replay",
                output_root=root / "runtime" / "sessions",
                bundle_manifest_path=build["bundle_manifest_path"],
            )
            self.assertEqual(session["status"], "ready")
            annotations = session["team_annotations"]
            self.assertEqual(annotations["frequency_basis"], "global")
            self.assertEqual(annotations["team_membership"], "not_applicable")
            self.assertIsNone(annotations["unseen_home_teams"])
            self.assertIsNone(annotations["unseen_away_teams"])
            index_payload = json.loads(
                Path(build["bundle_dir"], "team_index.json").read_text(encoding="utf-8")
            )
            self.assertEqual(index_payload["teams"], [])
            self.assertEqual(index_payload["frequency_basis"], "global")

    def test_bare_manifest_paths_resolve_from_bundle_root_not_cwd(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            build = self._build_bundle(root, "frequency")
            fixtures_path = self._write_fixtures(root)
            bundle_root = Path(build["bundle_dir"])
            # The bundle manifest only records bundle-relative payload paths.
            manifest_payload = json.loads(
                (bundle_root / "bundle_manifest.json").read_text(encoding="utf-8")
            )
            for entry in manifest_payload["files"]:
                self.assertFalse(Path(entry["path"]).is_absolute())
            # A different working directory must not change verification.
            previous_cwd = os.getcwd()
            os.chdir(tempfile.gettempdir())
            try:
                verification = verify_runtime_bundle(bundle_root / "bundle_manifest.json")
            finally:
                os.chdir(previous_cwd)
            self.assertEqual(verification["status"], "ready", verification["reasons"])
            session = run_snapshot_session(
                None,
                fixtures_path,
                mode="replay",
                output_root=root / "runtime" / "sessions",
                bundle_manifest_path=bundle_root / "bundle_manifest.json",
            )
            self.assertEqual(session["status"], "ready")


class BundleLineageTests(RuntimeBundleTestBase):
    def test_original_issue_cannot_bind_a_derived_bare_manifest(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            build = self._build_bundle(root, "frequency")
            fixtures_path = self._write_fixtures(root)
            session = run_snapshot_session(
                None,
                fixtures_path,
                mode="replay",
                output_root=root / "runtime" / "sessions",
                bundle_manifest_path=build["bundle_manifest_path"],
            )
            self.assertEqual(session["status"], "ready")
            run_dir = Path(session["prediction_run_dir"])
            original_issue_copy = (
                Path(build["bundle_dir"]) / "evidence" / "issue_audit.json"
            )
            external_copy = root / "runtime" / "inputs" / "original_issue_external.json"
            shutil.copyfile(original_issue_copy, external_copy)

            # B03 bare semantics stay strict: the original issue binds the ORIGINAL
            # manifest hash, so a run produced from the derived manifest rejects it.
            bare_audit = verify_snapshot_run(
                run_dir, output_root=root / "runtime" / "audits-bare", issue_evidence_path=external_copy
            )
            self.assertEqual(bare_audit["status"], "blocked")
            self.assertIn("issue_evidence_invalid", bare_audit["reasons"][0])
            self.assertFalse(bare_audit["predictions_usable"])

            # Rewriting the bundled issue copy to point at the derived hash is also blocked.
            bundle_root = Path(build["bundle_dir"])
            derived_manifest = json.loads(
                (bundle_root / "snapshot" / build["snapshot_id"] / "snapshot_manifest.json").read_text(
                    encoding="utf-8"
                )
            )
            derived_file_hash = file_sha256(
                bundle_root / "snapshot" / build["snapshot_id"] / "snapshot_manifest.json"
            )
            issue_payload = json.loads(original_issue_copy.read_text(encoding="utf-8"))
            issue_payload["snapshot"]["sha256"] = derived_file_hash
            issue_payload["snapshot"]["path"] = str(
                bundle_root / "snapshot" / build["snapshot_id"] / "snapshot_manifest.json"
            )
            write_json(issue_payload, original_issue_copy)

            def mutate(payload):
                new_issue_hash = file_sha256(original_issue_copy)
                payload["issue_evidence"]["file_sha256"] = new_issue_hash
                payload["original_snapshot"]["issue_source_file_sha256"] = new_issue_hash

            self._sync_bundle_records(bundle_root, {"evidence/issue_audit.json": original_issue_copy})
            self._rewrite_bundle_manifest(bundle_root, mutate)
            verification = verify_runtime_bundle(bundle_root / "bundle_manifest.json")
            self.assertEqual(verification["status"], "blocked", verification["reasons"])
            self.assertEqual(verification["checks"][-1]["check"], "issue_lineage")
            self.assertIn(
                "does not match the audited snapshot manifest file", verification["reasons"][0]
            )

    def _resync_derived_manifest(self, bundle_root: Path, snapshot_id: str, mutate_fields) -> Path:
        """Rewrite the derived manifest, recompute its writer hash, sync bundle records."""

        from matchquant.engine_snapshot_manifest import write_snapshot_manifest

        derived_manifest_path = bundle_root / "snapshot" / snapshot_id / "snapshot_manifest.json"
        payload = json.loads(derived_manifest_path.read_text(encoding="utf-8"))
        mutate_fields(payload)
        write_snapshot_manifest(payload, derived_manifest_path)
        self._sync_bundle_records(bundle_root, {f"snapshot/{snapshot_id}/snapshot_manifest.json": derived_manifest_path})

        def sync(bundle_payload):
            bundle_payload["derived_snapshot"]["manifest_file_sha256"] = file_sha256(derived_manifest_path)
            bundle_payload["derived_snapshot"]["manifest_content_sha256"] = json.loads(
                derived_manifest_path.read_text(encoding="utf-8")
            )["manifest_sha256"]

        self._rewrite_bundle_manifest(bundle_root, sync)
        return derived_manifest_path

    def test_forbidden_manifest_field_change_is_blocked_even_when_synced(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            build = self._build_bundle(root, "frequency")
            bundle_root = Path(build["bundle_dir"])

            # as_of is not on the allowed list: a fully synced rewrite is still refused.
            self._resync_derived_manifest(
                bundle_root,
                build["snapshot_id"],
                lambda payload: payload["as_of"].update(
                    {"value": "2024-01-08", "timestamp": "2024-01-08T00:00:00"}
                ),
            )
            verification = verify_runtime_bundle(bundle_root / "bundle_manifest.json")
            self.assertEqual(verification["status"], "blocked", verification["reasons"])
            self.assertEqual(verification["checks"][-1]["check"], "manifest_allowed_diff")

            # A fixed leaf set to a wrong value is refused even with everything synced.
            build2 = self._build_bundle(root / "case2", "frequency")
            bundle_root2 = Path(build2["bundle_dir"])
            self._resync_derived_manifest(
                bundle_root2,
                build2["snapshot_id"],
                lambda payload: payload["outputs"].update({"snapshot_dir": "../escape"}),
            )
            verification2 = verify_runtime_bundle(bundle_root2 / "bundle_manifest.json")
            self.assertEqual(verification2["status"], "blocked", verification2["reasons"])
            self.assertEqual(verification2["checks"][-1]["check"], "manifest_allowed_diff")

    def test_synced_state_hash_rewrite_is_blocked_by_the_original_copy(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            build = self._build_bundle(root, "frequency")
            bundle_root = Path(build["bundle_dir"])
            state_path = bundle_root / "snapshot" / build["snapshot_id"] / "model_states.json"

            # counts are not cross-checked against the model config, so the synced
            # forgery passes the strict loader; the ORIGINAL manifest copy still
            # records the original state hash and refuses the rewritten chain.
            from matchquant.engine_snapshot_manifest import write_snapshot_manifest
            from matchquant.engine_snapshot_support import canonical_json_bytes, configuration_hash
            import hashlib

            state_payload = json.loads(state_path.read_text(encoding="utf-8"))
            record = state_payload["models"][0]
            record["state"]["counts"]["D"] = 5.0
            record["config_sha256"] = configuration_hash(record["config"])
            record["state_sha256"] = hashlib.sha256(canonical_json_bytes(record["state"])).hexdigest()
            state_path.write_bytes(canonical_json_bytes(state_payload))
            new_state_hash = file_sha256(state_path)

            derived_manifest_path = bundle_root / "snapshot" / build["snapshot_id"] / "snapshot_manifest.json"
            derived_payload = json.loads(derived_manifest_path.read_text(encoding="utf-8"))
            derived_payload["state"]["sha256"] = new_state_hash
            derived_payload["models"][0]["state_sha256"] = record["state_sha256"]
            write_snapshot_manifest(derived_payload, derived_manifest_path)
            self._sync_bundle_records(
                bundle_root,
                {
                    f"snapshot/{build['snapshot_id']}/model_states.json": state_path,
                    f"snapshot/{build['snapshot_id']}/snapshot_manifest.json": derived_manifest_path,
                },
            )

            def sync(bundle_payload):
                bundle_payload["state_file_sha256"] = new_state_hash
                bundle_payload["derived_snapshot"]["state_file_sha256"] = new_state_hash
                bundle_payload["derived_snapshot"]["manifest_file_sha256"] = file_sha256(derived_manifest_path)
                bundle_payload["derived_snapshot"]["manifest_content_sha256"] = json.loads(
                    derived_manifest_path.read_text(encoding="utf-8")
                )["manifest_sha256"]
                bundle_payload["model"]["state_payload_sha256"] = record["state_sha256"]

            self._rewrite_bundle_manifest(bundle_root, sync)
            verification = verify_runtime_bundle(bundle_root / "bundle_manifest.json")
            self.assertEqual(verification["status"], "blocked", verification["reasons"])
            self.assertEqual(verification["checks"][-1]["check"], "bundle_manifest_structure")
            self.assertIn("state hashes disagree", verification["reasons"][0])

    def test_team_index_tamper_with_synced_hash_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            build = self._build_bundle(root, "elo")
            bundle_root = Path(build["bundle_dir"])
            team_index_path = bundle_root / "team_index.json"
            payload = json.loads(team_index_path.read_text(encoding="utf-8"))
            first_team = payload["teams"][0]
            payload["teams"] = [f"{first_team} City" if item == first_team else item for item in payload["teams"]]
            write_json(payload, team_index_path)

            def sync(bundle_payload):
                bundle_payload["team_index"]["file_sha256"] = file_sha256(team_index_path)

            self._sync_bundle_records(bundle_root, {"team_index.json": team_index_path})
            self._rewrite_bundle_manifest(bundle_root, sync)
            verification = verify_runtime_bundle(bundle_root / "bundle_manifest.json")
            self.assertEqual(verification["status"], "blocked", verification["reasons"])
            self.assertEqual(verification["checks"][-1]["check"], "team_index")

    def test_state_manifest_copy_and_multi_model_tamper_is_blocked(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            build = self._build_bundle(root, "frequency")
            bundle_root = Path(build["bundle_dir"])
            state_path = bundle_root / "snapshot" / build["snapshot_id"] / "model_states.json"
            state_path.write_text(
                state_path.read_text(encoding="utf-8").replace("frequency", "frequency2", 1),
                encoding="utf-8",
            )
            verification = verify_runtime_bundle(bundle_root / "bundle_manifest.json")
            self.assertEqual(verification["status"], "blocked", verification["reasons"])

            build2 = self._build_bundle(root / "copy-case", "frequency")
            bundle_root2 = Path(build2["bundle_dir"])
            copy_path = bundle_root2 / "evidence" / "original_snapshot_manifest.json"
            copy_payload = json.loads(copy_path.read_text(encoding="utf-8"))
            copy_payload["generated_at"] = "2020-01-01T00:00:00Z"
            write_json(copy_payload, copy_path)
            verification2 = verify_runtime_bundle(bundle_root2 / "bundle_manifest.json")
            self.assertEqual(verification2["status"], "blocked", verification2["reasons"])

            # A two-model snapshot cannot be bundled at all.
            history_path, manifest_path = self._write_history_evidence(root / "multi" / "research")
            built = build_engine_snapshot(
                history_path,
                manifest_path,
                "2024-01-07",
                model_names=["frequency", "elo"],
                model_configs={
                    "frequency": MODEL_CONFIGS["frequency"],
                    "elo": MODEL_CONFIGS["elo"],
                },
                output_root=root / "multi" / "research" / "snapshots",
            )
            fixtures_path = self._write_fixtures(root / "multi")
            issue_path = self._fresh_issue_evidence(root / "multi", built)
            multi_build = build_runtime_bundle(
                built["snapshot_manifest_path"], issue_path, output_root=root / "multi" / "jobs"
            )
            self.assertEqual(multi_build["status"], "blocked")
            self.assertIn("single-model", multi_build["reasons"][0])

    def test_bundle_manifest_schema_and_inventory_tamper_is_blocked(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            build = self._build_bundle(root, "frequency")
            bundle_root = Path(build["bundle_dir"])

            # Unknown schema.
            self._rewrite_bundle_manifest(
                bundle_root, lambda payload: payload.update({"schema_version": "made-up-bundle-v9"})
            )
            self.assertEqual(
                verify_runtime_bundle(bundle_root / "bundle_manifest.json")["status"], "blocked"
            )

            # Unknown extra field on the bundle manifest.
            build2 = self._build_bundle(root / "schema-case", "frequency")
            bundle_root2 = Path(build2["bundle_dir"])
            self._rewrite_bundle_manifest(bundle_root2, lambda payload: payload.update({"extra": 1}))
            self.assertEqual(
                verify_runtime_bundle(bundle_root2 / "bundle_manifest.json")["status"], "blocked"
            )

            # Missing payload file.
            build3 = self._build_bundle(root / "missing-case", "frequency")
            bundle_root3 = Path(build3["bundle_dir"])
            (bundle_root3 / "RUN.md").unlink()
            self.assertEqual(
                verify_runtime_bundle(bundle_root3 / "bundle_manifest.json")["status"], "blocked"
            )

            # Extra unlisted file, including an executable object.
            build4 = self._build_bundle(root / "extra-case", "frequency")
            bundle_root4 = Path(build4["bundle_dir"])
            (bundle_root4 / "extra_notes.txt").write_text("hello", encoding="utf-8")
            self.assertEqual(
                verify_runtime_bundle(bundle_root4 / "bundle_manifest.json")["status"], "blocked"
            )
            (bundle_root4 / "evil.exe").write_bytes(b"MZ")
            verification4 = verify_runtime_bundle(bundle_root4 / "bundle_manifest.json")
            self.assertEqual(verification4["status"], "blocked")
            self.assertTrue(
                any("evil.exe" in reason for reason in verification4["reasons"]), verification4["reasons"]
            )

            # Absolute path, parent-escaping path, and ADS path entries.
            for label, mutate in (
                ("absolute", lambda entry: entry.update({"path": str(bundle_root5 / "RUN.md")})),
                ("escape", lambda entry: entry.update({"path": "../../RUN.md"})),
                ("ads", lambda entry: entry.update({"path": "RUN.md:hidden"})),
            ):
                build5 = self._build_bundle(root / f"path-case-{label}", "frequency")
                bundle_root5 = Path(build5["bundle_dir"])

                def sync(bundle_payload, mutate=mutate):
                    for entry in bundle_payload["files"]:
                        if entry["path"] == "RUN.md":
                            mutate(entry)

                self._rewrite_bundle_manifest(bundle_root5, sync)
                verification5 = verify_runtime_bundle(bundle_root5 / "bundle_manifest.json")
                self.assertEqual(verification5["status"], "blocked", label)

            # A junction replacing a payload file is refused.
            build6 = self._build_bundle(root / "junction-case", "frequency")
            bundle_root6 = Path(build6["bundle_dir"])
            run_md = bundle_root6 / "RUN.md"
            run_md.unlink()
            exit_code = subprocess.run(
                ["cmd", "/c", "mklink", "/J", str(run_md), str(bundle_root6 / "evidence")],
                capture_output=True,
            ).returncode
            self.assertEqual(exit_code, 0)
            self.assertTrue(bool(os.lstat(run_md).st_file_attributes & stat.FILE_ATTRIBUTE_REPARSE_POINT))
            verification6 = verify_runtime_bundle(bundle_root6 / "bundle_manifest.json")
            self.assertEqual(verification6["status"], "blocked", verification6["reasons"])
            subprocess.run(["cmd", "/c", "rmdir", str(run_md)], capture_output=True)

    def test_build_requires_fresh_issue_bound_to_the_snapshot(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            built = self._build_snapshot(root, "frequency")
            stale_path = self._stale_issue_evidence(root, built)
            stale_build = build_runtime_bundle(
                built["snapshot_manifest_path"], stale_path, output_root=root / "jobs-stale"
            )
            self.assertEqual(stale_build["status"], "blocked")
            self.assertIn("fresh_at_issue", stale_build["reasons"][0])
            self.assertTrue(Path(stale_build["build_job_path"]).is_file())
            job = json.loads(Path(stale_build["build_job_path"]).read_text(encoding="utf-8"))
            self.assertEqual(job["status"], "blocked")
            self.assertTrue(Path(stale_build["build_report_path"]).is_file())

            # An issue bound to a different snapshot is refused.
            other_built = self._build_snapshot(root / "other-case", "frequency")
            other_issue = self._fresh_issue_evidence(root / "other-case", other_built)
            mismatched_build = build_runtime_bundle(
                built["snapshot_manifest_path"], other_issue, output_root=root / "jobs-mismatch"
            )
            self.assertEqual(mismatched_build["status"], "blocked")
            self.assertIn("issue_evidence_failed", mismatched_build["reasons"][0])


class BundleIsolationTests(RuntimeBundleTestBase):
    def test_output_roots_inside_bundle_inputs_or_source_are_refused(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            build = self._build_bundle(root, "frequency")
            bundle_root = Path(build["bundle_dir"])
            fixtures_path = self._write_fixtures(root)
            built_manifest = root / "research" / "snapshots"
            snapshot_manifest = next(built_manifest.glob("*/snapshot_manifest.json"))
            issue_source = root / "runtime" / "inputs" / "issue_fresh.json"

            session_cases = [
                bundle_root / "out",
                bundle_root,
            ]
            for output_root in session_cases:
                with self.subTest(output_root=str(output_root)):
                    with self.assertRaises((DataValidationError, OSError)):
                        run_snapshot_session(
                            None,
                            fixtures_path,
                            mode="replay",
                            output_root=output_root,
                            bundle_manifest_path=bundle_root / "bundle_manifest.json",
                        )
            self.assertFalse((bundle_root / "out").exists())

            build_cases = [
                snapshot_manifest.parent / "out",
                snapshot_manifest.parent,
                issue_source.parent / "bundle-jobs",
                Path(r"D:\Codex\matchquant_data") / "b04-must-not-write",
                Path(r"D:\Codex\matchquant_runs") / "b04-must-not-write",
                Path(__file__).resolve().parent.parent / "b04-must-not-write",
            ]
            for output_root in build_cases:
                with self.subTest(output_root=str(output_root)):
                    with self.assertRaises((DataValidationError, OSError)):
                        build_runtime_bundle(
                            snapshot_manifest, issue_source, output_root=output_root
                        )
            self.assertFalse((snapshot_manifest.parent / "out").exists())
            self.assertFalse((Path(__file__).resolve().parent.parent / "b04-must-not-write").exists())

            # Missing snapshot input: the declared file type still protects its archive dir.
            missing_snapshot = root / "research" / "snapshots" / "ghost" / "snapshot_manifest.json"
            with self.assertRaises((DataValidationError, OSError)):
                build_runtime_bundle(
                    missing_snapshot,
                    issue_source,
                    output_root=missing_snapshot.parent / "jobs",
                )
            self.assertFalse(missing_snapshot.parent.exists())

    def test_cli_exit_codes_and_mutual_exclusion(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            build = self._build_bundle(root, "frequency")
            fixtures_path = self._write_fixtures(root)
            built_manifest = root / "research" / "snapshots"
            snapshot_manifest = next(built_manifest.glob("*/snapshot_manifest.json"))
            issue_source = root / "runtime" / "inputs" / "issue_fresh.json"

            def run_cli(*argv):
                stderr = io_module.StringIO()
                stdout = io_module.StringIO()
                with contextlib.redirect_stderr(stderr), contextlib.redirect_stdout(stdout):
                    with self.assertRaises(SystemExit) as exit_context:
                        main(list(argv))
                return exit_context.exception.code, stdout.getvalue(), stderr.getvalue()

            # runtime-bundle-verify ready -> 0.
            code, out, _ = run_cli(
                "runtime-bundle-verify", "--bundle-manifest", str(build["bundle_manifest_path"])
            )
            self.assertEqual(code, 0, out)
            self.assertIn("ready", out)

            # Bare snapshot + bundle are mutually exclusive at parse time -> 2.
            code, _, err = run_cli(
                "snapshot-session",
                "--snapshot-manifest", str(snapshot_manifest),
                "--bundle-manifest", str(build["bundle_manifest_path"]),
                "--fixtures", str(fixtures_path),
                "--mode", "replay",
                "--output-root", str(root / "sessions-x"),
            )
            self.assertEqual(code, 2)
            self.assertIn("usage", err.lower())

            # Bundle + issue override are mutually exclusive at parse time -> 2.
            code, _, err = run_cli(
                "snapshot-session",
                "--bundle-manifest", str(build["bundle_manifest_path"]),
                "--issue-evidence", str(issue_source),
                "--fixtures", str(fixtures_path),
                "--mode", "replay",
                "--output-root", str(root / "sessions-x"),
            )
            self.assertEqual(code, 2)
            self.assertIn("usage", err.lower())

            code, _, err = run_cli(
                "snapshot-run-verify",
                "--forecast-run", str(root / "any-run"),
                "--output-root", str(root / "audits-x"),
                "--issue-evidence", str(issue_source),
                "--bundle-manifest", str(build["bundle_manifest_path"]),
            )
            self.assertEqual(code, 2)
            self.assertIn("usage", err.lower())

            # A missing bundle manifest is a blocked CLI run -> 1.
            code, _, _ = run_cli(
                "runtime-bundle-verify",
                "--bundle-manifest", str(root / "nope" / "bundle_manifest.json"),
            )
            self.assertEqual(code, 1)

            # A stale issue cannot produce a ready bundle via CLI -> 1.
            built2 = self._build_snapshot(root / "stale-case", "frequency")
            stale_path = self._stale_issue_evidence(root / "stale-case", built2)
            code, _, _ = run_cli(
                "runtime-bundle-build",
                "--snapshot-manifest", str(built2["snapshot_manifest_path"]),
                "--issue-audit", str(stale_path),
                "--output-root", str(root / "stale-case" / "jobs"),
            )
            self.assertEqual(code, 1)
            job_dirs = list((root / "stale-case" / "jobs").iterdir())
            self.assertEqual(len(job_dirs), 1)
            self.assertFalse((job_dirs[0] / "bundle" / "bundle_manifest.json").exists())

            # A ready build via CLI -> 0, and bundle-manifest last written.
            built3 = self._build_snapshot(root / "ready-case", "frequency")
            issue3 = self._fresh_issue_evidence(root / "ready-case", built3)
            code, out, _ = run_cli(
                "runtime-bundle-build",
                "--snapshot-manifest", str(built3["snapshot_manifest_path"]),
                "--issue-audit", str(issue3),
                "--output-root", str(root / "ready-case" / "jobs"),
            )
            self.assertEqual(code, 0, out)
            self.assertIn("ready", out)

            # Bundle session replay via CLI -> 0.
            code, out, _ = run_cli(
                "snapshot-session",
                "--bundle-manifest", str(build["bundle_manifest_path"]),
                "--fixtures", str(fixtures_path),
                "--mode", "replay",
                "--output-root", str(root / "sessions-cli"),
            )
            self.assertEqual(code, 0, out)

            # Bundle audit via CLI -> 0.
            session_dirs = sorted((root / "sessions-cli").iterdir())
            run_dir = next(
                entry
                for entry in session_dirs[-1].iterdir()
                if entry.name.startswith("m11_engine_snapshot_predict_")
            )
            code, out, _ = run_cli(
                "snapshot-run-verify",
                "--forecast-run", str(run_dir),
                "--bundle-manifest", str(build["bundle_manifest_path"]),
                "--output-root", str(root / "audits-cli"),
            )
            self.assertEqual(code, 0, out)

    def test_bundle_build_required_artifact_io_failure_stays_blocked(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            build_inputs_root = root / "case"
            built = self._build_snapshot(build_inputs_root, "frequency")
            fixtures_path = self._write_fixtures(build_inputs_root)
            del fixtures_path
            issue_path = self._fresh_issue_evidence(build_inputs_root, built)

            real_write_json = runtime_bundle_module.write_json

            def failing_write_json(payload, path, *args, **kwargs):
                if Path(str(path)).name == "bundle_manifest.json":
                    raise PermissionError("simulated bundle manifest write failure")
                return real_write_json(payload, path, *args, **kwargs)

            with patch.object(
                runtime_bundle_module, "write_json", side_effect=failing_write_json
            ):
                build = build_runtime_bundle(
                    built["snapshot_manifest_path"], issue_path, output_root=build_inputs_root / "jobs"
                )
            self.assertEqual(build["status"], "blocked")
            self.assertIn("assemble_failed", build["reasons"][0])
            self.assertIn("simulated bundle manifest write failure", build["reasons"][0])
            self.assertEqual(build["stages"]["assemble"], "failed")
            self.assertEqual(build["stages"]["self_verify"], "not_run")
            job = json.loads(Path(build["build_job_path"]).read_text(encoding="utf-8"))
            self.assertEqual(job["status"], "blocked")
            report = Path(build["build_report_path"]).read_text(encoding="utf-8")
            self.assertIn("blocked", report)

    def test_size_limit_blocks_and_keeps_the_candidate_for_ruling(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            built = self._build_snapshot(root, "frequency")
            issue_path = self._fresh_issue_evidence(root, built)
            tiny_limit = 1024
            with patch.object(
                runtime_bundle_module, "RUNTIME_BUNDLE_SIZE_LIMIT_BYTES", tiny_limit
            ), patch.object(
                runtime_bundle_validation_module, "RUNTIME_BUNDLE_SIZE_LIMIT_BYTES", tiny_limit
            ):
                build = build_runtime_bundle(
                    built["snapshot_manifest_path"], issue_path, output_root=root / "jobs"
                )
                self.assertEqual(build["status"], "blocked")
                self.assertIn("bundle_oversize", build["reasons"][0])
                self.assertGreater(build["total_bytes"], tiny_limit)
                self.assertTrue(build["oversize_files"])
                bundle_root = Path(build["bundle_dir"])
                self.assertTrue((bundle_root / "bundle_manifest.json").is_file())
                verification = verify_runtime_bundle(bundle_root / "bundle_manifest.json")
                self.assertEqual(verification["status"], "blocked")
                self.assertIn("above the fixed 5 MiB limit", verification["reasons"][0])
            # The untouched limit verifies the same candidate as blocked too.
            verification = verify_runtime_bundle(bundle_root / "bundle_manifest.json")
            self.assertEqual(verification["status"], "blocked")
            job = json.loads(Path(build["build_job_path"]).read_text(encoding="utf-8"))
            self.assertEqual(job["status"], "blocked")
            per_file = job["bundle_files"]
            self.assertTrue(all("bytes" in entry and "sha256" in entry for entry in per_file))

    def test_bundle_runtime_window_reads_no_sources_after_disconnection(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            model = "poisson"
            built = self._build_snapshot(root, model)
            fixtures_path = self._write_fixtures(root)
            results_path = self._results_file(root)
            issue_path = self._fresh_issue_evidence(root, built)
            build = build_runtime_bundle(
                built["snapshot_manifest_path"], issue_path, output_root=root / "jobs"
            )
            self.assertEqual(build["status"], "ready")
            bundle_root = Path(build["bundle_dir"])
            before_digest = self._bundle_files_digest(bundle_root)

            shutil.rmtree(root / "research")
            issue_path.unlink()

            with self._runtime_guard() as (opened, network_calls):
                session = run_snapshot_session(
                    None,
                    fixtures_path,
                    mode="replay",
                    output_root=root / "runtime" / "sessions",
                    bundle_manifest_path=bundle_root / "bundle_manifest.json",
                )
                standalone = verify_snapshot_run(
                    Path(session["prediction_run_dir"]),
                    output_root=root / "runtime" / "audits",
                    bundle_manifest_path=bundle_root / "bundle_manifest.json",
                )
                settlement = settle_forecast_run(
                    Path(session["prediction_run_dir"]),
                    results_path,
                    output_root=root / "runtime" / "settle",
                )
            self.assertEqual(session["status"], "ready")
            self.assertEqual(standalone["status"], "ready")
            self.assertEqual(settlement["status"], "complete")
            forbidden = [
                entry for entry in opened
                if any(label in entry.lower() for label in SOURCE_PATH_LABELS)
            ]
            self.assertEqual(forbidden, [])
            self.assertEqual(network_calls, [])
            self.assertEqual(self._bundle_files_digest(bundle_root), before_digest)

            # The bundle-internal evidence copy IS read (allowed); the original issue
            # source path was deleted above and never touched through any fallback.
            audit_payload = json.loads(
                Path(standalone["prediction_audit_path"]).read_text(encoding="utf-8")
            )
            self.assertEqual(audit_payload["issue_evidence"]["path"].endswith("issue_audit.json"), True)
            self.assertIn(str(bundle_root), audit_payload["issue_evidence"]["path"])


class BundleReworkTests(RuntimeBundleTestBase):
    """R1-R4 rework regressions: precheck-before-read, structured bad-bundle
    audits, honest build persistence, and the fixed v1 payload set."""

    @contextlib.contextmanager
    def _resolved_open_guard(self):
        """Record builtins.open/io.open calls with their resolved real targets."""

        import socket as socket_module

        opened: list[dict] = []
        real_open = builtins.open
        real_io_open = io_module.open

        def guarded(module_label, file, *args, **kwargs):
            if isinstance(file, (str, Path)):
                requested = str(file)
                try:
                    resolved = str(Path(requested).resolve())
                except OSError:
                    resolved = requested
                opened.append({"via": module_label, "requested": requested, "resolved": resolved})
            return real_open(file, *args, **kwargs)

        with (
            patch("builtins.open", side_effect=lambda f, *a, **k: guarded("builtins", f, *a, **k)),
            patch("io.open", side_effect=lambda f, *a, **k: guarded("io", f, *a, **k)),
            patch.object(urllib.request, "urlopen", side_effect=AssertionError("network used")),
            patch("socket.socket", side_effect=lambda *a, **k: (_ for _ in ()).throw(AssertionError("network used"))),
        ):
            yield opened

    def _outside_reads(self, opened: list[dict], bundle_root: Path) -> list[dict]:
        root_resolved = str(bundle_root.resolve()).casefold()
        outside = []
        for entry in opened:
            resolved = str(Path(entry["resolved"]).resolve()).casefold()
            if resolved != root_resolved and not resolved.startswith(root_resolved + os.sep):
                outside.append(entry)
        return outside

    def test_junction_parent_directory_blocks_before_any_content_read(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            build = self._build_bundle(root, "frequency")
            case_root = root / "junction-case"
            case_root.mkdir()
            bundle_case = case_root / "bundle"
            shutil.copytree(Path(build["bundle_dir"]), bundle_case)
            outside_evidence = case_root / "outside_evidence"
            shutil.copytree(bundle_case / "evidence", outside_evidence)
            junction = bundle_case / "evidence"
            shutil.rmtree(junction)
            self.assertEqual(
                subprocess.run(
                    ["cmd", "/c", "mklink", "/J", str(junction), str(outside_evidence)],
                    capture_output=True,
                ).returncode,
                0,
            )
            self.assertTrue(
                bool(os.lstat(junction).st_file_attributes & stat.FILE_ATTRIBUTE_REPARSE_POINT)
            )

            with self._resolved_open_guard() as opened:
                verification = verify_runtime_bundle(bundle_case / "bundle_manifest.json")
            self.assertEqual(verification["status"], "blocked", verification["reasons"])
            self.assertTrue(
                any("reparse" in reason for reason in verification["reasons"]), verification["reasons"]
            )
            self.assertEqual(self._outside_reads(opened, bundle_case), [])
            self.assertTrue(
                all(
                    check["check"] != "bundle_files_inventory" or check["status"] == "failed"
                    for check in verification["checks"]
                ),
                verification["checks"],
            )

            with contextlib.redirect_stderr(io_module.StringIO()), contextlib.redirect_stdout(io_module.StringIO()):
                with self.assertRaises(SystemExit) as cli_exit:
                    main(
                        [
                            "runtime-bundle-verify",
                            "--bundle-manifest",
                            str(bundle_case / "bundle_manifest.json"),
                        ]
                    )
            self.assertEqual(cli_exit.exception.code, 1)
            subprocess.run(["cmd", "/c", "rmdir", str(junction)], capture_output=True)

    def test_standalone_audit_with_bad_bundle_saves_structured_blocked_artifacts(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            build = self._build_bundle(root, "frequency")
            fixtures_path = self._write_fixtures(root)
            session = run_snapshot_session(
                None,
                fixtures_path,
                mode="replay",
                output_root=root / "runtime" / "sessions",
                bundle_manifest_path=build["bundle_manifest_path"],
            )
            self.assertEqual(session["status"], "ready")
            run_dir = Path(session["prediction_run_dir"])

            def make_missing(case_root: Path) -> Path:
                return case_root / "nope" / "bundle_manifest.json"

            def make_corrupt(case_root: Path) -> Path:
                case_root.mkdir(parents=True, exist_ok=True)
                target = case_root / "corrupt_bundle"
                shutil.copytree(Path(build["bundle_dir"]), target)
                (target / "bundle_manifest.json").write_text("{not-json", encoding="utf-8")
                return target / "bundle_manifest.json"

            def make_hash_wrong(case_root: Path) -> Path:
                case_root.mkdir(parents=True, exist_ok=True)
                target = case_root / "hash_wrong_bundle"
                shutil.copytree(Path(build["bundle_dir"]), target)
                run_md = target / "RUN.md"
                run_md.write_text(run_md.read_text(encoding="utf-8") + "\n", encoding="utf-8")
                return target / "bundle_manifest.json"

            def make_rejected_set(case_root: Path) -> Path:
                case_root.mkdir(parents=True, exist_ok=True)
                target = case_root / "rejected_set_bundle"
                shutil.copytree(Path(build["bundle_dir"]), target)
                (target / "unexpected.exe").write_bytes(b"MZ")

                def mutate(payload):
                    actual = file_sha256(target / "unexpected.exe")
                    payload["files"].append({"path": "unexpected.exe", "bytes": 2, "sha256": actual})
                    payload["size"]["payload_bytes"] = sum(entry["bytes"] for entry in payload["files"])
                    payload["size"]["file_count"] = len(payload["files"])
                    payload["size"]["actual_file_count"] = len(payload["files"]) + 1

                self._rewrite_bundle_manifest(target, mutate)
                return target / "bundle_manifest.json"

            cases = {
                "missing_manifest": make_missing,
                "corrupt_manifest_json": make_corrupt,
                "payload_hash_wrong": make_hash_wrong,
                "payload_set_rejected": make_rejected_set,
            }
            for name, make_manifest in cases.items():
                with self.subTest(case=name):
                    case_root = root / "rework" / name
                    case_root.mkdir(parents=True, exist_ok=True)
                    output_root = case_root / "audits"
                    bundle_manifest = make_manifest(case_root)
                    # API: structured blocked result, never an exception.
                    result = verify_snapshot_run(
                        run_dir, output_root=output_root, bundle_manifest_path=bundle_manifest
                    )
                    self.assertEqual(result["status"], "blocked", result["reasons"])
                    self.assertFalse(result["predictions_usable"])
                    self.assertTrue(
                        result["reasons"][0].startswith("bundle_verification_failed: ")
                        and len(result["reasons"][0]) > len("bundle_verification_failed: "),
                        result["reasons"],
                    )
                    audit_json = Path(result["prediction_audit_path"])
                    audit_md = Path(result["prediction_audit_report_path"])
                    self.assertTrue(audit_json.is_file(), name)
                    self.assertTrue(audit_md.is_file(), name)
                    payload = json.loads(audit_json.read_text(encoding="utf-8"))
                    self.assertEqual(payload["stages"]["bundle_verify"], "failed")
                    self.assertEqual(payload["stages"]["evidence_load"], "not_run")
                    self.assertEqual(payload["stages"]["repredict"], "not_run")
                    self.assertFalse(payload["predictions_usable"])
                    self.assertEqual(payload["bundle"]["verification_status"], "blocked")
                    self.assertIsNone(payload["bundle"]["validation_kind"])
                    self.assertIsNone(payload["bundle"]["derived"])
                    self.assertIsNone(payload["bundle"]["original"])
                    self.assertEqual(
                        payload["bundle"]["lineage_evidence"], "unavailable_bundle_verification_failed"
                    )
                    self.assertTrue(payload["bundle"]["verify_reasons"])
                    self.assertEqual(payload["issue_evidence"]["record_status"], "unavailable")
                    self.assertFalse(payload["issue_evidence"]["fresh_at_issue"])
                    # CLI: exit 1 with the real bundle failure reason.
                    stderr = io_module.StringIO()
                    stdout = io_module.StringIO()
                    with contextlib.redirect_stderr(stderr), contextlib.redirect_stdout(stdout):
                        with self.assertRaises(SystemExit) as cli_exit:
                            main(
                                [
                                    "snapshot-run-verify",
                                    "--forecast-run", str(run_dir),
                                    "--bundle-manifest", str(bundle_manifest),
                                    "--output-root", str(case_root / "audits-cli"),
                                ]
                            )
                    self.assertEqual(cli_exit.exception.code, 1, name)
                    combined = stdout.getvalue() + stderr.getvalue()
                    self.assertIn("blocked", combined)
                    self.assertIn("bundle_verification_failed", combined)

    def test_required_build_report_failure_blocks_delivery_but_keeps_candidate(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            built = self._build_snapshot(root, "frequency")
            issue_path = self._fresh_issue_evidence(root, built)
            real_write = runtime_bundle_module._write_utf8_text

            def failing_write(path, text):
                if Path(path).name == "build_report.md":
                    raise PermissionError("simulated required build report failure")
                return real_write(path, text)

            with patch.object(
                runtime_bundle_module, "_write_utf8_text", side_effect=failing_write
            ):
                build = build_runtime_bundle(
                    built["snapshot_manifest_path"], issue_path, output_root=root / "jobs"
                )
                self.assertEqual(build["status"], "blocked")
                self.assertEqual(build["stages"]["finalize"], "failed")
                self.assertIn("required_artifact_write_failed", build["reasons"][-1])
                self.assertIn("build_report", build["reasons"][-1])
                self.assertEqual(build["stages"]["self_verify"], "completed")
                self.assertFalse(Path(build["build_report_path"]).exists())
                stderr = io_module.StringIO()
                with contextlib.redirect_stderr(stderr), contextlib.redirect_stdout(io_module.StringIO()):
                    with self.assertRaises(SystemExit) as cli_exit:
                        main(
                            [
                                "runtime-bundle-build",
                                "--snapshot-manifest", str(built["snapshot_manifest_path"]),
                                "--issue-audit", str(issue_path),
                                "--output-root", str(root / "jobs-cli"),
                            ]
                        )
                self.assertEqual(cli_exit.exception.code, 1)
            job = json.loads(Path(build["build_job_path"]).read_text(encoding="utf-8"))
            self.assertEqual(job["status"], "blocked")
            self.assertEqual(job["stages"]["finalize"], "failed")
            self.assertIn("required_artifact_write_failed", job["reasons"][-1])
            # The candidate bundle itself stays intact and verifiable; that does
            # not upgrade the failed build delivery to ready.
            self.assertTrue((Path(build["bundle_dir"]) / "bundle_manifest.json").is_file())
            candidate = verify_runtime_bundle(Path(build["bundle_dir"]) / "bundle_manifest.json")
            self.assertEqual(candidate["status"], "ready", candidate["reasons"])
            self.assertNotEqual(build["status"], "ready")

    def test_build_job_record_write_failure_raises_honestly(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            built = self._build_snapshot(root, "frequency")
            issue_path = self._fresh_issue_evidence(root, built)
            real_write_json = runtime_bundle_module.write_json

            def failing_write_json(payload, path, *args, **kwargs):
                if Path(str(path)).name == "build_job.json":
                    raise PermissionError("simulated build job record failure")
                return real_write_json(payload, path, *args, **kwargs)

            with patch.object(
                runtime_bundle_module, "write_json", side_effect=failing_write_json
            ):
                with self.assertRaises(OSError) as raised:
                    build_runtime_bundle(
                        built["snapshot_manifest_path"], issue_path, output_root=root / "jobs"
                    )
                self.assertIn("build job record", str(raised.exception))
                self.assertIn("remain on disk", str(raised.exception))
                stderr = io_module.StringIO()
                with contextlib.redirect_stderr(stderr), contextlib.redirect_stdout(io_module.StringIO()):
                    with self.assertRaises(SystemExit) as cli_exit:
                        main(
                            [
                                "runtime-bundle-build",
                                "--snapshot-manifest", str(built["snapshot_manifest_path"]),
                                "--issue-audit", str(issue_path),
                                "--output-root", str(root / "jobs-cli"),
                            ]
                        )
                self.assertEqual(cli_exit.exception.code, 1)
                self.assertIn("build job record", stderr.getvalue())

    def test_required_bundle_manifest_hash_read_failure_blocks_delivery(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            built = self._build_snapshot(root, "frequency")
            issue_path = self._fresh_issue_evidence(root, built)
            real_sha = runtime_bundle_module.file_sha256

            def failing_sha(path, *args, **kwargs):
                if Path(str(path)).name == "bundle_manifest.json":
                    raise PermissionError("simulated required artifact hash read failure")
                return real_sha(path, *args, **kwargs)

            with patch.object(runtime_bundle_module, "file_sha256", side_effect=failing_sha):
                build = build_runtime_bundle(
                    built["snapshot_manifest_path"], issue_path, output_root=root / "jobs"
                )
            self.assertEqual(build["status"], "blocked")
            self.assertEqual(build["stages"]["finalize"], "failed")
            self.assertIn("required_artifact_write_failed", build["reasons"][-1])
            self.assertIn("bundle_manifest_sha256", build["reasons"][-1])
            self.assertIsNone(build["bundle_manifest_sha256"])
            job = json.loads(Path(build["build_job_path"]).read_text(encoding="utf-8"))
            self.assertEqual(job["status"], "blocked")
            self.assertIsNone(job["bundle_manifest_sha256"])

    def _sync_catalog(self, bundle_root: Path, mutate_files) -> None:
        def mutate(payload):
            mutate_files(payload)
            payload["size"]["payload_bytes"] = sum(entry["bytes"] for entry in payload["files"])
            payload["size"]["file_count"] = len(payload["files"])
            payload["size"]["actual_file_count"] = len(payload["files"]) + 1

        self._rewrite_bundle_manifest(bundle_root, mutate)

    def test_fixed_v1_payload_set_and_recorded_bytes_are_enforced(self):
        import hashlib

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            build = self._build_bundle(root, "frequency")
            control_root = Path(build["bundle_dir"])
            control = verify_runtime_bundle(control_root / "bundle_manifest.json")
            self.assertEqual(control["status"], "ready", control["reasons"])
            self.assertEqual(control["file_count"], 7)
            self.assertEqual(control["actual_file_count"], 8)
            snapshot_id = build["snapshot_id"]
            expected_payload = {
                "evidence/original_snapshot_manifest.json",
                "evidence/issue_audit.json",
                "team_index.json",
                "RUN.md",
                f"snapshot/{snapshot_id}/snapshot_manifest.json",
                f"snapshot/{snapshot_id}/model_states.json",
                f"snapshot/{snapshot_id}/snapshot_report.md",
            }
            self.assertEqual(
                {entry["path"] for entry in json.loads(
                    (control_root / "bundle_manifest.json").read_text(encoding="utf-8")
                )["files"]},
                expected_payload,
            )

            cases = {}
            # Listed extra executable (synthetic sentinel bytes, never executed).
            case = root / "listed_executable"
            shutil.copytree(control_root, case / "bundle")
            (case / "bundle" / "unexpected.exe").write_bytes(b"MZ")

            def add_exe(payload, case=case):
                payload["files"].append(
                    {"path": "unexpected.exe", "bytes": 2, "sha256": hashlib.sha256(b"MZ").hexdigest()}
                )

            cases["listed_executable"] = (case, add_exe)
            # Listed extra raw-results CSV.
            case = root / "listed_raw_csv"
            shutil.copytree(control_root, case / "bundle")
            (case / "bundle" / "unexpected_raw.csv").write_bytes(b"h,d\n")

            def add_csv(payload, case=case):
                payload["files"].append(
                    {
                        "path": "unexpected_raw.csv",
                        "bytes": 4,
                        "sha256": hashlib.sha256(b"h,d\n").hexdigest(),
                    }
                )

            cases["listed_raw_csv"] = (case, add_csv)
            # Required RUN.md removed on disk and dropped from the catalog.
            case = root / "missing_required_RUN"
            shutil.copytree(control_root, case / "bundle")
            (case / "bundle" / "RUN.md").unlink()

            def drop_run(payload):
                payload["files"] = [entry for entry in payload["files"] if entry["path"] != "RUN.md"]

            cases["missing_required_RUN"] = (case, drop_run)
            # Same bytes, but the recorded size lies (0) with the totals synced.
            case = root / "false_payload_size"
            shutil.copytree(control_root, case / "bundle")

            def fake_size(payload):
                for entry in payload["files"]:
                    if entry["path"] == "RUN.md":
                        entry["bytes"] = 0

            cases["false_payload_size"] = (case, fake_size)

            for name, (case, mutate) in cases.items():
                with self.subTest(case=name):
                    bundle_case = case / "bundle"
                    self._sync_catalog(bundle_case, mutate)
                    verification = verify_runtime_bundle(bundle_case / "bundle_manifest.json")
                    self.assertEqual(verification["status"], "blocked", verification["reasons"])
                    self.assertTrue(
                        any(
                            marker in verification["reasons"][0]
                            for marker in ("fixed v1 payload set", "required bundle payload is missing",
                                           "recorded byte size does not match")
                        ),
                        verification["reasons"],
                    )
                    stderr = io_module.StringIO()
                    with contextlib.redirect_stderr(stderr), contextlib.redirect_stdout(io_module.StringIO()):
                        with self.assertRaises(SystemExit) as cli_exit:
                            main(
                                [
                                    "runtime-bundle-verify",
                                    "--bundle-manifest",
                                    str(bundle_case / "bundle_manifest.json"),
                                ]
                            )
                    self.assertEqual(cli_exit.exception.code, 1, name)

    def test_normalization_colliding_paths_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            build = self._build_bundle(root, "frequency")
            bundle_case = root / "collide" / "bundle"
            bundle_case.parent.mkdir(parents=True)
            shutil.copytree(Path(build["bundle_dir"]), bundle_case)

            def mutate(payload):
                # A second listing that collides with RUN.md under Windows
                # normalization (same file, different case in the catalog).
                run_entry = next(entry for entry in payload["files"] if entry["path"] == "RUN.md")
                payload["files"].append(dict(run_entry, path="run.md"))
                payload["size"]["payload_bytes"] = sum(entry["bytes"] for entry in payload["files"])
                payload["size"]["file_count"] = len(payload["files"])
                payload["size"]["actual_file_count"] = len(payload["files"]) + 1

            self._rewrite_bundle_manifest(bundle_case, mutate)
            verification = verify_runtime_bundle(bundle_case / "bundle_manifest.json")
            self.assertEqual(verification["status"], "blocked", verification["reasons"])
            self.assertTrue(
                any("normalization" in reason for reason in verification["reasons"]),
                verification["reasons"],
            )


if __name__ == "__main__":
    unittest.main()
