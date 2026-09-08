import contextlib
import io
import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from matchquant.engine_snapshot import build_engine_snapshot, predict_from_engine_snapshot
from matchquant.engine_snapshot_status import check_engine_snapshot_status
from matchquant.io import file_sha256, write_json
from matchquant.schema import DataValidationError

from test_engine_snapshot import MODEL_CONFIGS, _fixtures_frame, _history_frame, _write_history_evidence


def _write_pointer(
    pointer_path: Path,
    canonical_path: Path,
    processed_manifest_path: Path,
    *,
    dataset_version: str,
) -> None:
    write_json(
        {
            "schema_version": "m2.1-current-pointer-v1",
            "dataset_version": dataset_version,
            "updated_at": "2026-09-08T00:00:00Z",
            "canonical": str(canonical_path.resolve()),
            "processed_manifest": str(processed_manifest_path.resolve()),
            "quality_report": None,
            "diff_report": None,
            "canonical_sha256": file_sha256(canonical_path),
        },
        pointer_path,
    )


class EngineSnapshotStatusTests(unittest.TestCase):
    def _case(self, root: Path) -> tuple[Path, Path, Path, dict]:
        history_path, history_manifest_path = _write_history_evidence(root, _history_frame())
        fixtures_path = root / "fixtures.csv"
        _fixtures_frame().to_csv(fixtures_path, index=False)
        built = build_engine_snapshot(
            history_path,
            history_manifest_path,
            "2024-01-07",
            model_names=["frequency"],
            model_configs={"frequency": MODEL_CONFIGS["frequency"]},
            output_root=root / "snapshots",
        )
        pointer_path = root / "pointer.json"
        _write_pointer(
            pointer_path,
            history_path,
            history_manifest_path,
            dataset_version="m2.1-test-snapshot",
        )
        return history_path, history_manifest_path, fixtures_path, {"built": built, "pointer": pointer_path}

    def test_fresh_status_writes_audit_without_mutating_inputs(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            history_path, history_manifest_path, _, paths = self._case(root)
            snapshot_dir = Path(paths["built"]["snapshot_dir"])
            before = {
                "pointer": paths["pointer"].read_bytes(),
                "snapshot": {path.name: file_sha256(path) for path in snapshot_dir.iterdir()},
                "history": history_path.read_bytes(),
                "history_manifest": history_manifest_path.read_bytes(),
            }

            result = check_engine_snapshot_status(
                paths["built"]["snapshot_manifest_path"],
                paths["pointer"],
                output_root=root / "runs",
            )

            self.assertEqual(result["status"], "fresh")
            self.assertEqual(result["comparison"]["canonical"]["path_match"], True)
            self.assertEqual(result["comparison"]["processed_manifest"]["hash_match"], True)
            self.assertTrue(Path(result["status_manifest_path"]).is_file())
            self.assertTrue(Path(result["status_report_path"]).is_file())
            self.assertEqual(paths["pointer"].read_bytes(), before["pointer"])
            self.assertEqual(history_path.read_bytes(), before["history"])
            self.assertEqual(history_manifest_path.read_bytes(), before["history_manifest"])
            self.assertEqual(
                {path.name: file_sha256(path) for path in snapshot_dir.iterdir()},
                before["snapshot"],
            )

    def test_verified_but_different_pointer_is_stale(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            history_path, history_manifest_path, _, paths = self._case(root)
            updated_dir = root / "updated"
            updated_dir.mkdir()
            updated_history = updated_dir / "history.csv"
            updated_manifest = updated_dir / "history_manifest.json"
            shutil.copyfile(history_path, updated_history)
            manifest = json.loads(history_manifest_path.read_text(encoding="utf-8"))
            manifest["dataset_version"] = "m2.1-updated"
            manifest["output"]["filename"] = str(updated_history.resolve())
            manifest["output"]["sha256"] = file_sha256(updated_history)
            write_json(manifest, updated_manifest)
            pointer_path = root / "updated-pointer.json"
            _write_pointer(pointer_path, updated_history, updated_manifest, dataset_version="m2.1-updated")

            result = check_engine_snapshot_status(
                paths["built"]["snapshot_manifest_path"],
                pointer_path,
                output_root=root / "runs",
            )

            self.assertEqual(result["status"], "stale")
            self.assertFalse(result["comparison"]["canonical"]["path_match"])
            self.assertTrue(result["comparison"]["canonical"]["hash_match"])
            self.assertIn("canonical_path_mismatch", result["reasons"])

    def test_invalid_pointer_is_unverifiable_and_still_writes_evidence(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _, _, _, paths = self._case(root)
            pointer = json.loads(paths["pointer"].read_text(encoding="utf-8"))
            pointer["canonical_sha256"] = "0" * 64
            write_json(pointer, paths["pointer"])
            before = paths["pointer"].read_bytes()

            result = check_engine_snapshot_status(
                paths["built"]["snapshot_manifest_path"],
                paths["pointer"],
                output_root=root / "runs",
            )

            self.assertEqual(result["status"], "unverifiable")
            self.assertEqual(result["current_pointer"]["status"], "invalid")
            self.assertTrue(result["current_pointer"]["errors"])
            self.assertEqual(result["current_pointer"]["canonical"]["recorded_sha256"], "0" * 64)
            self.assertTrue(Path(result["status_manifest_path"]).is_file())
            self.assertEqual(paths["pointer"].read_bytes(), before)

    def test_invalid_snapshot_is_unverifiable_without_mutating_pointer(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _, _, _, paths = self._case(root)
            snapshot_manifest = Path(paths["built"]["snapshot_manifest_path"])
            before = snapshot_manifest.read_bytes()
            snapshot_manifest.write_text(before.decode("utf-8").replace('"m11-engine-snapshot-v1"', '"tampered"'), encoding="utf-8")
            pointer_before = paths["pointer"].read_bytes()

            result = check_engine_snapshot_status(
                snapshot_manifest,
                paths["pointer"],
                output_root=root / "runs",
            )

            self.assertEqual(result["status"], "unverifiable")
            self.assertEqual(result["snapshot"]["status"], "invalid")
            self.assertTrue(result["snapshot"]["errors"])
            self.assertEqual(paths["pointer"].read_bytes(), pointer_before)

    def test_status_rejects_output_root_inside_snapshot(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _, _, _, paths = self._case(root)
            snapshot_dir = Path(paths["built"]["snapshot_dir"])
            before = {path.name: file_sha256(path) for path in snapshot_dir.iterdir()}

            with self.assertRaisesRegex(DataValidationError, "input directory"):
                check_engine_snapshot_status(
                    paths["built"]["snapshot_manifest_path"],
                    paths["pointer"],
                    output_root=snapshot_dir / "audit",
                )

            self.assertFalse((snapshot_dir / "audit").exists())
            self.assertEqual(
                {path.name: file_sha256(path) for path in snapshot_dir.iterdir()},
                before,
            )

    def test_stale_snapshot_remains_reproducible_after_status_check(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _, _, fixtures_path, paths = self._case(root)
            updated_dir = root / "updated"
            updated_dir.mkdir()
            updated_history = updated_dir / "history.csv"
            updated_manifest = updated_dir / "history_manifest.json"
            original_history = Path(paths["built"]["manifest"]["history"]["path"])
            original_manifest = Path(paths["built"]["manifest"]["history_manifest"]["path"])
            shutil.copyfile(original_history, updated_history)
            manifest = json.loads(original_manifest.read_text(encoding="utf-8"))
            manifest["dataset_version"] = "m2.1-updated"
            manifest["output"]["filename"] = str(updated_history.resolve())
            manifest["output"]["sha256"] = file_sha256(updated_history)
            write_json(manifest, updated_manifest)
            pointer_path = root / "updated-pointer.json"
            _write_pointer(pointer_path, updated_history, updated_manifest, dataset_version="m2.1-updated")
            snapshot_dir = Path(paths["built"]["snapshot_dir"])
            before = {path.name: file_sha256(path) for path in snapshot_dir.iterdir()}

            result = check_engine_snapshot_status(
                paths["built"]["snapshot_manifest_path"],
                pointer_path,
                output_root=root / "status-runs",
            )
            prediction = predict_from_engine_snapshot(
                paths["built"]["snapshot_manifest_path"],
                fixtures_path,
                output_root=root / "prediction-runs",
            )

            self.assertEqual(result["status"], "stale")
            self.assertTrue(Path(prediction["predictions_path"]).is_file())
            self.assertEqual(
                {path.name: file_sha256(path) for path in snapshot_dir.iterdir()},
                before,
            )

    def test_cli_status_is_offline_and_concise(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _, _, _, paths = self._case(root)
            from matchquant.cli import main

            output = io.StringIO()
            with patch("urllib.request.urlopen", side_effect=AssertionError("network")), contextlib.redirect_stdout(output):
                with self.assertRaises(SystemExit) as exit_info:
                    main(
                        [
                            "engine-snapshot-status",
                            "--snapshot-manifest",
                            paths["built"]["snapshot_manifest_path"],
                            "--current-pointer",
                            str(paths["pointer"]),
                            "--output-root",
                            str(root / "runs"),
                        ]
                    )

            self.assertEqual(exit_info.exception.code, 0)
            self.assertLessEqual(len(output.getvalue().strip().splitlines()), 12)


if __name__ == "__main__":
    unittest.main()
