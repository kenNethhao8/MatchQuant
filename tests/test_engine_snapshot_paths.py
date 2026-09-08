import tempfile
import unittest
from pathlib import Path

from matchquant.engine_snapshot import build_engine_snapshot, predict_from_engine_snapshot
from matchquant.schema import DataValidationError
from test_engine_snapshot import MODEL_CONFIGS, _fixtures_frame, _history_frame, _write_history_evidence


class EngineSnapshotPathTests(unittest.TestCase):
    def test_prediction_rejects_output_root_inside_snapshot(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            history_path, manifest_path = _write_history_evidence(root, _history_frame())
            fixtures_path = root / "fixtures.csv"
            _fixtures_frame().to_csv(fixtures_path, index=False)
            built = build_engine_snapshot(
                history_path,
                manifest_path,
                "2024-01-07",
                model_names=["frequency"],
                model_configs={"frequency": MODEL_CONFIGS["frequency"]},
                output_root=root / "snapshots",
            )
            snapshot_dir = Path(built["snapshot_dir"])
            nested = snapshot_dir / "nested-runs"
            with self.assertRaisesRegex(DataValidationError, "input directory"):
                predict_from_engine_snapshot(built["snapshot_manifest_path"], fixtures_path, output_root=nested)
            self.assertFalse(nested.exists())
            self.assertEqual({path.name for path in snapshot_dir.iterdir()}, {"model_states.json", "snapshot_manifest.json"})
