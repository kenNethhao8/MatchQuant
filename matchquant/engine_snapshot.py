"""Offline M11 engine-state snapshot build and snapshot-only prediction."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence
from uuid import uuid4

import pandas as pd

from .backtest import _parse_boundary
from .fixtures import FIXTURE_SCHEMA_VERSION, load_fixtures
from .forecast import (
    FORECAST_PREDICTION_COLUMNS,
    _fit_history,
    _fixtures_at_or_after_as_of,
    _history_before_as_of,
    _prediction_row,
)
from .io import file_sha256, load_matches, write_json, write_predictions
from .models import create_model
from .schema import DataValidationError
from .engine_snapshot_report import SNAPSHOT_DISCLAIMER, write_snapshot_prediction_report
from .engine_snapshot_support import (
    ENGINE_SNAPSHOT_SCHEMA_VERSION,
    ENGINE_STATE_SCHEMA_VERSION,
    canonical_json_bytes,
    export_model_state,
    import_model_state,
    model_states_payload,
)
from .engine_snapshot_manifest import load_engine_snapshot, new_engine_snapshot_dir, write_snapshot_manifest
from .engine_snapshot_build_support import (
    config_sources,
    history_manifest_evidence,
    reject_output_root,
    selection_evidence,
)

SNAPSHOT_PREDICT_SCHEMA_VERSION = "m11-engine-snapshot-predict-v1"


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _new_prediction_run_dir(output_root: str | Path, snapshot_dir: Path, fixtures_path: Path) -> Path:
    root = Path(output_root).expanduser().resolve()
    reject_output_root(root, snapshot_dir, fixtures_path)
    root.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = root / f"m11_engine_snapshot_predict_{stamp}_{uuid4().hex[:8]}"
    run_dir.mkdir()
    return run_dir


def build_engine_snapshot(
    history_path: str | Path,
    history_manifest_path: str | Path,
    as_of: str | pd.Timestamp,
    *,
    model_names: Sequence[str] | None = None,
    selection_run_config_path: str | Path | None = None,
    model_configs: Mapping[str, Mapping[str, Any]] | None = None,
    output_root: str | Path = Path(r"D:\Codex\matchquant_data\engine_snapshots"),
    season_mode: str | None = None,
    model_config_paths: Mapping[str, str | Path] | None = None,
) -> dict[str, Any]:
    """Build a two-file, immutable state snapshot from explicit M4 history evidence."""

    history_file = Path(history_path).expanduser().resolve()
    manifest_file = Path(history_manifest_path).expanduser().resolve()
    reject_output_root(Path(output_root).expanduser().resolve(), history_file, manifest_file)
    history = load_matches(history_file)
    history_hash, history_manifest, observed_cutoff = history_manifest_evidence(
        history_file, manifest_file, history
    )
    timeline, eligible, as_of_meta, cutoff = _history_before_as_of(history, as_of)
    names, configs, selection, selection_season_mode = selection_evidence(
        history_hash, as_of, model_names, model_configs, selection_run_config_path
    )
    actual_season_mode = season_mode or selection_season_mode or "carry"
    if actual_season_mode not in {"carry", "reset"}:
        raise ValueError("season_mode must be 'carry' or 'reset'")
    sources = config_sources(names, model_config_paths)
    if selection_run_config_path is not None:
        sources = {
            name: {
                "mode": "selection_run_config",
                "path": selection["path"],
                "sha256": selection["sha256"],
            }
            for name in names
        }

    records: list[dict[str, Any]] = []
    for name in names:
        model = create_model(name, configs[name])
        _fit_history(eligible, model, actual_season_mode)
        records.append(export_model_state(model, configs[name]))
    state_payload = model_states_payload(records)

    snapshot_dir = new_engine_snapshot_dir(output_root)
    state_path = snapshot_dir / "model_states.json"
    state_path.write_bytes(canonical_json_bytes(state_payload))
    state_hash = file_sha256(state_path)
    manifest_path = snapshot_dir / "snapshot_manifest.json"
    manifest_models = []
    for record in sorted(records, key=lambda item: str(item["model"])):
        manifest_models.append(
            {
                "model": record["model"],
                "config": record["config"],
                "config_sha256": record["config_sha256"],
                "state_sha256": record["state_sha256"],
                "config_source": sources[record["model"]],
                "state_codec": ENGINE_STATE_SCHEMA_VERSION,
            }
        )
    manifest = {
        "schema_version": ENGINE_SNAPSHOT_SCHEMA_VERSION,
        "snapshot_id": snapshot_dir.name,
        "generated_at": _utc_now(),
        "as_of": as_of_meta,
        "history": {
            "path": str(history_file),
            "sha256": history_hash,
            "rows": int(len(timeline)),
            "rows_used_before_as_of": int(len(eligible)),
            "rows_excluded_at_or_after_as_of": int(len(timeline) - len(eligible)),
            "observed_cutoff": observed_cutoff,
            "data_cutoff": history_manifest.get("data_cutoff"),
            "used_data_cutoff": str(eligible.iloc[-1]["date"]) if not eligible.empty else None,
        },
        "history_manifest": {
            "path": str(manifest_file),
            "sha256": file_sha256(manifest_file),
            "schema_version": history_manifest["schema_version"],
            "dataset_version": history_manifest.get("dataset_version"),
            "data_cutoff": history_manifest.get("data_cutoff"),
            "output_sha256": history_manifest["output"]["sha256"],
        },
        "model_names": list(names),
        "models": manifest_models,
        "state": {
            "path": str(state_path),
            "sha256": state_hash,
            "schema_version": ENGINE_STATE_SCHEMA_VERSION,
        },
        "outputs": {
            "snapshot_dir": str(snapshot_dir),
            "snapshot_manifest": str(manifest_path),
            "model_states": str(state_path),
        },
        "selection": selection,
        "season_mode": actual_season_mode,
        "validation": {
            "history_rows_strictly_before_as_of": int(len(eligible)),
            "fixture_results_used_for_model_updates": False,
            "history_copied": False,
            "input_files_unchanged": True,
            "network_accessed": False,
            "snapshot_is_immutable": True,
        },
        "disclaimer": SNAPSHOT_DISCLAIMER,
    }
    write_snapshot_manifest(manifest, manifest_path)
    manifest_hash = file_sha256(manifest_path)
    return {
        "snapshot_dir": str(snapshot_dir),
        "snapshot_manifest_path": str(manifest_path),
        "manifest_path": str(manifest_path),
        "model_states_path": str(state_path),
        "state_path": str(state_path),
        "manifest": json.loads(manifest_path.read_text(encoding="utf-8")),
        "manifest_sha256": manifest_hash,
        "state_sha256": state_hash,
        "model_names": list(names),
        "models": records,
        "history_sha256": history_hash,
        "history_manifest_sha256": file_sha256(manifest_file),
        "as_of": as_of_meta,
        "cutoff": str(cutoff),
    }


def predict_from_engine_snapshot(
    snapshot_manifest_path: str | Path,
    fixtures_path: str | Path,
    *,
    output_root: str | Path = Path(r"D:\Codex\matchquant_runs"),
) -> dict[str, Any]:
    """Predict from JSON state only; the history path recorded in the snapshot is not read."""

    loaded = load_engine_snapshot(snapshot_manifest_path)
    manifest = loaded["manifest"]
    fixtures_file = Path(fixtures_path).expanduser().resolve()
    if not fixtures_file.is_file():
        raise FileNotFoundError(f"fixture CSV does not exist: {fixtures_file}")
    fixtures_hash = file_sha256(fixtures_file)
    fixtures = load_fixtures(fixtures_file)
    as_of_meta = manifest["as_of"]
    as_of_timestamp, date_only = _parse_boundary(as_of_meta["value"], "snapshot as_of")
    cutoff = as_of_timestamp.normalize() if date_only else as_of_timestamp
    fixture_timeline = _fixtures_at_or_after_as_of(fixtures, as_of_meta, cutoff)

    predictions: list[dict[str, Any]] = []
    for record in loaded["models"]:
        model = import_model_state(record)
        for row in fixture_timeline.itertuples(index=False):
            predictions.append(_prediction_row(row, str(model.name), model.predict(row.home_team, row.away_team)))
    prediction_frame = pd.DataFrame(predictions, columns=FORECAST_PREDICTION_COLUMNS)
    if prediction_frame.empty:
        raise DataValidationError("fixture set is empty after snapshot prediction validation")

    snapshot_dir = Path(loaded["manifest_path"]).parent
    run_dir = _new_prediction_run_dir(output_root, snapshot_dir, fixtures_file)
    predictions_path = write_predictions(prediction_frame, run_dir / "predictions.csv")
    predictions_hash = file_sha256(predictions_path)
    report_path = run_dir / "snapshot_report.md"
    config_path = run_dir / "run_config.json"
    config = {
        "schema_version": SNAPSHOT_PREDICT_SCHEMA_VERSION,
        "command": "matchquant engine-snapshot-predict",
        "generated_at": _utc_now(),
        "as_of": as_of_meta,
        "snapshot": {
            "snapshot_id": manifest["snapshot_id"],
            "manifest_path": loaded["manifest_path"],
            "manifest_sha256": loaded["manifest_sha256"],
            "manifest_content_sha256": manifest["manifest_sha256"],
            "state_path": loaded["state_path"],
            "state_sha256": loaded["state_sha256"],
            "history_path": manifest["history"]["path"],
            "history_sha256": manifest["history"]["sha256"],
            "history_cutoff": manifest["history"]["used_data_cutoff"],
        },
        "fixtures": {
            "path": str(fixtures_file),
            "sha256": fixtures_hash,
            "rows": int(len(fixtures)),
            "schema": f"{FIXTURE_SCHEMA_VERSION}-score-free",
        },
        "model_names": [str(record["model"]) for record in loaded["models"]],
        "models": [
            {
                "model": record["model"],
                "config": record["config"],
                "config_sha256": record["config_sha256"],
                "state_sha256": record["state_sha256"],
            }
            for record in loaded["models"]
        ],
        "validation": {
            "all_fixture_dates_at_or_after_as_of": True,
            "fixture_results_used_for_model_updates": False,
            "history_read_during_prediction": False,
            "snapshot_mutated": False,
            "network_accessed": False,
            "probabilities_finite_non_negative_sum_to_one": True,
        },
        "predictions_sha256": predictions_hash,
        "outputs": {
            "run_dir": str(run_dir),
            "predictions": str(predictions_path),
            "predictions_sha256": predictions_hash,
            "snapshot_report": str(report_path),
            "run_config": str(config_path),
        },
        "disclaimer": SNAPSHOT_DISCLAIMER,
    }
    write_json(config, config_path)
    write_snapshot_prediction_report(report_path, config, prediction_frame)
    return {
        "run_dir": str(run_dir),
        "predictions_path": str(predictions_path),
        "snapshot_report_path": str(report_path),
        "report_path": str(report_path),
        "run_config_path": str(config_path),
        "predictions": prediction_frame,
        "config": config,
        "snapshot_manifest_sha256": loaded["manifest_sha256"],
        "fixtures_sha256": fixtures_hash,
    }


__all__ = [
    "SNAPSHOT_PREDICT_SCHEMA_VERSION",
    "build_engine_snapshot",
    "predict_from_engine_snapshot",
]
