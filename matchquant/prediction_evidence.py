"""Unified prediction-evidence adapter bridging M4 and single-model M11 runs into M6.

``load_prediction_evidence`` dispatches on the actual ``schema_version`` recorded in a
run directory's ``run_config.json``:

- ``m4-forecast-v1`` reuses the existing strict M4 loader unchanged (legacy hash
  semantics preserved) and maps the result onto the shared evidence shape.
- ``m11-engine-snapshot-predict-v1`` performs strict cross-file validation of the
  prediction run against its recorded immutable snapshot (manifest, state, model,
  config, as_of, fixtures, hashes) without reading any history file.

The adapter never rewrites a producer's run_config into a different schema, never
creates synthetic M4 fields for M11 runs, and never reads the history CSV or the
history processed manifest recorded in snapshot lineage.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from .backtest import _parse_boundary
from .engine_snapshot import SNAPSHOT_PREDICT_SCHEMA_VERSION
from .engine_snapshot_manifest import load_engine_snapshot
from .engine_snapshot_support import configuration_hash
from .fixtures import load_fixtures
from .forecast import FORECAST_SCHEMA_VERSION, _configuration_hash
from .io import file_sha256
from .models import MODEL_NAMES
from .schema import DataValidationError, has_explicit_time
from .settlement_validation import (
    _read_prediction_csv,
    _validate_prediction_dates,
    load_forecast_run,
)

PREDICTION_EVIDENCE_SCHEMAS = (FORECAST_SCHEMA_VERSION, SNAPSHOT_PREDICT_SCHEMA_VERSION)


def _read_json_object(path: Path, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise DataValidationError(f"could not read {label} '{path}': {exc}") from exc
    if not isinstance(payload, dict):
        raise DataValidationError(f"{label} root must be a JSON object: {path}")
    return payload


def _require_valid_sha256(value: object, message: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise DataValidationError(message)
    return value.lower()


def load_prediction_evidence(forecast_run_dir: str | Path) -> dict[str, Any]:
    """Load and strictly validate one M4 or single-model M11 prediction run."""

    run_dir = Path(forecast_run_dir).expanduser().resolve()
    if not run_dir.is_dir():
        raise FileNotFoundError(f"prediction run directory does not exist: {run_dir}")
    config_path = run_dir / "run_config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"prediction run_config.json does not exist: {config_path}")
    config = _read_json_object(config_path, "prediction run config")
    schema_version = config.get("schema_version")
    if schema_version == FORECAST_SCHEMA_VERSION:
        return _m4_evidence(run_dir)
    if schema_version == SNAPSHOT_PREDICT_SCHEMA_VERSION:
        return _m11_evidence(run_dir, config, config_path)
    raise DataValidationError(
        "unsupported prediction run schema_version; expected "
        f"{FORECAST_SCHEMA_VERSION} or {SNAPSHOT_PREDICT_SCHEMA_VERSION}: {schema_version!r}"
    )


def _m4_evidence(run_dir: Path) -> dict[str, Any]:
    """Adapt one M4 run using the existing legacy loader and semantics."""

    loaded = load_forecast_run(run_dir)
    config = loaded["config"]
    model_block = config.get("model") if isinstance(config.get("model"), Mapping) else {}
    recorded_config_sha256 = model_block.get("config_sha256")
    model_config_sha256: str | None = None
    limitations: list[str] = []
    if isinstance(recorded_config_sha256, str):
        recomputed = _configuration_hash(model_block.get("config"))
        if recorded_config_sha256 != recomputed:
            raise DataValidationError(
                "M4 run config model.config_sha256 does not match model.config contents"
            )
        model_config_sha256 = recorded_config_sha256
    else:
        limitations.append(
            "M4 run config does not record model.config_sha256; the model config content hash is not verified"
        )
    if loaded["prediction_hash_status"] == "unrecorded_legacy":
        limitations.append(
            "legacy M4 run does not record a predictions SHA-256; recorded-hash verification is unavailable (unrecorded_legacy)"
        )
    origin_evidence = {
        "kind": "m4_run_config",
        "history": config.get("history"),
        "fixtures": config.get("fixtures"),
        "season_mode": config.get("season_mode"),
        "selection": config.get("selection"),
        "history_evidence_status": "recorded_by_producer_not_reread_by_settlement",
        "model_config_hash_status": "verified" if model_config_sha256 else "unrecorded",
    }
    return {
        "producer_schema": FORECAST_SCHEMA_VERSION,
        "producer_command": config.get("command"),
        "run_dir": loaded["run_dir"],
        "run_config_path": loaded["config_path"],
        "run_config_sha256": loaded["config_hash"],
        "predictions_path": loaded["predictions_path"],
        "predictions_sha256": loaded["prediction_hash"],
        "prediction_hash_status": loaded["prediction_hash_status"],
        "predictions": loaded["predictions"],
        "model_name": loaded["model_name"],
        "model_config": model_block.get("config"),
        "model_config_sha256": model_config_sha256,
        "as_of": loaded["as_of"],
        "as_of_date_only": loaded["as_of_date_only"],
        "as_of_meta": loaded["as_of_meta"],
        "origin_evidence": origin_evidence,
        "limitations": limitations,
    }


def _m11_as_of_evidence(
    config: Mapping[str, Any], *, snapshot_as_of: Mapping[str, Any]
) -> tuple[pd.Timestamp, bool, dict[str, object]]:
    """Bind the M11 run's actually recorded as_of value/timestamp/date_only.

    Unlike the M4 helper, a missing or empty recorded field is rejected instead of
    being derived from ``as_of.value``; M11 producer evidence must be complete.
    """

    raw = config.get("as_of")
    if not isinstance(raw, Mapping):
        raise DataValidationError("M11 run config must contain an as_of object")
    for field in ("value", "timestamp", "date_only"):
        if field not in raw or raw[field] is None or (isinstance(raw[field], str) and not raw[field].strip()):
            raise DataValidationError(f"M11 run config as_of.{field} must be actually recorded and non-empty")
    if not isinstance(raw["date_only"], bool):
        raise DataValidationError("M11 run config as_of.date_only must be a boolean")
    try:
        timestamp, inferred_date_only = _parse_boundary(raw["value"], "snapshot prediction as_of")
    except (TypeError, ValueError) as exc:
        raise DataValidationError(str(exc)) from exc
    if bool(raw["date_only"]) != inferred_date_only:
        raise DataValidationError("M11 run config as_of.date_only is inconsistent with as_of.value")
    try:
        parsed_recorded = pd.Timestamp(raw["timestamp"])
    except (TypeError, ValueError) as exc:
        raise DataValidationError("M11 run config as_of.timestamp is not a parseable timestamp") from exc
    if pd.isna(parsed_recorded) or parsed_recorded != timestamp:
        raise DataValidationError("M11 run config as_of.timestamp does not match as_of.value")
    if str(raw["value"]) != str(snapshot_as_of.get("value")):
        raise DataValidationError("M11 run as_of.value does not match the snapshot manifest as_of")
    if str(raw["timestamp"]) != str(snapshot_as_of.get("timestamp")):
        raise DataValidationError("M11 run as_of.timestamp does not match the snapshot manifest as_of")
    if bool(raw["date_only"]) != bool(snapshot_as_of.get("date_only")):
        raise DataValidationError("M11 run as_of.date_only does not match the snapshot manifest as_of")
    return timestamp, inferred_date_only, {
        "value": str(raw["value"]),
        "timestamp": timestamp.isoformat(),
        "date_only": inferred_date_only,
    }


def _validate_used_data_cutoff(
    history: Mapping[str, Any], *, as_of: pd.Timestamp, date_only: bool
) -> pd.Timestamp | None:
    """Strict validity and temporal gate for the snapshot's recorded used_data_cutoff.

    ``rows_used_before_as_of > 0`` requires a legal, finite used cutoff; an empty
    training source keeps its legal ``null``. A date-only as_of requires the used
    cutoff strictly before the day boundary; an exact as_of requires an explicit
    used timestamp strictly before the boundary, and rejects a same-day date-only
    cutoff because it cannot prove time order. The full-input history cutoff is
    never compared here, and history files are never opened.
    """

    rows_used = history.get("rows_used_before_as_of")
    if isinstance(rows_used, bool) or not isinstance(rows_used, int):
        raise DataValidationError("snapshot history rows_used_before_as_of must be an integer")
    used_raw = history.get("used_data_cutoff")
    if rows_used > 0 and used_raw is None:
        raise DataValidationError(
            "snapshot history records rows_used_before_as_of > 0 but no used_data_cutoff; "
            "missing lineage must not be treated as an empty training source"
        )
    if rows_used == 0:
        if used_raw is not None:
            raise DataValidationError(
                "snapshot history records no training rows before as_of but a non-null used_data_cutoff; "
                "recorded lineage is inconsistent"
            )
        return None
    text = str(used_raw).strip()
    if not text:
        raise DataValidationError("snapshot used_data_cutoff is recorded but empty; lineage is inconsistent")
    try:
        used_timestamp = pd.Timestamp(text)
    except (TypeError, ValueError) as exc:
        raise DataValidationError(
            f"snapshot used_data_cutoff is not a parseable timestamp: {used_raw!r}"
        ) from exc
    if pd.isna(used_timestamp):
        raise DataValidationError(
            f"snapshot used_data_cutoff is not a valid finite timestamp: {used_raw!r}"
        )
    if has_explicit_time(text):
        boundary = as_of.normalize() if date_only else as_of
        if used_timestamp >= boundary:
            raise DataValidationError(
                "snapshot used_data_cutoff timestamp is not strictly before the as_of boundary; "
                f"recorded lineage would let the model state use matches at or after as_of: {text}"
            )
    else:
        if used_timestamp >= as_of.normalize():
            raise DataValidationError(
                "snapshot used_data_cutoff is a date-only value on or after the as_of day; "
                "it cannot prove that the last used match is strictly before the as_of boundary: "
                f"{text}"
            )
    return used_timestamp


def _m11_evidence(run_dir: Path, config: Mapping[str, Any], config_path: Path) -> dict[str, Any]:
    """Strictly validate one single-model M11 run against its recorded snapshot."""

    predictions_path = run_dir / "predictions.csv"
    if not predictions_path.is_file():
        raise FileNotFoundError(f"M11 predictions.csv does not exist: {predictions_path}")
    config_hash = file_sha256(config_path)
    prediction_hash = file_sha256(predictions_path)

    model_names = config.get("model_names")
    if (
        not isinstance(model_names, list)
        or not model_names
        or any(not isinstance(name, str) for name in model_names)
    ):
        raise DataValidationError("M11 run config must record model_names as a list of model names")
    if len(model_names) != 1:
        raise DataValidationError(
            "multi-model M11 prediction runs are not supported for settlement; "
            f"found models: {', '.join(model_names)}"
        )
    model_name = model_names[0]
    if model_name not in MODEL_NAMES:
        raise DataValidationError(f"M11 run config records an unsupported model name: {model_name}")
    models = config.get("models")
    if not isinstance(models, list) or len(models) != 1 or not isinstance(models[0], Mapping):
        raise DataValidationError(
            "M11 run config must record exactly one model evidence object for settlement"
        )
    run_model = models[0]
    if str(run_model.get("model")) != model_name:
        raise DataValidationError("M11 run config model evidence does not match model_names")

    snapshot_evidence = config.get("snapshot")
    if not isinstance(snapshot_evidence, Mapping):
        raise DataValidationError("M11 run config must record a snapshot evidence object")
    for field in (
        "snapshot_id",
        "manifest_path",
        "manifest_sha256",
        "manifest_content_sha256",
        "state_path",
        "state_sha256",
    ):
        if not snapshot_evidence.get(field):
            raise DataValidationError(f"M11 run config snapshot evidence is missing field: {field}")
    manifest_path = Path(str(snapshot_evidence["manifest_path"])).expanduser().resolve()
    if not manifest_path.is_file():
        raise FileNotFoundError(f"M11 snapshot manifest does not exist: {manifest_path}")

    loaded = load_engine_snapshot(manifest_path)
    manifest = loaded["manifest"]
    if Path(loaded["manifest_path"]) != manifest_path:
        raise DataValidationError("M11 run config snapshot manifest path is inconsistent")
    if snapshot_evidence["snapshot_id"] != manifest["snapshot_id"]:
        raise DataValidationError("M11 run config snapshot_id does not match the snapshot manifest")
    actual_manifest_hash = file_sha256(manifest_path)
    if actual_manifest_hash != _require_valid_sha256(
        snapshot_evidence["manifest_sha256"], "M11 snapshot manifest SHA-256 evidence is not a 64-character SHA-256"
    ):
        raise DataValidationError(
            "M11 recorded snapshot manifest SHA-256 does not match the actual manifest file; "
            "the snapshot appears modified or tampered"
        )
    if str(snapshot_evidence["manifest_content_sha256"]) != str(manifest["manifest_sha256"]):
        raise DataValidationError(
            "M11 recorded snapshot manifest content hash does not match the manifest contents"
        )
    state_path = Path(str(snapshot_evidence["state_path"])).expanduser().resolve()
    if Path(loaded["state_path"]) != state_path:
        raise DataValidationError("M11 run config snapshot state path is inconsistent")
    actual_state_hash = file_sha256(state_path)
    if actual_state_hash != _require_valid_sha256(
        snapshot_evidence["state_sha256"], "M11 snapshot state SHA-256 evidence is not a 64-character SHA-256"
    ):
        raise DataValidationError(
            "M11 recorded snapshot state SHA-256 does not match the actual state file; "
            "the snapshot appears modified or tampered"
        )

    manifest_model_names = manifest["model_names"]
    if len(manifest_model_names) != 1 or str(manifest_model_names[0]) != model_name:
        raise DataValidationError(
            "single-model M11 runs must be produced from a single-model snapshot; "
            f"the source snapshot contains models: {', '.join(manifest_model_names)}"
        )
    if len(manifest["models"]) != 1:
        raise DataValidationError(
            "single-model M11 runs must match a single-model snapshot manifest; "
            f"the source snapshot manifest records {len(manifest['models'])} models"
        )
    if len(loaded["models"]) != 1:
        raise DataValidationError(
            "single-model M11 runs must match a single-model snapshot state; "
            f"the snapshot state contains {len(loaded['models'])} model states"
        )
    snapshot_model = manifest["models"][0]
    if str(snapshot_model.get("model")) != model_name:
        raise DataValidationError("M11 snapshot model does not match the run model")
    if run_model.get("config") != snapshot_model.get("config"):
        raise DataValidationError("M11 run model config does not match the snapshot manifest model config")
    if str(run_model.get("config_sha256")) != str(snapshot_model.get("config_sha256")):
        raise DataValidationError(
            "M11 run model config hash does not match the snapshot manifest model config hash"
        )
    recomputed_config_hash = configuration_hash(run_model.get("config"))
    if recomputed_config_hash != str(run_model.get("config_sha256")):
        raise DataValidationError("M11 run model config_sha256 does not match the recorded model config contents")
    if str(run_model.get("state_sha256")) != str(snapshot_model.get("state_sha256")):
        raise DataValidationError(
            "M11 run per-model state hash does not match the snapshot manifest per-model state hash"
        )

    as_of, date_only, as_of_meta = _m11_as_of_evidence(config, snapshot_as_of=manifest["as_of"])

    history = manifest["history"]
    for field in ("history_path", "history_sha256", "history_cutoff"):
        if field not in snapshot_evidence:
            raise DataValidationError(f"M11 run config snapshot evidence is missing field: {field}")
    for field in ("history_path", "history_sha256"):
        if snapshot_evidence[field] is None or not str(snapshot_evidence[field]).strip():
            raise DataValidationError(f"M11 run config snapshot evidence field {field} must be recorded and non-empty")
    recorded_history_path = Path(str(snapshot_evidence["history_path"])).expanduser().resolve()
    snapshot_history_path = Path(str(history["path"])).expanduser().resolve()
    if recorded_history_path != snapshot_history_path:
        raise DataValidationError(
            "M11 run config recorded history_path does not match the snapshot manifest history path"
        )
    if str(snapshot_evidence["history_sha256"]).lower() != str(history["sha256"]).lower():
        raise DataValidationError(
            "M11 run config recorded history_sha256 does not match the snapshot manifest history hash"
        )
    used_data_cutoff = history.get("used_data_cutoff")
    recorded_history_cutoff = snapshot_evidence["history_cutoff"]
    if (str(recorded_history_cutoff) if recorded_history_cutoff is not None else None) != (
        str(used_data_cutoff) if used_data_cutoff is not None else None
    ):
        raise DataValidationError(
            "M11 run config recorded history_cutoff does not match the snapshot manifest used_data_cutoff"
        )
    _validate_used_data_cutoff(history, as_of=as_of, date_only=date_only)

    recorded_hashes: list[str] = []
    for value in (config.get("predictions_sha256"), (config.get("outputs") or {}).get("predictions_sha256") if isinstance(config.get("outputs"), Mapping) else None):
        if value is None:
            continue
        recorded_hashes.append(
            _require_valid_sha256(value, "M11 predictions SHA-256 evidence is not a 64-character SHA-256")
        )
    if not recorded_hashes:
        raise DataValidationError("M11 run config must record a predictions SHA-256")
    if len(set(recorded_hashes)) != 1:
        raise DataValidationError("M11 predictions SHA-256 evidence fields disagree")
    if recorded_hashes[0] != prediction_hash.lower():
        raise DataValidationError(
            "M11 predictions SHA-256 does not match predictions.csv; input appears modified or tampered"
        )
    outputs = config.get("outputs")
    if isinstance(outputs, Mapping) and outputs.get("predictions") is not None:
        if Path(str(outputs["predictions"])).expanduser().resolve() != predictions_path:
            raise DataValidationError("M11 run config outputs.predictions path does not match the run directory")

    predictions = _read_prediction_csv(predictions_path, producer_label="M11")
    if predictions["model"].astype(str).nunique() != 1 or str(predictions["model"].iloc[0]) != model_name:
        raise DataValidationError("predictions model column does not match the M11 run model")
    _validate_prediction_dates(predictions, as_of, date_only)

    fixtures_evidence = config.get("fixtures")
    if not isinstance(fixtures_evidence, Mapping):
        raise DataValidationError("M11 run config must record a fixtures evidence object")
    for field in ("path", "sha256", "rows"):
        if fixtures_evidence.get(field) is None:
            raise DataValidationError(f"M11 run config fixtures evidence is missing field: {field}")
    fixtures_path = Path(str(fixtures_evidence["path"])).expanduser().resolve()
    if not fixtures_path.is_file():
        raise FileNotFoundError(f"M11 recorded source fixtures CSV does not exist: {fixtures_path}")
    actual_fixtures_hash = file_sha256(fixtures_path)
    if actual_fixtures_hash != _require_valid_sha256(
        fixtures_evidence["sha256"], "M11 fixtures SHA-256 evidence is not a 64-character SHA-256"
    ):
        raise DataValidationError(
            "M11 recorded fixtures SHA-256 does not match the actual fixtures file; "
            "the input appears modified or tampered"
        )
    fixtures = load_fixtures(fixtures_path)
    if int(len(fixtures)) != int(fixtures_evidence["rows"]):
        raise DataValidationError("M11 recorded fixtures row count does not match the actual fixtures file")
    _validate_prediction_dates(fixtures, as_of, date_only, table_label="recorded source fixtures")
    if int(len(predictions)) != int(len(fixtures)):
        raise DataValidationError(
            "M11 predictions and recorded source fixtures do not cover the same fixture set"
        )
    merged = fixtures.merge(
        predictions.loc[:, ["fixture_id", "date", "season", "home_team", "away_team"]],
        on="fixture_id",
        how="outer",
        indicator=True,
        suffixes=("_fixture", "_prediction"),
    )
    unmatched = merged["_merge"] != "both"
    if unmatched.any():
        raise DataValidationError(
            "M11 predictions and recorded source fixtures have unmatched fixture_id values: "
            f"{', '.join(sorted(merged.loc[unmatched, 'fixture_id'].astype(str))[:5])}"
        )
    for column in ("date", "season", "home_team", "away_team"):
        mismatch = ~merged[f"{column}_fixture"].astype(str).eq(merged[f"{column}_prediction"].astype(str))
        if mismatch.any():
            raise DataValidationError(
                f"M11 prediction column '{column}' does not match the recorded source fixture at input rows "
                + ", ".join(str(index + 2) for index in mismatch.index[mismatch][:5])
            )

    origin_evidence = {
        "kind": "m11_engine_snapshot",
        "snapshot_dir": str(manifest_path.parent),
        "snapshot_id": manifest["snapshot_id"],
        "snapshot_manifest_path": str(manifest_path),
        "snapshot_manifest_sha256": actual_manifest_hash,
        "snapshot_manifest_content_sha256": manifest["manifest_sha256"],
        "state_path": str(state_path),
        "state_sha256": actual_state_hash,
        "history": history,
        "history_manifest": manifest["history_manifest"],
        "history_evidence_status": "recorded_in_snapshot_manifest_not_reread_by_settlement",
        "season_mode": manifest["season_mode"],
        "used_data_cutoff": used_data_cutoff,
        "fixtures": {
            "path": str(fixtures_path),
            "sha256": actual_fixtures_hash,
            "rows": int(len(fixtures)),
            "schema": fixtures_evidence.get("schema"),
            "hash_evidence_status": "verified",
        },
        "as_of_consistency": "verified_run_config_vs_snapshot_manifest",
    }
    limitations = [
        "M11 settlement does not read the history CSV or the history processed manifest; "
        "snapshot lineage is validated from recorded fields only.",
    ]
    return {
        "producer_schema": SNAPSHOT_PREDICT_SCHEMA_VERSION,
        "producer_command": config.get("command"),
        "run_dir": run_dir,
        "run_config_path": config_path,
        "run_config_sha256": config_hash,
        "predictions_path": predictions_path,
        "predictions_sha256": prediction_hash,
        "prediction_hash_status": "verified",
        "predictions": predictions,
        "model_name": model_name,
        "model_config": run_model.get("config"),
        "model_config_sha256": str(run_model.get("config_sha256")),
        "as_of": as_of,
        "as_of_date_only": date_only,
        "as_of_meta": as_of_meta,
        "origin_evidence": origin_evidence,
        "limitations": limitations,
    }


__all__ = [
    "PREDICTION_EVIDENCE_SCHEMAS",
    "load_prediction_evidence",
]
