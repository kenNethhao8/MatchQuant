"""Strict manifest and filesystem validation for M11 engine snapshots."""

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping
from uuid import uuid4

import pandas as pd

from .backtest import _parse_boundary
from .engine_snapshot_support import (
    ENGINE_SNAPSHOT_SCHEMA_VERSION,
    ENGINE_STATE_SCHEMA_VERSION,
    canonical_json_bytes,
    load_model_states,
)
from .io import file_sha256
from .models import MODEL_NAMES
from .schema import DataValidationError

_HISTORY_MANIFEST_SCHEMAS = {"m2-processed-manifest-v1", "m2.1-processed-manifest-v1"}
_SNAPSHOT_FIELDS = (
    "schema_version",
    "snapshot_id",
    "generated_at",
    "as_of",
    "history",
    "history_manifest",
    "model_names",
    "models",
    "state",
    "outputs",
    "selection",
    "season_mode",
    "validation",
    "disclaimer",
    "manifest_sha256",
)


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DataValidationError(f"could not read {label} '{path}': {exc}") from exc
    if not isinstance(payload, dict):
        raise DataValidationError(f"{label} root must be a JSON object: {path}")
    return payload


def _valid_sha256(value: Any) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-fA-F]{64}", value) is not None


def _payload_hash(payload: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def _recorded_path(value: Any, base: Path, label: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise DataValidationError(f"{label} path is missing")
    candidate = Path(value).expanduser()
    return (candidate if candidate.is_absolute() else base / candidate).resolve()


def _require_keys(payload: Mapping[str, Any], keys: tuple[str, ...], label: str) -> None:
    missing = [key for key in keys if key not in payload]
    if missing:
        raise DataValidationError(f"{label} is missing required fields: {', '.join(missing)}")


def _require_exact_keys(payload: Mapping[str, Any], keys: tuple[str, ...], label: str) -> None:
    _require_keys(payload, keys, label)
    extras = sorted(set(payload) - set(keys))
    if extras:
        raise DataValidationError(f"{label} has unknown fields: {', '.join(extras)}")


def _validate_generated_at(payload: Mapping[str, Any]) -> None:
    generated_at = payload.get("generated_at")
    if not isinstance(generated_at, str) or not generated_at.endswith("Z"):
        raise DataValidationError("snapshot generated_at must be a UTC ISO-8601 string")
    try:
        parsed = datetime.fromisoformat(generated_at[:-1] + "+00:00")
    except ValueError as exc:
        raise DataValidationError("snapshot generated_at is invalid") from exc
    if parsed.tzinfo is None or parsed.utcoffset() != timezone.utc.utcoffset(parsed):
        raise DataValidationError("snapshot generated_at must be UTC")


def _validate_as_of(payload: Mapping[str, Any]) -> None:
    raw = payload.get("as_of")
    if not isinstance(raw, Mapping):
        raise DataValidationError("snapshot as_of metadata must be an object")
    _require_keys(raw, ("value", "timestamp", "date_only"), "snapshot as_of metadata")
    if not isinstance(raw["date_only"], bool):
        raise DataValidationError("snapshot as_of date_only must be boolean")
    try:
        timestamp, date_only = _parse_boundary(raw["value"], "snapshot as_of")
        declared = pd.Timestamp(pd.to_datetime(raw["timestamp"], errors="raise", format="mixed"))
    except (TypeError, ValueError) as exc:
        raise DataValidationError("snapshot as_of metadata is invalid") from exc
    if bool(raw["date_only"]) != date_only or declared != timestamp:
        raise DataValidationError("snapshot as_of metadata is inconsistent")


def new_engine_snapshot_dir(output_root: str | Path) -> Path:
    root = Path(output_root).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    snapshot_dir = root / f"m11_engine_snapshot_{stamp}_{uuid4().hex[:8]}"
    snapshot_dir.mkdir()
    return snapshot_dir


def manifest_content_sha256(payload: Mapping[str, Any]) -> str:
    body = dict(payload)
    body.pop("manifest_sha256", None)
    return _payload_hash(body)


def write_snapshot_manifest(payload: Mapping[str, Any], path: str | Path) -> Path:
    manifest = dict(payload)
    manifest["manifest_sha256"] = manifest_content_sha256(manifest)
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(canonical_json_bytes(manifest))
    return output_path


def load_engine_snapshot(snapshot_manifest_path: str | Path) -> dict[str, Any]:
    """Load only immutable snapshot evidence and model state; never read history."""

    manifest_path = Path(snapshot_manifest_path).expanduser().resolve()
    if not manifest_path.is_file():
        raise FileNotFoundError(f"snapshot manifest does not exist: {manifest_path}")
    manifest = _read_json(manifest_path, "snapshot manifest")
    _require_exact_keys(manifest, _SNAPSHOT_FIELDS, "snapshot manifest")
    if manifest.get("schema_version") != ENGINE_SNAPSHOT_SCHEMA_VERSION:
        raise DataValidationError(f"snapshot manifest schema_version must be {ENGINE_SNAPSHOT_SCHEMA_VERSION}")
    if manifest.get("manifest_sha256") != manifest_content_sha256(manifest):
        raise DataValidationError("snapshot manifest hash does not match manifest contents")
    snapshot_id = manifest.get("snapshot_id")
    if not isinstance(snapshot_id, str) or not snapshot_id.startswith("m11_engine_snapshot_") or snapshot_id != manifest_path.parent.name:
        raise DataValidationError("snapshot_id does not match the manifest directory")
    if manifest_path.name != "snapshot_manifest.json":
        raise DataValidationError("snapshot manifest must be named snapshot_manifest.json")
    _validate_generated_at(manifest)
    _validate_as_of(manifest)

    history = manifest.get("history")
    history_manifest = manifest.get("history_manifest")
    for evidence, label in ((history, "history"), (history_manifest, "history manifest")):
        if not isinstance(evidence, Mapping):
            raise DataValidationError(f"snapshot {label} evidence must be an object")
        if not isinstance(evidence.get("path"), str) or not evidence["path"].strip() or not _valid_sha256(evidence.get("sha256")):
            raise DataValidationError(f"snapshot {label} evidence must contain a valid path and SHA-256")
    if not isinstance(history, Mapping) or not isinstance(history_manifest, Mapping):
        raise DataValidationError("snapshot history evidence is missing")
    _require_keys(
        history,
        (
            "path",
            "sha256",
            "rows",
            "rows_used_before_as_of",
            "rows_excluded_at_or_after_as_of",
            "observed_cutoff",
            "data_cutoff",
            "used_data_cutoff",
        ),
        "snapshot history",
    )
    for field in ("rows", "rows_used_before_as_of", "rows_excluded_at_or_after_as_of"):
        value = history[field]
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise DataValidationError(f"snapshot history {field} must be a non-negative integer")
    if history["rows_used_before_as_of"] + history["rows_excluded_at_or_after_as_of"] != history["rows"]:
        raise DataValidationError("snapshot history row counts are inconsistent")
    for field in ("observed_cutoff", "data_cutoff"):
        if not isinstance(history[field], str) or not history[field].strip():
            raise DataValidationError(f"snapshot history {field} evidence is missing")
    used_cutoff = history["used_data_cutoff"]
    if used_cutoff is not None and (not isinstance(used_cutoff, str) or not used_cutoff.strip()):
        raise DataValidationError("snapshot history used_data_cutoff evidence is invalid")
    _require_exact_keys(
        history_manifest,
        ("path", "sha256", "schema_version", "dataset_version", "data_cutoff", "output_sha256"),
        "snapshot history manifest",
    )
    if history_manifest["schema_version"] not in _HISTORY_MANIFEST_SCHEMAS:
        raise DataValidationError("snapshot history manifest schema_version is unsupported")
    if history_manifest["data_cutoff"] != history["data_cutoff"]:
        raise DataValidationError("snapshot history manifest data_cutoff does not match history evidence")
    if not _valid_sha256(history_manifest.get("output_sha256")):
        raise DataValidationError("snapshot history manifest output hash is missing or invalid")
    if history_manifest["output_sha256"].casefold() != history["sha256"].casefold():
        raise DataValidationError("snapshot history manifest output hash does not match history evidence")

    if not isinstance(manifest.get("selection"), Mapping):
        raise DataValidationError("snapshot selection evidence must be an object")
    if manifest.get("season_mode") not in {"carry", "reset"}:
        raise DataValidationError("snapshot season_mode must be 'carry' or 'reset'")
    if not isinstance(manifest.get("disclaimer"), str) or not manifest["disclaimer"].strip():
        raise DataValidationError("snapshot disclaimer is missing")

    state_evidence = manifest.get("state")
    if not isinstance(state_evidence, Mapping):
        raise DataValidationError("snapshot state evidence is missing")
    _require_exact_keys(state_evidence, ("path", "sha256", "schema_version"), "snapshot state")
    if state_evidence.get("schema_version") != ENGINE_STATE_SCHEMA_VERSION:
        raise DataValidationError("snapshot state schema_version is unsupported")
    state_path = _recorded_path(state_evidence.get("path"), manifest_path.parent, "snapshot state")
    if state_path.parent != manifest_path.parent or state_path.name != "model_states.json":
        raise DataValidationError("snapshot state file must be model_states.json in the snapshot directory")
    if not state_path.is_file():
        raise FileNotFoundError(f"snapshot state file does not exist: {state_path}")
    expected_state_hash = state_evidence.get("sha256")
    if not _valid_sha256(expected_state_hash):
        raise DataValidationError("snapshot state SHA-256 is missing or invalid")
    actual_state_hash = file_sha256(state_path)
    if actual_state_hash.lower() != str(expected_state_hash).lower():
        raise DataValidationError("snapshot state SHA-256 does not match the recorded hash")
    state_payload = load_model_states(state_path)
    records = state_payload["models"]

    manifest_names = manifest.get("model_names")
    if (
        not isinstance(manifest_names, list)
        or any(not isinstance(name, str) for name in manifest_names)
        or len(set(manifest_names)) != len(manifest_names)
        or any(name not in MODEL_NAMES for name in manifest_names)
        or set(manifest_names) != {str(item["model"]) for item in records}
    ):
        raise DataValidationError("snapshot manifest model_names do not match model state records")
    manifest_models = manifest.get("models")
    if not isinstance(manifest_models, list):
        raise DataValidationError("snapshot manifest field 'models' must be a list")
    expected = {
        str(item.get("model")): item
        for item in manifest_models
        if isinstance(item, Mapping) and item.get("model") is not None
    }
    if len(expected) != len(manifest_models) or set(expected) != {str(item["model"]) for item in records}:
        raise DataValidationError("snapshot manifest models do not match model state records")
    for record in records:
        evidence = expected[str(record["model"])]
        _require_exact_keys(
            evidence,
            ("model", "config", "config_sha256", "state_sha256", "config_source", "state_codec"),
            f"snapshot manifest model {record['model']}",
        )
        if not isinstance(evidence["config_source"], Mapping):
            raise DataValidationError(f"snapshot config source for {record['model']} must be an object")
        _require_exact_keys(
            evidence["config_source"],
            ("mode", "path", "sha256"),
            f"snapshot config source for {record['model']}",
        )
        if (
            evidence.get("config") != record.get("config")
            or evidence.get("config_sha256") != record.get("config_sha256")
            or evidence.get("state_sha256") != record.get("state_sha256")
            or evidence.get("state_codec") != ENGINE_STATE_SCHEMA_VERSION
        ):
            raise DataValidationError(f"snapshot manifest evidence for {record['model']} does not match state")

    declarations = manifest.get("validation")
    if not isinstance(declarations, Mapping):
        raise DataValidationError("snapshot validation declarations are missing")
    required = (
        "fixture_results_used_for_model_updates",
        "history_copied",
        "input_files_unchanged",
        "network_accessed",
        "snapshot_is_immutable",
    )
    if any(key not in declarations for key in required):
        raise DataValidationError("snapshot validation declarations are incomplete")
    if (
        declarations["network_accessed"] is not False
        or declarations["history_copied"] is not False
        or declarations["fixture_results_used_for_model_updates"] is not False
        or declarations["input_files_unchanged"] is not True
        or declarations["snapshot_is_immutable"] is not True
    ):
        raise DataValidationError("snapshot violates the offline/no-history-copy contract")
    return {
        "manifest": manifest,
        "manifest_path": str(manifest_path),
        "manifest_sha256": file_sha256(manifest_path),
        "state_payload": state_payload,
        "state_path": str(state_path),
        "state_sha256": actual_state_hash,
        "models": records,
    }


__all__ = [
    "load_engine_snapshot",
    "manifest_content_sha256",
    "new_engine_snapshot_dir",
    "write_snapshot_manifest",
]
