"""Input-evidence and selection helpers for M11 snapshot builds."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd

from .forecast import load_selection_run_config
from .io import file_sha256
from .models import MODEL_NAMES, create_model
from .schema import DataValidationError, REQUIRED_COLUMNS, has_explicit_time

_HISTORY_MANIFEST_SCHEMAS = {"m2-processed-manifest-v1", "m2.1-processed-manifest-v1"}


def _read_json(path: Path, label: str) -> dict[str, Any]:
    import json

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DataValidationError(f"could not read {label} '{path}': {exc}") from exc
    if not isinstance(payload, dict):
        raise DataValidationError(f"{label} root must be a JSON object: {path}")
    return payload


def _same_path(left: Path, right: Path) -> bool:
    return str(left).casefold() == str(right).casefold()


def _recorded_path(value: Any, base: Path, label: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise DataValidationError(f"{label} path is missing")
    candidate = Path(value).expanduser()
    return (candidate if candidate.is_absolute() else base / candidate).resolve()


def history_manifest_evidence(
    history_path: Path, manifest_path: Path, history: pd.DataFrame
) -> tuple[str, dict[str, Any], str]:
    """Validate the explicit M2/M2.1 history evidence used for fitting."""

    if not history_path.is_file():
        raise FileNotFoundError(f"history CSV does not exist: {history_path}")
    if not manifest_path.is_file():
        raise FileNotFoundError(f"history manifest does not exist: {manifest_path}")
    history_hash = file_sha256(history_path)
    manifest = _read_json(manifest_path, "history manifest")
    if manifest.get("schema_version") not in _HISTORY_MANIFEST_SCHEMAS:
        raise DataValidationError("history manifest has an unsupported schema_version")
    output = manifest.get("output")
    if not isinstance(output, Mapping):
        raise DataValidationError("history manifest output evidence is missing")
    output_path = _recorded_path(output.get("filename"), manifest_path.parent, "history manifest output")
    if not _same_path(output_path, history_path):
        raise DataValidationError("history manifest output filename does not match the explicit history path")
    recorded_hash = output.get("sha256")
    if not isinstance(recorded_hash, str) or recorded_hash.casefold() != history_hash.casefold():
        raise DataValidationError("history manifest output SHA-256 does not match the history CSV")
    rows = output.get("rows")
    if isinstance(rows, bool) or not isinstance(rows, int) or rows != len(history):
        raise DataValidationError("history manifest output row count does not match the history CSV")
    if output.get("columns") != list(REQUIRED_COLUMNS):
        raise DataValidationError("history manifest output columns do not match the M1 canonical schema")
    data_cutoff = manifest.get("data_cutoff")
    if not isinstance(data_cutoff, str) or not data_cutoff.strip():
        raise DataValidationError("history manifest data_cutoff evidence is missing")
    try:
        declared_cutoff = pd.Timestamp(pd.to_datetime(data_cutoff, errors="raise", format="mixed"))
        observed_cutoff = history["_timestamp"].max()
    except (TypeError, ValueError, KeyError) as exc:
        raise DataValidationError("history manifest data_cutoff is invalid") from exc
    if has_explicit_time(data_cutoff):
        cutoff_matches = declared_cutoff == observed_cutoff
    else:
        cutoff_matches = declared_cutoff.normalize() == observed_cutoff.normalize()
    if not cutoff_matches:
        raise DataValidationError(
            f"history manifest data_cutoff={data_cutoff!r} does not match observed history cutoff"
        )
    cutoff_index = history["_timestamp"].idxmax()
    return history_hash, manifest, str(history.loc[cutoff_index, "date"])


def normalise_names(model_names: Sequence[str] | None) -> list[str]:
    if model_names is None:
        names = list(MODEL_NAMES)
    elif isinstance(model_names, str):
        names = [model_names]
    else:
        names = [str(name).strip() for name in model_names]
    if not names or any(not name for name in names):
        raise ValueError("at least one model is required")
    if len(set(names)) != len(names):
        raise ValueError("model names must be unique")
    unknown = [name for name in names if name not in MODEL_NAMES]
    if unknown:
        raise ValueError(f"unknown model(s): {unknown}; choose from {MODEL_NAMES}")
    return names


def normalise_configs(
    names: Sequence[str], model_configs: Mapping[str, Mapping[str, Any]] | None
) -> dict[str, dict[str, Any]]:
    if model_configs is None:
        supplied: Mapping[str, Mapping[str, Any]] = {}
    elif not isinstance(model_configs, Mapping):
        raise TypeError("model_configs must be a mapping of model name to JSON object")
    else:
        supplied = model_configs
    unknown = [str(name) for name in supplied if str(name) not in names]
    if unknown:
        raise ValueError(f"model_configs contains models not selected: {unknown}")
    configs: dict[str, dict[str, Any]] = {}
    for name in names:
        raw = supplied.get(name, {})
        if not isinstance(raw, Mapping):
            raise TypeError(f"model config for {name} must be a JSON object")
        config = dict(raw)
        try:
            create_model(name, config)
        except (TypeError, ValueError) as exc:
            raise DataValidationError(f"model config is invalid for {name}: {exc}") from exc
        configs[name] = config
    return configs


def selection_evidence(
    history_hash: str,
    as_of: str | pd.Timestamp,
    model_names: Sequence[str] | None,
    model_configs: Mapping[str, Mapping[str, Any]] | None,
    selection_path: str | Path | None,
) -> tuple[list[str], dict[str, dict[str, Any]], dict[str, Any], str]:
    if selection_path is not None:
        if model_names is not None or model_configs:
            raise ValueError("selection run config and explicit model/configs are mutually exclusive")
        model, config, selection = load_selection_run_config(selection_path, history_hash, as_of)
        return [model], {model: config}, selection, str(selection.get("selection_season_mode") or "carry")
    names = normalise_names(model_names)
    configs = normalise_configs(names, model_configs)
    return names, configs, {
        "mode": "explicit_model_config",
        "path": None,
        "sha256": None,
        "schema_version": None,
        "evidence_status": "explicit_model_config",
        "selection_history_sha256": None,
        "current_history_sha256": history_hash,
        "history_hash_status": "not_applicable",
        "selection_season_mode": None,
    }, "carry"


def config_sources(
    names: Sequence[str], model_config_paths: Mapping[str, str | Path] | None
) -> dict[str, dict[str, Any]]:
    if model_config_paths is None:
        return {name: {"mode": "api", "path": None, "sha256": None} for name in names}
    if not isinstance(model_config_paths, Mapping):
        raise TypeError("model_config_paths must be a mapping")
    unknown = [str(name) for name in model_config_paths if str(name) not in names]
    if unknown:
        raise ValueError(f"model_config_paths contains models not selected: {unknown}")
    sources: dict[str, dict[str, Any]] = {}
    for name in names:
        value = model_config_paths.get(name)
        if value is None:
            sources[name] = {"mode": "api", "path": None, "sha256": None}
            continue
        path = Path(value).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"model config does not exist: {path}")
        sources[name] = {"mode": "explicit_file", "path": str(path), "sha256": file_sha256(path)}
    return sources


def reject_output_root(root: Path, *inputs: Path) -> None:
    for input_path in inputs:
        resolved_input = input_path.expanduser().resolve()
        if root == resolved_input or root == resolved_input.parent or resolved_input in root.parents:
            raise DataValidationError(f"output root cannot be an input directory: {root}")


__all__ = [
    "config_sources",
    "history_manifest_evidence",
    "normalise_configs",
    "normalise_names",
    "reject_output_root",
    "selection_evidence",
]
