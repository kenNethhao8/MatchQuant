"""Deterministic JSON codecs and integrity checks for M11 engine snapshots."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .models import (
    MODEL_NAMES,
    RESULTS,
    EloModel,
    FrequencyModel,
    PoissonModel,
    _TeamStats,
    create_model,
)
from .schema import DataValidationError

ENGINE_SNAPSHOT_SCHEMA_VERSION = "m11-engine-snapshot-v1"
ENGINE_STATE_SCHEMA_VERSION = "m11-engine-state-v1"

_MODEL_PARAMETERS = {
    "frequency": ("alpha",),
    "elo": (
        "initial_rating",
        "k_factor",
        "home_advantage",
        "rating_scale",
        "draw_base",
        "draw_scale",
    ),
    "poisson": (
        "prior_home_goals",
        "prior_away_goals",
        "prior_weight",
        "max_goals",
        "tail_tolerance",
    ),
}


def _json_value(value: Any, label: str = "value") -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise DataValidationError(f"{label} must contain only finite JSON numbers")
        return value
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise DataValidationError(f"{label} contains a non-string JSON key")
            result[key] = _json_value(item, f"{label}.{key}")
        return result
    if isinstance(value, (list, tuple)):
        return [_json_value(item, f"{label}[{index}]") for index, item in enumerate(value)]
    raise DataValidationError(f"{label} contains unsupported value type {type(value).__name__}")


def canonical_json_bytes(payload: Any) -> bytes:
    """Serialize a JSON-compatible payload deterministically, including a newline."""

    normalized = _json_value(payload)
    text = json.dumps(
        normalized,
        ensure_ascii=False,
        sort_keys=True,
        indent=2,
        allow_nan=False,
    )
    return (text + "\n").encode("utf-8")


def configuration_hash(config: Mapping[str, Any]) -> str:
    if not isinstance(config, Mapping):
        raise DataValidationError("model config must be a JSON object")
    normalized = _json_value(dict(config), "model config")
    compact = json.dumps(
        normalized,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(compact).hexdigest()


def _payload_hash(payload: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DataValidationError(f"could not read {label} '{path}': {exc}") from exc
    if not isinstance(payload, dict):
        raise DataValidationError(f"{label} root must be a JSON object: {path}")
    return payload


def _finite_number(value: Any, label: str) -> float:
    if isinstance(value, bool):
        raise DataValidationError(f"{label} must be a finite number")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise DataValidationError(f"{label} must be a finite number") from exc
    if not math.isfinite(number):
        raise DataValidationError(f"{label} must be a finite number")
    return number


def _non_negative_number(value: Any, label: str) -> float:
    number = _finite_number(value, label)
    if number < 0:
        raise DataValidationError(f"{label} must be non-negative")
    return number


def _non_negative_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise DataValidationError(f"{label} must be a non-negative integer")
    return int(value)


def _effective_config(model: Any) -> dict[str, Any]:
    name = str(getattr(model, "name", ""))
    if name not in _MODEL_PARAMETERS:
        raise DataValidationError(f"cannot snapshot unknown model '{name}'")
    return {field: getattr(model, field) for field in _MODEL_PARAMETERS[name]}


def _team_name(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise DataValidationError(f"{label} must be a non-empty team name")
    return value


def _model_state(model: Any) -> dict[str, Any]:
    name = str(getattr(model, "name", ""))
    if isinstance(model, FrequencyModel) and name == "frequency":
        values = np.asarray(model.counts, dtype=float)
        if values.shape != (3,):
            raise DataValidationError("frequency counts must contain exactly three values")
        return {
            "alpha": _finite_number(model.alpha, "frequency state alpha"),
            "counts": {
                result: _finite_number(values[index], f"frequency state counts.{result}")
                for index, result in enumerate(RESULTS)
            },
        }
    if isinstance(model, EloModel) and name == "elo":
        ratings = {
            _team_name(team, "elo state team"): _finite_number(rating, f"elo state rating {team!r}")
            for team, rating in model.ratings.items()
        }
        return {
            field: _finite_number(getattr(model, field), f"elo state {field}")
            for field in _MODEL_PARAMETERS["elo"]
        } | {"ratings": dict(sorted(ratings.items()))}
    if isinstance(model, PoissonModel) and name == "poisson":
        teams: dict[str, Any] = {}
        for team, stats in model.teams.items():
            team_name = _team_name(team, "poisson state team")
            matches = _non_negative_int(stats.matches, f"poisson state matches for {team_name!r}")
            teams[team_name] = {
                "goals_for": _non_negative_number(stats.goals_for, f"poisson state {team_name}.goals_for"),
                "goals_against": _non_negative_number(stats.goals_against, f"poisson state {team_name}.goals_against"),
                "matches": matches,
            }
        total_matches = _non_negative_int(model.matches, "poisson state matches")
        return {
            "prior_home_goals": _finite_number(model.prior_home_goals, "poisson state prior_home_goals"),
            "prior_away_goals": _finite_number(model.prior_away_goals, "poisson state prior_away_goals"),
            "prior_weight": _finite_number(model.prior_weight, "poisson state prior_weight"),
            "max_goals": _non_negative_int(model.max_goals, "poisson state max_goals"),
            "tail_tolerance": _finite_number(model.tail_tolerance, "poisson state tail_tolerance"),
            "total_home_goals": _non_negative_number(model.total_home_goals, "poisson state total_home_goals"),
            "total_away_goals": _non_negative_number(model.total_away_goals, "poisson state total_away_goals"),
            "matches": int(total_matches),
            "teams": dict(sorted(teams.items())),
        }
    raise DataValidationError(f"cannot snapshot unsupported model object '{name}'")


def export_model_state(model: Any, config: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Export one supported online model without private or executable state."""

    name = str(getattr(model, "name", ""))
    if name not in MODEL_NAMES:
        raise DataValidationError(f"cannot snapshot unknown model '{name}'")
    raw_config = _effective_config(model) if config is None else dict(config)
    normalized_config = _json_value(raw_config, f"{name} config")
    if not isinstance(normalized_config, dict):
        raise DataValidationError(f"{name} config must be a JSON object")
    try:
        create_model(name, normalized_config)
    except (TypeError, ValueError) as exc:
        raise DataValidationError(f"invalid {name} config: {exc}") from exc
    state = _model_state(model)
    return {
        "model": name,
        "config": normalized_config,
        "config_sha256": configuration_hash(normalized_config),
        "state": state,
        "state_sha256": _payload_hash(state),
    }


def _require_keys(payload: Mapping[str, Any], keys: tuple[str, ...], label: str) -> None:
    missing = [key for key in keys if key not in payload]
    if missing:
        raise DataValidationError(f"{label} is missing required fields: {', '.join(missing)}")


def _require_exact_keys(payload: Mapping[str, Any], keys: tuple[str, ...], label: str) -> None:
    _require_keys(payload, keys, label)
    extras = sorted(set(payload) - set(keys))
    if extras:
        raise DataValidationError(f"{label} has unknown fields: {', '.join(extras)}")


def _check_parameters(model: Any, state: Mapping[str, Any], fields: tuple[str, ...], label: str) -> None:
    for field in fields:
        expected = _finite_number(state.get(field), f"{label}.{field}")
        actual = _finite_number(getattr(model, field), f"{label}.{field}")
        if actual != expected:
            raise DataValidationError(f"{label}.{field} does not match model config")


def import_model_state(record: Mapping[str, Any]) -> Any:
    """Validate one state record and rebuild a fresh model instance."""

    if not isinstance(record, Mapping):
        raise DataValidationError("model state record must be an object")
    name = record.get("model")
    if not isinstance(name, str) or name not in MODEL_NAMES:
        raise DataValidationError(f"model state has unknown model '{name}'")
    if set(record) != {"model", "config", "config_sha256", "state", "state_sha256"}:
        raise DataValidationError(f"{name} state record has unknown or missing fields")
    config = record.get("config")
    state = record.get("state")
    if not isinstance(config, Mapping) or not isinstance(state, Mapping):
        raise DataValidationError(f"{name} state must contain config and state objects")
    if record.get("config_sha256") != configuration_hash(config):
        raise DataValidationError(f"{name} config hash does not match config")
    if record.get("state_sha256") != _payload_hash(state):
        raise DataValidationError(f"{name} state hash does not match state")
    try:
        model = create_model(str(name), dict(config))
    except (TypeError, ValueError) as exc:
        raise DataValidationError(f"invalid {name} config: {exc}") from exc

    if name == "frequency":
        _require_exact_keys(state, ("alpha", "counts"), "frequency state")
        _check_parameters(model, state, ("alpha",), "frequency state")
        counts = state["counts"]
        if not isinstance(counts, Mapping) or set(counts) != set(RESULTS):
            raise DataValidationError("frequency state counts must contain exactly H, D, and A")
        values = []
        for result in RESULTS:
            value = _finite_number(counts[result], f"frequency state counts.{result}")
            if value < 0:
                raise DataValidationError("frequency state counts must be non-negative")
            values.append(value)
        model.counts = np.asarray(values, dtype=float)
        return model

    if name == "elo":
        _require_exact_keys(state, (*_MODEL_PARAMETERS["elo"], "ratings"), "elo state")
        _check_parameters(model, state, _MODEL_PARAMETERS["elo"], "elo state")
        ratings = state["ratings"]
        if not isinstance(ratings, Mapping):
            raise DataValidationError("elo state ratings must be an object")
        model.ratings = {
            _team_name(team, "elo state team"): _finite_number(rating, f"elo state rating {team!r}")
            for team, rating in sorted(ratings.items())
        }
        return model

    _require_exact_keys(
        state,
        (*_MODEL_PARAMETERS["poisson"], "total_home_goals", "total_away_goals", "matches", "teams"),
        "poisson state",
    )
    _check_parameters(model, state, _MODEL_PARAMETERS["poisson"], "poisson state")
    total_matches = state["matches"]
    total_matches = _non_negative_int(total_matches, "poisson state matches")
    teams = state["teams"]
    if not isinstance(teams, Mapping):
        raise DataValidationError("poisson state teams must be an object")
    rebuilt: dict[str, _TeamStats] = {}
    for team, raw_stats in sorted(teams.items()):
        team_name = _team_name(team, "poisson state team")
        if not isinstance(raw_stats, Mapping):
            raise DataValidationError(f"poisson state stats for {team_name!r} must be an object")
        _require_exact_keys(raw_stats, ("goals_for", "goals_against", "matches"), f"poisson state {team_name}")
        matches = raw_stats["matches"]
        matches = _non_negative_int(matches, f"poisson state matches for {team_name!r}")
        rebuilt[team_name] = _TeamStats(
            goals_for=_non_negative_number(raw_stats["goals_for"], f"poisson state {team_name}.goals_for"),
            goals_against=_non_negative_number(raw_stats["goals_against"], f"poisson state {team_name}.goals_against"),
            matches=matches,
        )
    model.total_home_goals = _non_negative_number(state["total_home_goals"], "poisson state total_home_goals")
    model.total_away_goals = _non_negative_number(state["total_away_goals"], "poisson state total_away_goals")
    model.matches = total_matches
    model.teams = rebuilt
    return model


def model_states_payload(records: Mapping[str, Mapping[str, Any]] | list[Mapping[str, Any]]) -> dict[str, Any]:
    """Return a validated, deterministically ordered model-state file payload."""

    values = list(records.values()) if isinstance(records, Mapping) else list(records)
    validated: list[dict[str, Any]] = []
    names: set[str] = set()
    for record in values:
        normalized = _json_value(record, "model state record")
        if not isinstance(normalized, dict):
            raise DataValidationError("model state record must be an object")
        name = normalized.get("model")
        if not isinstance(name, str):
            raise DataValidationError("model state record has an invalid model name")
        if name in names:
            raise DataValidationError(f"duplicate model state: {name}")
        import_model_state(normalized)
        names.add(name)
        validated.append(normalized)
    if not validated:
        raise DataValidationError("at least one model state is required")
    validated.sort(key=lambda item: str(item["model"]))
    return {"schema_version": ENGINE_STATE_SCHEMA_VERSION, "models": validated}


def load_model_states(path: str | Path) -> dict[str, Any]:
    state_path = Path(path).expanduser().resolve()
    if not state_path.is_file():
        raise FileNotFoundError(f"model state file does not exist: {state_path}")
    payload = _read_json(state_path, "model state file")
    if payload.get("schema_version") != ENGINE_STATE_SCHEMA_VERSION:
        raise DataValidationError(f"model state schema_version must be {ENGINE_STATE_SCHEMA_VERSION}")
    records = payload.get("models")
    if not isinstance(records, list):
        raise DataValidationError("model state field 'models' must be a list")
    normalized = model_states_payload(records)
    if canonical_json_bytes(payload) != canonical_json_bytes(normalized):
        raise DataValidationError("model state file is not in deterministic canonical form")
    return normalized


def new_engine_snapshot_dir(output_root: str | Path) -> Path:
    """Compatibility wrapper for the manifest module's snapshot directory helper."""

    from .engine_snapshot_manifest import new_engine_snapshot_dir as implementation

    return implementation(output_root)


def manifest_content_sha256(payload: Mapping[str, Any]) -> str:
    """Compatibility wrapper for the manifest module's content hash helper."""

    from .engine_snapshot_manifest import manifest_content_sha256 as implementation

    return implementation(payload)


def write_snapshot_manifest(payload: Mapping[str, Any], path: str | Path) -> Path:
    """Compatibility wrapper for the manifest module's writer."""

    from .engine_snapshot_manifest import write_snapshot_manifest as implementation

    return implementation(payload, path)


def load_engine_snapshot(snapshot_manifest_path: str | Path) -> dict[str, Any]:
    """Compatibility wrapper for the manifest module's strict loader."""

    from .engine_snapshot_manifest import load_engine_snapshot as implementation

    return implementation(snapshot_manifest_path)

__all__ = [
    "ENGINE_SNAPSHOT_SCHEMA_VERSION", "ENGINE_STATE_SCHEMA_VERSION", "canonical_json_bytes",
    "configuration_hash", "export_model_state", "import_model_state", "load_engine_snapshot",
    "load_model_states", "manifest_content_sha256", "model_states_payload",
    "new_engine_snapshot_dir", "write_snapshot_manifest",
]
