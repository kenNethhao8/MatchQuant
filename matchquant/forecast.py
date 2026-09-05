"""Offline local Alpha forecasts from canonical history and pending fixtures."""

from __future__ import annotations

import hashlib
import json
import re
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from .backtest import _parse_boundary, _timeline
from .fixtures import validate_fixtures
from .io import file_sha256
from .models import MODEL_NAMES, RESULTS, create_model
from .schema import DataValidationError, validate_matches
from .selection import SELECTION_SCHEMA_VERSION
from .forecast_report import DISCLAIMER

FORECAST_SCHEMA_VERSION = "m4-forecast-v1"
FORECAST_PREDICTION_COLUMNS = (
    "fixture_id",
    "date",
    "season",
    "home_team",
    "away_team",
    "model",
    "home_win_prob",
    "draw_prob",
    "away_win_prob",
    "model_most_likely_result",
    "expected_home_goals",
    "expected_away_goals",
)

@dataclass
class ForecastRun:
    """In-memory result of one as-of forecast snapshot."""

    model_name: str
    predictions: pd.DataFrame
    as_of: dict[str, object]
    history_rows_total: int
    history_rows_used: int
    history_cutoff: str | None
    selection: dict[str, object]
    season_mode: str


def _configuration_hash(config: Mapping[str, Any]) -> str:
    serialized = json.dumps(config, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _read_json_object(path: Path, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DataValidationError(f"could not read {label} '{path}': {exc}") from exc
    if not isinstance(payload, dict):
        raise DataValidationError(f"{label} root must be a JSON object: {path}")
    return payload

def _valid_sha256(value: object) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-fA-F]{64}", value) is not None


def load_selection_run_config(path: str | Path, history_sha256: str) -> tuple[str, dict[str, Any], dict[str, object]]:
    """Extract and validate the selected M3 candidate from one run config."""

    selection_path = Path(path).expanduser().resolve()
    if not selection_path.is_file():
        raise FileNotFoundError(f"selection run config does not exist: {selection_path}")
    payload = _read_json_object(selection_path, "selection run config")
    if payload.get("schema_version") != SELECTION_SCHEMA_VERSION:
        raise DataValidationError(
            f"selection run config schema_version must be {SELECTION_SCHEMA_VERSION}"
        )
    selected = payload.get("selected_candidate")
    if not isinstance(selected, dict):
        raise DataValidationError("selection run config has no selected_candidate object")
    name = str(selected.get("candidate_name", "")).strip()
    model = str(selected.get("model", "")).strip()
    config = selected.get("config")
    if not name or model not in MODEL_NAMES or not isinstance(config, dict):
        raise DataValidationError(
            "selected_candidate must contain candidate_name, supported model, and config"
        )
    try:
        create_model(model, config)
    except (TypeError, ValueError) as exc:
        raise DataValidationError(f"selected candidate has invalid {model} config: {exc}") from exc

    selection_hash = payload.get("input_sha256")
    if not _valid_sha256(selection_hash):
        raise DataValidationError("selection run config input_sha256 must be a 64-character SHA-256")
    candidate_hash = selected.get("configuration_sha256")
    computed_candidate_hash = _configuration_hash(config)
    if candidate_hash is not None and candidate_hash != computed_candidate_hash:
        raise DataValidationError("selected candidate configuration_sha256 does not match config")
    history_match = str(selection_hash).lower() == history_sha256.lower()
    evidence = {
        "mode": "m3_run_config",
        "path": str(selection_path),
        "sha256": file_sha256(selection_path),
        "schema_version": payload["schema_version"],
        "selection_history_sha256": str(selection_hash),
        "current_history_sha256": history_sha256,
        "history_hash_matches": history_match,
        "history_hash_status": "matched" if history_match else "mismatch",
        "evidence_status": "matched" if history_match else "history_hash_mismatch",
        "selected_candidate": {
            "candidate_name": name,
            "model": model,
            "config": dict(config),
            "configuration_sha256": computed_candidate_hash,
        },
        "selection_season_mode": payload.get("season_mode"),
    }
    return model, dict(config), evidence

def load_model_config(path: str | Path, model: str) -> dict[str, Any]:
    """Read one explicit model JSON config and validate it against ``model``."""

    config_path = Path(path).expanduser().resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"model config does not exist: {config_path}")
    payload = _read_json_object(config_path, "model config")
    if "config" in payload and isinstance(payload["config"], dict):
        declared_model = payload.get("model")
        if declared_model is not None and str(declared_model) != model:
            raise DataValidationError(
                f"model config declares '{declared_model}', but --model is '{model}'"
            )
        config = dict(payload["config"])
    else:
        config = dict(payload)
    try:
        create_model(model, config)
    except (TypeError, ValueError) as exc:
        raise DataValidationError(f"model config is invalid for {model}: {exc}") from exc
    return config

def _reject_exact_boundary_with_date_only(
    frame: pd.DataFrame, boundary: Mapping[str, object], label: str
) -> None:
    if bool(boundary["date_only"]):
        return
    timestamp = boundary["timestamp"]
    if not isinstance(timestamp, pd.Timestamp):
        try:
            timestamp = pd.Timestamp(timestamp)
        except (TypeError, ValueError) as exc:
            raise TypeError(f"{label} timestamp is invalid") from exc
    ambiguous = frame["_date_only"] & frame["_timestamp"].dt.normalize().eq(timestamp.normalize())
    if ambiguous.any():
        raise ValueError(
            f"{label} contains a clock but the same day has date-only rows; "
            "use a calendar-date boundary or provide exact times for every row"
        )


def _history_before_as_of(
    history: pd.DataFrame, as_of: str | pd.Timestamp
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object], pd.Timestamp]:
    timestamp, date_only = _parse_boundary(as_of, "as_of")
    boundary = {"timestamp": timestamp, "date_only": date_only}
    timeline = _timeline(validate_matches(history), "match_id")
    _reject_exact_boundary_with_date_only(timeline, boundary, "as_of")
    timestamp = boundary["timestamp"]
    if not isinstance(timestamp, pd.Timestamp):
        raise TypeError("as_of timestamp is invalid")
    cutoff = timestamp.normalize() if bool(boundary["date_only"]) else timestamp
    eligible = timeline.loc[timeline["_batch_key"].lt(cutoff)].copy()
    as_of_meta = {
        "value": str(as_of),
        "timestamp": timestamp.isoformat(),
        "date_only": bool(boundary["date_only"]),
    }
    return timeline, eligible, as_of_meta, cutoff


def _fixtures_at_or_after_as_of(
    fixtures: pd.DataFrame, as_of_meta: Mapping[str, object], cutoff: pd.Timestamp
) -> pd.DataFrame:
    timeline = _timeline(validate_fixtures(fixtures), "fixture_id")
    _reject_exact_boundary_with_date_only(timeline, as_of_meta, "as_of")
    before = timeline["_timestamp"].lt(cutoff)
    if before.any():
        raise DataValidationError(
            "all fixtures must be on or after as_of; offending input rows "
            f"{', '.join(str(index + 1) for index in before.index[before][:5])}"
        )
    return timeline


def _fit_history(history: pd.DataFrame, model_object: Any, season_mode: str) -> int:
    model_object.reset()
    if history.empty:
        return 0
    current_season: str | None = None
    for _, batch in history.groupby("_batch_key", sort=False):
        seasons = batch["season"].astype(str).unique().tolist()
        if season_mode == "reset" and len(seasons) > 1:
            raise ValueError("season_mode='reset' cannot reset within a batch containing multiple seasons")
        season = seasons[0]
        if season_mode == "reset" and current_season is not None and season != current_season:
            model_object.reset()
        current_season = season
        for row in batch.itertuples(index=False):
            model_object.update(row.home_team, row.away_team, int(row.home_goals), int(row.away_goals))
    return int(len(history))


def _prediction_row(row: Any, model_name: str, prediction: Mapping[str, float]) -> dict[str, object]:
    probabilities = np.asarray(
        [prediction["home_win_prob"], prediction["draw_prob"], prediction["away_win_prob"]],
        dtype=float,
    )
    if probabilities.shape != (3,) or not np.all(np.isfinite(probabilities)):
        raise ValueError("model probabilities must be three finite values")
    if np.any(probabilities < 0) or not np.isclose(
        probabilities.sum(), 1.0, atol=1e-8, rtol=0.0
    ):
        raise ValueError("model probabilities must be non-negative and sum to 1")
    return {
        "fixture_id": row.fixture_id,
        "date": row.date,
        "season": row.season,
        "home_team": row.home_team,
        "away_team": row.away_team,
        "model": model_name,
        "home_win_prob": float(probabilities[0]),
        "draw_prob": float(probabilities[1]),
        "away_win_prob": float(probabilities[2]),
        "model_most_likely_result": RESULTS[int(np.argmax(probabilities))],
        "expected_home_goals": prediction.get("expected_home_goals"),
        "expected_away_goals": prediction.get("expected_away_goals"),
    }


def forecast_fixtures(
    history: pd.DataFrame,
    fixtures: pd.DataFrame,
    model: str | Any,
    as_of: str | pd.Timestamp,
    *,
    model_config: dict[str, Any] | None = None,
    season_mode: str = "carry",
    selection: dict[str, object] | None = None,
) -> ForecastRun:
    """Fit one online model before ``as_of`` and predict fixtures without updates."""

    if season_mode not in {"carry", "reset"}:
        raise ValueError("season_mode must be 'carry' or 'reset'")
    history_timeline, eligible, as_of_meta, cutoff = _history_before_as_of(history, as_of)
    fixture_timeline = _fixtures_at_or_after_as_of(fixtures, as_of_meta, cutoff)
    model_object = create_model(model, model_config) if isinstance(model, str) else model
    if not all(hasattr(model_object, name) for name in ("reset", "predict", "update", "name")):
        raise TypeError("model must be a supported model name or online model object")
    used = _fit_history(eligible, model_object, season_mode)
    predictions = [
        _prediction_row(
            row,
            str(model_object.name),
            model_object.predict(row.home_team, row.away_team),
        )
        for row in fixture_timeline.itertuples(index=False)
    ]
    prediction_frame = pd.DataFrame(predictions, columns=FORECAST_PREDICTION_COLUMNS)
    if prediction_frame.empty:
        raise ValueError("fixture set is empty after forecast validation")
    history_cutoff = str(eligible.iloc[-1]["date"]) if used else None
    return ForecastRun(
        model_name=str(model_object.name),
        predictions=prediction_frame,
        as_of=as_of_meta,
        history_rows_total=int(len(history_timeline)),
        history_rows_used=used,
        history_cutoff=history_cutoff,
        selection={} if selection is None else selection,
        season_mode=season_mode,
    )


def new_forecast_run_dir(output_root: str | Path) -> Path:
    root = Path(output_root).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = root / f"m4_forecast_{stamp}_{uuid.uuid4().hex[:8]}"
    run_dir.mkdir()
    return run_dir


def run_forecast(*args: Any, **kwargs: Any) -> dict[str, Any]:
    """Lazily import the persistence layer while preserving the public API."""

    from .forecast_runner import run_forecast as _run_forecast

    return _run_forecast(*args, **kwargs)


__all__ = [
    "DISCLAIMER",
    "FORECAST_PREDICTION_COLUMNS",
    "FORECAST_SCHEMA_VERSION",
    "ForecastRun",
    "forecast_fixtures",
    "load_model_config",
    "load_selection_run_config",
    "new_forecast_run_dir",
    "run_forecast",
]
