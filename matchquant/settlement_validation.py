"""Strict input validation for the M6 local settlement workflow."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from .backtest import _parse_boundary
from .fixtures import validate_fixtures
from .forecast import FORECAST_SCHEMA_VERSION
from .io import file_sha256
from .metrics import PROBABILITY_COLUMNS
from .models import MODEL_NAMES, RESULTS
from .schema import DataValidationError

PREDICTION_REQUIRED_COLUMNS = (
    "fixture_id",
    "date",
    "season",
    "home_team",
    "away_team",
    "model",
    *PROBABILITY_COLUMNS,
    "model_most_likely_result",
)


def _sample_indices(mask: pd.Series, limit: int = 5) -> str:
    values = [str(index + 2) for index in mask.index[mask][:limit]]
    suffix = ", ..." if int(mask.sum()) > limit else ""
    return ", ".join(values) + suffix


def _read_json_object(path: Path, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise DataValidationError(f"could not read {label} '{path}': {exc}") from exc
    if not isinstance(payload, dict):
        raise DataValidationError(f"{label} root must be a JSON object: {path}")
    return payload


def _valid_sha256(value: object) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-fA-F]{64}", value) is not None


def _read_prediction_csv(path: Path) -> pd.DataFrame:
    try:
        raw = pd.read_csv(path, dtype="string", keep_default_na=False)
    except (OSError, ValueError, pd.errors.ParserError) as exc:
        raise DataValidationError(f"could not read predictions CSV '{path}': {exc}") from exc
    try:
        frame = validate_fixtures(raw)
    except (TypeError, ValueError) as exc:
        raise DataValidationError(f"predictions CSV is not a valid M4 fixture table: {exc}") from exc
    missing = [column for column in PREDICTION_REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        raise DataValidationError(
            f"predictions CSV is missing required columns: {', '.join(missing)}"
        )

    model_values = frame["model"].astype("string").str.strip()
    missing_models = model_values.isna() | model_values.eq("")
    if missing_models.any():
        raise DataValidationError(
            "prediction column 'model' contains missing values at input rows "
            f"{_sample_indices(missing_models)}"
        )
    frame["model"] = model_values

    result_values = frame["model_most_likely_result"].astype("string").str.strip()
    invalid_results = result_values.isna() | ~result_values.isin(RESULTS)
    if invalid_results.any():
        raise DataValidationError(
            "prediction column 'model_most_likely_result' must contain only H, D, or A; "
            f"invalid input rows {_sample_indices(invalid_results)}"
        )
    frame["model_most_likely_result"] = result_values

    probabilities = frame.loc[:, PROBABILITY_COLUMNS].apply(pd.to_numeric, errors="coerce")
    values = probabilities.to_numpy(dtype=float)
    invalid_probabilities = (
        ~np.isfinite(values).all(axis=1)
        | (values < 0).any(axis=1)
        | (values > 1).any(axis=1)
        | ~np.isclose(values.sum(axis=1), 1.0, atol=1e-8, rtol=0.0)
    )
    if invalid_probabilities.any():
        raise DataValidationError(
            "prediction probabilities must be finite, non-negative, at most 1, and sum to 1; "
            f"invalid input rows {_sample_indices(pd.Series(invalid_probabilities))}"
        )
    for column in PROBABILITY_COLUMNS:
        frame[column] = probabilities[column].astype(float).to_numpy()
    expected = frame["model_most_likely_result"].map(
        {result: index for index, result in enumerate(RESULTS)}
    )
    if not np.array_equal(expected.to_numpy(dtype=int), np.argmax(values, axis=1)):
        raise DataValidationError(
            "model_most_likely_result does not match the largest H/D/A probability"
        )

    for column in ("expected_home_goals", "expected_away_goals"):
        if column not in frame.columns:
            frame[column] = pd.Series([None] * len(frame), dtype="object")
            continue
        numeric = pd.to_numeric(frame[column], errors="coerce")
        filled = frame[column].astype("string").str.strip().ne("")
        numeric_values = numeric.to_numpy(dtype=float, na_value=np.nan)
        invalid = filled & (numeric.isna() | ~np.isfinite(numeric_values) | numeric.lt(0))
        if invalid.any():
            raise DataValidationError(
                f"prediction column '{column}' contains invalid values at input rows "
                f"{_sample_indices(invalid)}"
            )
        frame[column] = numeric
    return frame


def _as_of_from_config(config: Mapping[str, Any]) -> tuple[pd.Timestamp, bool, dict[str, object]]:
    raw = config.get("as_of")
    if not isinstance(raw, Mapping) or raw.get("value") in (None, ""):
        raise DataValidationError("M4 run config must contain an as_of object with a value")
    value = raw["value"]
    try:
        timestamp, inferred_date_only = _parse_boundary(value, "forecast as_of")
    except (TypeError, ValueError) as exc:
        raise DataValidationError(str(exc)) from exc
    declared_date_only = raw.get("date_only")
    if not isinstance(declared_date_only, bool) or declared_date_only != inferred_date_only:
        raise DataValidationError("M4 run config as_of.date_only is inconsistent with as_of.value")
    recorded_timestamp = raw.get("timestamp")
    if recorded_timestamp is not None:
        try:
            parsed_recorded = pd.Timestamp(recorded_timestamp)
        except (TypeError, ValueError) as exc:
            raise DataValidationError("M4 run config as_of.timestamp is invalid") from exc
        if parsed_recorded != timestamp:
            raise DataValidationError("M4 run config as_of.timestamp does not match as_of.value")
    return timestamp, inferred_date_only, {
        "value": str(value),
        "timestamp": timestamp.isoformat(),
        "date_only": inferred_date_only,
    }


def _recorded_prediction_hash(config: Mapping[str, Any]) -> str | None:
    candidates: list[object] = [config.get("predictions_sha256")]
    outputs = config.get("outputs")
    if isinstance(outputs, Mapping):
        candidates.append(outputs.get("predictions_sha256"))
        prediction_output = outputs.get("predictions")
        if isinstance(prediction_output, Mapping):
            candidates.append(prediction_output.get("sha256"))
    hashes: list[str] = []
    for value in candidates:
        if value is None:
            continue
        if not _valid_sha256(value):
            raise DataValidationError(
                "M4 predictions SHA-256 evidence is not a 64-character SHA-256"
            )
        hashes.append(str(value).lower())
    if not hashes:
        return None
    if len(set(hashes)) != 1:
        raise DataValidationError("M4 predictions SHA-256 evidence fields disagree")
    return hashes[0]


def _validate_prediction_dates(frame: pd.DataFrame, as_of: pd.Timestamp, date_only: bool) -> None:
    if date_only:
        invalid = frame["_timestamp"].lt(as_of.normalize())
    else:
        ambiguous = frame["_date_only"] & frame["_timestamp"].dt.normalize().eq(as_of.normalize())
        if ambiguous.any():
            raise DataValidationError(
                "predictions contain date-only fixtures on the exact as_of day; temporal order is ambiguous"
            )
        invalid = frame["_timestamp"].lt(as_of)
    if invalid.any():
        raise DataValidationError(
            "predictions contain fixtures before as_of at input rows "
            f"{_sample_indices(invalid)}"
        )


def load_forecast_run(forecast_run_dir: str | Path) -> dict[str, Any]:
    """Read and strictly validate one M4 run without changing any input file."""

    run_dir = Path(forecast_run_dir).expanduser().resolve()
    if not run_dir.is_dir():
        raise FileNotFoundError(f"forecast run directory does not exist: {run_dir}")
    config_path = run_dir / "run_config.json"
    predictions_path = run_dir / "predictions.csv"
    if not config_path.is_file():
        raise FileNotFoundError(f"M4 run_config.json does not exist: {config_path}")
    if not predictions_path.is_file():
        raise FileNotFoundError(f"M4 predictions.csv does not exist: {predictions_path}")

    config_hash = file_sha256(config_path)
    prediction_hash = file_sha256(predictions_path)
    config = _read_json_object(config_path, "M4 run config")
    if config.get("schema_version") != FORECAST_SCHEMA_VERSION:
        raise DataValidationError(
            f"unsupported forecast run schema_version; expected {FORECAST_SCHEMA_VERSION}"
        )
    model_config = config.get("model")
    model_name = str(model_config.get("name", "")) if isinstance(model_config, Mapping) else ""
    if model_name not in MODEL_NAMES:
        raise DataValidationError("M4 run config must contain a supported model.name")
    recorded_hash = _recorded_prediction_hash(config)
    if recorded_hash is not None and recorded_hash != prediction_hash.lower():
        raise DataValidationError(
            "M4 predictions SHA-256 does not match predictions.csv; input appears modified or tampered"
        )
    as_of, date_only, as_of_meta = _as_of_from_config(config)
    predictions = _read_prediction_csv(predictions_path)
    if predictions["model"].astype(str).nunique() != 1 or str(predictions["model"].iloc[0]) != model_name:
        raise DataValidationError("predictions model does not match the M4 run config model.name")
    _validate_prediction_dates(predictions, as_of, date_only)

    if recorded_hash is None:
        hash_status = "unrecorded_legacy"
    else:
        hash_status = "verified"
    return {
        "run_dir": run_dir,
        "config_path": config_path,
        "predictions_path": predictions_path,
        "config_hash": config_hash,
        "prediction_hash": prediction_hash,
        "prediction_hash_status": hash_status,
        "config": config,
        "predictions": predictions,
        "model_name": model_name,
        "as_of": as_of,
        "as_of_date_only": date_only,
        "as_of_meta": as_of_meta,
    }


__all__ = ["PREDICTION_REQUIRED_COLUMNS", "load_forecast_run"]
