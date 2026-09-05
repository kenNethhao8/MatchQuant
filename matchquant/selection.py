"""Strict chronological candidate selection for the M3 research protocol."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from .backtest import run_walk_forward
from .calibration import calibration_diagnostics
from .models import MODEL_NAMES, create_model
from .schema import has_explicit_time, validate_matches

SELECTION_SCHEMA_VERSION = "m3-selection-v1"
CANDIDATE_CONFIG_VERSION = "m3-candidate-config-v1"


def _parse_boundary(value: str | pd.Timestamp, label: str) -> dict[str, object]:
    if value is None:
        raise ValueError(f"{label} is required")
    try:
        timestamp = pd.Timestamp(pd.to_datetime(value, errors="raise", format="mixed"))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid {label}: {value!r}") from exc
    date_only = not has_explicit_time(value)
    if date_only:
        timestamp = timestamp.normalize()
    return {"value": str(value), "timestamp": timestamp, "date_only": date_only}


def _effective_end(boundary: Mapping[str, object]) -> pd.Timestamp:
    timestamp = boundary["timestamp"]
    if not isinstance(timestamp, pd.Timestamp):
        raise TypeError("boundary timestamp is invalid")
    return timestamp + pd.Timedelta(days=1) if boundary["date_only"] else timestamp


def parse_validation_folds(specs: Iterable[str]) -> list[dict[str, str]]:
    """Parse ``START..END`` fold specs while preserving their supplied order."""

    values = [specs] if isinstance(specs, str) else list(specs)
    if not values:
        raise ValueError("at least one --validation-fold START..END is required")
    folds: list[dict[str, str]] = []
    for index, spec in enumerate(values, start=1):
        parts = re.split(r"\s*\.\.\s*", str(spec), maxsplit=1)
        if len(parts) != 2 or not all(part.strip() for part in parts):
            raise ValueError(f"validation fold {index} must use START..END")
        folds.append({"fold_id": f"validation_{index}", "start": parts[0].strip(), "end": parts[1].strip()})
    return folds


def _validate_protocol(
    folds: Sequence[Mapping[str, str]],
    final_test_start: str | pd.Timestamp,
    final_test_end: str | pd.Timestamp | None,
) -> tuple[list[dict[str, object]], dict[str, object], dict[str, object] | None]:
    final_start = _parse_boundary(final_test_start, "final_test_start")
    final_end = _parse_boundary(final_test_end, "final_test_end") if final_test_end is not None else None
    if final_end is not None and _effective_end(final_end) <= final_start["timestamp"]:
        raise ValueError("final_test_end must be after final_test_start")
    parsed: list[dict[str, object]] = []
    previous_end: pd.Timestamp | None = None
    for fold in folds:
        fold_id = str(fold.get("fold_id", ""))
        start = _parse_boundary(fold.get("start"), f"{fold_id or 'validation fold'} start")
        end = _parse_boundary(fold.get("end"), f"{fold_id or 'validation fold'} end")
        if _effective_end(end) <= start["timestamp"]:
            raise ValueError(f"{fold_id or 'validation fold'} must have END after START")
        if previous_end is not None and start["timestamp"] < previous_end:
            raise ValueError("validation folds must be chronological and non-overlapping")
        if _effective_end(end) > final_start["timestamp"]:
            raise ValueError("every validation fold must end no later than final_test_start")
        parsed.append(
            {
                "fold_id": fold_id or f"validation_{len(parsed) + 1}",
                "start": start,
                "end": end,
            }
        )
        previous_end = _effective_end(end)
    return parsed, final_start, final_end


def _configuration_hash(config: Mapping[str, Any]) -> str:
    serialized = json.dumps(config, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def load_candidates(path: str | Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Load and validate the small repository candidate configuration."""

    candidate_path = Path(path)
    try:
        payload = json.loads(candidate_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"could not read candidate config '{candidate_path}': {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError("candidate config root must be a JSON object")
    if payload.get("schema_version") != CANDIDATE_CONFIG_VERSION:
        raise ValueError(f"candidate config schema_version must be {CANDIDATE_CONFIG_VERSION}")
    raw_candidates = payload.get("candidates")
    if not isinstance(raw_candidates, list) or not raw_candidates:
        raise ValueError("candidate config field 'candidates' must be a non-empty list")
    candidates: list[dict[str, Any]] = []
    names: set[str] = set()
    for order, raw in enumerate(raw_candidates):
        if not isinstance(raw, dict):
            raise ValueError(f"candidate {order + 1} must be an object")
        name, model = str(raw.get("name", "")).strip(), str(raw.get("model", "")).strip()
        config = raw.get("config", {})
        if not name or name in names:
            raise ValueError("candidate names must be non-empty and unique")
        if model not in MODEL_NAMES:
            raise ValueError(f"candidate '{name}' has unknown model '{model}'")
        if not isinstance(config, dict):
            raise ValueError(f"candidate '{name}' config must be a JSON object")
        try:
            create_model(model, config)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"candidate '{name}' has invalid {model} config: {exc}") from exc
        names.add(name)
        candidates.append(
            {
                "candidate_order": order,
                "candidate_name": name,
                "model": model,
                "config": dict(config),
                "configuration_sha256": _configuration_hash(config),
            }
        )
    return candidates, payload


def _finite_metric(metrics: Mapping[str, object], key: str) -> float:
    value = float(metrics[key])
    if not np.isfinite(value):
        raise ValueError(f"metric '{key}' is non-finite")
    return value


def _fold_record(
    fold: Mapping[str, object], run_metrics: Mapping[str, object], n_matches: int
) -> dict[str, object]:
    overall = run_metrics["overall"]
    if not isinstance(overall, Mapping):
        raise ValueError("walk-forward metrics are missing overall values")
    return {
        "fold_id": str(fold["fold_id"]),
        "start": str(fold["start"]["value"]),
        "end": str(fold["end"]["value"]),
        "n_matches": int(n_matches),
        "log_loss": _finite_metric(overall, "log_loss"),
        "brier_score": _finite_metric(overall, "brier_score"),
        "accuracy": _finite_metric(overall, "accuracy"),
    }


def _aggregate(folds: Sequence[Mapping[str, object]]) -> dict[str, object]:
    if not folds:
        raise ValueError("validation produced no folds")
    total = int(sum(int(fold["n_matches"]) for fold in folds))
    if total <= 0:
        raise ValueError("validation produced no matches")
    return {
        "n_matches": total,
        "average_log_loss": float(
            sum(int(fold["n_matches"]) * float(fold["log_loss"]) for fold in folds) / total
        ),
        "average_brier_score": float(
            sum(int(fold["n_matches"]) * float(fold["brier_score"]) for fold in folds) / total
        ),
        "accuracy": float(
            sum(int(fold["n_matches"]) * float(fold["accuracy"]) for fold in folds) / total
        ),
    }


def _ranking_key(candidate: Mapping[str, object]) -> tuple[float, float, str, int]:
    validation = candidate["validation"]
    if not isinstance(validation, Mapping):
        raise ValueError("candidate validation summary is missing")
    return (
        float(validation["average_log_loss"]),
        float(validation["average_brier_score"]),
        str(candidate["candidate_name"]),
        int(candidate["candidate_order"]),
    )


def run_nested_selection(
    matches: pd.DataFrame,
    candidates: Sequence[Mapping[str, Any]],
    validation_folds: Iterable[Mapping[str, str]],
    final_test_start: str | pd.Timestamp,
    *,
    final_test_end: str | pd.Timestamp | None = None,
    season_mode: str = "carry",
    calibration_bins: int = 10,
) -> dict[str, object]:
    """Select on chronological validation folds, then evaluate once on final test."""

    normalized = validate_matches(matches)
    folds = list(validation_folds)
    parsed_folds, final_start, final_end = _validate_protocol(
        folds, final_test_start, final_test_end
    )
    if not candidates:
        raise ValueError("at least one candidate is required")
    candidate_records: list[dict[str, object]] = []
    candidate_names: set[str] = set()
    for candidate in candidates:
        name = str(candidate.get("candidate_name", "")).strip()
        model = str(candidate.get("model", "")).strip()
        config = candidate.get("config", {})
        if not name or name in candidate_names or model not in MODEL_NAMES or not isinstance(config, dict):
            raise ValueError("candidates must contain candidate_name, supported model, and config")
        try:
            create_model(model, config)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"candidate '{name}' has invalid {model} config: {exc}") from exc
        candidate_names.add(name)
        fold_records: list[dict[str, object]] = []
        for fold in parsed_folds:
            run = run_walk_forward(
                normalized,
                model,
                fold["start"]["value"],
                test_end=fold["end"]["value"],
                season_mode=season_mode,
                model_config=config,
            )
            fold_records.append(_fold_record(fold, run.metrics, run.test_matches))
        validation = _aggregate(fold_records)
        candidate_records.append(
            {
                "candidate_order": int(candidate.get("candidate_order", len(candidate_records))),
                "candidate_name": name,
                "model": model,
                "config": dict(config),
                "configuration_sha256": str(candidate.get("configuration_sha256") or _configuration_hash(config)),
                "folds": fold_records,
                "validation": validation,
            }
        )

    ranked = sorted(candidate_records, key=_ranking_key)
    ranking = [
        {
            "rank": index,
            "candidate_name": str(candidate["candidate_name"]),
            "candidate_order": int(candidate["candidate_order"]),
            "average_log_loss": float(candidate["validation"]["average_log_loss"]),
            "average_brier_score": float(candidate["validation"]["average_brier_score"]),
        }
        for index, candidate in enumerate(ranked, start=1)
    ]
    selected = ranked[0]
    final_run = run_walk_forward(
        normalized,
        str(selected["model"]),
        final_start["value"],
        test_end=final_end["value"] if final_end is not None else None,
        season_mode=season_mode,
        model_config=selected["config"],
    )
    calibration_summary, calibration_table = calibration_diagnostics(
        final_run.predictions, n_bins=calibration_bins
    )
    return {
        "schema_version": SELECTION_SCHEMA_VERSION,
        "validation_folds": [
            {"fold_id": str(fold["fold_id"]), "start": str(fold["start"]["value"]), "end": str(fold["end"]["value"])}
            for fold in parsed_folds
        ],
        "final_test": {
            "start": str(final_start["value"]),
            "end": str(final_end["value"]) if final_end is not None else None,
        },
        "selection_rule": {
            "primary": "validation pooled average multiclass log loss",
            "secondary": "validation pooled average multiclass Brier score",
            "final_tie_break": "lexicographic candidate_name, then candidate_order",
            "final_test_metrics_used_for_selection": False,
        },
        "candidates": candidate_records,
        "ranking": ranking,
        "selected_candidate": dict(selected),
        "final_run": final_run,
        "final_metrics": final_run.metrics,
        "calibration_summary": calibration_summary,
        "calibration_table": calibration_table,
    }
