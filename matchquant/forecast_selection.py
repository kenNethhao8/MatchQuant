"""M3 selection-evidence checks required by the M4 as-of protocol."""

from __future__ import annotations

from typing import Any, Mapping

import pandas as pd

from .backtest import _parse_boundary
from .schema import DataValidationError


def _parse_boundary_record(value: object, label: str) -> dict[str, object]:
    try:
        timestamp, date_only = _parse_boundary(value, label)
    except (TypeError, ValueError) as exc:
        raise DataValidationError(f"{label} is invalid: {value!r}") from exc
    return {"value": str(value), "timestamp": timestamp, "date_only": bool(date_only)}


def _effective_end(boundary: Mapping[str, object]) -> pd.Timestamp:
    timestamp = boundary["timestamp"]
    if not isinstance(timestamp, pd.Timestamp):
        raise DataValidationError("selection boundary timestamp is invalid")
    return timestamp + pd.Timedelta(days=1) if bool(boundary["date_only"]) else timestamp


def _serialise_boundary(boundary: Mapping[str, object]) -> dict[str, object]:
    timestamp = boundary["timestamp"]
    if not isinstance(timestamp, pd.Timestamp):
        raise DataValidationError("selection boundary timestamp is invalid")
    return {
        "value": str(boundary["value"]),
        "timestamp": timestamp.isoformat(),
        "date_only": bool(boundary["date_only"]),
    }


def _require_boundary_value(container: Mapping[str, Any], key: str, label: str) -> object:
    if key not in container or container[key] is None or str(container[key]).strip() == "":
        raise DataValidationError(f"{label} is missing")
    return container[key]


def _validate_final_test(
    payload: Mapping[str, Any], latest_validation_end: pd.Timestamp
) -> dict[str, object]:
    raw_final = payload.get("final_test")
    if not isinstance(raw_final, dict):
        raise DataValidationError("selection run config final_test must be an object")
    final_start = _parse_boundary_record(
        _require_boundary_value(raw_final, "start", "final_test.start"), "final_test.start"
    )
    if "end" not in raw_final:
        raise DataValidationError("final_test.end is missing; use null when there is no end")
    raw_end = raw_final["end"]
    final_end = None if raw_end is None else _parse_boundary_record(raw_end, "final_test.end")
    if final_end is not None and _effective_end(final_end) <= final_start["timestamp"]:
        raise DataValidationError("final_test.end must be after final_test.start")
    if latest_validation_end > final_start["timestamp"]:
        raise DataValidationError("final_test.start is before the validation folds end")
    return {
        "start": _serialise_boundary(final_start),
        "end": None if final_end is None else _serialise_boundary(final_end),
        "metrics_used_for_selection": False,
        "after_as_of_allowed": True,
        "note": "M3 final-test metrics are retained for audit and are not used to select the candidate.",
    }


def validate_selection_as_of(
    payload: Mapping[str, Any], as_of: str | pd.Timestamp
) -> dict[str, object]:
    """Validate M3 validation boundaries against one M4 as-of boundary."""

    rule = payload.get("selection_rule")
    if not isinstance(rule, dict):
        raise DataValidationError("selection run config selection_rule must be an object")
    if rule.get("final_test_metrics_used_for_selection") is not False:
        raise DataValidationError(
            "selection_rule.final_test_metrics_used_for_selection must be explicitly false"
        )

    raw_folds = payload.get("validation_folds")
    if not isinstance(raw_folds, list) or not raw_folds:
        raise DataValidationError("selection run config validation_folds must be a non-empty list")
    as_of_boundary = _parse_boundary_record(as_of, "as_of")
    as_of_timestamp = as_of_boundary["timestamp"]
    if not isinstance(as_of_timestamp, pd.Timestamp):
        raise DataValidationError("as_of timestamp is invalid")

    parsed: list[dict[str, object]] = []
    fold_ids: set[str] = set()
    previous_end: pd.Timestamp | None = None
    for index, raw_fold in enumerate(raw_folds, start=1):
        if not isinstance(raw_fold, dict):
            raise DataValidationError(f"validation_folds[{index}] must be an object")
        fold_id = str(raw_fold.get("fold_id", "")).strip()
        if not fold_id or fold_id in fold_ids:
            raise DataValidationError("validation fold IDs must be non-empty and unique")
        fold_ids.add(fold_id)
        start = _parse_boundary_record(
            _require_boundary_value(raw_fold, "start", f"validation fold {fold_id}.start"),
            f"validation fold {fold_id}.start",
        )
        end = _parse_boundary_record(
            _require_boundary_value(raw_fold, "end", f"validation fold {fold_id}.end"),
            f"validation fold {fold_id}.end",
        )
        effective_end = _effective_end(end)
        if effective_end <= start["timestamp"]:
            raise DataValidationError(f"validation fold {fold_id} END must be after START")
        if previous_end is not None and start["timestamp"] < previous_end:
            raise DataValidationError("validation folds must be chronological and non-overlapping")
        if (
            not bool(as_of_boundary["date_only"])
            and bool(end["date_only"])
            and end["timestamp"].normalize() == as_of_timestamp.normalize()
        ):
            raise DataValidationError(
                f"validation fold {fold_id} has an ambiguous date-only end on the same day as exact-time as_of"
            )
        if effective_end > as_of_timestamp:
            raise DataValidationError(
                f"validation fold {fold_id} ends after as_of; effective end is {effective_end.isoformat()}"
            )
        parsed.append(
            {
                "fold_id": fold_id,
                "start": _serialise_boundary(start),
                "end": _serialise_boundary(end),
                "effective_end_exclusive": effective_end.isoformat(),
                "_effective_end": effective_end,
            }
        )
        previous_end = effective_end

    latest = max(parsed, key=lambda fold: fold["_effective_end"])
    latest_end = dict(latest["end"])
    latest_end["fold_id"] = latest["fold_id"]
    latest_end["effective_end_exclusive"] = latest["effective_end_exclusive"]
    latest_validation_end = latest["_effective_end"]
    if not isinstance(latest_validation_end, pd.Timestamp):
        raise DataValidationError("latest validation end timestamp is invalid")
    for fold in parsed:
        fold.pop("_effective_end")

    final_test = _validate_final_test(payload, latest_validation_end)
    as_of_evidence = {
        "status": "validated",
        "as_of": _serialise_boundary(as_of_boundary),
        "validation_fold_count": len(parsed),
        "validation_folds": parsed,
        "latest_validation_end": latest_end,
        "selection_rule_final_test_metrics_used_for_selection": False,
        "final_test": final_test,
    }
    return {
        "selection_as_of_status": "validated",
        "as_of_evidence": as_of_evidence,
        "validation_latest_end": latest_end,
        "selection_rule": dict(rule),
        "final_test": final_test,
    }


__all__ = ["validate_selection_as_of"]
