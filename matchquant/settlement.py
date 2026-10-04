"""Offline M6 settlement of one local M4 forecast run."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import uuid

import pandas as pd

from .forecast import FORECAST_SCHEMA_VERSION
from .forecast_report import DISCLAIMER as FORECAST_DISCLAIMER
from .io import file_sha256, load_matches, write_json
from .metrics import PROBABILITY_COLUMNS, evaluate_predictions
from .models import result_from_score
from .prediction_evidence import load_prediction_evidence
from .schema import DataValidationError
from .settlement_report import write_settlement_report
from .settlement_validation import PREDICTION_REQUIRED_COLUMNS, load_forecast_run

SETTLEMENT_SCHEMA_VERSION = "m6-settlement-v1"
SETTLEMENT_METRICS_SCHEMA_VERSION = "m6-settlement-metrics-v1"
SETTLEMENT_COLUMNS = (
    *PREDICTION_REQUIRED_COLUMNS,
    "expected_home_goals",
    "expected_away_goals",
    "match_key",
    "settlement_status",
    "reason",
    "date_status",
    "result_match_id",
    "result_match_ids",
    "result_input_rows",
    "result_date",
    "result_season",
    "result_home_team",
    "result_away_team",
    "result_home_goals",
    "result_away_goals",
    "actual_result",
)
DISCLAIMER = (
    "M6 只对用户明确提供的本地最终赛果做离线审计结算；不是实时服务、投注建议、"
    "盈利承诺或未来表现保证。指标只代表本次安全匹配子集。"
)


def _match_key(row: Any) -> tuple[str, str, str]:
    return str(row.season), str(row.home_team), str(row.away_team)


def _display_key(key: tuple[str, str, str]) -> str:
    return "::".join(key)


def _result_time_issue(
    result: Any,
    as_of: pd.Timestamp,
    as_of_date_only: bool,
) -> str | None:
    result_date = result._timestamp
    if as_of_date_only:
        if result_date < as_of.normalize():
            return "result_before_forecast_as_of"
    elif result._date_only and result_date.normalize() == as_of.normalize():
        return "result_date_only_overlaps_exact_as_of"
    elif result_date < as_of:
        return "result_before_forecast_as_of"
    return None


def _base_row(prediction: Any, key: tuple[str, str, str]) -> dict[str, object]:
    return {
        "fixture_id": prediction.fixture_id,
        "date": prediction.date,
        "season": prediction.season,
        "home_team": prediction.home_team,
        "away_team": prediction.away_team,
        "model": prediction.model,
        "home_win_prob": float(prediction.home_win_prob),
        "draw_prob": float(prediction.draw_prob),
        "away_win_prob": float(prediction.away_win_prob),
        "model_most_likely_result": prediction.model_most_likely_result,
        "expected_home_goals": getattr(prediction, "expected_home_goals", None),
        "expected_away_goals": getattr(prediction, "expected_away_goals", None),
        "match_key": _display_key(key),
        "settlement_status": "unresolved",
        "reason": "no_unique_final_result_match",
        "date_status": "",
        "result_match_id": "",
        "result_match_ids": "",
        "result_input_rows": "",
        "result_date": "",
        "result_season": "",
        "result_home_team": "",
        "result_away_team": "",
        "result_home_goals": "",
        "result_away_goals": "",
        "actual_result": "",
    }


def settle_predictions(
    predictions: pd.DataFrame,
    results: pd.DataFrame,
    as_of: pd.Timestamp,
    *,
    as_of_date_only: bool,
) -> tuple[pd.DataFrame, dict[str, object], dict[str, int]]:
    """Match predictions to canonical results and score only safe unique matches."""

    result_groups = {
        key: group
        for key, group in results.groupby(["season", "home_team", "away_team"], sort=False)
    }
    rows: list[dict[str, object]] = []
    for prediction in predictions.itertuples(index=False):
        key = _match_key(prediction)
        prediction_date = pd.Timestamp(prediction.date)
        row = _base_row(prediction, key)
        candidates = result_groups.get(key)
        if candidates is None or candidates.empty:
            rows.append(row)
            continue
        row["result_match_ids"] = "|".join(str(value) for value in candidates["match_id"])
        row["result_input_rows"] = "|".join(str(int(index) + 2) for index in candidates.index)
        row["result_date"] = "|".join(str(value) for value in candidates["date"])
        if len(candidates) != 1:
            row["settlement_status"] = "ambiguous"
            row["reason"] = "multiple_results_match_same_season_home_away_key"
            rows.append(row)
            continue

        result = candidates.iloc[0]
        row["result_match_id"] = str(result["match_id"])
        row["result_season"] = str(result["season"])
        row["result_home_team"] = str(result["home_team"])
        row["result_away_team"] = str(result["away_team"])
        row["result_home_goals"] = int(result["home_goals"])
        row["result_away_goals"] = int(result["away_goals"])
        row["actual_result"] = result_from_score(
            int(result["home_goals"]), int(result["away_goals"])
        )
        row["date_status"] = (
            "same_date"
            if result["_timestamp"].normalize() == prediction_date.normalize()
            else "date_changed"
        )
        issue = _result_time_issue(result, as_of, as_of_date_only)
        if issue is not None:
            row["settlement_status"] = "time_conflict"
            row["reason"] = issue
        else:
            row["settlement_status"] = "settled"
            row["reason"] = "unique_exact_identity_match"
        rows.append(row)

    settlement = pd.DataFrame(rows, columns=SETTLEMENT_COLUMNS)
    status_names = ("settled", "unresolved", "ambiguous", "time_conflict")
    counts = {status: int(settlement["settlement_status"].eq(status).sum()) for status in status_names}
    counts["total_predictions"] = int(len(settlement))
    settled = settlement.loc[settlement["settlement_status"].eq("settled")]
    if settled.empty:
        metric_values: dict[str, object] = {
            "log_loss": None,
            "brier_score": None,
            "accuracy": None,
        }
    else:
        scored = settled.loc[:, ["season", "actual_result", *PROBABILITY_COLUMNS]]
        overall = evaluate_predictions(scored)["overall"]
        metric_values = {
            "log_loss": overall["log_loss"],
            "brier_score": overall["brier_score"],
            "accuracy": overall["accuracy"],
        }
    metric_values.update(
        {
            "total_predictions": counts["total_predictions"],
            "settled_predictions": counts["settled"],
            "coverage": (
                counts["settled"] / counts["total_predictions"]
                if counts["total_predictions"]
                else None
            ),
        }
    )
    return settlement, metric_values, counts


def _new_run_dir(output_root: str | Path) -> Path:
    root = Path(output_root).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = root / f"m6_settlement_{stamp}_{uuid.uuid4().hex[:8]}"
    run_dir.mkdir()
    return run_dir


def _path_is_within(path: Path, parent: Path) -> bool:
    return path == parent or parent in path.parents


def _timestamp_text(value: object) -> str:
    return pd.Timestamp(value).isoformat()


def settle_forecast_run(
    forecast_run_dir: str | Path,
    results_path: str | Path,
    *,
    output_root: str | Path = Path(r"D:\Codex\matchquant_runs"),
) -> dict[str, Any]:
    """Persist one M6 run while leaving all forecast and results inputs untouched."""

    forecast = load_prediction_evidence(forecast_run_dir)
    results_file = Path(results_path).expanduser().resolve()
    if not results_file.is_file():
        raise FileNotFoundError(f"canonical results CSV does not exist: {results_file}")
    results_hash = file_sha256(results_file)
    results = load_matches(results_file)
    output_root_path = Path(output_root).expanduser().resolve()
    if _path_is_within(output_root_path, forecast["run_dir"]):
        raise DataValidationError("M6 output root must not be the prediction run directory or one of its children")
    snapshot_dir = forecast["origin_evidence"].get("snapshot_dir")
    if snapshot_dir and _path_is_within(output_root_path, Path(snapshot_dir)):
        raise DataValidationError("M6 output root must not be the M11 snapshot directory or one of its children")
    data_root = Path(r"D:\Codex\matchquant_data").resolve()
    if _path_is_within(output_root_path, data_root):
        raise DataValidationError("M6 must not write under D:\\Codex\\matchquant_data")

    settlement, metrics, counts = settle_predictions(
        forecast["predictions"],
        results,
        forecast["as_of"],
        as_of_date_only=forecast["as_of_date_only"],
    )
    status = (
        "needs_review"
        if counts["ambiguous"] or counts["time_conflict"]
        else "partial"
        if counts["unresolved"]
        else "complete"
    )
    run_dir = _new_run_dir(output_root_path)
    settlement_path = run_dir / "settlement.csv"
    metrics_path = run_dir / "metrics.json"
    report_path = run_dir / "settlement_report.md"
    config_path = run_dir / "run_config.json"
    settlement.to_csv(settlement_path, index=False, encoding="utf-8-sig")
    metrics_payload = {
        "schema_version": SETTLEMENT_METRICS_SCHEMA_VERSION,
        "status": status,
        "counts": counts,
        "metrics": metrics,
        "log_loss": metrics["log_loss"],
        "brier_score": metrics["brier_score"],
        "accuracy": metrics["accuracy"],
        "total_predictions": metrics["total_predictions"],
        "settled_predictions": metrics["settled_predictions"],
        "coverage": metrics["coverage"],
        "metric_scope": "settled rows only; unresolved, ambiguous, and time_conflict rows excluded",
    }
    write_json(metrics_payload, metrics_path)

    result_cutoff = results["_timestamp"].max()
    result_start = results["_timestamp"].min()
    producer_schema = forecast["producer_schema"]
    if producer_schema == FORECAST_SCHEMA_VERSION:
        input_scope = "M4 run_config.json、predictions.csv"
    else:
        input_scope = (
            "M11 run_config.json、predictions.csv、snapshot_manifest.json、model_states.json "
            "与显式记录的 score-free fixtures"
        )
    limitations = [
        f"M6 只读取显式指定的 {input_scope} 和 M1 canonical 结果 CSV。",
        "零命中是预期的 unresolved，不代表输入错误；球队别名、改名和模糊匹配不会被猜测。",
        "指标不是实时预测、投注价值、盈利能力或未来表现证据。",
        *forecast["limitations"],
    ]
    config: dict[str, Any] = {
        "schema_version": SETTLEMENT_SCHEMA_VERSION,
        "command": "matchquant settle",
        "generated_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "status": status,
        "as_of": forecast["as_of_meta"],
        "forecast_run": {
            "directory": str(forecast["run_dir"]),
            "producer_schema": producer_schema,
            "run_config": {"path": str(forecast["run_config_path"]), "sha256": forecast["run_config_sha256"]},
            "predictions": {
                "path": str(forecast["predictions_path"]),
                "sha256": forecast["predictions_sha256"],
                "rows": int(len(forecast["predictions"])),
                "schema_version": producer_schema,
                "hash_evidence_status": forecast["prediction_hash_status"],
            },
            "model": forecast["model_name"],
            "producer_command": forecast["producer_command"],
            "model_config": forecast["model_config"],
            "model_config_sha256": forecast["model_config_sha256"],
            "origin_evidence": forecast["origin_evidence"],
        },
        "results": {
            "path": str(results_file),
            "sha256": results_hash,
            "rows": int(len(results)),
            "schema_version": "m1",
            "data_version": None,
            "provenance_status": "m1_csv_only_no_external_manifest",
            "date_range": {"start": _timestamp_text(result_start), "end": _timestamp_text(result_cutoff)},
            "data_cutoff": _timestamp_text(result_cutoff),
        },
        "matching": {
            "identity_key": ["season", "home_team", "away_team"],
            "comparison": "exact trimmed source text; no aliases or fuzzy matching",
            "date_rule": "date may differ for a unique identity match; same_date/date_changed is recorded",
            "duplicate_rule": "more than one result for one identity is ambiguous and excluded from metrics",
            "temporal_rule": "result before as_of is time_conflict and excluded; a unique date change is allowed",
        },
        "counts": counts,
        "metrics": metrics,
        "outputs": {
            "settlement": str(settlement_path),
            "metrics": str(metrics_path),
            "settlement_report": str(report_path),
            "run_config": str(config_path),
        },
        "input_mutation": {
            "forecast_run_modified": False,
            "results_input_modified": False,
            "matchquant_data_modified": False,
            "active_pointer_updated": False,
        },
        "limitations": limitations,
        "disclaimer": DISCLAIMER,
        "forecast_disclaimer": FORECAST_DISCLAIMER,
    }
    write_settlement_report(config, settlement, report_path)
    config["output_hashes"] = {
        "settlement": file_sha256(settlement_path),
        "metrics": file_sha256(metrics_path),
        "settlement_report": file_sha256(report_path),
    }
    write_json(config, config_path)
    return {
        "run_dir": str(run_dir),
        "settlement_path": str(settlement_path),
        "metrics_path": str(metrics_path),
        "settlement_report_path": str(report_path),
        "run_config_path": str(config_path),
        "status": status,
        "settlement": settlement,
        "metrics": metrics,
        "counts": counts,
        "config": config,
    }


__all__ = [
    "DISCLAIMER",
    "PREDICTION_REQUIRED_COLUMNS",
    "SETTLEMENT_COLUMNS",
    "SETTLEMENT_METRICS_SCHEMA_VERSION",
    "SETTLEMENT_SCHEMA_VERSION",
    "load_forecast_run",
    "settle_forecast_run",
    "settle_predictions",
]
