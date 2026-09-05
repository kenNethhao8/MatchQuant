"""Offline M8 publication-readiness verification for one M4 forecast snapshot."""

from __future__ import annotations

import json
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from .fixtures import FIXTURE_COLUMNS
from .forecast import _history_before_as_of
from .forecast_report import DISCLAIMER as FORECAST_DISCLAIMER
from .io import file_sha256, load_matches, write_json
from .release_verify_fixture import fixture_snapshot_checks
from .schema import DataValidationError, REQUIRED_COLUMNS
from .settlement_validation import load_forecast_run

RELEASE_VERIFY_SCHEMA_VERSION = "m8-release-verify-v1"
RELEASE_STATUSES = ("ready", "needs_review", "blocked")
_SHA256 = re.compile(r"^[0-9a-fA-F]{64}$")
_HISTORY_SCHEMAS = {"m2-processed-manifest-v1", "m2.1-processed-manifest-v1"}


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DataValidationError(f"could not read {label} '{path}': {exc}") from exc
    if not isinstance(payload, dict):
        raise DataValidationError(f"{label} root must be a JSON object: {path}")
    return payload


def _valid_hash(value: object) -> bool:
    return isinstance(value, str) and _SHA256.fullmatch(value) is not None


def _path(value: object, base: Path) -> Path | None:
    if value is None or str(value).strip() == "":
        return None
    candidate = Path(str(value))
    return (candidate if candidate.is_absolute() else base / candidate).resolve()


def _within(child: Path, parent: Path) -> bool:
    return child == parent or parent in child.parents


def _check(check_id: str, status: str, summary: str, reasons: list[str] | None = None, **evidence: Any) -> dict[str, Any]:
    return {"id": check_id, "status": status, "summary": summary, "reasons": reasons or [], "evidence": evidence}


def _timestamp(value: object) -> pd.Timestamp:
    return pd.Timestamp(pd.to_datetime(value, errors="raise", format="mixed"))


def _same_time(left: object, right: object) -> bool:
    try:
        return _timestamp(left) == _timestamp(right)
    except (TypeError, ValueError):
        return False


def _observed_cutoff(frame: pd.DataFrame | None) -> str | None:
    if frame is None or frame.empty:
        return None
    value = pd.Timestamp(frame["_timestamp"].max())
    return value.strftime("%Y-%m-%d") if value.hour == value.minute == value.second == value.microsecond == 0 else value.isoformat()


def _output_descriptor(payload: Mapping[str, Any]) -> dict[str, Any]:
    raw = payload.get("output")
    output = raw if isinstance(raw, Mapping) else {}
    path = next((output.get(key) for key in ("filename", "path", "canonical") if output.get(key)), None)
    digest = next((output.get(key) for key in ("sha256", "canonical_sha256") if output.get(key)), None)
    rows = next((output.get(key) for key in ("rows", "row_count") if output.get(key) is not None), None)
    return {"path": path, "sha256": digest, "rows": rows, "columns": output.get("columns")}


def _file_record(path: Path | None) -> dict[str, Any]:
    record: dict[str, Any] = {"path": str(path) if path else None, "exists": bool(path and path.is_file()), "sha256": None}
    if path and path.is_file():
        record["sha256"] = file_sha256(path)
    return record


def _recorded_prediction_hash(config: Mapping[str, Any]) -> str | None:
    values: list[object] = [config.get("predictions_sha256")]
    outputs = config.get("outputs")
    if isinstance(outputs, Mapping):
        values.append(outputs.get("predictions_sha256"))
        prediction = outputs.get("predictions")
        if isinstance(prediction, Mapping):
            values.append(prediction.get("sha256"))
    hashes = [str(value).lower() for value in values if value is not None]
    if any(not _valid_hash(value) for value in hashes) or len(set(hashes)) > 1:
        raise DataValidationError("M4 predictions SHA-256 evidence is invalid or fields disagree")
    return hashes[0] if hashes else None


def _m4_artifact_reference(forecast: dict[str, Any]) -> dict[str, Any]:
    outputs = forecast["config"].get("outputs")
    if not isinstance(outputs, Mapping):
        return _check("m4_artifact_reference", "needs_review", "M4 outputs artifact references 不完整", ["M4 run_config 缺少 outputs object；M6 已按实际 run 目录完成读取"])
    reasons: list[str] = []
    for key, actual in (("predictions", forecast["predictions_path"]), ("run_config", forecast["config_path"])):
        declared = _path(outputs.get(key), forecast["run_dir"])
        if declared is None:
            reasons.append(f"M4 outputs.{key} 缺失")
        elif declared != actual:
            reasons.append(f"M4 outputs.{key} 与实际 run artifact 路径不一致")
    return _check("m4_artifact_reference", "blocked" if reasons else "passed", "M4 outputs artifact 路径已与实际 run 文件核对", reasons, declared=dict(outputs), actual={"predictions": str(forecast["predictions_path"]), "run_config": str(forecast["config_path"])})


def _history_checks(history_path: Path, manifest_path: Path, history: pd.DataFrame | None, forecast: dict[str, Any] | None, history_hash: str | None) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    info: dict[str, Any] = {"path": str(history_path), "sha256": history_hash, "rows": len(history) if history is not None else None, "manifest_path": str(manifest_path), "manifest_sha256": file_sha256(manifest_path) if manifest_path.is_file() else None}
    try:
        payload = _read_json(manifest_path, "history manifest")
        info.update({"schema_version": payload.get("schema_version")})
    except (DataValidationError, OSError, ValueError) as exc:
        checks = [_check("history_manifest_structure", "blocked", "历史 manifest 无法作为 JSON object 使用", [str(exc)])]
        if forecast is not None and history is not None:
            checks.append(_check("m4_history_reference", "blocked", "无法核对 M4 history 引用", [str(exc)], actual_path=str(history_path), actual_sha256=history_hash))
        return checks, info

    descriptor = _output_descriptor(payload)
    output_path = _path(descriptor["path"], manifest_path.parent)
    info.update({"manifest_output": {**descriptor, "path": str(output_path) if output_path else None}, "dataset_version": payload.get("dataset_version"), "data_cutoff": payload.get("data_cutoff")})
    reasons: list[str] = []
    if payload.get("schema_version") not in _HISTORY_SCHEMAS:
        reasons.append(f"不支持的 history manifest schema_version: {payload.get('schema_version')!r}")
    if output_path is None:
        reasons.append("history manifest 缺少 output.filename/path")
    if not _valid_hash(descriptor["sha256"]):
        reasons.append("history manifest output.sha256 缺失或不是 64 位 SHA-256")
    if not isinstance(descriptor["rows"], int) or isinstance(descriptor["rows"], bool) or descriptor["rows"] < 0:
        reasons.append("history manifest output.rows 缺失或无效")
    if descriptor["columns"] is not None and descriptor["columns"] != list(REQUIRED_COLUMNS):
        reasons.append("history manifest output.columns 与 M1 canonical schema 不一致")
    if history_path != output_path:
        reasons.append("history manifest output 路径与显式 --history 不一致")
    if history_hash and descriptor["sha256"] and str(descriptor["sha256"]).lower() != history_hash.lower():
        reasons.append("history manifest output SHA-256 与实际 --history 不一致")
    if history is not None and isinstance(descriptor["rows"], int) and descriptor["rows"] != len(history):
        reasons.append(f"history manifest output.rows={descriptor['rows']}，实际 canonical 行数={len(history)}")
    checks = [_check("history_manifest_output", "blocked" if reasons else "passed", "history manifest 的 canonical output 已核对", reasons, output=info["manifest_output"])]

    metadata_reasons: list[str] = []
    version, cutoff = payload.get("dataset_version"), payload.get("data_cutoff")
    if not isinstance(version, str) or not version.strip():
        metadata_reasons.append("history manifest 缺少 dataset_version；M8 不推造版本号")
    if cutoff in (None, ""):
        metadata_reasons.append("history manifest 缺少 data_cutoff；M8 不用观测值替代 manifest 证据")
    elif history is not None:
        try:
            if not _same_time(cutoff, _observed_cutoff(history)):
                metadata_reasons.append(f"manifest data_cutoff={cutoff!r} 与实际 canonical cutoff={_observed_cutoff(history)!r} 不一致")
        except (TypeError, ValueError):
            metadata_reasons.append(f"manifest data_cutoff 不是可比较的日期/时间: {cutoff!r}")
    metadata_status = "needs_review" if metadata_reasons and all("缺少" in item for item in metadata_reasons) else "blocked" if metadata_reasons else "passed"
    checks.append(_check("history_manifest_metadata", metadata_status, "history manifest 的版本与 cutoff 证据已检查", metadata_reasons, dataset_version=version, data_cutoff=cutoff, observed_cutoff=_observed_cutoff(history)))

    if forecast is not None and history is not None:
        declared = forecast["config"].get("history")
        reasons = []
        if not isinstance(declared, Mapping):
            reasons.append("M4 run_config 缺少 history object")
        else:
            if _path(declared.get("path"), forecast["run_dir"]) != history_path:
                reasons.append("M4 run_config history.path 与显式 --history 不一致")
            if not _valid_hash(declared.get("sha256")) or str(declared.get("sha256")).lower() != str(history_hash).lower():
                reasons.append("M4 run_config history.sha256 与显式 history 实际 SHA-256 不一致")
            if declared.get("rows") != len(history):
                reasons.append(f"M4 run_config history.rows={declared.get('rows')!r}，实际行数={len(history)}")
        checks.append(_check("m4_history_reference", "blocked" if reasons else "passed", "M4 history 路径、hash 与行数已核对", reasons, declared=declared, actual_path=str(history_path), actual_sha256=history_hash))
    else:
        checks.append(_check("m4_history_reference", "blocked", "无法核对 M4 history 引用", ["M4 或 history canonical 未通过读取/校验"]))
    return checks, info


def _time_check(forecast: dict[str, Any] | None, history: pd.DataFrame | None, fixture: dict[str, Any], history_info: dict[str, Any]) -> dict[str, Any]:
    if forecast is None or history is None or fixture.get("frame") is None:
        return _check("time_boundary_crosscheck", "blocked", "时间链路无法完整交叉核对", ["M4、history 或 M5 fixture output 不完整"])
    try:
        timeline, eligible, _, _ = _history_before_as_of(history, forecast["as_of_meta"]["value"])
    except (TypeError, ValueError, DataValidationError) as exc:
        return _check("time_boundary_crosscheck", "blocked", "M4 as-of 与 history 无法按既有 date-only/精确时刻语义重放", [str(exc)])
    declared = forecast["config"].get("history") if isinstance(forecast["config"].get("history"), Mapping) else {}
    actual_used, actual_cutoff = len(eligible), _observed_cutoff(eligible)
    reasons: list[str] = []
    if declared.get("rows_used_before_as_of") != actual_used:
        reasons.append(f"M4 history.rows_used_before_as_of={declared.get('rows_used_before_as_of')!r}，实际={actual_used}")
    if not ((declared.get("data_cutoff") is None and actual_cutoff is None) or _same_time(declared.get("data_cutoff"), actual_cutoff)):
        reasons.append(f"M4 history.data_cutoff={declared.get('data_cutoff')!r}，实际纳入 history cutoff={actual_cutoff!r}")
    if declared.get("rows_excluded_at_or_after_as_of") != len(history) - actual_used:
        reasons.append("M4 history.rows_excluded_at_or_after_as_of 与实际排除行数不一致")
    flags = forecast["config"].get("validation")
    required_flags = {"all_fixture_dates_at_or_after_as_of": True, "fixture_results_used_for_model_updates": False, "probabilities_finite_non_negative_sum_to_one": True}
    if not isinstance(flags, Mapping) or any(flags.get(key) is not expected for key, expected in required_flags.items()):
        reasons.append("M4 validation 字段缺失、为 false，或与既有 M4 语义矛盾")
    frame = fixture["frame"]
    as_of, date_only = forecast["as_of"], forecast["as_of_date_only"]
    if date_only:
        before = frame["_timestamp"].lt(as_of.normalize())
    else:
        if (frame["_date_only"] & frame["_timestamp"].dt.normalize().eq(as_of.normalize())).any():
            reasons.append("fixture 在精确 as-of 同日使用 date-only，时刻不可确定")
        before = frame["_timestamp"].lt(as_of)
    if before.any():
        reasons.append("fixture 日期早于 M4 as-of")
    prediction_keys = forecast["predictions"].loc[:, FIXTURE_COLUMNS].sort_values(["date", "fixture_id"]).reset_index(drop=True).astype(str)
    fixture_keys = frame.loc[:, FIXTURE_COLUMNS].sort_values(["date", "fixture_id"]).reset_index(drop=True).astype(str)
    if not prediction_keys.equals(fixture_keys):
        reasons.append("M4 predictions 的 fixture 身份/日期与 M5 canonical output 不一致")
    quality = fixture.get("quality_payload") or {}
    pending_range = quality.get("pending_date_range")
    actual_range = {"earliest": str(frame["date"].min()), "latest": str(frame["date"].max())}
    if not isinstance(pending_range, Mapping) or dict(pending_range) != actual_range:
        reasons.append("M5 pending_date_range 与实际 fixture 日期范围不一致")
    completed_cutoff = quality.get("completed_data_cutoff")
    relation = "before_or_unknown"
    if completed_cutoff not in (None, ""):
        try:
            relation = "after_or_equal" if _timestamp(completed_cutoff) >= as_of else "before"
        except (TypeError, ValueError):
            reasons.append(f"M5 completed_data_cutoff 不是可比较日期: {completed_cutoff!r}")
    return _check("time_boundary_crosscheck", "blocked" if reasons else "passed", "M4 as-of、实际 history cutoff/rows-used、M5 日期范围已交叉核对", reasons, as_of=forecast["as_of_meta"], history={"rows_total": len(timeline), "rows_used_actual": actual_used, "rows_used_declared": declared.get("rows_used_before_as_of"), "cutoff_actual": actual_cutoff, "cutoff_declared": declared.get("data_cutoff"), "manifest_cutoff": history_info.get("data_cutoff")}, fixtures={"pending_date_range": dict(pending_range) if isinstance(pending_range, Mapping) else pending_range, "actual_date_range": actual_range, "completed_data_cutoff": completed_cutoff, "completed_cutoff_relation_to_as_of": relation}, chronology_note="history cutoff 在 as-of 之前只证明本次 replay 使用边界，不证明历史数据已最新")


def _selection_check(forecast: dict[str, Any] | None, history_hash: str | None) -> dict[str, Any]:
    if forecast is None:
        return _check("selection_evidence", "blocked", "无法读取 M4 selection evidence", ["M4 run 未通过严格校验"])
    config, selection = forecast["config"], forecast["config"].get("selection")
    top_status = config.get("selection_evidence_status")
    reasons: list[str] = []
    review: list[str] = []
    if not isinstance(selection, Mapping):
        reasons.append("M4 run_config 缺少 selection object")
        selection = {}
    if top_status != selection.get("evidence_status"):
        reasons.append("M4 selection_evidence_status 与 selection.evidence_status 不一致")
    if top_status == "history_hash_mismatch":
        review.append("M4 已记录 selection history hash mismatch；M8 不会静默提升或修复该证据")
    elif top_status not in {"matched", "explicit_model_config"}:
        reasons.append(f"M4 selection evidence status 不在允许集合内: {top_status!r}")
    mode = selection.get("mode")
    if mode == "m3_run_config":
        selected_hash = selection.get("selection_history_sha256")
        if not _valid_hash(selected_hash) or not history_hash:
            reasons.append("M3 selection history hash 缺失或无效")
        elif str(selected_hash).lower() != history_hash.lower() and top_status != "history_hash_mismatch":
            reasons.append("M3 selection history hash 不匹配但 M4 未标记 mismatch")
        if selection.get("history_hash_matches") is False and top_status != "history_hash_mismatch":
            reasons.append("M4 selection 标记 history_hash_matches=false 但 evidence status 不是 mismatch")
        if selection.get("history_hash_status") == "mismatch" and top_status != "history_hash_mismatch":
            reasons.append("M4 selection 标记 history_hash_status=mismatch 但 evidence status 不是 mismatch")
        if top_status == "history_hash_mismatch" and selection.get("history_hash_matches") is True:
            reasons.append("M4 selection 同时标记 history_hash_mismatch 与 history_hash_matches=true")
        if config.get("selection_as_of_status") != "validated" or selection.get("selection_as_of_status") != "validated":
            reasons.append("M3 selection as-of evidence 未明确 validated")
    elif mode == "explicit_model_config":
        if config.get("selection_as_of_status") != "not_applicable":
            reasons.append("explicit model config 的 selection_as_of_status 应为 not_applicable")
    else:
        reasons.append(f"未知 selection mode: {mode!r}")
    evidence_path = _path(selection.get("path"), forecast["run_dir"])
    if evidence_path is None or not evidence_path.is_file():
        review.append("selection evidence 原文件不可用；M8 保留 M4 已记录状态，不重新选择模型")
    elif selection.get("sha256") is not None:
        try:
            if not _valid_hash(selection.get("sha256")) or file_sha256(evidence_path).lower() != str(selection["sha256"]).lower():
                reasons.append("selection evidence 文件实际 hash 与 M4 记录不一致")
        except OSError as exc:
            reasons.append(f"selection evidence 文件无法读取: {exc}")
    status = "blocked" if reasons else "needs_review" if review else "passed"
    return _check("selection_evidence", status, "只保留并核对 M4 已有的 M3/explicit model selection evidence；未重新选择、训练或校准", reasons + review, mode=mode, m4_status=top_status, selection_as_of_status=config.get("selection_as_of_status"), evidence_path=str(evidence_path) if evidence_path else None, recorded_selection=dict(selection), selection_recomputed=False)


def _new_release_dir(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    while True:
        path = root / f"m8_release_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}_{uuid.uuid4().hex[:8]}"
        try:
            path.mkdir()
            return path
        except FileExistsError:
            continue


def _overall_status(checks: list[dict[str, Any]]) -> str:
    if any(item["status"] == "blocked" for item in checks):
        return "blocked"
    return "needs_review" if any(item["status"] == "needs_review" for item in checks) else "ready"


def verify_release(forecast_run_dir: str | Path, history_path: str | Path, history_manifest_path: str | Path, fixture_manifest_path: str | Path, *, output_root: str | Path = Path(r"D:\Codex\matchquant_runs")) -> dict[str, Any]:
    """Create one immutable M8 audit directory from explicit local evidence only."""
    forecast_dir = Path(forecast_run_dir).expanduser().resolve()
    history_file = Path(history_path).expanduser().resolve()
    history_manifest = Path(history_manifest_path).expanduser().resolve()
    fixture_manifest = Path(fixture_manifest_path).expanduser().resolve()
    output_root_path = Path(output_root).expanduser().resolve()
    data_root = Path(r"D:\Codex\matchquant_data").resolve()
    if output_root_path in {history_file.parent, history_manifest.parent} or any(_within(output_root_path, path) for path in (forecast_dir, fixture_manifest.parent, data_root)):
        raise DataValidationError("M8 output-root must not be an input directory, the M4 run, the M5 snapshot, or under D:\\Codex\\matchquant_data")

    checks: list[dict[str, Any]] = []
    forecast: dict[str, Any] | None = None
    history: pd.DataFrame | None = None
    history_hash: str | None = None
    try:
        forecast = load_forecast_run(forecast_dir)
        checks.append(_check("m4_run_validation", "passed", "M4 run_config、predictions schema 与 M6 严格规则已通过", prediction_hash_status=forecast["prediction_hash_status"], config_sha256=forecast["config_hash"], predictions_sha256=forecast["prediction_hash"]))
    except (DataValidationError, FileNotFoundError, OSError, TypeError, ValueError) as exc:
        checks.append(_check("m4_run_validation", "blocked", "M4 run 未通过 M6 严格验证", [str(exc)]))
    try:
        history_hash = file_sha256(history_file)
        history = load_matches(history_file)
    except (DataValidationError, FileNotFoundError, OSError, TypeError, ValueError) as exc:
        checks.append(_check("history_canonical_validation", "blocked", "显式 history canonical CSV 无法读取/校验", [str(exc)], path=str(history_file), sha256=history_hash))

    if forecast is not None:
        try:
            recorded = _recorded_prediction_hash(forecast["config"])
            status = "verified" if recorded else "unrecorded_legacy"
            checks.append(_check("m4_prediction_hash", "passed" if recorded else "needs_review", "M4 predictions SHA-256 证据已记录" if recorded else "旧 M4 缺少 predictions SHA-256，保留为 unrecorded_legacy", [] if recorded else ["不能把旧 M4 快照标为 ready"], recorded_sha256=recorded, actual_sha256=forecast["prediction_hash"], evidence_status=status))
        except DataValidationError as exc:
            checks.append(_check("m4_prediction_hash", "blocked", "M4 predictions hash evidence 无效", [str(exc)]))
        checks.append(_m4_artifact_reference(forecast))

    history_checks, history_info = _history_checks(history_file, history_manifest, history, forecast, history_hash)
    checks.extend(history_checks)
    fixture_checks, fixture_info = fixture_snapshot_checks(fixture_manifest, forecast)
    checks.extend(fixture_checks)
    checks.append(_time_check(forecast, history, fixture_info, history_info))
    checks.append(_selection_check(forecast, history_hash))
    status = _overall_status(checks)

    run_dir = _new_release_dir(output_root_path)
    report_path, manifest_path = run_dir / "release_report.md", run_dir / "release_manifest.json"
    inputs: dict[str, Any] = {
        "forecast_run": {"directory": str(forecast_dir), "run_config": _file_record(forecast["config_path"] if forecast else forecast_dir / "run_config.json"), "predictions": _file_record(forecast["predictions_path"] if forecast else forecast_dir / "predictions.csv")},
        "history": {"path": str(history_file), "sha256": history_hash, "rows": len(history) if history is not None else None},
        "history_manifest": history_info,
        "fixture_snapshot_manifest": {"path": str(fixture_manifest), "sha256": fixture_info.get("manifest_sha256"), **{key: fixture_info.get(key) for key in ("schema_version", "snapshot_id", "snapshot_dir")}},
        "fixture_output": fixture_info.get("output"),
        "fixture_raw_snapshot": fixture_info.get("raw"),
        "fixture_quality_report": fixture_info.get("quality"),
    }
    manifest: dict[str, Any] = {
        "schema_version": RELEASE_VERIFY_SCHEMA_VERSION,
        "command": "matchquant release-verify",
        "generated_at": _utc_now(),
        "status": status,
        "status_rules": {
            "ready": "所有硬完整性与时间核验通过，且没有 legacy prediction hash、selection mismatch 或缺失版本/cutoff 等需人工注意证据",
            "needs_review": "未发现硬矛盾，但存在旧 M4 未记录 prediction hash、selection history mismatch 或不完整版本证据；不得直接发布",
            "blocked": "存在 hash/path/rows/schema/quality/raw/时间等硬矛盾或无法读取的关键证据；不得发布",
            "final_decision": "M8 不执行人工审批；最终发布决定仍由用户作出",
        },
        "inputs": inputs,
        "as_of": forecast["as_of_meta"] if forecast else None,
        "prediction_hash_evidence": next((item["evidence"] for item in checks if item["id"] == "m4_prediction_hash"), None),
        "selection_evidence": next((item["evidence"] for item in checks if item["id"] == "selection_evidence"), None),
        "checks": checks,
        "outputs": {"release_directory": str(run_dir), "release_manifest": str(manifest_path), "release_report": str(report_path), "forecast_predictions": str(forecast["predictions_path"]) if forecast else None, "forecast_predictions_copied": False},
        "input_mutation": {"forecast_run_modified": False, "history_csv_modified": False, "history_manifest_modified": False, "fixture_snapshot_modified": False, "fixture_raw_modified": False, "fixture_quality_report_modified": False, "matchquant_data_modified": False, "current_pointer_updated": False},
        "limitations": ["本包只核验用户显式指定的本地 M4/M2/M2.1/M5 证据，不下载、不调用同步/准备命令，也不重跑 M4。", "history cutoff 在 as-of 之前不代表历史数据已最新；M5 snapshot 日期、复制时间或 source URL 不代表实时性。", "未执行人工审批；M8 不代表实时预测、概率已校准、盈利能力或投注建议，也不建立 source 再分发授权。"],
        "disclaimer": f"{FORECAST_DISCLAIMER} M8 是发布前可追溯性审计，不是实时性或盈利性证明。",
    }
    from .release_report import write_release_report

    write_release_report(manifest, report_path)
    manifest["outputs"]["release_report_sha256"] = file_sha256(report_path)
    write_json(manifest, manifest_path)
    return {"status": status, "run_dir": str(run_dir), "release_manifest_path": str(manifest_path), "release_report_path": str(report_path), "forecast_predictions_path": inputs["forecast_run"]["predictions"]["path"], "checks": checks, "prediction_hash_evidence": manifest["prediction_hash_evidence"], "selection_evidence": manifest["selection_evidence"], "manifest": manifest}


release_verify = verify_release


__all__ = ["RELEASE_STATUSES", "RELEASE_VERIFY_SCHEMA_VERSION", "release_verify", "verify_release"]
