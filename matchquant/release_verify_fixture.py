"""Read-only M5 fixture evidence checks used by the M8 audit."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from .epl import read_source_csv
from .epl_fixtures import FIXTURE_MANIFEST_VERSION, convert_fixture_source_frame
from .fixtures import FIXTURE_COLUMNS, FIXTURE_SCHEMA_VERSION, load_fixtures
from .io import file_sha256
from .schema import DataValidationError

_SHA256 = re.compile(r"^[0-9a-fA-F]{64}$")


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DataValidationError(f"could not read {label} '{path}': {exc}") from exc
    if not isinstance(payload, dict):
        raise DataValidationError(f"{label} root must be a JSON object: {path}")
    return payload


def _path(value: object, base: Path) -> Path | None:
    if value is None or str(value).strip() == "":
        return None
    candidate = Path(str(value))
    return (candidate if candidate.is_absolute() else base / candidate).resolve()


def _valid_hash(value: object) -> bool:
    return isinstance(value, str) and _SHA256.fullmatch(value) is not None


def _within(child: Path, parent: Path) -> bool:
    return child == parent or parent in child.parents


def _check(check_id: str, status: str, summary: str, reasons: list[str] | None = None, **evidence: Any) -> dict[str, Any]:
    return {"id": check_id, "status": status, "summary": summary, "reasons": reasons or [], "evidence": evidence}


def _descriptor(payload: Mapping[str, Any]) -> dict[str, Any]:
    raw = payload.get("output")
    output = raw if isinstance(raw, Mapping) else {}
    path = next((output.get(key) for key in ("filename", "path", "canonical") if output.get(key)), None)
    sha256 = next((output.get(key) for key in ("sha256", "canonical_sha256") if output.get(key)), payload.get("canonical_output_sha256"))
    rows = next((output.get(key) for key in ("rows", "row_count") if output.get(key) is not None), payload.get("canonical_output_row_count"))
    return {"path": path, "sha256": sha256, "rows": rows, "columns": output.get("columns")}


def _file_record(path: Path | None) -> dict[str, Any]:
    record = {"path": str(path) if path else None, "exists": bool(path and path.is_file()), "sha256": None}
    if path and path.is_file():
        record["sha256"] = file_sha256(path)
    return record


def fixture_snapshot_checks(manifest_path: Path, forecast: dict[str, Any] | None) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Validate M5 manifest, raw source, quality report, and pending output without writing."""
    info: dict[str, Any] = {"manifest_path": str(manifest_path), "manifest_sha256": file_sha256(manifest_path) if manifest_path.is_file() else None}
    try:
        payload = _read_json(manifest_path, "fixture snapshot manifest")
    except (DataValidationError, OSError, ValueError) as exc:
        return [_check("fixture_manifest_structure", "blocked", "M5 fixture snapshot manifest 无法使用", [str(exc)])], info

    snapshot_id = payload.get("snapshot_id")
    snapshot_dir = _path(payload.get("snapshot_dir"), manifest_path.parent) or manifest_path.parent
    info.update({"schema_version": payload.get("schema_version"), "snapshot_id": snapshot_id, "snapshot_dir": str(snapshot_dir)})
    reasons: list[str] = []
    if payload.get("schema_version") != FIXTURE_MANIFEST_VERSION:
        reasons.append(f"不是 M5 fixture snapshot manifest: {payload.get('schema_version')!r}")
    if payload.get("fixture_schema_version") != FIXTURE_SCHEMA_VERSION:
        reasons.append("M5 fixture_schema_version 与 M4 score-free schema 不一致")
    if not isinstance(snapshot_id, str) or not snapshot_id.strip():
        reasons.append("M5 manifest 缺少 snapshot_id")
    if manifest_path.parent != snapshot_dir or manifest_path.parent.name != snapshot_id:
        reasons.append("snapshot_id、snapshot_dir 与 manifest 实际目录不一致")
    checks = [_check("fixture_manifest_identity", "blocked" if reasons else "passed", "M5 snapshot 身份字段已核对", reasons, snapshot_id=snapshot_id, snapshot_dir=str(snapshot_dir))]

    descriptor = _descriptor(payload)
    output_path = _path(descriptor["path"], manifest_path.parent)
    top_hash, top_rows = payload.get("canonical_output_sha256"), payload.get("canonical_output_row_count")
    info["output"] = {**descriptor, "path": str(output_path) if output_path else None, "canonical_output_sha256": top_hash, "canonical_output_row_count": top_rows, "actual_sha256": None, "actual_rows": None}
    output_frame: pd.DataFrame | None = None
    reasons = []
    if output_path is None or not output_path.is_file():
        reasons.append("M5 manifest output 文件缺失")
    elif not _within(output_path, snapshot_dir):
        reasons.append("M5 output 不在其 snapshot_dir 内")
    if not _valid_hash(descriptor["sha256"]) or not _valid_hash(top_hash) or str(descriptor["sha256"]).lower() != str(top_hash).lower():
        reasons.append("M5 output.sha256 与 canonical_output_sha256 缺失或不一致")
    if not isinstance(descriptor["rows"], int) or isinstance(descriptor["rows"], bool) or descriptor["rows"] < 0 or top_rows != descriptor["rows"]:
        reasons.append("M5 output 行数与 canonical_output_row_count 缺失或不一致")
    if descriptor["columns"] is not None and descriptor["columns"] != list(FIXTURE_COLUMNS):
        reasons.append("M5 output.columns 与 score-free fixture schema 不一致")
    actual_hash = None
    if output_path and output_path.is_file():
        actual_hash = file_sha256(output_path)
        info["output"]["actual_sha256"] = actual_hash
        try:
            output_frame = load_fixtures(output_path)
        except (DataValidationError, OSError, ValueError) as exc:
            reasons.append(f"M5 canonical fixture CSV 无法通过 M4 schema 校验: {exc}")
        if not _valid_hash(descriptor["sha256"]) or actual_hash.lower() != str(descriptor["sha256"]).lower():
            reasons.append("M5 output SHA-256 与实际 fixture CSV 不一致")
        if output_frame is not None and len(output_frame) != descriptor["rows"]:
            reasons.append(f"M5 output.rows={descriptor['rows']}，实际 fixture 行数={len(output_frame)}")
        if output_frame is not None:
            info["output"]["actual_rows"] = len(output_frame)
    if forecast is not None:
        declared = forecast["config"].get("fixtures")
        info["m4_fixtures"] = declared
        if not isinstance(declared, Mapping):
            reasons.append("M4 run_config 缺少 fixtures object")
        else:
            if _path(declared.get("path"), forecast["run_dir"]) != output_path:
                reasons.append("M4 run_config fixtures.path 与 M5 output 路径不一致")
            if declared.get("schema") != f"{FIXTURE_SCHEMA_VERSION}-score-free":
                reasons.append("M4 run_config fixtures.schema 与 M5 fixture schema 不一致")
            if not _valid_hash(declared.get("sha256")) or str(declared.get("sha256")).lower() != str(actual_hash or "").lower():
                reasons.append("M4 run_config fixtures.sha256 与 M5 output 实际 hash 不一致")
            if output_frame is not None and declared.get("rows") != len(output_frame):
                reasons.append("M4 run_config fixtures.rows 与 M5 output 行数不一致")
    info["frame"] = output_frame
    checks.append(_check("fixture_output_reference", "blocked" if reasons else "passed", "M5 output 与 M4 fixtures 引用、schema、hash、行数已核对", reasons, output=info["output"], actual_sha256=actual_hash))

    raw_path, raw_hash = _path(payload.get("source_raw_snapshot"), manifest_path.parent), payload.get("source_sha256")
    info["raw"] = {**_file_record(raw_path), "declared_sha256": raw_hash}
    reasons = []
    if raw_path is None or not raw_path.is_file():
        reasons.append("M5 source_raw_snapshot 缺失")
    elif not _within(raw_path, snapshot_dir):
        reasons.append("M5 source_raw_snapshot 不在其 snapshot_dir 内")
    if not _valid_hash(raw_hash):
        reasons.append("M5 source_sha256 缺失或无效")
    if raw_path and raw_path.is_file() and _valid_hash(raw_hash) and file_sha256(raw_path).lower() != str(raw_hash).lower():
        reasons.append("M5 source_raw_snapshot 实际 hash 与 source_sha256 不一致")
    checks.append(_check("fixture_raw_evidence", "blocked" if reasons else "passed", "M5 raw source 快照与 hash 已核对", reasons, raw=info["raw"]))

    quality_path, quality_hash = _path(payload.get("quality_report"), manifest_path.parent), payload.get("quality_report_sha256")
    quality: dict[str, Any] | None = None
    info["quality"] = {"path": str(quality_path) if quality_path else None, "declared_sha256": quality_hash, "sha256": None}
    reasons = []
    if quality_path is None or not quality_path.is_file():
        reasons.append("M5 quality_report 文件缺失")
    elif not _within(quality_path, snapshot_dir):
        reasons.append("M5 quality_report 不在其 snapshot_dir 内")
    if not _valid_hash(quality_hash):
        reasons.append("M5 quality_report_sha256 缺失或无效")
    if quality_path and quality_path.is_file():
        info["quality"]["sha256"] = file_sha256(quality_path)
        if _valid_hash(quality_hash) and info["quality"]["sha256"].lower() != str(quality_hash).lower():
            reasons.append("M5 quality report 实际 hash 与 manifest 不一致")
        try:
            quality = _read_json(quality_path, "M5 quality report")
        except DataValidationError as exc:
            reasons.append(str(exc))
    info["quality_payload"] = quality
    if quality is not None:
        if quality.get("snapshot_id") != snapshot_id:
            reasons.append("quality report snapshot_id 与 M5 manifest 不一致")
        if quality.get("source_sha256") != raw_hash:
            reasons.append("quality report source_sha256 与 M5 manifest 不一致")
        quality_raw = _path(quality.get("source_raw_snapshot"), quality_path.parent) if quality.get("source_raw_snapshot") else None
        if quality_raw != raw_path:
            reasons.append("quality report source_raw_snapshot 与 M5 manifest 不一致")
        if quality.get("output") != payload.get("output"):
            reasons.append("quality report output 字段与 M5 manifest 不一致")
        if quality.get("status") != payload.get("status") or quality.get("status") != "ok":
            reasons.append("M5 quality/status 不是可供 M4 使用的 ok")
        for field in ("source_date_range", "completed_date_range", "pending_date_range", "completed_data_cutoff", "data_cutoff", "data_cutoff_status", "counts"):
            if quality.get(field) != payload.get(field):
                reasons.append(f"quality {field} 与 M5 manifest 不一致")
        if output_frame is not None:
            pending = quality.get("pending_date_range")
            actual_range = {"earliest": str(output_frame["date"].min()), "latest": str(output_frame["date"].max())}
            if not isinstance(pending, Mapping) or dict(pending) != actual_range:
                reasons.append(f"quality pending_date_range={pending!r} 与实际 fixture 日期范围={actual_range!r} 不一致")
    checks.append(_check("fixture_quality_evidence", "blocked" if reasons else "passed", "M5 quality report 的 hash 与输出字段已核对", reasons, quality=info["quality"]))

    if raw_path and raw_path.is_file() and output_frame is not None and isinstance(payload.get("season"), str):
        try:
            expected, audit = convert_fixture_source_frame(read_source_csv(raw_path), payload["season"])
            left = output_frame.loc[:, FIXTURE_COLUMNS].sort_values(["date", "fixture_id"]).reset_index(drop=True).astype(str)
            right = expected.loc[:, FIXTURE_COLUMNS].sort_values(["date", "fixture_id"]).reset_index(drop=True).astype(str)
            reasons = [] if left.equals(right) else ["M5 raw source 重算出的 pending fixture rows 与 canonical output 不一致"]
            for key in ("pending_rows", "completed_rows", "invalid_rows", "duplicate_stable_fixture_ids"):
                reported = quality.get(key) if isinstance(quality, Mapping) else None
                if reported is None and isinstance(quality, Mapping) and isinstance(quality.get("counts"), Mapping):
                    reported = quality["counts"].get(key)
                if reported != audit.get(key):
                    reasons.append(f"quality {key}={reported!r} 与 raw source 重算值={audit.get(key)!r} 不一致")
            for field in ("source_date_range", "completed_date_range", "pending_date_range", "completed_data_cutoff", "data_cutoff", "data_cutoff_status"):
                quality_value = quality.get(field) if isinstance(quality, Mapping) else None
                if quality_value != audit.get(field) or payload.get(field) != audit.get(field):
                    reasons.append(f"M5 {field} 与 raw source 重算值不一致")
            checks.append(_check("fixture_raw_output_self_consistency", "blocked" if reasons else "passed", "M5 raw source、quality report 与 fixture output 可相互复核", reasons, recomputed_pending_rows=len(expected), recomputed_completed_data_cutoff=audit.get("completed_data_cutoff"), pending_date_range=audit.get("pending_date_range")))
        except (DataValidationError, OSError, TypeError, ValueError) as exc:
            checks.append(_check("fixture_raw_output_self_consistency", "blocked", "M5 raw source 无法与 output 重算核对", [str(exc)]))
    else:
        checks.append(_check("fixture_raw_output_self_consistency", "blocked", "M5 raw source/output 缺少可重算证据", ["raw source、fixture output 或 season 字段缺失"]))
    return checks, info


__all__ = ["fixture_snapshot_checks"]
