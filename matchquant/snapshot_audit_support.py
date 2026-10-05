"""Shared read-only audit support for B03 snapshot sessions and run audits.

This module binds an optional M12 issue-evidence record to a strictly loaded
M11 snapshot, and enforces output isolation for snapshot sessions and audits.
It never opens history CSVs, history processed manifests, current pointers, or
any research-side path recorded inside an issue-evidence file; recorded paths
are compared in normalized form only.
"""

from __future__ import annotations

import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from .io import file_sha256
from .schema import DataValidationError

SESSION_SCHEMA_VERSION = "baseline-snapshot-session-v1"
AUDIT_SCHEMA_VERSION = "baseline-snapshot-run-audit-v1"
ISSUE_EVIDENCE_SCHEMA_VERSION = "m12-engine-snapshot-status-v1"
ACTIVE_LINK_STATUS = "not_checked_in_runtime"
SESSION_DIR_PREFIX = "baseline_snapshot_session_"
AUDIT_DIR_PREFIX = "baseline_snapshot_audit_"
EXPECTED_STAGE_ERRORS = (DataValidationError, FileNotFoundError, OSError, TypeError, ValueError)
FORMAL_ROOTS = (Path(r"D:\Codex\matchquant_data"), Path(r"D:\Codex\matchquant_runs"))
_M12_RECORD_STATUSES = ("fresh", "stale", "unverifiable")


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def read_json_object(path: Path, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise DataValidationError(f"could not read {label} '{path}': {exc}") from exc
    if not isinstance(payload, dict):
        raise DataValidationError(f"{label} root must be a JSON object: {path}")
    return payload


def _normalized_path(value: Any) -> str:
    return os.path.normcase(str(Path(str(value)).expanduser().resolve()))


def same_recorded_path(left: Any, right: Any) -> bool:
    """Compare two recorded paths after normalization only; never touch the files."""

    try:
        return _normalized_path(left) == _normalized_path(right)
    except (TypeError, ValueError, OSError):
        return False


def ensure_safe_output_root(output_root: str | Path, inputs: Sequence[str | Path]) -> Path:
    """Reject output roots that collide with formal roots, inputs, or prior sessions.

    For directory inputs (such as a prediction run directory) the root must not be
    the input itself or inside it; for file inputs the root must not be the file,
    its containing archive directory, or inside that directory. Being a sibling
    of an input under a common parent stays allowed.
    """

    root = Path(output_root).expanduser().resolve()
    for formal in FORMAL_ROOTS:
        if root == formal or formal in root.parents:
            raise DataValidationError(
                f"output root must not be inside the formal data or runs root: {root}"
            )
    for raw in inputs:
        input_path = Path(raw).expanduser().resolve()
        if root == input_path:
            raise DataValidationError(f"output root cannot be an input path: {root}")
        if input_path.is_dir():
            if input_path in root.parents:
                raise DataValidationError(f"output root cannot be inside an input directory: {root}")
        else:
            parent = input_path.parent
            if root == parent or parent in root.parents:
                raise DataValidationError(f"output root cannot be inside an input archive directory: {root}")
    for parent in (root, *root.parents):
        if parent.name.startswith(SESSION_DIR_PREFIX) or parent.name.startswith(AUDIT_DIR_PREFIX):
            raise DataValidationError(
                "output root must not be inside an existing snapshot session or audit directory: "
                f"{parent}"
            )
        if parent.is_dir() and parent.name.startswith("m11_engine_snapshot_"):
            raise DataValidationError(
                f"output root must not be inside an existing M11 snapshot or prediction run directory: {parent}"
            )
    return root


def new_output_dir(output_root: str | Path, prefix: str) -> Path:
    root = Path(output_root).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_dir = root / f"{prefix}{stamp}_{uuid.uuid4().hex[:8]}"
    output_dir.mkdir()
    return output_dir


def _parse_utc_z(value: Any, label: str) -> datetime:
    if not isinstance(value, str) or not value.endswith("Z"):
        raise DataValidationError(f"issue evidence {label} must be a UTC ISO-8601 string ending with Z")
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as exc:
        raise DataValidationError(f"issue evidence {label} is not a valid UTC timestamp: {value!r}") from exc
    if parsed.utcoffset() != timezone.utc.utcoffset(parsed):
        raise DataValidationError(f"issue evidence {label} must be UTC")
    return parsed


def _require_mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise DataValidationError(f"issue evidence {label} must be an object")
    return value


def _require_text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise DataValidationError(f"issue evidence {label} is missing or empty")
    return value


def _comparison_flags(comparison: Mapping[str, Any]) -> dict[str, bool]:
    canonical = _require_mapping(comparison.get("canonical"), "comparison.canonical")
    processed = _require_mapping(comparison.get("processed_manifest"), "comparison.processed_manifest")
    dataset = _require_mapping(comparison.get("dataset_version"), "comparison.dataset_version")
    return {
        "canonical_path_match": canonical.get("path_match") is True,
        "canonical_hash_match": canonical.get("hash_match") is True,
        "processed_manifest_path_match": processed.get("path_match") is True,
        "processed_manifest_hash_match": processed.get("hash_match") is True,
        "dataset_version_match": dataset.get("match") is True,
    }


def missing_issue_evidence() -> dict[str, Any]:
    return {
        "provided": False,
        "path": None,
        "sha256": None,
        "schema_version": None,
        "record_status": "missing",
        "record_generated_at": None,
        "record_reasons": [],
        "comparison_flags": None,
        "fresh_at_issue": False,
        "active_link_status": ACTIVE_LINK_STATUS,
        "pointer_status": None,
        "pointer_dataset_version": None,
        "source_access": "no_issue_evidence_provided",
        "limitations": [
            "未提供发行证据；历史发行状态缺失，不能伪造 lineage，也不能当作 fresh。"
        ],
    }


def load_issue_evidence(
    issue_evidence_path: str | Path,
    *,
    snapshot_id: str,
    manifest_sha256: str,
    state_sha256: str,
    history: Mapping[str, Any],
    history_manifest: Mapping[str, Any],
    audited_at: str,
) -> dict[str, Any]:
    """Validate one optional M12 ``m12-engine-snapshot-status-v1`` record against this snapshot.

    The record's file hash is computed and recorded; structure, field types, UTC
    ``generated_at`` (not after ``audited_at``), target binding, and internal
    consistency are checked. Paths recorded inside the evidence are compared in
    normalized form only and are never opened, so research-side files may stay
    disconnected. A malformed record, an unknown schema, a snapshot target
    mismatch, a future review time, or an internal contradiction raises
    ``DataValidationError`` (blocked); stale/unverifiable records are returned
    as historical limitations and must never be upgraded to fresh.
    """

    path = Path(issue_evidence_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"issue evidence JSON does not exist: {path}")
    evidence_hash = file_sha256(path)
    payload = read_json_object(path, "issue evidence")
    if payload.get("schema_version") != ISSUE_EVIDENCE_SCHEMA_VERSION:
        raise DataValidationError(
            "unsupported issue evidence schema_version; expected "
            f"{ISSUE_EVIDENCE_SCHEMA_VERSION}: {payload.get('schema_version')!r}"
        )
    record_status = payload.get("status")
    if record_status not in _M12_RECORD_STATUSES:
        raise DataValidationError(f"issue evidence record status is unknown: {record_status!r}")
    generated_at_text = _require_text(payload.get("generated_at"), "generated_at")
    generated_at = _parse_utc_z(generated_at_text, "generated_at")
    audited = _parse_utc_z(audited_at, "audited_at")
    if generated_at > audited:
        raise DataValidationError(
            "issue evidence generated_at is later than the audited_at of this audit: "
            f"{generated_at_text} > {audited_at}"
        )

    snapshot_block = _require_mapping(payload.get("snapshot"), "snapshot block")
    record_snapshot_id = _require_text(snapshot_block.get("snapshot_id"), "snapshot.snapshot_id")
    if record_snapshot_id != snapshot_id:
        raise DataValidationError(
            "issue evidence targets a different snapshot_id: "
            f"evidence={record_snapshot_id!r} audited={snapshot_id!r}"
        )
    record_manifest_hash = _require_text(snapshot_block.get("sha256"), "snapshot.sha256")
    if record_manifest_hash.casefold() != str(manifest_sha256).casefold():
        raise DataValidationError(
            "issue evidence recorded manifest SHA-256 does not match the audited snapshot manifest file"
        )
    record_state_hash = _require_text(snapshot_block.get("state_sha256"), "snapshot.state_sha256")
    if record_state_hash.casefold() != str(state_sha256).casefold():
        raise DataValidationError(
            "issue evidence recorded state SHA-256 does not match the audited snapshot state file"
        )
    recorded_history = _require_mapping(snapshot_block.get("history"), "snapshot.history")
    if not same_recorded_path(recorded_history.get("path"), history.get("path")):
        raise DataValidationError(
            "issue evidence recorded history path does not match the audited snapshot history path"
        )
    if str(recorded_history.get("sha256", "")).casefold() != str(history.get("sha256", "")).casefold():
        raise DataValidationError(
            "issue evidence recorded history SHA-256 does not match the audited snapshot history record"
        )
    recorded_history_manifest = _require_mapping(
        snapshot_block.get("history_manifest"), "snapshot.history_manifest"
    )
    if not same_recorded_path(recorded_history_manifest.get("path"), history_manifest.get("path")):
        raise DataValidationError(
            "issue evidence recorded history-manifest path does not match the audited snapshot record"
        )
    if str(recorded_history_manifest.get("sha256", "")).casefold() != str(
        history_manifest.get("sha256", "")
    ).casefold():
        raise DataValidationError(
            "issue evidence recorded history-manifest SHA-256 does not match the audited snapshot record"
        )

    comparison = payload.get("comparison")
    flags: dict[str, bool] | None = None
    if comparison is not None:
        comparison = _require_mapping(comparison, "comparison")
        flags = _comparison_flags(comparison)
        canonical = _require_mapping(comparison.get("canonical"), "comparison.canonical")
        if not same_recorded_path(canonical.get("snapshot_path"), history.get("path")):
            raise DataValidationError(
                "issue evidence comparison snapshot-side canonical path does not match the audited snapshot"
            )
        if str(canonical.get("snapshot_sha256", "")).casefold() != str(history.get("sha256", "")).casefold():
            raise DataValidationError(
                "issue evidence comparison snapshot-side canonical hash does not match the audited snapshot"
            )
        processed = _require_mapping(comparison.get("processed_manifest"), "comparison.processed_manifest")
        if not same_recorded_path(processed.get("snapshot_path"), history_manifest.get("path")):
            raise DataValidationError(
                "issue evidence comparison snapshot-side manifest path does not match the audited snapshot"
            )
        if str(processed.get("snapshot_sha256", "")).casefold() != str(
            history_manifest.get("sha256", "")
        ).casefold():
            raise DataValidationError(
                "issue evidence comparison snapshot-side manifest hash does not match the audited snapshot"
            )
        dataset = _require_mapping(comparison.get("dataset_version"), "comparison.dataset_version")
        if str(dataset.get("snapshot")) != str(history_manifest.get("dataset_version")):
            raise DataValidationError(
                "issue evidence comparison snapshot-side dataset_version does not match the audited snapshot"
            )

    reasons = payload.get("reasons")
    if not isinstance(reasons, list) or any(not isinstance(item, str) for item in reasons):
        raise DataValidationError("issue evidence reasons must be a list of strings")

    pointer = payload.get("current_pointer")
    pointer_status: str | None = None
    pointer_dataset_version: str | None = None
    if record_status == "fresh":
        pointer = _require_mapping(payload.get("current_pointer"), "current_pointer")
        pointer_status = str(pointer.get("status"))
        pointer_dataset_version = (
            str(pointer.get("dataset_version")) if pointer.get("dataset_version") is not None else None
        )
        if pointer_status != "valid":
            raise DataValidationError(
                "issue evidence claims fresh but its current pointer record is not valid; "
                "a fresh history association cannot be trusted without valid pointer evidence"
            )
        if comparison is None:
            raise DataValidationError("issue evidence claims fresh but records no comparison evidence")
        assert flags is not None
        if not all(flags.values()):
            raise DataValidationError(
                "issue evidence claims fresh but its comparison flags disagree; handwritten fresh "
                "status is not trusted"
            )
        if reasons:
            raise DataValidationError(
                "issue evidence claims fresh but records reasons; handwritten fresh status is not trusted"
            )
        pointer_canonical = _require_mapping(pointer.get("canonical"), "current_pointer.canonical")
        pointer_manifest = _require_mapping(
            pointer.get("processed_manifest"), "current_pointer.processed_manifest"
        )
        comparison_canonical = _require_mapping(comparison.get("canonical"), "comparison.canonical")
        comparison_processed = _require_mapping(
            comparison.get("processed_manifest"), "comparison.processed_manifest"
        )
        comparison_dataset = _require_mapping(comparison.get("dataset_version"), "comparison.dataset_version")
        if str(pointer_canonical.get("sha256", "")).casefold() != str(
            comparison_canonical.get("pointer_sha256", "")
        ).casefold():
            raise DataValidationError(
                "issue evidence canonical pointer hash disagrees between the pointer record and comparison"
            )
        if str(pointer_manifest.get("sha256", "")).casefold() != str(
            comparison_processed.get("pointer_sha256", "")
        ).casefold():
            raise DataValidationError(
                "issue evidence processed-manifest pointer hash disagrees between records"
            )
        if str(pointer.get("dataset_version")) != str(comparison_dataset.get("pointer")):
            raise DataValidationError(
                "issue evidence pointer dataset_version disagrees between the pointer record and comparison"
            )
    else:
        if not reasons:
            raise DataValidationError(
                f"issue evidence records {record_status} without reasons; the record is internally inconsistent"
            )
        if flags is not None and all(flags.values()):
            raise DataValidationError(
                f"issue evidence records {record_status} but its comparison shows no mismatch"
            )
        if isinstance(pointer, Mapping):
            pointer_status = str(pointer.get("status"))
            pointer_dataset_version = (
                str(pointer.get("dataset_version")) if pointer.get("dataset_version") is not None else None
            )

    limitations = [
        "发行证据只做记录内部一致性与目标绑定核验；不打开其记录的 pointer/history/processed manifest 路径。",
        "记录的实际文件 hash 只证明本次输入一致性，不构成发行来源认证，也不能发现所有字段同步伪造的历史。",
    ]
    if record_status == "fresh":
        limitations.append(
            "fresh_at_issue 仅表示发行时点的历史记录关联验证；不证明当前 active、今天最新或训练到目标比赛。"
        )
    else:
        limitations.append(
            f"历史发行记录状态为 {record_status}；只能作为历史限制展示，不能升格为 fresh。"
        )
    return {
        "provided": True,
        "path": str(path),
        "sha256": evidence_hash,
        "schema_version": ISSUE_EVIDENCE_SCHEMA_VERSION,
        "record_status": record_status,
        "record_generated_at": generated_at_text,
        "record_reasons": list(reasons),
        "comparison_flags": flags,
        "fresh_at_issue": record_status == "fresh",
        "active_link_status": ACTIVE_LINK_STATUS,
        "pointer_status": pointer_status,
        "pointer_dataset_version": pointer_dataset_version,
        "source_access": "recorded_paths_compared_normalized_only_not_opened",
        "limitations": limitations,
    }


__all__ = [
    "ACTIVE_LINK_STATUS",
    "AUDIT_DIR_PREFIX",
    "AUDIT_SCHEMA_VERSION",
    "EXPECTED_STAGE_ERRORS",
    "FORMAL_ROOTS",
    "ISSUE_EVIDENCE_SCHEMA_VERSION",
    "SESSION_DIR_PREFIX",
    "SESSION_SCHEMA_VERSION",
    "ensure_safe_output_root",
    "load_issue_evidence",
    "missing_issue_evidence",
    "new_output_dir",
    "read_json_object",
    "same_recorded_path",
    "utc_now",
]
