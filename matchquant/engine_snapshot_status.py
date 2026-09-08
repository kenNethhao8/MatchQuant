"""Read-only M12 freshness status for one explicit M11 snapshot and pointer."""

from __future__ import annotations

import json
import os
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from .engine_snapshot_build_support import reject_output_root
from .engine_snapshot_manifest import load_engine_snapshot
from .io import file_sha256, write_json
from .schema import DataValidationError

ENGINE_SNAPSHOT_STATUS_SCHEMA_VERSION = "m12-engine-snapshot-status-v1"
ENGINE_SNAPSHOT_STATUS_PREFIX = "m12_engine_snapshot_status_"
CURRENT_POINTER_SCHEMA_VERSION = "m2.1-current-pointer-v1"
_HISTORY_MANIFEST_SCHEMAS = {"m2-processed-manifest-v1", "m2.1-processed-manifest-v1"}
_SHA256 = re.compile(r"^[0-9a-fA-F]{64}$")
_POINTER_FIELDS = (
    "schema_version",
    "dataset_version",
    "canonical",
    "processed_manifest",
    "canonical_sha256",
)
_EVIDENCE_ERRORS = (DataValidationError, FileNotFoundError, OSError, TypeError, ValueError, KeyError, UnicodeError)


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _valid_sha256(value: Any) -> bool:
    return isinstance(value, str) and _SHA256.fullmatch(value) is not None


def _same_path(left: str | Path, right: str | Path) -> bool:
    return os.path.normcase(str(Path(left).expanduser().resolve())) == os.path.normcase(
        str(Path(right).expanduser().resolve())
    )


def _resolve_path(value: Any, base: Path, label: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise DataValidationError(f"{label} path is missing")
    candidate = Path(value).expanduser()
    return (candidate if candidate.is_absolute() else base / candidate).resolve()


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeError) as exc:
        raise DataValidationError(f"could not read {label} '{path}': {exc}") from exc
    if not isinstance(payload, dict):
        raise DataValidationError(f"{label} root must be a JSON object: {path}")
    return payload


def _path_record(path: Path) -> dict[str, Any]:
    record: dict[str, Any] = {"path": str(path), "exists": False, "sha256": None}
    try:
        record["exists"] = path.is_file()
        if record["exists"]:
            record["sha256"] = file_sha256(path)
    except OSError as exc:
        record["error"] = str(exc)
    return record


def _invalid_record(path: Path, error: Exception) -> dict[str, Any]:
    record = _path_record(path)
    record.update(status="invalid", errors=[str(error)])
    return record


def _partial_pointer_record(pointer_path: Path) -> dict[str, Any]:
    """Retain explicit pointer target paths and observed hashes after validation fails."""

    record = _path_record(pointer_path)
    if not pointer_path.is_file():
        return record
    try:
        payload = _read_json(pointer_path, "current pointer")
    except _EVIDENCE_ERRORS:
        return record
    record.update(
        schema_version=payload.get("schema_version"),
        dataset_version=payload.get("dataset_version"),
    )
    for field, label in (("canonical", "canonical"), ("processed_manifest", "processed_manifest")):
        value = payload.get(field)
        if not isinstance(value, str) or not value.strip():
            continue
        try:
            target = _resolve_path(value, pointer_path.parent, f"current pointer {label}")
        except _EVIDENCE_ERRORS:
            continue
        target_record = _path_record(target)
        if field == "canonical":
            target_record["recorded_sha256"] = payload.get("canonical_sha256")
        record[field] = target_record
    return record


def _load_pointer(pointer_path: Path) -> dict[str, Any]:
    if not pointer_path.is_file():
        raise FileNotFoundError(f"current pointer does not exist: {pointer_path}")
    payload = _read_json(pointer_path, "current pointer")
    missing = [field for field in _POINTER_FIELDS if field not in payload]
    if missing:
        raise DataValidationError(f"current pointer is missing required fields: {', '.join(missing)}")
    if payload.get("schema_version") != CURRENT_POINTER_SCHEMA_VERSION:
        raise DataValidationError(f"unsupported current pointer schema: {pointer_path}")
    if not isinstance(payload.get("dataset_version"), str) or not payload["dataset_version"].strip():
        raise DataValidationError("current pointer dataset_version must be a non-empty string")
    if not _valid_sha256(payload.get("canonical_sha256")):
        raise DataValidationError("current pointer canonical_sha256 is missing or invalid")

    canonical_path = _resolve_path(payload["canonical"], pointer_path.parent, "current pointer canonical")
    manifest_path = _resolve_path(
        payload["processed_manifest"], pointer_path.parent, "current pointer processed_manifest"
    )
    if not canonical_path.is_file():
        raise FileNotFoundError(f"current pointer canonical does not exist: {canonical_path}")
    if not manifest_path.is_file():
        raise FileNotFoundError(f"current pointer processed manifest does not exist: {manifest_path}")

    canonical_hash = file_sha256(canonical_path)
    if canonical_hash.casefold() != str(payload["canonical_sha256"]).casefold():
        raise DataValidationError(
            f"current pointer canonical SHA-256 mismatch: pointer={payload['canonical_sha256']} actual={canonical_hash}"
        )

    processed_manifest = _read_json(manifest_path, "current pointer processed manifest")
    if processed_manifest.get("schema_version") not in _HISTORY_MANIFEST_SCHEMAS:
        raise DataValidationError("current pointer processed manifest schema is unsupported")
    if processed_manifest.get("dataset_version") != payload["dataset_version"]:
        raise DataValidationError("current pointer dataset_version does not match processed manifest")
    output = processed_manifest.get("output")
    if not isinstance(output, Mapping):
        raise DataValidationError("current pointer processed manifest output evidence is missing")
    output_hash = output.get("sha256")
    if not _valid_sha256(output_hash):
        raise DataValidationError("current pointer processed manifest output hash is missing or invalid")
    output_path = _resolve_path(output.get("filename"), manifest_path.parent, "processed manifest output")
    if not _same_path(output_path, canonical_path):
        raise DataValidationError("current pointer processed manifest output does not match canonical")
    if str(output_hash).casefold() != canonical_hash.casefold():
        raise DataValidationError("processed manifest output hash does not match canonical")

    return {
        "status": "valid",
        "path": str(pointer_path),
        "sha256": file_sha256(pointer_path),
        "schema_version": payload["schema_version"],
        "dataset_version": payload["dataset_version"],
        "canonical": {
            "path": str(canonical_path),
            "sha256": canonical_hash,
            "recorded_sha256": payload["canonical_sha256"],
        },
        "processed_manifest": {
            "path": str(manifest_path),
            "sha256": file_sha256(manifest_path),
            "dataset_version": processed_manifest.get("dataset_version"),
            "data_cutoff": processed_manifest.get("data_cutoff"),
            "output_sha256": output_hash,
        },
    }


def _load_snapshot(snapshot_manifest_path: Path) -> dict[str, Any]:
    loaded = load_engine_snapshot(snapshot_manifest_path)
    manifest = loaded["manifest"]
    history = manifest["history"]
    history_manifest = manifest["history_manifest"]
    history_path = _resolve_path(history["path"], snapshot_manifest_path.parent, "snapshot history")
    history_manifest_path = _resolve_path(
        history_manifest["path"], snapshot_manifest_path.parent, "snapshot history manifest"
    )
    return {
        "status": "valid",
        "path": str(snapshot_manifest_path),
        "sha256": loaded["manifest_sha256"],
        "snapshot_id": manifest["snapshot_id"],
        "schema_version": manifest["schema_version"],
        "history": {
            "path": str(history_path),
            "sha256": history["sha256"],
            "data_cutoff": history.get("data_cutoff"),
            "used_data_cutoff": history.get("used_data_cutoff"),
        },
        "history_manifest": {
            "path": str(history_manifest_path),
            "sha256": history_manifest["sha256"],
            "dataset_version": history_manifest.get("dataset_version"),
            "data_cutoff": history_manifest.get("data_cutoff"),
            "output_sha256": history_manifest.get("output_sha256"),
        },
        "state_sha256": loaded["state_sha256"],
    }


def _compare(snapshot: Mapping[str, Any], pointer: Mapping[str, Any]) -> tuple[str, dict[str, Any], list[str]]:
    snapshot_history = snapshot["history"]
    snapshot_manifest = snapshot["history_manifest"]
    pointer_canonical = pointer["canonical"]
    pointer_manifest = pointer["processed_manifest"]
    comparisons = {
        "canonical": {
            "path_match": _same_path(snapshot_history["path"], pointer_canonical["path"]),
            "hash_match": str(snapshot_history["sha256"]).casefold() == str(pointer_canonical["sha256"]).casefold(),
            "snapshot_path": snapshot_history["path"],
            "pointer_path": pointer_canonical["path"],
            "snapshot_sha256": snapshot_history["sha256"],
            "pointer_sha256": pointer_canonical["sha256"],
        },
        "processed_manifest": {
            "path_match": _same_path(snapshot_manifest["path"], pointer_manifest["path"]),
            "hash_match": str(snapshot_manifest["sha256"]).casefold() == str(pointer_manifest["sha256"]).casefold(),
            "snapshot_path": snapshot_manifest["path"],
            "pointer_path": pointer_manifest["path"],
            "snapshot_sha256": snapshot_manifest["sha256"],
            "pointer_sha256": pointer_manifest["sha256"],
        },
        "dataset_version": {
            "match": snapshot_manifest.get("dataset_version") == pointer.get("dataset_version"),
            "snapshot": snapshot_manifest.get("dataset_version"),
            "pointer": pointer.get("dataset_version"),
        },
    }
    reasons: list[str] = []
    if not snapshot_manifest.get("dataset_version"):
        reasons.append("snapshot_dataset_version_missing")
    if not comparisons["canonical"]["path_match"]:
        reasons.append("canonical_path_mismatch")
    if not comparisons["canonical"]["hash_match"]:
        reasons.append("canonical_hash_mismatch")
    if not comparisons["processed_manifest"]["path_match"]:
        reasons.append("processed_manifest_path_mismatch")
    if not comparisons["processed_manifest"]["hash_match"]:
        reasons.append("processed_manifest_hash_mismatch")
    if not comparisons["dataset_version"]["match"]:
        reasons.append("dataset_version_mismatch")
    if not reasons:
        return "fresh", comparisons, reasons
    if "snapshot_dataset_version_missing" in reasons:
        return "unverifiable", comparisons, reasons
    return "stale", comparisons, reasons


def _new_audit_dir(output_root: str | Path, snapshot_manifest_path: Path, pointer_path: Path) -> Path:
    root = Path(output_root).expanduser().resolve()
    reject_output_root(root, snapshot_manifest_path.parent, pointer_path)
    root.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    audit_dir = root / f"{ENGINE_SNAPSHOT_STATUS_PREFIX}{stamp}_{uuid.uuid4().hex[:8]}"
    audit_dir.mkdir()
    return audit_dir


def _write_report(path: Path, audit: Mapping[str, Any]) -> Path:
    snapshot = audit["snapshot"]
    pointer = audit["current_pointer"]
    lines = [
        "# MatchQuant M12 engine snapshot status",
        "",
        f"- status：`{audit['status']}`",
        f"- snapshot manifest：`{snapshot['path']}`",
        f"- current pointer：`{pointer['path']}`",
        "",
    ]
    if audit.get("reasons"):
        lines.extend(["## Reasons", ""])
        lines.extend(f"- `{reason}`" for reason in audit["reasons"])
        lines.append("")
    if audit.get("comparison"):
        lines.extend(
            [
                "## Evidence",
                "",
                f"- canonical：path_match={audit['comparison']['canonical']['path_match']}；hash_match={audit['comparison']['canonical']['hash_match']}",
                f"- processed manifest：path_match={audit['comparison']['processed_manifest']['path_match']}；hash_match={audit['comparison']['processed_manifest']['hash_match']}",
                f"- dataset version：snapshot={audit['comparison']['dataset_version']['snapshot']}；pointer={audit['comparison']['dataset_version']['pointer']}",
                "",
            ]
        )
    lines.extend(
        [
            "## Scope",
            "",
            "本命令只读取用户显式提供的 M11 snapshot manifest、M2.1 current pointer 及其证据文件；不联网、不调用同步流程、不更新 pointer、不修改历史或快照，不自动寻找 latest 或重建快照。",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def check_engine_snapshot_status(
    snapshot_manifest_path: str | Path,
    current_pointer_path: str | Path,
    *,
    output_root: str | Path = Path(r"D:\Codex\matchquant_runs"),
) -> dict[str, Any]:
    """Audit one explicit M11 snapshot against one explicit M2.1 current pointer."""

    snapshot_path = Path(snapshot_manifest_path).expanduser().resolve()
    pointer_path = Path(current_pointer_path).expanduser().resolve()
    audit_dir = _new_audit_dir(output_root, snapshot_path, pointer_path)
    try:
        snapshot = _load_snapshot(snapshot_path)
    except _EVIDENCE_ERRORS as exc:
        snapshot = _invalid_record(snapshot_path, exc)
    try:
        pointer = _load_pointer(pointer_path)
    except _EVIDENCE_ERRORS as exc:
        pointer = _partial_pointer_record(pointer_path)
        pointer.update(status="invalid", errors=[str(exc)])

    comparison: dict[str, Any] | None = None
    reasons: list[str] = []
    if snapshot.get("status") == "valid" and pointer.get("status") == "valid":
        status, comparison, reasons = _compare(snapshot, pointer)
    else:
        status = "unverifiable"
        if snapshot.get("status") != "valid":
            reasons.append("snapshot_evidence_invalid")
        if pointer.get("status") != "valid":
            reasons.append("current_pointer_invalid")

    status_manifest_path = audit_dir / "snapshot_status.json"
    status_report_path = audit_dir / "snapshot_status.md"
    audit: dict[str, Any] = {
        "schema_version": ENGINE_SNAPSHOT_STATUS_SCHEMA_VERSION,
        "command": "matchquant engine-snapshot-status",
        "generated_at": _utc_now(),
        "status": status,
        "snapshot": snapshot,
        "current_pointer": pointer,
        "comparison": comparison,
        "reasons": reasons,
        "validation": {
            "inputs_unchanged": True,
            "snapshot_mutated": False,
            "current_pointer_mutated": False,
            "history_mutated": False,
            "network_accessed": False,
            "automatic_latest_lookup": False,
            "automatic_apply": False,
            "automatic_rebuild": False,
        },
        "outputs": {
            "audit_dir": str(audit_dir),
            "status_manifest": str(status_manifest_path),
            "status_report": str(status_report_path),
        },
        "limitations": [
            "fresh 只表示显式 snapshot history/history-manifest 证据与显式 current pointer 的路径、hash 和 dataset version 精确匹配。",
            "stale 快照仍可由现有 M11 predict 独立复现；本命令不自动重建快照。",
            "data cutoff 和本地 active pointer 不代表互联网实时性。",
        ],
    }
    write_json(audit, status_manifest_path)
    _write_report(status_report_path, audit)
    return {
        **audit,
        "audit_dir": str(audit_dir),
        "status_manifest_path": str(status_manifest_path),
        "status_report_path": str(status_report_path),
        "status_manifest_sha256": file_sha256(status_manifest_path),
    }


__all__ = [
    "CURRENT_POINTER_SCHEMA_VERSION",
    "ENGINE_SNAPSHOT_STATUS_PREFIX",
    "ENGINE_SNAPSHOT_STATUS_SCHEMA_VERSION",
    "check_engine_snapshot_status",
]
