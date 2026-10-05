"""Shared read-only audit support for B03 snapshot sessions and run audits.

This module binds an optional M12 issue-evidence record to a strictly loaded
M11 snapshot, and enforces output isolation for snapshot sessions and audits.

Issue-evidence validation recomputes every recorded equality from the record's
own snapshot-side and pointer-side values (paths, hashes, dataset_version,
cutoffs), binds both sides and the recorded cutoffs to the strictly loaded
manifest, and requires the recorded comparison flags to equal the recomputed
ones. Handwritten flags are never trusted, and bad field types never silently
pass as evidence. Recorded research-side paths are compared in normalized form
only and are never opened.
"""

from __future__ import annotations

import json
import os
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from .engine_snapshot_status import CURRENT_POINTER_SCHEMA_VERSION
from .engine_snapshot_support import ENGINE_SNAPSHOT_SCHEMA_VERSION
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
_SHA256_RE = re.compile(r"^[0-9a-fA-F]{64}$")

# Exact shapes of the frozen m12-engine-snapshot-status-v1 record sub-blocks.
_M12_SNAPSHOT_KEYS = (
    "status", "path", "sha256", "snapshot_id", "schema_version",
    "history", "history_manifest", "state_sha256",
)
_M12_SNAPSHOT_HISTORY_KEYS = ("path", "sha256", "data_cutoff", "used_data_cutoff")
_M12_SNAPSHOT_HISTORY_MANIFEST_KEYS = (
    "path", "sha256", "dataset_version", "data_cutoff", "output_sha256",
)
_M12_POINTER_KEYS = (
    "status", "path", "sha256", "schema_version", "dataset_version",
    "canonical", "processed_manifest",
)
_M12_POINTER_CANONICAL_KEYS = ("path", "sha256", "recorded_sha256")
_M12_POINTER_MANIFEST_KEYS = ("path", "sha256", "dataset_version", "data_cutoff", "output_sha256")
_M12_COMPARISON_KEYS = ("canonical", "processed_manifest", "dataset_version")
_M12_COMPARISON_SIDE_KEYS = (
    "path_match", "hash_match", "snapshot_path", "pointer_path", "snapshot_sha256", "pointer_sha256",
)
_M12_COMPARISON_DATASET_KEYS = ("match", "snapshot", "pointer")
_COMPARISON_FLAG_KEYS = (
    "canonical_path_match",
    "canonical_hash_match",
    "processed_manifest_path_match",
    "processed_manifest_hash_match",
    "dataset_version_match",
)


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

    if not isinstance(left, str) or not isinstance(right, str):
        return False
    try:
        return _normalized_path(left) == _normalized_path(right)
    except (TypeError, ValueError, OSError):
        return False


def ensure_safe_output_root(output_root: str | Path, inputs: Sequence[Any]) -> Path:
    """Reject output roots that collide with formal roots, inputs, or prior sessions.

    Each input is either a ``(path, kind)`` pair with ``kind`` in ``{"file", "dir"}``
    (the authoritative declared type, used regardless of whether the path exists)
    or a bare ``Path`` whose kind is inferred from the filesystem. Directory
    inputs (such as a prediction run directory) forbid the root being the input
    itself or inside it; file inputs forbid the root being the file, its
    containing archive directory, or inside that directory. Protection never
    depends on the input existing, so a missing run or fixture cannot be used to
    place outputs inside the declared input directory. Being a sibling of an
    input under a common parent stays allowed. Expected input failures are
    handled after this check by the structured blocked flow.
    """

    root = Path(output_root).expanduser().resolve()
    for formal in FORMAL_ROOTS:
        if root == formal or formal in root.parents:
            raise DataValidationError(
                f"output root must not be inside the formal data or runs root: {root}"
            )
    for item in inputs:
        if isinstance(item, tuple):
            input_path, kind = Path(item[0]).expanduser().resolve(), item[1]
            if kind not in {"file", "dir"}:
                raise ValueError(f"input kind must be 'file' or 'dir': {kind!r}")
        else:
            input_path = Path(item).expanduser().resolve()
            kind = "dir" if input_path.is_dir() else "file"
        if root == input_path:
            raise DataValidationError(f"output root cannot be an input path: {root}")
        if kind == "dir":
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
        raise DataValidationError(f"issue evidence {label} must be a non-empty string")
    return value


def _require_optional_text(value: Any, label: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise DataValidationError(f"issue evidence {label} must be null or a non-empty string")
    return value


def _require_hash(value: Any, label: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise DataValidationError(f"issue evidence {label} must be a 64-character SHA-256 string")
    return value


def _require_bool(value: Any, label: str) -> bool:
    if not isinstance(value, bool):
        raise DataValidationError(f"issue evidence {label} must be an actual boolean")
    return value


def _require_keys(payload: Mapping[str, Any], keys: tuple[str, ...], label: str) -> None:
    missing = [key for key in keys if key not in payload]
    if missing:
        raise DataValidationError(f"issue evidence {label} is missing required fields: {', '.join(missing)}")


def _require_exact_keys(payload: Mapping[str, Any], keys: tuple[str, ...], label: str) -> None:
    _require_keys(payload, keys, label)
    extras = sorted(set(payload) - set(keys))
    if extras:
        raise DataValidationError(f"issue evidence {label} has unknown fields: {', '.join(extras)}")


def _bind_text(value: Any, expected: Any, label: str, *, path: bool = False) -> None:
    recorded = _require_text(value, label)
    if path:
        if not same_recorded_path(recorded, expected):
            raise DataValidationError(
                f"issue evidence recorded {label} does not match the strictly loaded snapshot record"
            )
        return
    if recorded != str(expected):
        raise DataValidationError(
            f"issue evidence recorded {label} does not match the strictly loaded snapshot record"
        )


def _bind_optional_text(value: Any, expected: Any, label: str, *, detail: str | None = None) -> None:
    recorded = _require_optional_text(value, label)
    if (recorded if recorded is not None else None) != (expected if isinstance(expected, str) else None):
        suffix = detail or "does not match the strictly loaded snapshot record"
        raise DataValidationError(f"issue evidence recorded {label} {suffix}")


def _bind_hash(value: Any, expected: Any, label: str) -> None:
    recorded = _require_hash(value, label)
    if recorded.casefold() != str(expected).casefold():
        raise DataValidationError(
            f"issue evidence recorded {label} does not match the strictly loaded snapshot record"
        )


def _validated_pointer_record(pointer: Any) -> Mapping[str, Any]:
    """Type-check one valid M12 pointer record and recompute its internal equalities.

    Every recorded value is validated against its actual expected type and the
    fixed pointer schema; key presence alone is never accepted as evidence.
    """

    pointer = _require_mapping(pointer, "current_pointer record")
    _require_exact_keys(pointer, _M12_POINTER_KEYS, "current_pointer record")
    if _require_text(pointer.get("status"), "current_pointer.status") != "valid":
        raise DataValidationError(
            "issue evidence current_pointer record is not valid; a valid pointer record is required here"
        )
    if _require_text(pointer.get("schema_version"), "current_pointer.schema_version") != (
        CURRENT_POINTER_SCHEMA_VERSION
    ):
        raise DataValidationError(
            "issue evidence current_pointer record schema_version must be "
            f"{CURRENT_POINTER_SCHEMA_VERSION}: {pointer.get('schema_version')!r}"
        )
    _require_text(pointer.get("path"), "current_pointer.path")
    _require_hash(pointer.get("sha256"), "current_pointer.sha256")
    pointer_dataset_version = _require_text(pointer.get("dataset_version"), "current_pointer.dataset_version")
    pointer_canonical = _require_mapping(pointer.get("canonical"), "current_pointer.canonical")
    _require_exact_keys(pointer_canonical, _M12_POINTER_CANONICAL_KEYS, "current_pointer.canonical")
    canonical_hash = _require_hash(pointer_canonical.get("sha256"), "current_pointer.canonical.sha256")
    recorded_sha256 = _require_hash(
        pointer_canonical.get("recorded_sha256"), "current_pointer.canonical.recorded_sha256"
    )
    if recorded_sha256.casefold() != canonical_hash.casefold():
        raise DataValidationError(
            "issue evidence valid pointer record disagrees with itself: canonical recorded_sha256 does "
            "not equal the recorded canonical sha256"
        )
    _require_text(pointer_canonical.get("path"), "current_pointer.canonical.path")
    pointer_manifest = _require_mapping(pointer.get("processed_manifest"), "current_pointer.processed_manifest")
    _require_exact_keys(pointer_manifest, _M12_POINTER_MANIFEST_KEYS, "current_pointer.processed_manifest")
    _require_text(pointer_manifest.get("path"), "current_pointer.processed_manifest.path")
    _require_hash(pointer_manifest.get("sha256"), "current_pointer.processed_manifest.sha256")
    _bind_optional_text(
        pointer_manifest.get("dataset_version"),
        pointer_dataset_version,
        "current_pointer.processed_manifest.dataset_version vs current_pointer.dataset_version",
        detail="disagrees with itself; the valid pointer record is internally inconsistent",
    )
    _require_optional_text(pointer_manifest.get("data_cutoff"), "current_pointer.processed_manifest.data_cutoff")
    manifest_output_hash = _require_hash(
        pointer_manifest.get("output_sha256"), "current_pointer.processed_manifest.output_sha256"
    )
    if manifest_output_hash.casefold() != canonical_hash.casefold():
        raise DataValidationError(
            "issue evidence valid pointer record disagrees with itself: processed manifest output_sha256 "
            "does not equal the canonical sha256"
        )
    return pointer


def _typed_comparison_flags(comparison: Mapping[str, Any]) -> dict[str, bool]:
    canonical = _require_mapping(comparison.get("canonical"), "comparison.canonical")
    _require_exact_keys(canonical, _M12_COMPARISON_SIDE_KEYS, "comparison.canonical")
    processed = _require_mapping(comparison.get("processed_manifest"), "comparison.processed_manifest")
    _require_exact_keys(processed, _M12_COMPARISON_SIDE_KEYS, "comparison.processed_manifest")
    dataset = _require_mapping(comparison.get("dataset_version"), "comparison.dataset_version")
    _require_exact_keys(dataset, _M12_COMPARISON_DATASET_KEYS, "comparison.dataset_version")
    return {
        "canonical_path_match": _require_bool(canonical.get("path_match"), "comparison.canonical.path_match"),
        "canonical_hash_match": _require_bool(canonical.get("hash_match"), "comparison.canonical.hash_match"),
        "processed_manifest_path_match": _require_bool(
            processed.get("path_match"), "comparison.processed_manifest.path_match"
        ),
        "processed_manifest_hash_match": _require_bool(
            processed.get("hash_match"), "comparison.processed_manifest.hash_match"
        ),
        "dataset_version_match": _require_bool(dataset.get("match"), "comparison.dataset_version.match"),
    }


def _recompute_comparison_flags(
    snapshot_block: Mapping[str, Any], pointer: Mapping[str, Any]
) -> dict[str, bool]:
    """Recompute every comparison equality from the record's own two sides."""

    recorded_history = snapshot_block["history"]
    recorded_history_manifest = snapshot_block["history_manifest"]
    pointer_canonical = pointer["canonical"]
    pointer_manifest = pointer["processed_manifest"]
    return {
        "canonical_path_match": same_recorded_path(
            recorded_history.get("path"), pointer_canonical.get("path")
        ),
        "canonical_hash_match": str(recorded_history.get("sha256", "")).casefold()
        == str(pointer_canonical.get("sha256", "")).casefold(),
        "processed_manifest_path_match": same_recorded_path(
            recorded_history_manifest.get("path"), pointer_manifest.get("path")
        ),
        "processed_manifest_hash_match": str(recorded_history_manifest.get("sha256", "")).casefold()
        == str(pointer_manifest.get("sha256", "")).casefold(),
        "dataset_version_match": recorded_history_manifest.get("dataset_version")
        == pointer.get("dataset_version"),
    }


def _cross_bind_comparison(
    comparison: Mapping[str, Any], snapshot_block: Mapping[str, Any], pointer: Mapping[str, Any]
) -> None:
    """Bind each comparison side to the corresponding sub-record of the same JSON."""

    recorded_history = snapshot_block["history"]
    recorded_history_manifest = snapshot_block["history_manifest"]
    canonical = comparison["canonical"]
    processed = comparison["processed_manifest"]
    dataset = comparison["dataset_version"]
    _bind_text(canonical.get("snapshot_path"), str(recorded_history.get("path")), "comparison canonical snapshot_path", path=True)
    _bind_hash(canonical.get("snapshot_sha256"), recorded_history.get("sha256"), "comparison canonical snapshot_sha256")
    _bind_text(canonical.get("pointer_path"), str(pointer["canonical"].get("path")), "comparison canonical pointer_path", path=True)
    _bind_hash(canonical.get("pointer_sha256"), pointer["canonical"].get("sha256"), "comparison canonical pointer_sha256")
    _bind_text(processed.get("snapshot_path"), str(recorded_history_manifest.get("path")), "comparison manifest snapshot_path", path=True)
    _bind_hash(processed.get("snapshot_sha256"), recorded_history_manifest.get("sha256"), "comparison manifest snapshot_sha256")
    _bind_text(processed.get("pointer_path"), str(pointer["processed_manifest"].get("path")), "comparison manifest pointer_path", path=True)
    _bind_hash(processed.get("pointer_sha256"), pointer["processed_manifest"].get("sha256"), "comparison manifest pointer_sha256")
    _bind_optional_text(
        dataset.get("snapshot"),
        recorded_history_manifest.get("dataset_version"),
        "comparison dataset_version.snapshot",
    )
    _bind_optional_text(
        dataset.get("pointer"),
        pointer.get("dataset_version"),
        "comparison dataset_version.pointer",
    )


def _require_flags_equal_recomputed(recorded: Mapping[str, bool], recomputed: Mapping[str, bool]) -> None:
    for key in _COMPARISON_FLAG_KEYS:
        if recorded[key] != recomputed[key]:
            raise DataValidationError(
                f"issue evidence comparison flag {key}={recorded[key]} contradicts the equality recomputed "
                f"from the record's own snapshot and pointer values ({recomputed[key]}); recorded flags are "
                "not trusted as evidence"
            )


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
        "recomputed_flags": None,
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

    The record's file hash is computed and recorded. Structure, field types, UTC
    ``generated_at`` (not after ``audited_at``), the target binding, and every
    recorded equality are checked: comparison flags must equal the equalities
    recomputed from the record's own snapshot-side and pointer-side values, the
    pointer record must be internally consistent, and the record's history
    cutoffs and history-manifest source fields must match the strictly loaded
    manifest. Paths recorded inside the evidence are compared in normalized form
    only and are never opened, so research-side files may stay disconnected. A
    malformed record, an unknown schema, a snapshot target mismatch, a future
    review time, or any internal contradiction raises ``DataValidationError``
    (blocked); stale/unverifiable records are returned as historical limitations
    and must never be upgraded to fresh.
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

    # Snapshot record: exact shape, valid status, and full binding to the
    # strictly loaded manifest (including the recorded cutoffs).
    snapshot_block = _require_mapping(payload.get("snapshot"), "snapshot record")
    _require_exact_keys(snapshot_block, _M12_SNAPSHOT_KEYS, "snapshot record")
    if _require_text(snapshot_block.get("status"), "snapshot.status") != "valid":
        raise DataValidationError(
            "issue evidence snapshot record status must be 'valid' to bind this evidence to the audited "
            f"snapshot; recorded status: {snapshot_block.get('status')!r}"
        )
    if _require_text(snapshot_block.get("schema_version"), "snapshot.schema_version") != (
        ENGINE_SNAPSHOT_SCHEMA_VERSION
    ):
        raise DataValidationError(
            "issue evidence snapshot record schema_version must be "
            f"{ENGINE_SNAPSHOT_SCHEMA_VERSION}: {snapshot_block.get('schema_version')!r}"
        )
    _require_text(snapshot_block.get("path"), "snapshot.path")
    record_snapshot_id = _require_text(snapshot_block.get("snapshot_id"), "snapshot.snapshot_id")
    if record_snapshot_id != snapshot_id:
        raise DataValidationError(
            "issue evidence targets a different snapshot_id: "
            f"evidence={record_snapshot_id!r} audited={snapshot_id!r}"
        )
    record_manifest_hash = _require_hash(snapshot_block.get("sha256"), "snapshot.sha256")
    if record_manifest_hash.casefold() != str(manifest_sha256).casefold():
        raise DataValidationError(
            "issue evidence recorded manifest SHA-256 does not match the audited snapshot manifest file"
        )
    record_state_hash = _require_hash(snapshot_block.get("state_sha256"), "snapshot.state_sha256")
    if record_state_hash.casefold() != str(state_sha256).casefold():
        raise DataValidationError(
            "issue evidence recorded state SHA-256 does not match the audited snapshot state file"
        )
    recorded_history = _require_mapping(snapshot_block.get("history"), "snapshot.history record")
    _require_exact_keys(recorded_history, _M12_SNAPSHOT_HISTORY_KEYS, "snapshot.history record")
    _bind_text(recorded_history.get("path"), str(history.get("path")), "snapshot.history path", path=True)
    _bind_hash(recorded_history.get("sha256"), history.get("sha256"), "snapshot.history sha256")
    _bind_optional_text(recorded_history.get("data_cutoff"), history.get("data_cutoff"), "snapshot.history data_cutoff")
    _bind_optional_text(
        recorded_history.get("used_data_cutoff"),
        history.get("used_data_cutoff"),
        "snapshot.history used_data_cutoff",
    )
    recorded_history_manifest = _require_mapping(
        snapshot_block.get("history_manifest"), "snapshot.history_manifest record"
    )
    _require_exact_keys(
        recorded_history_manifest, _M12_SNAPSHOT_HISTORY_MANIFEST_KEYS, "snapshot.history_manifest record"
    )
    _bind_text(
        recorded_history_manifest.get("path"),
        str(history_manifest.get("path")),
        "snapshot.history_manifest path",
        path=True,
    )
    _bind_hash(
        recorded_history_manifest.get("sha256"),
        history_manifest.get("sha256"),
        "snapshot.history_manifest sha256",
    )
    _bind_optional_text(
        recorded_history_manifest.get("dataset_version"),
        history_manifest.get("dataset_version"),
        "snapshot.history_manifest dataset_version",
    )
    _bind_optional_text(
        recorded_history_manifest.get("data_cutoff"),
        history_manifest.get("data_cutoff"),
        "snapshot.history_manifest data_cutoff",
    )
    _bind_hash(
        recorded_history_manifest.get("output_sha256"),
        history_manifest.get("output_sha256"),
        "snapshot.history_manifest output_sha256",
    )

    reasons = payload.get("reasons")
    if not isinstance(reasons, list) or any(not isinstance(item, str) for item in reasons):
        raise DataValidationError("issue evidence reasons must be a list of strings")

    pointer = payload.get("current_pointer")
    comparison = payload.get("comparison")
    pointer_status: str | None = None
    pointer_dataset_version: str | None = None
    recorded_flags: dict[str, bool] | None = None
    recomputed_flags: dict[str, bool] | None = None

    if record_status == "fresh":
        pointer = _validated_pointer_record(pointer)
        if comparison is None:
            raise DataValidationError(
                "issue evidence claims fresh but records no comparison evidence; handwritten fresh status "
                "is not trusted"
            )
        comparison = _require_mapping(comparison, "comparison")
        _require_exact_keys(comparison, _M12_COMPARISON_KEYS, "comparison")
        recorded_flags = _typed_comparison_flags(comparison)
        _cross_bind_comparison(comparison, snapshot_block, pointer)
        recomputed_flags = _recompute_comparison_flags(snapshot_block, pointer)
        _require_flags_equal_recomputed(recorded_flags, recomputed_flags)
        if not all(recomputed_flags.values()):
            raise DataValidationError(
                "issue evidence claims fresh but the equality recomputed from its own snapshot and pointer "
                "records is not complete; handwritten fresh status is not trusted"
            )
        if reasons:
            raise DataValidationError(
                "issue evidence claims fresh but records reasons; handwritten fresh status is not trusted"
            )
    else:
        if not reasons:
            raise DataValidationError(
                f"issue evidence records {record_status} without reasons; the record is internally inconsistent"
            )
        pointer_is_valid_record = isinstance(pointer, Mapping) and pointer.get("status") == "valid"
        if record_status == "stale":
            # Per the M12 status semantics, stale requires both records valid and a
            # real comparison mismatch; a pointer-invalid record is unverifiable.
            if not pointer_is_valid_record:
                raise DataValidationError(
                    "issue evidence records stale without a valid current pointer record; a record whose "
                    "pointer evidence is invalid or missing is unverifiable, not stale"
                )
        if pointer_is_valid_record:
            pointer = _validated_pointer_record(pointer)
            if comparison is None:
                raise DataValidationError(
                    f"issue evidence records {record_status} with a valid pointer record but no comparison "
                    "evidence; the record is internally inconsistent"
                )
            comparison = _require_mapping(comparison, "comparison")
            _require_exact_keys(comparison, _M12_COMPARISON_KEYS, "comparison")
            recorded_flags = _typed_comparison_flags(comparison)
            _cross_bind_comparison(comparison, snapshot_block, pointer)
            recomputed_flags = _recompute_comparison_flags(snapshot_block, pointer)
            _require_flags_equal_recomputed(recorded_flags, recomputed_flags)
            if all(recomputed_flags.values()):
                raise DataValidationError(
                    f"issue evidence records {record_status} but the equality recomputed from its own "
                    "snapshot and pointer records shows no mismatch"
                )
        elif comparison is not None:
            raise DataValidationError(
                "issue evidence records comparison evidence without a valid current pointer record; the "
                "record is internally inconsistent"
            )

    if isinstance(pointer, Mapping):
        pointer_status = (
            str(pointer.get("status")) if isinstance(pointer.get("status"), str) else None
        )
        dataset_version_value = pointer.get("dataset_version")
        pointer_dataset_version = (
            str(dataset_version_value) if isinstance(dataset_version_value, str) else None
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
        "comparison_flags": recorded_flags,
        "recomputed_flags": recomputed_flags,
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
