"""Read-only B04 verification of one portable runtime bundle.

``verify_runtime_bundle`` re-derives every recorded fact from the bundle's own
bytes: payload inventory (paths, sizes, hashes, no reparse points, no missing or
extra files), the byte copy of the original snapshot manifest (shape and content
hash), the strictly loaded derived snapshot (existing M11 loader, never relaxed),
the fixed allowed manifest diff, the single-model consistency, the original M12
issue record bound to the ORIGINAL manifest hash through the B03 evidence
validator, and the state-derived team index. Recorded hashes never stand in for
recomputation, source paths recorded inside the bundle are provenance text that
is never opened, and no fallback to an original research directory exists.
"""

from __future__ import annotations

import json
import os
import re
import stat
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any, Mapping

from .engine_snapshot_manifest import (
    _SNAPSHOT_FIELDS,
    _validate_as_of,
    _validate_generated_at,
    load_engine_snapshot,
    manifest_content_sha256,
)
from .engine_snapshot_support import ENGINE_SNAPSHOT_SCHEMA_VERSION, ENGINE_STATE_SCHEMA_VERSION
from .io import file_sha256
from .models import MODEL_NAMES
from .schema import DataValidationError
from .snapshot_audit_support import (
    ACTIVE_LINK_STATUS,
    ISSUE_EVIDENCE_SCHEMA_VERSION,
    load_issue_evidence,
    read_json_object,
    utc_now,
)
from .team_index import TEAM_INDEX_SCHEMA_VERSION, verify_team_index

RUNTIME_BUNDLE_SCHEMA_VERSION = "baseline-runtime-bundle-v1"
BUNDLE_LINEAGE_VALIDATION_KIND = "bundle_original_issue_via_derived_manifest_lineage"
RUNTIME_BUNDLE_SIZE_LIMIT_BYTES = 5 * 1024 * 1024

_BUNDLE_MANIFEST_KEYS = (
    "schema_version",
    "command",
    "generated_at",
    "build_job_id",
    "snapshot_id",
    "model",
    "original_snapshot",
    "derived_snapshot",
    "state_file_sha256",
    "issue_evidence",
    "original_manifest_copy",
    "team_index",
    "files",
    "manifest_field_diffs",
    "size",
    "limitations",
)
_BUNDLE_MODEL_KEYS = ("model", "config", "config_sha256", "state_payload_sha256")
_BUNDLE_ORIGINAL_KEYS = (
    "snapshot_id",
    "schema_version",
    "manifest_file_sha256",
    "manifest_content_sha256",
    "state_file_sha256",
    "provenance_baseline_dir",
    "issue_source_file",
    "issue_source_file_sha256",
)
_BUNDLE_DERIVED_KEYS = (
    "snapshot_id",
    "manifest_relative_path",
    "manifest_file_sha256",
    "manifest_content_sha256",
    "state_relative_path",
    "state_file_sha256",
)
_BUNDLE_ISSUE_KEYS = (
    "relative_path",
    "file_sha256",
    "schema_version",
    "record_status",
    "record_generated_at",
    "recorded_snapshot_sha256",
    "original_manifest_file_sha256",
    "fresh_at_issue",
    "validation_kind",
)
_BUNDLE_MANIFEST_COPY_KEYS = ("relative_path", "file_sha256", "content_sha256")
_BUNDLE_TEAM_INDEX_KEYS = ("relative_path", "file_sha256", "schema_version")
_BUNDLE_SIZE_KEYS = ("payload_bytes", "file_count", "limit_bytes", "includes_bundle_manifest")
_BUNDLE_DIFFS_KEYS = ("allowed_leaf_paths", "observed")
_BUNDLE_FILE_ENTRY_KEYS = ("path", "bytes", "sha256")
_BUNDLE_DIFF_ENTRY_KEYS = ("field", "original", "derived", "change")

ALLOWED_MANIFEST_DIFF_FIELDS = frozenset(
    {
        "state.path",
        "outputs.snapshot_dir",
        "outputs.snapshot_manifest",
        "outputs.model_states",
        "outputs.snapshot_report",
        "manifest_sha256",
    }
)
ALLOWED_ADDED_FIELDS = frozenset({"outputs.snapshot_report"})
FIXED_DERIVED_LEAF_VALUES = {
    "state.path": "model_states.json",
    "outputs.snapshot_dir": ".",
    "outputs.snapshot_manifest": "snapshot_manifest.json",
    "outputs.model_states": "model_states.json",
    "outputs.snapshot_report": "snapshot_report.md",
}
_BUNDLE_MANIFEST_NAME = "bundle_manifest.json"
_ORIGINAL_OUTPUTS_KEYS = {"snapshot_dir", "snapshot_manifest", "model_states", "snapshot_report"}
_STATE_BLOCK_KEYS = ("path", "sha256", "schema_version")
_HISTORY_MANIFEST_SCHEMAS = {"m2-processed-manifest-v1", "m2.1-processed-manifest-v1"}
_VALIDATION_DECLARATION_KEYS = (
    "fixture_results_used_for_model_updates",
    "history_copied",
    "input_files_unchanged",
    "network_accessed",
    "snapshot_is_immutable",
)

_SHA256_RE = re.compile(r"^[0-9a-fA-F]{64}$")


class _BundleBlocked(Exception):
    """Internal: stop verification at the first failed gate with its reason."""


def _valid_sha256(value: Any) -> bool:
    return isinstance(value, str) and _SHA256_RE.fullmatch(value) is not None


def _require_exact_keys(payload: Mapping[str, Any], keys: tuple[str, ...], label: str) -> None:
    if not isinstance(payload, Mapping):
        raise DataValidationError(f"{label} must be a JSON object")
    missing = [key for key in keys if key not in payload]
    extras = sorted(set(payload) - set(keys))
    if missing:
        raise DataValidationError(f"{label} is missing required fields: {', '.join(missing)}")
    if extras:
        raise DataValidationError(f"{label} has unknown fields: {', '.join(extras)}")


def validated_bundle_relative_path(value: Any, label: str) -> str:
    """Reject absolute, drive/UNC, parent-escaping, ADS, or malformed payload paths."""

    if not isinstance(value, str) or not value.strip():
        raise DataValidationError(f"{label} must be a non-empty relative path string")
    if "\\" in value:
        raise DataValidationError(f"{label} must use forward slashes inside a bundle: {value!r}")
    parts = value.split("/")
    if any(part in ("", ".", "..") for part in parts):
        raise DataValidationError(f"{label} must be a normalised relative path without escaping: {value!r}")
    if any(":" in part for part in parts):
        raise DataValidationError(f"{label} must not contain drive letters or NTFS alternate data streams: {value!r}")
    if PurePosixPath(value).is_absolute() or PureWindowsPath(value).is_absolute():
        raise DataValidationError(f"{label} must be a relative path: {value!r}")
    if PureWindowsPath(value).drive or PureWindowsPath(value).root:
        raise DataValidationError(f"{label} must be a relative path without drive or root: {value!r}")
    return value


def _is_reparse_point(path: Path) -> bool:
    try:
        st = os.lstat(path)
    except OSError:
        return True
    attrs = getattr(st, "st_file_attributes", None)
    if attrs is not None:
        return bool(attrs & stat.FILE_ATTRIBUTE_REPARSE_POINT)
    return path.is_symlink()


def _actual_bundle_files(bundle_root: Path, errors: list[str]) -> set[str]:
    """Collect the actual bundle-relative file paths, flagging reparse directories."""

    actual: set[str] = set()
    for dirpath, dirnames, filenames in os.walk(bundle_root):
        here = Path(dirpath)
        for name in list(dirnames):
            if _is_reparse_point(here / name):
                errors.append(f"reparse directory inside the bundle: {here / name}")
                dirnames.remove(name)
        for name in filenames:
            relative = (here / name).relative_to(bundle_root).as_posix()
            actual.add(relative)
    return actual


def check_bundle_payload_integrity(
    bundle_root: Path,
    files: list[dict[str, Any]],
    *,
    bundle_manifest_name: str = _BUNDLE_MANIFEST_NAME,
) -> tuple[bool, list[str], set[str]]:
    """Re-hash every recorded payload file and detect missing/extra/reparse files.

    Read-only; used both by ``verify_runtime_bundle`` and by the bundle-mode
    session/audit final checks that observe the whole package unchanged.
    """

    errors: list[str] = []
    recorded: dict[str, dict[str, Any]] = {}
    for entry in files:
        if not isinstance(entry, Mapping):
            errors.append("bundle files entries must be JSON objects")
            continue
        try:
            path_text = validated_bundle_relative_path(entry.get("path"), "bundle files entry path")
        except DataValidationError as exc:
            errors.append(str(exc))
            continue
        if path_text in recorded:
            errors.append(f"duplicate bundle files entry: {path_text}")
            continue
        recorded[path_text] = dict(entry)
    if errors:
        return False, errors, set()

    for path_text, entry in sorted(recorded.items()):
        candidate = bundle_root / Path(*path_text.split("/"))
        if _is_reparse_point(candidate):
            errors.append(f"payload file must not be a symlink/junction/reparse object: {path_text}")
            continue
        if not candidate.is_file():
            errors.append(f"listed bundle file is missing: {path_text}")
            continue
        try:
            actual_hash = file_sha256(candidate)
        except OSError as exc:
            errors.append(f"listed bundle file could not be read: {path_text}: {exc}")
            continue
        if not _valid_sha256(entry.get("sha256")):
            errors.append(f"bundle files entry SHA-256 is not a 64-character SHA-256: {path_text}")
            continue
        if str(entry["sha256"]).casefold() != actual_hash.casefold():
            errors.append(f"bundle file hash does not match the recorded hash: {path_text}")

    actual = _actual_bundle_files(bundle_root, errors)
    expected = set(recorded) | {bundle_manifest_name}
    for missing in sorted(expected - actual):
        if missing != bundle_manifest_name or (bundle_root / bundle_manifest_name).exists():
            errors.append(f"listed bundle file is missing on disk: {missing}")
    for extra in sorted(actual - expected):
        errors.append(f"unlisted file inside the bundle: {extra}")

    return (not errors), errors, set(recorded)


def manifest_leaf_diff(original: Any, derived: Any, prefix: str = "") -> list[dict[str, Any]]:
    """Compute the changed/added/removed JSON leaf list between two manifests."""

    diffs: list[dict[str, Any]] = []
    if isinstance(original, Mapping) and isinstance(derived, Mapping):
        keys = set(original) | set(derived)
        for key in sorted(str(item) for item in keys):
            field = f"{prefix}.{key}" if prefix else str(key)
            if key not in original:
                diffs.append({"field": field, "original": None, "derived": derived[key], "change": "added"})
            elif key not in derived:
                diffs.append({"field": field, "original": original[key], "derived": None, "change": "removed"})
            else:
                diffs.extend(manifest_leaf_diff(original[key], derived[key], field))
        return diffs
    if isinstance(original, Mapping) or isinstance(derived, Mapping):
        diffs.append({"field": prefix, "original": original, "derived": derived, "change": "changed"})
        return diffs
    if original != derived:
        diffs.append({"field": prefix, "original": original, "derived": derived, "change": "changed"})
    return diffs


def _validate_manifest_payload_shape(payload: Mapping[str, Any], *, label: str) -> None:
    """Static manifest validation without touching any recorded file path.

    The original manifest byte copy is evidence: its JSON shape, field types and
    internal hashes are checked here, while its recorded source paths are never
    opened and the strict file-backed loader only ever runs on the derived
    snapshot inside the bundle.
    """

    _require_exact_keys(payload, _SNAPSHOT_FIELDS, label)
    if payload.get("schema_version") != ENGINE_SNAPSHOT_SCHEMA_VERSION:
        raise DataValidationError(
            f"{label} schema_version must be {ENGINE_SNAPSHOT_SCHEMA_VERSION}: "
            f"{payload.get('schema_version')!r}"
        )
    if payload.get("manifest_sha256") != manifest_content_sha256(payload):
        raise DataValidationError(f"{label} manifest hash does not match manifest contents")
    _validate_generated_at(payload)
    _validate_as_of(payload)
    snapshot_id = payload.get("snapshot_id")
    if not isinstance(snapshot_id, str) or not snapshot_id.startswith("m11_engine_snapshot_"):
        raise DataValidationError(f"{label} snapshot_id is missing or invalid")

    history = payload.get("history")
    history_manifest = payload.get("history_manifest")
    for evidence, evidence_label in ((history, "history"), (history_manifest, "history manifest")):
        if not isinstance(evidence, Mapping):
            raise DataValidationError(f"{label} {evidence_label} evidence must be an object")
        if not isinstance(evidence.get("path"), str) or not evidence["path"].strip() or not _valid_sha256(
            evidence.get("sha256")
        ):
            raise DataValidationError(f"{label} {evidence_label} evidence must contain a valid path and SHA-256")
    _require_exact_keys(
        history,
        (
            "path",
            "sha256",
            "rows",
            "rows_used_before_as_of",
            "rows_excluded_at_or_after_as_of",
            "observed_cutoff",
            "data_cutoff",
            "used_data_cutoff",
        ),
        f"{label} history",
    )
    for field in ("rows", "rows_used_before_as_of", "rows_excluded_at_or_after_as_of"):
        value = history[field]
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise DataValidationError(f"{label} history {field} must be a non-negative integer")
    if history["rows_used_before_as_of"] + history["rows_excluded_at_or_after_as_of"] != history["rows"]:
        raise DataValidationError(f"{label} history row counts are inconsistent")
    for field in ("observed_cutoff", "data_cutoff"):
        if not isinstance(history[field], str) or not history[field].strip():
            raise DataValidationError(f"{label} history {field} evidence is missing")
    used_cutoff = history["used_data_cutoff"]
    if used_cutoff is not None and (not isinstance(used_cutoff, str) or not used_cutoff.strip()):
        raise DataValidationError(f"{label} history used_data_cutoff evidence is invalid")
    _require_exact_keys(
        history_manifest,
        ("path", "sha256", "schema_version", "dataset_version", "data_cutoff", "output_sha256"),
        f"{label} history manifest",
    )
    if history_manifest["schema_version"] not in _HISTORY_MANIFEST_SCHEMAS:
        raise DataValidationError(f"{label} history manifest schema_version is unsupported")
    if history_manifest["data_cutoff"] != history["data_cutoff"]:
        raise DataValidationError(f"{label} history manifest data_cutoff does not match history evidence")
    if not _valid_sha256(history_manifest.get("output_sha256")):
        raise DataValidationError(f"{label} history manifest output hash is missing or invalid")
    if str(history_manifest["output_sha256"]).casefold() != str(history["sha256"]).casefold():
        raise DataValidationError(f"{label} history manifest output hash does not match history evidence")

    if not isinstance(payload.get("selection"), Mapping):
        raise DataValidationError(f"{label} selection evidence must be an object")
    if payload.get("season_mode") not in {"carry", "reset"}:
        raise DataValidationError(f"{label} season_mode must be 'carry' or 'reset'")
    if not isinstance(payload.get("disclaimer"), str) or not payload["disclaimer"].strip():
        raise DataValidationError(f"{label} disclaimer is missing")

    state_evidence = payload.get("state")
    _require_exact_keys(state_evidence, _STATE_BLOCK_KEYS, f"{label} state")
    if state_evidence.get("schema_version") != ENGINE_STATE_SCHEMA_VERSION:
        raise DataValidationError(f"{label} state schema_version is unsupported")
    if not _valid_sha256(state_evidence.get("sha256")):
        raise DataValidationError(f"{label} state SHA-256 is missing or invalid")
    if not isinstance(state_evidence.get("path"), str) or not state_evidence["path"].strip():
        raise DataValidationError(f"{label} state path must be a non-empty string")

    manifest_names = payload.get("model_names")
    if (
        not isinstance(manifest_names, list)
        or any(not isinstance(name, str) for name in manifest_names)
        or len(set(manifest_names)) != len(manifest_names)
        or any(name not in MODEL_NAMES for name in manifest_names)
    ):
        raise DataValidationError(f"{label} model_names must be a unique list of supported model names")
    manifest_models = payload.get("models")
    if not isinstance(manifest_models, list):
        raise DataValidationError(f"{label} field 'models' must be a list")
    if {str(item) for item in manifest_names} != {
        str(item.get("model")) for item in manifest_models if isinstance(item, Mapping) and item.get("model") is not None
    }:
        raise DataValidationError(f"{label} model_names do not match the manifest models list")
    for evidence in manifest_models:
        model_name = evidence.get("model") if isinstance(evidence, Mapping) else None
        _require_exact_keys(
            evidence,
            ("model", "config", "config_sha256", "state_sha256", "config_source", "state_codec"),
            f"{label} model {model_name}",
        )
        _require_exact_keys(
            evidence.get("config_source"),
            ("mode", "path", "sha256"),
            f"{label} config source for {model_name}",
        )
        if evidence.get("state_codec") != ENGINE_STATE_SCHEMA_VERSION:
            raise DataValidationError(f"{label} state codec for {model_name} is unsupported")
        if not _valid_sha256(evidence.get("config_sha256")) or not _valid_sha256(evidence.get("state_sha256")):
            raise DataValidationError(f"{label} model hashes for {model_name} are missing or invalid")

    declarations = payload.get("validation")
    if not isinstance(declarations, Mapping):
        raise DataValidationError(f"{label} validation declarations are missing")
    if any(key not in declarations for key in _VALIDATION_DECLARATION_KEYS):
        raise DataValidationError(f"{label} validation declarations are incomplete")
    if (
        declarations["network_accessed"] is not False
        or declarations["history_copied"] is not False
        or declarations["fixture_results_used_for_model_updates"] is not False
        or declarations["input_files_unchanged"] is not True
        or declarations["snapshot_is_immutable"] is not True
    ):
        raise DataValidationError(f"{label} violates the offline/no-history-copy contract")

    outputs = payload.get("outputs")
    if not isinstance(outputs, Mapping):
        raise DataValidationError(f"{label} outputs evidence must be an object")
    unexpected_outputs = sorted(set(outputs) - _ORIGINAL_OUTPUTS_KEYS)
    if unexpected_outputs:
        raise DataValidationError(
            f"{label} has an unexpected outputs shape; refusing to guess: {', '.join(unexpected_outputs)}"
        )


def _relative_provenance_matches_baseline(recorded_value: Any, other_value: Any, baseline_dir: str) -> bool:
    """Compare provenance texts; relative texts are resolved against the frozen
    original base directory, never against the current working directory."""

    if not isinstance(recorded_value, str) or not recorded_value.strip():
        return False
    if not isinstance(other_value, str) or not other_value.strip():
        return False
    if PureWindowsPath(recorded_value).is_absolute() and PureWindowsPath(other_value).is_absolute():
        return True
    if not isinstance(baseline_dir, str) or not baseline_dir.strip():
        return False
    try:
        frozen_recorded = (
            os.path.normcase(str(Path(recorded_value).expanduser()))
            if PureWindowsPath(recorded_value).is_absolute()
            else os.path.normcase(str(Path(baseline_dir, recorded_value)))
        )
        frozen_other = (
            os.path.normcase(str(Path(other_value).expanduser()))
            if PureWindowsPath(other_value).is_absolute()
            else os.path.normcase(str(Path(baseline_dir, other_value)))
        )
    except (TypeError, ValueError, OSError):
        return False
    return frozen_recorded == frozen_other


def verify_runtime_bundle(bundle_manifest_path: str | Path) -> dict[str, Any]:
    """Read-only verification of one runtime bundle; never writes anything."""

    manifest_file = Path(bundle_manifest_path).expanduser().resolve()
    checks: list[dict[str, str]] = []

    def record(name: str, passed: bool, detail: str = "") -> None:
        checks.append({"check": name, "status": "passed" if passed else "failed", "detail": detail})

    def blocked(check_name: str, reason: str) -> dict[str, Any]:
        record(check_name, False, reason)
        return {
            "schema_version": RUNTIME_BUNDLE_SCHEMA_VERSION,
            "command": "matchquant runtime-bundle-verify",
            "verified_at": utc_now(),
            "status": "blocked",
            "reasons": [reason],
            "bundle_manifest_path": str(manifest_file),
            "bundle_root": str(manifest_file.parent),
            "bundle_manifest_file_sha256": None,
            "snapshot_id": None,
            "original": None,
            "derived": None,
            "issue_validation": None,
            "team_index": None,
            "model": None,
            "checks": checks,
            "file_count": None,
            "total_bytes": None,
            "limitations": [
                "verify 是只读检查：不写入、不自动解压、不发布；blocked 表示该包不能当作可用运行包。",
                "整个包没有签名：完全一致的伪造包体无法被 verify 发现；hash 只证明本次输入一致性。",
            ],
        }

    if not manifest_file.is_file():
        return blocked("bundle_manifest_load", f"bundle manifest does not exist: {manifest_file}")
    bundle_root = manifest_file.parent
    try:
        bundle_manifest_file_sha256 = file_sha256(manifest_file)
        payload = read_json_object(manifest_file, "bundle manifest")
        _require_exact_keys(payload, _BUNDLE_MANIFEST_KEYS, "bundle manifest")
        if payload.get("schema_version") != RUNTIME_BUNDLE_SCHEMA_VERSION:
            raise DataValidationError(
                f"unsupported bundle manifest schema_version; expected "
                f"{RUNTIME_BUNDLE_SCHEMA_VERSION}: {payload.get('schema_version')!r}"
            )
    except (DataValidationError, OSError, ValueError, TypeError) as exc:
        return blocked("bundle_manifest_load", str(exc))
    record("bundle_manifest_load", True, f"schema_version={payload.get('schema_version')}")

    try:
        snapshot_id = payload.get("snapshot_id")
        if not isinstance(snapshot_id, str) or not snapshot_id.startswith("m11_engine_snapshot_"):
            raise DataValidationError(f"bundle manifest snapshot_id is missing or invalid: {snapshot_id!r}")
        model_block = payload.get("model")
        _require_exact_keys(model_block, _BUNDLE_MODEL_KEYS, "bundle manifest model")
        if model_block.get("model") not in MODEL_NAMES:
            raise DataValidationError(f"bundle manifest records an unsupported model: {model_block.get('model')!r}")
        for block_name, keys in (
            ("original_snapshot", _BUNDLE_ORIGINAL_KEYS),
            ("derived_snapshot", _BUNDLE_DERIVED_KEYS),
            ("issue_evidence", _BUNDLE_ISSUE_KEYS),
            ("original_manifest_copy", _BUNDLE_MANIFEST_COPY_KEYS),
            ("team_index", _BUNDLE_TEAM_INDEX_KEYS),
            ("size", _BUNDLE_SIZE_KEYS),
            ("manifest_field_diffs", _BUNDLE_DIFFS_KEYS),
        ):
            _require_exact_keys(payload.get(block_name), keys, f"bundle manifest {block_name}")
        if not isinstance(payload.get("files"), list) or not payload["files"]:
            raise DataValidationError("bundle manifest files must be a non-empty list")
        if not isinstance(payload.get("limitations"), list) or any(
            not isinstance(item, str) for item in payload["limitations"]
        ):
            raise DataValidationError("bundle manifest limitations must be a list of strings")
        original_block = payload["original_snapshot"]
        derived_block = payload["derived_snapshot"]
        issue_block = payload["issue_evidence"]
        copy_block = payload["original_manifest_copy"]
        team_block = payload["team_index"]
        size_block = payload["size"]
        for value, label in (
            (original_block.get("manifest_file_sha256"), "original manifest file hash"),
            (original_block.get("manifest_content_sha256"), "original manifest content hash"),
            (original_block.get("state_file_sha256"), "original state hash"),
            (derived_block.get("manifest_file_sha256"), "derived manifest file hash"),
            (derived_block.get("manifest_content_sha256"), "derived manifest content hash"),
            (derived_block.get("state_file_sha256"), "derived state hash"),
        ):
            if not _valid_sha256(value):
                raise DataValidationError(f"bundle manifest {label} is missing or invalid")
        if str(payload.get("state_file_sha256")).casefold() != str(
            original_block["state_file_sha256"]
        ).casefold() or str(payload.get("state_file_sha256")).casefold() != str(
            derived_block["state_file_sha256"]
        ).casefold():
            raise DataValidationError(
                "bundle manifest state hashes disagree between the top level, original, and derived records"
            )
        if original_block.get("snapshot_id") != snapshot_id or derived_block.get("snapshot_id") != snapshot_id:
            raise DataValidationError("bundle manifest snapshot ids disagree between the records")
        if original_block.get("schema_version") != ENGINE_SNAPSHOT_SCHEMA_VERSION:
            raise DataValidationError(
                f"bundle manifest original snapshot schema_version must be {ENGINE_SNAPSHOT_SCHEMA_VERSION}"
            )
        if not isinstance(original_block.get("provenance_baseline_dir"), str) or not original_block[
            "provenance_baseline_dir"
        ].strip():
            raise DataValidationError("bundle manifest provenance_baseline_dir must be a non-empty string")
        if not isinstance(original_block.get("issue_source_file"), str) or not original_block[
            "issue_source_file"
        ].strip():
            raise DataValidationError("bundle manifest issue_source_file must be a non-empty string")
        expected_manifest_rel = f"snapshot/{snapshot_id}/snapshot_manifest.json"
        expected_state_rel = f"snapshot/{snapshot_id}/model_states.json"
        if derived_block.get("manifest_relative_path") != expected_manifest_rel:
            raise DataValidationError(
                f"bundle manifest derived manifest path must be {expected_manifest_rel}: "
                f"{derived_block.get('manifest_relative_path')!r}"
            )
        if derived_block.get("state_relative_path") != expected_state_rel:
            raise DataValidationError(
                f"bundle manifest derived state path must be {expected_state_rel}: "
                f"{derived_block.get('state_relative_path')!r}"
            )
        if copy_block.get("relative_path") != "evidence/original_snapshot_manifest.json":
            raise DataValidationError("bundle manifest original manifest copy path is not the fixed evidence path")
        if issue_block.get("relative_path") != "evidence/issue_audit.json":
            raise DataValidationError("bundle manifest issue evidence path is not the fixed evidence path")
        if issue_block.get("schema_version") != ISSUE_EVIDENCE_SCHEMA_VERSION:
            raise DataValidationError(
                f"bundle manifest issue evidence schema_version must be {ISSUE_EVIDENCE_SCHEMA_VERSION}"
            )
        if issue_block.get("record_status") != "fresh" or issue_block.get("fresh_at_issue") is not True:
            raise DataValidationError(
                "bundle manifest issue evidence must record a fresh_at_issue record bound to the original snapshot"
            )
        if issue_block.get("validation_kind") != BUNDLE_LINEAGE_VALIDATION_KIND:
            raise DataValidationError(
                f"bundle manifest issue evidence validation_kind must be {BUNDLE_LINEAGE_VALIDATION_KIND}"
            )
        if str(issue_block.get("recorded_snapshot_sha256")).casefold() != str(
            original_block["manifest_file_sha256"]
        ).casefold() or str(issue_block.get("original_manifest_file_sha256")).casefold() != str(
            original_block["manifest_file_sha256"]
        ).casefold():
            raise DataValidationError(
                "the original issue record must stay bound to the original manifest file hash, "
                "never to the derived manifest hash"
            )
        if team_block.get("schema_version") != TEAM_INDEX_SCHEMA_VERSION:
            raise DataValidationError(
                f"bundle manifest team index schema_version must be {TEAM_INDEX_SCHEMA_VERSION}"
            )
        if size_block.get("includes_bundle_manifest") is not False:
            raise DataValidationError("bundle manifest size record must exclude the bundle manifest itself")
        if size_block.get("limit_bytes") != RUNTIME_BUNDLE_SIZE_LIMIT_BYTES:
            raise DataValidationError("bundle manifest size limit does not match the fixed 5 MiB contract")
        diffs_block = payload["manifest_field_diffs"]
        if list(diffs_block.get("allowed_leaf_paths") or []) != sorted(ALLOWED_MANIFEST_DIFF_FIELDS):
            raise DataValidationError(
                "bundle manifest allowed manifest diff fields do not match the fixed contract"
            )
        if not isinstance(diffs_block.get("observed"), list):
            raise DataValidationError("bundle manifest observed manifest diffs must be a list")
        for entry in diffs_block["observed"]:
            _require_exact_keys(entry, _BUNDLE_DIFF_ENTRY_KEYS, "bundle manifest observed diff entry")
    except (DataValidationError, TypeError, ValueError, KeyError, OSError) as exc:
        return blocked("bundle_manifest_structure", str(exc))
    record("bundle_manifest_structure", True, f"snapshot_id={snapshot_id}")

    files: list[dict[str, Any]] = payload["files"]
    try:
        payload_bytes = 0
        for entry in files:
            _require_exact_keys(entry, _BUNDLE_FILE_ENTRY_KEYS, "bundle files entry")
            validated_bundle_relative_path(entry.get("path"), "bundle files entry path")
            if isinstance(entry.get("bytes"), bool) or not isinstance(entry.get("bytes"), int) or entry["bytes"] < 0:
                raise DataValidationError(f"bundle files entry bytes must be a non-negative integer: {entry.get('path')!r}")
            payload_bytes += int(entry["bytes"])
        if size_block.get("file_count") != len(files) or size_block.get("payload_bytes") != payload_bytes:
            raise DataValidationError(
                "bundle manifest size record does not match the files list (file count or payload bytes)"
            )
    except (DataValidationError, TypeError, ValueError, KeyError) as exc:
        return blocked("bundle_files_list", str(exc))
    record("bundle_files_list", True, f"{len(files)} payload files, {payload_bytes} bytes")

    try:
        ok, integrity_errors, recorded_paths = check_bundle_payload_integrity(bundle_root, files)
        if not ok:
            raise DataValidationError("; ".join(integrity_errors))
    except (DataValidationError, OSError) as exc:
        return blocked("bundle_files_inventory", str(exc))
    record(
        "bundle_files_inventory",
        True,
        "every listed file present, hashed, and a regular non-reparse file; no unlisted files",
    )

    original_copy_path = bundle_root / Path(*copy_block["relative_path"].split("/"))
    try:
        actual_copy_hash = file_sha256(original_copy_path)
        if str(copy_block.get("file_sha256")).casefold() != actual_copy_hash.casefold():
            raise DataValidationError("original snapshot manifest copy hash does not match the recorded copy hash")
        copy_payload = read_json_object(original_copy_path, "original snapshot manifest copy")
        _validate_manifest_payload_shape(copy_payload, label="original snapshot manifest copy")
        if str(copy_payload["manifest_sha256"]).casefold() != str(
            copy_block.get("content_sha256")
        ).casefold() or str(copy_payload["manifest_sha256"]).casefold() != str(
            original_block["manifest_content_sha256"]
        ).casefold():
            raise DataValidationError(
                "original snapshot manifest copy content hash does not match the recorded original content hash"
            )
        if str(copy_payload["state"]["sha256"]).casefold() != str(payload["state_file_sha256"]).casefold():
            raise DataValidationError(
                "original snapshot manifest copy state hash does not match the recorded bundle state hash"
            )
        if copy_payload["snapshot_id"] != snapshot_id:
            raise DataValidationError("original snapshot manifest copy snapshot_id does not match the bundle")
        if original_block.get("issue_source_file_sha256") is not None and str(
            original_block["issue_source_file_sha256"]
        ).casefold() != str(issue_block.get("file_sha256")).casefold():
            raise DataValidationError(
                "bundle manifest original issue source hash does not match the recorded issue evidence hash"
            )
    except (DataValidationError, OSError, ValueError, TypeError, KeyError) as exc:
        return blocked("original_manifest_copy", str(exc))
    record(
        "original_manifest_copy",
        True,
        "byte copy verified against the recorded original file/content hashes and full JSON shape",
    )

    derived_manifest_path = bundle_root / Path(*derived_block["manifest_relative_path"].split("/"))
    derived_state_path = bundle_root / Path(*derived_block["state_relative_path"].split("/"))
    try:
        actual_derived_hash = file_sha256(derived_manifest_path)
        if str(derived_block.get("manifest_file_sha256")).casefold() != actual_derived_hash.casefold():
            raise DataValidationError(
                "derived snapshot manifest file hash does not match the recorded derived hash"
            )
        loaded = load_engine_snapshot(derived_manifest_path)
        if str(loaded["manifest"]["manifest_sha256"]).casefold() != str(
            derived_block.get("manifest_content_sha256")
        ).casefold():
            raise DataValidationError(
                "derived snapshot manifest content hash does not match the recorded derived content hash"
            )
        if str(loaded["state_sha256"]).casefold() != str(payload["state_file_sha256"]).casefold():
            raise DataValidationError(
                "derived snapshot state file hash does not match the recorded bundle state hash"
            )
        if loaded["manifest"]["snapshot_id"] != snapshot_id:
            raise DataValidationError("derived snapshot snapshot_id does not match the bundle")
        if len(loaded["manifest"]["model_names"]) != 1 or len(loaded["models"]) != 1:
            raise DataValidationError(
                "runtime bundles are single-model; the derived snapshot is not a single-model snapshot"
            )
        if loaded["manifest"]["state"]["sha256"].casefold() != str(
            copy_payload["state"]["sha256"]
        ).casefold():
            raise DataValidationError(
                "derived snapshot state hash differs from the original manifest copy state hash; "
                "only the fixed path leaves may change"
            )
    except (DataValidationError, FileNotFoundError, OSError, ValueError, TypeError) as exc:
        return blocked("derived_snapshot_strict_load", str(exc))
    record(
        "derived_snapshot_strict_load",
        True,
        "existing strict M11 loader passed on the derived manifest; state bytes match the original hash",
    )

    try:
        recomputed_diff = manifest_leaf_diff(copy_payload, loaded["manifest"])
        observed_diff = payload["manifest_field_diffs"]["observed"]
        if observed_diff != recomputed_diff:
            raise DataValidationError(
                "bundle manifest recorded manifest diffs do not match the diff recomputed from the "
                "original copy and the derived manifest"
            )
        for diff in recomputed_diff:
            if diff["change"] == "removed" or diff["field"] not in ALLOWED_MANIFEST_DIFF_FIELDS:
                raise DataValidationError(
                    f"forbidden manifest change between the original copy and the derived manifest: "
                    f"{diff['change']} {diff['field']}"
                )
            if diff["change"] == "added" and diff["field"] not in ALLOWED_ADDED_FIELDS:
                raise DataValidationError(
                    f"forbidden added manifest leaf between the original copy and the derived manifest: {diff['field']}"
                )
        for field, expected_value in FIXED_DERIVED_LEAF_VALUES.items():
            node: Any = loaded["manifest"]
            for part in field.split("."):
                if not isinstance(node, Mapping) or part not in node:
                    raise DataValidationError(f"derived manifest is missing the fixed leaf {field}")
                node = node[part]
            if node != expected_value:
                raise DataValidationError(
                    f"derived manifest fixed leaf {field} must be {expected_value!r}: {node!r}"
                )
    except (DataValidationError, TypeError, ValueError) as exc:
        return blocked("manifest_allowed_diff", str(exc))
    record(
        "manifest_allowed_diff",
        True,
        "original copy vs derived manifest differ only in the fixed path leaves and the recomputed content hash",
    )

    try:
        record_model = loaded["manifest"]["models"][0]
        if model_block.get("model") != record_model.get("model"):
            raise DataValidationError("bundle manifest model does not match the derived snapshot model")
        if model_block.get("config") != record_model.get("config"):
            raise DataValidationError("bundle manifest model config does not match the derived snapshot config")
        if str(model_block.get("config_sha256")).casefold() != str(record_model.get("config_sha256")).casefold():
            raise DataValidationError("bundle manifest model config hash does not match the derived snapshot")
        if str(model_block.get("state_payload_sha256")).casefold() != str(
            record_model.get("state_sha256")
        ).casefold():
            raise DataValidationError("bundle manifest model payload hash does not match the derived snapshot")
    except (DataValidationError, IndexError, TypeError, KeyError) as exc:
        return blocked("single_model_consistency", str(exc))
    record(
        "single_model_consistency",
        True,
        f"model={model_block.get('model')} config/payload hashes bound to the derived snapshot",
    )

    issue_copy_path = bundle_root / Path(*issue_block["relative_path"].split("/"))
    try:
        actual_issue_hash = file_sha256(issue_copy_path)
        if str(issue_block.get("file_sha256")).casefold() != actual_issue_hash.casefold():
            raise DataValidationError("issue evidence copy hash does not match the recorded issue hash")
        issue_payload = read_json_object(issue_copy_path, "issue evidence copy")
        issue_result = load_issue_evidence(
            issue_copy_path,
            snapshot_id=snapshot_id,
            manifest_sha256=str(original_block["manifest_file_sha256"]),
            state_sha256=str(original_block["state_file_sha256"]),
            history=copy_payload["history"],
            history_manifest=copy_payload["history_manifest"],
            audited_at=utc_now(),
        )
        if issue_result["record_status"] != "fresh":
            raise DataValidationError(
                "the bundled issue record is not fresh at issue; the lineage bridge requires the "
                "original record to be fresh"
            )
        for recorded_value, issue_value in (
            (
                copy_payload["history"]["path"],
                (issue_payload.get("snapshot") or {}).get("history", {}).get("path"),
            ),
            (
                copy_payload["history_manifest"]["path"],
                (issue_payload.get("snapshot") or {}).get("history_manifest", {}).get("path"),
            ),
        ):
            if not _relative_provenance_matches_baseline(
                recorded_value, issue_value, str(original_block["provenance_baseline_dir"])
            ):
                raise DataValidationError(
                    f"relative {label} provenance does not match the frozen original manifest base directory; "
                    "the current working directory is never used as a source baseline"
                )
    except (DataValidationError, FileNotFoundError, OSError, ValueError, TypeError) as exc:
        return blocked("issue_lineage", str(exc))
    record(
        "issue_lineage",
        True,
        "original M12 record re-validated through load_issue_evidence against the ORIGINAL manifest hash; fresh",
    )

    team_index_path = bundle_root / Path(*team_block["relative_path"].split("/"))
    try:
        actual_team_hash = file_sha256(team_index_path)
        if str(team_block.get("file_sha256")).casefold() != actual_team_hash.casefold():
            raise DataValidationError("team index file hash does not match the recorded team index hash")
        team_payload = read_json_object(team_index_path, "team index")
        verify_team_index(
            team_payload,
            loaded,
            state_file_sha256=str(payload["state_file_sha256"]),
        )
    except (DataValidationError, OSError, ValueError, TypeError) as exc:
        return blocked("team_index", str(exc))
    record(
        "team_index",
        True,
        f"team index recomputed from the strictly loaded state; model={team_payload['model']}, "
        f"{len(team_payload['teams'])} teams, membership={team_payload['team_membership']}",
    )

    try:
        actual_total = 0
        for relative in sorted(recorded_paths | {_BUNDLE_MANIFEST_NAME}):
            candidate = bundle_root / Path(*relative.split("/"))
            if candidate.is_file():
                actual_total += candidate.stat().st_size
        if actual_total > RUNTIME_BUNDLE_SIZE_LIMIT_BYTES:
            raise DataValidationError(
                f"the unpacked bundle is {actual_total} bytes, above the fixed 5 MiB limit "
                f"({RUNTIME_BUNDLE_SIZE_LIMIT_BYTES} bytes); oversized candidates need a designer ruling"
            )
    except (DataValidationError, OSError) as exc:
        return blocked("size_gate", str(exc))
    record(
        "size_gate",
        True,
        f"unpacked total {actual_total} bytes including the bundle manifest (limit {RUNTIME_BUNDLE_SIZE_LIMIT_BYTES})",
    )

    derived_context = {
        "snapshot_id": snapshot_id,
        "relative_path": derived_block["manifest_relative_path"],
        "resolved_manifest_path": str(derived_manifest_path),
        "resolved_state_path": str(derived_state_path),
        "file_sha256": str(derived_block["manifest_file_sha256"]),
        "content_sha256": str(derived_block["manifest_content_sha256"]),
        "state_sha256": str(payload["state_file_sha256"]),
    }
    original_context = {
        "snapshot_id": original_block["snapshot_id"],
        "schema_version": original_block["schema_version"],
        "manifest_file_sha256": str(original_block["manifest_file_sha256"]),
        "manifest_content_sha256": str(original_block["manifest_content_sha256"]),
        "state_file_sha256": str(original_block["state_file_sha256"]),
        "provenance_baseline_dir": original_block["provenance_baseline_dir"],
        "copy_relative_path": copy_block["relative_path"],
        "copy_file_sha256": actual_copy_hash,
    }
    issue_context = {
        "provided": True,
        "path": str(issue_copy_path),
        "sha256": actual_issue_hash,
        "schema_version": ISSUE_EVIDENCE_SCHEMA_VERSION,
        "record_status": issue_result["record_status"],
        "record_generated_at": issue_result["record_generated_at"],
        "record_reasons": issue_result["record_reasons"],
        "comparison_flags": issue_result["comparison_flags"],
        "recomputed_flags": issue_result["recomputed_flags"],
        "fresh_at_issue": issue_result["fresh_at_issue"],
        "active_link_status": ACTIVE_LINK_STATUS,
        "pointer_status": issue_result["pointer_status"],
        "pointer_dataset_version": issue_result["pointer_dataset_version"],
        "source_access": "bundle_internal_evidence_copy_no_source_fallback",
        "validation_kind": BUNDLE_LINEAGE_VALIDATION_KIND,
        "original_manifest_file_sha256": str(original_block["manifest_file_sha256"]),
        "derived_manifest_file_sha256": str(derived_block["manifest_file_sha256"]),
        "state_file_sha256": str(payload["state_file_sha256"]),
        "limitations": [
            "发行证据是包内原 M12 记录副本，经 B03 load_issue_evidence 重新绑定原始 manifest hash；"
            "fresh_at_issue 只表示发行时点的关联验证，不证明当前 active、今天最新或训练到目标比赛。",
            "原记录中的来源路径仅作规范化比较，运行端从不打开；不存在回退到原研究目录的通道。",
            "整个包没有签名来源认证；不能发现所有字段同步伪造的包体。",
        ],
    }
    return {
        "schema_version": RUNTIME_BUNDLE_SCHEMA_VERSION,
        "command": "matchquant runtime-bundle-verify",
        "verified_at": utc_now(),
        "status": "ready",
        "reasons": [],
        "bundle_manifest_path": str(manifest_file),
        "bundle_manifest_file_sha256": bundle_manifest_file_sha256,
        "bundle_root": str(bundle_root),
        "snapshot_id": snapshot_id,
        "original": original_context,
        "derived": derived_context,
        "issue_validation": issue_context,
        "team_index": {
            "path": str(team_index_path),
            "file_sha256": actual_team_hash,
            "payload": team_payload,
        },
        "model": {
            "model": model_block["model"],
            "config": model_block["config"],
            "config_sha256": str(model_block["config_sha256"]),
            "state_payload_sha256": str(model_block["state_payload_sha256"]),
        },
        "files": [dict(entry) for entry in files],
        "checks": checks,
        "file_count": len(files),
        "total_bytes": actual_total,
        "limitations": [
            "verify 只读完成：不写入、不自动解压 ZIP、不发布；包体未签名，hash 不构成来源认证。",
        ],
    }


__all__ = [
    "ALLOWED_MANIFEST_DIFF_FIELDS",
    "BUNDLE_LINEAGE_VALIDATION_KIND",
    "RUNTIME_BUNDLE_SCHEMA_VERSION",
    "RUNTIME_BUNDLE_SIZE_LIMIT_BYTES",
    "check_bundle_payload_integrity",
    "manifest_leaf_diff",
    "validated_bundle_relative_path",
    "verify_runtime_bundle",
]
