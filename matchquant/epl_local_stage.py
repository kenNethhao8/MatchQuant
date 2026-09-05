"""Immutable local E0 bytes staging for the M2.1 offline sync bridge."""

from __future__ import annotations

import os
import shutil
import stat
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .epl import normalize_season
from .io import file_sha256, write_json
from .schema import DataValidationError

LOCAL_STAGE_ADAPTER_VERSION = "m7-epl-local-file-stage-v1"
LOCAL_STAGE_MANIFEST_VERSION = "m7-local-stage-manifest-v1"
LOCAL_STAGE_TYPE = "local_file"
LOCAL_STAGE_SOURCE = "local-file"
COMPETITION = "English Premier League"
DIVISION = "E0"


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _normalise_upstream_retrieved_at(value: str | None) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text or "T" not in text.upper():
        raise ValueError("upstream_retrieved_at must be a timezone-aware ISO-8601 timestamp")
    candidate = text[:-1] + "+00:00" if text[-1:].upper() == "Z" else text
    try:
        parsed = datetime.fromisoformat(candidate)
    except ValueError as exc:
        raise ValueError("upstream_retrieved_at must be a timezone-aware ISO-8601 timestamp") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("upstream_retrieved_at must be a timezone-aware ISO-8601 timestamp")
    return parsed.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _snapshot(root: Path) -> Path:
    staging_root = root / "staging"
    staging_root.mkdir(parents=True, exist_ok=True)
    while True:
        name = f"epl_local_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}_{uuid.uuid4().hex[:12]}"
        path = staging_root / name
        try:
            path.mkdir()
            return path
        except FileExistsError:
            continue


def _read_only(path: Path) -> None:
    try:
        os.chmod(path, stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
    except OSError:
        pass


def _season_policy(season: str) -> dict[str, Any]:
    return {
        "in_progress_season": None,
        "declared_in_progress_season": None,
        "season_is_declared_in_progress": False,
        "strict_380_gate_applied": True,
        "match_count_policy": "strict_complete_season",
    }


def _manifest(
    snapshot: Path,
    source_path: Path,
    raw_path: Path | None,
    season: str,
    archived_at: str,
    source_url: str | None,
    source_label: str | None,
    upstream_retrieved_at: str | None,
    raw_hash: str | None,
    status: str,
    error: str | None,
) -> dict[str, Any]:
    retrieval_status = "user_provided" if upstream_retrieved_at else "unknown"
    policy = _season_policy(season)
    entry = {
        "season": season,
        **policy,
        "url": source_url,
        "source_label": source_label,
        "source_type": LOCAL_STAGE_TYPE,
        "retrieved_at": upstream_retrieved_at,
        "upstream_retrieved_at": upstream_retrieved_at,
        "upstream_retrieved_at_status": retrieval_status,
        "local_archived_at": archived_at,
        "raw_filename": raw_path.name if raw_path else f"E0_{season}.csv",
        "raw_path": str(raw_path) if raw_path else None,
        "sha256": raw_hash,
        "row_count": None,
        "data_cutoff": None,
        "http_status": None,
        "content_type": None,
        "parse_status": "not_run",
        "audit_status": "not_run",
        "download_status": "local_copy_ok" if status == "unverified" else "failed",
        "status": status,
        "error": error,
    }
    return {
        "schema_version": LOCAL_STAGE_MANIFEST_VERSION,
        "adapter_version": LOCAL_STAGE_ADAPTER_VERSION,
        "staging_type": LOCAL_STAGE_TYPE,
        "source_type": LOCAL_STAGE_TYPE,
        "source": LOCAL_STAGE_SOURCE,
        "source_label": source_label,
        "source_url": source_url,
        "source_urls": [source_url] if source_url else [],
        "competition": COMPETITION,
        "division": DIVISION,
        "snapshot_id": snapshot.name,
        "snapshot_dir": str(snapshot),
        "source_path": str(source_path),
        "local_archived_at": archived_at,
        "upstream_retrieved_at": upstream_retrieved_at,
        "upstream_retrieved_at_status": retrieval_status,
        "generated_at": archived_at,
        "requested_seasons": [season],
        "in_progress_season": None,
        "declared_in_progress_season": None,
        "season_policy": {season: policy},
        "data_cutoff": None,
        "status": status,
        "validation_status": "unverified" if status == "unverified" else "failed",
        "validation_scope": "M7 archives bytes only; epl-sync must perform source quality validation",
        "files": [entry],
        "error": error,
    }


def stage_local_epl_source(
    source_csv: str | Path,
    season: str,
    data_dir: str | Path = Path(r"D:\Codex\matchquant_data"),
    *,
    source_url: str | None = None,
    source_label: str | None = None,
    upstream_retrieved_at: str | None = None,
) -> dict[str, Any]:
    """Archive one local E0 file without parsing, converting, diffing, or applying it."""

    normalized = normalize_season(season)
    source_path = Path(source_csv).expanduser().resolve()
    root = Path(data_dir).expanduser().resolve()
    upstream_retrieved_at = _normalise_upstream_retrieved_at(upstream_retrieved_at)
    for name, value in (
        ("source_url", source_url),
        ("source_label", source_label),
    ):
        if value is not None and not str(value).strip():
            raise ValueError(f"{name} cannot be empty when provided")
    snapshot = _snapshot(root)
    raw_path = snapshot / "raw" / f"E0_{normalized}.csv"
    manifest_path = snapshot / "raw" / "source_manifest.json"
    archived_at = _utc_now()
    raw_hash: str | None = None
    status = "unverified"
    error: str | None = None
    try:
        if not source_path.is_file():
            raise FileNotFoundError(f"source CSV does not exist: {source_path}")
        source_hash_before = file_sha256(source_path)
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source_path, raw_path)
        raw_hash = file_sha256(raw_path)
        source_hash_after = file_sha256(source_path)
        if raw_hash != source_hash_before or source_hash_after != source_hash_before:
            raise DataValidationError("source CSV changed while it was being archived")
    except (DataValidationError, FileNotFoundError, OSError, ValueError) as exc:
        status = "failed"
        error = str(exc)
        if raw_path.is_file() and raw_hash is None:
            try:
                raw_hash = file_sha256(raw_path)
            except OSError:
                raw_hash = None

    manifest = _manifest(
        snapshot,
        source_path,
        raw_path if raw_path.is_file() else None,
        normalized,
        archived_at,
        str(source_url) if source_url is not None else None,
        str(source_label) if source_label is not None else None,
        str(upstream_retrieved_at) if upstream_retrieved_at is not None else None,
        raw_hash,
        status,
        error,
    )
    write_json(manifest, manifest_path)
    if raw_path.is_file():
        _read_only(raw_path)
    if status == "failed":
        raise DataValidationError(
            f"M7 local staging failed: {error}; manifest: {manifest_path}"
        )
    return {
        "status": status,
        "snapshot_id": snapshot.name,
        "snapshot_dir": str(snapshot),
        "manifest_path": str(manifest_path),
        "raw_path": str(raw_path),
        "raw_sha256": raw_hash,
        "manifest": manifest,
    }


__all__ = [
    "LOCAL_STAGE_ADAPTER_VERSION",
    "LOCAL_STAGE_MANIFEST_VERSION",
    "stage_local_epl_source",
]
