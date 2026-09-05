"""M3 run-directory and dataset provenance helpers."""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from .io import file_sha256


def _observed_cutoff(matches: pd.DataFrame) -> str | None:
    latest = matches["_timestamp"].max()
    if pd.isna(latest):
        return None
    timestamp = pd.Timestamp(latest)
    return timestamp.strftime("%Y-%m-%d") if timestamp.time() == datetime.min.time() else timestamp.isoformat()


def dataset_provenance(input_path: Path, matches: pd.DataFrame) -> dict[str, Any]:
    """Read an adjacent processed manifest without making it a runtime dependency."""

    manifest_path = input_path.parent / "epl_processed_manifest.json"
    observed_cutoff = _observed_cutoff(matches)
    provenance: dict[str, Any] = {
        "manifest_path": str(manifest_path),
        "manifest_present": manifest_path.is_file(),
        "manifest_sha256": None,
        "manifest_schema_version": None,
        "dataset_version": None,
        "dataset_version_status": "missing_manifest",
        "manifest_data_cutoff": None,
        "manifest_data_cutoff_status": "missing_manifest",
        "observed_input_cutoff": observed_cutoff,
        "data_cutoff": observed_cutoff,
        "data_cutoff_source": "input_observed",
        "status": "missing_manifest",
        "error": None,
    }
    if not manifest_path.is_file():
        return provenance
    provenance["manifest_sha256"] = file_sha256(manifest_path)
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        provenance.update(status="invalid_manifest", error=str(exc))
        return provenance
    if not isinstance(payload, dict):
        provenance.update(status="invalid_manifest", error="manifest root must be a JSON object")
        return provenance
    version, cutoff = payload.get("dataset_version"), payload.get("data_cutoff")
    provenance["manifest_schema_version"] = payload.get("schema_version")
    provenance["dataset_version"] = version
    provenance["dataset_version_status"] = "present" if version else "missing_manifest_field"
    provenance["manifest_data_cutoff"] = cutoff
    provenance["manifest_data_cutoff_status"] = "present" if cutoff else "missing_manifest_field"
    provenance["data_cutoff"] = cutoff or observed_cutoff
    provenance["data_cutoff_source"] = "manifest" if cutoff else "input_observed"
    provenance["status"] = "ok" if version and cutoff else "manifest_fields_incomplete"
    return provenance


def new_run_dir(output_root: Path) -> Path:
    output_root.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = output_root / f"m3_select_{stamp}_{uuid.uuid4().hex[:8]}"
    run_dir.mkdir()
    return run_dir
