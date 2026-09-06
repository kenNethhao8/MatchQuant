"""Evidence, artifact-discovery, and persistence helpers for M9 sessions."""

from __future__ import annotations

import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .forecast_session_report import write_forecast_session_report
from .io import file_sha256, write_json

SESSION_PREFIX = "m9_forecast_session_"


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def file_record(path: str | Path | None) -> dict[str, Any]:
    if path is None:
        return {"path": None, "exists": False, "sha256": None}
    candidate = Path(path).expanduser().resolve()
    record: dict[str, Any] = {
        "path": str(candidate),
        "exists": candidate.is_file(),
        "sha256": None,
    }
    if candidate.is_file():
        try:
            record["sha256"] = file_sha256(candidate)
        except OSError as exc:
            record["hash_error"] = str(exc)
    return record


def _directory_record(path: str | Path | None) -> dict[str, Any]:
    if path is None:
        return {"path": None, "exists": False, "sha256": None}
    candidate = Path(path).expanduser().resolve()
    return {"path": str(candidate), "exists": candidate.is_dir(), "sha256": None}


def artifact_bundle(paths: dict[str, str | Path | None]) -> dict[str, dict[str, Any]]:
    return {
        name: _directory_record(path) if name.endswith("_dir") else file_record(path)
        for name, path in paths.items()
    }


def partial_run_artifacts(run_dir: Path | None) -> dict[str, dict[str, Any]]:
    return artifact_bundle(
        {
            "run_dir": run_dir,
            "predictions": run_dir / "predictions.csv" if run_dir else None,
            "run_config": run_dir / "run_config.json" if run_dir else None,
            "report": run_dir / "forecast_report.md" if run_dir else None,
        }
    )


def partial_release_artifacts(run_dir: Path | None) -> dict[str, dict[str, Any]]:
    return artifact_bundle(
        {
            "run_dir": run_dir,
            "manifest": run_dir / "release_manifest.json" if run_dir else None,
            "report": run_dir / "release_report.md" if run_dir else None,
        }
    )


def discover_new_child(
    root: Path, prefix: str, before: set[Path]
) -> tuple[Path | None, dict[str, Any]]:
    """Return a child only when exactly one new matching directory exists."""

    candidates = []
    if root.is_dir():
        candidates = sorted(
            (
                candidate
                for candidate in root.iterdir()
                if candidate.is_dir()
                and candidate.name.startswith(prefix)
                and candidate not in before
            ),
            key=lambda candidate: candidate.name,
        )
    if len(candidates) == 1:
        status = "unique"
        bound_path = str(candidates[0].resolve())
    elif candidates:
        status = "ambiguous"
        bound_path = None
    else:
        status = "none"
        bound_path = None
    discovery = {
        "status": status,
        "prefix": prefix,
        "candidates": [str(candidate.resolve()) for candidate in candidates],
        "bound_path": bound_path,
    }
    return (candidates[0] if len(candidates) == 1 else None), discovery


def _declared_manifest_path(message: str) -> Path | None:
    match = re.search(r"manifest:\s*(.+?);\s*quality report:", message, re.IGNORECASE)
    if not match:
        return None
    return Path(match.group(1).strip()).expanduser().resolve()


def m5_failure_evidence(
    fixture_data_root: Path, before: set[Path], message: str
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    """Find M5 failure evidence without guessing among concurrent snapshots."""

    declared_manifest = _declared_manifest_path(message)
    if declared_manifest is not None:
        snapshot = declared_manifest.parent
        discovery = {
            "status": "declared",
            "prefix": "epl_fixtures_",
            "candidates": [],
            "bound_path": str(snapshot),
            "declared_manifest_path": str(declared_manifest),
        }
    else:
        snapshot, discovery = discover_new_child(
            fixture_data_root / "fixtures" / "snapshots",
            "epl_fixtures_",
            before,
        )
    raw_snapshot = None
    if snapshot and (snapshot / "raw").is_dir():
        raw_files = sorted(path for path in (snapshot / "raw").iterdir() if path.is_file())
        raw_snapshot = raw_files[0] if len(raw_files) == 1 else None
        if len(raw_files) > 1:
            discovery["raw_candidates"] = [str(path.resolve()) for path in raw_files]
            discovery["raw_status"] = "ambiguous"
    artifacts = artifact_bundle(
        {
            "snapshot_dir": snapshot,
            "manifest": snapshot / "fixture_snapshot_manifest.json" if snapshot else None,
            "quality_report": snapshot / "reports" / "fixture_quality_report.json" if snapshot else None,
            "raw_snapshot": raw_snapshot,
            "fixture_output": snapshot / "processed" / "epl_fixtures.csv" if snapshot else None,
        }
    )
    return artifacts, discovery


def new_forecast_session_dir(output_root: str | Path) -> Path:
    """Create one unique M9 session directory under an explicit output root."""

    root = Path(output_root).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    while True:
        name = f"{SESSION_PREFIX}{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}_{uuid.uuid4().hex[:8]}"
        path = root / name
        try:
            path.mkdir()
            return path
        except FileExistsError:
            continue


def selection_record(
    selection_run_config_path: Path | None, model_name: str | None, model_config_path: Path | None
) -> dict[str, Any]:
    if selection_run_config_path is not None:
        return {"mode": "m3_run_config", **file_record(selection_run_config_path)}
    return {"mode": "explicit_model_config", "model": model_name, **file_record(model_config_path)}


def initial_manifest(
    schema_version: str,
    session_dir: Path,
    history_file: Path,
    history_manifest: Path,
    fixture_source: Path,
    season: str,
    as_of: object,
    selection_run_config_path: Path | None,
    model_name: str | None,
    model_config_path: Path | None,
    fixture_data_root: Path,
    output_root: Path,
) -> dict[str, Any]:
    manifest_path = session_dir / "session_manifest.json"
    report_path = session_dir / "session_report.md"
    return {
        "schema_version": schema_version,
        "command": "matchquant forecast-session",
        "session_id": session_dir.name,
        "session_dir": str(session_dir),
        "generated_at_utc": utc_now(),
        "status": "blocked",
        "status_source": "pending_stages",
        "inputs": {
            "history": file_record(history_file),
            "history_manifest": file_record(history_manifest),
            "fixture_source": file_record(fixture_source),
            "model_selection": selection_record(selection_run_config_path, model_name, model_config_path),
            "season": str(season),
            "as_of": str(as_of),
            "fixture_data_root": str(fixture_data_root),
            "output_root": str(output_root),
        },
        "stages": {
            "fixture_snapshot": {"name": "M5", "status": "not_run", "artifacts": {}},
            "forecast": {"name": "M4", "status": "not_run", "artifacts": {}},
            "release_verification": {"name": "M8", "status": "not_run", "artifacts": {}},
        },
        "outputs": {
            "session_directory": str(session_dir),
            "session_manifest": str(manifest_path),
            "session_report": str(report_path),
            "session_report_sha256": None,
            "fixture_manifest": None,
            "forecast_run": None,
            "forecast_predictions": None,
            "forecast_run_config": None,
            "forecast_report": None,
            "release_run": None,
            "release_manifest": None,
            "release_report": None,
        },
        "input_mutation": {
            "network_accessed": False,
            "history_csv_modified": False,
            "history_manifest_modified": False,
            "fixture_source_modified": False,
            "model_selection_evidence_modified": False,
            "current_pointer_updated": False,
            "post_match_results_imported": False,
            "child_artifacts_copied_into_session": False,
        },
        "limitations": [
            "M9 是赛前本地封装，不证明数据或赛程实时，不代表概率已校准、盈利、投注价值或人工发布完成。",
            "M8 的 ready/needs_review/blocked 是发布状态唯一权威；M9 不提升或降低该状态。",
            "M9 不联网、不调用下载/同步/本地封存/结算命令，也不把 completed fixture 写入 history。",
        ],
    }


def persist(manifest: dict[str, Any], manifest_path: Path, report_path: Path) -> None:
    write_forecast_session_report(manifest, report_path)
    manifest["outputs"]["session_report_sha256"] = file_sha256(report_path)
    write_json(manifest, manifest_path)


def error_record(exc: Exception) -> dict[str, str]:
    return {"type": type(exc).__name__, "message": str(exc)}


def stage_paths(stage: dict[str, Any], key: str) -> str | None:
    artifact = stage.get("artifacts", {}).get(key, {})
    return artifact.get("path") if isinstance(artifact, dict) else None


def attach_stage_links(
    stage: dict[str, Any], artifacts: dict[str, dict[str, Any]], fields: dict[str, str]
) -> None:
    for field, artifact_name in fields.items():
        artifact = artifacts.get(artifact_name, {})
        stage[field] = artifact.get("path") if isinstance(artifact, dict) else None


__all__ = [
    "SESSION_PREFIX",
    "artifact_bundle",
    "attach_stage_links",
    "discover_new_child",
    "error_record",
    "file_record",
    "initial_manifest",
    "m5_failure_evidence",
    "new_forecast_session_dir",
    "partial_release_artifacts",
    "partial_run_artifacts",
    "persist",
    "selection_record",
    "stage_paths",
]
