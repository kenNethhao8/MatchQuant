"""Offline orchestration for one explicit M5 -> M4 -> M8 forecast session."""

from __future__ import annotations

import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from .epl_fixtures import prepare_epl_fixtures
from .forecast import run_forecast
from .forecast_session_report import write_forecast_session_report
from .io import file_sha256, write_json
from .release_verify import verify_release
from .schema import DataValidationError

FORECAST_SESSION_SCHEMA_VERSION = "m9-forecast-session-v1"
FORECAST_SESSION_STATUSES = ("ready", "needs_review", "blocked")
_SESSION_PREFIX = "m9_forecast_session_"


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _within(child: Path, parent: Path) -> bool:
    return child == parent or parent in child.parents


def _file_record(path: str | Path | None) -> dict[str, Any]:
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


def _artifact_bundle(paths: dict[str, str | Path | None]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for name, path in paths.items():
        result[name] = _directory_record(path) if name.endswith("_dir") else _file_record(path)
    return result


def _partial_run_artifacts(run_dir: Path | None) -> dict[str, dict[str, Any]]:
    return _artifact_bundle(
        {
            "run_dir": run_dir,
            "predictions": run_dir / "predictions.csv" if run_dir else None,
            "run_config": run_dir / "run_config.json" if run_dir else None,
            "report": run_dir / "forecast_report.md" if run_dir else None,
        }
    )


def _partial_release_artifacts(run_dir: Path | None) -> dict[str, dict[str, Any]]:
    return _artifact_bundle(
        {
            "run_dir": run_dir,
            "manifest": run_dir / "release_manifest.json" if run_dir else None,
            "report": run_dir / "release_report.md" if run_dir else None,
        }
    )


def _new_children(root: Path, prefix: str, before: set[Path]) -> list[Path]:
    if not root.is_dir():
        return []
    return sorted(
        (candidate for candidate in root.iterdir() if candidate.is_dir() and candidate.name.startswith(prefix) and candidate not in before),
        key=lambda candidate: candidate.name,
    )


def new_forecast_session_dir(output_root: str | Path) -> Path:
    """Create one unique M9 session directory under an explicit output root."""

    root = Path(output_root).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    while True:
        name = f"{_SESSION_PREFIX}{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}_{uuid.uuid4().hex[:8]}"
        path = root / name
        try:
            path.mkdir()
            return path
        except FileExistsError:
            continue


def _validate_output_root(
    output_root: Path,
    fixture_data_root: Path,
    history_file: Path,
    history_manifest: Path,
    fixture_source: Path,
) -> None:
    default_data_root = Path(r"D:\Codex\matchquant_data").resolve()
    if _within(output_root, default_data_root) or _within(output_root, fixture_data_root):
        raise DataValidationError(
            "M9 output-root must not be inside D:\\Codex\\matchquant_data or fixture-data-root"
        )
    if _within(fixture_data_root, output_root):
        raise DataValidationError("M9 output-root and fixture-data-root must not overlap")
    input_dirs = {history_file.parent, history_manifest.parent, fixture_source.parent}
    if output_root in input_dirs:
        raise DataValidationError("M9 output-root must not equal a history, manifest, or fixture source input directory")
    if any(parent.name.startswith(_SESSION_PREFIX) and parent.exists() for parent in (output_root, *output_root.parents)):
        raise DataValidationError("M9 output-root must not be inside an existing forecast-session directory")


def _selection_record(
    selection_run_config_path: Path | None, model_name: str | None, model_config_path: Path | None
) -> dict[str, Any]:
    if selection_run_config_path is not None:
        return {"mode": "m3_run_config", **_file_record(selection_run_config_path)}
    return {"mode": "explicit_model_config", "model": model_name, **_file_record(model_config_path)}


def _initial_manifest(
    session_dir: Path,
    history_file: Path,
    history_manifest: Path,
    fixture_source: Path,
    season: str,
    as_of: str | pd.Timestamp,
    selection_run_config_path: Path | None,
    model_name: str | None,
    model_config_path: Path | None,
    fixture_data_root: Path,
    output_root: Path,
) -> dict[str, Any]:
    manifest_path = session_dir / "session_manifest.json"
    report_path = session_dir / "session_report.md"
    return {
        "schema_version": FORECAST_SESSION_SCHEMA_VERSION,
        "command": "matchquant forecast-session",
        "session_id": session_dir.name,
        "session_dir": str(session_dir),
        "generated_at_utc": _utc_now(),
        "status": "blocked",
        "status_source": "pending_stages",
        "inputs": {
            "history": _file_record(history_file),
            "history_manifest": _file_record(history_manifest),
            "fixture_source": _file_record(fixture_source),
            "model_selection": _selection_record(selection_run_config_path, model_name, model_config_path),
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


def _persist(manifest: dict[str, Any], manifest_path: Path, report_path: Path) -> None:
    write_forecast_session_report(manifest, report_path)
    manifest["outputs"]["session_report_sha256"] = file_sha256(report_path)
    write_json(manifest, manifest_path)


def _error_record(exc: Exception) -> dict[str, str]:
    return {"type": type(exc).__name__, "message": str(exc)}


def _m5_failure_artifacts(fixture_data_root: Path, before: set[Path], message: str) -> dict[str, dict[str, Any]]:
    snapshots = _new_children(fixture_data_root / "fixtures" / "snapshots", "epl_fixtures_", before)
    snapshot = snapshots[-1] if snapshots else None
    if snapshot is None:
        match = re.search(r"manifest:\s*(.+?);\s*quality report:", message)
        if match:
            manifest_path = Path(match.group(1).strip()).expanduser().resolve()
            snapshot = manifest_path.parent
    raw_snapshot = None
    if snapshot and (snapshot / "raw").is_dir():
        raw_files = sorted(path for path in (snapshot / "raw").iterdir() if path.is_file())
        raw_snapshot = raw_files[0] if raw_files else None
    return _artifact_bundle(
        {
            "snapshot_dir": snapshot,
            "manifest": snapshot / "fixture_snapshot_manifest.json" if snapshot else None,
            "quality_report": snapshot / "reports" / "fixture_quality_report.json" if snapshot else None,
            "raw_snapshot": raw_snapshot,
            "fixture_output": snapshot / "processed" / "epl_fixtures.csv" if snapshot else None,
        }
    )


def _stage_paths(stage: dict[str, Any], key: str) -> str | None:
    artifact = stage.get("artifacts", {}).get(key, {})
    return artifact.get("path") if isinstance(artifact, dict) else None


def _attach_stage_links(stage: dict[str, Any], artifacts: dict[str, dict[str, Any]], fields: dict[str, str]) -> None:
    for field, artifact_name in fields.items():
        artifact = artifacts.get(artifact_name, {})
        stage[field] = artifact.get("path") if isinstance(artifact, dict) else None


def _result(
    manifest: dict[str, Any], manifest_path: Path, report_path: Path
) -> dict[str, Any]:
    stages = manifest["stages"]
    forecast = stages["forecast"]
    release = stages["release_verification"]
    fixture = stages["fixture_snapshot"]
    return {
        "status": manifest["status"],
        "session_id": manifest["session_id"],
        "session_dir": manifest["session_dir"],
        "session_manifest_path": str(manifest_path),
        "session_report_path": str(report_path),
        "fixture_snapshot_dir": _stage_paths(fixture, "snapshot_dir"),
        "fixture_manifest_path": _stage_paths(fixture, "manifest"),
        "forecast_run_dir": _stage_paths(forecast, "run_dir"),
        "forecast_predictions_path": _stage_paths(forecast, "predictions"),
        "forecast_run_config_path": _stage_paths(forecast, "run_config"),
        "release_dir": _stage_paths(release, "run_dir"),
        "release_manifest_path": _stage_paths(release, "manifest"),
        "release_report_path": _stage_paths(release, "report"),
        "stages": stages,
        "manifest": manifest,
    }


def run_forecast_session(
    history_path: str | Path,
    history_manifest_path: str | Path,
    fixture_source_csv: str | Path,
    season: str,
    as_of: str | pd.Timestamp,
    *,
    selection_run_config_path: str | Path | None = None,
    model_name: str | None = None,
    model_config_path: str | Path | None = None,
    season_mode: str | None = None,
    source_url: str | None = None,
    source_label: str | None = None,
    fixture_data_root: str | Path = Path(r"D:\Codex\matchquant_data"),
    output_root: str | Path = Path(r"D:\Codex\matchquant_runs"),
) -> dict[str, Any]:
    """Run M5, M4, then M8 with explicit local inputs and M9 evidence."""

    history_file = Path(history_path).expanduser().resolve()
    history_manifest = Path(history_manifest_path).expanduser().resolve()
    fixture_source = Path(fixture_source_csv).expanduser().resolve()
    fixture_root = Path(fixture_data_root).expanduser().resolve()
    output_root_path = Path(output_root).expanduser().resolve()
    selection_path = Path(selection_run_config_path).expanduser().resolve() if selection_run_config_path else None
    model_config = Path(model_config_path).expanduser().resolve() if model_config_path else None
    if selection_path is not None and model_name is not None:
        raise ValueError("selection run config and --model are mutually exclusive")
    if selection_path is not None and model_config is not None:
        raise ValueError("selection run config and --model-config are mutually exclusive")
    if selection_path is None and (model_name is None or model_config is None):
        raise ValueError("provide either selection evidence or explicit --model with --model-config")
    _validate_output_root(output_root_path, fixture_root, history_file, history_manifest, fixture_source)

    session_dir = new_forecast_session_dir(output_root_path)
    manifest_path = session_dir / "session_manifest.json"
    report_path = session_dir / "session_report.md"
    manifest = _initial_manifest(
        session_dir, history_file, history_manifest, fixture_source, season, as_of,
        selection_path, model_name, model_config, fixture_root, output_root_path,
    )
    _persist(manifest, manifest_path, report_path)

    m5_root = fixture_root / "fixtures" / "snapshots"
    m5_before = set(m5_root.iterdir()) if m5_root.is_dir() else set()
    try:
        fixture = prepare_epl_fixtures(
            fixture_source, season, fixture_root, source_url=source_url, source_label=source_label
        )
    except Exception as exc:
        stage = manifest["stages"]["fixture_snapshot"]
        stage.update(status="blocked", result_status="failed", error=_error_record(exc))
        stage["artifacts"] = _m5_failure_artifacts(fixture_root, m5_before, str(exc))
        _attach_stage_links(
            stage,
            stage["artifacts"],
            {"snapshot_dir": "snapshot_dir", "manifest_path": "manifest", "quality_report_path": "quality_report", "fixture_path": "fixture_output"},
        )
        manifest["outputs"]["fixture_manifest"] = stage["artifacts"]["manifest"]
        manifest["status_source"] = "M5 failure"
        _persist(manifest, manifest_path, report_path)
        return _result(manifest, manifest_path, report_path)

    fixture_artifacts = _artifact_bundle(
        {
            "snapshot_dir": fixture.get("snapshot_dir"),
            "manifest": fixture.get("manifest_path"),
            "quality_report": fixture.get("quality_report_path"),
            "diff_report": fixture.get("diff_report_path"),
            "raw_snapshot": fixture.get("manifest", {}).get("source_raw_snapshot"),
            "fixture_output": fixture.get("fixture_path"),
        }
    )
    fixture_stage = manifest["stages"]["fixture_snapshot"]
    fixture_stage["artifacts"] = fixture_artifacts
    fixture_stage["result_status"] = fixture.get("status")
    _attach_stage_links(
        fixture_stage,
        fixture_artifacts,
        {"snapshot_dir": "snapshot_dir", "manifest_path": "manifest", "quality_report_path": "quality_report", "fixture_path": "fixture_output"},
    )
    fixture_stage["source_sha256"] = fixture.get("source_sha256")
    manifest["outputs"]["fixture_manifest"] = fixture_artifacts["manifest"]
    if fixture.get("status") != "ok" or not fixture.get("fixture_path"):
        fixture_stage.update(
            status="blocked",
            error={"type": "DataValidationError", "message": "M5 did not produce a usable pending fixture CSV"},
        )
        manifest["status_source"] = "M5 no pending fixture"
        _persist(manifest, manifest_path, report_path)
        return _result(manifest, manifest_path, report_path)
    fixture_stage["status"] = "completed"
    _persist(manifest, manifest_path, report_path)

    m4_before = set(output_root_path.iterdir()) if output_root_path.is_dir() else set()
    try:
        forecast = run_forecast(
            history_file,
            fixture["fixture_path"],
            as_of,
            selection_run_config_path=selection_path,
            model_name=model_name,
            model_config_path=model_config,
            season_mode=season_mode,
            output_root=output_root_path,
        )
    except Exception as exc:
        stage = manifest["stages"]["forecast"]
        stage.update(status="blocked", result_status="failed", error=_error_record(exc))
        new_forecasts = _new_children(output_root_path, "m4_forecast_", m4_before)
        stage["artifacts"] = _partial_run_artifacts(new_forecasts[-1] if new_forecasts else None)
        _attach_stage_links(
            stage,
            stage["artifacts"],
            {"run_dir": "run_dir", "predictions_path": "predictions", "run_config_path": "run_config", "forecast_report_path": "report"},
        )
        manifest["status_source"] = "M4 failure"
        _persist(manifest, manifest_path, report_path)
        return _result(manifest, manifest_path, report_path)

    forecast_stage = manifest["stages"]["forecast"]
    forecast_stage["status"] = "completed"
    forecast_stage["result_status"] = "completed"
    forecast_stage["artifacts"] = _artifact_bundle(
        {
            "run_dir": forecast.get("run_dir"),
            "predictions": forecast.get("predictions_path"),
            "run_config": forecast.get("run_config_path"),
            "report": forecast.get("forecast_report_path"),
        }
    )
    _attach_stage_links(
        forecast_stage,
        forecast_stage["artifacts"],
        {"run_dir": "run_dir", "predictions_path": "predictions", "run_config_path": "run_config", "forecast_report_path": "report"},
    )
    forecast_stage["predictions_sha256"] = forecast.get("config", {}).get("predictions_sha256")
    manifest["outputs"]["forecast_run"] = forecast_stage["artifacts"]["run_dir"]
    manifest["outputs"]["forecast_predictions"] = forecast_stage["artifacts"]["predictions"]
    manifest["outputs"]["forecast_run_config"] = forecast_stage["artifacts"]["run_config"]
    manifest["outputs"]["forecast_report"] = forecast_stage["artifacts"]["report"]
    _persist(manifest, manifest_path, report_path)

    m8_before = set(output_root_path.iterdir()) if output_root_path.is_dir() else set()
    try:
        release = verify_release(
            forecast["run_dir"], history_file, history_manifest, fixture["manifest_path"], output_root=output_root_path
        )
    except Exception as exc:
        stage = manifest["stages"]["release_verification"]
        stage.update(status="blocked", result_status="failed", error=_error_record(exc))
        new_releases = _new_children(output_root_path, "m8_release_", m8_before)
        stage["artifacts"] = _partial_release_artifacts(new_releases[-1] if new_releases else None)
        _attach_stage_links(
            stage,
            stage["artifacts"],
            {"run_dir": "run_dir", "release_manifest_path": "manifest", "release_report_path": "report"},
        )
        manifest["status"] = "blocked"
        manifest["status_source"] = "M8 exception"
        _persist(manifest, manifest_path, report_path)
        return _result(manifest, manifest_path, report_path)

    release_stage = manifest["stages"]["release_verification"]
    release_stage["status"] = release["status"]
    release_stage["result_status"] = release["status"]
    release_stage["artifacts"] = _artifact_bundle(
        {
            "run_dir": release.get("run_dir"),
            "manifest": release.get("release_manifest_path"),
            "report": release.get("release_report_path"),
            "forecast_predictions": release.get("forecast_predictions_path"),
        }
    )
    _attach_stage_links(
        release_stage,
        release_stage["artifacts"],
        {"run_dir": "run_dir", "release_manifest_path": "manifest", "release_report_path": "report", "forecast_predictions_path": "forecast_predictions"},
    )
    manifest["outputs"]["release_run"] = release_stage["artifacts"]["run_dir"]
    manifest["outputs"]["release_manifest"] = release_stage["artifacts"]["manifest"]
    manifest["outputs"]["release_report"] = release_stage["artifacts"]["report"]
    review_reasons = [
        reason
        for check in release.get("checks", [])
        for reason in check.get("reasons", [])
        if check.get("status") in {"needs_review", "blocked"}
    ]
    if review_reasons:
        release_stage["reasons"] = review_reasons
    manifest["status"] = release["status"]
    manifest["status_source"] = "M8 release verification"
    _persist(manifest, manifest_path, report_path)
    return _result(manifest, manifest_path, report_path)


__all__ = [
    "FORECAST_SESSION_SCHEMA_VERSION",
    "FORECAST_SESSION_STATUSES",
    "new_forecast_session_dir",
    "run_forecast_session",
]
