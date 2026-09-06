"""Offline orchestration for one explicit M5 -> M4 -> M8 forecast session."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from .epl_fixtures import prepare_epl_fixtures
from .forecast import run_forecast
from .forecast_session_support import (
    SESSION_PREFIX as _SESSION_PREFIX,
    artifact_bundle as _artifact_bundle,
    attach_stage_links as _attach_stage_links,
    discover_new_child as _discover_new_child,
    error_record as _error_record,
    initial_manifest as _initial_manifest,
    m5_failure_evidence as _m5_failure_evidence,
    new_forecast_session_dir,
    partial_release_artifacts as _partial_release_artifacts,
    partial_run_artifacts as _partial_run_artifacts,
    persist as _persist,
    stage_paths as _stage_paths,
)
from .release_verify import verify_release
from .schema import DataValidationError

FORECAST_SESSION_SCHEMA_VERSION = "m9-forecast-session-v1"
FORECAST_SESSION_STATUSES = ("ready", "needs_review", "blocked")
_EXPECTED_STAGE_ERRORS = (DataValidationError, FileNotFoundError, OSError, TypeError, ValueError)


def _within(child: Path, parent: Path) -> bool:
    return child == parent or parent in child.parents


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
        FORECAST_SESSION_SCHEMA_VERSION,
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
    except _EXPECTED_STAGE_ERRORS as exc:
        stage = manifest["stages"]["fixture_snapshot"]
        stage.update(status="blocked", result_status="failed", error=_error_record(exc))
        stage["artifacts"], stage["artifact_discovery"] = _m5_failure_evidence(
            fixture_root, m5_before, str(exc)
        )
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
    except _EXPECTED_STAGE_ERRORS as exc:
        stage = manifest["stages"]["forecast"]
        stage.update(status="blocked", result_status="failed", error=_error_record(exc))
        forecast_dir, discovery = _discover_new_child(output_root_path, "m4_forecast_", m4_before)
        stage["artifact_discovery"] = discovery
        stage["artifacts"] = _partial_run_artifacts(forecast_dir)
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
    except _EXPECTED_STAGE_ERRORS as exc:
        stage = manifest["stages"]["release_verification"]
        stage.update(status="blocked", result_status="failed", error=_error_record(exc))
        release_dir, discovery = _discover_new_child(output_root_path, "m8_release_", m8_before)
        stage["artifact_discovery"] = discovery
        stage["artifacts"] = _partial_release_artifacts(release_dir)
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
