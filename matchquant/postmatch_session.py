"""Offline M10 orchestration: M7 local stage -> M2.1 -> M6 settlement."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .epl_local_stage import stage_local_epl_source
from .epl_sync import sync_epl
from .postmatch_session_support import (
    POSTMATCH_SESSION_PREFIX,
    active_pointer_evidence,
    artifact_bundle,
    discover_new_child,
    error_record,
    file_record,
    initial_manifest,
    load_m9_evidence,
    m7_failure_artifacts,
    new_postmatch_session_dir,
    partial_settlement_artifacts,
    persist,
    sync_failure_artifacts,
    within,
)
from .schema import DataValidationError
from .settlement import settle_forecast_run

POSTMATCH_SESSION_SCHEMA_VERSION = "m10-postmatch-session-v1"
POSTMATCH_SESSION_STATUSES = ("needs_review", "complete", "partial", "blocked")
_EXPECTED_ERRORS = (DataValidationError, FileNotFoundError, OSError, TypeError, ValueError)
_DEFAULT_DATA_ROOT = Path(r"D:\Codex\matchquant_data").resolve()


def _validate_output_root(output_root: Path, data_root: Path, m9_dir: Path) -> None:
    if within(output_root, data_root) or within(output_root, _DEFAULT_DATA_ROOT):
        raise DataValidationError("M10 output-root must not be inside D:\\Codex\\matchquant_data")
    if within(output_root, m9_dir):
        raise DataValidationError("M10 output-root must not be inside the M9 session directory")
    if any(path.name.startswith(POSTMATCH_SESSION_PREFIX) and path.is_dir() for path in (output_root, *output_root.parents)):
        raise DataValidationError("M10 output-root must not be inside an existing post-match session directory")


def _artifact_path(stage: dict[str, Any], name: str) -> str | None:
    artifact = stage.get("artifacts", {}).get(name, {})
    return artifact.get("path") if isinstance(artifact, dict) else None


def _result(manifest: dict[str, Any], manifest_path: Path, report_path: Path) -> dict[str, Any]:
    local_stage = manifest["stages"]["local_stage"]
    sync_stage = manifest["stages"]["sync"]
    settlement_stage = manifest["stages"]["settlement"]
    return {
        "status": manifest["status"],
        "session_id": manifest["session_id"],
        "session_dir": manifest["session_dir"],
        "manifest_path": str(manifest_path),
        "postmatch_manifest_path": str(manifest_path),
        "report_path": str(report_path),
        "postmatch_report_path": str(report_path),
        "staging_dir": sync_stage.get("staging_dir") or local_stage.get("snapshot_dir"),
        "snapshot_dir": local_stage.get("snapshot_dir"),
        "candidate_path": _artifact_path(sync_stage, "candidate"),
        "quality_report_path": _artifact_path(sync_stage, "quality_report"),
        "diff_report_path": _artifact_path(sync_stage, "diff_report"),
        "processed_manifest_path": _artifact_path(sync_stage, "processed_manifest"),
        "version_id": sync_stage.get("version_id"),
        "version_dir": sync_stage.get("version_dir") or _artifact_path(sync_stage, "version_dir"),
        "pointer_path": sync_stage.get("pointer_path"),
        "settlement_run_dir": settlement_stage.get("run_dir") or _artifact_path(settlement_stage, "run_dir"),
        "settlement_path": _artifact_path(settlement_stage, "settlement"),
        "metrics_path": _artifact_path(settlement_stage, "metrics"),
        "settlement_report_path": _artifact_path(settlement_stage, "report"),
        "settlement_run_config_path": _artifact_path(settlement_stage, "run_config"),
        "apply_eligibility": manifest["apply_eligibility"],
        "pointer_evidence": manifest["pointer_evidence"],
        "stages": manifest["stages"],
        "manifest": manifest,
    }


def _record_sync_stage(stage: dict[str, Any], result: dict[str, Any], status: str) -> None:
    diff = result.get("diff") if isinstance(result.get("diff"), dict) else {}
    stage.update(
        {
            "status": status,
            "result_status": result.get("status"),
            "staging_dir": result.get("staging_dir"),
            "snapshot_id": result.get("snapshot_id"),
            "counts": diff,
            "strict_380_gate_applied_by_season": result.get("strict_380_gate_applied_by_season"),
            "skipped_unfinished": result.get("skipped_unfinished"),
            "apply_blockers": result.get("apply_blockers") or [name for name in ("revised", "missing_from_candidate") if diff.get(name)],
            "candidate_sha256": result.get("candidate_sha256"),
            "quality_report_sha256": result.get("quality_report_sha256"),
            "diff_report_sha256": result.get("diff_report_sha256"),
            "version_id": result.get("version_id"),
            "version_dir": result.get("version_dir"),
            "pointer_path": result.get("pointer_path") or stage.get("pointer_path"),
        }
    )
    stage["artifacts"] = artifact_bundle(
        {
            "staging_dir": result.get("staging_dir"),
            "source_manifest": Path(result["staging_dir"]) / "raw" / "source_manifest.json" if result.get("staging_dir") else None,
            "candidate": result.get("candidate_path"),
            "processed_manifest": result.get("processed_manifest_path"),
            "quality_report": result.get("quality_report_path"),
            "diff_report": result.get("diff_report_path"),
            "version_dir": result.get("version_dir"),
            "version_manifest": result.get("version_manifest"),
            "pointer": result.get("pointer_path"),
        }
    )


def _finish_blocked(manifest: dict[str, Any], manifest_path: Path, report_path: Path, source: str) -> dict[str, Any]:
    manifest["status"] = "blocked"
    manifest["status_source"] = source
    persist(manifest, manifest_path, report_path)
    return _result(manifest, manifest_path, report_path)


def run_postmatch_session(
    session_manifest_path: str | Path,
    season: str,
    *,
    results_source_csv: str | Path | None = None,
    staging_dir: str | Path | None = None,
    apply: bool = False,
    in_progress_season: str | None = None,
    source_url: str | None = None,
    source_label: str | None = None,
    upstream_retrieved_at: str | None = None,
    data_root: str | Path = Path(r"D:\Codex\matchquant_data"),
    output_root: str | Path = Path(r"D:\Codex\matchquant_runs"),
    current_pointer_path: str | Path | None = None,
) -> dict[str, Any]:
    """Run one explicit offline M10 session without resolving any latest artifact."""

    if (results_source_csv is None) == (staging_dir is None):
        raise ValueError("provide exactly one of results_source_csv or staging_dir")
    if apply and results_source_csv is not None:
        raise ValueError("--apply requires --staging-dir from a prior reviewed dry run")
    if any(value is not None for value in (source_url, source_label, upstream_retrieved_at)) and staging_dir is not None:
        raise ValueError("source provenance options require --results-source")
    season_text = str(season).strip()
    if not season_text:
        raise ValueError("season must be explicit and non-empty")
    manifest_input = Path(session_manifest_path).expanduser().resolve()
    data_root_path = Path(data_root).expanduser().resolve()
    output_root_path = Path(output_root).expanduser().resolve()
    pointer_path = Path(current_pointer_path).expanduser().resolve() if current_pointer_path else data_root_path / "processed" / "epl_current.json"
    _validate_output_root(output_root_path, data_root_path, manifest_input.parent)

    m9: dict[str, Any] | None
    try:
        m9 = load_m9_evidence(manifest_input)
    except _EXPECTED_ERRORS as exc:
        m9 = {
            "manifest_path": manifest_input,
            "session_dir": manifest_input.parent,
            "m9_manifest": file_record(manifest_input),
        }
        session_dir = new_postmatch_session_dir(output_root_path)
        manifest_path, report_path = session_dir / "postmatch_manifest.json", session_dir / "postmatch_report.md"
        manifest = initial_manifest(
            POSTMATCH_SESSION_SCHEMA_VERSION, session_dir, m9, season_text,
            Path(results_source_csv).expanduser().resolve() if results_source_csv else None,
            Path(staging_dir).expanduser().resolve() if staging_dir else None,
            apply, in_progress_season, data_root_path, output_root_path, pointer_path,
            m9_error=error_record(exc),
        )
        manifest["status_source"] = "M9 validation"
        return _finish_blocked(manifest, manifest_path, report_path, "M9 validation")

    source_path = Path(results_source_csv).expanduser().resolve() if results_source_csv else None
    supplied_staging = Path(staging_dir).expanduser().resolve() if staging_dir else None
    session_dir = new_postmatch_session_dir(output_root_path)
    manifest_path, report_path = session_dir / "postmatch_manifest.json", session_dir / "postmatch_report.md"
    manifest = initial_manifest(
        POSTMATCH_SESSION_SCHEMA_VERSION, session_dir, m9, season_text, source_path, supplied_staging,
        apply, in_progress_season, data_root_path, output_root_path, pointer_path,
    )
    manifest["source_provenance"] = {
        "source_url": source_url,
        "source_label": source_label,
        "upstream_retrieved_at": upstream_retrieved_at,
    }
    persist(manifest, manifest_path, report_path)

    staging_path: Path
    if source_path is not None:
        staging_root = data_root_path / "staging"
        before = set(staging_root.iterdir()) if staging_root.is_dir() else set()
        try:
            staged = stage_local_epl_source(
                source_path, season_text, data_root_path, source_url=source_url,
                source_label=source_label, upstream_retrieved_at=upstream_retrieved_at,
            )
        except _EXPECTED_ERRORS as exc:
            stage = manifest["stages"]["local_stage"]
            stage.update(status="blocked", result_status="failed", error=error_record(exc))
            stage["artifacts"], stage["artifact_discovery"] = m7_failure_artifacts(data_root_path, before, str(exc))
            return _finish_blocked(manifest, manifest_path, report_path, "M7 failure")
        staging_path = Path(staged["snapshot_dir"]).expanduser().resolve()
        local_stage = manifest["stages"]["local_stage"]
        local_stage.update(
            status="completed", result_status=staged.get("status"), snapshot_dir=str(staging_path),
            manifest_path=staged.get("manifest_path"), raw_path=staged.get("raw_path"),
            raw_sha256=staged.get("raw_sha256"),
        )
        local_stage["artifacts"] = artifact_bundle(
            {"snapshot_dir": staging_path, "manifest": staged.get("manifest_path"), "raw_snapshot": staged.get("raw_path")}
        )
        manifest["inputs"]["staging_dir"] = artifact_bundle({"staging_dir": staging_path})["staging_dir"]
    else:
        if supplied_staging is None:
            raise ValueError("staging-dir is required for this mode")
        staging_path = supplied_staging
        local_stage = manifest["stages"]["local_stage"]
        local_stage["result_status"] = "explicit_staging_supplied"
        local_stage["artifacts"] = artifact_bundle({"staging_dir": staging_path})
    persist(manifest, manifest_path, report_path)

    sync_stage = manifest["stages"]["sync"]
    sync_stage["pointer_path"] = str(pointer_path)
    try:
        dry_run = sync_epl(
            seasons=[season_text], staging_dir=staging_path, data_dir=data_root_path, apply=False,
            in_progress_season=in_progress_season, baseline_path=m9["history_path"],
            processed_manifest_path=m9["history_manifest_path"], current_pointer_path=pointer_path,
        )
    except _EXPECTED_ERRORS as exc:
        sync_stage.update(status="blocked", result_status="failed", error=error_record(exc), staging_dir=str(staging_path))
        sync_stage["artifacts"] = sync_failure_artifacts(staging_path)
        return _finish_blocked(manifest, manifest_path, report_path, "M2.1 dry-run failure")
    _record_sync_stage(sync_stage, dry_run, "dry_run")
    blockers = [name for name in ("revised", "missing_from_candidate") if dry_run.get("diff", {}).get(name)]
    manifest["apply_eligibility"] = {
        "eligible": not blockers,
        "reason": "explicit --apply with this staging directory is still required" if not blockers else f"M2.1 apply blockers: {', '.join(blockers)}",
        "staging_dir": str(staging_path),
        "candidate_sha256": dry_run.get("candidate_sha256"),
    }
    if blockers:
        sync_stage.update(status="blocked", result_status="dry-run_apply_blocked", error={"type": "DataValidationError", "message": manifest["apply_eligibility"]["reason"]})
        return _finish_blocked(manifest, manifest_path, report_path, "M2.1 apply blockers")
    if not apply:
        manifest["status"] = "needs_review"
        manifest["status_source"] = "M2.1 dry-run awaiting explicit apply"
        manifest["stages"]["settlement"]["status"] = "not_run"
        persist(manifest, manifest_path, report_path)
        return _result(manifest, manifest_path, report_path)

    try:
        pointer = active_pointer_evidence(pointer_path, m9["baseline"])
    except _EXPECTED_ERRORS as exc:
        manifest["pointer_evidence"] = {"status": "blocked", "error": error_record(exc), "pointer_path": str(pointer_path)}
        sync_stage.update(status="blocked", result_status="pre_apply_pointer_error", error=error_record(exc))
        return _finish_blocked(manifest, manifest_path, report_path, "active pointer validation")
    manifest["pointer_evidence"] = pointer
    if pointer["status"] != "matched":
        sync_stage.update(
            status="blocked", result_status="pre_apply_pointer_mismatch",
            error={"type": "DataValidationError", "message": "active pointer does not match the M9 baseline"},
        )
        manifest["apply_eligibility"].update(eligible=False, reason="active pointer does not match the M9 baseline")
        return _finish_blocked(manifest, manifest_path, report_path, "active pointer mismatch")

    version_root = data_root_path / "processed" / "versions"
    versions_before = set(version_root.iterdir()) if version_root.is_dir() else set()
    try:
        applied = sync_epl(
            seasons=[season_text], staging_dir=staging_path, data_dir=data_root_path, apply=True,
            in_progress_season=in_progress_season, baseline_path=m9["history_path"],
            processed_manifest_path=m9["history_manifest_path"], current_pointer_path=pointer_path,
        )
    except _EXPECTED_ERRORS as exc:
        sync_stage.update(status="blocked", result_status="apply_failed", error=error_record(exc))
        sync_stage["artifacts"] = sync_failure_artifacts(staging_path)
        version_dir, discovery = discover_new_child(version_root, "m2.1-", versions_before)
        sync_stage["artifact_discovery"] = discovery
        if version_dir is not None:
            sync_stage["artifacts"]["version_dir"] = artifact_bundle({"version_dir": version_dir})["version_dir"]
        return _finish_blocked(manifest, manifest_path, report_path, "M2.1 apply failure")
    _record_sync_stage(sync_stage, applied, "applied")
    if applied.get("status") != "applied":
        sync_stage.update(status="blocked", result_status=applied.get("status"), error={"type": "DataValidationError", "message": "M2.1 did not return applied status"})
        return _finish_blocked(manifest, manifest_path, report_path, "M2.1 apply result")
    manifest["apply_eligibility"].update(applied=True)
    manifest["input_mutation"]["active_pointer_updated"] = True
    settlement_stage = manifest["stages"]["settlement"]
    m6_before = set(output_root_path.iterdir()) if output_root_path.is_dir() else set()
    try:
        settlement = settle_forecast_run(
            m9["forecast_run_dir"], applied["pointer"]["canonical"], output_root=output_root_path,
        )
    except _EXPECTED_ERRORS as exc:
        settlement_stage.update(status="blocked", result_status="failed", error=error_record(exc))
        settlement_dir, discovery = discover_new_child(output_root_path, "m6_settlement_", m6_before)
        settlement_stage["artifact_discovery"] = discovery
        settlement_stage["artifacts"] = partial_settlement_artifacts(settlement_dir)
        return _finish_blocked(manifest, manifest_path, report_path, "M6 failure after applied history")
    settlement_stage.update(status=settlement.get("status"), result_status=settlement.get("status"), run_dir=settlement.get("run_dir"), counts=settlement.get("counts"), metrics=settlement.get("metrics"))
    settlement_stage["artifacts"] = artifact_bundle(
        {"run_dir": settlement.get("run_dir"), "settlement": settlement.get("settlement_path"), "metrics": settlement.get("metrics_path"), "report": settlement.get("settlement_report_path"), "run_config": settlement.get("run_config_path")}
    )
    overall = settlement.get("status")
    if m9["status"] == "needs_review" or overall == "needs_review":
        overall = "needs_review"
    if overall not in {"complete", "partial", "needs_review"}:
        return _finish_blocked(manifest, manifest_path, report_path, "M6 returned unsupported status")
    manifest["status"] = overall
    manifest["status_source"] = "M6 settlement" if m9["status"] == "ready" else "M9 needs_review retained through M6"
    persist(manifest, manifest_path, report_path)
    return _result(manifest, manifest_path, report_path)


__all__ = ["POSTMATCH_SESSION_SCHEMA_VERSION", "POSTMATCH_SESSION_STATUSES", "run_postmatch_session"]
