"""Evidence and persistence helpers for the offline M10 post-match session."""

from __future__ import annotations

import json
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from .io import file_sha256, write_json
from .schema import DataValidationError

POSTMATCH_SESSION_PREFIX = "m10_postmatch_session_"
_SHA256 = re.compile(r"^[0-9a-fA-F]{64}$")
_POINTER_SCHEMA = "m2.1-current-pointer-v1"


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DataValidationError(f"could not read {label} '{path}': {exc}") from exc
    if not isinstance(payload, dict):
        raise DataValidationError(f"{label} root must be an object: {path}")
    return payload


def _resolve_path(value: object, base: Path) -> Path:
    if value is None or not str(value).strip():
        raise DataValidationError("recorded path is missing")
    candidate = Path(str(value)).expanduser()
    return (candidate if candidate.is_absolute() else base / candidate).resolve()


def file_record(path: str | Path | None) -> dict[str, Any]:
    candidate = Path(path).expanduser().resolve() if path is not None else None
    return {
        "path": str(candidate) if candidate else None,
        "exists": bool(candidate and candidate.is_file()),
        "sha256": file_sha256(candidate) if candidate and candidate.is_file() else None,
    }


def _directory_record(path: str | Path | None) -> dict[str, Any]:
    candidate = Path(path).expanduser().resolve() if path is not None else None
    return {"path": str(candidate) if candidate else None, "exists": bool(candidate and candidate.is_dir()), "sha256": None}


def within(child: Path, parent: Path) -> bool:
    return child == parent or parent in child.parents


def artifact_bundle(paths: Mapping[str, str | Path | None]) -> dict[str, dict[str, Any]]:
    return {
        name: _directory_record(path) if name.endswith("_dir") else file_record(path)
        for name, path in paths.items()
    }


def _require_file_record(value: object, label: str, base: Path) -> tuple[Path, dict[str, Any]]:
    if not isinstance(value, Mapping):
        raise DataValidationError(f"M9 {label} artifact record is missing")
    path = _resolve_path(value.get("path"), base)
    expected = value.get("sha256")
    if not isinstance(expected, str) or _SHA256.fullmatch(expected) is None:
        raise DataValidationError(f"M9 {label} SHA-256 evidence is missing or invalid")
    actual = file_record(path)
    if not actual["exists"]:
        raise FileNotFoundError(f"M9 {label} does not exist: {path}")
    if actual["sha256"].lower() != expected.lower():
        raise DataValidationError(f"M9 {label} SHA-256 does not match the recorded artifact")
    return path, actual


def _require_directory_record(value: object, label: str, base: Path) -> Path:
    if not isinstance(value, Mapping):
        raise DataValidationError(f"M9 {label} artifact record is missing")
    path = _resolve_path(value.get("path"), base)
    if not path.is_dir():
        raise FileNotFoundError(f"M9 {label} directory does not exist: {path}")
    return path


def load_m9_evidence(session_manifest_path: str | Path) -> dict[str, Any]:
    """Validate one M9 manifest and return only its recorded, hash-locked inputs."""

    manifest_path = Path(session_manifest_path).expanduser().resolve()
    manifest = _read_json(manifest_path, "M9 session manifest")
    if manifest.get("schema_version") != "m9-forecast-session-v1":
        raise DataValidationError("unsupported M9 session manifest schema")
    status = manifest.get("status")
    if status not in {"ready", "needs_review"}:
        raise DataValidationError("M10 requires an M9 session with ready or needs_review status")
    inputs = manifest.get("inputs")
    stages = manifest.get("stages")
    if not isinstance(inputs, Mapping) or not isinstance(stages, Mapping):
        raise DataValidationError("M9 session manifest must contain inputs and stages objects")
    forecast_stage = stages.get("forecast")
    if not isinstance(forecast_stage, Mapping) or forecast_stage.get("status") != "completed":
        raise DataValidationError("M9 forecast stage is not completed")
    history_path, history = _require_file_record(inputs.get("history"), "history", manifest_path.parent)
    history_manifest_path, history_manifest = _require_file_record(
        inputs.get("history_manifest"), "history manifest", manifest_path.parent
    )
    history_payload = _read_json(history_manifest_path, "M9 history manifest")
    output = history_payload.get("output")
    if isinstance(output, Mapping) and output.get("sha256"):
        if str(output["sha256"]).lower() != history["sha256"].lower():
            raise DataValidationError("M9 history manifest output SHA-256 does not match the recorded history")

    artifacts = forecast_stage.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise DataValidationError("M9 forecast artifacts are missing")
    run_dir = _require_directory_record(artifacts.get("run_dir"), "M4 run", manifest_path.parent)
    config_path, config = _require_file_record(artifacts.get("run_config"), "M4 run config", manifest_path.parent)
    predictions_path, predictions = _require_file_record(artifacts.get("predictions"), "M4 predictions", manifest_path.parent)
    expected_config = (run_dir / "run_config.json").resolve()
    expected_predictions = (run_dir / "predictions.csv").resolve()
    if config_path != expected_config or predictions_path != expected_predictions:
        raise DataValidationError("M9 forecast artifacts do not point to the recorded M4 run files")
    recorded_prediction_hash = forecast_stage.get("predictions_sha256")
    if recorded_prediction_hash is not None and str(recorded_prediction_hash).lower() != predictions["sha256"].lower():
        raise DataValidationError("M9 forecast prediction hash does not match predictions.csv")
    from .settlement_validation import load_forecast_run

    forecast = load_forecast_run(run_dir)
    if forecast["config_hash"].lower() != config["sha256"].lower() or forecast["prediction_hash"].lower() != predictions["sha256"].lower():
        raise DataValidationError("M9 M4 artifact hashes disagree with the loaded forecast run")
    declared_history = forecast["config"].get("history")
    if not isinstance(declared_history, Mapping):
        raise DataValidationError("M4 run config history evidence is missing")
    declared_history_path = _resolve_path(declared_history.get("path"), run_dir)
    declared_history_hash = declared_history.get("sha256")
    if declared_history_path != history_path or not isinstance(declared_history_hash, str) or declared_history_hash.lower() != history["sha256"].lower():
        raise DataValidationError("M4 history evidence does not match the M9 baseline")
    baseline = {
        "history": history,
        "history_manifest": history_manifest,
        "forecast_run_config": config,
        "forecast_predictions": predictions,
    }
    return {
        "manifest_path": manifest_path,
        "manifest": manifest,
        "status": status,
        "session_dir": manifest_path.parent,
        "history_path": history_path,
        "history_manifest_path": history_manifest_path,
        "forecast_run_dir": run_dir,
        "forecast_config_path": config_path,
        "forecast_predictions_path": predictions_path,
        "forecast": forecast,
        "baseline": baseline,
        "m9_manifest": file_record(manifest_path),
    }


def active_pointer_evidence(pointer_path: Path, baseline: Mapping[str, Any]) -> dict[str, Any]:
    """Compare the active pointer's two files with the locked M9 baseline hashes."""

    pointer_path = Path(pointer_path).expanduser().resolve()
    if not pointer_path.is_file():
        raise FileNotFoundError(f"active current pointer does not exist: {pointer_path}")
    pointer = _read_json(pointer_path, "active current pointer")
    if pointer.get("schema_version") != _POINTER_SCHEMA:
        raise DataValidationError(f"unsupported current pointer schema: {pointer_path}")
    canonical = _resolve_path(pointer.get("canonical"), pointer_path.parent)
    processed_manifest = _resolve_path(pointer.get("processed_manifest"), pointer_path.parent)
    canonical_record = file_record(canonical)
    processed_record = file_record(processed_manifest)
    pointer_hash = pointer.get("canonical_sha256")
    pointer_hash_valid = isinstance(pointer_hash, str) and _SHA256.fullmatch(pointer_hash) is not None
    pointer_canonical_match = bool(pointer_hash_valid and canonical_record["sha256"] and pointer_hash.lower() == canonical_record["sha256"].lower())
    canonical_match = canonical_record["sha256"] == baseline.get("history", {}).get("sha256")
    manifest_match = processed_record["sha256"] == baseline.get("history_manifest", {}).get("sha256")
    return {
        "status": "matched" if pointer_canonical_match and canonical_match and manifest_match else "mismatch",
        "pointer_path": str(pointer_path),
        "pointer_schema": pointer.get("schema_version"),
        "canonical": canonical_record,
        "processed_manifest": processed_record,
        "pointer_canonical_hash_matches_file": pointer_canonical_match,
        "canonical_hash_matches_m9": canonical_match,
        "processed_manifest_hash_matches_m9": manifest_match,
        "baseline_canonical_sha256": baseline.get("history", {}).get("sha256"),
        "baseline_processed_manifest_sha256": baseline.get("history_manifest", {}).get("sha256"),
    }


def discover_new_child(root: Path, prefix: str, before: set[Path]) -> tuple[Path | None, dict[str, Any]]:
    before_resolved = {Path(path).resolve() for path in before}
    candidates = sorted(
        path for path in root.iterdir()
        if path.is_dir() and path.name.startswith(prefix) and path.resolve() not in before_resolved
    ) if root.is_dir() else []
    if len(candidates) == 1:
        status, bound = "unique", str(candidates[0].resolve())
    elif candidates:
        status, bound = "ambiguous", None
    else:
        status, bound = "none", None
    return (candidates[0] if len(candidates) == 1 else None), {
        "status": status,
        "prefix": prefix,
        "candidates": [str(path.resolve()) for path in candidates],
        "bound_path": bound,
    }


def new_postmatch_session_dir(output_root: str | Path) -> Path:
    root = Path(output_root).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    while True:
        name = f"{POSTMATCH_SESSION_PREFIX}{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}_{uuid.uuid4().hex[:8]}"
        path = root / name
        try:
            path.mkdir()
            return path
        except FileExistsError:
            continue


def initial_manifest(
    schema_version: str,
    session_dir: Path,
    m9: dict[str, Any] | None,
    season: str,
    results_source: Path | None,
    staging_dir: Path | None,
    apply: bool,
    in_progress_season: str | None,
    data_root: Path,
    output_root: Path,
    pointer_path: Path,
    m9_error: dict[str, str] | None = None,
) -> dict[str, Any]:
    manifest_path, report_path = session_dir / "postmatch_manifest.json", session_dir / "postmatch_report.md"
    return {
        "schema_version": schema_version,
        "command": "matchquant postmatch-session",
        "session_id": session_dir.name,
        "session_dir": str(session_dir),
        "generated_at_utc": utc_now(),
        "status": "blocked",
        "status_source": "pending_stages",
        "inputs": {
            "session_manifest": file_record(m9["manifest_path"] if m9 else None),
            "season": str(season),
            "in_progress_season": in_progress_season,
            "results_source": file_record(results_source),
            "staging_dir": _directory_record(staging_dir),
            "data_root": str(data_root),
            "output_root": str(output_root),
            "current_pointer": file_record(pointer_path),
        },
        "m9": {
            "status": m9.get("status") if m9 else None,
            "session_dir": str(m9["session_dir"]) if m9 else None,
            "session_manifest": m9.get("m9_manifest") if m9 else None,
            "history": m9.get("baseline", {}).get("history") if m9 else None,
            "history_manifest": m9.get("baseline", {}).get("history_manifest") if m9 else None,
            "forecast_run_dir": str(m9.get("forecast_run_dir")) if m9 and m9.get("forecast_run_dir") else None,
            "forecast_run_config": m9.get("baseline", {}).get("forecast_run_config") if m9 else None,
            "forecast_predictions": m9.get("baseline", {}).get("forecast_predictions") if m9 else None,
            "error": m9_error,
        },
        "source_provenance": {"source_url": None, "source_label": None, "upstream_retrieved_at": None},
        "mode": {"apply_requested": bool(apply), "source_mode": results_source is not None, "staging_mode": staging_dir is not None},
        "stages": {
            "local_stage": {"name": "M7", "status": "not_run", "result_status": None, "artifacts": {}},
            "sync": {"name": "M2.1", "status": "not_run", "result_status": None, "artifacts": {}},
            "settlement": {"name": "M6", "status": "not_run", "result_status": None, "artifacts": {}},
        },
        "pointer_evidence": {"status": "not_run"},
        "apply_eligibility": {"eligible": False, "reason": "M2.1 dry-run has not completed"},
        "outputs": {
            "session_directory": str(session_dir),
            "postmatch_manifest": str(manifest_path),
            "postmatch_report": str(report_path),
            "postmatch_report_sha256": None,
        },
        "input_mutation": {
            "network_accessed": False,
            "m9_manifest_modified": False,
            "m4_forecast_modified": False,
            "history_csv_modified": False,
            "history_manifest_modified": False,
            "results_source_modified": False,
            "active_pointer_updated": False,
            "child_artifacts_copied_into_session": False,
        },
        "limitations": [
            "M10 完全离线；只处理用户显式提供的 M9 manifest、E0 结果 staging、M2.1 candidate 和 M6 settlement。",
            "M10 不声明历史或赛果当前，不声明概率已校准、盈利能力、投注价值或未来表现。",
            "M2.1 apply 是前向不可回滚的历史版本变更；若 M6 失败，需单独处理结算，不回滚已应用版本。",
        ],
        "status_rules": {
            "needs_review": "dry-run 待明确 apply、携带 M9 needs_review 或 M6 needs_review",
            "complete": "addition-only apply 后所有预测行安全结算且无 M9 review",
            "partial": "addition-only apply 后存在 unresolved 但无 ambiguous/time_conflict",
            "blocked": "输入、M7、M2.1、pointer coherence 或 M6 存在硬失败",
        },
    }


def error_record(exc: Exception) -> dict[str, str]:
    return {"type": type(exc).__name__, "message": str(exc)}


def m7_failure_artifacts(data_root: Path, before: set[Path], message: str) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    match = re.search(r"manifest:\s*(.+)$", message, re.IGNORECASE)
    declared = Path(match.group(1).strip()).expanduser().resolve() if match else None
    snapshot = declared.parent.parent if declared and declared.name == "source_manifest.json" else None
    staging_root = (data_root / "staging").resolve()
    if snapshot is not None and (not snapshot.name.startswith("epl_local_") or not within(snapshot, staging_root)):
        snapshot = None
    discovery = {"status": "declared", "prefix": "epl_local_", "candidates": [], "bound_path": str(snapshot)} if snapshot else None
    if snapshot is None:
        snapshot, discovery = discover_new_child(data_root / "staging", "epl_local_", before)
    artifacts = artifact_bundle({
        "snapshot_dir": snapshot,
        "manifest": snapshot / "raw" / "source_manifest.json" if snapshot else None,
        "raw_snapshot": None,
    })
    if snapshot and (snapshot / "raw").is_dir():
        raw_files = sorted(path for path in (snapshot / "raw").iterdir() if path.is_file() and path.name != "source_manifest.json")
        if len(raw_files) == 1:
            artifacts["raw_snapshot"] = file_record(raw_files[0])
        elif discovery is not None and len(raw_files) > 1:
            discovery["raw_candidates"] = [str(path.resolve()) for path in raw_files]
            discovery["raw_status"] = "ambiguous"
    return artifacts, discovery or {"status": "none", "prefix": "epl_local_", "candidates": [], "bound_path": None}


def sync_failure_artifacts(staging_dir: Path) -> dict[str, dict[str, Any]]:
    return artifact_bundle({
        "staging_dir": staging_dir,
        "source_manifest": staging_dir / "raw" / "source_manifest.json",
        "candidate": staging_dir / "processed" / "epl_canonical.csv",
        "processed_manifest": staging_dir / "processed" / "epl_processed_manifest.json",
        "quality_report": staging_dir / "reports" / "epl_quality_report.json",
        "diff_report": staging_dir / "reports" / "epl_sync_diff_report.json",
    })


def partial_settlement_artifacts(run_dir: Path | None) -> dict[str, dict[str, Any]]:
    return artifact_bundle({
        "run_dir": run_dir,
        "settlement": run_dir / "settlement.csv" if run_dir else None,
        "metrics": run_dir / "metrics.json" if run_dir else None,
        "report": run_dir / "settlement_report.md" if run_dir else None,
        "run_config": run_dir / "run_config.json" if run_dir else None,
    })


def persist(manifest: dict[str, Any], manifest_path: Path, report_path: Path) -> None:
    from .postmatch_session_report import write_postmatch_session_report

    write_postmatch_session_report(manifest, report_path)
    manifest["outputs"]["postmatch_report_sha256"] = file_sha256(report_path)
    write_json(manifest, manifest_path)


__all__ = [
    "POSTMATCH_SESSION_PREFIX", "active_pointer_evidence", "artifact_bundle", "discover_new_child",
    "error_record", "file_record", "initial_manifest", "load_m9_evidence", "m7_failure_artifacts",
    "new_postmatch_session_dir", "partial_settlement_artifacts", "persist", "sync_failure_artifacts",
    "within",
]
