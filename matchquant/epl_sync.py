"""Baseline resolution, staging reports, and version activation for M2.1."""

from __future__ import annotations

import json
import os
import shutil
import uuid
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from .epl import CANONICAL_COLUMNS, _season_list, normalize_season, read_source_csv
from .epl_sync_source import (
    CURRENT_POINTER_FILENAME,
    CURRENT_POINTER_VERSION,
    SYNC_ADAPTER_VERSION,
    SYNC_DIFF_REPORT_VERSION,
    SYNC_PROCESSED_MANIFEST_VERSION,
    SYNC_QUALITY_REPORT_VERSION,
    _analyse_source,
    _download_snapshot,
    _source_error,
    _utc_now,
    audit_sync_source_frame,
    compare_canonical,
    convert_sync_source_frame,
)
from .io import file_sha256, load_matches, write_json
from .schema import DataValidationError, validate_matches


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DataValidationError(f"could not read JSON '{path}': {exc}") from exc
    if not isinstance(payload, dict):
        raise DataValidationError(f"JSON root must be an object: {path}")
    return payload


def _cutoff(frame: pd.DataFrame) -> str | None:
    if frame.empty:
        return None
    latest = pd.to_datetime(frame["date"], errors="coerce").max()
    if pd.isna(latest):
        return None
    return latest.strftime("%Y-%m-%d") if latest.hour == 0 and latest.minute == 0 and latest.second == 0 else latest.isoformat()


def _season_cutoffs(frame: pd.DataFrame) -> dict[str, str | None]:
    if frame.empty:
        return {}
    return {str(season): _cutoff(part) for season, part in frame.groupby("season", sort=True)}


def _season_policy_by_season(
    selected: tuple[str, ...], reports: list[dict[str, Any]], in_progress_season: str | None
) -> dict[str, dict[str, Any]]:
    report_by_season = {str(report.get("season")): report for report in reports if report.get("season")}
    return {
        season: {
            "in_progress_season": in_progress_season,
            "declared_in_progress_season": in_progress_season,
            "season_is_declared_in_progress": season == in_progress_season,
            "strict_380_gate_applied": season != in_progress_season,
            "match_count_policy": "in_progress_exception" if season == in_progress_season else "strict_complete_season",
            "completed_match_count": report_by_season.get(season, {}).get("completed_match_count"),
            "expected_matches": report_by_season.get(season, {}).get("expected_matches", 380),
            "meets_expected_matches": report_by_season.get(season, {}).get("meets_expected_matches"),
        }
        for season in selected
    }


def _resolve_path(value: Any, base: Path) -> Path:
    if value is None or str(value).strip() == "":
        raise DataValidationError("current pointer is missing a required path")
    path = Path(str(value))
    return path if path.is_absolute() else base / path


def _load_baseline(
    data_dir: Path,
    canonical_path: Path | None,
    processed_manifest_path: Path | None,
    current_pointer_path: Path | None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    pointer_path = current_pointer_path or data_dir / "processed" / CURRENT_POINTER_FILENAME
    pointer: dict[str, Any] | None = None
    if canonical_path is None and processed_manifest_path is None and pointer_path.is_file():
        pointer = _read_json(pointer_path)
        if pointer.get("schema_version") != CURRENT_POINTER_VERSION:
            raise DataValidationError(f"unsupported current pointer schema: {pointer_path}")
        canonical_path = _resolve_path(pointer.get("canonical"), pointer_path.parent)
        processed_manifest_path = _resolve_path(pointer.get("processed_manifest"), pointer_path.parent)
    canonical_path = canonical_path or data_dir / "processed" / "epl_canonical.csv"
    processed_manifest_path = processed_manifest_path or data_dir / "processed" / "epl_processed_manifest.json"
    if not canonical_path.is_file():
        raise FileNotFoundError(f"active baseline canonical CSV does not exist: {canonical_path}")
    if not processed_manifest_path.is_file():
        raise FileNotFoundError(f"active baseline processed manifest does not exist: {processed_manifest_path}")
    baseline = load_matches(canonical_path)
    manifest = _read_json(processed_manifest_path)
    expected_hash = manifest.get("output", {}).get("sha256")
    actual_hash = file_sha256(canonical_path)
    if expected_hash and expected_hash != actual_hash:
        raise DataValidationError(f"baseline canonical SHA-256 mismatch: manifest={expected_hash} actual={actual_hash}")
    if pointer and pointer.get("canonical_sha256") and pointer["canonical_sha256"] != actual_hash:
        raise DataValidationError(f"current pointer SHA-256 mismatch: pointer={pointer['canonical_sha256']} actual={actual_hash}")
    version = (pointer or {}).get("dataset_version") or manifest.get("dataset_version") or f"m2-legacy-{actual_hash[:16]}"
    return baseline, {
        "version": str(version),
        "canonical_path": str(canonical_path),
        "processed_manifest_path": str(processed_manifest_path),
        "canonical_sha256": actual_hash,
        "processed_manifest_sha256": file_sha256(processed_manifest_path),
        "manifest": manifest,
        "data_cutoff": _cutoff(baseline),
        "season_cutoffs": _season_cutoffs(baseline),
        "current_pointer_path": str(pointer_path),
    }


def _quality_report(
    snapshot: Path,
    manifest: dict[str, Any],
    selected: tuple[str, ...],
    reports: list[dict[str, Any]],
    errors: list[str],
    in_progress_season: str | None,
) -> dict[str, Any]:
    warnings = [warning for report in reports for warning in report.get("warnings", [])]
    failed = bool(errors) or any(report.get("status") == "failed" for report in reports)
    season_policy = _season_policy_by_season(selected, reports, in_progress_season)
    return {
        "schema_version": SYNC_QUALITY_REPORT_VERSION,
        "adapter_version": SYNC_ADAPTER_VERSION,
        "source_manifest": str(snapshot / "raw" / "source_manifest.json"),
        "source_manifest_sha256": file_sha256(snapshot / "raw" / "source_manifest.json"),
        "source": manifest.get("source"),
        "competition": manifest.get("competition"),
        "division": manifest.get("division"),
        "snapshot_id": manifest.get("snapshot_id", snapshot.name),
        "generated_at": _utc_now(),
        "requested_seasons": list(selected),
        "in_progress_season": in_progress_season,
        "declared_in_progress_season": in_progress_season,
        "season_policy": season_policy,
        "strict_380_gate_applied_by_season": {season: policy["strict_380_gate_applied"] for season, policy in season_policy.items()},
        "source_urls": [str(url) for url in (manifest.get("source_urls") or [entry.get("url") for entry in manifest.get("files", [])])],
        "data_cutoff": max((report["latest_date"] for report in reports if report.get("latest_date")), default=None),
        "status": "failed" if failed else ("warning" if warnings else "ok"),
        "errors": errors,
        "seasons": reports,
        "totals": {
            "source_rows": int(sum(report.get("source_rows", 0) for report in reports)),
            "completed_match_count": int(sum(report.get("completed_match_count", 0) for report in reports)),
            "skipped_unfinished_count": int(sum(report.get("skipped_unfinished_count", 0) for report in reports)),
            "invalid_rows": int(sum(report.get("invalid_rows", 0) for report in reports)),
            "duplicate_rows": int(sum(report.get("duplicate_rows", 0) for report in reports)),
        },
    }


def _write_stage_outputs(
    snapshot: Path,
    baseline: pd.DataFrame,
    baseline_meta: dict[str, Any],
    selected: tuple[str, ...],
    apply_requested: bool,
    in_progress_season: str | None,
) -> dict[str, Any]:
    manifest = _read_json(snapshot / "raw" / "source_manifest.json")
    entries = manifest.get("files")
    if not isinstance(entries, list):
        raise DataValidationError("sync source manifest field 'files' must be a list")
    by_season: dict[str, dict[str, Any]] = {}
    for entry in entries:
        if not isinstance(entry, dict) or "season" not in entry:
            raise DataValidationError("sync source manifest contains an invalid file entry")
        season = normalize_season(str(entry["season"]))
        if season in by_season:
            raise DataValidationError(f"sync source manifest contains duplicate season {season}")
        by_season[season] = entry
    reports: list[dict[str, Any]] = []
    errors: list[str] = []
    staged_frames: list[pd.DataFrame] = []
    for season in selected:
        entry = by_season.get(season)
        if entry is None:
            message = f"season {season} is absent from sync source manifest"
            errors.append(message)
            reports.append({"season": season, "status": "failed", "errors": [{"kind": "missing_season", "message": message}], "warnings": []})
            continue
        filename = str(entry.get("raw_filename", ""))
        report: dict[str, Any] | None = None
        try:
            if not filename or Path(filename).name != filename:
                raise DataValidationError("sync source manifest has an unsafe or empty raw filename")
            if entry.get("download_status", entry.get("status")) == "failed" or entry.get("parse_status") == "failed":
                raise DataValidationError(str(entry.get("error") or "source file is unavailable or unparsed"))
            raw_path = snapshot / "raw" / filename
            if not raw_path.is_file():
                raise FileNotFoundError(f"raw file does not exist: {raw_path}")
            actual_hash = file_sha256(raw_path)
            if not entry.get("sha256") or actual_hash != entry.get("sha256"):
                raise DataValidationError(f"raw SHA-256 mismatch: manifest={entry.get('sha256')} actual={actual_hash}")
            source = read_source_csv(raw_path)
            report, converted = _analyse_source(source, season, in_progress_season)
            report.update(raw_filename=filename, source_url=entry.get("url"), retrieved_at=entry.get("retrieved_at"), raw_sha256=actual_hash)
            reports.append(report)
            if report["status"] == "failed" or converted is None:
                errors.append(f"{season}: {_source_error(report)}")
            else:
                staged_frames.append(converted)
        except (DataValidationError, FileNotFoundError, OSError, ValueError) as exc:
            if report is None:
                report = {"season": season, "status": "failed", "errors": [{"kind": "source_error", "message": str(exc)}], "warnings": []}
            reports.append(report)
            errors.append(f"{season}: {exc}")
    quality = _quality_report(snapshot, manifest, selected, reports, errors, in_progress_season)
    reports_dir, processed_dir = snapshot / "reports", snapshot / "processed"
    reports_dir.mkdir(exist_ok=True)
    processed_dir.mkdir(exist_ok=True)
    quality_path = reports_dir / "epl_quality_report.json"
    write_json(quality, quality_path)
    if errors:
        raise DataValidationError(f"EPL sync validation failed; quality report: {quality_path}; first error: {errors[0]}")
    carried = baseline.loc[~baseline["season"].isin(selected), CANONICAL_COLUMNS]
    staged = pd.concat(staged_frames, ignore_index=True) if staged_frames else pd.DataFrame(columns=CANONICAL_COLUMNS)
    staged_path = processed_dir / "epl_staged_canonical.csv"
    if not staged.empty:
        staged = validate_matches(staged).sort_values(["_timestamp", "match_id"], kind="mergesort").loc[:, CANONICAL_COLUMNS].reset_index(drop=True)
    staged.to_csv(staged_path, index=False, encoding="utf-8")
    staged_hash = file_sha256(staged_path)
    candidate = pd.concat([carried, staged], ignore_index=True)
    if candidate.empty:
        raise DataValidationError(f"EPL sync candidate is empty; quality report: {quality_path}")
    validated = validate_matches(candidate)
    candidate = validated.sort_values(["_timestamp", "match_id"], kind="mergesort").loc[:, CANONICAL_COLUMNS].reset_index(drop=True)
    candidate_path = processed_dir / "epl_canonical.csv"
    candidate.to_csv(candidate_path, index=False, encoding="utf-8")
    candidate_hash = file_sha256(candidate_path)
    quality.update(candidate_data_cutoff=_cutoff(candidate), candidate_rows=int(len(candidate)))
    write_json(quality, quality_path)
    quality_hash = file_sha256(quality_path)
    diff = compare_canonical(baseline, candidate)
    skipped_unfinished = [item for report in reports for item in report.get("skipped_unfinished", [])]
    diff["counts"]["skipped_unfinished"] = len(skipped_unfinished)
    diff["details"]["skipped_unfinished"] = skipped_unfinished
    diff["skipped_unfinished"] = skipped_unfinished
    diff_path = reports_dir / "epl_sync_diff_report.json"
    report_payload: dict[str, Any] = {
        "schema_version": SYNC_DIFF_REPORT_VERSION,
        "adapter_version": SYNC_ADAPTER_VERSION,
        "generated_at": _utc_now(),
        "mode": "apply" if apply_requested else "dry-run",
        "source": manifest.get("source"),
        "competition": manifest.get("competition"),
        "division": manifest.get("division"),
        "source_urls": manifest.get("source_urls") or [entry.get("url") for entry in entries],
        "retrieved_at": [entry.get("retrieved_at") for entry in entries],
        "requested_seasons": list(selected),
        "in_progress_season": in_progress_season,
        "declared_in_progress_season": in_progress_season,
        "season_policy": quality["season_policy"],
        "strict_380_gate_applied_by_season": quality["strict_380_gate_applied_by_season"],
        "snapshot_id": manifest.get("snapshot_id", snapshot.name),
        "staging_dir": str(snapshot),
        "baseline": {"version": baseline_meta["version"], "canonical": baseline_meta["canonical_path"], "processed_manifest": baseline_meta["processed_manifest_path"], "canonical_sha256": baseline_meta["canonical_sha256"], "processed_manifest_sha256": baseline_meta["processed_manifest_sha256"], "data_cutoff": baseline_meta["data_cutoff"], "season_cutoffs": baseline_meta["season_cutoffs"]},
        "candidate": {"canonical": str(candidate_path), "canonical_sha256": candidate_hash, "rows": int(len(candidate)), "data_cutoff": _cutoff(candidate), "season_cutoffs": _season_cutoffs(candidate)},
        "staged_input": {"canonical": str(staged_path), "canonical_sha256": staged_hash, "rows": int(len(staged)), "source_manifest_sha256": file_sha256(snapshot / "raw" / "source_manifest.json")},
        "data_cutoffs": {"baseline": baseline_meta["data_cutoff"], "source": manifest.get("data_cutoff"), "candidate": _cutoff(candidate), "source_seasons": {report["season"]: report.get("latest_date") for report in reports}},
        "quality_report": {"path": str(quality_path), "sha256": quality_hash},
        "diff": diff,
        "counts": diff["counts"],
        "details": diff["details"],
        "apply": {"requested": bool(apply_requested), "eligible": not diff["counts"]["revised"] and not diff["counts"]["missing_from_candidate"], "status": "pending", "blockers": [name for name in ("revised", "missing_from_candidate") if diff["counts"][name]]},
    }
    if report_payload["apply"]["blockers"] and apply_requested:
        report_payload["apply"]["status"] = "blocked_manual_review"
    elif not apply_requested:
        report_payload["apply"]["status"] = "dry_run_no_change"
    else:
        report_payload["apply"]["status"] = "applied"
    write_json(report_payload, diff_path)
    diff_hash = file_sha256(diff_path)
    processed_path = processed_dir / "epl_processed_manifest.json"
    processed = {
        "schema_version": SYNC_PROCESSED_MANIFEST_VERSION,
        "adapter_version": SYNC_ADAPTER_VERSION,
        "canonical_format_version": "m1-canonical-v1",
        "dataset_version": f"candidate-{snapshot.name}",
        "source_manifest": str(snapshot / "raw" / "source_manifest.json"),
        "source_manifest_sha256": file_sha256(snapshot / "raw" / "source_manifest.json"),
        "generated_at": _utc_now(),
        "requested_seasons": list(selected),
        "in_progress_season": in_progress_season,
        "declared_in_progress_season": in_progress_season,
        "season_policy": quality["season_policy"],
        "strict_380_gate_applied_by_season": quality["strict_380_gate_applied_by_season"],
        "seasons": sorted(str(value) for value in candidate["season"].unique()),
        "data_cutoff": _cutoff(candidate),
        "inputs": [{"season": report["season"], "raw_filename": report.get("raw_filename"), "url": report.get("source_url"), "sha256": report.get("raw_sha256")} for report in reports],
        "baseline": {"version": baseline_meta["version"], "canonical_sha256": baseline_meta["canonical_sha256"]},
        "output": {"filename": str(candidate_path), "sha256": candidate_hash, "rows": int(len(candidate)), "columns": list(CANONICAL_COLUMNS)},
        "quality_report": str(quality_path),
        "quality_report_sha256": quality_hash,
        "diff_report": str(diff_path),
        "diff_report_sha256": diff_hash,
        "quality_status": quality["status"],
        "apply": report_payload["apply"],
    }
    write_json(processed, processed_path)
    return {"manifest": manifest, "quality": quality, "diff": report_payload, "candidate": candidate, "candidate_path": candidate_path, "quality_path": quality_path, "diff_path": diff_path, "processed_path": processed_path, "candidate_hash": candidate_hash, "quality_hash": quality_hash, "diff_hash": diff_hash}


def _atomic_write_json(payload: dict[str, Any], path: Path) -> None:
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        write_json(payload, temporary)
        os.replace(temporary, path)
    except Exception:
        if temporary.exists():
            temporary.unlink()
        raise


def _apply_version(snapshot: Path, data_dir: Path, outputs: dict[str, Any], baseline_meta: dict[str, Any], pointer_path: Path) -> dict[str, Any]:
    version_id = f"m2.1-{snapshot.name}"
    version_dir = data_dir / "processed" / "versions" / version_id
    if version_dir.exists():
        raise DataValidationError(f"version directory already exists; refusing overwrite: {version_dir}")
    version_dir.mkdir(parents=True, exist_ok=False)
    version_canonical, version_quality = version_dir / "epl_canonical.csv", version_dir / "epl_quality_report.json"
    version_diff = version_dir / "epl_sync_diff_report.json"
    shutil.copyfile(outputs["candidate_path"], version_canonical)
    shutil.copyfile(outputs["quality_path"], version_quality)
    shutil.copyfile(outputs["diff_path"], version_diff)
    stage_manifest = _read_json(outputs["processed_path"])
    version_manifest = {**stage_manifest, "dataset_version": version_id, "output": {**stage_manifest["output"], "filename": str(version_canonical), "sha256": file_sha256(version_canonical)}, "quality_report": str(version_quality), "quality_report_sha256": file_sha256(version_quality), "diff_report": str(version_diff), "diff_report_sha256": file_sha256(version_diff), "baseline": {"version": baseline_meta["version"], "canonical_sha256": baseline_meta["canonical_sha256"]}}
    version_manifest_path = version_dir / "epl_processed_manifest.json"
    write_json(version_manifest, version_manifest_path)
    pointer_path.parent.mkdir(parents=True, exist_ok=True)
    pointer = {"schema_version": CURRENT_POINTER_VERSION, "dataset_version": version_id, "updated_at": _utc_now(), "canonical": str(version_canonical), "processed_manifest": str(version_manifest_path), "quality_report": str(version_quality), "diff_report": str(version_diff), "canonical_sha256": file_sha256(version_canonical)}
    _atomic_write_json(pointer, pointer_path)
    return {"version_id": version_id, "version_dir": str(version_dir), "version_manifest": str(version_manifest_path), "pointer_path": str(pointer_path), "pointer": pointer}


def sync_epl(
    seasons: Iterable[str] | None = None,
    data_dir: str | Path = Path(r"D:\Codex\matchquant_data"),
    timeout: float = 30.0,
    apply: bool = False,
    in_progress_season: str | None = None,
    staging_dir: str | Path | None = None,
    baseline_path: str | Path | None = None,
    processed_manifest_path: str | Path | None = None,
    current_pointer_path: str | Path | None = None,
) -> dict[str, Any]:
    """Download or consume one immutable staging snapshot and compare it offline."""
    root = Path(data_dir).expanduser().resolve()
    if staging_dir is None:
        selected = _season_list(seasons, ())
        declared_in_progress = normalize_season(in_progress_season) if in_progress_season is not None else None
    else:
        snapshot = Path(staging_dir).expanduser().resolve()
        if not snapshot.is_dir():
            raise FileNotFoundError(f"sync staging directory does not exist: {snapshot}")
        manifest = _read_json(snapshot / "raw" / "source_manifest.json")
        selected = _season_list(seasons if seasons is not None else manifest.get("requested_seasons"), ())
        declared_in_progress = normalize_season(in_progress_season) if in_progress_season is not None else None
    if declared_in_progress is not None and declared_in_progress not in selected:
        raise ValueError(f"in-progress season {declared_in_progress} must be one of requested seasons: {', '.join(selected)}")
    if staging_dir is None:
        snapshot, _ = _download_snapshot(selected, root, timeout, declared_in_progress)
    baseline, baseline_meta = _load_baseline(
        root,
        Path(baseline_path).expanduser().resolve() if baseline_path else None,
        Path(processed_manifest_path).expanduser().resolve() if processed_manifest_path else None,
        Path(current_pointer_path).expanduser().resolve() if current_pointer_path else None,
    )
    outputs = _write_stage_outputs(snapshot, baseline, baseline_meta, selected, apply, declared_in_progress)
    result: dict[str, Any] = {"status": "dry-run", "dry_run": not apply, "apply_requested": bool(apply), "snapshot_id": snapshot.name, "staging_dir": str(snapshot), "requested_seasons": list(selected), "in_progress_season": declared_in_progress, "declared_in_progress_season": declared_in_progress, "strict_380_gate_applied_by_season": outputs["quality"]["strict_380_gate_applied_by_season"], "baseline": baseline_meta, "candidate_path": str(outputs["candidate_path"]), "candidate_sha256": outputs["candidate_hash"], "quality_report_path": str(outputs["quality_path"]), "quality_report_sha256": outputs["quality_hash"], "diff_report_path": str(outputs["diff_path"]), "diff_report_sha256": outputs["diff_hash"], "processed_manifest_path": str(outputs["processed_path"]), "skipped_unfinished": outputs["quality"]["totals"]["skipped_unfinished_count"], "diff": outputs["diff"]["counts"]}
    if apply and outputs["diff"]["apply"]["blockers"]:
        result["status"] = "blocked"
        result["apply_blockers"] = outputs["diff"]["apply"]["blockers"]
        return result
    if apply:
        pointer_path = Path(current_pointer_path).expanduser().resolve() if current_pointer_path else root / "processed" / CURRENT_POINTER_FILENAME
        result.update(status="applied", **_apply_version(snapshot, root, outputs, baseline_meta, pointer_path))
    return result


__all__ = ["sync_epl", "compare_canonical", "audit_sync_source_frame", "convert_sync_source_frame"]
