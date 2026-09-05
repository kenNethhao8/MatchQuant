"""Immutable local snapshot persistence for the M5 E0 fixture adapter."""

from __future__ import annotations

import hashlib
import os
import re
import stat
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from .epl import normalize_season, read_source_csv
from .epl_fixtures import (
    FIXTURE_ADAPTER_VERSION,
    FIXTURE_DIFF_VERSION,
    FIXTURE_MANIFEST_VERSION,
    _analyse,
    _report_template,
    compare_fixtures,
)
from .fixtures import FIXTURE_COLUMNS, FIXTURE_SCHEMA_VERSION, load_fixtures, validate_fixtures
from .io import file_sha256, write_json
from .schema import DataValidationError


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _new_snapshot(root: Path) -> Path:
    snapshot_root = root / "fixtures" / "snapshots"
    snapshot_root.mkdir(parents=True, exist_ok=True)
    while True:
        snapshot_id = (
            f"epl_fixtures_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}_"
            f"{uuid.uuid4().hex[:8]}"
        )
        path = snapshot_root / snapshot_id
        try:
            path.mkdir()
            return path
        except FileExistsError:
            continue


def _safe_filename(source_path: Path) -> str:
    name = re.sub(r"[^A-Za-z0-9._-]+", "_", source_path.name).strip("._")
    return name or "source.csv"


def _read_only(path: Path) -> None:
    try:
        os.chmod(path, stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
    except OSError:
        pass


def _write_csv_atomic(frame: pd.DataFrame, path: Path) -> None:
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        frame.to_csv(temporary, index=False, encoding="utf-8")
        os.replace(temporary, path)
    except Exception:
        if temporary.exists():
            temporary.unlink()
        raise


def _manifest(
    report: dict[str, Any], snapshot: Path, source_path: Path, raw_path: Path | None,
    source_hash: str | None, created_at: str, baseline_path: Path | None,
    baseline_hash: str | None, output: dict[str, Any] | None, quality_path: Path,
    quality_hash: str, diff_path: Path | None,
) -> dict[str, Any]:
    diff_hash = file_sha256(diff_path) if diff_path and diff_path.is_file() else None
    return {
        "schema_version": FIXTURE_MANIFEST_VERSION,
        "adapter_version": FIXTURE_ADAPTER_VERSION,
        "fixture_schema_version": FIXTURE_SCHEMA_VERSION,
        "snapshot_id": snapshot.name,
        "snapshot_dir": str(snapshot),
        "created_at_utc": created_at,
        "source_path": str(source_path),
        "source_raw_snapshot": str(raw_path) if raw_path else None,
        "source_filename": raw_path.name if raw_path else None,
        "source_sha256": source_hash,
        "source_url": None,
        "source_label": None,
        "season": report.get("season"),
        "counts": report.get("counts", {}),
        "date_range": report.get("source_date_range", report.get("date_range")),
        "source_date_range": report.get("source_date_range"),
        "completed_date_range": report.get("completed_date_range"),
        "pending_date_range": report.get("pending_date_range"),
        "completed_data_cutoff": report.get("completed_data_cutoff"),
        "completed_data_cutoff_status": report.get("completed_data_cutoff_status"),
        "data_cutoff": report.get("data_cutoff"),
        "data_cutoff_status": report.get("data_cutoff_status"),
        "status": report.get("status"),
        "quality_report": str(quality_path),
        "quality_report_sha256": quality_hash,
        "baseline_fixtures": (
            {"path": str(baseline_path), "sha256": baseline_hash} if baseline_path else None
        ),
        "diff_report": str(diff_path) if diff_path else None,
        "diff_report_sha256": diff_hash,
        "output": output,
        "canonical_output_sha256": output.get("sha256") if output else None,
        "canonical_output_row_count": output.get("rows") if output else None,
        "limitations": report.get("limitations", []),
    }


def prepare_epl_fixtures(
    source_csv: str | Path,
    season: str,
    data_root: str | Path = Path(r"D:\Codex\matchquant_data"),
    *,
    source_url: str | None = None,
    source_label: str | None = None,
    baseline_fixtures: str | Path | None = None,
) -> dict[str, Any]:
    """Create one immutable local fixture snapshot without changing active data."""

    root = Path(data_root).expanduser().resolve()
    snapshot = _new_snapshot(root)
    created_at = _utc_now()
    source_path = Path(source_csv).expanduser().resolve()
    raw_path: Path | None = None
    source_hash: str | None = None
    baseline_path = Path(baseline_fixtures).expanduser().resolve() if baseline_fixtures else None
    baseline_hash: str | None = None
    quality_path = snapshot / "reports" / "fixture_quality_report.json"
    manifest_path = snapshot / "fixture_snapshot_manifest.json"
    diff_path: Path | None = None
    output_path: Path | None = None
    output: dict[str, Any] | None = None
    report = _initial_report(season, snapshot, source_path, created_at, baseline_path)
    try:
        normalized = normalize_season(season)
        report["season"] = normalized
        if source_url is not None and not str(source_url).strip():
            raise ValueError("source_url cannot be empty when provided")
        if source_label is not None and not str(source_label).strip():
            raise ValueError("source_label cannot be empty when provided")
        if not source_path.is_file():
            raise FileNotFoundError(f"source CSV does not exist: {source_path}")
        body = source_path.read_bytes()
        raw_path = snapshot / "raw" / _safe_filename(source_path)
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        raw_path.write_bytes(body)
        source_hash = hashlib.sha256(body).hexdigest()
        source = read_source_csv(raw_path)
        report, pending = _analyse(source, normalized)
        report.update(
            snapshot_id=snapshot.name,
            source_path=str(source_path),
            created_at_utc=created_at,
            baseline_fixtures=str(baseline_path) if baseline_path else None,
        )

        baseline_frame = None
        if baseline_path is not None:
            if not baseline_path.is_file():
                raise FileNotFoundError(f"baseline fixture CSV does not exist: {baseline_path}")
            baseline_hash = file_sha256(baseline_path)
            baseline_frame = load_fixtures(baseline_path)
        if report["status"] == "failed":
            raise DataValidationError(f"fixture source quality failed: {'; '.join(report['errors'])}")

        if report["status"] == "ok":
            candidate = validate_fixtures(pd.DataFrame(pending, columns=FIXTURE_COLUMNS))
            candidate = candidate.sort_values(["_timestamp", "fixture_id"], kind="mergesort")
            candidate = candidate.loc[:, FIXTURE_COLUMNS].reset_index(drop=True)
            output_path = snapshot / "processed" / "epl_fixtures.csv"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            _write_csv_atomic(candidate, output_path)
            output = {
                "path": str(output_path),
                "sha256": file_sha256(output_path),
                "rows": len(candidate),
                "columns": list(FIXTURE_COLUMNS),
            }
            report["output"] = output
            if baseline_frame is not None:
                diff_path = snapshot / "reports" / "fixture_diff_report.json"
                diff = compare_fixtures(baseline_frame, candidate)
                diff.update(
                    snapshot_id=snapshot.name,
                    baseline={"path": str(baseline_path), "sha256": baseline_hash, "rows": len(baseline_frame)},
                    candidate=output,
                    generated_at_utc=_utc_now(),
                )
                write_json(diff, diff_path)
        elif baseline_frame is not None:
            diff_path = snapshot / "reports" / "fixture_diff_report.json"
            write_json(
                {
                    "schema_version": FIXTURE_DIFF_VERSION,
                    "status": "not_run",
                    "reason": "no_pending_fixtures",
                    "snapshot_id": snapshot.name,
                    "baseline": {"path": str(baseline_path), "sha256": baseline_hash, "rows": len(baseline_frame)},
                    "candidate": None,
                    "manual_judgment_required": True,
                    "note": "No candidate fixture CSV exists; baseline missing rows were not classified.",
                },
                diff_path,
            )
        report["status"] = "no_pending_fixtures" if not report["pending_rows"] else "ok"
    except (DataValidationError, FileNotFoundError, OSError, TypeError, ValueError, pd.errors.ParserError) as exc:
        report["status"] = "failed"
        report.setdefault("errors", []).append(str(exc))
        if output_path and output_path.exists():
            output_path.unlink()
        output = None

    report.update(
        source_url=str(source_url) if source_url is not None else None,
        source_label=str(source_label) if source_label is not None else None,
        source_sha256=source_hash,
        source_raw_snapshot=str(raw_path) if raw_path else None,
        quality_report=str(quality_path),
        diff_report=str(diff_path) if diff_path else None,
        output=output,
    )
    quality_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(report, quality_path)
    quality_hash = file_sha256(quality_path)
    manifest = _manifest(
        report, snapshot, source_path, raw_path, source_hash, created_at,
        baseline_path, baseline_hash, output, quality_path, quality_hash, diff_path,
    )
    manifest["source_url"] = str(source_url) if source_url is not None else None
    manifest["source_label"] = str(source_label) if source_label is not None else None
    write_json(manifest, manifest_path)
    if raw_path and raw_path.is_file():
        _read_only(raw_path)
    if report["status"] == "failed":
        first_error = report.get("errors", ["unknown fixture snapshot error"])[0]
        raise DataValidationError(
            f"M5 fixture snapshot failed: {first_error}; manifest: {manifest_path}; quality report: {quality_path}"
        )
    return {
        "status": report["status"],
        "snapshot_id": snapshot.name,
        "snapshot_dir": str(snapshot),
        "manifest_path": str(manifest_path),
        "quality_report_path": str(quality_path),
        "diff_report_path": str(diff_path) if diff_path else None,
        "fixture_path": output["path"] if output else None,
        "source_sha256": source_hash,
        "counts": report["counts"],
        "manifest": manifest,
        "quality": report,
    }


def _initial_report(
    season: str, snapshot: Path, source_path: Path, created_at: str, baseline_path: Path | None
) -> dict[str, Any]:
    report = _report_template(str(season).strip() or None)
    report.update(
        snapshot_id=snapshot.name,
        source_path=str(source_path),
        created_at_utc=created_at,
        baseline_fixtures=str(baseline_path) if baseline_path else None,
        diff_report=None,
        output=None,
    )
    return report


__all__ = ["prepare_epl_fixtures"]
