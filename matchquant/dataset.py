"""Layer 2: Manifest-bound dataset preparation, versioning, and readiness checks."""

from __future__ import annotations

import json
import os
import tempfile
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import pandas as pd

from .epl import (
    CANONICAL_COLUMNS,
    convert_source_frame,
    normalize_season,
    read_source_csv,
)
from .io import file_sha256, write_json
from .profile import UserProfile, load_profile
from .rich_data import RICH_CANONICAL_COLUMNS, convert_raw_epl_season
from .schema import DataValidationError, validate_matches

DATASET_MANIFEST_VERSION = "m2.1-dataset-manifest-v1"
CURRENT_POINTER_VERSION = "m2.1-current-pointer-v1"


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _is_pending_row(row: pd.Series) -> bool:
    """Return True if row has valid teams and date but empty full-time scores."""
    date_val = str(row.get("Date", "")).strip()
    home_val = str(row.get("HomeTeam", "")).strip()
    away_val = str(row.get("AwayTeam", "")).strip()
    fthg_val = str(row.get("FTHG", "")).strip()
    ftag_val = str(row.get("FTAG", "")).strip()

    if not date_val or not home_val or not away_val:
        return False
    # If both scores are blank or NA
    return fthg_val == "" and ftag_val == ""


def filter_completed_rows(
    df: pd.DataFrame, season: str, is_in_progress: bool = False
) -> tuple[pd.DataFrame, int, list[str]]:
    """Separate completed rows from pending rows for an in-progress season."""
    if not is_in_progress:
        return df, 0, []

    pending_mask = df.apply(_is_pending_row, axis=1)
    pending_count = int(pending_mask.sum())

    # Check for partial scores (one empty, one filled)
    fthg_str = df["FTHG"].astype("string").fillna("").str.strip()
    ftag_str = df["FTAG"].astype("string").fillna("").str.strip()
    partial_score_mask = (fthg_str.eq("") & ftag_str.ne("")) | (fthg_str.ne("") & ftag_str.eq(""))
    errors: list[str] = []
    if partial_score_mask.any():
        bad_idx = df.index[partial_score_mask].tolist()[:5]
        errors.append(
            f"in-progress season {season} has rows with partial full-time scores at index {bad_idx}"
        )

    completed_df = df.loc[~pending_mask].copy().reset_index(drop=True)
    return completed_df, pending_count, errors


def build_dataset_from_manifest(
    season_snapshots: Mapping[str, Path | str],
    data_root: str | Path,
    output_canonical: str | Path | None = None,
    output_rich: str | Path | None = None,
    in_progress_season: str | None = None,
    source_checked_at: str | None = None,
    dataset_version: str | None = None,
    quality_report_path: str | Path | None = None,
    diff_report_path: str | Path | None = None,
) -> dict[str, Any]:
    """Build canonical and rich datasets bound strictly to explicitly selected season snapshots.

    Does not discover files from directory globbing.
    """
    root = Path(data_root)
    processed_dir = root / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)

    can_path = Path(output_canonical) if output_canonical else processed_dir / "epl_canonical.csv"
    rich_path = Path(output_rich) if output_rich else processed_dir / "epl_rich_canonical.csv"
    manifest_path = processed_dir / "epl_processed_manifest.json"
    current_pointer_path = processed_dir / "epl_current.json"

    sorted_seasons = sorted(season_snapshots.keys(), key=lambda s: normalize_season(s))
    if not sorted_seasons:
        raise DataValidationError("cannot build dataset from empty season snapshots")

    canonical_frames: list[pd.DataFrame] = []
    rich_frames: list[pd.DataFrame] = []
    snapshots_meta: dict[str, Any] = {}

    for season in sorted_seasons:
        norm_season = normalize_season(season)
        snap_path = Path(season_snapshots[season])
        if not snap_path.is_file():
            raise FileNotFoundError(f"season {norm_season} snapshot not found: {snap_path}")

        raw_df = read_source_csv(snap_path)
        raw_hash = file_sha256(snap_path)
        is_in_prog = in_progress_season is not None and norm_season == normalize_season(in_progress_season)

        completed_df, pending_count, partial_errors = filter_completed_rows(
            raw_df, norm_season, is_in_progress=is_in_prog
        )
        if partial_errors:
            raise DataValidationError("; ".join(partial_errors))

        # Check complete season gate (380 completed fixtures) if not in-progress
        if not is_in_prog:
            if len(completed_df) != 380:
                raise DataValidationError(
                    f"completed season {norm_season} has {len(completed_df)} rows, expected exactly 380"
                )

        # Convert canonical
        can_season, _ = convert_source_frame(completed_df, norm_season)
        canonical_frames.append(can_season)

        # Convert rich
        rich_season = convert_raw_epl_season(completed_df, norm_season)
        rich_frames.append(rich_season)

        snapshots_meta[norm_season] = {
            "season": norm_season,
            "raw_path": str(snap_path),
            "raw_sha256": raw_hash,
            "total_rows": int(len(raw_df)),
            "completed_rows": int(len(completed_df)),
            "pending_rows": pending_count,
            "is_in_progress": is_in_prog,
            "strict_380_gate_applied": not is_in_prog,
            "latest_date": can_season["date"].max() if not can_season.empty else None,
        }

    combined_can = pd.concat(canonical_frames, ignore_index=True)
    combined_can = validate_matches(combined_can)
    combined_can = combined_can.sort_values(["date", "match_id"], kind="mergesort").loc[:, CANONICAL_COLUMNS].reset_index(drop=True)
    can_path.parent.mkdir(parents=True, exist_ok=True)
    combined_can.to_csv(can_path, index=False, encoding="utf-8")
    can_hash = file_sha256(can_path)

    combined_rich = pd.concat(rich_frames, ignore_index=True)
    combined_rich = combined_rich.sort_values(["date", "kickoff_time", "match_id"], kind="mergesort").reset_index(
        drop=True
    )
    rich_path.parent.mkdir(parents=True, exist_ok=True)
    combined_rich.to_csv(rich_path, index=False, encoding="utf-8")
    rich_hash = file_sha256(rich_path)

    overall_cutoff = combined_can["date"].max() if not combined_can.empty else None
    version_id = dataset_version or f"m2.1-epl_{_utc_now().replace(':', '').replace('-', '')[:15]}"

    manifest = {
        "schema_version": DATASET_MANIFEST_VERSION,
        "dataset_version": version_id,
        "generated_at": _utc_now(),
        "source_checked_at": source_checked_at or _utc_now(),
        "seasons": sorted_seasons,
        "in_progress_season": in_progress_season,
        "data_cutoff": overall_cutoff,
        "canonical": {
            "path": str(can_path),
            "sha256": can_hash,
            "rows": int(len(combined_can)),
            "columns": list(CANONICAL_COLUMNS),
        },
        "rich": {
            "path": str(rich_path),
            "sha256": rich_hash,
            "rows": int(len(combined_rich)),
            "columns": list(RICH_CANONICAL_COLUMNS),
        },
        "season_snapshots": snapshots_meta,
    }
    write_json(manifest, manifest_path)

    # Atomically write current pointer epl_current.json compatible with M2.1 contract
    pointer = {
        "schema_version": CURRENT_POINTER_VERSION,
        "dataset_version": version_id,
        "updated_at": _utc_now(),
        "canonical": str(can_path),
        "canonical_sha256": can_hash,
        "canonical_rows": int(len(combined_can)),
        "rich": str(rich_path),
        "rich_sha256": rich_hash,
        "rich_rows": int(len(combined_rich)),
        "processed_manifest": str(manifest_path),
        "quality_report": str(quality_report_path) if quality_report_path else None,
        "diff_report": str(diff_report_path) if diff_report_path else None,
        "data_cutoff": overall_cutoff,
        "seasons": sorted_seasons,
        "in_progress_season": in_progress_season,
    }
    _atomic_write_json(pointer, current_pointer_path)

    return manifest


def _atomic_write_json(payload: Mapping[str, Any], target_path: Path) -> None:
    """Atomically write JSON to target path via temporary file and replace."""
    target_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=target_path.parent,
        prefix="tmp_cur_",
        suffix=".json",
        delete=False,
    ) as tmp:
        tmp_name = tmp.name
        json.dump(payload, tmp, indent=2, ensure_ascii=False)
    os.replace(tmp_name, target_path)


def load_current_dataset_pointer(data_root: str | Path) -> dict[str, Any] | None:
    """Load the current active dataset pointer if present."""
    pointer_path = Path(data_root) / "processed" / "epl_current.json"
    if not pointer_path.is_file():
        return None
    try:
        return json.loads(pointer_path.read_text(encoding="utf-8"))
    except Exception:
        return None


def check_data_readiness(
    data_root: str | Path,
    as_of: str | date | None = None,
    fixture_date: str | date | None = None,
    profile: UserProfile | None = None,
    allow_stale_fallback: bool | None = None,
) -> dict[str, Any]:
    """Audit local data coverage, freshness, and readiness before single-match analysis."""
    prof = profile or load_profile()
    effective_data_root = Path(data_root if data_root else prof.data_root)
    pointer = load_current_dataset_pointer(effective_data_root)

    warnings: list[str] = []
    errors: list[str] = []

    if pointer is None:
        return {
            "ready": False,
            "status": "not_ready",
            "reason": "no_dataset",
            "message": f"no active dataset found in '{effective_data_root / 'processed'}'; sync required",
            "dataset_version": None,
            "fetched_at": None,
            "source_checked_at": None,
            "latest_completed_match_date": None,
            "used_data_cutoff": None,
            "seasons": [],
            "in_progress_season": None,
            "completed_matches": 0,
            "data_age_days": None,
            "is_off_season": False,
            "warnings": ["active dataset pointer does not exist"],
            "errors": ["missing epl_current.json"],
            "rich_path": None,
            "canonical_path": None,
        }

    dataset_version = pointer.get("dataset_version")
    generated_at = pointer.get("updated_at") or pointer.get("generated_at")
    source_checked_at = pointer.get("source_checked_at") or generated_at

    # Canonical path & row extraction (supports string or dict pointer)
    canonical_val = pointer.get("canonical")
    if isinstance(canonical_val, dict):
        canonical_path = Path(canonical_val.get("path", ""))
        completed_matches = int(canonical_val.get("rows", 0))
    elif isinstance(canonical_val, str):
        canonical_path = Path(canonical_val)
        completed_matches = int(pointer.get("canonical_rows", 0))
    else:
        canonical_path = effective_data_root / "processed" / "epl_canonical.csv"
        completed_matches = 0

    # Rich path extraction
    rich_val = pointer.get("rich")
    if isinstance(rich_val, dict):
        rich_path = Path(rich_val.get("path", ""))
    elif isinstance(rich_val, str):
        rich_path = Path(rich_val)
    else:
        # Fallback to standard location
        candidate = effective_data_root / "processed" / "epl_rich_canonical.csv"
        if candidate.is_file():
            rich_path = candidate
        elif canonical_path.parent.is_dir() and (canonical_path.parent / "epl_rich_canonical.csv").is_file():
            rich_path = canonical_path.parent / "epl_rich_canonical.csv"
        else:
            rich_path = candidate

    # Load cutoff and seasons from manifest if not directly in pointer
    latest_completed_match = pointer.get("data_cutoff")
    seasons = pointer.get("seasons", [])
    in_prog = pointer.get("in_progress_season")

    proc_manifest_path = pointer.get("processed_manifest")
    if proc_manifest_path and Path(proc_manifest_path).is_file():
        try:
            m_data = json.loads(Path(proc_manifest_path).read_text(encoding="utf-8"))
            if not latest_completed_match:
                latest_completed_match = m_data.get("data_cutoff")
            if not seasons:
                seasons = m_data.get("seasons", [])
            if not in_prog:
                in_prog = m_data.get("in_progress_season")
            if completed_matches == 0:
                out = m_data.get("output", {})
                if isinstance(out, dict):
                    completed_matches = out.get("rows", 0)
        except Exception:
            pass

    if not rich_path.is_file():
        errors.append(f"rich dataset file does not exist: {rich_path}")
    if not canonical_path.is_file():
        errors.append(f"canonical dataset file does not exist: {canonical_path}")

    # Determine cutoff and data age
    as_of_date: date | None = None
    if as_of:
        if isinstance(as_of, str):
            try:
                as_of_date = datetime.strptime(as_of.strip(), "%Y-%m-%d").date()
            except ValueError:
                errors.append(f"invalid as_of date format: {as_of}")
        elif isinstance(as_of, date):
            as_of_date = as_of

    latest_date_obj: date | None = None
    if latest_completed_match:
        try:
            latest_date_obj = datetime.strptime(latest_completed_match, "%Y-%m-%d").date()
        except ValueError:
            pass

    ref_date = as_of_date or date.today()
    data_age_days: int | None = None
    if latest_date_obj is not None:
        data_age_days = (ref_date - latest_date_obj).days

    # Off-season check: June or July, or data_age_days > 45 in summer months
    is_off_season = False
    if ref_date.month in (6, 7) or (ref_date.month == 8 and ref_date.day < 15):
        is_off_season = True
        warnings.append("currently in EPL off-season window; absence of recent matches is expected")

    # Cutoff for used data: matches strictly before as_of
    used_data_cutoff = latest_completed_match
    if as_of_date and latest_date_obj:
        if as_of_date <= latest_date_obj:
            used_data_cutoff = f"< {as_of_date.isoformat()}"
        else:
            used_data_cutoff = latest_completed_match

    # Check stale condition
    stale_threshold_days = 30
    is_stale = False
    if data_age_days is not None and data_age_days > stale_threshold_days and not is_off_season:
        is_stale = True
        warnings.append(
            f"latest completed match is {data_age_days} days before cutoff (exceeds {stale_threshold_days} days threshold)"
        )

    # Permit stale fallback if explicitly authorized
    effective_allow_stale = (
        allow_stale_fallback if allow_stale_fallback is not None else prof.refresh_policy.allow_stale_fallback
    )

    ready = (len(errors) == 0)
    status = "ok"
    if errors:
        status = "failed"
        ready = False
    elif is_stale:
        if effective_allow_stale:
            status = "needs_review"
            ready = True
        else:
            status = "stale_review_required"
            ready = False

    return {
        "ready": ready,
        "status": status,
        "dataset_version": dataset_version,
        "fetched_at": generated_at,
        "source_checked_at": source_checked_at,
        "latest_completed_match_date": latest_completed_match,
        "used_data_cutoff": used_data_cutoff,
        "seasons": seasons,
        "in_progress_season": in_prog,
        "completed_matches": completed_matches,
        "data_age_days": data_age_days,
        "is_off_season": is_off_season,
        "is_stale": is_stale,
        "stale_fallback_used": (is_stale and effective_allow_stale),
        "warnings": warnings,
        "errors": errors,
        "rich_path": str(rich_path),
        "canonical_path": str(canonical_path),
    }
