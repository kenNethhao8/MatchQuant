"""Layer 1: Explicit on-demand source synchronization with staging, diffing, and addition-only gates."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import pandas as pd

from .dataset import build_dataset_from_manifest, filter_completed_rows, load_current_dataset_pointer
from .epl import (
    DEFAULT_EPL_SEASONS,
    audit_source_frame,
    normalize_season,
    read_source_csv,
    source_url,
    stable_match_id,
)
from .io import file_sha256, write_json
from .profile import UserProfile, load_profile
from .schema import DataValidationError

SYNC_DIFF_VERSION = "m2.1-sync-diff-v1"
QUALITY_REPORT_VERSION = "m2.1-quality-report-v1"


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _timestamp_slug() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


@dataclass
class SeasonDiff:
    season: str
    is_in_progress: bool
    strict_380_gate_applied: bool
    source_url: str
    raw_sha256: str | None = None
    http_status: int | None = None
    total_rows: int = 0
    completed_rows: int = 0
    pending_rows: int = 0
    additions_count: int = 0
    revisions_count: int = 0
    missing_baseline_count: int = 0
    unchanged_count: int = 0
    additions: list[str] = field(default_factory=list)
    revisions: list[dict[str, Any]] = field(default_factory=list)
    missing_baseline: list[str] = field(default_factory=list)
    validation_errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "season": self.season,
            "is_in_progress": self.is_in_progress,
            "strict_380_gate_applied": self.strict_380_gate_applied,
            "source_url": self.source_url,
            "raw_sha256": self.raw_sha256,
            "http_status": self.http_status,
            "total_rows": self.total_rows,
            "completed_rows": self.completed_rows,
            "pending_rows": self.pending_rows,
            "additions_count": self.additions_count,
            "revisions_count": self.revisions_count,
            "missing_baseline_count": self.missing_baseline_count,
            "unchanged_count": self.unchanged_count,
            "additions": self.additions[:20],
            "revisions": self.revisions[:20],
            "missing_baseline": self.missing_baseline[:20],
            "validation_errors": self.validation_errors,
        }


def _compare_season_records(
    baseline_df: pd.DataFrame | None,
    new_df: pd.DataFrame,
    season: str,
) -> tuple[list[str], list[dict[str, Any]], list[str], int]:
    """Compare baseline records with newly fetched records using stable_match_id."""
    additions: list[str] = []
    revisions: list[dict[str, Any]] = []
    missing_baseline: list[str] = []
    unchanged_count = 0

    new_dict: dict[str, pd.Series] = {}
    for _, row in new_df.iterrows():
        mid = stable_match_id(season, str(row["HomeTeam"]), str(row["AwayTeam"]))
        new_dict[mid] = row

    if baseline_df is None or baseline_df.empty:
        additions = list(new_dict.keys())
        return additions, revisions, missing_baseline, unchanged_count

    baseline_dict: dict[str, pd.Series] = {}
    for _, row in baseline_df.iterrows():
        mid = stable_match_id(season, str(row["home_team"]), str(row["away_team"]))
        baseline_dict[mid] = row

    # Check additions and revisions
    for mid, new_row in new_dict.items():
        if mid not in baseline_dict:
            additions.append(mid)
        else:
            base_row = baseline_dict[mid]
            # Compare key attributes: date, scores
            diff_fields: list[str] = []
            base_date = str(base_row.get("date", "")).strip()
            # New date format DD/MM/YYYY or similar
            try:
                new_date = pd.to_datetime(str(new_row.get("Date", "")).strip(), dayfirst=True).strftime("%Y-%m-%d")
            except Exception:
                new_date = str(new_row.get("Date", "")).strip()

            if base_date and new_date and base_date != new_date:
                diff_fields.append("date")

            try:
                base_hg = int(base_row.get("home_goals", -1))
                new_hg = int(pd.to_numeric(new_row.get("FTHG"), errors="coerce"))
                if base_hg != new_hg:
                    diff_fields.append("home_goals")
            except Exception:
                pass

            try:
                base_ag = int(base_row.get("away_goals", -1))
                new_ag = int(pd.to_numeric(new_row.get("FTAG"), errors="coerce"))
                if base_ag != new_ag:
                    diff_fields.append("away_goals")
            except Exception:
                pass

            if diff_fields:
                revisions.append({
                    "match_id": mid,
                    "diff_fields": diff_fields,
                    "baseline": {k: str(base_row.get(k)) for k in ("date", "home_team", "away_team", "home_goals", "away_goals")},
                    "new": {k: str(new_row.get(k)) for k in ("Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG")},
                })
            else:
                unchanged_count += 1

    # Check missing baseline
    for mid in baseline_dict:
        if mid not in new_dict:
            missing_baseline.append(mid)

    return additions, revisions, missing_baseline, unchanged_count


def sync_epl(
    data_root: str | Path,
    seasons: Sequence[str] | None = None,
    in_progress_season: str | None = None,
    dry_run: bool = True,
    apply: bool = False,
    timeout: float = 30.0,
    custom_fetcher: Callable[[str, float], bytes] | None = None,
    profile: UserProfile | None = None,
) -> dict[str, Any]:
    """Execute on-demand source synchronization.

    Defaults to dry-run. Keeps snapshots in timestamped staging.
    Enforces addition-only promotion rule.
    """
    root = Path(data_root)
    staging_slug = f"epl_sync_{_timestamp_slug()}_{os.urandom(6).hex()}"
    staging_dir = root / "staging" / staging_slug / "raw"
    staging_dir.mkdir(parents=True, exist_ok=True)

    norm_in_prog = normalize_season(in_progress_season) if in_progress_season else None

    # Determine seasons to cover
    if seasons is not None:
        target_seasons = sorted([normalize_season(s) for s in seasons])
    else:
        prof = profile or load_profile()
        start_year = int(normalize_season(prof.historical_start_season)[:4])
        # Up to in_progress_season or default end
        end_year = int(norm_in_prog[:4]) if norm_in_prog else 2025
        target_seasons = [f"{y}-{(y + 1) % 100:02d}" for y in range(start_year, end_year + 1)]

    # Load baseline matches for diff comparison
    pointer = load_current_dataset_pointer(root)
    baseline_df: pd.DataFrame | None = None
    if pointer:
        can_path = Path(pointer.get("canonical", root / "processed" / "epl_canonical.csv"))
        if can_path.is_file():
            try:
                baseline_df = pd.read_csv(can_path)
            except Exception:
                baseline_df = None

    seasons_downloaded: list[str] = []
    seasons_cached: list[str] = []
    season_diffs: dict[str, SeasonDiff] = {}
    season_snapshot_paths: dict[str, Path] = {}
    promotion_block_reasons: list[str] = []

    for season in target_seasons:
        is_in_prog = norm_in_prog is not None and season == norm_in_prog
        cached_raw_file = root / "raw" / f"E0_{season}.csv"

        # Check if already cached in raw/ with valid 380 matches (unless in-progress)
        is_cached = False
        if cached_raw_file.is_file() and not is_in_prog:
            try:
                raw_check = read_source_csv(cached_raw_file)
                audit = audit_source_frame(raw_check, season)
                if audit["status"] != "failed" and audit["match_count"] == 380:
                    is_cached = True
            except Exception:
                is_cached = False

        if is_cached:
            seasons_cached.append(season)
            season_snapshot_paths[season] = cached_raw_file
            season_diffs[season] = SeasonDiff(
                season=season,
                is_in_progress=False,
                strict_380_gate_applied=True,
                source_url=source_url(season),
                raw_sha256=file_sha256(cached_raw_file),
                http_status=304,  # Not modified / local cache
                total_rows=380,
                completed_rows=380,
                pending_rows=0,
                unchanged_count=380,
            )
            continue

        # Need to download this season into staging
        seasons_downloaded.append(season)
        staging_raw_file = staging_dir / f"E0_{season}.csv"
        url = source_url(season)
        diff = SeasonDiff(
            season=season,
            is_in_progress=is_in_prog,
            strict_380_gate_applied=not is_in_prog,
            source_url=url,
        )

        try:
            if custom_fetcher is not None:
                body = custom_fetcher(season, timeout)
                diff.http_status = 200
            else:
                req = urllib.request.Request(url, headers={"User-Agent": "MatchQuant/0.1 EPL research sync"})
                with urllib.request.urlopen(req, timeout=timeout) as resp:
                    body = resp.read()
                    diff.http_status = int(resp.getcode())

            staging_raw_file.write_bytes(body)
            diff.raw_sha256 = file_sha256(staging_raw_file)
            season_snapshot_paths[season] = staging_raw_file

            source_df = read_source_csv(staging_raw_file)
            diff.total_rows = len(source_df)

            completed_df, pending_count, partial_errors = filter_completed_rows(
                source_df, season, is_in_progress=is_in_prog
            )
            diff.completed_rows = len(completed_df)
            diff.pending_rows = pending_count
            if partial_errors:
                diff.validation_errors.extend(partial_errors)
                promotion_block_reasons.extend(partial_errors)

            # Audit frame quality
            audit = audit_source_frame(completed_df, season)
            if not is_in_prog and audit["match_count"] != 380:
                err = f"completed season {season} has {audit['match_count']} valid completed matches, expected 380"
                diff.validation_errors.append(err)
                promotion_block_reasons.append(err)

            if audit["invalid_rows"] > 0:
                err = f"season {season} has {audit['invalid_rows']} invalid rows in completed data"
                diff.validation_errors.append(err)
                promotion_block_reasons.append(err)

            # Compare against baseline partition for this season
            season_baseline = None
            if baseline_df is not None and "season" in baseline_df.columns:
                season_baseline = baseline_df[baseline_df["season"] == season]

            additions, revisions, missing_bl, unchanged_cnt = _compare_season_records(
                season_baseline, completed_df, season
            )
            diff.additions = additions
            diff.additions_count = len(additions)
            diff.revisions = revisions
            diff.revisions_count = len(revisions)
            diff.missing_baseline = missing_bl
            diff.missing_baseline_count = len(missing_bl)
            diff.unchanged_count = unchanged_cnt

            if revisions:
                promotion_block_reasons.append(
                    f"season {season} contains {len(revisions)} revisions to existing records"
                )
            if missing_bl:
                promotion_block_reasons.append(
                    f"season {season} is missing {len(missing_bl)} records previously in baseline"
                )

        except Exception as exc:
            diff.validation_errors.append(str(exc))
            promotion_block_reasons.append(f"failed to fetch/process season {season}: {exc}")

        season_diffs[season] = diff

    # Addition-only gate evaluation
    promotion_eligible = (len(promotion_block_reasons) == 0)
    should_apply = apply and not dry_run and promotion_eligible

    diff_report = {
        "schema_version": SYNC_DIFF_VERSION,
        "generated_at": _utc_now(),
        "staging_dir": str(staging_dir),
        "requested_seasons": target_seasons,
        "in_progress_season": norm_in_prog,
        "seasons_downloaded": seasons_downloaded,
        "seasons_cached": seasons_cached,
        "promotion_eligible": promotion_eligible,
        "promotion_block_reasons": promotion_block_reasons,
        "diff_summary": {
            "total_additions": sum(d.additions_count for d in season_diffs.values()),
            "total_revisions": sum(d.revisions_count for d in season_diffs.values()),
            "total_missing_baseline": sum(d.missing_baseline_count for d in season_diffs.values()),
            "total_unchanged": sum(d.unchanged_count for d in season_diffs.values()),
            "total_skipped_pending": sum(d.pending_rows for d in season_diffs.values()),
            "total_invalid": sum(len(d.validation_errors) for d in season_diffs.values()),
        },
        "seasons": {s: diff.to_dict() for s, diff in season_diffs.items()},
    }

    diff_report_path = staging_dir.parent / "epl_sync_diff_report.json"
    write_json(diff_report, diff_report_path)

    new_dataset_meta: dict[str, Any] | None = None
    if should_apply:
        # Atomic apply: copy new raw files to data_root/raw/ if not already there
        raw_target_dir = root / "raw"
        raw_target_dir.mkdir(parents=True, exist_ok=True)
        final_snapshots: dict[str, Path] = {}

        for season, snap_path in season_snapshot_paths.items():
            dest = raw_target_dir / f"E0_{season}.csv"
            # If from staging, copy to raw (immutable snapshot kept in staging)
            if snap_path != dest:
                shutil.copy2(snap_path, dest)
            final_snapshots[season] = dest

        new_dataset_meta = build_dataset_from_manifest(
            season_snapshots=final_snapshots,
            data_root=root,
            in_progress_season=norm_in_prog,
            source_checked_at=_utc_now(),
            dataset_version=f"m2.1-epl_sync_{_timestamp_slug()}_{os.urandom(4).hex()}",
            diff_report_path=diff_report_path,
        )

    # Status classification
    if not promotion_eligible:
        status = "review_required" if any(d.revisions_count > 0 or d.missing_baseline_count > 0 for d in season_diffs.values()) else "failed"
    elif should_apply:
        status = "applied"
    else:
        status = "dry_run_ok"

    return {
        "status": status,
        "action": "apply" if should_apply else "dry_run",
        "timestamp": _utc_now(),
        "staging_dir": str(staging_dir),
        "requested_seasons": target_seasons,
        "in_progress_season": norm_in_prog,
        "seasons_downloaded": seasons_downloaded,
        "seasons_cached": seasons_cached,
        "promotion_eligible": promotion_eligible,
        "promotion_block_reasons": promotion_block_reasons,
        "diff_summary": diff_report["diff_summary"],
        "diff_report_path": str(diff_report_path),
        "applied_dataset_version": new_dataset_meta.get("dataset_version") if new_dataset_meta else None,
        "dataset_cutoff": new_dataset_meta.get("data_cutoff") if new_dataset_meta else (pointer.get("data_cutoff") if pointer else None),
        "dataset_ready": (new_dataset_meta is not None) or (pointer is not None),
    }
