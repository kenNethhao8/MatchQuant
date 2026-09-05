"""Source parsing, comparison, and explicit download pieces for M2.1."""

from __future__ import annotations

import urllib.error
import urllib.request
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from .epl import (
    CANONICAL_COLUMNS,
    SOURCE_REQUIRED_COLUMNS,
    _clean_text,
    _invalid_score,
    _parse_source_dates,
    _season_list,
    _source_date_format,
    normalize_season,
    raw_filename,
    read_source_csv,
    source_url,
    stable_match_id,
)
from .io import file_sha256, write_json
from .schema import DataValidationError, validate_matches

SYNC_ADAPTER_VERSION = "m2.1-epl-football-data-v1"
SYNC_SOURCE_MANIFEST_VERSION = "m2.1-source-manifest-v1"
SYNC_QUALITY_REPORT_VERSION = "m2.1-quality-report-v1"
SYNC_DIFF_REPORT_VERSION = "m2.1-diff-report-v1"
SYNC_PROCESSED_MANIFEST_VERSION = "m2.1-processed-manifest-v1"
CURRENT_POINTER_VERSION = "m2.1-current-pointer-v1"
CURRENT_POINTER_FILENAME = "epl_current.json"


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _source_error(report: dict[str, Any]) -> str:
    errors = report.get("errors", [])
    if errors:
        first = errors[0]
        location = f" at source row {first['source_row']}" if "source_row" in first else ""
        return f"{first.get('kind', 'invalid row')}{location}"
    return "source contains invalid fields"


def _record(row: pd.Series) -> dict[str, Any]:
    values: dict[str, Any] = {}
    for column in CANONICAL_COLUMNS:
        value = row.name if column == "match_id" and column not in row.index else row[column]
        values[column] = value.item() if hasattr(value, "item") else value
    return values


def compare_canonical(baseline: pd.DataFrame, candidate: pd.DataFrame) -> dict[str, Any]:
    """Compare two validated canonical datasets by the existing stable match ID."""
    baseline = validate_matches(baseline).loc[:, CANONICAL_COLUMNS]
    candidate = validate_matches(candidate).loc[:, CANONICAL_COLUMNS]
    base = baseline.set_index("match_id")
    cand = candidate.set_index("match_id")
    base_ids, candidate_ids = set(base.index), set(cand.index)
    added_ids = sorted(candidate_ids - base_ids)
    missing_ids = sorted(base_ids - candidate_ids)
    common_ids = sorted(base_ids & candidate_ids)
    if common_ids:
        base_common, candidate_common = base.loc[common_ids], cand.loc[common_ids]
        same = pd.Series(True, index=common_ids)
        for column in CANONICAL_COLUMNS[1:]:
            same &= base_common[column].eq(candidate_common[column])
        unchanged_ids = [match_id for match_id in common_ids if bool(same.loc[match_id])]
        revised_ids = [match_id for match_id in common_ids if not bool(same.loc[match_id])]
    else:
        unchanged_ids, revised_ids = [], []
    details: dict[str, list[dict[str, Any]]] = {
        "added": [{"match_id": match_id, "record": _record(cand.loc[match_id])} for match_id in added_ids],
        "unchanged": [{"match_id": match_id, "record": _record(cand.loc[match_id])} for match_id in unchanged_ids],
        "revised": [],
        "missing_from_candidate": [{"match_id": match_id, "record": _record(base.loc[match_id])} for match_id in missing_ids],
    }
    for match_id in revised_ids:
        before, after = _record(base.loc[match_id]), _record(cand.loc[match_id])
        details["revised"].append(
            {
                "match_id": match_id,
                "changed_fields": [column for column in CANONICAL_COLUMNS[1:] if before[column] != after[column]],
                "baseline": before,
                "candidate": after,
            }
        )
    counts = {name: len(records) for name, records in details.items()}
    return {"counts": counts, "details": details, **{name: details[name] for name in details}}


def _row_error(source: pd.DataFrame, mask: pd.Series, kind: str, message: str) -> list[dict[str, Any]]:
    return [{"source_row": int(index) + 2, "kind": kind, "message": message} for index in source.index[mask]]


def _normalise_in_progress_season(value: str | None) -> str | None:
    return None if value is None else normalize_season(value)


def _season_policy(season: str, in_progress_season: str | None) -> dict[str, Any]:
    declared = _normalise_in_progress_season(in_progress_season)
    if declared is not None and declared != season:
        raise ValueError(f"in-progress season {declared} must match requested season {season}")
    is_in_progress = declared == season
    return {
        "in_progress_season": declared,
        "declared_in_progress_season": declared,
        "season_is_declared_in_progress": is_in_progress,
        "strict_380_gate_applied": not is_in_progress,
        "match_count_policy": "in_progress_exception" if is_in_progress else "strict_complete_season",
    }


def _analyse_source(source: pd.DataFrame, season: str, in_progress_season: str | None = None) -> tuple[dict[str, Any], pd.DataFrame | None]:
    normalized = normalize_season(season)
    policy = _season_policy(normalized, in_progress_season)
    source = source.reset_index(drop=True)
    report: dict[str, Any] = {
        "season": normalized,
        "source_rows": int(len(source)),
        "match_count": 0,
        "completed_match_count": 0,
        "skipped_unfinished_count": 0,
        "skipped_unfinished": [],
        "team_count": 0,
        "earliest_date": None,
        "latest_date": None,
        "source_date_format": None,
        "duplicate_rows": 0,
        "duplicate_fixture_rows": 0,
        "invalid_rows": 0,
        "invalid_date_rows": 0,
        "invalid_score_rows": 0,
        "invalid_team_rows": 0,
        "expected_matches": 380,
        "meets_expected_matches": False,
        **policy,
        "errors": [],
        "warnings": [],
        "status": "failed",
    }
    missing = [column for column in SOURCE_REQUIRED_COLUMNS if column not in source.columns]
    if missing:
        report["invalid_rows"] = len(source)
        report["errors"] = [{"kind": "missing_columns", "message": f"missing required source columns: {', '.join(missing)}"}]
        return report, None
    dates = _parse_source_dates(source["Date"])
    home, away = _clean_text(source["HomeTeam"]), _clean_text(source["AwayTeam"])
    home_score, away_score = _clean_text(source["FTHG"]), _clean_text(source["FTAG"])
    report["source_date_format"] = _source_date_format(source["Date"])
    invalid_date = dates.isna()
    empty_home, empty_away = home.isna() | home.eq(""), away.isna() | away.eq("")
    invalid_team = (empty_home | empty_away | home.str.casefold().eq(away.str.casefold())).fillna(True)
    empty_home_score, empty_away_score = home_score.isna() | home_score.eq(""), away_score.isna() | away_score.eq("")
    unfinished = empty_home_score & empty_away_score & ~invalid_date & ~invalid_team
    partial_score = empty_home_score ^ empty_away_score
    invalid_numeric_score = (~empty_home_score & _invalid_score(home_score)) | (~empty_away_score & _invalid_score(away_score))
    invalid_score = (partial_score | invalid_numeric_score).fillna(True)
    structural = ~invalid_date & ~invalid_team
    fixture = pd.DataFrame({"date": dates, "home_team": home, "away_team": away}, index=source.index)
    match_ids = pd.Series(
        [stable_match_id(normalized, h, a) if valid else "" for h, a, valid in zip(home, away, structural)],
        index=source.index,
        dtype="string",
    )
    duplicate_id = pd.Series(False, index=source.index)
    duplicate_fixture = pd.Series(False, index=source.index)
    if structural.any():
        duplicate_id.loc[structural] = match_ids.loc[structural].duplicated(keep=False)
        duplicate_fixture.loc[structural] = fixture.loc[structural].duplicated(subset=["date", "home_team", "away_team"], keep=False)
    invalid = (invalid_date | invalid_team | invalid_score | duplicate_id | duplicate_fixture).fillna(True)
    report.update(
        invalid_rows=int(invalid.sum()),
        invalid_date_rows=int(invalid_date.sum()),
        invalid_score_rows=int(invalid_score.sum()),
        invalid_team_rows=int(invalid_team.sum()),
        duplicate_rows=int(duplicate_id.sum()),
        duplicate_fixture_rows=int(duplicate_fixture.sum()),
        skipped_unfinished_count=int(unfinished.sum()),
        team_count=int(pd.concat([home.loc[structural], away.loc[structural]], ignore_index=True).nunique()),
    )
    report["skipped_unfinished"] = [
        {
            "season": normalized,
            "source_row": int(index) + 2,
            "date": str(source.loc[index, "Date"]).strip(),
            "home_team": str(home.loc[index]),
            "away_team": str(away.loc[index]),
            "reason": "both full-time score fields are empty",
        }
        for index in source.index[unfinished]
    ]
    errors: list[dict[str, Any]] = []
    errors.extend(_row_error(source, invalid_date, "invalid_date", "date is missing, malformed, or uses a mixed source format"))
    errors.extend(_row_error(source, invalid_team, "invalid_team", "home/away team is missing or both teams are equal"))
    errors.extend(_row_error(source, partial_score, "partial_score", "exactly one full-time score field is empty"))
    errors.extend(_row_error(source, invalid_numeric_score & ~partial_score, "invalid_score", "score must be a finite non-negative integer"))
    errors.extend(_row_error(source, duplicate_id, "duplicate_match_id", "season and home/away teams produce a duplicate match_id"))
    errors.extend(_row_error(source, duplicate_fixture, "duplicate_fixture", "date and home/away teams duplicate a fixture"))
    valid_completed = structural & ~invalid_score & ~duplicate_id & ~duplicate_fixture & ~unfinished
    report["completed_match_count"] = int(valid_completed.sum())
    report["match_count"] = report["completed_match_count"]
    report["meets_expected_matches"] = report["completed_match_count"] == 380
    valid_dates = dates.loc[valid_completed]
    if not valid_dates.empty:
        report["earliest_date"] = valid_dates.min().strftime("%Y-%m-%d")
        report["latest_date"] = valid_dates.max().strftime("%Y-%m-%d")
    if report["invalid_rows"]:
        report["warnings"].append(f"{report['invalid_rows']} row(s) fail sync validation")
    if report["skipped_unfinished_count"]:
        report["warnings"].append(f"{report['skipped_unfinished_count']} unfinished fixture(s) were excluded")
    if not report["meets_expected_matches"]:
        report["warnings"].append(f"completed match count {report['completed_match_count']} does not equal the 380-match full-season expectation")
        if report["strict_380_gate_applied"]:
            errors.append(
                {
                    "kind": "incomplete_season",
                    "message": "requested season must contain exactly 380 completed matches unless it is explicitly declared in-progress",
                    "completed_match_count": report["completed_match_count"],
                    "expected_matches": report["expected_matches"],
                }
            )
    report["errors"] = errors
    report["status"] = "failed" if errors else ("warning" if report["warnings"] else "ok")
    if errors:
        return report, None
    canonical = pd.DataFrame(
        {
            "match_id": [match_ids.loc[index] for index in source.index[valid_completed]],
            "date": dates.loc[valid_completed].dt.strftime("%Y-%m-%d").tolist(),
            "season": normalized,
            "home_team": home.loc[valid_completed].tolist(),
            "away_team": away.loc[valid_completed].tolist(),
            "home_goals": pd.to_numeric(home_score.loc[valid_completed], errors="raise").astype("int64").tolist(),
            "away_goals": pd.to_numeric(away_score.loc[valid_completed], errors="raise").astype("int64").tolist(),
        }
    )
    if canonical.empty:
        return report, canonical.loc[:, CANONICAL_COLUMNS].copy()
    return report, validate_matches(canonical).loc[:, CANONICAL_COLUMNS].copy()


def audit_sync_source_frame(source: pd.DataFrame, season: str, in_progress_season: str | None = None) -> dict[str, Any]:
    """Audit one source frame, applying the strict 380 gate unless explicitly exempted."""
    return _analyse_source(source, season, in_progress_season)[0]


def convert_sync_source_frame(source: pd.DataFrame, season: str, in_progress_season: str | None = None) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Convert completed rows and explicitly record unfinished rows."""
    report, canonical = _analyse_source(source, season, in_progress_season)
    if report["status"] == "failed" or canonical is None:
        raise DataValidationError(f"invalid EPL sync source {report['season']}: {_source_error(report)}")
    return canonical, report


def _snapshot_id() -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"epl_sync_{stamp}_{uuid.uuid4().hex[:12]}"


def _download_snapshot(
    seasons: tuple[str, ...], data_dir: Path, timeout: float, in_progress_season: str | None = None
) -> tuple[Path, dict[str, Any]]:
    if timeout <= 0:
        raise ValueError("sync timeout must be positive")
    snapshot = data_dir / "staging" / _snapshot_id()
    raw_dir = snapshot / "raw"
    raw_dir.mkdir(parents=True, exist_ok=False)
    entries: list[dict[str, Any]] = []
    for season in seasons:
        raw_path = raw_dir / raw_filename(season)
        entry: dict[str, Any] = {
            "season": season,
            **_season_policy(season, in_progress_season),
            "url": source_url(season),
            "retrieved_at": _utc_now(),
            "raw_filename": raw_path.name,
            "sha256": None,
            "http_status": None,
            "content_type": None,
            "row_count": None,
            "data_cutoff": None,
            "parse_status": "not_run",
            "audit_status": None,
            "completed_match_count": None,
            "skipped_unfinished_count": None,
            "download_status": "failed",
            "error": None,
        }
        try:
            request = urllib.request.Request(entry["url"], headers={"User-Agent": "MatchQuant/0.1 EPL sync"})
            with urllib.request.urlopen(request, timeout=timeout) as response:
                body = response.read()
                entry["http_status"] = int(response.getcode())
                entry["content_type"] = response.headers.get("Content-Type")
            if entry["http_status"] != 200:
                raise DataValidationError(f"unexpected HTTP status {entry['http_status']}")
            raw_path.write_bytes(body)
            entry["sha256"] = file_sha256(raw_path)
            source = read_source_csv(raw_path)
            report, _ = _analyse_source(source, season, in_progress_season)
            entry.update(
                row_count=int(len(source)),
                data_cutoff=report["latest_date"],
                source_date_format=report["source_date_format"],
                audit_status=report["status"],
                completed_match_count=report["completed_match_count"],
                skipped_unfinished_count=report["skipped_unfinished_count"],
                audit_warnings=report["warnings"],
                parse_status="ok",
                download_status="ok",
            )
        except urllib.error.HTTPError as exc:
            entry.update(http_status=exc.code, error=f"HTTP {exc.code}: {exc.reason}")
        except (urllib.error.URLError, TimeoutError, OSError, ValueError, DataValidationError) as exc:
            if raw_path.is_file() and entry["sha256"] is None:
                entry["sha256"] = file_sha256(raw_path)
            entry["parse_status"] = "failed" if raw_path.is_file() else "not_run"
            entry["error"] = str(exc)
        entries.append(entry)
    manifest = {
        "schema_version": SYNC_SOURCE_MANIFEST_VERSION,
        "adapter_version": SYNC_ADAPTER_VERSION,
        "source": "Football-Data.co.uk",
        "competition": "English Premier League",
        "division": "E0",
        "snapshot_id": snapshot.name,
        "generated_at": _utc_now(),
        "requested_seasons": list(seasons),
        **_season_policy_manifest(seasons, in_progress_season),
        "source_urls": [entry["url"] for entry in entries],
        "data_cutoff": max((entry["data_cutoff"] for entry in entries if entry["data_cutoff"]), default=None),
        "status": "ok" if all(entry["download_status"] == "ok" and entry["parse_status"] == "ok" and entry["audit_status"] != "failed" for entry in entries) else "failed",
        "files": entries,
    }
    manifest_path = snapshot / "raw" / "source_manifest.json"
    write_json(manifest, manifest_path)
    failures = [entry["season"] for entry in entries if entry["download_status"] != "ok" or entry["parse_status"] != "ok"]
    if failures:
        raise DataValidationError(f"EPL sync download failed for {', '.join(failures)}; staging: {snapshot}")
    return snapshot, manifest


def _season_policy_manifest(seasons: tuple[str, ...], in_progress_season: str | None) -> dict[str, Any]:
    return {
        "in_progress_season": in_progress_season,
        "declared_in_progress_season": in_progress_season,
        "season_policy": {season: _season_policy(season, in_progress_season) for season in seasons},
    }
