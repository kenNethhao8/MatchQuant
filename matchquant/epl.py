"""Football-Data.co.uk EPL download, audit, and M1 canonical conversion."""
from __future__ import annotations

import hashlib
import json
import re
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from .io import file_sha256, write_json
from .schema import DataValidationError, validate_matches

DEFAULT_EPL_SEASONS = tuple(f"{year}-{(year + 1) % 100:02d}" for year in range(2015, 2025))
SOURCE_REQUIRED_COLUMNS = ("Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG")
CANONICAL_COLUMNS = ("match_id", "date", "season", "home_team", "away_team", "home_goals", "away_goals")
SOURCE_BASE_URL = "https://www.football-data.co.uk/mmz4281"
ADAPTER_VERSION = "m2-epl-football-data-v1"
SOURCE_MANIFEST_VERSION = "m2-source-manifest-v1"
QUALITY_REPORT_VERSION = "m2-quality-report-v1"
PROCESSED_MANIFEST_VERSION = "m2-processed-manifest-v1"
def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
def normalize_season(value: str) -> str:
    """Normalize a season label without inferring it from match dates."""
    text = str(value).strip()
    compact = re.fullmatch(r"(\d{2})(\d{2})", text)
    match = re.fullmatch(r"(20\d{2})-(\d{2})", text) if not compact else None
    if compact:
        start, end = 2000 + int(compact.group(1)), compact.group(2)
    elif match:
        start, end = int(match.group(1)), match.group(2)
    else:
        raise ValueError("season must use YYYY-YY or YYXX format, e.g. 2015-16")
    if int(end) != (start + 1) % 100:
        raise ValueError(f"season has inconsistent years: {value!r}")
    return f"{start:04d}-{int(end):02d}"
def _season_list(values: Iterable[str] | None, default: Iterable[str] = ()) -> tuple[str, ...]:
    selected = list(default if values is None else values)
    if not selected:
        raise ValueError("at least one season is required")
    normalized = [normalize_season(value) for value in selected]
    if len(set(normalized)) != len(normalized):
        raise ValueError("season list contains duplicates")
    return tuple(sorted(normalized))
def source_url(season: str) -> str:
    normalized = normalize_season(season)
    return f"{SOURCE_BASE_URL}/{normalized[2:4]}{normalized[5:7]}/E0.csv"
def raw_filename(season: str) -> str:
    return f"E0_{normalize_season(season)}.csv"
def stable_match_id(season: str, home_team: str, away_team: str) -> str:
    """Build a revision-safe ID from season and stripped source team names."""
    payload = "\x1f".join([normalize_season(season), str(home_team).strip(), str(away_team).strip()])
    return "fd-e0-" + hashlib.sha256(payload.encode("utf-8")).hexdigest()
def read_source_csv(path: str | Path) -> pd.DataFrame:
    """Read a Football-Data CSV without dropping malformed rows."""
    source_path = Path(path)
    for encoding in ("utf-8-sig", "latin-1"):
        try:
            return pd.read_csv(source_path, dtype="string", keep_default_na=False, encoding=encoding)
        except UnicodeDecodeError:
            continue
        except (OSError, ValueError, pd.errors.ParserError) as exc:
            raise DataValidationError(f"could not parse source CSV '{source_path}': {exc}") from exc
    raise DataValidationError(f"could not decode source CSV '{source_path}'")
def _missing_source_columns(source: pd.DataFrame) -> list[str]:
    return [column for column in SOURCE_REQUIRED_COLUMNS if column not in source.columns]
def _clean_text(values: pd.Series) -> pd.Series:
    return values.astype("string").str.strip()
def _source_date_format(values: pd.Series) -> str | None:
    non_empty = _clean_text(values)
    non_empty = non_empty[non_empty.ne("")].dropna()
    if non_empty.empty:
        return None
    if non_empty.str.fullmatch(r"\d{2}/\d{2}/\d{4}").all():
        return "%d/%m/%Y"
    if non_empty.str.fullmatch(r"\d{2}/\d{2}/\d{2}").all():
        return "%d/%m/%y"
    return None
def _parse_source_dates(values: pd.Series) -> pd.Series:
    """Parse one source file's exact DD/MM/YYYY or DD/MM/YY format."""
    cleaned = _clean_text(values)
    date_format = _source_date_format(cleaned)
    if date_format is None:
        return pd.Series(pd.NaT, index=values.index, dtype="datetime64[ns]")
    return pd.to_datetime(cleaned, format=date_format, errors="coerce")
def _invalid_score(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(_clean_text(values), errors="coerce")
    numeric_array = numeric.astype("float64").to_numpy()
    finite = pd.Series(np.isfinite(numeric_array), index=values.index)
    return numeric.isna() | ~finite | (numeric < 0) | (numeric % 1 != 0)
def _team_conflicts(home: pd.Series, away: pd.Series) -> list[dict[str, Any]]:
    values = pd.concat([home, away], ignore_index=True).dropna().astype(str)
    groups: dict[str, set[str]] = {}
    for value in values.unique():
        groups.setdefault(re.sub(r"\s+", " ", value).casefold(), set()).add(value)
    return [
        {"normalized": key, "variants": sorted(variants)}
        for key, variants in sorted(groups.items())
        if len(variants) > 1
    ]
def audit_source_frame(source: pd.DataFrame, season: str) -> dict[str, Any]:
    """Return a non-destructive quality record for one source season."""
    normalized_season = normalize_season(season)
    report: dict[str, Any] = dict(
        season=normalized_season,
        source_rows=int(len(source)),
        match_count=0,
        team_count=0,
        earliest_date=None,
        latest_date=None,
        duplicate_rows=0,
        duplicate_fixture_rows=0,
        invalid_rows=0,
        invalid_date_rows=0,
        invalid_score_rows=0,
        invalid_team_rows=0,
        source_date_format=None,
        expected_matches=380,
        meets_expected_matches=False,
        team_name_conflicts=[],
        unresolved_aliases=[],
        warnings=[],
        status="failed",
    )
    missing = _missing_source_columns(source)
    if missing:
        report.update(invalid_rows=len(source), warnings=[f"missing required source columns: {', '.join(missing)}"])
        return report
    date_format = _source_date_format(source["Date"])
    dates, home, away = _parse_source_dates(source["Date"]), _clean_text(source["HomeTeam"]), _clean_text(source["AwayTeam"])
    report["source_date_format"] = date_format
    invalid_date = dates.isna()
    invalid_home, invalid_away = home.isna() | home.eq(""), away.isna() | away.eq("")
    same_team = home.str.casefold().eq(away.str.casefold())
    invalid_team = (invalid_home | invalid_away | same_team).fillna(True)
    invalid_score = (_invalid_score(source["FTHG"]) | _invalid_score(source["FTAG"])).fillna(True)
    invalid = (invalid_date | invalid_team | invalid_score).fillna(True)
    valid = ~invalid
    fixture = pd.DataFrame({"date": dates, "home_team": home, "away_team": away}, index=source.index)
    duplicate = pd.Series(False, index=source.index)
    if valid.any(): duplicate.loc[valid] = fixture.loc[valid].duplicated(subset=["date", "home_team", "away_team"], keep=False)
    valid_dates = dates.loc[valid]
    teams = pd.concat([home.loc[valid], away.loc[valid]], ignore_index=True).dropna()
    report.update(
        match_count=int(valid.sum()),
        invalid_rows=int(invalid.sum()),
        invalid_date_rows=int(invalid_date.sum()),
        invalid_score_rows=int(invalid_score.sum()),
        invalid_team_rows=int(invalid_team.sum()),
        duplicate_rows=int(duplicate.sum()),
        duplicate_fixture_rows=int(duplicate.sum()),
        team_count=int(teams.nunique()),
        meets_expected_matches=int(valid.sum()) == 380,
        team_name_conflicts=_team_conflicts(source.loc[valid, "HomeTeam"], source.loc[valid, "AwayTeam"]),
    )
    if not valid_dates.empty:
        report.update(
            earliest_date=valid_dates.min().strftime("%Y-%m-%d"),
            latest_date=valid_dates.max().strftime("%Y-%m-%d"),
        )
    warnings: list[str] = []
    if report["invalid_rows"]:
        warnings.append(f"{report['invalid_rows']} row(s) have invalid date, team, or score fields")
    if date_format is None:
        warnings.append("Date values are not in one exact DD/MM/YYYY or DD/MM/YY format")
    if report["duplicate_rows"]:
        warnings.append(f"{report['duplicate_rows']} row(s) duplicate a fixture")
    if not report["meets_expected_matches"]:
        warnings.append(f"valid match count {report['match_count']} does not equal the 380-match EPL expectation")
    if report["team_name_conflicts"]:
        warnings.append("team-name variants share a whitespace/case-normalized spelling; no alias was applied")
    report["warnings"] = warnings
    report["status"] = "failed" if report["invalid_rows"] or report["duplicate_rows"] else ("warning" if warnings else "ok")
    return report
def convert_source_frame(source: pd.DataFrame, season: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Map one audited source frame to M1's canonical schema."""
    normalized_season = normalize_season(season)
    report = audit_source_frame(source, normalized_season)
    if report["status"] == "failed":
        raise DataValidationError(f"invalid EPL source season {normalized_season}: {'; '.join(report['warnings'])}")
    dates, home, away = _parse_source_dates(source["Date"]), _clean_text(source["HomeTeam"]), _clean_text(source["AwayTeam"])
    home_goals = pd.to_numeric(_clean_text(source["FTHG"]), errors="raise").astype("int64")
    away_goals = pd.to_numeric(_clean_text(source["FTAG"]), errors="raise").astype("int64")
    iso_dates = dates.dt.strftime("%Y-%m-%d")
    canonical = pd.DataFrame(
        {
            "match_id": [stable_match_id(normalized_season, h, a) for h, a in zip(home, away)],
            "date": iso_dates,
            "season": normalized_season,
            "home_team": home,
            "away_team": away,
            "home_goals": home_goals,
            "away_goals": away_goals,
        }
    )
    return validate_matches(canonical).loc[:, CANONICAL_COLUMNS].copy(), report
def download_epl(
    seasons: Iterable[str] | None = None,
    data_dir: str | Path = Path(r"D:\Codex\matchquant_data"),
    timeout: float = 30.0,
) -> dict[str, Any]:
    """Download explicit EPL seasons and write a traceable source manifest."""
    selected = _season_list(seasons, DEFAULT_EPL_SEASONS)
    if timeout <= 0:
        raise ValueError("download timeout must be positive")
    raw_dir = Path(data_dir) / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = raw_dir / "source_manifest.json"
    if manifest_path.exists() or any((raw_dir / raw_filename(season)).exists() for season in selected):
        raise DataValidationError("raw snapshot already exists; use a new data-dir to preserve immutable files")
    entries: list[dict[str, Any]] = []
    for season in selected:
        path = raw_dir / raw_filename(season)
        entry: dict[str, Any] = dict(
            season=season,
            url=source_url(season),
            retrieved_at=_utc_now(),
            raw_filename=path.name,
            sha256=None,
            http_status=None,
            content_type=None,
            row_count=None,
            data_cutoff=None,
            parse_status="not_run",
            status="failed",
            error=None,
        )
        try:
            request = urllib.request.Request(entry["url"], headers={"User-Agent": "MatchQuant/0.1 EPL research downloader"})
            with urllib.request.urlopen(request, timeout=timeout) as response:
                body, entry["http_status"] = response.read(), int(response.getcode())
                entry["content_type"] = response.headers.get("Content-Type")
            if entry["http_status"] != 200:
                raise DataValidationError(f"unexpected HTTP status {entry['http_status']}")
            path.write_bytes(body)
            entry["sha256"] = file_sha256(path)
            source = read_source_csv(path)
            audit = audit_source_frame(source, season)
            entry.update(row_count=int(len(source)), data_cutoff=audit["latest_date"], source_date_format=audit["source_date_format"], audit_status=audit["status"], audit_match_count=audit["match_count"], audit_team_count=audit["team_count"], audit_duplicate_rows=audit["duplicate_rows"], audit_invalid_rows=audit["invalid_rows"], meets_expected_matches=audit["meets_expected_matches"], audit_warnings=audit["warnings"], parse_status="ok")
            if audit["status"] == "failed" or not audit["meets_expected_matches"]:
                raise DataValidationError("source audit failed the complete-season 380-match gate")
            entry["status"] = "ok"
        except urllib.error.HTTPError as exc:
            entry.update(http_status=exc.code, error=f"HTTP {exc.code}: {exc.reason}")
        except (urllib.error.URLError, TimeoutError, OSError, ValueError, DataValidationError) as exc:
            if entry["parse_status"] == "not_run":
                entry["parse_status"] = "failed" if path.is_file() else "not_run"
            if path.is_file() and entry["sha256"] is None:
                entry["sha256"] = file_sha256(path)
            entry["error"] = str(exc)
        entries.append(entry)
    manifest = {
        "schema_version": SOURCE_MANIFEST_VERSION,
        "adapter_version": ADAPTER_VERSION,
        "source": "Football-Data.co.uk",
        "competition": "English Premier League",
        "division": "E0",
        "generated_at": _utc_now(),
        "requested_seasons": list(selected),
        "data_cutoff": max((entry["data_cutoff"] for entry in entries if entry["data_cutoff"]), default=None),
        "status": "ok" if all(entry["status"] == "ok" for entry in entries) else "failed",
        "files": entries,
    }
    write_json(manifest, manifest_path)
    failures = [entry["season"] for entry in entries if entry["status"] != "ok"]
    if failures:
        raise DataValidationError(f"EPL download failed for {', '.join(failures)}; manifest: {manifest_path}")
    manifest["manifest_path"] = str(manifest_path)
    return manifest
def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DataValidationError(f"could not read JSON manifest '{path}': {exc}") from exc
    if not isinstance(payload, dict):
        raise DataValidationError(f"JSON manifest root must be an object: {path}")
    return payload
def prepare_epl(
    seasons: Iterable[str] | None = None,
    data_dir: str | Path = Path(r"D:\Codex\matchquant_data"),
    output_path: str | Path | None = None,
    quality_report_path: str | Path | None = None,
    processed_manifest_path: str | Path | None = None,
) -> dict[str, Any]:
    """Convert downloaded raw files offline and write quality/hash manifests."""
    root, raw_dir = Path(data_dir), Path(data_dir) / "raw"
    manifest_path, manifest = raw_dir / "source_manifest.json", _read_json(raw_dir / "source_manifest.json")
    entries = manifest.get("files")
    if not isinstance(entries, list):
        raise DataValidationError("source manifest field 'files' must be a list")
    by_season: dict[str, dict[str, Any]] = {}
    for entry in entries:
        if not isinstance(entry, dict) or "season" not in entry:
            raise DataValidationError("source manifest contains an invalid file entry")
        season = normalize_season(str(entry["season"]))
        if season in by_season:
            raise DataValidationError(f"source manifest contains duplicate season {season}")
        by_season[season] = entry
    selected = _season_list(seasons, sorted(by_season))
    quality_path = Path(quality_report_path) if quality_report_path else root / "reports" / "epl_quality_report.json"
    output = Path(output_path) if output_path else root / "processed" / "epl_canonical.csv"
    processed_path = Path(processed_manifest_path) if processed_manifest_path else root / "processed" / "epl_processed_manifest.json"
    quality: dict[str, Any] = {
        "schema_version": QUALITY_REPORT_VERSION,
        "adapter_version": ADAPTER_VERSION,
        "source_manifest": str(manifest_path),
        "generated_at": _utc_now(),
        "requested_seasons": list(selected),
        "data_cutoff": None,
        "status": "failed",
        "seasons": [],
        "totals": {},
    }
    input_records: list[dict[str, Any]] = []
    canonical_frames: list[pd.DataFrame] = []
    errors: list[str] = []
    for season in selected:
        entry = by_season.get(season)
        if entry is None:
            error = f"season {season} is absent from source manifest"
            errors.append(error)
            quality["seasons"].append({"season": season, "status": "failed", "warnings": [error]})
            continue
        filename = str(entry.get("raw_filename", ""))
        report: dict[str, Any] = {"season": season, "raw_filename": filename, "source_url": entry.get("url"), "raw_sha256": entry.get("sha256")}
        audit: dict[str, Any] | None = None
        try:
            if entry.get("status") != "ok" or entry.get("parse_status") != "ok":
                raise DataValidationError("source manifest marks this file as unavailable or unparsed")
            if Path(filename).name != filename or not filename:
                raise DataValidationError("source manifest has an unsafe or empty raw filename")
            raw_path = raw_dir / filename
            if not raw_path.is_file():
                raise FileNotFoundError(f"raw file does not exist: {raw_path}")
            actual_hash = file_sha256(raw_path)
            if actual_hash != entry.get("sha256"):
                raise DataValidationError(f"raw SHA-256 mismatch: manifest={entry.get('sha256')} actual={actual_hash}")
            source = read_source_csv(raw_path)
            audit = audit_source_frame(source, season)
            audit.update(report)
            quality["seasons"].append(audit)
            if audit["status"] == "failed" or not audit["meets_expected_matches"]:
                raise DataValidationError("season fails the complete-season 380-match gate")
            converted, _ = convert_source_frame(source, season)
            canonical_frames.append(converted)
            input_records.append({"season": season, "raw_filename": filename, "url": entry.get("url"), "sha256": actual_hash})
        except (DataValidationError, FileNotFoundError, OSError, ValueError) as exc:
            if audit is None:
                report.update(status="failed", warnings=[str(exc)])
                quality["seasons"].append(report)
            errors.append(f"{season}: {exc}")
    quality["totals"] = {
        "seasons": len(quality["seasons"]),
        "source_rows": int(sum(item.get("source_rows", 0) for item in quality["seasons"])),
        "match_count": int(sum(item.get("match_count", 0) for item in quality["seasons"])),
        "invalid_rows": int(sum(item.get("invalid_rows", 0) for item in quality["seasons"])),
        "duplicate_rows": int(sum(item.get("duplicate_rows", 0) for item in quality["seasons"])),
    }
    if errors:
        write_json(quality, quality_path)
        raise DataValidationError(f"EPL preparation failed; quality report: {quality_path}; first error: {errors[0]}")
    validated = validate_matches(pd.concat(canonical_frames, ignore_index=True))
    canonical = validated.sort_values(["_timestamp", "match_id"], kind="mergesort").loc[:, CANONICAL_COLUMNS].reset_index(drop=True)
    quality["data_cutoff"] = canonical["date"].max()
    output.parent.mkdir(parents=True, exist_ok=True)
    canonical.to_csv(output, index=False, encoding="utf-8")
    output_hash = file_sha256(output)
    quality["status"] = "warning" if any(item.get("status") == "warning" for item in quality["seasons"]) else "ok"
    write_json(quality, quality_path)
    processed = {
        "schema_version": PROCESSED_MANIFEST_VERSION,
        "adapter_version": ADAPTER_VERSION,
        "canonical_format_version": "m1-canonical-v1",
        "source_manifest": str(manifest_path),
        "source_manifest_sha256": file_sha256(manifest_path),
        "generated_at": _utc_now(),
        "seasons": list(selected),
        "data_cutoff": canonical["date"].max(),
        "inputs": input_records,
        "output": {"filename": str(output), "sha256": output_hash, "rows": int(len(canonical)), "columns": list(CANONICAL_COLUMNS)},
        "quality_report": str(quality_path),
        "quality_status": quality["status"],
    }
    write_json(processed, processed_path)
    return {
        "seasons": list(selected),
        "rows": int(len(canonical)),
        "status": quality["status"],
        "output_path": str(output),
        "quality_report_path": str(quality_path),
        "processed_manifest_path": str(processed_path),
        "quality": quality,
        "processed_manifest": processed,
    }
