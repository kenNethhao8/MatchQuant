"""Offline Football-Data E0 fixture snapshots for the M4 forecast input."""

from __future__ import annotations

import hashlib
import math
from typing import Any

import pandas as pd

from .epl import _source_date_format, normalize_season
from .fixtures import FIXTURE_COLUMNS, FIXTURE_SCHEMA_VERSION, validate_fixtures
from .schema import DataValidationError

FIXTURE_SOURCE_REQUIRED_COLUMNS = ("Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG")
FIXTURE_ADAPTER_VERSION = "m5-epl-football-data-fixture-v1"
FIXTURE_MANIFEST_VERSION = "m5-fixture-snapshot-manifest-v1"
FIXTURE_QUALITY_VERSION = "m5-fixture-quality-report-v1"
FIXTURE_DIFF_VERSION = "m5-fixture-diff-report-v1"


def stable_fixture_id(season: str, home_team: str, away_team: str) -> str:
    """Return an ID unchanged by a date rescheduling or score update."""

    payload = "\x1f".join(
        [normalize_season(season), str(home_team).strip(), str(away_team).strip()]
    )
    return "fd-e0-fixture-" + hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _clean(value: object) -> str:
    return "" if pd.isna(value) else str(value).strip()


def _parse_dates(values: pd.Series) -> tuple[pd.Series, str | None]:
    cleaned = values.astype("string").str.strip()
    date_format = _source_date_format(cleaned)
    if date_format is None:
        return pd.Series(pd.NaT, index=values.index, dtype="datetime64[ns]"), None
    return pd.to_datetime(cleaned, format=date_format, errors="coerce"), date_format


def _score(value: object) -> tuple[str, int | None]:
    text = _clean(value)
    if not text:
        return "empty", None
    numeric = pd.to_numeric(text, errors="coerce")
    try:
        number = float(numeric)
    except (TypeError, ValueError):
        return "invalid", None
    if not math.isfinite(number) or number < 0 or not number.is_integer():
        return "invalid", None
    return "valid", int(number)


def _report_template(season: str | None, source_rows: int = 0) -> dict[str, Any]:
    counts = {
        "source_rows": int(source_rows),
        "pending_rows": 0,
        "completed_rows": 0,
        "partial_score_rows": 0,
        "invalid_rows": 0,
        "invalid_date_rows": 0,
        "invalid_team_rows": 0,
        "invalid_score_rows": 0,
        "duplicate_stable_fixture_ids": 0,
        "duplicate_rows": 0,
        "conflicting_fixture_ids": 0,
    }
    return {
        "schema_version": FIXTURE_QUALITY_VERSION,
        "adapter_version": FIXTURE_ADAPTER_VERSION,
        "fixture_schema_version": FIXTURE_SCHEMA_VERSION,
        "season": season,
        "status": "failed",
        "source_date_format": None,
        "date_range": {"earliest": None, "latest": None},
        "source_date_range": {"earliest": None, "latest": None},
        "completed_date_range": {"earliest": None, "latest": None},
        "pending_date_range": {"earliest": None, "latest": None},
        "completed_data_cutoff": None,
        "completed_data_cutoff_status": "missing_no_completed_results",
        "data_cutoff": None,
        "data_cutoff_status": "missing_no_completed_results",
        "counts": counts,
        **counts,
        "invalid_row_details": [],
        "completed_row_details": [],
        "duplicate_details": [],
        "pending_fixture_ids": [],
        "completed_rows_excluded_from_fixture_output": True,
        "completed_rows_not_written_to_history": True,
        "errors": [],
        "warnings": [],
        "limitations": [
            "This is an offline local E0-compatible snapshot; source reachability and currentness are not established.",
            "No team alias or fixture identity guess is applied; source team text is retained.",
            "Source redistribution rights are not established by this workflow.",
        ],
    }


def _set_counts(report: dict[str, Any], **values: int) -> None:
    report["counts"].update({key: int(value) for key, value in values.items()})
    for key, value in values.items():
        report[key] = int(value)


def _date_range(values: list[str]) -> dict[str, str | None]:
    ordered = sorted(value for value in values if value)
    return {
        "earliest": ordered[0] if ordered else None,
        "latest": ordered[-1] if ordered else None,
    }


def _analyse(source: pd.DataFrame, season: str) -> tuple[dict[str, Any], list[dict[str, str]]]:
    if not isinstance(source, pd.DataFrame):
        raise TypeError("source must be a pandas DataFrame")
    normalized = normalize_season(season)
    report = _report_template(normalized, len(source))
    missing = [column for column in FIXTURE_SOURCE_REQUIRED_COLUMNS if column not in source.columns]
    if missing:
        message = f"missing required source columns: {', '.join(missing)}"
        report["errors"].append(message)
        _set_counts(report, invalid_rows=len(source))
        return report, []
    if source.empty:
        report["errors"].append("source CSV contains no rows")
        return report, []

    dates, date_format = _parse_dates(source["Date"])
    report["source_date_format"] = date_format
    valid_dates = dates.dropna()
    source_date_range = _date_range(valid_dates.dt.strftime("%Y-%m-%d").tolist())
    report["source_date_range"] = source_date_range
    report["date_range"] = dict(source_date_range)

    pending: list[dict[str, str]] = []
    identity_rows: dict[str, list[dict[str, Any]]] = {}
    counts = {key: 0 for key in report["counts"] if key != "source_rows"}
    for index, row in source.reset_index(drop=True).iterrows():
        source_row = index + 2
        date_value = dates.iloc[index]
        date_text = date_value.strftime("%Y-%m-%d") if not pd.isna(date_value) else None
        home, away = _clean(row["HomeTeam"]), _clean(row["AwayTeam"])
        home_state, home_goals = _score(row["FTHG"])
        away_state, away_goals = _score(row["FTAG"])
        reasons: list[str] = []
        if date_text is None:
            counts["invalid_date_rows"] += 1
            reasons.append("invalid_date")
        if not home or not away or home.casefold() == away.casefold():
            counts["invalid_team_rows"] += 1
            reasons.append("invalid_team")

        filled_count = int(home_state != "empty") + int(away_state != "empty")
        if filled_count == 1:
            counts["partial_score_rows"] += 1
            reasons.append("partial_score")
            if home_state == "invalid" or away_state == "invalid":
                counts["invalid_score_rows"] += 1
                reasons.append("invalid_score")
        elif filled_count == 2 and (home_state != "valid" or away_state != "valid"):
            counts["invalid_score_rows"] += 1
            reasons.append("invalid_score")

        fixture_id = None
        if home and away:
            fixture_id = stable_fixture_id(normalized, home, away)
            identity_rows.setdefault(fixture_id, []).append(
                {
                    "source_row": source_row,
                    "fixture_id": fixture_id,
                    "date": date_text,
                    "home_team": home,
                    "away_team": away,
                    "status": "pending" if filled_count == 0 else "completed" if filled_count == 2 and not reasons else "invalid",
                    "home_score": home_goals if home_state == "valid" else _clean(row["FTHG"]),
                    "away_score": away_goals if away_state == "valid" else _clean(row["FTAG"]),
                }
            )

        if not reasons and filled_count == 0 and date_text is not None:
            counts["pending_rows"] += 1
            pending.append(
                {
                    "fixture_id": fixture_id or stable_fixture_id(normalized, home, away),
                    "date": date_text,
                    "season": normalized,
                    "home_team": home,
                    "away_team": away,
                }
            )
            report["pending_fixture_ids"].append(pending[-1]["fixture_id"])
        elif not reasons and filled_count == 2:
            counts["completed_rows"] += 1
            report["completed_row_details"].append(
                {
                    "source_row": source_row,
                    "fixture_id": fixture_id,
                    "date": date_text,
                    "home_team": home,
                    "away_team": away,
                    "home_goals": home_goals,
                    "away_goals": away_goals,
                    "excluded_from_fixture_output": True,
                    "not_written_to_history": True,
                }
            )
        else:
            counts["invalid_rows"] += 1
            report["invalid_row_details"].append(
                {"source_row": source_row, "fixture_id": fixture_id, "reasons": sorted(set(reasons))}
            )

    for fixture_id, rows in identity_rows.items():
        if len(rows) < 2:
            continue
        counts["duplicate_stable_fixture_ids"] += 1
        counts["duplicate_rows"] += len(rows)
        signatures = {(item["date"], item["status"], item["home_score"], item["away_score"]) for item in rows}
        conflict = len(signatures) > 1
        if conflict:
            counts["conflicting_fixture_ids"] += 1
        report["duplicate_details"].append(
            {"fixture_id": fixture_id, "source_rows": [item["source_row"] for item in rows], "conflict": conflict, "rows": rows}
        )

    completed_date_range = _date_range(
        [item["date"] for item in report["completed_row_details"]]
    )
    pending_date_range = _date_range([item["date"] for item in pending])
    report["completed_date_range"] = completed_date_range
    report["pending_date_range"] = pending_date_range
    report["completed_data_cutoff"] = completed_date_range["latest"]
    report["completed_data_cutoff_status"] = (
        "known" if report["completed_data_cutoff"] else "missing_no_completed_results"
    )
    report["data_cutoff"] = report["completed_data_cutoff"]
    report["data_cutoff_status"] = report["completed_data_cutoff_status"]

    _set_counts(report, **counts)
    if report["completed_rows"]:
        report["warnings"].append(
            "completed rows are recorded for quality only; they are excluded from fixture output and history updates"
        )
    if report["source_date_format"] is None:
        report["errors"].append("Date values are not all in one exact DD/MM/YYYY or DD/MM/YY format")
    if report["partial_score_rows"]:
        report["errors"].append(f"{report['partial_score_rows']} row(s) have only one full-time score")
    if report["invalid_rows"]:
        report["errors"].append(f"{report['invalid_rows']} row(s) have invalid date, team, or score fields")
    if report["duplicate_stable_fixture_ids"]:
        report["errors"].append(
            f"{report['duplicate_stable_fixture_ids']} stable fixture ID group(s) are duplicated"
        )
    if report["invalid_rows"] or report["duplicate_stable_fixture_ids"]:
        report["status"] = "failed"
    elif not report["pending_rows"]:
        report["status"] = "no_pending_fixtures"
        report["warnings"].append("no_pending_fixtures: no score-free rows are ready for M4 forecast input")
    else:
        report["status"] = "ok"
    return report, pending


def audit_fixture_source_frame(source: pd.DataFrame, season: str) -> dict[str, Any]:
    """Audit a local E0-compatible frame without writing files."""

    return _analyse(source, season)[0]


def convert_fixture_source_frame(
    source: pd.DataFrame, season: str
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Convert only valid pending rows to the independent M4 fixture schema."""

    report, pending = _analyse(source, season)
    if report["status"] == "failed":
        raise DataValidationError(f"invalid E0 fixture source: {'; '.join(report['errors'])}")
    if report["status"] == "no_pending_fixtures":
        raise DataValidationError("no_pending_fixtures: source has no score-free fixture rows")
    frame = pd.DataFrame(pending, columns=FIXTURE_COLUMNS)
    validated = validate_fixtures(frame).sort_values(["_timestamp", "fixture_id"], kind="mergesort")
    return validated.loc[:, FIXTURE_COLUMNS].reset_index(drop=True), report


def compare_fixtures(baseline: pd.DataFrame, candidate: pd.DataFrame) -> dict[str, Any]:
    """Compare two validated fixture frames by explicit IDs, without fuzzy aliases.

    A changed source team spelling normally changes the stable ID and is therefore
    reported as added plus missing. ``revised`` is reserved for rows retaining the
    same fixture ID, such as a date change.
    """

    baseline_frame = validate_fixtures(baseline).loc[:, FIXTURE_COLUMNS]
    candidate_frame = validate_fixtures(candidate).loc[:, FIXTURE_COLUMNS]
    baseline_map = baseline_frame.set_index("fixture_id", drop=False)
    candidate_map = candidate_frame.set_index("fixture_id", drop=False)
    baseline_ids, candidate_ids = set(baseline_map.index), set(candidate_map.index)
    added, missing, unchanged, revised = [], [], [], []
    for fixture_id in sorted(candidate_ids - baseline_ids):
        added.append({"fixture_id": fixture_id, "candidate": candidate_map.loc[fixture_id].to_dict()})
    for fixture_id in sorted(baseline_ids - candidate_ids):
        missing.append(
            {
                "fixture_id": fixture_id,
                "baseline": baseline_map.loc[fixture_id].to_dict(),
                "manual_judgment_required": True,
                "possible_reasons": ["completed", "cancelled", "source_missing"],
            }
        )
    for fixture_id in sorted(baseline_ids & candidate_ids):
        before, after = baseline_map.loc[fixture_id], candidate_map.loc[fixture_id]
        changed = [
            column
            for column in ("date", "season", "home_team", "away_team")
            if str(before[column]) != str(after[column])
        ]
        if changed:
            revised.append(
                {
                    "fixture_id": fixture_id,
                    "changed_fields": changed,
                    "baseline": before.to_dict(),
                    "candidate": after.to_dict(),
                }
            )
        else:
            unchanged.append({"fixture_id": fixture_id})
    counts = {
        "added": len(added),
        "unchanged": len(unchanged),
        "revised": len(revised),
        "missing_from_candidate": len(missing),
    }
    return {
        "schema_version": FIXTURE_DIFF_VERSION,
        "status": "ok",
        "counts": counts,
        "details": {"added": added, "unchanged": unchanged, "revised": revised, "missing_from_candidate": missing},
        "missing_from_candidate_requires_manual_judgment": bool(missing),
        "team_text_change_semantics": (
            "stable fixture IDs include season and stripped source home/away text; "
            "changed team text is added plus missing/manual review unless the same ID is retained"
        ),
        "apply": {"performed": False, "pointer_updated": False, "baseline_deleted": False},
    }


def prepare_epl_fixtures(*args: Any, **kwargs: Any) -> dict[str, Any]:
    """Lazily import the snapshot writer while preserving one public entry point."""

    from .epl_fixture_snapshot import prepare_epl_fixtures as _prepare

    return _prepare(*args, **kwargs)


__all__ = [
    "FIXTURE_ADAPTER_VERSION",
    "FIXTURE_DIFF_VERSION",
    "FIXTURE_MANIFEST_VERSION",
    "FIXTURE_QUALITY_VERSION",
    "FIXTURE_SOURCE_REQUIRED_COLUMNS",
    "audit_fixture_source_frame",
    "compare_fixtures",
    "convert_fixture_source_frame",
    "prepare_epl_fixtures",
    "stable_fixture_id",
]
