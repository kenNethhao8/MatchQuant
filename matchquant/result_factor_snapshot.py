"""Offline build, strict load, and snapshot-only query for M13A result factors."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import stat
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path, PurePath, PureWindowsPath
from typing import Any, Mapping, Sequence
from uuid import uuid4

import pandas as pd

from .backtest import _parse_boundary
from .engine_snapshot_build_support import history_manifest_evidence, reject_output_root
from .engine_snapshot_support import canonical_json_bytes, configuration_hash
from .fixtures import FIXTURE_SCHEMA_VERSION, load_fixtures
from .forecast import _fixtures_at_or_after_as_of, _history_before_as_of
from .io import file_sha256, load_matches
from .result_factor_report import render_build_report, render_query_report
from .result_factors import (
    APPEARANCE_KEYS, AS_OF_KEYS, CONFIG_KEYS, ELO_CONFIG, ELO_CONFIG_KEYS,
    LEAGUE_PRIOR_KEYS, LEAGUE_SIDE_KEYS, RESULT_FACTOR_BUILD_REPORT_FILENAME,
    RESULT_FACTOR_CONFIG, RESULT_FACTOR_CONFIG_SHA256, RESULT_FACTOR_COVERAGE_STATUS,
    RESULT_FACTOR_MANIFEST_FILENAME, RESULT_FACTOR_MANIFEST_SCHEMA_VERSION,
    RESULT_FACTOR_MINIMUM_SAMPLE, RESULT_FACTOR_QUERY_COLUMNS, RESULT_FACTOR_QUERY_CSV_FILENAME,
    RESULT_FACTOR_QUERY_FILENAME, RESULT_FACTOR_QUERY_REPORT_FILENAME,
    RESULT_FACTOR_QUERY_SCHEMA_VERSION, RESULT_FACTOR_SNAPSHOT_MAX_BYTES,
    RESULT_FACTOR_STATE_FILENAME, RESULT_FACTOR_STATE_SCHEMA_VERSION, RESULT_FACTOR_WINDOWS,
    STATE_KEYS, TEAM_STATE_KEYS, VENUE_TOTALS_KEYS, VENUE_TOTAL_KEYS,
    build_result_factor_state, fixture_team_factors, flatten_fixture_factors,
)
from .schema import DataValidationError, has_explicit_time

_HISTORY_MANIFEST_SCHEMAS = {"m2-processed-manifest-v1", "m2.1-processed-manifest-v1"}
_MANIFEST_KEYS = (
    "schema_version", "snapshot_id", "generated_at", "as_of", "history", "config",
    "config_sha256", "state", "report", "inventory", "status", "reasons",
    "coverage_status", "experimental", "probability_adjustment_applied",
)
_HISTORY_KEYS = (
    "path", "sha256", "manifest_path", "manifest_sha256", "manifest_schema_version",
    "dataset_version", "data_cutoff", "rows_total", "rows_used", "used_data_cutoff",
    "output_sha256",
)
_STATE_DESCRIPTOR_KEYS = ("filename", "schema_version", "sha256", "bytes")
_REPORT_DESCRIPTOR_KEYS = ("filename", "sha256", "bytes")
_INVENTORY_KEYS = ("manifest_bytes", "total_bytes")
_QUERY_KEYS = (
    "schema_version", "generated_at", "status", "as_of", "snapshot", "fixtures",
    "fixture_count", "results", "reasons", "coverage_status", "experimental",
    "probability_adjustment_applied", "validation", "failure_stage",
)
_VALID_STATUSES = {"ready", "needs_review"}
_SHA256_RE = re.compile(r"[0-9a-f]{64}")


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _read_json_object(path: Path, label: str) -> dict[str, Any]:
    def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise DataValidationError(f"{label} contains duplicate JSON key '{key}'")
            result[key] = value
        return result

    try:
        value = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=reject_duplicate_keys)
    except (OSError, json.JSONDecodeError) as exc:
        raise DataValidationError(f"could not read {label} '{path}': {exc}") from exc
    if not isinstance(value, dict):
        raise DataValidationError(f"{label} root must be an object")
    return value


def _write_json_bytes(payload: Mapping[str, Any], path: Path) -> None:
    path.write_bytes(canonical_json_bytes(dict(payload)))


def _valid_sha256(value: object) -> bool:
    return isinstance(value, str) and _SHA256_RE.fullmatch(value) is not None


def _require_exact_keys(payload: Mapping[str, Any], keys: Sequence[str], label: str) -> None:
    missing = [key for key in keys if key not in payload]
    extra = sorted(set(payload) - set(keys))
    if missing or extra:
        pieces = []
        if missing:
            pieces.append(f"missing {', '.join(missing)}")
        if extra:
            pieces.append(f"unknown {', '.join(extra)}")
        raise DataValidationError(f"{label} fields are invalid: {'; '.join(pieces)}")


def _is_int(value: object, *, minimum: int = 0, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise DataValidationError(f"{label} must be an integer >= {minimum}")
    return value


def _is_finite_number(value: object, *, label: str, minimum: float | None = None, maximum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise DataValidationError(f"{label} must be a finite number")
    numeric = float(value)
    if minimum is not None and numeric < minimum:
        raise DataValidationError(f"{label} must be >= {minimum}")
    if maximum is not None and numeric > maximum:
        raise DataValidationError(f"{label} must be <= {maximum}")
    return numeric


def _valid_iso_timestamp(value: object, label: str) -> pd.Timestamp:
    if not isinstance(value, str) or not value.strip():
        raise DataValidationError(f"{label} must be a non-empty ISO date/time string")
    try:
        return pd.Timestamp(pd.to_datetime(value, errors="raise", format="mixed"))
    except (TypeError, ValueError) as exc:
        raise DataValidationError(f"{label} is invalid") from exc


def _validate_as_of(raw: object, label: str) -> dict[str, Any]:
    if not isinstance(raw, Mapping):
        raise DataValidationError(f"{label} must be an object")
    _require_exact_keys(raw, AS_OF_KEYS, label)
    value, timestamp_text, date_only = raw["value"], raw["timestamp"], raw["date_only"]
    if not isinstance(value, str) or not value.strip():
        raise DataValidationError(f"{label}.value must be a non-empty string")
    if not isinstance(timestamp_text, str) or not timestamp_text.strip():
        raise DataValidationError(f"{label}.timestamp must be a non-empty string")
    if not isinstance(date_only, bool):
        raise DataValidationError(f"{label}.date_only must be boolean")
    try:
        parsed, expected_date_only = _parse_boundary(value, label)
        declared = pd.Timestamp(pd.to_datetime(timestamp_text, errors="raise", format="mixed"))
    except (TypeError, ValueError) as exc:
        raise DataValidationError(f"{label} metadata is invalid") from exc
    if date_only is not expected_date_only or declared != parsed:
        raise DataValidationError(f"{label} metadata is inconsistent")
    return {"value": value, "timestamp": timestamp_text, "date_only": date_only}


def _validate_generated_at(value: object) -> None:
    if not isinstance(value, str) or not value.endswith("Z"):
        raise DataValidationError("snapshot generated_at must be UTC ISO-8601")
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as exc:
        raise DataValidationError("snapshot generated_at is invalid") from exc
    if parsed.tzinfo is None or parsed.utcoffset() != timezone.utc.utcoffset(parsed):
        raise DataValidationError("snapshot generated_at must be UTC")


def _validate_config(config: object, expected_hash: object, label: str) -> None:
    if not isinstance(config, Mapping):
        raise DataValidationError(f"{label} must be an object")
    _require_exact_keys(config, CONFIG_KEYS, label)
    elo = config["elo"]
    if not isinstance(elo, Mapping):
        raise DataValidationError(f"{label}.elo must be an object")
    _require_exact_keys(elo, ELO_CONFIG_KEYS, f"{label}.elo")
    for key, expected in ELO_CONFIG.items():
        value = elo[key]
        _is_finite_number(value, label=f"{label}.elo.{key}")
        if type(value) is not type(expected) or value != expected:
            raise DataValidationError(f"{label}.elo.{key} does not match the frozen M13A value")
    if config["season_mode"] != "carry":
        raise DataValidationError(f"{label}.season_mode must be 'carry'")
    windows = config["windows"]
    if not isinstance(windows, list) or any(type(v) is not int for v in windows) or windows != list(RESULT_FACTOR_WINDOWS):
        raise DataValidationError(f"{label}.windows must be [5, 10]")
    if type(config["minimum_sample_for_ready"]) is not int or config["minimum_sample_for_ready"] != RESULT_FACTOR_MINIMUM_SAMPLE:
        raise DataValidationError(f"{label}.minimum_sample_for_ready is inconsistent")
    if type(config["venue_prior_weight"]) is not int or config["venue_prior_weight"] != 5:
        raise DataValidationError(f"{label}.venue_prior_weight is inconsistent")
    if not _valid_sha256(expected_hash):
        raise DataValidationError(f"{label} SHA-256 is invalid")
    if configuration_hash(dict(config)) != expected_hash or expected_hash != RESULT_FACTOR_CONFIG_SHA256:
        raise DataValidationError(f"{label} SHA-256 does not match the frozen M13A configuration")


def _path_has_reparse_point(path: Path) -> bool:
    try:
        info = path.lstat()
    except FileNotFoundError:
        return False
    attributes = getattr(info, "st_file_attributes", 0)
    reparse_flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
    return path.is_symlink() or bool(attributes & reparse_flag)


def _reject_reparse_components(snapshot_root: Path, artifact_path: Path) -> None:
    if _path_has_reparse_point(snapshot_root):
        raise DataValidationError("snapshot directory must not be a symlink or reparse point")
    if _path_has_reparse_point(artifact_path):
        raise DataValidationError("snapshot artifact must not be a symlink or reparse point")


def _is_within(path: Path, root: Path) -> bool:
    try:
        return path == root or root in path.parents
    except (OSError, ValueError):
        return False


def _path_overlap(left: Path, right: Path) -> bool:
    return _is_within(left, right) or _is_within(right, left)


def _resolve_output_root(output_root: str | Path, *inputs: Path) -> Path:
    raw = os.fspath(output_root)
    if not isinstance(raw, str) or not raw.strip():
        raise DataValidationError("output_root must be a non-empty path")
    if ".." in PurePath(raw).parts or ".." in PureWindowsPath(raw).parts:
        raise DataValidationError("output_root must not contain '..' path components")
    root = Path(raw).expanduser().resolve()
    repo = Path(__file__).resolve().parents[1]
    project = repo.parent
    protected = (
        Path(r"D:\Codex\matchquant_data"), Path(r"D:\Codex\matchquant_runs"),
        project / "_unpacked", repo, repo / "docs" / "baseline",
        project / "handoff" / "codex_designs", project / "handoff" / "codex_reviews",
        project / "handoff" / "deepseek_reports", project / "handoff" / "zcode_reports",
    )
    for protected_root in protected:
        if _path_overlap(root, protected_root.resolve()):
            raise DataValidationError(f"output_root overlaps a protected read-only path: {root}")
    reject_output_root(root, *inputs)
    for input_path in inputs:
        if _path_overlap(root, input_path.resolve().parent):
            raise DataValidationError(f"output_root overlaps an input directory: {root}")
    return root


def _new_artifact_dir(output_root: Path, prefix: str) -> Path:
    output_root.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    artifact_dir = output_root / f"{prefix}_{stamp}_{uuid4().hex[:8]}"
    artifact_dir.mkdir()
    return artifact_dir


def _safe_artifact_child(snapshot_root: Path, filename: object, expected: str, label: str) -> Path:
    if not isinstance(filename, str) or not filename:
        raise DataValidationError(f"{label} filename must be a non-empty string")
    if ".." in PurePath(filename).parts or ".." in PureWindowsPath(filename).parts:
        raise DataValidationError(f"{label} filename must not contain '..'")
    candidate = Path(filename)
    if candidate.is_absolute() or PureWindowsPath(filename).is_absolute() or "/" in filename or "\\" in filename or ":" in filename:
        raise DataValidationError(f"{label} must be one fixed filename in the snapshot directory")
    if filename != expected:
        raise DataValidationError(f"{label} filename must be {expected}")
    path = snapshot_root / filename
    _reject_reparse_components(snapshot_root, path)
    resolved = path.resolve(strict=True)
    if resolved.parent != snapshot_root or resolved.name != expected:
        raise DataValidationError(f"{label} resolves outside the snapshot directory")
    return resolved


def _validate_hash_and_bytes(record: object, *, filename: str, expected_keys: Sequence[str], base: Path, label: str) -> Path:
    if not isinstance(record, Mapping):
        raise DataValidationError(f"{label} descriptor must be an object")
    _require_exact_keys(record, expected_keys, f"{label} descriptor")
    path = _safe_artifact_child(base, record.get("filename"), filename, label)
    size = _is_int(record.get("bytes"), label=f"{label}.bytes")
    digest = record.get("sha256")
    if not _valid_sha256(digest):
        raise DataValidationError(f"{label}.sha256 is invalid")
    if not path.is_file():
        raise FileNotFoundError(f"snapshot {label} is missing: {path}")
    if path.stat().st_size != size:
        raise DataValidationError(f"snapshot {label} byte count does not match")
    if file_sha256(path) != digest:
        raise DataValidationError(f"snapshot {label} SHA-256 does not match")
    return path


def _validate_appearance(appearance: object, *, as_of: Mapping[str, Any], label: str) -> tuple[pd.Timestamp, str]:
    if not isinstance(appearance, Mapping):
        raise DataValidationError(f"{label} must be an object")
    _require_exact_keys(appearance, APPEARANCE_KEYS, label)
    for field in ("match_id", "date", "season"):
        if not isinstance(appearance[field], str) or not appearance[field].strip():
            raise DataValidationError(f"{label}.{field} must be a non-empty string")
    if not isinstance(appearance["date_only"], bool):
        raise DataValidationError(f"{label}.date_only must be boolean")
    if not isinstance(appearance["venue"], str) or appearance["venue"] not in {"home", "away"}:
        raise DataValidationError(f"{label}.venue is invalid")
    gf = _is_int(appearance["goals_for"], label=f"{label}.goals_for")
    ga = _is_int(appearance["goals_against"], label=f"{label}.goals_against")
    points = _is_int(appearance["points"], label=f"{label}.points")
    expected_points = 3 if gf > ga else 1 if gf == ga else 0
    if points != expected_points:
        raise DataValidationError(f"{label}.points does not match its score")
    _is_finite_number(appearance["opponent_pre_batch_elo"], label=f"{label}.opponent_pre_batch_elo", minimum=-100000, maximum=100000)
    try:
        match_date = pd.Timestamp(pd.to_datetime(appearance["date"], errors="raise", format="mixed"))
        batch_time = _valid_iso_timestamp(appearance["batch_time"], f"{label}.batch_time")
    except (TypeError, ValueError) as exc:
        raise DataValidationError(f"{label} date/time is invalid") from exc
    if has_explicit_time(appearance["date"]) == appearance["date_only"]:
        raise DataValidationError(f"{label}.date_only does not match its date value")
    if match_date.normalize() != batch_time.normalize():
        raise DataValidationError(f"{label}.batch_time calendar date does not match date")
    try:
        as_of_ts, _ = _parse_boundary(as_of["value"], "snapshot as_of")
        if not batch_time < as_of_ts:
            raise DataValidationError(f"{label} batch must be strictly before as_of")
    except (TypeError, ValueError) as exc:
        if isinstance(exc, DataValidationError):
            raise
        raise DataValidationError(f"{label} could not be compared with as_of") from exc
    return batch_time, str(appearance["match_id"])


def _close_number(actual: object, expected: float | None, label: str) -> None:
    if expected is None:
        if actual is not None:
            raise DataValidationError(f"{label} must be null when there are no prior samples")
        return
    value = _is_finite_number(actual, label=label)
    if not math.isclose(value, expected, rel_tol=0.0, abs_tol=1e-10):
        raise DataValidationError(f"{label} does not match its cumulative totals")


def _validate_state(state: object, manifest: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(state, dict):
        raise DataValidationError("factor state root must be an object")
    _require_exact_keys(state, STATE_KEYS, "factor state")
    if state["schema_version"] != RESULT_FACTOR_STATE_SCHEMA_VERSION:
        raise DataValidationError(f"factor state schema_version must be {RESULT_FACTOR_STATE_SCHEMA_VERSION}")
    state_as_of = _validate_as_of(state["as_of"], "factor state as_of")
    manifest_as_of = _validate_as_of(manifest["as_of"], "factor manifest as_of")
    if state_as_of != manifest_as_of:
        raise DataValidationError("factor state and manifest as_of do not match")

    history = manifest["history"]
    rows_used = _is_int(state["rows_used"], label="factor state rows_used")
    rows_total = _is_int(history["rows_total"], label="factor manifest history.rows_total")
    if rows_total == 0 or rows_used > rows_total:
        raise DataValidationError("factor state rows_used is out of range")
    if rows_used != history["rows_used"]:
        raise DataValidationError("factor state rows_used does not match manifest history")
    cutoff = state["used_data_cutoff"]
    if cutoff is None:
        if rows_used != 0:
            raise DataValidationError("factor state used_data_cutoff is missing for non-empty history")
    else:
        if not isinstance(cutoff, str) or not cutoff.strip() or rows_used == 0:
            raise DataValidationError("factor state used_data_cutoff is inconsistent with rows_used")
        used_ts = _valid_iso_timestamp(cutoff, "factor state used_data_cutoff")
        as_of_ts, _ = _parse_boundary(state_as_of["value"], "snapshot as_of")
        if not used_ts < as_of_ts:
            raise DataValidationError("factor state used_data_cutoff must be strictly before as_of")
    if cutoff != history["used_data_cutoff"]:
        raise DataValidationError("factor state and manifest used_data_cutoff do not match")

    _validate_config(state["config"], state["config_sha256"], "factor state config")
    _validate_config(manifest["config"], manifest["config_sha256"], "factor manifest config")
    if state["config"] != manifest["config"] or state["config_sha256"] != manifest["config_sha256"]:
        raise DataValidationError("factor state and manifest config do not match")

    priors = state["league_priors"]
    if not isinstance(priors, Mapping):
        raise DataValidationError("factor state league_priors must be an object")
    _require_exact_keys(priors, LEAGUE_PRIOR_KEYS, "factor state league_priors")
    normalized_priors: dict[str, dict[str, Any]] = {}
    for venue in LEAGUE_PRIOR_KEYS:
        side = priors[venue]
        if not isinstance(side, Mapping):
            raise DataValidationError(f"factor state league_priors.{venue} must be an object")
        _require_exact_keys(side, LEAGUE_SIDE_KEYS, f"factor state league_priors.{venue}")
        matches = _is_int(side["matches"], label=f"league_priors.{venue}.matches")
        gf_sum = _is_int(side["goals_for_sum"], label=f"league_priors.{venue}.goals_for_sum")
        ga_sum = _is_int(side["goals_against_sum"], label=f"league_priors.{venue}.goals_against_sum")
        if matches != rows_used:
            raise DataValidationError(f"league prior {venue} match count does not equal rows_used")
        _close_number(side["goals_for_mean"], gf_sum / matches if matches else None, f"league_priors.{venue}.goals_for_mean")
        _close_number(side["goals_against_mean"], ga_sum / matches if matches else None, f"league_priors.{venue}.goals_against_mean")
        normalized_priors[venue] = dict(side)

    teams = state["teams"]
    if not isinstance(teams, Mapping):
        raise DataValidationError("factor state teams must be an object")
    aggregate = {
        "total_appearances": 0, "goals_for_sum": 0, "goals_against_sum": 0,
        "venue": {venue: {field: 0 for field in VENUE_TOTAL_KEYS} for venue in VENUE_TOTALS_KEYS},
    }
    for team_name, team in teams.items():
        if not isinstance(team_name, str) or not team_name.strip() or not isinstance(team, Mapping):
            raise DataValidationError("factor state team records require non-empty string keys and objects")
        _require_exact_keys(team, TEAM_STATE_KEYS, f"factor state team '{team_name}'")
        total = _is_int(team["total_appearances"], label=f"{team_name}.total_appearances")
        if total < 1 or total > rows_used:
            raise DataValidationError(f"{team_name}.total_appearances is out of range")
        gf_sum = _is_int(team["goals_for_sum"], label=f"{team_name}.goals_for_sum")
        ga_sum = _is_int(team["goals_against_sum"], label=f"{team_name}.goals_against_sum")
        _is_finite_number(team["final_elo"], label=f"{team_name}.final_elo", minimum=-100000, maximum=100000)
        venue_totals = team["venue_totals"]
        if not isinstance(venue_totals, Mapping):
            raise DataValidationError(f"{team_name}.venue_totals must be an object")
        _require_exact_keys(venue_totals, VENUE_TOTALS_KEYS, f"{team_name}.venue_totals")
        local_venue: dict[str, dict[str, int]] = {}
        for venue in VENUE_TOTALS_KEYS:
            record = venue_totals[venue]
            if not isinstance(record, Mapping):
                raise DataValidationError(f"{team_name}.venue_totals.{venue} must be an object")
            _require_exact_keys(record, VENUE_TOTAL_KEYS, f"{team_name}.venue_totals.{venue}")
            local_venue[venue] = {field: _is_int(record[field], label=f"{team_name}.{venue}.{field}") for field in VENUE_TOTAL_KEYS}
        if local_venue["home"]["matches"] + local_venue["away"]["matches"] != total:
            raise DataValidationError(f"{team_name} venue match counts do not sum to total_appearances")
        if local_venue["home"]["goals_for_sum"] + local_venue["away"]["goals_for_sum"] != gf_sum:
            raise DataValidationError(f"{team_name} venue goals_for sums do not match team total")
        if local_venue["home"]["goals_against_sum"] + local_venue["away"]["goals_against_sum"] != ga_sum:
            raise DataValidationError(f"{team_name} venue goals_against sums do not match team total")

        appearances = team["appearances"]
        if not isinstance(appearances, list):
            raise DataValidationError(f"{team_name}.appearances must be an array")
        expected_retained = min(total, max(RESULT_FACTOR_WINDOWS))
        if len(appearances) != expected_retained:
            raise DataValidationError(
                f"{team_name}.appearances must retain exactly the most recent {expected_retained} entries"
            )
        previous: tuple[pd.Timestamp, str] | None = None
        seen_ids: set[str] = set()
        tail = {"goals_for_sum": 0, "goals_against_sum": 0,
                "venue": {venue: {field: 0 for field in VENUE_TOTAL_KEYS} for venue in VENUE_TOTALS_KEYS}}
        for index, appearance in enumerate(appearances):
            batch_time, match_id = _validate_appearance(appearance, as_of=state_as_of, label=f"{team_name}.appearances[{index}]")
            if previous is not None and (batch_time, match_id) < previous:
                raise DataValidationError(f"{team_name}.appearances are not in stable chronology order")
            if match_id in seen_ids:
                raise DataValidationError(f"{team_name}.appearances contain duplicate match_id values")
            previous = (batch_time, match_id)
            seen_ids.add(match_id)
            venue = appearance["venue"]
            tail["goals_for_sum"] += appearance["goals_for"]
            tail["goals_against_sum"] += appearance["goals_against"]
            tail["venue"][venue]["matches"] += 1
            tail["venue"][venue]["goals_for_sum"] += appearance["goals_for"]
            tail["venue"][venue]["goals_against_sum"] += appearance["goals_against"]
        if tail["goals_for_sum"] > gf_sum or tail["goals_against_sum"] > ga_sum:
            raise DataValidationError(f"{team_name}.appearances exceed cumulative goal totals")
        for venue in VENUE_TOTALS_KEYS:
            for field in VENUE_TOTAL_KEYS:
                if tail["venue"][venue][field] > local_venue[venue][field]:
                    raise DataValidationError(f"{team_name}.appearances exceed cumulative {venue} {field}")
        aggregate["total_appearances"] += total
        aggregate["goals_for_sum"] += gf_sum
        aggregate["goals_against_sum"] += ga_sum
        for venue in VENUE_TOTALS_KEYS:
            for field in VENUE_TOTAL_KEYS:
                aggregate["venue"][venue][field] += local_venue[venue][field]

    if aggregate["total_appearances"] != rows_used * 2:
        raise DataValidationError("team appearance counts do not equal twice rows_used")
    if aggregate["goals_for_sum"] != normalized_priors["home"]["goals_for_sum"] + normalized_priors["away"]["goals_for_sum"]:
        raise DataValidationError("team goals_for sums do not match league score totals")
    if aggregate["goals_against_sum"] != normalized_priors["home"]["goals_against_sum"] + normalized_priors["away"]["goals_against_sum"]:
        raise DataValidationError("team goals_against sums do not match league score totals")
    for venue in VENUE_TOTALS_KEYS:
        for field in VENUE_TOTAL_KEYS:
            if aggregate["venue"][venue][field] != normalized_priors[venue][field]:
                raise DataValidationError(f"team cumulative {venue} {field} does not match league totals")
    expected_status = "needs_review" if rows_used < RESULT_FACTOR_MINIMUM_SAMPLE else "ready"
    if manifest["status"] != expected_status:
        raise DataValidationError("factor manifest status does not match rows_used")
    return state


def load_result_factor_snapshot(snapshot_manifest_path: str | Path) -> dict[str, Any]:
    """Load a bounded JSON snapshot; recorded history paths are never opened."""

    raw_path = Path(snapshot_manifest_path).expanduser()
    if not raw_path.is_file():
        raise FileNotFoundError(f"factor manifest does not exist: {raw_path}")
    manifest_path = raw_path.resolve(strict=True)
    if manifest_path.name != RESULT_FACTOR_MANIFEST_FILENAME:
        raise DataValidationError(f"factor manifest must be named {RESULT_FACTOR_MANIFEST_FILENAME}")
    manifest = _read_json_object(manifest_path, "factor manifest")
    _require_exact_keys(manifest, _MANIFEST_KEYS, "factor manifest")
    if manifest["schema_version"] != RESULT_FACTOR_MANIFEST_SCHEMA_VERSION:
        raise DataValidationError(f"factor manifest schema_version must be {RESULT_FACTOR_MANIFEST_SCHEMA_VERSION}")
    _validate_generated_at(manifest["generated_at"])
    manifest_as_of = _validate_as_of(manifest["as_of"], "factor manifest as_of")
    if not isinstance(manifest["snapshot_id"], str) or manifest["snapshot_id"] != manifest_path.parent.name:
        raise DataValidationError("factor snapshot_id must match its directory name")
    if not isinstance(manifest["status"], str) or manifest["status"] not in _VALID_STATUSES:
        raise DataValidationError("factor manifest status must be ready or needs_review")
    if not isinstance(manifest["reasons"], list) or any(not isinstance(item, str) for item in manifest["reasons"]):
        raise DataValidationError("factor manifest reasons must be an array of strings")
    if manifest["coverage_status"] != RESULT_FACTOR_COVERAGE_STATUS:
        raise DataValidationError("factor manifest coverage_status is invalid")
    if manifest["experimental"] is not True or manifest["probability_adjustment_applied"] is not False:
        raise DataValidationError("factor manifest must be experimental and must not adjust probabilities")

    history = manifest["history"]
    if not isinstance(history, Mapping):
        raise DataValidationError("factor manifest history must be an object")
    _require_exact_keys(history, _HISTORY_KEYS, "factor manifest history")
    for field in ("path", "manifest_path"):
        if not isinstance(history[field], str) or not history[field].strip():
            raise DataValidationError(f"factor manifest history.{field} must be a provenance string")
    for field in ("sha256", "manifest_sha256", "output_sha256"):
        if not _valid_sha256(history[field]):
            raise DataValidationError(f"factor manifest history.{field} must be a SHA-256")
    if history["sha256"] != history["output_sha256"]:
        raise DataValidationError("factor manifest history output hash does not match history hash")
    if not isinstance(history["manifest_schema_version"], str) or history["manifest_schema_version"] not in _HISTORY_MANIFEST_SCHEMAS:
        raise DataValidationError("factor manifest history-manifest schema is unsupported")
    if history["dataset_version"] is not None and (not isinstance(history["dataset_version"], str) or not history["dataset_version"].strip()):
        raise DataValidationError("factor manifest dataset_version must be a string or null")
    if not isinstance(history["data_cutoff"], str) or not history["data_cutoff"].strip():
        raise DataValidationError("factor manifest data_cutoff must be a non-empty string")
    _valid_iso_timestamp(history["data_cutoff"], "factor manifest history.data_cutoff")
    rows_total = _is_int(history["rows_total"], label="factor manifest history.rows_total")
    rows_used = _is_int(history["rows_used"], label="factor manifest history.rows_used")
    if rows_total == 0 or rows_used > rows_total:
        raise DataValidationError("factor manifest history row counts are out of range")
    used_cutoff = history["used_data_cutoff"]
    if used_cutoff is not None and (not isinstance(used_cutoff, str) or not used_cutoff.strip()):
        raise DataValidationError("factor manifest used_data_cutoff must be null or a string")
    if (rows_used == 0) != (used_cutoff is None):
        raise DataValidationError("factor manifest rows_used and used_data_cutoff are inconsistent")
    if used_cutoff is not None:
        used_ts = _valid_iso_timestamp(used_cutoff, "factor manifest history.used_data_cutoff")
        as_of_ts, _ = _parse_boundary(manifest_as_of["value"], "snapshot as_of")
        if not used_ts < as_of_ts:
            raise DataValidationError("factor manifest used_data_cutoff must be strictly before as_of")
    _validate_config(manifest["config"], manifest["config_sha256"], "factor manifest config")

    snapshot_root = manifest_path.parent.resolve(strict=True)
    state_record = manifest["state"]
    if not isinstance(state_record, Mapping):
        raise DataValidationError("factor manifest state descriptor must be an object")
    _require_exact_keys(state_record, _STATE_DESCRIPTOR_KEYS, "factor manifest state")
    if state_record["schema_version"] != RESULT_FACTOR_STATE_SCHEMA_VERSION:
        raise DataValidationError("factor state descriptor schema_version is unsupported")
    state_path = _validate_hash_and_bytes(
        state_record, filename=RESULT_FACTOR_STATE_FILENAME,
        expected_keys=_STATE_DESCRIPTOR_KEYS, base=snapshot_root, label="factor state",
    )
    report_path = _validate_hash_and_bytes(
        manifest["report"], filename=RESULT_FACTOR_BUILD_REPORT_FILENAME,
        expected_keys=_REPORT_DESCRIPTOR_KEYS, base=snapshot_root, label="factor build report",
    )
    inventory = manifest["inventory"]
    if not isinstance(inventory, Mapping):
        raise DataValidationError("factor manifest inventory must be an object")
    _require_exact_keys(inventory, _INVENTORY_KEYS, "factor manifest inventory")
    inventory_manifest_bytes = _is_int(inventory["manifest_bytes"], label="inventory.manifest_bytes")
    inventory_total = _is_int(inventory["total_bytes"], label="inventory.total_bytes")
    actual_manifest_bytes = manifest_path.stat().st_size
    actual_total = actual_manifest_bytes + state_path.stat().st_size + report_path.stat().st_size
    if inventory_manifest_bytes != actual_manifest_bytes or inventory_total != actual_total:
        raise DataValidationError("factor snapshot byte inventory does not match disk")
    if actual_total > RESULT_FACTOR_SNAPSHOT_MAX_BYTES:
        raise DataValidationError("factor snapshot exceeds the 5 MiB size limit")
    expected_names = {RESULT_FACTOR_MANIFEST_FILENAME, RESULT_FACTOR_STATE_FILENAME, RESULT_FACTOR_BUILD_REPORT_FILENAME}
    if {item.name for item in snapshot_root.iterdir()} != expected_names:
        raise DataValidationError("factor snapshot directory must contain only manifest, state, and build report")

    state = _read_json_object(state_path, "factor state")
    _validate_state(state, manifest)
    if state["as_of"] != manifest_as_of:
        raise DataValidationError("factor state and manifest as_of do not match")
    return {
        "manifest_path": str(manifest_path),
        "snapshot_dir": str(snapshot_root),
        "manifest": manifest,
        "manifest_sha256": file_sha256(manifest_path),
        "state_path": str(state_path),
        "state_sha256": state_record["sha256"],
        "report_path": str(report_path),
        "report_sha256": manifest["report"]["sha256"],
        "state": state,
    }


def _manifest_bytes_with_inventory(manifest: dict[str, Any], state_bytes: int, report_bytes: int) -> bytes:
    manifest["inventory"] = {"manifest_bytes": 0, "total_bytes": state_bytes + report_bytes}
    for _ in range(12):
        data = canonical_json_bytes(manifest)
        current = {"manifest_bytes": len(data), "total_bytes": state_bytes + report_bytes + len(data)}
        if manifest["inventory"] == current:
            return data
        manifest["inventory"] = current
    raise DataValidationError("factor manifest byte inventory did not stabilize")


def _write_blocked_build(
    artifact_dir: Path, error: BaseException, stage: str, *,
    as_of: Mapping[str, Any] | None = None, rows_total: int | None = None,
    rows_used: int | None = None, used_data_cutoff: str | None = None,
    team_count: int | None = None, state_sha256: str | None = None,
) -> dict[str, Any]:
    reasons = [f"{stage}: {error}"]
    report_path = artifact_dir / RESULT_FACTOR_BUILD_REPORT_FILENAME
    report_path.write_text(render_build_report(
        status="blocked", snapshot_dir=artifact_dir, as_of=as_of, rows_total=rows_total,
        rows_used=rows_used, used_data_cutoff=used_data_cutoff, team_count=team_count,
        state_sha256=state_sha256, reasons=reasons,
    ), encoding="utf-8")
    failure_path = artifact_dir / "factor_build_failure.json"
    _write_json_bytes({
        "schema_version": RESULT_FACTOR_MANIFEST_SCHEMA_VERSION, "status": "blocked",
        "failure_stage": stage, "reasons": reasons, "partial_dir": str(artifact_dir),
        "experimental": True, "probability_adjustment_applied": False,
    }, failure_path)
    # If a late write-time check failed after the manifest was emitted, make its
    # status visibly blocked. The unexpected extra failure file also prevents load.
    manifest_path = artifact_dir / RESULT_FACTOR_MANIFEST_FILENAME
    if manifest_path.is_file():
        try:
            partial = _read_json_object(manifest_path, "partial factor manifest")
            partial["status"] = "blocked"
            partial["reasons"] = reasons
            manifest_path.write_bytes(canonical_json_bytes(partial))
        except (OSError, DataValidationError, TypeError, ValueError):
            pass
    return {
        "status": "blocked", "reasons": reasons, "failure_stage": stage,
        "output_dir": str(artifact_dir), "manifest_path": None, "state_path": None,
        "report_path": str(report_path), "failure_path": str(failure_path),
        "rows_total": rows_total, "rows_used": rows_used, "used_data_cutoff": used_data_cutoff,
    }


def build_result_factor_snapshot(
    history_path: str | Path,
    history_manifest_path: str | Path,
    as_of: str | pd.Timestamp,
    *,
    output_root: str | Path,
) -> dict[str, Any]:
    """Build a read-only-history, immutable, bounded M13A result-factor snapshot."""

    history_file = Path(history_path).expanduser().resolve()
    source_manifest_file = Path(history_manifest_path).expanduser().resolve()
    root = _resolve_output_root(output_root, history_file, source_manifest_file)
    artifact_dir = _new_artifact_dir(root, "m13_result_factor_snapshot")
    stage = "history_load"
    parsed_as_of: dict[str, Any] | None = None
    rows_total: int | None = None
    rows_used: int | None = None
    used_cutoff: str | None = None
    team_count: int | None = None
    state_hash: str | None = None
    try:
        history_hash_start = file_sha256(history_file)
        source_manifest_hash_start = file_sha256(source_manifest_file)
        history = load_matches(history_file)
        verified_hash, source_manifest, _observed_cutoff = history_manifest_evidence(
            history_file, source_manifest_file, history
        )
        if verified_hash != history_hash_start or file_sha256(source_manifest_file) != source_manifest_hash_start:
            raise DataValidationError("history or history manifest changed while being loaded")
        stage = "as_of_filter"
        timeline, eligible, parsed_as_of, _cutoff = _history_before_as_of(history, as_of)
        rows_total, rows_used = int(len(timeline)), int(len(eligible))
        stage = "factor_calculation"
        state = build_result_factor_state(eligible, as_of=parsed_as_of, rows_total=rows_total)
        used_cutoff = state["used_data_cutoff"]
        team_count = len(state["teams"])
        state_bytes = canonical_json_bytes(state)
        state_hash = hashlib.sha256(state_bytes).hexdigest()
        status = "needs_review" if rows_used < RESULT_FACTOR_MINIMUM_SAMPLE else "ready"
        reasons = []
        if rows_used == 0:
            reasons.append("no_eligible_history")
        elif rows_used < RESULT_FACTOR_MINIMUM_SAMPLE:
            reasons.append("small_history_sample")
        report_bytes = render_build_report(
            status=status, snapshot_dir=artifact_dir, as_of=parsed_as_of,
            rows_total=rows_total, rows_used=rows_used, used_data_cutoff=used_cutoff,
            team_count=team_count, state_sha256=state_hash, reasons=reasons,
        ).encode("utf-8")
        if (file_sha256(history_file), file_sha256(source_manifest_file)) != (history_hash_start, source_manifest_hash_start):
            raise DataValidationError("history or history manifest changed during snapshot build")

        stage = "snapshot_write"
        manifest: dict[str, Any] = {
            "schema_version": RESULT_FACTOR_MANIFEST_SCHEMA_VERSION,
            "snapshot_id": artifact_dir.name,
            "generated_at": _utc_now(),
            "as_of": parsed_as_of,
            "history": {
                "path": str(history_file),
                "sha256": history_hash_start,
                "manifest_path": str(source_manifest_file),
                "manifest_sha256": source_manifest_hash_start,
                "manifest_schema_version": source_manifest["schema_version"],
                "dataset_version": source_manifest.get("dataset_version"),
                "data_cutoff": source_manifest["data_cutoff"],
                "rows_total": rows_total,
                "rows_used": rows_used,
                "used_data_cutoff": used_cutoff,
                "output_sha256": history_hash_start,
            },
            "config": deepcopy(RESULT_FACTOR_CONFIG),
            "config_sha256": RESULT_FACTOR_CONFIG_SHA256,
            "state": {
                "filename": RESULT_FACTOR_STATE_FILENAME,
                "schema_version": RESULT_FACTOR_STATE_SCHEMA_VERSION,
                "sha256": state_hash,
                "bytes": len(state_bytes),
            },
            "report": {
                "filename": RESULT_FACTOR_BUILD_REPORT_FILENAME,
                "sha256": hashlib.sha256(report_bytes).hexdigest(),
                "bytes": len(report_bytes),
            },
            "inventory": {"manifest_bytes": 0, "total_bytes": 0},
            "status": status,
            "reasons": reasons,
            "coverage_status": RESULT_FACTOR_COVERAGE_STATUS,
            "experimental": True,
            "probability_adjustment_applied": False,
        }
        manifest_bytes = _manifest_bytes_with_inventory(manifest, len(state_bytes), len(report_bytes))
        if manifest["inventory"]["total_bytes"] > RESULT_FACTOR_SNAPSHOT_MAX_BYTES:
            raise DataValidationError(f"factor snapshot exceeds 5 MiB ({manifest['inventory']['total_bytes']} bytes)")
        state_path = artifact_dir / RESULT_FACTOR_STATE_FILENAME
        report_path = artifact_dir / RESULT_FACTOR_BUILD_REPORT_FILENAME
        manifest_path = artifact_dir / RESULT_FACTOR_MANIFEST_FILENAME
        state_path.write_bytes(state_bytes)
        report_path.write_bytes(report_bytes)
        manifest_path.write_bytes(manifest_bytes)  # Manifest is written last.
        load_result_factor_snapshot(manifest_path)
        if (file_sha256(history_file), file_sha256(source_manifest_file)) != (history_hash_start, source_manifest_hash_start):
            raise DataValidationError("history or history manifest changed by snapshot completion")
        if manifest_path.stat().st_size + state_path.stat().st_size + report_path.stat().st_size > RESULT_FACTOR_SNAPSHOT_MAX_BYTES:
            raise DataValidationError("written factor snapshot exceeds the 5 MiB size limit")
        return {
            "status": status, "reasons": reasons, "output_dir": str(artifact_dir),
            "manifest_path": str(manifest_path), "state_path": str(state_path),
            "report_path": str(report_path), "manifest_sha256": file_sha256(manifest_path),
            "state_sha256": state_hash, "manifest": manifest, "rows_total": rows_total,
            "rows_used": rows_used, "used_data_cutoff": used_cutoff, "team_count": team_count,
            "coverage_status": RESULT_FACTOR_COVERAGE_STATUS, "experimental": True,
            "probability_adjustment_applied": False,
        }
    except (DataValidationError, FileNotFoundError, OSError, TypeError, ValueError) as exc:
        return _write_blocked_build(
            artifact_dir, exc, stage, as_of=parsed_as_of, rows_total=rows_total,
            rows_used=rows_used, used_data_cutoff=used_cutoff, team_count=team_count,
            state_sha256=state_hash,
        )


def _blank_query_payload(
    *, status: str, snapshot: Mapping[str, Any] | None, fixtures: Mapping[str, Any] | None,
    results: list[dict[str, Any]] | None = None, reasons: Sequence[str] = (),
    failure_stage: str | None = None,
) -> dict[str, Any]:
    payload = {
        "schema_version": RESULT_FACTOR_QUERY_SCHEMA_VERSION,
        "generated_at": _utc_now(),
        "status": status,
        "as_of": dict(snapshot["as_of"]) if snapshot is not None else None,
        "snapshot": dict(snapshot) if snapshot is not None else None,
        "fixtures": dict(fixtures) if fixtures is not None else None,
        "fixture_count": len(results or []),
        "results": list(results or []),
        "reasons": list(reasons),
        "coverage_status": RESULT_FACTOR_COVERAGE_STATUS,
        "experimental": True,
        "probability_adjustment_applied": False,
        "validation": {
            "score_free_fixture_schema": f"{FIXTURE_SCHEMA_VERSION}-score-free",
            "history_read_during_query": False,
            "history_manifest_read_during_query": False,
            "current_pointer_read": False,
            "results_used": False,
            "network_accessed": False,
            "snapshot_mutated": False,
        },
        "failure_stage": failure_stage,
    }
    _require_exact_keys(payload, _QUERY_KEYS, "factor query")
    return payload


def _write_blocked_query(
    artifact_dir: Path, error: BaseException, stage: str, *,
    snapshot: Mapping[str, Any] | None = None, fixtures: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    reason = f"{stage}: {error}"
    payload = _blank_query_payload(
        status="blocked", snapshot=snapshot, fixtures=fixtures,
        reasons=[reason], failure_stage=stage,
    )
    query_path = artifact_dir / RESULT_FACTOR_QUERY_FILENAME
    report_path = artifact_dir / RESULT_FACTOR_QUERY_REPORT_FILENAME
    _write_json_bytes(payload, query_path)
    report_path.write_text(render_query_report(payload, None), encoding="utf-8")
    return {
        "status": "blocked", "reasons": [reason], "failure_stage": stage,
        "output_dir": str(artifact_dir), "query_path": str(query_path),
        "csv_path": None, "report_path": str(report_path), "payload": payload,
    }


def _fixture_review_reasons(
    fixture_id: str, home_team: str, away_team: str,
    home: Mapping[str, Any], away: Mapping[str, Any],
) -> list[str]:
    reasons: list[str] = []
    for side, name, features in (("home", home_team, home), ("away", away_team, away)):
        if features["unknown_team"]:
            reasons.append(f"{fixture_id}:{side}_unknown_team:{name}")
        elif features["total_appearances"] < RESULT_FACTOR_MINIMUM_SAMPLE:
            reasons.append(f"{fixture_id}:{side}_small_sample:{features['total_appearances']}")
    return reasons


def query_result_factors(
    snapshot_manifest_path: str | Path,
    fixtures_path: str | Path,
    *,
    output_root: str | Path,
) -> dict[str, Any]:
    """Query one explicit snapshot and score-free fixtures without reading source history."""

    manifest_file = Path(snapshot_manifest_path).expanduser().resolve()
    fixtures_file = Path(fixtures_path).expanduser().resolve()
    root = _resolve_output_root(output_root, manifest_file, fixtures_file)
    artifact_dir = _new_artifact_dir(root, "m13_result_factor_query")
    stage = "snapshot_load"
    fixture_evidence: dict[str, Any] | None = None
    snapshot_evidence: dict[str, Any] | None = None
    try:
        manifest_hash_start = file_sha256(manifest_file)
        loaded = load_result_factor_snapshot(manifest_file)
        manifest, state = loaded["manifest"], loaded["state"]
        snapshot_evidence = {
            "snapshot_id": manifest["snapshot_id"],
            "manifest_path": str(manifest_file),
            "manifest_sha256": loaded["manifest_sha256"],
            "state_sha256": loaded["state_sha256"],
            "as_of": manifest["as_of"],
            "history_sha256": manifest["history"]["sha256"],
            "history_manifest_sha256": manifest["history"]["manifest_sha256"],
        }
        stage = "fixture_load"
        fixtures_hash_start = file_sha256(fixtures_file)
        fixtures = load_fixtures(fixtures_file)
        fixture_evidence = {
            "path": str(fixtures_file), "sha256": fixtures_hash_start,
            "rows": int(len(fixtures)), "schema_version": f"{FIXTURE_SCHEMA_VERSION}-score-free",
        }
        as_of = manifest["as_of"]
        _, date_only = _parse_boundary(as_of["value"], "snapshot as_of")
        as_of_timestamp = pd.Timestamp(as_of["timestamp"])
        cutoff = as_of_timestamp.normalize() if date_only else as_of_timestamp
        stage = "fixture_time_validation"
        _fixtures_at_or_after_as_of(fixtures, as_of, cutoff)

        stage = "factor_query"
        results: list[dict[str, Any]] = []
        output_rows: list[dict[str, Any]] = []
        reasons: list[str] = []
        for row in fixtures.to_dict(orient="records"):
            fixture_id = str(row["fixture_id"])
            home_team, away_team = str(row["home_team"]), str(row["away_team"])
            home = fixture_team_factors(state, home_team, "home", row["date"])
            away = fixture_team_factors(state, away_team, "away", row["date"])
            row_reasons = _fixture_review_reasons(fixture_id, home_team, away_team, home, away)
            row_status = "needs_review" if row_reasons else "ready"
            reasons.extend(row_reasons)
            identity = {key: str(row[key]) for key in ("fixture_id", "date", "season", "home_team", "away_team")}
            results.append({
                **identity, "status": row_status, "reasons": row_reasons,
                "home": home, "away": away, "probability_adjustment_applied": False,
            })
            output_rows.append(flatten_fixture_factors(identity, home, away))
        status = "needs_review" if reasons or state["rows_used"] == 0 else "ready"
        if state["rows_used"] == 0:
            reasons.insert(0, "no_eligible_history")
        payload = _blank_query_payload(
            status=status, snapshot=snapshot_evidence, fixtures=fixture_evidence,
            results=results, reasons=list(dict.fromkeys(reasons)),
        )
        frame = pd.DataFrame(output_rows, columns=RESULT_FACTOR_QUERY_COLUMNS)
        csv_path = artifact_dir / RESULT_FACTOR_QUERY_CSV_FILENAME
        report_path = artifact_dir / RESULT_FACTOR_QUERY_REPORT_FILENAME
        query_path = artifact_dir / RESULT_FACTOR_QUERY_FILENAME
        stage = "output_write"
        frame.to_csv(csv_path, index=False, encoding="utf-8-sig")
        report_path.write_text(render_query_report(payload, str(csv_path)), encoding="utf-8")
        if file_sha256(fixtures_file) != fixtures_hash_start:
            raise DataValidationError("fixtures changed during result-factor query")
        if file_sha256(manifest_file) != manifest_hash_start:
            raise DataValidationError("factor manifest changed during result-factor query")
        if file_sha256(loaded["state_path"]) != loaded["state_sha256"]:
            raise DataValidationError("factor state changed during result-factor query")
        if file_sha256(loaded["report_path"]) != loaded["report_sha256"]:
            raise DataValidationError("factor build report changed during result-factor query")
        _write_json_bytes(payload, query_path)
        return {
            "status": status, "reasons": payload["reasons"], "output_dir": str(artifact_dir),
            "query_path": str(query_path), "csv_path": str(csv_path), "report_path": str(report_path),
            "fixture_count": len(output_rows), "payload": payload, "results": results,
            "coverage_status": RESULT_FACTOR_COVERAGE_STATUS, "experimental": True,
            "probability_adjustment_applied": False,
        }
    except (DataValidationError, FileNotFoundError, OSError, TypeError, ValueError) as exc:
        return _write_blocked_query(
            artifact_dir, exc, stage, snapshot=snapshot_evidence, fixtures=fixture_evidence,
        )


__all__ = ["build_result_factor_snapshot", "load_result_factor_snapshot", "query_result_factors"]
