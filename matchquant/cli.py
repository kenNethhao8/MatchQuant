"""Command-line entry point for MatchQuant M1-M12."""

from __future__ import annotations

import argparse
import importlib.metadata
import importlib.resources
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

from .backtest import run_models
from .demo_export import export_synthetic_demo
from .epl import DEFAULT_EPL_SEASONS, download_epl, prepare_epl
from .epl_fixtures import prepare_epl_fixtures
from .epl_local_stage import stage_local_epl_source
from .epl_sync import sync_epl
from .engine_snapshot import build_engine_snapshot, predict_from_engine_snapshot
from .engine_snapshot_status import check_engine_snapshot_status
from .forecast import load_model_config, run_forecast
from .forecast_session import run_forecast_session
from .postmatch_session import run_postmatch_session
from .io import file_sha256, load_matches, write_json, write_predictions
from .m3 import dataset_provenance, new_run_dir
from .models import MODEL_NAMES
from .schema import DataValidationError
from .selection import SELECTION_SCHEMA_VERSION, load_candidates, parse_validation_folds, run_nested_selection
from .settlement import settle_forecast_run
from .release_verify import verify_release
from .runtime_bundle import build_runtime_bundle
from .runtime_bundle_validation import verify_runtime_bundle
from .snapshot_run_verify import verify_snapshot_run
from .snapshot_session import run_snapshot_session


def default_m3_candidate_config() -> Path:
    """Path of the packaged default M3 candidate configuration.

    The resource ships inside the wheel (``matchquant/resources/``), so the
    default also works from a real install outside the source checkout; an
    explicit ``--candidate-config`` keeps its old semantics and overrides it.
    """

    resource = importlib.resources.files("matchquant") / "resources" / "m3_candidates.json"
    with importlib.resources.as_file(resource) as path:
        return Path(path)


def _version_text() -> str:
    try:
        package_version = importlib.metadata.version("matchquant")
    except importlib.metadata.PackageNotFoundError:
        package_version = "0.0.0+uninstalled"
    return (
        f"matchquant {package_version} (baseline-v0.1 research build; research software, "
        "not a live prediction product)"
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="matchquant", description="Reproducible football 1X2 research engine")
    parser.add_argument(
        "--version",
        action="version",
        version=_version_text(),
        help="show the installed package version (metadata-driven) and the baseline-v0.1 research profile, then exit",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    run = subparsers.add_parser(
        "run",
        help="run an offline walk-forward backtest (online means batch-by-batch model-state updating, not the network)",
    )
    run.add_argument("--config", type=Path, help="optional JSON configuration")
    run.add_argument("--input", type=Path, help="canonical match CSV")
    run.add_argument("--test-start", help="first test date/time; date-only starts are calendar-day aligned")
    run.add_argument("--test-end", help="optional exclusive test end date/time")
    run.add_argument("--model", choices=[*MODEL_NAMES, "all"], help="model or all three")
    run.add_argument("--season-mode", choices=["carry", "reset"], help="carry or reset online state across seasons")
    run.add_argument("--output-dir", type=Path, help="directory for predictions, metrics, and run config")
    select = subparsers.add_parser(
        "select",
        help="M3 strict chronological model selection with calibration diagnostics (diagnostics only, not calibrated probabilities)",
    )
    select.add_argument("--input", type=Path, required=True, help="local canonical match CSV")
    select.add_argument(
        "--candidate-config",
        "--candidates",
        type=Path,
        default=default_m3_candidate_config(),
        help="candidate configuration JSON (default: the packaged m3_candidates.json resource)",
    )
    select.add_argument(
        "--validation-fold",
        dest="validation_folds",
        action="append",
        required=True,
        metavar="START..END",
        help="repeat for chronological, non-overlapping validation periods",
    )
    select.add_argument(
        "--final-test-start",
        "--test-start",
        dest="final_test_start",
        required=True,
        help="first final-test batch; final test is never used for selection",
    )
    select.add_argument("--final-test-end", "--test-end", dest="final_test_end")
    select.add_argument("--season-mode", choices=["carry", "reset"], default="carry")
    select.add_argument("--calibration-bins", type=int, default=10)
    select.add_argument("--output-root", type=Path, default=Path(r"D:\Codex\matchquant_runs"))
    forecast = subparsers.add_parser(
        "forecast",
        help="run one offline local fixture forecast (M4 research snapshot; not live data, not an Alpha/factor engine)",
    )
    forecast.add_argument("--history", type=Path, required=True, help="local canonical results CSV")
    forecast.add_argument("--fixtures", type=Path, required=True, help="local score-free pending fixture CSV")
    forecast.add_argument("--as-of", required=True, help="snapshot date/time; date-only excludes that full day")
    selection = forecast.add_mutually_exclusive_group(required=True)
    selection.add_argument(
        "--selection-run-config", "--selection-config", "--selection-run", "--m3-run-config",
        dest="selection_run_config", type=Path,
        help="explicit M3 selection run_config.json",
    )
    selection.add_argument("--model", choices=MODEL_NAMES, help="explicit model name; requires --model-config")
    forecast.add_argument("--model-config", "--config", type=Path, help="JSON config for the explicit --model")
    forecast.add_argument("--season-mode", choices=["carry", "reset"], help="online state handling across seasons")
    forecast.add_argument(
        "--output-root", "--output-dir", dest="output_root", type=Path,
        default=Path(r"D:\Codex\matchquant_runs"), help="root for a new timestamped forecast directory",
    )
    snapshot_build = subparsers.add_parser(
        "engine-snapshot-build",
        help="build an offline M11 model-state snapshot from explicit history evidence",
    )
    snapshot_build.add_argument("--history", type=Path, required=True, help="local canonical results CSV")
    snapshot_build.add_argument("--history-manifest", type=Path, required=True, help="local processed history manifest JSON")
    snapshot_build.add_argument("--as-of", required=True, help="snapshot date/time; date-only excludes that full day")
    snapshot_selection = snapshot_build.add_mutually_exclusive_group(required=True)
    snapshot_selection.add_argument(
        "--selection-run-config", "--selection-config", dest="selection_run_config", type=Path,
        help="explicit M3 selection run_config.json",
    )
    snapshot_selection.add_argument(
        "--model", action="append", choices=MODEL_NAMES, help="repeat for each explicit model",
    )
    snapshot_build.add_argument(
        "--model-config", action="append", type=Path,
        help="repeat in the same order as --model; each is a JSON model config",
    )
    snapshot_build.add_argument("--season-mode", choices=["carry", "reset"], help="online state handling across seasons")
    snapshot_build.add_argument(
        "--snapshot-root", type=Path, default=Path(r"D:\Codex\matchquant_data\engine_snapshots"),
        help="root for a new immutable M11 snapshot directory",
    )
    snapshot_predict = subparsers.add_parser(
        "engine-snapshot-predict",
        help="predict from an explicit M11 snapshot without loading history",
    )
    snapshot_predict.add_argument("--snapshot-manifest", type=Path, required=True, help="explicit M11 snapshot_manifest.json")
    snapshot_predict.add_argument("--fixtures", type=Path, required=True, help="local score-free pending fixture CSV")
    snapshot_predict.add_argument(
        "--output-root", type=Path, default=Path(r"D:\Codex\matchquant_runs"),
        help="root for a new timestamped M11 prediction directory",
    )
    snapshot_status = subparsers.add_parser(
        "engine-snapshot-status",
        help="audit one explicit M11 snapshot against one explicit M2.1 current pointer",
    )
    snapshot_status.add_argument("--snapshot-manifest", type=Path, required=True, help="explicit M11 snapshot_manifest.json")
    snapshot_status.add_argument("--current-pointer", type=Path, required=True, help="explicit M2.1 epl_current.json")
    snapshot_status.add_argument(
        "--output-root", type=Path, default=Path(r"D:\Codex\matchquant_runs"),
        help="root for a new timestamped M12 status audit directory",
    )
    snapshot_session = subparsers.add_parser(
        "snapshot-session",
        help="run one offline single-model M11 snapshot prediction session with a read-only audit",
    )
    session_source = snapshot_session.add_mutually_exclusive_group(required=True)
    session_source.add_argument(
        "--snapshot-manifest", type=Path, default=None,
        help="explicit M11 snapshot_manifest.json (bare mode)",
    )
    session_source.add_argument(
        "--bundle-manifest", type=Path, default=None,
        help="explicit runtime bundle_manifest.json (B04 bundle mode; mutually exclusive with --snapshot-manifest)",
    )
    snapshot_session.add_argument("--fixtures", type=Path, required=True, help="explicit score-free fixture CSV")
    snapshot_session.add_argument("--mode", choices=["replay", "research"], default="replay", help="replay (retrospective) or research (pending coverage review)")
    snapshot_session.add_argument(
        "--output-root", type=Path, required=True,
        help="root for the new unique session directory (must be outside formal roots, inputs, and the bundle)",
    )
    snapshot_session.add_argument(
        "--issue-evidence", type=Path, default=None,
        help="optional M12 snapshot_status.json (bare mode only; bundle mode reads the bundled record)",
    )
    snapshot_verify = subparsers.add_parser(
        "snapshot-run-verify",
        help="read-only audit of one existing single-model M11 snapshot prediction run",
    )
    snapshot_verify.add_argument("--forecast-run", type=Path, required=True, help="completed single-model M11 snapshot prediction run directory")
    snapshot_verify.add_argument(
        "--output-root", type=Path, required=True,
        help="root for the new unique audit directory (must be outside formal roots, inputs, and the bundle)",
    )
    verify_evidence = snapshot_verify.add_mutually_exclusive_group()
    verify_evidence.add_argument(
        "--issue-evidence", type=Path, default=None,
        help="optional M12 snapshot_status.json (m12-engine-snapshot-status-v1)",
    )
    verify_evidence.add_argument(
        "--bundle-manifest", type=Path, default=None,
        help="explicit runtime bundle_manifest.json (B04 bundle mode; mutually exclusive with --issue-evidence)",
    )
    bundle_build = subparsers.add_parser(
        "runtime-bundle-build",
        help="build one portable single-model runtime bundle from a fresh-at-issue M12 record",
    )
    bundle_build.add_argument("--snapshot-manifest", type=Path, required=True, help="explicit single-model M11 snapshot_manifest.json (read-only input)")
    bundle_build.add_argument(
        "--issue-audit", type=Path, required=True,
        help="explicit original M12 snapshot_status.json (m12-engine-snapshot-status-v1), fresh at issue",
    )
    bundle_build.add_argument(
        "--output-root", type=Path, required=True,
        help="root for the new unique build job directory (must be outside formal roots, inputs, and source code)",
    )
    bundle_verify = subparsers.add_parser(
        "runtime-bundle-verify",
        help="read-only verification of one runtime bundle against its own manifest",
    )
    bundle_verify.add_argument("--bundle-manifest", type=Path, required=True, help="explicit runtime bundle_manifest.json")
    demo_export = subparsers.add_parser(
        "demo-export",
        help=(
            "export the packaged synthetic demo (frequency bundle, fixtures, results, DEMO.md) to one new "
            "directory; offline, no training/prediction/settlement"
        ),
    )
    demo_export.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help=(
            "new directory to create; it must not already exist and cannot be inside the formal "
            "data/runs roots or the package/source tree"
        ),
    )
    download = subparsers.add_parser("epl-download", help="download explicit Football-Data.co.uk EPL seasons")
    download.add_argument(
        "--seasons", nargs="+", default=list(DEFAULT_EPL_SEASONS), help="seasons such as 2015-16 (default: ten complete seasons)"
    )
    download.add_argument("--data-dir", type=Path, default=Path(r"D:\Codex\matchquant_data"))
    download.add_argument("--timeout", type=float, default=30.0)
    prepare = subparsers.add_parser("epl-prepare", help="offline-convert downloaded EPL CSVs to M1 canonical CSV")
    prepare.add_argument("--seasons", nargs="+", help="seasons to convert; default: every season in source manifest")
    prepare.add_argument("--data-dir", type=Path, default=Path(r"D:\Codex\matchquant_data"))
    prepare.add_argument("--output", type=Path, help="canonical CSV path")
    prepare.add_argument("--quality-report", type=Path, help="quality JSON path")
    prepare.add_argument("--processed-manifest", type=Path, help="processed manifest JSON path")
    sync = subparsers.add_parser("epl-sync", help="explicit EPL staging sync and offline diff")
    sync_sources = sync.add_mutually_exclusive_group(required=True)
    sync_sources.add_argument("--seasons", nargs="+", help="seasons to download, such as 2025-26")
    sync_sources.add_argument("--staging-dir", type=Path, help="existing local staging snapshot; performs no network access")
    sync_mode = sync.add_mutually_exclusive_group()
    sync_mode.add_argument("--apply", action="store_true", help="activate only an addition-only candidate")
    sync_mode.add_argument("--dry-run", action="store_true", help="compare without changing the active pointer (default)")
    sync.add_argument("--data-dir", type=Path, default=Path(r"D:\Codex\matchquant_data"))
    sync.add_argument("--baseline", type=Path, help="optional baseline canonical CSV")
    sync.add_argument("--processed-manifest", type=Path, help="optional baseline processed manifest")
    sync.add_argument("--current-pointer", type=Path, help="optional current pointer path")
    sync.add_argument("--in-progress-season", help="explicitly allow an incomplete requested season, such as 2025-26")
    sync.add_argument("--timeout", type=float, default=30.0)
    local_stage = subparsers.add_parser(
        "epl-local-stage",
        help="offline-archive one local E0 result file for the M2.1 sync review",
    )
    local_stage.add_argument("--source-csv", type=Path, required=True, help="explicit local E0-compatible CSV")
    local_stage.add_argument("--season", required=True, help="explicit season such as 2025-26")
    local_stage.add_argument("--source-url", help="optional URL recorded as provenance only; no network access")
    local_stage.add_argument("--source-label", help="optional human-readable provenance label")
    local_stage.add_argument(
        "--upstream-retrieved-at",
        "--source-retrieved-at",
        dest="upstream_retrieved_at",
        help="optional user-provided upstream retrieval time; local archive time is never used here",
    )
    local_stage.add_argument(
        "--data-dir", type=Path, default=Path(r"D:\Codex\matchquant_data"),
        help="root for immutable M7 local staging snapshots",
    )
    fixture_prepare = subparsers.add_parser(
        "epl-fixtures-prepare",
        help="offline-copy and normalize one local Football-Data E0 fixture source",
    )
    fixture_prepare.add_argument("--source-csv", type=Path, required=True, help="explicit local E0-compatible CSV")
    fixture_prepare.add_argument("--season", required=True, help="explicit season such as 2025-26")
    fixture_prepare.add_argument("--source-url", help="optional URL recorded as provenance only; no network access")
    fixture_prepare.add_argument("--source-label", help="optional human-readable provenance label")
    fixture_prepare.add_argument("--baseline-fixtures", type=Path, help="optional local M4 fixture CSV for report-only diff")
    fixture_prepare.add_argument(
        "--data-root", "--data-dir", dest="data_root", type=Path,
        default=Path(r"D:\Codex\matchquant_data"),
        help="root for immutable fixture snapshots",
    )
    settle = subparsers.add_parser(
        "settle", help="offline-settle one M4 or single-model M11 prediction run with local M1 results"
    )
    settle.add_argument(
        "--forecast-run", "--forecast-run-dir", "--forecast-dir", dest="forecast_run",
        type=Path, required=True, help="completed M4 forecast or single-model M11 snapshot prediction run directory",
    )
    settle.add_argument(
        "--results", "--results-csv", dest="results", type=Path, required=True,
        help="explicit local M1 canonical final-results CSV",
    )
    settle.add_argument(
        "--output-root", type=Path, default=Path(r"D:\Codex\matchquant_runs"),
        help="root for a new timestamped M6 settlement directory",
    )
    release = subparsers.add_parser(
        "release-verify", help="offline M8 audit of one explicit M4/M2/M5 evidence chain",
    )
    release.add_argument(
        "--forecast-run", "--forecast-run-dir", dest="forecast_run", type=Path, required=True,
        help="explicit completed M4 forecast run directory",
    )
    release.add_argument("--history", "--history-csv", dest="history", type=Path, required=True, help="explicit M2/M2.1 canonical history CSV")
    release.add_argument("--history-manifest", type=Path, required=True, help="explicit M2/M2.1 processed manifest JSON")
    release.add_argument("--fixture-manifest", "--fixture-snapshot-manifest", dest="fixture_manifest", type=Path, required=True, help="explicit M5 fixture_snapshot_manifest.json")
    release.add_argument("--output-root", type=Path, default=Path(r"D:\Codex\matchquant_runs"), help="root for one new timestamped M8 audit directory")
    session = subparsers.add_parser(
        "forecast-session", help="offline explicit M5 -> M4 -> M8 pre-match session",
    )
    session.add_argument("--history", type=Path, required=True, help="local canonical results CSV")
    session.add_argument("--history-manifest", type=Path, required=True, help="local processed history manifest JSON")
    session.add_argument("--fixture-source", type=Path, required=True, help="local E0-compatible fixture source CSV")
    session.add_argument("--season", required=True, help="explicit season such as 2025-26")
    session.add_argument("--as-of", required=True, help="forecast snapshot date/time")
    session_selection = session.add_mutually_exclusive_group(required=True)
    session_selection.add_argument(
        "--selection-run-config", "--selection-config", "--m3-run-config",
        dest="selection_run_config", type=Path, help="explicit M3 selection run_config.json",
    )
    session_selection.add_argument("--model", choices=MODEL_NAMES, help="explicit model name; requires --model-config")
    session.add_argument("--model-config", "--config", type=Path, help="JSON config for the explicit --model")
    session.add_argument("--season-mode", choices=["carry", "reset"], help="online state handling across seasons")
    session.add_argument("--source-url", help="provenance text only; no network access")
    session.add_argument("--source-label", help="provenance text only; no network access")
    session.add_argument(
        "--fixture-data-root", type=Path, default=Path(r"D:\Codex\matchquant_data"),
        help="root for the new immutable M5 fixture snapshot",
    )
    session.add_argument(
        "--output-root", type=Path, default=Path(r"D:\Codex\matchquant_runs"),
        help="root for unique M4, M8, and M9 artifacts",
    )
    postmatch = subparsers.add_parser(
        "postmatch-session", help="offline explicit M7 -> M2.1 -> M6 post-match session",
    )
    postmatch.add_argument("--session-manifest", type=Path, required=True, help="explicit M9 session_manifest.json")
    postmatch_sources = postmatch.add_mutually_exclusive_group(required=True)
    postmatch_sources.add_argument("--results-source", type=Path, help="local E0 results CSV; stages only")
    postmatch_sources.add_argument("--staging-dir", type=Path, help="existing M7/M2.1 staging snapshot")
    postmatch.add_argument("--season", required=True, help="explicit season such as 2025-26")
    postmatch_mode = postmatch.add_mutually_exclusive_group()
    postmatch_mode.add_argument("--apply", action="store_true", help="apply the reviewed staging snapshot and settle M6")
    postmatch_mode.add_argument("--dry-run", action="store_true", help="stage and compare only (default)")
    postmatch.add_argument("--in-progress-season", help="explicit incomplete-season exception passed to M2.1")
    postmatch.add_argument("--source-url", help="provenance text only for a fresh local stage; no network access")
    postmatch.add_argument("--source-label", help="provenance text only for a fresh local stage")
    postmatch.add_argument("--upstream-retrieved-at", "--source-retrieved-at", dest="upstream_retrieved_at", help="optional timezone-aware upstream retrieval time")
    postmatch.add_argument("--data-root", type=Path, default=Path(r"D:\Codex\matchquant_data"), help="M7/M2.1 data root")
    postmatch.add_argument("--output-root", type=Path, default=Path(r"D:\Codex\matchquant_runs"), help="root for unique M10/M6 artifacts")
    postmatch.add_argument("--current-pointer", type=Path, help="optional active M2.1 pointer for apply coherence")
    return parser


def _read_config(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"could not read config '{path}': {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError("config root must be a JSON object")
    return payload


def _merged_options(args: argparse.Namespace) -> dict[str, Any]:
    options = _read_config(args.config)
    overrides = {
        "input": args.input,
        "test_start": args.test_start,
        "test_end": args.test_end,
        "model": args.model,
        "season_mode": args.season_mode,
        "output_dir": args.output_dir,
    }
    for key, value in overrides.items():
        if value is not None:
            options[key] = value
    if options.get("input") is None:
        raise ValueError("--input or config field 'input' is required")
    if options.get("test_start") is None:
        raise ValueError("--test-start or config field 'test_start' is required")
    options.setdefault("model", "all")
    options.setdefault("season_mode", "carry")
    options.setdefault("output_dir", Path(r"D:\Codex\matchquant_runs\latest"))
    options["input"] = Path(options["input"])
    options["output_dir"] = Path(options["output_dir"])
    options["model_configs"] = options.get("models", options.get("model_configs", {}))
    if not isinstance(options["model_configs"], dict):
        raise ValueError("config field 'models' must be a JSON object")
    return options


def _model_names(value: str) -> list[str]:
    return list(MODEL_NAMES) if value == "all" else [value]


def _run(args: argparse.Namespace) -> int:
    options = _merged_options(args)
    input_path = options["input"].expanduser().resolve()
    output_dir = options["output_dir"].expanduser().resolve()
    matches = load_matches(input_path)
    input_hash = file_sha256(input_path)
    data_cutoff = str(matches["date"].max())
    names = _model_names(str(options["model"]))
    configs = {name: options["model_configs"].get(name, {}) for name in names}
    runs = run_models(
        matches,
        names,
        options["test_start"],
        test_end=options.get("test_end"),
        season_mode=str(options["season_mode"]),
        model_configs=configs,
    )

    prediction_frame = pd.concat([runs[name].predictions for name in names], ignore_index=True)
    prediction_path = write_predictions(prediction_frame, output_dir / "predictions.csv")
    metrics_payload = {
        "schema_version": "m1",
        "input": str(input_path),
        "input_sha256": input_hash,
        "data_cutoff": data_cutoff,
        "test_start": str(options["test_start"]),
        "test_end": options.get("test_end"),
        "season_mode": str(options["season_mode"]),
        "models": {name: runs[name].metrics for name in names},
    }
    metrics_path = write_json(metrics_payload, output_dir / "metrics.json")
    config_payload = {
        "schema_version": "m1",
        "input": str(input_path),
        "input_sha256": input_hash,
        "data_cutoff": data_cutoff,
        "test_start": str(options["test_start"]),
        "test_end": options.get("test_end"),
        "season_mode": str(options["season_mode"]),
        "models": {name: configs[name] for name in names},
        "outputs": {
            "predictions": str(prediction_path),
            "metrics": str(metrics_path),
            "run_config": str(output_dir / "run_config.json"),
        },
    }
    config_path = write_json(config_payload, output_dir / "run_config.json")
    total_test = runs[names[0]].test_matches
    print("MatchQuant M1 complete")
    print(f"models={','.join(names)} test_matches={total_test}")
    print(f"predictions={prediction_path}")
    print(f"metrics={metrics_path}")
    print(f"run_config={config_path}")
    return 0


def _select(args: argparse.Namespace) -> int:
    input_path = args.input.expanduser().resolve()
    candidate_path = args.candidate_config.expanduser().resolve()
    matches = load_matches(input_path)
    candidates, candidate_payload = load_candidates(candidate_path)
    folds = parse_validation_folds(args.validation_folds)
    result = run_nested_selection(
        matches,
        candidates,
        folds,
        args.final_test_start,
        final_test_end=args.final_test_end,
        season_mode=args.season_mode,
        calibration_bins=args.calibration_bins,
    )
    run_dir = new_run_dir(args.output_root.expanduser().resolve())
    input_hash = file_sha256(input_path)
    candidate_hash = file_sha256(candidate_path)
    provenance = dataset_provenance(input_path, matches)
    selected = result["selected_candidate"]
    final_run = result["final_run"]
    predictions_path = write_predictions(final_run.predictions, run_dir / "predictions.csv")
    calibration_path = run_dir / "calibration_reliability.csv"
    result["calibration_table"].to_csv(calibration_path, index=False, encoding="utf-8")
    validation_path = write_json(
        {
            "schema_version": "m3-validation-v1",
            "input": str(input_path),
            "input_sha256": input_hash,
            "data_cutoff": provenance["data_cutoff"],
            "data_version": provenance["dataset_version"],
            "candidate_config": str(candidate_path),
            "candidate_config_sha256": candidate_hash,
            "validation_folds": result["validation_folds"],
            "selection_rule": result["selection_rule"],
            "candidates": result["candidates"],
            "ranking": result["ranking"],
            "selected_candidate": selected,
        },
        run_dir / "validation_metrics.json",
    )
    final_metrics_payload = {
        "schema_version": "m3-final-metrics-v1",
        "input": str(input_path),
        "input_sha256": input_hash,
        "data_cutoff": provenance["data_cutoff"],
        "data_version": provenance["dataset_version"],
        "data_provenance_status": provenance["status"],
        "candidate": {
            "candidate_name": selected["candidate_name"],
            "candidate_order": selected["candidate_order"],
            "model": selected["model"],
            "config": selected["config"],
            "configuration_sha256": selected["configuration_sha256"],
        },
        "test_boundary": result["final_test"],
        "training_matches": int(final_run.training_matches),
        "test_matches": int(final_run.test_matches),
        "metrics": result["final_metrics"],
        "calibration_status": "diagnostic_only_not_calibrated",
        "calibration": {
            **result["calibration_summary"],
            "reliability_table": str(calibration_path),
        },
    }
    final_metrics_path = write_json(final_metrics_payload, run_dir / "final_metrics.json")
    run_config_path = run_dir / "run_config.json"
    run_config = {
        "schema_version": SELECTION_SCHEMA_VERSION,
        "command": "matchquant select",
        "generated_at": pd.Timestamp.utcnow().floor("s").isoformat().replace("+00:00", "Z"),
        "input": str(input_path),
        "input_sha256": input_hash,
        "input_rows": int(len(matches)),
        "data_cutoff": provenance["data_cutoff"],
        "data_version": provenance["dataset_version"],
        "data_provenance": provenance,
        "candidate_config": {
            "path": str(candidate_path),
            "sha256": candidate_hash,
            "payload": candidate_payload,
        },
        "season_mode": args.season_mode,
        "validation_folds": result["validation_folds"],
        "final_test": result["final_test"],
        "selection_rule": result["selection_rule"],
        "validation_metrics": result["candidates"],
        "ranking": result["ranking"],
        "selected_candidate": selected,
        "final_test_metrics": final_metrics_payload["metrics"],
        "calibration": final_metrics_payload["calibration"],
        "outputs": {
            "run_dir": str(run_dir),
            "validation_metrics": str(validation_path),
            "predictions": str(predictions_path),
            "final_metrics": str(final_metrics_path),
            "calibration_reliability": str(calibration_path),
            "run_config": str(run_config_path),
        },
    }
    config_path = write_json(run_config, run_config_path)
    print("MatchQuant M3 selection complete")
    print(f"selected={selected['candidate_name']} model={selected['model']} final_test_matches={final_run.test_matches}")
    print(f"run_dir={run_dir}")
    print(f"validation_metrics={validation_path}")
    print(f"predictions={predictions_path}")
    print(f"final_metrics={final_metrics_path}")
    print(f"calibration_reliability={calibration_path}")
    print(f"run_config={config_path}")
    return 0


def _forecast(args: argparse.Namespace) -> int:
    result = run_forecast(
        args.history,
        args.fixtures,
        args.as_of,
        selection_run_config_path=args.selection_run_config,
        model_name=args.model,
        model_config_path=args.model_config,
        season_mode=args.season_mode,
        output_root=args.output_root,
    )
    run = result["run"]
    print("MatchQuant M4 forecast complete")
    print(f"model={run.model_name} fixtures={len(run.predictions)} history_rows_used={run.history_rows_used}")
    print(f"as_of={run.as_of['value']} selection_evidence={result['selection_evidence_status']}")
    print(f"predictions={result['predictions_path']}")
    print(f"forecast_report={result['forecast_report_path']}")
    print(f"run_config={result['run_config_path']}")
    return 0


def _engine_snapshot_build(args: argparse.Namespace) -> int:
    if args.selection_run_config is not None:
        if args.model_config:
            raise ValueError("--selection-run-config and --model-config are mutually exclusive")
        result = build_engine_snapshot(
            args.history,
            args.history_manifest,
            args.as_of,
            selection_run_config_path=args.selection_run_config,
            output_root=args.snapshot_root,
            season_mode=args.season_mode,
        )
    else:
        names = args.model or []
        config_paths = args.model_config or []
        if not names:
            raise ValueError("provide --model at least once or --selection-run-config")
        if len(config_paths) != len(names):
            raise ValueError("each --model must have one matching --model-config")
        configs: dict[str, dict[str, Any]] = {}
        sources: dict[str, Path] = {}
        for name, path in zip(names, config_paths):
            if name in configs:
                raise ValueError("--model values must be unique")
            configs[name] = load_model_config(path, name)
            sources[name] = path
        result = build_engine_snapshot(
            args.history,
            args.history_manifest,
            args.as_of,
            model_names=names,
            model_configs=configs,
            model_config_paths=sources,
            output_root=args.snapshot_root,
            season_mode=args.season_mode,
        )
    manifest = result["manifest"]
    print("MatchQuant M11 engine snapshot build complete")
    print(f"models={','.join(result['model_names'])} rows_used={manifest['history']['rows_used_before_as_of']} as_of={manifest['as_of']['value']}")
    print(f"snapshot_dir={result['snapshot_dir']}")
    print(f"snapshot_manifest={result['snapshot_manifest_path']}")
    print(f"model_states={result['model_states_path']}")
    print(f"snapshot_manifest_sha256={result['manifest_sha256']}")
    return 0


def _engine_snapshot_predict(args: argparse.Namespace) -> int:
    result = predict_from_engine_snapshot(
        args.snapshot_manifest,
        args.fixtures,
        output_root=args.output_root,
    )
    print("MatchQuant M11 engine snapshot predict complete")
    print(f"models={','.join(result['config']['model_names'])} predictions={len(result['predictions'])}")
    print(f"predictions={result['predictions_path']}")
    print(f"snapshot_report={result['snapshot_report_path']}")
    print(f"run_config={result['run_config_path']}")
    return 0


def _engine_snapshot_status(args: argparse.Namespace) -> int:
    result = check_engine_snapshot_status(
        args.snapshot_manifest,
        args.current_pointer,
        output_root=args.output_root,
    )
    print(f"MatchQuant M12 engine snapshot status {result['status']}")
    print(f"snapshot={args.snapshot_manifest}")
    print(f"current_pointer={args.current_pointer}")
    print(f"status_manifest={result['status_manifest_path']}")
    print(f"status_report={result['status_report_path']}")
    return 0 if result["status"] in {"fresh", "stale"} else 2


_SNAPSHOT_STATUS_EXIT = {"ready": 0, "needs_review": 2, "blocked": 1}
_SNAPSHOT_EXPECTED_ERRORS = (DataValidationError, FileNotFoundError, OSError, TypeError, ValueError)


def _artifact_note(output_root: Path, kind: str) -> str:
    """Honest stderr note: report what actually exists instead of claiming nothing was saved."""

    from .snapshot_audit_support import AUDIT_DIR_PREFIX, SESSION_DIR_PREFIX

    try:
        root = Path(output_root).expanduser().resolve()
        existing = (
            sorted(
                entry.name
                for entry in root.iterdir()
                if entry.is_dir()
                and (entry.name.startswith(SESSION_DIR_PREFIX) or entry.name.startswith(AUDIT_DIR_PREFIX))
            )
            if root.is_dir()
            else []
        )
    except OSError:
        existing = []
    if existing:
        preview = ", ".join(existing[:3]) + ("..." if len(existing) > 3 else "")
        return (
            f"note: {len(existing)} {kind} director{'y' if len(existing) == 1 else 'ies'} already exist "
            f"under {root} ({preview}); the failure happened after their creation and saved artifacts "
            "may be incomplete"
        )
    return f"no {kind} artifacts were saved (the output was refused, unwritable, or never created)"


def _print_write_errors(result: dict[str, Any], stream: Any) -> None:
    for error in result.get("output_write_errors") or []:
        print(f"note: some artifacts could not be written: {error}", file=stream)


def _snapshot_session(args: argparse.Namespace) -> int:
    try:
        result = run_snapshot_session(
            args.snapshot_manifest,
            args.fixtures,
            mode=args.mode,
            output_root=args.output_root,
            issue_evidence_path=args.issue_evidence,
            bundle_manifest_path=args.bundle_manifest,
        )
    except _SNAPSHOT_EXPECTED_ERRORS as exc:
        print(f"MatchQuant snapshot-session blocked: {exc}", file=sys.stderr)
        print(_artifact_note(args.output_root, "session"), file=sys.stderr)
        return 1
    _print_write_errors(result, sys.stderr)
    print(f"MatchQuant snapshot session {result['status']}")
    print(f"mode={result['mode']} audit_status={result['audit_status']}")
    if result["reasons"]:
        print(f"reasons={','.join(result['reasons'])}")
    _print_session_evidence_summary(result)
    print(f"session_dir={result['session_dir']}")
    print(f"prediction_run={result['prediction_run_dir']}")
    print(f"session_manifest={result['session_manifest_path']}")
    print(f"session_report={result['session_report_path']}")
    return _SNAPSHOT_STATUS_EXIT[result["status"]]


def _print_session_evidence_summary(result: dict[str, Any]) -> None:
    """Show model/as_of/used-cutoff/bundle identity from this session's strict evidence.

    Everything comes from the manifest built during this run; missing fields are
    printed as 'unavailable' instead of being backfilled from history files.
    """

    manifest = result.get("manifest") or {}
    snapshot = manifest.get("snapshot") or {}
    model_names = snapshot.get("model_names") or []
    as_of = snapshot.get("as_of")
    as_of_value = as_of.get("value") if isinstance(as_of, dict) else as_of
    print(
        "model={model} as_of={as_of} used_data_cutoff={cutoff}".format(
            model=",".join(str(name) for name in model_names) if model_names else "unavailable",
            as_of=as_of_value or "unavailable",
            cutoff=snapshot.get("used_data_cutoff") or "unavailable",
        )
    )
    bundle = manifest.get("bundle") or {}
    if bundle:
        print(
            "bundle_root={root} bundle_snapshot_id={snapshot} fresh_at_issue={fresh} "
            "active_link_status={link}".format(
                root=bundle.get("bundle_root") or "unavailable",
                snapshot=bundle.get("snapshot_id") or "unavailable",
                fresh=bundle.get("fresh_at_issue") if bundle.get("fresh_at_issue") is not None else "unavailable",
                link=bundle.get("active_link_status") or "unavailable",
            )
        )


def _snapshot_run_verify(args: argparse.Namespace) -> int:
    try:
        result = verify_snapshot_run(
            args.forecast_run,
            output_root=args.output_root,
            issue_evidence_path=args.issue_evidence,
            bundle_manifest_path=args.bundle_manifest,
        )
    except _SNAPSHOT_EXPECTED_ERRORS as exc:
        print(f"MatchQuant snapshot-run-verify blocked: {exc}", file=sys.stderr)
        print(_artifact_note(args.output_root, "audit"), file=sys.stderr)
        return 1
    print(f"MatchQuant snapshot run audit {result['status']}")
    if result["reasons"]:
        print(f"reasons={','.join(result['reasons'])}")
    print(f"forecast_run={result['run_dir']}")
    print(f"audit_dir={result['audit_dir']}")
    print(f"prediction_audit={result['prediction_audit_path']}")
    print(f"audit_report={result['prediction_audit_report_path']}")
    return _SNAPSHOT_STATUS_EXIT[result["status"]]


def _runtime_bundle_build(args: argparse.Namespace) -> int:
    try:
        result = build_runtime_bundle(
            args.snapshot_manifest,
            args.issue_audit,
            output_root=args.output_root,
        )
    except _SNAPSHOT_EXPECTED_ERRORS as exc:
        print(f"MatchQuant runtime-bundle-build blocked: {exc}", file=sys.stderr)
        return 1
    print(f"MatchQuant runtime bundle build {result['status']}")
    if result["reasons"]:
        print(f"reasons={'; '.join(result['reasons'])}")
    for error in result.get("output_write_errors") or []:
        print(
            f"note: some required build artifacts could not be persisted: {error}; "
            "existing artifacts stay on disk, and the job record lists the final delivery state",
            file=sys.stderr,
        )
    print(f"snapshot_id={result['snapshot_id']}")
    print(f"build_dir={result['build_dir']}")
    print(f"bundle_dir={result['bundle_dir']}")
    if result["bundle_manifest_sha256"]:
        print(f"bundle_manifest={result['bundle_manifest_path']}")
        print(f"bundle_manifest_sha256={result['bundle_manifest_sha256']}")
    print(f"files={result['file_count']} actual_files={result['actual_file_count']} payload_bytes={result['payload_bytes']} total_bytes={result['total_bytes']} limit_bytes={result['limit_bytes']}")
    return 0 if result["status"] == "ready" else 1


def _runtime_bundle_verify(args: argparse.Namespace) -> int:
    try:
        result = verify_runtime_bundle(args.bundle_manifest)
    except _SNAPSHOT_EXPECTED_ERRORS as exc:
        print(f"MatchQuant runtime-bundle-verify blocked: {exc}", file=sys.stderr)
        return 1
    print(f"MatchQuant runtime bundle verify {result['status']}")
    if result["reasons"]:
        print(f"reasons={'; '.join(result['reasons'])}")
    if result["snapshot_id"]:
        print(f"snapshot_id={result['snapshot_id']}")
    print(f"bundle_root={result['bundle_root']}")
    print(f"files={result['file_count']} total_bytes={result['total_bytes']}")
    return 0 if result["status"] == "ready" else 1


def _demo_export(args: argparse.Namespace) -> int:
    try:
        result = export_synthetic_demo(args.output_dir)
    except _SNAPSHOT_EXPECTED_ERRORS as exc:
        print(f"MatchQuant demo-export failed: {exc}", file=sys.stderr)
        return 1
    print("MatchQuant synthetic demo export complete")
    print(
        "synthetic=true model=frequency (tutorial only; not a real primary-model selection)"
    )
    print(f"demo_dir={result['demo_dir']}")
    print(f"bundle_manifest={result['bundle_manifest']}")
    print(f"bundle_manifest_sha256={result['bundle_manifest_sha256']}")
    print(f"fixtures={result['fixtures']}")
    print(f"results={result['results']}")
    print(f"demo_doc={result['demo_doc']}")
    print(f"demo_manifest={result['demo_manifest']}")
    print(
        f"files={result['files_exported']} bundle_verify={result['bundle_verify_status']} "
        f"snapshot_id={result['snapshot_id']}"
    )
    print("next: follow the demo README section or DIR/DEMO.md (verify -> session -> audit -> settle)")
    return 0


def _epl_download(args: argparse.Namespace) -> int:
    manifest = download_epl(args.seasons, args.data_dir, timeout=args.timeout)
    print("MatchQuant M2 EPL download complete")
    print(f"seasons={','.join(manifest['requested_seasons'])}")
    print(f"manifest={args.data_dir / 'raw' / 'source_manifest.json'}")
    return 0


def _epl_prepare(args: argparse.Namespace) -> int:
    result = prepare_epl(
        args.seasons,
        args.data_dir,
        output_path=args.output,
        quality_report_path=args.quality_report,
        processed_manifest_path=args.processed_manifest,
    )
    print("MatchQuant M2 EPL prepare complete")
    print(f"seasons={','.join(result['seasons'])} rows={result['rows']} status={result['status']}")
    print(f"canonical={result['output_path']}")
    print(f"quality_report={result['quality_report_path']}")
    print(f"processed_manifest={result['processed_manifest_path']}")
    return 0


def _epl_sync(args: argparse.Namespace) -> int:
    result = sync_epl(
        seasons=args.seasons,
        data_dir=args.data_dir,
        timeout=args.timeout,
        apply=args.apply,
        in_progress_season=args.in_progress_season,
        staging_dir=args.staging_dir,
        baseline_path=args.baseline,
        processed_manifest_path=args.processed_manifest,
        current_pointer_path=args.current_pointer,
    )
    mode = "apply" if result["apply_requested"] else "dry-run"
    counts = result["diff"]
    print(f"MatchQuant M2.1 EPL sync {result['status']}")
    print(f"source={result.get('source') or 'unknown'} competition={result.get('competition') or 'unknown'} division={result.get('division') or 'unknown'}")
    if result.get("source_label"):
        print(f"source_label={result['source_label']}")
    if result.get("source_urls"):
        print(f"source_urls={','.join(result['source_urls'])}")
    policy = result["in_progress_season"] or "none"
    print(f"mode={mode} seasons={','.join(result['requested_seasons'])} baseline={result['baseline']['version']} in_progress_season={policy}")
    print(f"staging={result['staging_dir']}")
    print(f"added={counts['added']} unchanged={counts['unchanged']} revised={counts['revised']} missing={counts['missing_from_candidate']} skipped_unfinished={result['skipped_unfinished']}")
    print(f"quality_report={result['quality_report_path']}")
    print(f"diff_report={result['diff_report_path']}")
    if result["status"] == "blocked":
        print(f"apply_blocked={','.join(result['apply_blockers'])} manual_review_required=true")
        return 2
    print(f"processed_manifest={result['processed_manifest_path']}")
    if result["status"] == "applied":
        print(f"version={result['version_id']} pointer={result['pointer_path']}")
    return 0


def _epl_local_stage(args: argparse.Namespace) -> int:
    result = stage_local_epl_source(
        args.source_csv,
        args.season,
        args.data_dir,
        source_url=args.source_url,
        source_label=args.source_label,
        upstream_retrieved_at=args.upstream_retrieved_at,
    )
    manifest = result["manifest"]
    print(f"MatchQuant M7 local staging {result['status']}")
    print(f"snapshot={result['snapshot_dir']}")
    print(f"raw={result['raw_path']}")
    print(f"raw_sha256={result['raw_sha256']}")
    print(f"manifest={result['manifest_path']}")
    print(f"validation={manifest['validation_status']} next=epl-sync --staging-dir")
    return 0


def _epl_fixtures_prepare(args: argparse.Namespace) -> int:
    result = prepare_epl_fixtures(
        args.source_csv,
        args.season,
        args.data_root,
        source_url=args.source_url,
        source_label=args.source_label,
        baseline_fixtures=args.baseline_fixtures,
    )
    counts = result["counts"]
    print(f"MatchQuant M5 EPL fixture snapshot {result['status']}")
    print(f"snapshot={result['snapshot_dir']}")
    print(f"source_sha256={result['source_sha256']}")
    print(
        "pending={pending_rows} completed={completed_rows} partial={partial_score_rows} invalid={invalid_rows}".format(
            **counts
        )
    )
    if result["fixture_path"]:
        print(f"fixtures={result['fixture_path']}")
    print(f"manifest={result['manifest_path']}")
    print(f"quality_report={result['quality_report_path']}")
    if result["diff_report_path"]:
        print(f"diff_report={result['diff_report_path']}")
    return 2 if result["status"] == "no_pending_fixtures" else 0


def _settle(args: argparse.Namespace) -> int:
    result = settle_forecast_run(args.forecast_run, args.results, output_root=args.output_root)
    print(f"MatchQuant M6 settlement {result['status']}")
    print(f"settled={result['counts']['settled']} total={result['counts']['total_predictions']} coverage={result['metrics']['coverage']}")
    print(f"settlement={result['settlement_path']}")
    print(f"metrics={result['metrics_path']}")
    print(f"settlement_report={result['settlement_report_path']}")
    print(f"run_config={result['run_config_path']}")
    return 2 if result["status"] == "needs_review" else 0


def _release_verify(args: argparse.Namespace) -> int:
    result = verify_release(
        args.forecast_run,
        args.history,
        args.history_manifest,
        args.fixture_manifest,
        output_root=args.output_root,
    )
    print(f"MatchQuant M8 release verification {result['status']}")
    print(f"release_dir={result['run_dir']}")
    print(f"release_manifest={result['release_manifest_path']}")
    print(f"release_report={result['release_report_path']}")
    print(f"forecast_predictions={result['forecast_predictions_path']}")
    counts = {status: sum(item["status"] == status for item in result["checks"]) for status in ("passed", "needs_review", "blocked")}
    print(f"checks_passed={counts['passed']} needs_review={counts['needs_review']} blocked={counts['blocked']}")
    return 0 if result["status"] == "ready" else 2


def _forecast_session(args: argparse.Namespace) -> int:
    result = run_forecast_session(
        args.history,
        args.history_manifest,
        args.fixture_source,
        args.season,
        args.as_of,
        selection_run_config_path=args.selection_run_config,
        model_name=args.model,
        model_config_path=args.model_config,
        season_mode=args.season_mode,
        source_url=args.source_url,
        source_label=args.source_label,
        fixture_data_root=args.fixture_data_root,
        output_root=args.output_root,
    )
    print(f"MatchQuant M9 forecast session {result['status']}")
    print(f"session_dir={result['session_dir']}")
    print(f"release_report={result['release_report_path'] or 'not_run'}")
    return 0 if result["status"] == "ready" else 2


def _postmatch_session(args: argparse.Namespace) -> int:
    result = run_postmatch_session(
        args.session_manifest,
        args.season,
        results_source_csv=args.results_source,
        staging_dir=args.staging_dir,
        apply=args.apply,
        in_progress_season=args.in_progress_season,
        source_url=args.source_url,
        source_label=args.source_label,
        upstream_retrieved_at=args.upstream_retrieved_at,
        data_root=args.data_root,
        output_root=args.output_root,
        current_pointer_path=args.current_pointer,
    )
    print(f"MatchQuant M10 post-match session {result['status']}")
    print(f"session_dir={result['session_dir']}")
    print(f"staging={result['staging_dir'] or 'not_run'}")
    print(f"sync={result['stages']['sync']['status']}")
    print(f"settlement={result['stages']['settlement']['status']}")
    print(f"report={result['postmatch_report_path']}")
    return 0 if result["status"] in {"complete", "partial"} else 2


def main(argv: list[str] | None = None) -> None:
    parser = _parser()
    args = parser.parse_args(argv)
    if args.command == "snapshot-session" and args.bundle_manifest is not None and args.issue_evidence is not None:
        parser.error(
            "--issue-evidence cannot be combined with --bundle-manifest; bundle sessions read "
            "the bundled issue record itself"
        )
    try:
        if args.command == "run":
            raise SystemExit(_run(args))
        if args.command == "select":
            raise SystemExit(_select(args))
        if args.command == "forecast":
            raise SystemExit(_forecast(args))
        if args.command == "engine-snapshot-build":
            raise SystemExit(_engine_snapshot_build(args))
        if args.command == "engine-snapshot-predict":
            raise SystemExit(_engine_snapshot_predict(args))
        if args.command == "engine-snapshot-status":
            raise SystemExit(_engine_snapshot_status(args))
        if args.command == "snapshot-session":
            raise SystemExit(_snapshot_session(args))
        if args.command == "snapshot-run-verify":
            raise SystemExit(_snapshot_run_verify(args))
        if args.command == "runtime-bundle-build":
            raise SystemExit(_runtime_bundle_build(args))
        if args.command == "runtime-bundle-verify":
            raise SystemExit(_runtime_bundle_verify(args))
        if args.command == "demo-export":
            raise SystemExit(_demo_export(args))
        if args.command == "epl-download":
            raise SystemExit(_epl_download(args))
        if args.command == "epl-prepare":
            raise SystemExit(_epl_prepare(args))
        if args.command == "epl-sync":
            raise SystemExit(_epl_sync(args))
        if args.command == "epl-local-stage":
            raise SystemExit(_epl_local_stage(args))
        if args.command == "epl-fixtures-prepare":
            raise SystemExit(_epl_fixtures_prepare(args))
        if args.command == "settle":
            raise SystemExit(_settle(args))
        if args.command == "release-verify":
            raise SystemExit(_release_verify(args))
        if args.command == "forecast-session":
            raise SystemExit(_forecast_session(args))
        if args.command == "postmatch-session":
            raise SystemExit(_postmatch_session(args))
        raise ValueError(f"unknown command: {args.command}")
    except (DataValidationError, FileNotFoundError, OSError, TypeError, ValueError) as exc:
        parser.error(str(exc))
