"""Command-line entry point for MatchQuant M1-M5."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from .backtest import run_models
from .epl import DEFAULT_EPL_SEASONS, download_epl, prepare_epl
from .epl_fixtures import prepare_epl_fixtures
from .epl_sync import sync_epl
from .forecast import run_forecast
from .io import file_sha256, load_matches, write_json, write_predictions
from .m3 import dataset_provenance, new_run_dir
from .models import MODEL_NAMES
from .schema import DataValidationError
from .selection import SELECTION_SCHEMA_VERSION, load_candidates, parse_validation_folds, run_nested_selection

DEFAULT_M3_CANDIDATE_CONFIG = Path(__file__).resolve().parent.parent / "examples" / "m3_candidates.json"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="matchquant", description="Reproducible football 1X2 research engine")
    subparsers = parser.add_subparsers(dest="command", required=True)
    run = subparsers.add_parser("run", help="run an online walk-forward backtest")
    run.add_argument("--config", type=Path, help="optional JSON configuration")
    run.add_argument("--input", type=Path, help="canonical match CSV")
    run.add_argument("--test-start", help="first test date/time; date-only starts are calendar-day aligned")
    run.add_argument("--test-end", help="optional exclusive test end date/time")
    run.add_argument("--model", choices=[*MODEL_NAMES, "all"], help="model or all three")
    run.add_argument("--season-mode", choices=["carry", "reset"], help="carry or reset online state across seasons")
    run.add_argument("--output-dir", type=Path, help="directory for predictions, metrics, and run config")
    select = subparsers.add_parser("select", help="M3 strict chronological model selection and calibration diagnostics")
    select.add_argument("--input", type=Path, required=True, help="local canonical match CSV")
    select.add_argument("--candidate-config", "--candidates", type=Path, default=DEFAULT_M3_CANDIDATE_CONFIG)
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
    forecast = subparsers.add_parser("forecast", help="run one offline local Alpha fixture forecast")
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
    print("source=Football-Data.co.uk competition=English Premier League division=E0")
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


def main(argv: list[str] | None = None) -> None:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "run":
            raise SystemExit(_run(args))
        if args.command == "select":
            raise SystemExit(_select(args))
        if args.command == "forecast":
            raise SystemExit(_forecast(args))
        if args.command == "epl-download":
            raise SystemExit(_epl_download(args))
        if args.command == "epl-prepare":
            raise SystemExit(_epl_prepare(args))
        if args.command == "epl-sync":
            raise SystemExit(_epl_sync(args))
        if args.command == "epl-fixtures-prepare":
            raise SystemExit(_epl_fixtures_prepare(args))
        raise ValueError(f"unknown command: {args.command}")
    except (DataValidationError, FileNotFoundError, OSError, TypeError, ValueError) as exc:
        parser.error(str(exc))
