"""Command-line entry point for MatchQuant M1."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from .backtest import run_models
from .epl import DEFAULT_EPL_SEASONS, download_epl, prepare_epl
from .io import file_sha256, load_matches, write_json, write_predictions
from .models import MODEL_NAMES
from .schema import DataValidationError


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


def main(argv: list[str] | None = None) -> None:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "run":
            raise SystemExit(_run(args))
        if args.command == "epl-download":
            raise SystemExit(_epl_download(args))
        if args.command == "epl-prepare":
            raise SystemExit(_epl_prepare(args))
        raise ValueError(f"unknown command: {args.command}")
    except (DataValidationError, FileNotFoundError, OSError, TypeError, ValueError) as exc:
        parser.error(str(exc))
