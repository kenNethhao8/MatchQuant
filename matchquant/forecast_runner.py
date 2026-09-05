"""Persistence and CLI-facing orchestration for local Alpha forecasts."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from .fixtures import FIXTURE_SCHEMA_VERSION, load_fixtures
from .forecast import (
    DISCLAIMER,
    FORECAST_SCHEMA_VERSION,
    _configuration_hash,
    forecast_fixtures,
    load_model_config,
    load_selection_run_config,
    new_forecast_run_dir,
)
from .forecast_report import write_forecast_report
from .io import file_sha256, load_matches, write_json, write_predictions
from .models import MODEL_NAMES


def run_forecast(
    history_path: str | Path,
    fixtures_path: str | Path,
    as_of: str | pd.Timestamp,
    *,
    selection_run_config_path: str | Path | None = None,
    model_name: str | None = None,
    model_config_path: str | Path | None = None,
    season_mode: str | None = None,
    output_root: str | Path = Path(r"D:\Codex\matchquant_runs"),
) -> dict[str, Any]:
    """Run and persist one local Alpha forecast with an auditable run config."""

    history_file = Path(history_path).expanduser().resolve()
    fixtures_file = Path(fixtures_path).expanduser().resolve()
    if selection_run_config_path is not None and model_name is not None:
        raise ValueError("selection run config and --model are mutually exclusive")
    if selection_run_config_path is not None and model_config_path is not None:
        raise ValueError("selection run config and --model-config are mutually exclusive")
    if selection_run_config_path is None and model_name is None:
        raise ValueError("provide either a selection run config or explicit --model with --model-config")
    if model_name is not None and model_config_path is None:
        raise ValueError("explicit --model requires --model-config JSON")

    history_hash = file_sha256(history_file)
    fixtures_hash = file_sha256(fixtures_file)
    history = load_matches(history_file)
    fixtures = load_fixtures(fixtures_file)
    if selection_run_config_path is not None:
        model, config, selection = load_selection_run_config(
            selection_run_config_path, history_hash, as_of
        )
        selected_season_mode = selection.get("selection_season_mode")
        evidence_path = Path(selection["path"])
        evidence_config_hash = selection["sha256"]
    else:
        model = str(model_name)
        if model not in MODEL_NAMES:
            raise ValueError(f"unknown model '{model}'; choose from {MODEL_NAMES}")
        config = load_model_config(model_config_path, model)
        evidence_path = Path(model_config_path).expanduser().resolve()
        evidence_config_hash = file_sha256(evidence_path)
        selection = {
            "mode": "explicit_model_config",
            "path": str(evidence_path),
            "sha256": evidence_config_hash,
            "schema_version": None,
            "selection_history_sha256": None,
            "current_history_sha256": history_hash,
            "history_hash_matches": None,
            "history_hash_status": "not_applicable",
            "evidence_status": "explicit_model_config",
            "selection_as_of_status": "not_applicable",
            "as_of_evidence": {
                "status": "not_applicable",
                "reason": "explicit model configuration; no M3 candidate selection was used",
            },
            "validation_latest_end": None,
            "selection_rule": None,
            "final_test": None,
            "selected_candidate": None,
            "selection_season_mode": None,
        }
        selected_season_mode = None
    actual_season_mode = season_mode or selected_season_mode or "carry"
    run = forecast_fixtures(
        history,
        fixtures,
        model,
        as_of,
        model_config=config,
        season_mode=actual_season_mode,
        selection=selection,
    )
    run_dir = new_forecast_run_dir(output_root)
    predictions_path = write_predictions(run.predictions, run_dir / "predictions.csv")
    config_path = run_dir / "run_config.json"
    report_path = run_dir / "forecast_report.md"
    config_payload: dict[str, Any] = {
        "schema_version": FORECAST_SCHEMA_VERSION,
        "command": "matchquant forecast",
        "generated_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "as_of": run.as_of,
        "history": {
            "path": str(history_file),
            "sha256": history_hash,
            "rows": int(len(history)),
            "rows_used_before_as_of": run.history_rows_used,
            "rows_excluded_at_or_after_as_of": int(len(history) - run.history_rows_used),
            "data_cutoff": run.history_cutoff,
        },
        "fixtures": {
            "path": str(fixtures_file),
            "sha256": fixtures_hash,
            "rows": int(len(fixtures)),
            "schema": f"{FIXTURE_SCHEMA_VERSION}-score-free",
        },
        "model": {
            "name": run.model_name,
            "config": config,
            "config_sha256": _configuration_hash(config),
        },
        "season_mode": run.season_mode,
        "selection": selection,
        "selection_evidence_status": selection["evidence_status"],
        "selection_as_of_status": selection["selection_as_of_status"],
        "selection_validation_latest_end": selection.get("validation_latest_end"),
        "selection_as_of_evidence": selection.get("as_of_evidence"),
        "validation": {
            "all_fixture_dates_at_or_after_as_of": True,
            "fixture_results_used_for_model_updates": False,
            "probabilities_finite_non_negative_sum_to_one": True,
        },
        "outputs": {
            "run_dir": str(run_dir),
            "predictions": str(predictions_path),
            "forecast_report": str(report_path),
            "run_config": str(config_path),
        },
        "disclaimer": DISCLAIMER,
    }
    write_json(config_payload, config_path)
    write_forecast_report(run, config_payload, report_path)
    return {
        "run_dir": str(run_dir),
        "predictions_path": str(predictions_path),
        "forecast_report_path": str(report_path),
        "run_config_path": str(config_path),
        "predictions": run.predictions,
        "run": run,
        "config": config_payload,
        "history_sha256": history_hash,
        "fixtures_sha256": fixtures_hash,
        "selection_evidence_status": selection["evidence_status"],
        "selection_as_of_status": selection["selection_as_of_status"],
        "evidence_config_path": str(evidence_path),
        "evidence_config_sha256": evidence_config_hash,
    }


__all__ = ["run_forecast"]
