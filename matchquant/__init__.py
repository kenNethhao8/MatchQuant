"""MatchQuant M1–M12: deterministic football research and local forecasts.

Public exports are resolved lazily so that importing the package — including
through the installed entry points — does not import numpy/pandas until the
CLI boundary has had a chance to run its narrow missing-dependency diagnosis
(see :mod:`matchquant.launcher`). In a normal environment every name in
``__all__`` still resolves exactly as before, with one deliberate detail: the
package attribute for an exported name is ALWAYS the exported object, even
when a same-named submodule (``release_verify``) gets bound to the package by
the import machinery after a transitive import. The custom module class below
intercepts those attribute reads, so ``from matchquant import release_verify``
keeps returning the original function regardless of import order — the same
observable behavior as the original eager ``__init__``.
"""

import sys
import types
from importlib import import_module
from typing import Any

_EXPORTS: dict[str, tuple[str, str]] = {
    "ModelRun": ("backtest", "ModelRun"),
    "run_models": ("backtest", "run_models"),
    "run_walk_forward": ("backtest", "run_walk_forward"),
    "calibration_diagnostics": ("calibration", "calibration_diagnostics"),
    "multiclass_ece": ("calibration", "multiclass_ece"),
    "reliability_table": ("calibration", "reliability_table"),
    "export_synthetic_demo": ("demo_export", "export_synthetic_demo"),
    "DEFAULT_EPL_SEASONS": ("epl", "DEFAULT_EPL_SEASONS"),
    "audit_source_frame": ("epl", "audit_source_frame"),
    "convert_source_frame": ("epl", "convert_source_frame"),
    "download_epl": ("epl", "download_epl"),
    "normalize_season": ("epl", "normalize_season"),
    "prepare_epl": ("epl", "prepare_epl"),
    "source_url": ("epl", "source_url"),
    "stable_match_id": ("epl", "stable_match_id"),
    "audit_sync_source_frame": ("epl_sync", "audit_sync_source_frame"),
    "compare_canonical": ("epl_sync", "compare_canonical"),
    "convert_sync_source_frame": ("epl_sync", "convert_sync_source_frame"),
    "sync_epl": ("epl_sync", "sync_epl"),
    "FIXTURE_ADAPTER_VERSION": ("epl_fixtures", "FIXTURE_ADAPTER_VERSION"),
    "FIXTURE_DIFF_VERSION": ("epl_fixtures", "FIXTURE_DIFF_VERSION"),
    "FIXTURE_MANIFEST_VERSION": ("epl_fixtures", "FIXTURE_MANIFEST_VERSION"),
    "FIXTURE_QUALITY_VERSION": ("epl_fixtures", "FIXTURE_QUALITY_VERSION"),
    "FIXTURE_SOURCE_REQUIRED_COLUMNS": ("epl_fixtures", "FIXTURE_SOURCE_REQUIRED_COLUMNS"),
    "audit_fixture_source_frame": ("epl_fixtures", "audit_fixture_source_frame"),
    "compare_fixtures": ("epl_fixtures", "compare_fixtures"),
    "convert_fixture_source_frame": ("epl_fixtures", "convert_fixture_source_frame"),
    "prepare_epl_fixtures": ("epl_fixtures", "prepare_epl_fixtures"),
    "stable_fixture_id": ("epl_fixtures", "stable_fixture_id"),
    "LOCAL_STAGE_ADAPTER_VERSION": ("epl_local_stage", "LOCAL_STAGE_ADAPTER_VERSION"),
    "LOCAL_STAGE_MANIFEST_VERSION": ("epl_local_stage", "LOCAL_STAGE_MANIFEST_VERSION"),
    "stage_local_epl_source": ("epl_local_stage", "stage_local_epl_source"),
    "build_engine_snapshot": ("engine_snapshot", "build_engine_snapshot"),
    "predict_from_engine_snapshot": ("engine_snapshot", "predict_from_engine_snapshot"),
    "ENGINE_SNAPSHOT_STATUS_SCHEMA_VERSION": ("engine_snapshot_status", "ENGINE_SNAPSHOT_STATUS_SCHEMA_VERSION"),
    "check_engine_snapshot_status": ("engine_snapshot_status", "check_engine_snapshot_status"),
    "ENGINE_SNAPSHOT_SCHEMA_VERSION": ("engine_snapshot_support", "ENGINE_SNAPSHOT_SCHEMA_VERSION"),
    "ENGINE_STATE_SCHEMA_VERSION": ("engine_snapshot_support", "ENGINE_STATE_SCHEMA_VERSION"),
    "export_model_state": ("engine_snapshot_support", "export_model_state"),
    "import_model_state": ("engine_snapshot_support", "import_model_state"),
    "load_engine_snapshot": ("engine_snapshot_support", "load_engine_snapshot"),
    "FIXTURE_COLUMNS": ("fixtures", "FIXTURE_COLUMNS"),
    "FIXTURE_REQUIRED_COLUMNS": ("fixtures", "FIXTURE_REQUIRED_COLUMNS"),
    "FIXTURE_SCHEMA_VERSION": ("fixtures", "FIXTURE_SCHEMA_VERSION"),
    "FixtureValidationError": ("fixtures", "FixtureValidationError"),
    "load_fixtures": ("fixtures", "load_fixtures"),
    "validate_fixtures": ("fixtures", "validate_fixtures"),
    "FORECAST_PREDICTION_COLUMNS": ("forecast", "FORECAST_PREDICTION_COLUMNS"),
    "FORECAST_SCHEMA_VERSION": ("forecast", "FORECAST_SCHEMA_VERSION"),
    "ForecastRun": ("forecast", "ForecastRun"),
    "forecast_fixtures": ("forecast", "forecast_fixtures"),
    "load_model_config": ("forecast", "load_model_config"),
    "load_selection_run_config": ("forecast", "load_selection_run_config"),
    "run_forecast": ("forecast", "run_forecast"),
    "FORECAST_SESSION_SCHEMA_VERSION": ("forecast_session", "FORECAST_SESSION_SCHEMA_VERSION"),
    "FORECAST_SESSION_STATUSES": ("forecast_session", "FORECAST_SESSION_STATUSES"),
    "new_forecast_session_dir": ("forecast_session", "new_forecast_session_dir"),
    "run_forecast_session": ("forecast_session", "run_forecast_session"),
    "POSTMATCH_SESSION_SCHEMA_VERSION": ("postmatch_session", "POSTMATCH_SESSION_SCHEMA_VERSION"),
    "POSTMATCH_SESSION_STATUSES": ("postmatch_session", "POSTMATCH_SESSION_STATUSES"),
    "run_postmatch_session": ("postmatch_session", "run_postmatch_session"),
    "file_sha256": ("io", "file_sha256"),
    "load_matches": ("io", "load_matches"),
    "write_json": ("io", "write_json"),
    "write_predictions": ("io", "write_predictions"),
    "evaluate_predictions": ("metrics", "evaluate_predictions"),
    "EloModel": ("models", "EloModel"),
    "FrequencyModel": ("models", "FrequencyModel"),
    "PoissonModel": ("models", "PoissonModel"),
    "create_model": ("models", "create_model"),
    "load_prediction_evidence": ("prediction_evidence", "load_prediction_evidence"),
    "DataValidationError": ("schema", "DataValidationError"),
    "REQUIRED_COLUMNS": ("schema", "REQUIRED_COLUMNS"),
    "validate_matches": ("schema", "validate_matches"),
    "load_candidates": ("selection", "load_candidates"),
    "parse_validation_folds": ("selection", "parse_validation_folds"),
    "run_nested_selection": ("selection", "run_nested_selection"),
    "SETTLEMENT_METRICS_SCHEMA_VERSION": ("settlement", "SETTLEMENT_METRICS_SCHEMA_VERSION"),
    "SETTLEMENT_SCHEMA_VERSION": ("settlement", "SETTLEMENT_SCHEMA_VERSION"),
    "load_forecast_run": ("settlement", "load_forecast_run"),
    "settle_forecast_run": ("settlement", "settle_forecast_run"),
    "settle_predictions": ("settlement", "settle_predictions"),
    "RELEASE_STATUSES": ("release_verify", "RELEASE_STATUSES"),
    "RELEASE_VERIFY_SCHEMA_VERSION": ("release_verify", "RELEASE_VERIFY_SCHEMA_VERSION"),
    "release_verify": ("release_verify", "release_verify"),
    "verify_release": ("release_verify", "verify_release"),
    "build_runtime_bundle": ("runtime_bundle", "build_runtime_bundle"),
    "RUNTIME_BUNDLE_SCHEMA_VERSION": ("runtime_bundle_validation", "RUNTIME_BUNDLE_SCHEMA_VERSION"),
    "verify_runtime_bundle": ("runtime_bundle_validation", "verify_runtime_bundle"),
    "SNAPSHOT_ACTIVE_LINK_STATUS": ("snapshot_audit_support", "ACTIVE_LINK_STATUS"),
    "SNAPSHOT_RUN_AUDIT_SCHEMA_VERSION": ("snapshot_audit_support", "AUDIT_SCHEMA_VERSION"),
    "SNAPSHOT_SESSION_SCHEMA_VERSION": ("snapshot_audit_support", "SESSION_SCHEMA_VERSION"),
    "verify_snapshot_run": ("snapshot_run_verify", "verify_snapshot_run"),
    "run_snapshot_session": ("snapshot_session", "run_snapshot_session"),
    "TEAM_INDEX_SCHEMA_VERSION": ("team_index", "TEAM_INDEX_SCHEMA_VERSION"),
}


def _resolve_export(name: str) -> Any:
    module_name, attribute = _EXPORTS[name]
    return getattr(import_module(f".{module_name}", __name__), attribute)


class _LazyExportModule(types.ModuleType):
    """Package module type that resolves public exports lazily.

    ``__getattribute__`` (not only PEP 562 ``__getattr__``) is required because
    importing a submodule binds that submodule as an attribute of the parent
    package: for the exported name ``release_verify`` that binding would
    otherwise shadow the original public function forever. Export names are
    always resolved to their declared object and re-cached, so import order
    cannot change the observable type or identity. Resolution is reentrant-safe:
    ``import_module`` consults ``sys.modules`` and the import lock, so nested
    package-attribute lookups during a transitive import degrade to plain
    attribute reads instead of recursing.
    """

    def __getattribute__(self, name: str) -> Any:
        if name in _EXPORTS:
            value = _resolve_export(name)
            object.__getattribute__(self, "__dict__")[name] = value
            return value
        return types.ModuleType.__getattribute__(self, name)


sys.modules[__name__].__class__ = _LazyExportModule


def __dir__() -> list[str]:
    # globals() is the module dict itself; no attribute access on the module
    # object happens here, so the custom __getattribute__ is not re-entered.
    return sorted(set(globals()) | set(_EXPORTS))


__all__ = [
    "DataValidationError",
    "FIXTURE_COLUMNS",
    "FIXTURE_REQUIRED_COLUMNS",
    "FIXTURE_SCHEMA_VERSION",
    "FORECAST_PREDICTION_COLUMNS",
    "FORECAST_SCHEMA_VERSION",
    "FORECAST_SESSION_SCHEMA_VERSION",
    "FORECAST_SESSION_STATUSES",
    "POSTMATCH_SESSION_SCHEMA_VERSION",
    "POSTMATCH_SESSION_STATUSES",
    "DEFAULT_EPL_SEASONS",
    "FIXTURE_ADAPTER_VERSION",
    "FIXTURE_DIFF_VERSION",
    "FIXTURE_MANIFEST_VERSION",
    "FIXTURE_QUALITY_VERSION",
    "FIXTURE_SOURCE_REQUIRED_COLUMNS",
    "ENGINE_SNAPSHOT_SCHEMA_VERSION",
    "ENGINE_STATE_SCHEMA_VERSION",
    "ENGINE_SNAPSHOT_STATUS_SCHEMA_VERSION",
    "LOCAL_STAGE_ADAPTER_VERSION",
    "LOCAL_STAGE_MANIFEST_VERSION",
    "EloModel",
    "FixtureValidationError",
    "FrequencyModel",
    "ForecastRun",
    "ModelRun",
    "PoissonModel",
    "REQUIRED_COLUMNS",
    "SETTLEMENT_METRICS_SCHEMA_VERSION",
    "SETTLEMENT_SCHEMA_VERSION",
    "RELEASE_STATUSES",
    "RELEASE_VERIFY_SCHEMA_VERSION",
    "SNAPSHOT_ACTIVE_LINK_STATUS",
    "SNAPSHOT_RUN_AUDIT_SCHEMA_VERSION",
    "SNAPSHOT_SESSION_SCHEMA_VERSION",
    "RUNTIME_BUNDLE_SCHEMA_VERSION",
    "TEAM_INDEX_SCHEMA_VERSION",
    "create_model",
    "build_engine_snapshot",
    "check_engine_snapshot_status",
    "audit_source_frame",
    "audit_sync_source_frame",
    "calibration_diagnostics",
    "compare_canonical",
    "compare_fixtures",
    "convert_source_frame",
    "convert_sync_source_frame",
    "convert_fixture_source_frame",
    "download_epl",
    "evaluate_predictions",
    "export_synthetic_demo",
    "file_sha256",
    "forecast_fixtures",
    "load_fixtures",
    "load_engine_snapshot",
    "load_model_config",
    "load_selection_run_config",
    "load_matches",
    "load_candidates",
    "load_forecast_run",
    "load_prediction_evidence",
    "multiclass_ece",
    "export_model_state",
    "import_model_state",
    "parse_validation_folds",
    "reliability_table",
    "run_models",
    "run_forecast",
    "run_snapshot_session",
    "verify_snapshot_run",
    "build_runtime_bundle",
    "verify_runtime_bundle",
    "predict_from_engine_snapshot",
    "new_forecast_session_dir",
    "run_forecast_session",
    "run_postmatch_session",
    "run_walk_forward",
    "run_nested_selection",
    "settle_forecast_run",
    "settle_predictions",
    "release_verify",
    "verify_release",
    "normalize_season",
    "prepare_epl",
    "prepare_epl_fixtures",
    "source_url",
    "stable_match_id",
    "stable_fixture_id",
    "stage_local_epl_source",
    "sync_epl",
    "validate_matches",
    "write_json",
    "write_predictions",
]
