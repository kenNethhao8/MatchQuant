"""MatchQuant M1-M6: deterministic football research and local forecasts."""

from .backtest import ModelRun, run_models, run_walk_forward
from .calibration import calibration_diagnostics, multiclass_ece, reliability_table
from .epl import (
    DEFAULT_EPL_SEASONS,
    audit_source_frame,
    convert_source_frame,
    download_epl,
    normalize_season,
    prepare_epl,
    source_url,
    stable_match_id,
)
from .epl_sync import audit_sync_source_frame, compare_canonical, convert_sync_source_frame, sync_epl
from .epl_fixtures import (
    FIXTURE_ADAPTER_VERSION,
    FIXTURE_DIFF_VERSION,
    FIXTURE_MANIFEST_VERSION,
    FIXTURE_QUALITY_VERSION,
    FIXTURE_SOURCE_REQUIRED_COLUMNS,
    audit_fixture_source_frame,
    compare_fixtures,
    convert_fixture_source_frame,
    prepare_epl_fixtures,
    stable_fixture_id,
)
from .fixtures import (
    FIXTURE_COLUMNS,
    FIXTURE_REQUIRED_COLUMNS,
    FIXTURE_SCHEMA_VERSION,
    FixtureValidationError,
    load_fixtures,
    validate_fixtures,
)
from .forecast import (
    FORECAST_PREDICTION_COLUMNS,
    FORECAST_SCHEMA_VERSION,
    ForecastRun,
    forecast_fixtures,
    load_model_config,
    load_selection_run_config,
    run_forecast,
)
from .io import file_sha256, load_matches, write_json, write_predictions
from .metrics import evaluate_predictions
from .models import EloModel, FrequencyModel, PoissonModel, create_model
from .schema import DataValidationError, REQUIRED_COLUMNS, validate_matches
from .selection import load_candidates, parse_validation_folds, run_nested_selection
from .settlement import (
    SETTLEMENT_METRICS_SCHEMA_VERSION,
    SETTLEMENT_SCHEMA_VERSION,
    load_forecast_run,
    settle_forecast_run,
    settle_predictions,
)

__all__ = [
    "DataValidationError",
    "FIXTURE_COLUMNS",
    "FIXTURE_REQUIRED_COLUMNS",
    "FIXTURE_SCHEMA_VERSION",
    "FORECAST_PREDICTION_COLUMNS",
    "FORECAST_SCHEMA_VERSION",
    "DEFAULT_EPL_SEASONS",
    "FIXTURE_ADAPTER_VERSION",
    "FIXTURE_DIFF_VERSION",
    "FIXTURE_MANIFEST_VERSION",
    "FIXTURE_QUALITY_VERSION",
    "FIXTURE_SOURCE_REQUIRED_COLUMNS",
    "EloModel",
    "FixtureValidationError",
    "FrequencyModel",
    "ForecastRun",
    "ModelRun",
    "PoissonModel",
    "REQUIRED_COLUMNS",
    "SETTLEMENT_METRICS_SCHEMA_VERSION",
    "SETTLEMENT_SCHEMA_VERSION",
    "create_model",
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
    "file_sha256",
    "forecast_fixtures",
    "load_fixtures",
    "load_model_config",
    "load_selection_run_config",
    "load_matches",
    "load_candidates",
    "load_forecast_run",
    "multiclass_ece",
    "parse_validation_folds",
    "reliability_table",
    "run_models",
    "run_forecast",
    "run_walk_forward",
    "run_nested_selection",
    "settle_forecast_run",
    "settle_predictions",
    "normalize_season",
    "prepare_epl",
    "prepare_epl_fixtures",
    "source_url",
    "stable_match_id",
    "stable_fixture_id",
    "sync_epl",
    "validate_matches",
    "write_json",
    "write_predictions",
]
