"""MatchQuant M1-M12: deterministic football research and local forecasts."""

from .backtest import ModelRun, run_models, run_walk_forward
from .calibration import calibration_diagnostics, multiclass_ece, reliability_table
from .demo_export import export_synthetic_demo
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
from .epl_local_stage import (
    LOCAL_STAGE_ADAPTER_VERSION,
    LOCAL_STAGE_MANIFEST_VERSION,
    stage_local_epl_source,
)
from .engine_snapshot import build_engine_snapshot, predict_from_engine_snapshot
from .engine_snapshot_status import (
    ENGINE_SNAPSHOT_STATUS_SCHEMA_VERSION,
    check_engine_snapshot_status,
)
from .engine_snapshot_support import (
    ENGINE_SNAPSHOT_SCHEMA_VERSION,
    ENGINE_STATE_SCHEMA_VERSION,
    export_model_state,
    import_model_state,
    load_engine_snapshot,
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
from .forecast_session import (
    FORECAST_SESSION_SCHEMA_VERSION,
    FORECAST_SESSION_STATUSES,
    new_forecast_session_dir,
    run_forecast_session,
)
from .postmatch_session import (
    POSTMATCH_SESSION_SCHEMA_VERSION,
    POSTMATCH_SESSION_STATUSES,
    run_postmatch_session,
)
from .io import file_sha256, load_matches, write_json, write_predictions
from .metrics import evaluate_predictions
from .models import EloModel, FrequencyModel, PoissonModel, create_model
from .prediction_evidence import load_prediction_evidence
from .schema import DataValidationError, REQUIRED_COLUMNS, validate_matches
from .selection import load_candidates, parse_validation_folds, run_nested_selection
from .settlement import (
    SETTLEMENT_METRICS_SCHEMA_VERSION,
    SETTLEMENT_SCHEMA_VERSION,
    load_forecast_run,
    settle_forecast_run,
    settle_predictions,
)
from .release_verify import (
    RELEASE_STATUSES,
    RELEASE_VERIFY_SCHEMA_VERSION,
    release_verify,
    verify_release,
)
from .runtime_bundle import build_runtime_bundle
from .runtime_bundle_validation import RUNTIME_BUNDLE_SCHEMA_VERSION, verify_runtime_bundle
from .snapshot_audit_support import (
    ACTIVE_LINK_STATUS as SNAPSHOT_ACTIVE_LINK_STATUS,
    AUDIT_SCHEMA_VERSION as SNAPSHOT_RUN_AUDIT_SCHEMA_VERSION,
    SESSION_SCHEMA_VERSION as SNAPSHOT_SESSION_SCHEMA_VERSION,
)
from .snapshot_run_verify import verify_snapshot_run
from .snapshot_session import run_snapshot_session
from .team_index import TEAM_INDEX_SCHEMA_VERSION

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
