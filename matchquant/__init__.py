"""MatchQuant v2.0: Quantitative Football Factor Engineering, Factor Lab & Domain Modeling."""

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
from .factor_engine import (
    ALL_FACTOR_COLUMNS,
    ChronologicalFactorEngine,
    FACTOR_SPECIFICATIONS,
    FUNDAMENTAL_FACTOR_COLUMNS,
    MARKET_FACTOR_COLUMNS,
    TeamDynamicState,
    build_historical_factor_panel,
)
from .factor_lab import (
    compute_quantile_profile,
    compute_vif_table,
    render_factor_lab_report,
    run_factor_lab,
)
from .fixtures import (
    FIXTURE_COLUMNS,
    FIXTURE_REQUIRED_COLUMNS,
    FIXTURE_SCHEMA_VERSION,
    FixtureValidationError,
    load_fixtures,
    validate_fixtures,
)
from .io import file_sha256, load_matches, write_json, write_predictions
from .metrics import evaluate_predictions
from .models import EloModel, FrequencyModel, PoissonModel, create_model
from .odds_provider import OddsProviderError, fetch_epl_odds, snapshot_to_market_input
from .quant_models import (
    CalibratedFactorSystem,
    FactorDixonColesModel,
    OrderedFactorModel,
    evaluate_probs_frame,
    fit_and_select_factor_system,
    fit_dixon_coles_factor_model,
    fit_ordered_factor_model,
)
from .quant_pipeline import (
    explain_matches,
    render_quant_research_report,
    run_quant_research_pipeline,
)
from .rich_data import (
    MARKET_BENCHMARK_COLUMNS,
    POST_MATCH_PITCH_COLUMNS,
    PRE_MATCH_METADATA_COLUMNS,
    RICH_CANONICAL_COLUMNS,
    build_rich_epl_dataset,
    convert_raw_epl_season,
    devig_power_probabilities,
    discover_raw_epl_files,
)
from .single_match import (
    SingleMatchError,
    build_query_features,
    normalize_fixture,
    run_single_match_report,
)
from .schema import DataValidationError, REQUIRED_COLUMNS, validate_matches
from .selection import load_candidates, parse_validation_folds, run_nested_selection

__version__ = "2.0.0"

__all__ = [
    "__version__",
    "ALL_FACTOR_COLUMNS",
    "CalibratedFactorSystem",
    "ChronologicalFactorEngine",
    "DEFAULT_EPL_SEASONS",
    "DataValidationError",
    "EloModel",
    "FACTOR_SPECIFICATIONS",
    "FIXTURE_COLUMNS",
    "FIXTURE_REQUIRED_COLUMNS",
    "FIXTURE_SCHEMA_VERSION",
    "FUNDAMENTAL_FACTOR_COLUMNS",
    "FactorDixonColesModel",
    "FixtureValidationError",
    "FrequencyModel",
    "MARKET_BENCHMARK_COLUMNS",
    "MARKET_FACTOR_COLUMNS",
    "ModelRun",
    "OddsProviderError",
    "OrderedFactorModel",
    "POST_MATCH_PITCH_COLUMNS",
    "PRE_MATCH_METADATA_COLUMNS",
    "PoissonModel",
    "REQUIRED_COLUMNS",
    "RICH_CANONICAL_COLUMNS",
    "TeamDynamicState",
    "SingleMatchError",
    "audit_source_frame",
    "build_historical_factor_panel",
    "build_query_features",
    "build_rich_epl_dataset",
    "calibration_diagnostics",
    "compute_quantile_profile",
    "compute_vif_table",
    "convert_raw_epl_season",
    "convert_source_frame",
    "create_model",
    "devig_power_probabilities",
    "discover_raw_epl_files",
    "download_epl",
    "evaluate_predictions",
    "evaluate_probs_frame",
    "explain_matches",
    "fetch_epl_odds",
    "file_sha256",
    "fit_and_select_factor_system",
    "fit_dixon_coles_factor_model",
    "fit_ordered_factor_model",
    "load_candidates",
    "load_fixtures",
    "load_matches",
    "multiclass_ece",
    "normalize_season",
    "normalize_fixture",
    "parse_validation_folds",
    "prepare_epl",
    "reliability_table",
    "render_factor_lab_report",
    "render_quant_research_report",
    "run_factor_lab",
    "run_models",
    "run_nested_selection",
    "run_quant_research_pipeline",
    "run_single_match_report",
    "run_walk_forward",
    "snapshot_to_market_input",
    "source_url",
    "stable_match_id",
    "validate_fixtures",
    "validate_matches",
    "write_json",
    "write_predictions",
]
