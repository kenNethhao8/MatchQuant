"""MatchQuant M1/M2: deterministic football prediction and EPL data preparation."""

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
from .io import file_sha256, load_matches, write_json, write_predictions
from .metrics import evaluate_predictions
from .models import EloModel, FrequencyModel, PoissonModel, create_model
from .schema import DataValidationError, REQUIRED_COLUMNS, validate_matches
from .selection import load_candidates, parse_validation_folds, run_nested_selection

__all__ = [
    "DataValidationError",
    "DEFAULT_EPL_SEASONS",
    "EloModel",
    "FrequencyModel",
    "ModelRun",
    "PoissonModel",
    "REQUIRED_COLUMNS",
    "create_model",
    "audit_source_frame",
    "audit_sync_source_frame",
    "calibration_diagnostics",
    "compare_canonical",
    "convert_source_frame",
    "convert_sync_source_frame",
    "download_epl",
    "evaluate_predictions",
    "file_sha256",
    "load_matches",
    "load_candidates",
    "multiclass_ece",
    "parse_validation_folds",
    "reliability_table",
    "run_models",
    "run_walk_forward",
    "run_nested_selection",
    "normalize_season",
    "prepare_epl",
    "source_url",
    "stable_match_id",
    "sync_epl",
    "validate_matches",
    "write_json",
    "write_predictions",
]
