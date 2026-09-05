"""MatchQuant M1: deterministic football prediction and backtesting."""

from .backtest import ModelRun, run_models, run_walk_forward
from .io import file_sha256, load_matches, write_json, write_predictions
from .metrics import evaluate_predictions
from .models import EloModel, FrequencyModel, PoissonModel, create_model
from .schema import DataValidationError, REQUIRED_COLUMNS, validate_matches

__all__ = [
    "DataValidationError",
    "EloModel",
    "FrequencyModel",
    "ModelRun",
    "PoissonModel",
    "REQUIRED_COLUMNS",
    "create_model",
    "evaluate_predictions",
    "file_sha256",
    "load_matches",
    "run_models",
    "run_walk_forward",
    "validate_matches",
    "write_json",
    "write_predictions",
]
