"""File input and structured output helpers."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .schema import DataValidationError, validate_matches


def load_matches(path: str | Path) -> pd.DataFrame:
    """Read a UTF-8 CSV and validate it against the canonical match schema."""

    csv_path = Path(path)
    if not csv_path.is_file():
        raise FileNotFoundError(f"input CSV does not exist: {csv_path}")
    try:
        frame = pd.read_csv(csv_path, dtype="string", keep_default_na=False)
    except (OSError, ValueError, pd.errors.ParserError) as exc:
        raise DataValidationError(f"could not read CSV '{csv_path}': {exc}") from exc
    return validate_matches(frame)


def file_sha256(path: str | Path) -> str:
    """Return the SHA-256 hash of the exact input file bytes."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    raise TypeError(f"object of type {type(value).__name__} is not JSON serializable")


def write_json(payload: dict[str, Any], path: str | Path) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default) + "\n",
        encoding="utf-8",
    )
    return output_path


def write_predictions(predictions: pd.DataFrame, path: str | Path) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(output_path, index=False, encoding="utf-8-sig")
    return output_path
