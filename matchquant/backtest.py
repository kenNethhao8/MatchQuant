"""Strict chronological online walk-forward execution."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping

import pandas as pd

from .metrics import evaluate_predictions
from .models import MODEL_NAMES, create_model, result_from_score
from .schema import DataValidationError, has_explicit_time, validate_matches

OUTPUT_COLUMNS = (
    "match_id",
    "date",
    "season",
    "home_team",
    "away_team",
    "home_goals",
    "away_goals",
    "actual_result",
    "model",
    "home_win_prob",
    "draw_prob",
    "away_win_prob",
    "expected_home_goals",
    "expected_away_goals",
)


@dataclass
class ModelRun:
    model_name: str
    predictions: pd.DataFrame
    metrics: dict[str, object]
    training_matches: int
    test_matches: int


def _parse_boundary(value: str | pd.Timestamp, label: str) -> tuple[pd.Timestamp, bool]:
    if value is None:
        raise ValueError(f"{label} is required for a strict walk-forward backtest")
    try:
        parsed = pd.to_datetime(value, errors="raise", format="mixed")
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid {label}: {value!r}") from exc
    timestamp = pd.Timestamp(parsed)
    date_only = not has_explicit_time(value)
    return (timestamp.normalize() if date_only else timestamp), date_only


def _timeline(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    calendar_day = frame["_timestamp"].dt.normalize()
    day_has_date_only = frame.groupby(calendar_day, sort=False)["_date_only"].transform("any")
    frame["_batch_key"] = frame["_timestamp"]
    frame.loc[day_has_date_only, "_batch_key"] = calendar_day.loc[day_has_date_only]
    return frame.sort_values(["_batch_key", "match_id"], kind="mergesort").reset_index(drop=True)


def _test_mask(
    frame: pd.DataFrame,
    test_start: str | pd.Timestamp,
    test_end: str | pd.Timestamp | None,
) -> pd.Series:
    start, start_date_only = _parse_boundary(test_start, "test_start")
    if not start_date_only:
        ambiguous = frame["_date_only"] & frame["_timestamp"].dt.normalize().eq(start.normalize())
        if ambiguous.any():
            raise ValueError(
                "test_start contains a clock but the same day has date-only matches; "
                "use a calendar-date test_start or supply exact times for every match"
            )
    mask = frame["_batch_key"].ge(start)
    if test_end is not None:
        end, end_date_only = _parse_boundary(test_end, "test_end")
        if not end_date_only:
            ambiguous = frame["_date_only"] & frame["_timestamp"].dt.normalize().eq(end.normalize())
            if ambiguous.any():
                raise ValueError(
                    "test_end contains a clock but the same day has date-only matches"
                )
        end_cutoff = end + pd.Timedelta(days=1) if end_date_only else end
        mask &= frame["_timestamp"].lt(end_cutoff)

    for _, batch in frame.groupby("_batch_key", sort=False):
        membership = mask.loc[batch.index]
        if membership.any() and not membership.all():
            raise ValueError("test boundary splits a prediction batch; use a batch-aligned boundary")
    if not mask.any():
        raise ValueError("test set is empty for the requested time boundary")
    return mask


def _prediction_row(row: Any, model_name: str, prediction: dict[str, float]) -> dict[str, object]:
    return {
        "match_id": row.match_id,
        "date": row.date,
        "season": row.season,
        "home_team": row.home_team,
        "away_team": row.away_team,
        "home_goals": int(row.home_goals),
        "away_goals": int(row.away_goals),
        "actual_result": result_from_score(int(row.home_goals), int(row.away_goals)),
        "model": model_name,
        "home_win_prob": prediction["home_win_prob"],
        "draw_prob": prediction["draw_prob"],
        "away_win_prob": prediction["away_win_prob"],
        "expected_home_goals": prediction.get("expected_home_goals"),
        "expected_away_goals": prediction.get("expected_away_goals"),
    }


def run_walk_forward(
    matches: pd.DataFrame,
    model: str | Any,
    test_start: str | pd.Timestamp,
    *,
    test_end: str | pd.Timestamp | None = None,
    season_mode: str = "carry",
    model_config: dict[str, Any] | None = None,
) -> ModelRun:
    """Run one model; each batch is predicted completely before any update."""

    if season_mode not in {"carry", "reset"}:
        raise ValueError("season_mode must be 'carry' or 'reset'")
    frame = _timeline(validate_matches(matches))
    mask = _test_mask(frame, test_start, test_end)
    first_test_index = frame.index[mask].min()
    training_mask = frame.index < first_test_index
    model_object = create_model(model, model_config) if isinstance(model, str) else model
    if not hasattr(model_object, "reset") or not hasattr(model_object, "predict") or not hasattr(model_object, "update"):
        raise TypeError("model must be a supported model name or online model object")
    model_object.reset()

    predictions: list[dict[str, object]] = []
    current_season: str | None = None
    eligible = training_mask | mask
    for _, batch in frame.loc[eligible].groupby("_batch_key", sort=False):
        seasons = batch["season"].astype(str).unique().tolist()
        if season_mode == "reset" and len(seasons) > 1:
            raise ValueError("season_mode='reset' cannot reset within a batch containing multiple seasons")
        season = seasons[0]
        if season_mode == "reset" and current_season is not None and season != current_season:
            model_object.reset()
        current_season = season

        batch_predictions: list[dict[str, float]] = []
        for row in batch.itertuples(index=False):
            batch_predictions.append(model_object.predict(row.home_team, row.away_team))

        is_test_batch = bool(mask.loc[batch.index].all())
        if is_test_batch:
            for row, prediction in zip(batch.itertuples(index=False), batch_predictions):
                predictions.append(_prediction_row(row, model_object.name, prediction))

        for row in batch.itertuples(index=False):
            model_object.update(row.home_team, row.away_team, int(row.home_goals), int(row.away_goals))

    prediction_frame = pd.DataFrame(predictions, columns=OUTPUT_COLUMNS)
    if prediction_frame.empty:
        raise ValueError("test set is empty after walk-forward execution")
    metrics = evaluate_predictions(prediction_frame)
    return ModelRun(
        model_name=model_object.name,
        predictions=prediction_frame,
        metrics=metrics,
        training_matches=int(training_mask.sum()),
        test_matches=int(mask.sum()),
    )


def run_models(
    matches: pd.DataFrame,
    model_names: Iterable[str],
    test_start: str | pd.Timestamp,
    *,
    test_end: str | pd.Timestamp | None = None,
    season_mode: str = "carry",
    model_configs: Mapping[str, dict[str, Any]] | None = None,
) -> dict[str, ModelRun]:
    """Run multiple models on the identical canonical test match set."""

    names = list(model_names)
    if not names:
        raise ValueError("at least one model is required")
    if len(set(names)) != len(names):
        raise ValueError("model names must be unique")
    unknown = [name for name in names if name not in MODEL_NAMES]
    if unknown:
        raise ValueError(f"unknown model(s): {unknown}; choose from {MODEL_NAMES}")

    normalized = validate_matches(matches)
    configs = {} if model_configs is None else model_configs
    runs: dict[str, ModelRun] = {}
    expected_ids: tuple[str, ...] | None = None
    for name in names:
        run = run_walk_forward(
            normalized,
            name,
            test_start,
            test_end=test_end,
            season_mode=season_mode,
            model_config=configs.get(name),
        )
        ids = tuple(run.predictions["match_id"].astype(str))
        if expected_ids is None:
            expected_ids = ids
        elif ids != expected_ids:
            raise DataValidationError("models did not receive the identical test match set")
        runs[name] = run
    return runs
