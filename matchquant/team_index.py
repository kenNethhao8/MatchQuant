"""B04 team index derived deterministically from one strictly loaded M11 state.

The index is exported from the model state that the strict snapshot loader
already validated. Team names are the state's own source text, deterministically
sorted; no aliasing, no mapping to current league membership, and no inference
about whether a team still competes. Frequency has no team learning state: its
index is empty and declares the global-frequency basis instead of pretending an
empty list is a team roster.
"""

from __future__ import annotations

from typing import Any, Mapping

from .schema import DataValidationError

TEAM_INDEX_SCHEMA_VERSION = "baseline-team-index-v1"

_TEAM_INDEX_KEYS = (
    "schema_version",
    "model",
    "algorithm_source",
    "state_file_sha256",
    "model_payload_sha256",
    "model_config_sha256",
    "team_membership",
    "frequency_basis",
    "teams",
    "notes",
)

_FREQUENCY_NOTES = (
    "frequency 模型没有球队学习状态：index 为空并声明全局频率（global）。",
    "不能把空球队列表当作球队清单，也不能把任何输入球队称为新队或虚构训练球队。",
    "已知/未知球队的区分对该模型不适用；每个 fixture 都使用同一全局频率分布。",
)
_ELO_NOTES = (
    "球队清单从 Elo 状态 ratings 键确定性导出，保留球队源文本；不猜测别名、不映射当前联赛成员。",
    "不在清单中的球队按模型配置使用 initial_rating 先验；清单中的球队也不证明其仍参加当前赛季。",
)
_POISSON_NOTES = (
    "球队清单从 Poisson 状态 teams 键确定性导出，保留球队源文本；不猜测别名、不映射当前联赛成员。",
    "不在清单中的球队按模型先验回退到收缩后的全局进球率；清单中的球队也不证明其仍参加当前赛季。",
)


def _single_model_record(loaded: Mapping[str, Any]) -> Mapping[str, Any]:
    records = loaded.get("models")
    if not isinstance(records, list) or len(records) != 1 or not isinstance(records[0], Mapping):
        count = len(records) if isinstance(records, list) else 0
        raise DataValidationError(
            f"team index requires a single-model M11 snapshot state; found {count} model records"
        )
    return records[0]


def build_team_index(loaded: Mapping[str, Any]) -> dict[str, Any]:
    """Export the deterministic team index from one strictly loaded single-model snapshot."""

    record = _single_model_record(loaded)
    model = str(record.get("model"))
    state = record.get("state")
    if not isinstance(state, Mapping):
        raise DataValidationError(f"{model} state must be an object")

    if model == "frequency":
        teams: list[str] = []
        membership = "not_applicable"
        basis: str | None = "global"
        source = "no_team_learning_state"
        notes = list(_FREQUENCY_NOTES)
    elif model == "elo":
        ratings = state.get("ratings")
        if not isinstance(ratings, Mapping):
            raise DataValidationError("elo state ratings must be an object")
        teams = sorted(str(team) for team in ratings)
        membership = "learned_from_state"
        basis = None
        source = "elo_state_ratings"
        notes = list(_ELO_NOTES)
    elif model == "poisson":
        team_states = state.get("teams")
        if not isinstance(team_states, Mapping):
            raise DataValidationError("poisson state teams must be an object")
        teams = sorted(str(team) for team in team_states)
        membership = "learned_from_state"
        basis = None
        source = "poisson_state_teams"
        notes = list(_POISSON_NOTES)
    else:
        raise DataValidationError(f"team index does not support model '{model}'")

    return {
        "schema_version": TEAM_INDEX_SCHEMA_VERSION,
        "model": model,
        "algorithm_source": source,
        "state_file_sha256": str(loaded.get("state_sha256")),
        "model_payload_sha256": str(record.get("state_sha256")),
        "model_config_sha256": str(record.get("config_sha256")),
        "team_membership": membership,
        "frequency_basis": basis,
        "teams": teams,
        "notes": notes,
    }


def verify_team_index(
    index: Mapping[str, Any],
    loaded: Mapping[str, Any],
    *,
    state_file_sha256: str,
) -> dict[str, Any]:
    """Recompute the index from the strictly loaded state and compare it in full.

    A tampered team index fails even when its recorded file hash was synced:
    the state-derived recompute, not the recorded bytes, is the evidence.
    """

    if not isinstance(index, Mapping):
        raise DataValidationError("team index must be a JSON object")
    missing = [key for key in _TEAM_INDEX_KEYS if key not in index]
    extras = sorted(set(index) - set(_TEAM_INDEX_KEYS))
    if missing:
        raise DataValidationError(f"team index is missing required fields: {', '.join(missing)}")
    if extras:
        raise DataValidationError(f"team index has unknown fields: {', '.join(extras)}")
    if index.get("schema_version") != TEAM_INDEX_SCHEMA_VERSION:
        raise DataValidationError(
            f"team index schema_version must be {TEAM_INDEX_SCHEMA_VERSION}: "
            f"{index.get('schema_version')!r}"
        )
    loaded_state_hash = str(loaded.get("state_sha256"))
    if str(state_file_sha256).casefold() != loaded_state_hash.casefold():
        raise DataValidationError(
            "team index state file hash does not match the strictly loaded snapshot state"
        )
    recomputed = build_team_index(loaded)
    recomputed["state_file_sha256"] = str(state_file_sha256)
    if dict(index) != recomputed:
        raise DataValidationError(
            "team index does not match the index recomputed from the strictly loaded model state; "
            "a recorded team index is never trusted over the state-derived one"
        )
    return recomputed


__all__ = ["TEAM_INDEX_SCHEMA_VERSION", "build_team_index", "verify_team_index"]
