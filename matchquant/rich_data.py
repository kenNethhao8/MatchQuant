"""Layer 1: Full-spectrum EPL Rich Match Data Extraction & Pre/Post-Match Isolation.

Unlocks the 50+ high-value pitch statistics (shots, shots on target, corners,
half-time scores, fouls, yellow/red cards, referee, kickoff time) and market
benchmarks (Pinnacle/Bet365 opening & closing odds, Over/Under 2.5, Asian Handicap)
that were previously discarded during M2 canonical conversion.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

from .epl import normalize_season, read_source_csv, stable_match_id
from .schema import DataValidationError

BASIC_INFO_COLUMNS: tuple[str, ...] = (
    "match_id",
    "season",
    "date",
    "kickoff_time",
    "home_team",
    "away_team",
    "home_goals",
    "away_goals",
    "actual_result",
    "referee",
)

# Backwards-compatible aliases for the grouped exports
PRE_MATCH_METADATA_COLUMNS: tuple[str, ...] = BASIC_INFO_COLUMNS

PITCH_PROCESS_COLUMNS: tuple[str, ...] = (
    "ht_home_goals",
    "ht_away_goals",
    "ht_result",
    "sh_home_goals",
    "sh_away_goals",
    "home_shots",
    "away_shots",
    "home_sot",
    "away_sot",
    "home_corners",
    "away_corners",
    "home_fouls",
    "away_fouls",
    "home_yellow",
    "away_yellow",
    "home_red",
    "away_red",
)

# Backwards-compatible alias
POST_MATCH_PITCH_COLUMNS: tuple[str, ...] = PITCH_PROCESS_COLUMNS

MARKET_BENCHMARK_COLUMNS: tuple[str, ...] = (
    "odds_open_h",
    "odds_open_d",
    "odds_open_a",
    "odds_close_h",
    "odds_close_d",
    "odds_close_a",
    "mkt_open_prob_home",
    "mkt_open_prob_draw",
    "mkt_open_prob_away",
    "mkt_close_prob_home",
    "mkt_close_prob_draw",
    "mkt_close_prob_away",
    "mkt_steam_home",
    "mkt_steam_draw",
    "mkt_steam_away",
    "mkt_ou25_over_prob",
    "mkt_ah_line",
)

RICH_CANONICAL_COLUMNS: tuple[str, ...] = (
    *BASIC_INFO_COLUMNS,
    *PITCH_PROCESS_COLUMNS,
    *MARKET_BENCHMARK_COLUMNS,
)

RICH_CANONICAL_COLUMNS: tuple[str, ...] = (
    *PRE_MATCH_METADATA_COLUMNS,
    *POST_MATCH_PITCH_COLUMNS,
    *MARKET_BENCHMARK_COLUMNS,
)


def devig_power_probabilities(
    odds_h: np.ndarray | pd.Series,
    odds_d: np.ndarray | pd.Series,
    odds_a: np.ndarray | pd.Series,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert decimal 1X2 odds into de-vigged probabilities using Power method.

    Power de-vigging raises raw implied probabilities (1/odds) to an exponent k >= 1
    such that p_H^k + p_D^k + p_A^k = 1. Unlike naive proportional normalization,
    Power de-vigging properly accounts for favorite-longshot bias in football markets.
    Falls back to proportional normalization if Newton iteration does not apply.
    """
    oh = np.asarray(odds_h, dtype=np.float64)
    od = np.asarray(odds_d, dtype=np.float64)
    oa = np.asarray(odds_a, dtype=np.float64)

    valid = (
        np.isfinite(oh)
        & np.isfinite(od)
        & np.isfinite(oa)
        & (oh > 1.0)
        & (od > 1.0)
        & (oa > 1.0)
    )
    n = len(oh)
    ph = np.full(n, np.nan, dtype=np.float64)
    pd_ = np.full(n, np.nan, dtype=np.float64)
    pa = np.full(n, np.nan, dtype=np.float64)

    if not np.any(valid):
        return ph, pd_, pa

    inv_h = 1.0 / oh[valid]
    inv_d = 1.0 / od[valid]
    inv_a = 1.0 / oa[valid]
    raw = np.stack([inv_h, inv_d, inv_a], axis=1)
    log_raw = np.log(np.clip(raw, 1e-9, 1.0))

    # Solve sum(raw_i ^ k) = 1 via 6 vectorized Newton-Raphson steps
    k = np.ones(raw.shape[0], dtype=np.float64)
    for _ in range(6):
        powered = np.exp(k[:, None] * log_raw)
        f_val = powered.sum(axis=1) - 1.0
        f_prime = (powered * log_raw).sum(axis=1)
        step = np.where(np.abs(f_prime) > 1e-12, f_val / f_prime, 0.0)
        k = np.clip(k - step, 0.5, 3.0)

    probs = np.exp(k[:, None] * log_raw)
    row_sums = probs.sum(axis=1, keepdims=True)
    probs = np.where(row_sums > 0, probs / row_sums, raw / raw.sum(axis=1, keepdims=True))

    ph[valid] = probs[:, 0]
    pd_[valid] = probs[:, 1]
    pa[valid] = probs[:, 2]
    return ph, pd_, pa


def _first_available_numeric(
    df: pd.DataFrame, candidates: Sequence[str], default: float = np.nan
) -> pd.Series:
    """Extract numeric series coalescing across candidate columns in priority order."""
    result = pd.Series(np.nan, index=df.index, dtype="float64")
    for col in candidates:
        if col in df.columns:
            vals = pd.to_numeric(df[col].astype("string").str.strip(), errors="coerce")
            result = result.fillna(vals)
    if not np.isnan(default):
        result = result.fillna(default)
    return result


def _parse_dates(series: pd.Series) -> pd.Series:
    cleaned = series.astype("string").str.strip()
    if cleaned.str.fullmatch(r"\d{2}/\d{2}/\d{4}").all():
        return pd.to_datetime(cleaned, format="%d/%m/%Y", errors="raise")
    if cleaned.str.fullmatch(r"\d{2}/\d{2}/\d{2}").all():
        return pd.to_datetime(cleaned, format="%d/%m/%y", errors="raise")
    parsed = pd.to_datetime(cleaned, dayfirst=True, errors="coerce")
    if parsed.isna().any():
        raise DataValidationError("Failed to parse one or more source match dates")
    return parsed


def convert_raw_epl_season(source: pd.DataFrame, season: str) -> pd.DataFrame:
    """Convert one raw Football-Data E0 DataFrame into the Rich Match Table."""
    normalized_season = normalize_season(season)

    # Drop completely empty trailing rows occasionally present in Football-Data CSVs
    if "Date" in source.columns and "HomeTeam" in source.columns:
        non_empty_mask = (
            source["Date"].astype("string").str.strip().ne("")
            & source["HomeTeam"].astype("string").str.strip().ne("")
        )
        df = source.loc[non_empty_mask].copy()
    else:
        raise DataValidationError(f"Season {normalized_season} is missing Date or HomeTeam")

    dates = _parse_dates(df["Date"]).dt.strftime("%Y-%m-%d")
    home_team = df["HomeTeam"].astype("string").str.strip()
    away_team = df["AwayTeam"].astype("string").str.strip()

    if "Time" in df.columns:
        kickoff_time = (
            df["Time"]
            .astype("string")
            .str.strip()
            .where(lambda s: s.str.fullmatch(r"\d{2}:\d{2}"), "")
        )
    else:
        kickoff_time = pd.Series("", index=df.index, dtype="string")

    if "Referee" in df.columns:
        referee = df["Referee"].astype("string").str.strip().replace("", "Unknown")
    else:
        referee = pd.Series("Unknown", index=df.index, dtype="string")

    match_ids = [
        stable_match_id(normalized_season, h, a)
        for h, a in zip(home_team, away_team)
    ]

    fthg = _first_available_numeric(df, ["FTHG"]).astype(int)
    ftag = _first_available_numeric(df, ["FTAG"]).astype(int)
    hthg = _first_available_numeric(df, ["HTHG"], default=0.0).astype(int)
    htag = _first_available_numeric(df, ["HTAG"], default=0.0).astype(int)

    result = np.where(fthg > ftag, "H", np.where(fthg < ftag, "A", "D"))
    ht_result = np.where(hthg > htag, "H", np.where(hthg < htag, "A", "D"))
    sh_home_goals = fthg - hthg
    sh_away_goals = ftag - htag

    hs = _first_available_numeric(df, ["HS"], default=0.0).astype(int)
    as_ = _first_available_numeric(df, ["AS"], default=0.0).astype(int)
    hst = _first_available_numeric(df, ["HST"], default=0.0).astype(int)
    ast = _first_available_numeric(df, ["AST"], default=0.0).astype(int)
    hc = _first_available_numeric(df, ["HC"], default=0.0).astype(int)
    ac = _first_available_numeric(df, ["AC"], default=0.0).astype(int)
    hf = _first_available_numeric(df, ["HF"], default=0.0).astype(int)
    af = _first_available_numeric(df, ["AF"], default=0.0).astype(int)
    hy = _first_available_numeric(df, ["HY"], default=0.0).astype(int)
    ay = _first_available_numeric(df, ["AY"], default=0.0).astype(int)
    hr = _first_available_numeric(df, ["HR"], default=0.0).astype(int)
    ar = _first_available_numeric(df, ["AR"], default=0.0).astype(int)

    # Opening 1X2 odds: prioritize Pinnacle (PSH/PSD/PSA), fallback to Market Avg then Bet365
    odds_open_h = _first_available_numeric(df, ["PSH", "AvgH", "BbAvH", "B365H", "BWH"])
    odds_open_d = _first_available_numeric(df, ["PSD", "AvgD", "BbAvD", "B365D", "BWD"])
    odds_open_a = _first_available_numeric(df, ["PSA", "AvgA", "BbAvA", "B365A", "BWA"])

    # Closing 1X2 odds: prioritize Pinnacle Closing (PSCH/PSCD/PSCA), fallback to AvgC / B365C / Opening
    odds_close_h = _first_available_numeric(df, ["PSCH", "AvgCH", "B365CH", "PSH", "B365H"])
    odds_close_d = _first_available_numeric(df, ["PSCD", "AvgCD", "B365CD", "PSD", "B365D"])
    odds_close_a = _first_available_numeric(df, ["PSCA", "AvgCA", "B365CA", "PSA", "B365A"])

    open_ph, open_pd, open_pa = devig_power_probabilities(odds_open_h, odds_open_d, odds_open_a)
    close_ph, close_pd, close_pa = devig_power_probabilities(odds_close_h, odds_close_d, odds_close_a)

    steam_h = close_ph - open_ph
    steam_d = close_pd - open_pd
    steam_a = close_pa - open_pa

    # Over/Under 2.5 implied probability
    odds_over25 = _first_available_numeric(df, ["P>2.5", "Avg>2.5", "BbAv>2.5", "B365>2.5"])
    odds_under25 = _first_available_numeric(df, ["P<2.5", "Avg<2.5", "BbAv<2.5", "B365<2.5"])
    inv_o = 1.0 / odds_over25
    inv_u = 1.0 / odds_under25
    mkt_ou25_over_prob = np.where(
        np.isfinite(inv_o) & np.isfinite(inv_u) & ((inv_o + inv_u) > 0),
        inv_o / (inv_o + inv_u),
        np.nan,
    )

    # Asian Handicap line (home team handicap, e.g. -1.5 means home favored by 1.5 goals)
    mkt_ah_line = _first_available_numeric(df, ["AHh", "BbAHh"], default=0.0)

    rich = pd.DataFrame(
        {
            "match_id": match_ids,
            "season": normalized_season,
            "date": dates,
            "kickoff_time": kickoff_time,
            "home_team": home_team,
            "away_team": away_team,
            "home_goals": fthg,
            "away_goals": ftag,
            "actual_result": result,
            "referee": referee,
            "ht_home_goals": hthg,
            "ht_away_goals": htag,
            "ht_result": ht_result,
            "sh_home_goals": sh_home_goals,
            "sh_away_goals": sh_away_goals,
            "home_shots": hs,
            "away_shots": as_,
            "home_sot": hst,
            "away_sot": ast,
            "home_corners": hc,
            "away_corners": ac,
            "home_fouls": hf,
            "away_fouls": af,
            "home_yellow": hy,
            "away_yellow": ay,
            "home_red": hr,
            "away_red": ar,
            "odds_open_h": odds_open_h,
            "odds_open_d": odds_open_d,
            "odds_open_a": odds_open_a,
            "odds_close_h": odds_close_h,
            "odds_close_d": odds_close_d,
            "odds_close_a": odds_close_a,
            "mkt_open_prob_home": open_ph,
            "mkt_open_prob_draw": open_pd,
            "mkt_open_prob_away": open_pa,
            "mkt_close_prob_home": close_ph,
            "mkt_close_prob_draw": close_pd,
            "mkt_close_prob_away": close_pa,
            "mkt_steam_home": steam_h,
            "mkt_steam_draw": steam_d,
            "mkt_steam_away": steam_a,
            "mkt_ou25_over_prob": mkt_ou25_over_prob,
            "mkt_ah_line": mkt_ah_line,
        }
    )

    rich = rich.loc[:, list(RICH_CANONICAL_COLUMNS)]

    rich = rich.sort_values(["date", "kickoff_time", "match_id"], kind="mergesort").reset_index(
        drop=True
    )
    return rich


def discover_raw_epl_files(data_root: str | Path) -> dict[str, Path]:
    """Discover all raw E0_YYYY-YY.csv files under data_root (including staging)."""
    root = Path(data_root)
    season_files: dict[str, Path] = {}

    for path in sorted(root.glob("raw/E0_*.csv")):
        season = normalize_season(path.stem.replace("E0_", ""))
        season_files[season] = path

    # Also check staging folders for newer seasons (e.g. 2025-26)
    for path in sorted(root.glob("staging/*/raw/E0_*.csv")):
        season = normalize_season(path.stem.replace("E0_", ""))
        season_files[season] = path

    if not season_files:
        raise FileNotFoundError(f"No E0_*.csv files found under {root}")
    return dict(sorted(season_files.items()))


def build_rich_epl_dataset(
    data_root: str | Path,
    output_csv: str | Path | None = None,
    seasons: Iterable[str] | None = None,
) -> pd.DataFrame:
    """Build the multi-season Rich Match Table from raw Football-Data E0 CSVs."""
    discovered = discover_raw_epl_files(data_root)
    if seasons is not None:
        requested = [normalize_season(s) for s in seasons]
        missing = [s for s in requested if s not in discovered]
        if missing:
            raise FileNotFoundError(f"Missing raw E0 CSV for seasons: {', '.join(missing)}")
        selected_items = [(s, discovered[s]) for s in sorted(requested)]
    else:
        selected_items = list(discovered.items())

    frames: list[pd.DataFrame] = []
    for season, path in selected_items:
        raw_df = read_source_csv(path)
        rich_season = convert_raw_epl_season(raw_df, season)
        frames.append(rich_season)

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.sort_values(
        ["date", "kickoff_time", "match_id"], kind="mergesort"
    ).reset_index(drop=True)

    if combined["match_id"].duplicated().any():
        dups = combined.loc[combined["match_id"].duplicated(), "match_id"].tolist()[:5]
        raise DataValidationError(f"Duplicate match_ids detected in rich dataset: {dups}")

    if output_csv is not None:
        out_path = Path(output_csv)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        combined.to_csv(out_path, index=False)

    return combined
