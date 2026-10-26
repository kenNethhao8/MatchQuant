#!/usr/bin/env python3
"""Prepare the small, public-facing dataset used by the MatchQuant showcase.

The source reports remain outside this Site checkout. This exporter keeps only
aggregate research metrics, one historical matchday, factor quintiles, and a
sampled Kelly equity curve; it does not copy the full prediction or trade files.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def round_num(value: Any, digits: int = 6) -> Any:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return round(value, digits)
    return value


def equity_series(strategy: dict[str, Any], max_points: int = 120) -> list[dict[str, Any]]:
    """Rebuild both cash paths in source trade order and retain a compact curve."""
    trades = strategy["trades"]
    if not trades:
        return []

    balances = {
        "fair": float(strategy["fair_odds"]["initial_bankroll"]),
        "raw": float(strategy["raw_odds"]["initial_bankroll"]),
    }
    start = {"date": trades[0]["date"], "index": 0, **balances}
    points = [start]
    stride = max(1, (len(trades) + max_points - 1) // max_points)

    for index, trade in enumerate(trades, start=1):
        for mode in ("fair", "raw"):
            stake = float(trade[f"bet_{mode}"])
            odds = float(trade[f"{mode}_odds"])
            profit = stake * (odds - 1.0) if trade["is_win"] else -stake
            balances[mode] += profit

        if index % stride == 0 or index == len(trades):
            points.append({
                "date": trade["date"],
                "index": index,
                "fair": round_num(balances["fair"], 4),
                "raw": round_num(balances["raw"], 4),
            })

    for mode in ("fair", "raw"):
        reported = float(strategy[f"{mode}_odds"]["final_bankroll"])
        if abs(balances[mode] - reported) > 1e-7:
            raise ValueError(f"{mode} equity reconstruction does not match the report")
    return points


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--research-run", required=True, type=Path)
    parser.add_argument("--output", type=Path, default=Path("dist/data/showcase-data.json"))
    args = parser.parse_args()

    research_run = args.research_run.resolve()
    quant = read_json(research_run / "quant_evaluation_summary.json")
    factor_lab = read_json(research_run / "factor_lab_metrics.json")

    with (research_run / "walk_forward_predictions.csv").open(
        "r", encoding="utf-8-sig", newline=""
    ) as stream:
        csv_rows = list(csv.DictReader(stream))
    if not csv_rows:
        raise ValueError("walk_forward_predictions.csv contains no rows")

    latest_date = max(row["date"] for row in csv_rows)
    latest_round = [row for row in csv_rows if row["date"] == latest_date]
    sample_predictions = quant.get("sample_predictions", [])
    sample_by_fixture = {
        (row["date"], row["home_team"], row["away_team"]): row
        for row in sample_predictions
    }
    csv_fixtures = {
        (row["date"], row["home_team"], row["away_team"]): row
        for row in latest_round
    }
    if set(sample_by_fixture) != set(csv_fixtures):
        raise ValueError("The sampled prediction fixtures do not match the latest CSV round")

    matchday = []
    for fixture in sample_predictions:
        csv_row = csv_fixtures[(fixture["date"], fixture["home_team"], fixture["away_team"])]
        score = f"{csv_row['home_goals']}-{csv_row['away_goals']}"
        if fixture["actual_score"].split(" ", 1)[0] != score:
            raise ValueError(f"Score mismatch for {fixture['home_team']} vs {fixture['away_team']}")
        expected = fixture["expected_goals"]
        matchday.append({
            "date": fixture["date"],
            "season": fixture["season"],
            "home_team": fixture["home_team"],
            "away_team": fixture["away_team"],
            "actual_score": fixture["actual_score"],
            "elo": fixture["elo_probs"],
            "track_a": fixture["track_a_fundamental_probs"],
            "track_m_open": fixture["track_m_open_tradable_probs"],
            "track_b": fixture["track_b_market_augmented_probs"],
            "expected_home_goals": expected["home_xg_proxy"],
            "expected_away_goals": expected["away_xg_proxy"],
            "prob_over25": expected["prob_over25"],
            "alpha_shift_home_pct": fixture["alpha_shift_home_pct"],
            "drivers": [
                {
                    "factor": driver["factor"],
                    "family": driver["family"],
                    "marginal_prob_h_pct": driver["marginal_prob_h_pct"],
                }
                for driver in fixture.get("top_non_elo_drivers", [])[:3]
            ],
        })

    selected_names = quant["summary"]["selected_fundamental_factors"]
    factor_by_name = {factor["name"]: factor for factor in factor_lab["factors"]}
    selected_factors = []
    for name in selected_names:
        factor = factor_by_name[name]
        profile = factor.get("quantile_profile", {})
        selected_factors.append({
            "name": name,
            "family": factor["family"],
            "track": factor["track"],
            "status": factor["status"],
            "description": factor.get("description", ""),
            "raw_ic_gd": factor.get("raw_ic_gd"),
            "orthogonal_tstat": factor.get("orthogonal_tstat"),
            "ic_ir": factor.get("ic_ir"),
            "monotonicity": profile.get("monotonicity"),
            "vif_selected": factor.get("vif_selected"),
            "quantiles": [
                {
                    "quantile": bucket["quantile"],
                    "count": bucket["count"],
                    "home_win_rate": bucket["home_win_rate"],
                    "draw_rate": bucket["draw_rate"],
                    "away_win_rate": bucket["away_win_rate"],
                    "avg_goal_diff": bucket["avg_goal_diff"],
                }
                for bucket in profile.get("buckets", [])
            ],
        })

    rejected_names = ("second_half_surge_diff", "discipline_risk_diff")
    rejected_examples = []
    for name in rejected_names:
        factor = factor_by_name[name]
        rejected_examples.append({
            "name": name,
            "description": factor.get("description", ""),
            "raw_ic_gd": factor.get("raw_ic_gd"),
            "orthogonal_tstat": factor.get("orthogonal_tstat"),
            "corr_with_elo": factor.get("corr_with_elo"),
            "status": factor["status"],
            "selection_reason": factor.get("selection_reason", ""),
        })

    walk_forward = quant["walk_forward_evaluation"]
    season_rows = []
    for season in walk_forward["eval_seasons"]:
        models = walk_forward["by_season"][season]
        season_rows.append({
            "season": season,
            "matches": models["elo"]["n_matches"],
            "elo": models["elo"],
            "track_a": models["track_a"],
            "track_m_open": models["track_m_open"],
        })

    strategy = quant["kelly_paper_trading"]["strategy_2_track_m_open"]
    kelly = {
        "matches_total": quant["kelly_paper_trading"]["n_matches_total"],
        "initial_bankroll": quant["kelly_paper_trading"]["initial_bankroll"],
        "kelly_fraction": quant["kelly_paper_trading"]["kelly_fraction"],
        "minimum_edge": quant["kelly_paper_trading"]["min_edge"],
        "max_bet_pct": quant["kelly_paper_trading"]["max_bet_pct"],
        "n_bets": strategy["n_bets"],
        "bet_win_rate_pct": strategy["win_rate_pct"],
        "clv_win_rate_pct": strategy["clv_win_rate_pct"],
        "avg_clv_bps": strategy["avg_clv_bps"],
        "fair_odds": strategy["fair_odds"],
        "raw_odds": strategy["raw_odds"],
        "equity_curve": equity_series(strategy),
    }

    data = {
        "snapshot": {
            "last_match_date": latest_date,
            "matchday_match_count": len(latest_round),
            "total_matches": quant["summary"]["total_matches"],
            "train_matches": quant["summary"]["train_matches"],
            "validation_matches": quant["summary"]["val_matches"],
            "holdout_matches": quant["summary"]["test_matches"],
            "train_seasons": quant["summary"]["train_seasons"],
            "validation_seasons": quant["summary"]["val_seasons"],
            "holdout_seasons": quant["summary"]["test_seasons"],
            "walk_forward_matches": walk_forward["n_eval_matches"],
            "walk_forward_seasons": walk_forward["eval_seasons"],
            "factor_lab_matches": factor_lab["n_matches_analyzed"],
            "factor_lab_seasons": factor_lab["seasons_analyzed"],
            "factor_candidate_count": len(factor_lab["factors"]),
            "fundamental_selected_count": len(selected_factors),
        },
        "walk_forward": {
            "seasons": season_rows,
            "pooled": {
                name: walk_forward["pooled"][name]
                for name in ("elo", "track_a", "track_m_open", "open", "close")
            },
        },
        "matchday": matchday,
        "factors": selected_factors,
        "rejected_examples": rejected_examples,
        "kelly": kelly,
        "sources": [
            "quant_evaluation_summary.json",
            "factor_lab_metrics.json",
            "walk_forward_predictions.csv",
        ],
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(data, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(f"Wrote {args.output} ({len(matchday)} historical fixtures, {len(selected_factors)} selected factors)")


if __name__ == "__main__":
    main()
