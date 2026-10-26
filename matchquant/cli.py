"""Clean CLI for MatchQuant v2.0 Quantitative Football Engine."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from . import __version__
from .backtest import run_models
from .epl import DEFAULT_EPL_SEASONS, download_epl, prepare_epl
from .factor_engine import build_historical_factor_panel
from .factor_lab import render_factor_lab_report, run_factor_lab
from .io import file_sha256, load_matches, write_json, write_predictions
from .models import MODEL_NAMES
from .quant_models import _clip_probs, fit_and_select_factor_system
from .quant_pipeline import run_quant_research_pipeline
from .rich_data import build_rich_epl_dataset
from .schema import DataValidationError
from .selection import load_candidates, parse_validation_folds, run_nested_selection


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="matchquant",
        description="MatchQuant v2.0: Quantitative Football Factor Engineering & Modeling Engine.",
    )
    parser.add_argument("--version", action="version", version=f"matchquant {__version__}")
    subparsers = parser.add_subparsers(dest="command")

    # 1. rich-prepare: Unlock all 44 pitch & market columns from raw E0 CSVs
    rich_prep = subparsers.add_parser(
        "rich-prepare",
        help="Build full-spectrum Rich Match Table from raw E0 CSVs (shots, SOT, corners, HT, cards, referee, odds).",
    )
    rich_prep.add_argument(
        "--data-root",
        default=r"D:\Codex\matchquant_data",
        help="Root directory containing raw/E0_*.csv and optional staging/*/raw/E0_*.csv.",
    )
    rich_prep.add_argument(
        "--output",
        default=r"D:\Codex\matchquant_data\processed\epl_rich_canonical.csv",
        help="Output CSV path for the Rich Match Table.",
    )
    rich_prep.add_argument("--seasons", nargs="+", default=None, help="Optional season filter.")

    # 2. factor-lab: Single-factor IC, Orthogonal Alpha over Elo, 5-quantile monotonicity & VIF
    flab = subparsers.add_parser(
        "factor-lab",
        help="Run Alphalens-style single-factor IC, Elo-orthogonal residual t-stats, 5-quantile & VIF diagnostics.",
    )
    flab.add_argument(
        "--input",
        default=r"D:\Codex\matchquant_data\processed\epl_rich_canonical.csv",
        help="Path to Rich Match Table CSV (or built automatically from --data-root if missing).",
    )
    flab.add_argument(
        "--data-root",
        default=r"D:\Codex\matchquant_data",
        help="Fallback raw data root if --input does not yet exist.",
    )
    flab.add_argument(
        "--output-dir",
        default=r"D:\Codex\matchquant_runs\factor_lab_v2",
        help="Directory to write factor_lab_report.md and factor_lab_metrics.json.",
    )

    # 3. quant-research: Full end-to-end Factor Lab + Dual-Track Modeling + Multi-Benchmark Validation
    qres = subparsers.add_parser(
        "quant-research",
        help="Run end-to-end factor engineering, Factor Lab screening, Dual-Track (Fundamental & Market) modeling, and out-of-sample evaluation.",
    )
    qres.add_argument(
        "--input",
        default=r"D:\Codex\matchquant_data\processed\epl_rich_canonical.csv",
        help="Path to Rich Match Table CSV.",
    )
    qres.add_argument(
        "--data-root",
        default=r"D:\Codex\matchquant_data",
        help="Fallback raw data root if --input does not yet exist.",
    )
    qres.add_argument(
        "--output-dir",
        default=r"D:\Codex\matchquant_runs\quant_v2_research",
        help="Directory to write reports, metrics, weights, and out-of-sample predictions.",
    )

    # 4. matchday-radar: Pre-match multi-model quantitative decision radar for the latest round
    radar_parser = subparsers.add_parser(
        "matchday-radar",
        help="Display pre-match quantitative radar for the latest round of matches (Elo, Track A, Track M-Open, xG, OrthoEdge, Top Drivers).",
    )
    radar_parser.add_argument(
        "--input",
        default=r"D:\Codex\matchquant_data\processed\epl_rich_canonical.csv",
        help="Path to Rich Match Table CSV.",
    )
    radar_parser.add_argument(
        "--data-root",
        default=r"D:\Codex\matchquant_data",
        help="Fallback raw data root if --input does not yet exist.",
    )
    radar_parser.add_argument(
        "--top-n",
        type=int,
        default=10,
        help="Number of latest matches to include in radar (default: 10).",
    )

    # 5. Legacy M1/M2/M3 commands preserved cleanly
    run_parser = subparsers.add_parser("run", help="Run baseline chronological walk-forward backtest.")
    run_parser.add_argument("--config", help="Path to JSON configuration file.")
    run_parser.add_argument("--input", help="Path to canonical matches CSV.")
    run_parser.add_argument("--test-start", dest="test_start", help="Inclusive test start date.")
    run_parser.add_argument("--test-end", dest="test_end", help="Exclusive test end date.")
    run_parser.add_argument("--model", choices=(*MODEL_NAMES, "all"), help="Baseline model.")
    run_parser.add_argument("--season-mode", dest="season_mode", choices=("carry", "reset"))
    run_parser.add_argument("--output-dir", dest="output_dir", help="Output directory.")

    dl_parser = subparsers.add_parser("epl-download", help="Download raw Football-Data.co.uk E0 CSVs.")
    dl_parser.add_argument("--seasons", nargs="+", default=list(DEFAULT_EPL_SEASONS))
    dl_parser.add_argument("--data-dir", default=r"D:\Codex\matchquant_data")

    prep_parser = subparsers.add_parser("epl-prepare", help="Prepare basic 7-column M1 canonical CSV.")
    prep_parser.add_argument("--seasons", nargs="+", default=list(DEFAULT_EPL_SEASONS))
    prep_parser.add_argument("--data-dir", default=r"D:\Codex\matchquant_data")
    prep_parser.add_argument("--output", default=None)

    sel_parser = subparsers.add_parser("select", help="Run nested chronological baseline selection.")
    sel_parser.add_argument("--input", required=True)
    sel_parser.add_argument("--candidate-config", default="examples/m3_candidates.json")
    sel_parser.add_argument("--validation-fold", dest="validation_folds", action="append", required=True)
    sel_parser.add_argument("--final-test-start", required=True)
    sel_parser.add_argument("--final-test-end", default=None)
    sel_parser.add_argument("--output-root", default=r"D:\Codex\matchquant_runs")
    sel_parser.add_argument("--output-dir", default=None)

    return parser


def _load_or_build_rich(input_csv: str, data_root: str) -> pd.DataFrame:
    in_path = Path(input_csv)
    if in_path.exists():
        return pd.read_csv(in_path)
    return build_rich_epl_dataset(data_root, output_csv=in_path)


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.command:
        parser.print_help()
        return 1

    try:
        if args.command == "rich-prepare":
            df = build_rich_epl_dataset(args.data_root, output_csv=args.output, seasons=args.seasons)
            print(
                f"Rich dataset built: rows={len(df)} cols={len(df.columns)} "
                f"seasons={df['season'].min()}..{df['season'].max()} output={args.output}"
            )
            return 0

        if args.command == "factor-lab":
            rich_df = _load_or_build_rich(args.input, args.data_root)
            panel, _ = build_historical_factor_panel(rich_df)
            lab = run_factor_lab(panel)
            out_dir = Path(args.output_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            write_json(lab, out_dir / "factor_lab_metrics.json")
            report_path = out_dir / "factor_lab_report.md"
            report_path.write_text(render_factor_lab_report(lab), encoding="utf-8")
            print(
                f"Factor Lab complete: analyzed={lab['n_matches_analyzed']} matches | "
                f"selected_fundamental={len(lab['selected_fundamental_factors'])} | "
                f"report={report_path}"
            )
            return 0

        if args.command == "quant-research":
            rich_df = _load_or_build_rich(args.input, args.data_root)
            bundle = run_quant_research_pipeline(rich_df, output_dir=args.output_dir)
            test_eval = bundle["test_evaluation"]["overall"]
            elo_ll = test_eval["1_baseline_elo"]["log_loss"]
            ta_ll = test_eval["4_track_a_fundamental_ensemble"]["log_loss"]
            tmo_ll = test_eval.get("5_track_m_open_tradable", {}).get("log_loss", float("nan"))
            tb_ll = test_eval["6_track_b_market_augmented"]["log_loss"]
            wf_pooled = bundle["walk_forward_evaluation"]["pooled"]
            wf_elo = wf_pooled["elo"]["log_loss"]
            wf_ta = wf_pooled["track_a"]["log_loss"]
            wf_tmo = wf_pooled.get("track_m_open", {}).get("log_loss", float("nan"))
            wf_tb = wf_pooled["track_b"]["log_loss"]
            print(
                f"Quant Research complete: matches={bundle['summary']['total_matches']} | "
                f"Holdout LogLoss: Elo={elo_ll:.5f}, TrackA={ta_ll:.5f} ({ta_ll - elo_ll:+.5f}), "
                f"TrackM-Open={tmo_ll:.5f} ({tmo_ll - elo_ll:+.5f}), "
                f"TrackB={tb_ll:.5f} ({tb_ll - elo_ll:+.5f}) | "
                f"Walk-Forward (1900m) LogLoss: Elo={wf_elo:.5f}, TrackA={wf_ta:.5f} ({wf_ta - wf_elo:+.5f}), "
                f"TrackM-Open={wf_tmo:.5f} ({wf_tmo - wf_elo:+.5f}), "
                f"TrackB={wf_tb:.5f} ({wf_tb - wf_elo:+.5f}) | "
                f"output_dir={args.output_dir}"
            )
            return 0

        if args.command == "matchday-radar":
            rich_df = _load_or_build_rich(args.input, args.data_root)
            panel, _ = build_historical_factor_panel(rich_df)
            n_target = max(1, int(args.top_n))
            target_df = panel.tail(n_target).copy()

            all_seasons = sorted(panel["season"].unique())
            target_season = str(target_df["season"].iloc[-1])
            k_idx = all_seasons.index(target_season) if target_season in all_seasons else (len(all_seasons) - 1)

            if k_idx >= 2:
                train_seasons = all_seasons[: k_idx - 1]
                val_season = all_seasons[k_idx - 1]
            else:
                train_seasons = all_seasons[:k_idx]
                val_season = all_seasons[k_idx]

            discovery_seasons = all_seasons[:k_idx] if k_idx > 0 else all_seasons
            lab = run_factor_lab(panel, discovery_seasons=discovery_seasons)
            dyn_fund = list(lab["selected_fundamental_factors"])
            dyn_m_open = dyn_fund + [c for c in ["mkt_open_prob_edge", "mkt_ah_supremacy"] if c in panel.columns]

            train_df = panel[panel["season"].isin(set(train_seasons))].copy()
            val_df = panel[panel["season"] == val_season].copy()

            sys_a, _ = fit_and_select_factor_system(train_df, val_df, dyn_fund, track="fundamental")
            sys_m_open, _ = fit_and_select_factor_system(train_df, val_df, dyn_m_open, track="tradable_m_open")

            val_ta_p = sys_a.predict_proba(val_df)
            val_open_cols = ["mkt_open_prob_home", "mkt_open_prob_draw", "mkt_open_prob_away"]
            val_open_p = _clip_probs(val_df[val_open_cols].to_numpy(dtype=float))
            ortho_coeffs = {}
            for c_idx, c_code in enumerate(["h", "d", "a"]):
                v_raw = val_ta_p[:, c_idx] - val_open_p[:, c_idx]
                v_open = val_open_p[:, c_idx]
                mask = np.isfinite(v_raw) & np.isfinite(v_open)
                slp, icpt = np.polyfit(v_open[mask], v_raw[mask], 1) if mask.sum() >= 20 else (0.0, 0.0)
                ortho_coeffs[c_code] = (float(slp), float(icpt))

            ta_p = sys_a.predict_proba(target_df)
            tmo_p = sys_m_open.predict_proba(target_df)
            _, lam_h, lam_a, p_o25 = sys_a.dixon_coles_model.predict_proba_and_goals(target_df)
            target_open_p = _clip_probs(target_df[val_open_cols].to_numpy(dtype=float))

            print("\n" + "=" * 140)
            print("  MatchQuant v2.0 赛前实盘量化雷达仪（Matchday Quantitative Decision Radar）")
            print("=" * 140)
            header = f"{'对阵双方 (开球日期)':<35} | {'Elo 胜率 (H/D/A)':<18} | {'Track A 基本面':<18} | {'Track M-Open':<18} | {'期望进球 (xG)':<15} | {'大球概率':<8} | {'去偏最优 Alpha':<16} | 核心非 Elo 驱动因子 (Top 3)"
            print(header)
            print("-" * 155)

            for idx, (_, row) in enumerate(target_df.iterrows()):
                match_label = f"{row['home_team']} vs {row['away_team']} ({str(row['date'])[:10]})"
                elo_str = f"{row['elo_prob_h']*100:.1f}/{row['elo_prob_d']*100:.1f}/{row['elo_prob_a']*100:.1f}%"
                ta_str = f"{ta_p[idx, 0]*100:.1f}/{ta_p[idx, 1]*100:.1f}/{ta_p[idx, 2]*100:.1f}%"
                tmo_str = f"{tmo_p[idx, 0]*100:.1f}/{tmo_p[idx, 1]*100:.1f}/{tmo_p[idx, 2]*100:.1f}%"
                xg_str = f"{lam_h[idx]:.2f} - {lam_a[idx]:.2f}"
                o25_str = f"{p_o25[idx]*100:.1f}%"

                best_edge_dir = "None"
                best_edge_val = -999.0
                for c_idx, c_code in enumerate(["h", "d", "a"]):
                    raw_e = ta_p[idx, c_idx] - target_open_p[idx, c_idx]
                    slp, icpt = ortho_coeffs[c_code]
                    ortho_e = raw_e - (icpt + slp * target_open_p[idx, c_idx])
                    if ortho_e > best_edge_val:
                        best_edge_val = ortho_e
                        best_edge_dir = c_code.upper()
                if best_edge_val >= 0.015:
                    edge_str = f"{best_edge_dir} ({best_edge_val*100:+.1f}%)"
                else:
                    edge_str = f"Neutral ({best_edge_val*100:+.1f}%)"

                attr = sys_a.ordered_model.explain_row(row)
                top_drivers = [
                    d for d in attr["top_factor_contributions"]
                    if d["factor"] != "elo_diff" and abs(d["marginal_prob_h_pct"]) >= 0.10
                ][:3]
                drivers_str = " ; ".join(f"{d['factor']}({d['marginal_prob_h_pct']:+.1f}%)" for d in top_drivers) or "基本面与Elo一致"

                print(f"{match_label:<35} | {elo_str:<18} | {ta_str:<18} | {tmo_str:<18} | {xg_str:<15} | {o25_str:<8} | {edge_str:<16} | {drivers_str}")

            print("=" * 155 + "\n")
            return 0

        if args.command == "run":
            cfg: dict[str, Any] = {}
            if args.config:
                cfg = json.loads(Path(args.config).read_text(encoding="utf-8"))
            input_path = args.input or cfg.get("input")
            test_start = args.test_start or cfg.get("test_start")
            test_end = args.test_end or cfg.get("test_end")
            model_arg = args.model or cfg.get("model", "all")
            season_mode = args.season_mode or cfg.get("season_mode", "carry")
            output_dir = Path(args.output_dir or cfg.get("output_dir", "outputs"))
            if not input_path or not test_start:
                raise ValueError("Both --input and --test-start are required")
            matches = load_matches(input_path)
            model_names = MODEL_NAMES if model_arg == "all" else (model_arg,)
            runs = run_models(
                matches,
                model_names=model_names,
                test_start=test_start,
                test_end=test_end,
                season_mode=season_mode,
                model_configs=cfg.get("model_configs"),
            )
            predictions = pd.concat([r.predictions for r in runs.values()], ignore_index=True)
            output_dir.mkdir(parents=True, exist_ok=True)
            write_predictions(predictions, output_dir / "predictions.csv")
            write_json(
                {
                    "input_path": str(Path(input_path).resolve()),
                    "input_sha256": file_sha256(input_path),
                    "models": {
                        name: r.metrics
                        for name, r in runs.items()
                    },
                },
                output_dir / "metrics.json",
            )
            print(f"Baseline backtest complete: output={output_dir}")
            return 0

        if args.command == "epl-download":
            download_epl(args.seasons, args.data_dir)
            return 0

        if args.command == "epl-prepare":
            prepare_epl(args.seasons, args.data_dir, args.output)
            return 0

        if args.command == "select":
            from datetime import datetime, timezone

            matches = load_matches(args.input)
            candidates, _ = load_candidates(args.candidate_config)
            folds = parse_validation_folds(args.validation_folds)
            sel = run_nested_selection(
                matches,
                candidates,
                folds,
                final_test_start=args.final_test_start,
                final_test_end=args.final_test_end,
            )
            if args.output_dir:
                out_dir = Path(args.output_dir)
            else:
                stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
                out_dir = Path(args.output_root) / f"m3_select_{stamp}"
            out_dir.mkdir(parents=True, exist_ok=True)
            write_json(
                {
                    "schema_version": sel["schema_version"],
                    "validation_folds": sel["validation_folds"],
                    "final_test": sel["final_test"],
                    "selection_rule": sel["selection_rule"],
                    "data_provenance": {"status": "missing_manifest", "input_sha256": file_sha256(args.input)},
                    "final_test_metrics": sel["final_metrics"],
                },
                out_dir / "run_config.json",
            )
            write_json(
                {
                    "candidates": sel["candidates"],
                    "ranking": sel["ranking"],
                    "selected_candidate": sel["selected_candidate"],
                },
                out_dir / "validation_metrics.json",
            )
            write_json(sel["final_metrics"], out_dir / "final_metrics.json")
            write_predictions(sel["final_run"].predictions, out_dir / "predictions.csv")
            sel["calibration_table"].to_csv(out_dir / "calibration_reliability.csv", index=False)
            print(f"Selection complete: selected={sel['selected_candidate']['candidate_name']} output={out_dir}")
            return 0

    except (DataValidationError, FileNotFoundError, ValueError, OSError) as exc:
        print(f"matchquant error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
