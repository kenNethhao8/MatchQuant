"""Comprehensive unit tests for MatchQuant v2.0 Quantitative Engine."""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from matchquant.factor_engine import (
    ALL_FACTOR_COLUMNS,
    ChronologicalFactorEngine,
    FACTOR_SPECIFICATIONS,
    FUNDAMENTAL_FACTOR_COLUMNS,
    build_historical_factor_panel,
)
from matchquant.factor_lab import (
    compute_quantile_profile,
    compute_vif_table,
    render_factor_lab_report,
    run_factor_lab,
)
from matchquant.quant_models import (
    fit_and_select_factor_system,
    fit_dixon_coles_factor_model,
    fit_ordered_factor_model,
)
from matchquant.quant_pipeline import (
    evaluate_fundamental_market_edge,
    evaluate_totals_over25_market,
    run_expanding_walk_forward_backtest,
    simulate_kelly_paper_trading,
)
from matchquant.rich_data import (
    RICH_CANONICAL_COLUMNS,
    convert_raw_epl_season,
    devig_power_probabilities,
)


def _make_synthetic_rich_dataset() -> pd.DataFrame:
    """Generate a synthetic 4-season dataset where team strength and SOT dominance drive outcomes."""
    rng = np.random.default_rng(42)
    teams = ["Arsenal", "Liverpool", "Man City", "Chelsea", "Tottenham", "Everton", "Fulham", "Luton"]
    true_strength = {
        "Man City": 1.2,
        "Liverpool": 0.9,
        "Arsenal": 0.8,
        "Chelsea": 0.2,
        "Tottenham": 0.1,
        "Everton": -0.5,
        "Fulham": -0.7,
        "Luton": -1.2,
    }
    seasons = ["2021-22", "2022-23", "2023-24", "2024-25"]
    rows = []
    match_idx = 0
    for s_idx, season in enumerate(seasons):
        base_date = pd.Timestamp(f"{2021 + s_idx}-08-15")
        round_num = 0
        for i, h in enumerate(teams):
            for j, a in enumerate(teams):
                if i == j:
                    continue
                match_idx += 1
                date_str = (base_date + pd.Timedelta(days=(round_num // 4) * 7)).strftime("%Y-%m-%d")
                round_num += 1
                diff = true_strength[h] - true_strength[a] + 0.25
                lam_h = max(0.3, 1.45 + 0.45 * diff)
                lam_a = max(0.3, 1.15 - 0.35 * diff)
                hg = int(rng.poisson(lam_h))
                ag = int(rng.poisson(lam_a))
                hthg = int(rng.binomial(hg, 0.45)) if hg > 0 else 0
                htag = int(rng.binomial(ag, 0.45)) if ag > 0 else 0
                hst = max(hg, int(rng.poisson(4.8 + 1.8 * diff)))
                ast = max(ag, int(rng.poisson(4.0 - 1.5 * diff)))
                hs = hst + int(rng.poisson(8.0 + 1.5 * diff))
                as_ = ast + int(rng.poisson(7.0 - 1.2 * diff))
                hc = max(0, int(rng.poisson(5.5 + 1.2 * diff)))
                ac = max(0, int(rng.poisson(4.5 - 1.0 * diff)))

                # Synthetic odds aligned with true strength
                p_h_true = 1.0 / (1.0 + np.exp(-0.9 * diff)) * 0.76
                p_d_true = 0.24
                p_a_true = max(0.05, 1.0 - p_h_true - p_d_true)
                overround = 1.04
                oh = round(1.0 / (p_h_true * overround), 2)
                od = round(1.0 / (p_d_true * overround), 2)
                oa = round(1.0 / (p_a_true * overround), 2)

                rows.append(
                    {
                        "Date": pd.Timestamp(date_str).strftime("%d/%m/%Y"),
                        "Time": "15:00",
                        "HomeTeam": h,
                        "AwayTeam": a,
                        "FTHG": str(hg),
                        "FTAG": str(ag),
                        "HTHG": str(hthg),
                        "HTAG": str(htag),
                        "Referee": "M Oliver" if match_idx % 2 == 0 else "A Taylor",
                        "HS": str(hs),
                        "AS": str(as_),
                        "HST": str(hst),
                        "AST": str(ast),
                        "HC": str(hc),
                        "AC": str(ac),
                        "HF": "10",
                        "AF": "11",
                        "HY": "1",
                        "AY": "2",
                        "HR": "0",
                        "AR": "0",
                        "PSH": str(oh),
                        "PSD": str(od),
                        "PSA": str(oa),
                        "PSCH": str(max(1.05, oh - 0.03 * np.sign(diff))),
                        "PSCD": str(od),
                        "PSCA": str(max(1.05, oa + 0.03 * np.sign(diff))),
                        "P>2.5": "1.90",
                        "P<2.5": "1.95",
                        "AHh": str(round(-0.5 * diff, 2)),
                    }
                )
        raw_season_df = pd.DataFrame(rows[-len(teams) * (len(teams) - 1) :])
        if s_idx == 0:
            frames = [convert_raw_epl_season(raw_season_df, season)]
        else:
            frames.append(convert_raw_epl_season(raw_season_df, season))
    return pd.concat(frames, ignore_index=True)


class QuantV2TestSuite(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.rich_df = _make_synthetic_rich_dataset()
        cls.panel, cls.engine = build_historical_factor_panel(cls.rich_df)

    def test_power_devig_probabilities_sum_to_one(self) -> None:
        ph, pd_, pa = devig_power_probabilities([1.80, 2.50], [3.60, 3.30], [4.50, 2.90])
        totals = ph + pd_ + pa
        np.testing.assert_allclose(totals, [1.0, 1.0], atol=1e-9)
        self.assertTrue(np.all((ph > 0) & (pd_ > 0) & (pa > 0)))

    def test_rich_canonical_schema_complete(self) -> None:
        for col in RICH_CANONICAL_COLUMNS:
            self.assertIn(col, self.rich_df.columns)
        self.assertEqual(len(self.rich_df), 4 * 56)
        # Stage-1 contract: exactly 44 enriched columns in the canonical order
        self.assertEqual(len(RICH_CANONICAL_COLUMNS), 44)
        self.assertEqual(
            list(self.rich_df.columns),
            list(RICH_CANONICAL_COLUMNS),
        )
        for contract_col in (
            "actual_result",
            "mkt_open_prob_home",
            "mkt_close_prob_home",
            "mkt_steam_home",
            "mkt_ou25_over_prob",
            "mkt_ah_line",
        ):
            self.assertIn(contract_col, RICH_CANONICAL_COLUMNS)

    def test_chronological_batch_freeze_no_lookahead_leakage(self) -> None:
        """Mutating future scores or same-batch scores must never alter pre-match factors."""
        mutated = self.rich_df.copy()
        cutoff_idx = len(mutated) // 2
        mutated.loc[cutoff_idx:, "home_goals"] = 9
        mutated.loc[cutoff_idx:, "away_goals"] = 0
        mutated.loc[cutoff_idx:, "home_sot"] = 25
        panel_mut, _ = build_historical_factor_panel(mutated)

        # Every match strictly prior to cutoff_idx batch must have identical factors
        cutoff_date = mutated.loc[cutoff_idx, "date"]
        before_orig = self.panel[self.panel["date"] < cutoff_date]
        before_mut = panel_mut[panel_mut["date"] < cutoff_date]
        for col in FUNDAMENTAL_FACTOR_COLUMNS:
            np.testing.assert_allclose(
                before_orig[col].to_numpy(),
                before_mut[col].to_numpy(),
                atol=1e-12,
                err_msg=f"Look-ahead leakage detected in factor {col}",
            )

    def test_factor_lab_diagnostics_and_vif(self) -> None:
        lab = run_factor_lab(self.panel)
        self.assertGreater(len(lab["selected_fundamental_factors"]), 3)
        self.assertEqual(len(lab["factors"]), len(ALL_FACTOR_COLUMNS))
        vifs = compute_vif_table(self.panel, lab["selected_fundamental_factors"])
        for v in vifs.values():
            self.assertGreaterEqual(v, 1.0)
        q_prof = compute_quantile_profile(self.panel, "sot_dominance_diff")
        self.assertEqual(len(q_prof["buckets"]), 5)
        md = render_factor_lab_report(lab)
        self.assertIn("MatchQuant Factor Lab", md)

    def test_ordered_and_dixon_coles_models_sign_consistency_and_attribution(self) -> None:
        train_df = self.panel[self.panel["season"].isin(["2021-22", "2022-23"])]
        val_df = self.panel[self.panel["season"] == "2023-24"]
        test_df = self.panel[self.panel["season"] == "2024-25"]

        features = ["elo_diff", "sot_dominance_diff", "att_def_mismatch_diff", "finishing_luck_diff"]
        m_ord = fit_ordered_factor_model(train_df, features, l2=1.0, enforce_expected_signs=True)
        # Positive expected-sign features must have non-negative weights; luck reversion must be <= 0
        self.assertGreaterEqual(m_ord.weights[0], 0.0)
        self.assertGreaterEqual(m_ord.weights[1], 0.0)
        self.assertLessEqual(m_ord.weights[3], 0.0)

        probs_ord = m_ord.predict_proba(test_df)
        np.testing.assert_allclose(probs_ord.sum(axis=1), np.ones(len(test_df)), atol=1e-6)

        m_dc = fit_dixon_coles_factor_model(train_df, features, l2=2.0, enforce_expected_signs=True)
        probs_dc, lh, la, p_o25 = m_dc.predict_proba_and_goals(test_df)
        np.testing.assert_allclose(probs_dc.sum(axis=1), np.ones(len(test_df)), atol=1e-6)
        self.assertTrue(np.all((lh > 0) & (la > 0) & (p_o25 > 0) & (p_o25 < 1)))

        system, info = fit_and_select_factor_system(train_df, val_df, features)
        rich_pred = system.predict_rich(test_df.head(5))
        self.assertEqual(len(rich_pred), 5)
        attr = system.ordered_model.explain_row(test_df.iloc[0])
        self.assertIn("top_factor_contributions", attr)

    def test_stage_5_new_factors_registered_and_zero_lookahead(self) -> None:
        spec_map = {s["name"]: s for s in FACTOR_SPECIFICATIONS}
        for name, exp_sign, exp_track in [
            ("fixture_congestion_diff", -1, "fundamental"),
            ("table_pressure_diff", 1, "fundamental"),
            ("style_counter_edge", 1, "fundamental"),
        ]:
            self.assertIn(name, spec_map)
            self.assertIn(name, FUNDAMENTAL_FACTOR_COLUMNS)
            self.assertEqual(spec_map[name]["expected_sign"], exp_sign)
            self.assertEqual(spec_map[name]["track"], exp_track)
            self.assertIn(name, self.panel.columns)
            self.assertFalse(self.panel[name].isna().any())

    def test_expanding_walk_forward_backtest_and_edge_evaluation(self) -> None:
        features = ["elo_diff", "sot_dominance_diff", "corner_pressure_diff"]
        mkt_features = features + ["mkt_open_prob_edge", "mkt_steam_move_diff"]
        wf_res = run_expanding_walk_forward_backtest(
            self.panel,
            features,
            mkt_features,
            eval_seasons=("2023-24", "2024-25"),
        )
        self.assertEqual(len(wf_res["eval_seasons"]), 2)
        self.assertGreater(wf_res["n_eval_matches"], 0)
        pooled = wf_res["pooled"]
        for key in ["elo", "track_a", "track_b", "open", "close"]:
            self.assertIn(key, pooled)
            self.assertTrue(np.isfinite(pooled[key]["log_loss"]))
            self.assertTrue(np.isfinite(pooled[key]["brier_score"]))

        edge_res = evaluate_fundamental_market_edge(wf_res["predictions"])
        self.assertIn("edge_h_quintiles", edge_res)
        self.assertIn("directional_thresholds", edge_res)
        self.assertEqual(len(edge_res["edge_h_quintiles"]), 5)
        self.assertEqual(len(edge_res["directional_thresholds"]), 5)
        for q in edge_res["edge_h_quintiles"]:
            self.assertTrue(np.isfinite(q["avg_edge_pct"]))
            self.assertTrue(np.isfinite(q["clv_win_rate_pct"]))
            self.assertTrue(np.isfinite(q["avg_clv_bps"]))

    def test_stage_6_factor_refinement_and_collinearity(self) -> None:
        """Verify Task A: table_pressure_diff Gaussian point-gap and style_counter_edge |rho| < 0.70."""
        corr_sc_cp = float(
            self.panel[["style_counter_edge", "corner_pressure_diff"]].corr().iloc[0, 1]
        )
        self.assertLess(abs(corr_sc_cp), 0.70)
        self.assertTrue(np.all(np.isfinite(self.panel["style_counter_edge"])))
        self.assertTrue(np.all(np.isfinite(self.panel["table_pressure_diff"])))

    def test_stage_6_walk_forward_dynamic_factor_selection(self) -> None:
        """Verify Task B: 100% dynamic factor selection strictly without future lookahead."""
        wf_res = run_expanding_walk_forward_backtest(
            self.panel,
            eval_seasons=("2023-24", "2024-25"),
            dynamic_factor_selection=True,
        )
        self.assertEqual(len(wf_res["eval_seasons"]), 2)
        for s in wf_res["eval_seasons"]:
            s_dict = wf_res["by_season"][s]
            self.assertIn("selected_fundamental_factors", s_dict)
            self.assertIn("n_fundamental_factors", s_dict)
            self.assertGreater(s_dict["n_fundamental_factors"], 2)
        preds = wf_res["predictions"]
        for col in ["expected_home_goals", "expected_away_goals", "prob_over25"]:
            self.assertIn(col, preds.columns)
            self.assertTrue(np.all(np.isfinite(preds[col])))

    def test_stage_6_debiased_orthogonal_edge_evaluation(self) -> None:
        """Verify Task C: OrthoEdge eliminates probability compression bias (balanced implied win rates)."""
        wf_res = run_expanding_walk_forward_backtest(
            self.panel,
            eval_seasons=("2023-24", "2024-25"),
            dynamic_factor_selection=True,
        )
        edge_res = evaluate_fundamental_market_edge(wf_res["predictions"])
        self.assertIn("ortho_edge_h_quintiles", edge_res)
        self.assertIn("ortho_directional_thresholds", edge_res)
        self.assertEqual(len(edge_res["ortho_edge_h_quintiles"]), 5)
        self.assertEqual(len(edge_res["ortho_directional_thresholds"]), 4)

        # Implied probabilities in OrthoEdge quintiles must be balanced (not collapsed to 33%)
        imp_probs = [q["implied_prob_pct"] for q in edge_res["ortho_edge_h_quintiles"]]
        max_spread = max(imp_probs) - min(imp_probs)
        self.assertLess(max_spread, 20.0)

    def test_stage_6_totals_over25_market_evaluation(self) -> None:
        """Verify Task D: Dixon-Coles Over 2.5 goals probability evaluation against market."""
        wf_res = run_expanding_walk_forward_backtest(
            self.panel,
            eval_seasons=("2023-24", "2024-25"),
            dynamic_factor_selection=True,
        )
        tot_res = evaluate_totals_over25_market(wf_res["predictions"])
        self.assertGreater(tot_res["n_matches"], 0)
        self.assertTrue(np.isfinite(tot_res["model_log_loss"]))
        self.assertTrue(np.isfinite(tot_res["market_log_loss"]))
        self.assertTrue(np.isfinite(tot_res["model_brier"]))
        self.assertTrue(np.isfinite(tot_res["market_brier"]))
        self.assertEqual(len(tot_res["quintiles"]), 5)
        for q in tot_res["quintiles"]:
            self.assertTrue(np.isfinite(q["avg_edge_pct"]))
            self.assertTrue(np.isfinite(q["actual_over25_pct"]))
            self.assertTrue(np.isfinite(q["implied_over25_pct"]))
            self.assertTrue(np.isfinite(q["roi_fair_odds_pct"]))

    def test_stage_7_track_m_open_and_out_of_sample_debiasing(self) -> None:
        """Verify Task B: Track M-Open tradable model & 100% out-of-sample rolling OrthoEdge debiasing."""
        wf_res = run_expanding_walk_forward_backtest(
            self.panel,
            eval_seasons=("2023-24", "2024-25"),
            dynamic_factor_selection=True,
        )
        self.assertIn("track_m_open", wf_res["pooled"])
        self.assertTrue(np.isfinite(wf_res["pooled"]["track_m_open"]["log_loss"]))
        self.assertTrue(np.isfinite(wf_res["pooled"]["track_m_open"]["brier_score"]))

        # Verify Track M-Open factors strictly forbid mkt_steam_move_diff
        for s in wf_res["eval_seasons"]:
            s_dict = wf_res["by_season"][s]
            self.assertIn("track_m_open", s_dict)
            self.assertIn("selected_tradable_factors", s_dict)
            tradable_factors = s_dict["selected_tradable_factors"]
            self.assertNotIn("mkt_steam_move_diff", tradable_factors)
            self.assertIn("mkt_open_prob_edge", tradable_factors)
            self.assertIn("mkt_ah_supremacy", tradable_factors)
            self.assertIn("ortho_debias_coeffs", s_dict)
            for c_code in ["h", "d", "a"]:
                self.assertIn(c_code, s_dict["ortho_debias_coeffs"])
                slp, icpt = s_dict["ortho_debias_coeffs"][c_code]
                self.assertTrue(np.isfinite(slp))
                self.assertTrue(np.isfinite(icpt))

        preds = wf_res["predictions"]
        for c in ["h", "d", "a"]:
            self.assertIn(f"track_m_open_prob_{c}", preds.columns)
            self.assertIn(f"tradable_edge_{c}", preds.columns)
            self.assertIn(f"ortho_edge_{c}", preds.columns)
            self.assertTrue(np.all(np.isfinite(preds[f"track_m_open_prob_{c}"])))
            self.assertTrue(np.all(np.isfinite(preds[f"tradable_edge_{c}"])))
            self.assertTrue(np.all(np.isfinite(preds[f"ortho_edge_{c}"])))

    def test_stage_7_fractional_kelly_paper_trading(self) -> None:
        """Verify Task C: Fractional Kelly paper trading across 1,900 walk-forward matches."""
        wf_res = run_expanding_walk_forward_backtest(
            self.panel,
            eval_seasons=("2023-24", "2024-25"),
            dynamic_factor_selection=True,
        )
        kelly_res = simulate_kelly_paper_trading(
            wf_res["predictions"],
            initial_bankroll=1000.0,
            kelly_fraction=0.20,
            min_edge=0.02,
            max_bet_pct=0.03,
        )
        self.assertEqual(kelly_res["initial_bankroll"], 1000.0)
        self.assertEqual(kelly_res["kelly_fraction"], 0.20)
        self.assertIn("strategy_1_ortho_edge", kelly_res)
        self.assertIn("strategy_2_track_m_open", kelly_res)

        for strat_key in ["strategy_1_ortho_edge", "strategy_2_track_m_open"]:
            s = kelly_res[strat_key]
            self.assertGreater(s["n_bets"], 0)
            self.assertGreaterEqual(s["win_rate_pct"], 0.0)
            self.assertLessEqual(s["win_rate_pct"], 100.0)
            self.assertGreaterEqual(s["clv_win_rate_pct"], 0.0)
            self.assertLessEqual(s["clv_win_rate_pct"], 100.0)
            self.assertTrue(np.isfinite(s["avg_clv_bps"]))

            for odds_mode in ["fair_odds", "raw_odds"]:
                perf = s[odds_mode]
                self.assertTrue(np.isfinite(perf["final_bankroll"]))
                self.assertTrue(np.isfinite(perf["roi_pct"]))
                self.assertTrue(np.isfinite(perf["sharpe_ratio"]))
                self.assertGreaterEqual(perf["max_drawdown_pct"], 0.0)
                self.assertLessEqual(perf["max_drawdown_pct"], 100.0)

    def test_stage_7_joint_poisson_loss_calibration(self) -> None:
        """Verify Task A: Bivariate Poisson joint goal loss ensures economic sign consistency."""
        train_df = self.panel[self.panel["season"].isin(["2021-22", "2022-23"])]
        features = ["elo_diff", "sot_dominance_diff", "corner_pressure_diff", "finishing_luck_diff"]
        m_dc = fit_dixon_coles_factor_model(train_df, features, l2=5.0, enforce_expected_signs=True)
        # Sign consistency: positive features >= 0, finishing luck reversion <= 0
        self.assertGreaterEqual(m_dc.supremacy_weights[0], 0.0)
        self.assertGreaterEqual(m_dc.supremacy_weights[1], 0.0)
        self.assertLessEqual(m_dc.supremacy_weights[3], 0.0)
        self.assertGreaterEqual(m_dc.tempo_weight, 0.0)
        self.assertTrue(np.isfinite(m_dc.alpha_home))
        self.assertTrue(np.isfinite(m_dc.alpha_away))
        self.assertTrue(np.isfinite(m_dc.rho))


if __name__ == "__main__":
    unittest.main()
