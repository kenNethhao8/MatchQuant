"""Layer 4 & 5: Domain-Specific Football Quantitative Models & Factor Attribution.

Replaces the naive unordered 48-column Softmax with models that respect the
true mathematical structure of football matches:
  1. FactorDixonColesModel: Factor-driven Poisson goal rates (lambda_H, lambda_A)
     with Dixon-Coles (1997) low-score correlation parameter rho for 0-0 / 1-1 draws.
  2. OrderedFactorModel: Proportional-odds ordered logit model (Away < Draw < Home)
     with tempo-modulated draw boundary width and economic sign consistency.
  3. FundamentalEnsembleModel (Track A): Pure pitch/tactical/schedule factor model
     (zero betting odds used as inputs) + per-factor marginal attribution.
  4. MarketAugmentedModel (Track B): Fuses fundamental pitch factors with opening
     market consensus and smart-money steam (opening-to-closing line movement).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit

from .factor_engine import FACTOR_SPECIFICATIONS, FUNDAMENTAL_FACTOR_COLUMNS, MARKET_FACTOR_COLUMNS
from .metrics import evaluate_predictions


def _clip_probs(probs: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(probs, dtype=np.float64), 1e-6, 1.0 - 1e-6)
    return p / p.sum(axis=1, keepdims=True)


def _multiclass_logloss(probs: np.ndarray, actual_results: Sequence[str]) -> float:
    p = _clip_probs(probs)
    idx_map = {"H": 0, "D": 1, "A": 2}
    indices = np.array([idx_map[r] for r in actual_results], dtype=int)
    chosen = p[np.arange(len(indices)), indices]
    return float(-np.mean(np.log(chosen)))


@dataclass
class OrderedFactorModel:
    """Ordered Logit model (Away < Draw < Home) with tempo-modulated draw boundary.

    In football, H/D/A are ordered outcomes driven by latent supremacy z = w^T x:
      P(A) = sigmoid(-c - z)
      P(D) = sigmoid(+c - z) - sigmoid(-c - z)
      P(H) = 1 - sigmoid(+c - z)
    where c > 0 is the half-width of the draw zone, modulated by expected_tempo
    and elo_abs_gap.
    """

    feature_names: tuple[str, ...]
    means: np.ndarray
    scales: np.ndarray
    weights: np.ndarray
    bias: float
    draw_log_width: float
    draw_tempo_coef: float
    l2: float

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        x_raw = df.loc[:, list(self.feature_names)].to_numpy(dtype=np.float64)
        x_raw = np.nan_to_num(x_raw, nan=0.0)
        x_std = (x_raw - self.means.reshape(1, -1)) / self.scales.reshape(1, -1)

        z = self.bias + x_std @ self.weights
        tempo = (
            df["expected_tempo"].to_numpy(dtype=np.float64)
            if "expected_tempo" in df.columns
            else np.zeros(len(df))
        )
        c = np.exp(np.clip(self.draw_log_width + self.draw_tempo_coef * tempo, -1.5, 1.5))

        p_a = expit(-c - z)
        p_a_plus_d = expit(c - z)
        p_d = np.maximum(1e-6, p_a_plus_d - p_a)
        p_h = np.maximum(1e-6, 1.0 - p_a_plus_d)
        return _clip_probs(np.column_stack([p_h, p_d, p_a]))

    def explain_row(self, row: Mapping[str, Any]) -> dict[str, Any]:
        """Decompose latent supremacy z and Home Win probability into per-factor contributions."""
        spec_map = {s["name"]: s for s in FACTOR_SPECIFICATIONS}
        contributions: list[dict[str, Any]] = []
        family_totals: dict[str, float] = {}

        z_total = self.bias
        for idx, fname in enumerate(self.feature_names):
            val = float(row.get(fname, 0.0))
            if not math.isfinite(val):
                val = 0.0
            z_val = (val - float(self.means[idx])) / float(self.scales[idx])
            contrib_z = float(self.weights[idx] * z_val)
            z_total += contrib_z
            fam = spec_map.get(fname, {}).get("family", "other")
            family_totals[fam] = family_totals.get(fam, 0.0) + contrib_z
            contributions.append(
                {
                    "factor": fname,
                    "family": fam,
                    "raw_value": val,
                    "z_score": float(z_val),
                    "weight": float(self.weights[idx]),
                    "latent_contribution": contrib_z,
                }
            )

        tempo = float(row.get("expected_tempo", 0.0))
        c = math.exp(min(1.5, max(-1.5, self.draw_log_width + self.draw_tempo_coef * tempo)))
        # Marginal slope dP(H)/dz = sigmoid(z - c) * (1 - sigmoid(z - c))
        s_h = float(expit(z_total - c))
        slope_h = s_h * (1.0 - s_h)

        for item in contributions:
            item["marginal_prob_h_pct"] = float(item["latent_contribution"] * slope_h * 100.0)

        family_prob_pct = {
            fam: float(val * slope_h * 100.0) for fam, val in family_totals.items()
        }
        contributions.sort(key=lambda d: abs(d["marginal_prob_h_pct"]), reverse=True)
        return {
            "latent_supremacy_z": float(z_total),
            "draw_boundary_c": float(c),
            "family_marginal_prob_h_pct": family_prob_pct,
            "top_factor_contributions": contributions,
        }


def fit_ordered_factor_model(
    train_df: pd.DataFrame,
    feature_names: Sequence[str],
    *,
    l2: float = 1.0,
    enforce_expected_signs: bool = True,
) -> OrderedFactorModel:
    """Fit the OrderedFactorModel via L-BFGS-B maximum likelihood with sign-consistency bounds."""
    f_list = tuple(feature_names)
    x_raw = train_df.loc[:, list(f_list)].to_numpy(dtype=np.float64)
    x_raw = np.nan_to_num(x_raw, nan=0.0)
    means = x_raw.mean(axis=0)
    scales = x_raw.std(axis=0)
    scales = np.where(scales < 1e-8, 1.0, scales)
    x_std = (x_raw - means.reshape(1, -1)) / scales.reshape(1, -1)

    tempo = (
        train_df["expected_tempo"].to_numpy(dtype=np.float64)
        if "expected_tempo" in train_df.columns
        else np.zeros(len(train_df))
    )
    y_map = {"H": 0, "D": 1, "A": 2}
    y = np.array([y_map[r] for r in train_df["actual_result"]], dtype=int)
    n_samples, n_features = x_std.shape

    spec_map = {s["name"]: s for s in FACTOR_SPECIFICATIONS}
    bounds: list[tuple[float | None, float | None]] = []
    for fname in f_list:
        exp_sign = int(spec_map.get(fname, {}).get("expected_sign", 0))
        if enforce_expected_signs and exp_sign > 0:
            bounds.append((0.0, 3.0))
        elif enforce_expected_signs and exp_sign < 0:
            bounds.append((-3.0, 0.0))
        else:
            bounds.append((-3.0, 3.0))
    # Bounds for bias, draw_log_width, draw_tempo_coef
    bounds.extend([(-2.0, 2.0), (-1.2, 1.0), (-0.5, 0.5)])

    theta0 = np.zeros(n_features + 3, dtype=np.float64)
    if "elo_diff" in f_list:
        theta0[f_list.index("elo_diff")] = 0.65
    theta0[n_features] = 0.25  # home bias
    theta0[n_features + 1] = -0.55  # draw_log_width (c ~ 0.58)
    theta0[n_features + 2] = -0.05  # slower tempo -> wider draw zone

    def objective(theta: np.ndarray) -> float:
        w = theta[:n_features]
        bias = theta[n_features]
        d_lw = theta[n_features + 1]
        d_tc = theta[n_features + 2]

        z = bias + x_std @ w
        c = np.exp(np.clip(d_lw + d_tc * tempo, -1.5, 1.5))

        p_a = expit(-c - z)
        p_a_d = expit(c - z)
        p_d = np.maximum(1e-7, p_a_d - p_a)
        p_h = np.maximum(1e-7, 1.0 - p_a_d)
        p_a = np.maximum(1e-7, p_a)
        probs = np.column_stack([p_h, p_d, p_a])
        probs = probs / probs.sum(axis=1, keepdims=True)

        nll = -np.mean(np.log(probs[np.arange(n_samples), y]))
        # Penalize non-Elo weights with L2 regularization
        reg = 0.5 * (l2 / max(1, n_samples)) * float(np.sum(w * w))
        return float(nll + reg)

    res = minimize(objective, theta0, method="L-BFGS-B", bounds=bounds, options={"maxiter": 500})
    theta_opt = res.x
    return OrderedFactorModel(
        feature_names=f_list,
        means=means,
        scales=scales,
        weights=theta_opt[:n_features],
        bias=float(theta_opt[n_features]),
        draw_log_width=float(theta_opt[n_features + 1]),
        draw_tempo_coef=float(theta_opt[n_features + 2]),
        l2=float(l2),
    )


@dataclass
class FactorDixonColesModel:
    """Factor-Driven Bivariate Poisson / Dixon-Coles (1997) Goal Model.

    Models expected goals:
      ln(lambda_H) = alpha_H + 0.5 * (w_sup^T x_std) + w_tempo * tempo
      ln(lambda_A) = alpha_A - 0.5 * (w_sup^T x_std) + w_tempo * tempo
    and applies the Dixon-Coles low-score dependence adjustment parameter rho
    to scorelines (0,0), (1,0), (0,1), (1,1).
    """

    feature_names: tuple[str, ...]
    means: np.ndarray
    scales: np.ndarray
    supremacy_weights: np.ndarray
    alpha_home: float
    alpha_away: float
    tempo_weight: float
    rho: float
    l2: float
    max_goals: int = 10

    def expected_goals(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        x_raw = df.loc[:, list(self.feature_names)].to_numpy(dtype=np.float64)
        x_raw = np.nan_to_num(x_raw, nan=0.0)
        x_std = (x_raw - self.means.reshape(1, -1)) / self.scales.reshape(1, -1)

        sup = x_std @ self.supremacy_weights
        tempo = (
            df["expected_tempo"].to_numpy(dtype=np.float64)
            if "expected_tempo" in df.columns
            else np.zeros(len(df))
        )
        lam_h = np.exp(np.clip(self.alpha_home + 0.5 * sup + self.tempo_weight * tempo, -1.5, 2.2))
        lam_a = np.exp(np.clip(self.alpha_away - 0.5 * sup + self.tempo_weight * tempo, -1.5, 2.2))
        return lam_h, lam_a

    def predict_proba_and_goals(
        self, df: pd.DataFrame
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return (probs_1x2 [N, 3], lam_h [N], lam_a [N], prob_over25 [N])."""
        lam_h, lam_a = self.expected_goals(df)
        n = len(df)
        max_g = self.max_goals
        goals = np.arange(max_g + 1, dtype=np.float64)
        fact = np.array([math.factorial(int(g)) for g in goals], dtype=np.float64)

        # Vectorized Poisson PMFs: shape [N, max_g + 1]
        pmf_h = np.exp(-lam_h[:, None]) * (lam_h[:, None] ** goals[None, :]) / fact[None, :]
        pmf_a = np.exp(-lam_a[:, None]) * (lam_a[:, None] ** goals[None, :]) / fact[None, :]

        # Joint matrix: shape [N, max_g + 1, max_g + 1]
        joint = pmf_h[:, :, None] * pmf_a[:, None, :]

        # Apply Dixon-Coles low-score correlation adjustment tau(x, y)
        rho = self.rho
        joint[:, 0, 0] *= np.maximum(0.0, 1.0 - lam_h * lam_a * rho)
        joint[:, 0, 1] *= np.maximum(0.0, 1.0 + lam_h * rho)
        joint[:, 1, 0] *= np.maximum(0.0, 1.0 + lam_a * rho)
        joint[:, 1, 1] *= np.maximum(0.0, 1.0 - rho)

        total_mass = joint.sum(axis=(1, 2), keepdims=True)
        joint = np.where(total_mass > 0, joint / total_mass, joint)

        tril_mask = np.tril(np.ones((max_g + 1, max_g + 1), dtype=bool), k=-1)
        diag_mask = np.eye(max_g + 1, dtype=bool)
        triu_mask = np.triu(np.ones((max_g + 1, max_g + 1), dtype=bool), k=1)
        over25_mask = (goals[:, None] + goals[None, :]) > 2.5

        p_h = joint[:, tril_mask].sum(axis=1)
        p_d = joint[:, diag_mask].sum(axis=1)
        p_a = joint[:, triu_mask].sum(axis=1)
        p_over25 = joint[:, over25_mask].sum(axis=1)

        probs = _clip_probs(np.column_stack([p_h, p_d, p_a]))
        return probs, lam_h, lam_a, p_over25

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        probs, _, _, _ = self.predict_proba_and_goals(df)
        return probs


def fit_dixon_coles_factor_model(
    train_df: pd.DataFrame,
    feature_names: Sequence[str],
    *,
    l2: float = 5.0,
    enforce_expected_signs: bool = True,
) -> FactorDixonColesModel:
    """Fit the Factor-Driven Dixon-Coles Goal Model by maximizing Poisson+DC log-likelihood."""
    f_list = tuple(f for f in feature_names if f not in ("elo_abs_gap", "expected_tempo"))
    x_raw = train_df.loc[:, list(f_list)].to_numpy(dtype=np.float64)
    x_raw = np.nan_to_num(x_raw, nan=0.0)
    means = x_raw.mean(axis=0)
    scales = x_raw.std(axis=0)
    scales = np.where(scales < 1e-8, 1.0, scales)
    x_std = (x_raw - means.reshape(1, -1)) / scales.reshape(1, -1)

    tempo = (
        train_df["expected_tempo"].to_numpy(dtype=np.float64)
        if "expected_tempo" in train_df.columns
        else np.zeros(len(train_df))
    )
    hg = train_df["home_goals"].to_numpy(dtype=np.float64)
    ag = train_df["away_goals"].to_numpy(dtype=np.float64)
    n_samples, n_features = x_std.shape

    spec_map = {s["name"]: s for s in FACTOR_SPECIFICATIONS}
    bounds: list[tuple[float | None, float | None]] = []
    for fname in f_list:
        exp_sign = int(spec_map.get(fname, {}).get("expected_sign", 0))
        if enforce_expected_signs and exp_sign > 0:
            bounds.append((0.0, 1.5))
        elif enforce_expected_signs and exp_sign < 0:
            bounds.append((-1.5, 0.0))
        else:
            bounds.append((-1.5, 1.5))
    # alpha_home, alpha_away, tempo_weight, rho
    bounds.extend([(-0.5, 1.2), (-0.5, 1.2), (0.0, 0.5), (-0.22, 0.15)])

    theta0 = np.zeros(n_features + 4, dtype=np.float64)
    if "elo_diff" in f_list:
        theta0[f_list.index("elo_diff")] = 0.35
    theta0[n_features] = math.log(max(0.5, float(np.mean(hg))))
    theta0[n_features + 1] = math.log(max(0.5, float(np.mean(ag))))
    theta0[n_features + 2] = 0.08
    theta0[n_features + 3] = -0.08

    m00 = (hg == 0) & (ag == 0)
    m01 = (hg == 0) & (ag == 1)
    m10 = (hg == 1) & (ag == 0)
    m11 = (hg == 1) & (ag == 1)

    y_map = {"H": 0, "D": 1, "A": 2}
    y_idx = np.array([y_map.get(r, 0) for r in train_df["actual_result"]], dtype=int)

    max_g = 8
    goals = np.arange(max_g + 1, dtype=np.float64)
    fact = np.array([math.factorial(int(g)) for g in goals], dtype=np.float64)
    tril_mask = np.tril(np.ones((max_g + 1, max_g + 1), dtype=bool), k=-1)
    diag_mask = np.eye(max_g + 1, dtype=bool)
    triu_mask = np.triu(np.ones((max_g + 1, max_g + 1), dtype=bool), k=1)

    def objective(theta: np.ndarray) -> float:
        w = theta[:n_features]
        a_h = theta[n_features]
        a_a = theta[n_features + 1]
        w_t = theta[n_features + 2]
        rho = theta[n_features + 3]

        sup = x_std @ w
        log_lh = np.clip(a_h + 0.5 * sup + w_t * tempo, -1.5, 2.2)
        log_la = np.clip(a_a - 0.5 * sup + w_t * tempo, -1.5, 2.2)
        lh = np.exp(log_lh)
        la = np.exp(log_la)

        # 1. Independent Poisson goal log-likelihood penalty
        nll_goals = np.mean((lh - hg * log_lh) + (la - ag * log_la))

        # 2. 1X2 cross-entropy loss via Dixon-Coles bivariate Poisson grid
        pmf_h = np.exp(-lh[:, None]) * (lh[:, None] ** goals[None, :]) / fact[None, :]
        pmf_a = np.exp(-la[:, None]) * (la[:, None] ** goals[None, :]) / fact[None, :]
        joint = pmf_h[:, :, None] * pmf_a[:, None, :]

        joint[:, 0, 0] *= np.maximum(0.0, 1.0 - lh * la * rho)
        joint[:, 0, 1] *= np.maximum(0.0, 1.0 + lh * rho)
        joint[:, 1, 0] *= np.maximum(0.0, 1.0 + la * rho)
        joint[:, 1, 1] *= np.maximum(0.0, 1.0 - rho)

        total_mass = joint.sum(axis=(1, 2), keepdims=True)
        joint = np.where(total_mass > 0, joint / total_mass, joint)

        p_h = joint[:, tril_mask].sum(axis=1)
        p_d = joint[:, diag_mask].sum(axis=1)
        p_a = joint[:, triu_mask].sum(axis=1)

        probs_1x2 = np.column_stack([p_h, p_d, p_a])
        probs_1x2 = np.clip(probs_1x2, 1e-7, 1.0)
        probs_1x2 = probs_1x2 / probs_1x2.sum(axis=1, keepdims=True)

        p_true = probs_1x2[np.arange(n_samples), y_idx]
        nll_1x2 = -float(np.mean(np.log(p_true)))

        penalty = 0.5 * (l2 / max(1, n_samples)) * float(np.sum(w * w))
        loss = 0.85 * nll_1x2 + 0.15 * float(nll_goals) + penalty
        return float(loss)

    res = minimize(objective, theta0, method="L-BFGS-B", bounds=bounds, options={"maxiter": 500})
    opt = res.x
    return FactorDixonColesModel(
        feature_names=f_list,
        means=means,
        scales=scales,
        supremacy_weights=opt[:n_features],
        alpha_home=float(opt[n_features]),
        alpha_away=float(opt[n_features + 1]),
        tempo_weight=float(opt[n_features + 2]),
        rho=float(opt[n_features + 3]),
        l2=float(l2),
    )


@dataclass
class CalibratedFactorSystem:
    """Complete trained factor system containing both Ordered Logit and Dixon-Coles models."""

    track: str
    feature_names: tuple[str, ...]
    ordered_model: OrderedFactorModel
    dixon_coles_model: FactorDixonColesModel
    weight_ordered: float
    weight_dc: float
    weight_elo: float

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        p_ord = self.ordered_model.predict_proba(df)
        p_dc = self.dixon_coles_model.predict_proba(df)
        p_elo = df[["elo_prob_h", "elo_prob_d", "elo_prob_a"]].to_numpy(dtype=np.float64)
        blended = (
            self.weight_ordered * p_ord
            + self.weight_dc * p_dc
            + self.weight_elo * p_elo
        )
        return _clip_probs(blended)

    def predict_rich(self, df: pd.DataFrame) -> pd.DataFrame:
        """Predict 1X2 probabilities, expected goals, Over 2.5 probability, and factor edge over Elo."""
        p_ord = self.ordered_model.predict_proba(df)
        p_dc, lam_h, lam_a, p_over25 = self.dixon_coles_model.predict_proba_and_goals(df)
        p_elo = df[["elo_prob_h", "elo_prob_d", "elo_prob_a"]].to_numpy(dtype=np.float64)
        probs = _clip_probs(
            self.weight_ordered * p_ord + self.weight_dc * p_dc + self.weight_elo * p_elo
        )

        out = df[
            ["match_id", "season", "date", "kickoff_time", "home_team", "away_team"]
        ].copy()
        out["elo_prob_h"] = p_elo[:, 0]
        out["elo_prob_d"] = p_elo[:, 1]
        out["elo_prob_a"] = p_elo[:, 2]
        out["home_win_prob"] = probs[:, 0]
        out["draw_prob"] = probs[:, 1]
        out["away_win_prob"] = probs[:, 2]
        out["expected_home_goals"] = lam_h
        out["expected_away_goals"] = lam_a
        out["prob_over25"] = p_over25
        out["alpha_shift_home_pct"] = (probs[:, 0] - p_elo[:, 0]) * 100.0
        out["most_likely_result"] = np.array(["H", "D", "A"])[np.argmax(probs, axis=1)]
        return out


def fit_and_select_factor_system(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    feature_names: Sequence[str],
    *,
    track: str = "fundamental",
) -> tuple[CalibratedFactorSystem, dict[str, Any]]:
    """Train Ordered Logit & Dixon-Coles models on train_df, select regularization & blend on val_df."""
    l2_candidates = (0.5, 2.0, 8.0, 25.0)
    best_ord: OrderedFactorModel | None = None
    best_ord_loss = float("inf")

    for l2 in l2_candidates:
        m_ord = fit_ordered_factor_model(train_df, feature_names, l2=l2, enforce_expected_signs=True)
        val_p = m_ord.predict_proba(val_df)
        loss = _multiclass_logloss(val_p, val_df["actual_result"].tolist())
        if loss < best_ord_loss:
            best_ord_loss = loss
            best_ord = m_ord

    best_dc: FactorDixonColesModel | None = None
    best_dc_loss = float("inf")
    for l2 in (1.0, 5.0, 15.0, 40.0):
        m_dc = fit_dixon_coles_factor_model(
            train_df, feature_names, l2=l2, enforce_expected_signs=True
        )
        val_p = m_dc.predict_proba(val_df)
        loss = _multiclass_logloss(val_p, val_df["actual_result"].tolist())
        if loss < best_dc_loss:
            best_dc_loss = loss
            best_dc = m_dc

    assert best_ord is not None and best_dc is not None
    p_ord_val = best_ord.predict_proba(val_df)
    p_dc_val = best_dc.predict_proba(val_df)
    p_elo_val = val_df[["elo_prob_h", "elo_prob_d", "elo_prob_a"]].to_numpy(dtype=np.float64)
    y_val = val_df["actual_result"].tolist()

    best_weights = (0.5, 0.3, 0.2)
    best_blend_loss = float("inf")
    for w_ord in np.linspace(0.0, 1.0, 11):
        for w_dc in np.linspace(0.0, 1.0 - w_ord, int(round((1.0 - w_ord) * 10)) + 1):
            w_elo = max(0.0, 1.0 - w_ord - w_dc)
            p_blend = _clip_probs(w_ord * p_ord_val + w_dc * p_dc_val + w_elo * p_elo_val)
            loss = _multiclass_logloss(p_blend, y_val)
            if loss < best_blend_loss:
                best_blend_loss = loss
                best_weights = (float(w_ord), float(w_dc), float(w_elo))

    # Refit on combined train + val with selected hyperparameters
    combined_df = pd.concat([train_df, val_df], ignore_index=True)
    final_ord = fit_ordered_factor_model(
        combined_df, feature_names, l2=best_ord.l2, enforce_expected_signs=True
    )
    final_dc = fit_dixon_coles_factor_model(
        combined_df, feature_names, l2=best_dc.l2, enforce_expected_signs=True
    )

    system = CalibratedFactorSystem(
        track=track,
        feature_names=tuple(feature_names),
        ordered_model=final_ord,
        dixon_coles_model=final_dc,
        weight_ordered=best_weights[0],
        weight_dc=best_weights[1],
        weight_elo=best_weights[2],
    )
    selection_info = {
        "track": track,
        "selected_ordered_l2": best_ord.l2,
        "selected_dc_l2": best_dc.l2,
        "selected_dixon_coles_rho": final_dc.rho,
        "blend_weights": {
            "ordered_logit": best_weights[0],
            "dixon_coles": best_weights[1],
            "elo_baseline": best_weights[2],
        },
        "validation_log_loss": {
            "elo_baseline": _multiclass_logloss(p_elo_val, y_val),
            "ordered_factor_only": best_ord_loss,
            "dixon_coles_factor_only": best_dc_loss,
            "blended_system": best_blend_loss,
        },
    }
    return system, selection_info


def evaluate_probs_frame(
    df: pd.DataFrame, probs: np.ndarray, model_name: str = "model"
) -> dict[str, float | int]:
    """Compute LogLoss, Brier Score, and Accuracy on a DataFrame + [N, 3] probability matrix."""
    p = _clip_probs(probs)
    eval_df = pd.DataFrame(
        {
            "season": df["season"].to_numpy(),
            "model": model_name,
            "actual_result": df["actual_result"].to_numpy(),
            "home_win_prob": p[:, 0],
            "draw_prob": p[:, 1],
            "away_win_prob": p[:, 2],
        }
    )
    res = evaluate_predictions(eval_df)
    return dict(res["overall"])  # type: ignore[arg-type]
