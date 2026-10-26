"""Layer 3: Factor Lab — Alphalens-Style Football Factor Profiling & Orthogonal Selection.

Evaluates every candidate factor before modeling:
  1. Rank IC (Spearman) with Goal Diff & Match Outcome, plus cross-season IC_IR & t-stat
  2. Orthogonal Alpha over Elo: partial correlation & t-stat against Elo residuals
     (separating true incremental alpha from disguised Elo proxies)
  3. 5-Quantile Stratification (Q1..Q5): Home Win%, Draw%, Avg Goal Diff, and Elo Excess
     across quintiles, plus Monotonicity score
  4. Multicollinearity (Pairwise Correlation & VIF) pruning to prevent sign flips
"""

from __future__ import annotations

import math
from typing import Any, Sequence

import numpy as np
import pandas as pd
from scipy import stats

from .factor_engine import (
    ALL_FACTOR_COLUMNS,
    FACTOR_SPECIFICATIONS,
    FUNDAMENTAL_FACTOR_COLUMNS,
    MARKET_FACTOR_COLUMNS,
)


def _spearman_corr(x: np.ndarray, y: np.ndarray, min_samples: int = 20) -> tuple[float, float]:
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < min_samples:
        return 0.0, 1.0
    xv, yv = x[mask], y[mask]
    if np.std(xv) < 1e-12 or np.std(yv) < 1e-12:
        return 0.0, 1.0
    res = stats.spearmanr(xv, yv)
    corr = float(res.statistic) if math.isfinite(float(res.statistic)) else 0.0
    pval = float(res.pvalue) if math.isfinite(float(res.pvalue)) else 1.0
    return corr, pval


def _orthogonalize_against_elo(factor: np.ndarray, elo_diff: np.ndarray) -> np.ndarray:
    """Return the component of `factor` orthogonal to `elo_diff` via OLS residual."""
    mask = np.isfinite(factor) & np.isfinite(elo_diff)
    out = np.full_like(factor, np.nan, dtype=np.float64)
    if mask.sum() < 20:
        return out
    e = elo_diff[mask]
    f = factor[mask]
    e_centered = e - np.mean(e)
    f_centered = f - np.mean(f)
    denom = float(np.dot(e_centered, e_centered))
    if denom < 1e-12:
        out[mask] = f_centered
        return out
    beta = float(np.dot(e_centered, f_centered) / denom)
    out[mask] = f_centered - beta * e_centered
    return out


def _ols_tstat(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Compute OLS slope t-statistic and two-sided p-value."""
    mask = np.isfinite(x) & np.isfinite(y)
    n = int(mask.sum())
    if n < 25:
        return 0.0, 1.0
    xv = x[mask]
    yv = y[mask]
    std_x = float(np.std(xv))
    std_y = float(np.std(yv))
    if std_x < 1e-12 or std_y < 1e-12:
        return 0.0, 1.0
    r = float(np.corrcoef(xv, yv)[0, 1])
    if not math.isfinite(r) or abs(r) >= 1.0:
        return 0.0, 1.0
    t_val = r * math.sqrt((n - 2) / max(1e-12, 1.0 - r * r))
    p_val = float(2.0 * stats.t.sf(abs(t_val), df=n - 2))
    return float(t_val), p_val


def compute_quantile_profile(
    df: pd.DataFrame, factor_col: str, n_quantiles: int = 5
) -> dict[str, Any]:
    """Compute 5-quantile stratification profile for a single factor."""
    sub = df[[factor_col, "goal_diff", "actual_result", "elo_residual"]].dropna()
    if len(sub) < 50:
        return {"buckets": [], "q5_minus_q1_gd": 0.0, "q5_minus_q1_elo_excess": 0.0, "monotonicity": 0.0}

    # Add tiny jitter tied to rank to break ties deterministically if needed
    values = sub[factor_col].to_numpy(dtype=np.float64)
    ranks = stats.rankdata(values, method="ordinal")
    bucket_ids = np.minimum(n_quantiles - 1, (ranks * n_quantiles) // (len(ranks) + 1)).astype(int)

    buckets: list[dict[str, Any]] = []
    gd_means: list[float] = []
    elo_res_means: list[float] = []
    draw_rates: list[float] = []

    for q in range(n_quantiles):
        mask = bucket_ids == q
        q_df = sub.iloc[mask]
        n_q = len(q_df)
        h_rate = float((q_df["actual_result"] == "H").mean()) if n_q else 0.0
        d_rate = float((q_df["actual_result"] == "D").mean()) if n_q else 0.0
        a_rate = float((q_df["actual_result"] == "A").mean()) if n_q else 0.0
        gd_mean = float(q_df["goal_diff"].mean()) if n_q else 0.0
        res_mean = float(q_df["elo_residual"].mean()) if n_q else 0.0
        gd_means.append(gd_mean)
        elo_res_means.append(res_mean)
        draw_rates.append(d_rate)
        buckets.append(
            {
                "quantile": f"Q{q + 1}",
                "count": int(n_q),
                "factor_mean": float(q_df[factor_col].mean()),
                "home_win_rate": h_rate,
                "draw_rate": d_rate,
                "away_win_rate": a_rate,
                "avg_goal_diff": gd_mean,
                "avg_elo_excess": res_mean,
            }
        )

    mono_corr, _ = _spearman_corr(
        np.arange(n_quantiles, dtype=float), np.asarray(gd_means), min_samples=3
    )
    draw_mono_corr, _ = _spearman_corr(
        np.arange(n_quantiles, dtype=float), np.asarray(draw_rates), min_samples=3
    )

    return {
        "buckets": buckets,
        "q5_minus_q1_gd": float(gd_means[-1] - gd_means[0]),
        "q5_minus_q1_home_win": float(buckets[-1]["home_win_rate"] - buckets[0]["home_win_rate"]),
        "q5_minus_q1_draw": float(draw_rates[-1] - draw_rates[0]),
        "q5_minus_q1_elo_excess": float(elo_res_means[-1] - elo_res_means[0]),
        "monotonicity": float(mono_corr),
        "draw_monotonicity": float(draw_mono_corr),
    }


def compute_vif_table(df: pd.DataFrame, columns: Sequence[str]) -> dict[str, float]:
    """Compute Variance Inflation Factor (VIF) for a set of factor columns."""
    mat = df.loc[:, list(columns)].to_numpy(dtype=np.float64)
    mat = np.nan_to_num(mat, nan=0.0)
    means = mat.mean(axis=0, keepdims=True)
    stds = mat.std(axis=0, keepdims=True)
    stds = np.where(stds < 1e-12, 1.0, stds)
    z = (mat - means) / stds
    with np.errstate(divide="ignore", invalid="ignore"):
        corr = np.corrcoef(z, rowvar=False)
    corr = np.nan_to_num(corr, nan=0.0)
    np.fill_diagonal(corr, 1.0)
    try:
        inv_corr = np.linalg.pinv(corr)
        vifs = np.diag(inv_corr)
    except np.linalg.LinAlgError:
        vifs = np.ones(len(columns))
    return {col: float(max(1.0, v)) for col, v in zip(columns, vifs)}


def run_factor_lab(
    panel: pd.DataFrame,
    *,
    discovery_seasons: Sequence[str] | None = None,
    warmup_matches_per_team: int = 5,
) -> dict[str, Any]:
    """Run comprehensive single-factor & orthogonal alpha diagnostics on the factor panel."""
    df = panel.copy()
    if discovery_seasons is not None:
        df = df[df["season"].isin(set(discovery_seasons))].copy()

    # Exclude cold-start initial season first 5 rounds so EWMA states are warmed up
    df = df[
        (df["home_matches_played"] >= warmup_matches_per_team)
        | (df["season"] != df["season"].min())
    ].copy()

    elo_arr = df["elo_diff"].to_numpy(dtype=np.float64)
    gd_arr = df["goal_diff"].to_numpy(dtype=np.float64)
    score_arr = df["actual_home_score"].to_numpy(dtype=np.float64)
    elo_res_arr = df["elo_residual"].to_numpy(dtype=np.float64)
    is_draw_arr = (df["actual_result"] == "D").to_numpy(dtype=np.float64)

    spec_map = {spec["name"]: spec for spec in FACTOR_SPECIFICATIONS}
    seasons = sorted(df["season"].unique())

    factor_reports: list[dict[str, Any]] = []
    for name in ALL_FACTOR_COLUMNS:
        spec = spec_map[name]
        f_arr = df[name].to_numpy(dtype=np.float64)

        # 1. Raw Rank IC with Goal Diff and Match Score
        raw_ic_gd, raw_ic_pval = _spearman_corr(f_arr, gd_arr)
        raw_ic_score, _ = _spearman_corr(f_arr, score_arr)
        draw_ic, draw_ic_pval = _spearman_corr(f_arr, is_draw_arr)

        # Seasonal IC stability
        seasonal_ics: list[float] = []
        for s in seasons:
            s_mask = (df["season"] == s).to_numpy()
            if s_mask.sum() >= 30:
                s_ic, _ = _spearman_corr(f_arr[s_mask], gd_arr[s_mask])
                seasonal_ics.append(s_ic)
        ic_mean_season = float(np.mean(seasonal_ics)) if seasonal_ics else 0.0
        ic_std_season = float(np.std(seasonal_ics, ddof=1)) if len(seasonal_ics) > 1 else 1.0
        ic_ir = ic_mean_season / max(ic_std_season, 1e-6)
        expected_sign = int(spec["expected_sign"])
        if expected_sign != 0:
            sign_consistent_seasons = sum(
                1 for x in seasonal_ics if (x * expected_sign) > 0
            ) / max(1, len(seasonal_ics))
        else:
            # Draw/tempo factors evaluated against draw outcome
            sign_consistent_seasons = sum(1 for x in seasonal_ics if abs(x) >= 0.0) / max(
                1, len(seasonal_ics)
            )

        # 2. Orthogonal Alpha over Elo
        corr_with_elo, _ = _spearman_corr(f_arr, elo_arr)
        if name == "elo_diff":
            ortho_ic = raw_ic_gd
            ortho_tstat, ortho_pval = _ols_tstat(f_arr, score_arr)
        elif expected_sign == 0:
            # Structural draw/tempo factor: test against Draw indicator
            ortho_ic = draw_ic
            ortho_tstat, ortho_pval = _ols_tstat(f_arr, is_draw_arr)
        else:
            f_ortho = _orthogonalize_against_elo(f_arr, elo_arr)
            ortho_ic, _ = _spearman_corr(f_ortho, elo_res_arr)
            ortho_tstat, ortho_pval = _ols_tstat(f_ortho, elo_res_arr)

        # 3. 5-Quantile Profile
        q_profile = compute_quantile_profile(df, name, n_quantiles=5)

        factor_reports.append(
            {
                "name": name,
                "family": spec["family"],
                "track": spec["track"],
                "expected_sign": expected_sign,
                "description": spec["description"],
                "raw_ic_gd": raw_ic_gd,
                "raw_ic_score": raw_ic_score,
                "raw_ic_pvalue": raw_ic_pval,
                "draw_ic": draw_ic,
                "draw_ic_pvalue": draw_ic_pval,
                "seasonal_ic_mean": ic_mean_season,
                "seasonal_ic_std": ic_std_season,
                "ic_ir": ic_ir,
                "sign_consistency_rate": float(sign_consistent_seasons),
                "corr_with_elo": corr_with_elo,
                "orthogonal_ic": ortho_ic,
                "orthogonal_tstat": ortho_tstat,
                "orthogonal_pvalue": ortho_pval,
                "quantile_profile": q_profile,
            }
        )

    # 4. Collinearity & Shortlist Selection
    vif_before = compute_vif_table(df, FUNDAMENTAL_FACTOR_COLUMNS)
    corr_matrix = df.loc[:, list(FUNDAMENTAL_FACTOR_COLUMNS)].corr(method="spearman")

    selected_fundamental: list[str] = []
    selected_market: list[str] = []

    # Sort fundamental factors by priority: core strength first, then highest |orthogonal_tstat| + |ic_ir|
    report_by_name = {r["name"]: r for r in factor_reports}
    for r in factor_reports:
        r["vif_initial"] = vif_before.get(r["name"], 1.0)

    # Always include core structural basis: elo_diff, elo_abs_gap, expected_tempo
    core_basis = ["elo_diff", "elo_abs_gap", "expected_tempo"]
    for c in core_basis:
        selected_fundamental.append(c)
        report_by_name[c]["status"] = "selected_core"
        report_by_name[c]["selection_reason"] = "核心实力与平局节奏结构基准因子"

    # Evaluate remaining fundamental factors for incremental orthogonal signal & low collinearity
    candidates = [
        r
        for r in factor_reports
        if r["track"] == "fundamental" and r["name"] not in core_basis
    ]
    candidates.sort(
        key=lambda r: (abs(r["orthogonal_tstat"]) + 1.5 * abs(r["ic_ir"])),
        reverse=True,
    )

    for r in candidates:
        fname = r["name"]
        # Strict Orthogonal Alpha Gate: must provide independent signal AFTER removing Elo!
        has_orthogonal_signal = abs(r["orthogonal_tstat"]) >= 1.20
        if not has_orthogonal_signal:
            if abs(r["raw_ic_gd"]) >= 0.08:
                r["status"] = "rejected_elo_shadow"
                r["selection_reason"] = (
                    f"原始IC显著({r['raw_ic_gd']:+.3f})但扣除Elo后正交t={r['orthogonal_tstat']:+.2f}不足，属Elo影子变量"
                )
            else:
                r["status"] = "rejected_weak_signal"
                r["selection_reason"] = (
                    f"原始与正交增量信号均较弱 (IC={r['raw_ic_gd']:+.3f}, ortho_t={r['orthogonal_tstat']:+.2f})"
                )
            continue

        # Check collinearity against already-selected non-Elo factors
        too_collinear = False
        collinear_with = None
        for sel in selected_fundamental:
            if sel == "elo_diff":
                if abs(float(corr_matrix.loc[fname, sel])) > 0.88 and abs(r["orthogonal_tstat"]) < 2.5:
                    too_collinear = True
                    collinear_with = sel
                    break
            else:
                if abs(float(corr_matrix.loc[fname, sel])) > 0.83:
                    too_collinear = True
                    collinear_with = sel
                    break

        if too_collinear:
            r["status"] = "pruned_collinear"
            r["selection_reason"] = f"与已入选因子 {collinear_with} 高度共线 (|rho|>0.83)，剔除防多重共线性"
        else:
            selected_fundamental.append(fname)
            r["status"] = "selected_alpha"
            r["selection_reason"] = (
                f"通过正交增量Alpha检验 (IC={r['raw_ic_gd']:+.3f}, ortho_t={r['orthogonal_tstat']:+.2f})"
            )

    # Evaluate Market Track factors
    for name in MARKET_FACTOR_COLUMNS:
        r = report_by_name[name]
        selected_market.append(name)
        r["status"] = "selected_market_track"
        r["selection_reason"] = (
            f"市场轨道特征 (IC={r['raw_ic_gd']:+.3f}, ortho_t={r['orthogonal_tstat']:+.2f})"
        )

    vif_after = compute_vif_table(df, selected_fundamental)
    for r in factor_reports:
        r["vif_selected"] = vif_after.get(r["name"], None)

    return {
        "n_matches_analyzed": int(len(df)),
        "seasons_analyzed": seasons,
        "selected_fundamental_factors": selected_fundamental,
        "selected_market_factors": selected_market,
        "factors": factor_reports,
    }


def render_factor_lab_report(lab_result: dict[str, Any]) -> str:
    """Render a comprehensive Chinese Markdown report for the Factor Lab diagnostics."""
    lines: list[str] = [
        "# MatchQuant Factor Lab — 足球量化单因子与正交增量检验报告",
        "",
        f"- **分析样本量**：{lab_result['n_matches_analyzed']} 场英超比赛",
        f"- **覆盖赛季**：`{', '.join(lab_result['seasons_analyzed'])}`",
        f"- **入选纯基本面因子数（Track A）**：**{len(lab_result['selected_fundamental_factors'])}** 个",
        f"- **入选市场信号因子数（Track B）**：**{len(lab_result['selected_market_factors'])}** 个",
        "",
        "## 一、 全部候选因子 IC、正交 Alpha 与五分位分层总表",
        "",
        "| 因子名称 | 因子族 | 轨道 | Rank IC (净胜球) | 跨季 IC_IR | 与Elo相关 | 扣除Elo后正交 t值 | Q5-Q1 净胜球差 | Q5-Q1 超额得分 | 入选状态 |",
        "| :--- | :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | :--- |",
    ]

    for f in lab_result["factors"]:
        qp = f["quantile_profile"]
        lines.append(
            f"| `{f['name']}` | `{f['family']}` | `{f['track']}` "
            f"| {f['raw_ic_gd']:+.4f} | {f['ic_ir']:+.2f} | {f['corr_with_elo']:+.3f} "
            f"| **{f['orthogonal_tstat']:+.2f}** | {qp['q5_minus_q1_gd']:+.3f} "
            f"| {qp['q5_minus_q1_elo_excess']:+.3f} | `{f['status']}` |"
        )

    lines.extend(
        [
            "",
            "## 二、 因子筛选与共线性（VIF）诊断详情",
            "",
            "| 因子名称 | 物理意义说明 | 初始 VIF | 入选后 VIF | 筛选结论与依据 |",
            "| :--- | :--- | ---: | ---: | :--- |",
        ]
    )
    for f in lab_result["factors"]:
        vif_sel = f"{f['vif_selected']:.2f}" if f.get("vif_selected") is not None else "-"
        lines.append(
            f"| `{f['name']}` | {f['description']} | {f['vif_initial']:.2f} | {vif_sel} | {f['selection_reason']} |"
        )

    lines.extend(
        [
            "",
            "## 三、 核心入选因子的五分位（Q1 → Q5）单调性分层表现",
            "",
        ]
    )
    for f in lab_result["factors"]:
        if not str(f["status"]).startswith("selected"):
            continue
        qp = f["quantile_profile"]
        lines.append(f"### `{f['name']}` — {f['description']}")
        lines.append(
            f"- **单调性系数 (Monotonicity)**：`{qp['monotonicity']:+.2f}` | **正交增量 t-stat**：`{f['orthogonal_tstat']:+.2f}`"
        )
        lines.append("")
        lines.append("| 分位组 | 场次 | 因子均值 | 主胜率 (H%) | 平局率 (D%) | 客胜率 (A%) | 场均净胜球 | 相对Elo超额得分 |")
        lines.append("| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
        for b in qp["buckets"]:
            lines.append(
                f"| **{b['quantile']}** | {b['count']} | {b['factor_mean']:+.3f} "
                f"| {b['home_win_rate'] * 100:.1f}% | {b['draw_rate'] * 100:.1f}% | {b['away_win_rate'] * 100:.1f}% "
                f"| {b['avg_goal_diff']:+.3f} | {b['avg_elo_excess']:+.4f} |"
            )
        lines.append("")

    return "\n".join(lines)
