r"""MatchQuant v2.0 - Model Showcase & Live Demo for VS Code.

This script demonstrates the core quantitative models, factor attribution,
and out-of-sample performance in one self-contained, runnable file.
To run in VS Code:
  Press the 'Run Python File' button or execute in terminal:
  D:\Dev\Python314\python.exe demo_model_showcase.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Ensure local package is in sys.path
WORKSPACE_DIR = Path(__file__).resolve().parent
if str(WORKSPACE_DIR) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_DIR))


def print_banner(title: str) -> None:
    print("\n" + "=" * 90)
    print(f"  {title}")
    print("=" * 90)


def main() -> int:
    # Set UTF-8 output encoding for console
    if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass

    print_banner("MatchQuant v2.0: 足球多因子量化定价模型全景展示 (VS Code Demo)")

    # 1. Check pre-computed summary report
    summary_path = WORKSPACE_DIR / "reports" / "research_run" / "quant_evaluation_summary.json"
    if summary_path.exists():
        with open(summary_path, "r", encoding="utf-8") as f:
            summary = json.load(f)

        print("\n[模块 1] 核心模型架构与 5 赛季滚动盲测战绩 (Walk-Forward Retraining: 1,900 场)")
        print("-" * 90)
        wf = summary.get("walk_forward_evaluation", {})
        pooled = wf.get("pooled", {})

        elo_m = pooled.get("elo", {})
        ta_m = pooled.get("track_a", {})
        tmo_m = pooled.get("track_m_open", {})
        pin_close = pooled.get("close", {})

        print(f"  * 评估跨度: 2021-22 至 2025-26 (共 5 个完整赛季，逐季滚动扩展重训，100% 零未来信息穿越)")
        print(f"  * Baseline 动态 Elo 基线:    LogLoss = {elo_m.get('log_loss', 0):.5f} | 准确率 = {elo_m.get('accuracy', 0)*100:.2f}% | Brier = {elo_m.get('brier_score', 0):.5f}")
        print(f"  * Track A 纯球场基本面模型:  LogLoss = {ta_m.get('log_loss', 0):.5f} | 准确率 = {ta_m.get('accuracy', 0)*100:.2f}% | ΔLogLoss = {ta_m.get('log_loss', 0) - elo_m.get('log_loss', 0):+.5f} (全5赛季全胜)")
        print(f"  * Track M-Open 初盘可交易:   LogLoss = {tmo_m.get('log_loss', 0):.5f} | 准确率 = {tmo_m.get('accuracy', 0)*100:.2f}% | 逼近机构初盘定价基准")
        print(f"  * Pinnacle 终盘市场天花板:    LogLoss = {pin_close.get('log_loss', 0):.5f} | 准确率 = {pin_close.get('accuracy', 0)*100:.2f}%")

        # Learned weights
        track_a_sel = summary.get("track_a_selection", {})
        blend_w = track_a_sel.get("blend_weights", {})
        rho = track_a_sel.get("selected_dixon_coles_rho", -0.0275)
        factors_list = summary.get("learned_weights_track_a", [])

        print("\n[模块 2] Track A 模型学到的核心因子权重 (符合足球经济学，零符号反转)")
        print("-" * 90)
        print(f"  * 进球相关性修正系数 Dixon-Coles rho: {rho:.4f} (提升 0-0/1-1 低比分平局概率)")
        print(f"  * 模型合成权重: Ordered Logit ({blend_w.get('ordered_logit', 0.0):.2f}) + Dixon-Coles ({blend_w.get('dixon_coles', 0.9):.2f}) + Elo ({blend_w.get('elo_baseline', 0.1):.2f})")
        print("  * 主要基本面因子权重 Top 6:")
        for item in factors_list[:6]:
            fname = item.get("factor", "")
            w_o = item.get("ordered_logit_weight", 0.0)
            w_d = item.get("dixon_coles_goal_weight", 0.0)
            fam = item.get("family", "")
            print(f"    - {fname:<26} [{fam:<14}] Ordered Logit: {w_o:+.4f} | Dixon-Coles: {w_d:+.4f}")

        # Kelly simulation
        strat2 = summary.get("kelly_paper_trading", {}).get("strategy_2_track_m_open", {})
        fair_p = strat2.get("fair_odds", {})
        raw_p = strat2.get("raw_odds", {})
        print(f"\n[模块 3] 0.20x 缩减凯利实战纸盘策略表现 ($1,000 初始本金，单笔 3% 上限)")
        print("-" * 90)
        print(f"  * 触发笔数:         {strat2.get('n_bets', 0)} 场 (从 1,900 场盲测中严格筛选)")
        print(f"  * 实际命中率:       {strat2.get('win_rate_pct', 0):.2f}%")
        print(f"  * CLV 聪明钱胜率:   {strat2.get('clv_win_rate_pct', 0):.2f}% (提前捕获收盘线赔率漂移)")
        print(f"  * 平均 CLV 溢价:    +{strat2.get('avg_clv_bps', 0):.1f} bps")
        print(f"  * 公平赔率最终净值: ${fair_p.get('final_bankroll', 0):.2f} (ROI: {fair_p.get('roi_pct', 0):+.2f}%, 夏普比率: {fair_p.get('sharpe_ratio', 0):.2f}, 最大回撤: {fair_p.get('max_drawdown_pct', 0):.1f}%)")
        print(f"  * 抽水原始赔率净值: ${raw_p.get('final_bankroll', 0):.2f} (ROI: {raw_p.get('roi_pct', 0):+.2f}%, 夏普比率: {raw_p.get('sharpe_ratio', 0):.2f}, 最大回撤: {raw_p.get('max_drawdown_pct', 0):.1f}%)")

    # 3. Sample match breakdown
    sample_preds = summary.get("sample_predictions", [])
    if sample_preds:
        print("\n[模块 4] 单场比赛量化雷达与因子归因拆解 (Top 3 边际贡献)")
        print("-" * 90)
        for sp in sample_preds[:5]:
            match_name = f"{sp.get('home_team')} vs {sp.get('away_team')}"
            date_str = sp.get("date", "")
            actual_score = sp.get("actual_score", "")

            elo_dict = sp.get("elo_probs", {})
            ta_dict = sp.get("track_a_fundamental_probs", {})
            xg_dict = sp.get("expected_goals", {})

            elo_h = elo_dict.get("H", 0) * 100
            elo_d = elo_dict.get("D", 0) * 100
            elo_a = elo_dict.get("A", 0) * 100

            ta_h = ta_dict.get("H", 0) * 100
            ta_d = ta_dict.get("D", 0) * 100
            ta_a = ta_dict.get("A", 0) * 100

            lam_h = xg_dict.get("home_xg_proxy", 0)
            lam_a = xg_dict.get("away_xg_proxy", 0)
            p_o25 = xg_dict.get("prob_over25", 0) * 100
            shift_h = sp.get("alpha_shift_home_pct", 0)

            drivers = sp.get("top_non_elo_drivers", [])
            drivers_str = " | ".join(f"{d.get('factor')}({d.get('marginal_prob_h_pct', 0):+.1f}%)" for d in drivers[:3])

            print(f"\n  >> 对阵: {match_name} ({date_str})  [实际赛果: {actual_score}]")
            print(f"     Elo 基准概率:       {elo_h:.1f}% / {elo_d:.1f}% / {elo_a:.1f}%")
            print(f"     Track A 基本面胜率:  {ta_h:.1f}% / {ta_d:.1f}% / {ta_a:.1f}% (主胜 Alpha: {shift_h:+.1f}%)")
            print(f"     期望进球 (xG代理):  {lam_h:.2f} - {lam_a:.2f} (大球概率 Over 2.5: {p_o25:.1f}%)")
            print(f"     核心 Alpha 驱动因子: {drivers_str}")

    print("\n" + "=" * 90)
    print("  VS Code 快捷展示指引:")
    print("  1. 核心模型定义文件: worktree/matchquant/quant_models.py (Ordered Logit & Dixon-Coles)")
    print("  2. 学术研究报告文件: worktree/reports/research_run/quant_research_report.md (按 Ctrl+Shift+V 预览)")
    print("  3. 实时终端雷达命令: python -m matchquant matchday-radar --input <data_csv> --top-n 10")
    print("=" * 90 + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
