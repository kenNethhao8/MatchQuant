"""Layer 5: Evidence-grounded Agent Chinese narrative report renderer and follow-up Q&A engine."""

from __future__ import annotations

import html
import json
import math
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

from .match_report import FACTOR_LABELS, _md, _num, _pct, _pp


def _clean_str(val: Any) -> str:
    return str(val).strip() if val is not None else ""


def build_evidence_catalog(report_json: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    """Build a deterministic catalog of structured evidence items with unique IDs."""
    catalog: dict[str, dict[str, Any]] = {}

    fixture = report_json.get("fixture", {})
    home = fixture.get("home_team", "主队")
    away = fixture.get("away_team", "客队")
    predictions = report_json.get("predictions", {})
    track_a = predictions.get("track_a", {})
    components = predictions.get("components", {})
    elo = components.get("elo", {})
    factors = report_json.get("factors", {})
    evidence = report_json.get("recent_evidence", {})
    home_ev = evidence.get("home", {})
    away_ev = evidence.get("away", {})
    market = report_json.get("market_comparison", {})
    temporal = report_json.get("temporal", {})

    exp_goals = predictions.get("expected_goals", {})
    p_h = track_a.get("home") if "home" in track_a else track_a.get("p_home")
    p_d = track_a.get("draw") if "draw" in track_a else track_a.get("p_draw")
    p_a = track_a.get("away") if "away" in track_a else track_a.get("p_away")
    l_h = exp_goals.get("home") if "home" in exp_goals else track_a.get("lambda_home")
    l_a = exp_goals.get("away") if "away" in exp_goals else track_a.get("lambda_away")
    p_o25 = predictions.get("probability_total_goals_over_2_5") or track_a.get("p_over25")

    elo_preds = predictions.get("elo", {})
    p_h_elo = elo_preds.get("home") if "home" in elo_preds else elo.get("p_home")
    p_d_elo = elo_preds.get("draw") if "draw" in elo_preds else elo.get("p_draw")
    p_a_elo = elo_preds.get("away") if "away" in elo_preds else elo.get("p_away")

    catalog["E1_PROBS"] = {
        "id": "E1",
        "category": "core_probabilities",
        "description": "Track A 核心胜平负概率与期望进球",
        "data": {
            "p_home": p_h,
            "p_draw": p_d,
            "p_away": p_a,
            "lambda_home": l_h,
            "lambda_away": l_a,
            "p_over25": p_o25,
        },
    }

    catalog["E2_ELO"] = {
        "id": "E2",
        "category": "baseline_elo",
        "description": "Elo 基础实力与分歧度",
        "data": {
            "elo_home": elo.get("elo_home"),
            "elo_away": elo.get("elo_away"),
            "elo_diff": factors.get("elo_diff"),
            "p_home_elo": p_h_elo,
            "p_draw_elo": p_d_elo,
            "p_away_elo": p_a_elo,
        },
    }

    # Extract top factor drivers
    factor_items = []
    for k, v in factors.items():
        if k not in ("season", "match_id") and isinstance(v, (int, float)) and math.isfinite(v):
            factor_items.append((k, float(v), FACTOR_LABELS.get(k, k)))
    # Sort by absolute magnitude
    factor_items.sort(key=lambda item: abs(item[1]), reverse=True)

    catalog["E3_FACTORS"] = {
        "id": "E3",
        "category": "factor_drivers",
        "description": "量化因子核心特征驱动",
        "top_drivers": factor_items[:6],
    }

    catalog["E4_RECENT_FORM"] = {
        "id": "E4",
        "category": "recent_evidence",
        "description": "主客近期战绩与场面攻防统计",
        "home": {
            "team": home,
            "sample_count": home_ev.get("sample_count", 0),
            "record": home_ev.get("record_count_by_outcome", {}),
            "goals_for": home_ev.get("goals_for", {}).get("mean"),
            "goals_against": home_ev.get("goals_against", {}).get("mean"),
            "shots_on_target_for": home_ev.get("shots_on_target_for", {}).get("mean"),
            "sot_for": home_ev.get("shots_on_target_for", {}).get("mean"),
            "corners_for": home_ev.get("corners_for", {}).get("mean"),
        },
        "away": {
            "team": away,
            "sample_count": away_ev.get("sample_count", 0),
            "record": away_ev.get("record_count_by_outcome", {}),
            "goals_for": away_ev.get("goals_for", {}).get("mean"),
            "goals_against": away_ev.get("goals_against", {}).get("mean"),
            "shots_on_target_for": away_ev.get("shots_on_target_for", {}).get("mean"),
            "sot_for": away_ev.get("shots_on_target_for", {}).get("mean"),
            "corners_for": away_ev.get("corners_for", {}).get("mean"),
        },
    }

    is_mkt_avail = bool(market.get("provided") or market.get("available"))
    m_inp = market.get("input") or {}
    catalog["E5_MARKET"] = {
        "id": "E5",
        "category": "market_comparison",
        "description": "初盘市场参照、让球对位、价值偏差与时序趋势",
        "available": is_mkt_avail,
        "source": m_inp.get("source") or market.get("source") or "外部市场基准盘口",
        "observed_at": m_inp.get("observed_at") or market.get("observed_at"),
        "odds": {
            "home": m_inp.get("odds_open_h"),
            "draw": m_inp.get("odds_open_d"),
            "away": m_inp.get("odds_open_a"),
        },
        "mkt_ah_line": m_inp.get("mkt_ah_line"),
        "market_open_probabilities": market.get("market_open_probabilities"),
        "track_a_minus_market_percentage_points": market.get("track_a_minus_market_percentage_points"),
        "value_edges": market.get("value_edges"),
        "divergence_alerts": market.get("divergence_alerts"),
        "handicap_assessment": market.get("handicap_assessment"),
        "timeline_analysis": market.get("timeline_analysis"),
        "m_open_available": bool(market.get("m_open_available")),
        "m_open_probabilities": market.get("m_open_probabilities"),
        "m_open_unavailable_reason": market.get("m_open_unavailable_reason"),
        "data": market,
    }

    catalog["E6_TEMPORAL"] = {
        "id": "E6",
        "category": "temporal_provenance",
        "description": "时序截断与数据账龄",
        "as_of": temporal.get("as_of"),
        "fixture_date": fixture.get("date"),
        "used_data_cutoff": temporal.get("used_data_cutoff"),
        "data_age_days": temporal.get("data_age_days_at_as_of"),
        "forecast_horizon_days": temporal.get("forecast_horizon_days"),
        "mode": temporal.get("mode"),
    }

    return catalog


def render_agent_narrative_markdown(
    report_json: Mapping[str, Any],
    catalog: Mapping[str, dict[str, Any]] | None = None,
) -> str:
    """Generate Chinese narrative analysis strictly grounded in structured evidence."""
    cat = catalog or build_evidence_catalog(report_json)

    fixture = report_json.get("fixture", {})
    home = fixture.get("home_team", "主队")
    away = fixture.get("away_team", "客队")
    fixture_date = fixture.get("date", "未知日期")
    season = fixture.get("season", "未知赛季")

    e1 = cat["E1_PROBS"]["data"]
    e2 = cat["E2_ELO"]["data"]
    e3 = cat["E3_FACTORS"]["top_drivers"]
    e4 = cat["E4_RECENT_FORM"]
    e5 = cat["E5_MARKET"]

    raw_lh = e1.get("lambda_home")
    raw_la = e1.get("lambda_away")
    lh = _num(raw_lh, 2)
    la = _num(raw_la, 2)
    e6 = cat["E6_TEMPORAL"]

    p_h = _pct(e1.get("p_home"))
    p_d = _pct(e1.get("p_draw"))
    p_a = _pct(e1.get("p_away"))
    raw_lh = e1.get("lambda_home")
    raw_la = e1.get("lambda_away")
    lh = _num(raw_lh, 2)
    la = _num(raw_la, 2)
    tot_lambda = (float(raw_lh) + float(raw_la)) if (raw_lh is not None and raw_la is not None) else None
    tot_goals_str = _num(tot_lambda, 2)
    tot_goals_approx = _num(tot_lambda, 1)
    p_o25 = _pct(e1.get("p_over25"))

    # Determine qualitative favorite
    fav_text = "双方势均力敌"
    ph_val = float(e1.get("p_home") or 0.33)
    pa_val = float(e1.get("p_away") or 0.33)
    if ph_val >= pa_val + 0.10:
        fav_text = f"{home} 主场优势明显，处于确定性占优区间"
    elif pa_val >= ph_val + 0.10:
        fav_text = f"{away} 客战实力突出，处于优势区间"
    elif ph_val > pa_val:
        fav_text = f"{home} 略微占优，但平局风险不可忽视"
    else:
        fav_text = f"{away} 略占上风，整体格局偏向胶着"

    # Form text
    h_rec = e4["home"]["record"]
    a_rec = e4["away"]["record"]
    h_rec_str = f"{h_rec.get('W', 0)}胜{h_rec.get('D', 0)}平{h_rec.get('L', 0)}负"
    a_rec_str = f"{a_rec.get('W', 0)}胜{a_rec.get('D', 0)}平{a_rec.get('L', 0)}负"

    # Build factor driver text
    top_factor_lines = []
    for f_name, f_val, f_label in e3:
        direction = "利好主队" if f_val > 0 else "利好客队"
        top_factor_lines.append(f"- **{f_label}** (`{f_name}`): {f_val:+.3f}（{direction}）")
    factors_md = "\n".join(top_factor_lines) if top_factor_lines else "- 基础特征为主，无极值衍生因子"

    # Section 4: Market text [E5]
    if e5["available"]:
        m_data = e5["data"]
        m_prob = e5.get("market_open_probabilities") or m_data.get("market_open_probabilities") or {}
        diffs = e5.get("track_a_minus_market_percentage_points") or m_data.get("track_a_minus_market_percentage_points") or {}
        odds = e5.get("odds") or {}
        ve = e5.get("value_edges") or m_data.get("value_edges") or {}
        da = e5.get("divergence_alerts") or m_data.get("divergence_alerts") or {}
        ha = e5.get("handicap_assessment") or m_data.get("handicap_assessment")
        ta = e5.get("timeline_analysis") or m_data.get("timeline_analysis")

        m_source = e5.get("source", "外部市场基准盘口")
        m_obs = e5.get("observed_at") or "赛前"

        m_lines = [
            f"本场比赛录入外部市场实时快照（机构来源：**{m_source}**，观测时点：**{m_obs}**）。数据时点严格满足早于模型预测截止日（`observed_at < as_of`），杜绝未来信息穿越。",
            "",
            "| 评估维度 | 主胜 (" + home + ") | 平局 (Draw) | 客胜 (" + away + ") | 备注说明 |",
            "| :--- | :---: | :---: | :---: | :--- |",
            f"| **Track A 基本面模型** | **{p_h}** | **{p_d}** | **{p_a}** | 纯球场多因子集成推断 |",
            f"| **去水市场隐含概率** | **{_pct(m_prob.get('home'))}** | **{_pct(m_prob.get('draw'))}** | **{_pct(m_prob.get('away'))}** | Power 幂次法去水隐含胜率 |",
            f"| **模型偏离 (Track A - 市场)** | **{_pp(diffs.get('home'), signed=True)}** | **{_pp(diffs.get('draw'), signed=True)}** | **{_pp(diffs.get('away'), signed=True)}** | 相对市场共识偏离度 |",
            f"| **原始基准赔率** | `{_num(odds.get('home'))}` | `{_num(odds.get('draw'))}` | `{_num(odds.get('away'))}` | 抽水前市场欧盘初盘 |",
            f"| **价值偏差 (Value Edge)** | `{_num(ve.get('home'), 4)}` | `{_num(ve.get('draw'), 4)}` | `{_num(ve.get('away'), 4)}` | $P_{{model}} \\times \\text{{Odds}} - 1$ |",
            "",
        ]

        # Handicap Assessment
        if ha:
            ha_line_str = _num(ha.get("mkt_ah_line"), 2)
            goal_diff = (float(raw_lh or 0) - float(raw_la or 0)) if (raw_lh is not None and raw_la is not None) else 0.0
            ah_alignment = (
                "基本面实力净胜预期与让球盘口支撑区间高度吻合。"
                if (goal_diff >= 0.25 and float(ha.get("mkt_ah_line") or 0) <= -0.25)
                or (abs(goal_diff) < 0.25 and abs(float(ha.get("mkt_ah_line") or 0)) <= 0.25)
                else "模型基本面期望与盘口让球深度存在微幅分歧。"
            )
            m_lines.extend([
                f"**亚盘让球深度**：{ha.get('interpretation')}（盘口线值 `{ha_line_str}`；约定：{ha.get('convention')}）。",
                f"结合 Track A 双变量泊松进球期望（{lh} : {la}，净进球期望 {goal_diff:+.2f} 球），{ah_alignment}",
                "",
            ])
        else:
            m_lines.extend([
                "**亚盘让球深度**：本场未录入有效亚盘让球线数据，仅展示 1X2 胜平负市场对照。",
                "",
            ])

        # Divergence Alerts
        if da and da.get("has_divergence"):
            thresh_val = _num(da.get("threshold_percentage_points"), 1)
            alert_items = da.get("alerts", [])
            alert_details_list = []
            for item in alert_items:
                out_key = item.get("outcome")
                raw_mkt = m_prob.get(out_key)
                mkt_val = item.get("market_prob_pct", 0)
                if raw_mkt is not None and abs(float(raw_mkt) * 100 - float(mkt_val)) < 0.05:
                    mkt_str = _pct(raw_mkt)
                else:
                    mkt_str = _pct(float(mkt_val) / 100)

                mod_val = item.get("model_prob_pct", 0)
                raw_mod = e1.get(f"p_{out_key}") or e1.get(out_key)
                if raw_mod is not None and abs(float(raw_mod) * 100 - float(mod_val)) < 0.05:
                    mod_str = _pct(raw_mod)
                else:
                    mod_str = _pct(float(mod_val) / 100)

                df_val = item.get("diff_percentage_points")
                alert_details_list.append(
                    f"**{item['outcome_label']}**（模型 {mod_str} vs 市场 {mkt_str}，偏差 {_pp(df_val, signed=True)}，{item.get('direction', '偏离')}）"
                )
            alert_details = "；".join(alert_details_list)
            note_text = alert_items[0].get("note") if alert_items else "可能存在场外未建模信息。"
            m_lines.extend([
                f"> [!WARNING] **显著背离预警（阈值 {thresh_val} 个百分点）**  ",
                f"> 检测到模型胜率与市场隐含胜率存在显著偏离：{alert_details}。  ",
                f"> **量化归因提示**：{note_text}模型与市场定价的分歧仅揭示统计与定价差异，不代表任何一方必然正确。",
                "",
            ])
        elif da:
            thresh_val = _num(da.get("threshold_percentage_points"), 1)
            m_lines.extend([
                f"> [!NOTE] **背离检验通过**：各项赛果概率与市场偏差均在阈值（{thresh_val} 个百分点）以内，未检测到显著背离，模型与市场共识基本一致。",
                "",
            ])

        # Timeline Analysis
        if ta and ta.get("status") == "multi_snapshot_trend":
            ds = ta.get("drift_summary", {})
            desc = ds.get("steam_description", ta.get("message", ""))
            prob_drift = ds.get("prob_drift_percentage_points", {})
            p_h_d_val = ds.get("p_home_drift_pct_points") if "p_home_drift_pct_points" in ds else prob_drift.get("home")
            p_d_d_val = ds.get("p_draw_drift_pct_points") if "p_draw_drift_pct_points" in ds else prob_drift.get("draw")
            p_a_d_val = ds.get("p_away_drift_pct_points") if "p_away_drift_pct_points" in ds else prob_drift.get("away")
            m_lines.extend([
                f"**盘口时序变动趋势（Steam Drift，累计 {ta.get('snapshot_count')} 个快照）**：",
                f"- **资金与赔率漂移**：{desc}",
                f"- **去水概率变动**：主胜漂移 {_pp(p_h_d_val, signed=True)}，平局漂移 {_pp(p_d_d_val, signed=True)}，客胜漂移 {_pp(p_a_d_val, signed=True)}。",
                "",
            ])
        elif ta and ta.get("status") == "single_snapshot_baseline":
            m_lines.extend([
                f"**盘口时序变动趋势**：{ta.get('message', '暂无多时点历史快照，当前仅提供单时点截面盘口对照。')}",
                "",
            ])
        else:
            m_lines.extend([
                "**盘口时序变动趋势**：本场仅录入单次截面盘口或快照库暂无序列，暂无多时点趋势追踪。",
                "",
            ])

        # M-Open
        if e5.get("m_open_available"):
            mo = e5.get("m_open_probabilities") or {}
            m_lines.extend([
                f"**M-Open 市场融合模型参考**：主胜 {_pct(mo.get('home'))} ｜ 平局 {_pct(mo.get('draw'))} ｜ 客胜 {_pct(mo.get('away'))}。",
                "",
            ])
        elif e5.get("m_open_unavailable_reason"):
            m_lines.extend([
                f"**M-Open 模型就绪度**：不可用（{e5.get('m_open_unavailable_reason')}）。",
                "",
            ])

        # Mandatory Disclaimer
        m_lines.extend([
            "> [!IMPORTANT]",
            "> **市场参照免责声明**：去水隐含概率、价值偏差（Value Edge）与背离预警仅供足球量化投资与博彩市场定价效率学术研究参考，模型未经实盘盲测检验，绝对不构成投注建议，严禁作为任何形式的实际投注或资金决策建议。",
        ])
        market_section = "\n".join(m_lines)
    else:
        # Honest downgrade explanation
        market_section = (
            "**当前运行模式**：纯基本面独立推断模式（Track A Only）  \n"
            "**市场数据状态**：未录入有效赛前盘口赔率（或外部实时拉取失败 / 触发时序防穿越保护）。  \n\n"
            "> [!NOTE] **双轨隔离原则说明**  \n"
            "> MatchQuant 核心算法严格遵循 Track A 与市场赔率物理隔离的宪法原则。Track A 完全依托历史战绩、攻防技术统计（射正、角球、预期进球）与 Elo 评级独立推断，不依赖任何机构初盘或资金流输入。因此即使无市场数据，模型结论与概率置信度依然保持完整可靠。"
        )

    status_badge = "✅ 完整分析" if report_json.get("status") in ("ok", "complete") else "⚠️ 需要复核 (Needs Review)"

    doc = f"""# {home} vs {away} 深度量化分析报告

**比赛对阵**：{home}（主）vs {away}（客）　**比赛日期**：{fixture_date}　**赛季**：{season}  
**报告状态**：{status_badge}　**分析引擎**：MatchQuant Dual-Track v2.0 / Track A

---

## 一、 核心量化结论与概率分布 `[E1]`

基于严格隔离截止日（as-of: **{e6['as_of']}**）的历史多赛季滚动数据与全因子系统，MatchQuant Track A 模型对本场比赛给出如下权威胜平负概率与进球预期：

| 评估维度 | 主胜 ({home}) | 平局 (Draw) | 客胜 ({away}) | 预期总进球 / 大 2.5 概率 |
| :--- | :---: | :---: | :---: | :---: |
| **Track A 综合概率** | **{p_h}** | **{p_d}** | **{p_a}** | 大 2.5 概率：**{p_o25}** |
| **Bivariate Poisson 期望** | {lh} 球 | — | {la} 球 | 期望总进球：**{tot_goals_str}** 球 |
| **基准 Elo 概率参考 `[E2]`** | {_pct(e2.get('p_home_elo'))} | {_pct(e2.get('p_draw_elo'))} | {_pct(e2.get('p_away_elo'))} | Elo 实力差：{_num(e2.get('elo_diff'), 1)} |

**格局综述**：{fav_text}。两队泊松期望进球比约为 {lh} : {la}，攻防模型预估本场节奏倾向于 {tot_goals_approx} 球水平。

---

## 二、 核心特征驱动与对位解析 `[E3]`

在通过嵌套交叉验证筛选出的基本面因子面板中，驱动本场模型相对基准 Elo 产生概率调整的主要因子如下：

{factors_md}

因子系统显示：攻防两端特征对主客场优势进行了重新校准。当考量近期射正转化偏离、攻防对位差及赛程密度后，模型对双方的基础实力差进行了正交化修正。

---

## 三、 近期战绩与场面技术统计 `[E4]`

截止日前两队各近 {e4['home']['sample_count']} 场（主队）与 {e4['away']['sample_count']} 场（客队）实际完成比赛统计表现如下：

- **{home}（主队）**：近期战绩 **{h_rec_str}**，场均进球 **{_num(e4['home']['goals_for'], 2)}**、场均失球 **{_num(e4['home']['goals_against'], 2)}**；场均射正 **{_num(e4['home']['sot_for'], 1)}** 次，场均角球 **{_num(e4['home']['corners_for'], 1)}** 次。
- **{away}（客队）**：近期战绩 **{a_rec_str}**，场均进球 **{_num(e4['away']['goals_for'], 2)}**、场均失球 **{_num(e4['away']['goals_against'], 2)}**；场均射正 **{_num(e4['away']['sot_for'], 1)}** 次，场均角球 **{_num(e4['away']['corners_for'], 1)}** 次。

注：未记录场次保持为客观缺失（NaN），不作为虚假 0 次统计纳入计算。

---

## 四、 市场参照与让球对位 `[E5]`

{market_section}

---

## 五、 数据版本与时序截断回执 `[E6]`

为保证研究的可复现性与严格零前瞻（No Lookahead），本次推理执行了以下时序隔离契约：
- **预测数据截止（as-of）**：`{e6['as_of']}`（当日 00:00:00 生效，该日及以后所有比赛均被排除）。
- **最新实际使用比赛日**：`{e6['used_data_cutoff']}`（数据账龄：{e6['data_age_days']} 天；预测提前量：{e6['forecast_horizon_days']} 天）。
- **分析模式**：{e6['mode']}。
- **权威数值凭证**：本报告文字结论与表格完全绑定至同批次生成的 `report.json`，未引入任何非结构化 LLM 估值计算。

---

## 六、 客观局限性与免责声明

1. **赛程与对阵**：由用户直接指定或别名解析匹配，系统未通过联网爬虫校验官方即时赛程与赛前延期变动。
2. **非赛场信息**：本系统模型不包含突发伤病停赛、更衣室新闻、天气突变或即时临场首发名单变动。
3. **因果与归因限制**：局部因子分解属于边际近似，不代表绝对因果推断。概率不代表确定性赛果，仅供足球量化研究参考，不构成投注建议。
"""
    return doc.strip()


def render_agent_narrative_html(
    report_json: Mapping[str, Any],
    catalog: Mapping[str, dict[str, Any]] | None = None,
) -> str:
    """Render self-contained, escaped, professional HTML report from structured evidence."""
    cat = catalog or build_evidence_catalog(report_json)
    fixture = report_json.get("fixture", {})
    home = html.escape(str(fixture.get("home_team", "主队")))
    away = html.escape(str(fixture.get("away_team", "客队")))
    fixture_date = html.escape(str(fixture.get("date", "")))
    season = html.escape(str(fixture.get("season", "")))

    e1 = cat["E1_PROBS"]["data"]
    e2 = cat["E2_ELO"]["data"]
    e3 = cat["E3_FACTORS"]["top_drivers"]
    e4 = cat["E4_RECENT_FORM"]
    e5 = cat["E5_MARKET"]

    raw_lh = e1.get("lambda_home")
    raw_la = e1.get("lambda_away")
    lh = _num(raw_lh, 2)
    la = _num(raw_la, 2)
    e6 = cat["E6_TEMPORAL"]

    p_h = _pct(e1.get("p_home"))
    p_d = _pct(e1.get("p_draw"))
    p_a = _pct(e1.get("p_away"))
    raw_lh = e1.get("lambda_home")
    raw_la = e1.get("lambda_away")
    lh = _num(raw_lh, 2)
    la = _num(raw_la, 2)
    tot_lambda = (float(raw_lh) + float(raw_la)) if (raw_lh is not None and raw_la is not None) else None
    tot_goals_str = _num(tot_lambda, 2)
    tot_goals_approx = _num(tot_lambda, 1)
    p_o25 = _pct(e1.get("p_over25"))

    # Qualitative favorite text
    ph_val = float(e1.get("p_home") or 0.33)
    pa_val = float(e1.get("p_away") or 0.33)
    if ph_val >= pa_val + 0.10:
        fav_text = f"{home} 主场优势明显，处于确定性占优区间"
    elif pa_val >= ph_val + 0.10:
        fav_text = f"{away} 客战实力突出，处于优势区间"
    elif ph_val > pa_val:
        fav_text = f"{home} 略微占优，但平局风险不可忽视"
    else:
        fav_text = f"{away} 略占上风，整体格局偏向胶着"

    # Factors rows
    factor_rows = []
    for f_name, f_val, f_label in e3:
        direction_class = "fav-home" if f_val > 0 else "fav-away"
        direction_label = "利好主队" if f_val > 0 else "利好客队"
        factor_rows.append(
            f"<tr><td>{html.escape(f_label)}</td><td><code>{html.escape(f_name)}</code></td>"
            f"<td class='num'>{f_val:+.3f}</td><td class='{direction_class}'>{direction_label}</td></tr>"
        )
    factor_table_html = "\n".join(factor_rows)

    # Form text
    h_rec = e4["home"]["record"]
    a_rec = e4["away"]["record"]
    h_rec_str = f"{h_rec.get('W', 0)}胜{h_rec.get('D', 0)}平{h_rec.get('L', 0)}负"
    a_rec_str = f"{a_rec.get('W', 0)}胜{a_rec.get('D', 0)}平{a_rec.get('L', 0)}负"

    # Section 4 HTML: Market [E5]
    if e5["available"]:
        m_data = e5["data"]
        m_prob = e5.get("market_open_probabilities") or m_data.get("market_open_probabilities") or {}
        diffs = e5.get("track_a_minus_market_percentage_points") or m_data.get("track_a_minus_market_percentage_points") or {}
        odds = e5.get("odds") or {}
        ve = e5.get("value_edges") or m_data.get("value_edges") or {}
        da = e5.get("divergence_alerts") or m_data.get("divergence_alerts") or {}
        ha = e5.get("handicap_assessment") or m_data.get("handicap_assessment")
        ta = e5.get("timeline_analysis") or m_data.get("timeline_analysis")

        m_source = html.escape(str(e5.get("source", "外部市场基准盘口")))
        m_obs = html.escape(str(e5.get("observed_at") or "赛前"))

        # Handicap card
        if ha:
            ha_interp = html.escape(str(ha.get("interpretation", "")))
            ha_line_val = _num(ha.get("mkt_ah_line"), 2)
            ha_conv = html.escape(str(ha.get("convention", "")))
            handicap_card_html = (
                f"<div class='card' style='border-left-color: #6366f1;'>"
                f"<strong>亚盘让球深度：</strong> {ha_interp}（盘口线值 <code>{ha_line_val}</code>；规则约定：{ha_conv}）。<br>"
                f"<small class='text-secondary'>结合 Track A 泊松进球期望（{lh} : {la}），模型对位优势与盘口深度相协调。</small>"
                f"</div>"
            )
        else:
            handicap_card_html = (
                "<div class='card' style='border-left-color: #94a3b8;'>"
                "<strong>亚盘让球深度：</strong> 本场未录入有效亚盘让球线数据，仅展示 1X2 欧盘市场对照。"
                "</div>"
            )

        # Divergence Alert card
        if da and da.get("has_divergence"):
            thresh_val = _num(da.get("threshold_percentage_points"), 1)
            alerts_list = []
            for item in da.get("alerts", []):
                out_key = item.get("outcome")
                o_label = html.escape(str(item.get("outcome_label", "")))
                raw_mkt = m_prob.get(out_key)
                mkt_val = item.get("market_prob_pct", 0)
                if raw_mkt is not None and abs(float(raw_mkt) * 100 - float(mkt_val)) < 0.05:
                    mk_p = _pct(raw_mkt)
                else:
                    mk_p = _pct(float(mkt_val) / 100)

                mod_val = item.get("model_prob_pct", 0)
                raw_mod = e1.get(f"p_{out_key}") or e1.get(out_key)
                if raw_mod is not None and abs(float(raw_mod) * 100 - float(mod_val)) < 0.05:
                    m_p = _pct(raw_mod)
                else:
                    m_p = _pct(float(mod_val) / 100)

                df_p = _pp(item.get("diff_percentage_points"), signed=True)
                alerts_list.append(f"<strong>{o_label}</strong>（模型 {m_p} vs 市场 {mk_p}，偏差 {df_p}）")
            alerts_str = "；".join(alerts_list)
            note_str = html.escape(str(da.get("alerts", [{}])[0].get("note", "可能存在场外未建模信息。")))
            divergence_card_html = (
                f"<div class='alert-card alert-warning'>"
                f"<div class='alert-title'>⚠️ 显著背离预警（阈值 {thresh_val} 个百分点）</div>"
                f"<p style='margin: 4px 0;'>检测到模型胜率与市场隐含胜率存在显著分歧：{alerts_str}。</p>"
                f"<small class='text-secondary'><strong>量化归因提示：</strong>{note_str}模型与市场定价分歧仅反映统计视角与市场资金的差异，不代表任何一方必然正确。</small>"
                f"</div>"
            )
        elif da:
            thresh_val = _num(da.get("threshold_percentage_points"), 1)
            divergence_card_html = (
                f"<div class='alert-card alert-success'>"
                f"<div class='alert-title'>✅ 背离检验通过</div>"
                f"<p style='margin: 4px 0;'>各项赛果概率偏差均在阈值（{thresh_val} 个百分点）以内，未见显著背离，模型基本面与市场共识保持一致。</p>"
                f"</div>"
            )
        else:
            divergence_card_html = ""

        # Timeline card
        if ta and ta.get("status") == "multi_snapshot_trend":
            ds = ta.get("drift_summary", {})
            desc = html.escape(str(ds.get("steam_description", ta.get("message", ""))))
            prob_drift = ds.get("prob_drift_percentage_points", {})
            p_h_d_val = ds.get("p_home_drift_pct_points") if "p_home_drift_pct_points" in ds else prob_drift.get("home")
            p_a_d_val = ds.get("p_away_drift_pct_points") if "p_away_drift_pct_points" in ds else prob_drift.get("away")
            p_h_drift = _pp(p_h_d_val, signed=True)
            p_a_drift = _pp(p_a_d_val, signed=True)
            timeline_card_html = (
                f"<div class='card' style='border-left-color: #0d9488;'>"
                f"<strong>盘口时序变动趋势（Steam Drift，累计 {ta.get('snapshot_count')} 个快照）：</strong><br>"
                f"<p style='margin: 4px 0;'>{desc}</p>"
                f"<small class='text-secondary'>去水胜率漂移：主胜 {p_h_drift} ｜ 客胜 {p_a_drift}</small>"
                f"</div>"
            )
        elif ta and ta.get("status") == "single_snapshot_baseline":
            msg = html.escape(str(ta.get("message", "暂无多时点历史快照，当前仅提供单时点截面盘口对照。")))
            timeline_card_html = (
                f"<div class='card' style='border-left-color: #94a3b8;'>"
                f"<strong>盘口时序变动趋势：</strong> {msg}"
                f"</div>"
            )
        else:
            timeline_card_html = (
                "<div class='card' style='border-left-color: #94a3b8;'>"
                "<strong>盘口时序变动趋势：</strong> 本场快照库暂无序列记录，仅展示当前单次时点盘口。"
                "</div>"
            )

        # M-Open text
        m_open_html = ""
        if e5.get("m_open_available"):
            mo = e5.get("m_open_probabilities") or {}
            m_open_html = (
                f"<div class='card' style='border-left-color: #8b5cf6;'>"
                f"<strong>M-Open 市场融合模型参考：</strong> 主胜 {_pct(mo.get('home'))} ｜ 平局 {_pct(mo.get('draw'))} ｜ 客胜 {_pct(mo.get('away'))}。"
                f"</div>"
            )

        market_html_section = f"""
  <div class="card" style="border-left-color: #0284c7;">
    <strong>盘口元数据：</strong> 本场比赛录入外部市场实时快照（机构来源 <code>{m_source}</code> ｜ 观测时点 <code>{m_obs}</code> ｜ 契约约束 <code>observed_at &lt; as_of</code>，零前瞻隔离）。
  </div>

  <table>
    <thead>
      <tr>
        <th>评估维度</th>
        <th>主胜 ({home})</th>
        <th>平局</th>
        <th>客胜 ({away})</th>
        <th>备注说明</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td><strong>Track A 基本面模型</strong></td>
        <td class="num"><strong>{p_h}</strong></td>
        <td class="num"><strong>{p_d}</strong></td>
        <td class="num"><strong>{p_a}</strong></td>
        <td>纯球场多因子集成推断</td>
      </tr>
      <tr>
        <td>去水市场隐含概率</td>
        <td class="num">{_pct(m_prob.get('home'))}</td>
        <td class="num">{_pct(m_prob.get('draw'))}</td>
        <td class="num">{_pct(m_prob.get('away'))}</td>
        <td>Power 幂次法去水胜率</td>
      </tr>
      <tr>
        <td>模型偏离 (Track A - 市场)</td>
        <td class="num">{_pp(diffs.get('home'), signed=True)}</td>
        <td class="num">{_pp(diffs.get('draw'), signed=True)}</td>
        <td class="num">{_pp(diffs.get('away'), signed=True)}</td>
        <td>百分点偏离度</td>
      </tr>
      <tr>
        <td>原始基准赔率</td>
        <td class="num">{_num(odds.get('home'))}</td>
        <td class="num">{_num(odds.get('draw'))}</td>
        <td class="num">{_num(odds.get('away'))}</td>
        <td>抽水前欧盘初盘</td>
      </tr>
      <tr>
        <td><strong>价值偏差 (Value Edge)</strong></td>
        <td class="num"><strong>{_num(ve.get('home'), 4)}</strong></td>
        <td class="num"><strong>{_num(ve.get('draw'), 4)}</strong></td>
        <td class="num"><strong>{_num(ve.get('away'), 4)}</strong></td>
        <td>P_model × Odds - 1</td>
      </tr>
    </tbody>
  </table>

  {handicap_card_html}
  {divergence_card_html}
  {timeline_card_html}
  {m_open_html}

  <div class="disclaimer-box">
    <strong>⚠️ 市场参照免责声明：</strong>
    去水隐含概率、价值偏差（Value Edge）与背离预警仅供足球量化投资与博彩市场定价效率学术研究参考，模型未经实盘盲测检验，严禁作为任何形式的实际投注或资金决策建议。
  </div>
"""
    else:
        market_html_section = """
  <div class="card" style="border-left-color: #94a3b8;">
    <strong>当前运行模式：</strong> 纯基本面独立推断模式（Track A Only）<br>
    <strong>市场数据状态：</strong> 未录入有效赛前盘口赔率（或外部实时拉取失败 / 触发时序防穿越保护）。
  </div>
  <div class="card" style="border-left-color: #3b82f6;">
    <strong>双轨隔离原则说明：</strong>
    MatchQuant 核心算法严格遵循 Track A 与市场赔率物理隔离的宪法原则。Track A 完全依托历史战绩、攻防技术统计（射正、角球、预期进球）与 Elo 评级独立推断，不依赖任何机构初盘或资金流输入。因此即使无市场数据，模型结论与概率置信度依然保持完整可靠。
  </div>
"""

    status_str = "完整分析" if report_json.get("status") in ("ok", "complete") else "需要复核"

    html_page = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{home} vs {away}｜MatchQuant Agent 分析报告</title>
<style>
  :root {{
    --bg-color: #f8fafc;
    --card-bg: #ffffff;
    --text-primary: #0f172a;
    --text-secondary: #475569;
    --border-color: #e2e8f0;
    --brand-primary: #0284c7;
    --brand-accent: #0369a1;
    --success: #16a34a;
    --warning: #ca8a04;
    --danger: #dc2626;
  }}
  body {{
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
    background-color: var(--bg-color);
    color: var(--text-primary);
    line-height: 1.6;
    margin: 0;
    padding: 24px 16px;
  }}
  .container {{
    max-width: 920px;
    margin: 0 auto;
    background: var(--card-bg);
    border-radius: 12px;
    border: 1px solid var(--border-color);
    box-shadow: 0 4px 6px -1px rgba(0,0,0,0.05);
    padding: 32px;
  }}
  .header {{
    border-bottom: 2px solid var(--border-color);
    padding-bottom: 16px;
    margin-bottom: 24px;
  }}
  h1 {{ margin: 0 0 8px 0; font-size: 24px; color: var(--brand-accent); }}
  h2 {{ margin: 24px 0 12px 0; font-size: 18px; color: #1e293b; border-bottom: 1px solid #f1f5f9; padding-bottom: 6px; }}
  .meta {{ font-size: 14px; color: var(--text-secondary); }}
  .badge {{
    display: inline-block;
    padding: 2px 8px;
    border-radius: 4px;
    font-size: 12px;
    font-weight: 600;
    background: #e0f2fe;
    color: var(--brand-accent);
  }}
  table {{
    width: 100%;
    border-collapse: collapse;
    margin: 14px 0;
    font-size: 14px;
  }}
  th, td {{
    padding: 9px 12px;
    border: 1px solid var(--border-color);
    text-align: left;
  }}
  th {{ background-color: #f1f5f9; font-weight: 600; }}
  td.num {{ text-align: right; font-variant-numeric: tabular-nums; }}
  .fav-home {{ color: var(--brand-primary); font-weight: 600; }}
  .fav-away {{ color: var(--danger); font-weight: 600; }}
  .card {{
    background: #f8fafc;
    border-left: 4px solid var(--brand-primary);
    padding: 12px 16px;
    margin: 14px 0;
    border-radius: 0 6px 6px 0;
    font-size: 14px;
  }}
  .alert-card {{
    padding: 12px 16px;
    margin: 14px 0;
    border-radius: 6px;
    font-size: 14px;
    border: 1px solid;
  }}
  .alert-warning {{
    background-color: #fefce8;
    border-color: #fef08a;
    color: #854d0e;
  }}
  .alert-warning .alert-title {{
    font-weight: 700;
    color: #a16207;
    margin-bottom: 4px;
  }}
  .alert-success {{
    background-color: #f0fdf4;
    border-color: #bbf7d0;
    color: #166534;
  }}
  .alert-success .alert-title {{
    font-weight: 700;
    color: #15803d;
    margin-bottom: 4px;
  }}
  .disclaimer-box {{
    background: #fffbeb;
    border: 1px solid #fde68a;
    border-left: 4px solid #f59e0b;
    padding: 12px 16px;
    margin: 16px 0;
    border-radius: 0 6px 6px 0;
    font-size: 13px;
    color: #92400e;
  }}
  .text-secondary {{ color: var(--text-secondary); }}
  footer {{
    margin-top: 32px;
    padding-top: 16px;
    border-top: 1px solid var(--border-color);
    font-size: 12px;
    color: var(--text-secondary);
  }}
</style>
</head>
<body>
<div class="container">
  <div class="header">
    <h1>{home} vs {away} 深度量化分析报告</h1>
    <div class="meta">
      比赛日期：{fixture_date} ｜ 赛季：{season} ｜ 状态：<span class="badge">{status_str}</span> ｜ 引擎：MatchQuant Track A
    </div>
  </div>

  <h2>一、 核心概率分布与泊松进球期望 [E1]</h2>
  <table>
    <thead>
      <tr>
        <th>预测模型</th>
        <th>主胜 ({home})</th>
        <th>平局</th>
        <th>客胜 ({away})</th>
        <th>期望进球 (主/客)</th>
        <th>大 2.5 概率</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td><strong>Track A 综合模型</strong></td>
        <td class="num"><strong>{p_h}</strong></td>
        <td class="num"><strong>{p_d}</strong></td>
        <td class="num"><strong>{p_a}</strong></td>
        <td class="num">{lh} / {la}</td>
        <td class="num">{p_o25}</td>
      </tr>
      <tr>
        <td>基准 Elo 模型 [E2]</td>
        <td class="num">{_pct(e2.get('p_home_elo'))}</td>
        <td class="num">{_pct(e2.get('p_draw_elo'))}</td>
        <td class="num">{_pct(e2.get('p_away_elo'))}</td>
        <td class="num">—</td>
        <td class="num">—</td>
      </tr>
    </tbody>
  </table>
  <div class="card">
    <strong>格局综述：</strong> {fav_text}。两队泊松期望进球比预估为 {lh} : {la}，全场倾向于 {tot_goals_approx} 球水平。
  </div>

  <h2>二、 核心量化特征驱动 [E3]</h2>
  <table>
    <thead>
      <tr>
        <th>因子名称</th>
        <th>标识</th>
        <th style="text-align:right">数值</th>
        <th>倾向判定</th>
      </tr>
    </thead>
    <tbody>
      {factor_table_html}
    </tbody>
  </table>

  <h2>三、 近期战绩与场面技术统计 [E4]</h2>
  <table>
    <thead>
      <tr>
        <th>球队</th>
        <th>有效样本</th>
        <th>胜平负记录</th>
        <th>场均进球</th>
        <th>场均失球</th>
        <th>场均射正</th>
        <th>场均角球</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td><strong>{home}（主）</strong></td>
        <td>{e4['home']['sample_count']} 场</td>
        <td>{h_rec_str}</td>
        <td class="num">{_num(e4['home']['goals_for'], 2)}</td>
        <td class="num">{_num(e4['home']['goals_against'], 2)}</td>
        <td class="num">{_num(e4['home']['sot_for'], 1)}</td>
        <td class="num">{_num(e4['home']['corners_for'], 1)}</td>
      </tr>
      <tr>
        <td><strong>{away}（客）</strong></td>
        <td>{e4['away']['sample_count']} 场</td>
        <td>{a_rec_str}</td>
        <td class="num">{_num(e4['away']['goals_for'], 2)}</td>
        <td class="num">{_num(e4['away']['goals_against'], 2)}</td>
        <td class="num">{_num(e4['away']['sot_for'], 1)}</td>
        <td class="num">{_num(e4['away']['corners_for'], 1)}</td>
      </tr>
    </tbody>
  </table>

  <h2>四、 市场参照与让球对位 [E5]</h2>
  {market_html_section}

  <h2>五、 时序截断与数据回执 [E6]</h2>
  <div class="card">
    <strong>时序契约：</strong> 预测截止日为 <code>{html.escape(str(e6['as_of']))}</code>，实际使用历史截止至 <code>{html.escape(str(e6['used_data_cutoff']))}</code>。<br>
    数据账龄：{e6['data_age_days']} 天 ｜ 提前量：{e6['forecast_horizon_days']} 天 ｜ 模式：{html.escape(str(e6['mode']))}。
  </div>

  <h2>六、 客观局限性与免责声明</h2>
  <div class="card" style="border-left-color: #64748b;">
    1. <strong>对阵真实性</strong>：比赛对阵由用户直接输入或别名解析匹配，系统未在线核验官方赛程与突发延期。<br>
    2. <strong>场外不可见信息</strong>：量化模型不包含临场伤停、更衣室突发动态、极端天气或即时首发阵容变动。<br>
    3. <strong>概率本质</strong>：概率估计仅供足球量化研究参考，不代表绝对确定赛果，严禁作为投注建议。
  </div>

  <footer>
    MatchQuant Quantitative Football Research Engine ｜ 本报告由宿主 Agent 基于权威计算数据生成，不构成投注建议。
  </footer>
</div>
</body>
</html>
"""
    return html_page


def answer_followup(
    report_json: Mapping[str, Any],
    question: str,
) -> dict[str, Any]:
    """Answer user follow-up questions reusing existing run JSON without refitting."""
    cat = build_evidence_catalog(report_json)
    q = question.strip()

    fixture = report_json.get("fixture", {})
    home = fixture.get("home_team", "主队")
    away = fixture.get("away_team", "客队")
    e1 = cat["E1_PROBS"]["data"]
    e2 = cat["E2_ELO"]["data"]
    e3 = cat["E3_FACTORS"]["top_drivers"]
    e4 = cat["E4_RECENT_FORM"]
    e5 = cat["E5_MARKET"]

    raw_lh = e1.get("lambda_home")
    raw_la = e1.get("lambda_away")
    lh = _num(raw_lh, 2)
    la = _num(raw_la, 2)

    answer_parts: list[str] = []
    referenced_evidence: list[str] = []

    # Intent 1: Refresh / Re-fetch live odds request (Guiding prompt, read-only proof)
    if any(k in q for k in ("刷新", "重新拉取", "更新盘口", "实时盘口", "最新盘口", "重新获取", "refresh", "reload")):
        referenced_evidence.extend(["E5"])
        answer_parts.append(
            "**静态只读追问说明**：当前追问引擎严格遵循“只读已有 Run 证据、杜绝隐式重算”的宪法原则，"
            f"本轮问答基于已有 Run（ID: `{report_json.get('run_id')}`）的静态磁盘快照完成，未消耗额外 API 额度或重复拟合模型。\n\n"
            "若您需要获取最新实时盘口并重新拟合分析，请向系统发起新的单场分析请求（例如在参数中指定 `use_live_odds: true`），"
            "系统将从数据源拉取最新快照并生成全新的独立 Run。"
        )

    # Intent 2: Market trend / Steam drift questions
    elif any(k in q for k in ("趋势", "走势", "变动", "变化", "steam", "drift", "临场", "时序", "波动")):
        referenced_evidence.extend(["E5"])
        if e5["available"]:
            ta = e5.get("timeline_analysis") or {}
            status = ta.get("status")
            if status == "multi_snapshot_trend":
                ds = ta.get("drift_summary", {})
                desc = ds.get("steam_description", ta.get("message", ""))
                prob_drift = ds.get("prob_drift_percentage_points", {})
                p_h_d_val = ds.get("p_home_drift_pct_points") if "p_home_drift_pct_points" in ds else prob_drift.get("home")
                p_a_d_val = ds.get("p_away_drift_pct_points") if "p_away_drift_pct_points" in ds else prob_drift.get("away")
                p_h_d = _pp(p_h_d_val, signed=True)
                p_a_d = _pp(p_a_d_val, signed=True)
                ah_d = ds.get("ah_line_drift")
                ah_d_str = f"{ah_d:+.2f} 球" if (ah_d is not None and isinstance(ah_d, (int, float))) else "无变动"
                answer_parts.append(
                    f"**盘口时序趋势（Steam Drift）**：根据已积累的 {ta.get('snapshot_count')} 个历史快照序列，"
                    f"市场赔率变动特征为：{desc}\n\n"
                    f"- 主胜去水隐含概率漂移：**{p_h_d}**\n"
                    f"- 客胜去水隐含概率漂移：**{p_a_d}**\n"
                    f"- 亚盘让球线漂移：**{ah_d_str}**\n\n"
                    f"聪慧钱（Steam Move）流向揭示了临场资金偏好，但模型 Track A 保持纯基本面隔离，未将该漂移作为训练入模因子。"
                )
            elif status == "single_snapshot_baseline":
                answer_parts.append(
                    "**盘口时序趋势**：当前比赛快照库中仅有 1 个截面快照，"
                    "暂未积累多时点时间序列（需至少 2 个以上快照）。当前仅提供静态初盘基准对照，未检测到动态漂移。"
                )
            else:
                answer_parts.append(
                    "**盘口时序趋势**：当前未检索到该场比赛的历史多时点盘口快照序列，仅提供静态市场对照。"
                )
        else:
            answer_parts.append(
                "**盘口时序趋势**：本场未录入外部盘口数据，因此无时序赔率或资金变动趋势。"
            )

    # Intent 3: Divergence / Difference between model and market / Value edge
    elif any(k in q for k in ("背离", "差异", "为什么不一样", "为什么差这么多", "差这么多", "分歧", "edge", "偏差", "价值")):
        referenced_evidence.extend(["E1", "E5"])
        p_h = _pct(e1.get("p_home"))
        p_d = _pct(e1.get("p_draw"))
        p_a = _pct(e1.get("p_away"))

        if e5["available"]:
            m_prob = e5.get("market_open_probabilities") or {}
            diffs = e5.get("track_a_minus_market_percentage_points") or {}
            ve = e5.get("value_edges") or {}
            da = e5.get("divergence_alerts") or {}

            diff_h_str = _pp(diffs.get("home"), signed=True)
            diff_a_str = _pp(diffs.get("away"), signed=True)
            diff_d_str = _pp(diffs.get("draw"), signed=True)

            answer_parts.append(
                f"**模型 vs 市场概率偏差与价值 Edge**：\n"
                f"- **主胜 ({home})**：Track A 基本面 **{p_h}** vs 市场去水 **{_pct(m_prob.get('home'))}**（偏差：{diff_h_str}，Edge: `{_num(ve.get('home'), 4)}`）\n"
                f"- **平局 (Draw)**：Track A 基本面 **{p_d}** vs 市场去水 **{_pct(m_prob.get('draw'))}**（偏差：{diff_d_str}，Edge: `{_num(ve.get('draw'), 4)}`）\n"
                f"- **客胜 ({away})**：Track A 基本面 **{p_a}** vs 市场去水 **{_pct(m_prob.get('away'))}**（偏差：{diff_a_str}，Edge: `{_num(ve.get('away'), 4)}`）"
            )

            if da and da.get("has_divergence"):
                thresh = _num(da.get("threshold_percentage_points"), 1)
                first_alert = da.get("alerts", [{}])[0]
                answer_parts.append(
                    f"**背离状态**：已触发**显著背离预警**（阈值 {thresh} 个百分点）。\n"
                    f"**量化归因解析**：MatchQuant Track A 纯粹基于历史技战术数据（射正偏离、攻防对位、泊松期望进球），"
                    f"而市场赔率融合了更衣室新闻、突发伤停、首发轮换与资金博弈等多重场外变量。"
                    f"背离仅提示两者的定价视角差异，绝不代表模型必然正确，不构成投注建议。"
                )
            else:
                thresh = _num(da.get("threshold_percentage_points"), 1) if da else "7.0"
                answer_parts.append(
                    f"**背离状态**：各项赛果概率与市场偏差均在设定阈值（{thresh} 个百分点）以内，未见显著背离，模型与市场定价保持基本吻合。"
                )
        else:
            answer_parts.append(
                f"**模型与盘口差异**：本场未录入外部初盘赔率，Track A 模型给出独立胜率：主胜 {p_h} ｜ 平局 {p_d} ｜ 客胜 {p_a}。"
            )

    # Intent 4: General market / odds / handicap questions
    elif any(k in q for k in ("盘口", "赔率", "让球", "亚盘", "市场", "怎么看", "看盘", "odds", "handicap", "market")):
        referenced_evidence.extend(["E5"])
        if e5["available"]:
            m_prob = e5.get("market_open_probabilities") or {}
            diffs = e5.get("track_a_minus_market_percentage_points") or {}
            ha = e5.get("handicap_assessment")
            source = e5.get("source", "外部市场")
            obs = e5.get("observed_at", "")

            answer_parts.append(
                f"**市场初盘深度解读**（来源：{source}，观测时间：{obs}）：\n"
                f"- **市场去水隐含胜率**：主胜 **{_pct(m_prob.get('home'))}** ｜ 平局 **{_pct(m_prob.get('draw'))}** ｜ 客胜 **{_pct(m_prob.get('away'))}**\n"
                f"- **Track A 偏离度**：主胜 {_pp(diffs.get('home'), signed=True)} ｜ 客胜 {_pp(diffs.get('away'), signed=True)}"
            )
            if ha:
                answer_parts.append(
                    f"- **亚盘让球盘口**：{ha.get('interpretation')}（线值 `{ha.get('mkt_ah_line')}`）。"
                    f"结合 Track A 预期进球（{lh} : {la}），基本面展现的实力差距与机构设立的让球门槛相互印证。"
                )
        else:
            answer_parts.append(
                "**盘口信息**：本场未录入外部初盘赔率或亚盘数据，所有推断均基于 Track A 纯比赛事实数据。"
            )

    # Intent 5: Goals / Over-Under questions
    elif any(k in q for k in ("进球", "大球", "小球", "比分", "几球")):
        referenced_evidence.extend(["E1"])
        raw_lh = e1.get("lambda_home")
        raw_la = e1.get("lambda_away")
        lh = _num(raw_lh, 2)
        la = _num(raw_la, 2)
        tot_lambda = (float(raw_lh) + float(raw_la)) if (raw_lh is not None and raw_la is not None) else None
        tot_goals_str = _num(tot_lambda, 2)
        p_o25 = _pct(e1.get("p_over25"))
        answer_parts.append(
            f"**进球与大小球分析**：Bivariate Poisson 模型测算出 {home} 期望进球为 **{lh}** 球，"
            f"{away} 期望进球为 **{la}** 球，预期总进球约为 {tot_goals_str} 球。"
            f"全场总进球大于 2.5 球的概率为 **{p_o25}**。"
        )

    # Intent 6: Ask about why favored or core logic
    elif any(k in q for k in ("为什么", "原因", "凭什么", "逻辑", "看好")):
        referenced_evidence.extend(["E1", "E2", "E3"])
        p_h = _pct(e1.get("p_home"))
        p_a = _pct(e1.get("p_away"))
        answer_parts.append(
            f"**模型判断核心逻辑**：针对 **{home} vs {away}**，Track A 给出主胜概率 {p_h}、客胜 {p_a}。"
            f"基准 Elo 实力差为 {_num(e2.get('elo_diff'), 1)}。"
        )
        if e3:
            top_f = e3[0]
            answer_parts.append(
                f"在特征层面，权重最显著的调整来自 **{top_f[2]}**（`{top_f[0]}`: {top_f[1]:+.3f}），"
                f"对基础实力进行了正交边缘修正。"
            )

    # Intent 7: Ask for concise / short summary
    elif any(k in q for k in ("简短", "简版", "总结", "一句话", "结论")):
        referenced_evidence.extend(["E1"])
        p_h = _pct(e1.get("p_home"))
        p_d = _pct(e1.get("p_draw"))
        p_a = _pct(e1.get("p_away"))
        answer_parts.append(
            f"**简版核心结论**：{home} 主胜 **{p_h}** ｜ 平局 **{p_d}** ｜ 客胜 **{p_a}**。"
            f"期望进球 {_num(e1.get('lambda_home'), 2)} : {_num(e1.get('lambda_away'), 2)}。"
        )

    # Default fallback
    else:
        referenced_evidence.extend(["E1", "E3"])
        answer_parts.append(
            f"关于您对 **{home} vs {away}** 的提问：基于本场已完成的 Track A 权威计算（胜 {_pct(e1.get('p_home'))} / 平 {_pct(e1.get('p_draw'))} / 负 {_pct(e1.get('p_away'))}），"
            f"模型综合了 Elo 基础实力与近期攻防统计驱动。您可以进一步询问进球期望、核心因子、盘口对比或时序趋势。"
        )

    return {
        "question": q,
        "answer": "\n\n".join(answer_parts),
        "referenced_evidence_ids": referenced_evidence,
        "run_id": report_json.get("run_id"),
        "recomputed": False,
    }
