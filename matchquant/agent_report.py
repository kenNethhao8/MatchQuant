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
            "sot_for": home_ev.get("shots_on_target_for", {}).get("mean"),
            "corners_for": home_ev.get("corners_for", {}).get("mean"),
        },
        "away": {
            "team": away,
            "sample_count": away_ev.get("sample_count", 0),
            "record": away_ev.get("record_count_by_outcome", {}),
            "goals_for": away_ev.get("goals_for", {}).get("mean"),
            "goals_against": away_ev.get("goals_against", {}).get("mean"),
            "sot_for": away_ev.get("shots_on_target_for", {}).get("mean"),
            "corners_for": away_ev.get("corners_for", {}).get("mean"),
        },
    }

    catalog["E5_MARKET"] = {
        "id": "E5",
        "category": "market_comparison",
        "description": "初盘市场参照与让球优势",
        "available": bool(market.get("available")),
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

    # Market text
    market_section = "未提供有效初盘市场数据，本场预测完全基于纯基本面与技术统计 Track A。"
    if e5["available"]:
        m_data = e5["data"]
        m_prob = m_data.get("market_open_probabilities", {})
        diff_h = m_data.get("track_a_minus_market_percentage_points", {}).get("home")
        diff_str = f"{diff_h:+.1f} 个百分点" if diff_h is not None else "一致"
        market_section = (
            f"市场初盘隐含主胜概率为 {_pct(m_prob.get('home'))}、平局 {_pct(m_prob.get('draw'))}、客胜 {_pct(m_prob.get('away'))}。"
            f"MatchQuant Track A 基本面模型相对市场初盘，主胜概率偏离 {diff_str}。"
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

Track A 保持纯基本面与比赛事实独立挖掘，不以市场终盘或资金流向作为先验输入，确保了定量推断的独立性。

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
3. **因果与归因限制**：局部因子分解属于边际近似，不代表绝对因果推断。概率不代表确定性赛果，仅供足球量化研究参考。
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

    factor_rows = []
    for f_name, f_val, f_label in e3:
        direction_class = "fav-home" if f_val > 0 else "fav-away"
        direction_label = "利好主队" if f_val > 0 else "利好客队"
        factor_rows.append(
            f"<tr><td>{html.escape(f_label)}</td><td><code>{html.escape(f_name)}</code></td>"
            f"<td class='num'>{f_val:+.3f}</td><td class='{direction_class}'>{direction_label}</td></tr>"
        )
    factor_table_html = "\n".join(factor_rows)

    status_str = "完整分析" if report_json.get("status") == "ok" else "需要复核"

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
    max-width: 900px;
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
    margin: 16px 0;
    font-size: 14px;
  }}
  th, td {{
    padding: 10px 12px;
    border: 1px solid var(--border-color);
    text-align: left;
  }}
  th {{ background-color: #f1f5f9; font-weight: 600; }}
  td.num {{ text-align: right; font-variant-numeric: tabular-nums; }}
  .fav-home {{ color: var(--brand-primary); font-weight: 600; }}
  .fav-away {{ color: #dc2626; font-weight: 600; }}
  .card {{
    background: #f8fafc;
    border-left: 4px solid var(--brand-primary);
    padding: 12px 16px;
    margin: 16px 0;
    border-radius: 0 6px 6px 0;
  }}
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

  <h2>三、 时序截断与数据回执 [E6]</h2>
  <div class="card">
    <strong>时序契约：</strong> 预测截止日为 <code>{html.escape(str(e6['as_of']))}</code>，实际使用历史截止至 <code>{html.escape(str(e6['used_data_cutoff']))}</code>。<br>
    数据账龄：{e6['data_age_days']} 天 ｜ 提前量：{e6['forecast_horizon_days']} 天 ｜ 模式：{html.escape(str(e6['mode']))}。
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

    answer_parts: list[str] = []
    referenced_evidence: list[str] = []

    # Case 1: Ask about why favored or core reason
    if any(k in q for k in ("为什么", "原因", "凭什么", "逻辑", "看好")):
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

    # Case 2: Ask about goals / over-under
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

    # Case 3: Ask about market / odds / handicap
    elif any(k in q for k in ("盘口", "赔率", "让球", "亚盘", "市场")):
        referenced_evidence.extend(["E5"])
        if e5["available"]:
            m = e5["data"]
            m_prob = m.get("market_open_probabilities", {})
            diff_h = m.get("track_a_minus_market_percentage_points", {}).get("home")
            answer_parts.append(
                f"**盘口与市场参照**：市场初盘隐含主胜概率 {_pct(m_prob.get('home'))}，"
                f"MatchQuant Track A 基本面概率偏离 {diff_h:+.1f} 个百分点。"
            )
        else:
            answer_parts.append(
                "**盘口信息**：本场未录入外部初盘赔率或亚盘数据，所有推断均基于 Track A 纯比赛事实数据。"
            )

    # Case 4: Ask for concise / short summary
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
            f"模型综合了 Elo 基础实力与近期攻防统计驱动。您可以进一步询问进球期望、核心因子或盘口对比。"
        )

    return {
        "question": q,
        "answer": "\n\n".join(answer_parts),
        "referenced_evidence_ids": referenced_evidence,
        "run_id": report_json.get("run_id"),
        "recomputed": False,
    }
