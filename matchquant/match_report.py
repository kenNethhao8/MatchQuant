"""Pure Markdown and standalone HTML renderers for single-match JSON results."""

from __future__ import annotations

import html
import math
from pathlib import Path
from typing import Any, Mapping, Sequence


FACTOR_LABELS = {
    "elo_diff": "主客 Elo 实力差",
    "elo_abs_gap": "双方 Elo 差距绝对值",
    "att_def_mismatch_diff": "攻防对位差",
    "venue_specialist_diff": "主客场表现差",
    "expected_tempo": "预期比赛节奏",
    "sot_dominance_diff": "近期射正优势差",
    "sot_momentum_diff": "射正动量差",
    "shot_volume_diff": "射门数量优势差",
    "shot_precision_diff": "射正率优势差",
    "corner_pressure_diff": "角球压迫差",
    "style_counter_edge": "角球与防守脆弱度对位",
    "finishing_luck_diff": "近期射正转化偏离差",
    "save_luck_diff": "失球与射正偏离差",
    "points_vs_dominance_gap": "积分与场面优势背离差",
    "first_half_edge_diff": "上半场净胜球表现差",
    "second_half_surge_diff": "下半场净胜球表现差",
    "discipline_risk_diff": "纪律风险差",
    "referee_home_bias": "裁判历史主队偏差",
    "promoted_coldstart_diff": "升班赛季冷启动差",
    "rest_advantage_diff": "休息时间优势差",
    "fixture_congestion_diff": "近 14 天赛程密度差",
    "table_pressure_diff": "积分榜压力差",
    "mkt_open_prob_edge": "初盘主客胜概率差",
    "mkt_steam_move_diff": "初终盘漂移差",
    "mkt_ah_supremacy": "亚盘主队让球优势",
}


def _num(value: Any, digits: int = 2) -> str:
    if value is None:
        return "未提供"
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return "未提供"
    return f"{parsed:.{digits}f}" if math.isfinite(parsed) else "未提供"


def _pct(value: Any, digits: int = 1) -> str:
    if value is None:
        return "未提供"
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return "未提供"
    return f"{parsed * 100:.{digits}f}%" if math.isfinite(parsed) else "未提供"


def _pp(value: Any, digits: int = 1, signed: bool = False) -> str:
    if value is None:
        return "未计算"
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return "未计算"
    if not math.isfinite(parsed):
        return "未计算"
    return f"{parsed:+.{digits}f} 个百分点" if signed else f"{parsed:.{digits}f} 个百分点"


def _md(value: Any) -> str:
    text = str(value)
    for source, target in (
        ("\\", "\\\\"),
        ("`", "\\`"),
        ("*", "\\*"),
        ("_", "\\_"),
        ("[", "\\["),
        ("]", "\\]"),
        ("<", "&lt;"),
        (">", "&gt;"),
        ("|", "\\|"),
    ):
        text = text.replace(source, target)
    return text


def _wdl(evidence: Mapping[str, Any]) -> str:
    counts = evidence.get("record_count_by_outcome", {})
    return f"{counts.get('W', 0)}胜 / {counts.get('D', 0)}平 / {counts.get('L', 0)}负"


def _recent_mean(evidence: Mapping[str, Any], key: str, label: str) -> str:
    item = evidence.get(key, {})
    mean = _num(item.get("mean"), 1)
    return f"{mean}（{item.get('valid_sample_count', 0)}/{evidence.get('sample_count', 0)} 场）"


def _recent_summary(evidence: Mapping[str, Any]) -> dict[str, str]:
    sample_count = int(evidence.get("sample_count", 0))
    gf = evidence.get("goals_for", {})
    ga = evidence.get("goals_against", {})
    return {
        "team": str(evidence.get("team", "")),
        "sample": f"{sample_count} 场",
        "wdl": _wdl(evidence),
        "goals_for": _num(gf.get("mean"), 1),
        "goals_against": _num(ga.get("mean"), 1),
        "sot": _recent_mean(evidence, "shots_on_target_for", "射正"),
        "corners": _recent_mean(evidence, "corners_for", "角球"),
    }


def render_report_markdown(report: Mapping[str, Any]) -> str:
    """Render the public Chinese report from the authoritative structured result."""

    fixture = report["fixture"]
    temporal = report["temporal"]
    prediction = report["predictions"]
    probabilities = prediction["track_a"]
    evidence = report["recent_evidence"]
    model = report["model"]
    selection = model["selection"]
    market = report["market_comparison"]
    status_label = "需要复核" if report["status"] == "needs_review" else "完整"
    mode_label = "历史回放" if temporal["mode"] == "replay" else "研究假设"
    home = _md(fixture["home_team"])
    away = _md(fixture["away_team"])
    resolution_note = ""
    if fixture.get("requested_home_team") != fixture["home_team"] or fixture.get("requested_away_team") != fixture["away_team"]:
        resolution_note = (
            f"名称解析：{_md(fixture.get('requested_home_team', fixture['home_team']))} → {home}；"
            f"{_md(fixture.get('requested_away_team', fixture['away_team']))} → {away}。"
        )
    lines = [
        f"# {home} vs {away}｜单场比赛研究报告",
        "",
        f"**状态：{status_label}**　**模式：{mode_label}**　比赛日期：{_md(fixture['date'])}　赛季：{_md(fixture['season'])}",
        "",
        "## 比赛与数据时间",
        "",
        f"对阵由用户指定，未在线核对官方赛程或当前联赛身份。{resolution_note}预测数据截止 **{_md(temporal['as_of'])} 当日开始**，因此该日及之后的历史行均未使用。最新使用比赛为 {_md(temporal['used_data_cutoff'])}，数据年龄 {_md(temporal['data_age_days_at_as_of'])} 天；预测时距比赛 {_md(temporal['forecast_horizon_days'])} 天。截止后未观测到的比赛不会被模拟。",
        "",
    ]
    if report.get("warnings"):
        lines.extend(["**复核提示：** " + "；".join(_md(item["message"]) for item in report["warnings"]), ""])

    lines.extend(
        [
            "## 模型判断",
            "",
            f"模型估计的最高概率结果为 **{_md(prediction['most_likely_outcome_label'])}**。这只是三种结果中的相对排序，不代表确定赛果；模型没有独立的现实校准声明。",
            "",
            "| 模型估计概率 | 主胜 | 平局 | 客胜 |",
            "| --- | ---: | ---: | ---: |",
            f"| Track A 基本面 | {_pct(probabilities['home'])} | {_pct(probabilities['draw'])} | {_pct(probabilities['away'])} |",
            "",
            "## 进球预期",
            "",
            f"模型预期进球：{home} **{_num(prediction['expected_goals']['home'])}**，{away} **{_num(prediction['expected_goals']['away'])}**；总进球至少 3 个的模型概率为 **{_pct(prediction['probability_total_goals_over_2_5'])}**。预期进球表示模型均值，不是承诺的准确比分。",
            "",
            "## 判断依据",
            "",
        ]
    )

    attribution = report["features"]["attribution"]
    if attribution["available"] and attribution["contributions"]:
        lines.extend(
            [
                "下表仅为“有序模型分量的局部近似贡献”。它们不相加等于最终集成概率差，也不是因果效应。",
                "",
                "| 因子 | 原始值 | 方向 | 主胜局部近似贡献 |",
                "| --- | ---: | --- | ---: |",
            ]
        )
        for item in attribution["contributions"]:
            lines.append(
                f"| {_md(FACTOR_LABELS.get(item['factor'], item['factor']))} | {_num(item['raw_value'], 3)} | {_md(item['direction'])} | {_pp(item['local_approx_home_win_contribution_pct_points'], signed=True)} |"
            )
        lines.append("")
    elif attribution["available"]:
        lines.extend(["当前有序模型分量未提供足以展示的非 Elo 贡献；下方近期统计只描述已记录比赛。", ""])
    else:
        lines.extend([_md(attribution["limitation"]), ""])

    home_evidence = _recent_summary(evidence["home"])
    away_evidence = _recent_summary(evidence["away"])
    lines.extend(
        [
            "### 最近五场可用历史",
            "",
            "| 球队 | 样本 | 胜平负 | 场均进球 | 场均失球 | 场均射正（有效场数） | 场均角球（有效场数） |",
            "| --- | ---: | --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for item in (home_evidence, away_evidence):
        lines.append(
            f"| {_md(item['team'])} | {item['sample']} | {item['wdl']} | {item['goals_for']} | {item['goals_against']} | {item['sot']} | {item['corners']} |"
        )
    lines.append("")
    for key in ("home", "away"):
        item = evidence[key]
        if item["records"]:
            match_list = "、".join(
                f"{_md(record['date'])} {record['result']} {_md(record['opponent'])} {record['goals_for']}-{record['goals_against']}"
                for record in item["records"]
            )
            lines.append(f"- {home if key == 'home' else away}：{match_list}")
    lines.append("")

    lines.extend(
        [
            "## 模型与赔率对照",
            "",
            "| 模型分量 | 主胜 | 平局 | 客胜 |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    for key, label in (("elo", "Elo 基线"), ("ordered_logit", "有序 Logit"), ("dixon_coles", "Dixon–Coles"), ("track_a", "Track A 集成")):
        values = prediction[key]
        lines.append(f"| {label} | {_pct(values['home'])} | {_pct(values['draw'])} | {_pct(values['away'])} |")
    differences = prediction["track_a_minus_elo_percentage_points"]
    lines.extend(
        [
            "",
            f"Track A 相较 Elo 的模型概率差：主胜 {_pp(differences['home'], signed=True)}、平局 {_pp(differences['draw'], signed=True)}、客胜 {_pp(differences['away'], signed=True)}。",
        ]
    )
    if market["provided"]:
        market_input = market["input"]
        market_probs = market.get("market_open_probabilities") or market_input["probabilities"]
        edge = market["track_a_minus_market_percentage_points"]
        lines.extend(
            [
                "",
                f"市场数据由用户提供，来源描述：{_md(market_input['source'])}；声明观测日 {_md(market_input['observed_at'])}（未独立核验）。Power 幂次法去水概率为主胜 {_pct(market_probs['home'])}、平局 {_pct(market_probs['draw'])}、客胜 {_pct(market_probs['away'])}。Track A 概率减去市场去水概率分别为 {_pp(edge['home'], signed=True)}、{_pp(edge['draw'], signed=True)}、{_pp(edge['away'], signed=True)}；这是概率差，不是已验证收益或未来 CLV。",
            ]
        )
        if market.get("value_edges"):
            ve = market["value_edges"]
            lines.extend(
                [
                    "",
                    f"价值偏差（Edge = P * Odds - 1）：主胜 {_num(ve.get('home'), 4)}、平局 {_num(ve.get('draw'), 4)}、客胜 {_num(ve.get('away'), 4)}。{_md(ve.get('disclaimer', ''))}",
                ]
            )
        if market.get("divergence_alerts"):
            da = market["divergence_alerts"]
            thresh_str = _num(da.get("threshold_percentage_points"), 1)
            if da.get("has_divergence"):
                alert_notes = "；".join(
                    f"{item['outcome_label']}（模型 {_pct(item['model_prob_pct']/100)} vs 市场 {_pct(item['market_prob_pct']/100)}，偏差 {_pp(item['diff_percentage_points'], signed=True)}）"
                    for item in da.get("alerts", [])
                )
                lines.extend(
                    [
                        "",
                        f"**显著背离预警（阈值 {thresh_str} 个百分点）：** 检测到模型胜率与市场隐含胜率存在显著分歧：{alert_notes}。{_md(da.get('alerts', [{}])[0].get('note', ''))}",
                    ]
                )
            else:
                lines.extend(
                    [
                        "",
                        f"背离检验：各项赛果概率偏差均在阈值（{thresh_str} 个百分点）以内，未见显著背离。",
                    ]
                )
        if market.get("handicap_assessment"):
            ha = market["handicap_assessment"]
            lines.extend(
                [
                    "",
                    f"亚盘让球盘口：{ha['interpretation']}（盘口线值 {ha['mkt_ah_line']}；约定：{ha['convention']}）。",
                ]
            )
        if market.get("timeline_analysis"):
            ta = market["timeline_analysis"]
            if ta.get("status") == "multi_snapshot_trend":
                ds = ta.get("drift_summary", {})
                lines.extend(
                    [
                        "",
                        f"**盘口时序变动趋势（{ta.get('snapshot_count')} 个快照）：** {_md(ds.get('steam_description', ta.get('message', '')))}",
                    ]
                )
            elif ta.get("status") == "single_snapshot_baseline":
                lines.extend(
                    [
                        "",
                        f"盘口时序趋势：{_md(ta.get('message', '暂无历史变动趋势，仅提供单时点截面对照。'))}",
                    ]
                )
        if market["m_open_available"]:
            m_open = market["m_open_probabilities"]
            lines.append(
                f"M-Open（初盘 1X2 + 亚盘）模型估计：主胜 {_pct(m_open['home'])}、平局 {_pct(m_open['draw'])}、客胜 {_pct(m_open['away'])}。"
            )
        else:
            lines.append(f"M-Open 未提供：{_md(market['m_open_unavailable_reason'])}")
    else:
        lines.append("未提供赛前赔率，未计算市场对照。")
    lines.append("")

    lines.extend(["## 哪些信息还没算进去", ""])
    lines.extend(f"- {_md(item)}" for item in report["limitations"])
    lines.extend(
        [
            "",
            "## 数据与方法",
            "",
            f"历史输入：{_md(Path(report['input_sources']['history']['path']).name)}（SHA-256 `{report['input_sources']['history']['sha256'][:12]}`）；实际使用截止 {_md(temporal['used_data_cutoff'])}。训练季：`{_md(', '.join(selection['partitions']['training_seasons']))}`；验证季：`{_md(selection['partitions']['validation_season'])}`。验证指标用于调参和模型选择，不是最终重拟合模型的盲测结果。模型标识：`{_md(model['model_identifier'])}`。",
            "",
        ]
    )
    return "\n".join(lines)


def _html(value: Any) -> str:
    return html.escape(str(value), quote=True)


def _html_table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    head = "".join(f"<th>{_html(item)}</th>" for item in headers)
    body = "".join(
        "<tr>" + "".join(f"<td>{_html(cell)}</td>" for cell in row) + "</tr>" for row in rows
    )
    return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"


def render_report_html(report: Mapping[str, Any]) -> str:
    """Render a self-contained printable HTML report with escaped dynamic text."""

    fixture = report["fixture"]
    temporal = report["temporal"]
    prediction = report["predictions"]
    evidence = report["recent_evidence"]
    attribution = report["features"]["attribution"]
    market = report["market_comparison"]
    status_label = "需要复核" if report["status"] == "needs_review" else "完整"
    mode_label = "历史回放" if temporal["mode"] == "replay" else "研究假设"
    title = f"{fixture['home_team']} vs {fixture['away_team']}｜单场比赛研究报告"
    recent_rows = []
    for key in ("home", "away"):
        item = _recent_summary(evidence[key])
        recent_rows.append(
            [item["team"], item["sample"], item["wdl"], item["goals_for"], item["goals_against"], item["sot"], item["corners"]]
        )
    component_rows = []
    for key, label in (("elo", "Elo 基线"), ("ordered_logit", "有序 Logit"), ("dixon_coles", "Dixon–Coles"), ("track_a", "Track A 集成")):
        p = prediction[key]
        component_rows.append([label, _pct(p["home"]), _pct(p["draw"]), _pct(p["away"])])
    warning_html = "".join(f"<li>{_html(item['message'])}</li>" for item in report.get("warnings", []))
    contribution_rows = [
        [
            FACTOR_LABELS.get(item["factor"], item["factor"]),
            _num(item["raw_value"], 3),
            item["direction"],
            _pp(item["local_approx_home_win_contribution_pct_points"], signed=True),
        ]
        for item in attribution["contributions"]
    ]
    recent_lists = []
    for key in ("home", "away"):
        item = evidence[key]
        rows = [
            [record["date"], record["venue"], record["opponent"], f"{record['goals_for']}-{record['goals_against']}", record["result"], _num(record["sot_for"], 1), _num(record["corners_for"], 1)]
            for record in item["records"]
        ]
        if rows:
            recent_lists.append(
                f"<h3>{_html(item['team'])} 最近比赛</h3>"
                + _html_table(["日期", "主/客", "对手", "比分", "结果", "射正", "角球"], rows)
            )
    limitations_html = "".join(f"<li>{_html(item)}</li>" for item in report["limitations"])
    if market["provided"]:
        market_input = market["input"]
        market_probs = market.get("market_open_probabilities") or market_input["probabilities"]
        edge = market["track_a_minus_market_percentage_points"]
        m_parts = [
            f"<p>市场数据由用户提供，来源描述：{_html(market_input['source'])}；声明观测日 {_html(market_input['observed_at'])}（未独立核验）。"
            f"Power 幂次法去水概率：主胜 {_pct(market_probs['home'])}、平局 {_pct(market_probs['draw'])}、客胜 {_pct(market_probs['away'])}。"
            f"Track A 减市场的概率差：主胜 {_pp(edge['home'], signed=True)}、平局 {_pp(edge['draw'], signed=True)}、客胜 {_pp(edge['away'], signed=True)}。</p>"
        ]
        if market.get("value_edges"):
            ve = market["value_edges"]
            m_parts.append(
                f"<p><strong>价值偏差（Edge = P * Odds - 1）：</strong> 主胜 {_num(ve.get('home'), 4)}、平局 {_num(ve.get('draw'), 4)}、客胜 {_num(ve.get('away'), 4)}。<br>"
                f"<small class='meta'>{_html(ve.get('disclaimer', ''))}</small></p>"
            )
        if market.get("divergence_alerts"):
            da = market["divergence_alerts"]
            thresh_str = _num(da.get("threshold_percentage_points"), 1)
            if da.get("has_divergence"):
                alert_notes = "；".join(
                    f"{item['outcome_label']}（模型 {_pct(item['model_prob_pct']/100)} vs 市场 {_pct(item['market_prob_pct']/100)}，偏差 {_pp(item['diff_percentage_points'], signed=True)}）"
                    for item in da.get("alerts", [])
                )
                m_parts.append(
                    f"<p><strong>显著背离预警（阈值 {thresh_str} 个百分点）：</strong> 检测到模型胜率与市场隐含胜率存在显著分歧：{alert_notes}。<br>"
                    f"<small class='meta'>{_html(da.get('alerts', [{}])[0].get('note', ''))}</small></p>"
                )
            else:
                m_parts.append(
                    f"<p><strong>背离检验：</strong> 各项赛果概率偏差均在阈值（{thresh_str} 个百分点）以内，未见显著背离。</p>"
                )
        if market.get("handicap_assessment"):
            ha = market["handicap_assessment"]
            m_parts.append(
                f"<p><strong>亚盘让球盘口：</strong> {_html(ha['interpretation'])}（盘口线值 {_num(ha['mkt_ah_line'], 2)}；约定：{_html(ha['convention'])}）。</p>"
            )
        if market.get("timeline_analysis"):
            ta = market["timeline_analysis"]
            if ta.get("status") == "multi_snapshot_trend":
                ds = ta.get("drift_summary", {})
                m_parts.append(
                    f"<p><strong>盘口时序变动趋势（{ta.get('snapshot_count')} 个快照）：</strong> {_html(ds.get('steam_description', ta.get('message', '')))}</p>"
                )
            elif ta.get("status") == "single_snapshot_baseline":
                m_parts.append(
                    f"<p><strong>盘口时序趋势：</strong> {_html(ta.get('message', '暂无历史变动趋势，仅提供单时点截面对照。'))}</p>"
                )
        if market["m_open_available"]:
            m = market["m_open_probabilities"]
            m_parts.append(f"<p>M-Open 模型估计：主胜 {_pct(m['home'])}、平局 {_pct(m['draw'])}、客胜 {_pct(m['away'])}。</p>")
        else:
            m_parts.append(f"<p>M-Open 未提供：{_html(market['m_open_unavailable_reason'])}</p>")
        market_html = "".join(m_parts)
    else:
        market_html = "<p>未提供赛前赔率，未计算市场对照。</p>"

    source = report["input_sources"]["history"]
    selection = report["model"]["selection"]
    resolution_note_html = ""
    if fixture.get("requested_home_team") != fixture["home_team"] or fixture.get("requested_away_team") != fixture["away_team"]:
        resolution_note_html = (
            f"名称解析：{_html(fixture.get('requested_home_team', fixture['home_team']))} → {_html(fixture['home_team'])}；"
            f"{_html(fixture.get('requested_away_team', fixture['away_team']))} → {_html(fixture['away_team'])}。"
        )
    differences = prediction["track_a_minus_elo_percentage_points"]
    warning_section = f"<aside><strong>复核提示</strong><ul>{warning_html}</ul></aside>" if warning_html else ""
    contribution_section = (
        "<p>以下为<strong>有序模型分量的局部近似贡献</strong>。它们不相加等于最终集成概率差，也不是因果效应。</p>"
        + _html_table(["因子", "原始值", "方向", "主胜局部近似贡献"], contribution_rows)
        if attribution["available"] and contribution_rows
        else f"<p>{_html(attribution['limitation'])}</p>"
    )
    html_document = f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{_html(title)}</title>
<style>
:root {{ color-scheme: light; --ink: #17212b; --muted: #5b6570; --line: #dce2e8; --accent: #166b55; --warn: #8a4b08; }}
* {{ box-sizing: border-box; }}
body {{ margin: 0; color: var(--ink); background: #f3f5f7; font: 16px/1.65 "Microsoft YaHei", "Noto Sans CJK SC", sans-serif; }}
main {{ max-width: 920px; margin: 28px auto; padding: 42px 50px; background: #fff; box-shadow: 0 8px 30px rgba(21, 34, 46, .08); }}
h1 {{ margin: 0 0 12px; font-size: 30px; line-height: 1.35; }} h2 {{ margin: 32px 0 10px; padding-bottom: 7px; border-bottom: 1px solid var(--line); font-size: 21px; }}
h3 {{ margin: 22px 0 8px; font-size: 16px; }} p {{ margin: 10px 0; }} .meta {{ color: var(--muted); }} .badge {{ display: inline-block; padding: 2px 10px; border-radius: 999px; color: var(--accent); background: #e8f3ef; font-size: 14px; }}
aside {{ margin-top: 16px; padding: 12px 16px; color: var(--warn); background: #fff5e8; border-left: 4px solid #e2a04a; }}
table {{ width: 100%; margin: 12px 0 20px; border-collapse: collapse; font-size: 14px; }} th, td {{ padding: 8px 10px; border-bottom: 1px solid var(--line); text-align: left; vertical-align: top; }} th {{ color: #3d4853; background: #f6f8fa; font-weight: 600; }} td:nth-child(n+2) {{ font-variant-numeric: tabular-nums; }}
ul {{ padding-left: 22px; }} li {{ margin: 5px 0; }} footer {{ margin-top: 30px; padding-top: 14px; border-top: 1px solid var(--line); color: var(--muted); font-size: 13px; overflow-wrap: anywhere; }}
@media (max-width: 640px) {{ main {{ margin: 0; padding: 26px 20px; box-shadow: none; }} h1 {{ font-size: 24px; }} table {{ display: block; overflow-x: auto; }} }}
@media print {{ body {{ background: #fff; }} main {{ max-width: none; margin: 0; padding: 12mm; box-shadow: none; }} h2, h3 {{ break-after: avoid; }} table {{ break-inside: avoid; }} }}
</style>
</head>
<body><main>
<h1>{_html(title)}</h1>
<p><span class="badge">状态：{_html(status_label)}</span> <span class="badge">模式：{_html(mode_label)}</span></p>
<p class="meta">日期：{_html(fixture['date'])}　赛季：{_html(fixture['season'])}　预测截止：{_html(temporal['as_of'])} 当日开始</p>
{warning_section}
<h2>比赛与数据时间</h2>
<p>对阵由用户指定，未在线核对官方赛程或当前联赛身份。{resolution_note_html}最新使用比赛为 {_html(temporal['used_data_cutoff'])}，数据年龄 {_html(temporal['data_age_days_at_as_of'])} 天；距比赛 {_html(temporal['forecast_horizon_days'])} 天。截止日及之后的历史行均未使用，截止后未观测到的比赛不会被模拟。</p>
<h2>模型判断</h2>
<p>模型估计的最高概率结果为 <strong>{_html(prediction['most_likely_outcome_label'])}</strong>。这是三种结果中的相对排序，不代表确定赛果；模型没有独立的现实校准声明。</p>
{_html_table(['模型估计概率', '主胜', '平局', '客胜'], [['Track A 基本面', _pct(prediction['track_a']['home']), _pct(prediction['track_a']['draw']), _pct(prediction['track_a']['away'])]])}
<h2>进球预期</h2>
<p>模型预期进球：{_html(fixture['home_team'])} <strong>{_num(prediction['expected_goals']['home'])}</strong>，{_html(fixture['away_team'])} <strong>{_num(prediction['expected_goals']['away'])}</strong>；总进球至少 3 个的模型概率为 <strong>{_pct(prediction['probability_total_goals_over_2_5'])}</strong>。预期进球是模型均值，不是准确比分承诺。</p>
<h2>判断依据</h2>
{contribution_section}
<h3>最近五场可用历史</h3>
{_html_table(['球队', '样本', '胜平负', '场均进球', '场均失球', '场均射正（有效场数）', '场均角球（有效场数）'], recent_rows)}
{''.join(recent_lists)}
<h2>模型与赔率对照</h2>
{_html_table(['模型分量', '主胜', '平局', '客胜'], component_rows)}
<p>Track A 相较 Elo 的模型概率差：主胜 {_pp(differences['home'], signed=True)}、平局 {_pp(differences['draw'], signed=True)}、客胜 {_pp(differences['away'], signed=True)}。</p>
{market_html}
<h2>哪些信息还没算进去</h2>
<ul>{limitations_html}</ul>
<h2>数据与方法</h2>
<p>训练季：{_html(', '.join(selection['partitions']['training_seasons']))}；验证季：{_html(selection['partitions']['validation_season'])}。验证指标用于调参与模型选择，不是最终重拟合模型的盲测结果。模型：<code>{_html(report['model']['model_identifier'])}</code>。</p>
<footer>历史输入：{_html(Path(source['path']).name)}（SHA-256 { _html(source['sha256'][:12]) }）｜实际使用截止 { _html(temporal['used_data_cutoff']) }｜状态 { _html(status_label) }。本报告仅供研究。</footer>
</main></body></html>
"""
    return html_document


__all__ = ["FACTOR_LABELS", "render_report_html", "render_report_markdown"]
