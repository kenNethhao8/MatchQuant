"""Chinese Markdown reports for M13A result-factor artifacts."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

from .result_factors import RESULT_FACTOR_COVERAGE_STATUS


def _text(value: object) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def render_build_report(
    *,
    status: str,
    snapshot_dir: Path,
    as_of: Mapping[str, Any] | None,
    rows_total: int | None,
    rows_used: int | None,
    used_data_cutoff: str | None,
    team_count: int | None,
    state_sha256: str | None,
    reasons: Sequence[str],
) -> str:
    lines = [
        "# M13A 赛果因子快照构建报告",
        "",
        f"- 状态：`{_text(status)}`",
        f"- 快照目录：`{_text(snapshot_dir)}`",
        f"- as_of：`{_text(as_of.get('value') if as_of else '未解析')}`",
        f"- 历史总行数：`{_text(rows_total if rows_total is not None else '未知')}`",
        f"- as_of 前使用行数：`{_text(rows_used if rows_used is not None else '未知')}`",
        f"- 实际使用截止：`{_text(used_data_cutoff) if used_data_cutoff is not None else 'null'}`",
        f"- 球队数：`{_text(team_count) if team_count is not None else '未知'}`",
        f"- state SHA-256：`{_text(state_sha256) if state_sha256 else '尚未生成'}`",
        f"- 覆盖状态：`{RESULT_FACTOR_COVERAGE_STATUS}`",
        "- experimental：`true`；probability_adjustment_applied：`false`",
        "",
        "## 定义与限制",
        "",
        "窗口固定为最近至多 5 场和 10 场，跨赛季 carry。每个批次先冻结所有球队与对手的批前 Elo，再按稳定 match_id 顺序更新现有 EloModel。主客场进失球均值使用对应场地的联盟先验和固定 5 场先验权重收缩。休息日是比赛日期之间的日历日差，不代表精确休息小时。",
        "",
        "快照只表示截至该 as_of 的本地历史计算结果；其后可能已有未收录比赛，因此不验证目标 fixture 日期的完整覆盖。因子是实验信息，不调整任何 1X2 概率，也不表示校准或因果关系。",
        "",
        "## 原因",
        "",
    ]
    lines.extend(f"- `{_text(reason)}`" for reason in reasons) if reasons else lines.append("- 无")
    return "\n".join(lines) + "\n"


def render_query_report(payload: Mapping[str, Any], csv_path: str | None) -> str:
    snapshot = payload.get("snapshot") or {}
    fixtures = payload.get("fixtures") or {}
    as_of = snapshot.get("as_of")
    as_of_value = as_of.get("value") if isinstance(as_of, Mapping) else as_of
    lines = [
        "# M13A 赛果因子查询报告",
        "",
        f"- 状态：`{_text(payload.get('status', 'blocked'))}`",
        f"- as_of：`{_text(as_of_value)}`",
        f"- fixture 数：`{_text(payload.get('fixture_count', 0))}`",
        f"- fixture 输入 SHA-256：`{_text(fixtures.get('sha256'))}`",
        f"- 因子 CSV：`{_text(csv_path) if csv_path else '未生成'}`",
        f"- 覆盖状态：`{RESULT_FACTOR_COVERAGE_STATUS}`",
        "- experimental：`true`；probability_adjustment_applied：`false`",
        "",
        "## 逐场摘要",
        "",
        "| fixture_id | 主队 | 客队 | 状态 | 原因 |",
        "|---|---|---|---|---|",
    ]
    for result in payload.get("results", []):
        lines.append(
            "| "
            + " | ".join(
                _text(value)
                for value in (
                    result.get("fixture_id"),
                    result.get("home_team"),
                    result.get("away_team"),
                    result.get("status"),
                    ", ".join(result.get("reasons", [])) or "无",
                )
            )
            + " |"
        )
    lines.extend([
        "",
        "查询只读取 factor manifest、其同目录 JSON state 和 score-free fixtures。manifest 中的历史路径只作为来源记录；查询没有读取历史、current pointer、赛果或网络，也没有改写快照。",
        "",
        "赛果因子不进入现有预测或结算流程。本报告不表示因子已训练、已校准、具有因果作用或提升未来预测效果。",
        "",
        "## 原因",
        "",
    ])
    reasons = payload.get("reasons", [])
    lines.extend(f"- `{_text(reason)}`" for reason in reasons) if reasons else lines.append("- 无")
    return "\n".join(lines) + "\n"


__all__ = ["render_build_report", "render_query_report"]
