"""Human-readable markdown output for one local Alpha forecast."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

DISCLAIMER = (
    "本地 Alpha 预测是研究软件的离线快照，不是实时赛程服务；概率仅表示模型输出的有限、非负且归一化的算术结果，"
    "不宣称已经校准，不构成投注建议、盈利承诺或未来表现保证。"
)


def _markdown_text(value: object) -> str:
    return str(value).replace("|", "/").replace("\r", " ").replace("\n", " ")


def write_forecast_report(run: Any, config: Mapping[str, Any], path: Path) -> Path:
    """Write a compact report without adding a reporting dependency."""

    selection = config["selection"]
    lines = [
        "# MatchQuant 本地 Alpha 预测报告",
        "",
        "本报告是由本地 canonical 历史结果和本地待赛 fixture CSV 生成的 as-of 快照。",
        "",
        "## 快照与证据",
        "",
        f"- as-of：`{_markdown_text(run.as_of['value'])}`（{'日期批次' if run.as_of['date_only'] else '精确时间批次'}）",
        f"- 历史输入：`{_markdown_text(config['history']['path'])}`，SHA-256 `{config['history']['sha256']}`",
        f"- 待赛输入：`{_markdown_text(config['fixtures']['path'])}`，SHA-256 `{config['fixtures']['sha256']}`",
        f"- 实际纳入历史：{run.history_rows_used}/{run.history_rows_total} 场；纳入历史 cutoff：`{run.history_cutoff or '无，冷启动'}`",
        f"- 模型：`{run.model_name}`；season mode：`{run.season_mode}`",
        f"- 选择证据状态：`{selection['evidence_status']}`",
        f"- 选择 as-of 证据：`{selection.get('selection_as_of_status', 'not_applicable')}`",
    ]
    latest_end = selection.get("validation_latest_end")
    if isinstance(latest_end, Mapping):
        lines.append(
            f"- M3 验证期最新结束边界：`{_markdown_text(latest_end.get('value'))}`；"
            f"有效结束边界：`{_markdown_text(latest_end.get('effective_end_exclusive'))}`"
        )
    as_of_evidence = selection.get("as_of_evidence")
    if isinstance(as_of_evidence, Mapping) and as_of_evidence.get("status") == "validated":
        lines.append(
            "- M3 final-test metrics 明确未用于选模；因此 final-test 可以位于本次 as-of 之后，"
            "仅作审计留痕。"
        )
    if selection.get("history_hash_status") == "mismatch":
        lines.extend(
            [
                "",
                "> 注意：M3 选择证据的 history hash 与本次 history 不一致。本次运行是明确请求下的快照，"
                "该选择证据不能视为本次 history 的匹配训练证据。",
            ]
        )
    lines.extend(
        [
            "",
            "## 预测结果",
            "",
            "H/D/A 分别表示主胜/平/客胜；“模型最可能结果”只表示三类模型概率中的最大者。",
            "",
            "| fixture_id | date | season | home_team | away_team | H | D | A | 模型最可能结果 |",
            "|---|---|---|---|---|---:|---:|---:|---|",
        ]
    )
    for row in run.predictions.itertuples(index=False):
        lines.append(
            "| "
            + " | ".join(
                [
                    _markdown_text(row.fixture_id),
                    _markdown_text(row.date),
                    _markdown_text(row.season),
                    _markdown_text(row.home_team),
                    _markdown_text(row.away_team),
                    f"{row.home_win_prob:.6f}",
                    f"{row.draw_prob:.6f}",
                    f"{row.away_win_prob:.6f}",
                    _markdown_text(row.model_most_likely_result),
                ]
            )
            + " |"
        )
    lines.extend(["", "## 限制与免责声明", "", f"{DISCLAIMER}", ""])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


__all__ = ["write_forecast_report"]
