"""Compact, non-claiming report for one M11 snapshot prediction run."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from .forecast_report import _markdown_text

SNAPSHOT_DISCLAIMER = (
    "M11 是本地研究状态快照，不是实时赛程服务；概率仅表示模型输出的有限、非负且归一化的算术结果，"
    "不宣称已经校准，不构成投注建议、盈利承诺或未来表现保证。"
)


def write_snapshot_prediction_report(
    path: Path, config: Mapping[str, Any], predictions: pd.DataFrame
) -> Path:
    snapshot = config["snapshot"]
    fixtures = config["fixtures"]
    lines = [
        "# MatchQuant M11 engine snapshot prediction",
        "",
        "本报告由不可变 M11 模型状态快照和显式本地 score-free fixture CSV 离线生成。",
        "",
        f"- snapshot：`{_markdown_text(snapshot['snapshot_id'])}`；manifest SHA-256 `{snapshot['manifest_sha256']}`",
        f"- as-of：`{_markdown_text(config['as_of']['value'])}`（{'日期批次' if config['as_of']['date_only'] else '精确时间批次'}）",
        f"- fixture 输入：`{_markdown_text(fixtures['path'])}`；SHA-256 `{fixtures['sha256']}`；行数 {fixtures['rows']}",
        f"- 模型：{', '.join(config['model_names'])}；快照预测阶段未读取历史或更新状态。",
        "",
        "## 结果",
        "",
        "H/D/A 分别表示主胜/平/客胜；模型最可能结果只表示三类概率中的最大者。",
        "",
        "| fixture_id | model | H | D | A | 模型最可能结果 |",
        "|---|---|---:|---:|---:|---|",
    ]
    for row in predictions.itertuples(index=False):
        lines.append(
            f"| {_markdown_text(row.fixture_id)} | {_markdown_text(row.model)} | {row.home_win_prob:.6f} | "
            f"{row.draw_prob:.6f} | {row.away_win_prob:.6f} | {row.model_most_likely_result} |"
        )
    lines.extend(["", "## 限制", "", SNAPSHOT_DISCLAIMER, ""])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


__all__ = ["SNAPSHOT_DISCLAIMER", "write_snapshot_prediction_report"]
