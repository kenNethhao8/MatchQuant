"""Chinese Markdown report for an M6 local settlement run."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import pandas as pd


def _text(value: object) -> str:
    return str(value).replace("|", "/").replace("\r", " ").replace("\n", " ")


def _metric_text(value: object) -> str:
    if value is None:
        return "不可用（已结算数为 0）"
    return f"{float(value):.8f}"


_M11_PRODUCER_TITLE = "M11 单模型引擎快照预测"


def _producer_title(forecast: Mapping[str, Any]) -> str:
    schema = str((forecast.get("predictions") or {}).get("schema_version", ""))
    if schema == "m11-engine-snapshot-predict-v1":
        return _M11_PRODUCER_TITLE
    return f"M4（{schema}）" if schema and schema != "m4-forecast-v1" else "M4"


def write_settlement_report(
    config: Mapping[str, Any], settlement: pd.DataFrame, path: str | Path
) -> Path:
    """Write a compact, auditable Chinese report for one M6 output directory."""

    counts = config["counts"]
    metrics = config["metrics"]
    forecast = config["forecast_run"]
    results = config["results"]
    producer = _producer_title(forecast)
    origin = forecast.get("origin_evidence") or {}
    lines = [
        "# MatchQuant M6 本地预测结算与评估报告",
        "",
        f"本报告只使用一个已生成的 {producer} 本地预测快照和用户明确指定的本地 M1 canonical 最终赛果。",
        "",
        "## 输入与时序证据",
        "",
        f"- {producer} run：`{_text(forecast['directory'])}`（producer schema：`{_text(forecast['predictions'].get('schema_version'))}`，command：`{_text(forecast.get('producer_command'))}`）",
        f"- run_config：`{_text(forecast['run_config']['path'])}`，SHA-256 `{forecast['run_config']['sha256']}`",
        f"- predictions：`{_text(forecast['predictions']['path'])}`，SHA-256 `{forecast['predictions']['sha256']}`",
        f"- predictions hash 证据：`{_text(forecast['predictions']['hash_evidence_status'])}`（旧 M4 未记录时为 `unrecorded_legacy`）",
        f"- 模型：`{_text(forecast['model'])}`（config SHA-256：`{_text(forecast.get('model_config_sha256') or '未记录')}`）",
    ]
    if origin.get("kind") == "m11_engine_snapshot":
        lines.extend(
            [
                f"- 快照 manifest：`{_text(origin.get('snapshot_manifest_path'))}`，SHA-256 `{_text(origin.get('snapshot_manifest_sha256'))}`",
                f"- 模型状态文件：`{_text(origin.get('state_path'))}`，SHA-256 `{_text(origin.get('state_sha256'))}`（快照不可变证据，结算只读）",
                f"- 快照 lineage 中的 history 证据：`{_text(origin.get('history_evidence_status'))}`；used data cutoff：`{_text(origin.get('used_data_cutoff') or '未记录')}`",
                f"- 预测源 fixtures：`{_text((origin.get('fixtures') or {}).get('path'))}`，SHA-256 `{_text((origin.get('fixtures') or {}).get('sha256'))}`（hash 已核验）",
            ]
        )
    lines.extend(
        [
            f"- 预测 as-of：`{_text(config['as_of']['value'])}`（{'日期边界' if config['as_of']['date_only'] else '精确时间边界'}）",
            f"- 结果 CSV：`{_text(results['path'])}`，SHA-256 `{results['sha256']}`",
            f"- 结果版本：`{_text(results.get('data_version') or '未由 M1 CSV 提供')}`（{_text(results.get('provenance_status'))}）",
            f"- 结果日期范围：`{_text(results['date_range']['start'])}` 至 `{_text(results['date_range']['end'])}`；数据 cutoff：`{_text(results['data_cutoff'])}`",
            "",
            "## 匹配规则",
            "",
            "- 身份键严格为 `season + home_team + away_team`；仅比较去除首尾空白后的原文本，不猜测别名，不做模糊匹配。",
            "- 唯一身份命中即使日期不同也可结算，并在逐场表中标记 `same_date` 或 `date_changed`。",
            "- 同一身份命中多条结果标为 `ambiguous`，不任意选择；零命中标为预期的 `unresolved`。",
            "- 结果早于预测 as-of 的行标为 `time_conflict`，不纳入指标；唯一改期结果仍可按身份结算并标记 `date_changed`。",
            "",
            "## 结算状态与指标",
            "",
            f"- run status：`{_text(config['status'])}`；总预测：{counts['total_predictions']}；已结算：{counts['settled']}；coverage：{_metric_text(metrics['coverage'])}",
            f"- unsettled：{counts['unresolved']}；ambiguous：{counts['ambiguous']}；time_conflict：{counts['time_conflict']}",
            f"- multiclass log loss：{_metric_text(metrics['log_loss'])}",
            f"- multiclass Brier：{_metric_text(metrics['brier_score'])}",
            f"- accuracy：{_metric_text(metrics['accuracy'])}",
            "",
            "指标只使用 `settled` 行；未结算、重复命中和时间矛盾不会被当作负样本或零概率。",
            "",
            "## 未结算与需复核明细",
            "",
            "| fixture_id | 状态 | match_key | date_status | 结果引用 | 原因 |",
            "|---|---|---|---|---|---|",
        ]
    )
    review_rows = settlement.loc[settlement["settlement_status"].ne("settled")]
    for row in review_rows.itertuples(index=False):
        lines.append(
            "| "
            + " | ".join(
                [
                    _text(row.fixture_id),
                    _text(row.settlement_status),
                    _text(row.match_key),
                    _text(row.date_status),
                    _text(row.result_match_ids),
                    _text(row.reason),
                ]
            )
            + " |"
        )
    if review_rows.empty:
        lines.append("| - | - | - | - | - | 无 |")
    lines.extend(
        [
            "",
            "## 限制与免责声明",
            "",
            "M6 不导入赛果，不联网，不调用 `epl-sync`，不更新 `matchquant_data` 或任何 active/current pointer，"
            "也不重跑模型、不选择模型、不校准概率、不修改预测或结果输入。结果输入由用户通过既有 M2/M2.1 "
            "或其他明确的本地 canonical 文件自行获得。",
            "",
            "本地结算与指标是研究审计结果，不是实时服务、投注建议、盈利承诺或未来表现保证。",
            "",
        ]
    )
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")
    return output_path


__all__ = ["write_settlement_report"]
