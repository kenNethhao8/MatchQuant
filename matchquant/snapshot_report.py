"""Chinese markdown reports for B03 snapshot sessions and read-only run audits."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from .forecast_report import _markdown_text

AUDIT_SCOPE_TEXT = (
    "本审计只校验单模型 M11 快照、score-free fixtures 与预测计算之间的一致性；"
    "不重新证明历史训练正确性，不读取 history CSV、历史 processed manifest 或 current pointer，不联网。"
)


def _status_line(status: Any) -> str:
    return f"- 状态：`{_markdown_text(str(status))}`"


def write_snapshot_audit_report(path: Path, audit: Mapping[str, Any]) -> Path:
    lines = [
        "# MatchQuant 快照预测运行只读审计",
        "",
        "本报告由显式 M11 快照预测 run 的只读审计离线生成。",
        "",
        _status_line(audit.get("status", "blocked")),
        f"- 审计时间：`{_markdown_text(str(audit.get('audited_at') or audit.get('generated_at') or 'unavailable'))}`",
    ]
    reasons = audit.get("reasons") or []
    if reasons:
        lines.extend(["", "## 理由", ""])
        lines.extend(f"- `{_markdown_text(str(reason))}`" for reason in reasons)
    run = audit.get("run")
    if isinstance(run, Mapping):
        lines.extend(["", "## 被审计的预测运行", ""])
        lines.append(f"- run 目录：`{_markdown_text(str(run.get('run_dir')))}`")
        lines.append(f"- producer schema：`{_markdown_text(str(run.get('producer_schema')))}`")
        lines.append(f"- 模型：`{_markdown_text(str(run.get('model_name')))}`")
        predictions = run.get("predictions")
        if isinstance(predictions, Mapping):
            lines.append(
                f"- predictions：`{_markdown_text(str(predictions.get('path')))}`；"
                f"SHA-256 `{_markdown_text(str(predictions.get('sha256')))}`；行数 {predictions.get('rows')}"
            )
    snapshot = audit.get("snapshot")
    if isinstance(snapshot, Mapping):
        lines.extend(["", "## 快照与时间边界", ""])
        lines.append(f"- snapshot_id：`{_markdown_text(str(snapshot.get('snapshot_id')))}`")
        as_of = snapshot.get("as_of")
        if isinstance(as_of, Mapping):
            lines.append(
                f"- as_of：`{_markdown_text(str(as_of.get('value')))}`"
                f"（{'日期批次' if as_of.get('date_only') else '精确时间批次'}）"
            )
        lines.append(f"- history data cutoff：`{_markdown_text(str(snapshot.get('history_data_cutoff')))}`")
        lines.append(f"- used data cutoff：`{_markdown_text(str(snapshot.get('used_data_cutoff')))}`")
        lines.append(f"- 训练行数 rows_used：{snapshot.get('rows_used_before_as_of')}")
    repredict = audit.get("repredict")
    if isinstance(repredict, Mapping):
        lines.extend(["", "## 重新预测对照", ""])
        lines.append(
            f"- 最大绝对概率差：{repredict.get('max_abs_probability_difference')}"
            "（容差 atol=1e-12, rtol=0）"
        )
        lines.append(f"- 行身份一致：{repredict.get('identity_match')}")
        lines.append("- 重新预测在隔离临时目录完成；原 run 未覆盖，重算文件不是原 run 的新版本。")
    issue = audit.get("issue_evidence")
    if isinstance(issue, Mapping):
        lines.extend(["", "## 发行证据", ""])
        lines.append(f"- 是否提供：{issue.get('provided')}")
        lines.append(f"- 记录状态：`{_markdown_text(str(issue.get('record_status')))}`")
        lines.append(f"- fresh_at_issue：{issue.get('fresh_at_issue')}")
        lines.append(f"- active_link_status：`{_markdown_text(str(issue.get('active_link_status')))}`")
    checks = audit.get("engine_checks")
    if isinstance(checks, list) and checks:
        lines.extend(["", "## 工程检查", "", "| 检查 | 状态 | 说明 |", "|---|---|---|"])
        for check in checks:
            if isinstance(check, Mapping):
                lines.append(
                    f"| {_markdown_text(str(check.get('check')))} | {_markdown_text(str(check.get('status')))} | "
                    f"{_markdown_text(str(check.get('detail') or ''))} |"
                )
    lines.extend(["", "## 限制", ""])
    for limitation in audit.get("limitations") or []:
        lines.append(f"- {_markdown_text(str(limitation))}")
    lines.extend(["", AUDIT_SCOPE_TEXT, ""])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def write_snapshot_session_report(path: Path, manifest: Mapping[str, Any]) -> Path:
    mode = str(manifest.get("mode"))
    status = str(manifest.get("status"))
    lines = [
        "# MatchQuant 快照预测会话报告",
        "",
        "本报告由显式单模型 M11 快照与显式 score-free fixture CSV 的离线会话生成。",
        "",
        f"- 会话状态：`{_markdown_text(status)}`",
        f"- 模式：`{_markdown_text(mode)}`"
        + ("（回溯研究 retrospective）" if mode == "replay" else "（本地待赛研究，未证明覆盖）" if mode == "research" else ""),
        f"- 会话目录：`{_markdown_text(str(manifest.get('session_id')))}`",
        f"- 生成时间：`{_markdown_text(str(manifest.get('generated_at')))}`",
    ]
    reasons = manifest.get("reasons") or []
    if reasons:
        lines.extend(["", "## 会话理由", ""])
        lines.extend(f"- `{_markdown_text(str(reason))}`" for reason in reasons)
    snapshot = manifest.get("snapshot")
    if isinstance(snapshot, Mapping):
        lines.extend(["", "## 快照与时间边界", ""])
        lines.append(f"- snapshot_id：`{_markdown_text(str(snapshot.get('snapshot_id')))}`")
        as_of = snapshot.get("as_of")
        if isinstance(as_of, Mapping):
            lines.append(
                f"- as_of：`{_markdown_text(str(as_of.get('value')))}`"
                f"（{'日期批次' if as_of.get('date_only') else '精确时间批次'}）"
            )
        lines.append(f"- history data cutoff：`{_markdown_text(str(snapshot.get('history_data_cutoff')))}`")
        lines.append(f"- used data cutoff：`{_markdown_text(str(snapshot.get('used_data_cutoff')))}`")
        lines.append(f"- 训练行数 rows_used：{snapshot.get('rows_used_before_as_of')}")
        lines.append(f"- 模型：`{_markdown_text(str(snapshot.get('model_names')))}`")
    run = manifest.get("prediction_run")
    if isinstance(run, Mapping):
        lines.extend(["", "## 预测运行", ""])
        lines.append(f"- run 目录：`{_markdown_text(str(run.get('run_dir')))}`")
        predictions = run.get("predictions")
        if isinstance(predictions, Mapping):
            lines.append(
                f"- predictions：`{_markdown_text(str(predictions.get('path')))}`；行数 {predictions.get('rows')}"
            )
    audit = manifest.get("audit")
    if isinstance(audit, Mapping):
        lines.extend(["", "## 子审计（原样保留，不升格）", ""])
        lines.append(f"- 审计状态：`{_markdown_text(str(audit.get('status')))}`")
        lines.append(f"- fresh_at_issue：{audit.get('fresh_at_issue')}")
        lines.append(f"- active_link_status：`{_markdown_text(str(audit.get('active_link_status')))}`")
        audit_reasons = audit.get("reasons") or []
        if audit_reasons:
            lines.extend(["", "子审计理由：", ""])
            lines.extend(f"- `{_markdown_text(str(reason))}`" for reason in audit_reasons)
    coverage = manifest.get("coverage")
    if isinstance(coverage, Mapping):
        lines.extend(["", "## 覆盖状态", ""])
        lines.append(f"- 覆盖：`{_markdown_text(str(coverage.get('status')))}`")
        for reason in coverage.get("reasons") or []:
            lines.append(f"- `{_markdown_text(str(reason))}`")
    lines.extend(["", "## 限制", ""])
    for limitation in manifest.get("limitations") or []:
        lines.append(f"- {_markdown_text(str(limitation))}")
    lines.extend(
        [
            "",
            "会话不自动结算、不运行 M12、不更新 pointer；结算需另行用实际 M11 run 目录与独立赛果文件调用既有 settle。",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


__all__ = ["write_snapshot_audit_report", "write_snapshot_session_report"]
