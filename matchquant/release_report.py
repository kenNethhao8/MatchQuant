"""Chinese Markdown renderer for the M8 release audit package."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping


def _text(value: object) -> str:
    return str(value).replace("|", "/").replace("\r", " ").replace("\n", " ")


def _path_text(value: object) -> str:
    return _text(value).replace("\\", "/")


def write_release_report(manifest: Mapping[str, Any], path: Path) -> Path:
    """Write a compact, evidence-first report without copying any input artifact."""

    checks = manifest.get("checks", [])
    counts = {status: sum(item.get("status") == status for item in checks) for status in ("passed", "needs_review", "blocked")}
    outputs = manifest.get("outputs", {})
    lines = [
        "# MatchQuant M8 本地预测发布审计报告",
        "",
        f"总体状态：`{_text(manifest.get('status'))}`",
        "",
        "本报告只回答：这份 M4 预测是否可由用户指定的历史版本、M5 赛程快照和 as-of 时间边界复核。它不执行人工审批，也不代表实时性、校准、盈利或投注价值。",
        "",
        "## 状态含义",
        "",
        "- `ready`：所有硬完整性/时间核验通过，且没有需要人工注意的 legacy 或 selection mismatch 证据。",
        "- `needs_review`：没有硬矛盾，但存在旧 M4 hash 未记录、selection history mismatch 或版本证据不完整；不得直接发布。",
        "- `blocked`：关键 hash、path、rows、schema、quality、raw 或时间证据存在矛盾/缺失；不得发布。",
        "- M8 未执行人工审批；最终发布决定仍由用户作出。",
        "",
        "## 审计统计",
        "",
        f"核验项：{len(checks)}；通过：{counts['passed']}；需复核：{counts['needs_review']}；阻断：{counts['blocked']}",
        "",
        "## as-of 与输入证据",
        "",
        f"- as-of：`{_text(manifest.get('as_of'))}`（date-only/精确时间语义保留在 release_manifest.json）",
    ]
    inputs = manifest.get("inputs", {})
    for label, key in (("M4 run directory", "forecast_run"), ("history canonical", "history"), ("history manifest", "history_manifest"), ("M5 fixture manifest", "fixture_snapshot_manifest"), ("M5 fixture output", "fixture_output"), ("M5 raw source", "fixture_raw_snapshot"), ("M5 quality report", "fixture_quality_report")):
        value = inputs.get(key, {})
        if isinstance(value, Mapping):
            input_path = value.get("directory") or value.get("manifest_path") or value.get("path")
            output = value.get("output")
            if not input_path and isinstance(output, Mapping):
                input_path = output.get("path") or output.get("filename")
            digest = value.get("sha256") or value.get("manifest_sha256")
            lines.append(f"- {label}：`{_text(input_path)}`；SHA-256：`{_text(digest)}`")
    lines.extend(["", "## 核验结果", "", "| id | 状态 | 摘要 | 原因 |", "|---|---|---|---|"])
    for item in checks:
        lines.append(
            "| " + " | ".join(
                [_text(item.get("id")), _text(item.get("status")), _text(item.get("summary")), _text("；".join(item.get("reasons", [])) or "-")]
            ) + " |"
        )
    selection = manifest.get("selection_evidence")
    lines.extend(["", "## 模型选择证据", "", f"`{_text(selection)}`", "", "M8 只保留 M4 已有的 M3 或 explicit model config 状态，不重新选择、训练、校准或提高证据等级。"])
    prediction_path = outputs.get("forecast_predictions") if isinstance(outputs, Mapping) else None
    if prediction_path:
        lines.extend(["", "## 原始预测文件", "", f"[打开原始 M4 predictions.csv](<{_path_text(prediction_path)}>)", "", "M8 未复制 predictions.csv；上面的链接指向原 M4 文件。"])
    lines.extend(["", "## 限制与声明", ""])
    for limitation in manifest.get("limitations", []):
        lines.append(f"- {_text(limitation)}")
    lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


__all__ = ["write_release_report"]
