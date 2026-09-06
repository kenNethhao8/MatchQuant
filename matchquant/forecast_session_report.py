"""Chinese Markdown renderer for one M9 local forecast session."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping


def _text(value: object) -> str:
    return str(value).replace("|", "/").replace("\r", " ").replace("\n", " ")


def _path_text(value: object) -> str:
    return _text(value).replace("\\", "/")


def _artifact_line(label: str, artifact: object) -> str:
    if not isinstance(artifact, Mapping):
        return f"- {label}：未生成"
    path = artifact.get("path")
    digest = artifact.get("sha256")
    if not path:
        return f"- {label}：未生成"
    link = f"[{label}](<{_path_text(path)}>)"
    return f"- {link}；存在：`{_text(artifact.get('exists'))}`；SHA-256：`{_text(digest)}`"


def write_forecast_session_report(manifest: Mapping[str, Any], path: Path) -> Path:
    """Render only the supplied M9 manifest; never inspect child artifacts."""

    inputs = manifest.get("inputs", {})
    stages = manifest.get("stages", {})
    lines = [
        "# MatchQuant M9 本地预测会话报告",
        "",
        f"总体状态：`{_text(manifest.get('status'))}`",
        "",
        "这是一次完全离线、显式输入的赛前封装：`M5 → M4 → M8`。",
        "M9 不联网、不更新历史或 current pointer、不导入赛后结果，也不执行人工发布。",
        "`ready` 只表示 M8 现有证据检查通过，不代表实时性、校准、盈利、投注价值或人工发布完成。",
        "",
        "## 显式输入",
        "",
        f"- 赛季：`{_text(inputs.get('season'))}`；as-of：`{_text(inputs.get('as_of'))}`",
        f"- fixture data root：`{_text(inputs.get('fixture_data_root'))}`",
        f"- output root：`{_text(inputs.get('output_root'))}`",
    ]
    for label, key in (
        ("history", "history"),
        ("history manifest", "history_manifest"),
        ("fixture source", "fixture_source"),
        ("model/selection evidence", "model_selection"),
    ):
        value = inputs.get(key)
        if isinstance(value, Mapping):
            lines.append(_artifact_line(label, value))

    lines.extend(["", "## 阶段与链接", "", "| 阶段 | 状态 | 说明 |", "|---|---|---|"])
    for stage_key, label in (
        ("fixture_snapshot", "M5 fixture snapshot"),
        ("forecast", "M4 forecast"),
        ("release_verification", "M8 release verification"),
    ):
        stage = stages.get(stage_key, {})
        status = stage.get("status") if isinstance(stage, Mapping) else "not_run"
        note = stage.get("error", {}).get("message") if isinstance(stage, Mapping) and isinstance(stage.get("error"), Mapping) else ""
        lines.append(f"| {label} | `{_text(status)}` | {_text(note or stage.get('result_status', ''))} |")
        if isinstance(stage, Mapping):
            for artifact_name, artifact in (stage.get("artifacts") or {}).items():
                if isinstance(artifact, Mapping) and artifact.get("path"):
                    lines.append(_artifact_line(f"{label} {artifact_name}", artifact))
            reasons = stage.get("reasons") or []
            for reason in reasons:
                lines.append(f"  - {_text(reason)}")

    lines.extend(
        [
            "",
            "## 不变更声明",
            "",
            "- 所有子产物均为新建唯一目录；M9 只链接，不复制 predictions、raw source 或第三方数据。",
            "- history、history manifest、fixture source、模型/选择证据、M5 raw/quality 与 current pointer 未被更新。",
            "- M9 未调用 `epl-download`、`epl-sync`、`epl-local-stage` 或 `settle`。",
            "",
            "## 赛后手动交接",
            "",
            "如需更新历史，用户必须明确执行 M7/M2.1 流程并审查其结果；之后再由用户明确执行 M6 settle。M9 不自动导入赛果或触发结算。",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


__all__ = ["write_forecast_session_report"]
