"""Pure Chinese Markdown rendering for one M10 manifest."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping


def _text(value: object) -> str:
    return str(value).replace("|", "/").replace("\r", " ").replace("\n", " ")


def _path(value: object) -> str:
    return _text(value).replace("\\", "/")


def _artifact_line(label: str, artifact: object) -> str:
    if not isinstance(artifact, Mapping) or not artifact.get("path"):
        return f"- {label}：未生成"
    return (
        f"- [{label}](<{_path(artifact['path'])}>)；存在：`{_text(artifact.get('exists'))}`；"
        f"SHA-256：`{_text(artifact.get('sha256'))}`"
    )


def _stage_note(stage: Mapping[str, Any]) -> str:
    error = stage.get("error")
    if isinstance(error, Mapping) and error.get("message"):
        return str(error["message"])
    result_status = stage.get("result_status")
    return str(result_status) if result_status else ""


def write_postmatch_session_report(manifest: Mapping[str, Any], path: Path) -> Path:
    """Render only the M10 manifest; child artifacts are never inspected."""

    inputs = manifest.get("inputs") if isinstance(manifest.get("inputs"), Mapping) else {}
    m9 = manifest.get("m9") if isinstance(manifest.get("m9"), Mapping) else {}
    stages = manifest.get("stages") if isinstance(manifest.get("stages"), Mapping) else {}
    eligibility = manifest.get("apply_eligibility") if isinstance(manifest.get("apply_eligibility"), Mapping) else {}
    lines = [
        "# MatchQuant M10 赛后会话报告",
        "",
        f"总体状态：`{_text(manifest.get('status'))}`",
        "",
        "M10 完全离线；默认只生成 M7/M2.1 审查证据，绝不自动激活 history 或结算。",
        "本报告是一次可追溯的本地交接记录，不声明历史或赛果当前，不声明概率已校准、盈利能力、投注价值或未来表现。",
        "",
        "## M9 锁定证据",
        "",
        f"- M9 状态：`{_text(m9.get('status'))}`；M9 会话目录：`{_path(m9.get('session_dir'))}`",
    ]
    for label, key in (
        ("M9 session manifest", "session_manifest"),
        ("history canonical", "history"),
        ("history processed manifest", "history_manifest"),
        ("M4 run config", "forecast_run_config"),
        ("M4 predictions", "forecast_predictions"),
    ):
        lines.append(_artifact_line(label, m9.get(key)))
    if m9.get("error"):
        lines.append(f"- M9 校验错误：`{_text(m9['error'])}`")

    lines.extend(["", "## 显式输入与模式", ""])
    lines.append(f"- 赛季：`{_text(inputs.get('season'))}`；in-progress 声明：`{_text(inputs.get('in_progress_season'))}`")
    lines.append(f"- data root：`{_path(inputs.get('data_root'))}`；output root：`{_path(inputs.get('output_root'))}`")
    lines.append(f"- 当前 pointer：`{_path(inputs.get('current_pointer', {}).get('path') if isinstance(inputs.get('current_pointer'), Mapping) else None)}`")
    lines.append(_artifact_line("results source", inputs.get("results_source")))
    lines.append(_artifact_line("explicit staging directory", inputs.get("staging_dir")))
    provenance = manifest.get("source_provenance")
    if isinstance(provenance, Mapping):
        lines.append(
            f"- provenance：label=`{_text(provenance.get('source_label'))}`；URL=`{_text(provenance.get('source_url'))}`；"
            f"upstream retrieved at=`{_text(provenance.get('upstream_retrieved_at'))}`"
        )

    lines.extend(["", "## 阶段与子产物链接", "", "| 阶段 | 状态 | 说明 |", "|---|---|---|"])
    for key, label in (("local_stage", "M7 local stage"), ("sync", "M2.1 sync"), ("settlement", "M6 settlement")):
        stage = stages.get(key) if isinstance(stages.get(key), Mapping) else {}
        lines.append(f"| {label} | `{_text(stage.get('status', 'not_run'))}` | {_text(_stage_note(stage))} |")
        artifacts = stage.get("artifacts") if isinstance(stage.get("artifacts"), Mapping) else {}
        for artifact_name, artifact in artifacts.items():
            lines.append(_artifact_line(f"{label} {artifact_name}", artifact))
        discovery = stage.get("artifact_discovery")
        if isinstance(discovery, Mapping):
            candidates = ", ".join(_path(item) for item in discovery.get("candidates", [])) or "无"
            lines.append(f"- 工件发现：`{_text(discovery.get('status'))}`；候选：`{_text(candidates)}`；绑定：`{_path(discovery.get('bound_path'))}`")

    lines.extend(["", "## M2.1 apply eligibility 与 pointer coherence", ""])
    lines.append(f"- addition-only eligible：`{_text(eligibility.get('eligible'))}`；说明：{_text(eligibility.get('reason'))}")
    pointer = manifest.get("pointer_evidence")
    if isinstance(pointer, Mapping):
        lines.append(f"- active pointer coherence：`{_text(pointer.get('status'))}`；canonical hash match：`{_text(pointer.get('canonical_hash_matches_m9'))}`；processed manifest hash match：`{_text(pointer.get('processed_manifest_hash_matches_m9'))}`")
    sync = stages.get("sync") if isinstance(stages.get("sync"), Mapping) else {}
    counts = sync.get("counts") if isinstance(sync.get("counts"), Mapping) else {}
    if counts:
        lines.append(f"- diff counts：added=`{_text(counts.get('added'))}` unchanged=`{_text(counts.get('unchanged'))}` revised=`{_text(counts.get('revised'))}` missing=`{_text(counts.get('missing_from_candidate'))}` skipped unfinished=`{_text(counts.get('skipped_unfinished'))}`")
    blockers = sync.get("apply_blockers") if isinstance(sync, Mapping) else None
    if blockers:
        lines.append(f"- apply blockers：`{_text(', '.join(map(str, blockers)))}`")

    mutation = manifest.get("input_mutation") if isinstance(manifest.get("input_mutation"), Mapping) else {}
    settlement_state = stages.get("settlement") if isinstance(stages.get("settlement"), Mapping) else {}
    if mutation.get("active_pointer_updated") and settlement_state.get("status") == "blocked":
        lines.extend(["", "## 前向失败说明", "", "M2.1 已成功激活新历史版本，但 M6 随后失败；M10 不回滚 pointer 或历史版本，请单独解决结算。"])
    lines.extend(["", "## 不变更声明", "", "M10 不联网，不调用下载、forecast、fixture prepare 或 scheduler；M9 manifest、M4 forecast、results source、baseline 与 M7 raw bytes 不被覆盖。", "M2.1 apply 仅通过显式已有 staging directory 进入；完成 M6 评估不等于实时预测、模型校准、盈利证明或投注建议。", ""])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


__all__ = ["write_postmatch_session_report"]
