"""B04 builder for one portable single-model runtime bundle.

``build_runtime_bundle`` accepts an explicitly provided M11 snapshot and an M12
issue record that is already strictly bound to that snapshot and fresh at issue,
then assembles a read-mostly bundle: the original manifest/state/issue bytes are
only ever copied, the derived manifest is produced by the existing manifest
writer with the fixed path-leaf changes, and ``bundle_manifest.json`` is written
last so an unfinished build is never marked usable. Build job artifacts stay
outside the bundle directory. The first version builds actual directories; it
does not pack archives, does not touch the formal roots, and never claims real
release or current coverage.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

from .engine_snapshot_manifest import load_engine_snapshot, write_snapshot_manifest
from .forecast_report import _markdown_text as _markdown_escape
from .io import file_sha256, write_json
from .runtime_bundle_validation import (
    ALLOWED_ADDED_FIELDS,
    ALLOWED_MANIFEST_DIFF_FIELDS,
    BUNDLE_LINEAGE_VALIDATION_KIND,
    BUNDLE_MANIFEST_NAME,
    RUNTIME_BUNDLE_SCHEMA_VERSION,
    RUNTIME_BUNDLE_SIZE_LIMIT_BYTES,
    fixed_bundle_payload_paths,
    manifest_leaf_diff,
    verify_runtime_bundle,
)
from .schema import DataValidationError
from .snapshot_audit_support import (
    ISSUE_EVIDENCE_SCHEMA_VERSION,
    ensure_safe_output_root,
    load_issue_evidence,
    new_output_dir,
    utc_now,
)
from .team_index import TEAM_INDEX_SCHEMA_VERSION, build_team_index

BUNDLE_BUILD_DIR_PREFIX = "runtime_bundle_build_"
_BUNDLE_DIR_NAME = "bundle"
_ORIGINAL_OUTPUTS_KEYS = {"snapshot_dir", "snapshot_manifest", "model_states", "snapshot_report"}


def _bundle_total_bytes(bundle_dir: Path) -> int:
    """Sum of all actual file sizes under the bundle directory (uncompressed)."""

    return sum(entry.stat().st_size for entry in bundle_dir.rglob("*") if entry.is_file())

_BASE_LIMITATIONS = (
    "运行包只保留派生快照、state、原 manifest/issue 字节副本与 team index；"
    "不含 raw/canonical CSV、赛果、训练文件、可执行对象或完整候选预测。",
    "build ready 只表示工程包完整；active_link_status 仍为 not_checked_in_runtime，"
    "fresh_at_issue 只表示原发行记录经派生桥关联，不证明当前覆盖。",
    "包体没有签名来源认证；hash 证明本次输入一致性，不能发现所有字段同步伪造的包体。",
)


def _write_utf8_text(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


def _reject_source_root_output(output_root: Path) -> None:
    source_root = Path(__file__).resolve().parent.parent
    if output_root == source_root or source_root in output_root.parents:
        raise DataValidationError(
            f"output root must not be inside the source code directory: {output_root}"
        )


def _write_derived_snapshot_report(path: Path, manifest: dict[str, Any], state_sha256: str) -> None:
    as_of = manifest["as_of"]
    lines = [
        "# 派生快照说明（runtime bundle 内）",
        "",
        "**本文件是 runtime bundle 新生成的派生说明，标明 derived；不宣称与原研究报告字节相同。**",
        "",
        f"- snapshot_id：`{manifest['snapshot_id']}`（保留原 snapshot_id）",
        f"- schema：`{manifest['schema_version']}`；派生 manifest 由既有 writer 生成",
        f"- as_of：`{as_of['value']}`（{'日期批次' if as_of['date_only'] else '精确时间批次'}）",
        f"- history data cutoff：`{manifest['history'].get('data_cutoff')}`",
        f"- used data cutoff：`{manifest['history'].get('used_data_cutoff')}`",
        f"- 模型：`{', '.join(manifest['model_names'])}`（单模型）",
        f"- state SHA-256：`{state_sha256}`（与原快照 state 字节一致）",
        "",
        "## 与原快照的关系",
        "",
        "- 原快照目录只读，未改写；bundle 内 state 为原字节复制。",
        "- 派生 manifest 仅按 B04 合同修改固定路径叶子（state.path、outputs.*）并由 writer 重算内容 hash；"
        "其余字段与原 manifest 深度一致。",
        "- 原始 manifest 与 M12 发行记录以字节副本保存在 `evidence/`；其中记录的来源路径仅为 provenance，"
        "运行端从不打开，也不回退到原研究目录。",
        "- 每次运行只从包内相对路径解析文件；来源路径不是运行文件候选。",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_bundle_run_md(path: Path, snapshot_id: str, model: str) -> None:
    lines = [
        "# MatchQuant runtime bundle 使用说明",
        "",
        "本包是一个单模型 M11 运行包：迁移到不同绝对目录后，预测会话、运行审计与结算依然离线完成，"
        "不需要原研究端目录。",
        "",
        "## 包内容",
        "",
        "- `bundle_manifest.json`：包清单（所有载荷文件的相对路径、大小与 SHA-256）。",
        f"- `snapshot/{snapshot_id}/`：派生快照 manifest 与原字节 state（保留原 snapshot_id）。",
        "- `evidence/original_snapshot_manifest.json`：原 manifest 字节副本（仅为证据，来源路径不执行）。",
        "- `evidence/issue_audit.json`：原 M12 fresh 发行记录字节副本（经派生桥关联到本包派生快照）。",
        "- `team_index.json`：从 state 确定性导出的球队索引。",
        "- `RUN.md`：本说明。",
        "",
        "## 命令顺序",
        "",
        "```text",
        "matchquant runtime-bundle-verify --bundle-manifest <包根>/bundle_manifest.json",
        "matchquant snapshot-session --bundle-manifest <包根>/bundle_manifest.json --fixtures <无比分CSV> --mode replay|research --output-root <会话输出根>",
        "matchquant snapshot-run-verify --forecast-run <M11-run> --bundle-manifest <包根>/bundle_manifest.json --output-root <审计输出根>",
        "matchquant settle --forecast-run <M11-run> --results <独立赛果CSV> --output-root <结算输出根>",
        "```",
        "",
        "## 边界",
        "",
        "- 全程离线；不读取 history CSV、历史 processed manifest 或 current pointer。",
        "- 会话/审计输出不得位于整个 bundle 目录内；每次运行所有包内文件起终 hash 不变。",
        f"- 模型 `{model}`：新球队按模型既有先验处理并在会话报告中标注；已知球队不证明其仍参加当前赛季。",
        "- active_link_status 固定为 not_checked_in_runtime；research 会话保持 needs_review。",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def _inventory_bundle_files(bundle_dir: Path) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for candidate in sorted(bundle_dir.rglob("*")):
        if not candidate.is_file():
            continue
        relative = candidate.relative_to(bundle_dir).as_posix()
        if relative == "bundle_manifest.json":
            continue
        entries.append(
            {"path": relative, "bytes": candidate.stat().st_size, "sha256": file_sha256(candidate)}
        )
    return entries


def build_runtime_bundle(
    snapshot_manifest_path: str | Path,
    issue_audit_path: str | Path,
    *,
    output_root: str | Path,
) -> dict[str, Any]:
    """Build one portable single-model runtime bundle and return its structured record."""

    snapshot_file = Path(snapshot_manifest_path).expanduser().resolve()
    issue_file = Path(issue_audit_path).expanduser().resolve()
    output_root_path = ensure_safe_output_root(
        output_root, [(snapshot_file, "file"), (issue_file, "file")]
    )
    _reject_source_root_output(output_root_path)

    generated_at = utc_now()
    job_dir = new_output_dir(output_root_path, BUNDLE_BUILD_DIR_PREFIX)
    bundle_dir = job_dir / _BUNDLE_DIR_NAME
    bundle_dir.mkdir()
    job_json_path = job_dir / "build_job.json"
    report_path = job_dir / "build_report.md"

    stages: dict[str, str] = {
        "snapshot_load": "not_run",
        "issue_evidence": "not_run",
        "assemble": "not_run",
        "size_gate": "not_run",
        "self_verify": "not_run",
        "finalize": "not_run",
    }
    result: dict[str, Any] = {
        "schema_version": RUNTIME_BUNDLE_SCHEMA_VERSION,
        "command": "matchquant runtime-bundle-build",
        "generated_at": generated_at,
        "status": "blocked",
        "reasons": [],
        "build_dir": str(job_dir),
        "bundle_dir": str(bundle_dir),
        "bundle_manifest_path": str(bundle_dir / "bundle_manifest.json"),
        "bundle_manifest_sha256": None,
        "snapshot_id": None,
        "original": None,
        "derived": None,
        "issue": None,
        "manifest_field_diffs": None,
        "file_count": None,
        "actual_file_count": None,
        "payload_bytes": None,
        "total_bytes": None,
        "limit_bytes": RUNTIME_BUNDLE_SIZE_LIMIT_BYTES,
        "stages": stages,
        "build_job_path": str(job_json_path),
        "build_report_path": str(report_path),
        "manifest": None,
        "limitations": list(_BASE_LIMITATIONS),
    }

    def _render_build_report_lines(job_persistence_error: str | None) -> list[str]:
        """Render the Chinese build report from the CURRENT delivery state.

        Called only after the delivery status/stages/reasons have been decided,
        and once more (bounded) after a job-record write failure, so a written
        report never shows a pre-final ready state.
        """

        status_text = "blocked" if result["status"] == "blocked" else "ready"
        lines = [
            "# MatchQuant runtime bundle 构建报告",
            "",
            f"- 构建状态：`{status_text}`",
            f"- 构建目录：`{job_dir}`",
            f"- 包目录：`{bundle_dir}`",
            f"- snapshot_id：`{result['snapshot_id']}`",
            f"- 包内载荷文件数：{result['file_count']}（实际文件数含 bundle_manifest.json 为 "
            f"{(result['file_count'] + 1) if result['file_count'] is not None else None}）；"
            f"载荷字节 {result['payload_bytes']}；解压总字节（含清单）{result.get('total_bytes')}"
            f"（上限 {RUNTIME_BUNDLE_SIZE_LIMIT_BYTES}）",
            "",
            "## 阶段",
            "",
            *(f"- `{name}`：`{state}`" for name, state in stages.items()),
        ]
        if result["reasons"]:
            lines.extend(["", "## 理由", ""])
            lines.extend(f"- `{reason}`" for reason in result["reasons"])
        if job_persistence_error is not None:
            lines.extend(
                [
                    "",
                    "## 产物持久化状态",
                    "",
                    f"- 构建作业记录（build_job.json）不可写：`{_markdown_escape(job_persistence_error)}`；"
                    "本报告是最终交付状态的持久记录（blocked），作业 JSON 未写出。",
                    "- 已有产物：包目录、已完成的工程阶段与合法候选包按实际保留；"
                    "独立 verify 仍可描述候选包的工程完整性，但不能把本次构建交付当作 ready。",
                ]
            )
        lines.extend(
            [
                "",
                "构建不修改原 snapshot/issue、正式两根或源码目录；build manifest/报告保存在包外作业目录。",
                "ready 只表示工程包完整，不代表真实发行或当前数据覆盖。",
                "",
            ]
        )
        return lines

    def _decide_delivery_state(errors: list[str]) -> None:
        """Fold required-artifact failures into the final persisted delivery state.

        Called before the report is rendered, so the report never renders a
        stale ready state for an already-known persistence failure.
        """

        if not errors:
            return
        result["output_write_errors"] = list(errors)
        stages["finalize"] = "failed"
        result["reasons"] = list(result["reasons"]) + [
            "required_artifact_write_failed: " + "; ".join(errors)
        ]
        if result["status"] == "ready":
            result["status"] = "blocked"
            result["limitations"] = list(result["limitations"]) + [
                "工程阶段与候选包按实际保留（独立 verify 仍可描述其工程完整性）；"
                "但必需构建产物持久化失败，交付状态为 blocked，不能当作 ready。"
            ]

    def finish() -> dict[str, Any]:
        """Persist the final delivery state; report/job/hash failures decide the status.

        The delivery status, stages, and reasons are decided BEFORE the report
        is rendered, so the written report always matches the returned status
        and the job record. A failing job write triggers one bounded report
        rewrite recording the unwritable job, then raises an honest expected IO
        error; if the report cannot be updated either, the error says which
        artifacts exist and warns that an earlier report must not be treated as
        the final delivery. No retries beyond that, and unknown errors are never
        swallowed.
        """

        delivery_errors: list[str] = []
        if Path(result["bundle_manifest_path"]).is_file():
            try:
                result["bundle_manifest_sha256"] = file_sha256(result["bundle_manifest_path"])
            except OSError as exc:
                result["bundle_manifest_sha256"] = None
                delivery_errors.append(f"bundle_manifest_sha256: {exc}")
        try:
            total = _bundle_total_bytes(bundle_dir)
            result["total_bytes"] = total
        except OSError as exc:
            total = None
            delivery_errors.append(f"bundle_size_inventory: {exc}")

        # Decide the final delivery state BEFORE rendering the report.
        _decide_delivery_state(delivery_errors)

        try:
            _write_utf8_text(report_path, "\n".join(_render_build_report_lines(None)))
        except OSError as exc:
            delivery_errors.append(f"build_report: {exc}")
            _decide_delivery_state(delivery_errors)

        job_payload = {
            "schema_version": RUNTIME_BUNDLE_SCHEMA_VERSION,
            "command": result["command"],
            "generated_at": generated_at,
            "kind": "runtime_bundle_build_job",
            "status": result["status"],
            "reasons": result["reasons"],
            "stages": stages,
            "inputs": {
                "snapshot_manifest": {"path": str(snapshot_file), "sha256": None},
                "issue_audit": {"path": str(issue_file), "sha256": None},
            },
            "bundle_dir": str(bundle_dir),
            "bundle_manifest_path": result["bundle_manifest_path"],
            "bundle_manifest_sha256": result["bundle_manifest_sha256"],
            "snapshot_id": result["snapshot_id"],
            "original": result["original"],
            "derived": result["derived"],
            "issue": result["issue"],
            "manifest_field_diffs": result["manifest_field_diffs"],
            "bundle_files": result["manifest"].get("files") if isinstance(result["manifest"], dict) else None,
            "file_count": result["file_count"],
            "payload_bytes": result["payload_bytes"],
            "actual_file_count": (result["file_count"] + 1) if result["file_count"] is not None else None,
            "total_bytes": result.get("total_bytes"),
            "limit_bytes": RUNTIME_BUNDLE_SIZE_LIMIT_BYTES,
            "size_note": "total_bytes 为解压后包内全部实际文件（含 bundle_manifest.json）之和；超限即 blocked。",
            "output_write_errors": result.get("output_write_errors", []),
            "limitations": result["limitations"],
        }
        try:
            write_json(job_payload, job_json_path)
        except OSError as exc:
            job_error = f"build_job: {exc}"
            delivery_errors.append(job_error)
            _decide_delivery_state(delivery_errors)
            updated_note = (
                f"the build report at '{report_path}' was updated to the final blocked delivery state "
                "(job record unwritable)"
            )
            try:
                _write_utf8_text(report_path, "\n".join(_render_build_report_lines(str(exc))))
            except OSError as report_exc:
                updated_note = (
                    f"the build report at '{report_path}' could NOT be updated either ({report_exc}); "
                    "any report already on disk may show a pre-final state and must not be treated as "
                    "the final delivery"
                )
            raise OSError(
                f"could not persist the runtime bundle build job record '{job_json_path}': {exc}; "
                f"the build directory {job_dir} and any completed bundle artifacts remain on disk, "
                f"but no build job record could be written; {updated_note}"
            ) from exc
        return result

    def fail(stage: str, reason: str) -> dict[str, Any]:
        stages[stage] = "failed"
        result["status"] = "blocked"
        result["reasons"] = [reason]
        return finish()

    # Stage: snapshot_load (strict original snapshot, single model, original hashes)
    try:
        if not snapshot_file.is_file():
            raise FileNotFoundError(f"snapshot manifest does not exist: {snapshot_file}")
        loaded = load_engine_snapshot(snapshot_file)
        original_manifest = loaded["manifest"]
        if len(original_manifest["model_names"]) != 1:
            raise DataValidationError(
                "runtime bundles are single-model; the snapshot contains models: "
                + ", ".join(original_manifest["model_names"])
            )
        stages["snapshot_load"] = "completed"
        result["snapshot_id"] = original_manifest["snapshot_id"]
        result["original"] = {
            "snapshot_id": original_manifest["snapshot_id"],
            "schema_version": original_manifest["schema_version"],
            "manifest_file_sha256": loaded["manifest_sha256"],
            "manifest_content_sha256": original_manifest["manifest_sha256"],
            "state_file_sha256": loaded["state_sha256"],
            "provenance_baseline_dir": str(snapshot_file.parent),
            "issue_source_file": str(issue_file),
            "issue_source_file_sha256": file_sha256(issue_file),
        }
    except (DataValidationError, FileNotFoundError, OSError, TypeError, ValueError) as exc:
        return fail("snapshot_load", f"snapshot_load_failed: {exc}")

    # Stage: issue_evidence (original M12 record must be fresh at issue and bound to the original hash)
    try:
        if not issue_file.is_file():
            raise FileNotFoundError(f"issue audit JSON does not exist: {issue_file}")
        issue_result = load_issue_evidence(
            issue_file,
            snapshot_id=original_manifest["snapshot_id"],
            manifest_sha256=loaded["manifest_sha256"],
            state_sha256=loaded["state_sha256"],
            history=original_manifest["history"],
            history_manifest=original_manifest["history_manifest"],
            audited_at=utc_now(),
        )
        if issue_result["record_status"] != "fresh":
            raise DataValidationError(
                "runtime bundle build requires a fresh_at_issue M12 record bound to the original snapshot; "
                f"the provided record is {issue_result['record_status']}"
            )
        stages["issue_evidence"] = "completed"
        result["issue"] = {
            "relative_path": "evidence/issue_audit.json",
            "file_sha256": file_sha256(issue_file),
            "schema_version": ISSUE_EVIDENCE_SCHEMA_VERSION,
            "record_status": issue_result["record_status"],
            "record_generated_at": issue_result["record_generated_at"],
            "recorded_snapshot_sha256": json.loads(issue_file.read_text(encoding="utf-8"))["snapshot"][
                "sha256"
            ],
            "original_manifest_file_sha256": loaded["manifest_sha256"],
            "fresh_at_issue": True,
            "validation_kind": BUNDLE_LINEAGE_VALIDATION_KIND,
        }
    except (DataValidationError, FileNotFoundError, OSError, TypeError, ValueError, KeyError) as exc:
        return fail("issue_evidence", f"issue_evidence_failed: {exc}")

    # Stage: assemble (copy bytes, write the derived manifest with the fixed path leaves)
    snapshot_id = original_manifest["snapshot_id"]
    model_record = original_manifest["models"][0]
    try:
        derived_manifest = copy.deepcopy(original_manifest)
        outputs = dict(derived_manifest["outputs"])
        unexpected = sorted(set(outputs) - _ORIGINAL_OUTPUTS_KEYS)
        if unexpected:
            raise DataValidationError(
                "unexpected original snapshot outputs shape; refusing to guess: " + ", ".join(unexpected)
            )
        outputs.update(
            {
                "snapshot_dir": ".",
                "snapshot_manifest": "snapshot_manifest.json",
                "model_states": "model_states.json",
                "snapshot_report": "snapshot_report.md",
            }
        )
        derived_manifest["outputs"] = outputs
        derived_manifest["state"] = {**derived_manifest["state"], "path": "model_states.json"}

        snapshot_dir = bundle_dir / "snapshot" / snapshot_id
        snapshot_dir.mkdir(parents=True)
        evidence_dir = bundle_dir / "evidence"
        evidence_dir.mkdir()
        (snapshot_dir / "model_states.json").write_bytes(Path(loaded["state_path"]).read_bytes())
        (evidence_dir / "original_snapshot_manifest.json").write_bytes(snapshot_file.read_bytes())
        (evidence_dir / "issue_audit.json").write_bytes(issue_file.read_bytes())
        write_snapshot_manifest(derived_manifest, snapshot_dir / "snapshot_manifest.json")
        _write_derived_snapshot_report(
            snapshot_dir / "snapshot_report.md", derived_manifest, loaded["state_sha256"]
        )
        derived_loaded = load_engine_snapshot(snapshot_dir / "snapshot_manifest.json")
        observed_diff = manifest_leaf_diff(original_manifest, derived_loaded["manifest"])
        for diff in observed_diff:
            if (
                diff["change"] == "removed"
                or diff["field"] not in ALLOWED_MANIFEST_DIFF_FIELDS
                or (diff["change"] == "added" and diff["field"] not in ALLOWED_ADDED_FIELDS)
            ):
                raise DataValidationError(
                    f"unexpected manifest change while deriving: {diff['change']} {diff['field']}"
                )
        team_index = build_team_index(derived_loaded)
        write_json(team_index, bundle_dir / "team_index.json")
        _write_bundle_run_md(bundle_dir / "RUN.md", snapshot_id, str(model_record["model"]))
        result["derived"] = {
            "snapshot_id": snapshot_id,
            "manifest_relative_path": f"snapshot/{snapshot_id}/snapshot_manifest.json",
            "manifest_file_sha256": file_sha256(snapshot_dir / "snapshot_manifest.json"),
            "manifest_content_sha256": derived_loaded["manifest"]["manifest_sha256"],
            "state_relative_path": f"snapshot/{snapshot_id}/model_states.json",
            "state_file_sha256": derived_loaded["state_sha256"],
        }
        result["manifest_field_diffs"] = {
            "allowed_leaf_paths": sorted(ALLOWED_MANIFEST_DIFF_FIELDS),
            "observed": observed_diff,
        }

        files = _inventory_bundle_files(bundle_dir)
        expected_payload = fixed_bundle_payload_paths(snapshot_id)
        produced_paths = {entry["path"] for entry in files}
        if produced_paths != set(expected_payload):
            raise DataValidationError(
                "the assembled bundle does not match the fixed v1 payload set: "
                f"missing={sorted(set(expected_payload) - produced_paths)} "
                f"unexpected={sorted(produced_paths - set(expected_payload))}"
            )
        bundle_manifest_payload = {
            "schema_version": RUNTIME_BUNDLE_SCHEMA_VERSION,
            "command": "matchquant runtime-bundle-build",
            "generated_at": generated_at,
            "build_job_id": job_dir.name,
            "snapshot_id": snapshot_id,
            "model": {
                "model": model_record["model"],
                "config": model_record["config"],
                "config_sha256": model_record["config_sha256"],
                "state_payload_sha256": model_record["state_sha256"],
            },
            "original_snapshot": result["original"],
            "derived_snapshot": result["derived"],
            "state_file_sha256": loaded["state_sha256"],
            "issue_evidence": result["issue"],
            "original_manifest_copy": {
                "relative_path": "evidence/original_snapshot_manifest.json",
                "file_sha256": loaded["manifest_sha256"],
                "content_sha256": original_manifest["manifest_sha256"],
            },
            "team_index": {
                "relative_path": "team_index.json",
                "file_sha256": file_sha256(bundle_dir / "team_index.json"),
                "schema_version": TEAM_INDEX_SCHEMA_VERSION,
            },
            "files": files,
            "manifest_field_diffs": result["manifest_field_diffs"],
            "size": {
                "payload_bytes": sum(entry["bytes"] for entry in files),
                "file_count": len(files),
                "actual_file_count": len(files) + 1,
                "limit_bytes": RUNTIME_BUNDLE_SIZE_LIMIT_BYTES,
                "includes_bundle_manifest": False,
            },
            "limitations": list(_BASE_LIMITATIONS),
        }
        result["file_count"] = len(files)
        result["actual_file_count"] = len(files) + 1
        result["payload_bytes"] = sum(entry["bytes"] for entry in files)
        # The bundle manifest is the last payload write; an unfinished package is
        # never marked usable, and a failed write keeps the build blocked.
        write_json(bundle_manifest_payload, bundle_dir / "bundle_manifest.json")
        result["manifest"] = bundle_manifest_payload
        stages["assemble"] = "completed"
    except (DataValidationError, FileNotFoundError, OSError, TypeError, ValueError) as exc:
        return fail("assemble", f"assemble_failed: {exc}")

    # Stage: size gate (unpacked total including the bundle manifest; blocked candidates stay on disk)
    total_bytes = sum(entry.stat().st_size for entry in bundle_dir.rglob("*") if entry.is_file())
    if total_bytes > RUNTIME_BUNDLE_SIZE_LIMIT_BYTES:
        oversize = sorted(
            (
                (entry.relative_to(bundle_dir).as_posix(), entry.stat().st_size)
                for entry in bundle_dir.rglob("*")
                if entry.is_file()
            ),
            key=lambda item: (-item[1], item[0]),
        )
        stages["size_gate"] = "failed"
        result["status"] = "blocked"
        result["reasons"] = [
            f"bundle_oversize: unpacked total {total_bytes} bytes exceeds the fixed 5 MiB limit "
            f"({RUNTIME_BUNDLE_SIZE_LIMIT_BYTES}); per-file sizes follow and the blocked candidate is kept "
            "for a designer ruling"
        ]
        result["oversize_files"] = [
            {"path": path, "bytes": size} for path, size in oversize
        ]
        stages["finalize"] = "completed"
        return finish()

    stages["size_gate"] = "completed"

    # Stage: self_verify (the finished package must verify ready through the public verifier)
    try:
        verification = verify_runtime_bundle(bundle_dir / "bundle_manifest.json")
        if verification["status"] != "ready":
            raise DataValidationError("; ".join(verification["reasons"]))
        stages["self_verify"] = "completed"
    except (DataValidationError, OSError, ValueError, TypeError) as exc:
        return fail("self_verify", f"self_verify_failed: {exc}")

    stages["finalize"] = "completed"
    result["status"] = "ready"
    return finish()


__all__ = ["BUNDLE_BUILD_DIR_PREFIX", "RUNTIME_BUNDLE_SCHEMA_VERSION", "build_runtime_bundle"]
