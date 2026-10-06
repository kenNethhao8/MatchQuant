"""Offline B03 snapshot prediction session with an embedded read-only audit.

``run_snapshot_session`` validates an explicit single-model M11 snapshot and an
explicit score-free fixture CSV, calls the existing M11 predictor into a fresh
session directory, audits the produced run through the shared read-only audit,
and writes a session manifest plus a Chinese report. It never settles, never
runs M12, never reads history or pointer files, and never touches the network.

B04 bundle mode: ``bundle_manifest_path`` is mutually exclusive with a bare
snapshot manifest and with ``issue_evidence_path``. The bundle is re-verified
from its own bytes first, prediction runs from the bundle's derived snapshot,
the embedded audit binds the run through the derived-manifest lineage bridge,
and every bundle payload file must hash unchanged across the session.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .engine_snapshot import predict_from_engine_snapshot
from .engine_snapshot_manifest import load_engine_snapshot
from .fixtures import FIXTURE_SCHEMA_VERSION, load_fixtures
from .io import file_sha256, write_json
from .runtime_bundle_validation import (
    BUNDLE_LINEAGE_VALIDATION_KIND,
    check_bundle_payload_integrity,
    verify_runtime_bundle,
)
from .schema import DataValidationError
from .snapshot_audit_support import (
    ACTIVE_LINK_STATUS,
    EXPECTED_STAGE_ERRORS,
    SESSION_DIR_PREFIX,
    SESSION_SCHEMA_VERSION,
    ensure_safe_output_root,
    new_output_dir,
    utc_now,
)
from .snapshot_report import write_snapshot_audit_report, write_snapshot_session_report
from .snapshot_run_verify import execute_snapshot_run_audit

_BASE_LIMITATIONS = (
    "会话不自动结算、不运行 M12、不读取 history CSV、历史 processed manifest 或 current pointer；"
    "不联网、不自动 apply、不重建快照。",
    "结算需另行使用会话引用的实际 M11 prediction run 目录和独立赛果文件调用既有 settle；"
    "不能把会话目录冒充 prediction run。",
    "active_link_status 固定为 not_checked_in_runtime；fresh_at_issue 只表示发行时点的关联验证，"
    "不证明当前 active、今天最新或训练到目标比赛。",
    "会话输出是状态与预测计算一致性证据，不重新证明历史训练正确性，也不构成校准、盈利或投注价值证明。",
)
_REPLAY_LIMITATION = "replay 模式是可复现的历史回溯研究（retrospective），不是当前实时预测。"
_RESEARCH_LIMITATION = (
    "research 模式未证明目标时点的赛果覆盖；待赛覆盖缺失必须保持可见，不能当作当前可用预测产品。"
)


def _write_blocked_audit_files(
    session_dir: Path, *, audit_command: str, audited_at: str, reasons: list[str]
) -> dict[str, Any]:
    """Leave explicit blocked audit artifacts when the session fails before the audit stage."""

    audit: dict[str, Any] = {
        "schema_version": "baseline-snapshot-run-audit-v1",
        "command": audit_command,
        "generated_at": audited_at,
        "audited_at": audited_at,
        "validation_scope": "single_model_m11_snapshot_run_computation_audit_no_history_no_network",
        "status": "blocked",
        "reasons": reasons,
        "stages": {
            "evidence_load": "not_run",
            "issue_evidence": "not_run",
            "repredict": "not_run",
            "inputs_unchanged": "not_run",
        },
        "run": None,
        "snapshot": None,
        "engine_checks": [
            {
                "check": "session_reached_audit_stage",
                "status": "not_run",
                "detail": "the session was blocked before a prediction run existed",
            }
        ],
        "repredict": None,
        "issue_evidence": None,
        "inputs": {},
        "inputs_unchanged": "not_run",
        "predictions_usable": False,
        "limitations": [*_BASE_LIMITATIONS],
        "outputs": {
            "audit_json": str(session_dir / "prediction_audit.json"),
            "audit_report": str(session_dir / "prediction_audit.md"),
        },
    }
    write_json(audit, session_dir / "prediction_audit.json")
    write_snapshot_audit_report(session_dir / "prediction_audit.md", audit)
    return audit


def _team_annotations(
    bundle_verification: dict[str, Any], fixtures: Any
) -> dict[str, Any]:
    """Display-only unseen-team annotations for one bundle session.

    The annotations never modify M11 prediction columns, the run schema, the
    model config, or the probabilities. For frequency the empty team index is
    never used to call any input team unseen: the global-frequency basis is
    declared instead.
    """

    index = bundle_verification["team_index"]["payload"]
    model = str(index["model"])
    if index["team_membership"] == "not_applicable":
        return {
            "frequency_basis": "global",
            "team_membership": "not_applicable",
            "unseen_home_teams": None,
            "unseen_away_teams": None,
            "unseen_team_prior": None,
            "note": (
                "frequency 模型没有球队学习状态：全部输入使用全局频率；"
                "不把输入球队称为新队，也不维护球队清单。"
            ),
        }
    known = set(index["teams"])
    unseen_home = sorted({str(value) for value in fixtures["home_team"]} - known)
    unseen_away = sorted({str(value) for value in fixtures["away_team"]} - known)
    config = bundle_verification["model"]["config"]
    if model == "elo":
        prior: dict[str, Any] = {
            "mechanism": "unseen teams use the model initial_rating prior",
            "initial_rating": config.get("initial_rating", 1500.0),
        }
    else:
        prior = {
            "mechanism": "unseen teams fall back to the shrunk global goal rates",
            "prior_home_goals": config.get("prior_home_goals"),
            "prior_away_goals": config.get("prior_away_goals"),
            "prior_weight": config.get("prior_weight"),
        }
    return {
        "frequency_basis": None,
        "team_membership": "learned_from_state",
        "unseen_home_teams": unseen_home,
        "unseen_away_teams": unseen_away,
        "unseen_team_prior": prior,
        "note": (
            "unseen 标注是独立会话注释：不修改 M11 预测列、run schema、模型配置或概率；"
            "已知球队也不证明其仍参加当前赛季。"
        ),
    }


def run_snapshot_session(
    snapshot_manifest_path: str | Path | None,
    fixtures_path: str | Path,
    *,
    mode: str = "replay",
    output_root: str | Path,
    issue_evidence_path: str | Path | None = None,
    bundle_manifest_path: str | Path | None = None,
) -> dict[str, Any]:
    """Run one offline snapshot prediction session and return its structured result.

    Exactly one of ``snapshot_manifest_path`` (bare B03 mode) or
    ``bundle_manifest_path`` (B04 bundle mode) must be provided; bundle mode
    cannot override the issue evidence because the original M12 record is read
    from the verified bundle itself.
    """

    if mode not in {"replay", "research"}:
        raise ValueError("mode must be 'replay' or 'research'")
    if bundle_manifest_path is not None:
        if snapshot_manifest_path is not None:
            raise ValueError(
                "bare snapshot and bundle modes are mutually exclusive; provide exactly one of "
                "snapshot_manifest_path or bundle_manifest_path"
            )
        if issue_evidence_path is not None:
            raise ValueError(
                "bundle sessions read the issue evidence inside the verified bundle; "
                "issue_evidence_path must be None"
            )
    elif snapshot_manifest_path is None:
        raise ValueError(
            "provide exactly one of snapshot_manifest_path or bundle_manifest_path"
        )
    bundle_file = (
        Path(bundle_manifest_path).expanduser().resolve() if bundle_manifest_path is not None else None
    )
    fixtures_file = Path(fixtures_path).expanduser().resolve()
    issue_file = (
        Path(issue_evidence_path).expanduser().resolve() if issue_evidence_path is not None else None
    )
    output_root_path = ensure_safe_output_root(
        output_root,
        [
            *([(bundle_file.parent, "dir")] if bundle_file is not None else []),
            *(
                [(Path(snapshot_manifest_path).expanduser().resolve(), "file")]
                if snapshot_manifest_path is not None
                else []
            ),
            (fixtures_file, "file"),
            *([(issue_file, "file")] if issue_file is not None else []),
        ],
    )
    generated_at = utc_now()
    session_dir = new_output_dir(output_root_path, SESSION_DIR_PREFIX)
    manifest_path = session_dir / "session_manifest.json"
    report_path = session_dir / "session_report.md"

    manifest: dict[str, Any] = {
        "schema_version": SESSION_SCHEMA_VERSION,
        "command": "matchquant snapshot-session",
        "session_id": session_dir.name,
        "generated_at": generated_at,
        "audited_at": None,
        "mode": mode,
        "validation_scope": "retrospective_replay" if mode == "replay" else "local_pending_research",
        "status": "blocked",
        "reasons": [],
        "stages": {
            "bundle_verify": "not_run" if bundle_file is not None else None,
            "snapshot_load": "not_run",
            "fixtures_load": "not_run",
            "prediction": "not_run",
            "audit": "not_run",
            "finalize": "not_run",
        },
        "inputs": {
            "snapshot_manifest": (
                {"path": str(Path(snapshot_manifest_path).expanduser().resolve()), "sha256": None}
                if snapshot_manifest_path is not None
                else None
            ),
            "bundle_manifest": (
                {"path": str(bundle_file), "sha256": None} if bundle_file is not None else None
            ),
            "fixtures": {"path": str(fixtures_file), "sha256": None, "rows": None},
            "issue_evidence": {
                "provided": issue_file is not None,
                "path": str(issue_file) if issue_file is not None else None,
                "sha256": None,
            },
        },
        "snapshot": None,
        "bundle": None,
        "team_index": None,
        "team_annotations": None,
        "prediction_run": None,
        "audit": None,
        "coverage": None,
        "active_link_status": ACTIVE_LINK_STATUS,
        "inputs_unchanged": "not_run",
        "predictions_usable": False,
        "limitations": [
            *_BASE_LIMITATIONS,
            _REPLAY_LIMITATION if mode == "replay" else _RESEARCH_LIMITATION,
        ],
        "outputs": {
            "session_dir": str(session_dir),
            "session_manifest": str(manifest_path),
            "session_report": str(report_path),
            "prediction_audit_json": str(session_dir / "prediction_audit.json"),
            "prediction_audit_md": str(session_dir / "prediction_audit.md"),
            "prediction_run_dir": None,
        },
    }
    if bundle_file is None:
        del manifest["stages"]["bundle_verify"]

    def persist() -> None:
        write_json(manifest, manifest_path)

    def finish() -> dict[str, Any]:
        output_errors: list[str] = []
        try:
            write_snapshot_session_report(report_path, manifest)
        except OSError as exc:
            output_errors.append(f"session_report: {exc}")
        manifest["output_hashes"] = {}
        for key, artifact_path in (
            ("prediction_audit_json", session_dir / "prediction_audit.json"),
            ("prediction_audit_md", session_dir / "prediction_audit.md"),
            ("session_report", report_path),
        ):
            try:
                manifest["output_hashes"][key] = (
                    file_sha256(artifact_path) if artifact_path.is_file() else None
                )
            except OSError as exc:
                manifest["output_hashes"][key] = None
                output_errors.append(f"{key}: {exc}")
        if output_errors:
            # A required session artifact (report or recorded hash) failed to be
            # written: the session delivery is blocked even when the engineering
            # stages and the embedded sub-audit completed. The M11 run and audit
            # evidence stay on disk and are reported as-is.
            manifest["output_write_errors"] = output_errors
            manifest["stages"]["finalize"] = "failed"
            manifest["reasons"] = list(manifest["reasons"]) + [
                "required_artifact_write_failed: " + "; ".join(output_errors)
            ]
            manifest["status"] = "blocked"
            manifest["predictions_usable"] = False
            manifest["coverage"] = {"status": "not_evaluated", "reasons": ["required_artifact_write_failed"]}
            manifest["limitations"] = list(manifest["limitations"]) + [
                "工程阶段与子审计结果按实际保留；但会话必需产物写入失败，交付状态为 blocked，不能当作 ready。"
            ]
        write_json(manifest, manifest_path)
        return {
            "status": manifest["status"],
            "mode": mode,
            "reasons": manifest["reasons"],
            "session_id": manifest["session_id"],
            "session_dir": str(session_dir),
            "session_manifest_path": str(manifest_path),
            "session_manifest_sha256": file_sha256(manifest_path),
            "session_report_path": str(report_path),
            "prediction_run_dir": manifest["outputs"]["prediction_run_dir"],
            "predictions_path": (
                manifest["prediction_run"]["predictions"]["path"]
                if isinstance(manifest["prediction_run"], dict)
                else None
            ),
            "bundle": manifest["bundle"],
            "team_index": manifest["team_index"],
            "team_annotations": manifest["team_annotations"],
            "prediction_audit_path": manifest["outputs"]["prediction_audit_json"],
            "prediction_audit_sha256": manifest["output_hashes"]["prediction_audit_json"],
            "prediction_audit_report_path": manifest["outputs"]["prediction_audit_md"],
            "audit_status": (
                manifest["audit"]["status"] if isinstance(manifest["audit"], dict) else "blocked"
            ),
            "audit_reasons": (
                manifest["audit"]["reasons"] if isinstance(manifest["audit"], dict) else []
            ),
            "predictions_usable": manifest["predictions_usable"],
            "output_write_errors": list(manifest.get("output_write_errors", [])),
            "manifest": manifest,
        }

    def block(stage: str, exc: Exception, *, write_audit: bool = True) -> dict[str, Any]:
        manifest["stages"][stage] = "failed"
        manifest["status"] = "blocked"
        manifest["reasons"] = [f"{stage}_failed: {exc}"]
        manifest["predictions_usable"] = False
        if write_audit:
            blocked_audit = _write_blocked_audit_files(
                session_dir,
                audit_command="matchquant snapshot-session",
                audited_at=utc_now(),
                reasons=list(manifest["reasons"]),
            )
            manifest["audit"] = {
                "status": blocked_audit["status"],
                "reasons": blocked_audit["reasons"],
                "fresh_at_issue": False,
                "active_link_status": ACTIVE_LINK_STATUS,
                "prediction_audit_json": {"path": manifest["outputs"]["prediction_audit_json"]},
                "prediction_audit_md": {"path": manifest["outputs"]["prediction_audit_md"]},
            }
        persist()
        return finish()

    # Stage: bundle_verify (bundle mode only; the whole package is re-verified from its own bytes)
    snapshot_file = Path(snapshot_manifest_path).expanduser().resolve() if snapshot_manifest_path is not None else None
    bundle_verification: dict[str, Any] | None = None
    if bundle_file is not None:
        try:
            bundle_verification = verify_runtime_bundle(bundle_file)
            if bundle_verification["status"] != "ready":
                raise DataValidationError(
                    "bundle verification failed: " + "; ".join(bundle_verification["reasons"])
                )
            derived_path = Path(bundle_verification["derived"]["resolved_manifest_path"])
            if not derived_path.is_file():
                raise FileNotFoundError(f"derived snapshot manifest does not exist: {derived_path}")
        except EXPECTED_STAGE_ERRORS as exc:
            return block("bundle_verify", exc)
        manifest["stages"]["bundle_verify"] = "completed"
        manifest["inputs"]["bundle_manifest"]["sha256"] = bundle_verification["bundle_manifest_file_sha256"]
        manifest["bundle"] = {
            "bundle_manifest_path": bundle_verification["bundle_manifest_path"],
            "bundle_manifest_file_sha256": bundle_verification["bundle_manifest_file_sha256"],
            "bundle_root": bundle_verification["bundle_root"],
            "snapshot_id": bundle_verification["snapshot_id"],
            "validation_kind": BUNDLE_LINEAGE_VALIDATION_KIND,
            "original_manifest_file_sha256": bundle_verification["original"]["manifest_file_sha256"],
            "derived_manifest_file_sha256": bundle_verification["derived"]["file_sha256"],
            "state_file_sha256": bundle_verification["derived"]["state_sha256"],
            "fresh_at_issue": bundle_verification["issue_validation"]["fresh_at_issue"],
            "active_link_status": bundle_verification["issue_validation"]["active_link_status"],
        }
        manifest["team_index"] = {
            "path": bundle_verification["team_index"]["path"],
            "file_sha256": bundle_verification["team_index"]["file_sha256"],
            "model": bundle_verification["team_index"]["payload"]["model"],
            "team_membership": bundle_verification["team_index"]["payload"]["team_membership"],
            "frequency_basis": bundle_verification["team_index"]["payload"]["frequency_basis"],
            "teams_count": len(bundle_verification["team_index"]["payload"]["teams"]),
        }
        snapshot_file = Path(bundle_verification["derived"]["resolved_manifest_path"])
    persist()

    # Stage: snapshot_load (expected input failures are recorded as structured blocked artifacts)
    try:
        if not snapshot_file.is_file():
            raise FileNotFoundError(f"snapshot manifest does not exist: {snapshot_file}")
        loaded = load_engine_snapshot(snapshot_file)
        loaded_manifest = loaded["manifest"]
        if len(loaded_manifest["model_names"]) != 1:
            raise DataValidationError(
                "snapshot session requires a single-model M11 snapshot; "
                f"the snapshot contains models: {', '.join(loaded_manifest['model_names'])}"
            )
        model_evidence = loaded_manifest["models"][0]
        manifest["stages"]["snapshot_load"] = "completed"
        if snapshot_manifest_path is not None:
            manifest["inputs"]["snapshot_manifest"]["sha256"] = loaded["manifest_sha256"]
        manifest["snapshot"] = {
            "snapshot_id": loaded_manifest["snapshot_id"],
            "manifest_path": str(snapshot_file),
            "manifest_file_sha256": loaded["manifest_sha256"],
            "manifest_content_sha256": loaded_manifest["manifest_sha256"],
            "state_path": loaded["state_path"],
            "state_sha256": loaded["state_sha256"],
            "as_of": loaded_manifest["as_of"],
            "model_names": list(loaded_manifest["model_names"]),
            "model_config": model_evidence["config"],
            "model_config_sha256": model_evidence["config_sha256"],
            "history_data_cutoff": loaded_manifest["history"].get("data_cutoff"),
            "used_data_cutoff": loaded_manifest["history"].get("used_data_cutoff"),
            "rows_used_before_as_of": loaded_manifest["history"].get("rows_used_before_as_of"),
            "history_manifest_dataset_version": loaded_manifest["history_manifest"].get(
                "dataset_version"
            ),
            "season_mode": loaded_manifest["season_mode"],
        }
    except EXPECTED_STAGE_ERRORS as exc:
        return block("snapshot_load", exc)
    persist()

    # Stage: fixtures_load
    try:
        if not fixtures_file.is_file():
            raise FileNotFoundError(f"fixtures CSV does not exist: {fixtures_file}")
        fixtures = load_fixtures(fixtures_file)
        manifest["stages"]["fixtures_load"] = "completed"
        manifest["inputs"]["fixtures"]["sha256"] = file_sha256(fixtures_file)
        manifest["inputs"]["fixtures"]["rows"] = int(len(fixtures))
        manifest["inputs"]["fixtures"]["schema"] = f"{FIXTURE_SCHEMA_VERSION}-score-free"
    except EXPECTED_STAGE_ERRORS as exc:
        return block("fixtures_load", exc)
    persist()

    # Bundle mode: display-only unseen-team annotations derived from the verified team index
    if bundle_verification is not None:
        manifest["team_annotations"] = _team_annotations(bundle_verification, fixtures)

    # Stage: prediction (existing M11 predictor into the session directory)
    try:
        predicted = predict_from_engine_snapshot(
            snapshot_file, fixtures_file, output_root=session_dir
        )
        run_dir = Path(predicted["run_dir"])
        manifest["stages"]["prediction"] = "completed"
        manifest["outputs"]["prediction_run_dir"] = str(run_dir)
        manifest["prediction_run"] = {
            "run_dir": str(run_dir),
            "producer_schema": predicted["config"]["schema_version"],
            "run_config": {
                "path": str(run_dir / "run_config.json"),
                "sha256": file_sha256(run_dir / "run_config.json"),
            },
            "predictions": {
                "path": str(run_dir / "predictions.csv"),
                "sha256": file_sha256(run_dir / "predictions.csv"),
                "rows": int(len(predicted["predictions"])),
            },
            "snapshot_report": {"path": str(predicted["snapshot_report_path"])},
        }
    except EXPECTED_STAGE_ERRORS as exc:
        return block("prediction", exc)
    persist()

    # Stage: audit (shared read-only audit of the actual produced run)
    audited_at = utc_now()
    manifest["audited_at"] = audited_at
    try:
        audit = execute_snapshot_run_audit(
            run_dir=run_dir,
            issue_evidence_path=issue_file,
            audited_at=audited_at,
            output_dir=session_dir,
            audit_command="matchquant snapshot-session",
            bundle_context=bundle_verification,
        )
    except EXPECTED_STAGE_ERRORS as exc:
        return block("audit", exc)
    manifest["stages"]["audit"] = (
        "completed" if not any(stage == "failed" for stage in audit["stages"].values()) else "failed"
    )
    issue_block = audit["issue_evidence"] if isinstance(audit["issue_evidence"], dict) else None
    manifest["audit"] = {
        "status": audit["status"],
        "reasons": audit["reasons"],
        "engine_checks": audit["engine_checks"],
        "issue_evidence": issue_block,
        "fresh_at_issue": bool(issue_block["fresh_at_issue"]) if issue_block else False,
        "active_link_status": ACTIVE_LINK_STATUS,
        "prediction_audit_json": {"path": manifest["outputs"]["prediction_audit_json"]},
        "prediction_audit_md": {"path": manifest["outputs"]["prediction_audit_md"]},
    }
    persist()

    # Stage: finalize (input immutability re-check and status derivation)
    unchanged_records: list[tuple[str, Path, str]] = []
    if bundle_verification is None:
        unchanged_records.append(
            ("snapshot_manifest", snapshot_file, manifest["inputs"]["snapshot_manifest"]["sha256"])
        )
    else:
        unchanged_records.append(
            ("bundle_manifest", bundle_file, manifest["inputs"]["bundle_manifest"]["sha256"])
        )
    unchanged_records.extend(
        [
            ("fixtures", fixtures_file, manifest["inputs"]["fixtures"]["sha256"]),
            (
                "run_config",
                run_dir / "run_config.json",
                manifest["prediction_run"]["run_config"]["sha256"],
            ),
            (
                "predictions",
                run_dir / "predictions.csv",
                manifest["prediction_run"]["predictions"]["sha256"],
            ),
        ]
    )
    if issue_file is not None and issue_block is not None:
        unchanged_records.append(("issue_evidence", issue_file, str(issue_block["sha256"])))
    inputs_unchanged = True
    try:
        for label, path, recorded in unchanged_records:
            if file_sha256(path) != recorded:
                inputs_unchanged = False
                manifest["reasons"].append(f"input_changed_during_session: {label}")
                break
    except OSError as exc:
        manifest["reasons"].append(f"input_verification_failed: {exc}")
        inputs_unchanged = False
    manifest["inputs_unchanged"] = inputs_unchanged

    bundle_unchanged: bool | None = None
    if bundle_verification is not None and inputs_unchanged:
        from .runtime_bundle_validation import fixed_bundle_payload_paths

        bundle_ok, bundle_errors, _recorded = check_bundle_payload_integrity(
            Path(bundle_verification["bundle_root"]),
            bundle_verification["files"],
            expected_payload_paths=fixed_bundle_payload_paths(bundle_verification["snapshot_id"]),
        )
        bundle_unchanged = bundle_ok
        if not bundle_ok:
            manifest["reasons"].append(
                "bundle_files_changed_during_session: " + "; ".join(bundle_errors)
            )
            inputs_unchanged = False

    audit_blocked = audit["status"] == "blocked" or any(
        stage == "failed" for stage in audit["stages"].values()
    )
    if not inputs_unchanged or audit_blocked:
        manifest["stages"]["finalize"] = "completed"
        manifest["status"] = "blocked"
        if audit_blocked and not any(
            reason.startswith(("input_changed_during_session", "input_verification_failed"))
            for reason in manifest["reasons"]
        ):
            manifest["reasons"].extend(f"audit_blocked: {reason}" for reason in audit["reasons"])
        manifest["predictions_usable"] = False
        manifest["coverage"] = {"status": "not_evaluated", "reasons": ["session_blocked"]}
        persist()
        return finish()

    if mode == "research":
        manifest["status"] = "needs_review"
        manifest["reasons"].append("coverage_to_fixture_time_not_proven")
        manifest["coverage"] = {
            "status": "not_proven",
            "reasons": ["coverage_to_fixture_time_not_proven"],
        }
    else:
        manifest["status"] = "ready"
        manifest["coverage"] = {"status": "retrospective_replay_not_live", "reasons": []}
    manifest["stages"]["finalize"] = "completed"
    manifest["predictions_usable"] = True
    if audit["status"] != "ready":
        manifest["limitations"].append(
            "子审计状态为 "
            f"{audit['status']}（理由：{', '.join(audit['reasons']) or 'none'}）；"
            "该子审计状态按实际保留，不因会话 ready 而升格。"
        )
    persist()
    return finish()


__all__ = ["SESSION_SCHEMA_VERSION", "run_snapshot_session"]
