"""Read-only B03 audit of one existing single-model M11 snapshot prediction run.

``verify_snapshot_run`` re-loads the run through the B02 strict prediction
evidence adapter, binds the optional M12 issue evidence, re-predicts from the
original snapshot and original score-free fixtures inside an isolated temporary
directory, compares probabilities with ``atol=1e-12, rtol=0``, and verifies the
input inventory is unchanged. The original run is never overwritten and the
recomputed files are never presented as a new version of it.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from .engine_snapshot import SNAPSHOT_PREDICT_SCHEMA_VERSION, predict_from_engine_snapshot
from .forecast import FORECAST_PREDICTION_COLUMNS
from .io import file_sha256, write_json
from .metrics import PROBABILITY_COLUMNS
from .prediction_evidence import load_prediction_evidence
from .schema import DataValidationError
from .settlement_validation import _read_prediction_csv
from .snapshot_audit_support import (
    AUDIT_DIR_PREFIX,
    AUDIT_SCHEMA_VERSION,
    EXPECTED_STAGE_ERRORS,
    ensure_safe_output_root,
    load_issue_evidence,
    missing_issue_evidence,
    new_output_dir,
    utc_now,
)
from .snapshot_report import AUDIT_SCOPE_TEXT, write_snapshot_audit_report

_PROBABILITY_TOLERANCE = {"atol": 1e-12, "rtol": 0.0}
_IDENTITY_COLUMNS = ("fixture_id", "date", "season", "home_team", "away_team", "model")

_BASE_LIMITATIONS = (
    "审计是状态与预测计算的一致性证明，不重新证明历史训练正确性。",
    "审计不读取 history CSV、历史 processed manifest 或 current pointer；不联网、不自动 apply、不重建快照。",
    "active_link_status 固定为 not_checked_in_runtime；fresh_at_issue 只表示发行时点的关联验证。",
)


def _preflight_recorded_paths(run_dir: Path) -> list[Path]:
    """Best-effort recorded input paths used only to reject unsafe output roots.

    The extracted paths are never used to accept evidence; the strict run-side
    validation still happens later through ``load_prediction_evidence``.
    """

    paths: list[Path] = []
    try:
        payload = json.loads((run_dir / "run_config.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return paths
    if not isinstance(payload, dict):
        return paths
    snapshot = payload.get("snapshot")
    if isinstance(snapshot, Mapping):
        for key in ("manifest_path", "state_path"):
            value = snapshot.get(key)
            if isinstance(value, str) and value.strip():
                paths.append(Path(value).expanduser())
    fixtures = payload.get("fixtures")
    if isinstance(fixtures, Mapping):
        value = fixtures.get("path")
        if isinstance(value, str) and value.strip():
            paths.append(Path(value).expanduser())
    return paths


def _safe_hash_inventory(paths: Mapping[str, Path]) -> tuple[dict[str, dict[str, Any]], list[str]]:
    """Hash every inventoried input; expected read failures become explicit unavailable entries."""

    inventory: dict[str, dict[str, Any]] = {}
    errors: list[str] = []
    for label, path in paths.items():
        try:
            inventory[label] = {"path": str(path), "sha256": file_sha256(path)}
        except OSError as exc:
            inventory[label] = {
                "path": str(path),
                "sha256": None,
                "hash_status": "unavailable",
                "error": str(exc),
            }
            errors.append(f"{label}: {type(exc).__name__}: {exc}")
    return inventory, errors


def _expected_goals_equal(left: pd.DataFrame, right: pd.DataFrame) -> bool:
    for column in ("expected_home_goals", "expected_away_goals"):
        first = left[column] if column in left.columns else pd.Series([None] * len(left))
        second = right[column] if column in right.columns else pd.Series([None] * len(right))
        first_numeric = pd.to_numeric(first, errors="coerce")
        second_numeric = pd.to_numeric(second, errors="coerce")
        both_missing = first_numeric.isna() & second_numeric.isna()
        if not bool((both_missing | (first_numeric == second_numeric)).all()):
            return False
    return True


def _blocked_audit(
    *,
    run_dir: Path,
    audit_command: str,
    audited_at: str,
    output_dir: Path,
    reasons: list[str],
    stages: dict[str, str],
    checks: list[dict[str, str]],
    run_block: dict[str, Any] | None,
    snapshot_block: dict[str, Any] | None,
    issue_block: dict[str, Any] | None,
    inputs_block: dict[str, Any],
) -> dict[str, Any]:
    audit: dict[str, Any] = {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "command": audit_command,
        "generated_at": audited_at,
        "audited_at": audited_at,
        "validation_scope": "single_model_m11_snapshot_run_computation_audit_no_history_no_network",
        "status": "blocked",
        "reasons": reasons,
        "stages": stages,
        "run": run_block,
        "snapshot": snapshot_block,
        "engine_checks": checks,
        "repredict": None,
        "issue_evidence": issue_block,
        "inputs": inputs_block,
        "inputs_unchanged": "not_run",
        "predictions_usable": False,
        "limitations": [*_BASE_LIMITATIONS, AUDIT_SCOPE_TEXT],
        "outputs": {
            "audit_json": str(output_dir / "prediction_audit.json"),
            "audit_report": str(output_dir / "prediction_audit.md"),
        },
    }
    _write_audit_files(audit, output_dir)
    return audit


def _write_audit_files(audit: dict[str, Any], output_dir: Path) -> None:
    report_path = output_dir / "prediction_audit.md"
    audit_json_path = output_dir / "prediction_audit.json"
    write_snapshot_audit_report(report_path, audit)
    audit["outputs"]["audit_report_sha256"] = file_sha256(report_path)
    write_json(audit, audit_json_path)


def execute_snapshot_run_audit(
    *,
    run_dir: Path,
    issue_evidence_path: Path | None,
    audited_at: str,
    output_dir: Path,
    audit_command: str,
) -> dict[str, Any]:
    """Run the shared read-only audit and write ``prediction_audit.json``/``.md``."""

    output_dir.mkdir(parents=True, exist_ok=True)
    stages = {
        "evidence_load": "not_run",
        "issue_evidence": "not_run",
        "input_inventory": "not_run",
        "repredict": "not_run",
        "inputs_unchanged": "not_run",
    }
    checks: list[dict[str, str]] = []

    def fail(
        reason: str,
        *,
        run_block: dict[str, Any] | None = None,
        snapshot_block: dict[str, Any] | None = None,
        issue_block: dict[str, Any] | None = None,
        inputs_block: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        return _blocked_audit(
            run_dir=run_dir,
            audit_command=audit_command,
            audited_at=audited_at,
            output_dir=output_dir,
            reasons=[reason],
            stages=stages,
            checks=checks,
            run_block=run_block,
            snapshot_block=snapshot_block,
            issue_block=issue_block,
            inputs_block=inputs_block or {},
        )

    run_config_path = run_dir / "run_config.json"
    predictions_path = run_dir / "predictions.csv"
    try:
        evidence = load_prediction_evidence(run_dir)
    except EXPECTED_STAGE_ERRORS as exc:
        stages["evidence_load"] = "failed"
        checks.append(
            {"check": "prediction_evidence_strict_load", "status": "failed", "detail": str(exc)}
        )
        return fail(f"prediction_evidence_load_failed: {exc}")

    stages["evidence_load"] = "completed"
    checks.append(
        {
            "check": "prediction_evidence_strict_load",
            "status": "passed",
            "detail": f"producer_schema={evidence['producer_schema']}",
        }
    )
    run_block = {
        "run_dir": str(run_dir),
        "producer_schema": evidence["producer_schema"],
        "producer_command": evidence["producer_command"],
        "run_config": {"path": str(evidence["run_config_path"]), "sha256": evidence["run_config_sha256"]},
        "predictions": {
            "path": str(evidence["predictions_path"]),
            "sha256": evidence["predictions_sha256"],
            "rows": int(len(evidence["predictions"])),
        },
        "model_name": evidence["model_name"],
        "model_config": evidence["model_config"],
        "model_config_sha256": evidence["model_config_sha256"],
    }
    if evidence["producer_schema"] != SNAPSHOT_PREDICT_SCHEMA_VERSION:
        checks.append(
            {
                "check": "single_model_m11_producer",
                "status": "failed",
                "detail": f"producer_schema={evidence['producer_schema']}",
            }
        )
        return fail(
            "snapshot run audit only accepts single-model M11 engine snapshot prediction runs; "
            f"producer_schema={evidence['producer_schema']}",
            run_block=run_block,
        )

    origin = evidence["origin_evidence"]
    snapshot_manifest_path = Path(str(origin["snapshot_manifest_path"]))
    state_path = Path(str(origin["state_path"]))
    fixtures_path = Path(str(origin["fixtures"]["path"]))
    snapshot_block = {
        "snapshot_id": origin["snapshot_id"],
        "manifest_path": str(snapshot_manifest_path),
        "manifest_file_sha256": origin["snapshot_manifest_sha256"],
        "manifest_content_sha256": origin["snapshot_manifest_content_sha256"],
        "state_path": str(state_path),
        "state_sha256": origin["state_sha256"],
        "as_of": evidence["as_of_meta"],
        "history_data_cutoff": origin["history"].get("data_cutoff"),
        "used_data_cutoff": origin["history"].get("used_data_cutoff"),
        "rows_used_before_as_of": origin["history"].get("rows_used_before_as_of"),
        "history_manifest_dataset_version": origin["history_manifest"].get("dataset_version"),
        "season_mode": origin.get("season_mode"),
    }
    checks.extend(
        {
            "check": name,
            "status": "passed",
            "detail": "enforced by load_prediction_evidence (B02 strict adapter)",
        }
        for name in (
            "snapshot_manifest_file_and_content_hash",
            "snapshot_state_hash",
            "single_model_set_consistency",
            "run_config_predictions_hash",
            "fixtures_hash_rows_and_row_identity",
            "as_of_three_field_binding",
            "used_cutoff_temporal_gate",
            "prediction_probabilities_valid",
            "fixture_boundary_at_or_after_as_of",
        )
    )

    issue_block: dict[str, Any]
    if issue_evidence_path is not None:
        if not Path(issue_evidence_path).is_file():
            stages["issue_evidence"] = "failed"
            checks.append(
                {
                    "check": "issue_evidence_validation",
                    "status": "failed",
                    "detail": f"issue evidence JSON does not exist: {issue_evidence_path}",
                }
            )
            return fail(
                f"issue_evidence_missing_or_unreadable: issue evidence JSON does not exist: {issue_evidence_path}",
                run_block=run_block,
                snapshot_block=snapshot_block,
                inputs_block={},
            )
        try:
            issue_block = load_issue_evidence(
                issue_evidence_path,
                snapshot_id=str(origin["snapshot_id"]),
                manifest_sha256=str(origin["snapshot_manifest_sha256"]),
                state_sha256=str(origin["state_sha256"]),
                history=origin["history"],
                history_manifest=origin["history_manifest"],
                audited_at=audited_at,
            )
        except EXPECTED_STAGE_ERRORS as exc:
            stages["issue_evidence"] = "failed"
            checks.append({"check": "issue_evidence_validation", "status": "failed", "detail": str(exc)})
            return fail(
                f"issue_evidence_invalid: {exc}",
                run_block=run_block,
                snapshot_block=snapshot_block,
                inputs_block={},
            )
        stages["issue_evidence"] = "completed"
        checks.append(
            {
                "check": "issue_evidence_validation",
                "status": "passed",
                "detail": f"record_status={issue_block['record_status']}",
            }
        )
    else:
        issue_block = missing_issue_evidence()
        stages["issue_evidence"] = "completed"

    inventory_paths = {
        "run_config": run_config_path,
        "predictions": predictions_path,
        "snapshot_manifest": snapshot_manifest_path,
        "snapshot_state": state_path,
        "fixtures": fixtures_path,
    }
    if issue_evidence_path is not None:
        inventory_paths["issue_evidence"] = Path(issue_evidence_path)
    inventory, inventory_errors = _safe_hash_inventory(inventory_paths)
    if inventory_errors:
        stages["input_inventory"] = "failed"
        checks.append(
            {
                "check": "input_inventory_hash",
                "status": "failed",
                "detail": "; ".join(inventory_errors),
            }
        )
        return fail(
            "input_read_failed: " + "; ".join(inventory_errors),
            run_block=run_block,
            snapshot_block=snapshot_block,
            issue_block=issue_block,
            inputs_block=inventory,
        )
    stages["input_inventory"] = "completed"
    checks.append({"check": "input_inventory_hash", "status": "passed", "detail": ""})

    repredict_block: dict[str, Any] | None = None
    try:
        with tempfile.TemporaryDirectory(prefix="m11_repredict_") as temporary:
            recomputed = predict_from_engine_snapshot(
                snapshot_manifest_path, fixtures_path, output_root=Path(temporary)
            )
            recomputed_frame = _read_prediction_csv(
                Path(recomputed["predictions_path"]), producer_label="M11 repredict"
            ).reset_index(drop=True)
            recomputed_hash = file_sha256(Path(recomputed["predictions_path"]))
            original = evidence["predictions"].reset_index(drop=True)
            if len(original) != len(recomputed_frame):
                raise DataValidationError(
                    f"repredicted row count {len(recomputed_frame)} does not match the original run "
                    f"row count {len(original)}"
                )
            identity_mismatch = [
                column
                for column in _IDENTITY_COLUMNS
                if not original[column].astype(str).eq(recomputed_frame[column].astype(str)).all()
            ]
            if identity_mismatch:
                raise DataValidationError(
                    "repredicted row identity does not match the original run for columns: "
                    + ", ".join(identity_mismatch)
                )
            differences = np.abs(
                original.loc[:, PROBABILITY_COLUMNS].to_numpy(dtype=float)
                - recomputed_frame.loc[:, PROBABILITY_COLUMNS].to_numpy(dtype=float)
            )
            max_abs_difference = float(differences.max()) if differences.size else 0.0
            probabilities_match = bool(
                np.all(differences <= _PROBABILITY_TOLERANCE["atol"])
            )
            most_likely_match = original["model_most_likely_result"].astype(str).eq(
                recomputed_frame["model_most_likely_result"].astype(str)
            ).all()
            goals_match = _expected_goals_equal(original, recomputed_frame)
            if not probabilities_match:
                raise DataValidationError(
                    "repredicted probabilities do not match the original predictions at "
                    f"atol=1e-12, rtol=0; max absolute difference {max_abs_difference!r}"
                )
            if not most_likely_match:
                raise DataValidationError(
                    "repredicted model_most_likely_result does not match the original predictions"
                )
            if not goals_match:
                raise DataValidationError(
                    "repredicted expected goals do not match the original predictions"
                )
    except EXPECTED_STAGE_ERRORS as exc:
        stages["repredict"] = "failed"
        checks.append({"check": "repredict_probabilities_match", "status": "failed", "detail": str(exc)})
        return fail(
            f"repredict_mismatch_or_failure: {exc}",
            run_block=run_block,
            snapshot_block=snapshot_block,
            issue_block=issue_block,
            inputs_block=inventory,
        )
    stages["repredict"] = "completed"
    checks.append(
        {
            "check": "repredict_probabilities_match",
            "status": "passed",
            "detail": f"max_abs_difference={max_abs_difference!r}",
        }
    )
    repredict_block = {
        "method": "isolated_temp_directory_rerun_of_existing_m11_predict",
        "original_run_unmodified": True,
        "recomputed_files_not_a_new_version_of_original_run": True,
        "recomputed_predictions_sha256": recomputed_hash,
        "identity_match": True,
        "max_abs_probability_difference": max_abs_difference,
        "tolerance": _PROBABILITY_TOLERANCE,
    }

    inputs_unchanged = True
    for label, record in inventory.items():
        path = Path(record["path"])
        try:
            current_hash = file_sha256(path)
        except OSError as exc:
            stages["inputs_unchanged"] = "failed"
            checks.append(
                {"check": "inputs_unchanged_after_audit", "status": "failed", "detail": f"{label}: {exc}"}
            )
            return fail(
                f"input_verification_failed: {label}: {exc}",
                run_block=run_block,
                snapshot_block=snapshot_block,
                issue_block=issue_block,
                inputs_block=inventory,
            )
        if current_hash != record["sha256"]:
            inputs_unchanged = False
            checks.append(
                {
                    "check": "inputs_unchanged_after_audit",
                    "status": "failed",
                    "detail": f"{label} changed during the audit",
                }
            )
            break
    if inputs_unchanged:
        stages["inputs_unchanged"] = "completed"
        checks.append({"check": "inputs_unchanged_after_audit", "status": "passed", "detail": ""})
    else:
        return fail(
            "inputs_changed_during_audit: the audited inputs are no longer the audited bytes",
            run_block=run_block,
            snapshot_block=snapshot_block,
            issue_block=issue_block,
            inputs_block=inventory,
        )

    reasons: list[str] = []
    if issue_block["record_status"] == "missing":
        reasons.append("issue_evidence_missing")
    elif issue_block["record_status"] == "stale":
        reasons.extend(["issue_evidence_stale", *issue_block["record_reasons"]])
    elif issue_block["record_status"] == "unverifiable":
        reasons.extend(["issue_evidence_unverifiable", *issue_block["record_reasons"]])
    status = "blocked" if any(stage == "failed" for stage in stages.values()) else (
        "ready" if not reasons else "needs_review"
    )
    limitations = [*_BASE_LIMITATIONS, *issue_block["limitations"], AUDIT_SCOPE_TEXT]
    audit: dict[str, Any] = {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "command": audit_command,
        "generated_at": audited_at,
        "audited_at": audited_at,
        "validation_scope": "single_model_m11_snapshot_run_computation_audit_no_history_no_network",
        "status": status,
        "reasons": reasons,
        "stages": stages,
        "run": run_block,
        "snapshot": snapshot_block,
        "engine_checks": checks,
        "repredict": repredict_block,
        "issue_evidence": issue_block,
        "inputs": inventory,
        "inputs_unchanged": inputs_unchanged,
        "predictions_usable": status != "blocked",
        "limitations": limitations,
        "outputs": {
            "audit_json": str(output_dir / "prediction_audit.json"),
            "audit_report": str(output_dir / "prediction_audit.md"),
        },
    }
    _write_audit_files(audit, output_dir)
    return audit


def verify_snapshot_run(
    run_dir: str | Path,
    *,
    output_root: str | Path,
    issue_evidence_path: str | Path | None = None,
) -> dict[str, Any]:
    """Audit one existing single-model M11 prediction run and write a new audit directory.

    Output isolation is validated first; the unique audit directory is created
    before input validation so that expected input/IO failures leave structured
    blocked JSON/report artifacts under the safe writable output.
    """

    run_dir_path = Path(run_dir).expanduser().resolve()
    issue_file = (
        Path(issue_evidence_path).expanduser().resolve() if issue_evidence_path is not None else None
    )
    output_root_path = ensure_safe_output_root(
        output_root,
        [run_dir_path, *([issue_file] if issue_file is not None else []), *_preflight_recorded_paths(run_dir_path)],
    )
    audited_at = utc_now()
    audit_dir = new_output_dir(output_root_path, AUDIT_DIR_PREFIX)
    if not run_dir_path.is_dir():
        audit = _blocked_audit(
            run_dir=run_dir_path,
            audit_command="matchquant snapshot-run-verify",
            audited_at=audited_at,
            output_dir=audit_dir,
            reasons=[f"run_directory_does_not_exist: {run_dir_path}"],
            stages={
                "evidence_load": "not_run",
                "issue_evidence": "not_run",
                "input_inventory": "not_run",
                "repredict": "not_run",
                "inputs_unchanged": "not_run",
            },
            checks=[
                {
                    "check": "prediction_evidence_strict_load",
                    "status": "not_run",
                    "detail": f"prediction run directory does not exist: {run_dir_path}",
                }
            ],
            run_block={"run_dir": str(run_dir_path)},
            snapshot_block=None,
            issue_block=None,
            inputs_block={},
        )
        return {
            "status": audit["status"],
            "reasons": audit["reasons"],
            "audit_dir": str(audit_dir),
            "prediction_audit_path": str(audit_dir / "prediction_audit.json"),
            "prediction_audit_sha256": file_sha256(audit_dir / "prediction_audit.json"),
            "prediction_audit_report_path": str(audit_dir / "prediction_audit.md"),
            "run_dir": str(run_dir_path),
            "predictions_usable": audit["predictions_usable"],
            "manifest": audit,
        }
    audit = execute_snapshot_run_audit(
        run_dir=run_dir_path,
        issue_evidence_path=issue_file,
        audited_at=audited_at,
        output_dir=audit_dir,
        audit_command="matchquant snapshot-run-verify",
    )
    audit_json_path = audit_dir / "prediction_audit.json"
    return {
        "status": audit["status"],
        "reasons": audit["reasons"],
        "audit_dir": str(audit_dir),
        "prediction_audit_path": str(audit_json_path),
        "prediction_audit_sha256": file_sha256(audit_json_path),
        "prediction_audit_report_path": str(audit_dir / "prediction_audit.md"),
        "run_dir": str(run_dir_path),
        "predictions_usable": audit["predictions_usable"],
        "manifest": audit,
    }


__all__ = ["execute_snapshot_run_audit", "verify_snapshot_run"]
