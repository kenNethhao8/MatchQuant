# M10 Local Post-match Session Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an explicit offline M10 post-match command that stages local results, reviews/apply-gates one M2.1 canonical update, and settles the exact M9 forecast without hidden data activation.

**Architecture:** `postmatch_session.py` sequences and gates M7/M2.1/M6. `postmatch_session_support.py` owns M9/pointer evidence validation, immutable file records, unique failure-artifact discovery, and M10 persistence. `postmatch_session_report.py` renders a Chinese report from the M10 manifest only. Existing M7, M2.1, and M6 functions remain authoritative; M10 neither reimplements their validation nor calls their network paths.

**Tech Stack:** Python 3, pandas, numpy, Python standard library, existing MatchQuant JSON/CSV helpers, unittest.

**Spec:** `docs/superpowers/specs/2026-09-06-m10-postmatch-session-design.md`

## Global Constraints

- Preserve M1 schema, M7/M2.1/M6/M9 interfaces, chronology rules, source manifests, and existing tests.
- Use no dependencies beyond Python, pandas, and numpy; keep each Python file under 400 lines and ideally under 300.
- M10 is offline-only: every `sync_epl` call must receive an explicit staging directory; never call the download path.
- Inputs are explicit: no automatic latest forecast/history/model/staging lookup and no inferred in-progress season.
- Default mode is dry run. `--apply` requires an explicit existing staging directory and may be the only route to an active-pointer change.
- Pass the exact M9 history and processed manifest to M2.1; before apply, require active pointer canonical and manifest hashes to match that baseline.
- Keep third-party/raw data under `D:\Codex\matchquant_data`, run/session artifacts under `D:\Codex\matchquant_runs`, and never write M10 output inside the data root.
- Preserve all failures and child artifacts. Never roll back applied data or overwrite/move input files, pointers, raw snapshots, forecasts, or manifests.
- Unit tests remain offline; concise CLI output is 20 lines or fewer.
- Do not merge or push. Commit only the feature branch after tests pass and wait for controller audit.

---

### Task 1: Define the M10 contract with isolated red tests

**Files:**
- Create: `tests/test_postmatch_session.py`
- Inspect: `tests/test_forecast_session.py`, `tests/test_epl_local_stage.py`, `tests/test_epl_sync.py`, `tests/test_settlement.py`

**Interfaces:**
- Consumes: public `run_forecast_session`, `stage_local_epl_source`, `sync_epl`, and `settle_forecast_run`.
- Produces: reusable M9-ready synthetic setup and expectations for `run_postmatch_session` plus CLI `postmatch-session`.

- [ ] **Step 1: Build one self-contained M9-ready data root.**

Create helpers that generate a canonical historical baseline, matching M2.1 processed manifest, active current pointer, an M9 session with one future fixture, and a completed E0 result file for that exact fixture. Use only temporary directories and local example data. The helper must return exact paths rather than globbing for a latest run.

```python
def build_ready_session(root: Path) -> dict[str, Path]:
    data_root = root / "data"
    runs_root = root / "runs"
    history, history_manifest, pointer = write_active_baseline(data_root)
    session = run_forecast_session(
        history, history_manifest, write_pending_e0(root / "input" / "E0_pending.csv"),
        "2025-26", "2025-08-02", model_name="frequency",
        model_config_path=write_frequency_config(root / "input" / "frequency.json"),
        fixture_data_root=data_root, output_root=runs_root,
    )
    return {
        "data_root": data_root,
        "runs_root": runs_root,
        "session_manifest": Path(session["session_manifest_path"]),
        "pointer": pointer,
        "results_source": write_completed_e0(root / "input" / "E0_results.csv"),
    }
```

- [ ] **Step 2: Write the dry-run behavior test before implementation.**

```python
result = run_postmatch_session(
    paths["session_manifest"], "2025-26",
    results_source_csv=paths["results_source"],
    in_progress_season="2025-26", data_root=paths["data_root"],
    output_root=paths["runs_root"],
)
self.assertEqual(result["status"], "needs_review")
self.assertEqual(result["stages"]["local_stage"]["status"], "completed")
self.assertEqual(result["stages"]["sync"]["status"], "dry_run")
self.assertEqual(result["stages"]["settlement"]["status"], "not_run")
self.assertTrue(result["apply_eligibility"]["eligible"])
self.assertEqual(file_sha256(paths["pointer"]), pointer_before)
self.assertFalse((paths["data_root"] / "processed" / "versions").exists())
```

- [ ] **Step 3: Write the separate apply/settle test before implementation.**

Use the exact staging directory returned by the dry-run call. Do not restage or rediscover it.

```python
applied = run_postmatch_session(
    paths["session_manifest"], "2025-26",
    staging_dir=dry_run["staging_dir"], apply=True,
    in_progress_season="2025-26", data_root=paths["data_root"],
    output_root=paths["runs_root"],
)
self.assertEqual(applied["stages"]["sync"]["status"], "applied")
self.assertIn(applied["stages"]["settlement"]["status"], {"complete", "partial"})
self.assertTrue(Path(applied["settlement_run_dir"]).is_dir())
self.assertNotEqual(file_sha256(paths["pointer"]), pointer_before)
```

- [ ] **Step 4: Add safety and failure tests before implementation.**

Cover all of the following exact cases:

```python
# direct raw source may stage/dry-run but can never apply in the same call
with self.assertRaisesRegex(ValueError, "--apply requires --staging-dir"):
    run_postmatch_session(
        paths["session_manifest"], "2025-26",
        results_source_csv=paths["results_source"], apply=True,
        data_root=paths["data_root"], output_root=paths["runs_root"],
    )

# tampered M9 history/prediction artifact or blocked M9 session halts before M7/M2.1 apply
self.assertEqual(result["status"], "blocked")
self.assertEqual(result["stages"]["sync"]["status"], "not_run")

# pointer mismatch blocks before apply and leaves its bytes unchanged
self.assertEqual(result["stages"]["settlement"]["status"], "not_run")

# a revised or missing M2.1 candidate blocks apply and never calls M6
# malformed E0 source retains M7 failure evidence; no M2.1/M6
# M9 needs_review remains an M10 needs_review after otherwise successful settlement
# output inside data root or an M9/M10 session is rejected before session creation
# two partial M6 children are recorded ambiguous, never bound by sorted-last guessing
```

- [ ] **Step 5: Run the focused file to establish RED.**

Run: `python -m unittest discover -s tests -p 'test_postmatch_session.py' -v`

Expected: import failure because M10 has not been implemented.

### Task 2: Add focused M10 evidence, M9 validation, and pointer coherence helpers

**Files:**
- Create: `matchquant/postmatch_session_support.py`
- Test: `tests/test_postmatch_session.py`

**Interfaces:**
- Consumes: `file_sha256`, `load_forecast_run`, standard-library JSON/path handling, and the M9 manifest layout.
- Produces: `POSTMATCH_SESSION_PREFIX`, `new_postmatch_session_dir`, `load_m9_evidence`, `active_pointer_evidence`, `discover_new_child`, `initial_manifest`, and `persist`.

- [ ] **Step 1: Implement exact file-record and path-boundary helpers.**

```python
def file_record(path: str | Path | None) -> dict[str, Any]:
    candidate = Path(path).expanduser().resolve() if path is not None else None
    return {
        "path": str(candidate) if candidate else None,
        "exists": bool(candidate and candidate.is_file()),
        "sha256": file_sha256(candidate) if candidate and candidate.is_file() else None,
    }

def within(child: Path, parent: Path) -> bool:
    return child == parent or parent in child.parents
```

Use `within` to reject only unsafe output nesting. Allow normal siblings under a common run root.

- [ ] **Step 2: Implement strict M9 evidence loading.**

```python
def load_m9_evidence(session_manifest_path: str | Path) -> dict[str, Any]:
    manifest = read_json(Path(session_manifest_path))
    if manifest.get("schema_version") != "m9-forecast-session-v1":
        raise DataValidationError("unsupported M9 session manifest schema")
    if manifest.get("status") not in {"ready", "needs_review"}:
        raise DataValidationError("M10 requires an M9 session with ready or needs_review status")
    forecast = manifest["stages"]["forecast"]
    if forecast.get("status") != "completed":
        raise DataValidationError("M9 forecast stage is not completed")
    # Verify recorded history/history-manifest/M4 config/prediction hashes before returning paths.
```

Call `load_forecast_run` after the M9 artifact/hash checks so M6 will receive exactly the recorded M4 run. Reject invalid/missing recorded paths rather than rebuilding choices from directory contents.

- [ ] **Step 3: Implement active-pointer coherence evidence.**

```python
def active_pointer_evidence(pointer_path: Path, baseline: dict[str, Any]) -> dict[str, Any]:
    pointer = read_json(pointer_path)
    canonical = resolve_pointer_path(pointer["canonical"], pointer_path.parent)
    processed_manifest = resolve_pointer_path(pointer["processed_manifest"], pointer_path.parent)
    canonical_match = file_sha256(canonical) == baseline["history"]["sha256"]
    manifest_match = file_sha256(processed_manifest) == baseline["history_manifest"]["sha256"]
    return {
        "status": "matched" if canonical_match and manifest_match else "mismatch",
        "pointer_path": str(pointer_path),
        "canonical": file_record(canonical),
        "processed_manifest": file_record(processed_manifest),
        "canonical_hash_matches_m9": canonical_match,
        "processed_manifest_hash_matches_m9": manifest_match,
    }
```

Require a valid M2.1 current-pointer schema and both hash matches before apply. Missing/malformed/mismatched pointers are explicit pre-apply blockers, not fallback triggers.

- [ ] **Step 4: Implement evidence persistence and safe partial-artifact discovery.**

```python
def discover_new_child(root: Path, prefix: str, before: set[Path]) -> tuple[Path | None, dict[str, Any]]:
    candidates = sorted(
        path for path in root.iterdir()
        if path.is_dir() and path.name.startswith(prefix) and path not in before
    ) if root.is_dir() else []
    if len(candidates) == 1:
        return candidates[0], {
            "status": "unique", "prefix": prefix,
            "candidates": [str(candidates[0].resolve())],
            "bound_path": str(candidates[0].resolve()),
        }
    return None, {
        "status": "ambiguous" if candidates else "none", "prefix": prefix,
        "candidates": [str(path.resolve()) for path in candidates], "bound_path": None,
    }
```

Create a unique M10 directory before M7. Persist `postmatch_manifest.json` plus `postmatch_report.md`; write the report before hashing it, then write the manifest. Do not use a generic `utils.py`, broad `except Exception`, or arbitrary last-directory selection.

- [ ] **Step 5: Run focused tests until these helpers are green.**

Run: `python -m unittest discover -s tests -p 'test_postmatch_session.py' -v`

Expected: helper validation tests pass; orchestration tests may still fail until Task 3.

- [ ] **Step 6: Commit the evidence boundary.**

```bash
git add matchquant/postmatch_session_support.py tests/test_postmatch_session.py
git commit -m "feat: add M10 post-match evidence guards"
```

### Task 3: Implement the dry-run/apply/settlement orchestration

**Files:**
- Create: `matchquant/postmatch_session.py`
- Modify: `matchquant/__init__.py`
- Test: `tests/test_postmatch_session.py`

**Interfaces:**
- Consumes: M10 support helpers plus public `stage_local_epl_source`, `sync_epl`, and `settle_forecast_run`.
- Produces: `POSTMATCH_SESSION_SCHEMA_VERSION`, `POSTMATCH_SESSION_STATUSES`, and `run_postmatch_session`.

- [ ] **Step 1: Enforce input/mode/output safety before creating artifacts.**

```python
if (results_source_csv is None) == (staging_dir is None):
    raise ValueError("provide exactly one of results_source_csv or staging_dir")
if apply and results_source_csv is not None:
    raise ValueError("--apply requires --staging-dir from a prior reviewed dry run")
if within(output_root_path, data_root_path):
    raise DataValidationError("M10 output-root must not be inside D:\\Codex\\matchquant_data")
if within(output_root_path, m9_session_dir) or any(
    parent.name.startswith(POSTMATCH_SESSION_PREFIX) for parent in (output_root_path, *output_root_path.parents)
):
    raise DataValidationError("M10 output-root must not be inside an input or existing post-match session directory")
```

Validate M9 evidence before any M7 write. When source mode is selected, require `season`; when staging mode is selected, validate the known source manifest season through M2.1 rather than guessing a file name.

- [ ] **Step 2: Create M10 evidence first and run M7 only for source mode.**

```python
session_dir = new_postmatch_session_dir(output_root_path)
manifest_path = session_dir / "postmatch_manifest.json"
report_path = session_dir / "postmatch_report.md"
manifest = initial_manifest(
    POSTMATCH_SESSION_SCHEMA_VERSION, session_dir, m9_evidence, season,
    results_source_csv, staging_dir, apply, in_progress_season,
    data_root_path, output_root_path, pointer_path,
)
persist(manifest, manifest_path, report_path)

if results_source_csv is not None:
    try:
        stage = stage_local_epl_source(
            results_source_csv, season, data_root_path,
            source_url=source_url, source_label=source_label,
            upstream_retrieved_at=upstream_retrieved_at,
        )
    except EXPECTED_STAGE_ERRORS as exc:
        record_m7_failure(manifest, exc, data_root_path)
        persist(manifest, manifest_path, report_path)
        return result(manifest, manifest_path, report_path)
    staging_path = Path(stage["snapshot_dir"])
else:
    staging_path = Path(staging_dir).expanduser().resolve()
```

Record hashes and known raw/manifest paths. For an M7 failure, parse only its declared manifest path or use a uniquely discoverable new snapshot; retain ambiguity rather than guessing.

- [ ] **Step 3: Always run M2.1 from the explicit staging directory in dry-run mode first.**

```python
sync_result = sync_epl(
    seasons=[season], staging_dir=staging_path, data_dir=data_root_path,
    apply=False, in_progress_season=in_progress_season,
    baseline_path=m9["history_path"],
    processed_manifest_path=m9["history_manifest_path"],
    current_pointer_path=pointer_path,
)
apply_blockers = [name for name in ("revised", "missing_from_candidate") if sync_result["diff"][name]]
if apply_blockers:
    manifest["stages"]["sync"].update(
        status="blocked", result_status="dry-run",
        error={"type": "DataValidationError", "message": f"M2.1 apply blockers: {', '.join(apply_blockers)}"},
    )
    manifest["status"] = "blocked"
    manifest["status_source"] = "M2.1 apply blockers"
    persist(manifest, manifest_path, report_path)
    return result(manifest, manifest_path, report_path)
```

Capture `candidate_sha256`, report paths/hashes, strict-380 policy, skipped unfinished rows, and diff counts. `sync_epl` exceptions must preserve partial known report paths and skip apply/M6. No network mock should be needed because `staging_dir` is always non-null.

- [ ] **Step 4: Make dry run terminal and apply mode deliberately forward-only.**

```python
if not apply:
    manifest["status"] = "needs_review"
    manifest["apply_eligibility"] = {"eligible": True, "reason": "explicit --apply with this staging directory is still required"}
    manifest["stages"]["settlement"]["status"] = "not_run"
    persist(manifest, manifest_path, report_path)
    return result(manifest, manifest_path, report_path)

pointer_evidence = active_pointer_evidence(pointer_path, m9)
if pointer_evidence["status"] != "matched":
    manifest["pointer_evidence"] = pointer_evidence
    manifest["stages"]["sync"].update(
        status="blocked", result_status="pre_apply_pointer_mismatch",
        error={"type": "DataValidationError", "message": "active pointer does not match the M9 baseline"},
    )
    manifest["status"] = "blocked"
    manifest["status_source"] = "active pointer mismatch"
    persist(manifest, manifest_path, report_path)
    return result(manifest, manifest_path, report_path)
applied = sync_epl(
    seasons=[season], staging_dir=staging_path, data_dir=data_root_path,
    apply=True, in_progress_season=in_progress_season,
    baseline_path=m9["history_path"],
    processed_manifest_path=m9["history_manifest_path"],
    current_pointer_path=pointer_path,
)
```

Do not apply if the preflight found a blocker. If the second `sync_epl` call returns `blocked`, record the result and do not settle. Never attempt a rollback.

- [ ] **Step 5: Settle only the M9-recorded forecast against the newly applied canonical output.**

```python
settlement = settle_forecast_run(
    m9["forecast_run_dir"],
    applied["pointer"]["canonical"],
    output_root=output_root_path,
)
manifest["status"] = (
    "needs_review" if m9["status"] == "needs_review" or settlement["status"] == "needs_review"
    else settlement["status"]
)
```

The only non-error final statuses are `complete`, `partial`, and `needs_review`. When M6 raises an expected error after apply, preserve the applied data version, set session `blocked`, record M6 evidence, and return without rollback.

- [ ] **Step 6: Export the narrow public API and run focused tests.**

```python
from .postmatch_session import (
    POSTMATCH_SESSION_SCHEMA_VERSION,
    POSTMATCH_SESSION_STATUSES,
    run_postmatch_session,
)
```

Run: `python -m unittest discover -s tests -p 'test_postmatch_session.py' -v`

Expected: dry-run, apply, pointer, failure, and invariance tests pass.

- [ ] **Step 7: Commit the orchestration.**

```bash
git add matchquant/postmatch_session.py matchquant/postmatch_session_support.py matchquant/__init__.py tests/test_postmatch_session.py
git commit -m "feat: add explicit M10 post-match session"
```

### Task 4: Render the report, wire the CLI, and document the complete local loop

**Files:**
- Create: `matchquant/postmatch_session_report.py`
- Modify: `matchquant/postmatch_session.py`
- Modify: `matchquant/cli.py`
- Modify: `README.md`
- Test: `tests/test_postmatch_session.py`

**Interfaces:**
- Consumes: only the M10 manifest for report rendering and explicit argparse fields for the CLI.
- Produces: `write_postmatch_session_report`, `matchquant postmatch-session`, and user-facing M10 documentation.

- [ ] **Step 1: Add a pure Chinese Markdown renderer.**

```python
def write_postmatch_session_report(manifest: Mapping[str, Any], path: Path) -> Path:
    lines = [
        "# MatchQuant M10 赛后会话报告",
        f"总体状态：`{manifest['status']}`",
        "M10 完全离线；默认只生成 M7/M2.1 审查证据，绝不自动激活 history 或结算。",
    ]
    # Render explicit M9/baseline/source/pointer evidence, M7/M2.1/M6 links,
    # apply eligibility, blockers, and no-profit/no-currentness disclaimer.
```

The renderer must not read or mutate child artifacts after orchestration. Show `--apply` consequence and a clear message if an apply already changed the pointer but M6 later failed.

- [ ] **Step 2: Add an explicit CLI parser and concise handler.**

```python
postmatch = subparsers.add_parser("postmatch-session", help="offline explicit M7 -> M2.1 -> M6 post-match session")
postmatch.add_argument("--session-manifest", type=Path, required=True)
source = postmatch.add_mutually_exclusive_group(required=True)
source.add_argument("--results-source", type=Path)
source.add_argument("--staging-dir", type=Path)
postmatch.add_argument("--season", required=True)
mode = postmatch.add_mutually_exclusive_group()
mode.add_argument("--apply", action="store_true")
mode.add_argument("--dry-run", action="store_true")
```

Add `--in-progress-season`, provenance options, `--data-root`, `--output-root`, and optional `--current-pointer`. Print no more than six status/path lines. Return `0` only for `complete` or `partial`; return `2` for `needs_review` or `blocked`.

- [ ] **Step 3: Add CLI smoke coverage.**

```python
completed = subprocess.run(
    [
        sys.executable, "-m", "matchquant", "postmatch-session",
        "--session-manifest", str(paths["session_manifest"]),
        "--results-source", str(paths["results_source"]),
        "--season", "2025-26", "--in-progress-season", "2025-26",
        "--data-root", str(paths["data_root"]),
        "--output-root", str(paths["runs_root"]),
    ],
    capture_output=True, text=True, check=False,
)
self.assertEqual(completed.returncode, 2)
self.assertIn("MatchQuant M10 post-match session needs_review", completed.stdout)
self.assertLessEqual(len(completed.stdout.strip().splitlines()), 20)
```

Add an apply CLI test using the dry-run staging path and assert it does not attempt a network call.

- [ ] **Step 4: Update README from M1-M9 to M1-M10.**

Document the two explicit commands, the no-direct-source-apply rule, pointer-coherence check, status mapping, artifact locations, forward-only failure behavior, and the complete pre-match (M9) -> post-match (M10) cycle. State that this still does not mean live data, calibrated probabilities, profitability, or a betting recommendation.

- [ ] **Step 5: Run complete verification and commit.**

Run: `python -m unittest discover -s tests -v`

Expected: all existing tests plus M10 pass offline.

Run: `python -m compileall -q matchquant tests`

Expected: exit 0.

Run: `python -m matchquant postmatch-session --help`

Expected: exit 0 and both input modes are visible.

Run: `git diff --check`

Expected: no whitespace errors.

```bash
git add README.md matchquant/postmatch_session.py matchquant/postmatch_session_support.py matchquant/postmatch_session_report.py matchquant/cli.py matchquant/__init__.py tests/test_postmatch_session.py
git commit -m "docs: document M10 local post-match workflow"
```

## Self-Review

- Spec coverage: Tasks 1-4 cover strict M9 linkage, M7 archival, M2.1 dry-run/apply gates, pointer coherence, M6 settlement, evidence, CLI, documentation, and regression tests.
- Placeholder scan: no task delegates safety or tests to an undefined later step; source/staging modes and apply semantics are explicit.
- Type consistency: the M10 public API, stage names, statuses, and artifact paths are defined before CLI/report tasks consume them.
- Scope check: M10 is one post-match orchestrator only; local static viewing, browser behavior, data download, models, and multi-league work remain later independent stages.

## Execution Handoff

This implementation will be executed in a user-owned Codex task on an app-managed worktree. The worker must read this plan, its M10 design spec, `AGENTS.md`, and `C:\\Users\\Administrator\\.Codex\\projects\\C--Users-Administrator\\memory\\feedback_code_constraints.md` before editing. Use TDD, keep files focused, commit only the feature branch, do not merge or push, and wait for controller audit.
