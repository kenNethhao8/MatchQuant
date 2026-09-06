# M9 Local Forecast Session Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add one explicit, offline M9 command that coordinates M5 fixture preparation, M4 forecasting, and M8 release verification into a traceable local pre-match session.

**Architecture:** `forecast_session.py` owns sequencing, status propagation, output-root safety, and the M9 session manifest. `forecast_session_report.py` only renders the Chinese report. Existing M5, M4, and M8 functions remain the authoritative validators and artifact creators; the CLI only translates explicit arguments and presents compact output.

**Tech Stack:** Python 3, pandas, numpy, existing MatchQuant JSON/CSV helpers, unittest.

**Spec:** `docs/superpowers/specs/2026-09-06-m9-local-forecast-session-design.md`

## Global Constraints

- Preserve M1 canonical schema, M5/M4/M8 interfaces, chronology rules, and existing tests.
- Use no dependencies beyond Python, pandas, and numpy.
- Offline only: do not call download, sync, local-stage, or settlement commands from M9.
- Every input path and model choice is explicit; no automatic latest-file discovery or history activation.
- Store M5 snapshots only in the explicit fixture data root and M4/M8/M9 run artifacts only in the explicit output root.
- Never overwrite or delete raw inputs, history, manifests, pointers, or child artifacts after a failure.
- Do not merge or push; commit only the feature branch after all tests pass.
- Guard correction: `output-root` is rejected when it is inside `D:\Codex\matchquant_data` or the explicit `fixture-data-root`, when the two roots overlap, when it equals a history/manifest/fixture-source parent directory, or when it is inside an existing `m9_forecast_session_*` directory. A normal sibling layout such as `<temp>\inputs` plus `<temp>\runs` remains valid; rejecting every descendant of the source file's parent would incorrectly reject that isolated layout.

---

### Task 1: Define the M9 contract with failing tests

**Files:**
- Create: `tests/test_forecast_session.py`
- Inspect: `tests/test_release_verify.py`, `tests/test_epl_fixtures.py`, `tests/test_forecast.py`

**Interfaces:**
- Consumes: existing `prepare_epl_fixtures`, `run_forecast`, and `verify_release` public functions.
- Produces: test expectations for public `run_forecast_session` and CLI `forecast-session`.

- [ ] **Step 1: Write a reusable synthetic-session fixture.**

```python
def build_inputs(root: Path, completed_date: str = "06/01/2024") -> dict[str, Path]:
    source = root / "local_e0.csv"
    source.write_text(
        Path("examples/synthetic_e0_source.csv").read_text(encoding="utf-8").replace(
            "08/01/2024,B,C,1,0", f"{completed_date},B,C,1,0"
        ),
        encoding="utf-8",
    )
    history = Path("examples/synthetic_matches.csv").resolve()
    history_manifest = root / "history_manifest.json"
    write_json(valid_m2_1_manifest(history), history_manifest)
    model_config = root / "frequency.json"
    model_config.write_text('{"alpha": 1.0}\n', encoding="utf-8")
    return {"history": history, "history_manifest": history_manifest, "source": source, "model": model_config}

def valid_m2_1_manifest(history: Path) -> dict[str, object]:
    frame = load_matches(history)
    return {
        "schema_version": "m2.1-processed-manifest-v1",
        "dataset_version": "m2.1-test-session",
        "data_cutoff": str(frame["date"].max()),
        "output": {
            "filename": str(history), "sha256": file_sha256(history),
            "rows": len(frame), "columns": list(REQUIRED_COLUMNS),
        },
    }
```

- [ ] **Step 2: Add failing ready-session test.**

```python
result = run_forecast_session(
    paths["history"], paths["history_manifest"], paths["source"], "2024-25", "2024-01-07",
    model_name="frequency", model_config_path=paths["model"],
    fixture_data_root=root / "fixture-data", output_root=root / "runs",
)
assert result["status"] == "ready"
assert Path(result["session_manifest_path"]).is_file()
assert Path(result["fixture_manifest_path"]).is_file()
assert Path(result["forecast_run_dir"]).is_dir()
assert Path(result["release_dir"]).is_dir()
```

- [ ] **Step 3: Add failing propagation and safety tests.**

```python
# completed cutoff after as_of propagates M8 needs_review
assert result["status"] == "needs_review"

# malformed E0 source blocks before M4/M8 and writes an M9 session report
assert result["status"] == "blocked"
assert result["stages"]["forecast"]["status"] == "not_run"

# output under matchquant_data is rejected before artifacts are written
with self.assertRaises(DataValidationError):
    run_forecast_session(
        paths["history"], paths["history_manifest"], paths["source"], "2024-25", "2024-01-07",
        model_name="frequency", model_config_path=paths["model"],
        fixture_data_root=data_root, output_root=data_root,
    )
```

- [ ] **Step 4: Run the focused test file to establish RED.**

Run: `python -m unittest discover -s tests -p 'test_forecast_session.py' -v`

Expected: failure because `matchquant.forecast_session` is not implemented.

### Task 2: Implement the orchestration and immutable M9 manifest

**Files:**
- Create: `matchquant/forecast_session.py`
- Modify: `matchquant/__init__.py`
- Test: `tests/test_forecast_session.py`

**Interfaces:**
- Consumes: existing M5 `prepare_epl_fixtures`, M4 `run_forecast`, M8 `verify_release`, `file_sha256`, and `write_json`; the exact child calls appear in Step 3 below.
- Produces: `FORECAST_SESSION_SCHEMA_VERSION`, `FORECAST_SESSION_STATUSES`, `run_forecast_session`, and `new_forecast_session_dir`.

- [ ] **Step 1: Add output-root and choice validation.**

```python
if selection_run_config_path is not None and model_name is not None:
    raise ValueError("selection run config and --model are mutually exclusive")
if selection_run_config_path is None and (model_name is None or model_config_path is None):
    raise ValueError("provide either selection evidence or explicit --model with --model-config")
if _within(output_root_path, data_root_path) or _within(output_root_path, fixture_data_root_path):
    raise DataValidationError("M9 output-root must not be inside matchquant_data or fixture-data-root")
if _within(fixture_data_root_path, output_root_path):
    raise DataValidationError("M9 output-root and fixture-data-root must not overlap")
if output_root_path in {history_file.parent, history_manifest.parent, fixture_source.parent}:
    raise DataValidationError("M9 output-root must not equal an input directory")
```

- [ ] **Step 2: Create a unique session directory before child stages.**

```python
session_dir = new_forecast_session_dir(output_root_path)
session_manifest = session_dir / "session_manifest.json"
session_report = session_dir / "session_report.md"
```

- [ ] **Step 3: Run M5, M4, then M8 with unchanged explicit inputs.**

```python
fixture = prepare_epl_fixtures(
    fixture_source_csv, season, fixture_data_root,
    source_url=source_url, source_label=source_label,
)
forecast = run_forecast(
    history_path, fixture["fixture_path"], as_of,
    selection_run_config_path=selection_run_config_path,
    model_name=model_name, model_config_path=model_config_path,
    season_mode=season_mode, output_root=output_root_path,
)
release = verify_release(forecast["run_dir"], history_path, history_manifest_path, fixture["manifest_path"], output_root=output_root_path)
```

- [ ] **Step 4: Persist stage records without masking errors.**

```python
stages = {
    "fixture_snapshot": {"status": fixture["status"], "manifest_path": fixture["manifest_path"], "sha256": fixture["source_sha256"]},
    "forecast": {"status": "completed", "run_dir": forecast["run_dir"], "predictions_sha256": forecast["config"]["predictions_sha256"]},
    "release_verification": {"status": release["status"], "run_dir": release["run_dir"], "manifest_path": release["release_manifest_path"]},
}
```

On a stage exception, write `status="blocked"`, the concrete exception, previous artifact paths, and `not_run` for later stages. Never remove successfully written prior artifacts.

- [ ] **Step 5: Export only the public M9 API.**

```python
from .forecast_session import (
    FORECAST_SESSION_SCHEMA_VERSION,
    FORECAST_SESSION_STATUSES,
    run_forecast_session,
)
```

- [ ] **Step 6: Run focused tests to establish GREEN.**

Run: `python -m unittest discover -s tests -p 'test_forecast_session.py' -v`

Expected: all M9 behavior tests pass.

### Task 3: Render the session report and wire the CLI

**Files:**
- Create: `matchquant/forecast_session_report.py`
- Modify: `matchquant/forecast_session.py`
- Modify: `matchquant/cli.py`
- Test: `tests/test_forecast_session.py`

**Interfaces:**
- Consumes: M9 manifest data only; it must not read or mutate child artifacts after orchestration.
- Produces: `write_forecast_session_report(manifest, path)` and `matchquant forecast-session`.

- [ ] **Step 1: Render a compact Chinese session report.**

```python
lines = [
    "# MatchQuant M9 本地预测会话报告",
    f"总体状态：`{manifest['status']}`",
    "M9 未更新历史、不联网、不执行人工发布，也不代表实时性或投注建议。",
]
```

Include explicit history/fixture paths and hashes, M5/M4/M8 artifact links, M8 status, and a post-match instruction that uses explicit M7/M2.1 plus M6 rather than M9.

- [ ] **Step 2: Add CLI parser arguments without hidden defaults.**

```python
session = subparsers.add_parser("forecast-session", help="offline explicit M5→M4→M8 pre-match session")
session.add_argument("--history", type=Path, required=True)
session.add_argument("--history-manifest", type=Path, required=True)
session.add_argument("--fixture-source", type=Path, required=True)
session.add_argument("--season", required=True)
session.add_argument("--as-of", required=True)
```

Mirror M4's mutually exclusive selection/model arguments, plus explicit fixture data root, output root, and optional provenance fields.

- [ ] **Step 3: Keep CLI output concise and status-bearing.**

```python
print(f"MatchQuant M9 forecast session {result['status']}")
print(f"session_dir={result['session_dir']}")
print(f"release_report={result['release_report_path']}")
return 0 if result["status"] == "ready" else 2
```

- [ ] **Step 4: Add CLI smoke assertions.**

```python
completed = subprocess.run(
    [
        sys.executable, "-m", "matchquant", "forecast-session",
        "--history", str(paths["history"]), "--history-manifest", str(paths["history_manifest"]),
        "--fixture-source", str(paths["source"]), "--season", "2024-25", "--as-of", "2024-01-07",
        "--model", "frequency", "--model-config", str(paths["model"]),
        "--fixture-data-root", str(root / "fixture-data"), "--output-root", str(root / "runs"),
    ],
    capture_output=True, text=True, check=False,
)
assert completed.returncode == 0
assert len(completed.stdout.strip().splitlines()) <= 12
```

- [ ] **Step 5: Run focused tests.**

Run: `python -m unittest discover -s tests -p 'test_forecast_session.py' -v`

Expected: report and CLI smoke tests pass.

### Task 4: Document the first user-facing pre-match flow and verify regression safety

**Files:**
- Modify: `README.md`
- Test: `tests/test_forecast_session.py`, full `tests/`

**Interfaces:**
- Consumes: final CLI and manifest schema from Tasks 2–3.
- Produces: M9 usage and a clear separation between pre-match M9 and post-match M7/M2.1/M6.

- [ ] **Step 1: Add an M9 README section.**

Document the explicit command, artifact locations, M5→M4→M8 order, `ready`/`needs_review`/`blocked` meanings, the prohibition on auto-sync/network/current-pointer changes, and the manual post-match handoff.

- [ ] **Step 2: Add input-invariance assertions.**

```python
before = {"history": file_sha256(history), "source": file_sha256(fixture_source)}
result = run_forecast_session(
    paths["history"], paths["history_manifest"], paths["source"], "2024-25", "2024-01-07",
    model_name="frequency", model_config_path=paths["model"],
    fixture_data_root=fixture_data_root, output_root=root / "runs",
)
after = {"history": file_sha256(history), "source": file_sha256(fixture_source)}
assert before == after
assert not (fixture_data_root / "processed" / "epl_current.json").exists()
```

- [ ] **Step 3: Run the complete offline suite and diff check.**

Run: `python -m unittest discover -s tests -v`

Expected: all tests pass with M1–M9 behavior covered.

Run: `git diff --check`

Expected: no whitespace errors.

- [ ] **Step 4: Commit the implementation.**

```bash
git add README.md matchquant/__init__.py matchquant/cli.py matchquant/forecast_session.py matchquant/forecast_session_report.py tests/test_forecast_session.py
git commit -m "feat: add explicit M9 forecast session workflow"
```

## Self-Review

- Spec coverage: Tasks 1–4 cover explicit inputs, M5→M4→M8 sequencing, immutable session evidence, failure propagation, output safety, CLI, tests, and README.
- Placeholder scan: no unfinished implementation choices or generic validation steps remain; function names, artifact names, and status rules are specified.
- Type consistency: Task 2 defines the public session API and manifest fields used by Tasks 3–4; all child calls use existing public M5/M4/M8 signatures.

## Execution Handoff

This implementation will be executed in a user-owned Codex task on an app-managed worktree. The worker must read this plan and the M9 design spec before editing, use test-first development, commit only its feature branch, and wait for controller audit before any merge.
