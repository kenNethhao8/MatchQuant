# M9 Local Forecast Session Design

## Goal

Let a local user create one auditable pre-match forecast session from explicit files in one command. M9 coordinates the already-approved M5 fixture snapshot, M4 forecast, and M8 release verification stages; it does not add a model, fetch data, update history, or settle results.

## User outcome

The user supplies one local E0-compatible fixture CSV, one explicit canonical history CSV and its M2/M2.1 processed manifest, an as-of boundary, and either M3 selection evidence or an explicit model configuration. M9 creates and links three immutable artifacts:

1. an M5 fixture snapshot under the explicit fixture data root;
2. an M4 forecast run under the explicit run root;
3. an M8 release audit under the explicit run root;
4. an M9 session directory under the explicit run root with a manifest and Chinese report linking the other artifacts.

The session status is exactly the M8 result when all three stages finish. `ready` means the existing M8 evidence checks passed; it never means current data, profitable betting, manual approval, or a live service. `needs_review` and `blocked` retain their existing M8 meanings.

## Command and explicit inputs

```powershell
python -m matchquant forecast-session `
  --history D:\Codex\matchquant_data\processed\epl_canonical.csv `
  --history-manifest D:\Codex\matchquant_data\processed\epl_processed_manifest.json `
  --fixture-source D:\YourLocalFiles\E0.csv `
  --season 2025-26 `
  --as-of 2026-09-06 `
  --selection-run-config D:\Codex\matchquant_runs\m3_select_20260906T000000Z_12345678\run_config.json
```

`--selection-run-config` is mutually exclusive with `--model` plus `--model-config`. Optional `--source-url` and `--source-label` are provenance text only. `--fixture-data-root` defaults to `D:\Codex\matchquant_data`; `--output-root` defaults to `D:\Codex\matchquant_runs`.

No argument may mean "latest". M9 never calls `epl-download`, `epl-sync`, `epl-local-stage`, or `settle`; therefore it cannot update `epl_current.json`, activate a history version, or import post-match results.

## Architecture

Create `matchquant/forecast_session.py` as the small orchestration boundary and `matchquant/forecast_session_report.py` as the renderer. Its public function is:

```python
def run_forecast_session(
    history_path: str | Path,
    history_manifest_path: str | Path,
    fixture_source_csv: str | Path,
    season: str,
    as_of: str | pd.Timestamp,
    *,
    selection_run_config_path: str | Path | None = None,
    model_name: str | None = None,
    model_config_path: str | Path | None = None,
    season_mode: str | None = None,
    source_url: str | None = None,
    source_label: str | None = None,
    fixture_data_root: str | Path = Path(r"D:\\Codex\\matchquant_data"),
    output_root: str | Path = Path(r"D:\\Codex\\matchquant_runs"),
) -> dict[str, Any]:
```

It calls existing public entry points in this exact order:

```text
prepare_epl_fixtures → run_forecast → verify_release
```

The M9 session directory is created first so that a failure can still leave one small, immutable session manifest and report. It records the explicit input paths and observed hashes, child artifact paths/hashes/statuses, exception text if a later stage cannot start, and no-mutation declarations. It links rather than copies predictions, raw source bytes, third-party data, or M4/M5/M8 reports.

## Failure and safety rules

- Reject an output root inside `D:\Codex\matchquant_data`, a history/manifest input directory, or a forecast-session directory. Never silently redirect output.
- M5 failure: retain M5's existing failure evidence, mark M9 `blocked`, skip M4/M8, and write a session report that names the halted stage.
- M4 failure: retain the M5 snapshot, mark M9 `blocked`, skip M8, and record the actual exception.
- M8 `needs_review` or `blocked`: preserve the M4 and M5 artifacts and report the same final status; do not downgrade or upgrade it.
- Do not delete child artifacts after a failure. They are forensic evidence.
- The only intentional M9 writes are a new M5 immutable fixture snapshot and new M4/M8/M9 immutable run artifacts. It must not overwrite input data, raw source files, history files, manifests, or pointers.

## Tests and documentation

Offline tests must cover a ready end-to-end synthetic session, a M8 needs-review propagation case, an invalid fixture source that stops before M4/M8, explicit-model and M3 selection modes, output-root safety, input invariance, and concise CLI output. README must show the M9 command, exact child-stage order, output locations, status meanings, and the post-match handoff to explicit M7/M2.1 plus M6 settlement.

## Out of scope

No browser UI, static site, database, scheduling, networking, history synchronization, odds, player/news factors, ML model, automatic selection, or final publication action belongs in M9. A local static viewer is a separate later product stage.
