# M10 Local Post-match Session Design

## Goal

Close the local post-match half of MatchQuant without weakening data ownership or M2.1's explicit-apply gate. M10 turns one reviewed M9 pre-match session plus a user-provided local E0 results export into an auditable M7 -> M2.1 -> M6 handoff. It is research workflow software, not a live data service, betting product, or automated data updater.

## User outcome

M9 already creates the pre-match evidence chain. M10 supplies the missing post-match chain:

1. archive one explicitly supplied local E0 results file through M7;
2. run M2.1 offline against the exact M9 history baseline and produce a reviewable candidate/diff;
3. only after a separate, explicit `--apply` invocation using that known staging snapshot, activate an addition-only version through the existing M2.1 gate;
4. settle the exact M9 forecast against the newly applied canonical version through M6;
5. create a small M10 manifest and Chinese report that link, rather than copy, every child artifact.

The first invocation is a preflight by default. It creates M7/M2.1 review evidence but never changes `epl_current.json`, an active pointer, or history versions. The second invocation may apply only when the user explicitly supplies `--apply` and an existing staging directory. M10 must never turn a raw results file directly into an applied version in one command.

## Command and explicit inputs

```powershell
# 1. Stage and review only: no active data change and no settlement
python -m matchquant postmatch-session `
  --session-manifest D:\Codex\matchquant_runs\m9_forecast_session_<...>\session_manifest.json `
  --results-source D:\YourLocalFiles\E0_results.csv `
  --season 2025-26 `
  --in-progress-season 2025-26 `
  --data-root D:\Codex\matchquant_data `
  --output-root D:\Codex\matchquant_runs

# 2. After reviewing the M10/M2.1 evidence, explicitly apply and settle
python -m matchquant postmatch-session `
  --session-manifest D:\Codex\matchquant_runs\m9_forecast_session_<...>\session_manifest.json `
  --staging-dir D:\Codex\matchquant_data\staging\epl_local_<...> `
  --season 2025-26 `
  --in-progress-season 2025-26 `
  --apply `
  --data-root D:\Codex\matchquant_data `
  --output-root D:\Codex\matchquant_runs
```

`--results-source` and `--staging-dir` are mutually exclusive and exactly one is required. `--season` is always explicit. `--apply` is rejected with `--results-source`; it requires `--staging-dir` so that the user has a stable, inspectable M7 snapshot before any history activation. `--dry-run` is an optional explicit spelling of the default preflight mode. `--in-progress-season` is passed unchanged to M2.1 and is never inferred from the date.

The M9 session manifest is the source of the forecast run, baseline canonical CSV, and baseline processed manifest. M10 does not search for a latest forecast, history version, staging directory, or model. Optional source URL/label/upstream timestamp are provenance text for a fresh M7 stage only and never trigger network access.

## Architecture

Add a small `postmatch_session.py` orchestration boundary, a focused `postmatch_session_support.py` evidence/persistence module, and a pure `postmatch_session_report.py` renderer. The public API is:

```python
def run_postmatch_session(
    session_manifest_path: str | Path,
    season: str,
    *,
    results_source_csv: str | Path | None = None,
    staging_dir: str | Path | None = None,
    apply: bool = False,
    in_progress_season: str | None = None,
    source_url: str | None = None,
    source_label: str | None = None,
    upstream_retrieved_at: str | None = None,
    data_root: str | Path = Path(r"D:\\Codex\\matchquant_data"),
    output_root: str | Path = Path(r"D:\\Codex\\matchquant_runs"),
    current_pointer_path: str | Path | None = None,
) -> dict[str, Any]:
```

The sequence is fixed:

```text
read/validate M9 -> M7 local stage (only with --results-source) -> M2.1 dry-run
  -> [explicit --apply only: active-pointer coherence check -> M2.1 apply -> M6 settle]
```

M10 calls existing public `stage_local_epl_source`, `sync_epl`, and `settle_forecast_run` functions. They remain the authority for raw archival, canonical conversion, source quality, addition-only activation, and forecast/result matching. M10 must pass `staging_dir` to every `sync_epl` call; it must never invoke the network-download path.

Before M2.1, M10 validates the M9 manifest schema and hashes for the recorded history, history manifest, M4 run config, and predictions. Only M9 sessions with a completed M4 stage and a final M9 status of `ready` or `needs_review` may continue. `blocked` M9 sessions never activate history or settle. M10 records a pre-existing M9 `needs_review` condition but never upgrades it to ready.

M10 passes the exact history and history manifest recorded by M9 as `baseline_path` and `processed_manifest_path` to M2.1. For `--apply`, it additionally reads the explicit/default current pointer and requires both its canonical SHA-256 and processed-manifest SHA-256 to match the M9 baseline. This prevents activating a candidate built from a stale or side-branch baseline. If the pointer is missing, malformed, or mismatched, M10 stops before M2.1 apply and M6.

## Evidence, statuses, and writes

M10 first creates a unique `m10_postmatch_session_<UTC>_<ID>` directory under the explicit output root. It contains only `postmatch_manifest.json` and a Chinese `postmatch_report.md`; child artifacts remain in their owning M7/M2.1/M6 locations. The manifest stores explicit input paths and hashes, derived M9 links, stage states, source provenance, pointer-coherence evidence, M2.1 counts/eligibility, M6 metrics references, and no-mutation declarations.

M10 overall statuses are:

- `needs_review`: a successful dry run awaiting the separate explicit `--apply`, a carried M9 `needs_review`, or an M6 `needs_review` outcome. The report states whether M2.1 is addition-only eligible.
- `complete`: M2.1 applied and M6 settled every forecast row without ambiguity/time conflict, with no carried M9 review condition.
- `partial`: M2.1 applied and M6 found unresolved forecast rows but no ambiguity/time conflict, with no carried M9 review condition.
- `blocked`: invalid/tampered inputs, M7/M2.1 failures, an M2.1 revised/missing apply blocker, pointer incoherence, or an expected M6 failure.

M10 never rolls back an applied history version. If M6 fails after a successful apply, it records `blocked`, preserves the newly activated M2.1 version and all session evidence, and tells the user to resolve settlement separately. Historical versions and pointers are forward-only evidence, not rollback targets.

Expected user/input/data exceptions may be captured into the session evidence. Unexpected programming exceptions must propagate. On M7 failure, retain its declared failure manifest when available; never guess a sibling artifact. On M2.1/M6 failure, record known deterministic paths or bind a partial new run only when there is exactly one candidate. Multiple candidates are reported as ambiguous and never arbitrarily selected.

## Safety rules

- Fully offline: M10 never calls `epl-download`, `forecast-session`, `forecast`, `epl-fixtures-prepare`, a browser, a scheduler, or any network function.
- M2.1 remains the only component that can update `processed/versions` and the active pointer, and it is reached only via explicit `--apply` plus an explicit existing staging directory.
- `output_root` must not be inside `D:\Codex\matchquant_data`, the M9 session directory, or an existing M10 session directory. A normal sibling under `D:\Codex\matchquant_runs` remains valid.
- Never overwrite/move the M9 manifest, M4 forecast, input results CSV, raw M7 copy, baseline history, baseline manifest, existing pointer, or child reports. The normal M7/M2.1/M6 new-artifact writes are the only intended mutations.
- No hidden "latest" resolution, date inference, team-alias guessing, auto model selection, or source URL inference.
- M10 does not claim data currentness, calibrated probabilities, profitability, betting value, or future performance.

## Tests and documentation

Offline tests must create a synthetic M9-ready session and a matching local completed E0 result file. Cover:

1. source-mode dry run: M7 + M2.1 evidence exists, no pointer/version change, M6 remains `not_run`, and M10 is `needs_review` with apply eligibility;
2. staging-mode explicit apply: pointer coherence succeeds, M2.1 applies, M6 settles the exact M9 forecast, and a complete/partial result links every artifact;
3. M9 hash/status validation, malformed raw results, M2.1 revision/missing blockers, pointer mismatch, and M6 failure all preserve evidence and prevent the unsafe later stage;
4. M9 `needs_review` is retained instead of upgraded;
5. output-root guards, immutable input hashes, no network path, no arbitrary partial-artifact binding, concise CLI output, and no direct-source `--apply`.

README must explain the two-command post-match workflow, the intentional dry-run/apply separation, every status, data/pointer consequences, and the difference between completion of evaluation and any claim about predictive or betting performance.

## Out of scope

No static or hosted UI, database, web server, scheduler, data downloader, automatic polling, multi-league expansion, odds/news/player factors, model training, calibration changes, automatic data repair, GitHub data publication, or new prediction model belongs in M10. A local static session viewer may follow only after this post-match loop is safely complete.
