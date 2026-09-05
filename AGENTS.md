# MatchQuant Project Rules

## Project purpose

MatchQuant is a local-first, reproducible research framework for football 1X2 probability forecasting and historical evaluation. It is research software, not a betting service or a live-prediction product.

## Scope and stage gates

- Preserve the approved M1 interface, chronology rules, and tests unless an explicit review approves a breaking change.
- M2 covers one reproducible EPL results-data source, local archival, canonical conversion, data-quality reporting, and a real-data baseline backtest.
- Build the explicit incremental-sync workflow only after the one-time M2 historical import has passed review. Do not add factor research, machine-learning models, player/news data, a web app, a database, or automated scheduling while completing M2.
- Each stage must finish with tests, a CLI smoke run, documented limitations, and a reviewable data/code version before the next stage begins.

## Local data ownership and Git rules

- Keep third-party raw and processed data under `D:\Codex\matchquant_data\`; keep backtest outputs under `D:\Codex\matchquant_runs\`.
- The Git repository contains code, schema, source definitions, manifests, hashes, configuration examples, and small synthetic fixtures. It must not contain downloaded third-party raw CSV files or large generated datasets by default.
- Do not push data or code to any remote, create a GitHub release, or publish a hosted artifact unless the user explicitly asks.
- Do not distribute third-party source files through GitHub until their redistribution terms have been independently verified and documented.
- Treat every downloaded raw file as immutable. Never silently overwrite it. A newer download is a new local snapshot identified by retrieval time and SHA-256.
- Store dataset provenance outside the canonical M1 match schema. A manifest must identify source URL, season, retrieval time, source-file hash, parser/format version, output hash, row count, and data cutoff.
- Long historical data belongs in versioned local files, not in agent memory. Agent memory may only retain pointers such as dataset version, cutoff, paths, hashes, and known limitations.

## Network and update rules

- `matchquant run` must be offline-only and consume an explicit local input dataset.
- Any network access must be triggered by an explicit data command. It must state the source, requested seasons, output location, and whether it will apply changes.
- An eventual sync command must use a staging copy, compare canonical match IDs and source hashes, report additions/revisions/skipped unfinished fixtures, and apply changes only through an explicit apply action.
- Do not claim the local data are current merely because a source is reachable. Report the local dataset cutoff and its version on every data-quality and backtest report.
- Do not silently skip unavailable, malformed, incomplete, duplicate, or corrected source rows. Fail clearly or record the issue in the quality report.

## Data and temporal integrity

- Keep the canonical M1 schema stable: `match_id`, `date`, `season`, `home_team`, `away_team`, `home_goals`, and `away_goals`.
- Assign season from the requested source season; never infer it from calendar months.
- Parse source dates with the documented source format. Preserve date-only semantics unless an authoritative kickoff time is available.
- Build stable match IDs deterministically. Reject duplicate IDs and report duplicate fixtures rather than guessing a merge.
- A completed EPL season should contain 380 matches. Report a count mismatch explicitly; do not silently treat it as a complete season.
- Every feature, model state update, and evaluation prediction may use only information available before the relevant match. Predict every same-day or same-timestamp batch before updating with any result from that batch.
- Never randomly shuffle chronological football data. Never select model parameters using a test period.

## Modeling and evaluation

- Return home-win, draw, and away-win probabilities that are finite, non-negative, and sum to one.
- Do not call valid probability arithmetic calibrated probability. Calibration requires separate evidence.
- Keep simple baselines available when adding later models. Compare all models on an identical chronological test set.
- Record model parameters, input-data hash, data version/cutoff, test boundary, and output paths for every run.
- Do not describe backtest metrics as proof of live profitability, betting value, or future performance.

## Engineering quality

- Keep the dependency set small. M1/M2 default to Python, pandas, and numpy unless a new dependency has a clear need and is documented.
- Unit tests must run offline. Network checks may be explicit smoke tests only.
- Validate new data adapters with fixture tests for field mapping, date parsing, invalid input, deterministic IDs, manifests, data-quality output, and canonical-schema compatibility.
- Keep CLI output concise and data-oriented. Save detailed evidence to files rather than printing long logs.
- Update README whenever commands, data layout, source assumptions, or reproducibility guarantees change.
