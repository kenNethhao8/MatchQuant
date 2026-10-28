# MatchQuant: on-demand data synchronization and an Agent Skill workflow

Implementation status: local commit `a020607` was rechecked on 2026-10-27 and still has not passed this stage's acceptance requirements. The supplied handover is byte-identical to the previous one; no new implementation is present in this checkout. See [the current requirement checklist](AGENT_SYNC_REQUIREMENTS_CHECKLIST_20261027.md) and [the detailed acceptance review and repair gates](AGENT_SYNC_REVIEW_20261026.md). Live remote verification was unavailable. The requirements below remain unchanged.

## Task

Extend the existing MatchQuant project so that a user can configure it once, invoke a MatchQuant Skill in a host Agent, describe a fixture in natural language, and receive a Chinese match analysis grounded in freshly checked historical data and the existing MatchQuant models.

The host Agent handles conversation, parameter clarification, tool orchestration, explanation, and writing. MatchQuant handles source access, data validation, chronology, factors, model calculation, and authoritative numerical results. The host Agent's language model supplies the language capability; this stage does not require embedding a second LLM API inside MatchQuant.

Complete the implementation, meaningful offline tests, installation and CLI checks, documentation, an installable local Skill, and sample artifacts. Do not stop at a plan or an instruction-only Skill. This specification explicitly authorizes the on-demand synchronization and local Skill stage described below. Preserve the repository's existing model and chronology guarantees.

The final execution receipt must be in Chinese and distinguish implemented behavior, actually executed checks, and remaining limitations.

## Problem to solve

The current single-match workflow accepts an explicit local rich historical CSV. If that CSV ends in May 2026 and a user requests a fixture in 2027, the system does not obtain the intervening matches. The historical training archive can still contain many seasons, but recent team state remains frozen at the end of that file. A stale-data warning does not supply the missing matches.

The desired experience is:

1. The user installs MatchQuant and its Skill, then configures data locations, the provider, aliases, refresh preferences, and report preferences once.
2. The user invokes the Skill and asks for one fixture, for example: "分析2027年1月15日曼城主场对维拉，按1月10日之前的数据，给我标准版。" This is an illustrative future request, not a verified fixture.
3. The Agent resolves the request using explicit configuration and reliable fixture metadata when available, and asks only for material missing or ambiguous details.
4. The Agent checks local coverage, calls an explicit data command when refresh is authorized, validates the resulting dataset version, and calls the existing prediction workflow with an explicit rich CSV and cutoff.
5. MatchQuant returns authoritative predictions and evidence. The Agent writes a moderate-length Chinese analysis from those results and delivers the artifacts.
6. Follow-up questions reuse the same run unless a change in analytical inputs requires recomputation.

The Git repository must contain the provider adapter, configuration examples, schemas, Skill, tests, and documentation. Downloaded third-party datasets remain in the user's local data directory. A new user must be able to acquire the data from the configured provider rather than receive the developer's personal data cache through Git.

## Workspace, rules, and current implementation

- Repository: `D:\matchquant\worktree`; the branch was `quant-v2` when this specification was prepared. Inspect the actual checkout and working tree first.
- Read `QUANT_MASTER_CHARTER.md`, `AGENTS.md`, `README.md`, `pyproject.toml`, this specification, and the existing `docs/stage_specs/SINGLE_MATCH_REPORT_LUNA_PROMPT.md`.
- The earlier single-match specification deliberately excluded a chatbot and implicit network access. This task adds a separate Agent orchestration layer and explicit data synchronization. The existing offline `match-report` contract remains available and unchanged in purpose.
- Preferred development interpreter: `D:\Dev\Python314\python.exe`. Verify it and the actual dependencies. Use a working Python >=3.11 if necessary and report the interpreter used. Follow the repository's `PYTHONPATH` instructions during development.
- Inspect and reuse `matchquant/epl.py`, `rich_data.py`, `single_match.py`, `match_report.py`, `cli.py`, `launcher.py`, `io.py`, `fixtures.py`, `factor_engine.py`, `factor_lab.py`, and `quant_models.py`, plus their tests.
- A previous implementation added `match-report`, `launcher.py`, the SciPy dependency, and single-match tests. Recheck the current files; do not assume the older missing-launcher problem still exists.
- At preparation time, `epl-download` exists and uses the Football-Data.co.uk EPL E0 source. Its complete-season gate requires 380 matches, and its default season range ends at 2024-25. Explicit season URL construction is already reusable. These facts do not establish current upstream availability; verify the provider during the explicit network smoke check.
- `rich-prepare` currently reads local raw/staging files. `discover_raw_epl_files` can choose a staged file by path ordering. The new workflow must bind to an explicitly selected source manifest, not rely on accidental staging-directory precedence.
- `epl-sync` is described in repository rules, but no implementation was found in the checked source. Verify this before introducing a new command. Historical documents and directory names do not prove an implementation exists.
- The current `match-report` filters `date < as_of`, uses date-only start-of-day semantics, supports partial latest-season state, and reports stale history with `needs_review`. Preserve its existing tests and limitations.
- Ordinary source hashes and provenance JSON are appropriate. Do not introduce bundle, release-verification, notarization, hash-chain, session-lock, or generic workflow frameworks prohibited by the charter.
- Keep third-party raw/processed data under a configured external data root; on this development machine the normal root is `D:\Codex\matchquant_data`. Keep runs under a configured external output root, normally `D:\Codex\matchquant_runs`.
- Preserve unrelated modified/untracked files, including existing showcase scripts, archives, Site packages, and stage specifications. Do not reset or clean the checkout.
- Leave code locally reviewable. Do not push, publish a release, deploy a Site, or redistribute downloaded source data.

## Architecture and responsibility boundaries

Implement the following layers with small, explicit interfaces. Reuse existing functions where they already provide the required behavior. This is one end-to-end task, delivered in the milestone order below.

| Layer | Responsibility | Main output |
| --- | --- | --- |
| 0. User profile | Runtime paths, provider settings, aliases, refresh authorization/preferences, report preferences | Validated local profile |
| 1. Provider synchronization | Acquire requested league seasons, preserve raw snapshots, distinguish additions and corrections | Source manifest, quality report, diff |
| 2. Dataset preparation | Select exact source snapshots, normalize rich data, validate coverage, publish a usable version | Rich CSV and processed manifest |
| 3. Prediction service | Apply the cutoff, calculate factors and probabilities using existing models | Authoritative single-match JSON and baseline reports |
| 4. Agent adapter and Skill | Resolve intent, check readiness, invoke explicit tools, manage follow-ups and statuses | Structured tool response and reusable Skill |
| 5. Evidence-based writing | Produce Chinese explanation from the returned facts and evidence | Conversational answer and Agent report |

The Agent must not recreate the numerical models in a prompt. The backend must not depend on an LLM to download, validate, filter, fit, or predict. The deterministic baseline reports remain usable when no Agent narrative is requested.

## Layer 0: user profile and setup

Provide a small, versioned profile format and a practical setup/diagnostic entry. Choose one simple existing configuration format; do not add a configuration framework.

The profile should cover:

- Data root and report root, independent of the developer's drive letters.
- Python/runtime or executable selection when the host needs it.
- Provider identifier, competition, requested historical start season, and source-specific settings.
- Explicit latest/in-progress season declaration when needed. Do not infer the incomplete-season exception merely from row counts or a calendar-month heuristic.
- Alias mapping or its path, including user-supplied Chinese-to-provider team names.
- Refresh policy, including whether analysis may automatically refresh, a reasonable source-check interval, and whether a user explicitly permits a stale-data fallback.
- Timezone and the documented date-only cutoff policy.
- Report language, length/style, and default model profile identifier/version.

Separate numerical/model settings from writing preferences. Changing report length or tone must not change probabilities. Per-request analytical overrides must not silently rewrite saved preferences.

First-time setup checks the runtime/package, paths, profile validity, and provider configuration. Let the user explicitly opt into automatic refresh during setup. A saved opt-in authorizes later routine refresh/apply operations within its documented addition-only policy; do not ask again for every normal analysis. A request such as "更新数据后分析" can authorize that refresh for the current request. Store no credentials in tracked files; use environment variables or the host's existing credential mechanism if a future provider needs them.

Do not hardcode the latest season or the year 2027 into production logic. Determine season identity from explicit user/profile information or verified competition/fixture metadata. The existing engine must continue to receive an explicit season.

## Layer 1: explicit on-demand source synchronization

Add or complete the narrowly scoped `epl-sync` capability after verifying what already exists. Follow `AGENTS.md`: default to dry-run, keep each network snapshot in a new timestamped/unique staging directory, and apply through an explicit action.

Required behavior:

1. Preserve the existing provider and reuse its URL/date/identity helpers. Do not invent a paid API, credentials, or an alternative source merely to make the demo work.
2. Fetch explicit requested seasons with bounded timeouts and limited, justified retry behavior. Detect HTTP errors, HTML/error bodies presented as CSV, decoding/parser failures, and schema changes.
3. Bootstrap a new user's historical archive, and refresh missing/current seasons in an existing archive. Do not redownload every sealed historical season on every fixture request.
4. The provider may return a whole season CSV. Compare the fetched season locally using stable match IDs; a dedicated upstream incremental API is not required. Compare requested season partitions with their corresponding baseline partitions, while retaining unchanged cached seasons.
5. Record additions, changed existing records, missing baseline fixtures, unchanged records, skipped pending rows, and validation failures. Date/score/technical-stat/market-field changes must not disappear behind a timestamp-only "new rows" filter.
6. Preserve raw files as immutable snapshots. Do not overwrite a prior download or the original M2 canonical/manifest.
7. Automatic application is addition-only after all hard checks pass. Corrections to existing records, missing baseline fixtures, or invalid data block automatic promotion and produce a useful review report. Do not silently implement a force-overwrite correction path.
8. Dry-run can write its new staging snapshot and reports, but must not change the current usable-dataset pointer. A successful explicit apply atomically updates only the current-version pointer after the new dataset is fully prepared and validated. Failure leaves the previous pointer usable and unchanged.
9. Every apply response identifies the baseline and new dataset versions, source snapshots, hashes, row counts, actual data cutoff, and output paths.
10. `match-report` and `run` remain offline-only. Network activity occurs through the explicit data command/tool that the Agent orchestrates.

Completed-season validation must preserve the EPL gates: 380 unique completed fixtures, appropriate 20-team/home-away coverage, valid scores/teams/dates, and no duplicate identities.

Support one explicitly declared latest in-progress season with fewer than 380 completed matches. The actual command must receive the declaration, even when the Agent obtained it from saved configuration or verified fixture metadata. Record the declaration and whether the strict 380 gate was applied in every relevant manifest and report.

For that declared in-progress season, rows with valid identity/date fields and both full-time scores empty may be classified as pending and excluded from completed history, with counts recorded. Partial scores, malformed identities, invalid values, and duplicates remain failures. Distinguish a season that has not started from a provider/network failure; do not claim unavailable future results exist.

## Layer 2: explicit dataset versions, rich conversion, and freshness

Bind each prepared dataset to an explicit source manifest and exact per-season snapshots. Add a manifest-based preparation path or a similarly small explicit interface. Preserve existing `rich-prepare` behavior for compatibility, while making the new Agent workflow independent of implicit raw-file discovery order.

The prepared version must include the rich CSV required by single-match inference and ordinary provenance/quality records. Preserve the basic M1 seven-column canonical schema; do not add provenance columns to it. If both basic and rich outputs are maintained, their version relationship must be explicit.

Use `processed/epl_current.json` or the existing equivalent current pointer if present; inspect compatibility before choosing fields. This pointer is a reference to a validated immutable prepared version, not a second mutable dataset.

Rich conversion requirements:

- Convert only validated completed records; pending fixtures must never become 0-0 completed matches.
- Preserve unavailable technical statistics and quotes as missing. Existing rich-converter defaults must not turn absent shot/corner statistics into observed zeros. The engine may use documented internal defaults where already supported, but report evidence must retain valid observation counts.
- Preserve missing Asian-handicap quotes as missing. Keep Track A independent of opening/closing odds, steam, and other market inputs.
- Maintain stable identities, deterministic ordering, documented source date parsing, and contiguous season coverage required by the existing engine.
- Do not repeatedly fit models or run the full research backtest during data preparation.

Provide a readiness/freshness result before prediction. At minimum distinguish:

- `fetched_at`: when the source snapshot was acquired.
- `source_checked_at`: when an actual successful source check was made.
- `latest_completed_match_date`: newest observed completed match.
- `used_data_cutoff`: newest match actually used after the prediction cutoff filter.
- Declared seasons, completed/pending counts, missing coverage, provider failures, and review reasons.

A recent successful HTTP response does not prove complete data coverage. An old latest-match date during an off-season does not by itself prove download failure. If reliable schedule metadata exists, compare expected completed fixtures with observed results and account for provider publication delay. If it does not exist, report that completeness cannot be independently established. Keep the existing data-age disclosure; record source freshness and match-data age separately rather than silently relabeling a stale-history result as complete.

On a failed refresh, default to an actionable source/readiness failure rather than silently using the old dataset for a normal complete report. A user-authorized stale fallback may still run the existing model, but the result must prominently remain `needs_review` and identify the failed refresh and actual old cutoff.

## Temporal contract: current prediction and historical reconstruction

Preserve the existing conservative engine contract:

- Fixture date and `as_of` are explicit ISO dates.
- `as_of` means the start of that calendar day; use only completed rows with `date < as_of`.
- Exclude the entire cutoff day before Factor Lab, fitting, selection, state updates, or recent evidence is calculated.
- Require the fixture date to be on or after `as_of`.
- Do not add intraday precision without authoritative timestamps and a separate justified change.

For a future fixture, data may include only results already available at the request's cutoff. If today is January 10 and the fixture is January 15, do not include or simulate January 11-14 outcomes. Newer source rows can be downloaded but must be excluded from that inference.

Keep match date, prediction cutoff, retrieval time, and historical availability/vintage distinct. Downloading a finalized historical CSV today and filtering it by old match dates does not prove that later corrections or fields were available at an old prediction time.

Support honest provenance modes using the available archives:

- A current/reconstructed snapshot can be used for a date-filtered reconstruction, clearly labeled as such when historical availability cannot be verified.
- If a strict archived replay is requested, select an eligible archived snapshot acquired before the relevant cutoff and validate its identity. If none exists, report that strict vintage reproduction is unavailable; do not silently substitute a later snapshot or claim verified point-in-time data.

Replay must not automatically fetch current data and describe it as an original historical snapshot. A separately requested refresh may acquire data for a labeled reconstruction. Record the provenance mode and vintage limitations in the JSON and user-facing report.

## Layer 3: preserve and expose the existing prediction service

Reuse the single-match service and model family. Make only narrow refactors needed for a stable tool interface and evidence references; do not fork a second forecasting implementation.

Preserve:

- At least two complete training seasons and one complete validation season before the cutoff.
- Training-only factor discovery; validation exclusion from discovery; honest labeling of tuning/selection metrics.
- Latest partial season updates historical team state without entering complete-season fitting/selection partitions.
- Recalculation of partitions after new complete seasons become available, with state cutoff and fitting cutoff reported separately.
- Query state isolation, no completed target row, same-day freezing, and existing season-transition behavior.
- Track A market isolation, finite normalized 1X2 probabilities, expected goals, and over-2.5 probability.
- Optional user-supplied opening market input with its existing cutoff and schema checks. No target closing odds or steam.
- Existing deterministic HTML/Markdown/JSON reports and both direct/fixture-CSV CLI inputs.

Source synchronization must cover the relevant league history, not just the two target clubs, because table/league/opponent state uses other fixtures.

New-season team identities require explicit handling. Reuse the existing promoted-team/cold-start policy for verified or explicitly declared new league participants where that policy is actually supported. Do not bypass unknown-team guards for arbitrary names or typos. If a first-match promoted team cannot be supported with the available identity/roster information, return a precise limitation. Evidence with zero prior observations must remain unavailable, not invented. Do not introduce another league's data or new factors to hide this boundary.

## Layer 4: structured adapter and installable Skill

Provide a small adapter that the host Agent can invoke with structured inputs and read without parsing human-oriented terminal logs. A JSON request/response script or a package CLI with a machine-readable mode is sufficient. Keep stdout machine-readable and send diagnostics elsewhere. Use proper argument passing rather than assembling unsafe shell strings from user text.

The request covers the selected profile, fixture identity, explicit date/season/cutoff, mode, refresh policy/override, optional aliases/market input, and report preference. The response includes a schema version, normalized request, status, run ID, synchronization/readiness summary, warnings/errors, dataset manifest, authoritative model-result path, and absolute artifact paths.

Preserve existing `match-report` exit behavior. The new adapter must provide useful structured outcomes for complete results, review-required results, missing/invalid inputs, and provider failures. Do not turn a provider exception into an unhandled traceback or a seemingly successful complete prediction.

Package a local Skill in the repository, for example `skills/matchquant/`, with concise `SKILL.md`, appropriate UI metadata, and only the helper scripts/references actually needed. Follow the target host's current official Skill requirements. Include an explicit way to install/copy it into a chosen Skill directory and a setup/diagnostic invocation. Use a temporary target for installation verification; do not silently replace an existing unrelated global Skill or change global Agent settings.

The Skill must teach the Agent how to:

1. Activate for single-match analysis, related follow-up questions, and setup/diagnostic requests.
2. Read the selected profile and interpret Chinese aliases and relative-date expressions with the configured timezone.
3. Obtain exact fixture date, season, and cutoff. When "今晚" or "曼城那场" is ambiguous, use a configured reliable fixture resolver or ask one concise bundled question. Do not invent a schedule, current league membership, or season from model memory.
4. Use fixture selection metadata only if the configured provider actually supplies it. Do not add an unverified schedule scraper or mandatory paid API. Explicit user-provided fixture information remains a supported path and must be labeled appropriately.
5. Check readiness and invoke explicit refresh/dry-run/apply tools according to saved authorization. Routine validated addition-only updates may proceed under saved opt-in. Revisions and blockers must be surfaced with their concrete reports.
6. Invoke the offline prediction service with the selected prepared version and cutoff.
7. Read the returned JSON and produce an evidence-grounded analysis, artifacts, and a brief data-update receipt.
8. Reuse an existing run for explanation, a shorter report, or a different presentation. Rerun when the fixture, cutoff, input-data version, market input, or model settings change. Store the last run ID/path in the conversation context; do not put datasets into Agent memory.

The user should not need to know the developer's paths, run shell commands manually for every query, or configure a separate language-model API solely for this local Skill. Make runtime availability and provider setup requirements explicit during installation.

## Layer 5: Agent writing and consistent reports

The host Agent uses the authoritative JSON and supporting evidence to write the final Chinese analysis. Keep the original model JSON identifiable and unchanged as a source of numerical truth.

Provide a supported way to render an Agent-authored analysis into Markdown and standalone HTML in the same run, such as a small structured narrative payload consumed by the existing renderer or a narrow companion renderer. Keep the deterministic baseline report available. Clearly name the Agent report artifacts so users and programs can distinguish them from original computed facts.

Report requirements:

- Default to roughly one to two readable pages, approximately 800-1,200 Chinese narrative characters plus compact tables, with less text when evidence is sparse.
- Include the model conclusion, supporting recent evidence, material model disagreement/uncertainty, optional market comparison, data-update/cutoff receipt, and limitations relevant to this fixture.
- Render probability tables and other numeric facts programmatically from the model result, or resolve explicit references/slots to those fields. Do not make the Agent recalculate, round inconsistently, or alter values to suit a preferred tone.
- Give factual paragraphs references to evidence IDs or JSON field paths. These references aid review; do not claim they mechanically prove every semantic statement in free prose.
- Use only supplied evidence for form, shots, corners, rest, factors, and other match-specific statements. Missing evidence stays unavailable with its actual valid observation count.
- Preserve existing attribution limitations. An ordered-component local approximation is not a causal or exact ensemble decomposition; zero ordered blend weight must not become an explanation of the final ensemble.
- Do not invent injuries, lineups, news, odds, official fixture verification, causal claims, or certainty. This stage does not require a news/injury provider. If such information is outside the returned evidence, explain that it was not included when relevant.
- A request to change length, audience, or tone changes expression, not numerical predictions.
- HTML must escape dynamic text and remain self-contained. Markdown, HTML, and conversational numerical statements must agree with the same run JSON.

A deterministic rendering test can check references, slots, escaping, and consistency. It cannot prove a real LLM's interpretation is correct. Include a realistic Agent exercise and report honestly which parts were exercised through an actual host.

## Scope and implementation discipline

The first version is a local Skill, a small adapter, explicit on-demand synchronization, and the existing local engine. An MCP server can later expose the same service interfaces; no MCP server, separate web application, database, scheduler, background daemon, or multi-user service is required now.

Do not replace the current models, invent factors, tune to a sample probability, add a new benchmark project, or create a persistent-model/cache framework. A simple optimization is optional only when a measured issue justifies it; it must not weaken cutoff or version isolation.

Keep existing commands and tests working. Any change to validation/default handling must be supported by a regression test and documented. Avoid unnecessary dependencies and broad rewrites.

## Milestone order

Execute these milestones sequentially within this task. Produce reviewable results as you go, and continue through all milestones unless there is a concrete external blocker requiring user information.

1. Inspect the actual code and record the capability gaps, proposed file changes, and interface choices.
2. Implement source synchronization, in-progress-season handling, explicit dataset versions, rich preparation, and readiness checks.
3. Connect profile and structured adapter to the existing prediction service; preserve its offline contract and tests.
4. Package the Skill, setup/install path, evidence-based writing workflow, and Agent-report rendering.
5. Complete focused tests, installation checks, one actual network smoke using the configured existing provider when available, sample analyses, documentation, and the Chinese execution receipt.

An unavailable provider must produce a real recorded failure and meaningful fallback behavior. Do not mislabel mocked/synthetic results as a live successful source check. Continue useful offline implementation and validation when network access is unavailable, then state the precise unverified acceptance item.

## Meaningful tests and acceptance checks

Keep unit/integration tests offline with temporary directories and controlled provider fixtures. Do not depend on a live provider for the standard suite. Test observable behavior and invariants rather than fixed prose or one hardcoded model probability.

Cover at least these boundaries:

1. A new user can bootstrap explicit required seasons. An existing user can add completed fixtures/current-season data without overwriting raw snapshots.
2. Dry-run leaves the current pointer unchanged. Valid explicit addition-only apply promotes a fully prepared version atomically. HTTP/parser/quality/preparation failures, revisions, missing baseline fixtures, and partial scores leave the previous version unchanged.
3. In-progress-season declarations are explicit and recorded. Valid pending rows are counted/excluded. Strict complete-season gates remain in force for completed seasons.
4. Old cached seasons are retained when only a current-season partition is fetched. Exact snapshot selection is independent of unrelated staging-directory names or traversal order.
5. Missing technical/market fields remain missing through rich conversion and report evidence. Zero observed shots is distinct from unavailable shots. Missing AH does not become a neutral line.
6. Use a controlled clock and clearly synthetic 2027 fixtures/results to simulate an installation whose last data are May 2026. Refresh adds the missing seasons/results; analysis state uses newly eligible results and reports the new version/cutoff. Production logic must not be hardcoded to the test years.
7. Changing/appending cutoff-day or future rows cannot change the eligible model state, discovery, fitted partitions, predictions, or recent evidence. Source hashes/excluded-row counts may change; compare appropriate numerical invariants rather than entire provenance objects.
8. A future fixture never uses results between the current cutoff and its future kickoff. The pending fixture never updates state. Existing query-order and market-isolation tests still pass.
9. A newly complete season changes fitting/validation partitions through the existing policy; a partial current season changes state without entering fitting partitions. No random splits or validation-label factor discovery.
10. Source-check age and match-data age are distinct. Off-season/no-new-results cases are reported honestly. Refresh failure is not silently converted into a complete result from stale data; explicitly allowed stale fallback is prominent and `needs_review`.
11. Strict archived replay rejects an unavailable eligible vintage. Later-snapshot reconstruction is labeled honestly. Retrieval time is not silently substituted for per-observation historical availability.
12. Adapter results are parseable structured responses with statuses and real artifact paths. Profile overrides, missing fields, aliases, ambiguous fixtures, unsupported teams, and provider failures produce useful behavior.
13. Follow-up explanation/format changes reuse the existing run; changed analytical inputs trigger recomputation. Numerical fields in reports resolve to the same authoritative JSON.
14. Agent narrative rendering validates evidence references/slots, escapes HTML, preserves zero-sample/missing evidence and attribution limitations, and generates readable Chinese artifacts.
15. Existing module CLI and installed console entry still work. Skill packaging/setup/install works in a temporary target and uses configurable paths.

Include at least one real end-to-end model fit from an adequate offline multi-season dataset, not only mocked probabilities. Reuse appropriate existing test helpers and selection grids to keep tests practical.

Run focused tests first. After changes stabilize, run the full existing suite once:

```powershell
$env:PYTHONPATH = "D:\matchquant\worktree"
& "D:\Dev\Python314\python.exe" -m unittest discover -s tests -v
```

Perform syntax/import checks appropriate to changed files, `git diff --check`, and an installation smoke in a temporary virtual environment. An editable installation using existing site packages is acceptable when declared dependencies are available; identify that setup honestly. Verify version/help, installed sync/adapter invocation, existing `match-report`, and the Skill install/setup path. Do not modify the user's global Python installation to perform the smoke test.

## Samples and actual network check

Use the actual execution date, not a hardcoded year, to select an available current/most recent season for the explicit provider smoke. Inspect the provider's current data and record actual outcomes. Real 2027 results cannot be fetched before they exist; the 2027 acceptance scenario must use clearly marked synthetic fixtures and a simulated clock when 2027 is still in the future.

Produce these demonstrations outside tracked datasets:

1. A clearly labeled synthetic 2027 stale-cache-to-refreshed-state example with before/after coverage, dataset versions, and a consistent report. Synthetic provider input must be isolated from real data roots.
2. An actual existing-provider synchronization dry-run, followed by explicit apply only if it passes the addition-only/bootstrap policy. Record source, seasons, retrieval times, counts, quality/diff results, and resulting version. If the source cannot be reached, retain the actual failure record and mark live sync verification incomplete.
3. A real-data single-match report using the explicitly selected prepared version. Prefer a verifiable fixture; a historical reconstruction is acceptable with its correct vintage label. If a future fixture is hypothetical/user supplied, label it and do not claim an official schedule was verified.
4. A host-Agent exercise using the packaged Skill that analyzes one fixture and answers one evidence-based follow-up. If the host cannot dynamically install/discover a Skill in the current session, test a temporary installation and explicitly invoke the Skill by its path if supported. State the exact interaction actually exercised; do not claim automatic activation was tested merely because the files validate.

Preserve all prior raw snapshots and report directories. Use unique run locations and link the actual generated files. Do not use a saved Showcase probability as an inference result.

## Documentation and completion criteria

Update README and focused references so a new user can:

- Install the package and Skill on their own machine and run setup/diagnostics.
- Configure the provider, local roots, aliases, refresh authorization, and report preferences.
- Bootstrap data, inspect a sync dry-run, apply a valid update, and understand how corrections are blocked/reviewed.
- Request a fixture in natural language and understand which fields are required or resolved.
- See the distinction between source freshness, completed-match coverage, prediction cutoff, fitting cutoff, and historical vintage.
- Understand what happens on provider failure, stale data, an incomplete season, a new team, and an ambiguous fixture.
- Open and reproduce deterministic and Agent-authored reports, read their JSON evidence, and ask follow-up questions.
- Use the offline CLI independently of the Agent.

The task is complete when the layered workflow is usable, source acquisition code and the Skill are versioned in Git, a fresh installation can bootstrap/update data, a configured Agent can orchestrate analysis without repeated path/command input, all chronology/model-isolation guarantees remain intact, checks are actually executed, and sample artifacts demonstrate the behavior. Explicitly list any externally blocked acceptance item instead of claiming it passed.

In the final Chinese receipt provide:

1. The original problem and implemented layers in a brief explanation.
2. Changed/new file links and the actual public commands/tool request shape.
3. The interpreter/environment, test results, installation results, and actual network-check outcome.
4. Source/season/version/cutoff details for the samples, distinguishing real and synthetic evidence.
5. Links to generated HTML, Markdown, authoritative JSON, synchronization/quality/diff records, and the installable Skill.
6. A short copyable natural-language example for first setup, one analysis, and one follow-up.
7. Unresolved limitations, including historical vintage or fixture-source coverage when relevant.

Do not report an installation, source refresh, Agent activation, test pass, numerical result, or sample file that you did not actually execute or generate.
