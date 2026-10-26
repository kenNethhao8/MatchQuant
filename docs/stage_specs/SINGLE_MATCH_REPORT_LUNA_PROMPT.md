# MatchQuant: single-match prediction and evidence-based report

## Task

Implement a complete local single-match research workflow in the existing MatchQuant repository. A user who has installed the project and prepared a local historical dataset must be able to specify two teams, a fixture date, a season, and a prediction cutoff, then receive a readable Simplified Chinese match report with calculated probabilities and supporting evidence.

Execute this task through implementation, meaningful offline tests, installation/CLI smoke checks, documentation, and a sample report. Do not stop after writing a plan. The requested work is authorized. Resolve routine implementation choices yourself and explain material limitations in the execution receipt.

The public report should take roughly one to two pages to read. Target approximately 800–1,200 Chinese characters of narrative plus compact tables; use less text when evidence is sparse. Do not pad the report to meet a character count.

## Workspace and current facts

- Repository: `D:\matchquant\worktree`, currently on branch `quant-v2`. Inspect the current working tree before changing anything.
- Read these first: `QUANT_MASTER_CHARTER.md`, `AGENTS.md`, `README.md`, `pyproject.toml`, `CODEX_SHOWCASE_HANDOVER.md`, and this specification.
- The project is a local research tool. This task explicitly authorizes the new prediction/report entry and the narrow supporting changes described here. Preserve the project’s chronology, data ownership, and model isolation rules.
- The charter prohibits resurrecting obsolete bundle/notarization/release-verification frameworks. Build a small computational workflow with ordinary JSON provenance. Do not add those frameworks.
- A global `takeover-context` skill was previously found to describe an unrelated Wanman project. Do not apply unrelated Wanman instructions to MatchQuant.
- The charter identifies `D:\Dev\Python314\python.exe` as the preferred interpreter. Check that it exists and has the necessary dependencies. If unavailable, use a working Python >=3.11 environment and state which interpreter you used.
- Existing `python -m matchquant` invokes `matchquant.cli:main` and works. At the time of this handoff, `pyproject.toml` points the installed `matchquant` command to `matchquant.launcher:main`, but `matchquant/launcher.py` is absent. Recheck and fix this installation mismatch narrowly.
- NumPy and pandas are declared dependencies. The quant modules import SciPy. Inspect imports and declare the dependencies actually required by the implemented workflow. Do not add scikit-learn or a new framework unless the source genuinely requires it.
- Existing commands include `rich-prepare`, `factor-lab`, `quant-research`, `matchday-radar`, `run`, `epl-download`, `epl-prepare`, and `select`.
- `matchday-radar` currently selects the last N rows of a historical factor panel and prints a terminal table. It does not accept an arbitrary pending fixture and does not write an individual narrative report. Preserve that command’s behavior.
- `quant-research` writes aggregate evaluation artifacts including `quant_research_report.md`, `quant_evaluation_summary.json`, and prediction CSVs. These are historical research outputs, not a reusable fitted model or a one-fixture prediction API.
- Old snapshot/session commands mentioned in resources or historical design documents are not proof that their implementations exist in this checkout. Inspect the actual code; do not build on absent modules.
- Preserve unrelated local/untracked files, including `demo_model_showcase.py`, `matchquant/quant_models.zip`, and existing Site sources/packages. Do not reset or clean the checkout.

## Existing code to reuse

Read the relevant implementation, not only the function signatures:

- `matchquant/rich_data.py`: rich historical schema, opening/closing odds fields, and `devig_power_probabilities`.
- `matchquant/factor_engine.py`: `ChronologicalFactorEngine`, season transitions, `compute_pre_match_factors`, `update_from_completed_batch`, and `build_historical_factor_panel`.
- `matchquant/factor_lab.py`: `run_factor_lab`, explicit `discovery_seasons`, factor screening, and separation of fundamental/market features.
- `matchquant/quant_models.py`: `fit_and_select_factor_system`, `CalibratedFactorSystem.predict_proba`, `predict_rich`, Dixon–Coles goal outputs, and `OrderedFactorModel.explain_row`.
- `matchquant/fixtures.py`: pending-fixture identity validation. Its current return value keeps the core fixture columns and internal chronology fields, so optional referee/market metadata requires a deliberate adapter rather than assuming it survives.
- `matchquant/io.py`: existing file hashing and JSON helpers.
- `matchquant/cli.py`, `matchquant/__main__.py`, and relevant tests.

Important source details to account for:

1. `build_historical_factor_panel` expects completed rows and reads target goals. Do not fabricate a pending row with zero goals and feed it through that function.
2. `compute_pre_match_factors` is described as non-mutating but calls `_get_or_init_team`, which can initialize or adjust team state. Inspect the actual behavior. Isolate query state and reject unknown teams before computing features.
3. The final probability system blends Ordered Logit, Dixon–Coles, and Elo. `OrderedFactorModel.explain_row` explains only the ordered component. Its `marginal_prob_h_pct` is a first-order local approximation, not an exact additive decomposition of the final ensemble.
4. `fit_and_select_factor_system` selects settings on validation data and then refits on train + validation. Its returned validation selection scores must not be replaced by in-sample scores from that final refit or labeled as an independent blind test.

## Intended user experience

Add a command named `match-report`. Support both direct arguments and a one-row, score-free fixture CSV. These are equivalent input methods, not two different forecasting implementations.

Example direct invocation, with paths and fixture names replaced by actual available inputs:

```powershell
python -m matchquant match-report --input "D:\data\epl_rich_canonical.csv" --home "Man City" --away "Aston Villa" --date 2026-05-24 --season 2025-26 --as-of 2026-05-24 --mode replay --output-root "D:\Codex\matchquant_runs\single_match"
```

Example CSV invocation:

```powershell
matchquant match-report --input "D:\data\epl_rich_canonical.csv" --fixtures "D:\data\one_fixture.csv" --as-of 2026-05-24 --mode replay --output-root "D:\Codex\matchquant_runs\single_match"
```

CLI requirements:

- Required common inputs: `--input`, `--as-of`, and `--output-root`. Require an explicit input file; do not silently discover/download history or fall back to a demo.
- Direct input requires all of `--home`, `--away`, `--date`, and `--season`.
- Alternatively, `--fixtures` accepts exactly one fixture using `fixture_id,date,season,home_team,away_team`, with optional `referee` and optional `kickoff_time` as descriptive metadata. Reject mixed direct/CSV input.
- For direct input, generate a stable fixture identifier from normalized identity. Internally adapt `fixture_id` to a model row identifier only where existing model APIs require it.
- `--mode research|replay` defaults to `research`. Always display the selected mode. Research means a user-specified hypothetical/pending fixture; it does not certify current data or a real scheduled match. Replay means an explicitly labeled historical reconstruction.
- Optional `--team-aliases PATH` accepts a small explicit JSON mapping from aliases to canonical names. Do not introduce an online name resolver or mandatory alias database.
- Optional `--market-input PATH` accepts the market JSON schema below.
- Optional `--max-data-age-days` defaults to 30; require a non-negative integer. This is a disclosure/review threshold, not proof that data within 30 days are current.
- Print a short status, matchup, cutoff, warnings count, and absolute output paths. Save detailed evidence to files.
- Both `python -m matchquant ...` and the installed `matchquant ...` command must work from a clean installation. Document shell-appropriate quoting and portable paths.
- A natural-language chatbot is not part of this stage. Document how a coding assistant can translate a request into this explicit command. A request without a fixture date/season/cutoff must be clarified; never guess a real match from “tonight” or team names alone.

## Input validation and temporal semantics

Implement one explicit, conservative time contract for this first release:

1. Accept ISO date-only `YYYY-MM-DD` for `--as-of` and fixture date. Treat `as_of` as the start of that calendar day. Do not add unverified intraday/time-zone forecasting semantics in this stage.
2. Use only completed historical rows with `date < as_of`. Exclude every row on the cutoff day, even if it has a kickoff time. This also freezes all same-day matches together.
3. Require `fixture_date >= as_of`. The fixture’s explicit season must be valid. Never infer its season from calendar month.
4. Filter history by the cutoff before constructing factors, running Factor Lab, selecting hyperparameters, fitting models, or computing recent form. Do not build the full-history state and then try to remove future influence.
5. Validate dates sufficiently to identify eligible rows; an unparseable historical date is an explicit error. Validate eligible history for required fields, duplicate identities, invalid scores, and malformed numeric inputs. Do not impute a missing completed score as 0–0.
6. Reject non-empty fixture scores or post-match statistics, including common raw/canonical aliases. Use a documented allowlist for pending-fixture input fields so unsupported target information is not silently accepted.
7. An already completed target fixture may exist later in the input history during replay. Its result is excluded by the cutoff and must never appear in the pre-match report or influence the model.
8. Resolve names against teams visible in eligible history. Permit conservative whitespace/case normalization only where it maps uniquely. Apply explicit aliases, validate their targets, and show the resolved identity.
9. Unknown or ambiguous names must fail with useful suggestions. Never turn a typo into an automatically initialized “promoted team.” New-team modeling is outside this stage.
10. Team presence in historical data does not prove current league membership or that the fixture is officially scheduled. State that the matchup is user supplied and has not been checked online.
11. Handle the fixture-season transition on an isolated query engine exactly once. Support the history’s current season or its immediate successor; reject skipped/unsupported season jumps with an explanation. Preserve existing carry/regression rules.
12. Never update the warmed engine using the pending fixture. Repeated requests or reversed query order must not change predictions. Use an independent in-memory copy for query feature generation if needed.
13. Report `used_data_cutoff`, data age at `as_of`, and forecast horizon (`fixture_date - as_of`). When history is stale under the configured threshold, output `needs_review` prominently. Also disclose that any intervening unobserved matches are not simulated.
14. These checks establish the tool’s chronology contract. Do not claim that a finalized historical CSV proves the historical availability/vintage of every individual source observation.

## Training and prediction workflow

Use this ordered pipeline, reusing the current model family and selection routines:

1. Load and validate the explicit rich historical input and pending fixture.
2. Apply the date cutoff, then build the pre-match historical factor panel and warm the historical engine only from eligible completed matches. Keep this state in memory.
3. Use an expanding historical training policy with complete EPL seasons before `as_of`: the latest eligible complete season is validation, and the preceding complete seasons are training. Require at least two training seasons plus one validation season. Validate a complete EPL season as 380 unique valid completed matches; do not call a partial season complete.
4. An incomplete current season may update the factor-engine state before `as_of`, but must not enter the fitting/selection partitions in this first release. Report fitting cutoff separately from the state’s latest data cutoff.
5. Run fundamental factor discovery using training seasons only. Pass the seasons explicitly. Do not use validation or future labels for factor discovery. Retain existing core controls, orthogonal screening, sign constraints, and collinearity rules. Do not hardcode the Showcase’s 12 factors or a particular fitted probability.
6. Fit/select Track A using the existing train/validation process. Hyperparameters and blend weights may be selected using validation, then the existing final refit on train + validation may be reused. Record selected features, parameters, blend weights, partition dates, and row counts.
7. Report returned validation selection metrics honestly as tuning/selection evidence. They are not an untouched test of the final refitted system. Do not rerun the five-season full research pipeline for every individual report.
8. Compute the pending fixture’s fundamental features from the isolated warmed engine. Expose Elo baseline probabilities and Track A ensemble probabilities. Use `predict_rich` or the equivalent existing components without inventing a completed target score.
9. Return finite non-negative home/draw/away probabilities summing to one, model expected home/away goals, and P(total goals > 2.5). Label goal rates as model expected goals, not independently observed/tracking-based xG.
10. Use the existing fixed model-selection grid. This stage has no performance optimization target, new factor invention, model replacement, or random chronological splitting.
11. If eligible data or model fitting are insufficient, fail explicitly. Do not silently substitute a synthetic fixture, a saved Showcase prediction, or an unrelated Elo-only report.

Track A must remain independent of all opening/closing odds, steam, and market-only factors, including during feature selection. If a narrow existing screening path violates this isolation, correct it with a focused regression test and document the reason.

## Optional market comparison

No market file is required to produce a complete fundamental report. When absent, display “未提供赛前赔率，未计算市场对照”. Never invent odds, edge, CLV, or an investment conclusion.

Accept an optional JSON object with these documented fields:

```json
{
  "home_team": "Man City",
  "away_team": "Aston Villa",
  "date": "2026-05-24",
  "observed_at": "2026-05-23",
  "source": "User-supplied bookmaker/source description",
  "odds_open_h": 2.1,
  "odds_open_d": 3.4,
  "odds_open_a": 3.6,
  "mkt_ah_line": -0.25
}
```

These values are schema examples, not actual market quotes. Do not include them as evidence in a real sample report.

- Match market identity to the resolved fixture. Require a declared source and observation date strictly before `as_of`, consistent with the date-only start-of-day cutoff.
- Validate decimal 1X2 odds as finite and >1. The Asian-handicap line is optional and finite when supplied; document the home-team sign convention. Reject partial/malformed odds rather than filling gaps with zero or future values.
- Use the existing Power method to calculate market-implied probabilities. Label the data as user supplied and the observation time as declared, not independently verified.
- With valid 1X2 odds, report `Track A probability - de-vigged market probability` in percentage points. State the calculation directly. This is a probability difference, not an independently validated expected profit or a future CLV.
- Track M-Open is optional. Run it only when all market predictors its fitted feature list requires are available and valid in both eligible historical fitting data and the fixture input. Existing M-Open uses opening 1X2 information plus Asian-handicap information. Missing AH must not become an unstated neutral zero; market comparison can still be shown while M-Open is marked unavailable.
- Fit optional M-Open using the same chronological partitions and fundamental selection. Use only opening market predictors. Never use Track B, closing odds, future steam, target results, or observed future CLV for a pre-match forecast.
- If M-Open is unavailable or fails while Track A is valid, keep the fundamental report, clearly state the market-track limitation, and set the appropriate review warning. Do not silently represent Track A as M-Open.
- Do not add Kelly stakes, bankroll projections, bet placement, or a “guaranteed value” verdict to a single-match report.

## Evidence and attribution

The report must be based on calculated outputs rather than generic football commentary:

- Show the two teams’ recent eligible matches (last five available per team) in a compact evidence summary: dates/sample counts, W/D/L, goals for/against, and available shots-on-target/corners statistics. All records must precede `as_of`. Disclose missing values and the actual valid sample count.
- Include canonical match IDs/dates and the historical factor row values in JSON so the evidence can be checked. Do not copy the complete historical dataset into report artifacts.
- Select up to three meaningful non-Elo drivers from the trained ordered component’s explanation and translate factor names into plain Chinese. Show direction, raw factor value, and the local approximate home-win contribution in percentage points where applicable.
- Explicitly label these contributions “有序模型分量的局部近似贡献”. They do not add up to the final ensemble’s probability difference and are not causal effects. Do not imply otherwise in the headline or tables.
- If the ordered component has zero ensemble weight, do not present its local contributions as explanations for the final prediction. Show factual feature observations and state the attribution limitation, or calculate a defensible final-model explanation within scope and test it. Do not add a new attribution framework dependency.
- Separate factual input statistics from the model’s interpretation. Do not turn a proxy into a claim about injuries, actual tactical plans, morale, lineup changes, or “motivation” known from news.
- No injury/player/news feeds are supplied by this repository. State that absence briefly where it affects the report. Do not browse or fabricate such information.
- Class names such as `CalibratedFactorSystem` do not establish calibrated real-world probabilities. Use “模型估计概率”; avoid unsupported confidence levels, calibration claims, or invented confidence intervals.

## Report content and files

Create a new run directory under the explicit output root. Do not overwrite a previous run or the existing `reports/research_run/` artifacts. Use a simple unique run name; do not introduce a manifest framework.

Required files:

1. `match_report.json`: the authoritative structured result, with a small schema version such as `single-match-report-v1`.
2. `match_report.md`: readable Simplified Chinese report.
3. `match_report.html`: the same report as a standalone, printable, responsive HTML document with inline CSS, no network requests, CDN dependencies, scripts, or externally loaded fonts. Escape all user-provided text.

All presentation numbers must come from the JSON result, with consistent rounding. Do not calculate a different version of the prediction inside the renderers. Write standards-compliant JSON without NaN or Infinity; unavailable values are null plus an explanation.

Suggested public report structure:

- **比赛与数据时间**: teams, user-supplied fixture date/season, research/replay label, prediction cutoff, latest used match date, and any freshness warning.
- **模型判断**: a short plain-language paragraph and a compact home/draw/away probability table. State which outcome has the largest estimated probability without turning it into certainty.
- **进球预期**: model expected goals for both teams and the probability of at least three total goals. Explain briefly that an expected goal value is an average, not a promised exact score.
- **判断依据**: up to three supported drivers and a concise recent-performance comparison. Include contrary evidence when the calculated values show it. No filler about a team’s history or reputation.
- **模型与赔率对照**: Elo/Track A/optional M-Open probabilities, component disagreement, and optional market probability differences. Clearly mark unavailable comparisons.
- **哪些信息还没算进去**: data gaps, stale inputs, unverified fixture, absent news/lineups, model limitations, and a brief research-use statement.
- A compact source/method footer with the input file identifier, actual used cutoff, train/validation seasons, and model/version identifier. Detailed numeric diagnostics belong in JSON rather than a long technical appendix in the public report.

The JSON must include at least:

- fixture identity and normalized/resolved names;
- requested `as_of`, mode, eligible history cutoff, data age, forecast horizon, and excluded future/same-day row count;
- input history path + SHA-256, optional alias/market source identity, package/model identifier, and actual fitting configuration;
- train/validation seasons, date ranges and row counts, selected features, blend weights, and honestly labeled validation-selection metrics;
- Elo and Track A probabilities, component probabilities/weights if exposed, goal outputs, optional M-Open/market results and availability reasons;
- raw inference feature values, attribution method/limitations, recent evidence records/aggregates;
- `status` (`complete` or `needs_review`) and explicit warning reasons.

Reuse simple existing hash helpers. One input hash is sufficient provenance for this scope; do not build signatures, chained hashes, certification, runtime bundles, or model-state publication machinery.

Use a documented exit policy: 0 for a complete report, 2 for a written report requiring review, and 1 for expected data/model errors. Preserve conventional argparse usage-error behavior. Invalid fixture input should not leave a report that appears successful. Unexpected programming exceptions must not be swallowed and described as ordinary missing data.

## Suggested implementation scope

Prefer a small design along these lines; adjust names if existing code makes a simpler equivalent possible:

- `matchquant/single_match.py`: input adaptation, cutoff/split policy, existing engine/model orchestration, and structured result.
- `matchquant/match_report.py`: pure Markdown/HTML rendering from the result JSON structure.
- An optional small `single_match_inputs.py` only if validation would otherwise obscure the computational flow.
- Minimal changes in `cli.py`, `pyproject.toml`, `README.md`, and `__init__.py` if exports are useful.
- A narrow helper/fix in `factor_engine.py`, `fixtures.py`, or `factor_lab.py` only where genuinely required for safe query state, metadata retention, or Track A isolation. Explain every such change.
- Focused tests in new single-match/report test files and existing CLI tests as appropriate.
- One small example pending-fixture CSV and optional example market JSON under existing example resources. Clearly label fictional/sample values. Preserve current canonical completed-match schemas and existing commands.

Do not add a web server, chat UI, LLM API dependency, MCP server, database, scheduler, downloader, new factor, new model family, saved-model cache, or a general platform architecture. Do not change the Showcase Site in this task.

## Execution sequence

1. Inspect repository state and the files above. Summarize the concrete approach briefly, then implement it. Do not seek approval for ordinary in-scope work.
2. Fix the installed command target and actual runtime dependency declarations.
3. Implement and test the input/cutoff/state boundaries before connecting model fitting.
4. Add the explicit training partitions and reuse existing selection/fitting/prediction components.
5. Implement structured results, factual evidence, optional market behavior, and the Chinese renderers.
6. Connect `match-report` to both CLI invocation forms; update installation and user documentation.
7. Run focused tests, one complete offline test suite, installation/CLI smoke checks, and an actual report generation. Address failures relevant to the task and accurately distinguish pre-existing failures.
8. Deliver a Chinese execution receipt with the exact usable commands, sample report links, evidence of tests, and remaining limitations.

Keep progress updates concise and explain findings that affect the implementation. Continue until the workflow is usable and reviewable.

## Required validation

Use deterministic offline tests that exercise behavior. Mock expensive fitting in focused orchestration/error tests where appropriate, but include at least one real end-to-end model fit and report generation. Do not assert a literal model probability that merely mirrors the implementation.

Cover these boundaries:

1. Changing/removing/appending valid future rows or changing same-cutoff-day results and statistics leaves eligible features, training selection, and predictions unchanged within a justified tight numerical tolerance. File hashes/excluded-row counts may change; do not compare the entire provenance JSON for equality.
2. Changing future target scores, closing odds, or steam cannot change the pre-match prediction. Changing any opening/closing market values cannot change Track A selection, prediction, or attribution.
3. Same-day historical matches remain excluded; no row order dependence or pending-query state contamination. Test repeated queries and different query order.
4. Probabilities are finite, non-negative, and sum to 1 within 1e-9; expected goals are finite and non-negative; over-2.5 probability is in [0,1].
5. Unknown/ambiguous teams, explicit aliases, identical teams, invalid date/season, fixture before cutoff, scored fixtures/post-match fields, duplicate eligible history, insufficient training seasons, and unsupported season jumps receive useful outcomes.
6. Missing market inputs still produce a fundamental report. Valid odds use the existing de-vigging function. Partial/invalid/future-dated odds fail validation. Missing AH explicitly disables a market track that requires it. No closing/steam field is accepted for a target forecast.
7. Partial current-season results update pre-cutoff team state but do not enter the complete-season fitting partitions. Factor discovery never receives validation/future labels. Validation selection evidence is not calculated using the final refitted model on its own training rows.
8. Stale-data review status, forecast horizon, source identity, and used cutoff are rendered correctly. Missing evidence stays unavailable rather than fabricated.
9. Report figures agree with the JSON; HTML escapes team names/source text; attribution is labeled as a component’s local approximation; a zero ordered blend weight is handled honestly. No target actual score leaks into a replay report.
10. The new command appears in help, both fixture input methods work, the console script works after an isolated installation, and old commands/tests remain usable.

Use the repository’s actual working interpreter. The standard suite is:

```powershell
python -m unittest discover -s tests -v
```

Run focused tests first, then the full suite once after changes stabilize. Use a temporary/isolated virtual environment for the installation smoke test so that global Python packages and the user’s existing installation are not overwritten. A venv using existing site packages and an editable `--no-deps` install may be used when all declared dependencies are already available.

For the end-to-end example:

- Prefer an explicitly labeled historical replay using an existing local rich EPL CSV and a fixture already supported by its teams/seasons, such as the final 2025–26 round if that dataset is present. Use a pre-fixture cutoff and do not include the actual score in the report.
- Determine real local dataset locations from project documentation and existing explicit paths. Do not download data or use the public Showcase’s precomputed probabilities as inference results.
- If real local data are unavailable, generate a deterministic synthetic multi-season dataset sufficient for the same training gates, use clearly fictional teams, and label every sample artifact as synthetic. Do not weaken production validation to make the demo pass.
- Write generated data and reports outside tracked source data, normally under `D:\Codex\matchquant_data\` and `D:\Codex\matchquant_runs\`. Do not overwrite the source input, existing reports, or prior runs.
- Allow new strict selection/training policy to produce different predictions or factor counts from the Showcase. Explain why; do not tune to reproduce its numbers.

## Documentation and acceptance

README must enable a new user to understand:

1. The required Python/dependencies and a working installation.
2. How they prepare their own local rich historical data using existing explicit data commands.
3. How to specify one fixture, its season, and the cutoff; supply aliases/market input when needed; and open the HTML report.
4. What data the report actually uses and what it cannot infer about schedules, lineups, injuries, current data coverage, or future profitability.
5. That a coding assistant can call this command from a natural-language request with explicit inputs, while the repository itself remains a command-line research tool.
6. How to reproduce the example report and interpret `needs_review`.

The stage is accepted only when a valid input produces all three consistent report files, invalid inputs fail clearly, chronology/market isolation tests pass, the installed command works, the sample report is readable Chinese with actual supporting evidence, and the implementation stays within the compact scope above.

Do not push to GitHub, publish a release, change repository branches destructively, or deploy any Site. Leave the changes locally reviewable. In your final Chinese receipt, provide changed-file links, the actual test/smoke outcomes, the real interpreter and commands used, a link to the sample HTML/Markdown/JSON, and any unresolved limitation. Do not claim a test, install, real-data run, or numerical result that you did not execute.
