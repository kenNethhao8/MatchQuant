# M11 Engine Snapshot Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans. Steps use checkbox (`- [ ]`) syntax and each task ends with an independent test gate.

**Goal:** Add an explicit offline M11 snapshot builder and snapshot-only prediction command that preserve M4 chronological results while avoiding full-history replay at query time.

**Architecture:** `engine_snapshot.py` owns build and predict orchestration. `engine_snapshot_support.py` owns deterministic JSON state codecs, manifest validation, hashing, and output directories. Existing M4 model/update logic remains authoritative; M11 serializes the post-update state and never copies or mutates history. CLI and README expose two explicit commands.

**Tech Stack:** Python 3, pandas, numpy, standard library JSON/CSV/pathlib, unittest. No new dependency.

**Spec:** `docs/superpowers/specs/2026-09-06-m11-engine-snapshot-design.md`

## Global Constraints

- Preserve M1 schema, chronology rules, M2.1/M4/M8/M9/M10 interfaces, and existing tests.
- M11 is offline-only; every input path is explicit and no build/predict path may call a download or network function.
- Keep raw and processed data under `D:\Codex\matchquant_data\`; keep snapshot artifacts there and prediction runs under `D:\Codex\matchquant_runs\`.
- Never copy history into a snapshot, overwrite input files, update `epl_current.json`, or silently resolve a latest file.
- Each Python file stays under 400 lines and ideally under 300; avoid `utils.py`, pickle, broad exception handling, and dead compatibility paths.
- Every snapshot/run records source paths, SHA-256 hashes, model configuration, cutoff, and mutation/network declarations.
- Unit tests remain offline and CLI output stays at or below 20 lines.
- Do not add Alpha factors, recent information, networking, UI, database, scheduler, calibration, or profitability claims in M11.

---

### Task 1: Add failing contract tests and compact fixtures

**Files:**
- Create: `tests/test_engine_snapshot.py`
- Inspect: `matchquant/forecast.py`, `matchquant/forecast_runner.py`, `matchquant/models.py`, `tests/test_forecast.py`, `tests/test_cli.py`

**Interfaces:**
- Consumes: existing `forecast_fixtures`, `run_forecast`, synthetic canonical history, explicit model configs, and canonical pending fixtures.
- Produces: reusable reference cases for all three models, snapshot round-trip tests, CLI smoke fixtures, and mutation/hash assertions.

- [ ] **Step 1: Build a self-contained reference case.**

Use only a temporary directory. Copy or construct a small canonical M1 history, create a matching processed history manifest with a valid output hash/cutoff, and create score-free fixtures after a chosen as-of date. Keep one post-cutoff historical row available so the future-row invariance test can alter it.

- [ ] **Step 2: Write the failing state/build contract tests.**

Cover: all three model states round-trip; explicit model/config selection; unique snapshot contains only manifest and state file; history is not copied; manifest hashes match; post-cutoff history changes do not alter the built state; invalid as-of and missing manifest evidence fail.

- [ ] **Step 3: Write the failing snapshot-predict contract tests.**

For every model, compare snapshot predictions against `forecast_fixtures` using the same history/as-of/config. Assert snapshot prediction accepts no history argument, records snapshot/fixture hashes, and rejects a fixture before cutoff, a fixture with scores, a corrupt snapshot, or a hash mismatch.

- [ ] **Step 4: Write CLI expectations.**

Exercise `engine-snapshot-build` and `engine-snapshot-predict` through `matchquant.cli.main` or subprocess as appropriate. Assert concise output, successful status, explicit paths, and a patched network entry point is never called.

- [ ] **Step 5: Run the focused file and confirm RED.**

Run: `python -m unittest discover -s tests -p 'test_engine_snapshot.py' -v`

Expected: import/implementation failures because M11 modules and commands do not yet exist.

---

### Task 2: Implement deterministic model state codecs and snapshot persistence

**Files:**
- Create: `matchquant/engine_snapshot_support.py`
- Modify: `matchquant/models.py` only if a small public state hook is required
- Test: `tests/test_engine_snapshot.py`

**Interfaces:**
- Consumes: `create_model`, current model public parameters/state, `file_sha256`, `write_json`, and M4 schema validators.
- Produces: deterministic state export/import, `new_engine_snapshot_dir`, file records, manifest validation, and strict snapshot loading.

- [ ] **Step 1: Define explicit JSON state codecs.**

Implement named codecs for Frequency, Elo, and Poisson. Export only finite JSON values with sorted team keys; reconstruct a fresh model through `create_model`, restore state, and validate the model name/config/state shape. Raise `DataValidationError` for unknown models, invalid numbers, malformed team records, or missing required fields.

- [ ] **Step 2: Add unique snapshot directories and file records.**

Create `m11_engine_snapshot_<UTC>_<ID>` under the explicit snapshot root. Reject a snapshot root inside the active data/run input directories when that would permit overwriting inputs. Persist `model_states.json`, then compute and record its SHA-256 in `snapshot_manifest.json`; never persist a copied history file.

- [ ] **Step 3: Add strict manifest loading.**

Require schema `m11-engine-snapshot-v1`, valid snapshot/state hashes, explicit as-of metadata, supported model configs, and an existing recorded state file. Return only the validated state/config evidence needed by the prediction path.

- [ ] **Step 4: Run the codec/persistence tests.**

Run: `python -m unittest discover -s tests -p 'test_engine_snapshot.py' -v`

Expected: state round-trip and manifest tests pass; orchestration tests remain pending or fail only on missing public APIs.

- [ ] **Step 5: Commit the isolated support layer.**

```powershell
git add matchquant/engine_snapshot_support.py tests/test_engine_snapshot.py
git commit -m "feat: add M11 engine snapshot state codecs"
```

---

### Task 3: Implement snapshot build and reference-equivalent prediction

**Files:**
- Create: `matchquant/engine_snapshot.py`
- Modify: `matchquant/__init__.py`
- Test: `tests/test_engine_snapshot.py`

**Interfaces:**
- Consumes: `load_matches`, history-manifest evidence, `_history_before_as_of`/existing M4 chronological semantics, `create_model`, state codecs, and `validate_fixtures`.
- Produces: `build_engine_snapshot` and `predict_from_engine_snapshot` with the signatures in the M11 spec.

- [ ] **Step 1: Implement explicit history evidence loading.**

Require the canonical history and processed manifest paths. Verify manifest output path, output hash, row count, required columns, and observed cutoff before building. Do not infer a missing version or silently use the active pointer.

- [ ] **Step 2: Implement chronological snapshot building.**

Use the existing M4 date/timestamp batch semantics, select only rows strictly before `as_of`, instantiate the requested models/configs, call `update` in chronological order, and serialize the resulting state. Record total rows, used rows, excluded rows, actual used cutoff, as-of metadata, model configuration hashes, and input mutation flags. Support an explicit M3 selection config by reusing its existing validation; do not re-select a model.

- [ ] **Step 3: Implement snapshot-only prediction.**

Load and validate the snapshot, read only the explicit canonical pending fixtures, enforce score-free rows and on/after cutoff, restore each selected model, call `predict` without any history replay, and write a unique run with predictions, run config, and a concise report. Include snapshot and fixture hashes and the exact cutoff in every output.

- [ ] **Step 4: Prove equivalence and isolation.**

Make the tests compare every model's snapshot output with the existing in-memory M4 reference. Add a test that deletes/renames the history path after snapshot creation and still predicts successfully, proving the query path uses only the snapshot; keep build-time history evidence mandatory.

- [ ] **Step 5: Run focused tests and commit.**

Run: `python -m unittest discover -s tests -p 'test_engine_snapshot.py' -v`

```powershell
git add matchquant/engine_snapshot.py matchquant/__init__.py tests/test_engine_snapshot.py
git commit -m "feat: build and serve M11 engine snapshots"
```

---

### Task 4: Add explicit CLI commands and documentation

**Files:**
- Modify: `matchquant/cli.py`
- Modify: `README.md`
- Test: `tests/test_engine_snapshot.py`

**Interfaces:**
- Consumes: `build_engine_snapshot`, `predict_from_engine_snapshot`, explicit paths, model/config selection options, and existing CLI error conventions.
- Produces: `matchquant engine-snapshot-build` and `matchquant engine-snapshot-predict`.

- [ ] **Step 1: Add the build parser.**

Require `--history`, `--history-manifest`, and `--as-of`. Accept either repeatable `--model` plus matching `--model-config` JSON paths or one explicit `--selection-run-config`; accept `--snapshot-root`. Reject ambiguous model/selection combinations at argparse or API validation.

- [ ] **Step 2: Add the predict parser.**

Require `--snapshot-manifest` and `--fixtures`; accept `--output-root`. Do not expose a history argument on this command. Print no more than six concise status/path lines and return nonzero on validation failure.

- [ ] **Step 3: Add CLI offline smoke tests.**

Build a snapshot then predict from it through the command path while patching the network entry point. Assert the run succeeds, the prediction output exists, the history is not required by prediction, and stdout is concise.

- [ ] **Step 4: Document the two-step workflow and limitations.**

Add an M11 section explaining snapshot build versus snapshot prediction, cutoff/hash evidence, no history copy, explicit refresh/build behavior, and that M11 is not Alpha, currentness, calibration, profitability, or betting evidence.

- [ ] **Step 5: Run complete verification and commit.**

Run:

```powershell
python -m unittest discover -s tests -v
python -m compileall -q matchquant tests
python -m matchquant engine-snapshot-build --help
python -m matchquant engine-snapshot-predict --help
git diff --check
```

Commit:

```powershell
git add README.md matchquant/cli.py tests/test_engine_snapshot.py
git commit -m "docs: document M11 engine snapshot workflow"
```

---

## Controller audit gate

The execution worker must not merge or push. The controller will inspect the diff, verify no raw/history copies or pointer mutation were introduced, run the full test suite independently, inspect snapshot/prediction manifests, and only then fast-forward-merge locally. M12 incremental updates and M13 Alpha factors remain out of scope until M11 equivalence passes.
