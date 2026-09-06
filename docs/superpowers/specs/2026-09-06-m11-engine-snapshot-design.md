# M11 Engine Snapshot Design

## Goal

把现有 M4 的显式历史重放逻辑编译为一个可审计、不可变的本地模型状态快照，并增加只读取快照的预测入口。用户查询时只需要快照和显式的待预测 fixture，不需要再次加载或下载十年历史；快照预测必须与同一 cutoff 下的 M4 历史重放保持一致。

M11 是轻量化基础层，不是 Alpha 因子层，也不是近期新闻或联网层。

## Current boundary

M1–M10 保持不变。当前 `forecast_fixtures` 在 `as_of` 前筛选历史并逐批调用模型 `update`，再调用 `predict`。M11 复用这套时间语义和模型实现，只把 `update` 完成后的模型状态持久化；不复制 raw/canonical history，不自动寻找最新文件，不修改 M2.1 current pointer。

## Proposed flow

```text
explicit history + history manifest + as_of + model config
    -> chronological M4 state build
    -> immutable M11 snapshot (state + hashes + cutoff)
    -> explicit local fixture CSV
    -> snapshot prediction run
```

快照是服务运行材料，不是替代历史归档。历史数据仍保留在 `D:\Codex\matchquant_data\`，回测和审计继续使用历史文件；快照默认写入 `D:\Codex\matchquant_data\engine_snapshots\`，预测结果写入 `D:\Codex\matchquant_runs\`。

## Public interfaces

```python
def build_engine_snapshot(
    history_path: str | Path,
    history_manifest_path: str | Path,
    as_of: str | pd.Timestamp,
    *,
    model_names: Sequence[str] | None = None,
    selection_run_config_path: str | Path | None = None,
    model_configs: Mapping[str, Mapping[str, Any]] | None = None,
    output_root: str | Path = Path(r"D:\Codex\matchquant_data\engine_snapshots"),
) -> dict[str, Any]: ...

def predict_from_engine_snapshot(
    snapshot_manifest_path: str | Path,
    fixtures_path: str | Path,
    *,
    output_root: str | Path = Path(r"D:\Codex\matchquant_runs"),
) -> dict[str, Any]: ...
```

The command layer exposes explicit `engine-snapshot-build` and `engine-snapshot-predict` commands. Build accepts either explicit model/configs or the existing M3 selection evidence. Predict accepts only a canonical M4 pending-fixture CSV and a snapshot manifest; it does not accept a history path and does not call any network path.

## Snapshot contract

Each unique snapshot directory contains only:

- `snapshot_manifest.json`: schema version, snapshot id, UTC creation time, as-of value and semantics, explicit history and history-manifest paths/hashes, total rows, rows used before as-of, observed cutoff, model names/config hashes, state-file hashes, and no-mutation/network declarations;
- `model_states.json`: JSON-compatible public state for every selected model, including validated model configuration and deterministic team/state values.

The state codec is explicit for the current models:

- Frequency: result counts and alpha;
- Elo: all parameters and sorted team ratings;
- Poisson: all parameters, global totals, match count, and sorted per-team goal statistics.

No private Python pickle, executable object, raw history copy, or unbounded opaque blob is allowed. JSON numbers must be finite, keys deterministic, and all state hashes recorded.

Predict validates the snapshot schema, every recorded hash, model/config/state consistency, and that every fixture is score-free and on/after the snapshot as-of boundary. It writes a unique run directory containing predictions, run config, and a concise report. The run records snapshot hash, fixture hash, snapshot cutoff, and model versions. It never mutates the snapshot or fixture input.

## Temporal and equivalence rules

- Build uses exactly the M4 `as_of` boundary and chronological update order; same-day/date-only batch semantics remain unchanged.
- Post-cutoff rows are not used to build state. Changing only post-cutoff scores must not change snapshot state or predictions.
- Snapshot prediction must match the reference `forecast_fixtures` output for the same model/config/history/as-of within a documented floating-point tolerance.
- A fixture before the snapshot cutoff, with known scores, or with invalid identity is a hard failure.
- Missing/corrupt/unknown model state or any hash mismatch is a hard failure; no automatic repair or latest-file lookup is allowed.
- M11 does not update model state after a prediction. Result-driven incremental state updates are a later stage.

## Non-goals

M11 does not add Alpha factors, injuries/news/odds/weather, network refresh, scheduled tasks, a database, a web UI, machine learning, calibration claims, live-currentness claims, or betting/profitability claims. M2.1 initial pointer activation remains a separately reviewed data operation.

## Acceptance gate

M11 is complete only when an offline test proves: all current models round-trip through JSON state; snapshot predictions agree with M4 reference predictions; future rows cannot affect state; invalid snapshots/fixtures fail explicitly; CLI build/predict are concise and offline; input files and active pointer remain unchanged; README and manifests document cutoff, hashes, paths, and limitations.
