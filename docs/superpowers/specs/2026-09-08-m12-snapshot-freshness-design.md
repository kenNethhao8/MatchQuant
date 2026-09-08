# M12 Snapshot Freshness Status Design

## Goal

为一个明确的 M11 engine snapshot 和一个明确的 M2.1 current pointer 生成只读、可审计的新鲜度报告。M12 是历史版本关联与重建门槛，不是第二套同步流程。

## Public interface

```python
def check_engine_snapshot_status(
    snapshot_manifest_path: str | Path,
    current_pointer_path: str | Path,
    *,
    output_root: str | Path = Path(r"D:\Codex\matchquant_runs"),
) -> dict[str, Any]: ...
```

CLI：`engine-snapshot-status --snapshot-manifest ... --current-pointer ... [--output-root ...]`。

输入路径全部显式给出。命令只读取 snapshot manifest/state、pointer、pointer 指向的 canonical 和 processed manifest；不调用 `epl-sync`、M7、M10 或任何网络路径。

## Status contract

- `fresh`：snapshot history 与 pointer 的 canonical/processed-manifest 路径、SHA-256 和 dataset version 完整且精确匹配。
- `stale`：snapshot 和 pointer 证据自身有效，但至少一项路径、hash 或 dataset version 不同。
- `unverifiable`：snapshot/pointer 缺失、篡改、schema/路径/hash 结构无效，或无法完成版本关联。

`stale` 不等于快照损坏。现有 M11 prediction 仍只读取快照自身证据，因此旧快照继续可复现。M12 不会自动寻找 latest、更新 pointer、重建快照或改变 M11 prediction 行为。

## Outputs and invariants

每次调用在显式 output root 下创建唯一 `m12_engine_snapshot_status_<UTC>_<ID>`，写出 `snapshot_status.json` 和 `snapshot_status.md`。JSON 记录输入路径/hash、snapshot/pointer 证据、逐项比较结果、分类理由，以及 `inputs_unchanged`、`snapshot_mutated`、`current_pointer_mutated`、`network_accessed`、`automatic_latest_lookup`、`automatic_apply`、`automatic_rebuild` 声明。

输出目录不得位于 snapshot 目录内。命令不修改 snapshot、pointer、历史 canonical 或 processed manifest。

## Acceptance gate

离线测试覆盖 fresh、verified-but-different stale、无效/hash 不符 pointer、无效 snapshot、输入不变、stale 后仍能用 M11 predict，以及 CLI 无网络 smoke。重建继续由用户主动执行显式 M11 build 命令。
