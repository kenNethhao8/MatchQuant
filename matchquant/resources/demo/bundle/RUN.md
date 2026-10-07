# MatchQuant runtime bundle 使用说明

本包是一个单模型 M11 运行包：迁移到不同绝对目录后，预测会话、运行审计与结算依然离线完成，不需要原研究端目录。

## 包内容

- `bundle_manifest.json`：包清单（所有载荷文件的相对路径、大小与 SHA-256）。
- `snapshot/m11_engine_snapshot_20261007T093810Z_fa40badd/`：派生快照 manifest 与原字节 state（保留原 snapshot_id）。
- `evidence/original_snapshot_manifest.json`：原 manifest 字节副本（仅为证据，来源路径不执行）。
- `evidence/issue_audit.json`：原 M12 fresh 发行记录字节副本（经派生桥关联到本包派生快照）。
- `team_index.json`：从 state 确定性导出的球队索引。
- `RUN.md`：本说明。

## 命令顺序

```text
matchquant runtime-bundle-verify --bundle-manifest <包根>/bundle_manifest.json
matchquant snapshot-session --bundle-manifest <包根>/bundle_manifest.json --fixtures <无比分CSV> --mode replay|research --output-root <会话输出根>
matchquant snapshot-run-verify --forecast-run <M11-run> --bundle-manifest <包根>/bundle_manifest.json --output-root <审计输出根>
matchquant settle --forecast-run <M11-run> --results <独立赛果CSV> --output-root <结算输出根>
```

## 边界

- 全程离线；不读取 history CSV、历史 processed manifest 或 current pointer。
- 会话/审计输出不得位于整个 bundle 目录内；每次运行所有包内文件起终 hash 不变。
- 模型 `frequency`：新球队按模型既有先验处理并在会话报告中标注；已知球队不证明其仍参加当前赛季。
- active_link_status 固定为 not_checked_in_runtime；research 会话保持 needs_review。
