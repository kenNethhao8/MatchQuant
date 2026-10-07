# 派生快照说明（runtime bundle 内）

**本文件是 runtime bundle 新生成的派生说明，标明 derived；不宣称与原研究报告字节相同。**

- snapshot_id：`m11_engine_snapshot_20261007T093810Z_fa40badd`（保留原 snapshot_id）
- schema：`m11-engine-snapshot-v1`；派生 manifest 由既有 writer 生成
- as_of：`2024-01-07`（日期批次）
- history data cutoff：`2024-01-06`
- used data cutoff：`2024-01-06`
- 模型：`frequency`（单模型）
- state SHA-256：`e1b57995c8608dd8cb5321ade08e6b351565c60c4932ec432784c7ffb8874e81`（与原快照 state 字节一致）

## 与原快照的关系

- 原快照目录只读，未改写；bundle 内 state 为原字节复制。
- 派生 manifest 仅按 B04 合同修改固定路径叶子（state.path、outputs.*）并由 writer 重算内容 hash；其余字段与原 manifest 深度一致。
- 原始 manifest 与 M12 发行记录以字节副本保存在 `evidence/`；其中记录的来源路径仅为 provenance，运行端从不打开，也不回退到原研究目录。
- 每次运行只从包内相对路径解析文件；来源路径不是运行文件候选。
