# MatchQuant 合成演示（frequency 单模型运行包）

**这是纯合成教学数据，不是真实联赛数据。** 所有球队（North United、South City、East Rovers、West Wanderers）与全部比分均为虚构。演示选择 frequency 模型仅仅为了让教程闭环最简单；它**不是**真实场景下的 primary model 选择结论，也不代表任何模型效果比较。

## 时间边界（三条时间各自独立，不能混用）

| 概念 | 本演示取值 | 含义 |
|---|---|---|
| snapshot `as_of` | `2024-01-07`（date-only） | 状态边界：2024-01-07 当天及之后的历史结果在建模时不可见 |
| `used_data_cutoff` | `2024-01-06` | 实际进入模型状态的最后一个历史日期 |
| fixture 日期 | `2024-01-07` | 两场待赛在同一天；先对整个批次完成预测，再另行用独立结果文件结算 |

`as_of`/`used_data_cutoff` 的准确数值以导出包内快照 manifest 与会话报告显示为准。

## bundle 内容

`bundle/` 是 B04 固定格式的完整运行包：7 个载荷 + `bundle_manifest.json`，共 8 个实际文件。它由真实生成流程构建（`build_engine_snapshot` → 真实 M12 `check_engine_snapshot_status` 得到 fresh 发行记录 → `build_runtime_bundle`），不是手写 JSON。运行端只从包内相对路径解析文件；把 `bundle/` 整体复制到其它目录（含空格路径）后，预测、审计与结算仍应离线完成且概率不变。

整个包没有签名来源认证：哈希只证明本次输入一致性，不能发现全体一致的伪造。

## 操作步骤

前提：已按 README 完成离线安装，且 `demo-export` 已把本目录导出到你的 `DIR`。以下命令中的 `matchquant` 指 venv 内的可执行入口（如 `D:\你的目录\.venv\Scripts\matchquant.exe`）。

1. 验证运行包（只读）：

   ```powershell
   matchquant runtime-bundle-verify --bundle-manifest "DIR\bundle\bundle_manifest.json"
   ```

2. 用 score-free fixtures 做 replay 预测会话（输出到新建空目录）：

   ```powershell
   matchquant snapshot-session --bundle-manifest "DIR\bundle\bundle_manifest.json" --fixtures "DIR\fixtures.csv" --mode replay --output-root "新建空目录"
   ```

3. 从该次会话的 `session_manifest.json` 读取 `outputs.prediction_run_dir`，再做独立只读审计：

   ```powershell
   matchquant snapshot-run-verify --forecast-run "<prediction_run_dir>" --bundle-manifest "DIR\bundle\bundle_manifest.json" --output-root "另一个新建空目录"
   ```

4. 用独立结果文件结算（`results.csv` 只在此步骤显式读取）：

   ```powershell
   matchquant settle --forecast-run "<prediction_run_dir>" --results "DIR\results.csv" --output-root "另一个新建空目录"
   ```

结算应显示 `complete`、2/2 场、coverage 1.0；概率指标按结果计算，不构成校准、盈利或投注建议。

## research 模式（可选）

`snapshot-session --mode research` 表示本地待赛研究：即使工程校验全部通过，因为目标时点的赛果覆盖未证明，状态恒为 `needs_review`（退出码 2），并保留审查理由与产物。它不能被包装成"当前可用预测产品"。replay 模式则是明确标记 retrospective 的历史回溯。

## 已知限制

- 全部数据为合成内容；不证明任何真实球队、真实赛程或真实数据的可用性。
- frequency 模型没有球队学习状态：预测使用全局历史频率，不维护球队清单，也不把输入球队称为"新队"。
- 概率仅保证有限、非负、和为 1；本项目不宣称概率已校准。
- 演示不含联网、训练、实时数据或投注功能；`demo-export` 也只导出文件，不训练、不预测、不结算。
