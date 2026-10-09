# MatchQuant 进度收拢与 baseline 交接入口

版本：设计 v1.0；评估日期：2026-10-04（Asia/Shanghai）。

本轮职责是设计、现状评估和交接。没有实现新的产品功能，没有恢复或激活真实数据，没有联网更新、apply、提交、推送或发布。本目录中的 B00–B07 是待实施计划；现有功能以代码和独立运行证据为准。

## 1. 结论与目标

项目已经有 M1–M12 研究引擎，具备历史回测、本地预测、赛果导入审查、旧工作流结算、模型状态快照及版本关联检查。下一步的主要工作是把这些能力交付成一个轻量、可复现、可结算的 baseline，而非重新实现 M11/M12。

本计划默认交付 **baseline-v0.1：轻量离线研究闭环**。用户端拿到本地运行包，明确给出无比分赛程后，可以加载模型状态得到 1X2 概率与证据；结果公布后，用独立的本地结果文件结算原预测；研究端保留完整历史和更新流程。Alpha、近期信息与页面的完整路线也已规划，但不阻塞 v0.1。

这是尚待用户补充的范围假设。若希望 baseline 本身包含 Alpha 或页面，按主计划末尾的扩展路线升级交付范围；环境恢复、证据契约和快照闭环仍先执行。

## 2. 从哪里读起

| 文件 | 用途 |
|---|---|
| [MASTER-PLAN.md](MASTER-PLAN.md) | 产品边界、系统设计、数据契约、执行顺序与 M13–M17 路线 |
| [AGENT-TASKS.md](AGENT-TASKS.md) | 可直接交给实现 agent 的任务卡、文件范围、命令与交接模板 |
| [EVALUATION.md](EVALUATION.md) | 现状评估、风险分级、独立验收矩阵、研究指标与通过规则 |
| [ZCODE-RULES.md](ZCODE-RULES.md) | 一个会话执行一个阶段、同项目新会话接续与高质量交付规则 |
| [现有 AGENTS.md](../../AGENTS.md) | 已生效的工程、时序、数据与本地 Git 规则 |

新入口以 `D:\matchquant\worktree` 为代码根目录。`D:\matchquant` 是迁移包与交接容器，不是 Git 仓库。`_unpacked\project` 是旧包展开的项目副本，不是本轮工作代码。不要根据目录名或旧 Markdown 的阶段标题判断完成状态。

## 3. 已核实进度

评估起点提交：`5aad2f1b3d2e8e007a3f7f9489d7a73f44ac4c94`。开始评估时 worktree 无未提交改动。

| 阶段 | 已有能力 | 本轮判断 |
|---|---|---|
| M1 | Frequency / Elo / Poisson；同批次先预测后更新；Log Loss / Brier / Accuracy | 已实现，合成 CLI 冒烟通过 |
| M2 | EPL 结果下载、不可变原始文件、canonical 转换、质量审计 | 已实现；迁移包有 3800 场历史 |
| M2.1 | staging、dry-run、addition-only apply、版本与 current pointer | 已实现；本机正式 data root 尚未恢复 |
| M3 | 时序验证、候选选择、独立 final test、校准诊断 | 已实现；旧回测证据是历史记录 |
| M4 / M5 | 历史重放预测、无比分本地赛程快照 | 已实现；M4 的 Alpha 是流程阶段称谓，不是 M13 因子引擎 |
| M6 / M7 / M8 | 旧 M4 结算、本地结果封存、预测发布审计 | 已实现；M6/M8 仍依赖 M4 产物契约 |
| M9 / M10 | 本地赛前会话、赛后 dry-run / 显式 apply / 结算 | 已实现；未接通 M11 输入 |
| M11 | JSON 模型状态构建与只读快照预测 | 已实现；三个模型本轮合成对比差值均为 0 |
| M12 | 显式快照与 current pointer 的 fresh / stale / unverifiable 检查 | 已实现；不能把 fresh 理解为数据实时或完整训练到今天 |
| M13–M17 | Alpha、近期信息、融合、用户体验、发布验收 | 旧路线中的后续设想；没有对应实现 |

独立全量测试：153 tests，33.111 秒，全部通过。评估解释器为 Codex bundled Python 3.12.14，numpy 2.3.5，pandas 3.0.1。直接调用 PATH 上的 `python` 会进入另一个缺少 numpy/pandas 的环境并失败；这次通过不等于项目已具备独立安装环境。

证据在仓库外：

- [测试日志](../../../assessment/2026-10-04/offline-tests.log)
- [结构化评估与合成冒烟结果](../../../assessment/2026-10-04/assessment.json)
- [本轮评估脚本](../../../assessment/2026-10-04/evaluate_baseline.py)

这些是评估材料，不能作为产品依赖或复制真实历史的依据。脚本仅在评估目录生成合成 manifest、快照和运行结果。

## 4. 数据现状与迁移差异

迁移展开的数据位于 `D:\matchquant\_unpacked\local_state\matchquant_data`；正式约定目录是 `D:\Codex\matchquant_data`。本轮检查时，旧 manifest 指向的正式 canonical 路径和正式 `epl_current.json` 都不存在。

历史 canonical：2015-16 至 2024-25，3800 场；cutoff 为 `2025-05-25`。文件实际 SHA-256 与 manifest 一致：

```text
9161416d01d78f3b1311b0bf49a67ce3df3fc05bfbf124644a3c6b820e0c36eb
```

旧 M2 processed manifest 未包含 `dataset_version`，其路径绑定旧正式目录。M11 build 会严格检查传入 history 路径是否匹配 manifest；不能直接把展开路径代入旧 manifest 并声称迁移完成。

包中 2025-26 staging 的旧 diff 为 added=380、unchanged=3800、revised=0、missing=0。它只是该旧快照的审查记录，不能由文件夹日期推断为今天的最新真实赛果。本轮未对它执行重新解析或 apply。

`experience_20260906` 的 M10 报告状态为 `needs_review`，M2.1 为 dry-run，M6 为 `not_run`。旧说明写的“最小闭环具备”是能力层面的描述，不能被解释为这份真实体验证据已完成结算或实时盲测。

旧 ZIP 对应 `aa9556f`，M11 尚为计划；当前代码另有 `82e7e2f` 的 M11 实现和 `5aad2f1` 的 M12 实现。根目录旧 follow-up 文档同时写“稳定 M1–M12”与“先执行 M11”，存在状态混用。本目录作为后续工作的最新设计入口，旧包保留原样作为历史证据。

## 5. baseline 的完成定义

必须同时通过：

1. 独立安装环境可以按说明运行全量离线测试与 CLI 冒烟。
2. 研究端输入的数据路径、hash、版本和 cutoff 可审计，原始迁移证据不被改写。
3. 用户端只加载运行包与无比分赛程；预测不读取完整历史，不联网。
4. 原预测能直接进入快照审计和结果结算，schema、hash、身份与时间规则连续。
5. 快照包可移动到另一目录，断开研究端文件后仍可预测；移动不会破坏来源追踪。
6. 全流程失败、未结算、stale 与不可核实状态明确；已有预测不被新结果回写。
7. 有合成闭环、真实历史回溯、同一输入的重复运行以及交付安装的独立验收证据。

工程完成不要求新因子提升成绩，也不把预测准确率或一次回测收益写成验收承诺。v0.1 可在历史回溯模式交付；缺少当前赛程与数据覆盖证据时必须保持 `research build`，不能称为当前可用预测产品。

## 6. 执行顺序和职责

```text
B00 环境与现状冻结
  → B01 正式数据恢复与版本建立
  → B02 M4/M11 预测证据与 M6 结算桥接
  → B03 快照会话与只读审计
  → B04 可迁移轻量运行包
  → B05 安装、示例与最小 CLI 体验
  → B06 真实回溯及工程独立验收
  → B07 交付归档
```

实现 agent 按任务卡做代码、测试、文档及提交候选；设计/评估 agent 负责审查契约、检查证据、复跑验收并给出 PASS / REWORK / BLOCKED 结论。每个实现会话只执行一个阶段，交付后评估，PASS后在同项目新会话执行下一阶段；未通过不进入依赖它的下一阶段。

本轮不启动实现 agent、不创建新 chat、不代发任务。可直接复制 [AGENT-TASKS.md](AGENT-TASKS.md) 中的首轮提示词交给你选定的 agent。
