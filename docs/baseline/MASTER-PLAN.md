# MatchQuant baseline 完整设计引导书

设计版本 v1.0，2026-10-04。代码起点与现状见 [入口](README.md)，执行粒度见 [任务卡](AGENT-TASKS.md)，验收标准见 [评估规程](EVALUATION.md)。本文件出现的新增模块、schema 与命令均是设计目标，不代表已经实现。

## 1. 交付体验与范围

baseline-v0.1 的使用者只需要安装环境、一个指定的轻量运行包、一个无比分赛程文件和输出目录。预测给出主胜/平/客胜概率、最可能结果、模型版本、训练信息截止时间与审查状态。得到赛果后，使用原预测与独立结果文件生成结算覆盖率及概率评分。

研究端负责保管十年历史、验证数据和重建模型状态；用户端不携带 canonical history、raw 下载文件或训练记录全集。信息不充分时输出明确状态。v0.1 的首次真实数据验证采用历史回溯，不要求立刻接入 2026-27 实时数据。

保留现有 Frequency、Elo、Poisson 模型，正式运行包只选择一个 primary model。其他模型可在研究报告中作对照。这样不需要在首版同时改造多模型同 fixture 去重、每模型结算和 UI 比较逻辑。

首版不新增 Alpha、新闻、赔率、机器学习、数据库、自动调度或网页。它们进入第 9 节的后续研究路线。既有 M1–M12 接口与旧文件保持可读，不重编号、不伪造旧 schema。

## 2. 系统分层和责任

```text
研究端
不可变 raw → M2/M2.1 canonical + manifest + current pointer
                         ↓ 显式 history、manifest、as_of、模型配置
                    M11 snapshot build
                         ↓ M12 版本关联审查 + 新运行包审计
                    portable runtime bundle

用户端（预测期间没有历史文件读取）
runtime bundle + score-free fixtures
    → snapshot-session → probabilities + session/evidence/report
                                   ↓ 独立 canonical results
                                settle → coverage + metrics + report

赛后维护（与上述结算分开）
local E0 results → M7 → M2.1 dry-run → 显式 apply → 新版本
    → M12 status → 显式 M11 rebuild → 新 bundle
```

结算可以只读显式结果 CSV，不要求推进 active history；历史更新继续走已有 M7/M2.1。任何预测、审计、结算命令都不得自动 apply、联网补数据或重建模型。新结果只影响未来新快照，不能改写旧会话。

M8 原本的发布审计要读取 M4 history。为快照运行新增专门的审计入口，不能简单改参数后复用 M8 并留下隐含 history 依赖。M9/M10 保持旧流程兼容；首版新增独立快照会话，避免首轮同时重写它们。

## 3. 时间、版本与状态的契约

### 3.1 必须分别记录的时间

| 字段/概念 | 含义与规则 |
|---|---|
| raw retrieval / local archive time | 获取或本地封存时间；不能用封存时间伪装上游获取时间 |
| history data cutoff | 输入 canonical 中最新的比赛时间，可能晚于本次回溯 as_of |
| used data cutoff | 实际进入模型状态的最后一个历史时间批次 |
| snapshot as_of | 状态边界，严格早于它的合法结果才能更新模型 |
| fixture date | 目标比赛时间，不早于 as_of；不能含真实比分 |
| generated_at / audited_at | 生成或审查时间，不代表上游数据完整覆盖至该时间 |

日期输入排除该日所有赛果；精确时间输入排除该时间批次。日期和精确时间混用造成先后不明时拒绝。每批先全预测再更新，不随机洗牌。赛季由输入指定，不从月份猜测。

M12 `fresh` 仅证明 snapshot 记录的历史证据与指定 pointer 匹配。一个从同一 CSV 取早期 as_of 的回溯快照也可能 fresh，所以必须同时展示 as_of 和 used cutoff。fresh 不证明训练到今天、数据源最新或近期比赛无遗漏。

### 3.2 两种运行模式

- `replay`：为可复现历史研究运行。允许明确选择旧快照与历史赛程，但报告必须标记 retrospective/replay；fixture 仍须满足边界规则，不能使用比分更新。
- `research`：本地待赛研究运行。必须显示 bundle 的发行审查时间、快照 as_of、used cutoff、是否检查过 active 关联及缺失信息。未证明覆盖至目标预测时点时为 `needs_review`，不得把缺口静默当成“没有新比赛”。

首版不提供“live”状态。用户端没有研究端 pointer 时，显示 `active_link_status=not_checked_in_runtime`，可以附带 `fresh_at_issue` 证据，但不得称其当前 fresh。未来若需要当前运行门槛，再制定有来源覆盖证据的更新策略；不要用任意 gap 天数替代赛果完整性审计。

### 3.3 用户状态和退出码

新快照会话采用 `ready / needs_review / blocked`；`ready` 表示该模式下完整通过工程审查。replay ready 仍是回溯研究。`needs_review` 可保留概率和理由；`blocked` 不生成可发布预测。

结算继续使用 `complete / partial / needs_review`；零 settled 时概率指标为 null，保留总预测数与 coverage。禁止把“未结算”填成零损失或完整成功。

新增命令建议退出码：0 为完成且 ready/complete；2 为完成但 needs_review/partial；1 为输入/IO/校验失败或 blocked。Python API 返回结构化状态。旧 CLI 退出码不在首版强制变更。M12 CLI 的成功运行和分类要分别处理，不能只据退出码推断 fresh。

## 4. 新增证据与公共接口

### 4.1 单模型预测证据适配层

当前 M6 `load_forecast_run` 只接受 `m4-forecast-v1`，M11 使用不同 schema，且模型证据为列表。设计一个显式按 schema 分派的 `load_prediction_evidence(run_dir)`，为 M4 与单模型 M11 产生统一内部对象：

```text
producer_schema / producer_command / run_dir
run_config_path + actual SHA-256
predictions_path + recorded/actual SHA-256 + validated table
model_name / model_config / model_config_sha256
as_of(value, timestamp, date_only)
origin_evidence（M4 history 或 M11 snapshot/state 的来源记录）
hash_evidence_status / limitations
```

M4 分支复用旧校验，保留 legacy unrecorded 标记；M11 分支必须验证 run hash、原 snapshot/state、配置、as_of 与模型对应。首版新会话和运行包只允许一个模型；多模型 M11 输入结算时给出明确错误，不能静默取第一模型。

结算复用 `settle_predictions` 的身份、改期、ambiguous 和 time_conflict 规则。输出中的 producer schema 写实际值，不能继续硬编码 M4。禁止把 M11 run_config 改成 M4 schema 来让旧校验放行。

### 4.2 快照会话

拟新增 API：

```python
run_snapshot_session(
    snapshot_manifest_path, fixtures_path, *,
    mode="replay", output_root, issue_evidence_path=None
) -> dict
```

拟新增 CLI：`snapshot-session --snapshot-manifest ... --fixtures ... --mode replay|research --output-root ...`。

工作顺序：验证输出路径 → 创建独立会话 → 校验单模型快照与赛程 → 调用现有 M11 predict → 只读审计证据 → 输出会话与中文报告。每个阶段保存完成/失败状态、原因和产物 hash。失败要保留已产生的证据，但不把部分产物标记为可用。

预测会话不调用 M12；M12 需要 pointer 及其历史文件，应在研究端发行时执行。若输入 issue evidence，只验证其 schema、hash、指向关系和时间，展示其历史状态，不重新读记录里的研究端路径。

输出：`session_manifest.json`、`session_report.md`、不可变 M11 run 目录引用、`prediction_audit.json/md`。manifest 记录 mode、原始输入/输出 hashes、as_of/cutoff、primary model、状态、原因、不变更声明及实际验证结果。自我声明不是测试证据。

### 4.3 快照运行审计

建议 API：`verify_snapshot_run(run_dir, *, output_root, issue_evidence_path=None)`。

检查 snapshot/state/schema/config/hashes、fixture 输入和预测行一一对应、概率合法、时间边界、单模型约束、输入修改、输出路径隔离。对比 snapshot 重新 predict 应在隔离临时输出完成，不能覆盖原运行。

它不需要 history，不执行历史训练正确性的重新证明。历史训练证据和 M4 等价性由研究端/测试端验收。缺少历史发行证据时报告 needs_review，而非伪造 lineage。

## 5. 可迁移运行包

### 5.1 内容与大小

建议结构：

```text
bundle_manifest.json
snapshot/<保留原 snapshot_id>/snapshot_manifest.json
snapshot/<保留原 snapshot_id>/model_states.json
evidence/issue_audit.json
evidence/selection_summary.json（如有）
team_index.json
RUN.md
```

模型仅一个。team index 从已验证状态导出，记录来源 hash；它帮助输入校验，不证明某队仍参加当前赛季。Elo/Poisson 对新队采用现有先验，报告 `unseen_team_prior`；不得静默改球队名称或猜别名。Frequency 没有球队学习状态，应明确其整体频率性质。

运行包不含 raw CSV、canonical history、完整候选预测或个人机器目录的可执行脚本。体积目标 ≤5 MiB，属于产品约束，不保证通过压缩做到；超限须解释实际字段与增长原因。安装 wheel/依赖另算，运行包大小与软件安装大小分开报告。

### 5.2 迁移策略

旧 M11 snapshot_manifest 内 state 路径可能是绝对路径，直接 zip 后搬走无法保证可读。首版采用**派生封装**：

1. 原始快照目录只读，不改写。
2. 在新 bundle 下保留 snapshot_id 子目录，将 state bytes 原样复制。
3. 使用现有 manifest writer 生成派生 manifest，仅将运行所需 state/outputs 路径改成包内相对路径；history、配置来源与 selection 原始路径保留为 provenance，不读取、不把它们包装为存在于用户端。
4. bundle manifest 记录原始 manifest 文件 hash、原始 content hash、派生 manifest hash、相同 state hash、允许修改的 JSON 字段清单及 schema 版本。派生 manifest 是新文件，不宣称与原始字节相同。
5. 重新加载派生快照并与原始预测对照；复制到不同绝对目录、断开原研究端文件后复测。

M11 snapshot_id 的目录约束和内容 hash 规则必须继续通过。若当前 loader 不能接受包内相对路径，只在独立 loader/新包 schema 中适配，不能降低旧 M11 hash 校验。

M12 版本关联审计在原研究端上完成；用户端读取发行摘要。派生 manifest 增加路径或 hash 差异，不得修改 M12 的 fresh 规则来强行通过。

bundle 自身列出每个必需文件的相对路径、大小与 SHA-256，拒绝路径越界、缺失、额外可执行对象及 hash 不符。hash 证明文件一致性，不能代替发行来源认证；首版本地交接不引入签名系统。

拟新增：`runtime-bundle-build --snapshot-manifest ... --issue-audit ... --output-root ...` 和 `runtime-bundle-verify --bundle-manifest ...`。最终 query 入口允许 `snapshot-session --bundle-manifest ...`，它与裸 `--snapshot-manifest` 互斥。bundle build 是独立发行行为，预测不自动打包。

## 6. 数据恢复和未来维护

优先按旧 manifest 恢复到 `D:\Codex\matchquant_data`，只复制到确认空的目标目录；原 `_unpacked` 归档保留。复制后核对全体 manifest/hash/行数，再开展历史回测。不能把原路径字符串批量替换掉当作正式迁移。

已有目标数据时停下覆盖步骤，生成差异和目标路径方案；如须迁移到新根目录，增加显式、带 parent hash 的新版本重建计划，由 M2 prepare 或专门转换工具产生新 manifest，旧证据不改。

旧 M2 manifest 无 dataset_version 不可手填。需要 M12 fresh 验收时，通过现有 M2.1 dry-run 生成质量/diff，审查后显式 apply 得到合法版本/pointer。apply 在实现轮仍需符合已有授权及 AGENTS 规则；本轮设计不构成对真实 active 数据写入的授权。合成隔离测试可自己构建测试 pointer，不能挪作真实来源证据。

原 2025-26 staging 不等于今天最新数据。需要当前数据时，先列请求赛季、来源、目标位置、预期 cutoff 与进行中声明，执行显式 data 命令；完整赛季要求380场，进行中只能显式声明一个赛季。revised/missing 继续阻断 apply，不能新写“自动接受修订”通道。

赛果维护循环：结果封存 → dry-run → 审查 → apply → M12 检查旧快照 → 显式重建 → 新发行包。旧运行、旧 bundle、原数据版本保留可重现。

## 7. 技术选型与执行次序

沿用 Python、numpy、pandas、unittest、标准库 JSON/CSV/pathlib。不新增模型依赖。安装优先建立项目 .venv，记录验证过的 Python 和库精确版本，不依赖用户全局 PATH 或本轮 Codex runtime 路径。

| 阶段 | 主要价值 | 输入依赖 | 完成证据 |
|---|---|---|---|
| B00 | 固定当前真值与可独立复跑环境 | 当前 worktree | 环境记录、全量日志、CLI smoke、起点 commit |
| B01 | 恢复数据链、建立合法版本 | B00 | hash 清单、质量报告、dry-run、合法 pointer 或明确暂缺 |
| B02 | M11 → M6 原预测结算 | B00 | 两种 producer 校验、回归、篡改/时间测试 |
| B03 | 快照预测与审计会话 | B02 | 成功/失败会话证据、无 history/network 验证 |
| B04 | 用户端搬家后仍可运行 | B03；B01 发行证据 | bundle lineage、迁移验证、无 history 文件测试 |
| B05 | 从安装到一次闭环的明确操作 | B04 | 安装/快速开始、CLI 示例、错误信息、版本标识 |
| B06 | 工程与模型表现分开验收 | B01–B05 | 独立评估矩阵、真实回溯、性能与限制报告 |
| B07 | 固定最终可交付材料 | B06 PASS | 代码版本、清单、checksums、移交包与已知限制 |

B01 与 B02 在设计上独立，但默认串行交接，避免多个 agent 同时修改共享文件和 active 数据。每个任务以 1 个可独立审查的改动为单位；不要给一个 agent“做到最好为止”的无边界指令。阶段评估失败返回原任务修正，不跳到新模型研究。

## 8. baseline 发布标准

v0.1 必须有合成端到端闭环和真实历史回溯。新运行包的概率结果与对应 M4 参考相同，容差 `atol=1e-12, rtol=0`；状态数据一致，元数据时间/目录可以不同。不能要求完整运行目录字节一样。

工程 PASS 允许 primary model 的效果未超越旧模型；这版的价值是可运行、轻量、证据连通和可迁移。新因子/融合若未通过研究门槛，生产配置仍使用原 baseline。

真实盲测必须在未来未见结果时锁定预测文件和配置，之后才能评估。已有回溯数据只能验证实现与重现已知指标，不能重新叫 untouched holdout。release manifest 必须写 research build、用途、数据 cutoff、scope、限制和缺失的实时覆盖证据。

## 9. 保留原设想：M13–M17 的完整扩展路线

### M13：结果型 Alpha 因子

进入条件：B06 通过，现有 baseline 固定。先做与来源结果一致的因子，不混入新闻。

首批定义：近5/10场积分率（胜3/平1/负0除以3n）、每场进/失球均值、主/客拆分收缩均值、date-only 休息天数、赛前对手 Elo 的近期均值。显式记录窗口、最小样本、收缩 prior、缺失值、计算版本、可用时间点。休息天数不能从日期差冒充精确小时。

因子状态独立于 canonical M1 schema；按批次在预测完成后更新。导出有限、明确的 JSON 状态，保留未加入因子的 baseline 对照。禁止把 Elo 数值当作“已经验证的 Alpha”。

交付：factor definition/config、状态构建与查询、时间变异测试、与 baseline 相同 fixture 列表的 walk-forward 表、分赛季分桶指标。初期可仅作解释不改变概率；改变概率留到 M15。研究证据不充分时因子维持实验标记。

### M14：本地近期信息包

进入条件：时间可用性契约已审查。先支持用户提供的 JSON/CSV，不依赖联网。

信息 schema 至少包含 event/fixture identity、source label/URL、raw hash、published_at、retrieved_at、available_at、valid_until、parser version、结构化内容和冲突/缺失状态。`available_at` 取发布时间与本次实际获得时间中的较晚者；无法证明过去可获得的信息只用于以后锁定的预测，不倒灌历史回溯。

raw 与解析结果分离保存，原文不可覆盖。伤停或阵容条目只有存在明确来源、对象、时间和事实状态才纳入；无信息、过期、冲突与解析失败是可区分状态，不用“零伤停”代替缺失。

交付：本地 import/validate/query 命令、信息包 manifest、expiry/conflict 测试、完全无信息回退。联网 adapter 后续单独立项，先验证来源许可、可获取字段、费用、失败行为与原文可存储性；不要先承诺某个来源永远可用。

### M15：透明融合与概率校准

进入条件：因子与信息有可回放、满足赛前可用性的样本。先冻结特征，再选择小范围融合候选。

首个候选采用有显式约束的概率混合：`p=(1-w)*p_baseline+w*p_candidate`，w 仅在 validation 选择；无合法 candidate/info 时 w=0。若需要训练 candidate，则单独说明模型、正则化、样本量和新增依赖。可先只用 M13，不强制新闻改变概率。

相同时间窗口比较 baseline、Alpha-only、info-only、full 四种消融；无信息行保留，不能按结果删除失败样本。选择规则使用 pooled Log Loss，Brier 辅助，校准单独诊断。校准训练只用预测时点之前的数据。研究未过门槛，正式 runtime 仍发行 baseline。

交付：固定候选清单、训练/验证/测试合同、选择与消融报告、配对置信区间、缺失/漂移报告、fallback 等价测试。M15 的概率合法性不能被写成校准证明。

### M16：本地用户体验

进入条件：B05 CLI 已验收、运行包契约稳定。优先做本地页面，保留 CLI 作为可复现入口。

页面用户输入主/客队、赛季、目标日期与明确快照；展示三类概率、as_of/used cutoff、来源审查、缺失/过期信息与导出审计包。页面调用同一公开 API，不自写另一套预测，不自动刷新网络或覆盖历史。训练端路径细节放在可展开证据中。

交付：页面设计、表单校验、状态映射、与 CLI 同输入等价验证、新电脑操作演示、无服务时的离线错误处理。首版不引入账户、云数据库、多人平台。

### M17：完整设想发布验收

对 M13–M16 实际进入交付的部分重新运行 B06 型验收，补充信息时间泄漏、过期/冲突回退、页面/CLI 一致性与来源许可检查。更新发行清单、版本、迁移说明及限制。未通过研究门槛的模块可保留实验入口，默认用户配置回退 baseline；未完成模块不列为已交付。

如果用户选择“完整研究闭环”作为 baseline，则在 B07 前插入 M13→M14→M15，并按门槛决定采用或回退；若选择“本地产品原型”，再插入 M16。即使融合回退，信息契约、页面和失败处理也须完成才能声称对应层级交付。
