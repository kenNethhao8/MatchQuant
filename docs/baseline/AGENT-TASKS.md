# MatchQuant 实现 agent 执行任务卡

设计 v1.0，2026-10-04。先读 [入口](README.md)、[主计划](MASTER-PLAN.md)、[评估规程](EVALUATION.md) 和仓库 `AGENTS.md`。本文件是执行说明，未勾选条目均未在本轮实现。

## 1. 通用执行合同

- 唯一实现代码根目录：`D:\matchquant\worktree`。起点 `5aad2f1b3d2e8e007a3f7f9489d7a73f44ac4c94`；开始时再次核对 HEAD 与未提交改动，保留别人的修改。
- 每个实现会话只负责一个阶段，遵守 [Zcode会话与交付规则](ZCODE-RULES.md)。阶段通过独立评估后，在同项目新会话执行下一阶段；大需求先拆解、另开会话。发现跨任务依赖，记录并返回设计评估，不顺手扩张为新模型或新产品。
- 先确认公开接口和失败规则，再写必要的契约/回归测试。沿用 unittest；不为纯文字改动堆积测试。
- 不更改 M1 canonical schema、同日先预测后更新语义、M2.1 修订阻断与显式 apply 规则；旧 M4/M9/M10 输入继续兼容。
- 数据和运行产物分别写入明确 data/run roots；不把第三方原始文件、完整真实数据、虚拟环境或大型输出提交 Git。
- 不推送、发布、创建远程 release 或合并他人分支。完成实现后交付可审查 diff/本地提交候选，由用户与评估方决定后续动作。
- 旧计划引用的 superpowers 执行工具不是当前任务的安装前置条件；本轮不要求或自动启动多 agent，按清晰任务卡即可执行。
- AGENTS 中 M2 的阶段限制用于 M2，不能误读为新批准 baseline 永远不能做快照会话；M13 等仍按阶段门槛另行进入。
- 每个任务交付实际命令、退出码、日志、产物路径/hash、剩余限制；manifest 写 `network_accessed=false` 不足以证明没有网络。

建议新任务证据放在 `D:\Codex\matchquant_runs\baseline_acceptance\<task-id>_<UTC>_<id>`。B00 未恢复正式 roots 前，可使用仓库外明确的评估临时目录。每次唯一目录，不覆盖本轮 `D:\matchquant\assessment\2026-10-04`。

## 2. B00 — 固定起点和独立安装环境

**依赖：**无。**产物：**environment.json、dependency lock/constraints、测试日志、合成 CLI run、差异/问题清单。**范围：**`.gitignore`、安装约束文件、README 安装段、必要的环境文档。初次运行若出现代码缺陷，先提交报告，不在环境任务内大改模型。

- [ ] 核对代码目录、HEAD、Git 状态、Python 路径、版本和 PATH 歧义。
- [ ] 创建项目专属 `.venv`，安装 editable project；锁定实际验证的依赖。避免只使用 `numpy>=...`/`pandas>=...` 然后声称所有未来版本兼容。
- [ ] `.gitignore` 增加 `.venv/`，锁文件可以提交，虚拟环境不能提交。
- [ ] 跑完整离线测试，记录实际测试数和通过情况；不得 skip 旧失败测试来凑通过。
- [ ] 运行 M1 synthetic CLI，验证三个模型的概率、指标和输出文件。
- [ ] 记录干净环境可复跑命令及 wheel/离线安装是否已验证；未验证时标明。

已有命令（在 repo 中执行，前提是解释器确实存在）：

```powershell
py -0p
git -c safe.directory=D:/matchquant/worktree status --short
git -c safe.directory=D:/matchquant/worktree rev-parse HEAD
# 选择已确认版本的 Python 建 .venv；不要假定 py -3.11 已安装。
.\.venv\Scripts\python.exe -m pip install -e .
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe -m matchquant run --input examples\synthetic_matches.csv --test-start 2024-01-07 --model all --output-dir <唯一合成运行目录>
```

**门槛：**独立 .venv 安装成功；全量测试全部通过；合成 CLI 退出0，三份模型结果均存在；日志记录解释器路径与依赖精确版本。不要把 bundled runtime 的153项通过替代本任务。

## 3. B01 — 恢复正式数据与建立版本链

**依赖：**B00。**范围：**恢复操作说明、hash/质量审计；必要时新增显式迁移 spec；不改旧 raw、旧 canonical 或旧 manifest。

- [ ] 盘点 `_unpacked\local_state` 的 source/processed/staging/run 文件与指向路径。
- [ ] 检查 `D:\Codex\matchquant_data` 和 `matchquant_runs` 是否为空/不存在；若非空，先出冲突报告，禁止覆盖。
- [ ] 在未来实现轮已有授权支持恢复的条件下按原根目录复制归档；原目录保留，核对全部必要文件 hash。复制只是恢复，不伪装新 source retrieval。
- [ ] load/validate canonical，确认 3800 行、10个赛季、cutoff和全季门槛；检查 source manifest 对原始 bytes 的关联。
- [ ] 按现有 M2.1 对明确的 staging 做离线 dry-run，检查 added/revised/missing、各赛季声明与质量。不要把旧 diff 直接当成新审查结果。
- [ ] 要建立真实 current pointer 时，将该轮质量/diff和精确 apply 命令作为独立审查产物；遵守该轮授权，不因这份计划自动推进真实 active 数据。
- [ ] 不伪造 `dataset_version`。缺少合法 pointer 时明确记录；合成 pointer 只用于隔离测试。
- [ ] 选定真实回溯的数据根、history/manifest路径和hash，冻结供B06使用。

现有命令示意（占位符由审查后的实际路径替换）：

```powershell
.\.venv\Scripts\python.exe -m matchquant epl-sync --staging-dir <明确的本地staging> --data-dir D:\Codex\matchquant_data --dry-run
# 下述是独立显式变更动作，只有该轮实际授权及审查通过后才执行。
.\.venv\Scripts\python.exe -m matchquant epl-sync --staging-dir <同一staging> --data-dir D:\Codex\matchquant_data --apply
```

**门槛：**路径与 hash 一致；正式数据可读；若激活则pointer与version一致且原资产不变；若暂未激活则只能给 partial，通过合成路线继续B02/B03，B04的真实发行和B06真实闭环等待合法数据版本。

## 4. B02 — M4/M11 预测证据和结算桥接

**依赖：**B00。**建议文件范围：**新增 `matchquant/prediction_evidence.py`、必要的M11校验模块；修改 `settlement_validation.py`、`settlement.py`、相关report；新增兼容/篡改/时序测试。旧公开API可保留包装，分派不得循环调用。

- [ ] 写统一内部 evidence 对象，按实际schema分派M4和单模型M11。
- [ ] M4分支保留全部旧约束、legacy hash标志和错误语义。
- [ ] M11分支校验原快照/state、model/config、as_of、fixtures与预测hash；未知schema拒绝。
- [ ] 保留实际 producer schema/command，不创建假M4 run_config。
- [ ] `settle_forecast_run` 接受两种producer，复用现有settle算法；报告中的source schema不硬编码M4。
- [ ] 新路径必须有完整hash证据；不适用的M4字段不能用dummy值凑齐。
- [ ] 首版拒绝多模型M11结算，给出明确说明；fixture唯一性规则仍严格。
- [ ] 覆盖正常结算、部分/零结算、重复identity、改期、时间冲突、预测文件修改、snapshot文件修改、未知schema与旧M4回归。

**门槛：**原M4→M6测试保持通过；同一M4与M11输入得到相同预测及结算指标；所有篡改与未来信息用例拒绝；不读取history文件完成M11证据校验；全量测试通过。

## 5. B03 — 快照会话与只读审计

**依赖：**B02。**建议范围：**`snapshot_session.py`、`snapshot_run_verify.py`、对应report/support、`cli.py`、`__init__.py`、必要 tests与README。不要把新代码塞进旧M9/M10一长段分支。

- [ ] 实现主计划中的 `run_snapshot_session` 与 `verify_snapshot_run`。
- [ ] CLI输入明确score-free canonical fixtures；E0 pending转换继续作为单独M5步骤，不强制首版包装它。
- [ ] 限制单模型，mode为replay/research；输出状态与退出码一致。
- [ ] M11 predict后审查完整产物/概率/行对应/hash/边界，再生成会话。
- [ ] 捕获预期校验/IO失败，保留结构化失败原因；不吞异常并标记成功。
- [ ] 快照与输入赛程不变；输出root在snapshot/输入归档内时拒绝。
- [ ] 真实发行证据缺失时明确needs_review；replay合成验收允许标记synthetic-ready，不能混成真实发行。
- [ ] 会话可引用B02结算；结算不要求先apply结果，也不要求用户端有history。
- [ ] 测试网络调用和history读取都为0，测试实际访问行为而不只看manifest自述。

设计目标命令（尚不存在）：

```powershell
matchquant snapshot-session --snapshot-manifest <明确manifest> --fixtures <无比分CSV> --mode replay --output-root <run-root>
matchquant snapshot-run-verify --forecast-run <M11-run> --output-root <audit-root>
matchquant settle --forecast-run <该M11-run> --results <独立canonical-results> --output-root <run-root>
```

**门槛：**一次synthetic snapshot→session→audit→settle成功；无比分/日期/hash/缺失/IO/输出路径失败有明确证据；原M9/M10全量回归通过。

## 6. B04 — 可迁移运行包

**依赖：**B03，真实包需要B01。**建议范围：**`runtime_bundle.py`、`runtime_bundle_validation.py`、`team_index.py`、CLI；必要时扩展snapshot session；测试与包schema文档。

- [ ] 实现主计划的派生封装与 lineage 契约，沿用现有M11 writer/codec。
- [ ] 记录原始/派生manifest hashes、state hash、改变字段清单与包内文件hash。
- [ ] 原始manifest/state不修改；派生manifest仅更改必要运行路径，保留历史来源。
- [ ] 所有运行文件通过bundle内相对路径定位，拒绝越界和缺文件；不能从来源绝对路径fallback寻找文件。
- [ ] 提取team index，保留球队源文本；新队先验显示明确，fixture不得假定球队仍参赛。
- [ ] 发行审计包含研究端M12关联结果、as_of/used cutoff、模型配置与测试依据；用户端只展示发行时状态。
- [ ] 新增bundle build/verify CLI，session的bundle参数与裸snapshot参数互斥。
- [ ] 在不同绝对目录解包，暂时移走原snapshot/history/manifest，断网预测与结算，结果仍一致。
- [ ] 比较包内文件清单与内容，确认无第三方raw/canonical CSV，报告文件数及实际字节数。

**门槛：**目标≤5MiB，超限单独评估；移动后无需原研究目录；概率与原快照一致；old M11 loader/schema/hash规则不被放宽；tamper/missing/越界测试拒绝；全量回归通过。

## 7. B05 — 可独立安装与最小使用体验

**依赖：**B04。**范围：**README、pyproject、安装约束/打包清单、CLI帮助/错误文案、小型合成示例与demo说明。v0.1不做网页。

- [ ] 明确安装支持版本，在新venv从最终交付物安装，不仅从源码cwd导入。
- [ ] 检查打包后的examples/config资源可用；目前setuptools只枚举 `matchquant*`，不能假定源码外examples天然随wheel分发。
- [ ] 快速开始按“verify bundle→score-free fixture→session→settle”给出复制可运行命令；准确区分现有命令和新命令。
- [ ] 提供一组synthetic fixtures/results与对应bundle构建说明。数据明确synthetic；同一日期完整预测后再结算，不能让demo自动读取比分。
- [ ] 增加research/replay说明、as_of/cutoff显示、unseen team和missing info标识、导出会话位置。
- [ ] 缺依赖、路径错误、hash错误、比分fixture、旧快照、空结果都有可操作错误文本；命令输出保持简洁。
- [ ] 软件版本与实际功能一致；baseline-v0.1是交付标识，Python包版本由最终发布规范明确，避免只有改版本号就叫完成。

**门槛：**一个新环境只依靠交付说明能完成synthetic闭环；从非源码目录调用CLI成功；用户端不加载历史；帮助和文档没有“已校准”“实时最新”等无证据声明。

## 8. B06 — 独立工程和研究验收

**依赖：**B01–B05。**范围：**独立验收脚本/测试、报告、预先冻结的评估合同；只修验收发现的阻塞缺陷，不在本轮调参追分。

- [ ] 执行EVALUATION的工程矩阵，记录PASS/FAIL/NOT_RUN，所有阻塞项必须通过。
- [ ] 用真实历史同一as_of/config生成M4与M11预测，逐fixture比较；同一批次先预测后统一纳入结果。
- [ ] 做2024-25历史回溯的重复性检查；已曝光窗口标记regression，不充当新holdout。
- [ ] bundle在迁移目录断网运行；hash/文件读取/不可变/无网络检查使用外部观察或patch。
- [ ] 记录安装、build、query、settle耗时和大小；按下面性能合同进行评估。
- [ ] 汇总失败、限制、模式覆盖、数据版本和最后代码hash，提交给设计评估方复核。

**门槛：**EVALUATION所有v0.1必需项PASS；新特性不会改变现有baseline算法；真实数据验证与性能有证据；非盲测数据不写未来性能结论。

## 9. B07 — 最终交付归档

**依赖：**B06 PASS。**范围：**发行清单、checksums、使用手册、代码版本与release report；第三方数据只保留本地指针，除非已确认可分发且用户要求。

- [ ] 固定代码版本、依赖锁定、bundle/schema版本及数据cutoff。
- [ ] 导出小型runtime、synthetic演示材料、安装说明和已知限制。
- [ ] 形成release_manifest：commit、文件hash、适用模式、测试报告路径、真实数据指针、许可检查状态。
- [ ] 更新README和迁移入口：新的工作目录、M11/M12已实现、旧ZIP属于旧代码、apply与最新数据状态。
- [ ] 保留旧迁移文档/包原样；不覆盖审计证据。最终包在全新目录再复测一次。
- [ ] 提交“已交付/实验/未实现”清单和下一步候选，不把M13–M17自动加到已完成。

**门槛：**评估方确认baseline-v0.1 research build可交付。保持本地，发布/上传须另有用户指令。

## 10. 每轮实现 agent 必须返回的模板

```text
任务：Bxx；起点/最终commit：...
完成项：...
修改文件与接口：...
测试：实际命令、解释器、测试数、退出码、日志绝对路径
CLI/闭环：实际输入路径/hash、as_of/cutoff、输出路径/hash、状态
不可变与无网络：验证方式、实际结果
数据变更：无 / 恢复副本 / 显式apply（列授权与精确version/pointer）
限制与未完成：...
需评估的决策：...
交接：HANDOFF.md绝对路径、NEXT-CHAT-PROMPT.md绝对路径
推荐下一任务：...（不自动执行）
```

评估方返回PASS / REWORK / BLOCKED；REWORK列具体缺陷和复测条件，BLOCKED说明缺失输入/授权/外部条件。测试通过只能解决工程问题，不能单独证明整个阶段完成。

## 11. 可复制的第一轮提示词

```text
你负责实现 MatchQuant baseline 计划中的 B00，先不要做 B01–B07。
代码根目录为 D:\matchquant\worktree。先读 AGENTS.md 和 docs\baseline\README.md、
MASTER-PLAN.md、AGENT-TASKS.md、EVALUATION.md，核对当前HEAD及用户修改。
目标是建立独立项目环境、固定依赖和起点证据，完整离线测试与synthetic CLI冒烟通过。
当前设计起点5aad2f1已有M11/M12，不要重新开发它们；PATH上的python缺依赖。
设计评估方已在bundled Python 3.12.14 / numpy2.3.5 / pandas3.0.1中通过153项测试，
但这是评估环境，不能替代你建立的交付环境。
不恢复/覆盖真实数据、不apply、不联网抓取数据、不新增Alpha/新闻/网页、不push或发布。
按任务卡交付可审查改动和实际验证证据，使用第10节模板返回，再等待设计评估。
```

后续阶段总是在同项目的新会话执行，携带HANDOFF、当前任务、最新评估结论、实际代码版本和证据路径。不要仅发“继续做”；同阶段返工可在原会话完成，上下文混乱时按ZCODE-RULES开同阶段恢复会话。
