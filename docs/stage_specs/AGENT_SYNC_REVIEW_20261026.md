# MatchQuant Agent Skill 与数据同步：收敛与验收复核

审查日期：2026-10-26。基准提交：`a020607`，本地分支 `quant-v2`。审查开始时工作区干净。依据为 `MATCHQUANT_AGENT_SKILL_AND_DATA_SYNC_PROMPT.md`、`AGENTS.md`、实际源码、用户提供的反重力交接材料，以及本次独立执行结果。没有以交接文字代替验收，也没有验证其远端 main/push 声明。

**结论：已具备分层实现的雏形，尚未达到“一次配置、预测前可靠更新、Agent 写作与追问、新用户可安装”的完整要求。本阶段不予通过。** 现有单场计算和日期截断仍可使用；新增同步与 Agent 工作流有可复现的核心缺口，不能标为生产可用。

## 1. 需求与实际分层状态

| 层 | 已有实现 | 验收结论 |
| --- | --- | --- |
| 用户配置 | JSON profile、中文别名、环境诊断、刷新/报告偏好字段 | 部分完成。指定 profile 保存路径已在本轮修正；刷新间隔、时区、报告偏好未完整参与运行；默认路径仍偏向开发机。 |
| 数据同步 | 显式 `epl-sync`、独立 staging、dry-run、比分/日期 diff、进行中赛季入口 | 未通过。apply 覆盖旧路径，单分区刷新丢历史，技术统计修订不拦截，无效 pending 与重复校验有缺口。 |
| 数据版本与就绪度 | 显式 snapshot 路径准备、current pointer、富化缺失保真 | 未通过。输出与 manifest 复用固定路径；失败时旧 pointer 的目标文件可已改变；缺少有效源检查年龄判断及完整版本证据。 |
| 单场预测服务 | 复用原模型，日期日初截断、完整赛季拟合、最新部分赛季更新状态 | 原有计算路径及回归测试通过。新增严格历史 vintage 选择未实现，不能把 replay 标签视为真实历史快照重放。 |
| Adapter 与 Skill | JSON 调用入口、源码 Skill、临时复制安装、中文请求操作说明 | 部分完成。wheel 缺少 Skill；缺少字段时猜赛季/截止日；正常结果没有 run_id；非法 profile 可逃逸 JSON 错误接口。 |
| Agent 报告与追问 | 固定 Python 伴生模板、E1–E6 目录、关键词追问且不重算 | 未通过目标。没有宿主 Agent 叙事输入；真实 JSON 字段读取错误，HTML 状态/内容与 Markdown 不一致，归因表达不可靠。 |

## 2. 本次实际检查和证据

本轮证据目录：

`D:\Codex\matchquant_runs\codex_review_a020607_b1_0eb2l`

### 2.1 现有测试与局部修正

- 指定解释器：`D:\Dev\Python314\python.exe`，Python 3.14.7。
- 初次运行完整套件：91 项，90 项通过、1 项错误。`test_t15_skill_installation_and_cli_smoke` 用系统 GBK 解码 UTF-8 子进程输出，产生 `UnicodeDecodeError`，继而 `json.loads(None)`。
- 同时发现 `setup` 更新指定 profile 后调用无参数 `prof.save()`，错误写入用户默认 profile。现有测试只检查响应内对象，没有重新读取指定文件，因此漏报。
- 本轮小范围修正：`adapter.py` 保存到请求的 `profile_path`；`tests/test_adapter.py` 隔离默认配置、断言指定文件实际持久化且未写默认文件、子进程显式使用 UTF-8 解码。
- 两项聚焦检查通过。修正后完整套件 **91/91 通过，28.926 秒**。此次设置独立 `MATCHQUANT_PROFILE_PATH`，没有通过 `PYTHONUTF8` 掩盖原解码缺陷。全量完成后，用户默认 profile 未再次生成。
- 以上通过只证明现有套件通过；下述验收反例没有被原套件覆盖。

日志：[修正后完整测试](D:/Codex/matchquant_runs/codex_review_a020607_b1_0eb2l/suite_after_scoped_fixes.log)、[首次测试](D:/Codex/matchquant_runs/codex_review_a020607_b1_0eb2l/existing_suite.log)。

### 2.2 独立验收反例

根据源码中已识别的可疑路径，编写了 14 个有针对性的验收检查；**14 个均复现未满足要求的行为**。它们是问题反例集合，不是随机抽样的整体成功率。

大部分同步检查使用隔离目录、合成 CSV 和真实同步/准备代码。R08/R09 只验证适配器编排，明确 mock 预测函数；并未把 mock 输出称为新模型运行。R09 还固定就绪度以隔离默认参数行为。真实模型拟合由原有完整测试覆盖。

可重跑脚本：[acceptance_probes.py](D:/Codex/matchquant_runs/codex_review_a020607_b1_0eb2l/acceptance_probes.py)。机器可读结果：[acceptance_probes.json](D:/Codex/matchquant_runs/codex_review_a020607_b1_0eb2l/acceptance_probes.json)。

```powershell
$env:PYTHONUTF8 = '1'
& 'D:\Dev\Python314\python.exe' 'D:\Codex\matchquant_runs\codex_review_a020607_b1_0eb2l\acceptance_probes.py'
```

脚本在审查目录下保留独立样本。重跑前应复制脚本到新的空证据目录，避免把已有探针数据误当作初始状态。

### 2.3 真实来源检查

实际执行了显式数据命令，使用独立数据根目录，**仅 dry-run，没有对用户原数据执行 apply**：

```powershell
python -m matchquant epl-sync `
  --data-root 'D:\Codex\matchquant_runs\codex_review_a020607_b1_0eb2l\live_provider_data' `
  --seasons 2026-27 --in-progress-season 2026-27 --dry-run --timeout 8
```

- 结果：`dry_run_ok`，实际返回 70 场已完成比赛、20 支球队。
- 比赛日期范围：2026-08-21 至 2026-10-19。
- 原始文件 SHA-256：`705ef4373f6c4c9ec09b67eb3ca3398d6a770813095efc5d629edb591e4ee41c`。
- 因显式声明进行中赛季，70 场不应被当作完整 380 场赛季。
- 原交接记录的是此前连接失败；本次来源已可访问。**网络当前可达不能解释或消除本次代码缺口，也不能单凭可达断言截至今天比赛已齐。**

证据：[实际命令结果](D:/Codex/matchquant_runs/codex_review_a020607_b1_0eb2l/live_provider_smoke.json)、[来源内容检查](D:/Codex/matchquant_runs/codex_review_a020607_b1_0eb2l/live_provider_inspection.json)。

### 2.4 新用户安装检查

实际构建 `matchquant-0.1.0-py3-none-any.whl`，安装到独立 `--target` 目录，再从仓库外用 `python -I` 显式加载该安装包。

- Python 包构建和安装成功。
- wheel 内没有 `SKILL.md`、Skill helper 或其 references。
- 对该已安装包调用默认 `install_skill(...)`：`FileNotFoundError`，因为它寻找安装位置旁边不存在的 `skills/matchquant`。
- 从源码目录复制 Skill 的测试，未覆盖这个新用户安装场景。

证据：[wheel 安装探针](D:/Codex/matchquant_runs/codex_review_a020607_b1_0eb2l/wheel_install_probe.json)。该 wheel 包含本轮 profile 保存路径小修正；其打包资源缺口与基准提交相同。

## 3. 阻断验收的问题及修复门禁

### P1-1：旧版本文件会被覆盖；失败的 apply 不能保住有效旧版本

位置：`sync.py:362` 将 staging 内容复制到固定 `raw/E0_<season>.csv`；`dataset.py:91`、`:93`、`:151`、`:159` 复用固定 canonical、rich 和 manifest 路径。

R01/R02：进行中赛季由 100 场增加为 101 场后，上次 raw 哈希改变，上次 pointer 记录的 canonical 哈希与该路径实际内容不符。旧 manifest 同样会被覆盖，无法独立复现旧数据版本。

R07：在 rich 写入时注入磁盘错误，pointer 字节确实没变，但它引用的 canonical 已经被写成新内容。仅原子替换 pointer 没有实现“旧版本仍可用”。

修复门禁：新 raw、canonical、rich 和 manifest 均写入新版本目录，完成所有验证后只原子切换 current pointer；失败不得改变旧 pointer 引用的任何文件。旧 M2 文件保持原样。R01/R02/R07 应转为满足。

另一个兼容性问题：保存 canonical 前没有裁剪 `validate_matches` 添加的内部列，输出实际含 `_timestamp`、`_date_only`，与七列 M1 契约及 manifest 的列声明不一致。

### P1-2：只同步当前赛季会丢弃历史分区

位置：`sync.py:186` 只组装本次 target_seasons 的 snapshots，`:366` 将其直接交给全量准备。

R03：先建立三个完整赛季共 1,140 场，再仅同步新赛季 100 场。期望当前版本为 1,240 场；实际变成 **100 场**，seasons 只剩新赛季。原 T4 测试把旧赛季再次全部传入，因此未测试“只请求新分区”的真实需求。

修复门禁：从基准版本 manifest 读取未请求的旧分区，与本次新分区合并准备；diff 只比较对应请求分区。空请求、重复声明、声明非最新进行中赛季等必须明确处理。

### P1-3：addition-only 与数据质量门禁不完整

位置：`sync.py:117` 只比较日期及全场比分；`dataset.py:33` 把非空字符串当作有效日期/球队；sync 未将 audit 的重复记录结果纳入 dry-run 阻断。

- R04：将既有比赛 HST 从 4 改成 5，结果 `applied`、revisions 为 0、unchanged 为 100。
- R05：`Date=not-a-date` 且主客队相同的空比分行被当作 pending 丢弃，结果仍 `applied`、invalid 为 0。
- R06：重复的已完成比赛在 dry-run 被去重进字典，结果仍 `promotion_eligible=true`；直到实际准备才可能报错。
- 380 行检查还不足以证明完整 EPL 赛季满足球队和主客覆盖要求。同步门禁应在准备/生效前一致验证。

修复门禁：对来源完整身份与日期先验证，再区分完成/未完成；跨全部用于模型的技术/市场字段识别修订；重复、半比分、缺失基线、非法 roster/覆盖在 dry-run 即阻断；完整 diff 不应仅保留前 20 条而丢失全部核查明细。

### P1-4：自动刷新、未来请求和历史 vintage 仍未闭环

位置：`adapter.py:194` 按月份推断 season，`:200` 将缺省 as_of 设为比赛日，`:213` 仅根据 ready/is_stale 决定刷新；`sync.py:191` 缺省结束年份固定为 2025。

- R08：源检查日期为 2026-01-01，配置 24 小时检查间隔，比赛数据只落后一天。分析返回 `ok` 且同步调用次数为 **0**。`source_check_interval_hours` 没有实际参与判定。
- R09：2027-01-15 的未来请求省略 season/as_of 时，编排直接向预测函数传入 `2026-27` 和 `2027-01-15`。这个检查隔离了编排行为，不证明真实未来数据已被使用；但它证明没有按任务书取得明确截止日或可配置当前日期政策。
- R10：profile 历史起始季为 2026-27，未声明 in-progress，缺省同步竟返回空 seasons 的 `dry_run_ok`。
- readiness 将 `updated_at` 当作 fetched/source_checked 的后备值，未真正计算源检查年龄；按月份推定休赛期可使过旧数据解除阻断，实际 used cutoff 也可能只是 `< YYYY-MM-DD` 占位文本。
- `single_match.py` 仍只接受 `research|replay`，没有交接材料声称的 `reconstruction` 模式。没有严格 archived replay 的快照选择、取得时间验证或 vintage 状态。当前 replay 仍是晚近 CSV 的日期截断重构。

修复门禁：分别判断源核查时间和实际比赛数据年龄；明确新赛季范围与进行中声明来源；缺参时按已授权的明确政策或一次追问补齐；当前预测不能默认使用未来比赛日作为“现在”；历史回放按任务书区分可验证 archive 和后取快照重构。刷新失败/修订阻断的 stale fallback 必须写进最终报告状态和证据。

### P1-5：新用户的 Skill 安装和宿主使用未通过

位置：`pyproject.toml` 仅打包 matchquant 包及其现有 resources；`skill_installer.py:16` 假设仓库旁边的 Skill 目录仍存在。

实际 wheel 安装后默认 Skill 安装失败，证据见 2.4。helper 的仓库 sys.path 回退层级也不正确，不能代替正式依赖安装。installer 默认 overwrite 并递归移除已有同名 Skill，尚未达到保护已有安装的要求。

修复门禁：选择一个明确可分发方式（包资源或独立、可定位的 Skill 工件），测试从仓库外的非 editable 安装；默认保留已有 Skill，覆盖必须明确；用实际宿主按路径或已安装 Skill 走一轮请求和追问，并区分复制验证与自动发现验证。

### P1-6：伴生报告没有实现 Agent 写作接口，且存在证据与格式不一致

位置：`agent_report.py:30` 读取不存在的顶层 `factors`；`:84` 以原始特征绝对值排序；`:177` 使用“确定性占优区间”；`:235` 固定声称“嵌套交叉验证”；`:322` 只把 `ok` 当完整。adapter 自动调用固定模板，没有 narrative payload 或宿主撰写结果入口。

- R11：真实 Arsenal/Chelsea JSON 的 `features.raw_inference_values.elo_diff=2.490456123582642`，证据目录却得到 null，E3 top_drivers 为空。
- 不同单位的原始因子绝对值不是模型贡献；因子正负也不普遍等于利好主/客，不能将其当作有权重的集成归因。
- R12：权威 JSON 为 `complete`，Markdown 标完整，HTML 却标“需要复核”；HTML 没有近期证据 E4 和市场 E5 等 Markdown 章节。两种报告并未共享同一完整内容。
- 正常 JSON 没有 run_id，adapter 返回 null；现有追问测试同时传 run_dir 掩盖了该缺口。使用 run_dir/report_json 的关键词追问可以工作，单独返回的 run_id 不可用。
- 当前固定模板可以生成概率表和近期统计，不能据此宣称宿主 LLM 已基于预设完成写作。现有“对话实录”记录存在、且 run_id=null，未提供宿主安装/激活证据，不能证明自然语言 Skill 全链路通过。

修复门禁：与真实 JSON schema 绑定证据，保留每项有效样本数；引用现有 attribution 及其权重/局部近似限制；用宿主提供的结构化叙事内容渲染统一 MD/HTML；数值表由权威 JSON 解析；稳定 run_id 和可回溯路径；用真实 schema 的集成样本替代仅自造另一套字段的 mock 测试。

## 4. 其他必须收口的问题

- R13：`execute_adapter_request` 在 try 外 load_profile，非法 profile JSON 直接抛异常，破坏结构化错误约定。
- R14：交接给出的 `python -m matchquant.adapter --action ...` 实际退出 0 但没有输出，因为该模块没有模块入口。可用入口是 `python -m matchquant agent-adapter ...`。README 本轮保留并使用后者。
- adapter 的 JSON 解析错误也需要走结构化错误返回；`needs_review` 与失败的退出码应按已约定接口统一。
- profile 布尔值不应以 `bool("false")` 解析为 true；保存前应验证字段。请求中允许的 overrides 应真正生效，timezone/报告风格/模型标识不能仅储存而不披露未使用。
- 样例配置 auto_refresh 默认打开与首次显式选择的体验需要收敛。
- Python distribution 版本是 0.1.0，运行包自报 2.0.0，需统一版本语义并记录在结果中。
- source manifest/质量报告的 URL、抓取时间、解析器版本、完整 diff、in-progress gate 策略等证据应落盘并与数据版本绑定，不能用 pointer 改写时间代替来源核查时间。

## 5. 本轮实际修改与配置副作用处理

本轮仅修正指定 profile 保存路径、隔离 setup 测试和 UTF-8 解码，并更新 README/任务书状态和本文档。上述 P1 核心缺口保留为明确的后续实现项；没有用文档改写把它们降低为已满足。

首次测试启动前未隔离用户默认配置，触发了现有 setup 保存缺陷。该次测试生成的 profile 指向临时目录，随后被诊断命令重建为空目录。已将这份测试配置移到审查目录保留备份，用户默认配置回到未初始化状态。**运行前的原始 profile 内容未事先采集，不能声称已恢复其原内容。** 后续聚焦与全量运行均隔离配置，最终确认没有再次生成用户默认 profile。

记录：[配置副作用处理](D:/Codex/matchquant_runs/codex_review_a020607_b1_0eb2l/profile_cleanup_record.json)。

## 6. 下一轮实施顺序与通过标准

1. **先修数据持久性和完整性**：不可变版本目录、只切指针、保留未请求历史、完整修订/身份/重复验证、七列 canonical。R01–R07 满足后才进入下一组。
2. **再修时序和刷新政策**：有效 source-check 间隔、明确 season/as_of、动态请求范围、归档快照选择、失败回退、真实覆盖披露。R08–R10 与新增 archive/fallback 测试满足。
3. **再收口用户入口**：可安装 Skill 资源、稳定 run_id、结构化错误、配置验证、实际宿主调用。wheel 场景及 R13/R14 满足。
4. **最后接 Agent 叙事和报告一致性**：真实 JSON schema、可信归因、宿主叙事输入、统一章节与状态、有效样本数。R11/R12 满足，并有一次宿主实际写作和不重算追问。
5. 稳定后执行完整离线套件、非 editable 安装、实际来源 dry-run/隔离 apply、真实与合成样本以及中文回执。2027 用合成时钟；真实网络只能验证当时实际存在的数据。

下一轮应继续遵循原完整任务书。本次通过的 91 项测试和成功来源访问都属于已有证据，不是免除缺失边界验收的理由。只有上述修复及对应实测完成，才能把本阶段状态改为通过。
