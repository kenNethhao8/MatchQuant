# MatchQuant 盘口数据接入计划书（主控：本会话，三阶段）

日期：2026-10-27　基线：`quant-v2` HEAD `a020607`（含工作区未提交修正）
流程：每个 Stage 由执行会话完成 → 用户把交付回执贴回主控会话 → 主控审核（通过/打回）→ 才可进入下一 Stage。

## 0. 必读约束与已核实的现状

执行会话开工前必须阅读：`QUANT_MASTER_CHARTER.md`、`AGENTS.md`、本文件。

### 0.1 红线（来自总纲，违反即打回）
1. **Track A 永不输入赔率**。盘口只能进入 Track B / M-Open / 独立的市场对照段。
2. **无穿越**。任何盘口快照必须带 `observed_at`，且严格早于预测截止（as-of）。
3. **新因子先过 Factor Lab 正交残差检验**，不得直接塞进模型。本计划不新增训练因子，只复用既有 `mkt_open_prob_edge`、`mkt_ah_supremacy`。
4. 不写 bundle、公证、hash chain 一类工程官僚代码。
5. 数值只来自代码输出，宿主 LLM 不得估算赔率或概率。

### 0.2 已核实事实（主控读源码所得，执行会话需自行复核）
- `single_match.py` **已支持** 市场输入：`market_input_path` 指向一个 JSON，必填字段 `home_team, away_team, date, observed_at, source, odds_open_h/d/a`，可选 `mkt_ah_line`。去水用 `devig_power_probabilities`（Power 法）。
- 有 `mkt_ah_line` 且历史数据满足 `_m_open_history_readiness` 时会拟合 **M-Open**；否则报告写明不可用原因。
- `adapter.py` 已把请求里的 `market_input_path` 透传。
- `observed_at` 目前是**日期粒度**，且必须 `< as_of`。因此"今天拉的盘口 + 预测今天以后的比赛"需要 `as_of` 设为盘口观察日之后的日期。这是已知张力，Stage 1 需给出明确规则。
- **不存在**临场 Steam/多时点序列的单场输入；`mkt_steam_*` 目前只存在于历史回测列。
- 本地数据最新比赛为 2026-05-24，2026-27 赛季未同步。不传 `as_of` 时 `analyze` 因数据陈旧返回 failed；用 `as_of=2026-05-24` 可跑，但 forecast horizon 为 160 天，且没有本赛季近况。**这是盘口分析前的前置阻断项，见 Stage 0。**
- 更正：此前口头说"需补 Track B 打通单场"。准确说法是单场目前只有 Track A + 可选 M-Open，没有 Track B（含 Steam）。Stage 2 是否做 Track B 取决于 Stage 1 能否取得多时点快照，见 Stage 2 的决策门。

### 0.3 不在范围内
赔率下注建议、凯利仓位、自动下单、爬取需要登录或违反条款的站点。报告只给"市场隐含概率与模型差异"，保留"不构成投注建议"声明。

---

## Stage 0（前置，半天）：让 2026-27 赛季数据可用

目的：没有本赛季近况，盘口对比没有意义。
任务：
1. 用现有显式命令对隔离目录先 `--dry-run` 再 `apply`：`epl-sync --seasons 2026-27 --in-progress-season 2026-27`。
2. **注意 Codex 清单 D06**：只同步新赛季会丢历史分区（1140+100 变 100）。**禁止直接对用户真实数据目录 apply。** 做法二选一：
   - A（推荐）：同时请求 `2015-16`…`2026-27` 全部赛季，再 apply；
   - B：先修 D06（保留未请求的旧分区）并加回归测试。
3. apply 前备份 `D:\Codex\matchquant_data\processed` 与 `raw`；apply 后核对总行数、赛季列表、2026-27 比赛数（应约 70+）、`as_of=2026-10-27` 下 `readiness` 为 ok。
4. 用 `analyze` 跑切尔西对曼联 2026-10-31（`as_of=2026-10-27`，`season=2026-27`）确认出完整报告。

验收：readiness ok；总行数 ≥ 原 4180 + 本赛季已赛场次；原 91 项测试仍通过；交付备份路径与前后行数对照。

**Stage 0 执行 Prompt：**
> 你是 MatchQuant 执行会话。先读 QUANT_MASTER_CHARTER.md、AGENTS.md、docs/stage_specs/MARKET_ODDS_INTEGRATION_PLAN_20261027.md 的 0 与 Stage 0。目标：让用户真实数据目录包含 2026-27 赛季已赛场次且不丢历史。步骤：先备份 processed/raw；用 epl-sync 请求 2015-16 至 2026-27 全部赛季先 dry-run，贴出差异摘要（新增场次、修订数、是否阻断），无修订阻断再 apply；不要为 D06 之外的问题改动同步代码。若只能靠修 D06 才安全，则先写失败测试再最小修复。完成后跑完整测试，并跑 as_of=2026-10-27 的切尔西对曼联分析。交付回执必须含：备份路径、前后行数与赛季列表、2026-27 场次与最新比赛日、readiness JSON、测试结果、未解决问题。不得声称未执行的验证。

---

## Stage 1（1~1.5 天）：实时盘口采集模块

目的：产出符合现有 `market_input_path` 契约的盘口快照，并留存时序。
任务：
1. **数据源选型（用户已定：首选 The Odds API，备选 API-Football 免费档）**：执行会话只需对首选源做最小调用实测并确认以下几点，任一不满足再启用备选并回报主控。核对项：`soccer_epl` 的 `h2h` 是否稳定返回；`spreads`（亚盘）对英超是否有数据；是否有 Pinnacle 或至少一家主流庄；返回里的更新时间字段；响应头剩余额度。免费档为 500 credits/月，实时赔率每次消耗 = 市场数 × 区域数，所以默认 `regions=uk` 或 `eu` 单区域、`markets=h2h,spreads`，并做本地缓存与最小刷新间隔以节省额度。原要求（候选源对比）保留如下：执行会话用最小调用实测候选源（如 The Odds API、API-Football 或其它免费额度源），对比：覆盖英超、是否含 Pinnacle/Bet365、是否含亚盘 `spreads`、免费额度与速率、条款是否允许、返回的更新时间字段。输出 1 页选型表，**停下来等主控与用户确认**再编码。
2. 新增 `matchquant/odds_provider.py`：
   - `fetch_odds(home, away, date, provider, api_key) -> list[OddsSnapshot]`；
   - 球队名映射复用 `UserProfile.resolve_team_alias` 与数据里的标准队名，映射失败明确报错，不模糊猜测；
   - API key 只从环境变量或 profile 引用读取，**禁止写入仓库、日志、报告**；
   - 网络错误、限流、无该场赔率都返回结构化错误，不崩溃。
3. 快照存储：`<data_root>/odds_snapshots/<fixture_id>/<UTC时间戳>.json`，只追加不覆盖，字段含 `source, bookmaker, observed_at_utc, odds_open_h/d/a, mkt_ah_line, raw_response_ref`。**这里的 "open" 字段名沿用契约，实际含义是"本快照时点赔率"，需在文档与字段说明中写清，避免与历史初盘混淆。**
4. 转换器 `snapshot_to_market_input(snapshot)`：生成现有契约 JSON。日期粒度规则：`observed_at` 取快照 UTC 日期；`as_of` 必须大于该日期；在 adapter 中当用户未显式给 `as_of` 且存在盘口时，默认取"盘口观察日 + 1 天"，并在响应里写明，不静默。
5. 亚盘符号约定与现有 `home_handicap_sign_convention`（主队让球为负）一致，并为转换加单元测试，覆盖 +/-0.25、0、±0.5 等。
6. adapter 新增动作 `odds`（拉取并落盘，返回快照路径与摘要）；`analyze` 支持 `use_live_odds: true`，先拉取再注入 `market_input_path`。拉取失败时按配置：降级为无盘口的 Track A 报告，并在响应与报告中**醒目标注**"盘口获取失败"，不得静默降级。
7. 测试（全部用本地假响应，不依赖外网）：映射、符号、日期规则、限流/空响应、key 不泄漏、快照不覆盖、`analyze` 降级路径。另做一次**真实网络 smoke**并保存脱敏记录。

验收：新测试全过，原 91 项不回归；真实 smoke 取得切尔西对曼联 2026-10-31 的至少一个快照；快照能通过现有 `_load_market_input` 校验；API key 全仓 grep 不出现。

**Stage 1 执行 Prompt：**
> 你是 MatchQuant 执行会话。读 QUANT_MASTER_CHARTER.md、AGENTS.md、本计划 0 与 Stage 1。严格遵守红线：Track A 不得接触赔率；快照必须带时间并早于 as_of；不得写入 API key。先完成数据源选型表并停下，等主控确认后再写代码。获批后按任务 2 到 7 实现，先写失败测试。字段 odds_open_* 在快照中表示"本快照时点赔率"，要在代码注释和说明中写清。不要修改预测模型与 Track A 路径。交付回执含：选型表与最终选择理由、新增/修改文件清单、测试输出（新增与全量）、真实 smoke 的脱敏记录路径、as_of 默认规则说明、key 泄漏 grep 结果、已知限制。未做的事项如实列出。

---

## Stage 2（0.5~1 天）：盘口分析进入单场结果

目的：让单场分析给出可信的"模型 vs 市场"对照；有多时点快照则给出趋势。

**决策门（Stage 2 开始前主控裁定）：**
- 门 A：仅有单时点快照 → 只做对照（模型概率 vs 去水市场概率、差值、M-Open 可用时的结果），**不做 Steam/趋势**。
- 门 B：已积累同一场 ≥2 个时点快照 → 另做趋势段（去水概率随时间变化，描述性指标，如首末变化、方向、速度）。趋势仅作描述，**不得作为新训练因子**；若要入模，须另立项走 Factor Lab。

任务：
1. 在 `single_match` 结果中补充（只用已有函数）：去水市场概率、`track_a_minus_market_percentage_points`（已存在则复核口径）、over/under 若有则展示隐含概率、M-Open 概率及不可用原因。
2. 趋势（门 B）：读取 `odds_snapshots/<fixture_id>/`，按 `observed_at_utc` 排序，输出时间序列与描述性变动；快照不足 2 个时明确写"无趋势数据"，不插值、不外推。
3. 背离提示：定义**可审计的阈值规则**（例如 |Track A − 市场| 超过 X 个百分点标记"显著背离"，X 写入配置并在报告中展示），文案只描述现象，列出可能原因为"可能包含模型未见信息（伤停、阵容、新闻）"，不断言原因。
4. 价值差（Edge）：可展示 `P_model × 赔率 − 1`，必须同时展示"这是模型与市场之差，模型未经市场结果盲测验证，不构成投注建议"，且仅当赔率为实盘快照时展示。不输出凯利仓位。
5. 回归测试：同一输入下 Track A 概率与加入盘口前**逐位相同**（证明隔离）；盘口字段改动不影响 Track A；时间穿越用例（observed_at ≥ as_of）必须被拒；趋势在快照缺失时的行为。

验收：隔离测试通过；全量测试通过；用 Stage 0 数据与 Stage 1 快照跑出切尔西对曼联完整 JSON；报告字段与数字一致。

**Stage 2 执行 Prompt：**
> 你是 MatchQuant 执行会话。读 QUANT_MASTER_CHARTER.md、AGENTS.md、本计划 0 与 Stage 2，并按主控已裁定的决策门（A 或 B）执行。核心要求：Track A 概率在加入盘口前后逐位相同，并有测试证明；不新增训练因子；趋势只描述不入模；背离阈值可配置且在报告中展示；任何数值来自代码输出。先写失败测试再实现。交付回执含：所选决策门、改动文件清单、隔离测试证明、全量测试输出、切尔西对曼联的 match_report.json 路径与市场字段摘录、未解决问题。

---

## Stage 3（0.5 天）：研报与追问呈现

目的：把 Stage 2 的结构化结果呈现给用户，并保证 Markdown / HTML / 追问一致。

任务：
1. 重写报告第 4 节 `[E5]`：展示盘口来源与观察时间、去水概率、与 Track A 的差异表、M-Open（如有）、趋势（门 B）、背离提示、局限声明。无盘口时显示获取失败或未提供的真实原因。
2. **同时修复已知 R03 缺口中与本节相关的部分**：Markdown 与 HTML 必须由同一份内容数据生成，状态、章节、数字一致；HTML 不得缺少 E5。（其余 R01/R02 等不在本阶段范围，不要顺手扩大。）
3. 追问：盘口类问题（"盘口怎么看""为什么和模型不一样""赔率变化趋势"）读取已有 `run_dir` 的 JSON 证据回答，不重算、不重新联网；用户要求"最新盘口"才触发重新拉取并明确说明会生成新 run。
4. `SKILL.md` 同步：说明何时调用 `odds`、`use_live_odds`、盘口失败如何向用户表达、as_of 规则。
5. 端到端演练并保存证据：自然语言请求 → `analyze`（含实时盘口）→ 报告 → 3 次追问（盘口差异、趋势或无趋势、重新拉取）。

验收：同一场的 MD 与 HTML 数字逐项一致（写自动比对测试）；无盘口 / 盘口失败 / 有盘口三种报告都可读且诚实；追问不重算；全量测试通过。

**Stage 3 执行 Prompt：**
> 你是 MatchQuant 执行会话。读 QUANT_MASTER_CHARTER.md、AGENTS.md、本计划 0 与 Stage 3。只做 E5 呈现、MD/HTML 一致性（限于本节所涉）、盘口追问与 SKILL.md 更新，不扩大范围到其他清单缺口。报告中所有数字来自 JSON。必须保存端到端演练证据。交付回执含：改动清单、MD/HTML 一致性测试、三种场景（有盘口、失败、无盘口）的报告路径、追问记录与是否重算的证据、全量测试输出、遗留问题。

---

## 审核清单（主控每个 Stage 回执使用）

| 项 | 检查 |
| --- | --- |
| 红线 | Track A 未接触赔率；observed_at < as_of；无 key 泄漏；无新训练因子 |
| 证据 | 测试日志、路径、命令输出真实存在；未执行的验证已如实标注 |
| 回归 | 原 91 项全部通过，新增测试覆盖本阶段要求 |
| 诚实度 | 失败降级有醒目标注；无趋势时不编造；不构成投注建议声明保留 |
| 范围 | 没有顺手修改无关模块 |

## 风险与待用户决定

1. **数据源与 key**：需要用户提供或注册 API key，及接受免费额度限制（Stage 1 选型后确认）。
2. **日期粒度张力**：当前契约 `observed_at` 为日期粒度且须早于 as_of，临场（开赛当日）盘口无法入同日预测。Stage 1 先用"观察日 + 1"规则兼容；若要精确到时分，需修改单场时序契约，属于较大改动，需单独立项。
3. **M-Open 就绪度**：需要历史数据里有足够的初盘与亚盘列；Stage 0 后须核对 `_m_open_history_readiness`，不满足时 Stage 2 只能给对照而无 M-Open。
4. **Track B 不在本计划内**：多时点 Steam 入模需先过 Factor Lab，另立项。
