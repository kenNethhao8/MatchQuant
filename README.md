# MatchQuant v2.0 — 足球量化因子工程与多模型研究引擎

> **纯粹量化研究架构**：全量技战术与市场特征解锁 $\rightarrow$ 5 大正交因子族（EWMA + 对手强度校准） $\rightarrow$ Factor Lab 单因子与 Elo 正交残差检验 $\rightarrow$ Dixon-Coles 进球模型 / 有序逻辑回归合成 $\rightarrow$ 双轨样本外验证与逐场因子归因拆解。

---

## 一、 核心架构（五层量化流水线）

| 层级 | 核心模块 | 功能职责 |
| :--- | :--- | :--- |
| **Layer 1: 全量数据层** | [`matchquant/rich_data.py`](matchquant/rich_data.py) | 从 Football-Data.co.uk 原始 `E0` CSV 中完整提取 **44 个标准化字段**（全/半场比分、主客射门、射正、角球、犯规、黄/红牌、主裁判、开球时间、Pinnacle 初/终盘与 Power 法去水隐含概率、大小球、亚盘让球），严格隔离赛前元信息与赛后统计。 |
| **Layer 2: 因子工程层** | [`matchquant/factor_engine.py`](matchquant/factor_engine.py) | 严格按 `(season, date, kickoff_time)` 批次先预测后更新（零未来信息泄露），通过双速指数衰减（EWMA）、跨赛季均值回归与**对手强度修正（Opponent-Adjusted）**计算 5 大因子族共 22 个衍生特征。 |
| **Layer 3: 因子检验实验室** | [`matchquant/factor_lab.py`](matchquant/factor_lab.py) | Alphalens 风格单因子诊断：计算每个因子的 `Rank IC`、跨赛季 `IC_IR`、**扣除 Elo 基准后的正交残差增量 $t$-统计量（`orthogonal_tstat`）**、五分位（$Q_1 \sim Q_5$）单调性分层回测与 `VIF` 共线性剔除。 |
| **Layer 4: 领域模型层** | [`matchquant/quant_models.py`](matchquant/quant_models.py) | 摒弃无序线性 Softmax，采用符合足球数学结构的两类领域模型：<br/>1. **`FactorDixonColesModel`**：因子驱动双变量泊松 + Dixon-Coles 低比分平局修正参数 $\rho$；<br/>2. **`OrderedFactorModel`**：有序逻辑回归（$\text{Away} < \text{Draw} < \text{Home}$）+ 预期节奏调节的动态平局阈值 + 经济学符号一致性约束。 |
| **Layer 5: 验证与归因层** | [`matchquant/quant_pipeline.py`](matchquant/quant_pipeline.py) | 双轨评估（**Track A 纯球场基本面模型** vs **Track B 叠加临场赔率漂移模型**），对标基础动态 Elo 与 Pinnacle 收盘去水概率天花板，并输出单场比赛的**因子边际概率贡献拆解**。 |

---

## 二、 五大量化因子族一览

1. **基础实力与比赛节奏族 (`strength`)**
   * `elo_diff`：主客动态 Elo 差值（含主场优势与升班马先验折价）
   * `elo_abs_gap`：主客实力绝对分差 $|\text{Elo}_H - \text{Elo}_A|$（实力越接近，平局率越高）
   * `att_def_mismatch_diff`：对手强度修正后的 EWMA 攻防错位差
   * `venue_specialist_diff`：指数衰减的主场专属 vs 客场专属净胜球差
   * `expected_tempo`：双方预期总射正与总进球节奏（慢节奏比赛显著推高 $0\text{-}0 / 1\text{-}1$ 平局概率）
2. **场面统治力与射门质量族 (`dominance`)**
   * `sot_dominance_diff`：对手强度校准后的慢速 EWMA 净射正差（$\text{HST} - \text{AST}$，纯技术统计中最强的 xG 代理变量）
   * `sot_momentum_diff`：快慢双速 EWMA 净射正动量差（捕捉近期场面统治力加速上升或滑坡）
   * `corner_pressure_diff`：对手强度校准后的 EWMA 净角球围攻压迫差（$\text{HC} - \text{AC}$）
   * `shot_volume_diff` / `shot_precision_diff`：总射门压制力与射正精度差
3. **运气回归与背离 Alpha 族 (`luck_reversion`)**
   * `points_vs_dominance_gap`：近期实际拿分率与场面射正统治力隐含拿分率之差（负向均值回归：识别“虚假连胜纸老虎”与“场面占优但欠债的倒霉强队”）
   * `finishing_luck_diff`：短期射正转化率（$\text{Goals}/\text{SOT}$）相对长期基线的超额偏离（负向均值回归）
   * `save_luck_diff`：短期门将扑救/抗射正转化率偏离
4. **比赛进程、半场韧性与纪律族 (`rhythm_discipline`)**
   * `first_half_edge_diff`：上半场 EWMA 净胜球差（抢开局能力）
   * `second_half_surge_diff`：下半场 EWMA 净胜球差（体能储备与替补深度）
   * `discipline_risk_diff`：犯规与红黄牌纪律风险指数
   * `referee_home_bias`：当值主裁判历史贝叶斯收缩后的主场净胜球偏倚
5. **赛程状态与市场聪明钱族 (`context_market`)**
   * `promoted_coldstart_diff`：升班马赛季前 10 轮冷启动折价因子
   * `rest_advantage_diff`：截断对数休息天数差
   * *市场轨道（Track B 专属，与纯基本面 Track A 严格物理隔离）*：`mkt_open_prob_edge`（Pinnacle 去水初盘差）、`mkt_steam_move_diff`（初盘至终盘的临场聪明钱漂移 Steam）、`mkt_ah_supremacy`（亚盘让球深度）

---

## 三、 快速运行命令

### 1. 解锁 11 赛季全量技战术与市场特征数据
```powershell
python -m matchquant rich-prepare `
  --data-root D:\Codex\matchquant_data `
  --output D:\matchquant\work\quant_v2\epl_rich_canonical.csv
```

### 2. 运行 Factor Lab 单因子检验与正交 Alpha 诊断
```powershell
python -m matchquant factor-lab `
  --input D:\matchquant\work\quant_v2\epl_rich_canonical.csv `
  --output-dir D:\matchquant\work\quant_v2\factor_lab
```

### 3. 运行端到端多因子建模、双轨样本外验证与归因报告
```powershell
python -m matchquant quant-research `
  --input D:\matchquant\work\quant_v2\epl_rich_canonical.csv `
  --output-dir D:\matchquant\work\quant_v2\research_run
```

### 4. 运行全量单元测试
```powershell
python -m unittest discover -s tests -v
```

---

## 四、 单场比赛研究报告（`match-report`）

`match-report` 使用你明确指定的本地富化历史 CSV，为一个用户指定的对阵生成中文 Markdown、独立 HTML 和结构化 JSON。它是本地研究命令，不会自动发现/下载数据，也不会查询赛程、伤停或新闻。

### 环境与安装

- Python 3.11 或更高版本。
- 项目运行依赖：NumPy、pandas 和 SciPy。SciPy 用于 Factor Lab 统计量与现有模型的参数拟合。
- 在仓库目录安装：`python -m pip install -e .`。安装后使用 `matchquant ...`；不安装时使用 `python -m matchquant ...`。

准备富化历史数据时，使用现有 `rich-prepare` 命令。它读取你本地 `D:\Codex\matchquant_data\raw` 和 `staging` 下已经准备的 EPL 原始 CSV；请把输出写到新文件，不要覆盖原始数据：

```powershell
python -m matchquant rich-prepare `
  --data-root "D:\Codex\matchquant_data" `
  --seasons 2015-16 2016-17 2017-18 2018-19 2019-20 2020-21 2021-22 2022-23 2023-24 2024-25 2025-26 `
  --output "D:\Codex\matchquant_data\processed\epl_rich_canonical.csv"
```

如果目标文件已存在，请换一个新输出路径。用户准备的数据至少需要比分和 MatchQuant 富化表中的技术统计字段；缺失的赛后统计会按因子引擎已有默认值参与状态更新，并在报告中披露。缺失比分或格式错误会导致命令失败。

### 指定一场比赛

直接传入主客队、比赛日期、赛季和截止日：

```powershell
python -m matchquant match-report `
  --input "D:\Codex\matchquant_data\processed\epl_rich_canonical.csv" `
  --home "Man City" --away "Aston Villa" `
  --date 2026-05-24 --season 2025-26 `
  --as-of 2026-05-23 --mode replay `
  --output-root "D:\Codex\matchquant_runs\single_match"
```

也可以用一行无比分 fixture CSV：

```powershell
python -m matchquant match-report `
  --input "D:\Codex\matchquant_data\processed\epl_rich_canonical.csv" `
  --fixtures "matchquant/resources/examples/single_match_fixture.csv" `
  --as-of 2026-05-23 --mode replay `
  --output-root "D:\Codex\matchquant_runs\single_match"
```

CSV 必须只有一行，列为 `fixture_id,date,season,home_team,away_team`，可选 `referee,kickoff_time`；不接受比分、赛后统计或其他列。日期只接受 `YYYY-MM-DD`。直接参数和 CSV 是互斥的。默认模式为 `research`；`replay` 只表示按历史截止日重建，不会读取目标比赛赛果。

截止日按当地日历日期的开始处理：只用 `date < --as-of` 的历史行，截止日当天所有比赛整体排除。fixture 日期必须不早于截止日，赛季必须明确填写。拟合需要至少三个完整 EPL 赛季（每季 380 场）：最新完整赛季用于验证，之前的完整赛季训练；截止日前的当前不完整赛季只更新因子状态，不进入拟合或调参。最终模型会在训练加验证样本上重拟合，所以 JSON 和报告把验证指标标成调参/选择依据，而不是最终模型的盲测成绩。

可选参数：

- `--team-aliases aliases.json`：显式 JSON 字典，例如 `{"Man City FC":"Man City"}`。所有目标名称都必须能唯一匹配历史球队。
- `--market-input market.json`：用户提供的初盘 1X2 赔率，可选提供主队亚盘线。声明观测日必须早于截止日；报告使用 Power 幂次法去水，并明确注明来源未独立核验。缺少亚盘线时仍展示 1X2 市场对照，但不会运行需要亚盘的 M-Open。未提供该文件时，报告写明“未提供赛前赔率，未计算市场对照”。
- `--max-data-age-days 30`：超过阈值时状态为 `needs_review`，命令退出码为 2。该阈值只是复核提示，不代表数据在阈值内就已更新。

市场 JSON 的字段示例（仅用于说明格式，不是实际报价）：

```json
{
  "home_team": "Man City",
  "away_team": "Aston Villa",
  "date": "2026-05-24",
  "observed_at": "2026-05-22",
  "source": "用户填写的来源说明",
  "odds_open_h": 2.1,
  "odds_open_d": 3.4,
  "odds_open_a": 3.6,
  "mkt_ah_line": -0.25
}
```

`mkt_ah_line` 可省略；约定主队让球为负值（例如 `-0.25` 表示主队让 0.25 球）。

命令输出的新目录中有 `match_report.html`、`match_report.md` 和 `match_report.json`。可用浏览器打开 HTML。完整报告状态退出码为 0；数据过旧或被请求的 M-Open 不可用时仍保留基本面报告，并以退出码 2 提醒复核；无效输入或拟合数据不足返回 1。

### 如何解读与复现示例

报告包含 Elo 基线、Track A 模型估计概率、模型期望进球、进球数超过 2.5 的概率、每队最近五场可用记录，以及至多三个有序 Logit 分量的局部近似因子贡献。局部贡献不是最终集成概率差的加法分解，也不是因果效应。报告不判断球队是否在官方赛程中，不推断首发、伤停、战术或动机，也不承诺盈利。输入文件的完整版本也不能证明其中每项观测在历史时点已可用。

样例 fixture 位于 `matchquant/resources/examples/single_match_fixture.csv`，采用一场 2025–26 赛季末轮的对阵名称，不含赛果。先按上一节命令准备包含 2025–26 数据的富化 CSV，再执行 CSV 示例命令即可重建。输出路径下的三份文件都来自同一 JSON 结果。

在 macOS/Linux shell 中用 `$PWD` 或你自己的数据路径替换 Windows 路径，并改用反斜杠续行，例如：

```sh
python -m matchquant match-report \
  --input "$PWD/data/epl_rich_canonical.csv" \
  --home "Man City" --away "Aston Villa" \
  --date 2026-05-24 --season 2025-26 \
  --as-of 2026-05-23 --mode replay \
  --output-root "$PWD/runs/single_match"
```

编码助手可以把自然语言请求转换成该命令，但必须拿到球队、ISO 比赛日期、赛季、数据截止日和显式历史文件路径；缺少日期、赛季或截止日时，应先向用户确认，不能把“今晚”或球队名称自行推断为真实赛程。MatchQuant 本身仍是命令行研究工具。

---

## 五、 Agent Skill 与按需数据同步（On-Demand Data Sync & Agent Skill）

为了实现“一次配置，自然语言交互，按需自动更新历史数据，调用模型计算并生成证据溯源中文报告”，MatchQuant 提供了完整的分层工作流（Layer 0 ~ Layer 5）：

### 1. 用户配置与诊断（Layer 0）

用户可在本地保存轻量级 JSON 配置文件（默认路径为 `~/.matchquant/profile.json`，或通过环境变量 `MATCHQUANT_PROFILE_PATH` 指定）：
- **数据与输出路径**：`data_root`（默认 `D:\Codex\matchquant_data`）、`report_root`（默认 `D:\Codex\matchquant_runs`）。
- **刷新授权与策略**：`auto_refresh: true` 一次性授权常规增量更新（无需每场比赛重复确认）；`source_check_interval_hours` 检查间隔；`allow_stale_fallback` 决定刷新失败或过旧时是否允许回退标记。
- **别名映射**：支持中文俱乐部昵称自动映射为官方英超球队名（如“曼城” $\rightarrow$ “Man City”）。

运行环境诊断命令：
```powershell
python -m matchquant agent-adapter --action setup
```

### 2. 显式按需数据同步引擎（Layer 1：`epl-sync`）

`epl-sync` 命令执行受控的数据拉取与比对，杜绝隐式网络抓取或覆盖历史原始文件：
- **默认 Dry-run 预检**：必须显式传入 `--apply` 才会将新数据晋级至当前版本。
- **不可变快照**：每次抓取的网络原始 CSV 存入时间戳隔离的 `staging/{timestamp}/raw/E0_{season}.csv`。
- **缓存与增量拉取**：已满足 380 场门禁的封存历史赛季直接使用本地缓存，仅拉取缺失或进行中的赛季。
- **结构化 Diff 与 Addition-only 门禁**：通过稳定 `match_id` 精确比对本地与抓取行，识别新增（additions）、变更（revisions）、缺失（missing baseline）与未完赛跳过（skipped pending）。**若出现既有历史修改或基线比赛缺失，系统硬性阻断自动合并**并生成人工复核报告。
- **进行中赛季声明**：通过 `--in-progress-season YYYY-YY` 显式声明，允许空比分行标记为 `pending` 排除在历史之外；已封存赛季严格执行 380 场门禁。

运行同步示例：
```powershell
# 预检对比（Dry-run）
python -m matchquant epl-sync --data-root "D:\Codex\matchquant_data" --seasons 2024-25 2025-26 --in-progress-season 2025-26 --dry-run

# 显式生效（Apply）
python -m matchquant epl-sync --data-root "D:\Codex\matchquant_data" --seasons 2024-25 2025-26 --in-progress-season 2025-26 --apply
```

### 3. 数据集版本与就绪度评估（Layer 2 & 3）

- **版本指针**：`processed/epl_current.json` 记录当前生效的基础与富化 CSV 路径、各赛季快照哈希、总场数与数据截止日。
- **缺失数据保真**：源数据确实未记录的射门、角球或亚盘数据在富化转换中保留为缺失值（NaN），有效样本数真实披露，绝不把缺失转为虚假的 0 次或平手盘。
- **数据就绪度检查**：
```powershell
python -m matchquant agent-adapter --action readiness
```

### 4. 机器可读 Agent 适配器与本地 Skill（Layer 4）

提供结构化标准输入输出接口（stdout 输出纯 JSON），宿主 Agent 无需解析易变的终端日志：
```powershell
python -m matchquant agent-adapter --action analyze --request-json '{"home":"阿森纳","away":"切尔西","date":"2026-05-18","season":"2025-26"}'
```

**本地 Skill 安装**：
技能包位于仓库 `skills/matchquant/`，包含规范的 `SKILL.md`，可复制安装到宿主 Agent 技能目录下：
```python
from matchquant.skill_installer import install_skill
install_skill(target_dir=r"C:\Users\Admin1\.gemini\config\skills")
```

### 5. 双轨报告与基于证据的中文追问（Layer 5）

每次分析在同一个运行目录下同时生成两套报告文件：
1. **基准报告**：`match_report.json`、`match_report.md`、`match_report.html`。
2. **Agent 伴生分析报告**：`agent_report.md`、`agent_report.html`（800~1,200 字，包含核心量化概率、近期技术面支撑、模型分歧、时序截断回执与局限性声明，断言绑定证据编号 `[E1]`~`[E6]`，数值全部由插槽程序化填充）。

**多轮追问复用**：
用户追问胜负逻辑、进球预期、盘口解读或简短总结时，无需重新拟合模型，适配器直接加载已生成的 `match_report.json` 返回证据检索结果：
```powershell
python -m matchquant agent-adapter --action followup --request-json '{"run_id":"...","question":"为什么模型看好主队？"}'
```
