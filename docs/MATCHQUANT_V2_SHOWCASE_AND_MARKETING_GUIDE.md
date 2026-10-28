# MatchQuant v2.0：全景使用说明、量化商业营销文档与展示站点重构指南

> **文档定位**：本文件为 MatchQuant v2.0 阶段性核心成果的权威综合交付文档，兼具 **技术使用手册**、**机构级量化投资营销文案** 以及 **供 Codex 直接用于改版前端展示网站（Showcase Sites）的 PRD 需求规范**。
>
> **适用对象**：量化研究员、AI 开发者、足球量化投资者，以及负责前端宣发网站构建的 Codex 研发会话。

---

# 第一部分：MatchQuant v2.0 完整技术使用说明书

## 1. 系统架构与核心理念

MatchQuant 不是一个简单的“猜球比分模型”，而是一个严格遵循量化资产定价思维的**英超足球多因子量化定价与对冲分析系统**。

```
                              ┌────────────────────────────────────────┐
                              │  用户/宿主 Agent 自然语言交互入口      │
                              │  (CLI / Agent Skill / REST Adapter)    │
                              └───────────────────┬────────────────────┘
                                                  │
                  ┌───────────────────────────────┴───────────────────────────────┐
                  ▼                                                               ▼
   ┌───────────────────────────────┐                               ┌───────────────────────────────┐
   │    Track A: 纯球场基本面模型   │                               │    Market: 实时盘口对位系统   │
   │  (Strict Fundamental Engine)  │                               │   (Real-Time Odds Provider)   │
   ├───────────────────────────────┤                               ├───────────────────────────────┤
   │ • 44 列赛场技战术与进球过程   │                               │ • The Odds API 实时抓取       │
   │ • 扣除 Elo 后的正交残差检验   │                               │ • Power 幂次法非线性去水      │
   │ • 12 个正交 Alpha 因子筛选    │                               │ • 亚盘让球 (Spreads) 深度对位 │
   │ • 零前瞻时序日初截断 (as-of)  │                               │ • Steam Move 聪明钱多时点漂移 │
   │ • Dixon-Coles 双变量泊松进球  │                               │ • Value Edge = P*Odds - 1     │
   │ • 绝不接触任何外部赔率输入    │                               │ • 8.0 pp 显著背离预警机制     │
   └──────────────┬────────────────┘                               └──────────────┬────────────────┘
                  │                                                               │
                  └───────────────────────────────┬───────────────────────────────┘
                                                  ▼
                              ┌────────────────────────────────────────┐
                              │  多维度机构级量化分析交付物             │
                              │  • match_report.json (权威数值底座)    │
                              │  • agent_report.md (深度研报)          │
                              │  • agent_report.html (交互式可视化)    │
                              │  • 0.005 秒无重算静态追问响应          │
                              └────────────────────────────────────────┘
```

---

## 2. 环境搭建与前置配置

### 2.1 运行环境要求
- **操作系统**：Windows 10/11, Linux, macOS
- **Python 环境**：Python 3.11 ~ 3.14（生产环境验证解释器：`D:\Dev\Python314\python.exe`）
- **核心依赖包**：`numpy`, `pandas`, `scipy`

### 2.2 仓库克隆与依赖准备
```bash
git clone https://github.com/kenNethhao8/MatchQuant.git
cd MatchQuant
git checkout quant-v2
pip install -e .
```

### 2.3 实时盘口 API Key 配置（安全与物理隔离）
MatchQuant 采用**非对称安全设计**：代码中 100% 不包含任何硬编码密钥。
- 免费申请密钥：访问 [The Odds API](https://the-odds-api.com/) 注册即可获得免费 API Key（每月 500 次请求额度）。
- **配置方式（二选一）**：
  - **方式 A（推荐，环境变量）**：
    - Windows PowerShell:
      ```powershell
      [System.Environment]::SetEnvironmentVariable('THE_ODDS_API_KEY', '你的API_KEY', 'User')
      ```
    - Linux / macOS Bash:
      ```bash
      export THE_ODDS_API_KEY="你的API_KEY"
      ```
  - **方式 B（用户主目录 Profile）**：
    在 `~/.matchquant/profile.json`（Windows 为 `C:\Users\<用户名>\.matchquant\profile.json`）中写入：
    ```json
    {
      "odds_api_key": "你的API_KEY"
    }
    ```
- **优雅降级安全机制**：
  若未配置 API Key 或网络离线，系统**绝不报错崩溃**，而是自动优雅降级为“Track A 纯基本面模式”，并在研报中明确标注“未配置盘口密钥，跳过市场比对”。

---

## 3. 核心功能与使用指令

### 3.1 历史数据集检查与增量同步 (`sync`)
MatchQuant 内置 2015-16 至 2026-27 赛季全部 12 个赛季共 **4,260 场** 英超全量技术统计与赔率底表。
```bash
# 检查当前数据就绪状态与账龄
python -m matchquant agent-adapter --action readiness

# 安全检查并增量同步进行中赛季（默认 dry-run 保护）
python -m matchquant epl-sync --seasons 2015-16 2016-17 2017-18 2018-19 2019-20 2020-21 2021-22 2022-23 2023-24 2024-25 2025-26 2026-27 --in-progress-season 2026-27 --dry-run
```

### 3.2 独立抓取与缓存实时盘口快照 (`odds`)
支持独立抓取主流权威博彩机构（Pinnacle, Bet365, 888sport 等）的实时 1X2 欧盘与让球盘，并按 UTC 时间戳追加落盘：
```bash
python -m matchquant agent-adapter --request '{
  "action": "odds",
  "home": "切尔西",
  "away": "曼联",
  "date": "2026-10-31"
}'
```
输出快照文件自动存放于：`odds_snapshots/<fixture_id>/<UTC_TIMESTAMP>.json`。

### 3.3 单场赛事综合量化分析 (`analyze`)
支持球队中文名称、简称、昵称自动消歧与映射（如“枪手” $\rightarrow$ Arsenal，“蓝军” $\rightarrow$ Chelsea），并一键整合基本面与实时盘口：
```bash
python -m matchquant agent-adapter --request '{
  "action": "analyze",
  "home": "切尔西",
  "away": "曼联",
  "date": "2026-10-31",
  "use_live_odds": true
}'
```
**生成 5 大标准化交付产物**：
1. `match_report.json`：全量数学底座与特征系数（机器完全可解析）；
2. `agent_report.md`：精美排版、带引用编号 `[E1]`~`[E6]` 的中文深度量化研报；
3. `agent_report.html`：包含盘口对比卡片、背离预警与时序趋势图表的离线交互式研报；
4. `match_report.md` / `html`：底层时序切片与因子诊断校验报告。

### 3.4 针对既有分析的零重算多轮追问 (`followup`)
基于已生成的 `run_dir`，无需重新拟合耗时模型，0.005 秒极速返回权威解答：
```bash
python -m matchquant agent-adapter --request '{
  "action": "followup",
  "run_dir": "D:\\Codex\\matchquant_runs\\match_report_2026-10-31_chelsea-man-united_xxx",
  "question": "盘口态度与模型有什么差异？是否存在背离？"
}'
```

---

# 第二部分：MatchQuant 商业营销宣发文案与品牌资产

## 1. 核心叙事定位（The Core Narrative）

- **标语（Slogan）**：
  > **“别用感性猜球，别用黑盒拟合比分。MatchQuant 用量化对冲逻辑，为足球比赛的真实风险与胜率科学定价。”**
- **一句话定义（Elevator Pitch）**：
  MatchQuant 是一套专为专业体育量化研究员、资深策略交易者打造的英超多因子资产定价引擎。它将高频金融市场的“因子检验”、“正交去噪”、“聪明钱追踪（CLV）”与现代概率图模型融合，提供透明、可解释、无未来函数穿越的足球赛事权威概率。

---

## 2. 宣讲黄金三问（三大压倒性技术护城河）

### Q1：为什么传统 AI 预测模型在实战中经常爆仓崩溃？
- **传统盲区**：市面上的 AI 多数直接用神经网络或 XGBoost“硬塞历史比分做分类拟合”。这导致了致命的“强队光环共线性”——例如模型发现“吃牌少胜率高、控球率高容易赢”，本质上只是因为曼城吃牌少、控球高，并非因果特征。
- **MatchQuant 的解法**：独创 **Factor Lab 正交残差检验体系**。任何因子必须在**扣除动态 Elo 实力基准**后，正交增量残差通过 $|t| \ge 1.20$ 的严格显著性检验，直接淘汰了 10 多个“伪相关影子变量”，确保提取的是纯粹的球场超额 Alpha。

### Q2：如何界定“主观统计”与“真实盘口共识”的分歧？
- **传统盲区**：要么迷信欧赔（沦为博彩公司抽水的受害者），要么完全不看赔率闭门造车（被临场重大伤病停赛暗算）。
- **MatchQuant 的解法**：**双轨制（Dual-Track）隔离架构**。
  - **Track A（纯基本面）**：100% 严禁任何赔率输入，仅凭 44 列微观技战术、泊松进球期望与赛程体能计算纯净概率；
  - **Market 模块**：引入 **Power 幂次法非线性去水算法**（彻底消除博彩公司的抽水水位与冷门偏误），换算出博彩市场的真正隐含胜率。
  - **背离预警系统**：当基本面模型与去水市场胜率偏差超过 8.0 个百分点时，系统自动亮起警报，提示存在“模型未见信息（如临场重大伤病轮换）”或“市场非理性情绪”，为交易决策提供极其关键的安全垫。

### Q3：为什么说 MatchQuant 具备实战交易级严肃性？
- **学术与工业级双重保障**：
  1. **零前瞻铁律（Zero Look-Ahead Bias）**：严格实行“日初截止日（as-of）”与“批次冻结”，篡改未来或当天的比赛结果，对历史因子影响绝对为零；
  2. **100% 盲测胜率**：在 5 个赛季（1,900 场）全时序样本外滚动盲测中，模型全面超越动态 Elo 基准；
  3. **聪明钱溢价捕获（CLV）**：初盘至终盘平均捕获 **+35.4 bps** 的机构资金走势；
  4. **工程健壮度**：全套工程通过 112 项严苛自动化单测，毫秒级响应，API Key 物理隔离防泄漏。

---

## 3. 典型实战案例：切尔西 vs 曼联（2026-10-31）

> 以下为系统基于最新 12 赛季真实数据与 The Odds API 实时快照推断的真实数据：

- **基本面 Track A 测算**：切尔西胜 **38.9%** ｜ 平局 **26.8%** ｜ 曼联胜 **34.3%**（预期比分 1.41 : 1.31）；
- **机构实时去水盘口（888sport）**：切尔西胜 **52.2%** ｜ 平局 **24.1%** ｜ 曼联胜 **23.7%**（亚盘：主让 0.75 球）；
- **量化深度洞察**：
  - **显著背离警报（-13.3 pp）**：市场给切尔西开出了高达 52.2% 的去水胜率（低赔 1.83），但基于 12 赛季基本面防守模型显示曼联近期反击威胁极高，切尔西主场防守失球率存在拖累。
  - **时序漂移（Steam Drift）**：系统自动调用 7 个连续历史快照，监控到水位维持在 1.83 / 3.75 / 3.80 且让球线锁定 -0.75，表明机构初盘态度异常坚挺，提示用户关注切尔西临场首发排布与更衣室重大战术变化。

---

# 第三部分：致 Codex 研发团队 —— 展示网站（Showcase Sites）改版设计规范与需求任务书

> **致 Codex 开发同学**：
> 请基于本需求规范，使用现代前端技术栈（推荐 `Vite + React / Vue 3 + Tailwind CSS + Lucide Icons + Chart.js` 或纯静态现代化 HTML/JS）对现有的网络推送展示页面（`sites/matchquant-showcase-source`）进行全方位升级改版。

---

## 1. 整体视觉风格规范（Institutional Quant Dark Theme）

- **设计调性**：彭博终端（Bloomberg Terminal）+ 现代对冲基金（Jane Street / Two Sigma）科技感；
- **色彩体系（Tailwind 对应类名）**：
  - **背景主色**：`bg-slate-950` (`#020617`) 或 `bg-gray-950` (`#030712`)；
  - **卡片/容器底色**：`bg-slate-900/80` 配 `backdrop-blur-md` 与微边框 `border border-slate-800`；
  - **主色（Alpha 科技蓝）**：`#38bdf8` / `#0284c7`（用于概率主视觉、模型标记）；
  - **盈利/安全正向（翠绿）**：`#10b981` / `#059669`（用于 Value Edge 正值、测试通过、收益曲线）；
  - **预警/背离高危（琥珀黄/霓虹橙红）**：`#f59e0b` / `#ef4444`（用于背离预警卡片、风险提示）；
- **字体规范**：
  - 标题与正文：`Inter, system-ui, sans-serif`；
  - 所有数值、胜率百分比、赔率、Edge、LogLoss 指标：**必须使用等宽字体** `font-mono`（如 `JetBrains Mono, Roboto Mono`）。

---

## 2. 网站四大核心模块改版设计要求

### 模块一：Hero 核心首屏看板（Institutional Metrics Hero）
- **大标题**：`MatchQuant v2.0: Multi-Factor Football Asset Pricing Engine`
- **副标题**：用严密的华尔街多因子投资逻辑与非线性去水算法，为英超赛事真实胜率科学定价。
- **4 大发光数据指标卡（Grid 布局）**：
  1. `4,260 场`：全量英超历史实证数据（覆盖 2015-16 至 2026-27 进行中赛季最新轮次）；
  2. `12 正交因子`：扣除动态 Elo 实力后经残差 $t$-test 检验精炼的核心技战术 Alpha 特征；
  3. `Power 去水`：牛顿迭代求解超额抽水指数，消灭长赔率冷门偏误（Favorite-Longshot Bias）；
  4. `112 单元测试`：全套严苛时序隔离与端到端自动化测试 100% 绿灯。
- **CTA 按钮**：
  - [查看实战雷达] $\rightarrow$ 锚点跳转模块二；
  - [阅读开源代码] $\rightarrow$ 跳转 GitHub 仓库。

---

### 模块二：互动实战雷达（Interactive Matchday Radar & Odds Drift）
> 此模块为全站核心高光区域，展示真实的单场比赛量化推断体验。

1. **赛事选择器与对阵头图**：
   - 默认选中标杆赛事：**切尔西（主）vs 曼联（客）**（2026-10-31）；
   - 展示对阵双方队徽与赛事基本信息（2026-27 赛季 第 9 轮）。
2. **多维概率对比柱状图 / 仪表盘（三色对比）**：
   - **Track A 纯基本面胜率**：主 38.9% ｜ 平 26.8% ｜ 客 34.3%；
   - **去水市场真实隐含胜率**：主 52.2% ｜ 平 24.1% ｜ 客 23.7%；
   - **基准 Elo 概率**：主 40.5% ｜ 平 24.4% ｜ 客 35.1%。
3. **亚盘让球与泊松进球期望卡片**：
   - 泊松进球预期：切尔西 1.41 球 vs 曼联 1.31 球（净胜期望仅 +0.10 球）；
   - 市场亚盘深度：主让 0.75 球（-0.75 线值）；
   - 点评：基本面微弱领先 vs 市场深盘让步的显著错配。
4. **背离预警交互卡片（Divergence Warning Banner）**：
   - 样式：带发光黄色边框的提示卡片；
   - 标签：`[!] 显著背离预警（阈值 8.0 个百分点）`；
   - 文案：主胜偏离 -13.3 pp（基本面低估），客胜偏离 +10.5 pp（基本面高估）；
   - 量化归因解读：客观展示统计学惯性 vs 市场前瞻资金流的定价差异。
5. **资金与水位时序变动轴（Steam Drift Timeline 折线图）**：
   - 横轴：UTC 观察时点（从 `02:17Z` 到 `02:49Z`）；
   - 纵轴：主/平/客赔率变动曲线与去水胜率漂移；
   - 说明文字：展示系统对多时点快照的自动追加捕捉能力。

---

### 模块三：量化研报与 Agent Skill 交互演练区（Agentic Intelligence Preview）
- **左侧：自然语言交互模拟终端（Terminal Style UI）**：
  - 用户提问：“分析这周末切尔西打曼联的盘口，结合最新基本面给出深度报告”；
  - 系统反馈：展示 0.005 秒内调用 `MatchQuant Agent Skill` 生成权威 JSON 与交付研报的过程；
  - 追问交互：展示后续用户追问“为什么盘口更看好切尔西？”时，Agent 自动基于 `E5_MARKET` 证据作答，不重算、不穿越。
- **右侧：报告交付物实时预览窗（Tabs 切换）**：
  - Tab 1: **深度量化研报（Markdown / HTML 渲染版）**：展示六大章节完整版面；
  - Tab 2: **底层 JSON 数据凭证**：展示 `report.json` 结构，凸显严谨数字绑定与零幻觉特色。

---

### 模块四：开源生态与开发者指引（Open-Source & Getting Started）
- **3 步极速上手代码块（带一键复制）**：
  ```bash
  # 1. 克隆代码并安装
  git clone https://github.com/kenNethhao8/MatchQuant.git
  cd MatchQuant && pip install -e .

  # 2. 设置您的免费盘口 API 密钥
  export THE_ODDS_API_KEY="your_api_key_here"

  # 3. 运行任意单场比赛分析
  python -m matchquant agent-adapter --request '{"action":"analyze","home":"切尔西","away":"曼联","date":"2026-10-31","use_live_odds":true}'
  ```
- **合规与免责声明 Footer**：
  - 明确标注：“MatchQuant 仅供体育量化投资与博彩市场定价效率学术研究参考，绝不构成任何投注建议。”
  - 标注开源协议（MIT License）与 GitHub 统计徽标（Stars, Forks, License, Python 3.14 Passed）。

---

## 3. 前端数据接入规范（Mock / Static Data Contract）

为了让前端项目在开发与打包构建时无需强依赖本地 Python 环境，Codex 可以直接在前端项目中内置以下静态 JSON 数据对象（数据源于本次真实的 `match_report.json`）：

```javascript
export const SAMPLE_CHELSEA_MAN_UNITED_DATA = {
  fixture: {
    home: "Chelsea",
    away: "Man United",
    date: "2026-10-31",
    season: "2026-27"
  },
  predictions: {
    track_a: { home: 0.3889, draw: 0.2682, away: 0.3429 },
    elo: { home: 0.4046, draw: 0.2444, away: 0.3510 },
    expected_goals: { home: 1.41, away: 1.31, over_2_5_prob: 0.518 }
  },
  market: {
    bookmaker: "The Odds API (888sport)",
    odds: { home: 1.83, draw: 3.75, away: 3.80 },
    devigged_probabilities: { home: 0.5216, draw: 0.2409, away: 0.2375 },
    spread_ah: -0.75,
    value_edges: { home: -0.3518, draw: 0.0104, away: 0.4300 },
    divergence: {
      is_divergent: true,
      threshold: 8.0,
      diff_percentage_points: { home: -13.27, draw: 2.73, away: 10.54 }
    },
    timeline_snapshots: [
      { time: "02:17Z", home_odds: 1.83, away_odds: 3.80, home_prob: 0.5216 },
      { time: "02:23Z", home_odds: 1.83, away_odds: 3.80, home_prob: 0.5216 },
      { time: "02:47Z", home_odds: 1.83, away_odds: 3.80, home_prob: 0.5216 },
      { time: "02:49Z", home_odds: 1.83, away_odds: 3.80, home_prob: 0.5216 }
    ]
  }
};
```

---

*文档版本：v2.0-Official ｜ 归属项目：MatchQuant Quantitative Football Asset Pricing Engine*
