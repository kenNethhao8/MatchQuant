# MatchQuant V1 — 产品展示站点（Showcase Sites）开发转交任务书

> **致 Codex 开发团队**：
> 本文件为 **MatchQuant V1（英超多因子量化定价与实战交易引擎）** 对外展示站点的完整业务逻辑、数据接口契约与前端交互设计任务书。
> 请阅读本任务书后，直接利用本项目现成的数据底表与报告产物，构建一个现代、专业、极具华尔街量化对冲质感的高水准展示网站。

---

## 一、 产品定位与核心叙事（The Story to Tell）

### 1. 一句话定位
> **“别人在用感性‘猜球’，传统 AI 在用黑盒‘拟合比分’，而 MatchQuant 在用华尔街多因子资产定价模型，为球赛的真实风险与胜率进行严密‘定价’。”**

### 2. 核心痛点与量化壁垒（宣讲黄金三问）
1. **为什么胜率高的球队下一场可能会崩？**  
   足球是极低比分的偶然性运动，靠偷鸡 1-0 连赢 3 场的“纸老虎”普遍被大众追捧。MatchQuant 独创**“拿分与统治力背离因子 (`points_vs_dominance_gap`)”**，精准量化“场面压迫 vs 实际积分”的背离度，在连胜虚假繁荣时反向做空。
2. **为什么传统 AI 因子全军覆没？**  
   传统统计发现“吃牌少胜率高、下半场进球多体能好”，其实全是被曼城、阿森纳的强队光环混淆了。MatchQuant 独创 **Factor Lab 正交残差检验**，**必须扣除 Elo 实力基准**，正交增量 $|t| \ge 1.20$ 才能入选，成功斩断了 10 多个“滥竽充数的伪因子”。
3. **如何精准预测且可直接实战？**  
   我们拒绝黑盒 Softmax 导致的符号反转，采用**有序逻辑回归 + 因子驱动双变量泊松（Dixon-Coles）进球模型**。构建 **Track M-Open 初盘可交易模型**，在 1,900 场全时序样本外盲测中，提前捕获机构聪明钱（CLV）走势胜率达 **`54.25%`**，平均溢价 **`+35.37 bps`**，凯利纸盘实现了正向复利收益！

---

## 二、 推荐技术栈与视觉设计规范

1. **建议技术栈**：
   - 方案 1（全功能单页 / 极简零配置）：`Vite + React / Vue 3 + Tailwind CSS + Lucide Icons + Recharts / Chart.js`；
   - 方案 2（Next.js / Astro 静态生成）：可直接一键部署至 Vercel / GitHub Pages；
   - 方案 3（轻量静态单文件）：直接用单个 `index.html` 引入 Tailwind CDN + Chart.js + Alpine.js，本地双击即可演示。
2. **视觉风格（Institutional Quant Dark Mode）**：
   - **主色调**：深邃量化黑/夜空灰（`#0a0e17`、`#111827`、`#1f2937`）；
   - **功能点缀色**：
     - Alpha 收益绿（`#10b981` / `#059669`）；
     - 警示/回撤橙红（`#ef4444` / `#f59e0b`）；
     - 机构深蓝/科技紫（`#3b82f6` / `#6366f1`）；
   - **排版质感**：等宽数字字体（`font-mono`，展示胜率、赔率、bps、LogLoss），精简发光边框，毛玻璃卡片（`backdrop-blur-md`）。

---

## 三、 展示站点六大功能板块设计（Site Map）

### 模块 1：Hero 头部 — 机构级震撼首屏
- **主标题**：`MatchQuant V1: Multi-Factor Football Asset Pricing Engine`
- **副标题**：从 11 个英超赛季、4,180 场比赛与 44 列微观技战术数据中，提炼 12 个正交 Alpha 因子，为足球比赛风险精准定价。
- **核心数据看板（4 个发光指标卡）**：
  - **4,180 场**：全量英超历史数据集（2015-16 至 2025-26）；
  - **100% 盲测胜率**：5 个赛季（1,900 场）逐季滚动重训，全部击败动态 Elo 基线；
  - **+35.4 bps**：初盘至终盘聪明钱（CLV）平均正向捕获溢价；
  - **12 个正交因子**：经 Factor Lab 残差检验淘汰伪因子后的精炼因子库。

---

### 模块 2：赛前实战雷达（Interactive Matchday Radar）
> **数据源**：`reports/research_run/walk_forward_predictions.csv`（最新一轮 10 场）或直接展示 CLI `matchday-radar` 的输出结构。

- **交互体验**：展示最新一轮英超对阵列表，点击任意一场对阵（如曼城 vs 阿斯顿维拉、阿森纳 vs 水晶宫）：
  - **胜平负三色概率条**：清晰对比 `Elo 基础胜率` vs `Track A 纯基本面胜率` vs `Track M-Open 初盘可交易胜率`；
  - **期望比分矩阵与进球数**：展示主客期望进球 $(\lambda_H, \lambda_A)$ 与大球概率（Over 2.5%）；
  - **单场因子归因瀑布图（Top Drivers Waterfall）**：
    - 清楚展示“为什么模型认为主胜概率比市场高了 4.1%？”
    - 展开显示：`sot_dominance_diff` (+4.8%)、`corner_pressure_diff` (+4.3%)、`venue_specialist_diff` (+2.5%) 的边际贡献拆解。

---

### 模块 3：Factor Lab 单因子体检展台（12 黄金因子 vs 伪因子被杀录）
> **数据源**：`reports/research_run/factor_lab_metrics.json` 与 `factor_lab_report.md`。

- **交互体验**：一个可切换“入选黄金因子（12个）”与“被淘汰伪因子（4个代表）”的交互式表格/卡片：
  - 点击因子卡片，展示其 **Q1 到 Q5 五分位单调性柱状图**（例如 `sot_dominance_diff` 从 Q1 的 20.2% 主胜率平滑单调递增到 Q5 的 72.8%）；
  - **震撼对比弹窗：“我们为什么淘汰了‘吃牌少’和‘下半场进球’？”**：
    - 呈现数据对比：下半场进球原始 IC 高达 `+0.365`，但扣除 Elo 后的正交增量 $t$ 仅 `+0.40`（不足 1.20 门槛）；证明它只是强队的“影子”，展现系统严肃量化的专业壁垒。

---

### 四、 核心数据文件路径与字段速查（Codex 直接读取）

展示站所需的所有分析结果已全部存放在本项目仓库的 `reports/research_run/` 目录下，直接读取解析即可：

| 文件路径 | 格式 | 核心用途与字段 |
| :--- | :--- | :--- |
| [`reports/research_run/quant_evaluation_summary.json`](reports/research_run/quant_evaluation_summary.json) | JSON | **全站数据核心中枢**：包含三轨模型在 5 赛季的 LogLoss、Accuracy、Brier Score；去偏 OrthoEdge 与 Track M-Open 的凯利纸盘回测数据（ROI、CLV、夏普、回撤）；以及单场归因样例。 |
| [`reports/research_run/factor_lab_metrics.json`](reports/research_run/factor_lab_metrics.json) | JSON | **因子体检全部原始指标**：22 个因子的 Rank IC、跨季 `IC_IR`、扣除 Elo 正交 $t$ 值、VIF、五分位（$Q_1\sim Q_5$）主客胜率与净胜球。 |
| [`reports/research_run/walk_forward_predictions.csv`](reports/research_run/walk_forward_predictions.csv) | CSV | **1,900 场逐场预测明细底表**：包含 `date, home_team, away_team, actual_result, elo_prob_*, track_a_prob_*, track_m_open_prob_*, mkt_open_prob_*, mkt_close_prob_*, expected_home_goals, expected_away_goals, prob_over25, ortho_edge_*`。 |
| [`reports/research_run/quant_research_report.md`](reports/research_run/quant_research_report.md) | Markdown | 完整的 9 章节学术/量化报告，供图文引用与直接渲染。 |

---

## 五、 给 Codex 的直接执行指令 Prompt

可以直接把下面这段指令发给 Codex：

```markdown
请阅读本项目中的 `CODEX_SHOWCASE_HANDOVER.md` 与 `reports/research_run/quant_evaluation_summary.json`，为 MatchQuant V1 构建一个高保真、科技感十足的多因子量化展示网站。
要求：
1. 采用深色量化对冲基金视觉风格（Dark Mode, 等宽字体，精致微光卡片）；
2. 包含 Hero 概览、赛前量化雷达（含胜率、期望进球与 Top3 因子归因）、12 黄金因子与被杀伪因子探索器、5 赛季逐季滚动盲测战绩（100% 胜率）、以及 0.20x 凯利实战纸盘收益曲线；
3. 数据全部来源于 `reports/research_run/` 下的 JSON 与 CSV 文件；
4. 保证响应式适配与极佳的交互动画体验。
```
