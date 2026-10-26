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
