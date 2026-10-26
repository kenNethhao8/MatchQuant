# MatchQuant v2.0 项目总纲（Master Charter）与分阶段协同执行规范

> **本文件为 MatchQuant 项目的最高架构宪法（Single Source of Truth）。**
> 任何新建会话的模型（无论是主架构师模型还是执行端 `3.8 flash` 模型）在开始任何任务前，**必须首先阅读并严格遵守本总纲**，严禁偏离初心或自行发明工程包装概念。

---

## 一、 项目初心与第一性原理（Core Quantitative Mission）

1. **唯一北极星目标**：
   用严密的量化投资思维（因子挖掘 $\rightarrow$ 单因子 IC 与正交残差检验 $\rightarrow$ 足球领域概率建模 $\rightarrow$ 样本外盲测与归因），将影响足球比赛结果的真实球场技战术、运气均值回归、赛程体能与市场聪明钱信息转化为**可验证、可解释、无穿越的预测信号**。
2. **双轨制建模原则（Dual-Track Isolation）**：
   - **Track A（纯球场基本面主模型）**：**100% 严禁输入任何博彩赔率**。纯粹依靠历史比分、射门、射正、角球、半场过程、红黄牌、休息天数与升班马状态预测胜平负与期望进球，直接验证基本面因子的真实 Alpha。
   - **Track B（市场融合增强模型）**：在 Track A 纯基本面因子之上，叠加去水后的机构初盘共识与临场聪明钱漂移（Steam Move），追求极致的赛前预测精度。
   - **评价标尺**：同时对比 **动态 Elo 基线（下限）**、**Pinnacle 幂次法去水初盘** 与 **Pinnacle 去水终盘（市场效率天花板）**。

---

## 二、 防偏离与防幻觉“五条铁律”（Anti-Drift Iron Laws）

执行模型在任何阶段**绝对禁止**触碰以下红线：

1. **铁律 1（严禁工程官僚主义）**：
   严禁编写任何形式的 `bundle`、`release_verify`、`snapshot_notarization`、`manifest_hash_chain`、`session_lock` 等合规公证代码。所有代码必须 100% 服务于数据清洗、因子计算、统计检验、概率模型与回测归因。
2. **铁律 2（严禁赛前时间穿越 / Look-Ahead Bias）**：
   计算任何一场比赛的赛前因子时，必须按比赛日/开球批次（`_batch_key`）严格冻结状态：**先统一提取同批次所有比赛的赛前因子，再统一用赛后结果更新球队与裁判状态**。篡改未来或同批次比分，绝不能改变当前赛前因子的哪怕小数点后第 12 位。
3. **铁律 3（严禁盲目堆砌未检验因子）**：
   任何新设计的因子，**不准直接塞进预测模型**。必须先进入 Factor Lab 接受**“扣除 Elo 实力后的正交残差检验（Orthogonal Residual $t$-test）”**。如果一个因子表面相关性高，但扣除 Elo 后正交 $|t| < 1.20$，说明它只是“强队影子变量”，必须淘汰。
4. **铁律 4（严禁共线性裸奔与符号反转）**：
   严禁将同一个统计量的 3/5/10 场窗口同时塞入线性层；严禁使用 `missing_flag` 0/1 指示器列污染权重；模型训练必须带有**足球经济学方向约束（Sign-Consistency Bounds）**，绝不允许出现“射正越多或拿分越多反而降低赢球概率”的符号反转。
5. **铁律 5（严禁使用无序 Softmax 忽略平局结构）**：
   足球赛果具有天然序数关系（客胜 `<` 平局 `<` 主胜），且平局受实力接近度与比赛总节奏支配。必须使用 **有序逻辑回归（Ordered Logit）** 或 **因子驱动的双变量泊松进球模型（Factor-Driven Dixon-Coles）**。

---

## 三、 “主架构师 + Flash 执行官”双模型协同工作模式（Workflow Protocol）

为彻底避免长上下文导致的记忆衰减与目标偏离，项目采用 **“阶段设计书驱动 + 独立新会话执行 + 阶段门禁审核”** 的闭环模式：

```
[主架构师模型] 出具《阶段 N 设计书》(Stage N Spec)
       │
       ▼
[用户开启新会话 -> 3.8 Flash 执行官]
  1. 读取 QUANT_MASTER_CHARTER.md + 《阶段 N 设计书》
  2. 仅在指定文件范围内编码与运行测试命令
  3. 严禁跨阶段超前开发，完成后输出《阶段 N 执行回执》
       │
       ▼
[主架构师模型] 审核代码、测试结果与量化指标报告
  ├── 若未达标：出具精确修复指令 -> Flash 修正
  └── 若达标：签署通过，出具《阶段 N+1 设计书》
```

### 给 `3.8 Flash` 执行模型的行为准则：
1. **每次新会话只干一件事**：只执行当前《阶段设计书》指定的任务，不碰未授权的文件。
2. **固定运行环境**：
   - 工作树目录：`D:\matchquant\worktree`（分支：`quant-v2`）
   - 指定 Python 解释器：`D:\Dev\Python314\python.exe`（已预装 `numpy`, `pandas`, `scipy`, `scikit-learn`）
   - 环境变量：执行命令前始终设置 `$env:PYTHONPATH = "D:\matchquant\worktree"`
3. **交付格式**：执行完毕后，向用户汇报：
   - 修改/新增的文件清单；
   - 单元测试与 CLI 命令的实际终端输出；
   - 阶段量化红线指标的实际数值（如 LogLoss、正交 $t$ 值、VIF 等）。

---

## 四、 全景五阶段演进路线图（5-Stage Master Roadmap）

整个 MatchQuant v2.0 量化体系划分为 **5 个独立可验收的阶段**。目前阶段一至四已在 `quant-v2` 分支完成基准实现并通过首轮验证，后续可按需对任一阶段进行复核微调，或直接推进阶段五实战深化：

| 阶段编号 | 阶段名称 | 核心负责模块 | 关键交付物 | 量化验收门禁（Gate Criteria） | 当前状态 |
| :---: | :--- | :--- | :--- | :--- | :---: |
| **阶段一** | **仓库净空与 44 列富化数据底座** | `matchquant/rich_data.py` | 清理 42 个旧公证模块；生成 11 赛季 4,180 场 `epl_rich_canonical.csv`（含射门/射正/角球/半场/红黄牌/裁判/去水初终盘/亚盘） | 1. 4,180 场主键与时间零缺失；<br>2. 幂次法去水概率和误差 $<10^{-6}$；<br>3. 单元测试全绿 | ✅ 已完成基线 |
| **阶段二** | **零穿越时序因子引擎与对手强度校准** | `matchquant/factor_engine.py` | 5 大族 22 个候选因子（含对手 Elo 强度校准、快慢双速 EWMA、跨季均值回归、升班马冷启动） | 1. 篡改未来或同批次比分，历史赛前因子差异 `= 0.0`；<br>2. 全量 22 因子无无穷大或未处理 NaN | ✅ 已完成基线 |
| **阶段三** | **单因子体检实验室与正交残差筛选** | `matchquant/factor_lab.py` | 输出 `factor_lab_report.md`（Rank IC、跨季 `IC_IR`、扣除 Elo 后正交 $t$ 值、五分位单调性、VIF 剪枝） | 1. 自动识别并拦截 $|t_{\text{ortho}}| < 1.20$ 的 Elo 影子伪因子；<br>2. 入选因子间 $|\rho| \le 0.83$ | ✅ 已完成基线 |
| **阶段四** | **足球双轨领域模型与单场因子归因** | `matchquant/quant_models.py`<br>`matchquant/quant_pipeline.py` | `OrderedFactorModel` + `FactorDixonColesModel`；Track A & B 样本外盲测报告 `quant_research_report.md` | 1. 因子权重 100% 零符号反转；<br>2. 样本外（760 场）Track A 纯基本面 LogLoss 优于 Elo 达 `-0.020+`；<br>3. Track B 优于 Pinnacle 初盘 | ✅ 已完成基线 |
| **阶段五** | **跨赛季滚动重训、新因子扩展与市场定价偏差（CLV/Edge）捕捉** | `matchquant/factor_engine.py`<br>`matchquant/quant_models.py`<br>`matchquant/quant_pipeline.py` | 1. 逐赛季滚动扩展窗口重训（Walk-Forward Retraining）；<br>2. 新增密集赛程负荷、积分榜保级/争冠战意因子并过 Factor Lab 检验；<br>3. 纯基本面 vs 初盘定价背离（Edge & CLV）量化回测报告 | 1. 新因子必须通过正交 $|t| \ge 1.20$ 方可入模；<br>2. 滚动重训下 Track A 持续跑赢 Elo；<br>3. 验证当 Track A 与初盘出现显著正背离时，能否捕获正向收盘线价值（CLV） | 🎯 待出设计书交由 Flash 执行 |

---

## 五、 核心代码目录契约（Repository Layout Contract）

在 `D:\matchquant\worktree`（`quant-v2` 分支）中，只允许保留以下清晰结构：

```text
D:\matchquant\worktree\
├── QUANT_MASTER_CHARTER.md        # 项目最高总纲（本文件）
├── docs/stage_specs/              # 存放各阶段设计书（Stage Design Specs）
├── matchquant/
│   ├── rich_data.py               # [Layer 1] 11赛季44列富化比赛与去水赔率提取
│   ├── factor_engine.py           # [Layer 2] 零穿越批冻结时序因子引擎（对手强度校准）
│   ├── factor_lab.py              # [Layer 3] 单因子IC、Elo正交残差t检验、五分位与VIF实验室
│   ├── quant_models.py            # [Layer 4] 有序Logit + 因子Dixon-Coles双变量泊松 + 归因
│   ├── quant_pipeline.py          # [Layer 5] 训练/验证/盲测全链路与Markdown报告生成
│   ├── cli.py                     # 极简CLI入口 (rich-prepare / factor-lab / quant-research)
│   └── (schema.py, models.py, backtest.py, metrics.py, calibration.py, selection.py, epl.py, fixtures.py, io.py)
└── tests/
    └── test_quant_v2.py           # 核心量化防穿越、去水、正交检验与模型符号一致性测试
```
