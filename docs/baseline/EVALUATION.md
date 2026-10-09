# MatchQuant 现状评估与 baseline 验收规程

评估日期2026-10-04；对象为现有代码和迁移资产，不是新的实现交付。需求范围默认为baseline-v0.1轻量离线研究闭环。

## 1. 本轮证据与结论

| 检查 | 观察结果 | 可以得出的结论 |
|---|---|---|
| 起点代码 | HEAD `5aad2f1b3d2e8e007a3f7f9489d7a73f44ac4c94`；初始Git clean | 当前代码包括M11/M12；旧ZIP不是最新代码 |
| PATH Python | Python3.14环境缺pandas/numpy，测试导入失败 | 默认环境不能运行；不能据此说模型实现失败 |
| 独立全量离线测试 | bundled Python3.12.14，numpy2.3.5，pandas3.0.1；153项，33.111秒，OK | 现有测试在这个评估环境通过；独立安装尚未验证 |
| M1合成CLI | all三模型，退出0 | 现有离线回测CLI在评估环境可运行 |
| M11对M4合成对比 | Frequency/Elo/Poisson的概率最大绝对差均0 | JSON快照路径与参考预测在这组输入一致 |
| M11→M6直接结算 | 三模型单独尝试均拒绝：`unsupported forecast run schema_version; expected m4-forecast-v1` | 快照预测尚不能进入现有结算契约 |
| 原canonical hash | 实际值与manifest一致；3800行，cutoff2025-05-25 | 迁移包的这份历史CSV字节一致；未重新审计每个raw来源文件 |
| 正式路径/pointer | manifest记录的正式canonical路径和正式pointer不存在 | 数据尚未完成按原根目录恢复；无法在正式资产上声称M12 fresh |
| 旧真实体验报告 | M10 needs_review、sync dry-run、M6 not_run | 这份体验没有完成真实赛后apply/settle |
| Alpha/近期信息/页面 | 无M13–M17实现证据 | 仍属于路线设想 |

完整证据：仓库外 `D:\matchquant\assessment\2026-10-04\offline-tests.log`、`assessment.json`、`evaluate_baseline.py` 和其中引用的合成运行目录。评估输出未写入正式数据或current pointer；synthetic输入hash前后不变。

本轮没有在新venv安装、重新解析所有真实raw、恢复data roots、执行真实apply、真实M11发行、完整历史逐批次快照回溯、性能benchmark或干净机器发布验收。不能把153项通过扩大为这些尚未做的检查已通过。

结论：**现有M1–M12引擎可在验证环境运行；baseline-v0.1尚未交付。** 首要阻塞是独立环境/数据恢复、M11结算与审计桥接、运行包迁移；先完成这些再评价新Alpha是否有价值。

## 2. 缺口分级

| 级别 | 缺口 | 对策/负责人任务 |
|---|---|---|
| P1 | 用户默认环境缺依赖；源码外安装未验证 | B00/B05 |
| P1 | 数据路径尚未恢复，旧manifest无dataset_version、无正式pointer | B01 |
| P1 | M11产物与M6 schema不兼容；M8/M9/M10仍绑定M4 | B02/B03，保留旧流程 |
| P1 | 旧快照可能包含绝对state路径，搬家不保证可读 | B04派生封装与迁移验收 |
| P1 | 暂无用户端无history的审计→结算完整证据 | B03–B06 |
| P2 | 文档混用包版本与worktree状态 | 最新入口和B07文档修订 |
| P2 | 旧真实体验是needs_review回溯demo，不能当完整发布证据 | B06重建真实回溯链并准确标记 |
| P2 | fresh、as_of、used cutoff和实时覆盖可能被混淆 | 主计划时间/状态合同，B03/B05显示 |
| P2 | pyproject无依赖上界，examples可能不随wheel安装 | B00锁定、B05安装验收 |
| P3 | 无Alpha、近期信息、页面 | M13–M17扩展，不阻塞v0.1 |

P1表示阻塞本次目标，不代表线上系统已发生事故。未知或没做的检查用NOT_RUN，不写FAIL来混淆已观察缺陷与缺失证据。

## 3. 工程验收矩阵（B06）

每行都保存可复跑命令、输入hash、代码版本、运行环境、输出和状态。全部必需项PASS才可交付v0.1。以下目标未在本轮整体执行。

| ID | 用例 | 必须满足 |
|---|---|---|
| E01 | 新venv安装 | 从最终源码/软件包安装；非源码cwd可调用CLI；依赖精确记录 |
| E02 | 全量离线测试 | 所有旧测试和有意义新测试通过，无临时skip或联网 |
| E03 | 三模型回归 | Frequency/Elo/Poisson原算法与时间规则保持，概率有限非负和为1 |
| E04 | 模型JSON round-trip | 支持模型配置/state一致，拒绝未知模型/非法数值/缺字段 |
| E05 | M4与M11等价 | 相同history/as_of/config/fixtures逐行比对，`atol=1e-12,rtol=0` |
| E06 | cutoff之后变异 | 改未来比分/加未来比赛不改变边界前state和预测；数据hash/manifest可相应变 |
| E07 | 同日与同timestamp泄漏 | 改同批次任一结果不能改变该批次其它预测；全预测完成后才更新 |
| E08 | selection时序 | validation结束在可用时间之前；final test不选参；相关缺失证据明确拒绝/审查 |
| E09 | 用户端无history | 断开history/processed manifest后，包加载、session、audit、settle仍可运行；未读归档路径 |
| E10 | 网络为0 | patch下载/HTTP/socket入口或观测调用；预测/审计/结算无网络调用，不只检字段 |
| E11 | 篡改/缺失 | 修改state/manifest/predictions/fixture/bundle文件或删必需文件时阻断，记录原因 |
| E12 | 输出目录隔离 | root在snapshot、bundle、输入归档内时拒绝；输入hash前后不变 |
| E13 | schema与模型数 | 旧M4与单模型M11成功；未知schema/多模型首版明确拒绝，无M4伪装 |
| E14 | 赛程时间/比分 | before-as_of、有真实比分、非法identity、重复fixture与时刻歧义拒绝 |
| E15 | 结算覆盖 | complete/partial/unresolved/ambiguous/time_conflict/date_changed行为正确；无匹配指标为null |
| E16 | 原预测锁定 | 新结果、历史apply或新快照不改旧run/bundle；能复现旧概率 |
| E17 | 状态准确 | needs_review/blocked与退出码、理由一致；fresh仅为发行时关联；replay非实时 |
| E18 | 路径迁移 | 不同绝对目录、原路径不可达下运行；派生lineage正确、state hash相同 |
| E19 | 真实数据链 | raw→canonical→manifest→版本/pointer→snapshot→bundle→session→settle可追踪 |
| E20 | 最小用户闭环 | 新环境照说明完成synthetic，再完成真实回溯；无内部隐含路径 |
| E21 | 大小/性能 | 记录包大小和以下性能合同；不通过时定位history读取或其它原因 |
| E22 | 失败中间产物 | IO失败、无待赛、审计拒绝保留阶段证据，不自动apply/rebuild，不产假成功 |
| E23 | 发行清单 | 功能范围、代码/依赖/schema/data版本、hash、许可状态和已知限制齐全 |

M12 fresh/stale/unverifiable分别测试；stale不破坏旧快照；无pointer不能生成fresh。M12的pointer检查需要历史文件，限定在研究端，不把它放进E09的用户查询路径。

## 4. 真实历史回溯与参考结果

### 4.1 已有历史指标（本轮读取旧记录，未复跑）

同一canonical SHA-256 `9161416d...e0c36eb`，carry模式，测试边界 `[2024-08-01,2025-06-01)`，380场。下表供重现回归，不能用于重新选参或声称未来表现。

| 模型/配置 | Log Loss ↓ | 三类Brier（平方和）↓ | Accuracy |
|---|---:|---:|---:|
| M1 Frequency默认 | 1.0810247852 | 0.6556525399 | 0.4078947368 |
| M1 Elo默认 | 1.0045134099 | 0.5988645946 | 0.5263157895 |
| M1 Poisson默认 | 1.0565088352 | 0.6373735998 | 0.4394736842 |
| M3 `elo_k32_draw027` | 0.9978217866 | 0.5941995193 | 0.5315789474 |

M3记录参数为k_factor=32、home_advantage=50、draw_base=0.27，multiclass ECE为0.0460981916，且标记 `diagnostic_only_not_calibrated`。manifest provenance在旧记录中为 `manifest_fields_incomplete`，data_version=null。

来源分别为 `D:\matchquant\_unpacked\local_state\matchquant_runs\epl_2024-25_holdout\metrics.json` 和 `controller_m3_audit\m3_select_20260905T071925Z_53d1239a\final_metrics.json`。后续复跑必须核对对应run_config和完整默认配置，不能仅凭这几项显式参数猜其它默认值。

### 4.2 baseline真实回溯步骤

1. 冻结hash一致的真实历史和模型配置；旧holdout只做实现回归。
2. 按日期/时间批次产生score-free fixtures，在该批次前as_of构建M11状态；建立与同状态M4参考的对照。
3. 锁定原预测hash，再用独立canonical结果结算；每批次结束后才允许未来批次状态加入该批结果。
4. 对照fixture/model/class顺序，验证逐概率与逐结算行，不只比较一个总体Accuracy。
5. 对全部380场重现walk-forward指标；允许构建多个快照，不能用一个赛季开始快照预测全季后声称它等价于在线更新回测。
6. 使用随机抽取批次时先固定seed和抽取规则；全集指标复跑仍要覆盖完整边界。每个批次记录as_of、used cutoff、hash和输出。

M4/M11合成等价已经通过，但不能代替本段真实回溯。重现旧指标的建议数值容差为1e-10；若库版本差异超过容差，先检查行集/参数/边界，再解释数值差异，不改算法迎合数字。

## 5. Alpha/融合的研究评估合同

本节用于M13–M15，不是v0.1工程发布门槛。

已查看2024-25结果，不能再次称其untouched。可将2015-16至2021-22用于状态初始化，2022-23与2023-24用于早期validation，但任何新增候选都要承认已知后期结果的研究暴露。新的2025-26数据只有在确认未被用于任何候选选择/人工改参数时才可作为新测试；无法确认则标记retrospective，另留未来锁定预测作为prospective test。

候选/因子/窗口/校准方法先记录到evaluation_contract，再运行。选参只用时间验证，final test只评一次；测试结果引发修改时，下一版需要新的验证与测试安排，不能反复使用同一test作为调参反馈。

主指标pooled multiclass Log Loss，辅指标Brier；Accuracy只作辅助。报告n、coverage、按赛季表现、每类reliability/ECE、冷启动/升班队/缺信息样本；不只给总平均。

研究“采用候选”默认门槛：

- 在validation pooled Log Loss优于baseline，且Brier不出现解释不了的明显退化。
- 在未参与选择的测试中 `ΔLL=LL_candidate-LL_baseline` 的均值<0；采用按自然周分块的配对bootstrap，固定seed、1000次，95%区间上界<0才称有改善证据。样本不足或跨周依赖明显时调整预注册block长度，并披露敏感性。
- Brier退化容忍阈值预注册，默认均值增量≤0.005；这个数是设计决策，不是经验已证明的通用标准。不能看test后调阈值。
- 全部时序/缺失/降级工程检查通过；新信息缺失时概率等于baseline。

未达到这些条件，结论为“证据不足，保持baseline默认”，不为完成复杂功能而强制发行候选。正常通过算法测试与概率求和不叫已校准；需要独立校准诊断与不同时间段证据。

不要求固定命中率、不承诺盈利，不把回测提高当成未来表现证明。模型候选超过原预注册清单时，应披露探索次数和选择偏差，而不是只保存赢家。

## 6. 性能、重复性与维护评估

性能在B00记录的同一Windows环境测量，记录CPU/内存、解释器、包大小、输入行数、cold/warm条件及是否含Python启动。

- runtime≤5MiB，不含软件依赖；必须没有完整历史文件，体积超限以文件构成定位。
- primary模型20场批量query包含CLI启动，10次重复后报告median/p95；默认体验目标p95≤5秒。同机不能达标时先分析，不能据单次慢启动直接宣布算法不合格。
- 从3800场历史build≤60秒是目标；build属于研究端，不进入单场查询时延。
- 同输入30次运行不修改输入；预测数值/排序稳定，UUID和时间戳允许不同；每次新输出目录可追溯。
- 维护循环能只执行显式dry-run→apply→status→rebuild，新包不会改变旧包；no_pending和source失败不隐式寻找替代数据。

性能目标是本计划的产品设计约束，本轮未benchmark。若机器条件影响结果，可先给REWORK并调整经记录的设备目标，但不能悄悄删掉未通过用例。

## 7. 评估方工作方式

每个任务先看实现方提交的契约和证据，再检查diff，独立运行针对本任务的必要检查与全量回归。评估不代做实现方未完成的代码，也不只看一个PASS截图。

评估返回：

```text
任务/版本：...
结论：PASS / REWORK / BLOCKED
已核实：...
未执行：...
阻塞缺陷：文件/API/触发输入/影响/修正验收
证据：命令、日志、输出和hash
范围与风险：...
允许进入的下一任务：...
```

PASS是该任务的设计与证据门槛通过；REWORK表示实现存在可修复缺陷；BLOCKED表示缺少必要输入、版本关联或该轮变更授权。不要因过程较难就判BLOCKED，也不要在没运行必需检查时判PASS。

最终报告分别给出“工程可交付”“研究改善证据”“当前数据覆盖”三项结论。v0.1可以工程PASS、无新增改善主张、当前覆盖未证明；它的交付标签必须仍是本地research build。
