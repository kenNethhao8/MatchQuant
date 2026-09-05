# MatchQuant M1–M6 — Research Engine

MatchQuant 是一个低成本、可复现的足球 1X2（主胜/平/客胜）预测与回测最小框架。M1 的核心运行和 M2 的数据适配只依赖 Python、pandas 和 numpy，不依赖 LLM、API Key、付费服务、数据库或网页。

## 输入格式

CSV 必须包含以下列：

```text
match_id,date,season,home_team,away_team,home_goals,away_goals
```

校验会拒绝缺失字段/值、重复 `match_id`、同赛季同日期同主客队的重复 fixture、非法日期、负数或非整数比分，以及主客队相同。`season` 原样作为输入字段使用，不从月份推断。

`date` 可以是 `YYYY-MM-DD` 或带开球时间的 ISO 日期时间。只有日期时，同一日所有比赛先完成预测，再统一用实际比分更新；带时间时按时间批次推进，同一时间戳也先全预测后更新。排序由时间批次和 `match_id` 决定，因此 CSV 行顺序不会改变结果。

合成样例位于 [`examples/synthetic_matches.csv`](examples/synthetic_matches.csv)，明确仅用于演示端到端流程。

## M2：真实英超数据

M2 使用 [Football-Data.co.uk 的 England results 页面](https://www.football-data.co.uk/englandm.php)及其公开的 [E0 CSV 端点](https://www.football-data.co.uk/mmz4281/2425/E0.csv)。该来源无需 Key；`E0` 是英格兰顶级联赛的代码。适配器只使用源字段 `Date`、`HomeTeam`、`AwayTeam`、`FTHG`、`FTAG`，球队名按源文本保留，不做未经证实的别名替换。

默认范围是十个完整赛季：2015-16 至 2024-25。当前进行中的赛季不在默认范围内，且命令会为每个请求赛季留下记录；下载后会执行同一套源数据质量审计，HTTP、解析、字段、非法行、重复 fixture 或非 380 场完整赛季都会写入失败 manifest 后使命令失败，不会静默跳过。

实际原始数据和处理产物默认写入 `D:\Codex\matchquant_data\`，不会写入 Git 仓库：

- `raw/E0_YYYY-YY.csv`：原始下载文件；
- `raw/source_manifest.json`：每个文件的赛季、URL、UTC 抓取时间、文件名、HTTP 状态、解析状态、SHA-256 和下载时审计结果（日期格式、行数、状态）；
- `processed/epl_canonical.csv`：可直接输入 M1 的 canonical CSV；
- `processed/epl_processed_manifest.json`：所有原始输入 hash、适配器/格式版本、输出 hash、行数和生成时间；
- `reports/epl_quality_report.json`：每赛季行数、球队数、日期范围、重复/无效行数、380 场预期和警告。

原始快照不可覆盖；如需重新抓取，请使用新的 `--data-dir` 保存新的本地快照，并以其 manifest/hash 区分版本。

网络下载必须显式触发；`run` 命令只读取本地文件：

```powershell
python -m matchquant epl-download `
  --seasons 2015-16 2016-17 2017-18 2018-19 2019-20 `
             2020-21 2021-22 2022-23 2023-24 2024-25 `
  --data-dir D:\Codex\matchquant_data

python -m matchquant epl-prepare `
  --seasons 2015-16 2016-17 2017-18 2018-19 2019-20 `
             2020-21 2021-22 2022-23 2023-24 2024-25 `
  --data-dir D:\Codex\matchquant_data `
  --output D:\Codex\matchquant_data\processed\epl_canonical.csv
```

`epl-prepare` 完全离线运行，并在写出前调用 M1 的 `validate_matches`。Football-Data 每个文件的日期格式会在严格的 `DD/MM/YYYY` 或 `DD/MM/YY` 中择一识别，混用或其它格式会失败，之后输出 `YYYY-MM-DD`；赛季始终由命令/manifest 指定，绝不从日期推断。`match_id` 只由规范化赛季和去除首尾空白的源主客队文本决定，不包含日期或比分，因此日期/比分修订仍可识别为同一场比赛；canonical 字段仍保留源球队显示文本。非法日期、比分、球队字段、重复 fixture、原始 hash 不匹配，或有效比赛数不是 380，都会使准备失败；失败时仍写出详细质量报告，但不会写出该输入的 canonical 或 processed manifest。可检测的大小写/空白球队名变体只记录为警告，不自动猜测别名。

以完整后期赛季 2024-25 为严格 holdout 的 M1 冒烟回测：

```powershell
python -m matchquant run `
  --input D:\Codex\matchquant_data\processed\epl_canonical.csv `
  --test-start 2024-08-01 `
  --test-end 2025-06-01 `
  --model all `
  --season-mode carry `
  --output-dir D:\Codex\matchquant_runs\epl_2024-25_holdout
```

这是研究数据管线和基线回测，不是已校准的概率系统；真实数据回测结果不代表可用于博彩，也不构成投注建议。已知限制包括：源数据是比赛日级别而非完整开球时间、没有阵容/伤停/赔率校准层、数据源未来可能修订，且适配器不会猜测球队别名。

## M2.1：显式增量同步与版本化

`epl-sync` 是唯一新增的增量同步入口。默认是 dry-run；`run` 仍只读取显式本地 CSV，绝不会联网。网络同步必须明确给出赛季，并把原始文件写入 `D:\Codex\matchquant_data\staging\epl_sync_<UTC时间>_<唯一ID>\` 下的新快照：

```powershell
python -m matchquant epl-sync `
  --seasons 2025-26 `
  --data-dir D:\Codex\matchquant_data `
  --dry-run
```

也可以用已有 staging 快照做完全离线的重算/审计：

```powershell
python -m matchquant epl-sync `
  --staging-dir D:\Codex\matchquant_data\staging\epl_sync_<...> `
  --data-dir D:\Codex\matchquant_data
```

同步只对请求赛季替换/追加候选，未请求赛季从 active baseline 原样带入。默认每个请求赛季都必须有恰好 380 场已完成比赛；不根据日期推断赛季是否进行中。只有在请求列表中明确指定唯一的 `--in-progress-season YYYY-YY` 后，该赛季才允许少于 380 场，且两项全场比分都为空的合法 fixture 会逐条记入 `skipped_unfinished`：

```powershell
python -m matchquant epl-sync `
  --seasons 2025-26 `
  --in-progress-season 2025-26 `
  --data-dir D:\Codex\matchquant_data `
  --dry-run
```

`--in-progress-season` 必须是请求赛季之一；部分比分、坏日期、坏球队/比分、重复 fixture 或重复 `match_id` 始终会使同步失败。质量、差异和 processed manifest 会记录声明值、每赛季策略以及 `strict_380_gate_applied`。M2 `epl-prepare` 的完整赛季 380 场门槛不变。

每次同步会在 staging 中留下 raw source manifest、候选 canonical、质量报告、差异报告和 processed manifest。差异报告包含 `added`、`unchanged`、`revised`、`missing_from_candidate`、`skipped_unfinished` 的数量和明细，并记录 baseline/candidate hash、来源 URL、抓取时间、数据 cutoff 和适配器版本。检测到 `revised` 或 `missing_from_candidate` 时，`--apply` 必须人工审查而被拒绝。

只有无修订、无缺失且所有硬校验通过时，才允许：

```powershell
python -m matchquant epl-sync `
  --seasons 2025-26 `
  --data-dir D:\Codex\matchquant_data `
  --apply
```

成功 apply 会新建 `processed\versions\m2.1-<staging快照名>\`，保存 canonical、processed manifest、质量/差异报告，再原子更新 `processed\epl_current.json`。旧的 M2 `processed\epl_canonical.csv`、manifest、raw 快照和报告不会被覆盖；初始 M2 数据无需迁移。

## M3：严格时序模型选择与概率校准诊断

M3 的 `select` 命令只读取显式给出的本地 canonical CSV、仓库内候选配置和相邻的可选 processed manifest，不联网、不调用 `epl-sync`，也不会修改 `D:\Codex\matchquant_data\`。默认输出为 `D:\Codex\matchquant_runs\m3_select_<UTC时间>_<唯一ID>\`。

候选配置位于 [`examples/m3_candidates.json`](examples/m3_candidates.json)，枚举 M1 已有的 Frequency、Elo、Poisson 及其少量参数组合；它不会改变 `run` 命令的默认模型或参数。示例：

```powershell
python -m matchquant select `
  --input D:\Codex\matchquant_data\processed\epl_canonical.csv `
  --candidate-config examples\m3_candidates.json `
  --validation-fold 2022-08-01..2023-07-31 `
  --validation-fold 2023-08-01..2024-07-31 `
  --final-test-start 2024-08-01 `
  --final-test-end 2025-06-01
```

每个 validation fold 都从该 fold 起点之前的历史重新按时间回放，保留同日/同时间批次先预测后更新；fold 必须按时间排列、不重叠，并在 final-test 起点前结束。候选按所有 validation 比赛的 pooled 平均 multiclass log loss 排名，Brier 只作次级 tie-break，最后用候选名的稳定字典序、再用配置顺序解决并列。选中候选之后只对 final-test 做一次回放；final-test 指标不会参与选择。日期形式的 fold 端点遵循 M1 的日历日批次语义。

每次 M3 run 保存 `run_config.json`（输入 CSV SHA-256、候选配置 hash、manifest/cutoff 状态、边界、全部候选指标、选择规则、最终指标和输出路径）、`validation_metrics.json`、选中候选的 `predictions.csv`、`final_metrics.json` 和 `calibration_reliability.csv`。后者为 home/draw/away 各自的固定等宽分桶表，并报告 multiclass ECE；空桶保留为零计数的有限值行以保持稳定表形。ECE 和 reliability 是诊断，不代表概率已经校准；现有 log loss/Brier/accuracy 仍保留。若相邻 manifest 缺少 `dataset_version`，run 会明确记录缺失，而不会伪造版本号。

## M4：本地 Alpha 预测工作流

M4 接受两个相互独立的本地 CSV：历史结果必须是 M1 canonical schema，待赛赛程必须是无比分的 fixture schema。两者不能混写。fixture 模板至少包含：

```text
fixture_id,date,season,home_team,away_team
```

仓库提供 [`examples/synthetic_fixtures.csv`](examples/synthetic_fixtures.csv) 作为小型合成端到端样例；它不是当前真实赛程。fixture 校验会拒绝缺列/空值、非法日期、重复 `fixture_id`、同赛季同日期同主客队重复、主客队相同、canonical `match_id`，以及 `home_goals`/`away_goals` 中任何已填写的比分。空的比分列可以存在于外部模板中，但不会被当作结果输入。

预测命令只读显式本地文件，不联网，不调用 `epl-sync`，也不会更新 `D:\Codex\matchquant_data\`：

```powershell
python -m matchquant forecast `
  --history examples\synthetic_matches.csv `
  --fixtures examples\synthetic_fixtures.csv `
  --as-of 2024-01-07 `
  --model poisson `
  --model-config examples\forecast_poisson.json `
  --output-root D:\Codex\matchquant_runs
```

`--model` 和 `--model-config` 是显式模型模式；JSON 可以是模型参数对象，也可以是包含 `model` 与 `config` 的对象。也可以把 M3 选择 run 的 `run_config.json` 显式交给预测命令，由程序提取 `selected_candidate`：

```powershell
python -m matchquant forecast `
  --history D:\Codex\matchquant_data\processed\epl_canonical.csv `
  --fixtures D:\Codex\my_fixtures.csv `
  --as-of 2025-08-01 `
  --selection-run-config D:\Codex\matchquant_runs\m3_select_<...>\run_config.json
```

两种模型选择方式互斥；M4 不设置新的隐式模型。选择 M3 run 时，M4 还会严格检查 `validation_folds`、`selection_rule` 和 `final_test` 边界：至少要有一个有效 validation fold，且每个 fold 的有效结束边界必须不晚于本次 `--as-of`。date-only 的 fold 结束日按整日结果处理，因此 date-only `--as-of` 不会把同日结果视为已知；精确时间 `--as-of` 遇到同日 date-only fold 结束边界会因时刻歧义直接拒绝。`selection_rule.final_test_metrics_used_for_selection` 必须明确为 `false`；M3 final-test 可以在 `--as-of` 之后存在，因为它不参与候选选择，但这一点会写入 `run_config.json` 和 `forecast_report.md`。若 M3 run 的 history SHA-256 与本次 `--history` 不一致，命令仍可在明确请求下运行，但会标记 `history_hash_mismatch`，不得把该选择证据理解为本次 history 的匹配训练证据。结构无效、验证期越过 as-of、候选缺失、模型配置无效或 hash 字段不合法会直接失败。

`--as-of` 是预测快照边界：日期形式（如 `2024-01-07`）排除该日及之后的所有历史结果；精确时间形式只纳入严格早于该时间批次的历史结果。同一精确时间戳先作为一个批次处理；如果边界所在日混有 date-only 记录，精确时间边界会被拒绝，避免把未知时刻误当成已知时刻。所有 fixture 必须不早于 `as-of`；一次 forecast 只用 as-of 之前的历史更新模型，任何待赛 fixture 都不会更新模型状态。

每次运行在 `D:\Codex\matchquant_runs\`（或 `--output-root`）下新建唯一的 `m4_forecast_<UTC时间>_<ID>` 目录，保存：

- `predictions.csv`：fixture 信息、H/D/A 概率、`model_most_likely_result`（仅表示“模型最可能结果”）及 Poisson 模型的期望进球字段；不含未来真实比分；
- `forecast_report.md`：中文快照说明、输入证据、逐 fixture 结果、限制与免责声明；
- `run_config.json`：历史/fixture/选择配置的路径和 SHA-256、as-of、实际历史 cutoff/行数、模型配置、选择证据状态、selection as-of 状态、M3 验证期最新结束边界、final-test 未参与选模的说明、`predictions.csv` SHA-256、输出路径和免责声明。

M4 输出是本地研究快照，不是当前实时赛程预测。概率经过有限、非负、和为 1 的合法性检查，但没有因此声明已经校准；没有外部赔率、阵容/伤停/新闻、机器学习、实时服务、数据库或自动调度，也不构成投注建议、盈利承诺或未来表现保证。若需要真实历史时点 replay，应在 `D:\Codex\matchquant_runs\` 临时生成无比分 fixture 输入，并在记录中明确 as-of 和输入 hash；它仍不是当前实时赛程预测。

## M5：本地英超赛程快照接入

M5 只接受用户明确传入的本地 Football-Data.co.uk `E0` 兼容 CSV。它不下载、不联网、不更新 `D:\Codex\matchquant_data\processed\epl_canonical.csv`、`epl_current.json` 或任何 active/current fixture pointer；完成赛果也不会自动写入 M2/M2.1 历史数据。仓库内的 [`examples/synthetic_e0_source.csv`](examples/synthetic_e0_source.csv) 只是小型合成测试输入，不代表当前英超赛程。

手动闭环如下：

1. 用户在本地下载或导出一个明确的 E0 风格 CSV，确认至少有 `Date,HomeTeam,AwayTeam,FTHG,FTAG`。M5 不检查网络可达性，也不把文件名推断为 URL；`--source-url` 或 `--source-label` 只有在用户明确提供时才写入 provenance。
2. 用显式赛季创建不可变 snapshot：

   ```powershell
   python -m matchquant epl-fixtures-prepare `
     --source-csv D:\Codex\my_input\E0.csv `
     --season 2025-26 `
     --source-label "local Football-Data E0 export" `
     --data-root D:\Codex\matchquant_data
   ```

   每次调用都会在 `D:\Codex\matchquant_data\fixtures\snapshots\` 新建唯一目录，复制原始 bytes、计算 SHA-256，并保存 manifest、质量报告和（仅当有合法 pending 行时）`processed\epl_fixtures.csv`。snapshot 记录输入路径、赛季、全源有效日期范围、completed 结果日期范围、pending 日期范围、completed-data cutoff、行分类和版本；raw copy 是只读快照且不被覆盖。`data_cutoff` 只表示 completed 结果的最新日期，不会把未来 pending 日期当作数据 cutoff。两个比分均为空才是 pending；两个比分均有效的 completed 行只进入质量证据。部分比分、坏日期/球队/比分、重复 stable fixture ID 和同 fixture 冲突会失败且不产生 fixture canonical。没有 pending 时状态为 `no_pending_fixtures`，只留证据，也不生成空 CSV。
3. 如果需要版本观察，可额外传入 `--baseline-fixtures`。M5 只按 `fixture_id` 报告 `added`、`unchanged`、`revised`、`missing_from_candidate`；stable ID 包含赛季和清理后的主/客队文本，因此源队名文本变化通常表现为 `added` + `missing_from_candidate`，并要求人工判断；只有 ID 实际保持不变时才报告 `revised`。M5 不做模糊别名、不删除 baseline、不 apply、不更新指针。
4. 人工明确选择 history、M3 selection evidence 或显式 model config，以及本次 `as-of`。然后把 M5 生成的 `processed\epl_fixtures.csv` 作为 M4 的 `--fixtures`：

   ```powershell
   python -m matchquant forecast `
     --history D:\Codex\matchquant_data\processed\epl_canonical.csv `
     --fixtures D:\Codex\matchquant_data\fixtures\snapshots\epl_fixtures_<...>\processed\epl_fixtures.csv `
     --as-of 2025-08-01 `
     --model poisson `
     --model-config examples\forecast_poisson.json `
     --output-root D:\Codex\matchquant_runs
   ```

   M4 仍只读显式本地文件；fixture 与 results schema 保持分离，预测运行另存自己的输入 hash、as-of 和报告。每一步都可从 raw snapshot、manifest、质量/差异报告和 M4 run config 审计。snapshot 的日期范围和本地版本不等于实时或完整赛程，也不证明来源具有再分发授权。

## M6：本地预测结算与评估闭环

M6 用一个已经完成的 M4 forecast run 和用户明确指定的本地 M1 canonical 最终赛果 CSV 做一次离线结算。它只读取 run 目录中的 `run_config.json`、`predictions.csv` 以及 `--results` 指定的结果文件；不下载赛果、不调用 `epl-sync`、不写入 `D:\Codex\matchquant_data\`，不更新 active/current pointer，不重跑模型，也不修改任何输入。

手动闭环示例（先完成 M4 forecast）：

```powershell
python -m matchquant settle `
  --forecast-run D:\Codex\matchquant_runs\m4_forecast_<...> `
  --results D:\Codex\my_results\epl_canonical.csv `
  --output-root D:\Codex\matchquant_runs
```

仓库内的小型合成端到端示例可以这样运行：

```powershell
python -m matchquant forecast `
  --history examples\synthetic_matches.csv `
  --fixtures examples\synthetic_fixtures.csv `
  --as-of 2024-01-07 `
  --model poisson `
  --model-config examples\forecast_poisson.json `
  --output-root D:\Codex\matchquant_runs

python -m matchquant settle `
  --forecast-run D:\Codex\matchquant_runs\m4_forecast_<...> `
  --results examples\synthetic_matches.csv `
  --output-root D:\Codex\matchquant_runs
```

每次结算会新建唯一的 `m6_settlement_<UTC时间>_<唯一ID>` 目录，保存 `settlement.csv`、`metrics.json`、`run_config.json` 和中文 `settlement_report.md`。身份匹配严格使用 `season + home_team + away_team` 的去首尾空白原文本，不猜测球队别名、不做模糊匹配；唯一命中允许改期并标记 `same_date`/`date_changed`，零命中保留为预期的 `unresolved`，重复命中标记 `ambiguous`。结果早于预测 as-of 或早于预测 fixture 日期的行标记 `time_conflict`，不进入指标。`metrics.json` 只对安全结算子集计算 multiclass log loss、三分类 Brier 和 accuracy；零场结算时三项预测指标为 `null`，coverage 按安全结算数/总预测数为 `0`。结果输入仍需由用户通过既有 M2/M2.1 或其他明确的本地 canonical 文件自行获得，M6 本身不导入赛果。

新生成的 M4 run 会在 `run_config.json` 中记录 `predictions.csv` 的 SHA-256；旧 M4 run 会明确记录 `unrecorded_legacy`，不会被 M6 偷改。M6 输出是离线研究审计，不是实时预测、投注建议、盈利承诺或未来表现保证。

## 三个模型

- `frequency`：带 Laplace 风格 `alpha` 平滑的历史 H/D/A 频率基线。
- `elo`：Elo 评分；通过显式 `draw_base`/`draw_scale` 产生平局概率，并用主胜期望分数 `P(H)+0.5P(D)` 更新 rating。
- `poisson`：基础独立 Poisson；在线收缩球队进球/失球率，输出 `expected_home_goals` 和 `expected_away_goals`。这两个字段是模型期望进球，不是观测 xG。`max_goals` 是比分枚举的最小上限，网格会继续扩大，直到两侧 Poisson 的遗漏尾部低于 `tail_tolerance`（默认 `1e-12`），避免高 lambda 时把明显的尾部质量直接归一化掉。

新球队使用模型先验/default rating。跨赛季默认 `carry`，也可用 `season_mode=reset` 在输入赛季发生变化时清空在线状态。概率经过合法性归一化，但本项目不宣称模型已经校准。

## 回测与指标

回测严格按时间推进：训练期比赛和测试期每个时间批次都先预测、后更新，测试期输出每场三类概率。指标包括总体和按输入 `season` 分组的：

- Log Loss：真实类别概率的负对数均值；
- 三分类 Brier Score：每场三个类别平方误差之和，再对比赛取平均；
- Accuracy：最大概率类别命中率。

测试边界必须显式提供。没有匹配到测试比赛会明确报错；没有训练比赛时允许冷启动测试，但结果应按冷启动解释。

## 运行

在项目根目录执行：

```powershell
python -m matchquant run `
  --input examples/synthetic_matches.csv `
  --test-start 2024-01-07 `
  --model all `
  --output-dir D:\Codex\matchquant_runs\synthetic
```

也可以使用配置示例：

```powershell
python -m matchquant run --config examples/config.json
```

输出目录包含：

- `predictions.csv`：每个测试比赛、每个模型的实际结果、三类概率和 Poisson 期望进球字段；
- `metrics.json`：总体/分赛季指标、测试边界和输入 SHA-256；
- `run_config.json`：实际运行配置、输入路径/哈希和输出路径。

## 测试

```powershell
python -m unittest discover -s tests -v
```

M2/M2.1/M3/M4/M5/M6 仍不包含网页、数据库、实时预测服务、因子引擎、外部赔率、球员/新闻数据或机器学习模型；历史回测、校准诊断、本地 Alpha 快照与本地结算不等于未来表现、盈利、投注价值或投注建议。
