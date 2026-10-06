# MatchQuant M1–M12 — Research Engine

MatchQuant 是一个低成本、可复现的足球 1X2（主胜/平/客胜）预测与回测最小框架。M1 的核心运行和 M2 的数据适配只依赖 Python、pandas 和 numpy，不依赖 LLM、API Key、付费服务、数据库或网页。

## 安装（baseline-v0.1，B00 验证）

不要依赖全局 PATH 上的 `python`；请在项目内建立独立虚拟环境。`constraints-baseline-v0.1.txt` 固定的依赖（numpy 2.3.5、pandas 3.0.1）要求 **Python >= 3.11**；本轮实际验证的解释器仅为 CPython 3.13.15，其它 3.11+ 版本未在此环境验证：

```powershell
# 使用本机已验证的解释器创建 .venv。本机 py 启动器选择器：py -V:Astral/CPython3.13.15；
# 注意 `py -3.13` 不识别 uv 注册的解释器。也可直接指定任意 Python >= 3.11 解释器路径。
py -V:Astral/CPython3.13.15 -m venv .venv   # 或：<python.exe> -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e . -c constraints-baseline-v0.1.txt
```

`constraints-baseline-v0.1.txt` 固定了 B00 实际验证的依赖版本（numpy 2.3.5、pandas 3.0.1）；editable 安装只联网获取依赖，之后所有测试与 `run`/`forecast`/`settle` 等命令均完全离线。后续命令请用 `.\.venv\Scripts\python.exe -m matchquant ...` 替代 README 示例中的 `python -m matchquant ...`。`.venv/` 不提交 Git。

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

## M7：本地赛果文件到 M2.1 staging 的桥接

`epl-local-stage` 是一个完全离线的原始文件封存入口。它只把用户明确指定的本地 E0 CSV 按 bytes 复制到 `D:\Codex\matchquant_data\staging\epl_local_<UTC时间>_<唯一ID>\raw\E0_YYYY-YY.csv`，计算 SHA-256，并写出兼容 `epl-sync --staging-dir` 的 `raw\source_manifest.json`。原输入不会被移动或修改，raw copy 会尽量设为只读；M7 不生成 canonical、不做 diff、不写 `processed\versions`、不更新 `processed\epl_current.json`，也不调用 M4/M5/M6。

M7 manifest 会明确标记 `status=unverified` 和 `validation_status=unverified`：M7 不重复 M2.1 的日期、比分、重复 fixture、completed/unfinished 或 380 场检查。`source_path` 是用户提供的绝对本地路径，`local_archived_at` 只表示本地封存时间，不能当作上游抓取时间；只有用户明确传入 `--upstream-retrieved-at`（别名 `--source-retrieved-at`）时，才会记录 `upstream_retrieved_at`。`--source-url` 和 `--source-label` 仅写 provenance，绝不会触发网络，也不会从文件名猜 URL。没有 URL 时 manifest、质量报告和差异报告的 `source_urls` 为空，不会伪造 URL。

手动闭环如下：

1. 用户在本地下载或导出一个 E0 CSV，保留原始文件。
2. 创建可审计的 M7 local staging snapshot：

   ```powershell
   python -m matchquant epl-local-stage `
     --source-csv D:\Codex\my_input\E0.csv `
     --season 2025-26 `
     --source-label "user-provided local E0 export" `
     --data-dir D:\Codex\matchquant_data
   ```

   若确实知道上游获取时间，可额外传 `--upstream-retrieved-at 2026-09-05T08:30:00Z`；该值必须是带时区的 ISO-8601 时间，系统会统一保存为 UTC `Z` 格式，不能把文件复制时间填到这个字段。M7 成功只代表 raw bytes 已封存，尚未代表 CSV 合法。
3. 把命令输出的 snapshot 路径交给现有 M2.1，完全离线生成质量和差异报告：

   ```powershell
   python -m matchquant epl-sync `
     --staging-dir D:\Codex\matchquant_data\staging\epl_local_<...> `
     --data-dir D:\Codex\matchquant_data `
     --dry-run
   ```

   正常完整赛季仍须 380 场。对于确实尚未结束的赛季，必须在这一步明确声明 `--in-progress-season 2025-26`；两项比分都为空的合法行会由 epl-sync 报告为 `skipped_unfinished`，M7 不会静默处理，部分比分和其它非法行仍会失败。
4. 只有 dry-run 证明 addition-only、无 `revised`、无 `missing_from_candidate` 且用户再次明确请求时，才执行同一个 snapshot 的 `epl-sync --apply`。由现有 M2.1 创建 version 和原子更新 pointer；M7 本身永远不 apply。

   ```powershell
   python -m matchquant epl-sync `
     --staging-dir D:\Codex\matchquant_data\staging\epl_local_<...> `
     --data-dir D:\Codex\matchquant_data `
     --in-progress-season 2025-26 `
     --apply
   ```

5. M5 `epl-fixtures-prepare` 与 M4/M6 继续使用各自明确的本地输入；M7 不把 completed rows 自动写入历史，也没有实时性声明。

缺失文件或封存 I/O 错误会留下 `status=failed` 的 manifest 后明确失败。格式错误、缺列或坏行可以先作为 `unverified` raw snapshot 留证，但必须由后续 `epl-sync --staging-dir` 生成完整失败质量报告；在质量通过前不得把它当成成功的历史数据。

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

M6 用一个已经完成的预测 run 和用户明确指定的本地 M1 canonical 最终赛果 CSV 做一次离线结算。`settle` 通过统一预测证据适配层（`matchquant.prediction_evidence.load_prediction_evidence`）按 run_config 的真实 `schema_version` 分派两种 producer：

- `m4-forecast-v1`：M4 forecast run。沿用既有 M4 严格校验与 legacy hash 语义；只读取 run 目录中的 `run_config.json`、`predictions.csv` 以及 `--results` 指定的结果文件。
- `m11-engine-snapshot-predict-v1`：单模型 M11 引擎快照预测 run。额外严格核验 run 与其记录的不可变快照的跨文件一致性：快照 manifest/状态文件 hash、manifest content hash、snapshot_id、逐模型 config/config_sha256/state_sha256（区分整份 state 文件 hash 与单模型 state payload hash）、run 与快照的 as_of（value/timestamp/date_only）一致、`used_data_cutoff` 与 as_of 边界的时序合法（date-only 边界必须严格早于 as_of；完整 history 文件含 as_of 后比赛不误拒），并读取显式记录的原 score-free fixtures 核对实际 hash、行数与逐 fixture 对应关系。多模型 M11 run 与未知 schema 明确拒绝；M11 结算不打开 history CSV 或 history processed manifest，断开历史文件后校验与结算仍可完成。

两种 producer 复用同一个 `settle_predictions` 结算算法：不下载赛果、不调用 `epl-sync`、不写入 `D:\Codex\matchquant_data\`，不更新 active/current pointer，不重跑模型，也不修改任何输入；结算输出 root 不得落入预测 run 目录、M11 快照目录或正式 data root。

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

每次结算会新建唯一的 `m6_settlement_<UTC时间>_<唯一ID>` 目录，保存 `settlement.csv`、`metrics.json`、`run_config.json` 和中文 `settlement_report.md`。身份匹配严格使用 `season + home_team + away_team` 的去首尾空白原文本，不猜测球队别名、不做模糊匹配；唯一命中允许改期并标记 `same_date`/`date_changed`，零命中保留为预期的 `unresolved`，重复命中标记 `ambiguous`。结果早于预测 as-of 的行标记 `time_conflict`，不进入指标；唯一身份的改期结果仍可结算并标记 `date_changed`。`metrics.json` 只对安全结算子集计算 multiclass log loss、三分类 Brier 和 accuracy；零场结算时三项预测指标为 `null`，coverage 按安全结算数/总预测数为 `0`。结果输入仍需由用户通过既有 M2/M2.1 或其他明确的本地 canonical 文件自行获得，M6 本身不导入赛果。

新生成的 M4 run 会在 `run_config.json` 中记录 `predictions.csv` 的 SHA-256；旧 M4 run 会明确记录 `unrecorded_legacy`，不会被 M6 偷改。M11 run 必须记录该 hash，M6 逐字节核验。结算 `run_config.json` 的 `forecast_run` 块记录真实 producer `schema_version` 与 `producer_command`，并新增 `model_config`、`model_config_sha256` 与 `origin_evidence` 字段（M4 记录 producer 声明的 history/selection 证据及其核验状态；M11 记录快照 manifest/state 路径与 hash、快照 lineage 的 history 证据状态和 `used_data_cutoff`）；中文结算报告按真实 producer 标注来源。M6 输出是离线研究审计，不是实时预测、投注建议、盈利承诺或未来表现保证。

## M8：本地预测发布审计包

M8 是 M4 本地预测快照的发布前可追溯性核验。它只读取用户显式指定的 M4 run、M2/M2.1 canonical history、processed manifest 和 M5 `fixture_snapshot_manifest.json`，不会下载、联网、调用 `epl-sync`/`epl-local-stage`/`epl-fixtures-prepare`，不会重跑 M4、重新选择模型、训练或校准。它也不会修改任何输入、`D:\Codex\matchquant_data\` 或 current pointer。

```powershell
python -m matchquant release-verify `
  --forecast-run D:\Codex\matchquant_runs\m4_forecast_<...> `
  --history D:\Codex\matchquant_data\processed\epl_canonical.csv `
  --history-manifest D:\Codex\matchquant_data\processed\epl_processed_manifest.json `
  --fixture-manifest D:\Codex\matchquant_data\fixtures\snapshots\epl_fixtures_<...>\fixture_snapshot_manifest.json `
  --output-root D:\Codex\matchquant_runs
```

所有输入路径都必须显式给出；M8 不猜测“最新”文件，也不从默认目录隐式替换输入。每次成功调用只在 `--output-root`（默认 `D:\Codex\matchquant_runs`）下创建唯一的 `m8_release_<UTC>_<ID>` 目录，其中包含 `release_manifest.json` 和中文 `release_report.md`。M8 只链接原始 M4 `predictions.csv`，不复制预测文件、raw source 或第三方数据。`--output-root` 不能是 M4 run、M5 snapshot、输入目录或 `D:\Codex\matchquant_data` 下的路径。

审计会严格核对：M6 对 M4 `run_config.json`、`predictions.csv`、schema 和 prediction SHA-256 的验证；M4 history/fixtures 的路径、hash、行数和 as-of；history manifest 的 output hash、行数、`data_cutoff`、`dataset_version`；M5 snapshot id、pending output、raw source/hash、quality report/hash、completed cutoff 和 pending 日期范围。历史 manifest 缺少 `dataset_version` 或 `data_cutoff` 时不会伪造值，最多为 `needs_review`；旧 M4 缺少 prediction hash 会保留 `unrecorded_legacy`，不能变成 `ready`；M4 已记录的 `history_hash_mismatch` 会原样保留为审计风险。若 M5 `completed_data_cutoff` 与 M4 date-only as-of 同日或更晚，或精确 as-of 落在同一无时刻日期/之后，M8 只标记 `needs_review`，不把它误称为模型泄漏；selection evidence 缺 hash 为 `needs_review`，非 64 位或与文件实际 hash 不一致则为 `blocked`。

状态含义如下：

- `ready`：所有硬完整性和时间核验通过，并且没有 legacy prediction hash、selection mismatch 或缺失版本/cutoff 等需人工注意的证据；仍需用户做最终发布决定。
- `needs_review`：没有已识别的硬矛盾，但存在需要人工注意的旧 hash、selection history mismatch 或不完整版本证据，不得直接发布。
- `blocked`：关键 path、hash、rows、schema、raw/quality 自洽性或时间边界存在矛盾，或关键证据无法读取，不得发布。

建议的手动顺序是：`M7` 或 `M2.1` 的可选历史更新（如确有需要）→ `M5 epl-fixtures-prepare` → `M4 forecast` → `M8 release-verify` → 后续 `M6 settle`。M8 不证明历史数据或赛程已“最新”，M5 的复制时间、文件日期范围和 source URL 也不证明实时性；它不代表实时预测、概率已校准、盈利能力或投注建议，不建立 source 文件的再分发授权。

## M9：本地预测会话闭环

M9 是一个完全离线、显式输入的赛前封装命令，严格按 `M5 epl-fixtures-prepare → M4 forecast → M8 release-verify` 执行。它不新增模型、不自动选择 history/manifest/fixture/模型、不联网、不调用 `epl-download`、`epl-sync`、`epl-local-stage` 或 `settle`，也不会更新 `epl_current.json`、active/current pointer 或历史结果。

```powershell
python -m matchquant forecast-session `
  --history D:\Codex\matchquant_data\processed\epl_canonical.csv `
  --history-manifest D:\Codex\matchquant_data\processed\epl_processed_manifest.json `
  --fixture-source D:\Codex\my_input\E0.csv `
  --season 2025-26 `
  --as-of 2026-09-06 `
  --model poisson `
  --model-config D:\Codex\my_input\poisson.json `
  --fixture-data-root D:\Codex\matchquant_data `
  --output-root D:\Codex\matchquant_runs
```

也可以用 M3 的选择证据替代显式模型配置，两种方式互斥：

```powershell
python -m matchquant forecast-session `
  --history D:\Codex\matchquant_data\processed\epl_canonical.csv `
  --history-manifest D:\Codex\matchquant_data\processed\epl_processed_manifest.json `
  --fixture-source D:\Codex\my_input\E0.csv `
  --season 2025-26 `
  --as-of 2026-09-06 `
  --selection-run-config D:\Codex\matchquant_runs\m3_select_<...>\run_config.json
```

每次调用会在 `--output-root` 下创建唯一的 `m9_forecast_session_<UTC>_<ID>` 目录，并在其中只写 `session_manifest.json` 和中文 `session_report.md`。M5 snapshot 留在显式 `--fixture-data-root` 下；M4 forecast 和 M8 release audit 留在显式 `--output-root` 下。M9 manifest/report 只链接子产物并保存路径、SHA-256、状态、失败原因和不变更声明，不复制 predictions、raw source 或第三方数据。即使 M5/M4 在中途失败，也会先创建 M9 目录并保留会话证据；后续未运行阶段标为 `not_run`。

`ready`、`needs_review`、`blocked` 原样来自 M8（M5/M4 无法完成时 M9 为 `blocked`）：

- `ready`：M8 硬完整性与时间核验通过；不代表实时性、校准、盈利、投注价值或人工发布完成。
- `needs_review`：M8 保留可人工复核的时间/版本/legacy 证据，不应直接发布。
- `blocked`：M8 或前置阶段存在无法安全继续的硬问题。

`--output-root` 不得位于 `D:\Codex\matchquant_data` 或显式 fixture data root 内/与其重叠，不得等于 history、history manifest 或 fixture source 所在目录，也不得落入已有的 `m9_forecast_session_*` 子目录。history、manifest、fixture source、M5 raw/quality、active/current pointer 均保持不变；M9 不导入赛后结果。赛后若要更新历史，用户必须明确执行 M7/M2.1 并审查结果，再明确执行 M6 settle；M9 不自动触发这条流程。

## M10：本地赛后会话闭环

M10 把一个显式的 M9 赛前会话、用户提供的本地 E0 最终赛果文件和既有 M2.1/M6 串成一个完全离线、可审计的赛后闭环。M10 不搜索“最新”会话、history、model 或 staging，也不联网、不下载、不重跑 M4；它只接受 M9 `session_manifest.json`、显式赛季和以下两种输入模式之一。

第一条命令只封存和预检：M7 把 E0 bytes 复制到新的 local staging snapshot，M2.1 随后对同一 snapshot 做 dry-run。它不会创建 history version、不会更新 `epl_current.json`，也不会执行 M6：

```powershell
python -m matchquant postmatch-session `
  --session-manifest D:\Codex\matchquant_runs\m9_forecast_session_<...>\session_manifest.json `
  --results-source D:\Codex\my_results\E0_results.csv `
  --season 2025-26 `
  --in-progress-season 2025-26 `
  --data-root D:\Codex\matchquant_data `
  --output-root D:\Codex\matchquant_runs
```

审查 M10 manifest、中文报告以及 M2.1 quality/diff/processed manifest 后，第二条命令必须使用上一步返回的同一个、明确指定的 staging 目录。只有这条单独的 `--apply` 调用才会先核对 active pointer 的 canonical 和 processed manifest 两个 SHA-256 是否仍与 M9 baseline 一致，再让 M2.1 创建 addition-only version 并原子更新 pointer；apply 成功后才以新 canonical 执行 M6：

```powershell
python -m matchquant postmatch-session `
  --session-manifest D:\Codex\matchquant_runs\m9_forecast_session_<...>\session_manifest.json `
  --staging-dir D:\Codex\matchquant_data\staging\epl_local_<...> `
  --season 2025-26 `
  --in-progress-season 2025-26 `
  --apply `
  --data-root D:\Codex\matchquant_data `
  --output-root D:\Codex\matchquant_runs
```

`--results-source` 与 `--staging-dir` 互斥且必须恰好一个；`--apply` 与 `--results-source` 互斥，不能把 raw E0 文件在同一调用中直接 apply。`--dry-run` 只是默认预检模式的显式写法。`--in-progress-season` 原样交给 M2.1，绝不从日期推断；source URL、label 和 upstream retrieval time 仅是 M7 provenance，不会触发网络。

每次调用在 `--output-root` 下创建唯一的 `m10_postmatch_session_<UTC>_<ID>`，其中只保存 `postmatch_manifest.json` 和中文 `postmatch_report.md`；M7、M2.1、M6 子产物留在各自目录并由 manifest 链接。M10 状态含义为：

- `needs_review`：dry-run 成功等待明确 apply，或携带 M9/M6 的 `needs_review`；不代表可以跳过人工审查。
- `complete`：M2.1 已 apply，M6 对所有预测行完成无歧义、无时间冲突的结算，且 M9 没有 review 状态。
- `partial`：M2.1 已 apply，M6 只留下 unresolved、没有 ambiguous/time conflict，且 M9 没有 review 状态。
- `blocked`：M9/hash、M7、M2.1、pointer coherence 或 M6 存在硬失败；后续不安全阶段不会运行。

M2.1 apply 是前向不可回滚的历史变更。如果 pointer 已更新而 M6 随后失败，M10 保留新版本、pointer、失败证据和会话报告，绝不自动回滚；用户应单独解决结算。M10 输出的是本次本地数据上的审计与评估完成情况，不证明历史或赛果当前、不证明概率已校准、不证明盈利或投注价值，也不是实时预测或投注建议。

## M11：引擎状态快照

M11 把显式 history 在一个明确 `as-of` 边界前的 M4 在线更新状态保存为不可变 JSON 快照。构建阶段必须同时给出 canonical history 和匹配的 processed manifest；程序核对路径、SHA-256、行数、列和 manifest `data_cutoff`，按 M4 的时间批次先预测后更新语义生成状态。快照目录只包含 `snapshot_manifest.json` 和 `model_states.json`，不会复制 history、raw 数据或可执行对象。

显式模型配置的构建示例（`--model` 与 `--model-config` 按出现顺序一一对应）：

```powershell
python -m matchquant engine-snapshot-build `
  --history D:\Codex\matchquant_data\processed\epl_canonical.csv `
  --history-manifest D:\Codex\matchquant_data\processed\epl_processed_manifest.json `
  --as-of 2025-08-01 `
  --model poisson `
  --model-config examples\forecast_poisson.json `
  --snapshot-root D:\Codex\matchquant_data\engine_snapshots
```

也可以传入一个已有的 M3 `run_config.json` 作为唯一选择证据：

```powershell
python -m matchquant engine-snapshot-build `
  --history D:\Codex\matchquant_data\processed\epl_canonical.csv `
  --history-manifest D:\Codex\matchquant_data\processed\epl_processed_manifest.json `
  --as-of 2025-08-01 `
  --selection-run-config D:\Codex\matchquant_runs\m3_select_<...>\run_config.json
```

构建输出的 manifest 记录 as-of 语义、历史和历史 manifest 的路径/hash、总行数、纳入/排除行数、实际 cutoff、模型配置/hash、状态文件/hash、不可变/不联网声明。预测阶段只需显式指定该 manifest 和 score-free 的 M4 fixture CSV：

```powershell
python -m matchquant engine-snapshot-predict `
  --snapshot-manifest D:\Codex\matchquant_data\engine_snapshots\m11_engine_snapshot_<...>\snapshot_manifest.json `
  --fixtures D:\Codex\my_fixtures.csv `
  --output-root D:\Codex\matchquant_runs
```

预测会严格校验快照 manifest、状态 hash、模型配置/状态一致性以及 fixture 不早于 as-of；它不接受 history 参数，不读取或更新历史，不更新模型状态，也不联网。预测运行保存 `predictions.csv`、`run_config.json` 和 `snapshot_report.md`，并记录快照 manifest/hash、fixture hash、cutoff 和输出 hash。快照预测应与同一 history/config/as-of 的 M4 `forecast_fixtures` 结果一致；未来新增或修改的 cutoff 后比赛不会改变已构建状态。M11 是轻量化基础层，不包含 Alpha 因子、近期信息、新闻、赔率、数据库、网页、调度或机器学习，也不证明数据当前、概率已校准、盈利能力或投注价值。

## M12：引擎快照新鲜度与历史版本关联

M12 只做一个完全离线、只读的审计：把用户明确给出的 M11 `snapshot_manifest.json` 与明确给出的 M2.1 `epl_current.json` 的 canonical/processed-manifest 路径、SHA-256 和 dataset version 进行比较。它不重写 M2.1/M7/M10，不联网，不更新 current pointer，不修改历史或已有 M11 快照，也不自动寻找 latest、apply 或重建快照。

```powershell
python -m matchquant engine-snapshot-status `
  --snapshot-manifest D:\Codex\matchquant_data\engine_snapshots\m11_engine_snapshot_<...>\snapshot_manifest.json `
  --current-pointer D:\Codex\matchquant_data\processed\epl_current.json `
  --output-root D:\Codex\matchquant_runs
```

每次调用都会在显式 output root 下创建唯一的 `m12_engine_snapshot_status_<UTC>_<ID>`，保存 `snapshot_status.json` 和简洁的 `snapshot_status.md`。`fresh` 只在两类证据完整且路径、hash、dataset version 精确匹配时返回；有效但关联到不同历史版本的快照返回 `stale`；缺失、篡改或结构不完整的 snapshot/pointer 返回 `unverifiable` 并保留失败证据。`stale` 不会使旧快照失效：旧快照仍可由 `engine-snapshot-predict` 独立复现。若要重建，用户必须主动再次执行显式的 M11 `engine-snapshot-build`。

## 快照会话与只读审计（B03）

B03 在 M11 之上新增两个公开 API（`matchquant.run_snapshot_session`、`matchquant.verify_snapshot_run`）和两条只读 CLI，把"单模型 M11 快照 → 预测 → 只读审计"串成一个离线会话。两者都只接受单模型 M11 输入，复用既有严格 snapshot loader、M11 predict 与 B02 预测证据适配层；不读取 history CSV、历史 processed manifest 或 current pointer，不联网、不结算、不运行 M12、不 apply、不重建快照。

```powershell
python -m matchquant snapshot-session `
  --snapshot-manifest <M11 snapshot_manifest.json> `
  --fixtures <score-free fixture CSV> `
  --mode replay `
  --output-root <会话输出根（必填）> `
  [--issue-evidence <M12 snapshot_status.json>]

python -m matchquant snapshot-run-verify `
  --forecast-run <已有单模型 M11 prediction run 目录> `
  --output-root <审计输出根（必填）> `
  [--issue-evidence <M12 snapshot_status.json>]
```

`snapshot-session` 在 `--output-root` 下创建唯一 `baseline_snapshot_session_<UTC>_<ID>` 目录，内含 `session_manifest.json`、中文 `session_report.md`、`prediction_audit.json`、`prediction_audit.md` 以及会话自己生成的原样 M11 prediction run 子目录。`--mode replay` 是可复现的历史回溯研究（报告明确 retrospective）；`--mode research` 表示本地待赛研究，即使工程通过且附带 fresh 发行证据，也因目标时点赛果覆盖未证明而保持 `needs_review`（理由 `coverage_to_fixture_time_not_proven`），不得当作当前可用预测产品。`snapshot-run-verify` 在 `--output-root` 下创建唯一 `baseline_snapshot_audit_<UTC>_<ID>`，输出新的审计 JSON/中文报告。

审计实际完成的工程检查包括：经 B02 严格证据适配层核验的快照 manifest/state hash、单模型集合一致、run/predictions/fixtures hash、as_of 三字段绑定与 used cutoff 时序门槛、fixture 与预测逐行身份对应、概率有限/范围/求和；随后用原快照与原 score-free fixtures 在隔离临时目录重新调用既有 M11 predict，与原预测逐行比较概率与身份（`atol=1e-12, rtol=0`）——只改预测值并同步更新记录 hash 仍会被该计算一致性检查拒绝；最后复核全部输入 hash 前后不变。原 run 不覆盖，重算文件不是原 run 的新版本。

可选 `--issue-evidence` 只接受现有 M12 输出 `snapshot_status.json`（schema `m12-engine-snapshot-status-v1`）。审计计算其文件 SHA-256，校验结构、字段实际类型、UTC `generated_at`（不得晚于本次审计时间）、目标快照绑定与记录内部一致性：record 子块除 exact-keys 外逐值校验（path 必须非空字符串、hash 必须为 64 位十六进制、pointer schema 固定为 `m2.1-current-pointer-v1`、snapshot schema 固定为 `m11-engine-snapshot-v1`，键存在不等于值有效）；比较标志不作为证据使用，而是从记录自身的 snapshot 侧与 pointer 侧实际 path/hash/dataset_version 重算相等关系，记录的标志必须与重算结果一致；pointer 记录自身的内部关系（recorded_sha256、output_sha256、dataset_version）同样重算；记录中的 history cutoff 与 history-manifest 来源字段必须与严格加载的真实 manifest 绑定。状态按既有 M12 状态机区分：stale 必须 snapshot/pointer 双方 valid 且比较确实存在差异，pointer 无效只能对应 unverifiable（其部分字段缺失按实际形状接受，不强制补齐），把 pointer 无效的记录手写为 stale 会被拒绝（blocked）。通过全部重算与绑定验证的 fresh 记录显示 `fresh_at_issue=true`，它只表示发行时点的历史关联验证，不证明当前 active、今天最新或训练到目标比赛；`active_link_status` 固定为 `not_checked_in_runtime`。stale/unverifiable 记录只能作为历史限制展示，不能升格为 fresh；证据中记录的研究端路径只做规范化比较，不会被打开，断开这些文件后运行端仍可完成审计。此版本没有签名或外部可信 hash，实际文件 hash 只证明本次输入一致性，不构成发行来源认证，也不能发现所有字段同步伪造的历史。

状态与退出码（正常解析后与 JSON 状态一致）：`ready` 退出 0（replay 工程通过，或 standalone 审计工程通过且已关联的历史发行记录 fresh）；`needs_review` 退出 2（standalone 审计工程通过但发行证据缺失/stale/unverifiable，或 research 工程通过但覆盖未证明）；`blocked` 退出 1（任一工程校验失败、输入/IO 错误或已提供证据损坏；此时保留结构化失败产物，`predictions_usable=false`）。replay 会话可在自身 `ready` 的同时保留 `needs_review` 的子审计状态与理由，两者同时显示，子审计不会被升格。输出根不得位于输入文件/目录内、既有 session/audit 或 M11 证据目录内，也不得写正式 `D:\Codex\matchquant_data` / `D:\Codex\matchquant_runs`；输入类型（run 为目录、snapshot/fixtures/issue 为文件）由调用方声明，不依赖文件是否存在，因此缺失的 run 或 fixture 也不能被用来把输出放进其声明的目录内。此类拒绝在创建任何文件前发生，stderr 说明原因且不会有失败报告。安全可写输出下，缺 snapshot/fixtures/run_config/显式 issue 证据、输入读取或清单 hash 失败等预期输入/IO 错误会先创建唯一目录再落入结构化 blocked 流程：JSON 与中文报告保留 failed/not_run 阶段、真实输入路径、unavailable hash 与不可用概率说明，stderr 不会把这类输入错误说成"输出被拒绝"。必需产物（中文报告或已记录产物的 hash 读取）写入失败时，会话整体为 `blocked`（退出 1、`predictions_usable=false`）：已完成的工程阶段、合法 M11 run 与子审计结果按实际保留并与会话交付失败明确区分，`output_write_errors` 记录未能保存的部分；manifest 本身无法写时抛出预期 IO 异常，stderr 诚实列出已有目录与产物，不伪造完整交付。注意区分：缺参数等 argparse 使用错误仍是既有 usage/退出 2，它不代表产生了 needs_review 审计。结算继续使用既有 `settle` 命令，传入会话引用的实际 M11 run 目录与独立赛果文件，不要把会话目录冒充 prediction run。

## 可迁移运行包（B04）

B04 在 B03 之上新增"可迁移运行包"：把一个单模型快照与其发行时 fresh 的 M12 记录封装成一个自包含目录。包迁移到不同绝对目录、原研究端文件（快照、state、history、processed manifest、pointer、原 issue JSON）全部断开后，验证、预测会话、运行审计与结算依然离线完成，概率与原快照/M4 参考 `atol=1e-12, rtol=0` 一致。运行端只从包内相对路径解析文件，不存在回退到来源绝对路径的通道。

```powershell
# 研究端构建（输入只读；fresh 发行记录是唯一入口）
python -m matchquant runtime-bundle-build `
  --snapshot-manifest <单模型 M11 snapshot_manifest.json> `
  --issue-audit <绑定该快照且 fresh 的 M12 snapshot_status.json> `
  --output-root <作业输出根（必填）>

# 只读验证（可在任意目录、迁移后执行）
python -m matchquant runtime-bundle-verify --bundle-manifest <包根>\bundle_manifest.json

# 包内预测会话（与裸 --snapshot-manifest 互斥；包内发行记录不可被 --issue-evidence 覆盖）
python -m matchquant snapshot-session `
  --bundle-manifest <包根>\bundle_manifest.json `
  --fixtures <score-free fixture CSV> `
  --mode replay `
  --output-root <会话输出根（必填）>

# 对包内派生快照产生的 M11 run 做独立审计
python -m matchquant snapshot-run-verify `
  --forecast-run <M11 prediction run 目录> `
  --bundle-manifest <包根>\bundle_manifest.json `
  --output-root <审计输出根（必填）>
```

包内布局（schema `baseline-runtime-bundle-v1`）：**首版固定 7 个载荷 + `bundle_manifest.json`，共 8 个实际文件，无扩展槽**——`snapshot/<原 snapshot_id>/` 下派生 manifest、原字节 `model_states.json`、派生说明 `snapshot_report.md`；`evidence/original_snapshot_manifest.json` 与 `evidence/issue_audit.json` 为原 manifest 与原 M12 记录的字节副本（来源路径仅为 provenance，从不执行）；`team_index.json`（`baseline-team-index-v1`，从严格加载的 state 确定性导出）；`RUN.md`。`bundle_manifest.json` 记录所有载荷的相对路径/大小/SHA-256（不记录自身最终 hash，包体无签名），`file_count=7` 为载荷数，`actual_file_count=8` 为含清单的实际文件数。构建作业的 manifest/报告保存在包外作业目录；`bundle_manifest.json` 最后写入，未完成包不标记可用。解压后全部实际文件之和 ≤5 MiB；超限输出 blocked 候选与逐文件大小说明，不静默删字段。构建交付状态由最终持久化统一决定：必需报告/作业记录/产物 hash 持久化失败时留下 blocked、`finalize=failed`、`required_artifact_write_failed` 与真实已有产物（合法候选包仍在磁盘上，独立 verify 可描述其工程完整性，但不能把失败的构建交付升格为 ready）；作业记录本身写失败时抛出诚实的预期 IO 异常并说明已有目录，CLI 退出 1。

`runtime-bundle-build` 只接受已严格绑定原快照且 `fresh` 的发行记录（stale/unverifiable/损坏/指向其他快照一律 blocked）；`runtime-bundle-verify` 只读重算全部证据，并按固定顺序设置门槛：先做**无内容读取的路径预检**——清单路径文本、Windows 规范化冲突、固定 v1 载荷集合（拒绝列入清单的额外文件与删去必需项，不按后缀黑名单）、每个载荷沿包根的父目录链、叶子对象与 resolve 后真实目标的包含关系——预检有任何错误即返回 blocked，不打开任何载荷内容；随后才逐文件以实际 stat 字节数与记录值比较并重算 SHA-256，最后枚举未列文件与 reparse 目录。原 manifest 副本检查 file/content hash 与完整 JSON 形状（其记录的来源路径从不打开）；派生快照经既有严格 loader（不放宽旧 M11 校验）；原副本与派生 manifest 的实际字段 diff 仅限合同固定叶子（`outputs.snapshot_report=snapshot_report.md` 是经裁定的唯一固定新增叶子，其它新增/删除仍拒绝）；单模型 config/payload hash 一致；原 issue 副本经 B03 `load_issue_evidence` 以原 manifest hash 重新核验（仍要求 fresh）；team index 从 state 重算比对。verify 不依赖包外来源，不读取原研究端路径，不自动解压 ZIP，也不发布。

`snapshot-session --bundle-manifest` 先对整个包重新 verify，再从包内派生快照预测；嵌入审计通过派生桥（`bundle_original_issue_via_derived_manifest_lineage`）展示原/派生两套 manifest hash 的绑定关系——原 issue 始终绑定原 manifest hash，不能改写为派生 hash，也不能把原 issue 当作派生 manifest 的裸 issue 参数（B03 裸语义保持严格拒绝）。standalone `snapshot-run-verify --bundle-manifest` 遇到缺失/损坏 JSON/hash 不符/路径或集合被拒绝的坏包时，不再中断为裸异常：在安全可写输出根正常返回结构化 blocked（`predictions_usable=false`、`bundle_verify=failed`、其余阶段 not_run），落盘 `prediction_audit.json/md`，bundle 区块保留真实包路径、verify 的 checks/reasons 与已知信息，未知 lineage 字段为 null/unavailable，不伪造 fresh 或成功派生桥；CLI 退出 1 并打印真实失败原因。会话输出不得位于整个 bundle 目录内；每次运行所有包内文件起终 hash 不变（共享完整性检查同样先做路径预检再读内容），变化即 blocked。Elo/Poisson 对未知球队使用既有先验并在会话报告中标注 `unseen_home_teams`/`unseen_away_teams`/`unseen_team_prior`（仅注释，不修改预测列、run schema、模型配置或概率）；frequency 没有球队学习状态，声明全局频率，不把空球队清单当作花名册。已知球队不证明其仍参加当前赛季。`fresh_at_issue` 只表示原发行记录经派生桥关联，`active_link_status` 固定 `not_checked_in_runtime`；research 会话保持 `needs_review`。已经生成的 M11 run 继续记录其当时实际绝对路径；迁移后在迁移目录生成新 run，不就地重写旧 run。整个包没有签名来源认证：完全一致的伪造包体无法被 verify 发现，hash 只证明本次输入一致性。



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

M2/M2.1/M3/M4/M5/M6/M8/M9/M10/M11/M12 仍不包含网页、数据库、实时预测服务、因子引擎、外部赔率、球员/新闻数据或机器学习模型；历史回测、校准诊断、本地 Alpha 快照、引擎状态快照、快照新鲜度审计、本地预测会话、本地赛后会话与本地结算不等于未来表现、盈利、投注价值或投注建议。
