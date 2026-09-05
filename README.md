# MatchQuant M1/M2 — Research Engine

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

M2 仅增加显式触发的 EPL 历史 CSV 下载、离线适配和质量审计；项目仍不包含网页、数据库、实时预测、因子引擎或机器学习模型，输入数据质量和概率校准也不等于博彩可用性。
