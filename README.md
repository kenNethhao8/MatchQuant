# MatchQuant M1 — Research Engine

MatchQuant M1 是一个低成本、可复现的足球 1X2（主胜/平/客胜）预测与回测最小框架。核心运行只依赖 Python、pandas 和 numpy，不依赖 LLM、API Key、付费服务、数据库或网页。

## 输入格式

CSV 必须包含以下列：

```text
match_id,date,season,home_team,away_team,home_goals,away_goals
```

校验会拒绝缺失字段/值、重复 `match_id`、同赛季同日期同主客队的重复 fixture、非法日期、负数或非整数比分，以及主客队相同。`season` 原样作为输入字段使用，不从月份推断。

`date` 可以是 `YYYY-MM-DD` 或带开球时间的 ISO 日期时间。只有日期时，同一日所有比赛先完成预测，再统一用实际比分更新；带时间时按时间批次推进，同一时间戳也先全预测后更新。排序由时间批次和 `match_id` 决定，因此 CSV 行顺序不会改变结果。

合成样例位于 [`examples/synthetic_matches.csv`](examples/synthetic_matches.csv)，明确仅用于演示端到端流程。

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

当前 M1 刻意不包含自动下载、网页、数据库、实时预测、因子引擎或复杂模型；输入数据质量和概率校准也仍需在后续阶段单独研究。
