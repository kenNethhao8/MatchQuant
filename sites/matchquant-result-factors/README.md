# MatchQuant 赛果因子面板

这是一个私有 Sites 静态演示页，只读取 `dist/data/demo.json`。数据由 MatchQuant M13A 的公开快照和查询 API 从合成历史样例生成；不连接实时赛程、球队新闻或生产数据，也不计算或修改 1X2 概率。

## 本地生成演示数据

在 MatchQuant 仓库根目录运行：

    .venv\Scripts\python.exe scripts\build_sites_demo.py

脚本只读取 `examples/result_factor_history.csv`、其 manifest 和本目录的 score-free 演示赛程。它在系统临时目录构建快照/查询，只把不含本机绝对路径的 JSON 写到 `dist/data/demo.json`。

演示历史截止 2024-08-15，as-of 为日期批次 2024-08-16；这不是当前赛程或实时数据。页面保留 M13A 的 `ready`、`needs_review`、`frozen_snapshot_coverage_not_verified` 和原始 reasons。M13A 因子仍是实验信息，不代表校准、因果关系或预测收益。

## Sites

纯静态 HTML，无 npm 依赖。Sites 配置将 `dist/` 作为唯一发布目录；本地 Sites 工作 checkout 被此目录下的 `.gitignore` 排除，不属于 MatchQuant 仓库源文件。
