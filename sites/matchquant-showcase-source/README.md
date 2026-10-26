# MatchQuant V1 Showcase

A buildless, static research showcase for MatchQuant. It presents historical model evaluation, one final-matchday attribution sample, Factor Lab quintiles, season-by-season walk-forward results, and the Track M-Open paper-book paths. The page does not run predictions in the browser or offer live picks.

## Public project links

- GitHub repository: https://github.com/kenNethhao8/MatchQuant
- Installation and methodology: https://github.com/kenNethhao8/MatchQuant/blob/main/README.md

## Prepare the display data

The exporter reads the report artifacts and writes a compact, aggregated JSON asset. It does not copy the full prediction CSV or trade log into the public Site.

```powershell
python scripts/build_showcase_data.py `
  --research-run D:\matchquant\worktree\reports\research_run `
  --output dist\data\showcase-data.json
```

Inputs: `quant_evaluation_summary.json`, `factor_lab_metrics.json`, and `walk_forward_predictions.csv`. The generated asset contains the reported summary metrics, one historical matchday, selected Q1–Q5 profiles, two rejected-factor examples, and a sampled equity curve.

## Preview

Serve the Site root with any static HTTP server and open `dist/`. No frontend package installation or build step is needed.

```powershell
python -m http.server 4178 --directory dist
```

## Research boundary

The evidence snapshot ends with the 2025–26 season. Walk-forward metrics describe historical out-of-sample evaluation, and Kelly figures are paper simulations. They are not evidence of live profitability or future performance. The un-de-vigged opening-odds paper path is shown beside the fair-odds path.
