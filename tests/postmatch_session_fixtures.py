import shutil
from pathlib import Path

import pandas as pd

from matchquant.epl import stable_match_id
from matchquant.forecast_session import run_forecast_session
from matchquant.io import file_sha256, load_matches, write_json
from matchquant.schema import REQUIRED_COLUMNS


EXAMPLES = Path(__file__).resolve().parents[1] / "examples"


def _write_baseline(root: Path) -> dict[str, Path]:
    data_root = root / "data"
    processed = data_root / "processed"
    processed.mkdir(parents=True)
    history = processed / "epl_canonical.csv"
    shutil.copyfile(EXAMPLES / "synthetic_matches.csv", history)
    frame = load_matches(history)
    frame["match_id"] = [stable_match_id(row.season, row.home_team, row.away_team) for row in frame.itertuples()]
    frame.loc[:, list(REQUIRED_COLUMNS)].to_csv(history, index=False)
    frame = load_matches(history)
    history_manifest = processed / "epl_processed_manifest.json"
    write_json(
        {
            "schema_version": "m2.1-processed-manifest-v1",
            "adapter_version": "m2.1-epl-football-data-v1",
            "dataset_version": "m2.1-test-baseline",
            "data_cutoff": str(frame["date"].max()),
            "output": {
                "filename": str(history),
                "sha256": file_sha256(history),
                "rows": len(frame),
                "columns": list(REQUIRED_COLUMNS),
            },
        },
        history_manifest,
    )
    pointer = processed / "epl_current.json"
    write_json(
        {
            "schema_version": "m2.1-current-pointer-v1",
            "dataset_version": "m2.1-test-baseline",
            "canonical": str(history),
            "processed_manifest": str(history_manifest),
            "quality_report": None,
            "diff_report": None,
            "canonical_sha256": file_sha256(history),
        },
        pointer,
    )
    return {"data_root": data_root, "history": history, "history_manifest": history_manifest, "pointer": pointer}


def _write_e0_results(path: Path, history: pd.DataFrame, *, revised: bool = False, omit_first: bool = False) -> Path:
    rows: list[list[object]] = []
    selected = history.loc[history["season"].eq("2024-25")].reset_index(drop=True)
    for index, row in selected.iterrows():
        if omit_first and index == 0:
            continue
        home_goals = int(row["home_goals"]) + (1 if revised and index == 0 else 0)
        rows.append(
            [
                pd.Timestamp(row["date"]).strftime("%d/%m/%Y"),
                row["home_team"],
                row["away_team"],
                home_goals,
                int(row["away_goals"]),
            ]
        )
    rows.append(["11/01/2024", "FutureHome", "FutureAway", 2, 1])
    pd.DataFrame(rows, columns=["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG"]).to_csv(path, index=False)
    return path


def build_ready_session(root: Path) -> dict[str, Path]:
    baseline = _write_baseline(root)
    input_root = root / "input"
    input_root.mkdir()
    fixture_source = input_root / "forecast_fixture.csv"
    fixture_source.write_text(
        "Date,HomeTeam,AwayTeam,FTHG,FTAG\n11/01/2024,FutureHome,FutureAway,,\n",
        encoding="utf-8",
    )
    model_config = input_root / "frequency.json"
    model_config.write_text('{"alpha": 1.0}\n', encoding="utf-8")
    runs_root = root / "runs"
    m9 = run_forecast_session(
        baseline["history"],
        baseline["history_manifest"],
        fixture_source,
        "2024-25",
        "2024-01-10",
        model_name="frequency",
        model_config_path=model_config,
        fixture_data_root=baseline["data_root"],
        output_root=runs_root,
    )
    if m9["status"] != "ready":
        raise AssertionError(f"synthetic M9 setup was not ready: {m9['status']}")
    results_source = _write_e0_results(input_root / "results.csv", load_matches(baseline["history"]))
    return {
        **baseline,
        "runs_root": runs_root,
        "session_manifest": Path(m9["session_manifest_path"]),
        "results_source": results_source,
    }
