"""Generate the private Sites demo from the public M13A API and synthetic inputs."""

from __future__ import annotations

import hashlib
import json
import tempfile
from pathlib import Path
from typing import Any

from matchquant.result_factor_snapshot import (
    build_result_factor_snapshot,
    query_result_factors,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
HISTORY = REPO_ROOT / "examples" / "result_factor_history.csv"
HISTORY_MANIFEST = REPO_ROOT / "examples" / "result_factor_history_manifest.json"
FIXTURES = REPO_ROOT / "sites" / "matchquant-result-factors" / "data" / "demo_fixtures.csv"
OUTPUT = REPO_ROOT / "sites" / "matchquant-result-factors" / "dist" / "data" / "demo.json"
AS_OF = "2024-08-16"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _build_demo() -> dict[str, Any]:
    history_manifest = json.loads(HISTORY_MANIFEST.read_text(encoding="utf-8"))
    if history_manifest.get("dataset_version") != "m13a-synthetic-result-factors-v1":
        raise ValueError("The Site demo accepts only the frozen synthetic M13A history.")

    with tempfile.TemporaryDirectory(
        prefix="matchquant-site-demo-", dir=REPO_ROOT.parent
    ) as temporary:
        work_root = Path(temporary)
        built = build_result_factor_snapshot(
            HISTORY,
            HISTORY_MANIFEST,
            AS_OF,
            output_root=work_root / "snapshot",
        )
        if built.get("status") != "ready" or not built.get("manifest_path"):
            raise RuntimeError(f"Synthetic snapshot build did not complete: {built.get('reasons')}")

        queried = query_result_factors(
            built["manifest_path"],
            FIXTURES,
            output_root=work_root / "query",
        )
        payload = queried.get("payload")
        if not isinstance(payload, dict) or queried.get("status") == "blocked":
            raise RuntimeError(f"Synthetic factor query failed: {queried.get('reasons')}")
        if payload.get("experimental") is not True or payload.get("probability_adjustment_applied") is not False:
            raise ValueError("M13A demo must remain experimental and must not adjust probabilities.")

        snapshot = payload["snapshot"]
        fixture_evidence = payload["fixtures"]
        return {
            "schema_version": "matchquant-sites-result-factor-demo-v1",
            "label": "合成演示数据",
            "generated_at": payload["generated_at"],
            "status": payload["status"],
            "as_of": payload["as_of"],
            "used_data_cutoff": built["used_data_cutoff"],
            "rows_total": built["rows_total"],
            "rows_used": built["rows_used"],
            "coverage_status": queried["coverage_status"],
            "experimental": queried["experimental"],
            "probability_adjustment_applied": queried["probability_adjustment_applied"],
            "fixture_count": queried["fixture_count"],
            "snapshot": {
                "snapshot_id": snapshot["snapshot_id"],
                "manifest_sha256": snapshot["manifest_sha256"],
                "state_sha256": snapshot["state_sha256"],
                "history_sha256": snapshot["history_sha256"],
                "history_manifest_sha256": snapshot["history_manifest_sha256"],
            },
            "source": {
                "dataset_version": history_manifest["dataset_version"],
                "history_file": "examples/result_factor_history.csv",
                "history_sha256": _sha256(HISTORY),
                "history_manifest_file": "examples/result_factor_history_manifest.json",
                "history_manifest_sha256": _sha256(HISTORY_MANIFEST),
                "fixture_file": "sites/matchquant-result-factors/data/demo_fixtures.csv",
                "fixture_sha256": fixture_evidence["sha256"],
                "fixture_rows": fixture_evidence["rows"],
            },
            "reasons": payload["reasons"],
            "results": payload["results"],
            "validation": payload["validation"],
        }


def main() -> int:
    demo = _build_demo()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(demo, ensure_ascii=False, indent=2) + "\n"
    if str(REPO_ROOT).lower() in serialized.lower():
        raise ValueError("Generated Site data contains a local absolute path.")
    temporary = OUTPUT.with_suffix(".json.tmp")
    temporary.write_text(serialized, encoding="utf-8")
    temporary.replace(OUTPUT)
    print(
        f"status={demo['status']} fixtures={demo['fixture_count']} "
        f"rows_used={demo['rows_used']} cutoff={demo['used_data_cutoff']} "
        f"output={OUTPUT.relative_to(REPO_ROOT).as_posix()} sha256={_sha256(OUTPUT)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
