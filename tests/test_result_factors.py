from __future__ import annotations

import builtins
import csv
import hashlib
import io
import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from matchquant import (
    build_result_factor_snapshot,
    load_result_factor_snapshot,
    query_result_factors,
)
from matchquant.engine_snapshot_support import canonical_json_bytes, configuration_hash
from matchquant.io import file_sha256
from matchquant.models import EloModel
from matchquant.result_factors import ELO_CONFIG, RESULT_FACTOR_CONFIG, RESULT_FACTOR_CONFIG_SHA256
from matchquant.schema import DataValidationError, REQUIRED_COLUMNS


def sample_rows() -> list[list[object]]:
    return [
        ["m1", "2023-08-01", "2023-24", "Alpha", "Beta", 2, 0],
        ["m2", "2023-08-01", "2023-24", "Gamma", "Alpha", 1, 1],
        ["m3", "2023-08-10", "2023-24", "Beta", "Alpha", 0, 1],
        ["m4", "2024-02-01", "2023-24", "Alpha", "Gamma", 0, 2],
        ["m5", "2024-08-15", "2024-25", "Beta", "Alpha", 1, 2],
    ]


def write_history(input_dir: Path, rows: list[list[object]]) -> tuple[Path, Path]:
    input_dir.mkdir(parents=True, exist_ok=True)
    history = input_dir / "history.csv"
    manifest = input_dir / "history_manifest.json"
    with history.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(REQUIRED_COLUMNS)
        writer.writerows(rows)
    dates = [pd.Timestamp(row[1]) for row in rows]
    cutoff = max(dates).isoformat()
    digest = hashlib.sha256(history.read_bytes()).hexdigest()
    payload = {
        "schema_version": "m2.1-processed-manifest-v1",
        "dataset_version": "m13a-synthetic-v1",
        "data_cutoff": cutoff,
        "output": {
            "filename": history.name,
            "sha256": digest,
            "rows": len(rows),
            "columns": list(REQUIRED_COLUMNS),
        },
    }
    manifest.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return history, manifest


def write_fixtures(input_dir: Path, rows: list[list[object]], *, scored: bool = False) -> Path:
    input_dir.mkdir(parents=True, exist_ok=True)
    fixtures = input_dir / "fixtures.csv"
    columns = ["fixture_id", "date", "season", "home_team", "away_team"]
    if scored:
        columns.append("home_goals")
    with fixtures.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(columns)
        writer.writerows(rows)
    return fixtures


def repack_snapshot(snapshot_dir: Path, mutate) -> Path:
    """Copy and semantically mutate one snapshot while synchronizing file hashes/size inventory."""
    target = snapshot_dir.parent / ("tampered_" + next(tempfile._get_candidate_names()))
    shutil.copytree(snapshot_dir, target)
    manifest_path = target / "factor_manifest.json"
    state_path = target / "team_factor_state.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    state = json.loads(state_path.read_text(encoding="utf-8"))
    manifest["snapshot_id"] = target.name
    mutate(manifest, state)
    state_bytes = canonical_json_bytes(state)
    state_path.write_bytes(state_bytes)
    manifest["state"]["sha256"] = hashlib.sha256(state_bytes).hexdigest()
    manifest["state"]["bytes"] = len(state_bytes)
    for _ in range(12):
        data = canonical_json_bytes(manifest)
        inventory = {
            "manifest_bytes": len(data),
            "total_bytes": len(data) + state_path.stat().st_size + (target / "factor_build_report.md").stat().st_size,
        }
        if manifest["inventory"] == inventory:
            break
        manifest["inventory"] = inventory
    manifest_path.write_bytes(canonical_json_bytes(manifest))
    return manifest_path


class ResultFactorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory(prefix="m13a test ")
        self.root = Path(self.temp.name)
        self.input_dir = self.root / "source inputs with spaces"
        self.build_root = self.root / "snapshot outputs"
        self.query_root = self.root / "query outputs"
        self.history, self.history_manifest = write_history(self.input_dir, sample_rows())

    def tearDown(self) -> None:
        self.temp.cleanup()

    def build(self, as_of: str = "2024-08-16", rows: list[list[object]] | None = None, root: Path | None = None):
        if rows is not None:
            self.history, self.history_manifest = write_history(self.input_dir, rows)
        return build_result_factor_snapshot(
            self.history, self.history_manifest, as_of,
            output_root=root or self.build_root,
        )

    def fixtures(self, rows: list[list[object]] | None = None, *, scored: bool = False) -> Path:
        if rows is None:
            rows = [["f1", "2024-08-18", "2024-25", "Alpha", "Beta"]]
        return write_fixtures(self.input_dir, rows, scored=scored)

    def query(self, manifest_path: str | Path, fixtures_path: Path | None = None, root: Path | None = None):
        return query_result_factors(
            manifest_path, fixtures_path or self.fixtures(), output_root=root or self.query_root,
        )

    def test_hand_calculated_windows_venue_rest_and_frozen_batch_elo(self) -> None:
        built = self.build()
        self.assertEqual(built["status"], "ready")
        self.assertEqual(built["rows_used"], 5)
        loaded = load_result_factor_snapshot(built["manifest_path"])
        state = loaded["state"]
        alpha = state["teams"]["Alpha"]
        beta = state["teams"]["Beta"]
        self.assertEqual(state["config"], RESULT_FACTOR_CONFIG)
        self.assertEqual(state["config_sha256"], RESULT_FACTOR_CONFIG_SHA256)
        self.assertEqual(state["as_of"]["value"], "2024-08-16")
        self.assertEqual(alpha["total_appearances"], 5)
        self.assertEqual(beta["total_appearances"], 3)
        self.assertEqual(len(alpha["appearances"]), 5)
        self.assertEqual(len(beta["appearances"]), 3)
        # Both Alpha appearances in the 1 August batch see the pre-batch Elo.
        same_batch = [item for item in alpha["appearances"] if item["batch_time"].startswith("2023-08-01")]
        self.assertEqual([item["match_id"] for item in same_batch], ["m1", "m2"])
        self.assertEqual([item["opponent_pre_batch_elo"] for item in same_batch], [1500.0, 1500.0])

        tracker = EloModel(**ELO_CONFIG)
        for row in sorted(sample_rows(), key=lambda item: (pd.Timestamp(item[1]), item[0])):
            tracker.update(str(row[3]), str(row[4]), int(row[5]), int(row[6]))
        self.assertAlmostEqual(alpha["final_elo"], tracker.ratings["Alpha"], places=12)
        self.assertAlmostEqual(beta["final_elo"], tracker.ratings["Beta"], places=12)

        fixture_path = self.fixtures()
        queried = self.query(built["manifest_path"], fixture_path)
        self.assertEqual(queried["status"], "ready")
        result = queried["results"][0]
        self.assertEqual(result["fixture_id"], "f1")
        self.assertEqual(result["home"]["recent_5"]["n"], 5)
        self.assertTrue(result["home"]["recent_5"]["window_complete"])
        self.assertEqual(result["home"]["recent_10"]["n"], 5)
        self.assertFalse(result["home"]["recent_10"]["window_complete"])
        self.assertFalse(result["home"]["recent_5"]["small_sample"])
        self.assertAlmostEqual(result["home"]["recent_5"]["points_rate"], 10 / 15)
        self.assertAlmostEqual(result["home"]["recent_5"]["goals_for_mean"], 6 / 5)
        self.assertAlmostEqual(result["home"]["recent_5"]["goals_against_mean"], 4 / 5)
        self.assertAlmostEqual(result["home"]["venue"]["goals_for_shrunk"], 6 / 7)
        self.assertAlmostEqual(result["home"]["venue"]["goals_against_shrunk"], 8 / 7)
        self.assertEqual(result["home"]["venue"]["sample_n"], 2)
        self.assertTrue(result["home"]["venue"]["prior_used"])
        self.assertAlmostEqual(result["away"]["recent_5"]["points_rate"], 0.0)
        self.assertEqual(result["away"]["recent_5"]["n"], 3)
        self.assertFalse(result["away"]["recent_5"]["small_sample"])
        self.assertFalse(result["away"]["recent_5"]["window_complete"])
        self.assertAlmostEqual(result["away"]["venue"]["goals_for_shrunk"], 1.0)
        self.assertAlmostEqual(result["away"]["venue"]["goals_against_shrunk"], 1.0)
        self.assertEqual(result["home"]["rest_calendar_days"], 3)
        self.assertEqual(result["away"]["rest_calendar_days"], 3)
        self.assertTrue(result["home"]["unverified_intervening_matches"])
        self.assertFalse(queried["payload"]["probability_adjustment_applied"])
        self.assertNotIn("home_win_prob", result)
        with (Path(queried["csv_path"])).open(encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(len(rows), 1)
        self.assertAlmostEqual(float(rows[0]["home_points_rate_5"]), 10 / 15)
        self.assertIn("away_venue_goals_against_shrunk", rows[0])

    def test_date_boundary_and_post_as_of_mutations_leave_factor_state_unchanged(self) -> None:
        original = self.build()
        original_loaded = load_result_factor_snapshot(original["manifest_path"])
        original_query = self.query(original["manifest_path"])
        changed_rows = sample_rows() + [
            ["m6", "2024-08-16", "2024-25", "Alpha", "Gamma", 4, 0],
            ["m7", "2025-01-03", "2024-25", "Gamma", "Beta", 0, 3],
        ]
        changed = self.build(rows=changed_rows, root=self.root / "future snapshots")
        changed_loaded = load_result_factor_snapshot(changed["manifest_path"])
        changed_query = self.query(changed["manifest_path"], root=self.root / "future queries")
        self.assertEqual(original["rows_used"], 5)
        self.assertEqual(changed["rows_used"], 5)
        self.assertEqual(changed["used_data_cutoff"], "2024-08-15")
        self.assertEqual(original["state_sha256"], changed["state_sha256"])
        self.assertEqual(original_loaded["state"], changed_loaded["state"])
        self.assertEqual(original_query["results"], changed_query["results"])
        self.assertEqual(file_sha256(self.history), changed_loaded["manifest"]["history"]["sha256"])

        exact_input = sample_rows() + [["m6", "2024-08-16", "2024-25", "Alpha", "Gamma", 1, 0]]
        exact = self.build(rows=exact_input, as_of="2024-08-16T12:00:00", root=self.root / "exact snapshots")
        self.assertEqual(exact["status"], "blocked")
        self.assertEqual(exact["failure_stage"], "as_of_filter")
        self.assertIsNone(exact["manifest_path"])

        # Exact as-of excludes an exact-time same-batch result; date-only as-of excludes the whole date.
        timed_rows = [[*row] for row in sample_rows()]
        timed_rows.append(["m6", "2024-08-16T12:00:00", "2024-25", "Alpha", "Gamma", 1, 0])
        timed = self.build(rows=timed_rows, as_of="2024-08-16", root=self.root / "date-only snapshots")
        self.assertEqual(timed["status"], "ready")
        self.assertEqual(timed["rows_used"], 5)

    def test_unknown_small_sample_empty_history_and_rest_edges(self) -> None:
        built = self.build()
        fixtures = self.fixtures([
            ["small", "2024-08-18", "2024-25", "Alpha", "Gamma"],
            ["unknown", "2024-08-20", "2024-25", "New FC", "Alpha"],
        ])
        query = self.query(built["manifest_path"], fixtures)
        self.assertEqual(query["status"], "needs_review")
        self.assertEqual([row["fixture_id"] for row in query["results"]], ["small", "unknown"])
        gamma = query["results"][0]["away"]
        self.assertEqual(gamma["recent_5"]["n"], 2)
        self.assertTrue(gamma["recent_5"]["small_sample"])
        self.assertFalse(gamma["recent_5"]["window_complete"])
        new_team = query["results"][1]["home"]
        self.assertTrue(new_team["unknown_team"])
        self.assertTrue(new_team["elo_prior"])
        self.assertEqual(new_team["elo_rating"], 1500.0)
        self.assertEqual(new_team["recent_5"]["n"], 0)
        self.assertIsNone(new_team["recent_5"]["points_rate"])
        self.assertIsNone(new_team["rest_calendar_days"])
        self.assertAlmostEqual(new_team["venue"]["goals_for_shrunk"], 0.8)
        self.assertTrue(new_team["venue"]["prior_used"])
        self.assertTrue(new_team["unverified_intervening_matches"])

        empty = self.build(as_of="2023-07-01", root=self.root / "empty snapshots")
        self.assertEqual(empty["status"], "needs_review")
        self.assertEqual(empty["rows_used"], 0)
        empty_query = self.query(empty["manifest_path"], self.fixtures([["none", "2023-07-05", "2023-24", "Unknown A", "Unknown B"]]), root=self.root / "empty queries")
        self.assertEqual(empty_query["status"], "needs_review")
        factors = empty_query["results"][0]
        self.assertIsNone(factors["home"]["venue"]["goals_for_shrunk"])
        self.assertIsNone(factors["home"]["rest_calendar_days"])
        self.assertIn("no_eligible_history", empty_query["reasons"])

        from matchquant.result_factors import calendar_rest_days
        self.assertIsNone(calendar_rest_days("2024-08-01", None))
        self.assertEqual(calendar_rest_days("2024-08-18", "2024-08-15"), 3)
        with self.assertRaises(DataValidationError):
            calendar_rest_days("2024-08-14", "2024-08-15")
        before_asof = self.query(built["manifest_path"], self.fixtures([["bad-date", "2024-08-15", "2024-25", "Alpha", "Beta"]]), root=self.root / "bad-date queries")
        self.assertEqual(before_asof["status"], "blocked")
        self.assertEqual(before_asof["failure_stage"], "fixture_time_validation")

    def test_window_cap_and_semantically_rehashed_payload_rejections(self) -> None:
        from datetime import date, timedelta
        long_rows = []
        start = date(2020, 1, 1)
        for index in range(12):
            day = start + timedelta(days=index * 7)
            if index % 2:
                home, away, hg, ag = "Beta", "Alpha", index % 3, (index + 1) % 3
            else:
                home, away, hg, ag = "Alpha", "Beta", index % 3, (index + 1) % 3
            long_rows.append([f"m{index:02d}", day.isoformat(), "2019-20", home, away, hg, ag])
        built = self.build(rows=long_rows, as_of="2021-01-01")
        loaded = load_result_factor_snapshot(built["manifest_path"])
        alpha = loaded["state"]["teams"]["Alpha"]
        self.assertEqual(alpha["total_appearances"], 12)
        self.assertEqual(len(alpha["appearances"]), 10)
        self.assertEqual(alpha["appearances"][0]["match_id"], "m02")
        self.assertLessEqual(loaded["manifest"]["inventory"]["total_bytes"], 5 * 1024 * 1024)
        source_manifest = loaded["manifest_path"]
        snapshot_dir = Path(source_manifest).parent

        def expect_reject(change) -> None:
            bad_manifest = repack_snapshot(snapshot_dir, change)
            with self.assertRaises((DataValidationError, FileNotFoundError)):
                load_result_factor_snapshot(bad_manifest)

        expect_reject(lambda manifest, state: state.__setitem__("rows_used", True))
        expect_reject(lambda manifest, state: state["teams"]["Alpha"]["appearances"].reverse())
        expect_reject(lambda manifest, state: state["teams"]["Alpha"]["appearances"].append(dict(state["teams"]["Alpha"]["appearances"][-1])))

        def truncated_window(manifest, state):
            state["teams"]["Alpha"]["appearances"].pop(0)
        expect_reject(truncated_window)

        def mismatch_asof(manifest, state):
            state["as_of"] = {"value": "2021-01-02", "timestamp": "2021-01-02T00:00:00", "date_only": True}
        expect_reject(mismatch_asof)

        def unsafe_path(manifest, state):
            manifest["state"]["filename"] = "../outside.json"
        expect_reject(unsafe_path)

        def rehashed_config_change(manifest, state):
            config = json.loads(json.dumps(manifest["config"]))
            config["elo"]["k_factor"] = 31
            digest = configuration_hash(config)
            manifest["config"] = config
            manifest["config_sha256"] = digest
            state["config"] = config
            state["config_sha256"] = digest
        expect_reject(rehashed_config_change)

        def future_appearance(manifest, state):
            appearance = state["teams"]["Alpha"]["appearances"][-1]
            appearance["date"] = "2021-01-02"
            appearance["batch_time"] = "2021-01-02T00:00:00"
        expect_reject(future_appearance)

    def test_query_is_offline_after_source_history_is_disconnected(self) -> None:
        built = self.build()
        fixtures = self.fixtures()
        detached = self.input_dir / "detached source"
        detached.mkdir()
        history_copy = detached / self.history.name
        manifest_copy = detached / self.history_manifest.name
        self.history.replace(history_copy)
        self.history_manifest.replace(manifest_copy)
        self.assertFalse(self.history.exists())
        self.assertFalse(self.history_manifest.exists())
        guarded_sources = {self.history.resolve(), self.history_manifest.resolve()}
        source_attempts: list[str] = []
        network_attempts: list[str] = []

        def deny_source(path: object, mode: str = "r") -> None:
            if not isinstance(path, (str, os.PathLike)):
                return
            try:
                normalized = Path(path).resolve()
            except (OSError, TypeError, ValueError):
                return
            if normalized in guarded_sources and "r" in mode:
                source_attempts.append(str(normalized))
                raise PermissionError("synthetic history source access denied by test guard")

        original_builtin_open = builtins.open
        original_io_open = io.open
        original_os_open = os.open

        def guarded_builtin_open(path, mode="r", *args, **kwargs):
            deny_source(path, mode)
            return original_builtin_open(path, mode, *args, **kwargs)

        def guarded_io_open(path, mode="r", *args, **kwargs):
            deny_source(path, mode)
            return original_io_open(path, mode, *args, **kwargs)

        def guarded_os_open(path, flags, *args, **kwargs):
            if flags & os.O_ACCMODE == os.O_RDONLY:
                deny_source(path, "r")
            return original_os_open(path, flags, *args, **kwargs)

        def deny_network(*args, **kwargs):
            network_attempts.append("connect")
            raise PermissionError("network disabled by test guard")

        with patch("builtins.open", guarded_builtin_open), patch("io.open", guarded_io_open), patch("os.open", guarded_os_open), patch.object(socket, "create_connection", deny_network), patch.object(socket.socket, "connect", deny_network):
            queried = self.query(built["manifest_path"], fixtures, root=self.root / "offline query outputs")
        self.assertEqual(queried["status"], "ready")
        self.assertEqual(source_attempts, [])
        self.assertEqual(network_attempts, [])
        self.assertFalse(queried["payload"]["validation"]["history_read_during_query"])
        self.assertFalse(queried["payload"]["validation"]["network_accessed"])

        # Negative control proves the same guard detects and blocks a source open.
        control_attempts: list[str] = []
        def control_open(path, mode="r", *args, **kwargs):
            if Path(path).resolve() in guarded_sources and "r" in mode:
                control_attempts.append(str(path))
                raise PermissionError("negative control blocked")
            return original_io_open(path, mode, *args, **kwargs)
        with patch("io.open", control_open):
            with self.assertRaises(PermissionError):
                self.history.open("rb")
        self.assertEqual(len(control_attempts), 1)

    def test_loader_rejects_reparse_state_leaf(self) -> None:
        built = self.build()
        state_path = Path(built["state_path"])
        saved_state = self.root / "saved state.json"
        state_path.replace(saved_state)
        target_dir = self.root / "reparse target"
        target_dir.mkdir()
        if os.name == "nt":
            powershell = shutil.which("powershell.exe") or shutil.which("powershell")
            self.assertIsNotNone(powershell)
            command = (
                f"New-Item -ItemType Junction -Path '{state_path}' -Target '{target_dir}' "
                "-ErrorAction Stop | Out-Null"
            )
            created = subprocess.run(
                [powershell, "-NoProfile", "-NonInteractive", "-Command", command],
                text=True, capture_output=True, check=False,
            )
            self.assertEqual(created.returncode, 0, created.stderr)
        else:
            os.symlink(target_dir, state_path, target_is_directory=True)
        try:
            with self.assertRaisesRegex(DataValidationError, "reparse"):
                load_result_factor_snapshot(built["manifest_path"])
        finally:
            if os.name == "nt":
                os.rmdir(state_path)
            else:
                state_path.unlink()

    def test_score_free_identity_order_repeatability_and_output_guards(self) -> None:
        built = self.build()
        fixtures = self.fixtures([
            ["f-second", "2024-08-20", "2024-25", "Beta", "Alpha"],
            ["f-first", "2024-08-18", "2024-25", "Alpha", "Beta"],
        ])
        start_hashes = (file_sha256(self.history), file_sha256(self.history_manifest), file_sha256(fixtures), file_sha256(built["manifest_path"]), file_sha256(built["state_path"]), file_sha256(built["report_path"]))
        one = self.query(built["manifest_path"], fixtures, root=self.root / "repeat one")
        two = self.query(built["manifest_path"], fixtures, root=self.root / "repeat two")
        self.assertEqual([r["fixture_id"] for r in one["results"]], ["f-second", "f-first"])
        self.assertEqual(one["results"], two["results"])
        self.assertNotEqual(one["query_path"], two["query_path"])
        end_hashes = (file_sha256(self.history), file_sha256(self.history_manifest), file_sha256(fixtures), file_sha256(built["manifest_path"]), file_sha256(built["state_path"]), file_sha256(built["report_path"]))
        self.assertEqual(start_hashes, end_hashes)
        scored = self.fixtures([["scored", "2024-08-18", "2024-25", "Alpha", "Beta", 2]], scored=True)
        bad = self.query(built["manifest_path"], scored, root=self.root / "score rejection")
        self.assertEqual(bad["status"], "blocked")
        self.assertEqual(bad["failure_stage"], "fixture_load")

        bad_path = self.root / "will not be created"
        with self.assertRaises(DataValidationError):
            build_result_factor_snapshot(self.history, self.history_manifest, "2024-08-16", output_root=str(self.root / "child" / ".." / "bad_path"))
        self.assertFalse((self.root / "bad_path").exists())
        with self.assertRaises(DataValidationError):
            build_result_factor_snapshot(self.history, self.history_manifest, "2024-08-16", output_root=self.input_dir / "nested output")
        self.assertFalse((self.input_dir / "nested output").exists())
        protected = Path(r"D:\Codex\matchquant_data\m13a-do-not-create")
        self.assertFalse(protected.exists())
        with self.assertRaises(DataValidationError):
            build_result_factor_snapshot(self.history, self.history_manifest, "2024-08-16", output_root=protected)
        self.assertFalse(protected.exists())

    def test_cli_exit_codes_and_reports(self) -> None:
        good = self.build()
        fixtures = self.fixtures()
        env = os.environ.copy()
        env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1])
        def run(*args: str) -> subprocess.CompletedProcess:
            return subprocess.run(
                [sys.executable, "-m", "matchquant", *args], cwd=Path(__file__).resolve().parents[1],
                env=env, text=True, capture_output=True, check=False,
            )
        built_cli = run(
            "result-factor-build", "--history", str(self.history), "--history-manifest", str(self.history_manifest),
            "--as-of", "2024-08-16", "--output-root", str(self.root / "cli builds"),
        )
        self.assertEqual(built_cli.returncode, 0, built_cli.stderr)
        manifest_path = next((self.root / "cli builds").glob("m13_result_factor_snapshot_*/factor_manifest.json"))

        review_build_cli = run(
            "result-factor-build", "--history", str(self.history), "--history-manifest", str(self.history_manifest),
            "--as-of", "2023-07-01", "--output-root", str(self.root / "cli review builds"),
        )
        self.assertEqual(review_build_cli.returncode, 2, review_build_cli.stderr)
        review_build_lines = dict(line.split("=", 1) for line in review_build_cli.stdout.splitlines() if "=" in line)
        self.assertEqual(review_build_lines["status"], "needs_review")
        self.assertEqual(review_build_lines["rows_used"], "0")
        review_build_manifest = json.loads(Path(review_build_lines["factor_manifest"]).read_text(encoding="utf-8"))
        self.assertEqual(review_build_manifest["status"], "needs_review")
        self.assertTrue(Path(review_build_lines["factor_build_report"]).is_file())

        blocked_build_cli = run(
            "result-factor-build", "--history", str(self.input_dir / "missing history.csv"),
            "--history-manifest", str(self.history_manifest), "--as-of", "2024-08-16",
            "--output-root", str(self.root / "cli blocked builds"),
        )
        self.assertEqual(blocked_build_cli.returncode, 1, blocked_build_cli.stderr)
        blocked_build_lines = dict(line.split("=", 1) for line in blocked_build_cli.stdout.splitlines() if "=" in line)
        self.assertEqual(blocked_build_lines["status"], "blocked")
        self.assertNotIn("factor_manifest", blocked_build_lines)
        blocked_build_dir = Path(blocked_build_lines["snapshot_dir"])
        self.assertEqual(json.loads((blocked_build_dir / "factor_build_failure.json").read_text(encoding="utf-8"))["status"], "blocked")
        self.assertTrue(Path(blocked_build_lines["factor_build_report"]).is_file())
        self.assertEqual(run("result-factor-build").returncode, 2)

        query_cli = run(
            "result-factor-query", "--factor-manifest", str(manifest_path), "--fixtures", str(fixtures),
            "--output-root", str(self.root / "cli queries"),
        )
        self.assertEqual(query_cli.returncode, 0, query_cli.stderr)
        query_path = next((self.root / "cli queries").glob("m13_result_factor_query_*/factor_query.json"))
        query_payload = json.loads(query_path.read_text(encoding="utf-8"))
        self.assertEqual(query_payload["status"], "ready")
        self.assertIn("probability_adjustment_applied：`false`", (query_path.parent / "factor_query_report.md").read_text(encoding="utf-8"))

        unknown_fixtures = self.fixtures([["unknown", "2024-08-18", "2024-25", "New FC", "Alpha"]])
        unknown_cli = run(
            "result-factor-query", "--factor-manifest", str(manifest_path), "--fixtures", str(unknown_fixtures),
            "--output-root", str(self.root / "cli unknown"),
        )
        self.assertEqual(unknown_cli.returncode, 2, unknown_cli.stderr)
        unknown_path = next((self.root / "cli unknown").glob("m13_result_factor_query_*/factor_query.json"))
        self.assertEqual(json.loads(unknown_path.read_text(encoding="utf-8"))["status"], "needs_review")

        bad_cli = run(
            "result-factor-query", "--factor-manifest", str(manifest_path), "--fixtures", str(self.input_dir / "missing.csv"),
            "--output-root", str(self.root / "cli bad input"),
        )
        self.assertEqual(bad_cli.returncode, 1, bad_cli.stderr)
        self.assertIn("status=blocked", bad_cli.stdout)
        bad_query_path = next((self.root / "cli bad input").glob("m13_result_factor_query_*/factor_query.json"))
        self.assertEqual(json.loads(bad_query_path.read_text(encoding="utf-8"))["status"], "blocked")
        self.assertTrue((bad_query_path.parent / "factor_query_report.md").is_file())
        usage_cli = run("result-factor-query", "--factor-manifest", str(manifest_path))
        self.assertEqual(usage_cli.returncode, 2)

    def test_required_report_write_failures_never_return_ready(self) -> None:
        from matchquant.result_factor_snapshot import RESULT_FACTOR_BUILD_REPORT_FILENAME, RESULT_FACTOR_QUERY_REPORT_FILENAME

        original_write_bytes = Path.write_bytes

        def fail_build_report(path: Path, payload: bytes) -> int:
            if path.name == RESULT_FACTOR_BUILD_REPORT_FILENAME:
                raise PermissionError("synthetic required build report write failure")
            return original_write_bytes(path, payload)

        with patch.object(Path, "write_bytes", fail_build_report):
            failed_build = self.build(root=self.root / "build report failure")
        self.assertEqual(failed_build["status"], "blocked")
        self.assertIsNone(failed_build["manifest_path"])
        self.assertEqual(json.loads(Path(failed_build["failure_path"]).read_text(encoding="utf-8"))["status"], "blocked")

        built = self.build(root=self.root / "query snapshot")
        fixtures = self.fixtures()
        original_write_text = Path.write_text

        def fail_query_report(path: Path, payload: str, *args, **kwargs) -> int:
            if path.name == RESULT_FACTOR_QUERY_REPORT_FILENAME:
                raise PermissionError("synthetic required query report write failure")
            return original_write_text(path, payload, *args, **kwargs)

        query_root = self.root / "query report failure"
        with patch.object(Path, "write_text", fail_query_report):
            with self.assertRaises(PermissionError):
                self.query(built["manifest_path"], fixtures, root=query_root)
        blocked_query_path = next(query_root.glob("m13_result_factor_query_*/factor_query.json"))
        self.assertEqual(json.loads(blocked_query_path.read_text(encoding="utf-8"))["status"], "blocked")
        self.assertFalse((blocked_query_path.parent / RESULT_FACTOR_QUERY_REPORT_FILENAME).exists())


if __name__ == "__main__":
    unittest.main()
