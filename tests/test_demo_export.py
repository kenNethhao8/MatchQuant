"""B05 tests: packaged resources, demo export, and resource-driven defaults.

All checks read resources through ``importlib.resources`` so they validate what
a real (non-editable) wheel install would see, not merely files in the source
checkout. The built-wheel ZIP listing itself is additionally verified by the
B05 acceptance script in the delivery directory.
"""

import hashlib
import importlib.resources
import json
import tempfile
import unittest
import unittest.mock as mock
from pathlib import Path

from matchquant import demo_export as demo_export_module
from matchquant.cli import default_m3_candidate_config
from matchquant.demo_export import export_synthetic_demo
from matchquant.schema import DataValidationError
from matchquant.selection import load_candidates


def _resource_bytes(relative: str) -> bytes:
    resource = importlib.resources.files("matchquant") / relative
    return resource.read_bytes()


class DemoResourceTests(unittest.TestCase):
    def test_payload_index_lists_consistent_files(self):
        payload = json.loads(_resource_bytes("resources/demo/demo_payload.json"))
        self.assertEqual(payload["schema_version"], "b05-demo-payload-v1")
        self.assertTrue(payload["synthetic"])
        self.assertEqual(payload["model"], "frequency")
        paths = {item["path"] for item in payload["files"]}
        expected_non_bundle = {"DEMO.md", "fixtures.csv", "results.csv"}
        self.assertTrue(expected_non_bundle.issubset(paths), paths)
        self.assertTrue(any(path.startswith("bundle/snapshot/") for path in paths), paths)
        for item in payload["files"]:
            data = _resource_bytes(f"resources/demo/{item['path']}")
            self.assertEqual(len(data), item["size"], item["path"])
            self.assertEqual(hashlib.sha256(data).hexdigest(), item["sha256"], item["path"])

    def test_bundle_payload_is_the_fixed_eight_files(self):
        payload = json.loads(_resource_bytes("resources/demo/demo_payload.json"))
        bundle_paths = sorted(
            item["path"] for item in payload["files"] if item["path"].startswith("bundle/")
        )
        self.assertEqual(len(bundle_paths), 8)
        self.assertIn("bundle/bundle_manifest.json", bundle_paths)

    def test_demo_fixtures_are_score_free_and_same_date(self):
        fixtures = _resource_bytes("resources/demo/fixtures.csv").decode("utf-8")
        results = _resource_bytes("resources/demo/results.csv").decode("utf-8")
        self.assertNotIn("home_goals", fixtures)
        self.assertIn("home_goals", results)
        fixture_rows = fixtures.strip().splitlines()
        result_rows = results.strip().splitlines()
        self.assertEqual(len(fixture_rows), 3)  # header + two fixtures
        self.assertEqual(len(result_rows), 3)
        first_date = fixture_rows[1].split(",")[1]
        self.assertEqual(first_date, fixture_rows[2].split(",")[1])
        self.assertIn("2024-01-07", first_date)

    def test_demo_doc_declares_synthetic_identity_and_boundaries(self):
        doc = _resource_bytes("resources/demo/DEMO.md").decode("utf-8")
        for expected in (
            "纯合成教学数据",
            "2024-01-07",
            "2024-01-06",
            "frequency",
            "runtime-bundle-verify",
            "snapshot-session",
            "settle",
            "不宣称",
        ):
            self.assertIn(expected, doc, expected)

    def test_default_m3_candidate_config_is_packaged_and_loadable(self):
        path = default_m3_candidate_config()
        self.assertTrue(path.is_file(), path)
        candidates, payload = load_candidates(path)
        self.assertTrue(candidates)
        self.assertEqual(payload["schema_version"], "m3-candidate-config-v1")

    def test_packaged_examples_match_repository_examples(self):
        for name in (
            "config.json",
            "forecast_poisson.json",
            "m3_candidates.json",
            "synthetic_e0_source.csv",
            "synthetic_fixtures.csv",
            "synthetic_matches.csv",
        ):
            packaged = _resource_bytes(f"resources/examples/{name}")
            repo = (Path(__file__).resolve().parent.parent / "examples" / name).read_bytes()
            self.assertEqual(packaged, repo, name)


class DemoExportTests(unittest.TestCase):
    def test_export_creates_complete_layout_and_verifies_bundle(self):
        with tempfile.TemporaryDirectory() as temporary:
            target = Path(temporary) / "demo export dir"
            result = export_synthetic_demo(target)
            self.assertTrue(result["synthetic"])
            self.assertEqual(result["model"], "frequency")
            self.assertEqual(result["bundle_verify_status"], "ready")
            demo_dir = Path(result["demo_dir"])
            self.assertEqual(demo_dir, target)
            for name in ("bundle", "fixtures.csv", "results.csv", "DEMO.md", "demo_manifest.json"):
                self.assertTrue((demo_dir / name).exists(), name)
            bundle_files = sorted(path for path in demo_dir.glob("bundle/**/*") if path.is_file())
            self.assertEqual(len(bundle_files), 8)
            self.assertTrue((demo_dir / "bundle" / "bundle_manifest.json").is_file())
            manifest = json.loads((demo_dir / "demo_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["schema_version"], "b05-demo-export-manifest-v1")
            self.assertTrue(manifest["synthetic"])
            self.assertEqual(manifest["bundle"]["verify_status"], "ready")
            self.assertEqual(manifest["package_version"], "0.1.0")
            exported = {item["path"]: item for item in manifest["files"]}
            self.assertNotIn("demo_manifest.json", exported)
            self.assertEqual(len(exported), 11)
            for relative, item in exported.items():
                data = (demo_dir / relative).read_bytes()
                self.assertEqual(len(data), item["size"], relative)
                self.assertEqual(hashlib.sha256(data).hexdigest(), item["sha256"], relative)

    def test_export_refuses_existing_directory_without_touching_it(self):
        with tempfile.TemporaryDirectory() as temporary:
            target = Path(temporary) / "existing"
            target.mkdir()
            sentinel = target / "keep.txt"
            sentinel.write_text("user data", encoding="utf-8")
            with self.assertRaises(DataValidationError) as caught:
                export_synthetic_demo(target)
            self.assertIn("already exists", str(caught.exception))
            self.assertEqual(sentinel.read_text(encoding="utf-8"), "user data")
            self.assertEqual(list(target.iterdir()), [sentinel])

    def test_export_refuses_protected_roots_before_creating_anything(self):
        package_dir = Path(demo_export_module.__file__).resolve().parent
        protected_targets = [
            Path(r"D:\Codex\matchquant_data") / "somewhere" / "demo",
            Path(r"D:\Codex\matchquant_runs") / "somewhere" / "demo",
            package_dir / "resources" / "demo" / "copy attempt",
            package_dir / "would_be_export",
            package_dir.parent / "would_be_export",
        ]
        for target in protected_targets:
            with self.assertRaises(DataValidationError) as caught:
                export_synthetic_demo(target)
            message = str(caught.exception)
            self.assertTrue(
                "lies inside" in message or "already exists" in message, (target, message)
            )
            self.assertFalse(target.exists(), target)

    def test_export_refuses_missing_parent_directory(self):
        with tempfile.TemporaryDirectory() as temporary:
            target = Path(temporary) / "not-created" / "demo"
            with self.assertRaises(DataValidationError) as caught:
                export_synthetic_demo(target)
            self.assertIn("parent directory does not exist", str(caught.exception))
            self.assertFalse(target.exists())

    def test_export_failure_leaves_partial_directory_without_success_manifest(self):
        # Inject a payload whose bundle manifest is missing: the export must
        # leave a real partial directory and never write demo_manifest.json.
        with tempfile.TemporaryDirectory() as temporary:
            target = Path(temporary) / "broken export"
            with mock.patch.object(
                demo_export_module,
                "_read_payload_bytes",
                return_value=[("bundle/extra.bin", b"unexpected payload")],
            ):
                with self.assertRaises(DataValidationError) as caught:
                    export_synthetic_demo(target)
            self.assertIn("INCOMPLETE", str(caught.exception))
            self.assertTrue((target / "bundle" / "extra.bin").is_file())
            self.assertFalse((target / "demo_manifest.json").exists())
            self.assertFalse((target / "bundle" / "bundle_manifest.json").exists())


if __name__ == "__main__":
    unittest.main()
