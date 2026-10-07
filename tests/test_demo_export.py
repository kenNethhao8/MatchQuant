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


def _catalog_entry(path: str, content: bytes = b"payload") -> dict:
    return {"path": path, "size": len(content), "sha256": hashlib.sha256(content).hexdigest()}


def _valid_catalog() -> list[dict]:
    entries = [
        _catalog_entry(path) for path in sorted(demo_export_module._expected_export_paths())
    ]
    entries += [
        _catalog_entry(f"bundle/snapshot/m11_demo_snapshot/{leaf}")
        for leaf in demo_export_module._SNAPSHOT_LEAF_NAMES
    ]
    return entries


class PayloadCatalogValidationTests(unittest.TestCase):
    """R1: the export catalog is strictly validated before any read or write."""

    def test_valid_catalog_passes_with_eleven_entries(self):
        validated = demo_export_module._validate_payload_catalog(_valid_catalog())
        self.assertEqual(len(validated), 11)

    def test_parent_traversal_entry_rejected(self):
        catalog = _valid_catalog() + [_catalog_entry("../../escape.txt", b"escaped")]
        with self.assertRaises(DataValidationError) as caught:
            demo_export_module._validate_payload_catalog(catalog)
        self.assertIn("canonical relative path", str(caught.exception))

    def test_absolute_windows_path_rejected(self):
        catalog = _valid_catalog() + [_catalog_entry("C:\\evil\\escape.txt", b"escaped")]
        with self.assertRaises(DataValidationError):
            demo_export_module._validate_payload_catalog(catalog)

    def test_unc_path_rejected(self):
        catalog = _valid_catalog() + [_catalog_entry("\\\\server\\share\\escape.txt", b"escaped")]
        with self.assertRaises(DataValidationError):
            demo_export_module._validate_payload_catalog(catalog)

    def test_ads_colon_path_rejected(self):
        catalog = _valid_catalog() + [_catalog_entry("bundle/RUN.md:stream", b"ads")]
        with self.assertRaises(DataValidationError):
            demo_export_module._validate_payload_catalog(catalog)

    def test_backslash_path_rejected(self):
        catalog = _valid_catalog() + [_catalog_entry("bundle\\RUN.md", b"backslash")]
        with self.assertRaises(DataValidationError):
            demo_export_module._validate_payload_catalog(catalog)

    def test_dot_component_rejected(self):
        catalog = _valid_catalog() + [_catalog_entry("bundle/./RUN.md", b"dot")]
        with self.assertRaises(DataValidationError):
            demo_export_module._validate_payload_catalog(catalog)

    def test_trailing_dot_component_rejected(self):
        catalog = _valid_catalog() + [_catalog_entry("bundle/snapshot/id/manifest.", b"dot")]
        with self.assertRaises(DataValidationError):
            demo_export_module._validate_payload_catalog(catalog)

    def test_non_string_path_rejected(self):
        catalog = _valid_catalog()
        catalog[0] = {"path": 5, "size": 1, "sha256": "a" * 64}
        with self.assertRaises(DataValidationError) as caught:
            demo_export_module._validate_payload_catalog(catalog)
        self.assertIn("must be a non-empty string", str(caught.exception))

    def test_exact_duplicate_rejected(self):
        catalog = _valid_catalog() + [_catalog_entry("DEMO.md", b"again")]
        with self.assertRaises(DataValidationError) as caught:
            demo_export_module._validate_payload_catalog(catalog)
        self.assertIn("duplicate", str(caught.exception))

    def test_windows_case_insensitive_duplicate_rejected(self):
        catalog = _valid_catalog() + [_catalog_entry("demo.md", b"case")]
        with self.assertRaises(DataValidationError) as caught:
            demo_export_module._validate_payload_catalog(catalog)
        self.assertIn("case-insensitive duplicates", str(caught.exception))

    def test_extra_entry_outside_fixed_set_rejected(self):
        catalog = _valid_catalog() + [_catalog_entry("extras/free.txt", b"extra")]
        with self.assertRaises(DataValidationError) as caught:
            demo_export_module._validate_payload_catalog(catalog)
        self.assertIn("outside the fixed export set", str(caught.exception))

    def test_wrong_snapshot_leaf_shape_rejected(self):
        catalog = _valid_catalog()
        catalog[-1] = _catalog_entry("bundle/snapshot/m11_demo_snapshot/model_states.bak")
        with self.assertRaises(DataValidationError) as caught:
            demo_export_module._validate_payload_catalog(catalog)
        self.assertIn("outside the fixed export set", str(caught.exception))

    def test_snapshot_entries_across_two_directories_rejected(self):
        catalog = _valid_catalog()
        catalog[-3] = _catalog_entry("bundle/snapshot/other_id/model_states.json")
        with self.assertRaises(DataValidationError) as caught:
            demo_export_module._validate_payload_catalog(catalog)
        self.assertIn("exactly one snapshot directory", str(caught.exception))

    def test_missing_snapshot_leaf_rejected(self):
        catalog = _valid_catalog()
        catalog.pop()
        with self.assertRaises(DataValidationError) as caught:
            demo_export_module._validate_payload_catalog(catalog)
        self.assertIn("snapshot payload entries", str(caught.exception))

    def test_missing_required_entry_rejected(self):
        catalog = [entry for entry in _valid_catalog() if entry["path"] != "results.csv"]
        with self.assertRaises(DataValidationError) as caught:
            demo_export_module._validate_payload_catalog(catalog)
        self.assertIn("missing required export entries", str(caught.exception))

    def test_bad_size_and_hash_types_rejected(self):
        for mutate in (
            lambda entry: {**entry, "size": "12"},
            lambda entry: {**entry, "size": True},
            lambda entry: {**entry, "size": -1},
            lambda entry: {**entry, "sha256": "not-hex"},
            lambda entry: {**entry, "sha256": 64},
        ):
            catalog = _valid_catalog()
            catalog[0] = mutate(catalog[0])
            with self.assertRaises(DataValidationError):
                demo_export_module._validate_payload_catalog(catalog)

    def test_tampered_index_rejected_before_target_creation(self):
        # The catalog validation (with the same correct-hash escape entry the
        # reviewer used) must fail before the target directory is created.
        with tempfile.TemporaryDirectory() as temporary:
            target = Path(temporary) / "escaped export"
            tampered = _valid_catalog() + [_catalog_entry("../../escape.txt", b"escaped")]
            with self.assertRaises(DataValidationError):
                demo_export_module._validate_payload_catalog(tampered)
            with mock.patch.object(
                demo_export_module,
                "_load_payload",
                side_effect=lambda: demo_export_module._validate_payload_catalog(tampered),
            ):
                with self.assertRaises(DataValidationError):
                    export_synthetic_demo(target)
            self.assertFalse(target.exists())
            self.assertFalse((Path(temporary) / "escape.txt").exists())


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

    def test_final_manifest_write_failure_names_stage_and_partial_directory(self):
        # R3: an injected PermissionError on the FINAL demo_manifest.json write
        # must produce the honest partial-export diagnosis on the real chain.
        real_write_text = Path.write_text

        def failing_write_text(self, *args, **kwargs):
            if self.name == "demo_manifest.json":
                raise PermissionError("injected final manifest failure")
            return real_write_text(self, *args, **kwargs)

        with tempfile.TemporaryDirectory() as temporary:
            target = Path(temporary) / "finalize failure export"
            with mock.patch.object(Path, "write_text", failing_write_text):
                with self.assertRaises(DataValidationError) as caught:
                    export_synthetic_demo(target)
            message = str(caught.exception)
            self.assertIn("INCOMPLETE", message)
            self.assertIn("must not be used", message)
            self.assertIn("demo_manifest.json", message)
            self.assertIn(str(target), message)
            self.assertIn("may be intact", message)  # bundle content is not declared invalid
            exported = sorted(path.name for path in target.rglob("*") if path.is_file())
            self.assertEqual(len(exported), 11)
            self.assertNotIn("demo_manifest.json", exported)

    def test_verify_unexpected_error_keeps_type_chain_and_note(self):
        # An unknown programming error during verification must NOT be
        # misclassified as a validation problem: the real exception type and
        # chain survive, with the partial-export context attached as a note.
        with tempfile.TemporaryDirectory() as temporary:
            target = Path(temporary) / "unexpected error export"
            with mock.patch.object(
                demo_export_module,
                "verify_runtime_bundle",
                side_effect=RuntimeError("unexpected engine bug"),
            ):
                with self.assertRaises(RuntimeError) as caught:
                    export_synthetic_demo(target)
            self.assertIn("unexpected engine bug", str(caught.exception))
            notes = " ".join(getattr(caught.exception, "__notes__", []))
            self.assertIn("INCOMPLETE", notes)
            self.assertIn(str(target), notes)
            self.assertFalse((target / "demo_manifest.json").exists())


if __name__ == "__main__":
    unittest.main()
