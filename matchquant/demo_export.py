"""Export the packaged synthetic demo (B05) into a new user directory.

``export_synthetic_demo`` copies the demo payload that ships inside the
matchquant wheel — the fixed 8-file frequency runtime bundle, score-free
fixtures, independent canonical results, and ``DEMO.md`` — into one new
directory. It never trains, predicts, settles, runs M12, touches the network,
or reads the formal data/run roots. The only results.csv access happens here,
as an explicit demo-resource export; prediction/audit windows never open it.

Safety contract:
- the target must be a new, not-yet-existing directory; nothing is overwritten;
- creation is refused (before any file is written) when the target already
  exists or lies inside the formal roots, the package/site-packages tree, the
  source checkout, or the demo payload resources;
- every payload file is read and hash-checked *before* the target directory is
  created, so payload problems leave no partial export;
- a failure after creation leaves the real partial directory on disk and raises
  with a message that says the export is incomplete and must not be used;
- the exported bundle is re-verified with ``verify_runtime_bundle`` before the
  final ``demo_manifest.json`` is written and success is announced.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import importlib.resources
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .io import file_sha256  # noqa: F401  (re-exported convenience for scripts)
from .runtime_bundle_validation import verify_runtime_bundle
from .schema import DataValidationError

DEMO_EXPORT_MANIFEST_SCHEMA_VERSION = "b05-demo-export-manifest-v1"
DEMO_PAYLOAD_SCHEMA_VERSION = "b05-demo-payload-v1"

FORMAL_DATA_ROOT = Path(r"D:\Codex\matchquant_data")
FORMAL_RUNS_ROOT = Path(r"D:\Codex\matchquant_runs")

_PAYLOAD_PACKAGE = "matchquant.resources"
_PAYLOAD_DIR = "demo"
_PAYLOAD_INDEX = "demo_payload.json"


def _is_within(child: Path, parent: Path) -> bool:
    """True when ``child`` is ``parent`` itself or lies anywhere beneath it."""

    child_text = str(child).lower()
    parent_text = str(parent).lower().rstrip("\\/")
    if child_text == parent_text:
        return True
    return child_text.startswith(parent_text + "\\") or child_text.startswith(parent_text + "/")


def _protected_roots() -> list[tuple[Path, str]]:
    package_dir = Path(__file__).resolve().parent
    payload_dir = package_dir / "resources" / _PAYLOAD_DIR
    return [
        (FORMAL_DATA_ROOT, "the formal data root"),
        (FORMAL_RUNS_ROOT, "the formal runs root"),
        (package_dir, "the matchquant package directory (installed or source)"),
        (package_dir.parent, "the package tree (site-packages or the source checkout)"),
        (payload_dir, "the packaged demo payload resources"),
    ]


def _load_payload() -> list[dict[str, Any]]:
    index = importlib.resources.files(_PAYLOAD_PACKAGE) / _PAYLOAD_DIR / _PAYLOAD_INDEX
    if not index.is_file():
        raise FileNotFoundError(
            f"the demo payload index is missing from the installed package resources: "
            f"{_PAYLOAD_DIR}/{_PAYLOAD_INDEX}"
        )
    payload = json.loads(index.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema_version") != DEMO_PAYLOAD_SCHEMA_VERSION:
        raise DataValidationError(
            "the demo payload index has an unexpected schema; the installed package is inconsistent"
        )
    files = payload.get("files")
    if not isinstance(files, list) or not files:
        raise DataValidationError(
            "the demo payload index lists no files; the installed package is inconsistent"
        )
    return files


def _read_payload_bytes(files: list[dict[str, Any]]) -> list[tuple[str, bytes]]:
    """Read and hash-check every payload file *before* the target is created."""

    demo_resources = importlib.resources.files(_PAYLOAD_PACKAGE) / _PAYLOAD_DIR
    prepared: list[tuple[str, bytes]] = []
    for item in files:
        relative = str(item["path"])
        expected_sha = str(item["sha256"])
        resource = demo_resources / relative
        if not resource.is_file():
            raise FileNotFoundError(
                f"demo payload file '{relative}' is missing from the installed package resources"
            )
        data = resource.read_bytes()
        actual_sha = hashlib.sha256(data).hexdigest()
        if actual_sha != expected_sha:
            raise DataValidationError(
                f"demo payload file '{relative}' does not match the packaged hash "
                f"(expected {expected_sha}, found {actual_sha}); the installed package is inconsistent"
            )
        prepared.append((relative, data))
    return prepared


def export_synthetic_demo(output_dir: str | Path) -> dict[str, Any]:
    """Export the packaged synthetic demo into the new directory ``output_dir``.

    Returns a structured result describing the export. Raises
    :class:`~matchquant.schema.DataValidationError` (or an OSError) for refused
    targets and export failures; a refused target is never created, and a
    failed export leaves the partial directory in place with an error that
    says it must not be used.
    """

    raw_target = Path(output_dir).expanduser()
    if raw_target.exists() or raw_target.is_symlink():
        raise DataValidationError(
            f"demo-export target already exists: {raw_target}; choose a new directory; "
            "the demo export never overwrites"
        )
    target = raw_target.resolve()
    if target.exists() or target.is_symlink():
        raise DataValidationError(
            f"demo-export target already exists after path resolution: {target}; "
            "choose a new directory; the demo export never overwrites"
        )
    for protected, label in _protected_roots():
        if _is_within(target, protected):
            raise DataValidationError(
                f"demo-export target '{target}' lies inside {label} ({protected}); refusing to create it there"
            )
    parent = target.parent
    if not parent.is_dir():
        raise DataValidationError(
            f"the parent directory does not exist or is not a directory: {parent}; "
            "create the parent directory first, then run demo-export again"
        )

    payload_files = _load_payload()
    prepared = _read_payload_bytes(payload_files)

    try:
        target.mkdir(parents=False)
    except OSError as exc:
        raise DataValidationError(f"could not create demo-export directory '{target}': {exc}") from exc

    written: list[dict[str, Any]] = []
    try:
        for relative, data in prepared:
            destination = target / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(data)
            written.append({"path": relative, "size": len(data), "sha256": hashlib.sha256(data).hexdigest()})
    except OSError as exc:
        raise DataValidationError(
            f"demo export is INCOMPLETE and must not be used: writing a payload file failed ({exc}); "
            f"the partial directory '{target}' is left on disk for inspection only"
        ) from exc

    bundle_manifest_path = target / "bundle" / "bundle_manifest.json"
    try:
        verification = verify_runtime_bundle(bundle_manifest_path)
    except Exception as exc:  # converted below, never swallowed
        raise DataValidationError(
            f"demo export is INCOMPLETE and must not be used: the exported bundle failed verification "
            f"({exc}); the partial directory '{target}' is left on disk for inspection only"
        ) from exc
    if verification["status"] != "ready":
        raise DataValidationError(
            f"demo export is INCOMPLETE and must not be used: the exported bundle did not verify ready "
            f"(status={verification['status']}; reasons={'; '.join(verification['reasons'])}); "
            f"the partial directory '{target}' is left on disk for inspection only"
        )

    manifest = {
        "schema_version": DEMO_EXPORT_MANIFEST_SCHEMA_VERSION,
        "synthetic": True,
        "generated_at_utc": datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z"),
        "package_version": _package_version(),
        "source": f"{_PAYLOAD_PACKAGE}.{_PAYLOAD_DIR} (packaged read-only resources)",
        "model": "frequency",
        "model_note": (
            "frequency is used for tutorial simplicity only; it is not a real primary-model selection"
        ),
        "bundle": {
            "root": str(target / "bundle"),
            "manifest": str(bundle_manifest_path),
            "manifest_sha256": verification["bundle_manifest_file_sha256"],
            "verify_status": verification["status"],
            "snapshot_id": verification["snapshot_id"],
            "file_count": verification["file_count"],
            "actual_file_count": verification["actual_file_count"],
            "total_bytes": verification["total_bytes"],
        },
        "files": sorted(written, key=lambda item: item["path"]),
        "usage": "follow DIR/DEMO.md: runtime-bundle-verify -> snapshot-session -> snapshot-run-verify -> settle",
        "limitations": [
            "all demo content is synthetic teaching material; it is not real league data",
            "the exported probabilities are not calibrated and carry no betting or profitability claim",
            "the bundle has no signature: hashes prove this export's internal consistency only",
        ],
    }
    manifest_path = target / "demo_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    return {
        "demo_dir": str(target),
        "bundle_root": str(target / "bundle"),
        "bundle_manifest": str(bundle_manifest_path),
        "bundle_manifest_sha256": verification["bundle_manifest_file_sha256"],
        "fixtures": str(target / "fixtures.csv"),
        "results": str(target / "results.csv"),
        "demo_doc": str(target / "DEMO.md"),
        "demo_manifest": str(manifest_path),
        "files_exported": len(written),
        "synthetic": True,
        "model": "frequency",
        "bundle_verify_status": verification["status"],
        "snapshot_id": verification["snapshot_id"],
    }


def _package_version() -> str:
    try:
        return importlib.metadata.version("matchquant")
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


__all__ = ["DEMO_EXPORT_MANIFEST_SCHEMA_VERSION", "export_synthetic_demo"]
