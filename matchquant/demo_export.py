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
- the payload index is fully validated BEFORE any payload file is read and
  BEFORE the target directory is created: the export set is fixed to the
  11 packaged payloads, every path must be a real string in strict canonical
  relative form (no absolute/drive/UNC/parent/ADS/backslash forms, no Windows
  case or normalization conflicts, no duplicates), and sizes/hashes must have
  valid types. A tampered index therefore fails without creating anything and
  without reading any payload file;
- a failure after creation (payload copy, bundle verification, or the final
  demo_manifest.json persistence) leaves the real partial directory on disk
  and raises with a message that names the stage, the actual directory, and
  says the export is INCOMPLETE and must not be used; unknown programming
  errors keep their real exception type and chain;
- the exported bundle is re-verified with ``verify_runtime_bundle`` before the
  final ``demo_manifest.json`` is written and success is announced.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import importlib.resources
import json
import re
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath, PureWindowsPath
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

# The export set is fixed: 8 bundle payloads (B04 fixed structure) plus the
# three standalone demo files. The catalog may not add, remove, duplicate, or
# re-point any of them.
_TOP_FIXED_PATHS = ("DEMO.md", "fixtures.csv", "results.csv")
_BUNDLE_FIXED_PATHS = (
    "bundle/RUN.md",
    "bundle/bundle_manifest.json",
    "bundle/team_index.json",
    "bundle/evidence/issue_audit.json",
    "bundle/evidence/original_snapshot_manifest.json",
)
_SNAPSHOT_LEAF_NAMES = ("model_states.json", "snapshot_manifest.json", "snapshot_report.md")
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")

_EXPECTED_EXPORT_ERRORS = (DataValidationError, FileNotFoundError, OSError, TypeError, ValueError)


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


def _validate_relative_payload_path(value: Any) -> str:
    """Validate one catalog path as a strict canonical relative POSIX form.

    Anything that is not a plain string, or that could escape the target or the
    packaged resources (absolute, drive/UNC, parent components, ADS colons,
    backslashes, empty/`.`/`..` components, trailing dots/spaces, or a
    non-canonical spelling) is rejected before anything is read or created.
    """

    if not isinstance(value, str) or not value:
        raise DataValidationError(
            f"demo payload path must be a non-empty string, got {value!r}; "
            "the installed package payload index is inconsistent"
        )
    if "\\" in value:
        raise DataValidationError(
            f"demo payload path must use forward separators only, got {value!r}"
        )
    if ":" in value:
        raise DataValidationError(
            f"demo payload path must not contain drive/ADS colons, got {value!r}"
        )
    windows_view = PureWindowsPath(value)
    if windows_view.is_absolute() or windows_view.drive:
        raise DataValidationError(
            f"demo payload path must be relative without drive/UNC prefix, got {value!r}"
        )
    parts = value.split("/")
    for part in parts:
        if part in ("", ".", ".."):
            raise DataValidationError(
                f"demo payload path must be a canonical relative path without empty, '.', "
                f"or '..' components, got {value!r}"
            )
        if part != part.strip() or part.endswith("."):
            raise DataValidationError(
                f"demo payload path components must not have trailing dots/spaces, got {value!r}"
            )
    if PurePosixPath(value).as_posix() == value:
        return value
    raise DataValidationError(
        f"demo payload path is not in canonical relative form, got {value!r}"
    )


def _validated_snapshot_leaf(path: str) -> tuple[str, str]:
    parts = path.split("/")
    if len(parts) != 4:
        raise DataValidationError(
            f"snapshot payload path must be bundle/snapshot/<snapshot_id>/<leaf>, got {path!r}"
        )
    if parts[0] != "bundle" or parts[1] != "snapshot":
        raise DataValidationError(
            f"snapshot payload path must start with bundle/snapshot/, got {path!r}"
        )
    leaf = parts[3]
    if leaf not in _SNAPSHOT_LEAF_NAMES:
        raise DataValidationError(
            f"snapshot payload leaf must be one of {', '.join(_SNAPSHOT_LEAF_NAMES)}, got {path!r}"
        )
    return parts[2], leaf


def _is_snapshot_payload_path(path: str) -> bool:
    parts = path.split("/")
    return (
        len(parts) == 4
        and parts[0] == "bundle"
        and parts[1] == "snapshot"
        and parts[2] != ""
        and parts[3] in _SNAPSHOT_LEAF_NAMES
    )


def _expected_export_paths() -> set[str]:
    return set(_TOP_FIXED_PATHS) | set(_BUNDLE_FIXED_PATHS)


def _validate_payload_catalog(files: Any) -> list[dict[str, Any]]:
    """Validate the whole export catalog: types, paths, and the fixed 11-file set."""

    if not isinstance(files, list) or not files:
        raise DataValidationError(
            "demo payload catalog must be a non-empty list; the installed package is inconsistent"
        )
    seen_paths: set[str] = set()
    seen_casefold: dict[str, str] = {}
    validated: list[dict[str, Any]] = []
    for item in files:
        if not isinstance(item, dict):
            raise DataValidationError(
                f"demo payload entry must be an object, got {type(item).__name__}"
            )
        path = _validate_relative_payload_path(item.get("path"))
        if path in seen_paths:
            raise DataValidationError(f"demo payload catalog contains a duplicate path: {path!r}")
        casefolded = path.casefold()
        if casefolded in seen_casefold:
            raise DataValidationError(
                f"demo payload catalog contains Windows case-insensitive duplicates: "
                f"{seen_casefold[casefolded]!r} and {path!r}"
            )
        seen_paths.add(path)
        seen_casefold[casefolded] = path
        size = item.get("size")
        if isinstance(size, bool) or not isinstance(size, int) or size < 0:
            raise DataValidationError(
                f"demo payload size must be a non-negative integer, got {size!r} for {path!r}"
            )
        sha256_value = item.get("sha256")
        if not isinstance(sha256_value, str) or not _SHA256_PATTERN.fullmatch(sha256_value):
            raise DataValidationError(
                f"demo payload sha256 must be a 64-hex-character string, got {sha256_value!r} for {path!r}"
            )
        validated.append({"path": path, "size": size, "sha256": sha256_value})

    expected_fixed = _expected_export_paths()
    catalog = set(seen_paths)
    missing = sorted(expected_fixed - catalog)
    if missing:
        raise DataValidationError(
            "demo payload catalog is missing required export entries: " + ", ".join(missing)
        )
    snapshot_paths = sorted(catalog - expected_fixed)
    malformed = [path for path in snapshot_paths if not _is_snapshot_payload_path(path)]
    if malformed:
        raise DataValidationError(
            "demo payload catalog lists entries outside the fixed export set: "
            + ", ".join(malformed)
        )
    if len(snapshot_paths) != len(_SNAPSHOT_LEAF_NAMES):
        raise DataValidationError(
            f"demo payload catalog must contain exactly {len(_SNAPSHOT_LEAF_NAMES)} snapshot "
            f"payload entries, found {len(snapshot_paths)}"
        )
    snapshot_ids: set[str] = set()
    snapshot_leaves: dict[str, set[str]] = {}
    for path in snapshot_paths:
        snapshot_id, leaf = _validated_snapshot_leaf(path)
        snapshot_ids.add(snapshot_id)
        snapshot_leaves.setdefault(snapshot_id, set()).add(leaf)
    if len(snapshot_ids) != 1:
        raise DataValidationError(
            "demo payload catalog must contain exactly one snapshot directory, found: "
            + ", ".join(sorted(snapshot_ids) or ["none"])
        )
    (snapshot_id,) = tuple(snapshot_ids)
    if snapshot_leaves[snapshot_id] != set(_SNAPSHOT_LEAF_NAMES):
        raise DataValidationError(
            f"snapshot payload directory {snapshot_id!r} must contain exactly "
            f"{', '.join(_SNAPSHOT_LEAF_NAMES)}"
        )
    return validated


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
    return _validate_payload_catalog(payload.get("files"))


def _read_payload_bytes(files: list[dict[str, Any]]) -> list[tuple[str, bytes]]:
    """Read and hash-check every payload file *before* the target is created.

    Paths were already validated to strict canonical relative form, so each
    source stays inside the packaged demo resources; the containment check is
    repeated after resolution as defence in depth.
    """

    demo_resources = importlib.resources.files(_PAYLOAD_PACKAGE) / _PAYLOAD_DIR
    resources_root = Path(str(demo_resources))
    prepared: list[tuple[str, bytes]] = []
    for item in files:
        relative = str(item["path"])
        expected_sha = str(item["sha256"])
        resource = demo_resources / relative
        if not resource.is_file():
            raise FileNotFoundError(
                f"demo payload file '{relative}' is missing from the installed package resources"
            )
        source_path = Path(str(resource))
        if resources_root and not _is_within(source_path, resources_root):
            raise DataValidationError(
                f"demo payload source for '{relative}' resolves outside the packaged resources: {source_path}"
            )
        data = resource.read_bytes()
        actual_sha = hashlib.sha256(data).hexdigest()
        if len(data) != item["size"] or actual_sha != expected_sha:
            raise DataValidationError(
                f"demo payload file '{relative}' does not match the packaged size/hash "
                f"(expected {item['size']} bytes / {expected_sha}, found {len(data)} bytes / {actual_sha}); "
                "the installed package is inconsistent"
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

    # Full catalog validation and payload reads happen before the target exists.
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
            if not _is_within(destination.resolve(), target):
                raise DataValidationError(
                    f"refusing to write outside the export target: '{relative}' resolves to "
                    f"{destination.resolve()}"
                )
            destination.write_bytes(data)
            written.append({"path": relative, "size": len(data), "sha256": hashlib.sha256(data).hexdigest()})
    except _EXPECTED_EXPORT_ERRORS as exc:
        raise DataValidationError(
            f"demo export is INCOMPLETE and must not be used: writing payload files failed "
            f"({exc}); the partial directory '{target}' is left on disk for inspection only"
        ) from exc

    bundle_manifest_path = target / "bundle" / "bundle_manifest.json"
    try:
        verification = verify_runtime_bundle(bundle_manifest_path)
    except _EXPECTED_EXPORT_ERRORS as exc:
        raise DataValidationError(
            f"demo export is INCOMPLETE and must not be used: the exported bundle failed verification "
            f"({exc}); the partial directory '{target}' is left on disk for inspection only"
        ) from exc
    except Exception as exc:
        # Unknown programming errors are not misclassified as validation
        # failures; the partial-export context is attached while the real
        # exception type and chain stay intact.
        exc.add_note(
            f"demo export is INCOMPLETE and must not be used: an unexpected error occurred after "
            f"writing payload files; the partial directory '{target}' is left on disk for inspection only"
        )
        raise
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
    try:
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    except OSError as exc:
        raise DataValidationError(
            f"demo export is INCOMPLETE and must not be used: the final demo_manifest.json could not be "
            f"written ({exc}); the partial directory '{target}' is left on disk for inspection only. "
            "The exported payload files (including the bundle) may be intact, but the demo delivery is "
            "not complete and must not be used as-is."
        ) from exc

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
