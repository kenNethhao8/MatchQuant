"""Skill installer utility for deploying the MatchQuant skill into target directories."""

from __future__ import annotations

import shutil
from pathlib import Path


def install_skill(
    target_dir: str | Path,
    source_dir: str | Path | None = None,
    overwrite: bool = True,
) -> Path:
    """Install MatchQuant skill into target skills directory."""
    if source_dir is None:
        source_dir = Path(__file__).resolve().parent.parent / "skills" / "matchquant"
    src = Path(source_dir)
    if not (src / "SKILL.md").is_file():
        raise FileNotFoundError(f"valid MatchQuant skill not found in source directory: {src}")

    dest = Path(target_dir) / "matchquant"
    if dest.exists() and not overwrite:
        raise FileExistsError(f"destination skill directory already exists: {dest}")

    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        shutil.rmtree(dest)

    shutil.copytree(src, dest)
    return dest


def verify_installed_skill(skill_dir: str | Path) -> dict[str, Any]:
    """Verify an installed skill's structure and manifest."""
    root = Path(skill_dir)
    skill_md = root / "SKILL.md"
    run_adapter = root / "scripts" / "run_adapter.py"
    profile_example = root / "references" / "profile_example.json"

    errors: list[str] = []
    if not skill_md.is_file():
        errors.append("SKILL.md missing")
    if not run_adapter.is_file():
        errors.append("scripts/run_adapter.py missing")
    if not profile_example.is_file():
        errors.append("references/profile_example.json missing")

    has_frontmatter = False
    if skill_md.is_file():
        text = skill_md.read_text(encoding="utf-8")
        if text.startswith("---") and "name: matchquant" in text:
            has_frontmatter = True
        else:
            errors.append("SKILL.md missing valid YAML frontmatter")

    return {
        "status": "ok" if not errors else "error",
        "skill_dir": str(root),
        "skill_md": str(skill_md),
        "has_frontmatter": has_frontmatter,
        "errors": errors,
    }
