"""Layer 0: User profile management and environment diagnostics."""

from __future__ import annotations

import json
import os
import shutil
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping

from .epl import normalize_season

PROFILE_SCHEMA_VERSION = "matchquant-profile-v1"
DEFAULT_PROFILE_DIR = Path.home() / ".matchquant"
DEFAULT_PROFILE_PATH = DEFAULT_PROFILE_DIR / "profile.json"

DEFAULT_TEAM_ALIASES: dict[str, str] = {
    # Current & frequent EPL teams
    "阿森纳": "Arsenal",
    "阿斯顿维拉": "Aston Villa",
    "维拉": "Aston Villa",
    "伯恩茅斯": "Bournemouth",
    "布伦特福德": "Brentford",
    "布莱顿": "Brighton",
    "切尔西": "Chelsea",
    "水晶宫": "Crystal Palace",
    "埃弗顿": "Everton",
    "富勒姆": "Fulham",
    "伊普斯维奇": "Ipswich",
    "莱斯特城": "Leicester",
    "莱斯特": "Leicester",
    "利物浦": "Liverpool",
    "曼城": "Man City",
    "曼彻斯特城": "Man City",
    "曼联": "Man United",
    "曼彻斯特联": "Man United",
    "纽卡斯尔联": "Newcastle",
    "纽卡斯尔": "Newcastle",
    "纽卡": "Newcastle",
    "诺丁汉森林": "Nottingham Forest",
    "森林": "Nottingham Forest",
    "南安普敦": "Southampton",
    "热刺": "Tottenham",
    "托特纳姆热刺": "Tottenham",
    "西汉姆联": "West Ham",
    "西汉姆": "West Ham",
    "狼队": "Wolves",
    # Historical / recent relegated teams
    "卢顿": "Luton",
    "伯恩利": "Burnley",
    "谢菲尔德联": "Sheffield United",
    "谢菲联": "Sheffield United",
    "利兹联": "Leeds",
    "利兹": "Leeds",
    "沃特福德": "Watford",
    "诺维奇": "Norwich",
    "西布罗姆维奇": "West Brom",
    "西布朗": "West Brom",
}


@dataclass
class RefreshPolicy:
    auto_refresh: bool = False
    source_check_interval_hours: int = 24
    allow_stale_fallback: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "auto_refresh": bool(self.auto_refresh),
            "source_check_interval_hours": int(self.source_check_interval_hours),
            "allow_stale_fallback": bool(self.allow_stale_fallback),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | None) -> RefreshPolicy:
        if not data:
            return cls()
        return cls(
            auto_refresh=bool(data.get("auto_refresh", False)),
            source_check_interval_hours=int(data.get("source_check_interval_hours", 24)),
            allow_stale_fallback=bool(data.get("allow_stale_fallback", False)),
        )


@dataclass
class ReportPreferences:
    language: str = "zh-CN"
    style: str = "standard"  # standard, concise, deep
    default_model_profile: str = "dual_track_v2"

    def to_dict(self) -> dict[str, Any]:
        return {
            "language": str(self.language),
            "style": str(self.style),
            "default_model_profile": str(self.default_model_profile),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | None) -> ReportPreferences:
        if not data:
            return cls()
        return cls(
            language=str(data.get("language", "zh-CN")),
            style=str(data.get("style", "standard")),
            default_model_profile=str(data.get("default_model_profile", "dual_track_v2")),
        )


@dataclass
class UserProfile:
    schema_version: str = PROFILE_SCHEMA_VERSION
    data_root: str = r"D:\Codex\matchquant_data"
    report_root: str = r"D:\Codex\matchquant_runs"
    python_executable: str = field(default_factory=lambda: sys.executable)
    provider: str = "football_data_epl"
    competition: str = "English Premier League"
    historical_start_season: str = "2015-16"
    in_progress_season: str | None = None
    team_aliases: dict[str, str] = field(default_factory=lambda: dict(DEFAULT_TEAM_ALIASES))
    refresh_policy: RefreshPolicy = field(default_factory=RefreshPolicy)
    timezone: str = "Asia/Shanghai"
    report_preferences: ReportPreferences = field(default_factory=ReportPreferences)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "data_root": str(self.data_root),
            "report_root": str(self.report_root),
            "python_executable": str(self.python_executable),
            "provider": str(self.provider),
            "competition": str(self.competition),
            "historical_start_season": str(self.historical_start_season),
            "in_progress_season": self.in_progress_season,
            "team_aliases": dict(self.team_aliases),
            "refresh_policy": self.refresh_policy.to_dict(),
            "timezone": str(self.timezone),
            "report_preferences": self.report_preferences.to_dict(),
        }

    def save(self, path: str | Path | None = None) -> Path:
        target = Path(path) if path else get_default_profile_path()
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(self.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
        return target

    def resolve_team_alias(self, name: str) -> str:
        """Resolve a team name or Chinese alias to standard canonical name."""
        cleaned = " ".join(str(name).strip().split())
        if not cleaned:
            return cleaned
        # Direct match in aliases
        if cleaned in self.team_aliases:
            return self.team_aliases[cleaned]
        # Case-insensitive match in aliases
        lower = cleaned.casefold()
        for alias, target in self.team_aliases.items():
            if alias.casefold() == lower:
                return target
        # Return as-is if no alias matches
        return cleaned

    def validate(self) -> list[str]:
        """Validate profile settings and return list of validation errors."""
        errors: list[str] = []
        if not self.data_root:
            errors.append("data_root cannot be empty")
        if not self.report_root:
            errors.append("report_root cannot be empty")
        try:
            normalize_season(self.historical_start_season)
        except Exception as exc:
            errors.append(f"invalid historical_start_season: {exc}")
        if self.in_progress_season:
            try:
                normalize_season(self.in_progress_season)
            except Exception as exc:
                errors.append(f"invalid in_progress_season: {exc}")
        if self.refresh_policy.source_check_interval_hours <= 0:
            errors.append("source_check_interval_hours must be positive")
        return errors


def get_default_profile_path() -> Path:
    env_path = os.environ.get("MATCHQUANT_PROFILE_PATH")
    if env_path:
        return Path(env_path)
    return DEFAULT_PROFILE_PATH


def load_profile(path: str | Path | None = None) -> UserProfile:
    """Load profile from path or default location, returning default if not found."""
    target = Path(path) if path else get_default_profile_path()
    if not target.exists():
        return UserProfile()
    try:
        data = json.loads(target.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ValueError(f"Failed to parse profile JSON at '{target}': {exc}") from exc

    if not isinstance(data, dict):
        raise ValueError(f"Profile JSON at '{target}' must be an object")

    hist_start = data.get("historical_start_season", "2015-16")
    in_prog = data.get("in_progress_season")
    aliases = dict(DEFAULT_TEAM_ALIASES)
    if isinstance(data.get("team_aliases"), dict):
        aliases.update({str(k): str(v) for k, v in data["team_aliases"].items()})

    refresh = RefreshPolicy.from_dict(data.get("refresh_policy"))
    report_prefs = ReportPreferences.from_dict(data.get("report_preferences"))

    return UserProfile(
        schema_version=data.get("schema_version", PROFILE_SCHEMA_VERSION),
        data_root=str(data.get("data_root", r"D:\Codex\matchquant_data")),
        report_root=str(data.get("report_root", r"D:\Codex\matchquant_runs")),
        python_executable=str(data.get("python_executable", sys.executable)),
        provider=str(data.get("provider", "football_data_epl")),
        competition=str(data.get("competition", "English Premier League")),
        historical_start_season=hist_start,
        in_progress_season=in_prog,
        team_aliases=aliases,
        refresh_policy=refresh,
        timezone=str(data.get("timezone", "Asia/Shanghai")),
        report_preferences=report_prefs,
    )


def diagnose_environment(profile: UserProfile | None = None) -> dict[str, Any]:
    """Diagnose local execution environment, paths, dependencies, and configuration."""
    prof = profile or load_profile()
    errors = prof.validate()
    warnings: list[str] = []

    # Check python interpreter
    python_path = Path(prof.python_executable)
    python_ok = python_path.is_file() or shutil.which(prof.python_executable) is not None
    if not python_ok:
        errors.append(f"python_executable does not exist or not found: {prof.python_executable}")

    # Check dependencies in current environment
    pkg_status: dict[str, str] = {}
    for pkg in ("numpy", "pandas", "scipy"):
        try:
            mod = __import__(pkg)
            pkg_status[pkg] = getattr(mod, "__version__", "installed")
        except ImportError:
            pkg_status[pkg] = "missing"
            errors.append(f"required python package missing: {pkg}")

    # Check paths
    data_root = Path(prof.data_root)
    report_root = Path(prof.report_root)

    data_root_writable = False
    try:
        data_root.mkdir(parents=True, exist_ok=True)
        test_file = data_root / ".write_test"
        test_file.write_text("ok", encoding="utf-8")
        test_file.unlink(missing_ok=True)
        data_root_writable = True
    except Exception as exc:
        errors.append(f"data_root '{data_root}' is not writable: {exc}")

    report_root_writable = False
    try:
        report_root.mkdir(parents=True, exist_ok=True)
        test_file = report_root / ".write_test"
        test_file.write_text("ok", encoding="utf-8")
        test_file.unlink(missing_ok=True)
        report_root_writable = True
    except Exception as exc:
        errors.append(f"report_root '{report_root}' is not writable: {exc}")

    # Check current dataset status
    current_pointer = data_root / "processed" / "epl_current.json"
    dataset_ready = current_pointer.is_file()
    dataset_info: dict[str, Any] = {}
    if dataset_ready:
        try:
            dataset_info = json.loads(current_pointer.read_text(encoding="utf-8"))
        except Exception:
            warnings.append("epl_current.json exists but could not be parsed")
    else:
        warnings.append("no prepared dataset found in data_root; bootstrap sync required")

    return {
        "status": "ok" if not errors else "error",
        "profile_schema_version": prof.schema_version,
        "python_executable": prof.python_executable,
        "python_executable_valid": python_ok,
        "python_version": sys.version.split()[0],
        "packages": pkg_status,
        "data_root": str(data_root),
        "data_root_writable": data_root_writable,
        "report_root": str(report_root),
        "report_root_writable": report_root_writable,
        "dataset_ready": dataset_ready,
        "dataset_info": dataset_info,
        "refresh_policy": prof.refresh_policy.to_dict(),
        "errors": errors,
        "warnings": warnings,
    }
