"""Layer 4: Structured, machine-readable Agent Adapter for host Agent orchestration."""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Mapping

from .agent_report import (
    answer_followup,
    build_evidence_catalog,
    render_agent_narrative_html,
    render_agent_narrative_markdown,
)
from .dataset import check_data_readiness, load_current_dataset_pointer
from .epl import normalize_season
from .io import write_json
from .odds_provider import OddsProviderError, fetch_epl_odds, snapshot_to_market_input
from .profile import UserProfile, diagnose_environment, load_profile
from .single_match import SingleMatchError, parse_iso_date, run_single_match_report
from .sync import sync_epl

ADAPTER_SCHEMA_VERSION = "matchquant-adapter-v1"


def execute_adapter_request(request: Mapping[str, Any]) -> dict[str, Any]:
    """Execute a structured request from the host Agent and return a structured response."""
    action = str(request.get("action", "analyze")).lower()
    profile_path = request.get("profile_path")
    prof = load_profile(profile_path)

    base_response: dict[str, Any] = {
        "schema_version": ADAPTER_SCHEMA_VERSION,
        "action": action,
        "status": "failed",
        "timestamp": datetime.now().isoformat(),
    }

    try:
        if action == "setup":
            return _handle_setup(request, prof, base_response)
        elif action == "sync":
            return _handle_sync(request, prof, base_response)
        elif action == "readiness":
            return _handle_readiness(request, prof, base_response)
        elif action == "analyze":
            return _handle_analyze(request, prof, base_response)
        elif action == "followup":
            return _handle_followup(request, prof, base_response)
        elif action == "odds":
            return _handle_odds(request, prof, base_response)
        else:
            base_response.update(
                status="failed",
                error=f"unknown adapter action: {action!r}. Supported actions: setup, sync, readiness, analyze, followup, odds",
            )
            return base_response
    except Exception as exc:
        base_response.update(
            status="failed",
            error=str(exc),
            error_type=exc.__class__.__name__,
        )
        return base_response


def _handle_setup(
    request: Mapping[str, Any], prof: UserProfile, response: dict[str, Any]
) -> dict[str, Any]:
    """Handle setup / environment diagnosis action."""
    # Check if updates were supplied
    updates = request.get("updates")
    if isinstance(updates, dict):
        if "data_root" in updates:
            prof.data_root = str(updates["data_root"])
        if "report_root" in updates:
            prof.report_root = str(updates["report_root"])
        if "auto_refresh" in updates:
            prof.refresh_policy.auto_refresh = bool(updates["auto_refresh"])
        if "allow_stale_fallback" in updates:
            prof.refresh_policy.allow_stale_fallback = bool(updates["allow_stale_fallback"])
        if "team_aliases" in updates and isinstance(updates["team_aliases"], dict):
            prof.team_aliases.update({str(k): str(v) for k, v in updates["team_aliases"].items()})
        if "in_progress_season" in updates:
            prof.in_progress_season = str(updates["in_progress_season"]) if updates["in_progress_season"] else None
        saved_path = prof.save(request.get("profile_path"))
        response["saved_profile_path"] = str(saved_path)

    diag = diagnose_environment(prof)
    response.update(
        status="ok" if diag["status"] == "ok" else "needs_review",
        diagnostics=diag,
        profile=prof.to_dict(),
    )
    return response


def _handle_sync(
    request: Mapping[str, Any], prof: UserProfile, response: dict[str, Any]
) -> dict[str, Any]:
    """Handle on-demand sync request."""
    data_root = request.get("data_root") or prof.data_root
    seasons = request.get("seasons")
    in_progress = request.get("in_progress_season") or prof.in_progress_season
    dry_run = bool(request.get("dry_run", True))
    apply_flag = bool(request.get("apply", False))
    timeout = float(request.get("timeout", 30.0))

    if apply_flag:
        dry_run = False

    sync_res = sync_epl(
        data_root=data_root,
        seasons=seasons,
        in_progress_season=in_progress,
        dry_run=dry_run,
        apply=apply_flag,
        timeout=timeout,
        profile=prof,
    )

    response.update(
        status=sync_res["status"],
        sync_result=sync_res,
    )
    return response


def _handle_readiness(
    request: Mapping[str, Any], prof: UserProfile, response: dict[str, Any]
) -> dict[str, Any]:
    """Handle data readiness audit action."""
    data_root = request.get("data_root") or prof.data_root
    as_of = request.get("as_of")
    fixture_date = request.get("fixture_date") or request.get("date")
    allow_stale = request.get("allow_stale_fallback")

    # Resolve aliases for home and away if provided
    home = request.get("home") or request.get("home_team")
    away = request.get("away") or request.get("away_team")
    resolved_home = prof.resolve_team_alias(home) if home else None
    resolved_away = prof.resolve_team_alias(away) if away else None

    readiness = check_data_readiness(
        data_root=data_root,
        as_of=as_of,
        fixture_date=fixture_date,
        profile=prof,
        allow_stale_fallback=allow_stale,
    )

    response.update(
        status=readiness["status"],
        ready=readiness["ready"],
        readiness=readiness,
        normalized_fixture={
            "requested_home": home,
            "resolved_home": resolved_home,
            "requested_away": away,
            "resolved_away": resolved_away,
            "as_of": as_of,
            "fixture_date": fixture_date,
        },
    )
    return response


def _handle_analyze(
    request: Mapping[str, Any], prof: UserProfile, response: dict[str, Any]
) -> dict[str, Any]:
    """Execute end-to-end single match analysis and artifact generation."""
    data_root = Path(request.get("data_root") or prof.data_root)
    report_root = Path(request.get("report_root") or prof.report_root)

    # Resolve teams and aliases
    raw_home = request.get("home") or request.get("home_team")
    raw_away = request.get("away") or request.get("away_team")
    if not raw_home or not raw_away:
        raise ValueError("both home and away teams are required for analysis")

    home = prof.resolve_team_alias(str(raw_home))
    away = prof.resolve_team_alias(str(raw_away))

    fixture_date = str(request.get("date") or request.get("fixture_date") or "").strip()
    if not fixture_date:
        raise ValueError("fixture date is required (YYYY-MM-DD)")

    season = request.get("season")
    if not season:
        # Determine season from fixture date if standard YYYY-MM-DD
        try:
            d = datetime.strptime(fixture_date, "%Y-%m-%d").date()
            start_y = d.year if d.month >= 7 else d.year - 1
            season = f"{start_y}-{(start_y + 1) % 100:02d}"
        except Exception:
            raise ValueError("could not infer season from fixture date; please specify season explicitly")
    norm_season = normalize_season(season)

    user_specified_as_of = bool(request.get("as_of"))
    as_of = str(request.get("as_of") or fixture_date).strip()
    mode = str(request.get("mode", "research"))

    # Live odds integration
    use_live_odds = bool(request.get("use_live_odds", False))
    market_input_path = request.get("market_input_path")
    live_odds_info: dict[str, Any] | None = None
    adapter_warnings: list[str] = []

    if use_live_odds:
        try:
            snapshot = fetch_epl_odds(
                home_team=home,
                away_team=away,
                fixture_date=fixture_date,
                api_key=request.get("api_key"),
                profile=prof,
            )
            market_input = snapshot_to_market_input(snapshot)
            snapshot_path = Path(snapshot["snapshot_path"])
            market_input_file = snapshot_path.parent / "market_input.json"
            market_input_file.write_text(
                json.dumps(market_input, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )

            observed_at_str = str(snapshot["observed_at"])
            observed_date = parse_iso_date(observed_at_str, "snapshot observed_at")

            if not user_specified_as_of:
                as_of = (observed_date + timedelta(days=1)).isoformat()
                adapter_warnings.append(
                    f"未显式指定 as_of，为满足市场盘口 observed_at < as_of 无穿越约束，默认将 as_of 设为盘口观察日次日 ({as_of})"
                )
                market_input_path = str(market_input_file.resolve())
            else:
                as_of_date = parse_iso_date(as_of, "as_of")
                if observed_date >= as_of_date:
                    adapter_warnings.append(
                        f"实时盘口观察时间 ({observed_at_str}) 不早于指定的 as_of ({as_of})，触发防穿越红线，已优雅降级为纯基本面 Track A 模式"
                    )
                    market_input_path = None
                else:
                    market_input_path = str(market_input_file.resolve())

            live_odds_info = {
                "status": "ok" if market_input_path else "downgraded",
                "snapshot_path": snapshot.get("snapshot_path"),
                "bookmaker": snapshot.get("bookmaker_title"),
                "observed_at_utc": snapshot.get("observed_at_utc"),
                "odds": {
                    "home": snapshot.get("odds_open_h"),
                    "draw": snapshot.get("odds_open_d"),
                    "away": snapshot.get("odds_open_a"),
                },
                "mkt_ah_line": snapshot.get("mkt_ah_line"),
            }
        except Exception as exc:
            adapter_warnings.append(
                f"实时盘口获取失败: {exc}，已优雅降级为纯基本面 Track A 模式"
            )
            live_odds_info = {
                "status": "failed",
                "error": str(exc),
                "error_type": exc.__class__.__name__,
            }

    # Check readiness
    auto_refresh = bool(request.get("auto_refresh", prof.refresh_policy.auto_refresh))
    allow_stale = bool(request.get("allow_stale_fallback", prof.refresh_policy.allow_stale_fallback))
    readiness = check_data_readiness(
        data_root=data_root,
        as_of=as_of,
        fixture_date=fixture_date,
        profile=prof,
        allow_stale_fallback=allow_stale,
    )

    sync_record: dict[str, Any] | None = None
    if not readiness["ready"] or readiness.get("is_stale"):
        if auto_refresh:
            # Trigger routine addition-only sync
            sync_record = sync_epl(
                data_root=data_root,
                in_progress_season=prof.in_progress_season,
                dry_run=False,
                apply=True,
                profile=prof,
            )
            # Re-check readiness after sync
            readiness = check_data_readiness(
                data_root=data_root,
                as_of=as_of,
                fixture_date=fixture_date,
                profile=prof,
                allow_stale_fallback=allow_stale,
            )

    if not readiness["ready"]:
        response.update(
            status="failed",
            error=f"data readiness check failed: {'; '.join(readiness.get('errors', []))}",
            readiness=readiness,
            sync_record=sync_record,
        )
        return response

    rich_csv_path = Path(readiness["rich_path"])
    if not rich_csv_path.is_file():
        raise FileNotFoundError(f"rich dataset not found at '{rich_csv_path}'")

    # Run single match prediction
    divergence_threshold = request.get("divergence_threshold_pct_points")
    max_data_age = request.get("max_data_age_days")
    extra_single_match_kwargs: dict[str, Any] = {}
    if divergence_threshold is not None:
        try:
            extra_single_match_kwargs["divergence_threshold_pct_points"] = float(divergence_threshold)
        except (ValueError, TypeError):
            pass
    if max_data_age is not None:
        try:
            extra_single_match_kwargs["max_data_age_days"] = int(max_data_age)
        except (ValueError, TypeError):
            pass

    report_json, base_artifacts = run_single_match_report(
        input_path=rich_csv_path,
        as_of=as_of,
        output_root=report_root,
        mode=mode,
        home=home,
        away=away,
        fixture_date=fixture_date,
        season=norm_season,
        market_input_path=market_input_path,
        snapshot_root=report_root / "odds_snapshots",
        **extra_single_match_kwargs,
    )

    run_dir = Path(base_artifacts["json"]).parent
    run_id = report_json.get("run_id") or run_dir.name
    report_json["run_id"] = run_id

    # Render Layer 5 Agent companion narrative reports
    catalog = build_evidence_catalog(report_json)
    agent_md = render_agent_narrative_markdown(report_json, catalog)
    agent_html = render_agent_narrative_html(report_json, catalog)

    agent_md_path = run_dir / "agent_report.md"
    agent_html_path = run_dir / "agent_report.html"
    agent_md_path.write_text(agent_md, encoding="utf-8")
    agent_html_path.write_text(agent_html, encoding="utf-8")

    all_artifacts = {
        "report_json": str(base_artifacts["json"]),
        "baseline_markdown": str(base_artifacts["markdown"]),
        "baseline_html": str(base_artifacts["html"]),
        "agent_markdown": str(agent_md_path),
        "agent_html": str(agent_html_path),
    }

    final_status = "ok" if report_json.get("status") in ("ok", "complete") else "needs_review"
    response.update(
        status=final_status,
        run_id=run_id,
        run_dir=str(run_dir),
        fixture={
            "home": home,
            "away": away,
            "requested_home": raw_home,
            "requested_away": raw_away,
            "date": fixture_date,
            "season": norm_season,
        },
        temporal={
            "as_of": as_of,
            "used_data_cutoff": report_json.get("temporal", {}).get("used_data_cutoff"),
            "data_age_days": report_json.get("temporal", {}).get("data_age_days_at_as_of"),
            "forecast_horizon_days": report_json.get("temporal", {}).get("forecast_horizon_days"),
            "mode": mode,
        },
        predictions={
            "track_a": report_json.get("predictions", {}).get("track_a", {}),
            "expected_goals": report_json.get("predictions", {}).get("expected_goals", {}),
            "probability_total_goals_over_2_5": report_json.get("predictions", {}).get("probability_total_goals_over_2_5"),
            "p_home": report_json.get("predictions", {}).get("track_a", {}).get("home"),
            "p_draw": report_json.get("predictions", {}).get("track_a", {}).get("draw"),
            "p_away": report_json.get("predictions", {}).get("track_a", {}).get("away"),
        },
        artifacts=all_artifacts,
        market_comparison=report_json.get("market_comparison"),
        readiness_summary={
            "dataset_version": readiness.get("dataset_version"),
            "latest_completed_match": readiness.get("latest_completed_match_date"),
            "data_age_days": readiness.get("data_age_days"),
        },
        sync_record=sync_record,
        warnings=list(report_json.get("warnings", [])) + adapter_warnings,
    )
    if live_odds_info is not None:
        response["live_odds"] = live_odds_info
    return response


def _handle_odds(
    request: Mapping[str, Any], prof: UserProfile, response: dict[str, Any]
) -> dict[str, Any]:
    """Handle standalone odds fetching and snapshot persistence."""
    raw_home = request.get("home") or request.get("home_team")
    raw_away = request.get("away") or request.get("away_team")
    if not raw_home or not raw_away:
        raise ValueError("both home and away teams are required for odds fetching")

    home = prof.resolve_team_alias(str(raw_home))
    away = prof.resolve_team_alias(str(raw_away))

    fixture_date = str(request.get("date") or request.get("fixture_date") or "").strip()
    if not fixture_date:
        raise ValueError("fixture date is required (YYYY-MM-DD)")

    api_key = request.get("api_key")
    regions = str(request.get("regions", "uk"))
    markets = str(request.get("markets", "h2h,spreads"))
    output_dir = request.get("output_dir")

    snapshot = fetch_epl_odds(
        home_team=home,
        away_team=away,
        fixture_date=fixture_date,
        api_key=api_key,
        regions=regions,
        markets=markets,
        output_dir=output_dir,
        profile=prof,
    )

    market_input = snapshot_to_market_input(snapshot)
    snapshot_path = Path(snapshot["snapshot_path"])
    market_input_file = snapshot_path.parent / "market_input.json"
    market_input_file.write_text(
        json.dumps(market_input, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    response.update(
        status="ok",
        fixture={
            "home": home,
            "away": away,
            "requested_home": raw_home,
            "requested_away": raw_away,
            "date": fixture_date,
        },
        snapshot=snapshot,
        snapshot_path=snapshot.get("snapshot_path"),
        market_input=market_input,
        market_input_path=str(market_input_file.resolve()),
        summary={
            "bookmaker": snapshot.get("bookmaker_title"),
            "observed_at_utc": snapshot.get("observed_at_utc"),
            "odds": {
                "home": snapshot.get("odds_open_h"),
                "draw": snapshot.get("odds_open_d"),
                "away": snapshot.get("odds_open_a"),
            },
            "mkt_ah_line": snapshot.get("mkt_ah_line"),
        },
    )
    return response


def _handle_followup(
    request: Mapping[str, Any], prof: UserProfile, response: dict[str, Any]
) -> dict[str, Any]:
    """Handle follow-up questions reusing existing run JSON without recomputing."""
    run_dir_or_json = request.get("run_dir") or request.get("report_json") or request.get("run_id")
    question = str(request.get("question", "")).strip()

    if not question:
        raise ValueError("question cannot be empty")
    if not run_dir_or_json:
        raise ValueError("run_id, run_dir, or report_json path is required for follow-up")

    target_json_path: Path | None = None
    candidate = Path(str(run_dir_or_json))
    if candidate.is_file() and candidate.name.endswith(".json"):
        target_json_path = candidate
    elif candidate.is_dir():
        for fname in ("match_report.json", "report.json"):
            if (candidate / fname).is_file():
                target_json_path = candidate / fname
                break
    else:
        # Search under report_root
        report_root = Path(prof.report_root)
        check_dir = report_root / str(run_dir_or_json)
        for fname in ("match_report.json", "report.json"):
            if (check_dir / fname).is_file():
                target_json_path = check_dir / fname
                break

    if not target_json_path or not target_json_path.is_file():
        raise FileNotFoundError(f"could not locate match_report.json for run reference '{run_dir_or_json}'")

    report_json = json.loads(target_json_path.read_text(encoding="utf-8"))
    followup_res = answer_followup(report_json, question)

    response.update(
        status="ok",
        run_id=report_json.get("run_id") or Path(target_json_path).parent.name,
        question=question,
        answer=followup_res["answer"],
        referenced_evidence_ids=followup_res["referenced_evidence_ids"],
        recomputed=False,
    )
    return response


def handle_adapter_cli(args: argparse.Namespace) -> int:
    """CLI handler for matchquant agent-adapter."""
    payload: dict[str, Any] = {}
    if getattr(args, "request_file", None):
        payload = json.loads(Path(args.request_file).read_text(encoding="utf-8"))
    elif getattr(args, "request_json", None):
        payload = json.loads(args.request_json)

    if getattr(args, "action", None) and "action" not in payload:
        payload["action"] = args.action

    result = execute_adapter_request(payload)
    # Output ONLY JSON to stdout
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0 if result.get("status") in ("ok", "needs_review", "applied", "dry_run_ok") else 1
