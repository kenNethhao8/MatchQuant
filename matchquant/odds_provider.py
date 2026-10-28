"""Layer 1: External market odds provider and snapshot persistence for MatchQuant.

Fetches 1X2 (H2H) and Asian Handicap (Spreads) odds from external odds providers
(default: The Odds API) without violating Track A fundamental model isolation.
All snapshots are stored immutably with explicit UTC timestamps to prevent
look-ahead bias (observed_at < as_of).
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from .profile import UserProfile, load_profile
from .single_match import MARKET_ALLOWED_FIELDS, parse_iso_date

ODDS_SNAPSHOT_SCHEMA_VERSION = "matchquant-odds-snapshot-v1"

# Priority list for authoritative and liquid bookmakers
PREFERRED_BOOKMAKERS = [
    "pinnacle",
    "bet365",
    "sport888",
    "williamhill",
    "unibet_uk",
    "betfair_sb_uk",
    "skybet",
    "ladbrokes_uk",
    "coral",
    "betway",
    "paddypower",
    "betfred_uk",
    "matchbook",
    "smarkets",
    "betvictor",
]

# Canonical team aliases mapping for EPL to bridge The Odds API full names
# and MatchQuant canonical names
ODDS_API_EPL_CANONICAL: dict[str, str] = {
    "arsenal": "Arsenal",
    "aston villa": "Aston Villa",
    "afc bournemouth": "Bournemouth",
    "bournemouth": "Bournemouth",
    "brentford": "Brentford",
    "brighton and hove albion": "Brighton",
    "brighton & hove albion": "Brighton",
    "brighton": "Brighton",
    "chelsea": "Chelsea",
    "crystal palace": "Crystal Palace",
    "everton": "Everton",
    "fulham": "Fulham",
    "ipswich town": "Ipswich",
    "ipswich": "Ipswich",
    "leicester city": "Leicester",
    "leicester": "Leicester",
    "liverpool": "Liverpool",
    "manchester city": "Man City",
    "man city": "Man City",
    "manchester united": "Man United",
    "man united": "Man United",
    "newcastle united": "Newcastle",
    "newcastle": "Newcastle",
    "nottingham forest": "Nottingham Forest",
    "southampton": "Southampton",
    "tottenham hotspur": "Tottenham",
    "tottenham": "Tottenham",
    "west ham united": "West Ham",
    "west ham": "West Ham",
    "wolverhampton wanderers": "Wolves",
    "wolves": "Wolves",
    "leeds united": "Leeds",
    "leeds": "Leeds",
    "burnley": "Burnley",
    "luton town": "Luton",
    "luton": "Luton",
    "sheffield united": "Sheffield United",
    "norwich city": "Norwich",
    "watford": "Watford",
    "west bromwich albion": "West Brom",
}


class OddsProviderError(RuntimeError):
    """Raised when odds retrieval, matching, or conversion fails."""
    pass


def get_odds_api_key(api_key: str | None = None) -> str | None:
    """Retrieve The Odds API key from argument, environment, or Windows user registry."""
    if api_key and str(api_key).strip():
        return str(api_key).strip()

    env_val = os.environ.get("THE_ODDS_API_KEY")
    if env_val and env_val.strip():
        return env_val.strip()

    # On Windows, try reading from user environment registry if process didn't inherit it
    if sys.platform == "win32":
        try:
            import winreg
            with winreg.OpenKey(winreg.HKEY_CURRENT_USER, r"Environment") as key:
                val, _ = winreg.QueryValueEx(key, "THE_ODDS_API_KEY")
                if val and str(val).strip():
                    return str(val).strip()
        except Exception:
            pass

    return None


def normalize_team_for_matching(name: str, profile: UserProfile | None = None) -> str:
    """Normalize a team name or alias to canonical English form."""
    raw = str(name).strip()
    if not raw:
        return ""

    if profile is not None:
        raw = profile.resolve_team_alias(raw)

    cleaned = " ".join(raw.lower().replace("-", " ").replace("&", "and").split())
    if cleaned in ODDS_API_EPL_CANONICAL:
        return ODDS_API_EPL_CANONICAL[cleaned]

    # Check without common suffixes/prefixes
    no_fc = re.sub(r"\b(fc|afc|cf)\b", "", cleaned).strip()
    no_fc = " ".join(no_fc.split())
    if no_fc in ODDS_API_EPL_CANONICAL:
        return ODDS_API_EPL_CANONICAL[no_fc]

    return raw


def _select_bookmaker(bookmakers: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Select the best bookmaker prioritizing authoritative and liquid providers with h2h market."""
    if not bookmakers:
        return None

    # Filter bookmakers that provide h2h market
    candidates = []
    for b in bookmakers:
        markets = {m.get("key") for m in b.get("markets", [])}
        if "h2h" in markets:
            candidates.append(b)

    if not candidates:
        return None

    pref_map = {name: idx for idx, name in enumerate(PREFERRED_BOOKMAKERS)}
    # Sort candidates by preference index; those not in preferred list go last
    candidates.sort(key=lambda b: pref_map.get(b.get("key", "").lower(), 9999))
    return candidates[0]


def _extract_odds_from_bookmaker(
    bookmaker: dict[str, Any],
    home_canonical: str,
    away_canonical: str,
    all_bookmakers: list[dict[str, Any]] | None = None,
) -> tuple[float, float, float, float | None]:
    """Extract h2h (home, draw, away) and spreads line from selected bookmaker."""
    markets_by_key: dict[str, dict[str, Any]] = {
        m.get("key"): m for m in bookmaker.get("markets", []) if isinstance(m, dict)
    }

    h2h_market = markets_by_key.get("h2h")
    if not h2h_market:
        raise OddsProviderError(f"bookmaker '{bookmaker.get('title')}' has no h2h market")

    h_odds: float | None = None
    d_odds: float | None = None
    a_odds: float | None = None

    for outcome in h2h_market.get("outcomes", []):
        name = outcome.get("name", "")
        price = float(outcome.get("price", 0.0))
        norm_name = normalize_team_for_matching(name)
        if norm_name == home_canonical or name.casefold() == home_canonical.casefold():
            h_odds = price
        elif norm_name == away_canonical or name.casefold() == away_canonical.casefold():
            a_odds = price
        elif name.casefold() in ("draw", "平局", "tie"):
            d_odds = price

    if h_odds is None or d_odds is None or a_odds is None:
        raise OddsProviderError(
            f"incomplete h2h outcomes from '{bookmaker.get('title')}': "
            f"home={h_odds}, draw={d_odds}, away={a_odds}"
        )

    if h_odds <= 1.0 or d_odds <= 1.0 or a_odds <= 1.0:
        raise OddsProviderError(
            f"decimal odds must be strictly > 1.0; got home={h_odds}, draw={d_odds}, away={a_odds}"
        )

    # Extract spreads (Asian Handicap)
    # Sign convention: home handicap is negative if giving goals (e.g. -0.25), positive if receiving.
    mkt_ah_line: float | None = None
    spread_market = markets_by_key.get("spreads")

    # If selected bookmaker lacks spreads, check if any other bookmaker in the fixture has spreads
    if not spread_market and all_bookmakers:
        for other_b in all_bookmakers:
            for m in other_b.get("markets", []):
                if m.get("key") == "spreads":
                    spread_market = m
                    break
            if spread_market:
                break

    if spread_market:
        for outcome in spread_market.get("outcomes", []):
            name = outcome.get("name", "")
            norm_name = normalize_team_for_matching(name)
            if norm_name == home_canonical or name.casefold() == home_canonical.casefold():
                point = outcome.get("point")
                if point is not None:
                    try:
                        mkt_ah_line = float(point)
                    except (ValueError, TypeError):
                        mkt_ah_line = None
                break

    return h_odds, d_odds, a_odds, mkt_ah_line


def fetch_epl_odds(
    home_team: str,
    away_team: str,
    fixture_date: str,
    api_key: str | None = None,
    *,
    regions: str = "uk",
    markets: str = "h2h,spreads",
    output_dir: Path | str | None = None,
    profile: UserProfile | None = None,
    max_retries: int = 2,
    timeout: float = 15.0,
) -> dict[str, Any]:
    """Fetch live EPL odds for a specific match from The Odds API and persist snapshot.

    Returns the immutable snapshot dictionary.
    """
    prof = profile or load_profile()
    key = get_odds_api_key(api_key)
    if not key:
        raise OddsProviderError(
            "The Odds API key is not configured. Please set the THE_ODDS_API_KEY environment variable."
        )

    canonical_home = normalize_team_for_matching(home_team, prof)
    canonical_away = normalize_team_for_matching(away_team, prof)

    if not canonical_home or not canonical_away:
        raise OddsProviderError(
            f"invalid team names: home={home_team!r}, away={away_team!r}"
        )

    target_date = parse_iso_date(fixture_date, "fixture_date").isoformat()

    # Query The Odds API
    base_url = "https://api.the-odds-api.com/v4/sports/soccer_epl/odds"
    params = {
        "apiKey": key,
        "regions": regions,
        "markets": markets,
        "oddsFormat": "decimal",
        "dateFormat": "iso",
    }
    url = f"{base_url}?{urllib.parse.urlencode(params)}"

    req = urllib.request.Request(
        url,
        headers={"User-Agent": "MatchQuant-OddsProvider/1.0", "Accept": "application/json"},
    )

    last_error: Exception | None = None
    data: list[dict[str, Any]] | None = None

    for attempt in range(max_retries + 1):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                if resp.status != 200:
                    raise OddsProviderError(
                        f"The Odds API returned unexpected HTTP status: {resp.status}"
                    )
                raw_bytes = resp.read()
                data = json.loads(raw_bytes.decode("utf-8"))
                break
        except urllib.error.HTTPError as http_err:
            if http_err.code == 401:
                raise OddsProviderError(
                    "The Odds API authentication failed (HTTP 401: Unauthorized). Check your API key."
                ) from http_err
            elif http_err.code == 429:
                raise OddsProviderError(
                    "The Odds API rate limit exceeded (HTTP 429: Too Many Requests)."
                ) from http_err
            else:
                last_error = OddsProviderError(
                    f"The Odds API request failed with HTTP {http_err.code}: {http_err.reason}"
                )
        except (urllib.error.URLError, TimeoutError, OSError) as net_err:
            last_error = OddsProviderError(
                f"The Odds API network communication error: {net_err}"
            )

        if attempt < max_retries:
            time.sleep(0.5 * (attempt + 1))

    if data is None:
        raise last_error or OddsProviderError("failed to retrieve data from The Odds API")

    if not isinstance(data, list):
        raise OddsProviderError("unexpected response format from The Odds API; expected a JSON list")

    # Match fixture by home/away teams and date
    matched_fixture: dict[str, Any] | None = None
    for item in data:
        item_home = normalize_team_for_matching(item.get("home_team", ""), prof)
        item_away = normalize_team_for_matching(item.get("away_team", ""), prof)
        commence_time = str(item.get("commence_time", ""))
        commence_date = commence_time[:10]

        if item_home == canonical_home and item_away == canonical_away:
            # Check date match: exact date or within 1 day (timezone boundary)
            if commence_date == target_date:
                matched_fixture = item
                break
            try:
                d1 = parse_iso_date(commence_date, "commence_date")
                d2 = parse_iso_date(target_date, "target_date")
                if abs((d1 - d2).days) <= 1:
                    matched_fixture = item
                    break
            except Exception:
                pass

    if matched_fixture is None:
        raise OddsProviderError(
            f"no odds found for fixture {canonical_home} vs {canonical_away} on {target_date}"
        )

    bookmakers = matched_fixture.get("bookmakers", [])
    selected_bm = _select_bookmaker(bookmakers)
    if not selected_bm:
        raise OddsProviderError(
            f"no valid bookmakers offering h2h odds found for {canonical_home} vs {canonical_away}"
        )

    h_odds, d_odds, a_odds, mkt_ah_line = _extract_odds_from_bookmaker(
        selected_bm,
        canonical_home,
        canonical_away,
        all_bookmakers=bookmakers,
    )

    now_utc = datetime.now(timezone.utc)
    observed_at_utc = now_utc.strftime("%Y-%m-%dT%H:%M:%SZ")
    observed_date_str = now_utc.strftime("%Y-%m-%d")

    bm_key = selected_bm.get("key", "unknown")
    bm_title = selected_bm.get("title", bm_key)

    fixture_id = f"{target_date}_{canonical_home}_{canonical_away}".replace(" ", "_")

    snapshot: dict[str, Any] = {
        "schema_version": ODDS_SNAPSHOT_SCHEMA_VERSION,
        "fixture_id": fixture_id,
        "home_team": canonical_home,
        "away_team": canonical_away,
        "date": target_date,
        "observed_at_utc": observed_at_utc,
        "observed_at": observed_date_str,
        "source": f"The Odds API ({bm_title})",
        "bookmaker": bm_key,
        "bookmaker_title": bm_title,
        "odds_open_h": round(h_odds, 4),
        "odds_open_d": round(d_odds, 4),
        "odds_open_a": round(a_odds, 4),
        "mkt_ah_line": round(mkt_ah_line, 4) if mkt_ah_line is not None else None,
        "home_handicap_sign_convention": "主队让球为负；例如 -0.25 表示主队让 0.25 球",
        "contract_note": "odds_open_* 表示本快照时点赔率，沿用既有契约字段名，避免与历史初盘混淆",
        "api_fixture_id": matched_fixture.get("id"),
        "commence_time": matched_fixture.get("commence_time"),
    }

    # Persist snapshot
    if output_dir:
        base_dir = Path(output_dir)
    else:
        base_dir = Path(prof.report_root) / "odds_snapshots"

    snapshot_dir = base_dir / fixture_id
    snapshot_dir.mkdir(parents=True, exist_ok=True)

    # Windows-safe UTC timestamp filename
    ts_filename = now_utc.strftime("%Y%m%dT%H%M%SZ") + ".json"
    snapshot_file = snapshot_dir / ts_filename

    # Ensure immutable persistence
    if not snapshot_file.exists():
        snapshot_file.write_text(
            json.dumps(snapshot, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    snapshot["snapshot_path"] = str(snapshot_file.resolve())
    return snapshot


def snapshot_to_market_input(snapshot: Mapping[str, Any]) -> dict[str, Any]:
    """Convert an immutable odds snapshot into single_match.py MARKET_ALLOWED_FIELDS format.

    Validates numeric ranges, team names, and produces strictly allowed fields.
    """
    for required in ("home_team", "away_team", "date", "observed_at", "source", "odds_open_h", "odds_open_d", "odds_open_a"):
        if required not in snapshot or snapshot[required] is None:
            raise OddsProviderError(f"snapshot missing required field: {required}")

    home_team = str(snapshot["home_team"]).strip()
    away_team = str(snapshot["away_team"]).strip()
    source = str(snapshot["source"]).strip()

    if not home_team or not away_team or not source:
        raise OddsProviderError("home_team, away_team, and source must be non-empty strings")

    fixture_date = parse_iso_date(str(snapshot["date"]), "snapshot date").isoformat()
    observed_at = parse_iso_date(str(snapshot["observed_at"]), "snapshot observed_at").isoformat()

    h_odds = float(snapshot["odds_open_h"])
    d_odds = float(snapshot["odds_open_d"])
    a_odds = float(snapshot["odds_open_a"])

    if h_odds <= 1.0 or d_odds <= 1.0 or a_odds <= 1.0:
        raise OddsProviderError(
            f"odds must be strictly greater than 1.0; got h={h_odds}, d={d_odds}, a={a_odds}"
        )

    market_input: dict[str, Any] = {
        "home_team": home_team,
        "away_team": away_team,
        "date": fixture_date,
        "observed_at": observed_at,
        "source": source,
        "odds_open_h": round(h_odds, 4),
        "odds_open_d": round(d_odds, 4),
        "odds_open_a": round(a_odds, 4),
    }

    mkt_ah_line = snapshot.get("mkt_ah_line")
    if mkt_ah_line is not None:
        try:
            line_val = float(mkt_ah_line)
            market_input["mkt_ah_line"] = round(line_val, 4)
        except (ValueError, TypeError):
            pass

    # Strict check against MARKET_ALLOWED_FIELDS
    extra = set(market_input) - MARKET_ALLOWED_FIELDS
    if extra:
        raise OddsProviderError(f"market input contains disallowed fields: {extra}")

    return market_input
