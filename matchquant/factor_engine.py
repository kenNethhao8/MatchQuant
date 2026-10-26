"""Layer 2: Chronological Football Quantitative Factor Engine.

Computes 5 orthogonal factor families from the Rich Match Table with strict
pre-match batch freezing (zero look-ahead bias):
  1. strength: Dynamic Elo, opponent-adjusted attack/defense mismatch, venue edge, tempo
  2. dominance: Opponent-adjusted net shots on target (SOT), SOT momentum, shot volume,
                shot precision, and corner siege pressure
  3. luck_reversion: Finishing conversion excess (xG debt proxy), save luck excess,
                     and points-vs-dominance divergence
  4. rhythm_discipline: 1st-half fast-start edge, 2nd-half surge/stamina, discipline risk,
                        and Bayesian-shrunk referee home bias
  5. context_market: Promoted-team cold-start discount, clipped log rest-days advantage,
                     plus isolated Market Track factors (opening edge, smart-money steam, AH)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np
import pandas as pd

FACTOR_SPECIFICATIONS: tuple[dict[str, Any], ...] = (
    # Family 1: Strength & Tempo
    {
        "name": "elo_diff",
        "family": "strength",
        "track": "fundamental",
        "expected_sign": 1,
        "description": "主客动态Elo评级差（含主场优势与升班马折价，除以100缩放）",
    },
    {
        "name": "elo_abs_gap",
        "family": "strength",
        "track": "fundamental",
        "expected_sign": 0,
        "description": "主客Elo实力绝对分差（衡量实力悬殊度，越小越易出平局）",
    },
    {
        "name": "att_def_mismatch_diff",
        "family": "strength",
        "track": "fundamental",
        "expected_sign": 1,
        "description": "对手强度校准后的EWMA主客攻防错位净胜球期望差",
    },
    {
        "name": "venue_specialist_diff",
        "family": "strength",
        "track": "fundamental",
        "expected_sign": 1,
        "description": "主队主场专属EWMA净胜球与客队客场专属EWMA净胜球之差",
    },
    {
        "name": "expected_tempo",
        "family": "strength",
        "track": "fundamental",
        "expected_sign": 0,
        "description": "双方预期场面总节奏（总射正+总进球期望，慢节奏对应高平局率）",
    },
    # Family 2: Pitch Dominance & Shot Quality
    {
        "name": "sot_dominance_diff",
        "family": "dominance",
        "track": "fundamental",
        "expected_sign": 1,
        "description": "对手强度修正后的慢速EWMA净射正差（HST-AST，核心xG代理指标）",
    },
    {
        "name": "sot_momentum_diff",
        "family": "dominance",
        "track": "fundamental",
        "expected_sign": 1,
        "description": "近期快速EWMA与慢速EWMA净射正之差（捕捉场面统治力加速上升/滑坡）",
    },
    {
        "name": "shot_volume_diff",
        "family": "dominance",
        "track": "fundamental",
        "expected_sign": 1,
        "description": "对手强度修正后的EWMA总射门差（HS-AS，前场推进压制力）",
    },
    {
        "name": "shot_precision_diff",
        "family": "dominance",
        "track": "fundamental",
        "expected_sign": 1,
        "description": "EWMA射正率（HST/HS）与抗射正率综合效率差",
    },
    {
        "name": "corner_pressure_diff",
        "family": "dominance",
        "track": "fundamental",
        "expected_sign": 1,
        "description": "对手强度修正后的EWMA净角球差（HC-AC，禁区围攻程度）",
    },
    {
        "name": "style_counter_edge",
        "family": "dominance",
        "track": "fundamental",
        "expected_sign": 1,
        "description": "中心化战术交互克制项（主客角球施压与对手防守吃牌脆弱度去共线净优势）",
    },
    # Family 3: Luck & Mean-Reversion Alpha
    {
        "name": "finishing_luck_diff",
        "family": "luck_reversion",
        "track": "fundamental",
        "expected_sign": -1,
        "description": "短期射正转化率超额偏离（正值代表近期把握机会运气过热，负值代表狂射不进欠债待反弹）",
    },
    {
        "name": "save_luck_diff",
        "family": "luck_reversion",
        "track": "fundamental",
        "expected_sign": -1,
        "description": "短期门将扑救率超额偏离（识别靠神扑续命或被射正即丢球的极端波动）",
    },
    {
        "name": "points_vs_dominance_gap",
        "family": "luck_reversion",
        "track": "fundamental",
        "expected_sign": -1,
        "description": "近期实际拿分率与射正统治力隐含拿分率之差（识别虚假连胜 vs 倒霉强队）",
    },
    # Family 4: Match Rhythm, Half-Time & Discipline
    {
        "name": "first_half_edge_diff",
        "family": "rhythm_discipline",
        "track": "fundamental",
        "expected_sign": 1,
        "description": "上半场EWMA净胜球差（HTHG-HTAG，抢开局能力）",
    },
    {
        "name": "second_half_surge_diff",
        "family": "rhythm_discipline",
        "track": "fundamental",
        "expected_sign": 1,
        "description": "下半场EWMA净胜球差（体能储备、板凳深度与后程发力能力）",
    },
    {
        "name": "discipline_risk_diff",
        "family": "rhythm_discipline",
        "track": "fundamental",
        "expected_sign": -1,
        "description": "主客队EWMA吃牌与犯规压力差（黄牌+2.5*红牌+0.08*犯规，负向纪律风险）",
    },
    {
        "name": "referee_home_bias",
        "family": "rhythm_discipline",
        "track": "fundamental",
        "expected_sign": 1,
        "description": "当值主裁历史贝叶斯收缩后的主场净胜球/判罚尺度偏倚",
    },
    # Family 5: Schedule Context & Market Signals
    {
        "name": "promoted_coldstart_diff",
        "family": "context_market",
        "track": "fundamental",
        "expected_sign": -1,
        "description": "升班马赛季前10轮冷启动指示差（主队升班马权重 - 客队升班马权重）",
    },
    {
        "name": "rest_advantage_diff",
        "family": "context_market",
        "track": "fundamental",
        "expected_sign": 1,
        "description": "截断对数休息天数优势：clip(ln(Rest_H) - ln(Rest_A), -1.0, 1.0)",
    },
    {
        "name": "fixture_congestion_diff",
        "family": "context_market",
        "track": "fundamental",
        "expected_sign": -1,
        "description": "密集赛程疲劳负荷差（过去14天英超比赛场次差 C14_H - C14_A，正值代表主队更透支，预期方向-1）",
    },
    {
        "name": "table_pressure_diff",
        "family": "context_market",
        "track": "fundamental",
        "expected_sign": 1,
        "description": "赛季中后程连续平滑积分紧迫度差（第12轮起高斯距离争冠/争四/保级战意差）",
    },
    # Market Track Factors (Isolated from pure fundamental track)
    {
        "name": "mkt_open_prob_edge",
        "family": "context_market",
        "track": "market",
        "expected_sign": 1,
        "description": "Pinnacle去水初盘主客胜隐含概率差（市场初始共识）",
    },
    {
        "name": "mkt_steam_move_diff",
        "family": "context_market",
        "track": "market",
        "expected_sign": 1,
        "description": "临场聪明钱漂移（终盘去水概率 - 初盘去水概率的主客净变动，捕捉临场伤停/首发信息）",
    },
    {
        "name": "mkt_ah_supremacy",
        "family": "context_market",
        "track": "market",
        "expected_sign": 1,
        "description": "亚盘让球盘口隐含主队净胜球优势（-mkt_ah_line）",
    },
)

FUNDAMENTAL_FACTOR_COLUMNS: tuple[str, ...] = tuple(
    spec["name"] for spec in FACTOR_SPECIFICATIONS if spec["track"] == "fundamental"
)
MARKET_FACTOR_COLUMNS: tuple[str, ...] = tuple(
    spec["name"] for spec in FACTOR_SPECIFICATIONS if spec["track"] == "market"
)
ALL_FACTOR_COLUMNS: tuple[str, ...] = tuple(spec["name"] for spec in FACTOR_SPECIFICATIONS)


@dataclass
class TeamDynamicState:
    """Per-team online state updated strictly after each match batch."""

    elo: float = 1500.0
    matches_total: int = 0
    matches_in_season: int = 0
    is_promoted: bool = False
    last_match_date: pd.Timestamp | None = None
    match_dates_in_season: list[pd.Timestamp] = field(default_factory=list)

    # Opponent-adjusted slow EWMA (alpha = 0.12, ~8 match half-life)
    adj_goals_for_slow: float = 0.0
    adj_goals_against_slow: float = 0.0
    adj_sot_net_slow: float = 0.0
    adj_shots_net_slow: float = 0.0
    adj_corners_net_slow: float = 0.0

    # Fast EWMA (alpha = 0.28, ~3.5 match half-life)
    adj_sot_net_fast: float = 0.0
    goals_for_fast: float = 1.35
    goals_against_fast: float = 1.35
    sot_for_fast: float = 4.50
    sot_against_fast: float = 4.50
    shots_for_slow: float = 13.0
    sot_for_slow: float = 4.50
    sot_against_slow: float = 4.50
    goals_for_slow: float = 1.35
    goals_against_slow: float = 1.35
    points_rate_fast: float = 0.45

    # Venue-specific EWMA net goals
    home_net_goals_ewma: float = 0.25
    away_net_goals_ewma: float = -0.25

    # Half-time & 2nd-half EWMA net goals
    ht_net_goals_ewma: float = 0.0
    sh_net_goals_ewma: float = 0.0

    # Discipline EWMA
    discipline_ewma: float = 2.50

    def shrink_between_seasons(self, carry_factor: float = 0.72) -> None:
        """Apply mean reversion between seasons while preserving structural hierarchy."""
        self.elo = 1500.0 + carry_factor * (self.elo - 1500.0)
        self.matches_in_season = 0
        self.is_promoted = False
        self.last_match_date = None
        self.match_dates_in_season = []

        self.adj_goals_for_slow *= carry_factor
        self.adj_goals_against_slow *= carry_factor
        self.adj_sot_net_slow *= carry_factor
        self.adj_shots_net_slow *= carry_factor
        self.adj_corners_net_slow *= carry_factor
        self.adj_sot_net_fast = self.adj_sot_net_slow

        self.home_net_goals_ewma = 0.25 + carry_factor * (self.home_net_goals_ewma - 0.25)
        self.away_net_goals_ewma = -0.25 + carry_factor * (self.away_net_goals_ewma + 0.25)
        self.ht_net_goals_ewma *= carry_factor
        self.sh_net_goals_ewma *= carry_factor
        self.discipline_ewma = 2.50 + carry_factor * (self.discipline_ewma - 2.50)


@dataclass
class RefereeState:
    matches: int = 0
    home_goal_diff_sum: float = 0.0
    cards_sum: float = 0.0


class ChronologicalFactorEngine:
    """Stateful factor generator with strict batch-freeze chronology."""

    def __init__(
        self,
        *,
        elo_k: float = 24.0,
        elo_home_advantage: float = 55.0,
        elo_draw_base: float = 0.26,
        elo_draw_scale: float = 400.0,
        promoted_initial_elo: float = 1435.0,
        alpha_slow: float = 0.12,
        alpha_fast: float = 0.28,
        alpha_venue: float = 0.15,
        season_carry: float = 0.72,
    ) -> None:
        self.elo_k = float(elo_k)
        self.elo_home_advantage = float(elo_home_advantage)
        self.elo_draw_base = float(elo_draw_base)
        self.elo_draw_scale = float(elo_draw_scale)
        self.promoted_initial_elo = float(promoted_initial_elo)
        self.alpha_slow = float(alpha_slow)
        self.alpha_fast = float(alpha_fast)
        self.alpha_venue = float(alpha_venue)
        self.season_carry = float(season_carry)
        self.reset()

    def reset(self) -> None:
        self.teams: dict[str, TeamDynamicState] = {}
        self.referees: dict[str, RefereeState] = {}
        self.season_table: dict[str, dict[str, int]] = {}
        self.current_season: str | None = None
        self.previous_season_teams: set[str] = set()
        self.current_season_teams: set[str] = set()
        self.league_matches: int = 0
        self.league_home_goals_ewma: float = 1.52
        self.league_away_goals_ewma: float = 1.22
        self.league_home_sot_ewma: float = 4.85
        self.league_away_sot_ewma: float = 4.05
        self.league_home_shots_ewma: float = 14.0
        self.league_away_shots_ewma: float = 11.5
        self.league_home_corners_ewma: float = 5.7
        self.league_away_corners_ewma: float = 4.7

    def _on_season_change(self, new_season: str) -> None:
        if self.current_season is not None and new_season != self.current_season:
            self.previous_season_teams = set(self.current_season_teams)
            self.current_season_teams = set()
            self.season_table = {}
            for team_state in self.teams.values():
                team_state.shrink_between_seasons(self.season_carry)
        self.current_season = new_season

    def _get_or_init_team(self, team_name: str) -> TeamDynamicState:
        if team_name not in self.teams:
            # First time seen: if we already have at least one completed season,
            # this team is newly promoted!
            is_promoted = len(self.previous_season_teams) > 0
            init_elo = self.promoted_initial_elo if is_promoted else 1500.0
            init_att = -0.15 if is_promoted else 0.0
            init_def = 0.18 if is_promoted else 0.0
            init_sot = -0.85 if is_promoted else 0.0
            self.teams[team_name] = TeamDynamicState(
                elo=init_elo,
                is_promoted=is_promoted,
                adj_goals_for_slow=init_att,
                adj_goals_against_slow=init_def,
                adj_sot_net_slow=init_sot,
                adj_sot_net_fast=init_sot,
            )
        else:
            state = self.teams[team_name]
            if (
                len(self.previous_season_teams) > 0
                and team_name not in self.previous_season_teams
                and state.matches_in_season == 0
            ):
                # Re-promoted team returning after relegation
                state.is_promoted = True
                state.elo = min(state.elo, self.promoted_initial_elo + 15.0)
        return self.teams[team_name]

    def elo_probabilities(self, home_elo: float, away_elo: float) -> tuple[float, float, float]:
        diff = home_elo + self.elo_home_advantage - away_elo
        p_draw = self.elo_draw_base * math.exp(-abs(diff) / self.elo_draw_scale)
        p_decisive = 1.0 - p_draw
        p_home_given_decisive = 1.0 / (1.0 + math.pow(10.0, -diff / 400.0))
        p_home = p_decisive * p_home_given_decisive
        p_away = p_decisive * (1.0 - p_home_given_decisive)
        return p_home, p_draw, p_away

    def compute_pre_match_factors(self, row: Mapping[str, Any]) -> dict[str, Any]:
        """Compute all pre-match factors for a single fixture WITHOUT mutating state."""
        home_name = str(row["home_team"])
        away_name = str(row["away_team"])
        ref_name = str(row.get("referee", "Unknown"))
        match_date = pd.Timestamp(row["date"])

        h = self._get_or_init_team(home_name)
        a = self._get_or_init_team(away_name)

        # 1. Elo & Strength
        elo_diff_raw = (h.elo + self.elo_home_advantage) - a.elo
        elo_diff = elo_diff_raw / 100.0
        elo_abs_gap = abs(elo_diff_raw) / 100.0
        elo_ph, elo_pd, elo_pa = self.elo_probabilities(h.elo, a.elo)
        elo_expected_score = elo_ph + 0.5 * elo_pd

        # Attack vs Defense mismatch (opponent-adjusted)
        home_exp_edge = h.adj_goals_for_slow + a.adj_goals_against_slow
        away_exp_edge = a.adj_goals_for_slow + h.adj_goals_against_slow
        att_def_mismatch_diff = home_exp_edge - away_exp_edge

        venue_specialist_diff = h.home_net_goals_ewma - a.away_net_goals_ewma
        expected_tempo = (
            (h.sot_for_slow + a.sot_for_slow) / 9.0
            + (h.goals_for_slow + a.goals_for_slow) / 2.7
        ) - 2.0

        # 2. Pitch Dominance & Shot Quality
        sot_dominance_diff = h.adj_sot_net_slow - a.adj_sot_net_slow
        h_sot_mom = h.adj_sot_net_fast - h.adj_sot_net_slow
        a_sot_mom = a.adj_sot_net_fast - a.adj_sot_net_slow
        sot_momentum_diff = h_sot_mom - a_sot_mom

        shot_volume_diff = (h.adj_shots_net_slow - a.adj_shots_net_slow) / 5.0

        h_prec = h.sot_for_slow / max(h.shots_for_slow, 5.0)
        a_prec = a.sot_for_slow / max(a.shots_for_slow, 5.0)
        shot_precision_diff = (h_prec - a_prec) * 10.0

        corner_pressure_diff = (h.adj_corners_net_slow - a.adj_corners_net_slow) / 3.0

        # Style counter edge: Centered interaction between aerial corner siege and opponent fragility/discipline
        c_h = h.adj_corners_net_slow
        c_a = a.adj_corners_net_slow
        disc_excess_h = h.discipline_ewma - 2.50
        disc_excess_a = a.discipline_ewma - 2.50
        vuln_excess_h = (h.goals_against_slow / max(h.sot_against_slow, 2.0)) - 0.30
        vuln_excess_a = (a.goals_against_slow / max(a.sot_against_slow, 2.0)) - 0.30

        h_style_edge = c_h * (0.20 + disc_excess_a + 2.0 * vuln_excess_a)
        a_style_edge = c_a * (0.20 + disc_excess_h + 2.0 * vuln_excess_h)
        style_counter_edge = float((h_style_edge - a_style_edge) / 3.0)

        # 3. Luck & Mean-Reversion Alpha
        # Finishing conversion excess: fast conversion rate minus slow baseline conversion rate
        h_conv_fast = h.goals_for_fast / max(h.sot_for_fast, 2.0)
        h_conv_slow = h.goals_for_slow / max(h.sot_for_slow, 2.0)
        a_conv_fast = a.goals_for_fast / max(a.sot_for_fast, 2.0)
        a_conv_slow = a.goals_for_slow / max(a.sot_for_slow, 2.0)
        finishing_luck_diff = (h_conv_fast - h_conv_slow) - (a_conv_fast - a_conv_slow)

        # Save luck excess: lower goals conceded per SOT than normal = keeper running hot
        h_save_excess = (h.goals_against_slow / max(h.sot_against_slow, 2.0)) - (
            h.goals_against_fast / max(h.sot_against_fast, 2.0)
        )
        a_save_excess = (a.goals_against_slow / max(a.sot_against_slow, 2.0)) - (
            a.goals_against_fast / max(a.sot_against_fast, 2.0)
        )
        save_luck_diff = h_save_excess - a_save_excess

        # Points rate vs SOT dominance implied points rate
        h_implied_pts = 1.0 / (1.0 + math.exp(-0.35 * h.adj_sot_net_fast))
        a_implied_pts = 1.0 / (1.0 + math.exp(-0.35 * a.adj_sot_net_fast))
        points_vs_dominance_gap = (h.points_rate_fast - h_implied_pts) - (
            a.points_rate_fast - a_implied_pts
        )

        # 4. Rhythm, Half-Time & Discipline
        first_half_edge_diff = h.ht_net_goals_ewma - a.ht_net_goals_ewma
        second_half_surge_diff = h.sh_net_goals_ewma - a.sh_net_goals_ewma
        discipline_risk_diff = h.discipline_ewma - a.discipline_ewma

        ref_state = self.referees.get(ref_name, RefereeState())
        league_avg_gd = self.league_home_goals_ewma - self.league_away_goals_ewma
        referee_home_bias = (
            (ref_state.home_goal_diff_sum + 15.0 * league_avg_gd) / (ref_state.matches + 15.0)
            - league_avg_gd
        )

        # 5. Context & Market Signals
        h_prom_weight = (
            max(0.0, (10 - h.matches_in_season) / 10.0) if h.is_promoted else 0.0
        )
        a_prom_weight = (
            max(0.0, (10 - a.matches_in_season) / 10.0) if a.is_promoted else 0.0
        )
        promoted_coldstart_diff = h_prom_weight - a_prom_weight

        if h.last_match_date is not None and a.last_match_date is not None:
            rest_h = max(1.0, min(14.0, float((match_date - h.last_match_date).days)))
            rest_a = max(1.0, min(14.0, float((match_date - a.last_match_date).days)))
            rest_advantage_diff = float(np.clip(math.log(rest_h) - math.log(rest_a), -1.0, 1.0))
        else:
            rest_advantage_diff = 0.0

        # Fixture congestion (EPL matches played in the past 14 natural days)
        c14_h = sum(1 for d in h.match_dates_in_season if 0 < (match_date - d).days <= 14)
        c14_a = sum(1 for d in a.match_dates_in_season if 0 < (match_date - d).days <= 14)
        fixture_congestion_diff = float(c14_h - c14_a)

        # Table pressure & motivation differential (round 12+)
        if min(h.matches_in_season, a.matches_in_season) < 12 or len(self.season_table) < 5:
            table_pressure_diff = 0.0
        else:
            sorted_teams = sorted(
                self.season_table.items(),
                key=lambda kv: (-kv[1]["pts"], -kv[1]["gd"], -kv[1]["gf"], kv[0]),
            )
            rank_map = {t_name: r + 1 for r, (t_name, _) in enumerate(sorted_teams)}
            pts_4 = sorted_teams[min(3, len(sorted_teams) - 1)][1]["pts"]
            pts_17 = sorted_teams[min(16, len(sorted_teams) - 1)][1]["pts"]
            pts_18 = sorted_teams[min(17, len(sorted_teams) - 1)][1]["pts"]

            def _calc_team_motivation(t_name: str, team_state: TeamState) -> float:
                if t_name not in self.season_table:
                    return 0.0
                entry = self.season_table[t_name]
                r = rank_map.get(t_name, 10)
                pts = entry["pts"]
                mp = entry["mp"]
                rem = max(1, 38 - mp)
                sigma_r = 1.15 * math.sqrt(rem)

                # Top / UCL Urgency (Gaussian Point-Gap to 4th place)
                if r <= 7:
                    gap_top = max(0, pts_4 - pts)
                    u_top = math.exp(-(gap_top ** 2) / (2.0 * (sigma_r ** 2)))
                else:
                    u_top = 0.0

                # Relegation Urgency (Gaussian Point-Gap to 17/18th cutoff for 13th+)
                if r >= 13:
                    if pts + 3 * rem < pts_17:
                        u_rel = 0.0
                    else:
                        delta_rel = abs(pts - 0.5 * (pts_17 + pts_18))
                        u_rel = 0.85 * math.exp(-(delta_rel ** 2) / (2.0 * (sigma_r ** 2)))
                else:
                    u_rel = 0.0

                w_progress = (mp / 38.0) ** 2
                base_motiv = w_progress * max(u_top, u_rel)
                sot_mod = 1.0 + 0.15 * math.tanh(team_state.adj_sot_net_fast)
                return float(base_motiv * sot_mod)

            table_pressure_diff = float(
                _calc_team_motivation(home_name, h) - _calc_team_motivation(away_name, a)
            )

        # Market track features (may be NaN for future fixtures without odds)
        open_ph = float(row.get("mkt_open_prob_home", np.nan))
        open_pa = float(row.get("mkt_open_prob_away", np.nan))
        mkt_open_prob_edge = (
            (open_ph - open_pa) if (math.isfinite(open_ph) and math.isfinite(open_pa)) else 0.0
        )

        steam_h = float(row.get("mkt_steam_home", np.nan))
        steam_a = float(row.get("mkt_steam_away", np.nan))
        mkt_steam_move_diff = (
            (steam_h - steam_a) * 10.0
            if (math.isfinite(steam_h) and math.isfinite(steam_a))
            else 0.0
        )

        ah_line = float(row.get("mkt_ah_line", 0.0))
        mkt_ah_supremacy = -ah_line if math.isfinite(ah_line) else 0.0

        return {
            "home_elo_pre": h.elo,
            "away_elo_pre": a.elo,
            "elo_prob_h": elo_ph,
            "elo_prob_d": elo_pd,
            "elo_prob_a": elo_pa,
            "elo_expected_score": elo_expected_score,
            "home_matches_played": h.matches_total,
            "away_matches_played": a.matches_total,
            # 22 Fundamental Factors
            "elo_diff": elo_diff,
            "elo_abs_gap": elo_abs_gap,
            "att_def_mismatch_diff": att_def_mismatch_diff,
            "venue_specialist_diff": venue_specialist_diff,
            "expected_tempo": expected_tempo,
            "sot_dominance_diff": sot_dominance_diff,
            "sot_momentum_diff": sot_momentum_diff,
            "shot_volume_diff": shot_volume_diff,
            "shot_precision_diff": shot_precision_diff,
            "corner_pressure_diff": corner_pressure_diff,
            "style_counter_edge": style_counter_edge,
            "finishing_luck_diff": finishing_luck_diff,
            "save_luck_diff": save_luck_diff,
            "points_vs_dominance_gap": points_vs_dominance_gap,
            "first_half_edge_diff": first_half_edge_diff,
            "second_half_surge_diff": second_half_surge_diff,
            "discipline_risk_diff": discipline_risk_diff,
            "referee_home_bias": referee_home_bias,
            "promoted_coldstart_diff": promoted_coldstart_diff,
            "rest_advantage_diff": rest_advantage_diff,
            "fixture_congestion_diff": fixture_congestion_diff,
            "table_pressure_diff": table_pressure_diff,
            # 3 Market Track Factors
            "mkt_open_prob_edge": mkt_open_prob_edge,
            "mkt_steam_move_diff": mkt_steam_move_diff,
            "mkt_ah_supremacy": mkt_ah_supremacy,
        }

    def update_from_completed_batch(self, batch: pd.DataFrame) -> None:
        """Update team, referee, and league states AFTER all pre-match predictions in the batch."""
        # First freeze pre-batch baselines so same-batch matches do not contaminate each other
        snapshots: list[tuple[Mapping[str, Any], float, float, float, float, float, float]] = []
        for row in batch.to_dict(orient="records"):
            h_name = str(row["home_team"])
            a_name = str(row["away_team"])
            h = self._get_or_init_team(h_name)
            a = self._get_or_init_team(a_name)
            snapshots.append(
                (
                    row,
                    h.elo,
                    a.elo,
                    h.adj_goals_for_slow,
                    h.adj_goals_against_slow,
                    a.adj_goals_for_slow,
                    a.adj_goals_against_slow,
                )
            )

        for (
            row,
            h_elo,
            a_elo,
            h_att,
            h_def,
            a_att,
            a_def,
        ) in snapshots:
            h_name = str(row["home_team"])
            a_name = str(row["away_team"])
            ref_name = str(row.get("referee", "Unknown"))
            match_date = pd.Timestamp(row["date"])

            self.current_season_teams.add(h_name)
            self.current_season_teams.add(a_name)

            h = self.teams[h_name]
            a = self.teams[a_name]

            hg = float(row["home_goals"])
            ag = float(row["away_goals"])
            hthg = float(row.get("ht_home_goals", 0))
            htag = float(row.get("ht_away_goals", 0))
            shhg = float(row.get("sh_home_goals", hg - hthg))
            shag = float(row.get("sh_away_goals", ag - htag))
            hs = float(row.get("home_shots", 13.0))
            as_ = float(row.get("away_shots", 11.0))
            hst = float(row.get("home_sot", 4.5))
            ast = float(row.get("away_sot", 4.0))
            hc = float(row.get("home_corners", 5.5))
            ac = float(row.get("away_corners", 4.5))
            hf = float(row.get("home_fouls", 10.5))
            af = float(row.get("away_fouls", 11.0))
            hy = float(row.get("home_yellow", 1.5))
            ay = float(row.get("away_yellow", 1.7))
            hr = float(row.get("home_red", 0.05))
            ar = float(row.get("away_red", 0.06))

            # Opponent strength offset based on pre-match Elo difference
            elo_gap = (h_elo - a_elo) / 400.0

            # Expected goals given opponent pre-batch defense/attack
            exp_hg = max(0.3, self.league_home_goals_ewma + 0.5 * (h_att + a_def))
            exp_ag = max(0.3, self.league_away_goals_ewma + 0.5 * (a_att + h_def))
            adj_hg = hg - exp_hg + 0.5 * h_att
            adj_ag = ag - exp_ag + 0.5 * a_att

            # Opponent-adjusted net SOT, shots, corners
            expected_net_sot = (self.league_home_sot_ewma - self.league_away_sot_ewma) - 1.8 * elo_gap
            adj_net_sot_h = (hst - ast) - expected_net_sot
            adj_net_sot_a = -adj_net_sot_h

            expected_net_shots = (
                self.league_home_shots_ewma - self.league_away_shots_ewma
            ) - 4.0 * elo_gap
            adj_net_shots_h = (hs - as_) - expected_net_shots
            adj_net_shots_a = -adj_net_shots_h

            expected_net_corners = (
                self.league_home_corners_ewma - self.league_away_corners_ewma
            ) - 1.5 * elo_gap
            adj_net_corners_h = (hc - ac) - expected_net_corners
            adj_net_corners_a = -adj_net_corners_h

            # Update Slow & Fast EWMAs for Home Team
            aslow = self.alpha_slow
            afast = self.alpha_fast
            aven = self.alpha_venue

            h.adj_goals_for_slow = (1 - aslow) * h.adj_goals_for_slow + aslow * adj_hg
            h.adj_goals_against_slow = (1 - aslow) * h.adj_goals_against_slow + aslow * adj_ag
            h.adj_sot_net_slow = (1 - aslow) * h.adj_sot_net_slow + aslow * adj_net_sot_h
            h.adj_sot_net_fast = (1 - afast) * h.adj_sot_net_fast + afast * adj_net_sot_h
            h.adj_shots_net_slow = (1 - aslow) * h.adj_shots_net_slow + aslow * adj_net_shots_h
            h.adj_corners_net_slow = (1 - aslow) * h.adj_corners_net_slow + aslow * adj_net_corners_h

            h.goals_for_fast = (1 - afast) * h.goals_for_fast + afast * hg
            h.goals_against_fast = (1 - afast) * h.goals_against_fast + afast * ag
            h.sot_for_fast = (1 - afast) * h.sot_for_fast + afast * hst
            h.sot_against_fast = (1 - afast) * h.sot_against_fast + afast * ast
            h.shots_for_slow = (1 - aslow) * h.shots_for_slow + aslow * hs
            h.sot_for_slow = (1 - aslow) * h.sot_for_slow + aslow * hst
            h.sot_against_slow = (1 - aslow) * h.sot_against_slow + aslow * ast
            h.goals_for_slow = (1 - aslow) * h.goals_for_slow + aslow * hg
            h.goals_against_slow = (1 - aslow) * h.goals_against_slow + aslow * ag

            h_pts = 1.0 if hg > ag else (1.0 / 3.0 if hg == ag else 0.0)
            a_pts = 1.0 if ag > hg else (1.0 / 3.0 if hg == ag else 0.0)
            h.points_rate_fast = (1 - afast) * h.points_rate_fast + afast * h_pts
            h.home_net_goals_ewma = (1 - aven) * h.home_net_goals_ewma + aven * (hg - ag)
            h.ht_net_goals_ewma = (1 - aslow) * h.ht_net_goals_ewma + aslow * (hthg - htag)
            h.sh_net_goals_ewma = (1 - aslow) * h.sh_net_goals_ewma + aslow * (shhg - shag)
            h_disc = hy + 2.5 * hr + 0.08 * hf
            h.discipline_ewma = (1 - aslow) * h.discipline_ewma + aslow * h_disc
            h.matches_total += 1
            h.matches_in_season += 1
            h.last_match_date = match_date
            h.match_dates_in_season.append(match_date)

            # Update Slow & Fast EWMAs for Away Team
            a.adj_goals_for_slow = (1 - aslow) * a.adj_goals_for_slow + aslow * adj_ag
            a.adj_goals_against_slow = (1 - aslow) * a.adj_goals_against_slow + aslow * adj_hg
            a.adj_sot_net_slow = (1 - aslow) * a.adj_sot_net_slow + aslow * adj_net_sot_a
            a.adj_sot_net_fast = (1 - afast) * a.adj_sot_net_fast + afast * adj_net_sot_a
            a.adj_shots_net_slow = (1 - aslow) * a.adj_shots_net_slow + aslow * adj_net_shots_a
            a.adj_corners_net_slow = (1 - aslow) * a.adj_corners_net_slow + aslow * adj_net_corners_a

            a.goals_for_fast = (1 - afast) * a.goals_for_fast + afast * ag
            a.goals_against_fast = (1 - afast) * a.goals_against_fast + afast * hg
            a.sot_for_fast = (1 - afast) * a.sot_for_fast + afast * ast
            a.sot_against_fast = (1 - afast) * a.sot_against_fast + afast * hst
            a.shots_for_slow = (1 - aslow) * a.shots_for_slow + aslow * as_
            a.sot_for_slow = (1 - aslow) * a.sot_for_slow + aslow * ast
            a.sot_against_slow = (1 - aslow) * a.sot_against_slow + aslow * hst
            a.goals_for_slow = (1 - aslow) * a.goals_for_slow + aslow * ag
            a.goals_against_slow = (1 - aslow) * a.goals_against_slow + aslow * hg

            a.points_rate_fast = (1 - afast) * a.points_rate_fast + afast * a_pts
            a.away_net_goals_ewma = (1 - aven) * a.away_net_goals_ewma + aven * (ag - hg)
            a.ht_net_goals_ewma = (1 - aslow) * a.ht_net_goals_ewma + aslow * (htag - hthg)
            a.sh_net_goals_ewma = (1 - aslow) * a.sh_net_goals_ewma + aslow * (shag - shhg)
            a_disc = ay + 2.5 * ar + 0.08 * af
            a.discipline_ewma = (1 - aslow) * a.discipline_ewma + aslow * a_disc
            a.matches_total += 1
            a.matches_in_season += 1
            a.last_match_date = match_date
            a.match_dates_in_season.append(match_date)

            # Update Season Standings table
            h_tab_pts = 3 if hg > ag else (1 if hg == ag else 0)
            a_tab_pts = 3 if ag > hg else (1 if ag == hg else 0)
            h_tbl = self.season_table.setdefault(h_name, {"pts": 0, "gd": 0, "gf": 0, "mp": 0})
            h_tbl["pts"] += h_tab_pts
            h_tbl["gd"] += int(hg - ag)
            h_tbl["gf"] += int(hg)
            h_tbl["mp"] += 1
            a_tbl = self.season_table.setdefault(a_name, {"pts": 0, "gd": 0, "gf": 0, "mp": 0})
            a_tbl["pts"] += a_tab_pts
            a_tbl["gd"] += int(ag - hg)
            a_tbl["gf"] += int(ag)
            a_tbl["mp"] += 1

            # Update Elo with margin-of-victory / SOT dampened multiplier
            ph, pd_, _ = self.elo_probabilities(h_elo, a_elo)
            exp_h = ph + 0.5 * pd_
            act_h = 1.0 if hg > ag else (0.5 if hg == ag else 0.0)
            goal_diff = abs(hg - ag)
            mov_mult = math.log(max(goal_diff, 1.0) + 1.0) * (
                2.2 / (2.2 + 0.001 * abs(h_elo - a_elo))
            )
            delta_elo = self.elo_k * mov_mult * (act_h - exp_h)
            h.elo += delta_elo
            a.elo -= delta_elo

            # Update Referee state
            ref_state = self.referees.setdefault(ref_name, RefereeState())
            ref_state.matches += 1
            ref_state.home_goal_diff_sum += hg - ag
            ref_state.cards_sum += hy + ay + 2.0 * (hr + ar)

            # Update League priors slowly
            l_alpha = 0.005
            self.league_home_goals_ewma = (1 - l_alpha) * self.league_home_goals_ewma + l_alpha * hg
            self.league_away_goals_ewma = (1 - l_alpha) * self.league_away_goals_ewma + l_alpha * ag
            self.league_home_sot_ewma = (1 - l_alpha) * self.league_home_sot_ewma + l_alpha * hst
            self.league_away_sot_ewma = (1 - l_alpha) * self.league_away_sot_ewma + l_alpha * ast
            self.league_home_shots_ewma = (1 - l_alpha) * self.league_home_shots_ewma + l_alpha * hs
            self.league_away_shots_ewma = (1 - l_alpha) * self.league_away_shots_ewma + l_alpha * as_
            self.league_home_corners_ewma = (1 - l_alpha) * self.league_home_corners_ewma + l_alpha * hc
            self.league_away_corners_ewma = (1 - l_alpha) * self.league_away_corners_ewma + l_alpha * ac
            self.league_matches += 1


def build_historical_factor_panel(
    rich_matches: pd.DataFrame,
    *,
    engine: ChronologicalFactorEngine | None = None,
) -> tuple[pd.DataFrame, ChronologicalFactorEngine]:
    """Build the complete pre-match factor panel across all historical matches.

    Strict batch-freeze guarantee: all matches sharing the same (season, date, kickoff_time)
    batch have their pre-match factors computed before ANY match in that batch updates state.
    """
    if engine is None:
        engine = ChronologicalFactorEngine()

    df = rich_matches.copy()
    if "kickoff_time" not in df.columns:
        df["kickoff_time"] = ""
    df["kickoff_time"] = df["kickoff_time"].fillna("").astype(str)
    df = df.sort_values(["date", "kickoff_time", "match_id"], kind="mergesort").reset_index(
        drop=True
    )

    records: list[dict[str, Any]] = []
    for (season, date_str, ko_time), batch in df.groupby(
        ["season", "date", "kickoff_time"], sort=False
    ):
        engine._on_season_change(str(season))

        # Step 1: Compute pre-match factors for EVERY match in the batch first
        for row in batch.to_dict(orient="records"):
            factors = engine.compute_pre_match_factors(row)
            hg = int(row["home_goals"])
            ag = int(row["away_goals"])
            act_score = 1.0 if hg > ag else (0.5 if hg == ag else 0.0)
            goal_diff = hg - ag
            record = {
                "match_id": row["match_id"],
                "season": row["season"],
                "date": date_str,
                "kickoff_time": ko_time,
                "home_team": row["home_team"],
                "away_team": row["away_team"],
                "referee": row.get("referee", "Unknown"),
                "home_goals": hg,
                "away_goals": ag,
                "goal_diff": goal_diff,
                "actual_result": "H" if hg > ag else ("A" if ag > hg else "D"),
                "actual_home_score": act_score,
                "elo_residual": act_score - factors["elo_expected_score"],
                "mkt_open_prob_home": row.get("mkt_open_prob_home", np.nan),
                "mkt_open_prob_draw": row.get("mkt_open_prob_draw", np.nan),
                "mkt_open_prob_away": row.get("mkt_open_prob_away", np.nan),
                "mkt_close_prob_home": row.get("mkt_close_prob_home", np.nan),
                "mkt_close_prob_draw": row.get("mkt_close_prob_draw", np.nan),
                "mkt_close_prob_away": row.get("mkt_close_prob_away", np.nan),
                "mkt_ou25_over_prob": row.get("mkt_ou25_over_prob", np.nan),
                "odds_open_h": row.get("odds_open_h", np.nan),
                "odds_open_d": row.get("odds_open_d", np.nan),
                "odds_open_a": row.get("odds_open_a", np.nan),
                "odds_close_h": row.get("odds_close_h", np.nan),
                "odds_close_d": row.get("odds_close_d", np.nan),
                "odds_close_a": row.get("odds_close_a", np.nan),
                "odds_ou25_over": row.get("odds_ou25_over", row.get("P>2.5", np.nan)),
                "odds_ou25_under": row.get("odds_ou25_under", row.get("P<2.5", np.nan)),
                **factors,
            }
            records.append(record)

        # Step 2: Update online state after the entire batch has been predicted
        engine.update_from_completed_batch(batch)

    return pd.DataFrame(records), engine
