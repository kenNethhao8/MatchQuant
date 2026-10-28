"""Tests for Layer 5: Agent narrative report rendering and follow-up Q&A (T14 & Stage 3)."""

from __future__ import annotations

import json
import unittest

from matchquant.agent_report import (
    answer_followup,
    build_evidence_catalog,
    render_agent_narrative_html,
    render_agent_narrative_markdown,
)


class AgentReportTests(unittest.TestCase):
    def setUp(self) -> None:
        self.mock_report_json = {
            "schema_version": "single-match-report-v1",
            "run_id": "test_run_20261026_01",
            "status": "ok",
            "fixture": {
                "home_team": "Arsenal <Gunners>",
                "away_team": "Chelsea & Blues",
                "date": "2026-05-10",
                "season": "2025-26",
            },
            "temporal": {
                "as_of": "2026-05-10",
                "used_data_cutoff": "2026-05-03",
                "data_age_days_at_as_of": 7,
                "forecast_horizon_days": 0,
                "mode": "research",
            },
            "predictions": {
                "track_a": {
                    "p_home": 0.524,
                    "p_draw": 0.256,
                    "p_away": 0.220,
                    "lambda_home": 1.85,
                    "lambda_away": 1.05,
                    "p_over25": 0.548,
                },
                "components": {
                    "elo": {
                        "elo_home": 1780.0,
                        "elo_away": 1690.0,
                        "p_home": 0.490,
                        "p_draw": 0.260,
                        "p_away": 0.250,
                    }
                },
            },
            "factors": {
                "elo_diff": 90.0,
                "sot_dominance_diff": 1.82,
                "venue_specialist_diff": 0.45,
                "expected_tempo": 2.75,
            },
            "recent_evidence": {
                "home": {
                    "sample_count": 6,
                    "record_count_by_outcome": {"W": 4, "D": 1, "L": 1},
                    "goals_for": {"mean": 2.1},
                    "goals_against": {"mean": 0.9},
                    "shots_on_target_for": {"mean": 5.8},
                    "corners_for": {"mean": 6.2},
                },
                "away": {
                    "sample_count": 6,
                    "record_count_by_outcome": {"W": 3, "D": 2, "L": 1},
                    "goals_for": {"mean": 1.5},
                    "goals_against": {"mean": 1.1},
                    "shots_on_target_for": {"mean": 4.5},
                    "corners_for": {"mean": 4.8},
                },
            },
            "market_comparison": {
                "provided": True,
                "available": True,
                "input": {
                    "source": "Pinnacle Closing/Open",
                    "observed_at": "2026-05-09",
                    "odds_open_h": 1.95,
                    "odds_open_d": 3.60,
                    "odds_open_a": 4.00,
                    "mkt_ah_line": -0.50,
                },
                "market_open_probabilities": {"home": 0.500, "draw": 0.260, "away": 0.240},
                "track_a_minus_market_percentage_points": {"home": 2.4, "draw": -0.4, "away": -2.0},
                "value_edges": {
                    "home": 0.0218,
                    "draw": -0.0784,
                    "away": -0.1200,
                    "disclaimer": "仅供量化研究参考，不构成投注建议",
                },
                "divergence_alerts": {
                    "has_divergence": False,
                    "threshold_percentage_points": 7.0,
                    "alerts": [],
                },
                "handicap_assessment": {
                    "mkt_ah_line": -0.50,
                    "interpretation": "主队让 0.50 球",
                    "convention": "主队让球为负",
                },
                "timeline_analysis": {
                    "available": True,
                    "status": "multi_snapshot_trend",
                    "snapshot_count": 3,
                    "drift_summary": {
                        "p_home_drift_pct_points": 1.8,
                        "p_draw_drift_pct_points": -0.5,
                        "p_away_drift_pct_points": -1.3,
                        "ah_line_drift": 0.0,
                        "steam_direction": "home",
                        "steam_description": "市场聪明钱向主队流动，主胜赔率持续下压（2.05 -> 1.95）。",
                    },
                },
                "m_open_available": False,
                "m_open_unavailable_reason": "历史训练分区缺少 M-Open 对应特征",
            },
        }

    def test_t14_evidence_catalog_extraction(self) -> None:
        """T14: Evidence catalog extraction provides structured IDs and data."""
        catalog = build_evidence_catalog(self.mock_report_json)
        self.assertIn("E1_PROBS", catalog)
        self.assertIn("E2_ELO", catalog)
        self.assertIn("E3_FACTORS", catalog)
        self.assertIn("E4_RECENT_FORM", catalog)
        self.assertIn("E5_MARKET", catalog)
        self.assertIn("E6_TEMPORAL", catalog)

        e1 = catalog["E1_PROBS"]["data"]
        self.assertAlmostEqual(e1["p_home"], 0.524)
        self.assertAlmostEqual(e1["lambda_home"], 1.85)

        e5 = catalog["E5_MARKET"]
        self.assertTrue(e5["available"])
        self.assertEqual(e5["source"], "Pinnacle Closing/Open")
        self.assertEqual(e5["observed_at"], "2026-05-09")
        self.assertEqual(e5["odds"]["home"], 1.95)
        self.assertAlmostEqual(e5["market_open_probabilities"]["home"], 0.500)
        self.assertAlmostEqual(e5["track_a_minus_market_percentage_points"]["home"], 2.4)
        self.assertEqual(e5["value_edges"]["home"], 0.0218)
        self.assertEqual(e5["handicap_assessment"]["mkt_ah_line"], -0.50)
        self.assertEqual(e5["timeline_analysis"]["snapshot_count"], 3)

    def test_t14_markdown_narrative_length_and_slot_consistency(self) -> None:
        """T14: Markdown narrative has appropriate character length and matches authoritative values."""
        md = render_agent_narrative_markdown(self.mock_report_json)
        self.assertGreater(len(md), 600)
        self.assertLess(len(md), 3500)
        self.assertIn("52.4%", md)
        self.assertIn("25.6%", md)
        self.assertIn("22.0%", md)
        self.assertIn("1.85", md)
        self.assertIn("1.05", md)
        self.assertIn("[E1]", md)
        self.assertIn("[E3]", md)
        self.assertIn("[E4]", md)
        self.assertIn("[E5]", md)
        self.assertIn("[E6]", md)

    def test_t14_html_escaping_and_self_contained_structure(self) -> None:
        """T14: HTML rendering escapes special characters and produces valid self-contained HTML."""
        html_out = render_agent_narrative_html(self.mock_report_json)
        self.assertIn("&lt;Gunners&gt;", html_out)
        self.assertIn("&amp; Blues", html_out)
        self.assertNotIn("<Gunners>", html_out)
        self.assertTrue(html_out.startswith("<!DOCTYPE html>"))
        self.assertIn("</html>", html_out)
        self.assertIn("52.4%", html_out)

    def test_t14_followup_qa_reuses_run_without_recomputation(self) -> None:
        """T14: Follow-up questions reuse existing run data and reference evidence IDs."""
        res_why = answer_followup(self.mock_report_json, "为什么看好主队？")
        self.assertFalse(res_why["recomputed"])
        self.assertIn("E1", res_why["referenced_evidence_ids"])
        self.assertIn("52.4%", res_why["answer"])

        res_goals = answer_followup(self.mock_report_json, "两队进球预期是多少？")
        self.assertFalse(res_goals["recomputed"])
        self.assertIn("1.85", res_goals["answer"])
        self.assertIn("1.05", res_goals["answer"])

        res_short = answer_followup(self.mock_report_json, "给我一个简短总结")
        self.assertFalse(res_short["recomputed"])
        self.assertIn("简版核心结论", res_short["answer"])

    def test_markdown_and_html_chapter_and_numbers_consistency(self) -> None:
        """Verify that Markdown and HTML have consistent 6 chapters and bit-exact numbers."""
        catalog = build_evidence_catalog(self.mock_report_json)
        md = render_agent_narrative_markdown(self.mock_report_json, catalog)
        html_out = render_agent_narrative_html(self.mock_report_json, catalog)

        # 1. Verify all 6 chapters exist in both Markdown and HTML
        chapters = [
            ("一、 核心量化结论与概率分布 `[E1]`", "一、 核心概率分布与泊松进球期望 [E1]"),
            ("二、 核心特征驱动与对位解析 `[E3]`", "二、 核心量化特征驱动 [E3]"),
            ("三、 近期战绩与场面技术统计 `[E4]`", "三、 近期战绩与场面技术统计 [E4]"),
            ("四、 市场参照与让球对位 `[E5]`", "四、 市场参照与让球对位 [E5]"),
            ("五、 数据版本与时序截断回执 `[E6]`", "五、 时序截断与数据回执 [E6]"),
            ("六、 客观局限性与免责声明", "六、 客观局限性与免责声明"),
        ]
        for md_chap, html_chap in chapters:
            self.assertIn(md_chap, md)
            self.assertIn(html_chap, html_out)

        # 2. Check exact matching numbers across both
        # Track A: 52.4%, 25.6%, 22.0%
        for val in ("52.4%", "25.6%", "22.0%"):
            self.assertIn(val, md)
            self.assertIn(val, html_out)

        # Market Probabilities: 50.0%, 26.0%, 24.0%
        for val in ("50.0%", "26.0%", "24.0%"):
            self.assertIn(val, md)
            self.assertIn(val, html_out)

        # Percentage point diff: +2.4 个百分点
        self.assertIn("+2.4 个百分点", md)
        self.assertIn("+2.4 个百分点", html_out)

        # Value Edge: 0.0218
        self.assertIn("0.0218", md)
        self.assertIn("0.0218", html_out)

        # Handicap: 主队让 0.50 球
        self.assertIn("主队让 0.50 球", md)
        self.assertIn("主队让 0.50 球", html_out)

        # Steam Drift: 聪明钱向主队流动
        self.assertIn("聪明钱向主队流动", md)
        self.assertIn("聪明钱向主队流动", html_out)

        # Disclaimer exists in both
        self.assertIn("不构成投注建议", md)
        self.assertIn("不构成投注建议", html_out)

    def test_no_market_honest_downgrade_consistency(self) -> None:
        """When market data is missing, both MD and HTML must honestly display Track A only mode."""
        no_mkt_json = dict(self.mock_report_json)
        no_mkt_json["market_comparison"] = {
            "provided": False,
            "available": False,
        }

        catalog = build_evidence_catalog(no_mkt_json)
        md = render_agent_narrative_markdown(no_mkt_json, catalog)
        html_out = render_agent_narrative_html(no_mkt_json, catalog)

        # Must state Track A Only
        self.assertIn("纯基本面独立推断模式", md)
        self.assertIn("纯基本面独立推断模式", html_out)
        self.assertIn("双轨隔离原则", md)
        self.assertIn("双轨隔离原则", html_out)

        # E5 chapter header must still exist
        self.assertIn("四、 市场参照与让球对位 `[E5]`", md)
        self.assertIn("四、 市场参照与让球对位 [E5]", html_out)

    def test_divergence_alert_rendering(self) -> None:
        """When divergence alert is triggered, both MD and HTML must display the warning alert."""
        div_json = json.loads(json.dumps(self.mock_report_json))
        div_json["market_comparison"]["divergence_alerts"] = {
            "has_divergence": True,
            "threshold_percentage_points": 7.0,
            "alerts": [
                {
                    "outcome": "home",
                    "outcome_label": "主胜",
                    "model_prob_pct": 52.4,
                    "market_prob_pct": 40.0,
                    "diff_percentage_points": 12.4,
                    "direction": "高估",
                    "note": "基本面指标显著高于市场共识。",
                }
            ],
        }

        catalog = build_evidence_catalog(div_json)
        md = render_agent_narrative_markdown(div_json, catalog)
        html_out = render_agent_narrative_html(div_json, catalog)

        self.assertIn("显著背离预警", md)
        self.assertIn("显著背离预警", html_out)
        self.assertIn("+12.4 个百分点", md)
        self.assertIn("+12.4 个百分点", html_out)
        self.assertIn("基本面指标显著高于市场共识", md)
        self.assertIn("基本面指标显著高于市场共识", html_out)

    def test_followup_market_qa_intents_without_recomputation(self) -> None:
        """Test market-related followup questions hit E5 and never recompute."""
        # Query 1: How to read the market / handicap
        res_odds = answer_followup(self.mock_report_json, "盘口和让球怎么看？")
        self.assertFalse(res_odds["recomputed"])
        self.assertIn("E5", res_odds["referenced_evidence_ids"])
        self.assertIn("50.0%", res_odds["answer"])
        self.assertIn("主队让 0.50 球", res_odds["answer"])

        # Query 2: Why model differs from market / Divergence & Edge
        res_diff = answer_followup(self.mock_report_json, "为什么模型和盘口差这么多？有背离吗？")
        self.assertFalse(res_diff["recomputed"])
        self.assertIn("E5", res_diff["referenced_evidence_ids"])
        self.assertIn("+2.4 个百分点", res_diff["answer"])
        self.assertIn("0.0218", res_diff["answer"])
        self.assertIn("背离", res_diff["answer"])

        # Query 3: Odds timeline / Steam drift trend
        res_trend = answer_followup(self.mock_report_json, "赔率变动走势和聪明钱趋势如何？")
        self.assertFalse(res_trend["recomputed"])
        self.assertIn("E5", res_trend["referenced_evidence_ids"])
        self.assertIn("聪明钱向主队流动", res_trend["answer"])
        self.assertIn("+1.8 个百分点", res_trend["answer"])

        # Query 4: Ask to refresh live odds -> guiding prompt without refitting
        res_refresh = answer_followup(self.mock_report_json, "帮我刷新最新盘口")
        self.assertFalse(res_refresh["recomputed"])
        self.assertIn("E5", res_refresh["referenced_evidence_ids"])
        self.assertIn("只读已有 Run 证据", res_refresh["answer"])
        self.assertIn("use_live_odds", res_refresh["answer"])

    def test_market_rounding_and_snapshot_wording_consistency(self) -> None:
        """Verify that table and alert have consistent rounding (e.g. 23.7479% -> 23.7%, not 23.8%) and wording."""
        rep = json.loads(json.dumps(self.mock_report_json))
        raw_mkt_away = 0.23747862629058455  # 23.7479%
        rep["market_comparison"]["market_open_probabilities"]["away"] = raw_mkt_away
        rep["market_comparison"]["track_a_minus_market_percentage_points"]["away"] = 10.54
        rep["market_comparison"]["divergence_alerts"] = {
            "has_divergence": True,
            "threshold_percentage_points": 8.0,
            "alerts": [
                {
                    "outcome": "away",
                    "outcome_label": "客胜",
                    "model_prob_pct": 34.29,
                    "market_prob_pct": 23.7479,  # 4 decimals or matched with m_prob
                    "diff_percentage_points": 10.54,
                    "direction": "高估",
                    "note": "测试背离提示",
                }
            ],
        }

        catalog = build_evidence_catalog(rep)
        md = render_agent_narrative_markdown(rep, catalog)
        html_out = render_agent_narrative_html(rep, catalog)

        # 1. Wording verification: "录入外部市场实时快照" must be present, "录入外部市场初盘数据" must not
        self.assertIn("录入外部市场实时快照", md)
        self.assertIn("录入外部市场实时快照", html_out)
        self.assertNotIn("录入外部市场初盘数据", md)
        self.assertNotIn("录入外部市场初盘数据", html_out)

        # 2. Rounding consistency: both table and alert must show 23.7% and never 23.8%
        self.assertIn("23.7%", md)
        self.assertIn("23.7%", html_out)
        self.assertNotIn("23.8%", md)
        self.assertNotIn("23.8%", html_out)


if __name__ == "__main__":
    unittest.main()
