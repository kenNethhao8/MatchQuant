"""Tests for Layer 5: Agent narrative report rendering and follow-up Q&A (T14)."""

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
                "available": True,
                "market_open_probabilities": {"home": 0.500, "draw": 0.260, "away": 0.240},
                "track_a_minus_market_percentage_points": {"home": 2.4, "draw": -0.4, "away": -2.0},
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

    def test_t14_markdown_narrative_length_and_slot_consistency(self) -> None:
        """T14: Markdown narrative has appropriate character length and matches authoritative values."""
        md = render_agent_narrative_markdown(self.mock_report_json)

        # Check character count (Chinese narrative around 800 - 1500 chars)
        self.assertGreater(len(md), 600)
        self.assertLess(len(md), 2500)

        # Check that authoritative numbers are rendered exactly
        self.assertIn("52.4%", md)
        self.assertIn("25.6%", md)
        self.assertIn("22.0%", md)
        self.assertIn("1.85", md)
        self.assertIn("1.05", md)
        self.assertIn("[E1]", md)
        self.assertIn("[E3]", md)
        self.assertIn("[E6]", md)

    def test_t14_html_escaping_and_self_contained_structure(self) -> None:
        """T14: HTML rendering escapes special characters and produces valid self-contained HTML."""
        html_out = render_agent_narrative_html(self.mock_report_json)

        # Check escaping of <Gunners> and & Blues
        self.assertIn("&lt;Gunners&gt;", html_out)
        self.assertIn("&amp; Blues", html_out)
        self.assertNotIn("<Gunners>", html_out)

        # Check structural elements
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


if __name__ == "__main__":
    unittest.main()
