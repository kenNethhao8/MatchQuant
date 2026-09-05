import tempfile
import unittest
from pathlib import Path

import pandas as pd

from matchquant.fixtures import FixtureValidationError, load_fixtures, validate_fixtures


def fixture_frame():
    return pd.DataFrame(
        [
            ["f1", "2024-01-07", "2024-25", "A", "B"],
            ["f2", "2024-01-07T12:30:00", "2024-25", "C", "D"],
        ],
        columns=["fixture_id", "date", "season", "home_team", "away_team"],
    )


class FixtureSchemaTests(unittest.TestCase):
    def test_valid_fixture_data_preserves_batch_metadata(self):
        result = validate_fixtures(fixture_frame())
        self.assertEqual(result["fixture_id"].tolist(), ["f1", "f2"])
        self.assertEqual(result["_date_only"].tolist(), [True, False])
        self.assertIn("_timestamp", result)

    def test_missing_empty_and_invalid_values_are_rejected(self):
        with self.assertRaisesRegex(FixtureValidationError, "missing required fixture columns"):
            validate_fixtures(fixture_frame().drop(columns=["season"]))
        with self.assertRaisesRegex(FixtureValidationError, "missing values"):
            bad = fixture_frame()
            bad.loc[0, "home_team"] = ""
            validate_fixtures(bad)
        with self.assertRaisesRegex(FixtureValidationError, "invalid dates"):
            bad = fixture_frame()
            bad.loc[0, "date"] = "not-a-date"
            validate_fixtures(bad)

    def test_duplicate_ids_duplicate_fixtures_and_same_team_are_rejected(self):
        duplicate_id = pd.concat([fixture_frame(), fixture_frame().iloc[[0]]], ignore_index=True)
        with self.assertRaisesRegex(FixtureValidationError, "duplicate fixture_id"):
            validate_fixtures(duplicate_id)

        duplicate_fixture = fixture_frame()
        duplicate_fixture.loc[1, ["fixture_id", "date", "home_team", "away_team"]] = [
            "f3",
            "2024-01-07",
            "A",
            "B",
        ]
        with self.assertRaisesRegex(FixtureValidationError, "duplicate fixtures"):
            validate_fixtures(duplicate_fixture)

        same_team = fixture_frame()
        same_team.loc[0, "away_team"] = "a"
        with self.assertRaisesRegex(FixtureValidationError, "cannot be the same"):
            validate_fixtures(same_team)

    def test_known_scores_are_rejected_but_empty_template_score_columns_are_allowed(self):
        empty_scores = fixture_frame()
        empty_scores["home_goals"] = ""
        empty_scores["away_goals"] = ""
        self.assertEqual(len(validate_fixtures(empty_scores)), 2)
        known_score = fixture_frame()
        known_score["home_goals"] = [1, ""]
        with self.assertRaisesRegex(FixtureValidationError, "known scores"):
            validate_fixtures(known_score)
        canonical_marker = fixture_frame()
        canonical_marker["match_id"] = ["m1", "m2"]
        with self.assertRaisesRegex(FixtureValidationError, "canonical match_id"):
            validate_fixtures(canonical_marker)

    def test_example_fixture_csv_loads(self):
        fixtures = load_fixtures(Path("examples/synthetic_fixtures.csv"))
        self.assertEqual(len(fixtures), 3)
        self.assertEqual(set(fixtures["season"]), {"2024-25"})

    def test_missing_fixture_file_is_explicit(self):
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaises(FileNotFoundError):
                load_fixtures(Path(temporary) / "missing.csv")


if __name__ == "__main__":
    unittest.main()
