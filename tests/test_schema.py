import unittest

import pandas as pd

from matchquant.io import load_matches
from matchquant.schema import DataValidationError, validate_matches


class SchemaTests(unittest.TestCase):
    def setUp(self):
        self.valid = pd.DataFrame(
            [
                ["m1", "2024-01-01", "S1", "A", "B", 1, 0],
                ["m2", "2024-01-02T12:30:00", "S1", "B", "C", 0, 0],
            ],
            columns=["match_id", "date", "season", "home_team", "away_team", "home_goals", "away_goals"],
        )

    def test_valid_data_is_normalized(self):
        result = validate_matches(self.valid)
        self.assertEqual(result["home_goals"].tolist(), [1, 0])
        self.assertEqual(result["away_goals"].tolist(), [0, 0])
        self.assertEqual(result["_date_only"].tolist(), [True, False])

    def test_required_columns_are_enforced(self):
        with self.assertRaisesRegex(DataValidationError, "missing required columns"):
            validate_matches(self.valid.drop(columns=["season"]))

    def test_invalid_values_are_rejected(self):
        bad = self.valid.copy()
        bad.loc[0, "home_goals"] = -1
        with self.assertRaisesRegex(DataValidationError, "non-negative integer"):
            validate_matches(bad)

        bad = self.valid.copy()
        bad.loc[0, "away_team"] = "A"
        with self.assertRaisesRegex(DataValidationError, "cannot be the same"):
            validate_matches(bad)

        bad = self.valid.copy()
        bad.loc[0, "date"] = "not-a-date"
        with self.assertRaisesRegex(DataValidationError, "invalid dates"):
            validate_matches(bad)

        bad = self.valid.copy()
        bad.loc[0, "season"] = ""
        with self.assertRaisesRegex(DataValidationError, "missing values"):
            validate_matches(bad)

    def test_empty_dataset_is_rejected(self):
        with self.assertRaisesRegex(DataValidationError, "dataset is empty"):
            validate_matches(self.valid.iloc[0:0])

    def test_duplicate_id_and_fixture_are_rejected(self):
        duplicate_id = pd.concat([self.valid, self.valid.iloc[[0]]], ignore_index=True)
        with self.assertRaisesRegex(DataValidationError, "duplicate match_id"):
            validate_matches(duplicate_id)

        duplicate_fixture = self.valid.copy()
        duplicate_fixture.loc[1, "match_id"] = "m3"
        duplicate_fixture.loc[1, "date"] = duplicate_fixture.loc[0, "date"]
        duplicate_fixture.loc[1, "home_team"] = duplicate_fixture.loc[0, "home_team"]
        duplicate_fixture.loc[1, "away_team"] = duplicate_fixture.loc[0, "away_team"]
        with self.assertRaisesRegex(DataValidationError, "duplicate fixtures"):
            validate_matches(duplicate_fixture)

    def test_example_csv_loads(self):
        frame = load_matches("examples/synthetic_matches.csv")
        self.assertEqual(len(frame), 12)
        self.assertEqual(frame["season"].nunique(), 2)
