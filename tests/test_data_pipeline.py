"""Focused tests for data loading and preprocessing."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.loader import (
    build_master_dataset,
    load_bureau_data,
    load_previous_application_data,
)
from src.data.preprocessor import (
    Preprocessor,
    categorize_features,
    data_quality_report,
)


class LoaderTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        root = Path(self.temp_dir.name)
        self.application_path = root / "application.csv"
        self.bureau_path = root / "bureau.csv"
        self.previous_path = root / "previous.csv"

        pd.DataFrame(
            {"SK_ID_CURR": [1, 2, 3], "TARGET": [0, 1, 0]}
        ).to_csv(self.application_path, index=False)
        pd.DataFrame(
            {
                "SK_ID_CURR": [1, 1, 2],
                "AMT_CREDIT_SUM": [100.0, 300.0, 500.0],
                "CREDIT_DAY_OVERDUE": [0, 10, 0],
            }
        ).to_csv(self.bureau_path, index=False)
        pd.DataFrame(
            {
                "SK_ID_CURR": [1, 1, 2],
                "AMT_APPLICATION": [50.0, 150.0, 250.0],
                "NAME_CONTRACT_STATUS": ["Approved", "Refused", "Approved"],
            }
        ).to_csv(self.previous_path, index=False)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_aggregations(self) -> None:
        bureau = load_bureau_data(self.bureau_path).set_index("SK_ID_CURR")
        previous = load_previous_application_data(self.previous_path).set_index(
            "SK_ID_CURR"
        )

        self.assertEqual(bureau.loc[1, "bureau_loan_count"], 2)
        self.assertEqual(bureau.loc[1, "bureau_avg_credit_sum"], 200.0)
        self.assertEqual(bureau.loc[1, "bureau_overdue_count"], 1)
        self.assertEqual(previous.loc[1, "previous_application_count"], 2)
        self.assertEqual(previous.loc[1, "previous_avg_amount"], 100.0)
        self.assertEqual(previous.loc[1, "previous_approval_rate"], 0.5)

    def test_master_dataset_keeps_all_applicants(self) -> None:
        master = build_master_dataset(
            self.application_path, self.bureau_path, self.previous_path
        ).set_index("SK_ID_CURR")

        self.assertEqual(len(master), 3)
        self.assertEqual(master.loc[3, "bureau_loan_count"], 0)
        self.assertEqual(master.loc[3, "previous_application_count"], 0)


class PreprocessorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.frame = pd.DataFrame(
            {
                "SK_ID_CURR": [1, 2, 3, 4],
                "TARGET": [0, 0, 0, 1],
                "DAYS_EMPLOYED": [365243, -1000, -2000, -3000],
                "AMT_INCOME_TOTAL": [100.0, np.nan, 300.0, 500.0],
                "NAME_EDUCATION_TYPE": ["A", "A", None, "B"],
                "MOSTLY_MISSING": [np.nan, np.nan, np.nan, 1.0],
            }
        )

    def test_preprocessor_handles_retiree_placeholder(self) -> None:
        transformed = Preprocessor().fit_transform(self.frame)

        self.assertNotIn("SK_ID_CURR", transformed.columns)
        self.assertNotIn("TARGET", transformed.columns)
        self.assertNotIn("MOSTLY_MISSING", transformed.columns)
        self.assertIn("is_retiree_placeholder", transformed.columns)
        self.assertEqual(transformed["is_retiree_placeholder"].tolist(), [1, 0, 0, 0])
        self.assertFalse(transformed.isna().any().any())

    def test_quality_helpers(self) -> None:
        groups = categorize_features(self.frame)
        report = data_quality_report(self.frame)

        self.assertIn("AMT_INCOME_TOTAL", groups["financial"])
        self.assertEqual(report["columns"].loc["AMT_INCOME_TOTAL", "missing_percentage"], 25)
        self.assertEqual(report["target_imbalance_ratio"], 3.0)


class RiskBandTests(unittest.TestCase):
    """Band edges and the review margin are pure functions of the thresholds."""

    def test_bands_use_configured_thresholds(self) -> None:
        from src.ml.predict import risk_band
        from src.utils.config import RISK_THRESHOLD_HIGH, RISK_THRESHOLD_LOW

        self.assertEqual(risk_band(RISK_THRESHOLD_LOW - 0.01), "Low")
        self.assertEqual(risk_band(RISK_THRESHOLD_LOW), "Medium")
        self.assertEqual(risk_band(RISK_THRESHOLD_HIGH - 0.01), "Medium")
        self.assertEqual(risk_band(RISK_THRESHOLD_HIGH), "High")

    def test_borderline_flag_tracks_the_margin(self) -> None:
        from src.ml.predict import BORDERLINE_MARGIN, borderline_assessment
        from src.utils.config import RISK_THRESHOLD_LOW

        just_inside = borderline_assessment(RISK_THRESHOLD_LOW + BORDERLINE_MARGIN / 2)
        self.assertTrue(just_inside["borderline_review"])
        self.assertEqual(just_inside["nearest_threshold"], "low")

        well_clear = borderline_assessment(RISK_THRESHOLD_LOW / 2)
        self.assertFalse(well_clear["borderline_review"])


if __name__ == "__main__":
    unittest.main()
