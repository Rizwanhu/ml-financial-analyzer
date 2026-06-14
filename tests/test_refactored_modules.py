from __future__ import annotations

import sys
import unittest
from pathlib import Path
import pandas as pd

# Setup paths to import from src/
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT / "src"))

from classification_banking import (
    classify_transactions_df,
    clean_amount as clean_amount_class,
    predict_mcc,
)
from forecasting_banking import (
    clean_amount as clean_amount_fore,
    generate_day_index_timeline,
    predict_future_spend,
)


class TestRefactoredModules(unittest.TestCase):
    def test_classification_amount_cleaning(self) -> None:
        amounts = pd.Series(["$12.34", "-$45.67", "($100.00)", "invalid", ""])
        cleaned = clean_amount_class(amounts)
        self.assertEqual(cleaned[0], 12.34)
        self.assertEqual(cleaned[1], -45.67)
        self.assertEqual(cleaned[2], -100.00)
        self.assertEqual(cleaned[3], 0.0)
        self.assertEqual(cleaned[4], 0.0)

    def test_forecasting_amount_cleaning(self) -> None:
        amounts = pd.Series(["$12.34", "-$45.67", "invalid", ""])
        cleaned = clean_amount_fore(amounts)
        self.assertEqual(cleaned[0], 12.34)
        self.assertEqual(cleaned[1], -45.67)
        self.assertEqual(cleaned[2], 0.0)
        self.assertEqual(cleaned[3], 0.0)

    def test_forecasting_day_index_timeline(self) -> None:
        df = pd.DataFrame({
            "date_only": pd.to_datetime(["2026-06-01", "2026-06-02", "2026-06-03"]).date,
            "amount": [10.0, 20.0, 30.0]
        })
        timeline = generate_day_index_timeline(df)
        self.assertIn("day_index", timeline.columns)
        self.assertListEqual(list(timeline["day_index"]), [0, 1, 2])

    def test_classification_prediction_runs(self) -> None:
        # Test classifier loading and execution since we trained the model
        model_path = PROJECT_ROOT / "artifacts" / "models" / "classification_model.joblib"
        if model_path.exists():
            from classification_banking import load_classification_model
            model = load_classification_model(model_path)
            predictions = predict_mcc(model, [100.0, 200.0])
            self.assertEqual(len(predictions), 2)

            df = pd.DataFrame({"amount": ["$100.00", "$200.00"]})
            classified_df = classify_transactions_df(df, model=model)
            self.assertIn("Predicted_MCC", classified_df.columns)
            self.assertIn("Category_Name", classified_df.columns)
            self.assertEqual(len(classified_df), 2)
        else:
            self.skipTest("Classification model not trained yet")

    def test_forecasting_prediction_runs(self) -> None:
        # Test forecaster loading and execution since we trained the model
        model_path = PROJECT_ROOT / "artifacts" / "models" / "forecaster.joblib"
        if model_path.exists():
            from forecasting_banking import load_forecaster
            model = load_forecaster(model_path)
            predictions = predict_future_spend(model, last_index=10, steps=5)
            self.assertEqual(len(predictions), 5)
        else:
            self.skipTest("Forecasting model not trained yet")


if __name__ == "__main__":
    unittest.main()
