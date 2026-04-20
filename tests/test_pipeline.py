from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd

# Allow tests to import from src/ and scripts/ directly.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT / "src"))
sys.path.append(str(PROJECT_ROOT))

from ai_accounting_assistant.preprocessing import add_time_features, clean_transactions, create_forecast_series
from scripts.train import generate_synthetic_transactions


class TestPreprocessing(unittest.TestCase):
    def test_clean_and_time_features(self) -> None:
        df = pd.DataFrame(
            [
                {"date": "2025-01-01", "amount": -100.5, "description": "Rent", "category": "rent"},
                {"date": "2025-01-02", "amount": "-50", "description": "Power Bill", "category": "utilities"},
            ]
        )
        cleaned = clean_transactions(df)
        prepared = add_time_features(cleaned)
        self.assertIn("month", prepared.columns)
        self.assertIn("day_of_week", prepared.columns)
        self.assertEqual(len(prepared), 2)

    def test_forecast_series_creation(self) -> None:
        df = generate_synthetic_transactions(rows=600, seed=1)
        cleaned = clean_transactions(df)
        prepared = add_time_features(cleaned)
        monthly = create_forecast_series(prepared)
        self.assertGreater(len(monthly), 10)
        self.assertTrue({"lag_1", "lag_2", "rolling_mean_3", "amount"}.issubset(monthly.columns))


if __name__ == "__main__":
    unittest.main()

