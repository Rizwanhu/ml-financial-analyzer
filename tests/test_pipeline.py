from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd

# Allow tests to import from src/ and scripts/ directly.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT / "src"))
sys.path.append(str(PROJECT_ROOT))

from fraud_banking.features import build_feature_frame
from fraud_banking.inference import load_model


class TestFraudPipeline(unittest.TestCase):
    def test_feature_frame_builds(self) -> None:
        tx = pd.DataFrame(
            [
                {
                    "id": 1,
                    "date": "2010-01-01 00:01:00",
                    "client_id": 825,
                    "card_id": 4524,
                    "amount": "$-77.00",
                    "use_chip": "Swipe Transaction",
                    "merchant_id": 0,
                    "merchant_city": "X",
                    "merchant_state": "ND",
                    "zip": 58523,
                    "mcc": 5499,
                    "errors": None,
                }
            ]
        )
        users = pd.DataFrame(
            [
                {
                    "id": 825,
                    "current_age": 53,
                    "retirement_age": 66,
                    "gender": "Female",
                    "per_capita_income": "$29278",
                    "yearly_income": "$59696",
                    "total_debt": "$127613",
                    "credit_score": 787,
                    "num_credit_cards": 5,
                }
            ]
        )
        cards = pd.DataFrame(
            [
                {
                    "id": 4524,
                    "card_brand": "Visa",
                    "card_type": "Debit",
                    "has_chip": "YES",
                    "num_cards_issued": 2,
                    "credit_limit": "$24295",
                    "year_pin_last_changed": 2008,
                    "card_on_dark_web": "No",
                }
            ]
        )
        feats = build_feature_frame(tx, users=users, cards=cards)
        self.assertGreater(feats.shape[1], 5)
        self.assertEqual(len(feats), 1)

    def test_model_artifact_loads_if_present(self) -> None:
        # Only validates that loading works once the user trains.
        try:
            load_model()
        except FileNotFoundError:
            self.skipTest("Model not trained yet (run python scripts/train.py)")


if __name__ == "__main__":
    unittest.main()

