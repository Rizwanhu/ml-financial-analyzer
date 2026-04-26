from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT / "src"))

from fraud_banking.inference import predict_from_transactions_df


def main() -> None:
    demo = pd.DataFrame(
        [
            {
                "id": 1,
                "date": "2010-01-01 00:01:00",
                "client_id": 825,
                "card_id": 4524,
                "amount": "$-77.00",
                "use_chip": "Swipe Transaction",
                "merchant_id": 12345,
                "merchant_city": "Beulah",
                "merchant_state": "ND",
                "zip": 58523,
                "mcc": 5499,
                "errors": None,
            }
        ]
    )
    try:
        scored = predict_from_transactions_df(demo)
        print("Integration test completed.")
        print(scored[["id", "fraud_proba", "fraud_pred"]].to_string(index=False))
    except FileNotFoundError:
        print("Model artifact not found. Train first: python scripts/train.py")


if __name__ == "__main__":
    main()

