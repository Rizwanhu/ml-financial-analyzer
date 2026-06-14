from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT / "src"))

from forecasting_banking.train import train_forecasting


def main() -> None:
    parser = argparse.ArgumentParser(description="Train spending forecasting model.")
    parser.add_argument(
        "--chunksize",
        type=int,
        default=250000,
        help="CSV chunksize for streaming the large transactions file.",
    )
    args = parser.parse_args()

    csv_path = PROJECT_ROOT / "finance_dataset" / "transactions_data.csv"
    model_path = PROJECT_ROOT / "artifacts" / "models" / "forecaster.joblib"

    print("Starting forecasting model training on daily totals...")
    saved_path = train_forecasting(
        csv_path,
        model_path=model_path,
        chunksize=args.chunksize,
    )
    print(f"✅ Re-trained on Daily Totals! Forecaster saved to {saved_path}")


if __name__ == "__main__":
    main()