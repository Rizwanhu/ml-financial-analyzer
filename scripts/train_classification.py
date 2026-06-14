from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT / "src"))

from classification_banking.train import train_classification


def main() -> None:
    parser = argparse.ArgumentParser(description="Train transaction classification model.")
    parser.add_argument(
        "--max_rows",
        type=int,
        default=10000,
        help="Cap transaction rows kept for training (for speed/memory).",
    )
    parser.add_argument(
        "--chunksize",
        type=int,
        default=250000,
        help="CSV chunksize for streaming the large transactions file.",
    )
    args = parser.parse_args()

    csv_path = PROJECT_ROOT / "finance_dataset" / "transactions_data.csv"
    model_path = PROJECT_ROOT / "artifacts" / "models" / "classification_model.joblib"

    print("Starting classification model training...")
    saved_path = train_classification(
        csv_path,
        model_path=model_path,
        max_rows=args.max_rows,
        chunksize=args.chunksize,
    )
    print(f"✅ Classification model saved to {saved_path}")


if __name__ == "__main__":
    main()