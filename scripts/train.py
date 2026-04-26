from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT / "src"))

from fraud_banking.config import DATASET_DIR
from fraud_banking.data import default_dataset_paths
from fraud_banking.train import train_fraud_model


def main() -> None:
    parser = argparse.ArgumentParser(description="Train fraud detection model on Kaggle dataset.")
    parser.add_argument(
        "--max_labeled_rows",
        type=int,
        default=1_000_000,
        help="Cap labeled rows kept for training (for speed/memory).",
    )
    parser.add_argument(
        "--chunksize",
        type=int,
        default=250_000,
        help="CSV chunksize for streaming the large transactions file.",
    )
    args = parser.parse_args()

    paths = default_dataset_paths(DATASET_DIR)
    result = train_fraud_model(paths, max_labeled_rows=args.max_labeled_rows, chunksize=args.chunksize)
    print("Training completed successfully.")
    print(f"Model saved to: {result.model_path}")
    print(f"Metrics saved to: {result.report_path}")


if __name__ == "__main__":
    main()

