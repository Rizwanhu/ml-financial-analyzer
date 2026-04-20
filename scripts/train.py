from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT / "src"))

from ai_accounting_assistant.config import PROCESSED_DATA_DIR, REPORT_DIR, ensure_directories
from ai_accounting_assistant.pipeline import train_all
from ai_accounting_assistant.utils.io import load_csv, save_csv


def generate_synthetic_transactions(rows: int = 2000, seed: int = 42) -> pd.DataFrame:
    """Generate synthetic accounting transactions for demo/training."""
    rng = np.random.default_rng(seed)
    categories = ["rent", "salary", "utilities", "travel", "marketing", "office"]
    templates = {
        "rent": ["monthly office rent", "warehouse lease payment", "building rent"],
        "salary": ["employee payroll", "consultant salary", "staff wage transfer"],
        "utilities": ["internet bill", "electricity payment", "water service charge"],
        "travel": ["flight ticket", "hotel booking", "client visit transport"],
        "marketing": ["social media ads", "campaign spend", "promotional expense"],
        "office": ["stationery purchase", "printer ink", "office supplies"],
    }

    dates = pd.date_range("2022-01-01", "2025-12-31", freq="D")
    chosen_dates = rng.choice(dates, size=rows, replace=True)
    chosen_categories = rng.choice(categories, size=rows, replace=True)

    records = []
    for i in range(rows):
        cat = chosen_categories[i]
        base_amount = {
            "rent": -2500,
            "salary": -1800,
            "utilities": -300,
            "travel": -700,
            "marketing": -900,
            "office": -150,
        }[cat]
        amount = float(base_amount + rng.normal(0, abs(base_amount) * 0.15))
        desc = rng.choice(templates[cat])
        records.append(
            {
                "date": pd.Timestamp(chosen_dates[i]).strftime("%Y-%m-%d"),
                "amount": amount,
                "description": desc,
                "category": cat,
            }
        )

    # Inject a few positive cash inflows to improve cashflow diversity.
    for _ in range(rows // 12):
        idx = rng.integers(0, rows)
        records[idx]["amount"] = float(abs(records[idx]["amount"]) * rng.uniform(1.5, 3.0))
        records[idx]["description"] = "customer payment received"
        records[idx]["category"] = "income"

    return pd.DataFrame(records)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train all models for AI Accounting Assistant.")
    parser.add_argument(
        "--data",
        type=str,
        default="",
        help="Optional path to a CSV dataset. If omitted, synthetic data is used.",
    )
    args = parser.parse_args()

    ensure_directories()
    if args.data:
        df = load_csv(Path(args.data))
    else:
        df = generate_synthetic_transactions()
        save_csv(df, PROCESSED_DATA_DIR / "synthetic_transactions.csv", index=False)

    artifacts = train_all(df)

    metrics_df = pd.DataFrame([artifacts.metrics])
    save_csv(metrics_df, REPORT_DIR / "training_metrics.csv", index=False)

    print("Training completed successfully.")
    print(metrics_df.to_string(index=False))
    print(f"Classification model: {artifacts.classification_pipeline_path}")
    print(f"Anomaly model: {artifacts.anomaly_model_path}")
    print(f"Forecast model: {artifacts.forecast_model_path}")


if __name__ == "__main__":
    main()

