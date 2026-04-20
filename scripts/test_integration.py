from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT / "src"))

from ai_accounting_assistant.pipeline import load_models, run_inference


def build_demo_input() -> pd.DataFrame:
    """Create small demo set for integration verification."""
    return pd.DataFrame(
        [
            {"date": "2025-08-02", "amount": -2200, "description": "monthly office rent", "category": "rent"},
            {"date": "2025-08-03", "amount": -320, "description": "internet bill", "category": "utilities"},
            {"date": "2025-08-07", "amount": 4800, "description": "customer payment received", "category": "income"},
            {"date": "2025-08-09", "amount": -910, "description": "social media ads", "category": "marketing"},
        ]
    )


def main() -> None:
    models = load_models()
    output = run_inference(build_demo_input(), models)
    tx = output["transactions_with_predictions"]
    forecast = output["next_month_cashflow_prediction"]

    print("Integration test completed.")
    print(tx[["description", "predicted_category", "anomaly_flag", "anomaly_score"]].to_string(index=False))
    print(f"Next month cashflow prediction: {forecast:.2f}")


if __name__ == "__main__":
    main()

