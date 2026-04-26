from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import FastAPI

# Allow imports from src/ when running from repo root.
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT / "src"))

from fraud_banking.inference import predict_from_transactions_df


app = FastAPI(title="Fraud Detection API", version="1.0.0")


@app.get("/health")
def health() -> dict[str, Any]:
    return {"ok": True}


@app.post("/predict")
def predict(payload: dict[str, Any]) -> dict[str, Any]:
    """
    Expects a single transaction in Kaggle `transactions_data.csv` shape.
    Returns fraud probability + prediction.
    """
    df = pd.DataFrame([payload])
    scored = predict_from_transactions_df(df)
    row = scored.iloc[0]
    return {
        "fraud_proba": float(row["fraud_proba"]),
        "fraud_pred": int(row["fraud_pred"]),
    }

