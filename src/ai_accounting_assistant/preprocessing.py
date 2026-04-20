from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd


REQUIRED_COLUMNS: List[str] = [
    "date",
    "amount",
    "description",
    "category",
]


def validate_schema(df: pd.DataFrame) -> None:
    """Ensure required columns are present before model operations."""
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def clean_transactions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and normalize transaction data.

    This function intentionally remains simple so students can extend it.
    """
    validate_schema(df)
    out = df.copy()

    # Standardize text and dates for downstream feature engineering.
    out["description"] = out["description"].fillna("").astype(str).str.strip().str.lower()
    out["category"] = out["category"].fillna("unknown").astype(str).str.strip().str.lower()
    out["date"] = pd.to_datetime(out["date"], errors="coerce")

    # Handle missing or invalid numeric values robustly.
    out["amount"] = pd.to_numeric(out["amount"], errors="coerce")
    out["amount"] = out["amount"].fillna(out["amount"].median())

    # Drop rows with invalid dates because time features are needed later.
    out = out.dropna(subset=["date"]).reset_index(drop=True)
    return out


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add calendar features for both classification and forecasting tasks."""
    out = df.copy()
    out["year"] = out["date"].dt.year
    out["month"] = out["date"].dt.month
    out["day"] = out["date"].dt.day
    out["day_of_week"] = out["date"].dt.dayofweek
    out["is_month_end"] = out["date"].dt.is_month_end.astype(int)
    return out


def create_forecast_series(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build monthly cashflow time series from transaction-level data.

    Positive and negative values are preserved in `amount`, so the monthly
    sum represents net cash flow for each month.
    """
    out = df.copy()
    out["month_start"] = out["date"].values.astype("datetime64[M]")
    monthly = out.groupby("month_start", as_index=False)["amount"].sum()
    monthly = monthly.sort_values("month_start").reset_index(drop=True)
    monthly["lag_1"] = monthly["amount"].shift(1)
    monthly["lag_2"] = monthly["amount"].shift(2)
    monthly["rolling_mean_3"] = monthly["amount"].rolling(3).mean()
    monthly = monthly.dropna().reset_index(drop=True)

    # Keep float for model compatibility.
    for col in ["amount", "lag_1", "lag_2", "rolling_mean_3"]:
        monthly[col] = monthly[col].astype(np.float64)
    return monthly

