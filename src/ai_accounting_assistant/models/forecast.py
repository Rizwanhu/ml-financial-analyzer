from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split


@dataclass
class ForecastResult:
    model_name: str
    metrics: Dict[str, float]
    model: RandomForestRegressor


FEATURES = ["lag_1", "lag_2", "rolling_mean_3"]
TARGET = "amount"


def train_cashflow_forecaster(monthly_df: pd.DataFrame, random_state: int = 42) -> ForecastResult:
    """
    Train monthly cashflow prediction model.

    This uses lag features and rolling means. For production forecasting,
    this can be replaced with ARIMA/Prophet/LSTM as needed.
    """
    if len(monthly_df) < 10:
        raise ValueError("Need at least 10 monthly records after lagging for stable training.")

    X = monthly_df[FEATURES]
    y = monthly_df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=random_state
    )

    model = RandomForestRegressor(n_estimators=300, random_state=random_state)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    rmse = float(np.sqrt(mean_squared_error(y_test, preds)))
    mae = float(mean_absolute_error(y_test, preds))
    return ForecastResult(
        model_name="random_forest_regressor",
        metrics={"rmse": rmse, "mae": mae},
        model=model,
    )


def forecast_next_month(model: RandomForestRegressor, lag_1: float, lag_2: float, rolling_mean_3: float) -> float:
    """Predict net cashflow for the next month from lag features."""
    X = pd.DataFrame(
        [{"lag_1": lag_1, "lag_2": lag_2, "rolling_mean_3": rolling_mean_3}]
    )
    return float(model.predict(X)[0])

