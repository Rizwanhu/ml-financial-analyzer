from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import pandas as pd
from joblib import dump, load

from ai_accounting_assistant.config import MODEL_DIR, ensure_directories
from ai_accounting_assistant.models.anomaly import detect_anomalies, fit_isolation_forest
from ai_accounting_assistant.models.classification import (
    train_advanced_random_forest,
    train_baseline_logistic,
)
from ai_accounting_assistant.models.forecast import forecast_next_month, train_cashflow_forecaster
from ai_accounting_assistant.preprocessing import add_time_features, clean_transactions, create_forecast_series


@dataclass
class TrainingArtifacts:
    classification_pipeline_path: Path
    anomaly_model_path: Path
    anomaly_scaler_path: Path
    forecast_model_path: Path
    metrics: Dict[str, float]


def train_all(df: pd.DataFrame) -> TrainingArtifacts:
    """Train all project models and persist artifacts."""
    ensure_directories()

    cleaned = clean_transactions(df)
    prepared = add_time_features(cleaned)

    baseline = train_baseline_logistic(prepared)
    advanced = train_advanced_random_forest(prepared)
    anomaly = fit_isolation_forest(prepared)
    monthly = create_forecast_series(prepared)
    forecast = train_cashflow_forecaster(monthly)

    classification_path = MODEL_DIR / "expense_classifier.joblib"
    anomaly_path = MODEL_DIR / "anomaly_detector.joblib"
    scaler_path = MODEL_DIR / "anomaly_scaler.joblib"
    forecast_path = MODEL_DIR / "cashflow_forecaster.joblib"

    # Persist advanced classifier by default because it usually performs better.
    dump(advanced.pipeline, classification_path)
    dump(anomaly.model, anomaly_path)
    dump(anomaly.scaler, scaler_path)
    dump(forecast.model, forecast_path)

    metrics = {
        "baseline_accuracy": baseline.metrics["accuracy"],
        "advanced_accuracy": advanced.metrics["accuracy"],
        "anomaly_rate": anomaly.metrics["anomaly_rate"],
        "forecast_rmse": forecast.metrics["rmse"],
        "forecast_mae": forecast.metrics["mae"],
    }

    return TrainingArtifacts(
        classification_pipeline_path=classification_path,
        anomaly_model_path=anomaly_path,
        anomaly_scaler_path=scaler_path,
        forecast_model_path=forecast_path,
        metrics=metrics,
    )


def load_models() -> Dict[str, object]:
    """Load all persisted models for app/runtime inference."""
    return {
        "classifier": load(MODEL_DIR / "expense_classifier.joblib"),
        "anomaly_model": load(MODEL_DIR / "anomaly_detector.joblib"),
        "anomaly_scaler": load(MODEL_DIR / "anomaly_scaler.joblib"),
        "forecast_model": load(MODEL_DIR / "cashflow_forecaster.joblib"),
    }


def run_inference(df: pd.DataFrame, models: Dict[str, object]) -> Dict[str, object]:
    """Run full integrated inference: category, anomaly, and forecast."""
    cleaned = clean_transactions(df)
    prepared = add_time_features(cleaned)

    classifier = models["classifier"]
    pred_df = prepared.copy()
    pred_df["predicted_category"] = classifier.predict(
        pred_df[["description", "amount", "month", "day_of_week"]]
    )

    anomaly_out = detect_anomalies(pred_df, models["anomaly_model"], models["anomaly_scaler"])

    monthly = create_forecast_series(prepared)
    if monthly.empty:
        next_month_forecast = 0.0
    else:
        latest = monthly.iloc[-1]
        next_month_forecast = forecast_next_month(
            model=models["forecast_model"],
            lag_1=float(latest["lag_1"]),
            lag_2=float(latest["lag_2"]),
            rolling_mean_3=float(latest["rolling_mean_3"]),
        )

    return {
        "transactions_with_predictions": anomaly_out,
        "next_month_cashflow_prediction": next_month_forecast,
    }

