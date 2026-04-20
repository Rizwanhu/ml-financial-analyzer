from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler


@dataclass
class AnomalyResult:
    model_name: str
    metrics: Dict[str, float]
    model: IsolationForest
    scaler: StandardScaler


def _build_features(df: pd.DataFrame) -> pd.DataFrame:
    return df[["amount", "month", "day_of_week", "is_month_end"]].copy()


def fit_isolation_forest(df: pd.DataFrame, contamination: float = 0.02, random_state: int = 42) -> AnomalyResult:
    """Train anomaly detector using Isolation Forest."""
    X = _build_features(df)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = IsolationForest(
        n_estimators=200,
        contamination=contamination,
        random_state=random_state,
    )
    model.fit(X_scaled)

    # Decision function > 0 generally indicates normal points.
    scores = model.decision_function(X_scaled)
    anomalies = model.predict(X_scaled)  # -1 for anomaly, +1 for normal
    anomaly_rate = float((anomalies == -1).mean())

    return AnomalyResult(
        model_name="isolation_forest",
        metrics={"mean_score": float(scores.mean()), "anomaly_rate": anomaly_rate},
        model=model,
        scaler=scaler,
    )


def detect_anomalies(df: pd.DataFrame, model: IsolationForest, scaler: StandardScaler) -> pd.DataFrame:
    """Apply trained anomaly model and return rows with model outputs."""
    X = _build_features(df)
    X_scaled = scaler.transform(X)
    out = df.copy()
    out["anomaly_flag"] = model.predict(X_scaled)
    out["anomaly_score"] = model.decision_function(X_scaled)
    return out

