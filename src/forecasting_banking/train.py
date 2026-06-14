from __future__ import annotations

from pathlib import Path
import joblib
from sklearn.linear_model import LinearRegression

from .data import compute_daily_totals_streaming, generate_day_index_timeline

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = PROJECT_ROOT / "artifacts" / "models"


def train_forecasting(
    csv_path: Path,
    model_path: Path | None = None,
    *,
    chunksize: int = 250000,
) -> Path:
    """
    Train a Linear Regression forecaster model on daily aggregated transactions.
    Supports streaming chunking parameters to process huge files.
    """
    if model_path is None:
        model_path = MODELS_DIR / "forecaster.joblib"

    # 1. Aggregates to daily totals using streaming
    df_daily = compute_daily_totals_streaming(csv_path, chunksize=chunksize)

    if df_daily.empty:
        raise RuntimeError("No data available to train forecasting model.")

    # 2. Add chronological day_index sequence
    df_timeline = generate_day_index_timeline(df_daily)

    X = df_timeline[["day_index"]]
    y = df_timeline["amount"]

    # 3. Train Linear Regression model
    model = LinearRegression()
    model.fit(X, y)

    # 4. Save to artifacts/models/
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)
    return model_path
