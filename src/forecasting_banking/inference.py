from __future__ import annotations

from pathlib import Path
from typing import Any
import joblib
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = PROJECT_ROOT / "artifacts" / "models"


def load_forecaster(model_path: Path | None = None) -> Any:
    """
    Loads the trained Linear Regression forecaster.
    """
    path = model_path or (MODELS_DIR / "forecaster.joblib")
    if not path.exists():
        raise FileNotFoundError(f"Forecasting model file not found at {path}. Please run train script first.")
    return joblib.load(path)


def predict_future_spend(model: Any, last_index: int, steps: int = 10) -> np.ndarray:
    """
    Predicts spending for future day indices starting from last_index.
    """
    future_indices = pd.DataFrame({"day_index": range(last_index, last_index + steps)})
    return model.predict(future_indices)
