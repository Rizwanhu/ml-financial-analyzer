from __future__ import annotations

from pathlib import Path
from typing import Any
import joblib
import numpy as np
import pandas as pd

from .data import clean_amount, load_mcc_mapping

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = PROJECT_ROOT / "artifacts" / "models"
MCC_JSON_PATH = PROJECT_ROOT / "finance_dataset" / "mcc_codes.json"


def load_classification_model(model_path: Path | None = None) -> Any:
    """
    Loads the trained Random Forest classifier.
    """
    path = model_path or (MODELS_DIR / "classification_model.joblib")
    if not path.exists():
        raise FileNotFoundError(f"Model file not found at {path}. Please run train script first.")
    return joblib.load(path)


def predict_mcc(model: Any, amounts: list[float] | np.ndarray | pd.Series) -> np.ndarray:
    """
    Predicts MCC codes for a list, numpy array, or pandas series of transaction amounts.
    """
    # Ensure inputs are formatted as a DataFrame feature block for the model
    X = pd.DataFrame({"amount": amounts})
    return model.predict(X)


def classify_transactions_df(
    df: pd.DataFrame,
    model: Any = None,
    *,
    model_path: Path | None = None,
    mcc_map_path: Path | None = None,
) -> pd.DataFrame:
    """
    Predicts MCC codes and maps them to Category Names for a dataframe.
    Requires an 'amount' column. Returns a new DataFrame with 'Predicted_MCC' and 'Category_Name'.
    """
    if "amount" not in df.columns:
        raise ValueError("DataFrame must contain an 'amount' column.")

    if model is None:
        model = load_classification_model(model_path)

    mcc_map = load_mcc_mapping(mcc_map_path or MCC_JSON_PATH)

    out = df.copy()
    cleaned_amounts = clean_amount(out["amount"])

    # Run prediction
    out["Predicted_MCC"] = predict_mcc(model, cleaned_amounts)

    # Map to human-readable names
    out["Category_Name"] = out["Predicted_MCC"].astype(str).map(mcc_map).fillna("Other/Misc Services")
    return out
