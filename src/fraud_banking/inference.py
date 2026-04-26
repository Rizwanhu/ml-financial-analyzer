from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from .config import DATASET_DIR, MODELS_DIR
from .data import load_cards, load_users
from .features import build_feature_frame


@dataclass(frozen=True)
class LoadedModel:
    pipeline: Any


def load_model(model_path: Path | None = None) -> LoadedModel:
    path = model_path or (MODELS_DIR / "fraud_pipeline.joblib")
    obj = joblib.load(path)
    return LoadedModel(pipeline=obj["pipeline"])


def predict_from_transactions_df(
    transactions_df: pd.DataFrame,
    *,
    dataset_dir: Path = DATASET_DIR,
    model_path: Path | None = None,
) -> pd.DataFrame:
    """
    Predict fraud probabilities for a dataframe in the same shape as transactions_data.csv.

    Enriches with user/card tables from the Kaggle dataset folder.
    Returns the input with added columns:
      - fraud_proba
      - fraud_pred (0/1 with threshold 0.5)
    """
    users = load_users(dataset_dir / "users_data.csv")
    cards = load_cards(dataset_dir / "cards_data.csv")
    model = load_model(model_path)

    X = build_feature_frame(transactions_df, users=users, cards=cards)
    proba = model.pipeline.predict_proba(X)[:, 1]

    out = transactions_df.copy()
    out["fraud_proba"] = proba
    out["fraud_pred"] = (out["fraud_proba"] >= 0.5).astype(int)
    return out

