from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd


_MONEY_RE = re.compile(r"[^0-9\.\-]+")


def parse_money(value: Any) -> float:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return float("nan")
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value)
    s = str(value).strip()
    if not s:
        return float("nan")
    s = _MONEY_RE.sub("", s)
    try:
        return float(s)
    except ValueError:
        return float("nan")


def safe_to_datetime(value: Any) -> pd.Timestamp | pd.NaT:
    try:
        return pd.to_datetime(value, errors="coerce")
    except Exception:
        return pd.NaT


@dataclass(frozen=True)
class FeatureSpec:
    numeric: tuple[str, ...]
    categorical: tuple[str, ...]


FEATURES = FeatureSpec(
    numeric=(
        "amount",
        "mcc",
        "zip",
        "hour",
        "day_of_week",
        "month",
        "current_age",
        "retirement_age",
        "per_capita_income",
        "yearly_income",
        "total_debt",
        "credit_score",
        "num_credit_cards",
        "num_cards_issued",
        "credit_limit",
        "year_pin_last_changed",
        "has_error",
    ),
    categorical=(
        "use_chip",
        "merchant_state",
        "gender",
        "card_brand",
        "card_type",
        "has_chip",
        "card_on_dark_web",
    ),
)


def build_feature_frame(
    transactions: pd.DataFrame,
    users: pd.DataFrame,
    cards: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build a model-ready feature frame from raw dataset tables.

    Intentionally avoids PII / high-cardinality identifiers:
    - excludes address, card_number, cvv, merchant_id, merchant_city, merchant_id
    """
    tx = transactions.copy()

    # Amount is in $ strings (e.g. "$-77.00")
    tx["amount"] = tx["amount"].map(parse_money)

    dt = tx["date"].map(safe_to_datetime)
    tx["hour"] = dt.dt.hour.astype("float64")
    tx["day_of_week"] = dt.dt.dayofweek.astype("float64")
    tx["month"] = dt.dt.month.astype("float64")

    # Errors is a string or NaN.
    tx["has_error"] = (~tx["errors"].isna()).astype("int64")

    # Standardize ids to int for merges.
    tx["client_id"] = pd.to_numeric(tx["client_id"], errors="coerce")
    tx["card_id"] = pd.to_numeric(tx["card_id"], errors="coerce")

    users_small = users[
        [
            "id",
            "current_age",
            "retirement_age",
            "gender",
            "per_capita_income",
            "yearly_income",
            "total_debt",
            "credit_score",
            "num_credit_cards",
        ]
    ].copy()
    users_small = users_small.rename(columns={"id": "client_id"})
    for c in ["per_capita_income", "yearly_income", "total_debt"]:
        users_small[c] = users_small[c].map(parse_money)

    cards_small = cards[
        [
            "id",
            "card_brand",
            "card_type",
            "has_chip",
            "num_cards_issued",
            "credit_limit",
            "year_pin_last_changed",
            "card_on_dark_web",
        ]
    ].copy()
    cards_small = cards_small.rename(columns={"id": "card_id"})
    cards_small["credit_limit"] = cards_small["credit_limit"].map(parse_money)

    merged = tx.merge(users_small, on="client_id", how="left").merge(cards_small, on="card_id", how="left")

    # Keep only the columns the model expects.
    keep_cols = list(FEATURES.numeric) + list(FEATURES.categorical)
    for col in keep_cols:
        if col not in merged.columns:
            merged[col] = np.nan
    return merged[keep_cols]

