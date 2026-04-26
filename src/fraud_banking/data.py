from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import pandas as pd


@dataclass(frozen=True)
class DatasetPaths:
    transactions_csv: Path
    users_csv: Path
    cards_csv: Path
    labels_json: Path


def default_dataset_paths(dataset_dir: Path) -> DatasetPaths:
    return DatasetPaths(
        transactions_csv=dataset_dir / "transactions_data.csv",
        users_csv=dataset_dir / "users_data.csv",
        cards_csv=dataset_dir / "cards_data.csv",
        labels_json=dataset_dir / "train_fraud_labels.json",
    )


def load_users(users_csv: Path) -> pd.DataFrame:
    return pd.read_csv(users_csv)


def load_cards(cards_csv: Path) -> pd.DataFrame:
    return pd.read_csv(cards_csv)


def load_labels_map(labels_json: Path) -> dict[str, str]:
    """
    Kaggle file format: {"target": { "<transaction_id>": "Yes"/"No", ... } }
    """
    with labels_json.open("r", encoding="utf-8") as f:
        data = json.load(f)
    target = data.get("target", {})
    if not isinstance(target, dict):
        raise ValueError("Unexpected train_fraud_labels.json format: 'target' is not a dict")
    return target


def iter_labeled_transactions(
    transactions_csv: Path,
    labels_map: dict[str, str],
    *,
    chunksize: int = 250_000,
    max_rows: int | None = None,
) -> Iterator[pd.DataFrame]:
    """
    Stream labeled transactions from the huge CSV.

    Yields chunks containing:
    - all original tx columns
    - label column "is_fraud" (0/1)
    """
    seen = 0
    for chunk in pd.read_csv(transactions_csv, chunksize=chunksize):
        # Use string keys to match labels_map.
        ids = chunk["id"].astype(str)
        labels = ids.map(labels_map.get)
        # Keep only rows that have labels (train labels may not cover all ids).
        mask = labels.notna()
        labeled = chunk.loc[mask].copy()
        labeled["is_fraud"] = (labels[mask].astype(str).str.lower() == "yes").astype("int64")

        if not labeled.empty:
            yield labeled

        seen += len(chunk)
        if max_rows is not None and seen >= max_rows:
            break

