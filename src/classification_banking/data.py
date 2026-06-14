from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator
import pandas as pd


def clean_amount(series: pd.Series) -> pd.Series:
    """
    Clean currency amount strings (e.g., '$77.00', '($77.00)') into floats.
    Handles '$', ',', ')', '(' (converting to negative), and empty values safely.
    """
    cleaned = series.astype(str)
    # Replicates original replace logic:
    cleaned = cleaned.str.replace(r"[\$,)]", "", regex=True)
    cleaned = cleaned.str.replace(r"[(]", "-", regex=True)
    cleaned = cleaned.replace(r"^\s*$", "0", regex=True)
    return pd.to_numeric(cleaned, errors="coerce").fillna(0.0)


def load_mcc_mapping(mcc_json_path: Path) -> dict[str, str]:
    """
    Loads merchant category codes mapping from JSON file.
    """
    if not mcc_json_path.exists():
        return {}
    with mcc_json_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def iter_classification_data(
    csv_path: Path,
    *,
    chunksize: int = 250_000,
    max_rows: int | None = None,
) -> Iterator[pd.DataFrame]:
    """
    Streams and cleans transaction data for classification.
    Yields dataframes containing:
      - 'amount': cleaned float values
      - 'mcc': merchant category codes
    """
    seen = 0
    # Stream in chunks
    for chunk in pd.read_csv(csv_path, chunksize=chunksize):
        if "amount" not in chunk.columns or "mcc" not in chunk.columns:
            raise ValueError("Transactions CSV must contain 'amount' and 'mcc' columns.")

        cleaned_chunk = pd.DataFrame({
            "amount": clean_amount(chunk["amount"]),
            "mcc": chunk["mcc"]
        })
        yield cleaned_chunk

        seen += len(chunk)
        if max_rows is not None and seen >= max_rows:
            break
