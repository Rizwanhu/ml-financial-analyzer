from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd


def load_csv(path: Path) -> pd.DataFrame:
    """Load a CSV file with basic safety checks."""
    if not path.exists():
        raise FileNotFoundError(f"CSV file not found: {path}")
    return pd.read_csv(path)


def save_csv(df: pd.DataFrame, path: Path, index: bool = False) -> None:
    """Save a dataframe to CSV, creating parent folders when needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=index)


def optional_load_csv(path: Path) -> Optional[pd.DataFrame]:
    """Load CSV if it exists, otherwise return None."""
    if path.exists():
        return pd.read_csv(path)
    return None

