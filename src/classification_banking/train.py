from __future__ import annotations

from pathlib import Path
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from .data import iter_classification_data

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = PROJECT_ROOT / "artifacts" / "models"


def train_classification(
    csv_path: Path,
    model_path: Path | None = None,
    *,
    max_rows: int = 10000,
    chunksize: int = 250000,
    random_state: int = 42,
) -> Path:
    """
    Train a Random Forest Classifier on transaction amounts to predict MCC codes.
    Uses data streaming chunking parameters to limit memory usage.
    """
    if model_path is None:
        model_path = MODELS_DIR / "classification_model.joblib"

    # Stream and accumulate chunks of cleaned data up to max_rows
    chunks: list[pd.DataFrame] = []
    total_loaded = 0
    for chunk in iter_classification_data(csv_path, chunksize=chunksize, max_rows=max_rows):
        chunks.append(chunk)
        total_loaded += len(chunk)
        if total_loaded >= max_rows:
            break

    if not chunks:
        raise RuntimeError("No transaction data loaded for training.")

    df = pd.concat(chunks, axis=0, ignore_index=True).head(max_rows)

    X = df[["amount"]]
    y = df["mcc"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state
    )

    model = RandomForestClassifier(n_estimators=10, random_state=random_state)
    model.fit(X_train, y_train)

    # Save trained model to artifacts/models/ folder
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)
    return model_path
