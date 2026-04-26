from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from .config import MODELS_DIR, REPORTS_DIR, ensure_directories
from .data import DatasetPaths, iter_labeled_transactions, load_cards, load_labels_map, load_users
from .features import FEATURES, build_feature_frame
from .metrics import compute_report


@dataclass(frozen=True)
class TrainResult:
    model_path: Path
    report_path: Path


def _build_pipeline() -> Pipeline:
    numeric_features = list(FEATURES.numeric)
    categorical_features = list(FEATURES.categorical)

    numeric = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )
    categorical = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=10)),
        ]
    )
    pre = ColumnTransformer(
        transformers=[
            ("num", numeric, numeric_features),
            ("cat", categorical, categorical_features),
        ]
    )

    clf = LogisticRegression(
        max_iter=2000,
        n_jobs=-1,
        class_weight="balanced",
        solver="saga",
    )

    return Pipeline(steps=[("preprocess", pre), ("model", clf)])


def train_fraud_model(
    paths: DatasetPaths,
    *,
    max_labeled_rows: int = 1_000_000,
    chunksize: int = 250_000,
    random_state: int = 42,
) -> TrainResult:
    ensure_directories()

    users = load_users(paths.users_csv)
    cards = load_cards(paths.cards_csv)
    labels_map = load_labels_map(paths.labels_json)

    # Stream labeled rows; keep a capped sample for local training speed.
    X_parts: list[pd.DataFrame] = []
    y_parts: list[np.ndarray] = []
    kept = 0

    for tx_chunk in iter_labeled_transactions(
        paths.transactions_csv,
        labels_map,
        chunksize=chunksize,
    ):
        y = tx_chunk["is_fraud"].to_numpy(dtype=int, copy=True)
        feats = build_feature_frame(tx_chunk, users=users, cards=cards)
        X_parts.append(feats)
        y_parts.append(y)
        kept += len(feats)
        if kept >= max_labeled_rows:
            break

    if not X_parts:
        raise RuntimeError("No labeled transactions were found. Check dataset files and labels JSON.")

    X = pd.concat(X_parts, axis=0, ignore_index=True)
    y_all = np.concatenate(y_parts, axis=0)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y_all,
        test_size=0.2,
        random_state=random_state,
        stratify=y_all,
    )

    pipe = _build_pipeline()
    pipe.fit(X_train, y_train)

    y_proba = pipe.predict_proba(X_test)[:, 1]
    report = compute_report(y_test, y_proba, threshold=0.5)

    model_path = MODELS_DIR / "fraud_pipeline.joblib"
    report_path = REPORTS_DIR / "fraud_metrics.json"

    joblib.dump(
        {
            "pipeline": pipe,
            "feature_spec": {"numeric": list(FEATURES.numeric), "categorical": list(FEATURES.categorical)},
        },
        model_path,
    )
    report_path.write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")

    return TrainResult(model_path=model_path, report_path=report_path)

