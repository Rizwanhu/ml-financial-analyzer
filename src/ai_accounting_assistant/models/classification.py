from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


FEATURE_COLUMNS: List[str] = ["description", "amount", "month", "day_of_week"]
TARGET_COLUMN: str = "category"


@dataclass
class ClassificationResult:
    model_name: str
    metrics: Dict[str, float]
    report: str
    pipeline: Pipeline


def _build_preprocessor() -> ColumnTransformer:
    text_features = Pipeline(
        steps=[
            ("tfidf", TfidfVectorizer(max_features=300, ngram_range=(1, 2))),
        ]
    )

    categorical_features = OneHotEncoder(handle_unknown="ignore")

    return ColumnTransformer(
        transformers=[
            ("text", text_features, "description"),
            ("categorical", categorical_features, ["month", "day_of_week"]),
            ("amount", "passthrough", ["amount"]),
        ]
    )


def train_baseline_logistic(df: pd.DataFrame, random_state: int = 42) -> ClassificationResult:
    """Baseline expense classifier with Logistic Regression."""
    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state, stratify=y
    )

    pipe = Pipeline(
        steps=[
            ("preprocessor", _build_preprocessor()),
            ("model", LogisticRegression(max_iter=1000, n_jobs=None)),
        ]
    )
    pipe.fit(X_train, y_train)
    preds = pipe.predict(X_test)

    return ClassificationResult(
        model_name="logistic_regression",
        metrics={"accuracy": float(accuracy_score(y_test, preds))},
        report=classification_report(y_test, preds, zero_division=0),
        pipeline=pipe,
    )


def train_advanced_random_forest(df: pd.DataFrame, random_state: int = 42) -> ClassificationResult:
    """Advanced expense classifier with Random Forest."""
    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state, stratify=y
    )

    pipe = Pipeline(
        steps=[
            ("preprocessor", _build_preprocessor()),
            ("model", RandomForestClassifier(n_estimators=200, random_state=random_state)),
        ]
    )
    pipe.fit(X_train, y_train)
    preds = pipe.predict(X_test)

    return ClassificationResult(
        model_name="random_forest",
        metrics={"accuracy": float(accuracy_score(y_test, preds))},
        report=classification_report(y_test, preds, zero_division=0),
        pipeline=pipe,
    )

