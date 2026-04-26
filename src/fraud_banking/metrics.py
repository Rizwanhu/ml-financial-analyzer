from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


@dataclass(frozen=True)
class ClassificationReport:
    roc_auc: float
    pr_auc: float
    f1: float
    precision: float
    recall: float
    threshold: float
    confusion_matrix: list[list[int]]

    def to_dict(self) -> dict:
        return asdict(self)


def compute_report(y_true: np.ndarray, y_proba: np.ndarray, *, threshold: float = 0.5) -> ClassificationReport:
    y_pred = (y_proba >= threshold).astype(int)
    return ClassificationReport(
        roc_auc=float(roc_auc_score(y_true, y_proba)),
        pr_auc=float(average_precision_score(y_true, y_proba)),
        f1=float(f1_score(y_true, y_pred, zero_division=0)),
        precision=float(precision_score(y_true, y_pred, zero_division=0)),
        recall=float(recall_score(y_true, y_pred, zero_division=0)),
        threshold=float(threshold),
        confusion_matrix=confusion_matrix(y_true, y_pred).astype(int).tolist(),
    )

