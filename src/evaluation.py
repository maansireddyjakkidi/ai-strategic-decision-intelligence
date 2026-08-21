from typing import Dict

import numpy as np

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def evaluate_predictions(
    y_true,
    y_pred,
    y_probability,
) -> Dict[str, float]:
    """
    Calculate the evaluation metrics used in the study.

    Returns
    -------
    dict
        Accuracy, precision, recall, F1 and ROC-AUC.
    """

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(
            y_true,
            y_pred,
            zero_division=0,
        ),
        "recall": recall_score(
            y_true,
            y_pred,
            zero_division=0,
        ),
        "f1": f1_score(
            y_true,
            y_pred,
            zero_division=0,
        ),
        "roc_auc": roc_auc_score(
            y_true,
            y_probability,
        ),
    }


def calculate_percentage_change(
    baseline: float,
    degraded: float,
) -> float:
    """
    Calculate percentage change from a baseline metric.

    A negative value indicates performance degradation.
    """

    if baseline == 0:
        return np.nan

    return ((degraded - baseline) / baseline) * 100