from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    mean_squared_error, mean_absolute_error, r2_score
)
import numpy as np

def evaluate_classification(y_true, y_pred, y_proba=None, logger=None):
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average="binary"),
        "recall": recall_score(y_true, y_pred, average="binary"),
        "f1": f1_score(y_true, y_pred, average="binary"),
    }
    if y_proba is not None:
        metrics["auc"] = roc_auc_score(y_true, y_proba)

    if logger:
        for k, v in metrics.items():
            logger.info(f"[classification] {k}: {v:.4f}")

    return metrics


def evaluate_regression(y_true, y_pred, logger=None):
    metrics = {
        "rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
        "mae": mean_absolute_error(y_true, y_pred),
        "r2": r2_score(y_true, y_pred),
    }

    if logger:
        for k, v in metrics.items():
            logger.info(f"[regression] {k}: {v:.4f}")

    return metrics

