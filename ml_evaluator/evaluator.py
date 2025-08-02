from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    mean_squared_error, mean_absolute_error, r2_score
)
import numpy as np

def evaluate_classification(y_true, y_pred, y_proba=None, logger=None):
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have same length")
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
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have same length")
    metrics = {
        "rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
        "mae": mean_absolute_error(y_true, y_pred),
        "r2": r2_score(y_true, y_pred),
    }

    if logger:
        for k, v in metrics.items():
            logger.info(f"[regression] {k}: {v:.4f}")

    return metrics

def evaluate_model(model, X, y, task: str = "classification", logger=None):
    y_pred = model.predict(X)
    y_proba = None

    if task == "classification" and hasattr(model, "predict_proba"):
        try:
            y_proba = model.predict_proba(X)[:, 1]
        except Exception:
            pass

        return evaluate_classification(y, y_pred, y_proba, logger=logger)

    elif task == "regression":
        return evaluate_regression(y, y_pred, logger=logger)

    else:
        raise ValueError(f"Unsupported task: {task}")
