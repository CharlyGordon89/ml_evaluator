from typing import List, Dict, Optional, Literal, Union
from sklearn.base import BaseEstimator
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
    mean_absolute_error, mean_squared_error, r2_score
)


def evaluate_model(
    model: BaseEstimator,
    X: np.ndarray,
    y: np.ndarray,
    task: Literal["classification", "regression"],
    metrics: Optional[list] = None,
    average: str = "macro",
    logger: Optional[object] = None) -> dict:
    
    """
    Unified evaluation entry point for scikit-learn models.

    Args:
        model: Trained scikit-learn model
        X: Features
        y: Ground truth labels/values
        task: Either 'classification' or 'regression'
        metrics: Optional list of metrics
        average: Averaging method for multi-class metrics
        logger: Optional logger

    Returns:
        Dict of evaluated metrics
    """
    y_pred = model.predict(X)
    y_prob = None

    if task == "classification" and hasattr(model, "predict_proba"):
        try:
            y_prob = model.predict_proba(X)
        except Exception:
            pass

        results = evaluate_classification(y, y_pred, y_prob=y_prob, metrics=metrics, average=average)

    elif task == "regression":
        results = evaluate_regression(y, y_pred, metrics=metrics)

    else:
        raise ValueError(f"Unsupported task type: {task}")

    if logger:
        logger.log_metrics(results)

    return results


def evaluate_classification(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None,
    metrics: Optional[List[str]] = None,
    average: str = "macro") -> Dict[str, float | list]:
    
    """
    Evaluate classification predictions using selected metrics.
    Supports multi-class and binary classification.

    Args:
        y_true: Ground truth labels.
        y_pred: Predicted class labels.
        y_prob: Predicted probabilities (for AUC).
        metrics: List of metric names to compute.
        average: Averaging method for multi-class metrics.

    Returns:
        Dictionary of metric results.
    """
    metrics = metrics or ["accuracy", "precision", "recall", "f1", "auc", "confusion_matrix"]
    results = {}

    if len(y_true) != len(y_pred):
        raise ValueError("Length of y_true and y_pred must match.")

    if "accuracy" in metrics:
        results["accuracy"] = accuracy_score(y_true, y_pred)

    if "precision" in metrics:
        results["precision"] = precision_score(y_true, y_pred, average=average, zero_division=0)

    if "recall" in metrics:
        results["recall"] = recall_score(y_true, y_pred, average=average, zero_division=0)

    if "f1" in metrics:
        results["f1"] = f1_score(y_true, y_pred, average=average, zero_division=0)

    if "auc" in metrics:
        if y_prob is None:
            raise ValueError("AUC requires probability predictions (y_prob).")
        results["auc"] = roc_auc_score(y_true, y_prob, multi_class="ovr" if len(set(y_true)) > 2 else "raise")

    if "confusion_matrix" in metrics:
        results["confusion_matrix"] = confusion_matrix(y_true, y_pred).tolist()

    return results


def evaluate_regression(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metrics: Optional[List[str]] = None) -> Dict[str, float]:
    
    """
    Evaluate regression predictions using selected metrics.

    Args:
        y_true: Ground truth values.
        y_pred: Predicted values.
        metrics: List of metric names to compute.

    Returns:
        Dictionary of metric results.
    """
    metrics = metrics or ["mae", "mse", "rmse", "r2"]
    results = {}

    if len(y_true) != len(y_pred):
        raise ValueError("Length of y_true and y_pred must match.")

    if "mae" in metrics:
        results["mae"] = mean_absolute_error(y_true, y_pred)

    if "mse" in metrics:
        results["mse"] = mean_squared_error(y_true, y_pred)

    if "rmse" in metrics:
        results["rmse"] = mean_squared_error(y_true, y_pred, squared=False)

    if "r2" in metrics:
        results["r2"] = r2_score(y_true, y_pred)

    return results
