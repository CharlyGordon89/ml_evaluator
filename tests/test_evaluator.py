from ml_evaluator.evaluator import evaluate_classification, evaluate_regression

def test_classification():
    y_true = [0, 1, 1, 0]
    y_pred = [0, 1, 1, 1]
    y_proba = [0.1, 0.9, 0.8, 0.6]
    metrics = evaluate_classification(y_true, y_pred, y_proba)
    assert "accuracy" in metrics and "auc" in metrics

def test_regression():
    y_true = [3.0, 2.5, 4.0, 5.0]
    y_pred = [2.8, 2.7, 4.2, 5.1]
    metrics = evaluate_regression(y_true, y_pred)
    assert "rmse" in metrics and "r2" in metrics

