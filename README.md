# 📊 ml_evaluator

**Modular, reusable evaluation engine for ML pipelines.**  
Provides standardized evaluation metrics for classification and regression models — seamlessly integrated with scikit-learn and ready for extension to deep learning or time series workflows.

---

## ✅ Features

- 🧠 **Classification**: Accuracy, Precision, Recall, F1 Score, AUC, Confusion Matrix
- 📈 **Regression**: MAE, MSE, RMSE, R²
- 🧪 **Unified interface** via `evaluate_model()` — abstracts task-specific logic
- 🔌 **Plug-and-play**: Works with any scikit-learn-compatible model
- 🛡️ Robust input validation and flexible config (metric selection, `average`)
- 🔄 Ready for integration with logging, experiment tracking, and MLOps tools

---

## 🚀 Installation

```bash
git clone https://github.com/your-org/ml_evaluator.git
cd ml_evaluator
pip install -e .
```

---

## 🧠 Usage

### Classification Example

```python
from ml_evaluator import evaluate_classification
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification

X, y = make_classification(n_samples=200, n_features=4, random_state=42)
model = LogisticRegression().fit(X, y)

y_pred = model.predict(X)
y_prob = model.predict_proba(X)

metrics = evaluate_classification(
    y_true=y,
    y_pred=y_pred,
    y_prob=y_prob,
    metrics=["accuracy", "precision", "recall", "f1", "auc", "confusion_matrix"]
)

print(metrics)
```

### Regression Example

```python
from ml_evaluator import evaluate_regression
from sklearn.linear_model import LinearRegression
from sklearn.datasets import make_regression

X, y = make_regression(n_samples=200, n_features=4, random_state=42)
model = LinearRegression().fit(X, y)

y_pred = model.predict(X)

metrics = evaluate_regression(y_true=y, y_pred=y_pred, metrics=["mae", "mse", "rmse", "r2"])
print(metrics)
```

### Unified Interface

```python
from ml_evaluator import evaluate_model

metrics = evaluate_model(model, X, y, task="classification", metrics=["accuracy", "auc"])
```

---

## 🧪 Testing

```bash
pytest -v tests/
```

---

## 🛠️ Extensibility

- Add new metric support via a future `metric registry` (planned)
- Easily extend for deep learning or time series models
- Integrate with `ml-logger` to auto-log evaluation results

---

## 📦 Requirements

```
scikit-learn>=1.1,<1.4
numpy>=1.21
```

---

## 📄 License

MIT License © 2025 Ruslan Mamedov
