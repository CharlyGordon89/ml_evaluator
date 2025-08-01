# ml_evaluator

**Modular, reusable evaluation module for ML pipelines.**  
Provides standardized performance metrics for classification and regression models in real-world ML projects.

---

## ✅ Features

- 📊 Classification metrics: Accuracy, Precision, Recall, F1 Score, AUC, Confusion Matrix
- 📈 Regression metrics: MAE, MSE, RMSE, R²
- 🔌 Plug-and-play interface: Works seamlessly with scikit-learn models and `y_true`, `y_pred`
- 📦 Reusab

# Example: Loading model + data from template paths
model = joblib.load("artifacts/models/model.pkl")
y_pred = model.predict(X_test)
metrics = evaluate_classification(y_test, y_pred)