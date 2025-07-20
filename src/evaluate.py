import pandas as pd
import joblib
import os
import json
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error

# Load test data
X_test = pd.read_csv("data/processed/X_test.csv")
y_test = pd.read_csv("data/processed/y_test.csv")
y_test = y_test.values.ravel()

# Load models
models = {
    "Linear Regression": "models/linear_regression.pkl",
    "Random Forest": "models/random_forest.pkl"
}

print("\nEvaluation on Test Set\n")

results = {}
for name, path in models.items():
    model = joblib.load(path)
    preds = model.predict(X_test)
    mse = mean_squared_error(y_test, preds)
    rmse = np.sqrt(mse)  #
    r2 = r2_score(y_test, preds)
    print(f"{name} -> RMSE: {rmse:.2f}, R²: {r2:.2f}")
    results[name] = {"rmse": round(rmse, 2), "r2": round(r2, 2)}

# Save metrics to JSON
with open("evaluation_metrics.json", "w") as f:
    json.dump(results, f, indent=2)

print("\nEvaluation metrics saved to 'evaluation_metrics.json'")
