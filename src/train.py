import os
import yaml
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# ------------------ Load Params ------------------
with open("params.yaml", "r") as f:
    params = yaml.safe_load(f)

test_size = params["train"]["test_size"]
random_state = params["train"]["random_state"]
n_estimators = params["train"]["rf_n_estimators"]
max_depth = params["train"]["rf_max_depth"]

# MLflow experiment name
experiment_name = "UsedCarPricePrediction"

# ------------------ Paths ------------------
processed_dir = "data/processed"
models_dir = "models"
plots_dir = "saved_plots"
os.makedirs(models_dir, exist_ok=True)
os.makedirs(plots_dir, exist_ok=True)

# ------------------ Load Processed Data ------------------
X_train = pd.read_csv(os.path.join(processed_dir, "X_train.csv"))
X_test = pd.read_csv(os.path.join(processed_dir, "X_test.csv"))
y_train = pd.read_csv(os.path.join(processed_dir, "y_train.csv"))
y_test = pd.read_csv(os.path.join(processed_dir, "y_test.csv"))

y_train = y_train.values.ravel()
y_test = y_test.values.ravel()

# ------------------ MLflow Setup ------------------
mlflow.set_experiment(experiment_name)
models_r2 = {}

# ------------------ Linear Regression ------------------
with mlflow.start_run(run_name="Linear Regression"):
    model = LinearRegression()
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)

    mlflow.log_param("model", "Linear Regression")
    mlflow.log_metric("rmse", rmse)
    mlflow.log_metric("r2_score", r2)
    mlflow.sklearn.log_model(model, "linear_regression_model")

    pickle.dump(model, open(os.path.join(models_dir, "linear_regression.pkl"), "wb"))

    plt.figure()
    plt.scatter(y_test, preds, edgecolors='k', color='blue')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title("Linear Regression")
    plot_path = os.path.join(plots_dir, "lr_plot.png")
    plt.savefig(plot_path)
    plt.close()
    mlflow.log_artifact(plot_path, artifact_path="plots")

    models_r2["Linear Regression"] = r2

# ------------------ Random Forest ------------------
with mlflow.start_run(run_name="Random Forest Regressor") as run:
    model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=random_state)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)

    mlflow.log_param("model", "Random Forest Regressor")
    mlflow.log_param("n_estimators", n_estimators)
    mlflow.log_param("max_depth", max_depth)
    mlflow.log_metric("rmse", rmse)
    mlflow.log_metric("r2_score", r2)
    mlflow.sklearn.log_model(model, "random_forest_model")

    pickle.dump(model, open(os.path.join(models_dir, "random_forest.pkl"), "wb"))

    plt.figure()
    plt.scatter(y_test, preds, edgecolors='k', color='orange')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title("Random Forest")
    plot_path = os.path.join(plots_dir, "rf_plot.png")
    plt.savefig(plot_path)
    plt.close()
    mlflow.log_artifact(plot_path, artifact_path="plots")

    result = mlflow.register_model(
        model_uri=f"runs:/{run.info.run_id}/random_forest_model",
        name="RFCarPriceRegressor"
    )

    models_r2["Random Forest"] = r2

# ------------------ Summary ------------------
best_model = max(models_r2, key=models_r2.get)
print("\nTraining Complete!")
print(f"Best model based on R² score: {best_model}")
