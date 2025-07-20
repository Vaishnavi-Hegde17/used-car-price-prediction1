import os
import pandas as pd
import numpy as np
import yaml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

# ------------------- Load Params -------------------
with open("params.yaml", "r") as f:
    params = yaml.safe_load(f)

test_size = params["train"]["test_size"]
random_state = params["train"]["random_state"]
scaling_enabled = params.get("scaling", {}).get("enabled", False)

# ------------------- Paths -------------------
raw_data_path = "data/cardata.csv"
processed_path = "data/processed"
os.makedirs(processed_path, exist_ok=True)

# ------------------- Load Dataset -------------------
df = pd.read_csv(raw_data_path)

# ------------------- Preprocessing -------------------
le = LabelEncoder()
df["Fuel_Type"] = le.fit_transform(df["Fuel_Type"])
df["Seller_Type"] = le.fit_transform(df["Seller_Type"])
df["Transmission"] = le.fit_transform(df["Transmission"])

X = df[["Year", "Present_Price", "Kms_Driven", "Fuel_Type", "Seller_Type", "Transmission", "Owner"]]
y = df["Selling_Price"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size, random_state=random_state
)

# Scaling logic (used for training but not saved)
if scaling_enabled:
    scaler = StandardScaler()
    _ = scaler.fit_transform(X_train)
    _ = scaler.transform(X_test)

# ------------------- Save Processed Data -------------------
X_train.to_csv(os.path.join(processed_path, "X_train.csv"), index=False)
X_test.to_csv(os.path.join(processed_path, "X_test.csv"), index=False)
y_train.to_csv(os.path.join(processed_path, "y_train.csv"), index=False)
y_test.to_csv(os.path.join(processed_path, "y_test.csv"), index=False)

print("\n Preprocessing Complete.. Data saved in 'data/processed/'")
