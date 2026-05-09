import os
import pickle
import yaml
import numpy as np
import pandas as pd
import scipy.sparse as sp

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, OrdinalEncoder
from sklearn.model_selection import train_test_split

# Params load
params = yaml.safe_load(open("params.yaml"))

# Preprocessing: Load raw data, transform features, and save processed data
def preprocessing():
    df = pd.read_csv("data/raw/churn.csv")

    print(f"Data Loaded — Shape: {df.shape}")

    # ── Fix TotalCharges ──────────────────────────────────────────────────────
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    median_total_charges = df["TotalCharges"].median()
    df["TotalCharges"] = df["TotalCharges"].fillna(median_total_charges)

    # ── Log-transform skewed column ───────────────────────────────────────────
    df["TotalCharges"] = np.log1p(df["TotalCharges"])

    # ── Features / Target ─────────────────────────────────────────────────────
    x = df.drop(columns=["Churn", "customerID"])
    y = df["Churn"].map({"Yes": 1, "No": 0})

    print(f"Features: {x.shape} | Target: {y.shape}")

    # ── Column groups ─────────────────────────────────────────────────────────
    binary_cols = [
        "gender",
        "Partner",
        "Dependents",
        "PhoneService",
        "PaperlessBilling"
    ]

    categorical_cols = [
        "MultipleLines",
        "InternetService",
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies",
        "Contract",
        "PaymentMethod",
    ]

    numeric_cols = [
        "tenure",
        "MonthlyCharges",
        "TotalCharges"
    ]

    pass_through_cols = ["SeniorCitizen"]

    # ── Preprocessor ──────────────────────────────────────────────────────────
    preprocessor = ColumnTransformer([
        ("binary", OrdinalEncoder(), binary_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
        ("num", StandardScaler(), numeric_cols),
        ("pass", "passthrough", pass_through_cols)
    ])

    x = preprocessor.fit_transform(x)

    print(f"After preprocessing — Shape: {x.shape}")

    # ── Train-test split ─────────────────────────────────────────────────────
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=params["data"]["test_size"],
        random_state=params["data"]["random_state"],
        stratify=y
    )

    print(f"Train: {x_train.shape} | Test: {x_test.shape}")

    # ── Processed data save ──────────────────────────────────────────────────
    os.makedirs("data/processed", exist_ok=True)

    np.save("data/processed/x_train.npy", x_train)
    np.save("data/processed/x_test.npy", x_test)

    y_train.to_csv("data/processed/y_train.csv", index=False)
    y_test.to_csv("data/processed/y_test.csv", index=False)

    # ── Preprocessor save ────────────────────────────────────────────────────
    os.makedirs("data_&_model", exist_ok=True)

    with open("data_&_model/preprocessor.pkl", "wb") as f:
        pickle.dump(preprocessor, f)

    print("Processed data saved to data/processed/")
    print("Preprocessor saved to data_&_model/preprocessor.pkl")


if __name__ == "__main__":
    preprocessing()