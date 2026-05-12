from dotenv import load_dotenv
import os
import sys
import warnings
import logging
import shutil

# ─────────────────────────────────────────────────────────────
# UTF-8 Fix
# ─────────────────────────────────────────────────────────────
os.environ['PYTHONIOENCODING'] = 'utf-8'

if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

# ─────────────────────────────────────────────────────────────
# Load Environment Variables
# ─────────────────────────────────────────────────────────────
load_dotenv()

# Secure Environment Variables
os.environ["MLFLOW_TRACKING_USERNAME"] = os.getenv("MLFLOW_TRACKING_USERNAME", "")
os.environ["MLFLOW_TRACKING_PASSWORD"] = os.getenv("MLFLOW_TRACKING_PASSWORD", "")
os.environ["DAGSHUB_TOKEN"] = os.getenv("DAGSHUB_TOKEN", "")

# ─────────────────────────────────────────────────────────────
# Suppress Warnings & Logs
# ─────────────────────────────────────────────────────────────
warnings.filterwarnings("ignore")
logging.getLogger("mlflow").setLevel(logging.ERROR)

# ─────────────────────────────────────────────────────────────
# Imports
# ─────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
import yaml
import dagshub
import mlflow
import mlflow.sklearn

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    f1_score,
    precision_score,
    recall_score
)

# ─────────────────────────────────────────────────────────────
# DagsHub Authentication
# ─────────────────────────────────────────────────────────────
dagshub.auth.add_app_token(os.environ["DAGSHUB_TOKEN"])

dagshub.init(
    repo_owner="Muneebkhan1457",
    repo_name="Teleco-Customer-Churn-",
    mlflow=True
)

# MLflow Tracking URI
mlflow.set_tracking_uri(
    "https://dagshub.com/Muneebkhan1457/Teleco-Customer-Churn-.mlflow"
)

# ─────────────────────────────────────────────────────────────
# Load Parameters
# ─────────────────────────────────────────────────────────────
with open("params.yaml", "r") as f:
    params = yaml.safe_load(f)


def train():

    # ─────────────────────────────────────────────────────────
    # Load Data
    # ─────────────────────────────────────────────────────────
    X_train = np.load("data/processed/x_train.npy")
    X_test = np.load("data/processed/x_test.npy")

    y_train = pd.read_csv("data/processed/y_train.csv").squeeze()
    y_test = pd.read_csv("data/processed/y_test.csv").squeeze()

    print(f"Train Shape: {X_train.shape}")
    print(f"Test Shape : {X_test.shape}")

    # ─────────────────────────────────────────────────────────
    # MLflow Experiment
    # ─────────────────────────────────────────────────────────
    mlflow.set_experiment("Telco-Churn-Prediction")

    # ─────────────────────────────────────────────────────────
    # Models
    # ─────────────────────────────────────────────────────────
    models = {

        "LogisticRegression": LogisticRegression(
            max_iter=params["model"]["lr_max_iter"],
            class_weight="balanced",
            random_state=params["model"]["random_state"]
        ),

        "RandomForest": RandomForestClassifier(
            n_estimators=params["model"]["rf_n_estimators"],
            class_weight="balanced",
            random_state=params["model"]["random_state"],
            n_jobs=-1
        ),

        "XGBoost": XGBClassifier(
            n_estimators=params["model"]["xgb_n_estimators"],
            max_depth=params["model"]["xgb_max_depth"],
            learning_rate=params["model"]["xgb_lr"],
            subsample=params["model"]["xgb_subsample"],
            colsample_bytree=params["model"]["xgb_colsample"],
            scale_pos_weight=params["model"]["scale_pos_weight"],
            eval_metric="logloss",
            random_state=params["model"]["random_state"]
        )
    }

    # ─────────────────────────────────────────────────────────
    # Training Loop
    # ─────────────────────────────────────────────────────────
    for model_name, model in models.items():

        print(f"\nTraining {model_name}...")

        with mlflow.start_run(run_name=model_name):

            # ────────────────────────────────────────────────
            # Train Model
            # ────────────────────────────────────────────────
            model.fit(X_train, y_train)

            # ────────────────────────────────────────────────
            # Predictions
            # ────────────────────────────────────────────────
            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)[:, 1]

            # ────────────────────────────────────────────────
            # Metrics
            # ────────────────────────────────────────────────
            accuracy = accuracy_score(y_test, y_pred)
            roc_auc = roc_auc_score(y_test, y_prob)
            f1 = f1_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)

            # ────────────────────────────────────────────────
            # Log Parameters
            # ────────────────────────────────────────────────
            mlflow.log_param("model_name", model_name)
            mlflow.log_params(params["model"])

            # ────────────────────────────────────────────────
            # Log Metrics
            # ────────────────────────────────────────────────
            mlflow.log_metric("accuracy", accuracy)
            mlflow.log_metric("roc_auc", roc_auc)
            mlflow.log_metric("f1_score", f1)
            mlflow.log_metric("precision", precision)
            mlflow.log_metric("recall", recall)

            # ────────────────────────────────────────────────
            # Save & Upload Model
            # ────────────────────────────────────────────────
            model_path = f"temp_model_{model_name}"

            if os.path.exists(model_path):
                shutil.rmtree(model_path)

            mlflow.sklearn.save_model(model, path=model_path)

            mlflow.log_artifacts(
                model_path,
                artifact_path="model"
            )

            # Cleanup
            shutil.rmtree(model_path)

            # ────────────────────────────────────────────────
            # Test Artifact Upload
            # ────────────────────────────────────────────────
            with open("test.txt", "w") as f:
                f.write("Connection successful!")

            mlflow.log_artifact("test.txt")

            if os.path.exists("test.txt"):
                os.remove("test.txt")

            # ────────────────────────────────────────────────
            # Print Results
            # ────────────────────────────────────────────────
            print(f"Accuracy : {accuracy:.4f}")
            print(f"ROC-AUC  : {roc_auc:.4f}")
            print(f"F1 Score : {f1:.4f}")
            print(f"Precision: {precision:.4f}")
            print(f"Recall   : {recall:.4f}")

    print("\nAll models trained successfully!")


if __name__ == "__main__":
    train()