import pandas as pd
import numpy as np
import pickle
import mlflow.sklearn
import dagshub

os.environ["DAGSHUB_TOKEN"] = os.getenv("DAGSHUB_TOKEN", "")

# Initialize DagsHub
dagshub.init(repo_owner='Muneebkhan1457', repo_name='Teleco-Customer-Churn-', mlflow=True)

def predict():
    # 1. Model Load 
    model = mlflow.sklearn.load_model(
        model_uri="models:/Telco-Churn-Model/latest"
    )
    print("Model Loaded!")

    # 2. Preprocessor Load 
    with open("data_&_model/preprocessor.pkl", "rb") as f:
        preprocessor = pickle.load(f)
    print("Preprocessor Loaded!")

    # 3. New Unseen Data
    new_data = pd.DataFrame([
        {
            "gender": "Male", "SeniorCitizen": 0, "Partner": "Yes",
            "Dependents": "No", "tenure": 12, "PhoneService": "Yes",
            "MultipleLines": "No", "InternetService": "Fiber optic",
            "OnlineSecurity": "No", "OnlineBackup": "No",
            "DeviceProtection": "No", "TechSupport": "No",
            "StreamingTV": "Yes", "StreamingMovies": "No",
            "Contract": "Month-to-month", "PaperlessBilling": "Yes",
            "PaymentMethod": "Electronic check",
            "MonthlyCharges": 70.5, "TotalCharges": 846.0
        },
        {
            "gender": "Female", "SeniorCitizen": 0, "Partner": "No",
            "Dependents": "No", "tenure": 1, "PhoneService": "Yes",
            "MultipleLines": "No", "InternetService": "DSL",
            "OnlineSecurity": "No", "OnlineBackup": "No",
            "DeviceProtection": "No", "TechSupport": "No",
            "StreamingTV": "No", "StreamingMovies": "No",
            "Contract": "Month-to-month", "PaperlessBilling": "Yes",
            "PaymentMethod": "Mailed check",
            "MonthlyCharges": 45.65, "TotalCharges": 45.65
        }
    ])

    print(f"\nNew Data:\n{new_data}")

    # 4. Preprocess data
    # (Applying same log transform as training)
    new_data["TotalCharges"] = np.log1p(new_data["TotalCharges"])
    new_data_transformed = preprocessor.transform(new_data)

    # 5. Prediction 
    predictions = model.predict(new_data_transformed)

    print("\n--- Predictions ---")
    for i, pred in enumerate(predictions):
        label = "Yes" if pred == 1 else "No"
        print(f"Record {i+1} — Predicted Churn: {label}")


if __name__ == "__main__":
    predict()
