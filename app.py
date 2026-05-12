from dotenv import load_dotenv
import os
import sys

os.environ['PYTHONIOENCODING'] = 'utf-8'
load_dotenv()

os.environ['MLFLOW_TRACKING_USERNAME'] = os.getenv('MLFLOW_TRACKING_USERNAME', 'Muneebkhan1457')
os.environ['MLFLOW_TRACKING_PASSWORD'] = os.getenv('MLFLOW_TRACKING_PASSWORD', 'ab7b436dbc3d1c4caf44d2d81a8f5d27a7e903ad')

import pickle
import numpy as np
import pandas as pd
import mlflow.sklearn
import dagshub
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import warnings
warnings.filterwarnings("ignore")
    import os
    os.environ["DAGSHUB_TOKEN"] = os.getenv("DAGSHUB_TOKEN", "")

# ── DagsHub Init ──────────────────────────────────────────────────────────────
dagshub.init(repo_owner='Muneebkhan1457', repo_name='Teleco-Customer-Churn-', mlflow=True)
mlflow.set_tracking_uri("https://dagshub.com/Muneebkhan1457/Teleco-Customer-Churn-.mlflow")

# ── Load model & preprocessor at startup ─────────────────────────────────────
print("Loading model from local path...")
model = mlflow.sklearn.load_model("data_&_model/best_model")
print("Model loaded!")

print("Loading preprocessor...")
with open("data_&_model/preprocessor.pkl", "rb") as f:
    preprocessor = pickle.load(f)
print("Preprocessor loaded!")

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Telco Customer Churn Prediction API",
    description="Predicts whether a telecom customer will churn based on account features.",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Request Schema ─────────────────────────────────────────────────────────────
class CustomerFeatures(BaseModel):
    gender: str = Field(..., example="Male")
    SeniorCitizen: int = Field(..., example=0, ge=0, le=1)
    Partner: str = Field(..., example="Yes")
    Dependents: str = Field(..., example="No")
    tenure: int = Field(..., example=12, ge=0)
    PhoneService: str = Field(..., example="Yes")
    MultipleLines: str = Field(..., example="No")
    InternetService: str = Field(..., example="Fiber optic")
    OnlineSecurity: str = Field(..., example="No")
    OnlineBackup: str = Field(..., example="No")
    DeviceProtection: str = Field(..., example="No")
    TechSupport: str = Field(..., example="No")
    StreamingTV: str = Field(..., example="Yes")
    StreamingMovies: str = Field(..., example="No")
    Contract: str = Field(..., example="Month-to-month")
    PaperlessBilling: str = Field(..., example="Yes")
    PaymentMethod: str = Field(..., example="Electronic check")
    MonthlyCharges: float = Field(..., example=70.5, ge=0)
    TotalCharges: float = Field(..., example=846.0, ge=0)

# ── Response Schema ────────────────────────────────────────────────────────────
class PredictionResponse(BaseModel):
    churn_prediction: str
    churn_probability: float
    message: str

# ── Health Check ──────────────────────────────────────────────────────────────
@app.get("/", tags=["Health"])
def root():
    return {"status": "ok", "message": "Telco Churn Prediction API is running!"}

@app.get("/health", tags=["Health"])
def health():
    return {"status": "healthy", "model": "Telco-Churn-Model", "version": "latest"}

# ── Predict ───────────────────────────────────────────────────────────────────
@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
def predict(customer: CustomerFeatures):
    try:
        # Build DataFrame
        input_df = pd.DataFrame([customer.model_dump()])

        # Apply log transform (same as training)
        input_df["TotalCharges"] = np.log1p(input_df["TotalCharges"])

        # Preprocess
        transformed = preprocessor.transform(input_df)

        # Predict
        pred = int(model.predict(transformed)[0])
        prob = float(model.predict_proba(transformed)[0][1])

        churn_label = "Yes" if pred == 1 else "No"
        message = (
            "⚠️ This customer is likely to churn. Consider retention actions."
            if pred == 1
            else "✅ This customer is likely to stay."
        )

        return PredictionResponse(
            churn_prediction=churn_label,
            churn_probability=round(prob, 4),
            message=message
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")
