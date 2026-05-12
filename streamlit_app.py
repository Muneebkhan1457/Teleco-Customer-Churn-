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
import streamlit as st
import warnings
warnings.filterwarnings("ignore")

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Telco Churn Predictor",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 16px;
        margin-bottom: 2rem;
        text-align: center;
        color: white;
        box-shadow: 0 8px 32px rgba(102, 126, 234, 0.3);
    }

    .main-header h1 { font-size: 2.5rem; font-weight: 700; margin: 0; }
    .main-header p  { font-size: 1.1rem; opacity: 0.9; margin-top: 0.5rem; }

    .predict-btn > button {
        background: linear-gradient(135deg, #667eea, #764ba2) !important;
        color: white !important;
        border: none !important;
        border-radius: 10px !important;
        padding: 0.75rem 2rem !important;
        font-size: 1.1rem !important;
        font-weight: 600 !important;
        width: 100% !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 15px rgba(102,126,234,0.4) !important;
    }
    .predict-btn > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 25px rgba(102,126,234,0.5) !important;
    }

    .result-box {
        border-radius: 16px;
        padding: 2rem;
        text-align: center;
        margin-top: 1.5rem;
        animation: fadeIn 0.5s ease;
    }
    .result-churn    { background: linear-gradient(135deg, #ff6b6b, #ee5a24); color: white; }
    .result-no-churn { background: linear-gradient(135deg, #2ecc71, #27ae60); color: white; }
    .result-box h2  { font-size: 2rem; font-weight: 700; }
    .result-box p   { font-size: 1.1rem; opacity: 0.9; }

    .metric-card {
        background: #f8f9fa;
        border-radius: 12px;
        padding: 1.2rem;
        text-align: center;
        border-left: 4px solid #667eea;
    }
    .metric-card h3 { color: #667eea; font-size: 1.8rem; font-weight: 700; margin: 0; }
    .metric-card p  { color: #6c757d; margin: 0; font-size: 0.9rem; }

    @keyframes fadeIn { from { opacity: 0; transform: translateY(10px); } to { opacity: 1; transform: translateY(0); } }

    div[data-testid="stSelectbox"] label,
    div[data-testid="stNumberInput"] label,
    div[data-testid="stSlider"] label { font-weight: 500; color: #374151; }

    .section-title {
        font-weight: 600;
        color: #374151;
        font-size: 1rem;
        border-bottom: 2px solid #667eea;
        padding-bottom: 0.4rem;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# ── Load Model & Preprocessor (cached) ────────────────────────────────────────
@st.cache_resource(show_spinner="Loading model from DagsHub...")
def load_artifacts():
    dagshub.init(repo_owner='Muneebkhan1457', repo_name='Teleco-Customer-Churn-', mlflow=True)
    mlflow.set_tracking_uri("https://dagshub.com/Muneebkhan1457/Teleco-Customer-Churn-.mlflow")
    model = mlflow.sklearn.load_model("models:/Telco-Churn-Model/latest")
    with open("data_&_model/preprocessor.pkl", "rb") as f:
        preprocessor = pickle.load(f)
    return model, preprocessor

model, preprocessor = load_artifacts()

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>📡 Telco Customer Churn Predictor</h1>
    <p>Powered by Machine Learning · MLflow · DagsHub · DVC</p>
</div>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📋 About")
    st.info(
        "This app predicts whether a telecom customer is likely to **churn** "
        "based on their account and service details. \n\n"
        "Fill in the customer details on the right and click **Predict**."
    )
    st.markdown("---")
    st.markdown("### 🧠 Model Info")
    st.markdown("- **Model:** Logistic Regression (Best ROC-AUC)")
    st.markdown("- **Registry:** DagsHub MLflow")
    st.markdown("- **Pipeline:** DVC Automated")
    st.markdown("---")
    st.markdown("### 📊 Dataset")
    st.markdown("- **Source:** Telco Customer Churn (Kaggle)")
    st.markdown("- **Records:** 7,043 customers")
    st.markdown("- **Target:** Churn (Yes / No)")

# ── Input Form ────────────────────────────────────────────────────────────────
col_left, col_right = st.columns([1.2, 1], gap="large")

with col_left:
    st.markdown("### 👤 Customer Information")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<p class="section-title">Demographics</p>', unsafe_allow_html=True)
        gender         = st.selectbox("Gender", ["Male", "Female"], key="gender")
        senior_citizen = st.selectbox("Senior Citizen", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No", key="senior")
        partner        = st.selectbox("Partner", ["Yes", "No"], key="partner")
        dependents     = st.selectbox("Dependents", ["Yes", "No"], key="dependents")
        tenure         = st.slider("Tenure (months)", 0, 72, 12, key="tenure")

    with c2:
        st.markdown('<p class="section-title">Phone Services</p>', unsafe_allow_html=True)
        phone_service  = st.selectbox("Phone Service", ["Yes", "No"], key="phone")
        multiple_lines = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"], key="lines")
        st.markdown('<p class="section-title">Internet Services</p>', unsafe_allow_html=True)
        internet       = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"], key="internet")
        online_sec     = st.selectbox("Online Security", ["Yes", "No", "No internet service"], key="sec")
        online_backup  = st.selectbox("Online Backup", ["Yes", "No", "No internet service"], key="backup")

    c3, c4 = st.columns(2)
    with c3:
        st.markdown('<p class="section-title">Add-on Services</p>', unsafe_allow_html=True)
        device_prot   = st.selectbox("Device Protection", ["Yes", "No", "No internet service"], key="device")
        tech_support  = st.selectbox("Tech Support", ["Yes", "No", "No internet service"], key="tech")
        streaming_tv  = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"], key="tv")
        streaming_mov = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"], key="movies")

    with c4:
        st.markdown('<p class="section-title">Billing</p>', unsafe_allow_html=True)
        contract       = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"], key="contract")
        paperless      = st.selectbox("Paperless Billing", ["Yes", "No"], key="paperless")
        payment        = st.selectbox("Payment Method", [
            "Electronic check", "Mailed check",
            "Bank transfer (automatic)", "Credit card (automatic)"
        ], key="payment")
        monthly_charges = st.number_input("Monthly Charges ($)", min_value=0.0, max_value=200.0, value=70.5, step=0.5, key="monthly")
        total_charges   = st.number_input("Total Charges ($)", min_value=0.0, max_value=10000.0, value=846.0, step=1.0, key="total")

# ── Predict ───────────────────────────────────────────────────────────────────
with col_right:
    st.markdown("### 🔮 Prediction")

    # Stats
    m1, m2, m3 = st.columns(3)
    with m1:
        st.markdown('<div class="metric-card"><h3>7043</h3><p>Customers Trained On</p></div>', unsafe_allow_html=True)
    with m2:
        st.markdown('<div class="metric-card"><h3>84.9%</h3><p>ROC-AUC Score</p></div>', unsafe_allow_html=True)
    with m3:
        st.markdown('<div class="metric-card"><h3>4</h3><p>Models Compared</p></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown('<div class="predict-btn">', unsafe_allow_html=True)
    predict_btn = st.button("🚀 Predict Churn", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    if predict_btn:
        with st.spinner("Analyzing customer data..."):
            try:
                input_df = pd.DataFrame([{
                    "gender": gender,
                    "SeniorCitizen": senior_citizen,
                    "Partner": partner,
                    "Dependents": dependents,
                    "tenure": tenure,
                    "PhoneService": phone_service,
                    "MultipleLines": multiple_lines,
                    "InternetService": internet,
                    "OnlineSecurity": online_sec,
                    "OnlineBackup": online_backup,
                    "DeviceProtection": device_prot,
                    "TechSupport": tech_support,
                    "StreamingTV": streaming_tv,
                    "StreamingMovies": streaming_mov,
                    "Contract": contract,
                    "PaperlessBilling": paperless,
                    "PaymentMethod": payment,
                    "MonthlyCharges": monthly_charges,
                    "TotalCharges": total_charges,
                }])

                input_df["TotalCharges"] = np.log1p(input_df["TotalCharges"])
                transformed = preprocessor.transform(input_df)
                pred = int(model.predict(transformed)[0])
                prob = float(model.predict_proba(transformed)[0][1])

                if pred == 1:
                    st.markdown(f"""
                    <div class="result-box result-churn">
                        <h2>⚠️ Will Churn</h2>
                        <p>Churn Probability: <strong>{prob:.1%}</strong></p>
                        <p>This customer is at high risk of leaving. Consider retention offers.</p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="result-box result-no-churn">
                        <h2>✅ Will Stay</h2>
                        <p>Churn Probability: <strong>{prob:.1%}</strong></p>
                        <p>This customer is likely to remain. Keep up the good service!</p>
                    </div>
                    """, unsafe_allow_html=True)

                st.markdown("#### 📊 Probability Breakdown")
                prob_df = pd.DataFrame({
                    "Outcome": ["Stay", "Churn"],
                    "Probability": [1 - prob, prob]
                })
                st.bar_chart(prob_df.set_index("Outcome"))

            except Exception as e:
                st.error(f"❌ Prediction failed: {e}")

    else:
        st.markdown("""
        <div style="text-align:center; padding: 3rem; color: #9ca3af; border: 2px dashed #e5e7eb; border-radius: 16px; margin-top: 1rem;">
            <div style="font-size: 3rem;">🤖</div>
            <p style="font-size: 1.1rem; margin-top: 0.5rem;">Fill in customer details and click <strong>Predict Churn</strong></p>
        </div>
        """, unsafe_allow_html=True)
