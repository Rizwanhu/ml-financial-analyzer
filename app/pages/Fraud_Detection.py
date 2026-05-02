from __future__ import annotations

import sys
import pandas as pd  
import streamlit as st 
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2] 
sys.path.append(str(PROJECT_ROOT / "src"))

# Step 2: Import internal modules
from fraud_banking.config import REPORTS_DIR
from fraud_banking.inference import predict_from_transactions_df, load_model

# Page Configuration
st.set_page_config(page_title="Fraud Detection Banking Demo", layout="wide")
st.title("Financial Fraud Detection (Kaggle Transactions Dataset)")
st.caption("Train the model with `python scripts/train.py`, then use this UI for batch scoring and live checks.")

@st.cache_resource
def get_models():
    return load_model()

# Sidebar Info
with st.sidebar:
    st.subheader("Model")
    st.write("Artifact: `artifacts/models/fraud_pipeline.joblib`")
    st.write("Metrics: `artifacts/reports/fraud_metrics.json`")

# Display Model Quality
st.subheader("Model quality (from latest training run)")
metrics_path = REPORTS_DIR / "fraud_metrics.json"
if metrics_path.exists():
    st.json(metrics_path.read_text(encoding="utf-8"))
else:
    st.info("No metrics found yet. Train first: `python scripts/train.py`.")

# Load Model
try:
    get_models()
except Exception as exc:
    st.error(f"Model load failed: {exc}")

st.divider()

# --- BATCH SCORING SECTION ---
st.subheader("Batch scoring (upload a CSV)")
st.write("Expected columns: `id,date,client_id,card_id,amount,use_chip,merchant_state,zip,mcc,errors`.")
uploaded_file = st.file_uploader("Upload transactions CSV", type=["csv"])

if uploaded_file is not None:
    try:
        # Load and Score
        df = pd.read_csv(uploaded_file)
        scored = predict_from_transactions_df(df)
        
        # Add human-readable labels for clarity
        scored['Status'] = scored['fraud_pred'].apply(lambda x: "⚠️ FRAUD" if x == 1 else "✅ NORMAL")
        
        # Calculate summary metrics
        total_tx = len(scored)
        fraud_count = int(scored['fraud_pred'].sum())
        
        # Display Metrics
        m1, m2, m3 = st.columns(3)
        m1.metric("Total Transactions", total_tx)
        m2.metric("Detected Frauds", fraud_count, delta=fraud_count, delta_color="inverse")
        m3.metric("Fraud Rate", f"{(fraud_count/total_tx)*100:.2f}%")

        # Display Results Table
        st.write("### Scored Results (Top 200 by Risk)")
        st.dataframe(
            scored.sort_values("fraud_proba", ascending=False).head(200),
            column_order=("Status", "fraud_proba", "amount", "date", "merchant_state", "mcc", "errors"),
            use_container_width=True
        )

        # Download option
        st.download_button(
            "Download scored CSV",
            data=scored.to_csv(index=False).encode("utf-8"),
            file_name="scored_transactions.csv",
            mime="text/csv",
        )
    except Exception as exc:
        st.error(f"Failed to score: {exc}")

st.divider()

# --- LIVE SAMPLE CHECK SECTION ---
st.subheader("Live sample check")
col1, col2, col3 = st.columns(3)
with col1:
    sample_id = st.number_input("id", min_value=1, value=1, step=1)
    sample_client_id = st.number_input("client_id", min_value=1, value=825, step=1)
    sample_card_id = st.number_input("card_id", min_value=1, value=4524, step=1)
    sample_amount = st.text_input("amount (e.g. $-77.00)", value="$-77.00")
with col2:
    sample_date = st.text_input("date (e.g. 2010-01-01 00:01:00)", value="2010-01-01 00:01:00")
    sample_use_chip = st.selectbox("use_chip", ["Swipe Transaction", "Chip Transaction", "Online Transaction"])
    sample_state = st.text_input("merchant_state", value="ND")
    sample_zip = st.number_input("zip", min_value=0, value=58523, step=1)
with col3:
    sample_mcc = st.number_input("mcc", min_value=0, value=5499, step=1)
    sample_errors = st.text_input("errors (optional)", value="")

if st.button("Score transaction"):
    row = pd.DataFrame(
        [
            {
                "id": int(sample_id),
                "date": sample_date,
                "client_id": int(sample_client_id),
                "card_id": int(sample_card_id),
                "amount": sample_amount,
                "use_chip": sample_use_chip,
                "merchant_id": 0,
                "merchant_city": "",
                "merchant_state": sample_state,
                "zip": float(sample_zip),
                "mcc": int(sample_mcc),
                "errors": sample_errors if sample_errors.strip() else None,
            }
        ]
    )
    
    scored = predict_from_transactions_df(row)
    proba = float(scored.loc[0, "fraud_proba"])
    pred = int(scored.loc[0, "fraud_pred"])
    
    st.divider()
    if pred == 1:
        st.error(f"### ⚠️ Prediction: FRAUD DETECTED")
        st.write(f"**Prediction Score:** {proba:.4f}")
    else:
        st.success(f"### ✅ Prediction: NOT FRAUD")
        st.write(f"**Prediction Score:** {proba:.4f}")