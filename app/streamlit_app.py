from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT / "src"))

from fraud_banking.config import REPORTS_DIR
from fraud_banking.inference import predict_from_transactions_df, load_model


st.set_page_config(page_title="Fraud Detection Banking Demo", layout="wide")
st.title("Financial Fraud Detection (Kaggle Transactions Dataset)")
st.caption("Train the model with `python scripts/train.py`, then use this UI for batch scoring and live checks.")


@st.cache_resource
def get_models():
    return load_model()

with st.sidebar:
    st.subheader("Model")
    st.write("Artifact: `artifacts/models/fraud_pipeline.joblib`")
    st.write("Metrics: `artifacts/reports/fraud_metrics.json`")

st.subheader("Model quality (from latest training run)")
metrics_path = REPORTS_DIR / "fraud_metrics.json"
if metrics_path.exists():
    st.json(metrics_path.read_text(encoding="utf-8"))
else:
    st.info("No metrics found yet. Train first: `python scripts/train.py`.")

try:
    get_models()
except Exception as exc:
    st.error(f"Model load failed: {exc}")

st.divider()

st.subheader("Batch scoring (upload a CSV)")
st.write("Expected columns match `transactions_data.csv` (at minimum: `id,date,client_id,card_id,amount,use_chip,merchant_state,zip,mcc,errors`).")
uploaded_file = st.file_uploader("Upload transactions CSV", type=["csv"])
if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        scored = predict_from_transactions_df(df)
        st.dataframe(scored.sort_values("fraud_proba", ascending=False).head(200))
        st.download_button(
            "Download scored CSV",
            data=scored.to_csv(index=False).encode("utf-8"),
            file_name="scored_transactions.csv",
            mime="text/csv",
        )
    except Exception as exc:
        st.error(f"Failed to score: {exc}")

st.divider()

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
    st.metric("Fraud probability", f"{proba:.4f}")
    st.write("Prediction:", "FRAUD" if pred == 1 else "NOT FRAUD")

