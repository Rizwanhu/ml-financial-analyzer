from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT / "src"))

from ai_accounting_assistant.pipeline import load_models, run_inference


st.set_page_config(page_title="AI Accounting Assistant", layout="wide")
st.title("AI Accounting Assistant")
st.write(
    "Upload a CSV file with columns: `date`, `amount`, `description`, `category` "
    "to run classification, anomaly detection, and cashflow prediction."
)


@st.cache_resource
def get_models():
    return load_models()


uploaded_file = st.file_uploader("Upload transactions CSV", type=["csv"])
if uploaded_file is None:
    st.info("Train models first (`python scripts/train.py`), then upload data here.")
    st.stop()

try:
    df = pd.read_csv(uploaded_file)
    models = get_models()
    results = run_inference(df, models)
    tx_df = results["transactions_with_predictions"]
    forecast = results["next_month_cashflow_prediction"]

    st.subheader("Predicted Transactions")
    st.dataframe(tx_df)

    st.subheader("Anomaly Alerts")
    anomalies = tx_df[tx_df["anomaly_flag"] == -1]
    if anomalies.empty:
        st.success("No anomalies detected in uploaded data.")
    else:
        st.warning(f"Detected {len(anomalies)} anomalous transaction(s).")
        st.dataframe(anomalies)

    st.subheader("Cashflow Forecast")
    st.metric("Next Month Net Cashflow (Predicted)", f"{forecast:,.2f}")
except Exception as exc:
    st.error(f"Failed to process input: {exc}")

