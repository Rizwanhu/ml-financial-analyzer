import streamlit as st
import pandas as pd
import joblib
import sys
import numpy as np
from pathlib import Path

# --- PATH SETUP ---
PROJECT_ROOT = Path(__file__).resolve().parents[2] 
sys.path.append(str(PROJECT_ROOT / "src"))

st.set_page_config(page_title="Spending Forecast", layout="wide")
st.title("📈 Spending Forecast")
st.caption("Analyzing historical patterns to predict future financial activity.")

@st.cache_resource
def load_forecast_model():
    model_path = PROJECT_ROOT / "artifacts" / "models" / "forecaster.joblib"
    return joblib.load(model_path) if model_path.exists() else None

model = load_forecast_model()

if model is None:
    st.error("⚠️ Forecasting model not found! Run `python scripts/train_forecasting.py` first.")
    st.stop()

st.subheader("Analyze Spending Trends")
uploaded_file = st.file_uploader("Upload transactions CSV to forecast", type=["csv"])

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        df['amount'] = df['amount'].astype(str).str.replace(r'[^0-9.\-]', '', regex=True)
        df['amount'] = df['amount'].replace('', '0').astype(float)
        
        # 1. Historical Analysis
        st.write("### Historical Spending Pattern")
        st.line_chart(df['amount'])
        
        # 2. Prediction Logic
        last_index = len(df)
        future_indices = np.array(range(last_index, last_index + 10)).reshape(-1, 1)
        predictions = model.predict(future_indices)
        
        # 3. Actionable Insights
        st.divider()
        st.subheader("🚀 Financial Outlook")
        
        m1, m2, m3 = st.columns(3)
        total_predicted = predictions.sum()
        avg_predicted = predictions.mean()
        
        # Trend check
        is_increasing = predictions[-1] > predictions[0]
        trend_status = "Increasing 📈" if is_increasing else "Decreasing/Stable 📉"
        
        m1.metric("Total Next 10 Spend", f"${total_predicted:.2f}")
        m2.metric("Avg Per Transaction", f"${avg_predicted:.2f}")
        m3.metric("Spending Trend", trend_status)
        
        if is_increasing:
            st.warning("⚠️ **Alert:** Your spending trend is moving upward. Consider reviewing your budget.")
        else:
            st.success("✅ **Good News:** Your spending trend appears stable or downward.")

        # Visualizing the forecast
        forecast_df = pd.DataFrame({"Predicted Amount": predictions})
        st.area_chart(forecast_df)
        
    except Exception as e:
        st.error(f"Error in forecasting: {e}")