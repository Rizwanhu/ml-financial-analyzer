import streamlit as st

st.set_page_config(page_title="Financial Analyzer Hub", layout="wide")

st.title("🏦 AI Financial Analysis Suite")
st.write("Welcome! This dashboard uses Machine Learning to analyze banking data.")

col1, col2, col3 = st.columns(3)

with col1:
    st.info("### ⚠️ Fraud Detection\nIdentify suspicious transactions using Anomaly Detection.")
with col2:
    st.success("### 📊 Classification\nCategorize transactions into merchant types automatically.")
with col3:
    st.warning("### 📈 Forecasting\nPredict future transaction amounts using Linear Regression.")

st.sidebar.success("Select a feature above to begin.")