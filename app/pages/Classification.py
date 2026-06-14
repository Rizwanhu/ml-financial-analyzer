import streamlit as st
import pandas as pd
import sys
from pathlib import Path

# --- PATH SETUP ---
PROJECT_ROOT = Path(__file__).resolve().parents[2] 
sys.path.append(str(PROJECT_ROOT / "src"))

from classification_banking import (
    load_classification_model,
    classify_transactions_df,
    predict_mcc,
    load_mcc_mapping,
)

# --- PAGE CONFIG ---
st.set_page_config(page_title="Transaction Classification", layout="wide")
st.title("📊 Transaction Classification")
st.caption("Categorizing transactions into human-readable merchant types.")

# --- HELPERS ---
@st.cache_resource
def get_model():
    try:
        return load_classification_model()
    except FileNotFoundError:
        return None

@st.cache_data
def get_mcc_map():
    mcc_path = PROJECT_ROOT / "finance_dataset" / "mcc_codes.json"
    return load_mcc_mapping(mcc_path)

model = get_model()
mcc_map = get_mcc_map()

if model is None:
    st.error("⚠️ Classification model not found! Please run `python scripts/train_classification.py` first.")
    st.stop()

# --- BATCH CLASSIFICATION ---
st.subheader("Batch Classification")
uploaded_file = st.file_uploader("Upload transactions CSV to categorize", type=["csv"])

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        
        if 'amount' in df.columns:
            # Use modular classification dataframe scoring
            df = classify_transactions_df(df, model=model)
            
            # Show results with readable names
            st.write("### Categorized Results")
            st.dataframe(df[['amount', 'date', 'merchant_state', 'Category_Name']].head(100), use_container_width=True)
            
            # Summary Chart with Names instead of Codes
            st.write("### Spending Distribution by Category")
            category_counts = df['Category_Name'].value_counts()
            st.bar_chart(category_counts)
            
            st.download_button(
                "Download Categorized CSV",
                data=df.to_csv(index=False).encode("utf-8"),
                file_name="categorized_transactions.csv",
                mime="text/csv",
            )
        else:
            st.error("The CSV must contain an 'amount' column.")
    except Exception as e:
        st.error(f"Error: {e}")

st.divider()

# --- LIVE TEST ---
st.subheader("Live Category Check")
col1, col2 = st.columns(2)
with col1:
    test_amount = st.number_input("Transaction Amount ($)", value=100.0, step=10.0)

if st.button("Classify Transaction"):
    prediction = predict_mcc(model, [test_amount])[0]
    category_name = mcc_map.get(str(prediction), "Unknown Category")
    
    with col2:
        st.metric("Predicted Category", category_name)
        st.info(f"The model identifies code {prediction} as **{category_name}**.")