import streamlit as st
import pandas as pd
import joblib
import sys
import json
from pathlib import Path

# --- PATH SETUP ---
PROJECT_ROOT = Path(__file__).resolve().parents[2] 
sys.path.append(str(PROJECT_ROOT / "src"))

# --- PAGE CONFIG ---
st.set_page_config(page_title="Transaction Classification", layout="wide")
st.title("📊 Transaction Classification")
st.caption("Categorizing transactions into human-readable merchant types.")

# --- HELPERS ---
@st.cache_resource
def load_classification_model():
    model_path = PROJECT_ROOT / "artifacts" / "models" / "classification_model.joblib"
    return joblib.load(model_path) if model_path.exists() else None

@st.cache_data
def load_mcc_names():
    mcc_path = PROJECT_ROOT / "finance_dataset" / "mcc_codes.json"
    if mcc_path.exists():
        with open(mcc_path, 'r') as f:
            return json.load(f)
    return {}

model = load_classification_model()
mcc_map = load_mcc_names()

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
            temp_df = df.copy()
            temp_df['amount'] = temp_df['amount'].astype(str).str.replace(r'[^0-9.\-]', '', regex=True)
            temp_df['amount'] = temp_df['amount'].replace('', '0').astype(float)
            
            # Predict
            df['Predicted_MCC'] = model.predict(temp_df[['amount']])
            
            # Translate Code to Name (e.g., 5411 -> Groceries)
            df['Category_Name'] = df['Predicted_MCC'].astype(str).map(mcc_map).fillna("Other/Misc Services")
            
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
    prediction = model.predict([[test_amount]])[0]
    category_name = mcc_map.get(str(prediction), "Unknown Category")
    
    with col2:
        st.metric("Predicted Category", category_name)
        st.info(f"The model identifies code {prediction} as **{category_name}**.")