import streamlit as st
import pandas as pd
import joblib
import sys
from pathlib import Path

# --- PATH SETUP ---
# Go up 2 levels to reach project root from app/pages/
PROJECT_ROOT = Path(__file__).resolve().parents[2] 
sys.path.append(str(PROJECT_ROOT / "src"))

# --- PAGE CONFIG ---
st.set_page_config(page_title="Transaction Classification", layout="wide")
st.title("📊 Transaction Classification")
st.caption("Automatically categorize transactions based on amount and merchant data.")

# --- MODEL LOADING ---
@st.cache_resource
def load_classification_model():
    model_path = PROJECT_ROOT / "artifacts" / "models" / "classification_model.joblib"
    if model_path.exists():
        return joblib.load(model_path)
    return None

model = load_classification_model()

if model is None:
    st.error("⚠️ Classification model not found! Please run `python scripts/train_classification.py` first.")
    st.stop()

# --- BATCH CLASSIFICATION ---
st.subheader("Batch Classification")
uploaded_file = st.file_uploader("Upload transactions CSV to categorize", type=["csv"])

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        
        # Simple Preprocessing (matching what we did in training)
        # We clean the amount column just in case it has symbols
        if 'amount' in df.columns:
            temp_df = df.copy()
            
            # 1. Force the column to be strings first
            temp_df['amount'] = temp_df['amount'].astype(str)
            
            # 2. Advanced cleaning: Remove $, commas, and extra spaces
            # This regex replaces anything that IS NOT a digit, a dot, or a minus sign with an empty string
            temp_df['amount'] = temp_df['amount'].str.replace(r'[^0-9.\-]', '', regex=True)
            
            # 3. Handle empty strings (just in case some rows become blank)
            temp_df['amount'] = temp_df['amount'].replace('', '0')
            
            # 4. Final conversion
            temp_df['amount'] = temp_df['amount'].astype(float)
            
            # Predict
            features = temp_df[['amount']]
            df['Predicted_MCC'] = model.predict(features)
            
            # Show results
            st.write("### Categorized Results")
            st.dataframe(df.head(100), use_container_width=True)
            
            # Summary Chart
            st.write("### Spending Distribution by Category (MCC)")
            category_counts = df['Predicted_MCC'].value_counts()
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
        st.error(f"Error processing file: {e}")

st.divider()

# --- LIVE TEST ---
st.subheader("Live Category Check")
col1, col2 = st.columns(2)

with col1:
    test_amount = st.number_input("Transaction Amount ($)", value=100.0, step=10.0)

if st.button("Classify Transaction"):
    # Reshape input for the model
    prediction = model.predict([[test_amount]])
    
    with col2:
        st.metric("Predicted MCC Code", f"{prediction[0]}")
        st.info(f"The model identifies this as category code {prediction[0]} based on the transaction value.")