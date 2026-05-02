import pandas as pd
import joblib
from pathlib import Path
from sklearn.linear_model import LinearRegression

def train_forecasting():
    # 1. Load Data
    df = pd.read_csv("finance_dataset/transactions_data.csv").head(5000)
    
    # 2. Preprocessing
    df['amount'] = df['amount'].replace('[\$,)]', '', regex=True).replace('[(]', '-', regex=True).astype(float)
    
    # 3. Features (X = Transaction ID) and Target (y = Amount)
    X = df[['id']] 
    y = df['amount']
    
    # 4. Train
    model = LinearRegression()
    model.fit(X, y)
    
    # 5. Save
    model_path = Path("artifacts/models/forecaster.joblib")
    joblib.dump(model, model_path)
    print(f"✅ Forecasting model saved to {model_path}")

if __name__ == "__main__":
    train_forecasting()