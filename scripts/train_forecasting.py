import pandas as pd
import joblib
from pathlib import Path
from sklearn.linear_model import LinearRegression

def train_forecasting():
    df = pd.read_csv("finance_dataset/transactions_data.csv")
    
    # Clean amount
    df['amount'] = df['amount'].astype(str).str.replace(r'[^0-9.\-]', '', regex=True)
    df['amount'] = df['amount'].replace('', '0').astype(float)
    
    # GROUP BY DATE - This makes the trend much clearer!
    # We take just the date part of the timestamp
    df['date_only'] = pd.to_datetime(df['date']).dt.date
    daily_spend = df.groupby('date_only')['amount'].sum().reset_index()
    
    # Use day sequence (1, 2, 3...) as X
    daily_spend['day_index'] = range(len(daily_spend))
    X = daily_spend[['day_index']]
    y = daily_spend['amount']
    
    model = LinearRegression()
    model.fit(X, y)
    
    model_path = Path("artifacts/models/forecaster.joblib")
    joblib.dump(model, model_path)
    print(f"✅ Re-trained on Daily Totals!")

if __name__ == "__main__":
    train_forecasting()