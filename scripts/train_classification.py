import pandas as pd
import joblib
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

def train_classification():
    # 1. Load Data
    df = pd.read_csv("finance_dataset/transactions_data.csv").head(10000) # Use a subset for speed
    
    # 2. Basic Preprocessing
    # Convert '$77.00' string to float 77.00
    df['amount'] = df['amount'].replace('[\$,)]', '', regex=True).replace('[(]', '-', regex=True).astype(float)
    
    # 3. Define Features (X) and Target (y)
    X = df[['amount']] 
    y = df['mcc']
    
    # 4. Train Model
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    model = RandomForestClassifier(n_estimators=10)
    model.fit(X_train, y_train)
    
    # 5. Save Model
    model_path = Path("artifacts/models/classification_model.joblib")
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)
    print(f"✅ Classification model saved to {model_path}")

if __name__ == "__main__":
    train_classification()