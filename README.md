# Smart AI Banking Dashboard

An enterprise-grade, multi-page Smart AI Banking Dashboard built using Python and Streamlit. This application integrates production-grade machine learning pipelines with an interactive frontend interface, supporting memory-safe data streaming configurations for massive datasets.

---

## Project Workflow & Architecture

The system decouples machine learning ingestion, training, and scoring logic from the user interface. It is organized into modular packages inside `src/` that ingest raw datasets, process features in memory-safe streams, and save model artifacts to a unified folder for the Streamlit app to serve.


graph TD
    Data[finance_dataset/] -->|Chunked Ingestion| Ingest[Streaming Ingestion Engine]
    Ingest -->|Clean & Preprocess| Train[Modular Train Pipeline]
    Train -->|Fit Model| Artifacts[artifacts/models/]
    Artifacts -->|Load Model| UI[Streamlit UI Pages]
    User[Uploaded CSV / Input] -->|Interactive Check| UI
    UI -->|Serve Forecasts / Classes / Fraud| Dashboard[Dashboard Visualization]
```

### End-to-End Operational Flow
1. **Data Ingestion & Grouping**: Raw CSVs (which can exceed 1.2 GB) are processed in memory-safe blocks using custom chunking generators.
2. **Preprocessing & Feature Engineering**: Amounts are normalized, currency signs stripped, brackets converted to negative float values, and records enriched (e.g. merging profile profiles, or grouping by daily timelines).
3. **Training & Serialization**: Models are trained and serialized using `joblib` into `artifacts/models/`.
4. **Interactive Dashboard**: The Streamlit interface loads the serialized model weights from the artifact path to classify uploads, check transactions live, and plot forecasting Outlooks.

---

## The 3 Core AI Models

| Model | Algorithm | Input Features | Output Target | Modular Package Location |
| :--- | :--- | :--- | :--- | :--- |
| **Fraud Detection** | Class-Balanced Logistic Regression | Client demographics, Card brands, Chip usage, Time (hour/day), and Transaction amount | Anomaly Probability (Fraud vs Normal) | [`src/fraud_banking/`](ml-financial-analyzer/src/fraud_banking/) |
| **Transaction Classification** | Random Forest Classifier | Normalized transaction `amount` | Merchant Category Code (MCC) | [`src/classification_banking/`](ml-financial-analyzer/src/classification_banking/) |
| **Spending Forecasting** | Linear Regression | Chronological timeline sequence (`day_index`) | Total Daily Aggregated Spend | [`src/forecasting_banking/`](ml-financial-analyzer/src/forecasting_banking/) |

---

### Detailed Model Descriptions

#### 1. Fraud Detection
* **Ingestion Style**: Streams chunks of transaction records, merging each row on-the-fly with cards (`cards_data.csv`) and user profiles (`users_data.csv`).
* **Preprocessing**: Columns are normalized using `SimpleImputer` and `OneHotEncoder` via scikit-learn's `ColumnTransformer`.
* **Classifier**: Class-balanced `LogisticRegression` to handle high fraud imbalances.

#### 2. Transaction Classification
* **Ingestion Style**: Processes chunks, applies currency regex cleaning to normalize money columns, and extracts `amount` and `mcc`.
* **Classifier**: `RandomForestClassifier` (10 estimators) trained to map purchase amount levels to human-readable categories. 
* **Translation**: Uses a JSON translation engine ([mcc_codes.json](file:///ml-financial-analyzer/finance_dataset/mcc_codes.json)) to output names like "Groceries", "Entertainment", or "Airlines" in the UI.

#### 3. Spending Forecasting
* **Ingestion Style**: Streams the transaction file, groups transactions by date inside each chunk, and aggregates running totals.
* **Timeline Generator**: Maps chronological calendar days to a sequential integer sequence (`day_index` from `0, 1, 2...`).
* **Regressor**: A `LinearRegression` model fitted to daily totals to predict long-term financial trends.

---

## Setup & Running the Dashboard

### 1. Pre-requisites & Environment
Create a virtual environment and install dependencies:
```bash
# Create and activate virtual environment
python -m venv venv
venv\Scripts\activate

# Install required packages
pip install -r requirements.txt
```

### 2. Training the Models (Command Line Interface)
All models support custom chunk sizes to run on low-resource environments. Run the scripts below to populate your model artifacts:

```bash
# Train Model 1: Fraud Detection
python scripts/train.py --max_labeled_rows 1000000 --chunksize 250000

# Train Model 2: Transaction Classification
python scripts/train_classification.py --max_rows 10000 --chunksize 250000

# Train Model 3: Spending Forecasting
python scripts/train_forecasting.py --chunksize 250000
```
*Trained model brains are stored in the unified directory: `artifacts/models/`*

### 3. Verification & Testing
Ensure the refactored pipelines perform as expected by running the tests:
```bash
# Run existing integration checks
python scripts/test_integration.py

# Run unit tests (including the new refactored package tests)
python -m unittest discover tests
```

### 4. Running the Dashboard UI
Start the interactive Streamlit dashboard:
```bash
streamlit run app/streamlit_app.py
```
This launches a browser session where you can:
* Upload customer csv logs to flag fraudulent transactions.
* Batch-classify uploads into merchant types or run a live category check.
* Forecast future spend trends and view your Financial Outlook.
