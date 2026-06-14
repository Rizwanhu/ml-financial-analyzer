# Code Overview — ML Financial Analyzer

This document explains how the project is organized, what each machine learning module does, how models are trained, and which dataset fields drive each task. It is meant for readers who want to understand the system without reading Python source line by line.

---

## What this project does

The platform turns raw banking transaction rows into three kinds of insight:

| Module | User question | ML role |
|--------|----------------|---------|
| **Fraud detection** | Is this transaction suspicious? | Supervised classification (fraud vs normal) |
| **Classification** | What type of spending is this? | Supervised classification (merchant category / MCC) |
| **Forecasting** | Where is spending headed? | Regression over time (future spend trend) |

A **Streamlit multi-page dashboard** (`app/streamlit_app.py` and `app/pages/`) is the main interface. Each page loads a trained model saved under `artifacts/models/` and scores uploaded CSV data or single live inputs.

There is also a **FastAPI** service (`api/main.py`) that exposes fraud scoring over HTTP.

---

## High-level architecture

```
finance_dataset/          Raw Kaggle-style CSV + JSON (not committed to git)
        │
        ▼
Training scripts          Build and save .joblib models
        │
        ▼
artifacts/models/         Persisted models used at runtime
        │
        ▼
Streamlit pages / API     Load models → preprocess → predict → show results
```

The codebase splits into two layers:

1. **`fraud_banking`** (`src/fraud_banking/`) — Production-style fraud pipeline: streaming large CSVs, joining user/card tables, rich feature engineering, sklearn `Pipeline`, metrics export.
2. **Dashboard helpers** (`scripts/train_classification.py`, `scripts/train_forecasting.py`) — Lighter training scripts wired directly to the Classification and Forecasting pages.
3. **`ai_accounting_assistant`** (`src/ai_accounting_assistant/`) — A second, more integrated ML stack (expense categories from text, anomaly detection, monthly cashflow forecasting). It is designed for transaction data with columns `date`, `amount`, `description`, and `category`. The function `train_all()` trains all three models in one call, but the current Streamlit pages use the simpler scripts above unless you wire them yourself.

---

## File map: which code files each model uses

| Model | Training | Core logic | UI | Dataset files |
|-------|----------|------------|-----|----------------|
| **Fraud** | `scripts/train.py` | `src/fraud_banking/*` (7 modules) | `app/pages/Fraud_Detection.py` | `transactions_data.csv`, `users_data.csv`, `cards_data.csv`, `train_fraud_labels.json` |
| **Classification** | `scripts/train_classification.py` (single file) | same file | `app/pages/Classification.py` | `transactions_data.csv` only; `mcc_codes.json` for display labels |
| **Forecasting** | `scripts/train_forecasting.py` (single file) | same file | `app/pages/forecasting.py` | `transactions_data.csv` only |

Fraud is split across many files because it joins three tables, streams a huge CSV, and reuses the same feature logic at train and predict time. Classification and forecasting keep almost all logic in one training script each; the Streamlit pages only load the saved model and repeat the same cleaning steps.

---

## File-by-file code explanation

### Module 1 — Fraud detection (multi-file package)

Fraud is the only module with a dedicated Python package under `src/fraud_banking/`. Everything flows: **config → load data → build features → train → save → infer → show in UI**.

#### `scripts/train.py` — entry point you run from the terminal

- Adds `src/` to Python’s path so `fraud_banking` can be imported.
- Parses optional arguments: `--max_labeled_rows` (cap how many labeled rows to train on) and `--chunksize` (how many transaction rows to read per chunk from disk).
- Builds paths to the four dataset files via `default_dataset_paths(DATASET_DIR)`.
- Calls `train_fraud_model(...)` and prints where the model and metrics were saved.

This file does not contain ML math; it only wires CLI arguments to the trainer.

#### `src/fraud_banking/config.py` — paths and folders

- Computes `PROJECT_ROOT` (repo root).
- Defines `DATASET_DIR` → `finance_dataset/`.
- Defines `MODELS_DIR` → `artifacts/models/` and `REPORTS_DIR` → `artifacts/reports/`.
- `ensure_directories()` creates those folders before training writes files.

Every other fraud file imports these constants so paths stay consistent.

#### `src/fraud_banking/data.py` — reading CSV and JSON

**`DatasetPaths`** — A small dataclass holding four paths: transactions CSV, users CSV, cards CSV, labels JSON.

**`default_dataset_paths(dataset_dir)`** — Returns the standard Kaggle filenames inside `finance_dataset/`.

**`load_users` / `load_cards`** — Simple `pd.read_csv` for the two lookup tables. Used at training and again at inference so each transaction can be enriched with client and card attributes.

**`load_labels_map`** — Opens `train_fraud_labels.json`, reads the `"target"` object, and returns a Python dict: transaction id string → `"Yes"` or `"No"`. That dict is the only source of fraud ground truth.

**`iter_labeled_transactions`** — The important piece for scale:

- Reads `transactions_data.csv` in chunks (default 250k rows) so the full file never has to fit in RAM.
- For each chunk, maps each row’s `id` through the labels dict.
- Drops rows with no label (unlabeled transactions never enter training).
- Adds column `is_fraud` (1 if label is yes, 0 otherwise).
- Yields labeled chunks one at a time to the trainer.

#### `src/fraud_banking/features.py` — turning raw rows into model columns

**`parse_money`** — Strips `$`, commas, etc. from amount strings and returns a float. Handles missing values as NaN.

**`safe_to_datetime`** — Parses `date` safely for time-based features.

**`FeatureSpec` / `FEATURES`** — Declares exactly which column names the model expects after engineering: 17 numeric and 7 categorical. This list is also saved inside the joblib bundle so train and predict stay aligned.

**`build_feature_frame(transactions, users, cards)`** — Core feature engineering:

1. Copy transactions; convert `amount` with `parse_money`.
2. From `date`, create `hour`, `day_of_week`, `month`.
3. Create `has_error` = 1 if `errors` is not null.
4. Normalize `client_id` and `card_id` for merging.
5. Select a subset of columns from `users` and rename `id` → `client_id`; parse income/debt money columns.
6. Select card columns and rename `id` → `card_id`; parse `credit_limit`.
7. Left-merge users then cards onto transactions.
8. Return only the columns listed in `FEATURES` (fill missing with NaN).

PII like full address, card number, merchant city, and merchant id are deliberately excluded.

#### `src/fraud_banking/train.py` — training loop and sklearn pipeline

**`_build_pipeline()`** — Builds a sklearn `Pipeline`:

- **Numeric branch:** median imputation for all numeric `FEATURES`.
- **Categorical branch:** most-frequent imputation, then `OneHotEncoder` (ignore unknown categories; rare categories grouped via `min_frequency=10`).
- **Classifier:** `LogisticRegression` with `class_weight="balanced"`, `solver="saga"`, many iterations — suited to imbalanced fraud data.

**`train_fraud_model(paths, ...)`** — Full training procedure:

1. `ensure_directories()`.
2. Load users, cards, label map.
3. Loop `iter_labeled_transactions`: for each chunk, build `y = is_fraud`, build `X = build_feature_frame(...)`, append to lists until `max_labeled_rows` reached.
4. Concatenate all chunks into one big `X` and `y`.
5. `train_test_split` 80/20, stratified by `y`.
6. `pipe.fit(X_train, y_train)`.
7. Predict probabilities on test set; call `compute_report`.
8. Save joblib dict: `pipeline` + `feature_spec`.
9. Write metrics JSON to `artifacts/reports/fraud_metrics.json`.

#### `src/fraud_banking/metrics.py` — evaluation numbers

**`compute_report(y_true, y_proba)`** — Takes true 0/1 labels and predicted fraud probabilities:

- Applies threshold 0.5 for hard predictions.
- Computes ROC-AUC, PR-AUC (important when fraud is rare), F1, precision, recall, confusion matrix.
- Returns a dataclass converted to JSON for the dashboard “Model quality” panel.

#### `src/fraud_banking/inference.py` — scoring new transactions

**`load_model`** — Loads `fraud_pipeline.joblib`, unwraps the `pipeline` object inside.

**`predict_from_transactions_df(transactions_df)`** — Runtime scoring:

1. Load `users_data.csv` and `cards_data.csv` from `finance_dataset/` again (required every time you score).
2. Load trained pipeline.
3. `X = build_feature_frame(transactions_df, users, cards)` — same logic as training.
4. `predict_proba` → column `fraud_proba`; threshold 0.5 → `fraud_pred`.

Returns the original dataframe plus those two columns.

#### `app/pages/Fraud_Detection.py` — Streamlit UI

- Puts `src/` on the path and imports `load_model`, `predict_from_transactions_df`, `REPORTS_DIR`.
- **Sidebar:** shows model artifact path and metrics file path.
- **Metrics block:** if `fraud_metrics.json` exists, displays it as JSON; otherwise tells user to run `python scripts/train.py`.
- **Batch scoring:** file uploader → `read_csv` → `predict_from_transactions_df` → adds human `Status` column → metrics (total tx, fraud count, rate) → sort by `fraud_proba` → table + download button.
- **Live check:** form fields matching Kaggle transaction columns → build one-row DataFrame → score → show red/green result with probability.

Expected upload columns: `id`, `date`, `client_id`, `card_id`, `amount`, `use_chip`, `merchant_state`, `zip`, `mcc`, `errors`.

#### `api/main.py` — optional REST API

- FastAPI app with `/health` and `/post /predict`.
- Accepts one transaction as JSON (same shape as a CSV row).
- Wraps payload in a DataFrame, calls `predict_from_transactions_df`, returns `fraud_proba` and `fraud_pred`.

#### `scripts/test_integration.py` — smoke test

- Builds one hard-coded sample transaction (matching test fixtures).
- Calls `predict_from_transactions_df`; prints fraud columns or tells you to train first.

#### `tests/test_pipeline.py` — unit tests

- **`test_feature_frame_builds`** — Tiny fake transaction + user + card; asserts `build_feature_frame` returns one row and multiple columns.
- **`test_model_artifact_loads_if_present`** — Skips if model not trained yet; otherwise checks joblib loads.

**Fraud data flow (summary):**

```
train_fraud_labels.json ──► is_fraud label
transactions_data.csv ──┐
users_data.csv ───────────┼──► data.py (load/stream)
cards_data.csv ───────────┘         │
                                    ▼
                            features.py (build_feature_frame)
                                    │
                                    ▼
                            train.py (LogisticRegression Pipeline)
                                    │
                                    ▼
                    fraud_pipeline.joblib + fraud_metrics.json
                                    │
                    inference.py ◄────┘
                                    │
              Fraud_Detection.py / api/main.py
```

---

### Module 2 — Classification (mostly one training file + one UI file)

Almost all training logic lives in **`scripts/train_classification.py`**. The UI is **`app/pages/Classification.py`**. No `src/` package.

#### `scripts/train_classification.py` — train and save (entire ML pipeline in one file)

Line-by-line responsibility:

1. **Import** pandas, joblib, sklearn `RandomForestClassifier`, `train_test_split`.
2. **`train_classification()` function:**
   - **Load:** `pd.read_csv("finance_dataset/transactions_data.csv").head(10000)` — only first 10k rows for speed; only this one dataset file.
   - **Clean amount:** regex removes `$` and `)`; `(` becomes minus sign; cast to float. Same idea as fraud’s money parsing but inline.
   - **Features X:** dataframe with single column `amount`.
   - **Target y:** column `mcc` (merchant category code integer).
   - **Split:** 80% train, 20% test (no stratify).
   - **Model:** `RandomForestClassifier(n_estimators=10)` — small forest, fast training.
   - **Fit** on `X_train`, `y_train`.
   - **Save:** `joblib.dump` to `artifacts/models/classification_model.joblib`; create parent folders if missing.
3. **`if __name__ == "__main__"`** — runs `train_classification()` when you execute the script.

**Dataset attributes used:**

| Column | Role |
|--------|------|
| `amount` | Only input feature |
| `mcc` | Label to predict |

Not used in training: `date`, `client_id`, `card_id`, `users_data.csv`, `cards_data.csv`, fraud labels.

#### `app/pages/Classification.py` — Streamlit UI

- **`load_classification_model`** (cached): loads `classification_model.joblib` if it exists; else `None` and page stops with error telling user to run training script.
- **`load_mcc_names`** (cached): reads `finance_dataset/mcc_codes.json` — **not used for training**, only to turn predicted MCC numbers into names like “Grocery Stores” in charts and live test.
- **Batch upload:**
  - User CSV must have `amount`.
  - Same amount cleaning as training (strip non-numeric chars, float).
  - `model.predict` on `[['amount']]` → column `Predicted_MCC`.
  - Map codes through `mcc_map` → `Category_Name`.
  - Show table, bar chart of category counts, download button.
- **Live test:** user enters dollar amount → `predict([[amount]])` → show category name from JSON map.

**Classification data flow:**

```
transactions_data.csv (amount, mcc)
        │
        ▼
train_classification.py  →  classification_model.joblib
        │
        ▼
Classification.py  +  mcc_codes.json (labels for humans only)
```

---

### Module 3 — Forecasting (mostly one training file + one UI file)

Same pattern as classification: training is **`scripts/train_forecasting.py`**, UI is **`app/pages/forecasting.py`**.

#### `scripts/train_forecasting.py` — train and save (entire ML pipeline in one file)

1. **Import** pandas, joblib, `LinearRegression`.
2. **`train_forecasting()` function:**
   - **Load:** full `transactions_data.csv` (no row cap unlike classification).
   - **Clean amount:** string replace non-numeric characters, empty → `0`, float.
   - **Date:** `pd.to_datetime(df['date'])`, then `date_only` = calendar date (drops time of day).
   - **Aggregate:** `groupby('date_only')['amount'].sum()` — one total spend per day across all transactions in the file.
   - **Feature engineering:** `day_index` = 0, 1, 2, … sequential index over sorted days (not month name or weekday — just “which day row in the series”).
   - **X** = `day_index`, **y** = daily total `amount`.
   - **Model:** `LinearRegression().fit(X, y)` — learns a straight-line trend of spend vs day number.
   - **Save:** `artifacts/models/forecaster.joblib`.
3. **Main guard** runs the function.

**Dataset attributes used:**

| Column | Role |
|--------|------|
| `date` | Group transactions by calendar day |
| `amount` | Summed per day; also the regression target |

Not used: `mcc`, user/card tables, fraud labels, `mcc_codes.json`.

#### `app/pages/forecasting.py` — Streamlit UI

- **`load_forecast_model`** (cached): loads `forecaster.joblib` or errors.
- **Upload CSV:**
  - Clean `amount` same way as training.
  - **Historical chart:** `st.line_chart(df['amount'])` — plots every transaction amount in upload order (not the same as daily aggregates used in training; this is a quick visual).
  - **Forecast:** `last_index = len(df)`; build future `day_index` values `last_index … last_index+9`; `model.predict` → 10 future values.
  - **Metrics:** sum of next 10 predictions, average, trend up/down by comparing first vs last predicted point.
  - Warning/success message based on trend; area chart of predictions.

**Important nuance:** Training learns from **daily totals** in the Kaggle file, but the UI line chart shows **raw per-row amounts** from the user’s upload. Predictions still use `day_index` continuation from upload row count, which approximates “days ahead” only if each uploaded row is one day or you treat row count like the training script’s day sequence.

**Forecasting data flow:**

```
transactions_data.csv (date, amount)
        │
        ▼
train_forecasting.py  →  daily sums  →  LinearRegression  →  forecaster.joblib
        │
        ▼
forecasting.py  →  predict next 10 day indices  →  charts + outlook metrics
```

---

## Dataset: what you need and where it lives

All banking models expect data under **`finance_dataset/`** at the project root (this folder is gitignored; you download it separately, typically from the Kaggle “Banking Transaction Fraud” style dataset).

### Files used

| File | Role |
|------|------|
| `transactions_data.csv` | Every card swipe / purchase: amount, time, merchant, MCC, errors, links to client and card |
| `users_data.csv` | Demographics and financial profile per `client_id` |
| `cards_data.csv` | Card product details per `card_id` |
| `train_fraud_labels.json` | Ground truth: transaction `id` → `"Yes"` or `"No"` (fraud) |
| `mcc_codes.json` | Maps numeric MCC codes to readable category names (for the UI only) |

### Core transaction attributes (`transactions_data.csv`)

These columns appear throughout training and inference:

- **`id`** — Unique transaction identifier; used to attach fraud labels from JSON.
- **`date`** — Timestamp string; parsed to derive hour, day of week, and month for fraud features and forecasting.
- **`client_id`** — Join key into `users_data.csv`.
- **`card_id`** — Join key into `cards_data.csv`.
- **`amount`** — Often stored as a string with `$` (e.g. `$-77.00`); cleaned to a float (negative = debit/spend).
- **`use_chip`** — How the card was used (swipe, chip, online); categorical feature for fraud.
- **`merchant_state`**, **`zip`** — Location signals for fraud.
- **`mcc`** — Merchant Category Code; used as the **target** for the dashboard classification model and as a **numeric feature** for fraud.
- **`errors`** — Optional error text; converted to a binary `has_error` flag for fraud.

Fields such as `merchant_id`, `merchant_city`, and full addresses are intentionally **not** fed into the fraud model to limit PII and high-cardinality noise.

### User attributes (`users_data.csv`) — joined for fraud only

After merge on `client_id`:

- **`current_age`**, **`retirement_age`**, **`gender`**
- **`per_capita_income`**, **`yearly_income`**, **`total_debt`** (money strings cleaned to floats)
- **`credit_score`**, **`num_credit_cards`**

### Card attributes (`cards_data.csv`) — joined for fraud only

After merge on `card_id`:

- **`card_brand`**, **`card_type`**, **`has_chip`**
- **`num_cards_issued`**, **`credit_limit`**, **`year_pin_last_changed`**
- **`card_on_dark_web`**

### Fraud labels (`train_fraud_labels.json`)

Structure: `{ "target": { "<transaction_id>": "Yes" | "No", ... } }`.

Only transactions whose `id` appears in this map are used for **supervised fraud training**. The training loop streams the huge transactions CSV in chunks, keeps labeled rows, and stops after a configurable cap (default one million rows) for speed and memory.

---

## How the three dashboard models are trained

Each module has its own training entry point and saved artifact. Run training after placing files in `finance_dataset/`.

### 1. Fraud detection

**Train:** `python scripts/train.py`  
**Saved model:** `artifacts/models/fraud_pipeline.joblib`  
**Metrics:** `artifacts/reports/fraud_metrics.json`  
**UI:** `app/pages/Fraud_Detection.py`

**What happens step by step:**

1. Load `users_data.csv` and `cards_data.csv` into memory.
2. Load the fraud label map from `train_fraud_labels.json`.
3. Stream `transactions_data.csv` in chunks (default 250,000 rows per chunk).
4. For each chunk, keep only rows with a known label; set `is_fraud` to 1 if label is `"yes"`, else 0.
5. **Feature engineering** (`build_feature_frame`): merge user and card fields onto each transaction; parse `amount`; extract `hour`, `day_of_week`, `month` from `date`; set `has_error` from whether `errors` is missing.
6. Concatenate chunks until `max_labeled_rows` (default 1,000,000) is reached.
7. **Train/test split** — 80% train, 20% test, stratified on fraud label.
8. **Model** — sklearn `Pipeline`: median/mode imputation → one-hot encoding for categoricals → **Logistic Regression** with `class_weight="balanced"` (important because fraud is rare).
9. Evaluate on holdout: ROC-AUC, PR-AUC, F1, precision, recall, confusion matrix at threshold 0.5.
10. Save the full pipeline plus feature column list into one joblib file.

**Attributes used for fraud (after joins):**

| Type | Fields |
|------|--------|
| Numeric | `amount`, `mcc`, `zip`, `hour`, `day_of_week`, `month`, ages, incomes, debt, `credit_score`, card counts/limits, `year_pin_last_changed`, `has_error` |
| Categorical | `use_chip`, `merchant_state`, `gender`, `card_brand`, `card_type`, `has_chip`, `card_on_dark_web` |

**At inference time**, the same joins and feature builder run on new rows; the model outputs `fraud_proba` (probability) and `fraud_pred` (1 if probability ≥ 0.5).

---

### 2. Transaction classification (expense / merchant type)

**Train:** `python scripts/train_classification.py`  
**Saved model:** `artifacts/models/classification_model.joblib`  
**UI:** `app/pages/Classification.py` (uses `mcc_codes.json` to show human-readable names)

**What happens step by step:**

1. Read the first **10,000 rows** of `transactions_data.csv` (subset for fast local training).
2. Clean **`amount`**: strip `$` and parentheses, convert to float.
3. **Features (X):** only **`amount`** (single numeric column).
4. **Target (y):** **`mcc`** — the merchant category code the model learns to predict.
5. **Train/test split** — 80% / 20%, no stratification in the script.
6. **Model** — **Random Forest Classifier** with 10 trees (`n_estimators=10`).
7. Save the classifier to joblib.

**How the dataset is used here:** The model does **not** use description text, date, or user/card tables in this path. It learns a coarse mapping from transaction size to MCC. The dashboard then maps predicted MCC integers to labels like “Groceries” via `mcc_codes.json`.

**Note:** The separate `ai_accounting_assistant` classification path is richer: it expects labeled **`category`** text and uses **`description`** (TF-IDF), **`amount`**, **`month`**, and **`day_of_week`**, with a Random Forest behind a text+categorical preprocessor. That path is trained via `train_all()` when you supply a compatible CSV—not the default Classification page model.

---

### 3. Spending forecast

**Train:** `python scripts/train_forecasting.py`  
**Saved model:** `artifacts/models/forecaster.joblib`  
**UI:** `app/pages/forecasting.py`

**What happens step by step:**

1. Load full `transactions_data.csv`.
2. Clean **`amount`** to float (same string stripping as classification).
3. Derive **`date_only`** from **`date`** (calendar day).
4. **Aggregate:** sum `amount` per day → one row per day with total daily spend.
5. **Features (X):** **`day_index`** — integer sequence 0, 1, 2, … over days (not calendar features).
6. **Target (y):** daily total spend.
7. **Model** — **Linear Regression** fitting spend vs day index.
8. Save model to joblib.

**How the dataset is used:** Only **`date`** and **`amount`** matter. The model captures a simple trend: “as days progress in the file, typical daily total spend follows a line.” The UI extrapolates the next 10 day-indices and shows totals, averages, and trend direction.

**Note:** The `ai_accounting_assistant` forecasting module works differently: it builds **monthly net cashflow**, creates **`lag_1`**, **`lag_2`**, and **`rolling_mean_3`**, and trains a **Random Forest Regressor** (300 trees) to predict next month’s net flow. That requires enough months of history (at least 10 after lagging) and is saved as `cashflow_forecaster.joblib` when using `train_all()`.

---

## Runtime flow (dashboard)

1. User opens a Streamlit page (Fraud, Classification, or Forecasting).
2. Page loads the corresponding `.joblib` from `artifacts/models/` (cached in memory).
3. User uploads a CSV or fills live fields.
4. Preprocessing mirrors training (amount parsing, column checks).
5. **Fraud** additionally reloads `users_data.csv` and `cards_data.csv` from `finance_dataset/` to enrich features—so those files must stay present for scoring even after training.
6. Predictions and charts are shown; fraud and classification support CSV download.

---

## Optional integrated stack (`ai_accounting_assistant`)

If you use a **personal ledger-style** CSV with columns `date`, `amount`, `description`, and `category`, the module `src/ai_accounting_assistant/pipeline.py` can train everything at once:

| Model | Algorithm | Input attributes | Output |
|-------|-----------|------------------|--------|
| Expense classifier | Logistic Regression (baseline) + Random Forest (saved) | `description` (TF-IDF), `amount`, `month`, `day_of_week` | `category` |
| Anomaly detector | Isolation Forest on scaled features | `amount`, `month`, `day_of_week`, `is_month_end` | anomaly flag / score |
| Cashflow forecaster | Random Forest Regressor | `lag_1`, `lag_2`, `rolling_mean_3` from monthly sums of `amount` | next month net cashflow |

Preprocessing (`preprocessing.py`) validates schema, normalizes text categories, drops bad dates, and builds monthly lag features from transaction-level `amount` (credits and debits net together).

This stack is the one described in the root `README.md` (`python scripts/train.py` there refers to **fraud only**; there is no single script in-repo that calls `train_all()`—you would invoke it from your own driver or notebook once your labeled expense CSV is ready).

---

## Artifacts and directories

| Path | Contents |
|------|----------|
| `finance_dataset/` | Raw training data (CSV + JSON) |
| `artifacts/models/` | Trained `.joblib` files |
| `artifacts/reports/` | Fraud metrics JSON |
| `src/fraud_banking/` | Fraud data loading, features, train, inference |
| `src/ai_accounting_assistant/` | Alternate full pipeline (classify, anomaly, forecast) |
| `app/pages/` | Streamlit UIs per module |
| `api/main.py` | REST fraud prediction |

---

## Training commands (quick reference)

```text
# Fraud (full pipeline, recommended for security module)
python scripts/train.py

# Classification (MCC from amount, dashboard page)
python scripts/train_classification.py

# Forecasting (daily spend trend, dashboard page)
python scripts/train_forecasting.py

# Smoke test fraud inference
python scripts/test_integration.py

# Launch dashboard
streamlit run app/streamlit_app.py
```

---

## Design choices worth knowing

- **Fraud** is the most data-intensive module: multi-table joins, streaming, class imbalance handling, and probability-based metrics.
- **Dashboard classification** is intentionally simple (amount → MCC) for fast demos; real expense tagging would benefit from description and date features (`ai_accounting_assistant` path).
- **Dashboard forecasting** uses linear regression on day index, not calendar seasonality; the monthly lag-based forecaster in `ai_accounting_assistant` is closer to “budget planning” semantics.
- Large CSV and JSON files stay out of git; you must provide `finance_dataset/` locally before training or fraud inference.

Together, these pieces implement the product story: **safety** (fraud), **awareness** (where money goes via MCC/categories), and **forward view** (spend trend / cashflow projection).
