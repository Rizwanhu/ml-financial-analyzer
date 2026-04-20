# AI Accounting Assistant

This repository implements the project described in `guide.md` using a modular machine learning architecture:

- Expense classification (baseline Logistic Regression + advanced Random Forest)
- Anomaly detection (Isolation Forest)
- Cashflow prediction (Random Forest regressor on monthly lag features)
- End-to-end integration through a Streamlit dashboard

## Quick Start

1. Create and activate a virtual environment.
2. Install dependencies:
   - `pip install -r requirements.txt`
3. Train models:
   - `python scripts/train.py`
4. Run integration check:
   - `python scripts/test_integration.py`
5. Run tests:
   - `python -m unittest discover tests`
6. Launch dashboard:
   - `streamlit run app/streamlit_app.py`

For full technical details, read `docs/IMPLEMENTATION_DETAILS.md`.

