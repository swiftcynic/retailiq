
# Milestone 5 — Streamlit Dashboard

## Files to add or replace

- NEW `dashboard/app.py`: Interactive Streamlit application
- NEW `src/retailiq/dashboard_data.py`: Snowflake read queries
- NEW `src/retailiq/inventory.py`: Hypothetical inventory calculator
- NEW `tests/test_inventory.py`: Inventory calculation tests
- UPDATE `pyproject.toml` and `requirements.txt`
- NEW `docs/dashboard.md`: Dashboard instructions

## Running the dashboard

From the project root:

```bash
source .venv/bin/activate
python -m pip install -e '.[dev]'
pytest -q
streamlit run dashboard/app.py
```

Open the URL printed by Streamlit, typically:

http://localhost:8501

The dashboard requires:
- Existing `.env` Snowflake credentials
- A successful Milestone 4 prediction run
- Access to the RetailIQ Snowflake tables

## Dashboard features

### 1. Demand Forecasts

- Filter by saved model run, store and product
- Download selected forecasts
- View historical weekly demand
- View predicted seven-day demand

### 2. Inventory Simulation

- Enter hypothetical available stock
- Configure supplier lead time
- Adjust safety stock coverage
- Calculate recommended replenishment quantities

The inventory figures are simulated, not actual retailer
inventory records.

### 3. Model Performance

- Display historical evaluation WAPE from Snowflake
- Compare baseline and XGBoost evaluation metrics
- Use local evaluation.json for additional metrics

## Cost management

Snowflake queries are cached for five minutes.

The app does not schedule automatic refreshes.

After using the dashboard:

1. Stop Streamlit with Ctrl+C.
2. Suspend the Snowflake warehouse:

```bash
python -m retailiq.cli suspend
```

## Limitations

- Forecasts use historical 2016 data.
- Inventory quantities are hypothetical.
- The model predicts seven-day totals, not individual days.
- The inventory calculator assumes uniform daily demand.
- The dashboard is intended for local execution.
- Snowflake permissions are required for reading forecasts,
  model runs and historical sales.
