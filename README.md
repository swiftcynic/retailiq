# RetailIQ — Milestone 1: Snowflake Data Engineering

This is a runnable starter repository for ingesting and validating M5 daily retail sales in Snowflake. Forecasting and dashboard modules are **not implemented yet**.

## Before you begin

1. Create a [Snowflake trial](https://signup.snowflake.com/) and use a standard X-Small warehouse. Do not enable paid billing for this exercise.
2. Download the **M5 Forecasting - Accuracy** files from [Kaggle](https://www.kaggle.com/competitions/m5-forecasting-accuracy/data) (Kaggle login and competition rules may apply).
3. Place `sales_train_validation.csv` and `calendar.csv` in `data/`. These data are deliberately not committed to the repository.
4. Use Python 3.11 and run all commands from the repository root.

## Install

```bash
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\\Scripts\\activate
python -m pip install -e '.[dev]'
cp .env.example .env       # Windows: copy .env.example .env
```

Edit `.env` with your Snowflake account identifier and username. Password is optional; the connector will open a browser login flow when no password is provided. Never commit `.env`.

## Run

```bash
python -m retailiq.cli setup
python -m retailiq.cli ingest --series 25 --batch-size 10
python -m retailiq.cli check
python -m retailiq.cli suspend
pytest -q
```

Expected dataset shape for the original `sales_train_validation.csv`: each series has 1,913 daily columns. `--series 25` ingests roughly 47,825 rows. The exact quantity is verified from your files at runtime.

## Validation query

```sql
SELECT ITEM_ID, STORE_ID, MIN(SALE_DATE), MAX(SALE_DATE), SUM(UNITS_SOLD)
FROM RETAILIQ.STAGING.DAILY_SALES
GROUP BY 1, 2
ORDER BY 1, 2
LIMIT 10;
```

## Design notes and constraints

- `RAW.DAILY_SALES` is a typed, long-format ingestion table. The source wide CSV is transformed locally in bounded batches before loading.
- Rerunning a batch replaces rows for the included item/store series, so retries do not create duplicate daily sales. One account should run the ingestion at a time.
- `STAGING.DAILY_SALES` is a validated consolidation view; `MART.WEEKLY_DEMAND` is an aggregate.
- The simple `INGEST_LOG` logs completed batches; it is not yet a production-grade lineage solution.
- The default 25 series cover only a small subset and should not be presented as representative of all products.
- On some accounts browser authentication and role permissions need adjustment. Setup requires create database and warehouse permissions.
- Connection commands require Snowflake access; offline unit tests run without credentials.
- Do not use a trial account for always-on or scheduled jobs. Monitor compute credits in the Snowflake console; warehouse resource monitors are not a complete serverless spending cap.

## Next milestone

Snowpark feature engineering (lags and rolling windows) with leakage-safe chronological splits; then XGBoost, MLflow, forecast tables, inventory simulation, Streamlit.

## Milestone 2: Snowpark feature engineering

After running `setup`, `ingest`, and `check`, create the feature views with:

```bash
python -m retailiq.cli features
python -m retailiq.cli suspend
pytest -q
```

This uses Snowpark's `Session.sql()` to execute `sql/03_features.sql` and verify
row counts. Results are views in `RETAILIQ.MART`:

- `DAILY_DEMAND_FEATURES`: one row per eligible item/store/date forecast origin;
  includes historical features and optional future labels.
- `TRAINING_FEATURES`: only origins with seven full future days of demand labels.

Inspect the views with the validation queries in
[`docs/feature_engineering.md`](docs/feature_engineering.md).
The next milestone is a chronological train/validation/test split, naive
baseline and local XGBoost training, using Snowflake feature export.

**Cost note:** View creation is cheap, but querying views runs window functions
on Snowflake compute. Keep your 25-series pilot and avoid repeated large queries.
