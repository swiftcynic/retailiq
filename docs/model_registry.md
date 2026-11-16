# Milestone 7 — Explainability and MLflow Model Registry

## Changed files only

- NEW `src/retailiq/explain.py` — saved-model feature importance CSV and chart
- NEW `src/retailiq/register_model.py` — logs and registers the fitted sklearn/XGBoost pipeline
- NEW `tests/test_explain.py` — offline feature-importance test
- NEW `docs/model_registry.md` — these instructions

No existing project files need replacement.

## Prerequisites

Start from the project root with your Python environment active:

```bash
python -m pip install mlflow matplotlib
```

Earlier training should have created:

- `artifacts/demand_xgboost.joblib`
- `artifacts/evaluation.json`
- (optional) `artifacts/evaluation_predictions.csv`

Only unpickle artifacts that you have created locally or that come from a trusted source.

## Run

```bash
python -m retailiq.explain
python -m retailiq.register_model
pytest -q
```

The scripts do not access Snowflake and do not retrain the model. Registering repeatedly produces additional MLflow model versions, so run the registration command only when you intend to create a version.

Start the UI from project root (after stopping any old MLflow server):

```bash
mlflow ui --backend-store-uri "sqlite:///$(pwd)/artifacts/mlflow/mlflow.db"
```

In the **Model training** UI, navigate to experiment `RetailIQ Demand Forecasting`, run `xgboost-registered-model` to see the logged explainability chart and metrics. Find the registered pipeline named `RetailIQ-Demand-XGBoost` under **Models**.

## Interpretation

The chart shows XGBoost's built-in split-based feature importance for the transformed (including one-hot-encoded) features. It is not SHAP, does not imply causation, and can differ from permutation importance.

## Privacy and cost

Both commands operate locally on existing model artifacts, using SQLite. No Snowflake credits are consumed. `artifacts/` should remain gitignored because it contains model serializations, a local database, and MLflow runs.
