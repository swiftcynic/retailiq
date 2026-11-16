# Milestone 8: Rolling backtests + TreeSHAP

## Commands

Run from the project root with the activated environment:

```bash
python -m pip install -e '.[dev]'
python -m pip install matplotlib
python -m retailiq.backtest --fetch --folds 8
python -m retailiq.cli suspend
python -m retailiq.shap_report
pytest -q
```

`--fetch` reads `RETAILIQ.MART.TRAINING_FEATURES` from Snowflake once and creates
`artifacts/backtests/feature_snapshot.csv`. **All subsequent experiments can run
without Snowflake:**

```bash
python -m retailiq.backtest --folds 12
python -m retailiq.shap_report
```

Only use 12 folds if at least 12 weekly evaluation origins are available.

The saved snapshot can include future target-null rows; the runner excludes
those before training/evaluation. Do not commit data snapshots, fitted models,
or customer-like data to GitHub.

## Artifacts

- `artifacts/backtests/summary.json`: pooled MAE, RMSE, WAPE and improvement.
- `artifacts/backtests/fold_metrics.csv`: metrics for each origin.
- `artifacts/backtests/series_metrics.csv`: error metrics for every item-store series.
- `artifacts/backtests/predictions.csv`: held-out predictions for each fold.
- `artifacts/backtests/model_last_fold.joblib`: model fitted at last validation origin (kept separate from production model).
- `artifacts/backtests/shap/shap_importance.csv`: mean absolute/signed TreeSHAP contribution by transformed feature.
- `artifacts/backtests/shap/shap_importance.png`: top 15 feature contributions.
- `artifacts/backtests/shap/example_explanation.json`: additivity check for one prediction.

## Method

The backtest uses seven-day targets and nonoverlapping weekly origins.
For each held-out origin, retraining uses only training origins whose labels
end on or before the held-out origin. This prevents **target overlap** and
restricts the model to information available at the evaluation date.
The same fixed features and XGBoost hyperparameters from `model.py` are used.

Importantly, this is **expanding-window** evaluation, not hyperparameter tuning.
You should keep a final untouched test period if using these folds to select
models or model settings. Report performance using pooled predictions, not
an unweighted average of per-fold WAPEs. Series with zero total demand have
undefined WAPE (represented as missing).

TreeSHAP describes contributions to the fitted model's *raw predicted seven-day
units*, before the `max(0, prediction)` clipping applied in production.
The contributions are not causal effects and can be split across one-hot
encoded product/store indicator columns. Different correlated features can
share credit in unintuitive ways.

## Scaling to more series

The ingestion command from Milestone 1 may append duplicate rows when rerun.
Before increasing the series count, confirm ingestion is idempotent or rebuild
RAW/STAGING safely. `clean_snapshot` fails fast on duplicate
`(ITEM_ID, STORE_ID, AS_OF_DATE)` combinations.

This model's current feature view supports per-product/store comparisons, not
category-level reporting because category isn't included in its exported
feature columns. Category reporting requires adding a verified category
mapping from original M5 IDs; don't infer categories from item labels blindly.
