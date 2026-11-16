# Milestone 6 — Quality gates and reproducible orchestration

Only changed/new files were delivered; retain Milestone 5 files.

## Commands

```bash
python -m retailiq.cli quality
python -m retailiq.cli pipeline
pytest -q

# Optional local experiment tracking
python -m pip install mlflow
python -m retailiq.cli pipeline --track-mlflow
mlflow ui --backend-store-uri ./artifacts/mlruns
```

`pipeline` runs four sequential steps: read-only data-quality checks, rebuild Snowflake feature views, train locally, and batch-predict to Snowflake. It **does not ingest** CSVs, preventing repeated data loads. It suspends `RETAILIQ_WH` on both success and failure, and writes a local JSON report under `artifacts/pipeline_runs/`. Reports are ignored by Git; do not put credentials in them.

The warehouse is used only for feature queries and stored predictions. Model training runs on your computer. Avoid running `pipeline` while your Streamlit dashboard is in active use, since warehouse suspension may interrupt a dashboard query.

## Data-quality checks

`quality` checks for empty sales, duplicate item/store/date keys, null keys, and null/negative unit sales. The checks inspect `RETAILIQ.STAGING.DAILY_SALES`, not the M5 source CSV. This is a minimal gate, not a full end-to-end data contract; add date continuity and volume drift checks in a later increment.

## GitHub Actions

`.github/workflows/tests.yml` executes pytest without Snowflake credentials. Do not commit `.env`, source CSVs, or trained `joblib` files. The GitHub Action is *not* scheduled to access Snowflake, avoiding trial-credit usage.

## Safety and limitations

- The project predicts historical April 2016 demand, not real-time current demand.
- The training workflow is a fixed chronological holdout with four evaluated origins, **not** retraining at every origin.
- Running the pipeline retrains your local model and overwrites its previous `artifacts/demand_xgboost.joblib` and evaluation files. If you need to compare runs, archive them first.
- Inference `MERGE` is idempotent for the same model version, forecast origin, and product-store series.
- A failed pipeline writes a failure report, suspends the warehouse when possible, and exits nonzero.
- No recurring cloud pipeline is enabled; you explicitly run it to control cost.

## Future enhancements

More rigorous rolling backtests, data-contract tests, and explicit compute usage reporting can be added after this pipeline is verified against Snowflake.
