# Milestone 3 — Local model training

Run from the project root after `python -m retailiq.cli features`:

```bash
python -m pip install -e '.[dev]'
python -m retailiq.cli train
pytest -q
python -m retailiq.cli suspend
```

`train` reads `RETAILIQ.MART.TRAINING_FEATURES` using the X-Small Snowflake warehouse, closes the connection, and trains XGBoost **locally**. Outputs appear under `artifacts/` (gitignored):

- `demand_xgboost.joblib` — trained preprocessing and model pipeline
- `evaluation.json` — training/evaluation counts, baseline and XGBoost metrics
- `evaluation_predictions.csv` — out-of-sample model and baseline forecasts

**Evaluation design:** The four latest available weekly-spaced forecast origins are held out. A single model is trained using only rows whose *entire 7-day target window* is complete by the earliest held-out forecast origin. No future label or overlapping test target is used for training. The seasonal baseline is `7 * AVG_7`, using only the previous seven completed days (including origin day). Evaluation is on 7-day item-store demand; this is a simple fixed-model backtest, not a fully retrained walk-forward evaluation.

**Limitations:** 25 item-store series is enough for a pipeline proof but not a convincing model benchmark. Seasonal naive may outperform XGBoost; report this without hiding it. Expansion to more series, additional calendar/price features, and a rolling retrained evaluation are future improvements. Model artifacts produced by joblib should never be loaded from untrusted sources. The training query does not include SQL time limits; use resource monitors and avoid running training repeatedly.
