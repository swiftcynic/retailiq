"""Local XGBoost demand forecasting, with Snowflake as the feature source.

Usage: python -m retailiq.cli train
"""
from __future__ import annotations

import json
from datetime import timedelta
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBRegressor

from .snowflake_io import connect

NUMERIC_FEATURES = [
    "LAG_1", "LAG_7", "LAG_14", "LAG_28", "AVG_7", "AVG_28",
    "STDDEV_28", "DAY_OF_WEEK", "MONTH_OF_YEAR",
]
CATEGORICAL_FEATURES = ["ITEM_ID", "STORE_ID"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
TARGET = "TARGET_NEXT_7D"
QUERY = """
SELECT ITEM_ID, STORE_ID, AS_OF_DATE,
       LAG_1, LAG_7, LAG_14, LAG_28, AVG_7, AVG_28, STDDEV_28,
       DAY_OF_WEEK, MONTH_OF_YEAR, TARGET_NEXT_7D
FROM RETAILIQ.MART.TRAINING_FEATURES
"""


def fetch_training_data() -> pd.DataFrame:
    """Fetch a small training mart to train locally (avoids Snowflake ML compute)."""
    with connect() as conn:
        with conn.cursor() as cur:
            cur.execute("USE WAREHOUSE RETAILIQ_WH")
            cur.execute(QUERY)
            frame = cur.fetch_pandas_all()
    return frame


def prepare_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Normalise Snowflake types and fail early on invalid training data."""
    required = set(FEATURES + ["AS_OF_DATE", TARGET])
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing training columns: {sorted(missing)}")
    df = frame.copy()
    df["AS_OF_DATE"] = pd.to_datetime(df["AS_OF_DATE"])
    for col in NUMERIC_FEATURES + [TARGET]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    if df[TARGET].isna().any() or (df[TARGET] < 0).any():
        raise ValueError("Invalid demand targets")
    for col in CATEGORICAL_FEATURES:
        if df[col].isna().any():
            raise ValueError(f"Null category: {col}")
        df[col] = df[col].astype(str)
    if df["AS_OF_DATE"].isna().any():
        raise ValueError("Invalid forecast origin date")
    # The earliest valid feature row may have a missing LAG_28.
    df[NUMERIC_FEATURES] = df[NUMERIC_FEATURES].fillna(0.0)
    return df


def chronological_split(df: pd.DataFrame, evaluation_weeks: int = 4):
    """Fit once before the first evaluated origin; evaluate every 7 days.

    Training labels have fully resolved by the earliest test origin. Since each
    label spans the NEXT 7 days, origins used to train must be at least 7 days
    earlier. No random splitting or overlapping held-out weekly targets.
    """
    if evaluation_weeks < 1:
        raise ValueError("evaluation_weeks must be >= 1")
    dates = pd.DatetimeIndex(df["AS_OF_DATE"].drop_duplicates().sort_values())
    if dates.empty:
        raise ValueError("No training data")
    last = dates[-1]
    test_dates = [last - pd.Timedelta(days=7 * i) for i in range(evaluation_weeks)]
    if not all(d in dates for d in test_dates):
        raise ValueError("Not enough continuous history for evaluation")
    first_test = min(test_dates)
    cutoff = first_test - pd.Timedelta(days=7)
    train = df[df["AS_OF_DATE"] <= cutoff].copy()
    test = df[df["AS_OF_DATE"].isin(test_dates)].copy()
    if len(train) < 100 or len(test) == 0:
        raise ValueError("Insufficient training/test rows; ingest more history")
    assert train["AS_OF_DATE"].max() + timedelta(days=7) <= test["AS_OF_DATE"].min()
    return train, test, cutoff


def build_model() -> Pipeline:
    preprocessing = ColumnTransformer(
        transformers=[
            ("numeric", "passthrough", NUMERIC_FEATURES),
            ("category", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ]
    )
    regressor = XGBRegressor(
        n_estimators=250, max_depth=4, learning_rate=0.05,
        subsample=0.9, colsample_bytree=0.9,
        objective="reg:squarederror", n_jobs=2, random_state=42,
        tree_method="hist",
    )
    return Pipeline([("preprocess", preprocessing), ("model", regressor)])


def metrics(actual, predictions) -> dict:
    y = np.asarray(actual, dtype=float)
    pred = np.maximum(0.0, np.asarray(predictions, dtype=float))
    if len(y) == 0 or len(y) != len(pred):
        raise ValueError("Metrics need matching nonempty arrays")
    denom = np.abs(y).sum()
    return {
        "mae": round(float(mean_absolute_error(y, pred)), 4),
        "rmse": round(float(np.sqrt(mean_squared_error(y, pred))), 4),
        "wape": round(float(np.abs(y - pred).sum() / denom), 6) if denom > 0 else None,
    }


def train(frame: pd.DataFrame, output_dir: str | Path = "artifacts") -> dict:
    df = prepare_frame(frame)
    training, test, cutoff = chronological_split(df)
    model = build_model()
    model.fit(training[FEATURES], training[TARGET])
    predicted = model.predict(test[FEATURES])
    # Prior 7 completed days are known at forecast origin; no future input.
    baseline = 7.0 * test["AVG_7"].to_numpy(dtype=float)
    results = {
        "training_rows": len(training),
        "evaluation_rows": len(test),
        "evaluation_origins": [d.strftime("%Y-%m-%d") for d in sorted(test["AS_OF_DATE"].unique())],
        "last_training_origin": cutoff.strftime("%Y-%m-%d"),
        "baseline": metrics(test[TARGET], baseline),
        "xgboost": metrics(test[TARGET], predicted),
    }
    if results["baseline"]["wape"] not in (None, 0):
        results["wape_improvement_percent"] = round(
            100 * (results["baseline"]["wape"] - results["xgboost"]["wape"])
            / results["baseline"]["wape"], 2
        )
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, out / "demand_xgboost.joblib")
    (out / "evaluation.json").write_text(json.dumps(results, indent=2) + "\n")
    predictions = test[["ITEM_ID", "STORE_ID", "AS_OF_DATE", TARGET]].copy()
    predictions["BASELINE_FORECAST"] = np.maximum(0, baseline)
    predictions["XGBOOST_FORECAST"] = np.maximum(0, predicted)
    predictions.to_csv(out / "evaluation_predictions.csv", index=False)
    return results
