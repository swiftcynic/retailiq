"""Cost-conscious, reproducible seven-day batch forecasting into Snowflake.

Execute with ``python -m retailiq.cli predict`` after Milestone 3 training.
Joblib artifacts must be locally produced and trusted (pickle-based format).
"""
from __future__ import annotations

import hashlib
import json
import uuid
from datetime import timedelta
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from snowflake.connector.pandas_tools import write_pandas

from .model import FEATURES, NUMERIC_FEATURES, CATEGORICAL_FEATURES
from .snowflake_io import connect, execute_sql_file

FORECAST_QUERY = """
SELECT ITEM_ID, STORE_ID, AS_OF_DATE,
       LAG_1, LAG_7, LAG_14, LAG_28, AVG_7, AVG_28, STDDEV_28,
       DAY_OF_WEEK, MONTH_OF_YEAR
FROM RETAILIQ.MART.DAILY_DEMAND_FEATURES
WHERE AS_OF_DATE = (SELECT MAX(AS_OF_DATE) FROM RETAILIQ.MART.DAILY_DEMAND_FEATURES)
"""
SQL_FILE = Path(__file__).resolve().parents[2] / "sql" / "04_predictions.sql"


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prepare_inference(frame: pd.DataFrame) -> pd.DataFrame:
    """Coerce Snowflake types, reject duplicates and never use target labels."""
    required = set(FEATURES + ["AS_OF_DATE"])
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Missing inference columns: {sorted(missing)}")
    if frame.empty:
        raise ValueError("No feature rows found; run the features command first")
    df = frame[list(FEATURES) + ["AS_OF_DATE"]].copy()
    df["AS_OF_DATE"] = pd.to_datetime(df["AS_OF_DATE"], errors="coerce")
    if df["AS_OF_DATE"].isna().any():
        raise ValueError("Invalid AS_OF_DATE")
    if df["AS_OF_DATE"].dt.normalize().nunique() != 1:
        raise ValueError("Expected one common forecast origin across all series")
    for col in CATEGORICAL_FEATURES:
        if df[col].isna().any():
            raise ValueError(f"Null category: {col}")
        df[col] = df[col].astype(str)
    if df.duplicated(["ITEM_ID", "STORE_ID", "AS_OF_DATE"]).any():
        raise ValueError("Duplicate item-store forecast origins")
    for col in NUMERIC_FEATURES:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        if np.isinf(df[col].to_numpy(dtype=float)).any():
            raise ValueError(f"Infinite numeric feature: {col}")
    # Same missing-value policy as Milestone 3 train.prepare_frame.
    df[NUMERIC_FEATURES] = df[NUMERIC_FEATURES].fillna(0.0)
    return df


def forecast_frame(frame: pd.DataFrame, model, model_version: str, run_id: str) -> pd.DataFrame:
    df = prepare_inference(frame)
    values = np.asarray(model.predict(df[FEATURES]), dtype=float)
    if values.ndim != 1 or len(values) != len(df) or not np.isfinite(values).all():
        raise ValueError("Model produced invalid forecasts")
    origin = df["AS_OF_DATE"].dt.normalize()
    return pd.DataFrame({
        "RUN_ID": run_id,
        "MODEL_VERSION": model_version,
        "ITEM_ID": df["ITEM_ID"].to_numpy(),
        "STORE_ID": df["STORE_ID"].to_numpy(),
        "FORECAST_ORIGIN": origin.dt.date.to_numpy(),
        "FORECAST_START_DATE": (origin + pd.Timedelta(days=1)).dt.date.to_numpy(),
        "FORECAST_END_DATE": (origin + pd.Timedelta(days=7)).dt.date.to_numpy(),
        "PREDICTED_UNITS": np.maximum(0.0, values),
    })


def load_metadata(artifact_dir: Path) -> dict:
    path = artifact_dir / "evaluation.json"
    if not path.exists():
        return {}
    result = json.loads(path.read_text())
    return {
        "training_cutoff": result.get("last_training_origin"),
        "evaluation_wape": result.get("xgboost", {}).get("wape"),
    }


def write_forecasts(conn, forecasts: pd.DataFrame, sha256: str, metadata: dict):
    """Atomic MERGEs; same model/origin/series reruns update, not duplicate.

    We deliberately preserve the original RUN_ID/CREATED_AT when a matching
    prediction exists, since a rerun of identical input is not a new forecast.
    """
    with conn.cursor() as cur:
        cur.execute("USE WAREHOUSE RETAILIQ_WH")
        cur.execute("USE DATABASE RETAILIQ")
        cur.execute("USE SCHEMA MART")
    execute_sql_file(conn, SQL_FILE)
    with conn.cursor() as cur:
        cur.execute("CREATE OR REPLACE TEMPORARY TABLE BATCH_FORECASTS ("
                    "RUN_ID VARCHAR(64), MODEL_VERSION VARCHAR(64), ITEM_ID VARCHAR(128), "
                    "STORE_ID VARCHAR(128), FORECAST_ORIGIN DATE, FORECAST_START_DATE DATE, "
                    "FORECAST_END_DATE DATE, PREDICTED_UNITS FLOAT)")
    ok, _, loaded, _ = write_pandas(conn, forecasts, "BATCH_FORECASTS",
                                    database="RETAILIQ", schema="MART",
                                    quote_identifiers=False)
    if not ok or loaded != len(forecasts):
        raise RuntimeError(f"Forecast upload incomplete: {loaded}/{len(forecasts)}")
    with conn.cursor() as cur:
        # Explicit transaction: both forecast and metadata MERGEs succeed or neither.
        cur.execute("BEGIN")
        try:
            cur.execute("""
                MERGE INTO RETAILIQ.MART.DEMAND_FORECASTS t
                USING BATCH_FORECASTS s
                ON t.MODEL_VERSION = s.MODEL_VERSION
                AND t.ITEM_ID = s.ITEM_ID AND t.STORE_ID = s.STORE_ID
                AND t.FORECAST_ORIGIN = s.FORECAST_ORIGIN
                WHEN MATCHED THEN UPDATE SET
                    t.PREDICTED_UNITS = s.PREDICTED_UNITS,
                    t.FORECAST_START_DATE = s.FORECAST_START_DATE,
                    t.FORECAST_END_DATE = s.FORECAST_END_DATE
                WHEN NOT MATCHED THEN INSERT
                    (RUN_ID, MODEL_VERSION, ITEM_ID, STORE_ID, FORECAST_ORIGIN,
                     FORECAST_START_DATE, FORECAST_END_DATE, PREDICTED_UNITS)
                VALUES
                    (s.RUN_ID, s.MODEL_VERSION, s.ITEM_ID, s.STORE_ID, s.FORECAST_ORIGIN,
                     s.FORECAST_START_DATE, s.FORECAST_END_DATE, s.PREDICTED_UNITS)
            """)
            # A run is unique by model version and forecast origin for a full snapshot.
            cur.execute("""
                MERGE INTO RETAILIQ.MART.MODEL_RUNS t
                USING (SELECT %s AS RUN_ID, %s AS MODEL_VERSION,
                              %s AS MODEL_FILE_SHA256, %s::DATE AS FORECAST_ORIGIN,
                              %s AS SERIES_COUNT, %s::DATE AS TRAINING_CUTOFF,
                              %s AS EVALUATION_WAPE) s
                ON t.MODEL_VERSION = s.MODEL_VERSION
                AND t.FORECAST_ORIGIN = s.FORECAST_ORIGIN
                WHEN NOT MATCHED THEN INSERT
                    (RUN_ID, MODEL_VERSION, MODEL_FILE_SHA256, FORECAST_ORIGIN,
                     SERIES_COUNT, TRAINING_CUTOFF, EVALUATION_WAPE)
                VALUES (s.RUN_ID, s.MODEL_VERSION, s.MODEL_FILE_SHA256,
                        s.FORECAST_ORIGIN, s.SERIES_COUNT, s.TRAINING_CUTOFF,
                        s.EVALUATION_WAPE)
            """, (
                forecasts["RUN_ID"].iloc[0], forecasts["MODEL_VERSION"].iloc[0],
                sha256, str(forecasts["FORECAST_ORIGIN"].iloc[0]), len(forecasts),
                metadata.get("training_cutoff"), metadata.get("evaluation_wape"),
            ))
            cur.execute("COMMIT")
        except Exception:
            cur.execute("ROLLBACK")
            raise


def predict(artifact_dir: str | Path = "artifacts", dry_run: bool = False) -> dict:
    path = Path(artifact_dir)
    model_path = path / "demand_xgboost.joblib"
    if not model_path.is_file():
        raise FileNotFoundError(f"Trained model not found: {model_path}. Run train first.")
    sha256 = file_hash(model_path)
    model_version = sha256[:16]
    # Never load downloaded/untrusted joblib artifacts.
    model = joblib.load(model_path)
    metadata = load_metadata(path)
    with connect() as conn:
        with conn.cursor() as cur:
            cur.execute("USE WAREHOUSE RETAILIQ_WH")
            cur.execute(FORECAST_QUERY)
            features = cur.fetch_pandas_all()
        run_id = uuid.uuid4().hex
        forecasts = forecast_frame(features, model, model_version, run_id)
        if not dry_run:
            write_forecasts(conn, forecasts, sha256, metadata)
    return {
        "run_id": run_id,
        "model_version": model_version,
        "forecast_origin": str(forecasts["FORECAST_ORIGIN"].iloc[0]),
        "forecast_start_date": str(forecasts["FORECAST_START_DATE"].iloc[0]),
        "forecast_end_date": str(forecasts["FORECAST_END_DATE"].iloc[0]),
        "series": len(forecasts),
        "total_predicted_units": round(float(forecasts["PREDICTED_UNITS"].sum()), 2),
        "dry_run": dry_run,
    }
