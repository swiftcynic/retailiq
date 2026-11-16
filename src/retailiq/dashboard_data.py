"""Bounded read-only Snowflake queries for the local Streamlit dashboard."""
from __future__ import annotations

import pandas as pd

from .snowflake_io import connect


RUN_SQL = """
SELECT MODEL_VERSION, FORECAST_ORIGIN, MIN(FORECAST_START_DATE) AS FORECAST_START_DATE,
       MAX(FORECAST_END_DATE) AS FORECAST_END_DATE, COUNT(*) AS SERIES_COUNT,
       ROUND(SUM(PREDICTED_UNITS), 2) AS TOTAL_PREDICTED_UNITS
FROM RETAILIQ.MART.DEMAND_FORECASTS
GROUP BY MODEL_VERSION, FORECAST_ORIGIN
ORDER BY FORECAST_ORIGIN DESC, MODEL_VERSION DESC
LIMIT 30
"""

FORECAST_SQL = """
SELECT ITEM_ID, STORE_ID, MODEL_VERSION, FORECAST_ORIGIN,
       FORECAST_START_DATE, FORECAST_END_DATE, PREDICTED_UNITS
FROM RETAILIQ.MART.DEMAND_FORECASTS
WHERE MODEL_VERSION = %s AND FORECAST_ORIGIN = %s::DATE
ORDER BY STORE_ID, ITEM_ID
LIMIT 5000
"""

HISTORY_SQL = """
SELECT ITEM_ID, STORE_ID, SALE_DATE, UNITS_SOLD
FROM RETAILIQ.STAGING.DAILY_SALES
WHERE SALE_DATE >= DATEADD(day, -91, %s::DATE)
  AND SALE_DATE <= %s::DATE
  AND ITEM_ID = %s AND STORE_ID = %s
ORDER BY SALE_DATE
LIMIT 100
"""

METADATA_SQL = """
SELECT MODEL_VERSION, FORECAST_ORIGIN, TRAINING_CUTOFF,
       EVALUATION_WAPE, SERIES_COUNT, CREATED_AT
FROM RETAILIQ.MART.MODEL_RUNS
WHERE MODEL_VERSION = %s AND FORECAST_ORIGIN = %s::DATE
ORDER BY CREATED_AT DESC LIMIT 1
"""


def _read(query: str, params: tuple = ()) -> pd.DataFrame:
    with connect() as conn:
        with conn.cursor() as cur:
            cur.execute("USE WAREHOUSE RETAILIQ_WH")
            cur.execute(query, params)
            return cur.fetch_pandas_all()


def get_runs() -> pd.DataFrame:
    return _read(RUN_SQL)


def get_forecasts(model_version: str, forecast_origin: str) -> pd.DataFrame:
    return _read(FORECAST_SQL, (model_version, forecast_origin))


def get_history(origin: str, item_id: str, store_id: str) -> pd.DataFrame:
    return _read(HISTORY_SQL, (origin, origin, item_id, store_id))


def get_metadata(model_version: str, forecast_origin: str) -> pd.DataFrame:
    return _read(METADATA_SQL, (model_version, forecast_origin))
