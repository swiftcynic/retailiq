"""Read-only data-quality gates on the Snowflake training source."""
from .snowflake_io import connect

# Checks are evaluated in the warehouse so no full dataset is downloaded.
QUALITY_QUERY = """
SELECT COUNT(*) AS ROWS_CHECKED,
       COUNT(DISTINCT ITEM_ID || ':' || STORE_ID || ':' || TO_VARCHAR(SALE_DATE)) AS DISTINCT_KEYS,
       COUNT_IF(ITEM_ID IS NULL OR STORE_ID IS NULL OR SALE_DATE IS NULL) AS NULL_KEYS,
       COUNT_IF(UNITS_SOLD IS NULL OR UNITS_SOLD < 0) AS INVALID_UNITS,
       COUNT(DISTINCT ITEM_ID || ':' || STORE_ID) AS SERIES
FROM RETAILIQ.STAGING.DAILY_SALES
"""


def validate_counts(rows_checked, distinct_keys, null_keys, invalid_units, series):
    """Pure validation for unit testing and re-use."""
    fields = (rows_checked, distinct_keys, null_keys, invalid_units, series)
    if any(v is None for v in fields):
        raise ValueError("Quality query returned null counters")
    summary = dict(rows_checked=int(rows_checked), distinct_keys=int(distinct_keys),
                   null_keys=int(null_keys), invalid_units=int(invalid_units),
                   series=int(series))
    if summary["rows_checked"] < 1 or summary["series"] < 1:
        raise ValueError("No retail sales records found")
    if summary["distinct_keys"] != summary["rows_checked"]:
        raise ValueError("Duplicate item/store/date sales keys detected")
    if summary["null_keys"] or summary["invalid_units"]:
        raise ValueError(f"Invalid retail sales records: {summary}")
    return summary


def run_quality_checks():
    with connect() as conn:
        with conn.cursor() as cur:
            cur.execute("USE WAREHOUSE RETAILIQ_WH")
            cur.execute(QUALITY_QUERY)
            result = cur.fetchone()
    return validate_counts(*result)
