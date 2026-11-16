"""Pure functions for the M5 wide-to-long ingestion step."""
import pandas as pd

REQUIRED_SALES_COLUMNS = {"item_id", "store_id"}
REQUIRED_CALENDAR_COLUMNS = {"d", "date"}


def to_daily_sales(wide: pd.DataFrame, calendar: pd.DataFrame) -> pd.DataFrame:
    """Convert M5 daily sales from d_1, d_2, ... into a validated long table."""
    if not REQUIRED_SALES_COLUMNS.issubset(wide.columns):
        raise ValueError("sales file must contain item_id and store_id")
    if not REQUIRED_CALENDAR_COLUMNS.issubset(calendar.columns):
        raise ValueError("calendar file must contain d and date")
    day_columns = [c for c in wide.columns if c.startswith("d_") and c[2:].isdigit()]
    if not day_columns:
        raise ValueError("No daily sales columns found (expected d_1, d_2, ...)")
    long = wide.melt(id_vars=["item_id", "store_id"], value_vars=day_columns,
                     var_name="DAY_KEY", value_name="UNITS_SOLD")
    dates = calendar[["d", "date"]].drop_duplicates(subset=["d"])
    long = long.merge(dates, left_on="DAY_KEY", right_on="d", how="left", validate="many_to_one")
    if long["date"].isna().any():
        raise ValueError("Some day keys have no matching calendar date")
    long["UNITS_SOLD"] = pd.to_numeric(long["UNITS_SOLD"], errors="raise")
    if long["UNITS_SOLD"].isna().any() or (long["UNITS_SOLD"] < 0).any():
        raise ValueError("Missing or negative sales found")
    if not (long["UNITS_SOLD"] % 1 == 0).all():
        raise ValueError("Sales units must be integers")
    result = long.rename(columns={"item_id": "ITEM_ID", "store_id": "STORE_ID", "date": "SALE_DATE"})
    result["SALE_DATE"] = pd.to_datetime(result["SALE_DATE"], errors="raise").dt.date
    result["UNITS_SOLD"] = result["UNITS_SOLD"].astype("int64")
    result = result[["ITEM_ID", "STORE_ID", "DAY_KEY", "SALE_DATE", "UNITS_SOLD"]]
    if result.duplicated(["ITEM_ID", "STORE_ID", "SALE_DATE"]).any():
        raise ValueError("Duplicate item-store-date rows found")
    return result
