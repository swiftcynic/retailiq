# Milestone 2 — Forecast feature engineering

## Forecast origin

Each row is an item-store series at `AS_OF_DATE` after that day's sales are known.
Historical features use values at or before `AS_OF_DATE`. The seven-day label is
`SUM(UNITS_SOLD)` over the *next* seven days. For production predictions, use
`DAILY_DEMAND_FEATURES` and drop the target columns; for training, use
`TRAINING_FEATURES`, which excludes rows without a full future label.

## Leakage controls

- Partition window functions by `ITEM_ID, STORE_ID`, ordered by date.
- Lags use past data; rolling means and standard deviations include only the
  completed origin day and earlier days.
- Require 28 consecutive days of history for each origin, with additional
  series continuity checks recommended if source data can contain internal gaps.
- A full future 7-day period is required before a row has `LABEL_AVAILABLE=TRUE`.
- Train/validation/test splits MUST use `AS_OF_DATE`, not random samples; purge
  7-day overlapping forecast targets at each split boundary.
- Never include `TARGET_NEXT_7D` or `LABEL_AVAILABLE` in XGBoost inputs.

## Inspect the views

```sql
SELECT * FROM RETAILIQ.MART.DAILY_DEMAND_FEATURES
ORDER BY ITEM_ID, STORE_ID, AS_OF_DATE DESC LIMIT 10;

SELECT COUNT(*), MIN(AS_OF_DATE), MAX(AS_OF_DATE)
FROM RETAILIQ.MART.TRAINING_FEATURES;

SELECT COUNT(*) AS N_BAD
FROM RETAILIQ.MART.DAILY_DEMAND_FEATURES
WHERE AVG_7 < 0 OR AVG_28 < 0 OR TARGET_NEXT_7D < 0;
```

## Important caveats

The first 28 complete historical days guarantee enough data for 28-day
aggregates, but `LAG_28` requires 28 **previous** observations and is null on
the first eligible origin. This is expected. Choose to filter or impute before
training. Day-level M5 series are dense; for arbitrary data, enforce a per-series
calendar and reject gaps explicitly before constructing rolling windows.

`DAILY_DEMAND_FEATURES` is a view, so repeated reads recompute window functions
and cost warehouse compute. For higher volumes, materialise a date-partitioned
feature table using an incremental process. For now, small datasets and
short-lived queries minimise credits.
