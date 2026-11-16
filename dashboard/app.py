"""Run from project root: streamlit run dashboard/app.py"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from dotenv import load_dotenv

from retailiq.dashboard_data import get_forecasts, get_history, get_metadata, get_runs
from retailiq.inventory import recommend_order

load_dotenv()
st.set_page_config(page_title="RetailIQ | Demand Intelligence", page_icon="📦", layout="wide")
st.title("📦 RetailIQ")
st.caption("Snowflake-backed retail forecasting • Historical M5 data • 7-day demand predictions")


@st.cache_data(ttl=300, show_spinner=False)
def runs():
    return get_runs()


@st.cache_data(ttl=300, show_spinner=False)
def forecasts(version: str, origin: str):
    return get_forecasts(version, origin)


@st.cache_data(ttl=300, show_spinner=False)
def history(origin: str, item: str, store: str):
    return get_history(origin, item, store)


@st.cache_data(ttl=300, show_spinner=False)
def metadata(version: str, origin: str):
    return get_metadata(version, origin)


with st.sidebar:
    st.header("Controls")
    if st.button("Refresh Snowflake data"):
        st.cache_data.clear()
        st.rerun()
    st.caption("Queries run on demand and are cached for 5 minutes. An active Snowflake warehouse consumes trial credits.")

try:
    batches = runs()
except Exception as exc:
    st.error("Unable to read Snowflake. Verify .env, warehouse access and that the Milestone 4 predict command completed.")
    st.exception(exc)
    st.stop()

if batches.empty:
    st.warning("No saved forecasts yet. Run: python -m retailiq.cli predict")
    st.stop()

batches = batches.copy()
batches["FORECAST_ORIGIN"] = pd.to_datetime(batches["FORECAST_ORIGIN"]).dt.strftime("%Y-%m-%d")
batch_keys = [(str(r.MODEL_VERSION), str(r.FORECAST_ORIGIN)) for r in batches.itertuples()]
with st.sidebar:
    version, origin = st.selectbox("Forecast batch", options=batch_keys,
                                   format_func=lambda x: f"{x[1]} · {x[0][:8]}")

try:
    df = forecasts(version, origin)
except Exception as exc:
    st.error(f"Unable to fetch forecast batch: {exc}")
    st.stop()
if df.empty:
    st.warning("Selected batch has no forecast rows")
    st.stop()

stores = sorted(df["STORE_ID"].astype(str).unique().tolist())
with st.sidebar:
    store = st.selectbox("Store", ["All stores"] + stores)
    store_rows = df if store == "All stores" else df.loc[df["STORE_ID"].astype(str) == store]
    products = sorted(store_rows["ITEM_ID"].astype(str).unique().tolist())
    item = st.selectbox("Product", ["All products"] + products)

filtered = store_rows if item == "All products" else store_rows.loc[store_rows["ITEM_ID"].astype(str) == item]
start = pd.Timestamp(df.iloc[0]["FORECAST_START_DATE"]).strftime("%d %b %Y")
end = pd.Timestamp(df.iloc[0]["FORECAST_END_DATE"]).strftime("%d %b %Y")
st.info(f"Forecast: {start}–{end} (historical M5 period, **not a current forecast**).")

c1, c2, c3 = st.columns(3)
c1.metric("Predicted units (selected)", f"{filtered['PREDICTED_UNITS'].sum():,.1f}")
c2.metric("Selected series", f"{len(filtered):,}")
c3.metric("Model version", version[:8])

t1, t2, t3 = st.tabs(["📈 Demand forecasts", "📦 Inventory simulation", "🧪 Model performance"])
with t1:
    st.subheader("Forecasts by product and store")
    display = filtered[["STORE_ID", "ITEM_ID", "PREDICTED_UNITS"]].sort_values("PREDICTED_UNITS", ascending=False)
    st.dataframe(display, width="stretch", hide_index=True)
    st.download_button("Download selected forecasts (CSV)", display.to_csv(index=False), "retailiq_forecasts.csv", "text/csv")
    if len(filtered) == 1:
        row = filtered.iloc[0]
        try:
            past = history(origin, str(row["ITEM_ID"]), str(row["STORE_ID"]))
            if not past.empty:
                past["SALE_DATE"] = pd.to_datetime(past["SALE_DATE"])
                past["UNITS_SOLD"] = pd.to_numeric(past["UNITS_SOLD"])
                # Seven-day sums aligned to the selected forecast origin.
                past["DAYS_BEFORE_ORIGIN"] = (pd.Timestamp(origin) - past["SALE_DATE"]).dt.days
                past["WEEK_BEFORE"] = past["DAYS_BEFORE_ORIGIN"] // 7
                weekly = (past[past["WEEK_BEFORE"] < 13]
                          .groupby("WEEK_BEFORE")["UNITS_SOLD"].sum().sort_index(ascending=False))
                fig = go.Figure()
                fig.add_bar(x=[f"-{int(k)+1} wk" for k in weekly.index], y=weekly.values, name="Observed 7-day demand")
                fig.add_bar(x=["Forecast +1 wk"], y=[float(row["PREDICTED_UNITS"])], name="XGBoost forecast")
                fig.update_layout(title="Historical 7-day sales vs next 7-day forecast", yaxis_title="Units", xaxis_title="Window")
                st.plotly_chart(fig, width="stretch")
                st.caption("Observed windows are consecutive seven-day intervals ending at the forecast origin; the last bar is a model prediction, not an observed outcome.")
            else:
                st.info("No sales history available for this item-store combination")
        except Exception as exc:
            st.warning(f"Could not load historical sales: {exc}")
    else:
        st.caption("Select a single store and product in the sidebar to compare its historical demand with the forecast.")

with t2:
    st.subheader("Inventory replenishment scenario")
    st.warning("Scenario only: M5 does not contain actual inventory levels or replenishment lead times. All inventory inputs below are hypothetical.")
    if len(filtered) != 1:
        st.info("Choose a single store and product in the sidebar to simulate its stock requirements.")
    else:
        predicted = float(filtered.iloc[0]["PREDICTED_UNITS"])
        left, middle, right = st.columns(3)
        with left:
            stock = st.number_input("Assumed on-hand units", min_value=0, value=20, step=1)
            on_order = st.number_input("Assumed incoming units", min_value=0, value=0, step=1)
        with middle:
            lead = st.number_input("Assumed lead time (days)", min_value=0, max_value=90, value=7, step=1)
            safety = st.number_input("Assumed safety coverage (days)", min_value=0, max_value=90, value=3, step=1)
        with right:
            backorders = st.number_input("Assumed backorders", min_value=0, value=0, step=1)
        result = recommend_order(predicted, int(stock), int(lead), int(safety), int(on_order), int(backorders))
        a, b, c = st.columns(3)
        a.metric("Suggested order", f"{result['recommended_order']:,} units")
        b.metric("Estimated lead-time demand", f"{result['lead_time_demand']:.1f}")
        c.metric("Safety stock target", f"{result['safety_stock']:.1f}")
        st.caption("Policy assumes uniform daily demand = 7-day prediction ÷ 7. Order = max(0, ceil(lead-time demand + safety stock − inventory position)). It is not a measured stockout risk or optimised policy.")

with t3:
    st.subheader("Model evaluation metadata")
    try:
        runs_df = metadata(version, origin)
        if runs_df.empty:
            st.info("No model run metadata saved for this batch")
        else:
            record = runs_df.iloc[0]
            if pd.notna(record["EVALUATION_WAPE"]):
                st.metric("Backtest XGBoost WAPE", f"{100*float(record['EVALUATION_WAPE']):.2f}%")
            st.write("Model version:", str(record["MODEL_VERSION"]))
            st.write("Training cutoff:", str(record["TRAINING_CUTOFF"]))
            st.caption("The WAPE shown is historical evaluation metadata, not the error of this forecast batch.")
    except Exception as exc:
        st.warning(f"Could not load model metadata: {exc}")
    evaluation_path = Path("artifacts/evaluation.json")
    if evaluation_path.is_file():
        try:
            m = json.loads(evaluation_path.read_text())
            if "baseline" in m and "xgboost" in m:
                st.markdown("#### Local training evaluation")
                values = pd.DataFrame([
                    {"Model": "Baseline", "WAPE (%)": 100*float(m["baseline"]["wape"])},
                    {"Model": "XGBoost", "WAPE (%)": 100*float(m["xgboost"]["wape"])}])
                st.bar_chart(values.set_index("Model"))
                st.dataframe(values, hide_index=True)
                st.caption("Only compare these results with the saved model if evaluation.json belongs to the same model training run.")
        except (ValueError, KeyError, TypeError) as exc:
            st.warning(f"Could not parse local evaluation metrics: {exc}")
    else:
        st.info("Optional: keep artifacts/evaluation.json locally to show baseline comparison.")

st.divider()
st.caption("RetailIQ portfolio demo • No live retail inventory or real-time sales • Source: historical M5 data")
