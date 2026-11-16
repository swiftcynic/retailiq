"""Expanding-window, leakage-aware seven-day demand backtests.

Run `python -m retailiq.backtest --fetch` once to snapshot Snowflake data.
Then rerun experiments offline without further warehouse queries.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from dotenv import load_dotenv

from .model import FEATURES, TARGET, build_model, fetch_training_data, metrics, prepare_frame


def clean_snapshot(frame: pd.DataFrame) -> pd.DataFrame:
    """Exclude unknown future labels, then apply exactly the training preprocessing."""
    if TARGET not in frame.columns:
        raise ValueError(f"Missing {TARGET}")
    frame = frame.copy()
    frame[TARGET] = pd.to_numeric(frame[TARGET], errors="coerce")
    frame = frame.dropna(subset=[TARGET])
    df = prepare_frame(frame)
    if df.duplicated(["ITEM_ID", "STORE_ID", "AS_OF_DATE"]).any():
        raise ValueError("Duplicate item/store/origin rows: verify Snowflake ingestion")
    return df.sort_values(["AS_OF_DATE", "ITEM_ID", "STORE_ID"]).reset_index(drop=True)


def backtest(frame: pd.DataFrame, n_folds: int = 8, min_training_rows: int = 100,
             output_dir: str | Path = "artifacts/backtests") -> dict:
    """Retrain at each origin, using only labels fully observed at that origin.

    For a seven-day target, a training origin T is eligible only if T+7d <=
    the evaluated origin. Test origin dates are seven days apart.
    """
    if n_folds < 2:
        raise ValueError("n_folds must be at least 2")
    df = clean_snapshot(frame)
    origins = pd.DatetimeIndex(df["AS_OF_DATE"].unique()).sort_values()
    if origins.empty:
        raise ValueError("No labelled forecast origins")
    last = origins[-1]
    requested = [last - pd.Timedelta(days=7*i) for i in range(n_folds)][::-1]
    if any(t not in origins for t in requested):
        raise ValueError("Not enough complete weekly origins for requested folds")

    fold_rows = []
    prediction_frames = []
    most_recent_model = None
    for fold_no, origin in enumerate(requested, start=1):
        cutoff = origin - pd.Timedelta(days=7)
        training = df.loc[df["AS_OF_DATE"] <= cutoff]
        testing = df.loc[df["AS_OF_DATE"] == origin]
        if len(training) < min_training_rows or testing.empty:
            raise ValueError(f"Fold {fold_no}: insufficient training/test rows")
        assert training["AS_OF_DATE"].max() + pd.Timedelta(days=7) <= origin
        model = build_model()
        model.fit(training[FEATURES], training[TARGET])
        predicted = np.maximum(0, model.predict(testing[FEATURES]).astype(float))
        baseline = np.maximum(0, 7 * testing["AVG_7"].to_numpy(dtype=float))
        truth = testing[TARGET].to_numpy(dtype=float)
        fold_rows.append({
            "fold": fold_no, "origin": origin.strftime("%Y-%m-%d"),
            "training_cutoff": cutoff.strftime("%Y-%m-%d"),
            "training_rows": int(len(training)), "test_rows": int(len(testing)),
            **{f"baseline_{k}": v for k, v in metrics(truth, baseline).items()},
            **{f"xgboost_{k}": v for k, v in metrics(truth, predicted).items()},
        })
        predictions = testing[["ITEM_ID", "STORE_ID", "AS_OF_DATE", TARGET]].copy()
        predictions["BASELINE_FORECAST"] = baseline
        predictions["XGBOOST_FORECAST"] = predicted
        predictions["FOLD"] = fold_no
        prediction_frames.append(predictions)
        most_recent_model = model

    all_predictions = pd.concat(prediction_frames, ignore_index=True)
    fold_metrics = pd.DataFrame(fold_rows)
    truth = all_predictions[TARGET].to_numpy(dtype=float)
    summary = {
        "folds": n_folds, "evaluation_rows": int(len(all_predictions)),
        "distinct_series": int(all_predictions[["ITEM_ID", "STORE_ID"]].drop_duplicates().shape[0]),
        "evaluation_origins": [d.strftime("%Y-%m-%d") for d in requested],
        "baseline": metrics(truth, all_predictions["BASELINE_FORECAST"]),
        "xgboost": metrics(truth, all_predictions["XGBOOST_FORECAST"]),
        "note": "Rolling-origin expanding-window backtest; aggregate metrics pool all held-out predictions.",
    }
    bw, xw = summary["baseline"]["wape"], summary["xgboost"]["wape"]
    summary["wape_improvement_percent"] = round(100*(bw-xw)/bw, 2) if bw not in (None, 0) and xw is not None else None

    per_series = []
    for (item, store), group in all_predictions.groupby(["ITEM_ID", "STORE_ID"]):
        y = group[TARGET].to_numpy(dtype=float)
        per_series.append({"ITEM_ID": item, "STORE_ID": store, "rows": len(group),
                           "baseline_wape": metrics(y, group["BASELINE_FORECAST"])["wape"],
                           "xgboost_wape": metrics(y, group["XGBOOST_FORECAST"])["wape"]})

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    fold_metrics.to_csv(output / "fold_metrics.csv", index=False)
    all_predictions.to_csv(output / "predictions.csv", index=False)
    pd.DataFrame(per_series).to_csv(output / "series_metrics.csv", index=False)
    joblib.dump(most_recent_model, output / "model_last_fold.joblib")
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main():
    parser = argparse.ArgumentParser(description="Leakage-safe seven-day rolling backtest")
    parser.add_argument("--fetch", action="store_true", help="Fetch one Snowflake snapshot (costs credits)")
    parser.add_argument("--input", default="artifacts/backtests/feature_snapshot.csv")
    parser.add_argument("--output-dir", default="artifacts/backtests")
    parser.add_argument("--folds", type=int, default=8)
    args = parser.parse_args()
    snapshot = Path(args.input)
    if args.fetch:
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        fetch_training_data().to_csv(snapshot, index=False)
        print(f"Saved snapshot: {snapshot}")
    if not snapshot.exists():
        parser.error("Snapshot missing. Run with --fetch once to read Snowflake.")
    report = backtest(pd.read_csv(snapshot), args.folds, output_dir=args.output_dir)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    load_dotenv()
    main()
