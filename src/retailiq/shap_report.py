"""Local TreeSHAP interpretation of the last rolling-backtest XGBoost model.

Uses XGBoost pred_contribs (TreeSHAP), so the shap package is not required.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb

from .model import FEATURES, TARGET
from .backtest import clean_snapshot


def compute_contributions(model, features: pd.DataFrame):
    """Return per-feature SHAP contributions, bias and raw model predictions."""
    if features.empty:
        raise ValueError("Cannot explain empty dataset")
    preprocess = model.named_steps["preprocess"]
    estimator = model.named_steps["model"]
    encoded = preprocess.transform(features[FEATURES])
    names = [str(n) for n in preprocess.get_feature_names_out()]
    matrix = xgb.DMatrix(encoded, feature_names=names)
    result = estimator.get_booster().predict(matrix, pred_contribs=True)
    result = np.asarray(result)
    if result.ndim != 2 or result.shape[1] != len(names)+1:
        raise ValueError("Unexpected XGBoost contribution dimensions")
    raw_predictions = np.asarray(estimator.predict(encoded), dtype=float)
    if not np.allclose(result.sum(axis=1), raw_predictions, atol=1e-3, rtol=1e-4):
        raise ValueError("TreeSHAP values do not sum to model predictions")
    return names, result[:, :-1], result[:, -1], raw_predictions


def create_report(snapshot_path: str | Path = "artifacts/backtests/feature_snapshot.csv",
                  model_path: str | Path = "artifacts/backtests/model_last_fold.joblib",
                  output_dir: str | Path = "artifacts/backtests/shap",
                  sample_size: int = 200) -> dict:
    if sample_size < 1:
        raise ValueError("sample_size must be positive")
    snapshot = clean_snapshot(pd.read_csv(snapshot_path))
    last_origin = snapshot["AS_OF_DATE"].max()
    heldout = snapshot[snapshot["AS_OF_DATE"] == last_origin].copy()
    if heldout.empty:
        raise ValueError("No held-out rows for last origin")
    heldout = heldout.sample(n=min(sample_size, len(heldout)), random_state=42)
    # joblib is pickle-based: use only models that you trained locally.
    model = joblib.load(model_path)
    names, values, biases, raw = compute_contributions(model, heldout)
    report = (pd.DataFrame({"feature": names, "mean_abs_shap": np.abs(values).mean(axis=0),
                            "mean_signed_shap": values.mean(axis=0)})
              .sort_values("mean_abs_shap", ascending=False).reset_index(drop=True))
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    report.to_csv(output / "shap_importance.csv", index=False)
    top = report.head(15).iloc[::-1]
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(top["feature"], top["mean_abs_shap"])
    ax.set_xlabel("Mean absolute TreeSHAP contribution (seven-day units)")
    ax.set_title(f"RetailIQ — {len(heldout)} sampled rows at {last_origin.date()}")
    fig.tight_layout()
    fig.savefig(output / "shap_importance.png", dpi=160)
    plt.close(fig)
    # One row audit demonstrates the additive decomposition of a raw forecast.
    first = {"origin": last_origin.strftime("%Y-%m-%d"), "item_id": str(heldout.iloc[0]["ITEM_ID"]),
             "store_id": str(heldout.iloc[0]["STORE_ID"]),
             "expected_value": float(biases[0]), "raw_prediction": float(raw[0]),
             "contribution_sum": float(biases[0] + values[0].sum()),
             "note": "Raw model output may be negative; deployed forecasts are clipped at zero."}
    (output / "example_explanation.json").write_text(json.dumps(first, indent=2) + "\n")
    return {"origin": first["origin"], "explained_rows": int(len(heldout)),
            "features": len(names), "top_feature": str(report.iloc[0]["feature"]),
            "chart": str(output / "shap_importance.png")}


def main():
    parser = argparse.ArgumentParser(description="Local XGBoost TreeSHAP report")
    parser.add_argument("--snapshot", default="artifacts/backtests/feature_snapshot.csv")
    parser.add_argument("--model", default="artifacts/backtests/model_last_fold.joblib")
    parser.add_argument("--output-dir", default="artifacts/backtests/shap")
    parser.add_argument("--sample-size", type=int, default=200)
    args = parser.parse_args()
    print(json.dumps(create_report(args.snapshot, args.model, args.output_dir, args.sample_size), indent=2))


if __name__ == "__main__":
    main()
