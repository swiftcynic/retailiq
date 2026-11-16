"""Offline feature-importance report for the trusted RetailIQ XGBoost pipeline.

Run: python -m retailiq.explain
No Snowflake access or model retraining is required.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .model import FEATURES


def feature_importance(model) -> pd.DataFrame:
    """Return XGBoost's fitted split-importance scores by transformed feature.

    This is model-specific feature importance, NOT SHAP values or a causal effect.
    """
    preprocessor = model.named_steps["preprocess"]
    booster = model.named_steps["model"]
    feature_names = preprocessor.get_feature_names_out()
    importances = np.asarray(booster.feature_importances_, dtype=float)
    if len(feature_names) != len(importances):
        raise ValueError("Preprocessing features and XGBoost importance lengths differ")
    if not np.isfinite(importances).all():
        raise ValueError("Feature importances contain non-finite values")
    return (pd.DataFrame({"feature": feature_names, "importance": importances})
            .sort_values("importance", ascending=False)
            .reset_index(drop=True))


def generate_report(model_path: str | Path = "artifacts/demand_xgboost.joblib",
                    output_dir: str | Path = "artifacts/explainability") -> dict:
    path = Path(model_path)
    if not path.is_file():
        raise FileNotFoundError(f"Saved model missing: {path}")
    # joblib uses pickle: load only the model you trained locally, not external files.
    model = joblib.load(path)
    importance = feature_importance(model)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    csv_path = output / "feature_importance.csv"
    png_path = output / "feature_importance.png"
    importance.to_csv(csv_path, index=False)

    top = importance.head(15).iloc[::-1]
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(top["feature"], top["importance"])
    ax.set_xlabel("XGBoost feature importance (normalized)")
    ax.set_title("RetailIQ: top 15 transformed features")
    fig.tight_layout()
    fig.savefig(png_path, dpi=160)
    plt.close(fig)
    return {"features": int(len(importance)), "csv": str(csv_path),
            "chart": str(png_path), "top_feature": str(importance.iloc[0]["feature"])}


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate offline model feature importance")
    parser.add_argument("--model", default="artifacts/demand_xgboost.joblib")
    parser.add_argument("--output-dir", default="artifacts/explainability")
    args = parser.parse_args()
    import json
    print(json.dumps(generate_report(args.model, args.output_dir), indent=2))


if __name__ == "__main__":
    main()
