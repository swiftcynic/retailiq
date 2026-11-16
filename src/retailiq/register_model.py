"""Register an existing, locally trained RetailIQ pipeline in local MLflow.

Run: python -m retailiq.register_model
This does not call Snowflake or retrain the estimator.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd

from .model import FEATURES, NUMERIC_FEATURES, CATEGORICAL_FEATURES
from .explain import generate_report
from .predict import file_hash

REGISTRY_NAME = "RetailIQ-Demand-XGBoost"
EXPERIMENT_NAME = "RetailIQ Demand Forecasting"


def register(artifact_dir: str | Path = "artifacts") -> dict:
    try:
        import mlflow
        import mlflow.sklearn
        from mlflow.tracking import MlflowClient
    except ImportError as exc:
        raise RuntimeError("Install MLflow: python -m pip install mlflow") from exc

    folder = Path(artifact_dir).resolve()
    model_path = folder / "demand_xgboost.joblib"
    metrics_path = folder / "evaluation.json"
    if not model_path.exists() or not metrics_path.exists():
        raise FileNotFoundError("Train RetailIQ first; expected joblib model and evaluation.json")

    db_path = folder / "mlflow" / "mlflow.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    uri = f"sqlite:///{db_path.as_posix()}"
    mlflow.set_tracking_uri(uri)
    mlflow.set_registry_uri(uri)
    mlflow.set_experiment(EXPERIMENT_NAME)

    # Treat the local model artifact as trusted; joblib deserialisation executes pickle.
    model = joblib.load(model_path)
    metrics = json.loads(metrics_path.read_text())
    digest = file_hash(model_path)

    # Register full preprocessing+estimator pipeline. Input example uses the same
    # raw schema required by the batch inference code.
    example = pd.DataFrame([{**{x: 0.0 for x in NUMERIC_FEATURES},
                             **{x: "example" for x in CATEGORICAL_FEATURES}}])[FEATURES]

    report = generate_report(model_path, folder / "explainability")
    with mlflow.start_run(run_name="xgboost-registered-model") as run:
        mlflow.log_params({"horizon_days": 7, "training_rows": metrics["training_rows"],
                           "evaluation_rows": metrics["evaluation_rows"]})
        for model_name in ("baseline", "xgboost"):
            for metric_name, value in metrics[model_name].items():
                if value is not None:
                    mlflow.log_metric(f"{model_name}_{metric_name}", float(value))
        if metrics.get("wape_improvement_percent") is not None:
            mlflow.log_metric("wape_improvement_percent",
                              float(metrics["wape_improvement_percent"]))
        mlflow.set_tags({"model_file_sha256": digest, "forecast_horizon_days": "7",
                         "evaluation": "chronological_holdout"})
        mlflow.log_artifact(str(metrics_path), artifact_path="evaluation")
        prediction_csv = folder / "evaluation_predictions.csv"
        if prediction_csv.exists():
            mlflow.log_artifact(str(prediction_csv), artifact_path="evaluation")
        mlflow.log_artifact(report["csv"], artifact_path="explainability")
        mlflow.log_artifact(report["chart"], artifact_path="explainability")

        # MLflow 3 API. 'name' replaces older 'artifact_path'.
        logged_model = mlflow.sklearn.log_model(
            sk_model=model, name="demand_pipeline", input_example=example,
            registered_model_name=REGISTRY_NAME,
            serialization_format="skops",
            skops_trusted_types=[
                "xgboost.core.Booster",
                "xgboost.sklearn.XGBRegressor",
            ],
            
        )

    client = MlflowClient(tracking_uri=uri, registry_uri=uri)
    versions = client.search_model_versions(f"name='{REGISTRY_NAME}'")
    matches = [v for v in versions if v.run_id == run.info.run_id]
    version = max(matches, key=lambda v: int(v.version)).version if matches else None
    return {"experiment_run_id": run.info.run_id, "registered_model": REGISTRY_NAME,
            "registered_version": version, "model_hash": digest[:16],
            "feature_importance_chart": report["chart"], "tracking_db": str(db_path),
            "mlflow_model_uri": getattr(logged_model, "model_uri", None)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Register saved RetailIQ model in MLflow")
    parser.add_argument("--artifact-dir", default="artifacts")
    args = parser.parse_args()
    print(json.dumps(register(args.artifact_dir), indent=2))


if __name__ == "__main__":
    main()
    