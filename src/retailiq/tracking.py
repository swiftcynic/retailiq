
"""Optional local MLflow experiment tracking; no hosted service needed."""

from pathlib import Path


def log_training_run(
    metrics: dict,
    tracking_dir: str | Path = "artifacts/mlflow",
) -> str:
    """Log training metrics to a local SQLite-backed MLflow store."""

    try:
        import mlflow
    except ImportError as exc:
        raise RuntimeError(
            "Install optional MLflow using: python -m pip install mlflow"
        ) from exc

    # Create a local directory for MLflow data.
    path = Path(tracking_dir).resolve()
    path.mkdir(parents=True, exist_ok=True)

    # SQLite database for experiment tracking.
    database_path = path / "mlflow.db"
    tracking_uri = f"sqlite:///{database_path.as_posix()}"

    mlflow.set_tracking_uri(tracking_uri)

    mlflow.set_experiment("RetailIQ Demand Forecasting")

    with mlflow.start_run(run_name="xgboost-seven-day") as run:
        mlflow.log_params({
            "horizon_days": 7,
            "training_rows": metrics["training_rows"],
            "evaluation_rows": metrics["evaluation_rows"],
        })

        for name in ("baseline", "xgboost"):
            for metric, value in metrics[name].items():
                if value is not None:
                    mlflow.log_metric(
                        f"{name}_{metric}",
                        float(value),
                    )

        improvement = metrics.get("wape_improvement_percent")
        if improvement is not None:
            mlflow.log_metric(
                "wape_improvement_percent",
                float(improvement),
            )

        mlflow.set_tag("evaluation", "chronological_holdout")

        return run.info.run_id
