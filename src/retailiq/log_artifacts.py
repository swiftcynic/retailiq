
"""Log previously generated evaluation artifacts into MLflow."""

from pathlib import Path

import mlflow

from .mlflow_artifacts import log_evaluation_artifacts


def main():
    db_path = Path("artifacts/mlflow/mlflow.db").resolve()

    mlflow.set_tracking_uri(
        f"sqlite:///{db_path.as_posix()}"
    )

    mlflow.set_experiment("RetailIQ Demand Forecasting")

    with mlflow.start_run(run_name="xgboost-evaluation-report"):
        log_evaluation_artifacts()


if __name__ == "__main__":
    main()
