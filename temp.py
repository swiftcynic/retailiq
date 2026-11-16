
from pathlib import Path
from mlflow.tracking import MlflowClient

db_path = Path("artifacts/mlflow/mlflow.db").resolve()
tracking_uri = f"sqlite:///{db_path.as_posix()}"

print("Database exists:", db_path.exists())
print("Database path:", db_path)
print("Tracking URI:", tracking_uri)

if db_path.exists():
    client = MlflowClient(tracking_uri=tracking_uri)

    experiments = client.search_experiments()

    for exp in experiments:
        runs = client.search_runs([exp.experiment_id])
        print(
            f"Experiment: {exp.name} | "
            f"ID: {exp.experiment_id} | "
            f"Runs: {len(runs)}"
        )

