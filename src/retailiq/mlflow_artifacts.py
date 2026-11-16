"""Log RetailIQ evaluation artifacts to MLflow."""

from pathlib import Path

import json

import matplotlib.pyplot as plt
import pandas as pd
import mlflow


def log_evaluation_artifacts(
	artifact_dir: str = "artifacts",
):
	artifact_path = Path(artifact_dir)
	
	metrics_path = artifact_path / "evaluation.json"
	predictions_path = artifact_path / "evaluation_predictions.csv"
	
	if not metrics_path.exists():
		raise FileNotFoundError(metrics_path)
		
	if not predictions_path.exists():
		raise FileNotFoundError(predictions_path)
		
	with open(metrics_path) as f:
		metrics = json.load(f)
		
	predictions = pd.read_csv(predictions_path)
	
	print("Available prediction columns:")
	print(predictions.columns.tolist())
	
	# Save the evaluation metrics as an MLflow artifact.
	mlflow.log_dict(metrics, "evaluation/metrics.json")
	
	# Plot baseline and XGBoost WAPE.
	labels = ["Seasonal Baseline", "XGBoost"]
	
	values = [
		metrics["baseline"]["wape"] * 100,
		metrics["xgboost"]["wape"] * 100,
	]
	
	fig, ax = plt.subplots(figsize=(7, 4))
	
	ax.bar(labels, values)
	
	ax.set_ylabel("WAPE (%)")
	ax.set_title("Forecasting Model Comparison")
	
	for index, value in enumerate(values):
		ax.text(
			index,
			value + 1,
			f"{value:.2f}%",
			ha="center",
		)
		
	fig.tight_layout()
	
	mlflow.log_figure(
		fig,
		"evaluation/model_comparison.png",
	)
	plt.close(fig)
	
	# Store predictions as a downloadable artifact.
	mlflow.log_artifact(
		str(predictions_path),
		artifact_path="evaluation",
	)
	
	print("Evaluation artifacts logged successfully.")
	