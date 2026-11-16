"""Orchestrate quality -> features -> train -> predict, logging a local run report.

The runner does not re-ingest historical CSV data and does not run unattended.
"""
from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def execute_pipeline(artifacts_dir: str | Path = "artifacts", suspend: bool = True,
                     quality_fn=None, features_fn=None, train_fn=None,
                     predict_fn=None, suspend_fn=None, track_mlflow: bool = False,
                     tracking_fn=None):
    """Dependency injection permits offline testing without Snowflake.

    Write run report on success or failure. Always attempt warehouse suspension.
    """
    from .quality import run_quality_checks
    from .features import create_features
    from .model import fetch_training_data, train
    from .predict import predict
    from .warehouse import suspend_warehouse

    quality_fn = quality_fn or run_quality_checks
    features_fn = features_fn or create_features
    train_fn = train_fn or (lambda out: train(fetch_training_data(), output_dir=out))
    predict_fn = predict_fn or (lambda out: predict(artifact_dir=out))
    suspend_fn = suspend_fn or suspend_warehouse

    artifact_root = Path(artifacts_dir)
    log_dir = artifact_root / "pipeline_runs"
    log_dir.mkdir(parents=True, exist_ok=True)
    report = {"pipeline_run_id": uuid.uuid4().hex, "started_at": utc_now(),
              "status": "running", "steps": {}}
    failure = None
    try:
        for name, fn in (
            ("quality", quality_fn),
            ("features", features_fn),
            ("training", lambda: train_fn(artifact_root)),
            ("prediction", lambda: predict_fn(artifact_root)),
        ):
            started = utc_now()
            value = fn()
            report["steps"][name] = {"started_at": started, "completed_at": utc_now(),
                                      "result": value}
        if track_mlflow:
            if tracking_fn is None:
                from .tracking import log_training_run
                tracking_fn = log_training_run
            started = utc_now()
            run_id = tracking_fn(report["steps"]["training"]["result"],
                                 artifact_root / "mlruns")
            report["steps"]["mlflow"] = {"started_at": started, "completed_at": utc_now(),
                                          "run_id": run_id}
        report["status"] = "success"
    except Exception as exc:
        report["status"] = "failed"
        report["error"] = {"type": type(exc).__name__, "message": str(exc)}
        failure = exc
    finally:
        if suspend:
            try:
                report["warehouse"] = suspend_fn()
            except Exception as exc:
                report["warehouse_error"] = f"{type(exc).__name__}: {exc}"
                if failure is None:
                    failure = exc
                    report["status"] = "failed"
        report["finished_at"] = utc_now()
        log_path = log_dir / f"{report['pipeline_run_id']}.json"
        log_path.write_text(json.dumps(report, indent=2, default=str) + "\n")
        report["report_path"] = str(log_path)
    if failure is not None:
        raise RuntimeError(f"Pipeline failed; see {report['report_path']}") from failure
    return report
