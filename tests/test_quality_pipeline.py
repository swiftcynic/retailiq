import json
import pytest
from retailiq.quality import validate_counts
from retailiq.pipeline import execute_pipeline


def test_valid_data():
    assert validate_counts(100, 100, 0, 0, 3)["series"] == 3


@pytest.mark.parametrize("values", [(0, 0, 0, 0, 0), (10, 9, 0, 0, 2),
                                      (10, 10, 1, 0, 2), (10, 10, 0, 1, 2)])
def test_invalid_data(values):
    with pytest.raises(ValueError):
        validate_counts(*values)


def test_pipeline_success(tmp_path):
    calls = []
    def step(name):
        def go(*args):
            calls.append(name)
            return {"ok": True}
        return go
    result = execute_pipeline(tmp_path, quality_fn=step("quality"),
                              features_fn=step("features"), train_fn=step("training"),
                              predict_fn=step("prediction"), suspend_fn=step("suspend"))
    assert calls == ["quality", "features", "training", "prediction", "suspend"]
    assert result["status"] == "success"
    assert json.loads((tmp_path / "pipeline_runs" / (result["pipeline_run_id"] + ".json")).read_text())["status"] == "success"


def test_pipeline_failure_suspends_and_logs(tmp_path):
    calls = []
    def fail():
        raise ValueError("bad data")
    with pytest.raises(RuntimeError, match="Pipeline failed"):
        execute_pipeline(tmp_path, quality_fn=fail, features_fn=lambda: calls.append("features"),
                         suspend_fn=lambda: calls.append("suspended"))
    assert calls == ["suspended"]
    logs = list((tmp_path / "pipeline_runs").glob("*.json"))
    assert len(logs) == 1
    assert json.loads(logs[0].read_text())["status"] == "failed"


def test_pipeline_optional_tracking(tmp_path):
    calls = []
    def train(_):
        return {"training_rows": 1, "evaluation_rows": 1,
                "baseline": {"wape": 0.5}, "xgboost": {"wape": 0.4}}
    def track(metrics, path):
        calls.append((metrics["training_rows"], path.name))
        return "mlflow-test-id"
    result = execute_pipeline(tmp_path, suspend=False, quality_fn=lambda: {},
                              features_fn=lambda: {}, train_fn=train,
                              predict_fn=lambda _: {}, track_mlflow=True,
                              tracking_fn=track)
    assert calls == [(1, "mlruns")]
    assert result["steps"]["mlflow"]["run_id"] == "mlflow-test-id"
