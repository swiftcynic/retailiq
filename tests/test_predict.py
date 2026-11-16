"""Offline inference safety and schema tests (no Snowflake credentials required)."""
import numpy as np
import pandas as pd
import pytest

from retailiq.model import FEATURES, NUMERIC_FEATURES
from retailiq.predict import forecast_frame, prepare_inference, file_hash, SQL_FILE


class FakeModel:
    def predict(self, features):
        assert list(features.columns) == FEATURES
        return np.array([-2.5, 12.75])


def sample():
    rows = []
    for item in ["A", "B"]:
        row = dict.fromkeys(NUMERIC_FEATURES, 2.0)
        row.update(ITEM_ID=item, STORE_ID="store1", AS_OF_DATE="2016-04-24",
                   TARGET_NEXT_7D=999999)  # deliberately ignored by inference
        rows.append(row)
    return pd.DataFrame(rows)


def test_inference_excludes_target_and_keeps_training_feature_order():
    data = prepare_inference(sample())
    assert "TARGET_NEXT_7D" not in data.columns
    assert data[FEATURES].columns.tolist() == FEATURES


def test_forecast_dates_and_nonnegative_predictions():
    result = forecast_frame(sample(), FakeModel(), "abc123", "run123")
    assert result.PREDICTED_UNITS.tolist() == [0.0, 12.75]
    assert result.FORECAST_START_DATE.iloc[0].isoformat() == "2016-04-25"
    assert result.FORECAST_END_DATE.iloc[0].isoformat() == "2016-05-01"
    assert result.MODEL_VERSION.tolist() == ["abc123", "abc123"]


def test_duplicate_origins_rejected():
    df = pd.concat([sample(), sample().iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="Duplicate"):
        prepare_inference(df)


def test_mixed_origins_rejected():
    df = sample()
    df.loc[1, "AS_OF_DATE"] = "2016-04-23"
    with pytest.raises(ValueError, match="one common"):
        prepare_inference(df)


def test_nonfinite_model_output_rejected():
    class BadModel:
        def predict(self, features):
            return np.array([np.nan, 1.0])
    with pytest.raises(ValueError, match="invalid forecasts"):
        forecast_frame(sample(), BadModel(), "v", "r")


def test_model_version_hash_is_deterministic(tmp_path):
    p = tmp_path / "model.bin"
    p.write_bytes(b"test-model")
    assert file_hash(p) == file_hash(p)
    assert len(file_hash(p)) == 64


def test_prediction_sql_contract():
    sql = SQL_FILE.read_text().upper()
    assert "DEMAND_FORECASTS" in sql
    assert "MODEL_RUNS" in sql
    assert "MODEL_VERSION" in sql
