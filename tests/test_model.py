import numpy as np
import pandas as pd
import pytest
from retailiq.model import (
    NUMERIC_FEATURES, TARGET, chronological_split, metrics, prepare_frame, train,
)


def sample_frame(n_days=110, n_series=3):
    records = []
    for series in range(n_series):
        for d in pd.date_range('2025-01-01', periods=n_days):
            row = {f: 2.0 for f in NUMERIC_FEATURES}
            row.update(ITEM_ID=f'item{series}', STORE_ID='store', AS_OF_DATE=d,
                       TARGET_NEXT_7D=float((d.day % 7) + series))
            records.append(row)
    return pd.DataFrame(records)


def test_chronological_split_prevents_label_leakage():
    df = prepare_frame(sample_frame())
    training, test, cutoff = chronological_split(df)
    assert training.AS_OF_DATE.max() + pd.Timedelta(days=7) <= test.AS_OF_DATE.min()
    assert len(test.AS_OF_DATE.unique()) == 4
    assert len(test) == 12


def test_no_target_in_features():
    from retailiq.model import FEATURES
    assert TARGET not in FEATURES
    assert 'AS_OF_DATE' not in FEATURES


def test_metrics_nonnegative_and_zero_demand():
    assert metrics([0, 0], [-1, 0])["wape"] is None
    assert metrics([1, 3], [1, 1])["wape"] == 0.5


def test_short_history_fails():
    with pytest.raises(ValueError, match='Insufficient'):
        chronological_split(prepare_frame(sample_frame(30, 1)))


def test_train_writes_artifacts(tmp_path):
    result = train(sample_frame(), output_dir=tmp_path)
    assert result['training_rows'] > 100
    assert (tmp_path / 'demand_xgboost.joblib').exists()
    assert (tmp_path / 'evaluation.json').exists()
    assert (tmp_path / 'evaluation_predictions.csv').exists()
