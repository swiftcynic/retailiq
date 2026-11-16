"""Offline tests for rolling-origin validation and XGBoost TreeSHAP additivity."""
import numpy as np
import pandas as pd
import pytest

from retailiq.model import FEATURES, TARGET, build_model
from retailiq.backtest import backtest, clean_snapshot
from retailiq.shap_report import compute_contributions


def sample_data(days=110, items=3):
    origins = pd.date_range("2016-01-01", periods=days, freq="D")
    rows = []
    for i in range(items):
        for j, date in enumerate(origins):
            rows.append({"ITEM_ID": f"item{i}", "STORE_ID": "store1", "AS_OF_DATE": date,
                         "LAG_1": j % 8, "LAG_7": (j + 2) % 8,
                         "LAG_14": (j + 4) % 8, "LAG_28": (j + 1) % 8,
                         "AVG_7": 3.0+i, "AVG_28": 3.5+i, "STDDEV_28": 1.1,
                         "DAY_OF_WEEK": date.dayofweek, "MONTH_OF_YEAR": date.month,
                         TARGET: float((j % 8 + i) * 7)})
    return pd.DataFrame(rows)


def test_rolling_backtest_cutoffs_and_outputs(tmp_path):
    result = backtest(sample_data(), n_folds=3, output_dir=tmp_path)
    assert result["folds"] == 3
    assert result["evaluation_rows"] == 9
    fold = pd.read_csv(tmp_path / "fold_metrics.csv")
    assert (pd.to_datetime(fold["training_cutoff"]) + pd.Timedelta(days=7)
            <= pd.to_datetime(fold["origin"])).all()
    assert (tmp_path / "model_last_fold.joblib").exists()


def test_duplicate_rows_rejected():
    df = sample_data()
    with pytest.raises(ValueError, match="Duplicate"):
        clean_snapshot(pd.concat([df, df.head(1)], ignore_index=True))


def test_unobserved_labels_excluded():
    df = sample_data()
    df.loc[df.index[-2:], TARGET] = np.nan
    assert len(clean_snapshot(df)) == len(df) - 2


def test_tree_shap_adds_to_raw_predictions():
    df = clean_snapshot(sample_data(days=35))
    model = build_model()
    model.fit(df[FEATURES], df[TARGET])
    names, contributions, bias, predictions = compute_contributions(model, df.head(4))
    assert contributions.shape == (4, len(names))
    np.testing.assert_allclose(contributions.sum(axis=1) + bias, predictions,
                               atol=1e-3, rtol=1e-4)
