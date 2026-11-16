"""Offline tests for the RetailIQ feature-importance report."""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBRegressor

from retailiq.explain import feature_importance, generate_report
from retailiq.model import FEATURES, NUMERIC_FEATURES, CATEGORICAL_FEATURES


def test_importance_matches_fitted_columns(tmp_path):
    import joblib
    rows = 48
    x = pd.DataFrame({k: np.linspace(0, 10, rows) for k in NUMERIC_FEATURES})
    x["ITEM_ID"] = ["A", "B"] * (rows // 2)
    x["STORE_ID"] = ["S1"] * rows
    y = 2 * x["LAG_7"] + (x["ITEM_ID"] == "B").astype(int)
    model = Pipeline([
        ("preprocess", ColumnTransformer([
            ("numeric", "passthrough", NUMERIC_FEATURES),
            ("category", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ])),
        ("model", XGBRegressor(n_estimators=5, n_jobs=1, random_state=42)),
    ]).fit(x[FEATURES], y)
    importance = feature_importance(model)
    assert len(importance) == len(model.named_steps["preprocess"].get_feature_names_out())
    assert importance["importance"].ge(0).all()
    file = tmp_path / "model.joblib"
    joblib.dump(model, file)
    result = generate_report(file, tmp_path / "explain")
    assert (tmp_path / "explain" / "feature_importance.csv").exists()
    assert (tmp_path / "explain" / "feature_importance.png").exists()
    assert result["features"] == len(importance)
