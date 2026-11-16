"""Offline tests for feature SQL contracts; no Snowflake account required."""
from pathlib import Path
from retailiq.features import FEATURE_SQL, split_statements, snowpark_settings
import pytest


def test_feature_views_have_separate_training_selection():
    sql = FEATURE_SQL.read_text().upper()
    assert "DAILY_DEMAND_FEATURES" in sql
    assert "TRAINING_FEATURES" in sql
    assert "WHERE LABEL_AVAILABLE = TRUE" in sql
    assert "ROWS BETWEEN 1 FOLLOWING AND 7 FOLLOWING" in sql
    assert "DATEDIFF('DAY', AS_OF_DATE, TARGET_END_DATE) = 7" in sql


def test_features_exclude_future_data_except_label():
    sql = FEATURE_SQL.read_text().upper()
    for feature in ["LAG_1", "LAG_7", "LAG_14", "LAG_28", "AVG_7", "AVG_28", "STDDEV_28"]:
        assert feature in sql
    assert "ROWS BETWEEN 27 PRECEDING AND CURRENT ROW" in sql
    assert "ROWS BETWEEN 6 PRECEDING AND CURRENT ROW" in sql


def test_sql_statement_splitter():
    parsed = split_statements("-- comment; with semicolon\nSELECT 1;\n-- another\nSELECT 2;")
    assert parsed == ["SELECT 1", "SELECT 2"]


def test_snowpark_auth_settings(monkeypatch):
    monkeypatch.setenv("SNOWFLAKE_ACCOUNT", "test-account")
    monkeypatch.setenv("SNOWFLAKE_USER", "test-user")
    monkeypatch.delenv("SNOWFLAKE_PASSWORD", raising=False)
    settings = snowpark_settings()
    assert settings["authenticator"] == "externalbrowser"
    assert settings["warehouse"] == "RETAILIQ_WH"
    assert "password" not in settings


def test_missing_credentials(monkeypatch):
    monkeypatch.delenv("SNOWFLAKE_ACCOUNT", raising=False)
    monkeypatch.delenv("SNOWFLAKE_USER", raising=False)
    with pytest.raises(RuntimeError, match="SNOWFLAKE_ACCOUNT"):
        snowpark_settings()
