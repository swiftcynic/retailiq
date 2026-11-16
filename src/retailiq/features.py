"""Execute versioned Snowflake feature SQL using Snowpark; no local data download.

A forecast origin is a completed sales date. Features are only available on or
before that date; labels are the seven subsequent days of actual demand.
"""
import os
from pathlib import Path


FEATURE_SQL = Path(__file__).resolve().parents[2] / "sql" / "03_features.sql"


def snowpark_settings():
    """Build Snowpark auth config without accidentally logging secrets."""
    account = os.getenv("SNOWFLAKE_ACCOUNT")
    user = os.getenv("SNOWFLAKE_USER")
    if not account or not user:
        raise RuntimeError("Set SNOWFLAKE_ACCOUNT and SNOWFLAKE_USER in .env")
    result = {
        "account": account,
        "user": user,
        "warehouse": "RETAILIQ_WH",
        "database": "RETAILIQ",
        "schema": "MART",
        "client_session_keep_alive": False,
    }
    if os.getenv("SNOWFLAKE_PASSWORD"):
        result["password"] = os.environ["SNOWFLAKE_PASSWORD"]
    else:
        result["authenticator"] = "externalbrowser"
    if os.getenv("SNOWFLAKE_ROLE"):
        result["role"] = os.environ["SNOWFLAKE_ROLE"]
    return result


def split_statements(sql):
    """Split our simple DDL script, ignoring full-line SQL comments."""
    without_comments = "\n".join(
        line for line in sql.splitlines() if not line.lstrip().startswith("--")
    )
    return [s.strip() for s in without_comments.split(";") if s.strip()]


def create_features():
    """Create Snowflake views and report feature row counts via Snowpark."""
    try:
        from snowflake.snowpark import Session
    except ImportError as exc:
        raise RuntimeError("Install Snowpark: pip install -e '.[dev]'") from exc

    session = Session.builder.configs(snowpark_settings()).create()
    try:
        for statement in split_statements(FEATURE_SQL.read_text()):
            session.sql(statement).collect()
        result = session.sql("""
            SELECT COUNT(*) AS TOTAL_ORIGINS,
                   COUNT_IF(LABEL_AVAILABLE) AS TRAINABLE_ORIGINS,
                   COUNT(DISTINCT ITEM_ID || ':' || STORE_ID) AS SERIES
            FROM RETAILIQ.MART.DAILY_DEMAND_FEATURES
        """).collect()[0]
        return {"total_origins": result[0], "trainable_origins": result[1], "series": result[2]}
    finally:
        session.close()
