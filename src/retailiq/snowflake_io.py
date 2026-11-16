"""Snowflake connection, SQL execution and validated batch loading."""
import os
from pathlib import Path
from snowflake import connector
from snowflake.connector.pandas_tools import write_pandas


def connect():
    required = ["SNOWFLAKE_ACCOUNT", "SNOWFLAKE_USER"]
    missing = [k for k in required if not os.environ.get(k)]
    if missing:
        raise RuntimeError(f"Set these environment variables: {', '.join(missing)}")
    options = dict(account=os.environ["SNOWFLAKE_ACCOUNT"], user=os.environ["SNOWFLAKE_USER"],
                   client_session_keep_alive=False)
    if os.environ.get("SNOWFLAKE_PASSWORD"):
        options["password"] = os.environ["SNOWFLAKE_PASSWORD"]
    else:
        options["authenticator"] = "externalbrowser"
    if os.environ.get("SNOWFLAKE_ROLE"):
        options["role"] = os.environ["SNOWFLAKE_ROLE"]
    return connector.connect(**options)


def execute_sql_file(conn, path):
    """Execute our intentionally simple, semicolon-separated SQL setup files."""
    statements = [stmt.strip() for stmt in Path(path).read_text().split(";") if stmt.strip()]
    with conn.cursor() as cursor:
        for statement in statements:
            cursor.execute(statement)


def load_dataframe(conn, frame, batch_id):
    """Load a single, uniquely named batch. Rerunning a batch replaces its rows."""
    with conn.cursor() as cursor:
        cursor.execute("USE WAREHOUSE RETAILIQ_WH")
        cursor.execute("USE DATABASE RETAILIQ")
        cursor.execute("USE SCHEMA RAW")
        # Stage data in a transient load table and replace only the selected item/store keys.
        cursor.execute("CREATE OR REPLACE TEMP TABLE BATCH_DAILY_SALES LIKE DAILY_SALES")
    success, chunks, loaded, _ = write_pandas(conn, frame, table_name="BATCH_DAILY_SALES",
                                             database="RETAILIQ", schema="RAW", quote_identifiers=False)
    if not success or loaded != len(frame):
        raise RuntimeError(f"Snowflake upload incomplete: {loaded}/{len(frame)} rows")
    with conn.cursor() as cursor:
        # Transactional replacement of the selected series makes repeated loads idempotent.
        cursor.execute("BEGIN")
        try:
            cursor.execute("""DELETE FROM DAILY_SALES d WHERE EXISTS (
                         SELECT 1 FROM BATCH_DAILY_SALES b
                         WHERE b.ITEM_ID=d.ITEM_ID AND b.STORE_ID=d.STORE_ID)""")
            cursor.execute("INSERT INTO DAILY_SALES SELECT * FROM BATCH_DAILY_SALES")
            cursor.execute("INSERT INTO INGEST_LOG (BATCH_ID, ROWS_LOADED) VALUES (%s, %s)", (batch_id, loaded))
            cursor.execute("COMMIT")
        except Exception:
            cursor.execute("ROLLBACK")
            raise
