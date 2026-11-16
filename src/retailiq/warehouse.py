"""Safe, idempotent warehouse suspension for local development."""
from .snowflake_io import connect


def suspend_warehouse():
    with connect() as conn:
        with conn.cursor() as cur:
            cur.execute("SHOW WAREHOUSES LIKE 'RETAILIQ_WH'")
            rows = cur.fetchall()
            if not rows:
                return "missing"
            columns = [field[0].lower() for field in cur.description]
            state = dict(zip(columns, rows[0])).get("state", "").upper()
            if state in {"STARTED", "RESIZING"}:
                cur.execute("ALTER WAREHOUSE RETAILIQ_WH SUSPEND")
                return "suspended"
            return f"inactive ({state})"
