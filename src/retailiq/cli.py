"""Run python -m retailiq.cli setup|ingest|check|suspend."""
import argparse
import uuid
from pathlib import Path
import pandas as pd
from dotenv import load_dotenv
from .transform import to_daily_sales
from .snowflake_io import connect, execute_sql_file, load_dataframe


def main():
    load_dotenv()
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["setup", "ingest", "check", "features", "train", "predict", "quality", "pipeline", "suspend"])
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--track-mlflow", action="store_true",
                        help="Log pipeline evaluation metrics to local MLflow")
    parser.add_argument("--series", type=int, default=25, help="Number of product-store time series; start small")
    parser.add_argument("--batch-size", type=int, default=10)
    args = parser.parse_args()
    if args.series < 1 or args.batch_size < 1:
        parser.error("--series and --batch-size must be positive")
    if args.command == "quality":
        import json
        from .quality import run_quality_checks
        print(json.dumps(run_quality_checks(), indent=2))
        return
    if args.command == "pipeline":
        import json
        from .pipeline import execute_pipeline
        print(json.dumps(execute_pipeline(track_mlflow=args.track_mlflow), indent=2, default=str))
        return
    if args.command == "features":
        from .features import create_features
        result = create_features()
        print("Snowpark feature views created:", result)
        return
    if args.command == "predict":
        import json
        from .predict import predict
        print(json.dumps(predict(), indent=2))
        return
    if args.command == "train":
        import json
        from .model import fetch_training_data, train
        result = train(fetch_training_data())
        print(json.dumps(result, indent=2))
        return
    if args.command == "suspend":
        from .warehouse import suspend_warehouse
        print("Warehouse:", suspend_warehouse())
        return
    with connect() as conn:
        if args.command == "setup":
            execute_sql_file(conn, "sql/01_setup.sql")
            execute_sql_file(conn, "sql/02_staging.sql")
            print("Created database, schemas, tables, and views")
        elif args.command == "ingest":
            data_dir = Path(args.data_dir)
            sales_path = data_dir / "sales_train_validation.csv"
            calendar_path = data_dir / "calendar.csv"
            if not sales_path.exists() or not calendar_path.exists():
                parser.error("Download and extract M5 data into data/ (see README)")
            calendar = pd.read_csv(calendar_path, usecols=["d", "date"])
            sales = pd.read_csv(sales_path, nrows=args.series)
            # Batching prevents one large local frame from consuming unnecessary memory.
            for start in range(0, len(sales), args.batch_size):
                frame = to_daily_sales(sales.iloc[start:start + args.batch_size], calendar)
                load_dataframe(conn, frame, f"m5-{start}-{uuid.uuid4().hex[:8]}")
                print(f"Loaded batch {start // args.batch_size + 1}: {len(frame):,} rows")
        elif args.command == "check":
            with conn.cursor() as cur:
                cur.execute("USE WAREHOUSE RETAILIQ_WH")
                cur.execute("SELECT COUNT(*), COUNT(DISTINCT ITEM_ID || ':' || STORE_ID) FROM RETAILIQ.STAGING.DAILY_SALES")
                rows, series = cur.fetchone()
                print(f"Daily sales: {rows:,} rows; {series:,} item-store series")
                cur.execute("SELECT MIN(SALE_DATE), MAX(SALE_DATE), SUM(UNITS_SOLD) FROM RETAILIQ.STAGING.DAILY_SALES")
                print("Date range and total units:", cur.fetchone())


if __name__ == "__main__":
    main()
