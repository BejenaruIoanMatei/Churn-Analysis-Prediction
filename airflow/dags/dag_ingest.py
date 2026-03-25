from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
import os
import pandas as pd
from sqlalchemy import create_engine

KAGGLE_DATASET = "blastchar/telco-customer-churn"
DATA_PATH = "/opt/airflow/data"
CSV_FILE = f"{DATA_PATH}/WA_Fn-UseC_-Telco-Customer-Churn.csv"

PG_CONN = (
    f"postgresql+psycopg2://{os.environ['POSTGRES_USER']}:"
    f"{os.environ['POSTGRES_PASSWORD']}@postgres:5432/"
    f"{os.environ['POSTGRES_DB']}"
)

def download_dataset():
    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi()
    api.authenticate()
    api.dataset_download_files(
        KAGGLE_DATASET,
        path=DATA_PATH,
        unzip=True
    )
    print(f"Dataset: {DATA_PATH}")

def ingest_to_postgres():
    df = pd.read_csv(CSV_FILE)
    
    df.columns = [c.lower().replace(" ", "_") for c in df.columns]
    df["totalcharges"] = pd.to_numeric(df["totalcharges"], errors="coerce")
    
    engine = create_engine(PG_CONN)
    df.to_sql(
        name="raw_churn",
        con=engine,
        schema="public",
        if_exists="replace",
        index=False
    )
    print(f"{len(df)} rows loaded in PostgreSQL -> table raw_churn")

with DAG(
    dag_id="dag_01_ingest_churn",
    start_date=datetime(2024, 1, 1),
    schedule=None,
    catchup=False,
    tags=["churn", "ingest"],
) as dag:

    t1 = PythonOperator(
        task_id="download_from_kaggle",
        python_callable=download_dataset,
    )

    t2 = PythonOperator(
        task_id="ingest_to_postgres",
        python_callable=ingest_to_postgres,
    )

    t1 >> t2