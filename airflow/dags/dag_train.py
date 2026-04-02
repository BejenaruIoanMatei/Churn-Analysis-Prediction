from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
import sys

sys.path.append('/opt/airflow/scripts')
from pipeline import train_and_analyze

with DAG(
    dag_id="dag_02_train_and_report",
    start_date=datetime(2024, 1, 1),
    schedule=None,
    catchup=False,
    tags=["ml", "train"],
) as dag:

    train_task = PythonOperator(
        task_id="train_xgb_model",
        python_callable=train_and_analyze,
    )

    train_task