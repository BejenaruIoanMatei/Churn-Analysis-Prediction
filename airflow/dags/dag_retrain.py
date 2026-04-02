from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.operators.empty import EmptyOperator
from datetime import datetime
import sys

sys.path.append('/opt/airflow/scripts')
from pipeline import train_and_analyze

F1_THRESHOLD = 0.75

def check_model_performance(**context):
    """
    Gets the most recent F1 from model_metrics
    F1 < treshold => 'retrain' (branch)
    F1 >= treshold => 'skip_retrain' (branch)
    """
    import os
    from sqlalchemy import create_engine
    import pandas as pd

    user = os.getenv('POSTGRES_USER')
    password = os.getenv('POSTGRES_PASSWORD')
    db = os.getenv('POSTGRES_DB', 'churn_db')

    engine = create_engine(f"postgresql+psycopg2://{user}:{password}@postgres:5432/{db}")

    df = pd.read_sql(
        "SELECT f1_score, run_date FROM model_metrics ORDER BY run_date DESC LIMIT 1",
        engine
    )

    if df.empty:
        print("No metrics found — triggering retrain")
        return 'retrain'

    latest_f1 = float(df.iloc[0]['f1_score'])
    run_date = df.iloc[0]['run_date']

    print(f"Latest F1: {latest_f1:.4f} (from {run_date})")
    print(f"Threshold: {F1_THRESHOLD}")

    context['ti'].xcom_push(key='latest_f1', value=latest_f1)

    if latest_f1 < F1_THRESHOLD:
        print(f"F1 {latest_f1:.4f} < {F1_THRESHOLD} → RETRAINING")
        return 'retrain'
    else:
        print(f"F1 {latest_f1:.4f} >= {F1_THRESHOLD} → SKIP")
        return 'skip_retrain'


def notify_result(**context):
    """
    Displays info regarding the run (what are the results)
    """
    ti = context['ti']
    latest_f1 = ti.xcom_pull(task_ids='check_performance', key='latest_f1')

    if latest_f1 is None:
        print("Retrain triggered — no previous metrics found")
        return

    if latest_f1 < F1_THRESHOLD:
        print(f"Model retrained (not ok boss) — F1 was {latest_f1:.4f}, below threshold {F1_THRESHOLD}")
    else:
        print(f"Model OK — F1 is {latest_f1:.4f}, above threshold {F1_THRESHOLD}. No retrain needed.")


with DAG(
    dag_id="dag_03_retrain_monitor",
    start_date=datetime(2024, 1, 1),
    schedule="@weekly",
    catchup=False,
    tags=["ml", "monitor", "retrain"],
) as dag:

    check_performance = BranchPythonOperator(
        task_id="check_performance",
        python_callable=check_model_performance,
    )

    retrain = PythonOperator(
        task_id="retrain",
        python_callable=train_and_analyze,
    )

    skip_retrain = EmptyOperator(
        task_id="skip_retrain",
    )

    notify = PythonOperator(
        task_id="notify_result",
        python_callable=notify_result,
        trigger_rule="none_failed_min_one_success",
    )

    check_performance >> [retrain, skip_retrain] >> notify