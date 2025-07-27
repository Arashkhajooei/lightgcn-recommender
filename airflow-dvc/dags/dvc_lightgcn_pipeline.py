from airflow import DAG
from airflow.operators.bash import BashOperator
import pendulum

dag = DAG(
    dag_id='dvc_lightgcn_pipeline',
    start_date=pendulum.now().subtract(days=1),
    schedule=None,  # Replaces schedule_interval
    catchup=False,
    description='Run LightGCN training & evaluation using DVC',
)

with dag:
    run_dvc = BashOperator(
        task_id='run_dvc_repro',
        bash_command='cd /opt/airflow/lightgcn-recommender && dvc repro',
        dag=dag,
    )

    run_eval = BashOperator(
        task_id='run_evaluation',
        bash_command='cd /opt/airflow/lightgcn-recommender && python scripts/evaluate.py',
        dag=dag,
    )

    push_dvc = BashOperator(
        task_id='push_dvc_outputs',
        bash_command='cd /opt/airflow/lightgcn-recommender && dvc push',
        dag=dag,
    )

    run_dvc >> run_eval >> push_dvc
