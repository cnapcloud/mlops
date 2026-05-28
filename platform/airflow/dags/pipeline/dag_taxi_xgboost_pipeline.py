from datetime import datetime

from airflow import DAG
from airflow.providers.cncf.kubernetes.operators.pod import KubernetesPodOperator


IMAGE = "cnapcloud/nyc-mlops:latest"

default_args = {
    "owner": "mlops",
}


with DAG(
    dag_id="taxi_xgboost_pipeline",
    start_date=datetime(2025, 1, 1),
    schedule=None,
    catchup=False,
    default_args=default_args,
    tags=["mlops", "xgboost", "ray"],
) as dag:

    # Step 1
    analyze = KubernetesPodOperator(
        task_id="step1_analyze",
        name="step1-analyze",
        namespace="default",

        image=IMAGE,

        cmds=["python"],
        arguments=[
            "/app/step1_analyze.py",
            "--input", "data/raw/",
            "--output", "reports/analysis",
        ],

        get_logs=True,
        is_delete_operator_pod=True,
    )

    # Step 2
    validate = KubernetesPodOperator(
        task_id="step2_validate",
        name="step2-validate",
        namespace="default",

        image=IMAGE,

        cmds=["python"],
        arguments=[
            "/app/step2_validate.py",
            "--input", "data/raw/",
            "--output", "reports/validation",
        ],

        get_logs=True,
        is_delete_operator_pod=True,
    )

    # Step 3
    train = KubernetesPodOperator(
        task_id="step3_train",
        name="step3-train",
        namespace="default",

        image=IMAGE,

        cmds=["python"],
        arguments=[
            "/app/step3_train.py",
            "--input", "data/raw/",
            "--mlflow-uri", "http://mlflow.mlops.svc.cluster.local",

            # distributed mode
            "--ray-address", "ray://raycluster-head-svc.default.svc.cluster.local:10001",
            "--num-workers", "2",
        ],

        get_logs=True,
        is_delete_operator_pod=True,
    )

    # Step 4
    evaluate = KubernetesPodOperator(
        task_id="step4_evaluate",
        name="step4-evaluate",
        namespace="default",

        image=IMAGE,

        cmds=["python"],
        arguments=[
            "/app/step4_evaluate.py",
            "--test-data", "data/raw/",
            "--mlflow-uri", "http://mlflow.mlops.svc.cluster.local",
            "--threshold", "0.0",
        ],

        get_logs=True,
        is_delete_operator_pod=True,
    )

    # Step 5
    register = KubernetesPodOperator(
        task_id="step5_register",
        name="step5-register",
        namespace="default",

        image=IMAGE,

        cmds=["python"],
        arguments=[
            "/app/step5_register.py",
            "--mlflow-uri", "http://mlflow.mlops.svc.cluster.local",
            "--auto-promote",
        ],

        get_logs=True,
        is_delete_operator_pod=True,
        logging_interval=3,
        startup_timeout_seconds=600,
    )

    analyze >> validate >> train >> evaluate >> register