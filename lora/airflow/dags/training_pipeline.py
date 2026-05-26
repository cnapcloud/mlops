"""Airflow DAG that runs each stage in a Kubernetes pod."""

from __future__ import annotations

import os
from datetime import datetime

from airflow import DAG
from airflow.providers.cncf.kubernetes.operators.pod import KubernetesPodOperator


def _pod_task(task_id: str, module_name: str) -> KubernetesPodOperator:
    return KubernetesPodOperator(
        task_id=task_id,
        name=task_id.replace("_", "-"),
        namespace=os.getenv("MLOPS_AIRFLOW_NAMESPACE", "default"),
        image=os.getenv("MLOPS_PIPELINE_IMAGE", "lora-pipeline:latest"),
        cmds=["python"],
        arguments=["-m", f"wrappers.{module_name}"],
        env_vars={
            "HF_TOKEN": os.getenv("HF_TOKEN", ""),
            "MLFLOW_TRACKING_URI": os.getenv("MLFLOW_TRACKING_URI", ""),
            "RAY_ADDRESS": os.getenv("RAY_ADDRESS", ""),
        },
        get_logs=True,
        is_delete_operator_pod=True,
        do_xcom_push=True,
        image_pull_policy=os.getenv("MLOPS_PIPELINE_IMAGE_PULL_POLICY", "IfNotPresent"),
    )


with DAG(
    dag_id="training_pipeline",
    start_date=datetime(2026, 1, 1),
    schedule=None,
    catchup=False,
    tags=["mlops", "lora", "kubernetes-pod-operator"],
) as dag:
    analysis = _pod_task("analysis", "airflow_analysis")
    validation = _pod_task("validation", "airflow_validation")
    train = _pod_task("train", "airflow_train")
    evaluate = _pod_task("evaluate", "airflow_eval")
    promote = _pod_task("promote", "airflow_promote")

    analysis >> validation >> train >> evaluate >> promote