"""Airflow DAG that runs each stage in a Kubernetes pod."""

from __future__ import annotations

import os
from datetime import datetime

from airflow import DAG
from airflow.providers.cncf.kubernetes.operators.pod import KubernetesPodOperator
from kubernetes.client import models as k8s

def _pod_task(task_id: str, module_name: str, arguments: list[str] | None = None, startup_timeout: int = 600) -> KubernetesPodOperator:
    """
     Kubernetes Secret을 환경 변수로 매핑
     kubectl  create secret generic hf-secret \
     --from-literal=HF_TOKEN=your_hf_token -n mlops
    """
    secret_env = k8s.V1EnvVar(
        name="HF_TOKEN",
        value_from=k8s.V1EnvVarSource(
            secret_key_ref=k8s.V1SecretKeySelector(
                name="hf-secret",  # k8s에 생성한 secret 이름
                key="HF_TOKEN"     # secret 내부의 key 이름
            )
        )
    )

    return KubernetesPodOperator(
        task_id=task_id,
        name=task_id.replace("_", "-"),
        namespace=os.getenv("MLOPS_AIRFLOW_NAMESPACE", "default"),
        image=os.getenv("MLOPS_PIPELINE_IMAGE", "cnapcloud/lora-pipeline:latest"),
        cmds=["python"],
        arguments=["-m", f"wrappers.{module_name}", *(arguments or [])],
        env_vars=[secret_env],
        get_logs=True,
        is_delete_operator_pod=True,
        do_xcom_push=True,
        image_pull_policy=os.getenv("MLOPS_PIPELINE_IMAGE_PULL_POLICY", "Always"),
        startup_timeout_seconds=startup_timeout,
    )


with DAG(
    dag_id="training_pipeline",
    start_date=datetime(2026, 1, 1),
    schedule=None,
    catchup=False,
    tags=["mlops", "lora", "kubernetes-pod-operator"],
) as dag:
    seed = _pod_task("seed", "airflow_seed")
    analysis = _pod_task("analysis", "airflow_analysis")
    validation = _pod_task("validation", "airflow_validation")
    train = _pod_task("train", "airflow_train")
    evaluate = _pod_task(
        "evaluate",
        "airflow_eval",
        arguments=["{{ ti.xcom_pull(task_ids='train') | tojson }}"],
    )
    promote = _pod_task("promote", "airflow_promote")

    seed >> analysis >> validation >> train >> evaluate >> promote