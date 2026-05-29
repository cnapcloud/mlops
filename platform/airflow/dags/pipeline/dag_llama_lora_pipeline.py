"""Airflow DAG that runs each stage in a Kubernetes pod."""

from __future__ import annotations

import os
from datetime import datetime

from airflow import DAG
from airflow.providers.cncf.kubernetes.operators.pod import KubernetesPodOperator
from airflow.providers.standard.operators.python import ShortCircuitOperator
from kubernetes.client import models as k8s


def _pod_task(task_id: str, module_name: str, arguments: list[str] | None = None) -> KubernetesPodOperator:
    
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
    
    # Volume Mount 정의 (Pod 내부 컨테이너 안에서 보일 경로)
    volume_mount = k8s.V1VolumeMount(
        name="my-pvc-volume",           # 아래 Volume 이름과 일치해야 합니다.
        mount_path="/mnt/data",         # 컨테이너 내부에 마운트될 경로
        read_only=False
    )

    # Volume 정의 (실제 쿠버네티스 PVC 연결)
    volume = k8s.V1Volume(
        name="my-pvc-volume",
        persistent_volume_claim=k8s.V1PersistentVolumeClaimVolumeSource(
            claim_name="lora-training-pvc"   # 실제 쿠버네티스에 생성되어 있는 PVC 이름
        )
    )
    
    hf_homn_env = k8s.V1EnvVar(name="HF_HOME", value="/mnt/data/hf_home")
    airflow_env = k8s.V1EnvVar(name="AIRFLOW", value="True")
    
    return KubernetesPodOperator(
        task_id=task_id,
        name=task_id.replace("_", "-"),
        namespace=os.getenv("MLOPS_AIRFLOW_NAMESPACE", "default"),
        image=os.getenv("MLOPS_PIPELINE_IMAGE", "cnapcloud/lora-pipeline:latest"),
        cmds=["python"],
        arguments=["-m", f"wrappers.{module_name}", *(arguments or [])],
        env_vars=[
            secret_env,
            hf_homn_env,
            airflow_env
        ],
        get_logs=True,
        is_delete_operator_pod=True,
        do_xcom_push=True,
        image_pull_policy=os.getenv("MLOPS_PIPELINE_IMAGE_PULL_POLICY", "Always"),
        volumes=[volume],
        volume_mounts=[volume_mount],
        logging_interval=3,
        startup_timeout_seconds=600,
    )

def _check_promoted(**context):
    result = context["ti"].xcom_pull(task_ids="evaluate")
    if isinstance(result, str):
        import json
        result = json.loads(result)
    return result.get("promoted") is True


with DAG(
    dag_id="llama_lora_pipeline",
    start_date=datetime(2026, 1, 1),
    schedule=None,
    catchup=False,
    tags=["mlops", "lora", "ray"],
) as dag:
    seed = _pod_task("seed", "airflow_seed")
    analysis = _pod_task("analysis", "airflow_analysis")
    validation = _pod_task("validation", "airflow_validation")
    train = _pod_task("train", "airflow_train")
    check_promote = ShortCircuitOperator(
        task_id="check_promote",
        python_callable=_check_promoted,
    )
    evaluate = _pod_task(
        "evaluate",
        "airflow_eval",
        arguments=["{{ ti.xcom_pull(task_ids='train') | tojson }}"],
    )
    promote = _pod_task("promote", "airflow_promote")

    seed >> analysis >> validation >> train >> evaluate >> check_promote >>  promote