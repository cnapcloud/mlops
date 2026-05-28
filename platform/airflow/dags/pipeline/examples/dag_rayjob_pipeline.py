import os

from airflow.sdk import DAG, task
from datetime import datetime
from common.rayjob import RayJobSensor
import manifests

MANIFEST_FILE_PATH = os.path.join(manifests.MANIFEST_ROOT, "example_job.yaml")

with open(MANIFEST_FILE_PATH, "r", encoding="utf-8") as f:
    EXAMPLE_JOB_MANIFEST = f.read()

with DAG(
    dag_id="example_rayjob_pipeline",
    start_date=datetime(2025, 1, 1),
    schedule=None,
    catchup=False,
) as dag:

    @task(task_id="register_model")
    def register_model(job_name: str):
        """
        [pseudo code]

        학습 결과 artifact 경로 조회
        ─ S3 / MinIO 경로: s3://mlops/runs/{job_name}/model/

        MLflow Model Registry 등록
        ─ mlflow.set_tracking_uri(MLFLOW_URI)
        ─ mlflow.register_model(artifact_uri, model_name)
        ─ client.transition_model_version_stage(...)  # Staging or Production

        KServe InferenceService 배포 (선택)
        ─ InferenceService CRD body 구성
        ─ api.create_namespaced_custom_object(group="serving.kserve.io", ...)
        """
        print(f"[pseudo] registering model for job: {job_name}")

    rayjob_task = RayJobSensor(
        task_id="run_and_wait_rayjob",
        manifest=EXAMPLE_JOB_MANIFEST,
        namespace="default",
        poke_interval=30,  # 30초마다 상태 체크
        timeout=3600       # 최대 1시간 대기
    )

    rayjob_task >> register_model(rayjob_task.output)