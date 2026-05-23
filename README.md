# MLOps Platform

**MLflow, Apache Airflow, KubeRay**를 기반으로 MLOps 플랫폼을 구성하고,
NYC Yellow Taxi 데이터셋을 활용하여 데이터 수집부터 전처리, 학습, 모델 등록, 서빙까지
End-to-End 머신러닝 파이프라인을 구현한 프로젝트입니다.

## 구성

- `taxi/`: NYC Taxi 예측용 기본 MLOps 파이프라인
- `kuberay/`: Ray 클러스터, Ray Job, Serve 예제
- `airflow/`: DAG 기반 오케스트레이션 예제
- `mlflow/`: MLflow 및 PostgreSQL PVC/Helm 설치 예제

## 접속 주소

- Airflow: http://airflow.cnapcloud.com
- MLflow: http://mlflow.cnapcloud.com

## 1. 설치

Helm 설치는 각 서비스의 `helm/` 폴더로 이동해서 실행합니다.

- `kuberay/helm/install.sh`
- `mlflow/helm/install.sh`
- `airflow/helm/install.sh`

예:

```bash
cd kuberay/helm && ./install.sh
cd mlflow/helm && ./install.sh
cd airflow/helm && ./install.sh
```

## 2. KubeRay

Ray 클러스터를 먼저 띄운 뒤, 브라우저로 LoadBalancer 주소에 접속합니다.

```bash
kubectl create -f ray-shared-pvc.yaml
kubectl create -f ray-cluster.yaml
kubectl apply -f rayjob/example-job.yaml
```

## 3. NYC Taxi 데모

```bash
cd taxi
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python pipeline.py --input data/raw/ --mode local --mlflow-uri http://localhost:5000
```

## 4. Airflow 파이프라인

주요 DAG는 다음 위치에 있습니다.

- `airflow/dags/pipeline/dag_lora_pipeline.py`
- `airflow/dags/pipeline/dag_lora_mlflow_pipeline.py`
- `airflow/dags/pipeline/nyc_taxi_pipeline.py`

Airflow UI에서 DAG를 trigger 합니다.

## 참고

- `airflow/`는 git sync 기반으로 DAG를 배포합니다.
- `taxi/`는 로컬 실행과 MLflow 연동 예제를 함께 보여줍니다.