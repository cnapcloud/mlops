# MLOps Platform

이 저장소는 MLflow, Apache Airflow, KubeRay를 기반으로 MLOps 플랫폼을 구성하고,
각 컴포넌트를 활용하기 위한 예제를 함께 제공합니다.

NYC Yellow Taxi 데이터셋을 활용하여 데이터 수집부터 전처리, 학습, 모델 등록, 서빙까지의
End-to-End 머신러닝 파이프라인을 구현하였습니다.

본 파이프라인은 로컬 실행과 Airflow 기반 실행 두 가지 방식으로 제공됩니다.

로컬 환경에서는 각 단계를 직접 실행할 수 있도록 구성되어 있으며,
Airflow 환경에서는 KubernetesPodOperator를 통해 모든 Task를 컨테이너 기반으로 오케스트레이션합니다.

학습 단계는 KubeRay와 연동하여 분산 학습 환경에서 수행되며,
학습된 모델은 MLflow에 저장 및 관리됩니다.

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