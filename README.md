# MLOps Platform

MLflow, Apache Airflow, KubeRay를 기반으로 구성한 MLOps 플랫폼입니다.
"데이터 분석 → 데이터 검증 → 학습 → 모델 평가 → 모델 등록"까지의 End-to-End 파이프라인을 제공하며,
로컬 실행과 Airflow 기반 실행 두 가지 방식을 지원합니다.

---

## 디렉토리 구성

```
.
├── platform/               # 플랫폼 인프라
│   ├── minio/              # 모델 아티팩트 오브젝트 스토리지
│   ├── mlflow/             # 실험 추적 및 모델 레지스트리
│   ├── kuberay/            # Ray 클러스터 / Job / Serve
│   └── airflow/            # DAG 기반 파이프라인 오케스트레이션
└── trainings/              # 학습 예제
    ├── llama-lora/         # LLM LoRA 파인튜닝 (Llama 3.2)
    └── taxi-xgboost/       # NYC Taxi 요금 예측 (XGBoost)
```

---

## 접속 주소

| 서비스 | URL |
|--------|-----|
| MinIO Console | http://console.cnapcloud.com |
| MinIO API | http://minio-api.cnapcloud.com |
| MLflow | http://mlflow.cnapcloud.com |
| KubeRay | http://kuberay.cnapcloud.com |
| Airflow | http://airflow.cnapcloud.com |

---

## Part 1. 플랫폼 설치

각 서비스는 `platform/` 하위 디렉토리에서 순서대로 설치합니다.

### Step 1. MinIO

```bash
cd platform/minio && make apply
```

설치 후 콘솔(http://console.cnapcloud.com)에 접속하여 MLflow용 버킷을 생성합니다.

| 항목 | 값 |
|------|----|
| Bucket | `mlflow` |
| AWS_ACCESS_KEY_ID | `mLuTMCv1SVSfycZgX4th` |
| AWS_SECRET_ACCESS_KEY | `pMmJbYQO4o12gFy1rsTX9mzN9ELgQplOQ1nrwHLf` |

> MinIO는 디스크 사용량이 **75~80%** 를 초과하면 `SlowDownWrite` 오류와 함께 쓰기를 거부합니다.
> 설치 전 여유 공간을 확인하세요.
> ```bash
> df -h /data
> ```

### Step 2. MLflow

```bash
cd platform/mlflow/helm && ./install.sh
```

### Step 3. KubeRay

```bash
cd platform/kuberay/helm && ./install.sh
```

Ray 클러스터와 공유 PVC를 생성합니다.

```bash
cd platform/kuberay/raycluster
kubectl create -f ray-shared-pvc.yaml
kubectl create -f ray-cluster.yaml
```

### Step 4. Airflow

```bash
cd platform/airflow/helm && ./install.sh
```

> DAG는 Git Sync 방식으로 배포됩니다. `platform/airflow/dags/` 경로의 파일이 자동으로 반영됩니다.

---

## Part 2. 학습 예제

### 2-1. LLM LoRA 파인튜닝 (Llama 3.2-1B)

`trainings/llama-lora/` — HuggingFace PEFT 기반 LoRA 파인튜닝 파이프라인입니다.
학습 결과는 MLflow에 등록되고 아티팩트는 MinIO에 저장됩니다.

**로컬 실행**

```bash
cd trainings/llama-lora
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 전체 파이프라인 실행
python -m src.cli.run_pipeline

# 단계별 실행
python -m src.cli.run_seed        # 데이터 준비
python -m src.cli.run_analysis    # 데이터 분석
python -m src.cli.run_validation  # 데이터 검증
python -m src.cli.run_train       # 학습
python -m src.cli.run_eval        # 평가
python -m src.cli.run_promote     # 모델 등록
```

---

### 2-2. NYC Taxi 요금 예측 (XGBoost)

`trainings/taxi-xgboost/` — NYC Yellow Taxi 데이터셋을 활용한 정형 데이터 파이프라인입니다.

**로컬 실행**

```bash
cd trainings/taxi-xgboost
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 전체 파이프라인 실행
python pipeline.py --input data/raw/ --mode local --mlflow-uri http://mlflow.cnapcloud.com

# 단계별 실행
python step1_analyze.py    # 데이터 분석
python step2_validate.py   # 데이터 검증
python step3_train.py      # 모델 학습
python step4_evaluate.py   # 모델 평가
python step5_register.py   # 모델 등록
```

**Airflow 실행**

**Airflow 실행**

Airflow UI(http://airflow.cnapcloud.com)에서 DAG를 trigger합니다.

| DAG | 경로 | 설명 |
|-----|------|------|
| `ddag_llama_lora_pipeline.py` | `platform/airflow/dags/pipeline/` | LLaMA 학습 |
| `dag_taxi_xgboost_pipeline.py` | `platform/airflow/dags/pipeline/examples/` | NYC Taxi 요금 예측 |
| `dag_lora_pipeline.py` | `platform/airflow/dags/pipeline/examples/` | LoRA 실행 예제 |
| `dag_rayjob_pipeline.py` | `platform/airflow/dags/pipeline/examples/` | RayJob 실행 예제 |

---

## 아키텍처

llama-lora, taxi-xgboost는 kuberay 기반으로 분산학습을 수행하고 학습 결과를 mlflow에 저장합니다.

```
Airflow DAG
    │
    ├── KubernetesPodOperator ──→ trainings/llama-lora  (LoRA 학습)
    │        │
    │        └── KubeRay TorchTrainer  (분산 학습)
    │
    ├── KubernetesPodOperator ──→ trainings/taxi-xgboost (XGBoost 학습)
    │        │
    │        └── KubeRay TorchTrainer  (분산 학습)    
    │
    └── 학습 결과 ──→ MLflow Model Registry ──→ MinIO (아티팩트)
```


## LLaMA 학습 파이프라인 사전 준비
LLaMA 모델은 HuggingFace에서 다운로드하여 PVC에 저장해두고 재사용합니다.
llama-lora 파이프라인을 trigger하기 전에 아래 작업을 먼저 수행합니다.

```
#모델 저장용 PVC 생성
kubectl apply -f platform/airflow/dags/manifests/lora-pvc.yaml

#HuggingFace 토큰 Secret 생성
kubectl create secret generic hf-secret \
  --from-literal=HF_TOKEN=<your_hf_token>
```
토큰은 HuggingFace → Settings → Access Tokens에서 발급받으며, LLaMA 모델 접근 권한(gated model)이 사전 승인된 계정이어야 합니다.

> KubeRay 클러스터와 llama-lora 파이프라인 최초 실행 시에는 모델 다운로드가 포함되어 실행 시간이 오래 걸립니다.
> 이후 실행부터는 PVC에 캐싱된 모델을 재사용합니다.