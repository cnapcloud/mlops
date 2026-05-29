# lora-pipeline

LoRA 기반 MLOps 파이프라인 프로젝트입니다. 데이터 분석/검증부터 KubeRay 학습, MLflow 모델 평가/승격까지 하나의 흐름으로 구성되어 있습니다.

## 주요 기능

- Task 0: 초기 데이터 생성 및 MinIO 업로드
- Task 1: 데이터 분석 (`analysis_report.json` 생성)
- Task 2: 데이터 품질 검증 (기준 미달 시 중단)
- Task 3: KubeRay 분산 학습 + MLflow 모델 등록
- Task 4: 신규 모델 vs Staging 모델 성능 비교 후 Staging 승격
- Task 5: Smoke Test 통과 시 Production 승격

## 프로젝트 구조

```text
.
├── airflow/
│   └── dags/
│       └── training_pipeline.py
├── docs/
│   └── senario.txt
├── src/
│   ├── cli/
│   ├── common/
│   ├── data/
│   ├── training/
│   └── wrappers/
├── tests/
├── Dockerfile
├── Makefile
└── pyproject.toml
```

## 요구 사항

- Python 3.12+
- Docker (이미지 빌드/푸시 시)
- Ray/MLflow 접속 정보
- MinIO 접속 정보
- Hugging Face 로그인 및 HF Token 발급 필요
  - https://huggingface.co/settings/tokens 에서 토큰 생성 후 HF_TOKEN 환경변수에 설정
- LLaMA 모델 접근 권한 요청 필요
  - https://huggingface.co/meta-llama/Llama-3.2-1B 에서 Meta 라이선스 동의 후 접근 승인 요청

## 빠른 시작 (로컬)

### 1) 가상환경 생성 + 의존성 설치

```bash
make venv
make install
```

Task 1 데이터 분석은 MinIO에 저장된 원본 JSON을 읽습니다. 로컬 실행 시 해당 object가 있어야 합니다.

### 1-1) 로컬 설정 파일 준비

`config.properties.example`을 복사해 `config-local.properties`를 만들고, 로컬 환경에 맞게 아래 값을 설정합니다.

- `HF_TOKEN`
- `RAY_ADDRESS`
- `MLFLOW_TRACKING_URI`
- `MINIO_URL`
- `MINIO_ACCESS_KEY`
- `MINIO_SECRET_KEY`
- `MINIO_INSECURE`
- `MINIO_BUCKET`
- `MINIO_RAW_OBJECT_KEY`

예시:

```bash
cp config.properties.example config-local.properties
```

### 1-2) 환경변수 설정

```bash
export CONFIG=local
export HF_TOKEN=
```

### 2) MinIO 초기 데이터 적재

```bash
make seed-data
```

### 3) 전체 파이프라인 실행

```bash
make run-pipeline
```

### 4) 특정 스테이지만 실행

```bash
make run-pipeline PIPELINE_STAGE=analysis
make run-pipeline PIPELINE_STAGE=validation
make run-pipeline PIPELINE_STAGE=train
make run-pipeline PIPELINE_STAGE=evaluate
make run-pipeline PIPELINE_STAGE=promote
```

## CLI 직접 실행

```bash
.venv/bin/python3 -m cli.run_seed
.venv/bin/python3 -m cli.run_analysis
.venv/bin/python3 -m cli.run_validation
.venv/bin/python3 -m cli.run_train
.venv/bin/python3 -m cli.run_eval
.venv/bin/python3 -m cli.run_promote
.venv/bin/python3 -m cli.run_pipeline
```

## Airflow + KubernetesPodOperator

- DAG 파일: `airflow/dags/training_pipeline.py`
- 오퍼레이터: `KubernetesPodOperator`
- 순서: `seed -> analysis -> validation -> train -> evaluate -> promote`
- 각 pod는 `wrappers.airflow_*` 모듈을 실행하고, 결과 메타데이터를 `/airflow/xcom/return.json`에 기록합니다.

필수 환경 변수(예시):

- `MLOPS_PIPELINE_IMAGE`
- `MLOPS_AIRFLOW_NAMESPACE`
- `MLOPS_PIPELINE_IMAGE_PULL_POLICY`
- `HF_TOKEN`
- `MLFLOW_TRACKING_URI`
- `RAY_ADDRESS`

## Docker 이미지 빌드/푸시

기본 이미지명은 `lora-pipeline:latest` 입니다.

```bash
make build
make push
make build-push
```

레지스트리/태그 지정 예시:

```bash
make build REGISTRY=ghcr.io/my-org IMAGE_NAME=lora-pipeline IMAGE_TAG=v1
make push REGISTRY=ghcr.io/my-org IMAGE_NAME=lora-pipeline IMAGE_TAG=v1
```

최종 이미지명 확인:

```bash
make print-image
```

## 테스트

```bash
.venv/bin/python3 -m pytest -q
```

## 참고 문서

- 시나리오: `docs/senario.txt`
- 리팩토링 가이드: `guide.md`