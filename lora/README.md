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
- Ray/MLflow 접속 가능한 환경
- MinIO 접속 정보(`MINIO_URL`, `MINIO_ACCESS_KEY`, `MINIO_SECRET_KEY`, `MINIO_BUCKET`, `MINIO_RAW_OBJECT_KEY`)

환경값 예시는 [config.properties.example](config.properties.example) 를 참고해서 `./config.properties` 또는 `/etc/lora/config.properties` 로 복사해 사용하면 됩니다.

## 빠른 시작 (로컬)

### 1) 가상환경 생성 + 의존성 설치

```bash
make venv
make install
```

Task 1 데이터 분석은 MinIO에 저장된 원본 JSON을 읽습니다. 로컬 실행 시 해당 object가 있어야 합니다.

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