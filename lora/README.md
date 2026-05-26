# lora-pipeline

LoRA 기반 MLOps 파이프라인 프로젝트입니다. 데이터 분석/검증부터 KubeRay 학습, MLflow 모델 평가/승격까지 하나의 흐름으로 구성되어 있습니다.

## 주요 기능

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

## 빠른 시작 (로컬)

### 1) 가상환경 생성 + 의존성 설치

```bash
make venv
make install
```

### 2) 전체 파이프라인 실행

```bash
make run-pipeline
```

### 3) 특정 스테이지만 실행

```bash
make run-pipeline PIPELINE_STAGE=analysis
make run-pipeline PIPELINE_STAGE=validation
make run-pipeline PIPELINE_STAGE=train
make run-pipeline PIPELINE_STAGE=evaluate
make run-pipeline PIPELINE_STAGE=promote
```

## CLI 직접 실행

```bash
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
- 순서: `analysis -> validation -> train -> evaluate -> promote`
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