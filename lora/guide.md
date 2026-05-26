# MLOps Pipeline 리팩토링 요구사항

현재 프로젝트는 Airflow Task 중심 구조(`task_01_*`, `task_02_*`)로 구성되어 있으며,
비즈니스 로직, orchestration, logging, MLflow 처리 등이 하나의 파일에 혼합되어 있다.

목표는 다음과 같다:

- Airflow 의존성 제거
- 로컬 실행 가능 구조 확보
- 테스트 용이성 향상
- Kubernetes/Airflow orchestration 분리
- MLOps 실무형 패키지 구조 적용
- 유지보수성 향상

---

## 1. 리팩토링 핵심 원칙

### 1-1. 비즈니스 로직과 orchestration 분리

**현재:**
```
task_05_register.py
  ├── smoke test 로직
  ├── MLflow stage 전환
  ├── logging
  ├── Airflow/XCom 의존
  └── CLI 실행
```

**변경 목표:**
```
순수 비즈니스 로직
    ↓
wrapper/orchestration layer
```

### 1-2. Airflow 종속 제거

비즈니스 로직은 다음을 몰라야 한다:

- Airflow
- KubernetesPodOperator
- XCom
- DAG
- Pod lifecycle

즉, **순수 Python 함수**로 유지한다.

```python
result = run_smoke_test(...)
```

### 1-3. 로컬 실행 가능 구조

모든 핵심 기능은 로컬에서 직접 실행 가능해야 한다.

```bash
python -m cli.run_eval
# 또는
python src/training/train.py
```

Airflow 없이도 실행 가능해야 한다.

### 1-4. XCom은 wrapper에서만 처리

XCom 저장은 orchestration wrapper에서만 수행한다.

```python
with open("/airflow/xcom/return.json", "w") as f:
    json.dump(result, f)
```

비즈니스 로직 내부에서는 XCom을 직접 다루지 않는다.

---

## 2. 패키지 구조 요구사항

**기존 구조는 제거되었고, 현재는 `src/` 도메인 기반 구조를 사용한다.**

### 목표 패키지 구조

```
project/
│
├── airflow/
│   └── dags/
│       └── training_pipeline.py
│
├── src/
│   ├── common/
│   ├── data/
│   ├── training/
│   ├── wrappers/
│   └── cli/
│
├── tests/
├── pyproject.toml
└── Dockerfile
```

서빙 관련 실행 로직은 별도 top-level 패키지로 두지 않고 `src/training/`, `src/wrappers/`, `src/cli/` 안에서 분리한다.
즉, Airflow는 DAG와 wrapper 호출만 담당하고, 실제 inference/승격 로직은 도메인 함수로 유지한다.

---

## 3. 파일별 역할 정의

### `common/`

공통 유틸리티. 포함 항목:

- logging 설정
- config
- MLflow helper
- 공통 exception

### `data/`

데이터 관련 처리. 포함 항목:

- 데이터 분석
- validation
- preprocessing

### `training/`

모델 lifecycle 관련 처리. 포함 항목:

- 학습
- 평가
- 모델 등록

### `src/training/` + `src/cli/`

평가/승격/서빙 성격의 실행 흐름은 도메인 로직과 로컬 진입점으로 나눠서 관리한다.

- `src/training/evaluate.py`: inference 또는 평가용 순수 함수
- `src/training/promote.py`: production promotion 로직
- `src/cli/run_eval.py`: 로컬 실행 진입점
- `src/cli/run_promote.py`: 로컬 실행 진입점

Airflow/Kubernetes 종속 레이어는 여기서 직접 다루지 않는다.

### `wrappers/`

Airflow/Kubernetes 종속 레이어. **여기서만** 다음을 처리한다:

- XCom에 전달할 최소 메타데이터
- `/airflow/xcom/return.json`
- KubernetesPodOperator
- env vars

Airflow wrapper는 도메인 함수를 호출한 뒤 URI/메타데이터만 XCom에 넘기는 얇은 껍데기여야 한다.

### `cli/`

로컬 실행 진입점.

```bash
python -m cli.run_eval
```

---

## 4. Logging 요구사항

**현재:**

```python
logging.getLogger(__name__)
```

직접 실행 시 `__main__`으로 출력된다.

**리팩토링 후:** 명시적 logger name 사용.

```python
log = logging.getLogger("smoke_test")
```

**출력 예시:**
```
2026-05-26 10:39:04 [INFO] smoke_test - [1/3] 테스트 중
```

---

## 5. 함수 구조 요구사항

**비즈니스 함수는 `dict` 반환:**

```python
def run_smoke_test(...) -> dict:
    return {
        "passed": True,
        "results": [...]
    }
```

**CLI entrypoint는 orchestration만 수행:**

```python
if __name__ == "__main__":
    main()
```

**Airflow wrapper는 XCom에 최소 메타데이터만 저장:**

```python
metadata = run()

with open("/airflow/xcom/return.json", "w") as f:
    json.dump(metadata, f)
```

---

## 6. MLflow 처리 요구사항

MLflow 관련 로직은 **공통 util 또는 orchestration layer**로 이동한다.

이동 대상:

- model load
- stage transition
- registry access

비즈니스 로직 내부에 직접 혼합하지 않는다.

---

## 7. 테스트 요구사항

**pytest 기반** 테스트 가능 구조로 변경.

```python
def test_smoke():
    result = run_smoke_test(...)
    assert result["passed"]
```

Airflow 없이 테스트 가능해야 한다.

---

## 8. pyproject.toml 요구사항

다음 구성 포함:

- setuptools 기반 패키지 구성
- editable install 지원
- pytest
- black
- ruff
- src layout

```bash
pip install -e .
```

가능해야 한다.

---

## 9. 최종 목표

최종적으로 프로젝트는 다음 특성을 가져야 한다:

| 항목 | 목표 |
|------|------|
| Airflow 역할 | orchestration만 담당 |
| 비즈니스 로직 | 순수 Python 유지 |
| 로컬 실행 | 가능 |
| 테스트 | Kubernetes 비의존 |
| XCom | 최소화 (URI/메타데이터만 전달) |
| 실제 데이터 | S3/MLflow 사용 |
| 패키지 구조 | 도메인 기반 적용 |
| 전체 방향 | 실무형 MLOps 구조 |
