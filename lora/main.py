"""
main.py
-------
MLOps 파이프라인 진입점

태스크 실행 순서:
  Task 1. Data Analysis      → 데이터 기초 통계 분석
  Task 2. Data Validation    → 품질 기준 검사 (실패 시 중단)
  Task 3. Train              → KubeRay 분산 학습 + MLflow 등록
  Task 4. Evaluate           → 신규 vs Staging 성능 비교 + 승격 (실패 시 중단)
  Task 5. Smoke Test         → 추론 검증 + Production 승격

실행 방법:
  python main.py

환경 변수:
  HF_TOKEN              HuggingFace 액세스 토큰 (필수)
  RAY_ADDRESS           KubeRay 클러스터 주소 (기본: ray://192.168.0.182:10001)
  MLFLOW_TRACKING_URI   MLflow 서버 주소 (기본: http://mlflow:5000)
  TRAIN_EPOCHS          학습 에포크 수 (기본: 3)
  RAY_NUM_WORKERS       분산 워커 수 (기본: 1)
  (전체 설정은 config.py 참조)
"""

import logging
import sys
import time
from datetime import datetime

# 파이프라인 태스크 임포트
import task_01_data_analysis  as t1
import task_02_data_validation as t2
import task_03_train_local           as t3
import task_04_evaluate        as t4
import task_05_smoke_test      as t5
from helper import _run_task, _abort, _print_final_summary


# ─────────────────────────────────────────────
# 로깅 설정
# ─────────────────────────────────────────────
def _setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(f"pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
        ],
    )

log = logging.getLogger("main")


# ─────────────────────────────────────────────
# 파이프라인 실행
# ─────────────────────────────────────────────
def main() -> None:
    _setup_logging()
    pipeline_start = time.time()

    log.info("▶▶▶ MLOps 파이프라인 시작 ◀◀◀")
    log.info("시작 시각: %s", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    results = {}  # 각 태스크 결과 보관

    # ──────────────────────────────────────────
    # Task 1: Data Analysis
    # ──────────────────────────────────────────
    results["t1"] = _run_task("Task 1: Data Analysis", t1.run)

    if results["t1"]["status"] != "success":
        _abort("Task 1 실패", results)

    # ──────────────────────────────────────────
    # Task 2: Data Validation
    # ──────────────────────────────────────────
    results["t2"] = _run_task("Task 2: Data Validation", t2.run)

    if results["t2"]["status"] != "success":
        _abort("Task 2 실행 오류", results)

    if not results["t2"]["passed"]:
        _abort("데이터 품질 검증 실패 → 파이프라인 중단", results)

    # ──────────────────────────────────────────
    # Task 3: Train + MLflow 등록
    # ──────────────────────────────────────────
    results["t3"] = _run_task("Task 3: Train", t3.run)

    if results["t3"]["status"] != "success":
        _abort("Task 3 학습 실패", results)

    # ──────────────────────────────────────────
    # Task 4: Evaluate & Staging 승격
    # ──────────────────────────────────────────
    results["t4"] = _run_task(
        "Task 4: Evaluate",
        t4.run,
        train_result=results["t3"],
    )

    if results["t4"]["status"] != "success":
        _abort("Task 4 평가 실패", results)

    if not results["t4"]["promoted"]:
        _abort(
            f"신규 모델이 기준 미달 → Staging 미승격\n  판정: {results['t4']['decision']}",
            results,
        )

    # ──────────────────────────────────────────
    # Task 5: Smoke Test + Production 승격
    # ──────────────────────────────────────────
    results["t5"] = _run_task(
        "Task 5: Smoke Test",
        t5.run,
        eval_result=results["t4"],
    )

    if results["t5"]["status"] != "success":
        _abort("Task 5 실행 오류", results)

    if not results["t5"]["promoted"]:
        _abort(
            f"Smoke Test 실패 → Production 미승격\n  판정: {results['t5']['decision']}",
            results,
        )

    # ──────────────────────────────────────────
    # 파이프라인 완료
    # ──────────────────────────────────────────
    elapsed = time.time() - pipeline_start
    _print_final_summary(results, elapsed)
    log.info("▶▶▶ 파이프라인 성공적으로 완료 ◀◀◀")
    sys.exit(0)


# 헬퍼 함수는 helper.py로 이동했습니다.


if __name__ == "__main__":
    main()
