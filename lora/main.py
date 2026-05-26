"""Legacy pipeline entrypoint redirected to the refactored src layout."""

from __future__ import annotations

import logging
import sys
import time
from datetime import datetime

from helper import _abort, _print_final_summary, _run_task
from common.logging import setup_logging
from data.analysis import run as run_analysis
from data.validation import run as run_validation
from training.evaluate import run as run_evaluate
from training.promote import run as run_promote
from training.train import run as run_train

log = logging.getLogger("main")


def main() -> None:
    setup_logging("main")
    pipeline_start = time.time()

    log.info("▶▶▶ MLOps 파이프라인 시작 ◀◀◀")
    log.info("시작 시각: %s", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    results: dict = {}
    results["t1"] = _run_task("Task 1: Data Analysis", run_analysis)
    if results["t1"]["status"] != "success":
        _abort("Task 1 실패", results)

    results["t2"] = _run_task("Task 2: Data Validation", run_validation)
    if results["t2"]["status"] != "success":
        _abort("Task 2 실행 오류", results)
    if not results["t2"]["passed"]:
        _abort("데이터 품질 검증 실패 → 파이프라인 중단", results)

    results["t3"] = _run_task("Task 3: Train", run_train)
    if results["t3"]["status"] != "success":
        _abort("Task 3 학습 실패", results)

    results["t4"] = _run_task("Task 4: Evaluate", run_evaluate, train_result=results["t3"])
    if results["t4"]["status"] != "success":
        _abort("Task 4 평가 실패", results)
    if not results["t4"]["promoted"]:
        _abort(f"신규 모델이 기준 미달 → Staging 미승격\n  판정: {results['t4']['decision']}", results)

    results["t5"] = _run_task("Task 5: Smoke Test", run_promote)
    if results["t5"]["status"] != "success":
        _abort("Task 5 실행 오류", results)
    if not results["t5"]["promoted"]:
        _abort(f"Smoke Test 실패 → Production 미승격\n  판정: {results['t5']['decision']}", results)

    elapsed = time.time() - pipeline_start
    _print_final_summary(results, elapsed)
    log.info("▶▶▶ 파이프라인 성공적으로 완료 ◀◀◀")
    sys.exit(0)


if __name__ == "__main__":
    main()
