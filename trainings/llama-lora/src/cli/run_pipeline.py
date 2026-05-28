"""CLI dispatcher for the full pipeline or individual stages."""

from __future__ import annotations

import argparse
import logging
import sys
import time
from datetime import datetime
from typing import Dict

from common.logging import setup_logging
from data.analysis import run as run_analysis
from data.validation_data import run as run_validation
from data.seed_data import run as run_seed
from training.evaluate import run as run_evaluate
from training.promote import run as run_promote
from training.train import run as run_train

log = logging.getLogger("pipeline")

def _run_task(name: str, func, **kwargs) -> dict:
    log.info("")
    log.info("━" * 60)
    log.info("▶ %s", name)
    log.info("━" * 60)

    start = time.time()
    result = func(**kwargs) if kwargs else func()
    elapsed = time.time() - start

    status = result.get("status", "unknown")
    log.info("◀ %s 완료 | status=%s | 소요시간=%.1fs", name, status, elapsed)
    return result


def _abort(reason: str, results: Dict) -> None:
    log.error("")
    log.error("╔══════════════════════════════════════════════════════╗")
    log.error("║              파이프라인 중단                          ║")
    log.error("╠══════════════════════════════════════════════════════╣")
    log.error("  사유: %s", reason)
    log.error("╚══════════════════════════════════════════════════════╝")
    _print_partial_summary(results)
    sys.exit(1)


def _print_final_summary(results: Dict, elapsed: float) -> None:
    t3 = results.get("t3", {})
    t4 = results.get("t4", {})
    t5 = results.get("t5", {})

    log.info("")
    log.info("╔══════════════════════════════════════════════════════╗")
    log.info("║              파이프라인 완료 요약                         ║")
    log.info("╠══════════════════════════════════════════════════════╣")
    log.info("  모델명      : %s", t3.get("model_name", "-"))
    log.info("  등록 버전   : v%s", t3.get("model_version", "-"))
    log.info("  MLflow RunID: %s", t3.get("run_id", "-"))
    log.info("  학습 메트릭 : %s", t3.get("metrics", {}))
    log.info("  신규 PPL    : %.4f", t4.get("new_perplexity", 0.0))
    log.info(
        "  이전 PPL    : %s",
        f"{t4.get('prev_perplexity', 0.0):.4f}" if t4.get("prev_perplexity") else "없음 (최초)",
    )
    log.info("  Smoke Test  : %d건 전체 통과", len(t5.get("test_results", [])))
    log.info("  Production  : ✅ 승격 완료")
    log.info("  총 소요시간  : %.1f분", elapsed / 60)
    log.info("╚══════════════════════════════════════════════════════╝")


def _print_partial_summary(results: Dict) -> None:
    task_names = {
        "t0": "Task 0 Data Seed",
        "t1": "Task 1 Data Analysis",
        "t2": "Task 2 Data Validation",
        "t3": "Task 3 Train",
        "t4": "Task 4 Evaluate",
        "t5": "Task 5 Smoke Test",
    }
    log.info("[ 태스크별 실행 결과 ]")
    for key, name in task_names.items():
        if key in results:
            status = results[key].get("status", "unknown")
            mark = "✅" if status == "success" else "❌"
            log.info("  %s %s: %s", mark, name, status)
        else:
            log.info("  ⬜ %s: 미실행", name)


def _run_full_pipeline() -> None:
    pipeline_start = time.time()

    log.info("▶▶▶ MLOps 파이프라인 시작 ◀◀◀")
    log.info("시작 시각: %s", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    results: dict = {}
    results["t0"] = _run_task("Task 0: Data Seed", run_seed)
    if results["t0"]["status"] != "success":
        _abort("Task 0 실패", results)

    results["t1"] = _run_task("Task 1: Data Analysis", run_analysis)
    if results["t1"]["status"] != "success":
        _abort("Task 1 실패", results)

    results["t2"] = _run_task("Task 2: Data Validation", run_validation)
    # if results["t2"]["status"] != "success":
    #    _abort("Task 2 실행 오류", results)
    #if not results["t2"]["passed"]:
    #    _abort("데이터 품질 검증 실패 → 파이프라인 중단", results)

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


def _run_single_stage(stage: str) -> None:
    if stage == "seed":
        run_seed()
    elif stage == "analysis":
        run_analysis()
    elif stage == "validation":
        run_validation()
    elif stage == "train":
        run_train()
    elif stage == "evaluate":
        run_evaluate()
    elif stage == "promote":
        run_promote()
    else:
        raise ValueError(f"알 수 없는 stage: {stage}")


def main() -> None:
    setup_logging("pipeline")

    parser = argparse.ArgumentParser(description="MLOps pipeline CLI")
    parser.add_argument(
        "stage",
        nargs="?",
        default="all",
        choices=["all", "seed", "analysis", "validation", "train", "evaluate", "promote"],
        help="실행할 stage (기본값: all)",
    )
    args = parser.parse_args()

    if args.stage == "all":
        _run_full_pipeline()
    else:
        _run_single_stage(args.stage)


if __name__ == "__main__":
    main()
