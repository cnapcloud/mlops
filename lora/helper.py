import logging
import sys
import time
from typing import Dict

# 동일한 로거 이름을 사용해 main과 로그를 공유합니다.
log = logging.getLogger("main")


def _run_task(name: str, func, **kwargs) -> dict:
    """태스크를 실행하고 소요 시간을 로깅합니다."""
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
    """파이프라인을 중단하고 요약을 출력합니다."""
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
    log.info("║              파이프라인 완료 요약                     ║")
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
    """중단 시점까지의 태스크 상태 출력."""
    task_names = {
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
