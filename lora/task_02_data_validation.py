"""
task_02_data_validation.py
--------------------------
Task 2: 데이터 품질 검증

수행 내용:
  - Task 1 분석 리포트(analysis_report.json)를 읽어서 품질 기준 검사
  - 검사 항목:
      1. Null 비율 < VALIDATION_MAX_NULL_RATIO (기본 5%)
      2. 중복 비율 < VALIDATION_MAX_DUP_RATIO  (기본 10%)
      3. 유효 샘플 수 >= VALIDATION_MIN_SAMPLES (기본 50건)
      4. 평균 추정 토큰 수 >= VALIDATION_MIN_AVG_TOKENS (기본 10)
  - 전체 통과 시 → 다음 Task 진행
  - 하나라도 실패 시 → 파이프라인 중단

반환값 (dict):
  - status        : "success" | "failed"
  - passed        : bool
  - checks        : 각 항목별 통과 여부 dict
  - report_path   : 저장된 검증 리포트 경로
"""

import json
import logging
import os

log = logging.getLogger(__name__)


def run() -> dict:
    log.info("=" * 60)
    log.info("Task 2: Data Validation 시작")
    log.info("=" * 60)

    try:
        from config import (
            ANALYSIS_REPORT_PATH,
            VALIDATION_REPORT_PATH,
            ARTIFACT_DIR,
            VALIDATION_MAX_NULL_RATIO,
            VALIDATION_MAX_DUP_RATIO,
            VALIDATION_MIN_SAMPLES,
            VALIDATION_MIN_AVG_TOKENS,
        )

        # ── 1. 분석 리포트 로드 ──────────────────────────────────────
        if not os.path.exists(ANALYSIS_REPORT_PATH):
            raise FileNotFoundError(
                f"분석 리포트를 찾을 수 없습니다: {ANALYSIS_REPORT_PATH}\n"
                "Task 1이 먼저 실행되어야 합니다."
            )

        with open(ANALYSIS_REPORT_PATH, "r", encoding="utf-8") as f:
            analysis = json.load(f)

        summary = analysis["summary"]
        log.info("분석 리포트 로드 완료: %s", ANALYSIS_REPORT_PATH)

        # ── 2. 각 항목 검사 ──────────────────────────────────────────
        checks = _run_checks(
            summary=summary,
            max_null_ratio=VALIDATION_MAX_NULL_RATIO,
            max_dup_ratio=VALIDATION_MAX_DUP_RATIO,
            min_samples=VALIDATION_MIN_SAMPLES,
            min_avg_tokens=VALIDATION_MIN_AVG_TOKENS,
        )

        # ── 3. 결과 판정 ─────────────────────────────────────────────
        passed = all(c["passed"] for c in checks.values())
        _print_result(checks, passed)

        # ── 4. 검증 리포트 저장 ──────────────────────────────────────
        report = {
            "passed": passed,
            "checks": checks,
            "thresholds": {
                "max_null_ratio":   VALIDATION_MAX_NULL_RATIO,
                "max_dup_ratio":    VALIDATION_MAX_DUP_RATIO,
                "min_samples":      VALIDATION_MIN_SAMPLES,
                "min_avg_tokens":   VALIDATION_MIN_AVG_TOKENS,
            },
        }
        os.makedirs(ARTIFACT_DIR, exist_ok=True)
        with open(VALIDATION_REPORT_PATH, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        log.info("검증 리포트 저장: %s", VALIDATION_REPORT_PATH)

        return {
            "status": "success",
            "passed": passed,
            "checks": checks,
            "report_path": VALIDATION_REPORT_PATH,
        }

    except Exception as e:
        log.error("Task 2 실패: %s", e, exc_info=True)
        return {"status": "failed", "passed": False, "error": str(e)}


# ─────────────────────────────────────────────
# 내부 함수
# ─────────────────────────────────────────────

def _run_checks(
    summary: dict,
    max_null_ratio: float,
    max_dup_ratio: float,
    min_samples: int,
    min_avg_tokens: float,
) -> dict:
    """각 품질 기준을 검사하고 결과 dict를 반환합니다."""

    checks = {}

    # 검사 1: Null 비율
    null_ratio = summary["null_ratio"]
    checks["null_ratio"] = {
        "description": f"Null 비율 < {max_null_ratio * 100:.0f}%",
        "actual":      round(null_ratio * 100, 2),
        "threshold":   max_null_ratio * 100,
        "passed":      null_ratio < max_null_ratio,
    }

    # 검사 2: 중복 비율
    dup_ratio = summary["duplicate_ratio"]
    checks["duplicate_ratio"] = {
        "description": f"중복 비율 < {max_dup_ratio * 100:.0f}%",
        "actual":      round(dup_ratio * 100, 2),
        "threshold":   max_dup_ratio * 100,
        "passed":      dup_ratio < max_dup_ratio,
    }

    # 검사 3: 최소 샘플 수
    valid_count = summary["valid_count"]
    checks["min_samples"] = {
        "description": f"유효 샘플 수 >= {min_samples}건",
        "actual":      valid_count,
        "threshold":   min_samples,
        "passed":      valid_count >= min_samples,
    }

    # 검사 4: 평균 토큰 수
    avg_tokens = summary["avg_tokens_estimated"]
    checks["avg_tokens"] = {
        "description": f"평균 추정 토큰 수 >= {min_avg_tokens}",
        "actual":      round(avg_tokens, 2),
        "threshold":   min_avg_tokens,
        "passed":      avg_tokens >= min_avg_tokens,
    }

    return checks


def _print_result(checks: dict, passed: bool) -> None:
    log.info("─" * 40)
    log.info("[ 데이터 검증 결과 ]")
    for name, result in checks.items():
        status_mark = "✅ PASS" if result["passed"] else "❌ FAIL"
        log.info(
            "  %s  %s  (실제값: %s, 기준: %s)",
            status_mark,
            result["description"],
            result["actual"],
            result["threshold"],
        )
    log.info("─" * 40)
    if passed:
        log.info("  최종 판정: ✅ 검증 통과 → 학습 진행")
    else:
        log.warning("  최종 판정: ❌ 검증 실패 → 파이프라인 중단")
    log.info("─" * 40)
