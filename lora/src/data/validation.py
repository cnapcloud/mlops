"""Data validation domain logic."""

from __future__ import annotations

import json
import logging
import os

from common.config import (
    ANALYSIS_REPORT_PATH,
    ARTIFACT_DIR,
    VALIDATION_MAX_DUP_RATIO,
    VALIDATION_MAX_NULL_RATIO,
    VALIDATION_MIN_AVG_TOKENS,
    VALIDATION_MIN_SAMPLES,
    VALIDATION_REPORT_PATH,
)

log = logging.getLogger("data.validation")


def run() -> dict:
    try:
        if not os.path.exists(ANALYSIS_REPORT_PATH):
            raise FileNotFoundError(
                f"분석 리포트를 찾을 수 없습니다: {ANALYSIS_REPORT_PATH}\nTask 1이 먼저 실행되어야 합니다."
            )

        with open(ANALYSIS_REPORT_PATH, "r", encoding="utf-8") as handle:
            analysis = json.load(handle)

        summary = analysis["summary"]
        log.info("분석 리포트 로드 완료: %s", ANALYSIS_REPORT_PATH)

        checks = _run_checks(
            summary=summary,
            max_null_ratio=VALIDATION_MAX_NULL_RATIO,
            max_dup_ratio=VALIDATION_MAX_DUP_RATIO,
            min_samples=VALIDATION_MIN_SAMPLES,
            min_avg_tokens=VALIDATION_MIN_AVG_TOKENS,
        )

        passed = all(check["passed"] for check in checks.values())
        _print_result(checks, passed)

        report = {
            "passed": passed,
            "checks": checks,
            "thresholds": {
                "max_null_ratio": VALIDATION_MAX_NULL_RATIO,
                "max_dup_ratio": VALIDATION_MAX_DUP_RATIO,
                "min_samples": VALIDATION_MIN_SAMPLES,
                "min_avg_tokens": VALIDATION_MIN_AVG_TOKENS,
            },
        }
        os.makedirs(ARTIFACT_DIR, exist_ok=True)
        with open(VALIDATION_REPORT_PATH, "w", encoding="utf-8") as handle:
            json.dump(report, handle, ensure_ascii=False, indent=2)
        log.info("검증 리포트 저장: %s", VALIDATION_REPORT_PATH)

        return {"status": "success", "passed": passed, "checks": checks, "report_path": VALIDATION_REPORT_PATH}
    except Exception as exc:
        log.error("Task 2 실패: %s", exc, exc_info=True)
        return {"status": "failed", "passed": False, "error": str(exc)}


def _run_checks(summary: dict, max_null_ratio: float, max_dup_ratio: float, min_samples: int, min_avg_tokens: float) -> dict:
    checks = {}
    null_ratio = summary["null_ratio"]
    checks["null_ratio"] = {
        "description": f"Null 비율 < {max_null_ratio * 100:.0f}%",
        "actual": round(null_ratio * 100, 2),
        "threshold": max_null_ratio * 100,
        "passed": null_ratio < max_null_ratio,
    }

    dup_ratio = summary["duplicate_ratio"]
    checks["duplicate_ratio"] = {
        "description": f"중복 비율 < {max_dup_ratio * 100:.0f}%",
        "actual": round(dup_ratio * 100, 2),
        "threshold": max_dup_ratio * 100,
        "passed": dup_ratio < max_dup_ratio,
    }

    valid_count = summary["valid_count"]
    checks["min_samples"] = {
        "description": f"유효 샘플 수 >= {min_samples}건",
        "actual": valid_count,
        "threshold": min_samples,
        "passed": valid_count >= min_samples,
    }

    avg_tokens = summary["avg_tokens_estimated"]
    checks["avg_tokens"] = {
        "description": f"평균 추정 토큰 수 >= {min_avg_tokens}",
        "actual": round(avg_tokens, 2),
        "threshold": min_avg_tokens,
        "passed": avg_tokens >= min_avg_tokens,
    }

    return checks


def _print_result(checks: dict, passed: bool) -> None:
    log.info("─" * 40)
    log.info("[ 데이터 검증 결과 ]")
    for _, result in checks.items():
        status_mark = "✅ PASS" if result["passed"] else "❌ FAIL"
        log.info("  %s  %s  (실제값: %s, 기준: %s)", status_mark, result["description"], result["actual"], result["threshold"])
    log.info("─" * 40)
    if passed:
        log.info("  최종 판정: ✅ 검증 통과 → 학습 진행")
    else:
        log.warning("  최종 판정: ❌ 검증 실패 → 파이프라인 중단")
    log.info("─" * 40)
