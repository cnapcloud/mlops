"""
task_01_data_analysis.py
------------------------
Task 1: 학습 데이터 기초 분석

수행 내용:
  - 샘플 데이터 100건 생성 (실제 환경에서는 외부 데이터 로드로 교체)
  - 텍스트 길이, 토큰 수, null/중복 비율 통계 계산
  - 분석 리포트를 artifacts/analysis_report.json 으로 저장
  - 분석 결과 요약 출력

반환값 (dict):
  - status        : "success" | "failed"
  - report_path   : 저장된 리포트 경로
  - summary       : 핵심 통계 요약 dict
"""

import json
import logging
import os
from collections import Counter

log = logging.getLogger(__name__)


def run() -> dict:
    try:
        # ── 1. 데이터 로드 ──────────────────────────────────────────
        raw_texts = _load_raw_data()
        log.info("원본 데이터 로드 완료: %d 건", len(raw_texts))

        # ── 2. 기초 통계 계산 ────────────────────────────────────────
        analysis = _analyze(raw_texts)

        # ── 3. 리포트 저장 ───────────────────────────────────────────
        report_path = _save_report(analysis)

        # ── 4. 요약 출력 ─────────────────────────────────────────────
        _print_summary(analysis)

        return {
            "status": "success",
            "report_path": report_path,
            "summary": analysis["summary"],
        }

    except Exception as e:
        log.error("Task 1 실패: %s", e, exc_info=True)
        return {"status": "failed", "error": str(e)}


# ─────────────────────────────────────────────
# 내부 함수
# ─────────────────────────────────────────────

def _load_raw_data() -> list[str]:
    """
    데모용 샘플 데이터 생성.
    실제 환경에서는 S3, PVC, DB 등에서 로드하는 코드로 교체.
    품질 검증 시나리오를 위해 의도적으로 null/중복 데이터를 소량 포함.
    """
    texts = []

    # 정상 데이터 88건
    for i in range(88):
        texts.append(
            f"질문 {i}: KubeRay 분산 학습 테스트 시나리오입니다. "
            f"답변 {i}: Kubernetes 위에서 Ray 클러스터를 구성하여 "
            f"분산 학습을 수행하는 정상 데이터입니다."
        )

    # 중복 데이터 7건 (동일 문장 반복)
    for _ in range(7):
        texts.append(
            "질문 0: KubeRay 분산 학습 테스트 시나리오입니다. "
            "답변 0: Kubernetes 위에서 Ray 클러스터를 구성하여 "
            "분산 학습을 수행하는 정상 데이터입니다."
        )

    # null/빈 데이터 5건
    for _ in range(5):
        texts.append(None)

    return texts


def _analyze(texts: list) -> dict:
    """텍스트 리스트에 대한 기초 통계를 계산합니다."""
    total = len(texts)

    # null 분석
    null_items  = [t for t in texts if t is None or (isinstance(t, str) and t.strip() == "")]
    valid_texts = [t for t in texts if t is not None and isinstance(t, str) and t.strip() != ""]

    null_count = len(null_items)
    null_ratio = null_count / total if total > 0 else 0.0

    # 중복 분석
    counter     = Counter(valid_texts)
    dup_texts   = [t for t, cnt in counter.items() if cnt > 1]
    dup_count   = sum(cnt - 1 for cnt in counter.values() if cnt > 1)
    dup_ratio   = dup_count / len(valid_texts) if valid_texts else 0.0

    # 길이 통계 (단어 기준 — 토크나이저 없는 경량 분석)
    word_counts = [len(t.split()) for t in valid_texts]
    char_counts = [len(t) for t in valid_texts]

    avg_words = sum(word_counts) / len(word_counts) if word_counts else 0
    max_words = max(word_counts) if word_counts else 0
    min_words = min(word_counts) if word_counts else 0
    avg_chars = sum(char_counts) / len(char_counts) if char_counts else 0

    # 토큰 수 추정 (단어 수 × 1.3 — 간이 추정)
    avg_tokens_estimated = avg_words * 1.3

    summary = {
        "total_count":              total,
        "valid_count":              len(valid_texts),
        "null_count":               null_count,
        "null_ratio":               round(null_ratio, 4),
        "duplicate_count":          dup_count,
        "duplicate_ratio":          round(dup_ratio, 4),
        "avg_word_count":           round(avg_words, 2),
        "max_word_count":           max_words,
        "min_word_count":           min_words,
        "avg_char_count":           round(avg_chars, 2),
        "avg_tokens_estimated":     round(avg_tokens_estimated, 2),
        "unique_duplicate_texts":   len(dup_texts),
    }

    return {
        "summary": summary,
        "duplicate_samples": dup_texts[:3],  # 중복 샘플 최대 3건만 기록
    }


def _save_report(analysis: dict) -> str:
    from config import ANALYSIS_REPORT_PATH, ARTIFACT_DIR

    os.makedirs(ARTIFACT_DIR, exist_ok=True)
    with open(ANALYSIS_REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(analysis, f, ensure_ascii=False, indent=2)

    log.info("분석 리포트 저장: %s", ANALYSIS_REPORT_PATH)
    return ANALYSIS_REPORT_PATH


def _print_summary(analysis: dict) -> None:
    s = analysis["summary"]
    log.info("─" * 40)
    log.info("[ 데이터 분석 요약 ]")
    log.info("  전체 건수       : %d", s["total_count"])
    log.info("  유효 건수       : %d", s["valid_count"])
    log.info("  Null 비율       : %.1f%%", s["null_ratio"] * 100)
    log.info("  중복 비율       : %.1f%%", s["duplicate_ratio"] * 100)
    log.info("  평균 단어 수    : %.1f", s["avg_word_count"])
    log.info("  추정 평균 토큰  : %.1f", s["avg_tokens_estimated"])
    log.info("─" * 40)
