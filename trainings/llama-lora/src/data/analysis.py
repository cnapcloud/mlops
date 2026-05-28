"""Data analysis domain logic."""

from __future__ import annotations

import json
import logging
import os
from collections import Counter

from common.config import (
    ANALYSIS_REPORT_PATH,
    ARTIFACT_DIR,
    MINIO_BUCKET,
    MINIO_RAW_OBJECT_KEY,
)
from common.minio import create_minio_client, get_json_object

log = logging.getLogger("data.analysis")


def run() -> dict:
    try:
        raw_texts = _load_raw_data()
        log.info("Raw data loaded successfully: %d records", len(raw_texts))

        analysis = _analyze(raw_texts)
        report_path = _save_report(analysis)
        _print_summary(analysis)

        return {"status": "success", "report_path": report_path, "summary": analysis["summary"]}
    except Exception as exc:
        log.error("Task 1 failed: %s", exc, exc_info=True)
        return {"status": "failed", "error": str(exc)}


def _load_raw_data() -> list[str | None]:
    client = create_minio_client()
    payload = get_json_object(client, MINIO_BUCKET, MINIO_RAW_OBJECT_KEY)

    if isinstance(payload, list):
        return payload

    if isinstance(payload, dict):
        candidate = payload.get("texts") or payload.get("data") or payload.get("items")
        if isinstance(candidate, list):
            return candidate

    raise ValueError(
        f"MinIO object must contain a JSON list or a mapping with texts/data/items: s3://{MINIO_BUCKET}/{MINIO_RAW_OBJECT_KEY}"
    )


def _analyze(texts: list[str | None]) -> dict:
    total = len(texts)
    null_items = [t for t in texts if t is None or (isinstance(t, str) and t.strip() == "")]
    valid_texts = [t for t in texts if t is not None and isinstance(t, str) and t.strip() != ""]

    null_count = len(null_items)
    null_ratio = null_count / total if total > 0 else 0.0

    counter = Counter(valid_texts)
    dup_texts = [t for t, cnt in counter.items() if cnt > 1]
    dup_count = sum(cnt - 1 for cnt in counter.values() if cnt > 1)
    dup_ratio = dup_count / len(valid_texts) if valid_texts else 0.0

    word_counts = [len(t.split()) for t in valid_texts]
    char_counts = [len(t) for t in valid_texts]
    avg_words = sum(word_counts) / len(word_counts) if word_counts else 0
    max_words = max(word_counts) if word_counts else 0
    min_words = min(word_counts) if word_counts else 0
    avg_chars = sum(char_counts) / len(char_counts) if char_counts else 0
    avg_tokens_estimated = avg_words * 1.3

    summary = {
        "total_count": total,
        "valid_count": len(valid_texts),
        "null_count": null_count,
        "null_ratio": round(null_ratio, 4),
        "duplicate_count": dup_count,
        "duplicate_ratio": round(dup_ratio, 4),
        "avg_word_count": round(avg_words, 2),
        "max_word_count": max_words,
        "min_word_count": min_words,
        "avg_char_count": round(avg_chars, 2),
        "avg_tokens_estimated": round(avg_tokens_estimated, 2),
        "unique_duplicate_texts": len(dup_texts),
    }

    return {"summary": summary, "duplicate_samples": dup_texts[:3]}


def _save_report(analysis: dict) -> str:
    os.makedirs(ARTIFACT_DIR, exist_ok=True)
    with open(ANALYSIS_REPORT_PATH, "w", encoding="utf-8") as handle:
        json.dump(analysis, handle, ensure_ascii=False, indent=2)
    log.info("Analysis report saved to: %s", ANALYSIS_REPORT_PATH)
    return ANALYSIS_REPORT_PATH


def _print_summary(analysis: dict) -> None:
    summary = analysis["summary"]
    log.info("─" * 40)
    log.info("[ Data Analysis Summary ]")
    log.info("  Total Records         : %d", summary["total_count"])
    log.info("  Valid Records         : %d", summary["valid_count"])
    log.info("  Null Ratio            : %.1f%%", summary["null_ratio"] * 100)
    log.info("  Duplicate Ratio       : %.1f%%", summary["duplicate_ratio"] * 100)
    log.info("  Avg Word Count        : %.1f", summary["avg_word_count"])
    log.info("  Estimated Avg Tokens  : %.1f", summary["avg_tokens_estimated"])
    log.info("─" * 40)