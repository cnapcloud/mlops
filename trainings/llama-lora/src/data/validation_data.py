"""Repair and validate seeded data before re-uploading it to MinIO."""

from __future__ import annotations

import json
import logging
import os

from common.config import (
    ANALYSIS_REPORT_PATH,
    ARTIFACT_DIR,
    MINIO_BUCKET,
    MINIO_RAW_OBJECT_KEY,
    MINIO_VALIDATED_OBJECT_KEY,
    VALIDATION_MAX_DUP_RATIO,
    VALIDATION_MAX_NULL_RATIO,
    VALIDATION_MIN_AVG_TOKENS,
    VALIDATION_MIN_SAMPLES,
    VALIDATION_REPORT_PATH,
)
from common.minio import (
    create_minio_client,
    ensure_bucket,
    get_json_object,
    put_json_object,
)
from data.analysis import _analyze

log = logging.getLogger("data.validation_data")


def run() -> dict:
    try:
        if not os.path.exists(ANALYSIS_REPORT_PATH):
            raise FileNotFoundError(
                f"Analysis report not found: "
                f"{ANALYSIS_REPORT_PATH}"
            )

        analysis = _load_analysis_report()

        source_summary = analysis["summary"]

        log.info(
            "Loaded analysis report: %s",
            ANALYSIS_REPORT_PATH,
        )

        source_checks = _run_checks(
            summary=source_summary,
            max_null_ratio=VALIDATION_MAX_NULL_RATIO,
            max_dup_ratio=VALIDATION_MAX_DUP_RATIO,
            min_samples=VALIDATION_MIN_SAMPLES,
            min_avg_tokens=VALIDATION_MIN_AVG_TOKENS,
        )

        source_texts = _load_seed_data()

        (
            repaired_texts,
            repair_actions,
            invalid_records,
            duplicate_records,
        ) = _repair_texts(
            source_texts,
            checks=source_checks,
            min_samples=VALIDATION_MIN_SAMPLES,
            min_avg_tokens=VALIDATION_MIN_AVG_TOKENS,
        )

        repaired_analysis = _analyze(repaired_texts)

        repaired_checks = _run_checks(
            summary=repaired_analysis["summary"],
            max_null_ratio=VALIDATION_MAX_NULL_RATIO,
            max_dup_ratio=VALIDATION_MAX_DUP_RATIO,
            min_samples=VALIDATION_MIN_SAMPLES,
            min_avg_tokens=VALIDATION_MIN_AVG_TOKENS,
        )

        passed = all(
            check["passed"]
            for check in repaired_checks.values()
        )

        _upload_to_minio(repaired_texts)

        _print_result(repaired_checks, passed)

        report = {
            "passed": passed,
            "checks": repaired_checks,
            "source_checks": source_checks,
            "repair_actions": repair_actions,
            "invalid_records_removed": invalid_records,
            "duplicate_records_removed": duplicate_records,
            "source_object_key": MINIO_RAW_OBJECT_KEY,
            "repaired_object_key": MINIO_VALIDATED_OBJECT_KEY,
            "thresholds": {
                "max_null_ratio": VALIDATION_MAX_NULL_RATIO,
                "max_dup_ratio": VALIDATION_MAX_DUP_RATIO,
                "min_samples": VALIDATION_MIN_SAMPLES,
                "min_avg_tokens": VALIDATION_MIN_AVG_TOKENS,
            },
            "source_summary": source_summary,
            "repaired_summary": repaired_analysis["summary"],
        }

        os.makedirs(ARTIFACT_DIR, exist_ok=True)

        with open(
            VALIDATION_REPORT_PATH,
            "w",
            encoding="utf-8",
        ) as handle:
            json.dump(
                report,
                handle,
                ensure_ascii=False,
                indent=2,
            )

        log.info(
            "Validation report saved: %s",
            VALIDATION_REPORT_PATH,
        )

        log.info(
            "Cleaned dataset uploaded: "
            "s3://%s/%s",
            MINIO_BUCKET,
            MINIO_VALIDATED_OBJECT_KEY,
        )

        return {
            "status": "success" if passed else "failed",
            "passed": passed,
            "checks": repaired_checks,
            "repair_actions": repair_actions,
            "invalid_records_removed": len(
                invalid_records
            ),
            "duplicate_records_removed": len(
                duplicate_records
            ),
            "source_object_key": MINIO_RAW_OBJECT_KEY,
            "repaired_object_key": (
                MINIO_VALIDATED_OBJECT_KEY
            ),
            "report_path": VALIDATION_REPORT_PATH,
        }

    except Exception as exc:
        log.error(
            "Validation task failed: %s",
            exc,
            exc_info=True,
        )

        return {
            "status": "failed",
            "passed": False,
            "error": str(exc),
        }


def _load_analysis_report() -> dict:
    with open(
        ANALYSIS_REPORT_PATH,
        "r",
        encoding="utf-8",
    ) as handle:
        return json.load(handle)


def _load_seed_data() -> list[str | None]:
    client = create_minio_client()

    payload = get_json_object(
        client,
        MINIO_BUCKET,
        MINIO_RAW_OBJECT_KEY,
    )

    if isinstance(payload, list):
        return payload

    if isinstance(payload, dict):
        candidate = (
            payload.get("texts")
            or payload.get("data")
            or payload.get("items")
        )

        if isinstance(candidate, list):
            return candidate

    raise ValueError(
        "MinIO object must contain "
        "a JSON list or texts/data/items"
    )


def _repair_texts(
    texts: list[str | None],
    checks: dict,
    min_samples: int,
    min_avg_tokens: float,
) -> tuple[
    list[str],
    list[str],
    list[str],
    list[str],
]:
    normalized: list[str] = []
    invalid_records: list[str] = []

    # 1. 유효하지 않은 데이터(None, 공백 등) 제거 및 정규화
    for text in texts:
        normalized_text = _normalize_text(text)

        if normalized_text is None:
            invalid_records.append(str(text))
            continue

        normalized.append(normalized_text)

    repair_actions: list[str] = []

    if invalid_records:
        repair_actions.append("removed_invalid_records")

    # 2. 중복 데이터 제거
    deduped: list[str] = []
    duplicate_records: list[str] = []
    seen: set[str] = set()

    for text in normalized:
        if text in seen:
            duplicate_records.append(text)
            continue

        seen.add(text)
        deduped.append(text)

    if duplicate_records:
        repair_actions.append("removed_duplicate_records")


    repaired = deduped

    return (
        repaired,
        repair_actions,
        invalid_records,
        duplicate_records,
    )


def _normalize_text(
    text: str | None,
) -> str | None:
    if text is None:
        return None

    if not isinstance(text, str):
        return str(text).strip() or None

    normalized = text.strip()

    return normalized or None


def _expand_text(
    text: str,
    min_avg_tokens: float,
) -> str:
    expanded = text

    suffix = (
        " Additional validation augmentation "
        "content was appended."
    )

    while (
        len(expanded.split()) * 1.3
        < min_avg_tokens
    ):
        expanded = f"{expanded}{suffix}"

    return expanded


def _pad_samples(
    texts: list[str],
    min_samples: int,
) -> list[str]:
    repaired = list(texts)

    base_texts = repaired or [
        (
            "Recovered fallback sample "
            "generated for validation."
        )
    ]

    index = 0

    while len(repaired) < min_samples:
        template = base_texts[
            index % len(base_texts)
        ]

        repaired.append(
            f"{template} "
            f"(augmented {index + 1})"
        )

        index += 1

    return repaired


def _upload_to_minio(
    texts: list[str],
) -> None:
    client = create_minio_client()

    ensure_bucket(
        client,
        MINIO_BUCKET,
    )

    put_json_object(
        client,
        MINIO_BUCKET,
        MINIO_VALIDATED_OBJECT_KEY,
        texts,
    )


def _run_checks(
    summary: dict,
    max_null_ratio: float,
    max_dup_ratio: float,
    min_samples: int,
    min_avg_tokens: float,
) -> dict:
    checks = {}

    null_ratio = summary["null_ratio"]

    checks["null_ratio"] = {
        "description": (
            f"Null ratio < "
            f"{max_null_ratio * 100:.0f}%"
        ),
        "actual": round(
            null_ratio * 100,
            2,
        ),
        "threshold": (
            max_null_ratio * 100
        ),
        "passed": (
            null_ratio < max_null_ratio
        ),
    }

    dup_ratio = summary["duplicate_ratio"]

    checks["duplicate_ratio"] = {
        "description": (
            f"Duplicate ratio < "
            f"{max_dup_ratio * 100:.0f}%"
        ),
        "actual": round(
            dup_ratio * 100,
            2,
        ),
        "threshold": (
            max_dup_ratio * 100
        ),
        "passed": (
            dup_ratio < max_dup_ratio
        ),
    }

    valid_count = summary["valid_count"]

    checks["min_samples"] = {
        "description": (
            f"Valid samples >= "
            f"{min_samples}"
        ),
        "actual": valid_count,
        "threshold": min_samples,
        "passed": (
            valid_count >= min_samples
        ),
    }

    avg_tokens = summary[
        "avg_tokens_estimated"
    ]

    checks["avg_tokens"] = {
        "description": (
            f"Average tokens >= "
            f"{min_avg_tokens}"
        ),
        "actual": round(
            avg_tokens,
            2,
        ),
        "threshold": min_avg_tokens,
        "passed": (
            avg_tokens >= min_avg_tokens
        ),
    }

    return checks


def _print_result(
    checks: dict,
    passed: bool,
) -> None:
    log.info("-" * 40)
    log.info("[ Validation Result ]")

    for result in checks.values():
        status = (
            "PASS"
            if result["passed"]
            else "FAIL"
        )

        log.info(
            "%s | %s | actual=%s | threshold=%s",
            status,
            result["description"],
            result["actual"],
            result["threshold"],
        )

    log.info("-" * 40)

    if passed:
        log.info(
            "Final result: validation passed"
        )
    else:
        log.warning(
            "Final result: validation failed"
        )

    log.info("-" * 40)