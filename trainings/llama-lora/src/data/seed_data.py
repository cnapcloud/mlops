"""Seed initial MinIO data."""

from __future__ import annotations

import logging

from common.config import MINIO_BUCKET, MINIO_RAW_OBJECT_KEY, TRAINING_DATA
from common.minio import create_minio_client, ensure_bucket, put_json_object

log = logging.getLogger("data.seed_data")


def run() -> dict:
    try:
        seed_data = _build_seed_data()
        _upload_to_minio(seed_data)
        log.info("Initial data upload completed: s3://%s/%s (%d건)", MINIO_BUCKET, MINIO_RAW_OBJECT_KEY, len(seed_data))
        return {
            "status": "success",
            "bucket": MINIO_BUCKET,
            "object_key": MINIO_RAW_OBJECT_KEY,
            "record_count": len(seed_data),
        }
    except Exception as exc:
        log.error("Task 0 failed: %s", exc, exc_info=True)
        raise


def _build_seed_data() -> list[str | None]:
    texts: list[str | None] = []
    for i in range(20):
        texts.append(
            f"질문 {i}: {TRAINING_DATA[0]} "
            f"답변 {i}: {TRAINING_DATA[1]}"
        )
    for _ in range(7):
        texts.append(
            f"질문 0: {TRAINING_DATA[0]} "
            f"답변 0: {TRAINING_DATA[1]}"
        )
    for _ in range(4):
        texts.append(None)
    return texts


def _upload_to_minio(texts: list[str | None]) -> None:
    client = create_minio_client()
    ensure_bucket(client, MINIO_BUCKET)
    put_json_object(client, MINIO_BUCKET, MINIO_RAW_OBJECT_KEY, texts)