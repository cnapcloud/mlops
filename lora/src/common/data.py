"""Data-related common utilities.

Contains helpers for preparing training/evaluation texts and loading
validated inputs from MinIO.
"""
from __future__ import annotations

from typing import List
import logging

from common import config
from common.minio import create_minio_client, get_json_object

log = logging.getLogger(__name__)


def ensure_eos_suffix(text: str, eos_token: str | None) -> str:
    if not eos_token:
        raise ValueError("tokenizer.eos_token is not set; cannot append EOS to training samples")

    normalized = text.rstrip()
    if normalized.endswith(eos_token):
        return normalized

    return f"{normalized}{eos_token}"


def load_validated_minio_data() -> List[str]:
    client = create_minio_client()
    payload = get_json_object(client, config.MINIO_BUCKET, config.MINIO_VALIDATED_OBJECT_KEY)

    if isinstance(payload, list):
        return [str(item).strip() for item in payload if item is not None and str(item).strip()]

    if isinstance(payload, dict):
        candidate = payload.get("texts") or payload.get("data") or payload.get("items")
        if isinstance(candidate, list):
            return [str(item).strip() for item in candidate if item is not None and str(item).strip()]

    raise ValueError(
        f"MinIO object must contain a JSON list or a mapping with texts/data/items: s3://{config.MINIO_BUCKET}/{config.MINIO_VALIDATED_OBJECT_KEY}"
    )
