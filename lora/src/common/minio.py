"""Shared MinIO helpers for reading and writing JSON payloads."""

from __future__ import annotations

import json
from typing import Any

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError

from common.config import MINIO_ACCESS_KEY, MINIO_SECRET_KEY, MINIO_URL


def create_minio_client() -> Any:
    return boto3.client(
        "s3",
        endpoint_url=MINIO_URL,
        aws_access_key_id=MINIO_ACCESS_KEY,
        aws_secret_access_key=MINIO_SECRET_KEY,
        config=Config(signature_version="s3v4"),
        region_name="us-east-1",
    )


def ensure_bucket(client: Any, bucket_name: str) -> None:
    try:
        client.head_bucket(Bucket=bucket_name)
    except ClientError as exc:
        error_code = exc.response.get("Error", {}).get("Code", "")
        if error_code not in {"404", "NoSuchBucket", "NotFound"}:
            raise
        client.create_bucket(Bucket=bucket_name)


def put_json_object(client: Any, bucket_name: str, object_key: str, payload: Any) -> None:
    client.put_object(
        Bucket=bucket_name,
        Key=object_key,
        Body=json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8"),
        ContentType="application/json",
    )


def get_json_object(client: Any, bucket_name: str, object_key: str) -> Any:
    response = client.get_object(Bucket=bucket_name, Key=object_key)
    body = response["Body"].read().decode("utf-8")
    return json.loads(body)