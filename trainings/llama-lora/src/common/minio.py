"""Shared MinIO helpers for reading and writing JSON payloads."""

from __future__ import annotations

import json
from typing import Any

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError
import urllib3
from urllib3.exceptions import InsecureRequestWarning

from common.config import MINIO_ACCESS_KEY, MINIO_INSECURE, MINIO_SECRET_KEY, MINIO_URL


def create_minio_client() -> Any:
    # If the user intentionally disables TLS verification, suppress the
    # InsecureRequestWarning from urllib3 so the logs aren't noisy.
    if MINIO_INSECURE:
        urllib3.disable_warnings(InsecureRequestWarning)

    return boto3.client(
        "s3",
        endpoint_url=MINIO_URL,
        aws_access_key_id=MINIO_ACCESS_KEY,
        aws_secret_access_key=MINIO_SECRET_KEY,
        # MinIO works reliably with path-style requests for bucket operations.
        config=Config(signature_version="s3v4", s3={"addressing_style": "path"}),
        region_name="us-east-1",
        verify=not MINIO_INSECURE,
    )


def ensure_bucket(client: Any, bucket_name: str) -> None:
    try:
        response = client.list_buckets()
        bucket_names = {bucket.get("Name") for bucket in response.get("Buckets", [])}
        if bucket_name in bucket_names:
            return
        client.create_bucket(Bucket=bucket_name)
    except ClientError as exc:
        error_code = exc.response.get("Error", {}).get("Code", "")
        if error_code not in {"BucketAlreadyOwnedByYou", "BucketAlreadyExists"}:
            raise


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
