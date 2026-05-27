"""Airflow wrapper for seeding initial MinIO data."""

from __future__ import annotations

import json
from pathlib import Path

from common.logging import setup_logging
from common.mlflow_utils import extract_metadata
from data.seed_data import run


def execute() -> dict:
    result = run()
    metadata = extract_metadata(result, ["status", "bucket", "object_key", "record_count"])
    _write_xcom(metadata)
    return metadata

def _write_xcom(payload: dict) -> None:
    xcom_path = Path("/airflow/xcom/return.json")
    xcom_path.parent.mkdir(parents=True, exist_ok=True)
    xcom_path.write_text(json.dumps(payload), encoding="utf-8")

def main() -> None:
    setup_logging("seed")
    execute()

if __name__ == "__main__":
    main()