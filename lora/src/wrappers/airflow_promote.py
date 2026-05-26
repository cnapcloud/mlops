"""Airflow wrapper for production promotion."""

from __future__ import annotations

import json
from pathlib import Path

from common.mlflow_utils import extract_metadata
from training.promote import run


def execute() -> dict:
    result = run()
    metadata = extract_metadata(result, ["status", "promoted", "decision"])
    _write_xcom(metadata)
    return metadata


def _write_xcom(payload: dict) -> None:
    xcom_path = Path("/airflow/xcom/return.json")
    xcom_path.parent.mkdir(parents=True, exist_ok=True)
    xcom_path.write_text(json.dumps(payload), encoding="utf-8")
