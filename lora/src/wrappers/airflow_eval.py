"""Airflow wrapper for evaluation."""

from __future__ import annotations

import json
from pathlib import Path

from common.mlflow_utils import extract_metadata
from training.evaluate import run


def execute(train_result: dict) -> dict:
    result = run(train_result)
    metadata = extract_metadata(result, ["status", "promoted", "new_version", "decision"])
    _write_xcom(metadata)
    return metadata


def _write_xcom(payload: dict) -> None:
    xcom_path = Path("/airflow/xcom/return.json")
    xcom_path.parent.mkdir(parents=True, exist_ok=True)
    xcom_path.write_text(json.dumps(payload), encoding="utf-8")
