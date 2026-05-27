"""Airflow wrapper for evaluation."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from common.mlflow_utils import extract_metadata
from training.evaluate import run


def execute(train_result: dict | None = None) -> dict:
    result = run(train_result)
    metadata = extract_metadata(result, ["status", "promoted", "new_version", "decision"])
    _write_xcom(metadata)
    return metadata


def _write_xcom(payload: dict) -> None:
    xcom_path = Path("/airflow/xcom/return.json")
    xcom_path.parent.mkdir(parents=True, exist_ok=True)
    xcom_path.write_text(json.dumps(payload), encoding="utf-8")

def main() -> None:
    train_result = None
    if len(sys.argv) > 1 and sys.argv[1].strip():
        train_result = json.loads(sys.argv[1])
    execute(train_result=train_result)


if __name__ == "__main__":
    main()
