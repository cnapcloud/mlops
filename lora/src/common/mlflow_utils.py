"""Small MLflow helpers kept out of domain code."""

from __future__ import annotations

from common.config import MLFLOW_TRACKING_URI
from mlflow import MlflowClient


def extract_metadata(result: dict, keys: list[str]) -> dict:
    return {key: result.get(key) for key in keys}

def get_latest_version(model_name: str) -> str | None:
    client = MlflowClient(MLFLOW_TRACKING_URI)
    versions = client.search_model_versions(f"name='{model_name}'")
    if not versions:
        return None
    latest = max(versions, key=lambda mv: int(mv.version))
    return latest