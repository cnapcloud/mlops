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

def find_model_info_by_run_id(run_id: str) -> tuple[str, str, str]:
    """run_id를 기반으로 등록된 모델 이름과 버전을 역추적하여 models:/ URI를 생성합니다."""
    client = MlflowClient(MLFLOW_TRACKING_URI)
    
    # run_id 조건을 걸어 모든 모델 레지스트리 버전을 검색
    # (어떤 모델 이름 밑에 등록되어 있는지 모르는 상태에서 전체 검색)
    filter_string = f"run_id = '{run_id}'"
    model_versions = client.search_model_versions(filter_string=filter_string)
    
    if not model_versions:
        raise RuntimeError(
            f"해당 run_id('{run_id}')로 등록된 모델을 레지스트리에서 찾을 수 없습니다. "
            f"모델 등록(mlflow.register_model)이 선행되었는지 확인해주세요."
        )
        
    # 동일한 run_id로 여러 이름/버전이 등록될 수 있으므로, 
    # 보통 가장 최근에 등록된 첫 번째 객체를 사용합니다.
    matched_model = model_versions[0]
    
    model_name = matched_model.name
    model_version = matched_model.version
    model_uri = f"models:/{model_name}/{model_version}"
    
    print(f"🎯 발견된 모델 정보 -> 이름: {model_name}, 버전: {model_version}")
    return model_uri, model_version, model_name