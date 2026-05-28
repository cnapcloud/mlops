"""Shared configuration values for the pipeline."""

from __future__ import annotations

import os
import logging
from pathlib import Path


log = logging.getLogger(__name__)

def _load_properties() -> dict[str, str]:
    # 1. 경로 및 환경변수 기본 설정
    repo_root = Path(__file__).resolve().parents[2]
    dirs = [Path("/etc/lora"), Path.cwd(), repo_root]
    
    # 대소문자 구분 없이 CONFIG 환경변수 가져오기
    config_name = (os.environ.get("CONFIG") or os.environ.get("config") or "").strip().lower()
    
    # 2. 파일 목록 정의 (기본 파일 + 환경별 파일)
    files = ["config.properties"]
    if config_name in {"local", "dev", "prd"}:
        files.append(f"config-{config_name}.properties")
        
    # 3. 디렉토리와 파일을 조합하여 우선순위대로 후보 생성
    # 기본 파일들의 dirs 경로들 -> 환경 파일들의 dirs 경로들 순서로 배열됨
    candidates = [d / f for f in files for d in dirs]
    
    # 4. 로그 출력용 env_candidates 계산 (후반부 3개 경로가 환경 파일에 해당)
    env_candidates = candidates[3:] if len(files) > 1 else []
    
    log.info("Config load candidates: %s", ", ".join(str(path) for path in candidates))
    log.info("Selected env config: %s", env_candidates[-1] if env_candidates else "none")

    # 5. 프로퍼티 파일 로드 및 병합
    loaded: dict[str, str] = {}
    for path in candidates:
        if not path.exists():
            continue

        for line in path.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            # 주석 및 잘못된 형식 필터링
            if not stripped or stripped.startswith(("#", ";")) or "=" not in stripped:
                continue

            key, value = stripped.split("=", 1)
            loaded[key.strip()] = value.strip()

    return loaded

_PROPERTIES = _load_properties()


def _get(name: str, default: str) -> str:
    return os.environ.get(name, _PROPERTIES.get(name, default))


def _get_int(name: str, default: int) -> int:
    return int(_get(name, str(default)))


def _get_float(name: str, default: float) -> float:
    return float(_get(name, str(default)))


def _get_bool(name: str, default: bool) -> bool:
    value = _get(name, str(default)).strip().lower()
    if value in {"true", "1", "yes", "y", "on"}:
        return True
    if value in {"false", "0", "no", "n", "off"}:
        return False
    raise ValueError(f"Invalid boolean value for {name}: {value}")


HF_TOKEN = _get("HF_TOKEN", "")
HF_HOME = _get("HF_HOME", "/shared/hf_home")
MODEL_ID = _get("MODEL_ID", "meta-llama/Llama-3.2-1B")
MODEL_NAME = _get("MODEL_NAME", "llm-finetune")

RAY_ADDRESS = _get("RAY_ADDRESS", "ray://raycluster-head-svc.default.svc:10001")
RAY_STORAGE = _get("RAY_STORAGE_PATH", "/shared/ray-checkpoints")
RAY_NUM_WORKERS = _get_int("RAY_NUM_WORKERS", 1)
USE_GPU = _get_bool("USE_GPU", False)

# Resources per worker defaults — can be overridden in config.properties or env
# `WORKER_CPUS` sets number of CPU cores to request per worker when not using GPU
# `WORKER_GPUS` sets number of GPUs to request per worker when using GPU
RAY_WORKER_CPUS = _get_int("RAY_WORKER_CPUS", 4)
RAY_WORKER_GPUS = _get_int("RAY_WORKER_GPUS", 1)

TRAIN_EPOCHS = _get_int("TRAIN_EPOCHS", 3)
TRAIN_BATCH = _get_int("TRAIN_BATCH", 1)
MAX_SEQ_LEN = _get_int("MAX_SEQ_LEN", 256)
LEARNING_RATE = _get_float("LEARNING_RATE", 2e-4)
LORA_R = _get_int("LORA_R", 8)
LORA_ALPHA = _get_int("LORA_ALPHA", 16)

MLFLOW_TRACKING_URI = _get("MLFLOW_TRACKING_URI", "http://minio.mlops.svc:9000")
MLFLOW_MODEL_NAME = _get("MLFLOW_MODEL_NAME", "llm-finetune")
MLFLOW_EXPERIMENT = _get("MLFLOW_EXPERIMENT", "llm-finetune-pipeline")

MINIO_URL = _get("MINIO_URL", "http://minio.mlops.svc:9000")
MINIO_ACCESS_KEY = _get("MINIO_ACCESS_KEY", "minioadmin")
MINIO_SECRET_KEY = _get("MINIO_SECRET_KEY", "minioadmin")
MINIO_INSECURE = _get_bool("MINIO_INSECURE", False)
MINIO_BUCKET = _get("MINIO_BUCKET", "lora-data")
MINIO_RAW_OBJECT_KEY = _get("MINIO_RAW_OBJECT_KEY", "raw/seed_data.json")
MINIO_VALIDATED_OBJECT_KEY = _get("MINIO_VALIDATED_OBJECT_KEY", "validated/seed_data.json")

VALIDATION_MAX_NULL_RATIO = _get_float("VALIDATION_MAX_NULL_RATIO", 0.05)
VALIDATION_MAX_DUP_RATIO = _get_float("VALIDATION_MAX_DUP_RATIO", 0.10)
VALIDATION_MIN_SAMPLES = _get_int("VALIDATION_MIN_SAMPLES", 50)
VALIDATION_MIN_AVG_TOKENS = _get_int("VALIDATION_MIN_AVG_TOKENS", 10)

TRAINING_DATA = [
    "MLOps 파이프라인의 핵심 구성 요소를 설명하세요.",
    "데이터 준비, 학습, 평가, 배포 단계로 구성됩니다.",
]
EVAL_SAMPLE_COUNT = _get_int("EVAL_SAMPLE_COUNT", 5)
EVAL_MIN_IMPROVEMENT_RATIO = _get_float("EVAL_MIN_IMPROVEMENT_RATIO", 0.01)

SMOKE_TEST_PROMPTS = [
    "질문 0 : MLOps 파이프라인의 핵심 구성 요소를 설명하세요.",
    "질문 1 : MLOps 파이프라인의 핵심 구성 요소를 설명하세요.",
    "질문 2 : MLOps 파이프라인의 핵심 구성 요소를 설명하세요.",
]
SMOKE_MAX_LATENCY_SEC = _get_float("SMOKE_MAX_LATENCY_SEC", 30.0)
SMOKE_MAX_NEW_TOKENS = _get_int("SMOKE_MAX_NEW_TOKENS", 50)

ARTIFACT_DIR = _get("ARTIFACT_DIR", "./artifacts")
ANALYSIS_REPORT_PATH = f"{ARTIFACT_DIR}/analysis_report.json"
VALIDATION_REPORT_PATH = f"{ARTIFACT_DIR}/validation_report.json"
