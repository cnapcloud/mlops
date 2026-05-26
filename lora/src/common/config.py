"""Shared configuration values for the pipeline."""

from __future__ import annotations

import os
from pathlib import Path


def _load_properties() -> dict[str, str]:
    # Check /etc first, then current working directory, then repository root
    repo_root_cfg = Path(__file__).resolve().parents[2] / "config.properties"
    candidates = [Path("/etc/lora/config.properties"), Path.cwd() / "config.properties", repo_root_cfg]
    loaded: dict[str, str] = {}

    for path in candidates:
        if not path.exists():
            continue

        for line in path.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#") or stripped.startswith(";"):
                continue
            if "=" not in stripped:
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

RAY_ADDRESS = _get("RAY_ADDRESS", "ray://192.168.0.182:10001")
RAY_STORAGE = _get("RAY_STORAGE_PATH", "/shared/ray-checkpoints")
RAY_NUM_WORKERS = _get_int("RAY_NUM_WORKERS", 1)
USE_GPU = _get_bool("USE_GPU", False)

TRAIN_EPOCHS = _get_int("TRAIN_EPOCHS", 3)
TRAIN_BATCH = _get_int("TRAIN_BATCH", 1)
MAX_SEQ_LEN = _get_int("MAX_SEQ_LEN", 256)
LEARNING_RATE = _get_float("LEARNING_RATE", 2e-4)
LORA_R = _get_int("LORA_R", 8)
LORA_ALPHA = _get_int("LORA_ALPHA", 16)

MLFLOW_TRACKING_URI = _get("MLFLOW_TRACKING_URI", "http://mlflow.cnapcloud.com")
MLFLOW_MODEL_NAME = _get("MLFLOW_MODEL_NAME", "llm-finetune")
MLFLOW_EXPERIMENT = _get("MLFLOW_EXPERIMENT", "llm-finetune-pipeline")

MINIO_URL = _get("MINIO_URL", "http://localhost:9000")
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

EVAL_SAMPLE_COUNT = _get_int("EVAL_SAMPLE_COUNT", 5)
EVAL_MIN_IMPROVEMENT_RATIO = _get_float("EVAL_MIN_IMPROVEMENT_RATIO", 0.01)

SMOKE_TEST_PROMPTS = [
    "질문 {0}: KubeRay 분산 학습 테스트 시나리오입니다.",
    "질문 {1}: KubeRay 분산 학습 테스트 시나리오입니다.",
    "질문 {2}: KubeRay 분산 학습 테스트 시나리오입니다.",
]
SMOKE_MAX_LATENCY_SEC = _get_float("SMOKE_MAX_LATENCY_SEC", 30.0)
SMOKE_MAX_NEW_TOKENS = _get_int("SMOKE_MAX_NEW_TOKENS", 50)

ARTIFACT_DIR = _get("ARTIFACT_DIR", "./artifacts")
ANALYSIS_REPORT_PATH = f"{ARTIFACT_DIR}/analysis_report.json"
VALIDATION_REPORT_PATH = f"{ARTIFACT_DIR}/validation_report.json"
