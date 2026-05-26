"""Shared configuration values for the pipeline."""

from __future__ import annotations

import os

HF_TOKEN = os.environ.get("HF_TOKEN", "")
HF_HOME = os.environ.get("HF_HOME", "/shared/hf_home")
MODEL_ID = os.environ.get("MODEL_ID", "meta-llama/Llama-3.2-1B")
MODEL_NAME = os.environ.get("MODEL_NAME", "llm-finetune")

RAY_ADDRESS = os.environ.get("RAY_ADDRESS", "ray://192.168.0.182:10001")
RAY_STORAGE = os.environ.get("RAY_STORAGE_PATH", "/shared/ray-checkpoints")
RAY_NUM_WORKERS = int(os.environ.get("RAY_NUM_WORKERS", "1"))
USE_GPU = os.environ.get("USE_GPU", "false").lower() == "true"

TRAIN_EPOCHS = int(os.environ.get("TRAIN_EPOCHS", "3"))
TRAIN_BATCH = int(os.environ.get("TRAIN_BATCH", "1"))
MAX_SEQ_LEN = int(os.environ.get("MAX_SEQ_LEN", "256"))
LEARNING_RATE = float(os.environ.get("LEARNING_RATE", "2e-4"))
LORA_R = int(os.environ.get("LORA_R", "8"))
LORA_ALPHA = int(os.environ.get("LORA_ALPHA", "16"))

MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow.cnapcloud.com")
MLFLOW_MODEL_NAME = os.environ.get("MLFLOW_MODEL_NAME", "llm-finetune")
MLFLOW_EXPERIMENT = os.environ.get("MLFLOW_EXPERIMENT", "llm-finetune-pipeline")

VALIDATION_MAX_NULL_RATIO = float(os.environ.get("VALIDATION_MAX_NULL_RATIO", "0.05"))
VALIDATION_MAX_DUP_RATIO = float(os.environ.get("VALIDATION_MAX_DUP_RATIO", "0.10"))
VALIDATION_MIN_SAMPLES = int(os.environ.get("VALIDATION_MIN_SAMPLES", "50"))
VALIDATION_MIN_AVG_TOKENS = int(os.environ.get("VALIDATION_MIN_AVG_TOKENS", "10"))

EVAL_SAMPLE_COUNT = int(os.environ.get("EVAL_SAMPLE_COUNT", "5"))
EVAL_MIN_IMPROVEMENT_RATIO = float(os.environ.get("EVAL_MIN_IMPROVEMENT_RATIO", "0.01"))

SMOKE_TEST_PROMPTS = [
    "질문 {0}: KubeRay 분산 학습 테스트 시나리오입니다.",
    "질문 {1}: KubeRay 분산 학습 테스트 시나리오입니다.",
    "질문 {2}: KubeRay 분산 학습 테스트 시나리오입니다.",
]
SMOKE_MAX_LATENCY_SEC = float(os.environ.get("SMOKE_MAX_LATENCY_SEC", "30.0"))
SMOKE_MAX_NEW_TOKENS = int(os.environ.get("SMOKE_MAX_NEW_TOKENS", "50"))

ARTIFACT_DIR = os.environ.get("ARTIFACT_DIR", "./artifacts")
ANALYSIS_REPORT_PATH = f"{ARTIFACT_DIR}/analysis_report.json"
VALIDATION_REPORT_PATH = f"{ARTIFACT_DIR}/validation_report.json"
