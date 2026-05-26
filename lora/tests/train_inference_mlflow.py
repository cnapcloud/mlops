import logging
import os
import tempfile

import mlflow
import torch

from datasets import Dataset
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Trainer,
)

from common import config

log = logging.getLogger(__name__)

# ==========================================
# Inference → MLflow에서 통합 로드 (pyfunc 방식)
# ==========================================
def inference(device: torch.device, run_id: str | None = None, model_name: str | None = None):
    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)

    # 1. MLflow Model URI 결정
    if run_id:
        # 아티팩트 경로가 "model"로 저장되었으므로 주소를 맞춰줍니다.
        model_uri = f"runs:/{run_id}/model"
        log.info("run_id로 모델 로드: %s", model_uri)
    else:
        name = model_name or config.MLFLOW_MODEL_NAME
        # 레지스트리에 등록된 최신 버전을 가리키는 정석 URI 표현법입니다.
        model_uri = f"models:/{name}/latest"
        log.info("Registry 최신 버전 모델 로드: %s", model_uri)

    # 2. 모델 로드 
    # (MLflow가 알아서 base 모델 위에 LoRA를 얹어 완성된 파이프라인으로 가져옵니다)
    log.info("MLflow에서 모델 복원 중...")
    loaded_model = mlflow.pyfunc.load_model(model_uri)
    log.info("모델 로드 완료")

    # 3. 추론 진행
    prompt = "질문 {1}: KubeRay 분산 학습 테스트 시나리오입니다."
    
    # pyfunc 모델의 predict는 텍스트 리스트를 입력으로 받으며, 
    # 내부적으로 tokenizer와 generate가 전부 수행됩니다.
    # 단, 세부 generate 파라미터(temperature 등)는 학습 시 log_model 할 때 
    # signature로 고정하지 않았다면 기본값으로 작동합니다.
    result = loaded_model.predict([prompt])
    
    print("\n" + "=" * 80)
    print(result[0])
    print("=" * 80)
    
if __name__ == "__main__":
    import logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    log = logging.getLogger(__name__)
    inference(device="mps", run_id=None, model_name="llm-finetune")