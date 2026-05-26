"""
# 기본 실행 — MPS 자동 감지 (Apple Silicon)
python train_mlflow.py

# inference 모드 + device 지정 (MLflow run_id 또는 registered model 사용)
python3 train_mlflow.py --mode inference --run-id <mlflow_run_id>
python3 train_mlflow.py --mode inference --model-name <registered_model_name>

# train 모드 + device 지정
python3 train_mlflow.py --mode train --device mps
python3 train_mlflow.py --mode train --device cuda
python3 train_mlflow.py --mode train --device cpu
"""

import argparse
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
    TrainingArguments,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
log = logging.getLogger(__name__)

from common import config


# ==========================================
# Device 선택 유틸
# ==========================================
def get_device(prefer: str | None = None) -> torch.device:
    """
    prefer: "cpu" | "mps" | "cuda" | None
      - None 또는 "auto" → mps > cuda > cpu 순으로 자동 감지
      - 명시하면 해당 device가 실제로 사용 가능한지 확인 후 반환
        (불가능하면 fallback 없이 RuntimeError)
    """
    prefer = (prefer or "auto").lower()

    if prefer == "auto":
        if torch.backends.mps.is_available():
            device = torch.device("mps")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
    elif prefer == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS를 요청했지만 이 환경에서 사용할 수 없습니다.")
        device = torch.device("mps")
    elif prefer == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA를 요청했지만 이 환경에서 사용할 수 없습니다.")
        device = torch.device("cuda")
    elif prefer == "cpu":
        device = torch.device("cpu")
    else:
        raise ValueError(f"알 수 없는 device 값: '{prefer}' (cpu | mps | cuda | auto)")

    log.info("device=%s (prefer=%s)", device, prefer)
    return device


def _training_args_for_device(device: torch.device, output_dir: str) -> TrainingArguments:
    """device 종류에 따라 TrainingArguments 플래그를 자동으로 설정."""
    is_cuda = device.type == "cuda"
    is_cpu  = device.type == "cpu"

    return TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=config.TRAIN_BATCH,
        per_device_eval_batch_size=config.TRAIN_BATCH,
        num_train_epochs=config.TRAIN_EPOCHS,
        learning_rate=config.LEARNING_RATE,
        logging_steps=1,
        save_strategy="epoch",
        eval_strategy="steps",
        eval_steps=3,
        bf16=False,
        fp16=(is_cuda and config.USE_GPU),
        use_cpu=is_cpu,
        lr_scheduler_type="cosine",
        warmup_steps=0.1,
        max_grad_norm=1.0,
        dataloader_pin_memory=is_cuda,
        push_to_hub=False,
        report_to="none",
    )


# ==========================================
# Dataset
# ==========================================
def build_dataset(tokenizer):
    raw_data = [
        {"text": f"질문 {i}: KubeRay 분산 학습 테스트 시나리오입니다. 답변 {i}: 정상 작동 중입니다.{tokenizer.eos_token}"}
        for i in range(10)
    ]
    dataset = Dataset.from_dict({"text": [d["text"] for d in raw_data]})

    def tokenize(example):
        model_inputs = tokenizer(
            example["text"],
            truncation=True,
            max_length=256, # 256을 넘는 것만 자르고, 짧은 건 그대로 둡니다.
        )
        # 복사본을 만들어 둡니다. (콜레이터가 이 내부 패딩을 찾아내 -100으로 바꿀 것입니다)
        model_inputs["labels"] = model_inputs["input_ids"].copy() 
        return model_inputs

    dataset = dataset.map(tokenize, batched=False)
    split = dataset.train_test_split(test_size=0.1, seed=42)
    
    return split["train"], split["test"]

# ==========================================
# Train → MLflow에 저장
# ==========================================
def train(device: torch.device) -> str:
    """학습 후 MLflow에 모델을 저장하고 run_id를 반환."""

    hf_token = config.HF_TOKEN or os.environ.get("HF_TOKEN", "")
    if not hf_token:
        log.warning("HF_TOKEN not set; attempting anonymous access")

    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    mlflow.set_experiment(config.MLFLOW_EXPERIMENT)

    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_ID)
    if tokenizer.pad_token is None:
      tokenizer.pad_token = tokenizer.eos_token

    torch_dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

    model = AutoModelForCausalLM.from_pretrained(
        config.MODEL_ID,
        torch_dtype=torch_dtype,
    )
    model.to(device)

    peft_config = LoraConfig(
        r=config.LORA_R,
        lora_alpha=config.LORA_ALPHA,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    model.gradient_checkpointing_enable()

    train_ds, eval_ds = build_dataset(tokenizer)
    # tokenize 함수에서 수동으로 하던 고정 패딩 및 -100 라벨 마스킹을 
    # 배치(Batch) 단위로 '자동 + 동적(Dynamic)' 처리하기 위해 DataCollatorForSeq2Seq 사용
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer, 
        padding=True, 
        label_pad_token_id=-100
    )

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=config.LEARNING_RATE)

    safe_model_id = config.MODEL_ID.replace("/", "_")
    output_dir = f"{config.ARTIFACT_DIR}/{safe_model_id}/train"
    training_args = _training_args_for_device(device, output_dir)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=data_collator,
        optimizers=(optimizer, None),
    )

    with mlflow.start_run() as run:
        run_id = run.info.run_id
        log.info("MLflow Run=%s", run_id)

        # 하이퍼파라미터 기록
        mlflow.log_params({
            "MODEL_ID":          config.MODEL_ID,
            "TRAIN_EPOCHS":      config.TRAIN_EPOCHS,
            "TRAIN_BATCH":       config.TRAIN_BATCH,
            "MAX_SEQ_LEN":       config.MAX_SEQ_LEN,
            "LEARNING_RATE":     config.LEARNING_RATE,
            "LORA_R":            config.LORA_R,
            "LORA_ALPHA":        config.LORA_ALPHA,
            "device":            device.type,
        })

        # 학습
        trainer.train()

        # eval metrics 기록
        metrics = trainer.evaluate()
        log.info("eval=%s", metrics)
        mlflow.log_metrics(metrics)

        # ----------------------------------------
        # MLflow에 모델 저장 (디스크 저장 없음)
        # LoRA → base 병합 (PEFT merge_and_unload)
        # ----------------------------------------   
        log.info("LoRA 어댑터 가중치만 MLflow에 저장 중...")
        model.gradient_checkpointing_disable()

        with tempfile.TemporaryDirectory() as tmp_dir:
            # 어댑터 가중치만 저장 (adapter_config.json + adapter_model.safetensors)
            model.save_pretrained(tmp_dir)
            tokenizer.save_pretrained(tmp_dir)

            # Artifact로 업로드
            mlflow.log_artifacts(tmp_dir, artifact_path="lora_adapter")
            mlflow.log_param("base_model_id", config.MODEL_ID)
               
            # MLflow Model Registry에 등록 (모델과 토크나이저를 함께 패키징)
            log.info("[Rank 0] MLflow에 모델 등록 중 (Model Name: %s)...", config.MLFLOW_MODEL_NAME)
            raw_model = model.module if hasattr(model, "module") else model
            components = { "model": raw_model, "tokenizer": tokenizer }
                    
            mlflow.transformers.log_model(
                transformers_model=components,
                task="text-generation",
                name="model",
                registered_model_name=config.MLFLOW_MODEL_NAME,
            )

        log.info("MLflow LoRA 어댑터 저장 완료 (run_id=%s)", run_id)
        
    return run_id


# ==========================================
# Inference → MLflow에서 로드
# ==========================================
def inference(device: torch.device, run_id: str | None = None, model_name: str | None = None):
    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)

    # ------------------------------------------
    # MLflow URI 결정 → 어댑터 다운로드
    # ------------------------------------------
    if run_id:
        adapter_uri = f"runs:/{run_id}/lora_adapter"
        log.info("Experiment에서 run_id로 어댑터 로드: %s", adapter_uri)
        
        # MLflow에서 어댑터 로컬 임시 경로로 다운로드
        with tempfile.TemporaryDirectory() as tmp_dir:
            adapter_path = mlflow.artifacts.download_artifacts(
                artifact_uri=adapter_uri,
                dst_path=tmp_dir,
            )
            log.info("어댑터 다운로드 완료: %s", adapter_path)

            # base 모델은 로컬에서 로드 (config.MODEL_ID)
            torch_dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

            tokenizer = AutoTokenizer.from_pretrained(
                adapter_path,
                clean_up_tokenization_spaces=False,
            )

            base_model = AutoModelForCausalLM.from_pretrained(
                config.MODEL_ID,
                torch_dtype=torch_dtype,
                local_files_only=True,
            )
            model = PeftModel.from_pretrained(base_model, adapter_path)
    else:
        log.info("Registry 최신 버전 모델 로드: models:/%s/latest", model_name or config.MLFLOW_MODEL_NAME)
        name = model_name or config.MLFLOW_MODEL_NAME
        model_uri = f"models:/{name}/latest"
        components = mlflow.transformers.load_model(model_uri, return_type="components")
        model = components["model"]
        tokenizer = components["tokenizer"]

    model.to(device)
    log.info("모델 로드 완료 (device=%s)", device)

    # ------------------------------------------
    # 추론
    # ------------------------------------------
    prompt = "질문 {1}: KubeRay 분산 학습 테스트 시나리오입니다"
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=50,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated = outputs[0][inputs["input_ids"].shape[1]:]
    result = tokenizer.decode(generated, skip_special_tokens=True)
    print("\n" + "=" * 80)
    print(result)
    print("=" * 80)

# ==========================================
# Main
# ==========================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=["train", "inference"],
        default="inference",
        help="실행 모드 (기본값: inference)",
    )
    parser.add_argument(
        "--device",
        choices=["cpu", "mps", "cuda", "auto"],
        default="auto",
        help="device 선택 (기본값: auto = mps 우선)",
    )
    parser.add_argument(
        "--run-id",
        default=None,
        help="추론 시 사용할 MLflow run_id (없으면 Registry latest 사용)",
    )
    parser.add_argument(
        "--model-name",
        default=None,
        help="추론 시 사용할 MLflow 등록 모델명 (없으면 config.MLFLOW_MODEL_NAME 사용)",
    )
    args = parser.parse_args()

    device = get_device(args.device)

    if args.mode == "train":
        run_id = train(device)
        log.info("학습 완료. 추론하려면: --mode inference --run-id %s", run_id)
    else:
        inference(device, run_id=args.run_id, model_name=args.model_name)