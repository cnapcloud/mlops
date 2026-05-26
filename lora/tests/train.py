"""
# 기본 실행 — MPS 자동 감지 (Apple Silicon)
python train.py

# inference 모드 + device 지정
python train.py --device mps
python train.py --device cuda
python train.py --device cpu

# train 모드 + device 지정
python train.py --mode train --device mps
python train.py --mode train --device cuda
python train.py --mode train --device cpu
"""


import argparse
import logging
import os

import mlflow
import mlflow.transformers as mlflow_tf
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
        # CUDA면 fp16, MPS/CPU면 float32 유지
        bf16=False,
        fp16=(is_cuda and config.USE_GPU),
        use_cpu=is_cpu,
        lr_scheduler_type="cosine",
        warmup_steps=0.1,
        max_grad_norm=1.0,
        dataloader_pin_memory=is_cuda,  # pin_memory는 CUDA에서만 유효
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
# Train
# ==========================================
def train(device: torch.device):
    hf_token = config.HF_TOKEN or os.environ.get("HF_TOKEN", "")
    if not hf_token:
        log.warning("HF_TOKEN not set; attempting anonymous access (may fail for private models)")

    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    mlflow.set_experiment(config.MLFLOW_EXPERIMENT)

    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_ID, local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # MPS는 bfloat16 미지원 → float32 사용
    torch_dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

    model = AutoModelForCausalLM.from_pretrained(
        config.MODEL_ID,
        torch_dtype=torch_dtype,
        local_files_only=True,
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
    trainer.train()
    trainer.save_model(f"{config.ARTIFACT_DIR}/lora_output")
    tokenizer.save_pretrained(f"{config.ARTIFACT_DIR}/lora_output")


# ==========================================
# Inference
# ==========================================
def inference(device: torch.device):
    tokenizer = AutoTokenizer.from_pretrained(
        f"{config.ARTIFACT_DIR}/lora_output",
        clean_up_tokenization_spaces=False,
    )

    torch_dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

    base_model = AutoModelForCausalLM.from_pretrained(
        config.MODEL_ID,
        torch_dtype=torch_dtype,
        local_files_only=True,
    )

    model = PeftModel.from_pretrained(base_model, f"{config.ARTIFACT_DIR}/lora_output")
    model.to(device)
    model.eval()

    prompt = "질문{1}: KubeRay 분산 학습 테스트 시나리오입니다"
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=50,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
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
        default="auto",           # auto → mps > cuda > cpu 자동 감지
        help="학습/추론에 사용할 device (기본값: auto = mps 우선)",
    )
    args = parser.parse_args()

    device = get_device(args.device)

    if args.mode == "train":
        train(device)
    else:
        inference(device)