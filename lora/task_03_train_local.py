import logging
import os

import mlflow
import mlflow.transformers as mlflow_tf
import torch

from datasets import Dataset
from peft import (
    LoraConfig,
    PeftModel,
    get_peft_model,
)

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

# ==========================================
# Logging
# ==========================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)

log = logging.getLogger(__name__)

# Use centralized config.py for environment-overridable constants
import config


# ==========================================
# Dataset
# ==========================================
def build_dataset(tokenizer):
    # 1. raw_data 생성 (요청하신 100개 데이터 및 EOS 토큰 반영)
    raw_data = [
        {"text": f"질문 {i}: KubeRay 분산 학습 테스트 시나리오입니다. 답변 {i}: 정상 작동 중입니다.{tokenizer.eos_token}"}
        for i in range(10)
    ]
    
    # 2. Dataset 객체 생성
    dataset = Dataset.from_dict({"text": [d["text"] for d in raw_data]})

    # 3. 토큰화 및 패딩 마스킹 처리 함수
    def tokenize(example):
        tokens = tokenizer(
            example["text"],
            truncation=True,
            padding="max_length",
            max_length=256,
        )

        # 기존 labels 복사
        tokens["labels"] = tokens["input_ids"].copy()
        
        # PyTorch 스타일의 마스킹을 map 함수에 맞게 리스트 컴프리헨션으로 구현
        # attention_mask가 0인 위치(패딩)는 학습에서 제외하도록 -100으로 변경
        tokens["labels"] = [
            label if mask == 1 else -100 
            for label, mask in zip(tokens["labels"], tokens["attention_mask"])
        ]

        return tokens

    # 4. map 적용 및 데이터셋 분할
    dataset = dataset.map(tokenize, batched=False)
    split = dataset.train_test_split(test_size=0.1, seed=42) # 일관된 분할을 위해 seed 추가
    
    return split["train"], split["test"]

# ==========================================
# Train
# ==========================================
def train():
    # load config constants
    hf_token = config.HF_TOKEN or os.environ.get("HF_TOKEN", "")
    if not hf_token:
        log.warning("HF_TOKEN not set; attempting anonymous access (may fail for private models)")

    device = torch.device("cuda" if torch.cuda.is_available() and config.USE_GPU else "cpu")
    log.info("device=%s", device)

    # ======================================
    # MLflow 설정
    # ======================================
    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    mlflow.set_experiment(config.MLFLOW_EXPERIMENT)

    # Tokenizer
    tokenizer_kwargs = {}
    if hf_token:
        tokenizer_kwargs["use_auth_token"] = hf_token

    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_ID, **tokenizer_kwargs)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token


    model = AutoModelForCausalLM.from_pretrained(
        config.MODEL_ID,
        dtype=torch.float32,  # gpu인 경우 torch.float16
    )

    # LoRA Config (config.py의 LORA_R, LORA_ALPHA 반영)
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

    # Dataset (config.py의 MAX_SEQ_LEN이 반영되도록 build_dataset 수정 필요 혹은 전역 참조)
    train_ds, eval_ds = build_dataset(tokenizer)

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    # Optimizer (config.py의 LEARNING_RATE 반영)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=config.LEARNING_RATE,
    )

    # ======================================
    # TrainingArguments 재설정
    # ======================================
    # 안전하게 경로명을 빌드하고 config.py 설정을 주입합니다.
    safe_model_id = config.MODEL_ID.replace("/", "_")
    output_dir = f"{config.ARTIFACT_DIR}/{safe_model_id}/train"

    training_args = TrainingArguments(
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
        fp16=(torch.cuda.is_available() and config.USE_GPU),
        lr_scheduler_type="cosine",
        warmup_steps=0.1,
        max_grad_norm=1.0,
        dataloader_pin_memory=False,
        push_to_hub=False,
        report_to="none", # MLflow 로깅을 수동으로 처리
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=data_collator,
        optimizers=(optimizer, None),
    )
    trainer.train()


# Inference
def inference():

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    tokenizer = AutoTokenizer.from_pretrained(f"{config.ARTIFACT_DIR}/lora_output")
    base_model = AutoModelForCausalLM.from_pretrained(
        config.MODEL_ID,
        torch_dtype=torch.float32,
    )

    model = PeftModel.from_pretrained(
        base_model,
        f"{config.ARTIFACT_DIR}/lora_output",
    )

    model.to(device)
    model.eval()
    prompt = "질문 {1}: KubeRay 분산 학습 테스트 시나리오입니다"
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
    ).to(device)

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
    if tokenizer.eos_token_id in generated:
        eos_index = (generated == tokenizer.eos_token_id).nonzero(as_tuple=True)[0][0].item()
        generated = generated[:eos_index]
        
    result = tokenizer.decode(generated, skip_special_tokens=True)

    print("\n")
    print("=" * 80)
    print(result)
    print("=" * 80)


# Main
if __name__ == "__main__":
    train()
    inference()
