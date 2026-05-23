import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
from datasets import Dataset
from torch.utils.data import DataLoader, Dataset as TorchDataset

# ==========================================
# 1. 데이터셋 준비
# ==========================================
def prepare_dataset(tokenizer):
    raw_data = [
        {"text": f"질문 {i}: KubeRay 분산 학습 테스트 시나리오입니다. 답변 {i}: 정상 작동 중입니다.{tokenizer.eos_token}"}
        for i in range(100)
    ]
    return Dataset.from_dict({"text": [d["text"] for d in raw_data]})


# ==========================================
# 2. PyTorch Dataset 래퍼
# ==========================================
class TextDataset(TorchDataset):
    def __init__(self, hf_dataset, tokenizer, max_length=256):
        self.data = hf_dataset
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = self.data[idx]["text"]
        tokenized = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        input_ids = tokenized["input_ids"].squeeze(0)
        attention_mask = tokenized["attention_mask"].squeeze(0)
        labels = input_ids.clone()
        labels[attention_mask == 0] = -100
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }


# ==========================================
# 3. 학습 함수
# ==========================================
def train_local(config):

    # 강제로 CPU 디바이스 지정
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"현재 강제 지정된 디바이스: {device}")

    model_id = "meta-llama/Llama-3.2-1B"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id
    )

    peft_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, peft_config)
    model.to(device)
    model.print_trainable_parameters()

    # Gradient Checkpointing 활성화
    # float32로 올리면 메모리(RAM) 사용량이 늘어나 스왑 지옥에 빠질 수 있으므로, 
    # 중간 계산값을 저장하지 않고 재계산하는 이 옵션을 켜서 RAM 부담을 절반으로 낮춥니다.
    model.gradient_checkpointing_enable()

    hf_dataset = prepare_dataset(tokenizer)
    dataset = TextDataset(hf_dataset, tokenizer)
    
    # 4. [수정] 데이터 로더 단일 스레드 고정
    dataloader = DataLoader(
        dataset, 
        batch_size=config["batch_size"], 
        shuffle=True,
        num_workers=0,
        pin_memory=False
    )

    # 오직 LoRA 레이어(학습 대상) 파라미터만 옵티마이저에 전달
    # 기존 model.parameters()는 동결된 Llama 가중치 10억 개를 매번 스캔하게 하여 
    # CPU 점유율을 뚝 떨어뜨리는 원인이 됩니다. 이를 원천 차단합니다.
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=2e-4)

    # ==========================================
    # 4. 학습 루프
    # ==========================================
    for epoch in range(config["epochs"]):
        model.train()
        total_loss = 0.0

        for step, batch in enumerate(dataloader):
            batch = {k: v.to(device) for k, v in batch.items()}
            
            # Forward
            outputs = model(**batch)
            loss = outputs.loss

            # Backward & Optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            # 첫 스텝이 풀렸는지 바로 확인하기 위해 출력 빈도를 1스텝 단위로 변경
            print(f"Epoch {epoch} | Step {step} | Loss: {loss.item():.4f}")

        avg_loss = total_loss / len(dataloader)
        print(f"=== Epoch {epoch} 완료 | Avg Loss: {avg_loss:.4f} ===")

    # ==========================================
    # 5. 모델 저장
    # ==========================================
    save_path = "./lora_output"
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    print(f"모델 저장 완료: {save_path}")


# ==========================================
# 6. 실행
# ==========================================
if __name__ == "__main__":
    import os
    HF_TOKEN = os.environ.get("HF_TOKEN")
    import sys
    
    if not HF_TOKEN:
        print("ERROR: HF_TOKEN environment variable is not set. Aborting.", file=sys.stderr)
      #  sys.exit(1)

    train_local({
        "epochs": 3,
        "batch_size": 2,
    })