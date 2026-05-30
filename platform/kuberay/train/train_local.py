import sys
import os
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
def train_local(config, force_cpu=False):

    # --mode cpu 인자가 들어오면 무조건 cpu로 고정
    if force_cpu:
        device = torch.device("cpu")
    else:
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        
    print(f"현재 지정된 디바이스: {device}")

    model_id = "meta-llama/Llama-3.2-1B"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token

    # CPU 환경일 때 torch.float32 강제
    torch_dtype = torch.float32 if device.type == "cpu" else torch.bfloat16
    print(f"설정된 데이터 타입(dtype): {torch_dtype}")

    model = AutoModelForCausalLM.from_pretrained(
        model_id, 
        dtype=torch_dtype,
        low_cpu_mem_usage=True if device.type == "cpu" else False
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
    model.gradient_checkpointing_enable()

    hf_dataset = prepare_dataset(tokenizer)
    dataset = TextDataset(hf_dataset, tokenizer)
    
    # 데이터 로더 설정
    dataloader = DataLoader(
        dataset, 
        batch_size=config["batch_size"], 
        shuffle=True,
        num_workers=0,
        pin_memory=False
    )

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
# 6. 실행 (인자 처리 파트)
# ==========================================
if __name__ == "__main__":
    HF_TOKEN = os.environ.get("HF_TOKEN")
    
    if not HF_TOKEN:
        print("ERROR: HF_TOKEN environment variable is not set. Aborting.", file=sys.stderr)

    # [수정] '--mode cpu' 또는 '--mode=cpu'가 들어왔는지 검사
    args = [arg.lower() for arg in sys.argv]
    is_cpu_forced = False

    # 공백 구분 방식 (--mode cpu) 또는 붙여쓰기 방식 (--mode=cpu) 둘 다 대응
    if "--mode=cpu" in args:
        is_cpu_forced = True
    elif "--mode" in args:
        mode_idx = args.index("--mode")
        if mode_idx + 1 < len(args) and args[mode_idx + 1] == "cpu":
            is_cpu_forced = True

    if is_cpu_forced:
        print("⚠️ 실행 인자 [--mode cpu]가 감지되어 CPU 학습을 강제합니다.")

    train_local({
        "epochs": 3,
        "batch_size": 2,
    }, force_cpu=is_cpu_forced)