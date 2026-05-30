import os
import ray
from ray import train as rt  # 직관성을 위해 rt로 별칭 지정
from ray.train import RunConfig, ScalingConfig
from ray.train.torch import TorchTrainer, TorchConfig

# ==========================================
# 1. 분산 학습용 데이터셋 준비 (100건 예시)
# ==========================================
def prepare_dataset():
    from datasets import Dataset
    from transformers import AutoTokenizer
    
    model_id = "meta-llama/Llama-3.2-1B"
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_auth_token=os.environ.get("HF_TOKEN"))
    
    raw_data = [
        {"text": f"질문 {i}: KubeRay 분산 학습 테스트 시나리오입니다. 답변 {i}: 정상 작동 중입니다.{tokenizer.eos_token}"} 
        for i in range(10)
    ]
    hf_dataset = Dataset.from_dict({"text": [d["text"] for d in raw_data]})
    return ray.data.from_huggingface(hf_dataset)

# ==========================================
# 2. 각 워커 Pod에서 독립적으로 실행될 학습 함수
# ==========================================
def train_func_per_worker(config):
    import json
    import tempfile
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, get_peft_model
    from ray.train import Checkpoint  # 체크포인트 객체 생성을 위해 추가

    world_size = rt.get_context().get_world_size()
    world_rank = rt.get_context().get_world_rank()

    epochs = config["epochs"]
    storage_path = config["storage_path"]
    model_id = config["model_id"]

    device = torch.device(
        "cuda" if torch.cuda.is_available() 
        else "mps" if torch.backends.mps.is_available() 
        else "cpu"
    )  
    print(f"[Rank {world_rank}] 현재 지정된 디바이스: {device}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token

    # CPU 환경일 때 torch.float32 강제
    torch_dtype = torch.float32 if device.type == "cpu" else torch.bfloat16
    print(f"[Rank {world_rank}] 설정된 데이터 타입(dtype): {torch_dtype}")

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch_dtype,
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
    model = rt.torch.prepare_model(model)

    # Gradient Checkpointing 활성화 (RAM 스wap 지옥 방지)
    model.gradient_checkpointing_enable()

    local_dataset_shard = rt.get_dataset_shard("train_set")
    train_dataloader = local_dataset_shard.iter_batches(
        batch_size=1,
        prefetch_batches=2,
    )

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=2e-4)

    def collate_fn(batch):
        texts = [str(text) for text in batch["text"]]
        tokenized = tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=256,
            return_tensors="pt"
        )
        labels = tokenized["input_ids"].clone()
        labels[tokenized["attention_mask"] == 0] = -100
        tokenized["labels"] = labels
        return tokenized

    # 공통 제어 변수 사전 선언
    checkpoint_dir = None
    last_loss = 0.0
    current_device = rt.torch.get_device()
    
    # 🛠️ 개선: 고속 로컬 컨텍스트 매니저 안에서 안전하게 학습을 진행합니다.
    with tempfile.TemporaryDirectory() as local_temp_dir:
        for epoch in range(epochs):
            model.train()
            total_loss = 0.0
            steps = 0

            for batch in train_dataloader:
                inputs = collate_fn(batch)
                inputs = {k: v.to(current_device) for k, v in inputs.items()}
                outputs = model(**inputs)
                loss = outputs.loss
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                steps += 1
                # per-step 로그 출력
                print(f"[Rank {world_rank}] Epoch {epoch} | Step {steps} | Loss: {loss.item():.6f}")

            last_loss = total_loss / steps if steps > 0 else 0.0
            # 로그: 에폭 완료 요약
            lr = None
            try:
                lr = optimizer.param_groups[0].get("lr") if hasattr(optimizer, "param_groups") and len(optimizer.param_groups) > 0 else None
            except Exception:
                lr = None

            print(f"[Rank {world_rank}] Epoch {epoch} completed. Local Avg Loss: {last_loss:.6f} | Steps: {steps} | LR: {lr}")

            # 각 epoch마다 Ray에 메트릭을 보고 (중간/최종 구분 없이)
            metrics = {"loss": last_loss, "epoch": epoch, "steps": steps}
            if lr is not None:
                metrics["lr"] = lr
            rt.report(metrics)

        # 🛠️ 개선: 모든 에폭이 완전히 끝난 후, 데드락 방지를 위해 Early Return 없이 0번만 디렉토리를 굽습니다.
        if world_rank == 0:
            print("[Rank 0] 최종 결과 백업을 위한 마스터 디렉토리 생성 중...")
            try:
                # 공용 스토리지 경로 구성 (오타 방지를 위해 깔끔하게 os.path.join 처리)
                shared_output_dir = os.path.join(storage_path, model_id.replace('/', '_'))
                checkpoint_dir = os.path.join(shared_output_dir, "final_result_checkpoint")
                os.makedirs(checkpoint_dir, exist_ok=True)
                
                # 가벼운 매너용 종이 보증서(metadata.json) 하나를 구워둡니다.
                with open(os.path.join(checkpoint_dir, "metadata.json"), "w", encoding="utf-8") as handle:
                    json.dump(
                        {"loss": last_loss, "epoch": epochs - 1, "model_id": model_id},
                        handle, ensure_ascii=False, indent=2
                    )
                print(f"[Rank 0] 마스터 체크포인트 디렉토리 생성 완료: {checkpoint_dir}")
            except Exception as e:
                print(f"[Rank 0] 디렉토리 생성 실패: {str(e)}")
                raise
        else:
            print(f"[Rank {world_rank}] 다른 워커가 마스터 노드(Rank 0)의 파일 영구화 작업을 기다립니다.")

        # 🛠️ 개선: 삼항 연산자로 Rank 0만 상자를 채우고, 나머지는 None을 들고 대기소로 진입
        ray_checkpoint = Checkpoint.from_directory(checkpoint_dir) if checkpoint_dir else None

        # 모든 워커가 다 함께 발을 맞춰 최종 결과 도장을 찍고 안전하게 함께 퇴근합니다.
        rt.report(
            {
                "loss": last_loss, 
                "epoch": epochs - 1
            },
            checkpoint=ray_checkpoint,
        )

    print(f"[Rank {world_rank}] 워커 프로세스가 안전하게 정상 종료되었습니다.")

# ==========================================
# 4. KubeRay 분산 오케스트레이션 세팅 및 실행
# ==========================================
if __name__ == "__main__":
    import sys
    
    HF_TOKEN = os.environ.get("HF_TOKEN")
    if not HF_TOKEN:
        print("ERROR: HF_TOKEN environment variable is not set. Aborting.", file=sys.stderr)
        sys.exit(1)
        
    RAY_REMOTE_ADDRESS = "ray://192.168.0.182:10001"
    
    # 에러 유발 요인이었던 경로 맨 앞의 등호(=) 부호를 철저히 격리하고 배제합니다.
    RAY_STORAGE_PATH = "/mnt/data/lora_out" 
    MODEL_ID = "meta-llama/Llama-3.2-1B"

    runtime_env = {
        "pip": [
            "transformers",
            "peft",
            "datasets",
            "torch",
            "xgboost",
        ],
        "env_vars": {
            "HF_TOKEN": HF_TOKEN,
            "HF_HOME": "/mnt/data/hf_home",
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
            "RAY_ENABLE_RECORD_ACTOR_TASK_LOGGING": "1"
        },
    }

    ray.init(
        address=RAY_REMOTE_ADDRESS,
        ignore_reinit_error=True,
        runtime_env=runtime_env
    )

    print(f"Connecting to remote KubeRay cluster at: {RAY_REMOTE_ADDRESS}...")

    ray_dataset = prepare_dataset()

    scaling_config = ScalingConfig(
        num_workers=1,
        use_gpu=False,
        resources_per_worker={"CPU": 4},
    )

    trainer = TorchTrainer(
        train_loop_per_worker=train_func_per_worker,
        # 🛠️ 개선: 워커 내부에서 유연하게 주소 경로를 참조할 수 있도록 주입해 줍니다.
        train_loop_config={
            "epochs": 3,
            "storage_path": RAY_STORAGE_PATH,
            "model_id": MODEL_ID
        },
        scaling_config=scaling_config,
        torch_config=TorchConfig(backend="gloo"),
        datasets={"train_set": ray_dataset},
        run_config=RunConfig(storage_path=RAY_STORAGE_PATH) # Ray 마스터 백업 시스템용
    )

    print("--- Starting KubeRay Distributed CPU Training Pipeline ---")
    result = trainer.fit()
    print("--- Pipeline Completed Successfully ---")
    
    # 💡 이제 checkpoint_dir 덕분에 유실 없이 결과를 안전하게 받아옵니다.
    print(f"최종 취합된 메트릭 결과 기록창: {result.metrics}")
    print(f"최종 저장된 Ray 시스템 체크포인트 위치: {result.checkpoint}")